# -*- coding: utf-8 -*-
"""
Device/CANFDDevice.py
=====================
CANFD 总线协议封装（python-can + CANable slcan）。
从 h7_canfd_上位机/canfd_test_pc/Device/CANFDDevice.py 复制（阶段2/3 已验证），
供 d21_canfd（阶段4，D21 上位机 CANFD 通道）使用。

CAN ID 分配（ID 越小优先级越高，PC 侧 0x010 最小 = 优先级最高）：
  PC_TX_ID      = 0x010  PC -> STM32 请求帧（Modbus 请求 / 手动 / 周期）
  STM32_ECHO_ID = 0x020  STM32 -> PC 响应帧（Modbus 响应 / 原样回显）
  STM32_TS_ID   = 0x030  STM32 -> PC 5ms 周期时间戳帧
"""
import time
import can
from can.interfaces.slcan import slcanBus
from can.util import dlc2len, len2dlc
import serial
import sys
import traceback


class _DeadlineSlcanBus(slcanBus):
    """Keep python-can's slcan framing; replace its unbounded serial flush."""

    def __init__(self, *args, write_timeout, **kwargs):
        self._write_timeout = write_timeout
        try:
            super().__init__(*args, **kwargs)
        except BaseException:
            self._is_shutdown = True
            if hasattr(self, "serialPortOrig"):
                self._close_port()
            raise

    def _close_port(self):
        primary = sys.exc_info()[1]
        try:
            self.serialPortOrig.close()
        except Exception:
            if primary is None:
                raise
            traceback.print_exc()

    def _write(self, string):
        port = self.serialPortOrig
        timeout = port.write_timeout
        if timeout is None:
            timeout = self._write_timeout  # also bounds initialization commands
        if timeout <= 0:
            raise serial.SerialTimeoutException("Serial write deadline expired")
        deadline = time.monotonic() + timeout
        port.write_timeout = timeout
        data = string.encode() + self.LINE_TERMINATOR
        if port.write(data) != len(data):
            raise serial.SerialTimeoutException("Incomplete slcan serial write")
        # pyserial.flush() uses tcdrain() on Linux, ignoring write_timeout.
        # Poll the output queue instead, sharing the write's remaining budget.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise serial.SerialTimeoutException("Serial drain deadline expired")
            if not port.out_waiting:
                return
            time.sleep(min(0.001, remaining))

    def shutdown(self):
        can.BusABC.shutdown(self)
        try:
            self.serialPortOrig.write_timeout = self._write_timeout
            self.close()  # slcan C command uses the same bounded write/drain
        finally:
            self._close_port()

# ---- CAN ID 分配 ----
PC_TX_ID      = 0x010
STM32_ECHO_ID = 0x020
STM32_TS_ID   = 0x030
TS_MAGIC      = 0x5A


class CANFDDevice:
    """
    CANFD 设备封装（CANable slcan / python-can）。
    行为对齐 D21 的 d21_rs485：connect/disconnect/connected + 帧级收发 + 解析工具。
    """

    def __init__(self, channel: str, bitrate: int = 1000000,
                 data_bitrate: int = 2000000, tty_baudrate: int = 10000000,
                 fd: bool = True, brs: bool = True, timeout: float = 0.1):
        self.channel = channel            # COM 口
        self.bitrate = bitrate            # 仲裁段波特率（1Mbps）
        self.data_bitrate = data_bitrate  # 数据段波特率（2Mbps，BRS）
        self.tty_baudrate = tty_baudrate  # CANable USB 串口波特率
        self.fd = fd                      # 是否 CAN FD
        self.brs = brs                    # 是否比特率切换
        self.timeout = timeout
        self._bus = None
        self.connected = False

    def connect(self):
        """打开 CANable（slcan）总线。"""
        if self.connected:
            return
        # slcan ignores the generic data_bitrate keyword. BitTimingFd selects
        # its supported S8/Y2 commands; only the rates (not these segments/clock)
        # are transmitted by the backend.
        timing = can.BitTimingFd.from_sample_point(
            f_clock=80000000, nom_bitrate=self.bitrate, nom_sample_point=80,
            data_bitrate=self.data_bitrate, data_sample_point=80,
        ) if self.fd else None
        self._bus = _DeadlineSlcanBus(
            channel=self.channel,
            tty_baudrate=self.tty_baudrate,
            rtscts=False,
            bitrate=self.bitrate,
            timing=timing,
            write_timeout=self.timeout,
        )
        self.connected = True

    def disconnect(self):
        """关闭总线。"""
        try:
            if self._bus is not None:
                self._bus.shutdown()
        finally:
            self._bus = None
            self.connected = False

    # ---------- 发送 ----------
    def send_frame(self, arb_id: int, data, is_fd=None, brs=None, timeout=None):
        """发送一帧。data 为 bytes/bytearray；is_fd/brs 缺省用构造参数。"""
        if not self.connected:
            raise RuntimeError("CANFD device not connected")
        data = bytes(data)
        is_fd = self.fd if is_fd is None else is_fd
        if len(data) > (64 if is_fd else 8):
            raise ValueError("Payload exceeds CAN frame capacity")
        if is_fd:
            # Pad after the embedded Modbus CRC, never inside the RTU frame.
            data = data.ljust(dlc2len(len2dlc(len(data))), b"\x00")
        msg = can.Message(
            arbitration_id=arb_id,
            data=data,
            is_extended_id=False,
            is_fd=is_fd,
            bitrate_switch=self.brs if brs is None else brs,
        )
        self._bus.send(msg, timeout=self.timeout if timeout is None else timeout)

    # ---------- 接收 ----------
    def recv(self, timeout: float = 0.0):
        """接收一帧；无数据返回 None。timeout 为阻塞秒数（0=非阻塞）。"""
        if not self.connected:
            return None
        return self._bus.recv(timeout=timeout)

    # ---------- 帧解析工具 ----------
    @staticmethod
    def parse_timestamp_frame(data):
        """
        解析时间戳帧：data[0]=0x5A, data[1:2]=计数(LE), data[3:6]=ms(LE)。
        返回 (ok, counter, ms)；格式不符返回 (False, None, None)。
        """
        if len(data) < 7 or data[0] != TS_MAGIC:
            return False, None, None
        counter = data[1] | (data[2] << 8)
        ms = data[3] | (data[4] << 8) | (data[5] << 16) | (data[6] << 24)
        return True, counter, ms

    @staticmethod
    def build_timestamp_payload(seq: int, ms: int):
        """构造 8 字节时间戳 payload：[0]=0x5A, [1:2]=seq(LE), [3:6]=ms(LE), [7]=0。"""
        seq &= 0xFFFF
        ms &= 0xFFFFFFFF
        return bytes([
            TS_MAGIC,
            seq & 0xFF, (seq >> 8) & 0xFF,
            ms & 0xFF, (ms >> 8) & 0xFF, (ms >> 16) & 0xFF, (ms >> 24) & 0xFF,
            0,
        ])

    def __del__(self):
        try:
            self.disconnect()
        except Exception:
            pass
