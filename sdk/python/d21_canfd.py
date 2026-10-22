# -*- coding: utf-8 -*-
import threading
import time
from D21ModbusCore import D21ModbusCore
from CANFDDevice import CANFDDevice

CANFD_MODBUS_REQ_ID = 0x010
CANFD_MODBUS_RESP_ID = 0x020

class d21_canfd(D21ModbusCore):
    def __init__(self, hand_id=0x01, modbus_port="COM3", timeout=0.1,
                 lock=None, tty_baudrate=10000000):   # 新增 tty_baudrate，默认 10M
        super().__init__(hand_id=hand_id, timeout=timeout)
        self.max_response_len = 59
        self.lock = lock
        self.device = CANFDDevice(channel=modbus_port, tty_baudrate=tty_baudrate)  # 传递
        self._evt = threading.Event()
        self._resp = None
        self._pending_func = None
        self._inflight = threading.Lock()
        self._rx_stop = threading.Event()
        self._rx_thread = None
        self.connected = False

    # ================= 连接 / 断开 =================
    def connect(self):
        """打开 CANable 总线并启动接收线程。"""
        if self.connected:
            return
        self.device.connect()
        self.connected = True
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def close(self):
        """停止接收线程并关闭总线。"""
        self._rx_stop.set()
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=1.0)
            self._rx_thread = None
        if self.device is not None:
            self.device.disconnect()
        self.connected = False

    # ================= 接收线程：持续 drain 总线 + 分发 =================
    def _rx_loop(self):
        while not self._rx_stop.is_set():
            if not self.connected:
                time.sleep(0.01)
                continue
            try:
                msg = self.device.recv(timeout=0.01)
            except Exception:
                time.sleep(0.01)
                continue
            if msg is None:
                continue
            self.on_rx(msg)

    def on_rx(self, msg) -> None:
        """收到 0x020 帧：仅接受与当前在途请求匹配的响应（过滤周期/手动发送的原样回显、时间戳）。"""
        if msg.arbitration_id != CANFD_MODBUS_RESP_ID:
            return
        data = bytes(msg.data)
        if len(data) < 5:
            return
        func = self._pending_func
        if func is None:
            return
        if data[0] != self.slave:
            return
        if data[1] != func and data[1] != (func | 0x80):
            return
        self._resp = data
        self._evt.set()

    # ================= 底层传输原语（CANFD 实现） =================
    def _transact(self, req: bytes, timeout=None, is_write: bool = False) -> bytes:
        """发 0x010 请求帧 → 等待匹配的 0x020 响应帧 → 返回完整响应字节。"""
        timeout = self.timeout if timeout is None else timeout
        func = req[1]
        with self._inflight:
            self._pending_func = func
            try:
                deadline = time.time() + timeout
                self._evt.clear()
                self._resp = None
                self._send(req)
                while True:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        raise TimeoutError("CANFD Modbus 响应超时")
                    if not self._evt.wait(remaining):
                        raise TimeoutError("CANFD Modbus 响应超时")
                    resp = self._resp
                    # 极少情况：误匹配/过短帧，等待下一帧真正的响应
                    if resp is None or len(resp) < 5:
                        self._evt.clear()
                        self._resp = None
                        continue
                    return resp
            finally:
                self._pending_func = None

    def _send(self, data: bytes):
        """发送 CANFD 请求帧；若配置了总线锁则加锁发送（与各 Worker 串行）。"""
        if self.lock is not None:
            with self.lock:
                self.device.send_frame(CANFD_MODBUS_REQ_ID, data, True, True)
        else:
            self.device.send_frame(CANFD_MODBUS_REQ_ID, data, True, True)
