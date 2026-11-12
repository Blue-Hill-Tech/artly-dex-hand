# -*- coding: utf-8 -*-
import os
import threading
import time
from D21ModbusCore import D21ModbusCore
from CANFDDevice import CANFDDevice
from slcan_fd import SlcanFD

# "fast": SlcanFD (pyserial only, synchronous, pipelined transactions); "python-can": the original CANFDDevice.
TRANSPORT_ENV = "AAC_CANFD_TRANSPORT"

CANFD_MODBUS_REQ_ID = 0x010
CANFD_MODBUS_RESP_ID = 0x020

class d21_canfd(D21ModbusCore):
    def __init__(self, hand_id=0x01, modbus_port="COM3", timeout=0.1,
                 lock=None, tty_baudrate=10000000, register_map=None, data_bitrate=None, transport=None):
        super().__init__(hand_id=hand_id, timeout=timeout, register_map=register_map)
        # data phase rate follows the firmware (v1 2 Mbit/s, v2 5 Mbit/s) unless given
        data_bitrate = data_bitrate or self.REGISTER_MAPS[self.register_map]["canfd_data_bitrate"]
        self.max_response_len = 59
        self.lock = lock
        transport = transport or os.environ.get(TRANSPORT_ENV, "fast")
        if transport not in ("fast", "python-can"):
            raise ValueError(f"{TRANSPORT_ENV} must be fast or python-can, not {transport!r}")
        self.pipelined = transport == "fast"   # transact_many() available: replies to back-to-back requests
        if self.pipelined:
            self.device = SlcanFD(channel=modbus_port, data_bitrate=data_bitrate, tty_baudrate=tty_baudrate,
                                  timeout=timeout, request_id=CANFD_MODBUS_REQ_ID, response_id=CANFD_MODBUS_RESP_ID)
        else:
            self.device = CANFDDevice(channel=modbus_port, data_bitrate=data_bitrate, tty_baudrate=tty_baudrate,
                                      timeout=timeout)
        self._evt = threading.Event()
        self._resp = None
        self._pending_func = None
        self._inflight = threading.Lock()
        self._rx_stop = threading.Event()
        self._rx_thread = None
        self._rx_error = None
        self.connected = False

    # ================= 连接 / 断开 =================
    def connect(self):
        """打开 CANable 总线并启动接收线程。"""
        if self.connected:
            return
        self.check_receive_error()
        self.device.connect()
        self.connected = True
        if self.pipelined:
            return                      # synchronous transport: no receive thread
        self._rx_stop.clear()
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()

    def close(self):
        """停止接收线程并关闭总线。"""
        self._rx_stop.set()
        if self._rx_thread is not None and self._rx_thread.ident is not None:
            self._rx_thread.join(timeout=1.0)
        try:
            if self.device is not None:
                self.device.disconnect()
            if self._rx_thread is not None:
                if self._rx_thread.is_alive():
                    raise RuntimeError("CANFD receive thread did not stop within 1 second")
                self._rx_thread = None
        finally:
            self.connected = False

    # ================= 接收线程：持续 drain 总线 + 分发 =================
    def _rx_loop(self):
        while not self._rx_stop.is_set():
            if not self.connected:
                time.sleep(0.01)
                continue
            try:
                msg = self.device.recv(timeout=0.01)
                if msg is not None:
                    self.on_rx(msg)
            except Exception as error:
                # Retain the receive traceback and wake any blocked transaction.
                self._rx_error = (error, error.__traceback__)
                self._evt.set()
                return

    def check_receive_error(self):
        if self._rx_error is not None:
            error, traceback = self._rx_error
            raise error.with_traceback(traceback)

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
    def transact_many(self, requests, timeout=None, overlap=False):
        """Replies to several requests sent back to back (fast transport only); TimeoutError if one is missing."""
        if not self.pipelined:
            raise RuntimeError("pipelined transactions need the fast CAN FD transport")
        timeout = self.timeout if timeout is None else timeout
        deadline = time.monotonic() + timeout
        if not self._inflight.acquire(timeout=timeout):
            raise TimeoutError("CANFD transaction lock deadline expired")
        try:
            if self.lock is not None and not self.lock.acquire(timeout=max(0, deadline - time.monotonic())):
                raise TimeoutError("CANFD bus lock deadline expired")
            try:
                return self.device.transact(requests, timeout=max(0.0, deadline - time.monotonic()), overlap=overlap)
            finally:
                if self.lock is not None:
                    self.lock.release()
        finally:
            self._inflight.release()

    def _transact(self, req: bytes, timeout=None, is_write: bool = False) -> bytes:
        """发 0x010 请求帧 → 等待匹配的 0x020 响应帧 → 返回完整响应字节。"""
        if self.pipelined:
            try:
                return self.transact_many([req], timeout)[0]
            except TimeoutError:
                self.device.drain()
                raise TimeoutError("CANFD Modbus 响应超时")
        timeout = self.timeout if timeout is None else timeout
        func = req[1]
        deadline = time.monotonic() + timeout
        if not self._inflight.acquire(timeout=timeout):
            raise TimeoutError("CANFD transaction lock deadline expired")
        self._pending_func = func
        try:
            self._evt.clear()
            self._resp = None
            self.check_receive_error()
            self._send(req, deadline)
            while True:
                self.check_receive_error()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("CANFD Modbus 响应超时")
                if not self._evt.wait(remaining):
                    self.check_receive_error()
                    raise TimeoutError("CANFD Modbus 响应超时")
                self.check_receive_error()
                resp = self._resp
                # 极少情况：误匹配/过短帧，等待下一帧真正的响应
                if resp is None or len(resp) < 5:
                    self._evt.clear()
                    self._resp = None
                    continue
                return resp
        finally:
            self._pending_func = None
            self._inflight.release()

    def _send(self, data: bytes, deadline):
        """Share the transaction budget across bus-lock wait, write and drain."""
        if self.lock is not None:
            if not self.lock.acquire(timeout=max(0, deadline - time.monotonic())):
                raise TimeoutError("CANFD bus lock deadline expired")
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("CANFD transmission deadline expired")
            self.device.send_frame(CANFD_MODBUS_REQ_ID, data, True, True, timeout=remaining)
        finally:
            if self.lock is not None:
                self.lock.release()
