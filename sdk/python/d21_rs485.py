# -*- coding: utf-8 -*-
import serial
import struct
from D21ModbusCore import D21ModbusCore


class d21_rs485(D21ModbusCore):
    """RS485 串口 Modbus RTU 客户端（仅负责串口通信）"""

    def __init__(self, hand_id=0x01, modbus_port="COM3", baudrate=500000, timeout=0.1):
        """
        初始化串口连接
        :param hand_id: 从站地址（默认 0x01）
        :param modbus_port: 串口号
        :param baudrate: 波特率（默认 115200）
        :param timeout: 串口超时（秒）
        """
        super().__init__(hand_id=hand_id, timeout=timeout)
        self.serial = serial.Serial(
            port=modbus_port,
            baudrate=baudrate,
            bytesize=8,
            parity='N',
            stopbits=1,
            timeout=timeout
        )
        if not self.serial.is_open:
            raise ConnectionError(f"无法打开串口 {modbus_port}")
        self.connected = True
        self.max_response_len = 250   # RS485 支持较长响应

    # ---------- 实现抽象方法 _transact ----------
    def _transact(self, req: bytes, timeout=None, is_write: bool = False) -> bytes:
        """
        发送 Modbus 请求帧，接收并返回完整响应帧（含地址/功能码/长度/数据/CRC）。
        若写操作，返回响应帧（固定8字节）。
        """
        old_timeout = self.serial.timeout
        if timeout is not None:
            self.serial.timeout = timeout
        try:
            self.serial.reset_input_buffer()
            self.serial.write(req)
            self.serial.flush()

            if is_write:
                # 写响应固定8字节
                response = self.serial.read(8)
                if len(response) < 8:
                    raise RuntimeError("写响应不完整")
                return response
            else:
                # 读响应：先读3字节头
                header = self.serial.read(3)
                if len(header) < 3:
                    raise RuntimeError("响应头不足")
                slave, func, data_len = struct.unpack('>BBB', header)
                # 读取数据 + CRC
                data = self.serial.read(data_len + 2)
                if len(data) < data_len + 2:
                    raise RuntimeError("数据不完整")
                return header + data
        finally:
            if timeout is not None:
                self.serial.timeout = old_timeout

    # ---------- 关闭连接 ----------
    def close(self):
        """关闭串口"""
        if self.serial.is_open:
            self.serial.close()
        self.connected = False

