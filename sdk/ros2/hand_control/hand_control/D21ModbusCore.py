# -*- coding: utf-8 -*-
"""
Device/D21ModbusCore.py
=======================
Modbus RTU 协议 + D21 业务层（485 / CANFD 双通道共享核心）。

仅包含 d21_rs485 中实际存在的方法，与 HandDevice 功能完全一致。
子类需实现 _transact(req, timeout, is_write) 和 close()。
"""
import struct
import math


class D21ModbusCore:
    """Modbus RTU 协议 + D21 灵巧手业务层（485/CANFD 共享）"""

    # ------- 关节角度范围映射（度） -------
    _ANGLE_LIMITS = {
        0: (0, 100),      # Thumb abduction
        1: (-35, 35),     # Thumb rotation
        2: (0, 50),       # Thumb CMC
        3: (0, 70),       # Thumb MCP
        4: (-20, 20),     # Index abduction
        5: (0, 90),       # Index MCP
        6: (0, 100),      # Index PIP
        7: (-20, 20),     # Middle abduction
        8: (0, 90),       # Middle MCP
        9: (0, 100),      # Middle PIP
        10: (-20, 20),    # Ring abduction
        11: (0, 90),      # Ring MCP
        12: (0, 100),     # Ring PIP
        13: (-20, 20),    # Pinky abduction
        14: (0, 90),      # Pinky MCP
        15: (0, 100),     # Pinky PIP
    }

    def __init__(self, hand_id=0x01, timeout=0.1):
        self.slave = hand_id
        self.timeout = timeout
        self.connected = False
        self.max_response_len = 250   # 由子类覆盖（485为250，CANFD为59）

    # ================= 底层传输原语（子类实现） =================
    def _transact(self, req: bytes, timeout=None, is_write: bool = False) -> bytes:
        raise NotImplementedError

    # ---------- CRC16 ----------
    @staticmethod
    def _crc16(data: bytes) -> int:
        crc = 0xFFFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x0001:
                    crc = (crc >> 1) ^ 0xA001
                else:
                    crc >>= 1
        return crc

    # ================= 请求构建 + 响应解析 =================
    def _send_request(self, func_code, address, quantity_or_value, description="",
                      is_write=False, is_multiple=False, write_values=None, timeout=None):
        if not is_write:
            req = struct.pack('>BBHH', self.slave, func_code, address, quantity_or_value)
        else:
            if not is_multiple:
                req = struct.pack('>BBHH', self.slave, func_code, address, quantity_or_value)
            else:
                count = quantity_or_value
                req = struct.pack('>BBHHB', self.slave, func_code, address, count, count * 2)
                for v in write_values:
                    req += struct.pack('>H', v)
        crc = self._crc16(req)
        req += struct.pack('<H', crc)

        resp = self._transact(req, timeout, is_write=is_write)

        if is_write:
            return self._parse_write_response(resp, func_code, description)
        return self._parse_response(resp, func_code, description)

    def _parse_response(self, resp: bytes, func_code, description=""):
        if len(resp) < 5:
            raise RuntimeError(f"{description}: Response header incomplete")
        slave, resp_func = resp[0], resp[1]
        if slave != self.slave:
            raise RuntimeError(f"{description}: Slave address mismatch")
        if resp_func == (func_code | 0x80):
            err_code = resp[2] if len(resp) > 3 else 0
            raise RuntimeError(f"{description}: Modbus exception 0x{func_code:02X}, error code {err_code}")
        if resp_func != func_code:
            raise RuntimeError(f"{description}: Function code mismatch")
        data_len = resp[2]
        total = 3 + data_len
        if len(resp) < total + 2:
            raise RuntimeError(f"{description}: Data incomplete")
        recv_crc = struct.unpack('<H', resp[total:total + 2])[0]
        calc_crc = self._crc16(resp[:total])
        if recv_crc != calc_crc:
            raise RuntimeError(f"{description}: CRC check failed")
        return resp[3:total]

    def _parse_write_response(self, resp: bytes, func_code, description=""):
        if len(resp) < 8:
            raise RuntimeError(f"{description}: Write response incomplete (expected 8 bytes, received {len(resp)})")
        slave = resp[0]
        resp_func = resp[1]
        if slave != self.slave:
            raise RuntimeError(f"{description}: Slave address mismatch")
        if resp_func == (func_code | 0x80):
            err_code = resp[2] if len(resp) > 2 else 0
            raise RuntimeError(f"{description}: Modbus exception 0x{func_code:02X}, error code {err_code}")
        recv_crc = struct.unpack('<H', resp[-2:])[0]
        calc_crc = self._crc16(resp[:-2])
        if recv_crc != calc_crc:
            raise RuntimeError(f"{description}: CRC check failed")
        return b''

    # ---------- 16-bit read (holding registers) ----------
    def _read_16bit_holding(self, address, count, description=""):
        max_count = self.max_response_len // 2
        if count <= max_count:
            data = self._send_request(0x03, address, count, description)
            return [struct.unpack('>H', data[i:i + 2])[0] for i in range(0, len(data), 2)]
        out = []
        off = 0
        while off < count:
            n = min(count - off, max_count)
            data = self._send_request(0x03, address + off, n, description)
            out.extend(struct.unpack('>H', data[i:i + 2])[0] for i in range(0, len(data), 2))
            off += n
        return out

    # ---------- 16-bit read (input registers) ----------
    def _read_16bit_input(self, address, count, description=""):
        max_count = self.max_response_len // 2
        if count <= max_count:
            data = self._send_request(0x04, address, count, description)
            return [struct.unpack('>H', data[i:i + 2])[0] for i in range(0, len(data), 2)]
        out = []
        off = 0
        while off < count:
            n = min(count - off, max_count)
            data = self._send_request(0x04, address + off, n, description)
            out.extend(struct.unpack('>H', data[i:i + 2])[0] for i in range(0, len(data), 2))
            off += n
        return out

    # ---------- Read tactile sensor 8-bit data (function code 0x05) ----------
    def _read_8bit_fast(self, address, byte_count, description=""):
        if byte_count <= self.max_response_len:
            data = self._send_request(0x05, address, byte_count, description)
            return [b for b in data]
        out = []
        off = 0
        while off < byte_count:
            n = min(byte_count - off, self.max_response_len)
            chunk = self._send_request(0x05, address + off, n, description)
            out.extend(b for b in chunk)
            off += n
        return out

    # ---------- Write operations ----------
    def _write_single_holding(self, address, value, description="", timeout=None):
        self._send_request(0x06, address, value, description, is_write=True, timeout=timeout)

    def _write_multiple_holding(self, address, values, description="", timeout=None):
        self._send_request(0x10, address, len(values), description,
                           is_write=True, is_multiple=True, write_values=values,
                           timeout=timeout)

    # ======================= 用户接口（与 d21_rs485 完全一致） ======================

    # ---------- 角度 → 寄存器值 ----------
    def _angle_to_register(self, joint_index, angle_deg):
        min_deg, max_deg = self._ANGLE_LIMITS[joint_index]
        if angle_deg < min_deg or angle_deg > max_deg:
            raise ValueError(
                f"Joint {joint_index} angle {angle_deg:.1f}° out of allowed range [{min_deg}°, {max_deg}°]"
            )
        reg = int(round((angle_deg - min_deg) / (max_deg - min_deg) * 2000))
        return max(0, min(2000, reg))

    # ---------- 写位置指令 ----------
    def write_single_joint_position(self, joint_index, angle_deg):
        if not 0 <= joint_index <= 15:
            raise ValueError(f"Joint index must be 0~15, received {joint_index}")
        reg_value = self._angle_to_register(joint_index, angle_deg)
        self._write_single_holding(joint_index, reg_value, f"write joint {joint_index}")

    def write_joint_position_commands(self, angles_deg):
        if len(angles_deg) != 16:
            raise ValueError("Need exactly 16 angle values")
        reg_values = [self._angle_to_register(i, angles_deg[i]) for i in range(16)]
        self._write_multiple_holding(0, reg_values, "write all joints")

    def write_thumb_position_commands(self, angles_deg):
        if len(angles_deg) != 4:
            raise ValueError("Need exactly 4 angle values")
        reg_values = [self._angle_to_register(i, angles_deg[i]) for i in range(4)]
        self._write_multiple_holding(0, reg_values, "write thumb joints")

    def write_index_position_commands(self, angles_deg):
        if len(angles_deg) != 3:
            raise ValueError("Need exactly 3 angle values")
        reg_values = [self._angle_to_register(4 + i, angles_deg[i]) for i in range(3)]
        self._write_multiple_holding(4, reg_values, "write index joints")

    def write_middle_position_commands(self, angles_deg):
        if len(angles_deg) != 3:
            raise ValueError("Need exactly 3 angle values")
        reg_values = [self._angle_to_register(7 + i, angles_deg[i]) for i in range(3)]
        self._write_multiple_holding(7, reg_values, "write middle joints")

    def write_ring_position_commands(self, angles_deg):
        if len(angles_deg) != 3:
            raise ValueError("Need exactly 3 angle values")
        reg_values = [self._angle_to_register(10 + i, angles_deg[i]) for i in range(3)]
        self._write_multiple_holding(10, reg_values, "write ring joints")

    def write_pinky_position_commands(self, angles_deg):
        if len(angles_deg) != 3:
            raise ValueError("Need exactly 3 angle values")
        reg_values = [self._angle_to_register(13 + i, angles_deg[i]) for i in range(3)]
        self._write_multiple_holding(13, reg_values, "write pinky joints")

    # ---------- 速度 ----------
    def _percent_to_register(self, percent):
        if not 0 <= percent <= 100:
            raise ValueError(f"Speed percentage {percent} out of range [0, 100]")
        reg = int(round(percent / 100.0 * 2000))
        return max(0, min(2000, reg))

    def write_single_finger_speed(self, finger_index, speed_percent):
        if not 1 <= finger_index <= 5:
            raise ValueError(f"Finger index must be 1~5, received {finger_index}")
        reg_value = self._percent_to_register(speed_percent)
        self._write_single_holding(finger_index + 15, reg_value, f"write finger {finger_index} speed")

    def write_joint_speed_commands(self, speeds_percent):
        if len(speeds_percent) != 5:
            raise ValueError("Need exactly 5 speed percentages")
        reg_values = [self._percent_to_register(p) for p in speeds_percent]
        self._write_multiple_holding(16, reg_values, "write all speeds")

    # ---------- 控制模式（寄存器21，写入1=位置，2=力矩） ----------
    def set_control_mode(self, mode):
        if mode not in (1, 2):
            raise ValueError(f"Control mode must be 1 or 2, received {mode}")
        self._write_single_holding(21, mode, "set control mode")

    # ---------- 力矩参数 ----------
    def set_single_torque(self, joint_index, torque_reg):
        if not 1 <= joint_index <= 16:
            raise ValueError(f"Joint index must be 1~16, received {joint_index}")
        if not -1000 <= torque_reg <= 1000:
            raise ValueError(f"Torque value {torque_reg} out of range [-1000, 1000]")
        address = 67 + joint_index
        self._write_single_holding(address, int(torque_reg + 1000), f"set joint {joint_index} torque")

    def set_all_torques(self, torques_reg):
        if len(torques_reg) != 16:
            raise ValueError(f"Need exactly 16 torque values, got {len(torques_reg)}")
        reg_values = []
        for idx, val in enumerate(torques_reg, start=1):
            if not -1000 <= val <= 1000:
                raise ValueError(f"Torque value at joint {idx} is {val}, out of range [-1000, 1000]")
            reg_values.append(int(val + 1000))
        self._write_multiple_holding(68, reg_values, "set all torques")

    # ---------- 传感器零标定 ----------
    def calibrate_sensor(self, finger_index):
        if not 1 <= finger_index <= 5:
            raise ValueError(f"Finger index must be 1~5, received {finger_index}")
        self._write_single_holding(47, finger_index, f"calibrate finger {finger_index} sensor", timeout=1.0)

    # ---------- 设备 ID ----------
    def set_device_id(self, new_id):
        if not 1 <= new_id <= 254:
            raise ValueError("ID must be 1~254")
        self._write_single_holding(58, new_id, "set device ID", timeout=0.5)
        self.slave = new_id

    # ---------- 风扇 ----------
    def set_fan_mode(self, mode):
        if not 0 <= mode <= 5:
            raise ValueError("Fan mode must be 0~5")
        self._write_single_holding(59, mode, "set fan mode")

    # ---------- 保存配置 ----------
    def save_config(self):
        self._write_single_holding(60, 1, "save config", timeout=1.0)

    # ---------- 整手使能 ----------
    def set_hand_enable(self, enable):
        if enable not in (0, 1):
            raise ValueError(f"Hand enable must be 0 or 1, received {enable}")
        self._write_single_holding(66, enable, "set hand enable")

    # ---------- 读取传感器 ----------
    def _register_to_angle(self, joint_index, reg_val):
        if not 0 <= joint_index <= 15:
            raise ValueError(f"Joint index must be 0~15, received {joint_index}")
        min_deg, max_deg = self._ANGLE_LIMITS[joint_index]
        reg_clamped = max(0, min(2000, reg_val))
        return min_deg + (reg_clamped / 2000.0) * (max_deg - min_deg)

    def read_joint_positions(self):
        raw = self._read_16bit_input(0, 16, "joint positions")
        return [self._register_to_angle(i, raw[i]) for i in range(16)]

    def read_joint_velocities(self):
        raw = self._read_16bit_input(16, 16, "joint velocities")
        return [v / 5.0 for v in raw]

    def read_motor_currents(self):
        raw = self._read_16bit_input(32, 16, "motor currents")
        return [(reg - 1000) / 1000.0 * 1.0 for reg in raw]

    def read_temp(self):
        raw = self._read_16bit_input(48, 16, "temperature")
        return [t / 10.0 for t in raw]

    def read_hardware_version(self):
        return self._read_16bit_input(64, 1, "hardware_version")[0]

    def read_software_version(self):
        return self._read_16bit_input(65, 1, "software_version")[0]

    # ---------- 触觉传感器 ----------
    @staticmethod
    def _map_tactile_xy(reg):
        if reg <= 128:
            return reg / 10.0
        else:
            return (reg - 256) / 10.0

    @staticmethod
    def _map_tactile_z(reg):
        return reg / 10.0

    def read_tactile_forces(self):
        raw = self._read_8bit_fast(0, 15, "tactile forces")
        mapped = []
        for i in range(0, len(raw), 3):
            x = self._map_tactile_xy(raw[i])
            y = self._map_tactile_xy(raw[i + 1])
            z = self._map_tactile_z(raw[i + 2])
            mapped.extend([x, y, z])
        return mapped

    def read_thumb_forces(self):
        raw = self._read_8bit_fast(0, 3, "thumb forces")
        return [self._map_tactile_xy(raw[0]), self._map_tactile_xy(raw[1]), self._map_tactile_z(raw[2])]

    def read_index_forces(self):
        raw = self._read_8bit_fast(3, 3, "index forces")
        return [self._map_tactile_xy(raw[0]), self._map_tactile_xy(raw[1]), self._map_tactile_z(raw[2])]

    def read_middle_forces(self):
        raw = self._read_8bit_fast(6, 3, "middle forces")
        return [self._map_tactile_xy(raw[0]), self._map_tactile_xy(raw[1]), self._map_tactile_z(raw[2])]

    def read_ring_forces(self):
        raw = self._read_8bit_fast(9, 3, "ring forces")
        return [self._map_tactile_xy(raw[0]), self._map_tactile_xy(raw[1]), self._map_tactile_z(raw[2])]

    def read_pinky_forces(self):
        raw = self._read_8bit_fast(12, 3, "pinky forces")
        return [self._map_tactile_xy(raw[0]), self._map_tactile_xy(raw[1]), self._map_tactile_z(raw[2])]

    # ---------- 压力分布 ----------
    def read_thumb_x_pressures(self):
        raw = self._read_8bit_fast(15, 116, "thumb_x")
        return [self._map_tactile_xy(v) for v in raw]

    def read_thumb_y_pressures(self):
        raw = self._read_8bit_fast(131, 116, "thumb_y")
        return [self._map_tactile_xy(v) for v in raw]

    def read_thumb_z_pressures(self):
        raw = self._read_8bit_fast(247, 116, "thumb_z")
        return [self._map_tactile_z(v) for v in raw]

    def read_thumb_norm_pressures(self):
        x_raw = self._read_8bit_fast(15, 116, "thumb_x")
        y_raw = self._read_8bit_fast(131, 116, "thumb_y")
        z_raw = self._read_8bit_fast(247, 116, "thumb_z")
        mag = []
        for x, y, z in zip(x_raw, y_raw, z_raw):
            fx = self._map_tactile_xy(x)
            fy = self._map_tactile_xy(y)
            fz = self._map_tactile_z(z)
            mag.append(math.sqrt(fx ** 2 + fy ** 2 + fz ** 2))
        return [round(v, 1) for v in mag]

    def read_index_x_pressures(self):
        raw = self._read_8bit_fast(363, 52, "index_x")
        return [self._map_tactile_xy(v) for v in raw]

    def read_index_y_pressures(self):
        raw = self._read_8bit_fast(415, 52, "index_y")
        return [self._map_tactile_xy(v) for v in raw]

    def read_index_z_pressures(self):
        raw = self._read_8bit_fast(467, 52, "index_z")
        return [self._map_tactile_z(v) for v in raw]

    def read_index_norm_pressures(self):
        x_raw = self._read_8bit_fast(363, 52, "index_x")
        y_raw = self._read_8bit_fast(415, 52, "index_y")
        z_raw = self._read_8bit_fast(467, 52, "index_z")
        mag = []
        for x, y, z in zip(x_raw, y_raw, z_raw):
            fx = self._map_tactile_xy(x)
            fy = self._map_tactile_xy(y)
            fz = self._map_tactile_z(z)
            mag.append(math.sqrt(fx ** 2 + fy ** 2 + fz ** 2))
        return [round(v, 1) for v in mag]

    def read_middle_x_pressures(self):
        raw = self._read_8bit_fast(519, 52, "middle_x")
        return [self._map_tactile_xy(v) for v in raw]

    def read_middle_y_pressures(self):
        raw = self._read_8bit_fast(571, 52, "middle_y")
        return [self._map_tactile_xy(v) for v in raw]

    def read_middle_z_pressures(self):
        raw = self._read_8bit_fast(623, 52, "middle_z")
        return [self._map_tactile_z(v) for v in raw]

    def read_middle_norm_pressures(self):
        x_raw = self._read_8bit_fast(519, 52, "middle_x")
        y_raw = self._read_8bit_fast(571, 52, "middle_y")
        z_raw = self._read_8bit_fast(623, 52, "middle_z")
        mag = []
        for x, y, z in zip(x_raw, y_raw, z_raw):
            fx = self._map_tactile_xy(x)
            fy = self._map_tactile_xy(y)
            fz = self._map_tactile_z(z)
            mag.append(math.sqrt(fx ** 2 + fy ** 2 + fz ** 2))
        return [round(v, 1) for v in mag]

    def read_ring_x_pressures(self):
        raw = self._read_8bit_fast(675, 52, "ring_x")
        return [self._map_tactile_xy(v) for v in raw]

    def read_ring_y_pressures(self):
        raw = self._read_8bit_fast(727, 52, "ring_y")
        return [self._map_tactile_xy(v) for v in raw]

    def read_ring_z_pressures(self):
        raw = self._read_8bit_fast(779, 52, "ring_z")
        return [self._map_tactile_z(v) for v in raw]

    def read_ring_norm_pressures(self):
        x_raw = self._read_8bit_fast(675, 52, "ring_x")
        y_raw = self._read_8bit_fast(727, 52, "ring_y")
        z_raw = self._read_8bit_fast(779, 52, "ring_z")
        mag = []
        for x, y, z in zip(x_raw, y_raw, z_raw):
            fx = self._map_tactile_xy(x)
            fy = self._map_tactile_xy(y)
            fz = self._map_tactile_z(z)
            mag.append(math.sqrt(fx ** 2 + fy ** 2 + fz ** 2))
        return [round(v, 1) for v in mag]

    def read_pinky_x_pressures(self):
        raw = self._read_8bit_fast(831, 52, "pinky_x")
        return [self._map_tactile_xy(v) for v in raw]

    def read_pinky_y_pressures(self):
        raw = self._read_8bit_fast(883, 52, "pinky_y")
        return [self._map_tactile_xy(v) for v in raw]

    def read_pinky_z_pressures(self):
        raw = self._read_8bit_fast(935, 52, "pinky_z")
        return [self._map_tactile_z(v) for v in raw]

    def read_pinky_norm_pressures(self):
        x_raw = self._read_8bit_fast(831, 52, "pinky_x")
        y_raw = self._read_8bit_fast(883, 52, "pinky_y")
        z_raw = self._read_8bit_fast(935, 52, "pinky_z")
        mag = []
        for x, y, z in zip(x_raw, y_raw, z_raw):
            fx = self._map_tactile_xy(x)
            fy = self._map_tactile_xy(y)
            fz = self._map_tactile_z(z)
            mag.append(math.sqrt(fx ** 2 + fy ** 2 + fz ** 2))
        return [round(v, 1) for v in mag]

    # ---------- 读保持寄存器（回读配置） ----------
    def read_commanded_joint_positions(self):
        raw = self._read_16bit_holding(0, 16, "read commanded positions")
        return [self._register_to_angle(i, raw[i]) for i in range(16)]

    def read_single_commanded_joint_position(self, joint_index):
        if not 0 <= joint_index <= 15:
            raise ValueError("Joint index must be 0~15")
        reg = self._read_16bit_holding(joint_index, 1, f"read commanded joint {joint_index}")[0]
        return self._register_to_angle(joint_index, reg)

    def read_finger_commanded_positions(self, finger_index):
        if not 1 <= finger_index <= 5:
            raise ValueError("Finger index must be 1~5")
        start_map = {1: 0, 2: 4, 3: 7, 4: 10, 5: 13}
        count_map = {1: 4, 2: 3, 3: 3, 4: 3, 5: 3}
        start = start_map[finger_index]
        count = count_map[finger_index]
        raw = self._read_16bit_holding(start, count, f"read finger {finger_index} commanded positions")
        return [self._register_to_angle(start + i, raw[i]) for i in range(count)]

    def read_single_finger_speed(self, finger_index):
        if not 1 <= finger_index <= 5:
            raise ValueError("Finger index must be 1~5")
        reg = self._read_16bit_holding(15 + finger_index, 1, f"read finger {finger_index} speed")[0]
        return reg / 10.0

    def read_all_finger_speeds(self):
        raw = self._read_16bit_holding(16, 5, "read all finger speeds")
        return [r / 10.0 for r in raw]

    def read_control_mode(self):
        return self._read_16bit_holding(21, 1, "read control mode")[0]

    def read_single_torque(self, joint_index):
        if not 1 <= joint_index <= 16:
            raise ValueError(f"Joint index must be 1~16, received {joint_index}")
        addr = 67 + joint_index
        reg = self._read_16bit_holding(addr, 1, f"read joint {joint_index} torque")[0]
        return reg - 1000

    def read_all_torques(self):
        raw = self._read_16bit_holding(68, 16, "read all torques")
        return [r - 1000 for r in raw]

    def read_device_id(self):
        return self._read_16bit_holding(58, 1, "read device ID")[0]

    def read_fan_mode(self):
        return self._read_16bit_holding(59, 1, "read fan mode")[0]

    def read_hand_enable(self):
        return self._read_16bit_holding(66, 1, "read hand enable")[0]

    # -------- 断开连接（子类必须实现）--------
    def close(self):
        raise NotImplementedError("Subclass must implement close()")
