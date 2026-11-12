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

    # ------- 寄存器换算表（固件版本） / register scaling per firmware -------
    # v1: original firmware, 0-2000 registers (spec section 4).
    # v2: new firmware, 0-40000 registers (vendor table 2026-11): 20x finer position/speed resolution, signed
    #     velocity readback, current and torque centred on 20000.
    #   full_scale       position / speed register at 100 % of the range
    #   signed_zero      register value of 0 for current, torque and (v2) velocity
    #   velocity         (scale, offset): deg/s = (raw - offset) * scale; v1 is unsigned
    #   torque_max       largest |torque| accepted by the register (v2 recommends |torque| <= 10000)
    #   thumb_torque_max same for motors 1-4: the v1.1 manual (table 3.5) limits them to 10000-30000, i.e. +-10000
    #   temperature      v2 only: input 48-63, 0-1500 -> deg C = raw / 10 (manual 3.2.4 / 3.4.1)
    #   stall_current    v2 only: holding 84-99, 0-40000 -> 0-1 A, position mode only (manual 3.1.4)
    #   canfd_data_bitrate / rs485_baudrate   bus rates fixed by that firmware (protocol v1.1 change log: main
    #                    controller v2.0 moved CAN FD data phase 2M -> 5M and RS-485 500k -> 4M)
    REGISTER_MAPS = {
        "v1": dict(full_scale=2000, signed_zero=1000, velocity=(400.0 / 2000, 0), torque_max=1000,
                   thumb_torque_max=1000, canfd_data_bitrate=2000000, rs485_baudrate=500000),
        "v2": dict(full_scale=40000, signed_zero=20000, velocity=(400.0 / 20000, 20000), torque_max=20000,
                   thumb_torque_max=10000, canfd_data_bitrate=5000000, rs485_baudrate=4000000),
    }
    DEFAULT_REGISTER_MAP = "v2"

    def __init__(self, hand_id=0x01, timeout=0.1, register_map=None):
        self.slave = hand_id
        self.timeout = timeout
        self.register_map = register_map or self.DEFAULT_REGISTER_MAP
        if self.register_map not in self.REGISTER_MAPS:
            raise ValueError(f"register_map must be one of {sorted(self.REGISTER_MAPS)}, got {self.register_map!r}")
        regmap = self.REGISTER_MAPS[self.register_map]
        self.full_scale = regmap["full_scale"]
        self.signed_zero = regmap["signed_zero"]
        self.velocity_scale, self.velocity_offset = regmap["velocity"]
        self.torque_max = regmap["torque_max"]
        self.thumb_torque_max = regmap["thumb_torque_max"]
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
            return self._parse_write_response(resp, req, description)
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

    def _parse_write_response(self, resp: bytes, request: bytes, description=""):
        if len(resp) < 5:
            raise RuntimeError(f"{description}: Write response header incomplete")
        if resp[0] != request[0]:
            raise RuntimeError(f"{description}: Slave address mismatch")
        func_code = request[1]
        exception = resp[1] == (func_code | 0x80)
        if not exception and resp[1] != func_code:
            raise RuntimeError(f"{description}: Function code mismatch")
        size = 5 if exception else 8
        if len(resp) < size:
            raise RuntimeError(f"{description}: Write response incomplete (expected {size} bytes, received {len(resp)})")
        # CAN FD can pad an exception to 8 bytes; CRC covers the RTU frame only.
        recv_crc = struct.unpack('<H', resp[size - 2:size])[0]
        if recv_crc != self._crc16(resp[:size - 2]):
            raise RuntimeError(f"{description}: CRC check failed")
        if exception:
            raise RuntimeError(f"{description}: Modbus exception 0x{func_code:02X}, error code {resp[2]}")
        # FC06 echoes register/value; FC10 echoes starting register/count.
        if resp[2:6] != request[2:6]:
            raise RuntimeError(f"{description}: Write acknowledgement echo mismatch")
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
        reg = int(round((angle_deg - min_deg) / (max_deg - min_deg) * self.full_scale))
        return max(0, min(self.full_scale, reg))

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
        reg = int(round(percent / 100.0 * self.full_scale))
        return max(0, min(self.full_scale, reg))

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
        address = 67 + joint_index
        self._write_single_holding(address, self.torque_to_register(torque_reg, joint_index),
                                   f"set joint {joint_index} torque")

    def torque_limit(self, motor=None):
        """Largest |torque| the register accepts; motors 1-4 (thumb) are limited to +-10000 on v2."""
        return self.thumb_torque_max if motor is not None and 1 <= motor <= 4 else self.torque_max

    def torque_to_register(self, torque, motor=None):
        """Signed torque value (v1 -1000..1000, v2 -20000..20000, thumb motors 1-4 +-10000) -> register."""
        limit = self.torque_limit(motor)
        if not -limit <= torque <= limit:
            where = f" for motor {motor}" if motor is not None else ""
            raise ValueError(f"Torque value {torque} out of range [-{limit}, {limit}]{where}")
        return int(round(torque)) + self.signed_zero

    def set_all_torques(self, torques_reg):
        if len(torques_reg) != 16:
            raise ValueError(f"Need exactly 16 torque values, got {len(torques_reg)}")
        reg_values = [self.torque_to_register(val, motor) for motor, val in enumerate(torques_reg, 1)]
        self._write_multiple_holding(68, reg_values, "set all torques")

    # ---------- 堵转电流（v2, holding 84-99） ----------
    def _require_v2(self, what):
        if self.register_map == "v1":
            raise NotImplementedError(f"{what} needs the new firmware (register map v2)")

    def set_all_stall_currents(self, amps):
        """Position-mode stall current per motor 1-16 in A (0-1): holding 84-99, 0-40000 (v1.1 manual 3.1.4)."""
        self._require_v2("stall current")
        if len(amps) != 16:
            raise ValueError(f"Need exactly 16 stall currents, got {len(amps)}")
        regs = []
        for a in amps:
            if not 0 <= a <= 1:
                raise ValueError(f"Stall current {a} A out of range [0, 1]")
            regs.append(int(round(a * self.full_scale)))
        self._write_multiple_holding(84, regs, "set stall currents")

    def read_all_stall_currents(self):
        self._require_v2("stall current")
        raw = self._read_16bit_holding(84, 16, "stall currents")
        return [r / float(self.full_scale) for r in raw]

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
        if not 0 <= reg_val <= self.full_scale:
            raise ValueError(f"Joint {joint_index} register out of range: {reg_val} (map {self.register_map}, "
                             f"0-{self.full_scale})")
        return min_deg + (reg_val / float(self.full_scale)) * (max_deg - min_deg)

    def detect_register_map(self):
        """'v1' if every joint position register is <= 2000, else 'v2'.

        Old firmware never reports more than 2000. On new firmware every joint at once would have to sit in the
        lowest 5 % of its range (all abductions at -18..-20 deg, thumb rotation below -31.5 deg) to look like v1."""
        raw = self._read_16bit_input(0, 16, "joint positions (map check)")
        return "v1" if max(raw) <= self.REGISTER_MAPS["v1"]["full_scale"] else "v2"

    def read_joint_positions(self):
        raw = self._read_16bit_input(0, 16, "joint positions")
        if len(raw) != 16:
            raise ValueError(f"Expected 16 joint position registers, received {len(raw)}")
        return [self._register_to_angle(i, raw[i]) for i in range(16)]

    def read_joint_velocities(self):
        raw = self._read_16bit_input(16, 16, "joint velocities")
        # deg/s; v1 is unsigned (magnitude only), v2 is signed
        return [(v - self.velocity_offset) * self.velocity_scale for v in raw]

    def read_motor_currents(self):
        raw = self._read_16bit_input(32, 16, "motor currents")
        return [(reg - self.signed_zero) / float(self.signed_zero) for reg in raw]   # A, -1..1

    def read_firmware_info(self):
        """Protocol v1.1 input registers 100-103 (new firmware only): main controller and driver version and date.

        Version = major*100 + minor (v2.0 -> 200); date = (year-2000)<<9 | month<<5 | day."""
        raw = self._read_16bit_input(100, 4, "firmware versions")

        def date(v):
            return f"{2000 + (v >> 9):04d}-{(v >> 5) & 0xF:02d}-{v & 0x1F:02d}"

        return {"main_version": f"v{raw[0] // 100}.{raw[0] % 100}", "main_date": date(raw[1]),
                "driver_version": f"v{raw[2] // 100}.{raw[2] % 100}", "driver_date": date(raw[3])}

    def read_motor_temperatures(self):
        """Motor 1-16 temperature in deg C: input 48-63, raw / 10 (v1.1 manual 3.2.4; new firmware only)."""
        self._require_v2("motor temperature")
        raw = self._read_16bit_input(48, 16, "motor temperatures")
        return [r / 10.0 for r in raw]

    def read_hardware_version(self):
        """Old firmware only: input 64/65 are not version registers on v2 (use read_firmware_info)."""
        if self.register_map != "v1":
            raise NotImplementedError("input 64 is not a version register on the new firmware; use read_firmware_info()")
        return self._read_16bit_input(64, 1, "hardware_version")[0]

    def read_software_version(self):
        if self.register_map != "v1":
            raise NotImplementedError("input 65 is not a version register on the new firmware; use read_firmware_info()")
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
        return reg * 100.0 / self.full_scale   # percent (the vendor SDK divided by 10, inconsistent with the write)

    def read_all_finger_speeds(self):
        raw = self._read_16bit_holding(16, 5, "read all finger speeds")
        return [r * 100.0 / self.full_scale for r in raw]   # percent

    def read_control_mode(self):
        return self._read_16bit_holding(21, 1, "read control mode")[0]

    def read_single_torque(self, joint_index):
        if not 1 <= joint_index <= 16:
            raise ValueError(f"Joint index must be 1~16, received {joint_index}")
        addr = 67 + joint_index
        reg = self._read_16bit_holding(addr, 1, f"read joint {joint_index} torque")[0]
        return reg - self.signed_zero

    def read_all_torques(self):
        raw = self._read_16bit_holding(68, 16, "read all torques")
        return [r - self.signed_zero for r in raw]

    def read_device_id(self):
        return self._read_16bit_holding(58, 1, "read device ID")[0]

    def read_fan_mode(self):
        return self._read_16bit_holding(59, 1, "read fan mode")[0]

    def read_hand_enable(self):
        return self._read_16bit_holding(66, 1, "read hand enable")[0]

    # -------- 断开连接（子类必须实现）--------
    def close(self):
        raise NotImplementedError("Subclass must implement close()")
