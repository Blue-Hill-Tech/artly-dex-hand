"""Small degree-valued adapter for the bundled D21 CAN FD SDK."""

import importlib
import math
import os
from pathlib import Path
import sys
import time
import traceback


SDK_DIR = Path(__file__).resolve().parents[1] / "aac" / "Python SDK"


def load_sdk_module(name):
    # Vendor modules use bare sibling imports. Resolve them independently of cwd.
    sys.path.insert(0, str(SDK_DIR))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(SDK_DIR))


_CORE = load_sdk_module("D21ModbusCore").D21ModbusCore
ANGLE_LIMITS = _CORE._ANGLE_LIMITS
REGISTER_MAPS = _CORE.REGISTER_MAPS
# Firmware register map: the SDK default (v2, new firmware, 0-40000; v1 is the old firmware's 0-2000) unless
# D21_REGISTER_MAP pins one. AACHand.open() auto-detects by trying the default first, then the other map.
REGISTER_MAP_ENV = "D21_REGISTER_MAP"


def default_register_map():
    value = os.environ.get(REGISTER_MAP_ENV, "").strip() or _CORE.DEFAULT_REGISTER_MAP
    if value not in REGISTER_MAPS:
        raise ValueError(f"{REGISTER_MAP_ENV} must be one of {sorted(REGISTER_MAPS)}, got {value!r}")
    return value


def command_resolution_deg(register_map):
    """Largest angle step of one position register count over all joints (v1 0.05 deg, v2 0.0025 deg)."""
    full_scale = REGISTER_MAPS[register_map]["full_scale"]
    return max(hi - lo for lo, hi in ANGLE_LIMITS.values()) / full_scale
JOINT_NAMES = ("thumb_abduction", "thumb_rotation", "thumb_cmc", "thumb_mcp") + tuple(
    f"{finger}_{joint}"
    for finger in ("index", "middle", "ring", "pinky")
    for joint in ("abduction", "mcp", "pip")
)


def finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def positive_number(value, name):
    value = finite_number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def validate_vector(values, length, name):
    if not isinstance(values, (list, tuple)) or len(values) != length:
        raise ValueError(f"{name} requires exactly {length} values")
    return [finite_number(v, f"{name}[{i}]") for i, v in enumerate(values)]


def validate_positions(values):
    values = validate_vector(values, 16, "angle")
    for i, value in enumerate(values):
        lo, hi = ANGLE_LIMITS[i]
        if not lo <= value <= hi:
            raise ValueError(f"{JOINT_NAMES[i]} must be in [{lo}, {hi}] degrees")
    return values


def validate_speeds(values):
    values = validate_vector(values, 5, "finger_speeds_percent")
    if any(not 0 <= v <= 100 for v in values):
        raise ValueError("finger_speeds_percent must be in [0, 100]")
    return values


def rs485_sdk(tty_baudrate=None, **kwargs):
    """The vendor RS-485 client with the CAN FD client's interface: it opens the port in its constructor and has no
    connect(); the baud rate follows the register map (v2 4 Mbaud, v1 500000) unless given."""
    sdk = load_sdk_module("d21_rs485").d21_rs485(**kwargs)
    sdk.connect = lambda: None
    return sdk


def resolve_serial_device(path):
    """The configured adapter path, or the one attached adapter of the same model when that exact unit is absent.

    /dev/serial/by-id names embed the unit's serial number; two CANable2s are swapped on the bench, so a config
    naming one unit should still open the other when it is the only CANable2 present."""
    if os.path.exists(path):
        return path
    directory, name = os.path.split(path)
    # usb-<Vendor>_<Model>_<serial>-if<N>: same model = same prefix before the serial, same interface suffix
    stem, _, interface = name.rpartition("-")
    model = stem.rpartition("_")[0]
    attached = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
    siblings = [f for f in attached if model and f.startswith(model + "_") and f.endswith("-" + interface)]
    if len(siblings) == 1:
        found = os.path.join(directory, siblings[0])
        print(f"[aac] {path} is not attached; using the one adapter of that model present: {found}")
        return found
    raise FileNotFoundError(f"{path} is not attached and {len(siblings)} other adapters of that model are "
                            f"(attached: {', '.join(attached) or 'none'}); plug it in or name one in the config")


def cleanup(actions, primary=None):
    """Attempt every cleanup; retain the primary traceback and print secondaries."""
    first = primary
    for action in actions:
        try:
            action()
        except BaseException as error:
            if first is None:
                first = error
            else:
                traceback.print_exception(type(error), error, error.__traceback__)
    if primary is None and first is not None:
        raise first.with_traceback(first.__traceback__)


class AACHand:
    BUS_RATE_KEYS = {"canfd": ("canfd_data_bitrate", "CAN FD"), "rs485": ("rs485_baudrate", "RS-485")}

    def __init__(self, serial_device, device_address, timeout_s, sdk_factory=None, register_map=None, bus="canfd"):
        if bus not in self.BUS_RATE_KEYS:
            raise ValueError(f"bus must be one of {sorted(self.BUS_RATE_KEYS)}")
        self.bus = bus
        if not isinstance(serial_device, str) or not serial_device.strip() or "REPLACE" in serial_device:
            raise ValueError("serial_device must name the actual CAN FD or RS-485 adapter serial port")
        if type(device_address) is not int or not 1 <= device_address <= 254:
            raise ValueError("device_address must be an integer in [1, 254]")
        self.timeout_s = positive_number(timeout_s, "connection_timeout_s")
        self.serial_device = serial_device
        self.device_address = device_address
        self.sdk_factory = sdk_factory
        # Firmware map: given explicitly (argument or D21_REGISTER_MAP) it is enforced; otherwise open() tries the
        # default map first and falls back to the other one when nothing answers at its CAN FD data rate.
        self.auto_register_map = not (register_map or os.environ.get(REGISTER_MAP_ENV, "").strip())
        self.set_register_map(register_map or default_register_map())
        self.sdk = None
        self.pipeline_ok = self.pipeline_fallbacks = 0

    def set_register_map(self, register_map):
        if register_map not in REGISTER_MAPS:
            raise ValueError(f"register_map must be one of {sorted(REGISTER_MAPS)}")
        self.register_map = register_map
        self.command_resolution_deg = command_resolution_deg(register_map)
        # Torque in mA (bench, v1: 1 register unit = 1 mA). v2 assumes torque full scale = current full scale,
        # 20000 = 1 A, so 20 units per mA. NOT verified on the new firmware: check with torque_probe.py after flashing.
        self.torque_units_per_ma = REGISTER_MAPS[register_map]["signed_zero"] / 1000.0
        # v1.1 manual table 3.5: thumb motors 1-4 accept only +-10000 counts on v2 (+-500 mA under that scale).
        self.thumb_torque_ma_max = REGISTER_MAPS[register_map]["thumb_torque_max"] / self.torque_units_per_ma

    def open(self):
        factory = self.sdk_factory or (rs485_sdk if self.bus == "rs485" else load_sdk_module("d21_canfd").d21_canfd)
        other = "v1" if self.register_map == "v2" else "v2"
        candidates = [self.register_map] + ([other] if self.auto_register_map else [])
        rate_key, bus_name = self.BUS_RATE_KEYS[self.bus]
        rates = " or ".join(f"{REGISTER_MAPS[m][rate_key] / 1e6:g} M" + ("bit/s" if self.bus == "canfd" else "baud")
                            for m in candidates)
        last_error = None
        if self.sdk_factory is None:
            self.serial_device = resolve_serial_device(self.serial_device)
        for register_map in candidates:
            self.set_register_map(register_map)
            self.sdk = factory(hand_id=self.device_address, modbus_port=self.serial_device,
                               timeout=self.timeout_s, tty_baudrate=10000000, register_map=register_map)
            try:
                self.sdk.connect()
                self.check_register_map()
                return self.read_positions()
            except TimeoutError as error:
                last_error = error
                cleanup([self.close], primary=error)
            except BaseException as error:
                cleanup([self.close], primary=error)
                raise
        if self.auto_register_map:
            hint = "check power, address and wiring"
        else:
            hint = (f"if the hand runs the other firmware, set {REGISTER_MAP_ENV}={other} (or unset it); "
                    f"otherwise check power, address and wiring")
        raise TimeoutError(f"D21 at {self.serial_device} did not answer over {bus_name} at {rates} "
                           f"({'/'.join(candidates)} firmware): {hint}.") from last_error

    def check_register_map(self):
        """Refuse a hand whose firmware does not match the register map: wrong scaling sends 20x wrong targets."""
        detect = getattr(self.sdk, "detect_register_map", None)
        if detect is None:
            return
        detected = detect()
        if detected in REGISTER_MAPS and detected != self.register_map:
            hint = (f"flash the new firmware, or run with {REGISTER_MAP_ENV}=v1 for the old one" if detected == "v1"
                    else f"unset {REGISTER_MAP_ENV} (the hand runs the new firmware)")
            raise RuntimeError(f"D21 at {self.serial_device} reports {detected} position registers but the code is "
                               f"set to {self.register_map}: {hint}")

    def set_position_mode(self):
        self.sdk.set_control_mode(1)

    def set_torque_mode(self):
        self.sdk.set_control_mode(2)
        mode = self.sdk.read_control_mode()
        if mode != 2:
            raise RuntimeError(f"control mode reads {mode} after requesting torque mode")

    # Torque register groups the firmware acts on. Old firmware (v1): one write of all 16 registers (68-83) is
    # stored but ignored, writes of up to 15 work (bench, 2026-10-23), so motors 1-15 then motor 16. New firmware
    # (v2): a single 16-register write is acted on (bench over RS-485, 2026-11-02), saving one ~2 ms round trip.
    TORQUE_GROUPS_BY_MAP = {"v1": ((1, 15), (16, 1)), "v2": ((1, 16),)}
    TORQUE_GROUPS = TORQUE_GROUPS_BY_MAP["v1"]

    def torque_registers(self, torques):
        """16 motor torques in mA -> the 16 torque register values (68-83), after the per-motor range check."""
        values = validate_vector(torques, 16, "motor_torque")
        for motor, value in enumerate(values, 1):
            limit = min(1000.0, self.thumb_torque_ma_max) if motor <= 4 else 1000.0
            if not -limit <= value <= limit:
                raise ValueError(f"motor_torque[{motor - 1}] (motor {motor}) must be in [-{limit:g}, {limit:g}] mA")
        zero = REGISTER_MAPS[self.register_map]["signed_zero"]
        return [int(round(v * self.torque_units_per_ma)) + zero for v in values]

    def write_motor_torques(self, torques):
        """16 motor torques in mA (v1: 1 register unit = 1 mA, bench-measured), -1000..1000."""
        regs = self.torque_registers(torques)
        for first, count in self.TORQUE_GROUPS_BY_MAP.get(self.register_map, self.TORQUE_GROUPS):
            self.sdk._write_multiple_holding(67 + first, regs[first - 1:first - 1 + count],
                                             f"torque motors {first}-{first + count - 1}")

    # RS-485 pipelining (bench 2026-11-02, v2): the hand queues a request that arrives while it is still working on
    # the previous one, so the torque write and the position read can share one ~1.8 ms reply delay. The second
    # request must follow the first by >= 0.5 ms (0.3 ms lost 1 % of replies, 0 ms merges the frames): 3000/3000
    # ticks OK, 2.76 ms per tick against 4.35 ms sequential.
    PIPELINE_GAP_S = 0.0005

    def can_pipeline(self):
        """Pipelined torque write + position read: the RS-485 client or the fast CAN FD transport, and firmware
        that takes one 16-register write."""
        return (self.sdk is not None and len(self.TORQUE_GROUPS_BY_MAP.get(self.register_map, self.TORQUE_GROUPS)) == 1
                and (self.bus == "rs485" and hasattr(self.sdk, "serial") or self.canfd_pipelined()))

    def canfd_pipelined(self):
        return self.bus == "canfd" and self.sdk is not None and getattr(self.sdk, "pipelined", False)

    PIPELINE_TIMEOUT_S = 0.02   # a lost CAN FD reply costs this before the sequential fallback
    # CAN FD: send the next request of an exchange before the previous reply is back (after SlcanFD's gap). Off by
    # default: frames with no gap wedged the hand's bridge (2026-11-04); set True only after a bench run that ends
    # with the hand still answering in order (hand/tests cannot check that).
    CANFD_OVERLAP = False

    def _pipelined_reads(self, requests, description):
        """Replies to back-to-back reads over the fast CAN FD transport, as register lists (one per request)."""
        replies = self.sdk.transact_many(requests, timeout=self.PIPELINE_TIMEOUT_S, overlap=self.CANFD_OVERLAP)
        out = []
        for req, rep in zip(requests, replies):
            data = self.sdk._parse_response(rep, req[1], description)
            count = int.from_bytes(req[4:6], "big")
            if len(data) != 2 * count:
                raise ValueError(f"{description}: expected {count} registers, got {len(data) // 2}")
            out.append([int.from_bytes(data[2 * i:2 * i + 2], "big") for i in range(count)])
        return out

    def _frame(self, pdu):
        crc = self.sdk._crc16(pdu)
        return pdu + bytes([crc & 0xFF, crc >> 8])

    def can_read_velocity(self):
        """The new firmware's velocity readback (input 16-31) is signed, so it can feed a damping term; v1's is not."""
        return self.register_map == "v2"

    def _decode_state(self, raw):
        positions = validate_vector([self.sdk._register_to_angle(i, r) for i, r in enumerate(raw[:16])], 16,
                                    "measured_angle")
        if len(raw) < 32:
            return positions, None
        scale, offset = REGISTER_MAPS[self.register_map]["velocity"]
        velocities = validate_vector([(r - offset) * scale for r in raw[16:32]], 16, "measured_velocity")
        return positions, velocities

    def read_state(self):
        """(positions deg, velocities deg/s) in one request: input 0-31. The reply delay barely depends on the size,
        so the velocities cost ~0.1 ms more than positions alone. Velocities are the firmware's own (1 kHz loop)."""
        if not self.can_read_velocity():
            return self.read_positions(), None
        if self.canfd_pipelined():
            try:        # two 16-register reads in one USB write (a 32-register reply would not fit a frame)
                slave = self.device_address
                a, b = self._pipelined_reads([self._frame(bytes([slave, 0x04, 0, 0, 0, 16])),
                                              self._frame(bytes([slave, 0x04, 0, 16, 0, 16]))], "positions + velocities")
                return self._decode_state(a + b)
            except (ValueError, TimeoutError, RuntimeError):
                self.sdk.device.drain()
        return self._decode_state(self.sdk._read_16bit_input(0, 32, "joint positions + velocities"))

    def read_positions_and_currents(self):
        """(positions deg, motor currents A) for the position-mode loops that need both (admittance, compliance).
        New firmware over RS-485: one request for input 0-47 (positions, velocities, currents), since the reply
        delay barely depends on its size: one ~2 ms round trip instead of two. Otherwise two requests."""
        zero = REGISTER_MAPS[self.register_map]["signed_zero"]
        if self.register_map == "v2" and self.bus == "rs485" and self.sdk is not None and hasattr(self.sdk, "serial"):
            raw = self.sdk._read_16bit_input(0, 48, "joint positions + velocities + motor currents")
            positions, _ = self._decode_state(raw[:16])
            currents = validate_vector([(r - zero) / float(zero) for r in raw[32:48]], 16, "motor_current")
            return positions, currents
        if self.register_map == "v2" and self.canfd_pipelined():
            try:        # positions and currents as two back-to-back reads
                slave = self.device_address
                pos, cur = self._pipelined_reads([self._frame(bytes([slave, 0x04, 0, 0, 0, 16])),
                                                  self._frame(bytes([slave, 0x04, 0, 32, 0, 16]))], "positions + currents")
                positions, _ = self._decode_state(pos)
                return positions, validate_vector([(r - zero) / float(zero) for r in cur], 16, "motor_current")
            except (ValueError, TimeoutError, RuntimeError):
                self.sdk.device.drain()
        return self.read_positions(), self.read_currents()

    def write_torques_and_read_positions(self, torques):
        """Write 16 motor torques (mA) and read the 16 joint positions in one pipelined RS-485 exchange.

        Returns (positions, pipelined). On any missing or malformed reply the line is drained and the exchange is
        redone sequentially (write, then read), so the caller always gets a fresh pose; pipelined is then False.
        Without pipelining support it is the sequential exchange."""
        positions, _, pipelined = self.write_torques_and_read_state(torques, velocities=False)
        return positions, pipelined

    def write_torques_and_read_state(self, torques, velocities=True):
        """write_torques_and_read_positions, also returning the firmware joint velocities (deg/s) when asked and
        supported: (positions, velocities or None, pipelined)."""
        velocities = velocities and self.can_read_velocity()
        count = 32 if velocities else 16
        regs = self.torque_registers(torques)
        if not self.can_pipeline():
            self.write_motor_torques(torques)
            if velocities:
                return (*self.read_state(), False)
            return self.read_positions(), None, False
        slave = self.device_address
        write = self._frame(bytes([slave, 0x10, 0, 68, 0, 16, 32]) + b"".join(r.to_bytes(2, "big") for r in regs))
        if self.canfd_pipelined():
            # One USB write carries the torque write and the read(s); the hand answers each as it arrives. Replies
            # must fit a 64-byte frame, so positions and velocities are two 16-register reads.
            reads = [self._frame(bytes([slave, 0x04, 0, 0, 0, 16]))]
            if velocities:
                reads.append(self._frame(bytes([slave, 0x04, 0, 16, 0, 16])))
            try:
                replies = self.sdk.transact_many([write] + reads, timeout=self.PIPELINE_TIMEOUT_S, overlap=self.CANFD_OVERLAP)
                self.sdk._parse_write_response(replies[0], write, "torque motors 1-16 (pipelined)")
                raw = []
                for req, rep in zip(reads, replies[1:]):
                    data = self.sdk._parse_response(rep, 0x04, "pipelined state read")
                    if len(data) != 32:
                        raise ValueError("pipelined state read: short reply")
                    raw.extend(int.from_bytes(data[2 * i:2 * i + 2], "big") for i in range(16))
                positions, vel = self._decode_state(raw)
                self.pipeline_ok += 1
                return positions, vel, True
            except (ValueError, TimeoutError, RuntimeError):
                self.pipeline_fallbacks += 1
                self.sdk.device.drain()
                self.sdk._write_multiple_holding(68, regs, "torque motors 1-16 (pipeline fallback)")
                if velocities:
                    return (*self.read_state(), False)
                return self.read_positions(), None, False
        read = self._frame(bytes([slave, 0x04, 0, 0, 0, count]))
        size = 5 + 2 * count
        port = self.sdk.serial
        try:
            port.reset_input_buffer()
            port.write(write)
            port.flush()
            until = time.perf_counter() + self.PIPELINE_GAP_S
            while time.perf_counter() < until:
                pass
            port.write(read)
            port.flush()
            echo = port.read(8)
            reply = port.read(size)
            if (len(echo) != 8 or echo[:6] != write[:6] or self._frame(echo[:6]) != echo
                    or len(reply) != size or reply[:3] != bytes([slave, 0x04, 2 * count])
                    or self._frame(reply[:size - 2]) != reply):
                raise ValueError("pipelined RS-485 exchange: missing or malformed reply")
            raw = [int.from_bytes(reply[3 + 2 * i:5 + 2 * i], "big") for i in range(count)]
            positions, vel = self._decode_state(raw)
            self.pipeline_ok += 1
            return positions, vel, True
        except (ValueError, TimeoutError):
            self.pipeline_fallbacks += 1
            time.sleep(0.005)              # let any late reply arrive, then drop it
            port.reset_input_buffer()
            self.sdk._write_multiple_holding(68, regs, "torque motors 1-16 (pipeline fallback)")
            if velocities:
                return (*self.read_state(), False)
            return self.read_positions(), None, False

    def set_speeds(self, speeds):
        speeds = validate_speeds(speeds)
        self.sdk.write_joint_speed_commands(speeds)

    def write_positions(self, positions):
        positions = validate_positions(positions)
        self.sdk.write_joint_position_commands(positions)

    def read_positions(self):
        return validate_vector(self.sdk.read_joint_positions(), 16, "measured_angle")

    def read_currents(self):
        # 16 motor currents in amps; motors, not joints (protocol spec section 3.5).
        return validate_vector(self.sdk.read_motor_currents(), 16, "motor_current")

    def check_health(self):
        # CAN FD: errors from the background receive thread surface here. RS-485 is synchronous: each request
        # raises its own error, so its client has nothing to poll.
        check = getattr(self.sdk, "check_receive_error", None)
        if check is not None:
            check()

    def close(self):
        if self.sdk is not None:
            sdk, self.sdk = self.sdk, None
            sdk.close()
