"""Talk to the D21 with NO vendor code: raw serial -> slcan text -> CAN FD -> Modbus RTU, all by hand.

Nothing from hand/aac/Python SDK or python-can is imported. The only library is pyserial, used to
open the CANable's USB serial port. Every byte that goes out is built and CRC'd in this file and
printed, both as the Modbus frame and as the slcan line the CANable turns into a CAN FD frame:

  slcan line:  b 010 E <hex data> \\r      b = CAN FD with bit-rate switch, 010 = request ID,
                                            E = DLC code for 48 bytes, data padded with 00 after the CRC
  reply line:  b 020 <dlc> <hex data> \\r   020 = response ID

Test: torque mode (Example 9), then ONE torque write of --count registers starting at motor 1
(+--torque on motors 1-2, 0 on the rest), then motor 1-2 currents for --hold seconds, then zero
torque with a 15-register write plus a 1-register write. The hand stays in torque mode.
Stop hand.hand_controller first; nothing else may have the adapter open.

    hand/.hand/bin/python dex-retargetor-cpp/python/raw_canfd.py --count 16
    hand/.hand/bin/python dex-retargetor-cpp/python/raw_canfd.py --count 15
"""
import argparse
import json
import time
from pathlib import Path

import serial  # pyserial only

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
REQ_ID, RESP_ID = 0x010, 0x020
FD_LENGTHS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 20, 24, 32, 48, 64]  # CAN FD DLC code -> bytes


def crc16_modbus(data):
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0xA001 if crc & 1 else crc >> 1
    return crc


def with_crc(body):
    crc = crc16_modbus(body)
    return body + bytes([crc & 0xFF, crc >> 8])  # Modbus sends the CRC low byte first


def hx(data):
    return ' '.join(f'{b:02X}' for b in data)


class RawCanFd:
    def __init__(self, port, verbose=True):
        self.verbose = verbose
        self.ser = serial.Serial(port, baudrate=10000000, timeout=0.01, write_timeout=0.5)
        time.sleep(2.0)                      # the CANable needs a moment after the port opens
        self.ser.reset_input_buffer()
        for cmd in ('C', 'S8', 'Y2', 'O'):   # close, 1 Mbit/s nominal, 2 Mbit/s data, open
            self.ser.write(cmd.encode() + b'\r')
            time.sleep(0.05)
        self.ser.reset_input_buffer()
        self.buffer = b''

    def close(self):
        try:
            self.ser.write(b'C\r')
        finally:
            self.ser.close()

    def send(self, modbus):
        dlc = next(code for code, n in enumerate(FD_LENGTHS) if n >= len(modbus))
        padded = modbus + bytes(FD_LENGTHS[dlc] - len(modbus))
        line = f'b{REQ_ID:03X}{dlc:X}' + padded.hex().upper()
        if self.verbose:
            print(f'    -> modbus {hx(modbus)}')
            print(f'       slcan  {line}\\r')
        self.ser.write(line.encode() + b'\r')

    def receive(self, timeout=0.2):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.buffer += self.ser.read(256)
            while b'\r' in self.buffer:
                raw, self.buffer = self.buffer.split(b'\r', 1)
                line = raw.replace(b'\x07', b'').decode(errors='replace')
                if len(line) >= 5 and line[0] in 'bd' and int(line[1:4], 16) == RESP_ID:
                    n = FD_LENGTHS[int(line[4], 16)]
                    return bytes.fromhex(line[5:5 + 2 * n])
        raise TimeoutError('no reply from the hand')

    def transact(self, modbus, expect_len):
        self.send(modbus)
        reply = self.receive()[:expect_len]
        crc_ok = reply[-2:] == with_crc(reply[:-2])[-2:]
        if self.verbose:
            print(f'    <- modbus {hx(reply)}   CRC {"ok" if crc_ok else "BAD"}')
        if reply[1] & 0x80:
            raise RuntimeError(f'Modbus exception {reply[2]} for function {modbus[1]:#04x}')
        return reply


def write_multiple(bus, start, values):
    body = bytes([0x01, 0x10]) + start.to_bytes(2, 'big') + len(values).to_bytes(2, 'big') + bytes([2 * len(values)])
    body += b''.join(v.to_bytes(2, 'big') for v in values)
    return bus.transact(with_crc(body), 8)


def read_input(bus, start, count):
    reply = bus.transact(with_crc(bytes([0x01, 0x04]) + start.to_bytes(2, 'big') + count.to_bytes(2, 'big')),
                         5 + 2 * count)
    return [int.from_bytes(reply[3 + 2 * i:5 + 2 * i], 'big') for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--count', type=int, default=16, help='registers in the single torque write (1-16)')
    parser.add_argument('--torque', type=int, default=200, help='torque on motors 1-2 (1 unit = 1 mA)')
    parser.add_argument('--hold', type=float, default=3.0)
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    args = parser.parse_args()
    if not 1 <= args.count <= 16 or not -500 <= args.torque <= 500:
        parser.error('--count must be 1-16 and --torque within +-500')
    port = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']['serial_device']

    bus = RawCanFd(port)
    try:
        print('\nExample 9: torque mode (register 21 = 2)')
        bus.transact(bytes.fromhex('01 06 00 15 00 02 19 CF'), 8)

        values = [args.torque + 1000] * min(2, args.count) + [1000] * max(0, args.count - 2)
        print(f'\nONE write of {args.count} registers from register 68: motors 1-2 = {args.torque:+d}, others 0')
        write_multiple(bus, 68, values)

        print(f'\nmotor 1-2 currents (input registers 32-33) for {args.hold:.0f} s:')
        bus.verbose = False
        start = time.monotonic()
        while time.monotonic() - start < args.hold:
            raw = read_input(bus, 32, 2)
            amps = [(v - 1000) / 1000 for v in raw]
            print(f'  t={time.monotonic() - start:4.1f}s  m1 {amps[0]:+.3f} A  m2 {amps[1]:+.3f} A')
            time.sleep(0.25)
        bus.verbose = True
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        try:
            print('\nzero torque (15 registers, then 1):')
            bus.verbose = True
            write_multiple(bus, 68, [1000] * 15)
            write_multiple(bus, 83, [1000])
            print('hand left in torque mode at zero torque')
        finally:
            bus.close()


if __name__ == '__main__':
    main()
