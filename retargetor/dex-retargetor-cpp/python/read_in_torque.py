"""Does any position register follow a finger that is moved BY HAND while in torque mode?

Uses the spec's exact frames:
  Example 6 (FC 04): read ACTUAL joint positions, addresses 0-2    request 01 04 00 00 00 03 B0 0B
  Example 9 (FC 06): torque mode on                                request 01 06 00 15 00 02 19 CF
plus, for comparison, the FC 03 read of the position COMMAND registers 0-2.

Sequence: send Example 6 in position mode, switch to torque mode with zero torque, then for
--duration seconds send Example 6 every 0.1 s and print the raw values and degrees. Push the
thumb by hand during that time. Afterwards the hand is left in torque mode (zero torque).

    hand/.hand/bin/python dex-retargetor-cpp/python/read_in_torque.py --duration 15
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT / 'hand' / 'aac' / 'Python SDK'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d21_torque import zero_torques  # noqa: E402

EXAMPLE_6 = bytes.fromhex('01 04 00 00 00 03 B0 0B')   # read input 0-2: ACTUAL positions (spec Example 6)
EXAMPLE_6_RESP_HEAD = bytes.fromhex('01 04 06')        # documented reply header: 6 data bytes follow
EXAMPLE_9 = bytes.fromhex('01 06 00 15 00 02 19 CF')   # control mode = 2 (spec Example 9)
COMMAND_0_2 = bytes.fromhex('01 03 00 00 00 03')       # read holding 0-2: position COMMANDS, CRC appended
JOINTS = ('thumb_abduction', 'thumb_rotation', 'thumb_cmc')


def hx(b):
    return ' '.join(f'{x:02X}' for x in b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--duration', type=float, default=15.0)
    parser.add_argument('--torque', type=int, default=0,
                        help='torque on motors 1-2 (thumb abduction pair) in register units (1 = 1 mA); 0 = none, push by hand')
    parser.add_argument('--torque-at', type=float, default=1.0, help='seconds into the window when the torque is applied')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    args = parser.parse_args()
    if not -500 <= args.torque <= 500:
        parser.error('--torque must be within +-500 for this test')

    from d21_canfd import d21_canfd
    role = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']
    if role['device_address'] != 1:
        sys.exit('the spec frames are for slave ID 1')
    sdk = d21_canfd(hand_id=1, modbus_port=role['serial_device'], timeout=0.1, tty_baudrate=10000000)
    sdk.connect()
    crc = sdk._crc16(COMMAND_0_2)
    command_req = COMMAND_0_2 + bytes([crc & 0xFF, crc >> 8])

    def regs(resp):
        n = resp[2]
        return [int.from_bytes(resp[3 + 2 * i:5 + 2 * i], 'big') for i in range(n // 2)]

    def read_both():
        act = regs(sdk._transact(EXAMPLE_6))      # the spec's exact actual-position request
        cmd = regs(sdk._transact(command_req))
        return cmd, act

    def fmt(values):
        return ' '.join(f'{JOINTS[j][6:9]}={v:4d}({sdk._register_to_angle(j, v):5.1f})' for j, v in enumerate(values))

    try:
        print(f'control mode: {sdk.read_control_mode()}')
        print('\nExample 9: torque mode')
        print(f'  request  : {hx(EXAMPLE_9)}')
        resp = sdk._transact(EXAMPLE_9, is_write=True)
        print(f'  response : {hx(resp[:8])}  {"MATCH" if resp[:8] == EXAMPLE_9 else "DIFFERENT"}; mode reads {sdk.read_control_mode()}')
        print('\nExample 6 (now in torque mode)')
        print(f'  request  : {hx(EXAMPLE_6)}')
        resp = sdk._transact(EXAMPLE_6)
        print(f'  response : {hx(resp[:11])}  header {"MATCH" if resp[:3] == EXAMPLE_6_RESP_HEAD else "DIFFERENT"} '
              f'(documented 01 04 06 + 6 data bytes)')
        cmd, act = read_both()
        print(f'  Example 6 actual positions: {fmt(act)}')
        print(f'  FC03 command positions    : {fmt(cmd)}')

        zero_torques(sdk)
        if args.torque:
            # Spec Example 10 layout: FC10 to registers 68-69 (motors 1-2, the thumb pair; same sign = abduction).
            body = bytes.fromhex('01 10 00 44 00 02 04') + (1000 + args.torque).to_bytes(2, 'big') * 2
            crc = sdk._crc16(body)
            torque_frame = body + bytes([crc & 0xFF, crc >> 8])
            print(f'\nzero torque for {args.torque_at:.1f} s, then {args.torque:+d} on motors 1-2 (thumb abduction pair), '
                  f'reading Example 6 for {args.duration:.0f} s in total; raw register values (degrees):')
        else:
            torque_frame = None
            print(f'\nzero torque. PUSH THE THUMB BY HAND for {args.duration:.0f} s; raw register values (degrees):')
        start = time.monotonic()
        last_cmd = last_act = None
        changes_cmd = changes_act = 0
        while time.monotonic() - start < args.duration:
            if torque_frame is not None and time.monotonic() - start >= args.torque_at:
                resp = sdk._transact(torque_frame, is_write=True)
                print(f'  >>> torque frame {hx(torque_frame)} -> reply {hx(resp[:8])}')
                torque_frame = None
            cmd, act = read_both()
            i = sdk.read_motor_currents()
            changes_cmd += last_cmd is not None and cmd != last_cmd
            changes_act += last_act is not None and act != last_act
            last_cmd, last_act = cmd, act
            print(f'  t={time.monotonic() - start:5.1f}s  Example 6 actual: {fmt(act)} | FC03 cmd: {fmt(cmd)} | m1 {i[0]:+.2f}A m2 {i[1]:+.2f}A')
            time.sleep(0.1)
        print(f'\nvalue changes during the push window: Example 6 actual registers {changes_act}, FC03 command registers {changes_cmd}')
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        try:
            zero_torques(sdk)
            print(f'hand left in mode {sdk.read_control_mode()} (2 = torque); start the AAC controller or power-cycle for position mode')
        finally:
            sdk.close()


if __name__ == '__main__':
    main()
