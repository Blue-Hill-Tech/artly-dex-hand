"""Position mode with the spec's exact frames: Example 3 writes, Example 6 reads.

  Example 3 (FC 10): set addresses 0-1 (thumb abduction, rotation) to 500
                     request 01 10 00 00 00 02 04 01 F4 01 F4 B3 B6, response 01 10 00 00 00 02 41 C8
  Example 6 (FC 04): read actual positions of addresses 0-2
                     request 01 04 00 00 00 03 B0 0B

Register 500 = 25.0 deg thumb abduction, -17.5 deg thumb rotation (value/2000 of the joint range).
Sequence: make sure the hand is in position mode, read Example 6 once, send Example 3, then send
Example 6 every 0.1 s for --duration seconds and print the raw values and degrees while the thumb
moves. With --push N, keeps reading for N more seconds so you can push the thumb by hand and watch
the position loop hold it. Stop hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/position_examples.py
    hand/.hand/bin/python dex-retargetor-cpp/python/position_examples.py --duration 4 --push 10
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

EXAMPLE_3 = bytes.fromhex('01 10 00 00 00 02 04 01 F4 01 F4 B3 B6')
EXAMPLE_3_RESP = bytes.fromhex('01 10 00 00 00 02 41 C8')
EXAMPLE_6 = bytes.fromhex('01 04 00 00 00 03 B0 0B')
EXAMPLE_6_RESP_HEAD = bytes.fromhex('01 04 06')
JOINTS = ('thumb_abduction', 'thumb_rotation', 'thumb_cmc')


def hx(b):
    return ' '.join(f'{x:02X}' for x in b)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--duration', type=float, default=4.0, help='seconds to read after each move')
    parser.add_argument('--values', default='500',
                        help="register values to send one after another with the Example 3 frame layout, e.g. '200,1800,200' "
                             "(500 is the exact documented frame; 0-2000 = full joint range)")
    parser.add_argument('--push', type=float, default=0.0, help='extra seconds to read while you push the thumb')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    args = parser.parse_args()

    from d21_canfd import d21_canfd
    role = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']
    if role['device_address'] != 1:
        sys.exit('the spec frames are for slave ID 1')
    sdk = d21_canfd(hand_id=1, modbus_port=role['serial_device'], timeout=0.1, tty_baudrate=10000000)
    sdk.connect()

    def regs(resp):
        return [int.from_bytes(resp[3 + 2 * i:5 + 2 * i], 'big') for i in range(resp[2] // 2)]

    def read6():
        return regs(sdk._transact(EXAMPLE_6))

    def fmt(values):
        return ' '.join(f'{JOINTS[j][6:9]}={v:4d}({sdk._register_to_angle(j, v):5.1f})' for j, v in enumerate(values))

    def watch(label, seconds):
        start, last, changes = time.monotonic(), None, 0
        while time.monotonic() - start < seconds:
            act = read6()
            changes += last is not None and act != last
            last = act
            print(f'  {label} t={time.monotonic() - start:4.1f}s  Example 6 actual: {fmt(act)}')
            time.sleep(0.1)
        return last, changes

    try:
        mode = sdk.read_control_mode()
        print(f'control mode: {mode}')
        if mode != 1:
            print('  hand is in torque mode: zeroing torque and switching to position mode. The fingers move to'
                  ' their last position targets; keep the hand clear.')
            zero_torques(sdk)
            sdk.set_control_mode(1)
            time.sleep(1.5)
            print(f'  control mode now: {sdk.read_control_mode()}')

        print('\nExample 6 before the move')
        resp = sdk._transact(EXAMPLE_6)
        print(f'  request  : {hx(EXAMPLE_6)}')
        print(f'  response : {hx(resp[:11])}  header {"MATCH" if resp[:3] == EXAMPLE_6_RESP_HEAD else "DIFFERENT"}')
        print(f'  actual   : {fmt(regs(resp))}')

        for value in [int(v) for v in args.values.split(',')]:
            if not 0 <= value <= 2000:
                sys.exit(f'register value {value} outside 0-2000')
            # Example 3 layout: FC10, start 0, 2 registers, 4 data bytes, value twice, CRC. At 500 this is
            # byte for byte the documented frame; other values use the same layout with the vendor's CRC.
            body = bytes.fromhex('01 10 00 00 00 02 04') + value.to_bytes(2, 'big') * 2
            crc = sdk._crc16(body)
            frame = body + bytes([crc & 0xFF, crc >> 8])
            assert value != 500 or frame == EXAMPLE_3
            print(f'\nExample 3 layout: thumb abduction and rotation -> register {value} '
                  f'({sdk._register_to_angle(0, value):.1f} deg, {sdk._register_to_angle(1, value):.1f} deg)'
                  + (' [exact documented frame]' if value == 500 else ''))
            print(f'  request  : {hx(frame)}')
            resp = sdk._transact(frame, is_write=True)
            if value == 500:
                print(f'  response : {hx(resp[:8])}  {"MATCH" if resp[:8] == EXAMPLE_3_RESP else "DIFFERENT"} (documented {hx(EXAMPLE_3_RESP)})')
            else:
                print(f'  response : {hx(resp[:8])}  (echo of start/count: {"OK" if resp[:6] == frame[:6] else "DIFFERENT"})')
            print(f'  reading Example 6 for {args.duration:.0f} s while the thumb moves:')
            final, changes = watch('move', args.duration)
            print(f'  value changes: {changes}; final abd={final[0]} rot={final[1]} (target {value}, {value}) '
                  f'-> {"reached" if all(abs(v - value) <= 10 for v in final[:2]) else "NOT reached"}')
        if args.push > 0:
            print(f'\nPUSH THE THUMB BY HAND for {args.push:.0f} s (position mode: the loop should resist and the reading should follow):')
            _, changes = watch('push', args.push)
            print(f'  value changes while pushing: {changes}')
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        sdk.close()
        print('done; hand stays in position mode holding the last targets')


if __name__ == '__main__':
    main()
