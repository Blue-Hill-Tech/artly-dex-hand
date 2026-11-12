"""Find out why the spec example moved the thumb when our earlier torque tests did not.

The spec example differs from the earlier tests in two ways; each variant changes one of them:
  A  write --count torque registers (default all 16) in ONE frame, once
  B  write 2 registers REPEATEDLY every 50 ms    (earlier tests rewrote continuously; the example wrote once)
  C  write 2 registers once, as the example did  (control: should move again)
  R  write registers for motors --start .. --start+--count-1 in ONE frame (FC10, or FC06 with --single),
     pushing --torque on the motors in --push and 0 on the rest; prints the pushed motors' currents

Results (2026-10-23): B and C move the thumb; A acts on --count up to 15 and ignores 16 (registers 68-83).
R separates "any write touching register 83 (motor 16) is ignored" from "16-register writes are ignored":
    torque_isolate.py R --start 15 --count 2 --push 16          # registers 82-83, push pinky PIP
    torque_isolate.py R --start 16 --count 1 --push 16 --single # FC06 to register 83 alone
    torque_isolate.py R --start 13 --count 1 --push 13 --single # control: ring PIP, register 80

Result (2026-10-23): B and C move the thumb, A with 16 registers does not (stored, but not acted on).
`A --count N` finds the largest write the firmware acts on: try 4, 8, 12.

Each variant: torque mode on, apply --torque (default +200) to motors 1 and 2 for --hold seconds while printing
their currents, then zero torque with firmware-accepted writes. The hand is left in torque mode.
Stop hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/torque_isolate.py C --torque 300
    hand/.hand/bin/python dex-retargetor-cpp/python/torque_isolate.py A --count 4
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('variant', choices=('A', 'B', 'C', 'R'))
    parser.add_argument('--start', type=int, default=1, help='variant R: first motor (1-16) in the frame')
    parser.add_argument('--push', default='', help="variant R: motors to push, e.g. '16' or '15,16' (default: all in frame)")
    parser.add_argument('--single', action='store_true', help='variant R: use FC06 single-register writes')
    parser.add_argument('--torque', type=int, default=200)
    parser.add_argument('--hold', type=float, default=3.0)
    parser.add_argument('--count', type=int, default=16,
                        help='variant A: number of registers (from motor 1) in the single write; 2..16')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    args = parser.parse_args()
    if not 0 < abs(args.torque) <= 1000:
        parser.error('--torque must be in [-1000, 1000] and non-zero')
    if args.variant == 'R':
        if not 1 <= args.start <= 16 or not 1 <= args.count or args.start + args.count - 1 > 16:
            parser.error('variant R: motors --start .. --start+--count-1 must lie in 1..16')
        frame = list(range(args.start, args.start + args.count))
        pushed = [int(m) for m in args.push.split(',')] if args.push else frame
        if not set(pushed) <= set(frame):
            parser.error('--push motors must be inside the frame')
        if args.single and args.count != 1:
            parser.error('--single writes one register: use --count 1')
    elif not 2 <= args.count <= 16:
        parser.error('--count must be in [2, 16]')

    from d21_canfd import d21_canfd
    role = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']
    sdk = d21_canfd(hand_id=role['device_address'], modbus_port=role['serial_device'], timeout=0.1,
                    tty_baudrate=10000000)
    sdk.connect()

    def write_two(value):
        sdk._write_multiple_holding(68, [value + 1000, value + 1000], 'motors 1-2')

    def write_count(value, count):
        # One FC10 frame: motors 1-2 = value, motors 3..count = 0 (register 1000).
        sdk._write_multiple_holding(68, [value + 1000, value + 1000] + [1000] * (count - 2), f'motors 1-{count}')

    watch = pushed if args.variant == 'R' else [1, 2]

    def write_range(value):
        if args.single:
            sdk._write_single_holding(67 + args.start, value + 1000 if args.start in pushed else 1000,
                                      f'motor {args.start}')
        else:
            regs = [value + 1000 if m in pushed else 1000 for m in frame]
            sdk._write_multiple_holding(67 + args.start, regs, f'motors {frame[0]}-{frame[-1]}')

    def status(label):
        i = sdk.read_motor_currents()
        currents = '  '.join(f'm{m} {i[m - 1]:+.3f} A' for m in watch)
        if args.variant == 'R':
            print(f'  {label:<8} current {currents}')
        else:
            q = sdk.read_joint_positions()
            print(f'  {label:<8} current {currents} | thumb_abduction {q[0]:6.2f} deg')

    try:
        print(f'mode before: {sdk.read_control_mode()}')
        status('before')
        sdk.set_control_mode(2)
        print(f'mode now: {sdk.read_control_mode()}')
        if args.variant == 'A':
            print(f'A: writing {args.count} registers in one frame, once ({args.torque} on motors 1-2, 0 elsewhere)')
            write_count(args.torque, args.count)
        elif args.variant == 'R':
            kind = 'FC06 single write' if args.single else f'one FC10 frame of {args.count} register(s)'
            print(f'R: {kind}: registers {67 + frame[0]}-{67 + frame[-1]} (motors {frame[0]}-{frame[-1]}), '
                  f'{args.torque} on motors {pushed}')
            write_range(args.torque)
        elif args.variant == 'B':
            print(f'B: writing 2 registers ({args.torque}) every 50 ms')
        else:
            print(f'C: writing 2 registers ({args.torque}) once, like the spec example')
            write_two(args.torque)
        start = time.monotonic()
        next_status = next_write = start
        while time.monotonic() - start < args.hold:
            now = time.monotonic()
            if args.variant == 'B' and now >= next_write:
                write_two(args.torque)
                next_write += 0.05
            if now >= next_status:
                status(f't={now - start:4.1f}s')
                next_status += 0.5
            time.sleep(0.005)
        readback = sdk.read_all_torques()
        print('torque registers read back: ' + ', '.join(f'm{m}={readback[m - 1]}' for m in watch))
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        try:
            zero_torques(sdk)  # 2-register groups: a 16-register zero would be ignored
            status('zeroed')
            print(f'hand left in mode {sdk.read_control_mode()} (2 = torque); start the AAC controller or power-cycle')
        finally:
            sdk.close()


if __name__ == '__main__':
    main()
