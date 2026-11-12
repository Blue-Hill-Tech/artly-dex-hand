"""Torque mode only: switch to torque mode, push, print live feedback, stay in torque mode.

No position-mode moves before or after. On exit (end, error or Ctrl+C) all 16 torques are
set to zero and the hand is LEFT IN TORQUE MODE, so the fingers are free. The AAC controller
selects position mode again when it starts; or power-cycle the hand.

    hand/.hand/bin/python dex-retargetor-cpp/python/torque_only.py --motors 1,2 --torque 500 --duration 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
from hand.backends.aac import AACHand, JOINT_NAMES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d21_torque import write_torques, zero_torques  # noqa: E402

# Joints each motor drives (protocol spec section 3.5).
MOTOR_JOINTS = {1: [0, 1], 2: [0, 1], 3: [2], 4: [3], 5: [4, 5], 6: [4, 5], 7: [6], 8: [7, 8], 9: [7, 8],
                10: [9], 11: [10, 11], 12: [10, 11], 13: [12], 14: [13, 14], 15: [13, 14], 16: [15]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--motors', default='1,2', help="'7', '1,2' (same sign) or '5,-6' (opposite sign)")
    parser.add_argument('--torque', type=int, default=500)
    parser.add_argument('--duration', type=float, default=3.0, help='seconds to hold the torque')
    parser.add_argument('--max-torque', type=int, default=500)
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    parser.add_argument('--probe-feedback', action='store_true',
                        help='also print joint velocity and commanded-position registers, to see what updates in torque mode')
    parser.add_argument('--pos-read', action='store_true',
                        help="also read the first pushed joint's position with 16-, 3- and 1-register requests")
    args = parser.parse_args()

    motors = [(abs(int(m)), 1 if int(m) > 0 else -1) for m in args.motors.split(',')]
    if any(not 1 <= m <= 16 for m, _ in motors):
        parser.error('motors must be 1-16')
    if abs(args.torque) > args.max_torque or args.max_torque > 1000:
        parser.error(f'|torque| must be <= --max-torque ({args.max_torque}), which is at most 1000')
    if not 0 < args.duration <= 10:
        parser.error('--duration must be in (0, 10] s')
    tau = [0] * 16
    for m, s in motors:
        tau[m - 1] = s * args.torque
    joints = sorted({j for m, _ in motors for j in MOTOR_JOINTS[m]})

    role = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']
    hand = AACHand(role['serial_device'], role['device_address'], 0.1)
    hand.open()
    sdk = hand.sdk

    def show(label):
        q, i = sdk.read_joint_positions(), sdk.read_motor_currents()
        cur = '  '.join(f'm{m}: {i[m - 1]:+.3f} A' for m, _ in motors)
        pos = '  '.join(f'{JOINT_NAMES[j]}: {q[j]:6.2f}' for j in joints)
        line = f'  {label:<14} {cur}   |   {pos}'
        if args.probe_feedback:
            # Which feedback still updates in torque mode? Positions freeze; check the others.
            vel_raw = sdk._read_16bit_input(16, 16, 'velocities')          # spec: 0-2000 -> 0-400 deg/s, unsigned
            cmd_raw = sdk._read_16bit_holding(0, 16, 'commanded positions')  # position-mode targets (echo)
            line += '   | vel ' + ' '.join(f'{JOINT_NAMES[j][:8]}={vel_raw[j] / 5:.0f}' for j in joints)
            line += '  cmdpos ' + ' '.join(f'{sdk._register_to_angle(j, cmd_raw[j]):.1f}' for j in joints)
        if args.pos_read:
            # Does the read SIZE matter (as it did for torque writes)? Same joint, three request sizes:
            # 16 registers (SDK default), 3 registers (spec Example 6 style), 1 register.
            j = joints[0]
            raw16 = sdk._read_16bit_input(0, 16, 'pos 16')[j]
            raw3 = sdk._read_16bit_input(max(0, j - 1), 3, 'pos 3')[j - max(0, j - 1)]
            raw1 = sdk._read_16bit_input(j, 1, 'pos 1')[0]
            line += (f'   | {JOINT_NAMES[j]} raw: 16-reg read={raw16} 3-reg read={raw3} 1-reg read={raw1} '
                     f'(= {sdk._register_to_angle(j, raw1):.2f} deg)')
        print(line)

    try:
        print(f'mode before: {sdk.read_control_mode()}')
        sdk.set_control_mode(2)
        print(f'mode now: {sdk.read_control_mode()} (torque)')
        zero_torques(sdk)
        show('zero torque')
        time.sleep(0.5)
        show('zero torque')
        print(f'pushing {args.torque:+d} on motors {args.motors} for {args.duration:.1f} s')
        start = next_print = time.monotonic()
        next_write = start
        while time.monotonic() - start < args.duration:
            # Refresh the torque registers every 50 ms (the vendor example writes once); a tight
            # loop would flood the firmware with hundreds of identical writes.
            if time.monotonic() >= next_write:
                write_torques(sdk, tau, motors=[m for m, _ in motors])
                next_write += 0.05
            if time.monotonic() >= next_print:
                show(f't={time.monotonic() - start:4.1f}s')
                next_print += 0.25
            time.sleep(0.005)
        print(f'torque registers read back: {[sdk.read_all_torques()[m - 1] for m, _ in motors]}')
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        try:
            zero_torques(sdk)
            show('zero torque')
            print(f'all torques zero; hand left in mode {sdk.read_control_mode()} (2 = torque, fingers free)')
        finally:
            hand.close()


if __name__ == '__main__':
    main()
