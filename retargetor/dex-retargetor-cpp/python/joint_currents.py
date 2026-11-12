"""Put the hand in position mode and print the current of each joint, continuously.

The hand reports one current per MOTOR (input registers 32-47). Joints map onto motors as in the
protocol spec, section 3.5:
  single-motor joints  thumb CMC (m3), thumb MCP (m4), each PIP (m7, m10, m13, m16): joint = that motor
  motor pairs          thumb (m1, m2), index (m5, m6), middle (m8, m9), ring (m11, m12), pinky (m14, m15):
                       same-sign part  (a + b) / 2 -> flexion (thumb: abduction)
                       opposite part   (a - b) / 2 -> abduction (thumb: rotation)
Currents are printed in mA as read (sign as the firmware reports it; in torque mode a positive command
reads back negative). Stop hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/joint_currents.py
    hand/.hand/bin/python dex-retargetor-cpp/python/joint_currents.py --rate 20 --motors --log /tmp/currents.csv
"""
import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from hand.backends.aac import AACHand, JOINT_NAMES  # noqa: E402
from d21_torque import zero_torques  # noqa: E402

SHORT = ['th_abd', 'th_rot', 'th_cmc', 'th_mcp', 'in_abd', 'in_mcp', 'in_pip', 'mi_abd', 'mi_mcp', 'mi_pip',
         'ri_abd', 'ri_mcp', 'ri_pip', 'pi_abd', 'pi_mcp', 'pi_pip']
# (same-sign joint, opposite-sign joint, motor a, motor b), all 0-based
PAIRS = ((0, 1, 0, 1), (5, 4, 4, 5), (8, 7, 7, 8), (11, 10, 10, 11), (14, 13, 13, 14))
SINGLES = ((2, 2), (3, 3), (6, 6), (9, 9), (12, 12), (15, 15))  # (joint, motor)


def joint_currents(motor_ma):
    joint = [0.0] * 16
    for same, opposite, a, b in PAIRS:
        joint[same] = (motor_ma[a] + motor_ma[b]) / 2
        joint[opposite] = (motor_ma[a] - motor_ma[b]) / 2
    for j, m in SINGLES:
        joint[j] = motor_ma[m]
    return joint


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--rate', type=float, default=10.0, help='lines per second (max ~200)')
    parser.add_argument('--duration', type=float, default=0.0, help='seconds; 0 = until Ctrl+C')
    parser.add_argument('--motors', action='store_true', help='also print the 16 raw motor currents')
    parser.add_argument('--log', default=None, help='optional CSV of every sample')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    args = parser.parse_args()
    if not 0 < args.rate <= 200:
        parser.error('--rate must be in (0, 200]')

    role = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']
    hand = AACHand(role['serial_device'], role['device_address'], 0.1)
    hand.open()
    sdk = hand.sdk
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    log = open(args.log, 'w', newline='') if args.log else None
    writer = csv.writer(log) if log else None
    if writer:
        writer.writerow(['t'] + [f'joint_{n}_mA' for n in JOINT_NAMES] + [f'motor_{m}_mA' for m in range(1, 17)])
    try:
        mode = sdk.read_control_mode()
        if mode != 1:
            print(f'hand is in mode {mode}: zeroing torque and switching to position mode '
                  '(fingers move to their last position targets; keep the hand clear)')
            zero_torques(sdk)
        hand.set_position_mode()
        print(f'control mode: {sdk.read_control_mode()} (1 = position)')
        print('joint currents in mA (pairs split into flexion / abduction parts):')
        print('  t      ' + ' '.join(f'{n:>7}' for n in SHORT))
        period, start = 1.0 / args.rate, time.monotonic()
        next_tick = start
        while not stop and (args.duration <= 0 or time.monotonic() - start < args.duration):
            now = time.monotonic()
            motor = [1000.0 * a for a in hand.read_currents()]
            joint = joint_currents(motor)
            print(f'  {now - start:5.1f}  ' + ' '.join(f'{v:+7.0f}' for v in joint))
            if args.motors:
                print('   motors ' + ' '.join(f'm{m + 1}:{v:+.0f}' for m, v in enumerate(motor)))
            if writer:
                writer.writerow([f'{now - start:.4f}'] + [f'{v:.1f}' for v in joint] + [f'{v:.1f}' for v in motor])
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    finally:
        hand.close()
        if log:
            log.close()
            print(f'log: {args.log}')
        print('stopped; hand stays in position mode')


if __name__ == '__main__':
    main()
