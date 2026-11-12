"""Hold a pose compliantly in POSITION mode (no glove, no torque mode).

The hand holds its pose; push a finger and it gives way, release it and it returns.
Use this to see and tune the compliance on hardware before using it in teleop.
Stop hand.hand_controller first: only one process may own the CAN adapter.

    hand/.hand/bin/python dex-retargetor-cpp/python/compliance_hold.py
    hand/.hand/bin/python dex-retargetor-cpp/python/compliance_hold.py --pose mid --gain 250 --log /tmp/hold.csv
"""
import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import dexrt  # noqa: E402
from hand.backends.aac import AACHand, ANGLE_LIMITS, JOINT_NAMES  # noqa: E402

LIMITS = np.array([ANGLE_LIMITS[i] for i in range(16)], dtype=float)
# Fingers half-bent, thumb parked clear of the index finger, no spread.
MID_POSE = np.array([0, 0, 0, 0] + [0, 45, 50] * 4, dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    parser.add_argument('--hand-serial', default='AAC-001')
    parser.add_argument('--pose', choices=('current', 'mid'), default='mid', help='pose to hold')
    parser.add_argument('--rate', type=float, default=50.0, help='loop rate (the hand updates feedback at ~50 Hz)')
    parser.add_argument('--speed', type=float, default=30.0, help='finger speed %% (how fast it yields and returns)')
    parser.add_argument('--gain', type=float, default=None, help='deg/s per amp above idle (default 150)')
    parser.add_argument('--return-rate', type=float, default=None, help='1/s pull back to the pose (default 2)')
    parser.add_argument('--error-clamp', type=float, default=None, help='max command lead in deg (default 15)')
    parser.add_argument('--duration', type=float, default=0.0, help='seconds; 0 = until Ctrl+C')
    parser.add_argument('--log', default=None, help='optional CSV log')
    args = parser.parse_args()
    if not 1 <= args.speed <= 100 or not 5 <= args.rate <= 200:
        parser.error('--speed must be 1-100 and --rate 5-200')
    overrides = {k: v for k, v in (('gain_dps_per_a', args.gain), ('return_rate', args.return_rate),
                                   ('error_clamp_deg', args.error_clamp)) if v is not None}
    compliance = dexrt.Compliance(**overrides)
    signs = np.array(dexrt.default_compliance_config()['readback_sign'])

    role = json.loads(Path(args.config).read_text())['hand_roles'][args.hand_serial]
    hand = AACHand(role['serial_device'], role['device_address'], 0.1)
    hand.open()
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    log = open(args.log, 'w', newline='') if args.log else None
    writer = csv.writer(log) if log else None
    if writer:
        writer.writerow(['t'] + [f'{k}_{n}' for k in ('target', 'meas', 'cmd', 'offset', 'effort', 'source')
                                 for n in JOINT_NAMES] + [f'cur_{m}' for m in range(1, 17)])
    try:
        hand.set_position_mode()
        hand.set_speeds([args.speed] * 5)
        if args.pose == 'mid':
            print('moving to the hold pose')
            hand.write_positions(MID_POSE.tolist())
            time.sleep(2.0)
        # Command frame: abduction reads back mirrored, so convert the measured pose before holding it.
        target = np.clip(signs * np.array(hand.read_positions()), LIMITS[:, 0], LIMITS[:, 1])
        hand.write_positions(target.tolist())
        print('calibrating idle current (keep hands off for 1 s)')
        for _ in range(int(args.rate)):
            compliance.add_idle_sample(hand.read_currents())
            time.sleep(1.0 / args.rate)
        idle = compliance.finish_idle()
        print('idle current per motor (A):', np.round(idle, 3).tolist())
        # Learning wiggle: every joint moves 12 deg and back, slowly, three times, so the controller
        # sees each motor driving for long enough to learn its current sign. Keep hands off.
        print('learning motor current signs: slow wiggle of every joint, three times (keep hands off)')
        hand.set_speeds([15.0] * 5)
        up = np.clip(target + 12.0, LIMITS[:, 0], LIMITS[:, 1])
        for pose in (up, target) * 3:
            hand.write_positions(pose.tolist())
            end = time.monotonic() + 1.0
            while time.monotonic() < end:
                compliance.step(pose, hand.read_positions(), hand.read_currents(), time.monotonic())
                time.sleep(1.0 / args.rate)
        hand.set_speeds([args.speed] * 5)
        learned = compliance.drive_signs()
        known = [JOINT_NAMES[j] for j in range(16) if learned[j, 0] != 0]
        print(f'current sign learned for {len(known)}/16 joints; others fall back to displacement')
        if len(known) < 16:
            print('  not learned:', ', '.join(n for n in JOINT_NAMES if n not in known))
        compliance.reset()
        print('holding. Push a finger: it should give way, then return when released. Ctrl+C to stop.')

        period, start = 1.0 / args.rate, time.monotonic()
        next_tick = next_print = start
        last_written = target.copy()
        while not stop and (args.duration <= 0 or time.monotonic() - start < args.duration):
            now = time.monotonic()
            currents = hand.read_currents()
            out = compliance.step(target, hand.read_positions(), currents, now)
            cmd = out['q_cmd']
            if np.abs(cmd - last_written).max() >= 0.05:  # register resolution is 0.035-0.05 deg
                hand.write_positions(cmd.tolist())
                last_written = cmd
            if writer:
                row = np.concatenate([target, out['q_meas'], cmd, out['offset'], out['effort'], out['source'], currents])
                writer.writerow([f'{now - start:.4f}'] + [f'{v:.3f}' for v in row])
            if now >= next_print:
                moving = [f'{JOINT_NAMES[j]} {out["offset"][j]:+.1f}{"c" if out["source"][j] == 2 else "d" if out["source"][j] == 1 else ""}'
                          for j in np.argsort(-np.abs(out['offset']))[:3] if abs(out['offset'][j]) >= 0.5]
                print(f'  t={now - start:5.1f}s  yielding: {", ".join(moving) or "-"}')
                next_print += 1.0
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    finally:
        # Hold the pose the finger is actually at; stay in position mode.
        try:
            hand.write_positions(np.clip(signs * np.array(hand.read_positions()), LIMITS[:, 0], LIMITS[:, 1]).tolist())
        finally:
            hand.close()
            if log:
                log.close()
                print('log:', args.log)
        print('stopped; hand holds its current pose in position mode')


if __name__ == '__main__':
    main()
