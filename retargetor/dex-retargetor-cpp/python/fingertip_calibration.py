"""Calibrate torque command (mA) -> fingertip force (N), one finger at a time.

The finger is driven in torque mode against a rigid block; its own fingertip sensor (tactile Z,
0.1 N steps, 0-25.5 N) measures the contact force. Result: newtons at the fingertip per mA,
plus the friction offset, for the impedance controller.

Procedure (the script prompts for each step):
  1. position mode: the current pose is stored as the return pose
  2. fingertip touching nothing -> the finger's tactile sensor is zeroed (register 47)
  3. place a rigid block where the fingertip pad will press straight into it when the finger flexes
  4. torque mode: each --levels value is applied to the finger's motors for --hold s, then zero
     for --rest s; force and motor currents are sampled throughout
  5. torque zeroed, position mode restored (the finger returns to the stored pose)

Torques use the 15+1 register writes the firmware acts on (d21_torque.py). Stop
hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/fingertip_calibration.py index
    hand/.hand/bin/python dex-retargetor-cpp/python/fingertip_calibration.py middle --levels 100,200,300,400
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT / 'hand' / 'aac' / 'Python SDK'))
sys.path.insert(0, str(HERE))
from d21_torque import write_torques, zero_torques  # noqa: E402

# Finger -> tactile sensor index (register 47 / force group) and the motors that flex it
# (spec section 3.5: MCP flexion = both motors of the pair with the same sign; PIP = single motor).
FINGERS = {
    'thumb': dict(sensor=1, motors=[3, 4]),          # CMC + MCP flexion (abduction pair 1-2 left alone)
    'index': dict(sensor=2, motors=[5, 6, 7]),
    'middle': dict(sensor=3, motors=[8, 9, 10]),
    'ring': dict(sensor=4, motors=[11, 12, 13]),
    'pinky': dict(sensor=5, motors=[14, 15, 16]),
}
FORCE_MAX_N = 25.5


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('finger', choices=list(FINGERS))
    parser.add_argument('--levels', default='100,200,300', help='torque commands in mA, comma separated')
    parser.add_argument('--direction', type=int, choices=(1, -1), default=1,
                        help='sign that presses the fingertip into the block (default +1; use -1 if it moves away)')
    parser.add_argument('--motors', default='', help="override the finger's motors, e.g. '7' (PIP only)")
    parser.add_argument('--hold', type=float, default=1.5, help='seconds per level')
    parser.add_argument('--rest', type=float, default=1.0, help='seconds at zero torque between levels')
    parser.add_argument('--max-torque', type=int, default=500)
    parser.add_argument('--out', default=None, help='result JSON (default /tmp/fingertip_cal_<finger>_<time>.json)')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    args = parser.parse_args()

    finger = FINGERS[args.finger]
    motors = [int(m) for m in args.motors.split(',')] if args.motors else finger['motors']
    levels = [int(v) for v in args.levels.split(',')]
    if any(not 1 <= m <= 16 for m in motors):
        parser.error('motors must be 1-16')
    if any(not 0 < v <= args.max_torque for v in levels) or args.max_torque > 1000:
        parser.error(f'levels must be in (0, --max-torque={args.max_torque}], max 1000')
    if args.hold < 0.5 or args.rest < 0.3:
        parser.error('--hold must be >= 0.5 s and --rest >= 0.3 s')
    stamp = time.strftime('%H%M%S')
    out = Path(args.out or f'/tmp/fingertip_cal_{args.finger}_{stamp}.json')

    from d21_canfd import d21_canfd
    role = json.loads(Path(args.config).read_text())['hand_roles']['AAC-001']
    sdk = d21_canfd(hand_id=role['device_address'], modbus_port=role['serial_device'], timeout=0.1,
                    tty_baudrate=10000000)
    sdk.connect()
    group_start = 3 * (finger['sensor'] - 1)

    def force():
        raw = sdk._read_8bit_fast(group_start, 3, 'fingertip force')
        return [sdk._map_tactile_xy(raw[0]), sdk._map_tactile_xy(raw[1]), sdk._map_tactile_z(raw[2])]

    def torques(value):
        tau = [0] * 16
        for m in motors:
            tau[m - 1] = args.direction * value
        return tau

    samples = []  # (t, phase, level, fx, fy, fz, currents of the finger's motors...)
    in_torque = False
    start_pose = None
    t0 = time.monotonic()

    def sample(phase, level, duration, refresh=None):
        rows, end, next_write = [], time.monotonic() + duration, time.monotonic()
        while time.monotonic() < end:
            if refresh is not None and time.monotonic() >= next_write:
                write_torques(sdk, refresh, motors=motors)
                next_write += 0.05
            f = force()
            i = sdk.read_motor_currents()
            row = [time.monotonic() - t0, phase, level, *f, *[i[m - 1] for m in motors]]
            rows.append(row)
            samples.append(row)
        return rows

    try:
        mode = sdk.read_control_mode()
        if mode != 1:
            print(f'hand is in mode {mode}; zeroing torque and switching to position mode first '
                  '(fingers move to their last position targets)')
            zero_torques(sdk)
            sdk.set_control_mode(1)
            time.sleep(1.0)
        start_pose = sdk.read_joint_positions()
        sdk.write_joint_position_commands(start_pose)  # the pose to return to afterwards

        input(f'\n[1/3] Make sure the {args.finger} fingertip touches NOTHING, then press Enter to zero its sensor.')
        sdk.calibrate_sensor(finger['sensor'])
        time.sleep(1.0)
        base = np.array([force() for _ in range(20)])
        print(f'      sensor zeroed; unloaded reading Fz {base[:, 2].mean():.2f} N (noise {base[:, 2].std():.2f} N)')

        input(f'[2/3] Place a rigid block so the {args.finger} fingertip PAD presses straight into it when the finger '
              f'flexes (touching or a few mm away). Keep your hands clear. Press Enter to start.')
        print(f'[3/3] torque mode: motors {motors}, direction {args.direction:+d}, levels {levels} mA, '
              f'{args.hold:.1f} s each')
        # Vendor order (spec examples 9-10, and every script that moved the hand): switch to torque
        # mode FIRST, then write torques. Torque writes are only valid in mode 2.
        sdk.set_control_mode(2)
        in_torque = True
        mode = sdk.read_control_mode()
        print(f'      control mode reads {mode}')
        if mode != 2:
            raise RuntimeError(f'control mode reads {mode} after requesting torque mode')
        zero_torques(sdk)
        sample('zero', 0, args.rest)

        results = []
        for level in levels:
            rows = sample('push', level, args.hold, refresh=torques(level))
            zero_torques(sdk)
            settled = np.array([r[3:] for r in rows[len(rows) // 2:]], dtype=float)  # second half of the hold
            fz, fmag = settled[:, 2].mean(), np.linalg.norm(settled[:, :3], axis=1).mean()
            current = np.abs(settled[:, 3:]).mean(axis=0)
            results.append(dict(level_mA=level, force_z_N=float(fz), force_norm_N=float(fmag),
                                force_z_std_N=float(settled[:, 2].std()),
                                motor_current_A={str(m): float(c) for m, c in zip(motors, current)}))
            sat = '  SENSOR SATURATED' if settled[:, 2].max() >= FORCE_MAX_N - 0.1 else ''
            print(f'  {level:4d} mA -> Fz {fz:5.2f} N  (|F| {fmag:5.2f} N, sd {settled[:, 2].std():.2f}) | '
                  + '  '.join(f'm{m} {c:.3f} A' for m, c in zip(motors, current)) + sat)
            if level == levels[0] and fz < 0.2:
                print('  !! almost no force at the first level: the fingertip may be moving AWAY from the block '
                      '(try --direction -1), missing the block, or below the friction level (try higher --levels)')
            sample('rest', 0, args.rest)

        x = np.array([r['level_mA'] for r in results], dtype=float)
        y = np.array([r['force_z_N'] for r in results])
        slope, offset = (np.polyfit(x, y, 1) if len(x) >= 2 else (y[0] / x[0], 0.0))
        print(f'\nresult for {args.finger}: {slope * 1000:.2f} N per A at the fingertip '
              f'({slope:.4f} N/mA), offset {offset:+.2f} N'
              + (f' -> no force below ~{-offset / slope:.0f} mA (friction)' if slope > 0 and offset < 0 else ''))
        out.write_text(json.dumps(dict(finger=args.finger, motors=motors, direction=args.direction,
                                       newton_per_mA=float(slope), offset_N=float(offset), levels=results,
                                       note='fingertip normal force (tactile Z) vs torque command per motor; '
                                            'depends on finger pose and where the pad touches'), indent=2))
        print(f'saved {out}')
    except KeyboardInterrupt:
        print('\ninterrupted')
    finally:
        try:
            if in_torque:
                zero_torques(sdk)
                if start_pose is not None:
                    sdk.write_joint_position_commands(start_pose)
                sdk.set_control_mode(1)
                print('torque zeroed; position mode restored (the finger moves back to its start pose). '
                      'If any finger stops responding, power-cycle the hand.')
        finally:
            sdk.close()
            if samples:
                csv_path = out.with_suffix('.csv')
                with open(csv_path, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['t', 'phase', 'level_mA', 'fx_N', 'fy_N', 'fz_N'] + [f'current_m{m}_A' for m in motors])
                    w.writerows(samples)
                print(f'samples: {csv_path}')


if __name__ == '__main__':
    main()
