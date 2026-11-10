"""Calibrate the current-based admittance (teleop/hand_control/admittance.py): free-motion current and current sign.

Position mode. Each joint is swept slowly through its range with nothing touching the hand, up then
down, while position and motor current are sampled. Per joint it writes to the calibration YAML:
  current_sign    +1/-1 so that a positive joint current drives the joint positive (from the sign of
                  current x velocity while the servo moves the joint)
  i_free_coef_a   cubic fit of the signed joint current vs command-frame angle (deg), amps
  hysteresis_a    mean |I_up - I_down| over the range: a starting point for tau_db_nm (x kt_nm_per_a)
Joints not swept keep their previous entry. Stop hand.hand_controller first.

    hand/.hand/bin/python dex-retargetor-cpp/python/admittance_calibrate.py                 # all 16 joints
    hand/.hand/bin/python dex-retargetor-cpp/python/admittance_calibrate.py --joints index_pip index_mcp
    hand/.hand/bin/python dex-retargetor-cpp/python/admittance_calibrate.py --watch           # print tau_ext live
--watch loads teleop/calibration/acc/admittance/admittance.yaml (+ its calibration) and prints the external torque estimate while
you push each finger: validation step 1, closed push and open push must flip the sign.
"""
import argparse
import datetime
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
from hand.backends.aac import AACHand, ANGLE_LIMITS, JOINT_NAMES  # noqa: E402
from teleop.hand_control.admittance import DEFAULT_CONFIG, joint_currents, load_admittance  # noqa: E402
from teleop.hand_control.hand_controller_aac import readback_sign  # noqa: E402

SHORT = ['th_abd', 'th_rot', 'th_cmc', 'th_mcp', 'in_abd', 'in_mcp', 'in_pip', 'mi_abd', 'mi_mcp', 'mi_pip',
         'ri_abd', 'ri_mcp', 'ri_pip', 'pi_abd', 'pi_mcp', 'pi_pip']
DEFAULT_OUTPUT = ROOT / 'teleop' / 'calibration' / 'acc' / 'admittance' / 'admittance_calibration.yaml'
HOME_POSE = ROOT / 'teleop' / 'calibration' / 'acc' / 'friction' / 'aac_friction_home_left.json'   # the user's collision-free pose (fingers spread)
ABD_SPAN = 6.0     # finger abductions sweep only +-ABD_SPAN deg around the home angle: wider sweeps can hit a neighbour
FINGER_ABDUCTION = {4: 'index', 7: 'middle', 10: 'ring', 13: 'pinky'}  # joint -> finger
FINGER_FLEXION = {'index': (5, 6), 'middle': (8, 9), 'ring': (11, 12), 'pinky': (14, 15)}  # (mcp, pip)


def home_pose():
    """The saved home pose (teleop/calibration/acc/friction/aac_friction_home_left.json) if present, else flat."""
    if HOME_POSE.exists():
        pose = json.loads(HOME_POSE.read_text())['pose_deg']
        return [min(max(float(pose[n]), lo), hi) for n, (lo, hi) in zip(JOINT_NAMES, ANGLE_LIMITS.values())]
    return [0.0] * 16


def base_pose(j, curl_deg):
    """Pose the rest of the hand holds while joint j sweeps: the home pose, except that for a finger's abduction
    the other three fingers curl (mcp, pip = curl_deg, clipped to each joint's limit) when curl_deg > 0."""
    pose = home_pose()
    finger = FINGER_ABDUCTION.get(j)
    if finger is not None and curl_deg > 0:
        for other, (mcp, pip) in FINGER_FLEXION.items():
            if other != finger:
                pose[mcp] = min(curl_deg, ANGLE_LIMITS[mcp][1])
                pose[pip] = min(curl_deg, ANGLE_LIMITS[pip][1])
    return pose


def sweep_joint(hand, j, steps, hold_s, fraction, rate, stop, curl_deg, sign):
    """Sweep joint j up then down from its base pose; returns arrays (t, q_deg, I_joint_raw_a, dq_deg_s)."""
    lo, hi = ANGLE_LIMITS[j]
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * fraction
    if j in FINGER_ABDUCTION:     # small sweep around the home angle, kept inside the limits
        home = home_pose()[j]
        a, b = max(lo + 1.0, home - ABD_SPAN), min(hi - 1.0, home + ABD_SPAN)
        mid, half = (a + b) / 2, (b - a) / 2
    setpoints = np.linspace(mid - half, mid + half, steps)
    setpoints = np.concatenate([setpoints, setpoints[::-1]])
    base = base_pose(j, curl_deg)
    hand.write_positions(base)  # settle the rest of the hand before the first setpoint
    time.sleep(1.5)
    t, q, cur = [], [], []
    period = 1.0 / rate
    for target in setpoints:
        if stop:
            break
        pose = list(base)
        pose[j] = float(target)
        hand.write_positions(pose)
        end = time.monotonic() + hold_s
        while time.monotonic() < end and not stop:
            now = time.monotonic()
            raw, motor = hand.read_positions_and_currents()
            t.append(now)
            q.append(sign[j] * raw[j])
            cur.append(joint_currents(motor)[j])
            delay = period - (time.monotonic() - now)
            if delay > 0:
                time.sleep(delay)
    hand.write_positions(base)
    t, q, cur = (np.array(v, dtype=float) for v in (t, q, cur))
    dq = np.gradient(q, t) if len(t) > 2 else np.zeros_like(q)
    return t, q, cur, dq


def fit_joint(q, cur, dq, moving_dps=5.0):
    moving = np.abs(dq) > moving_dps
    drive = float(np.sum(cur[moving] * dq[moving])) if moving.any() else 0.0
    sign = -1 if drive < 0 else 1
    signed = sign * cur
    coef = np.polyfit(q, signed, 3) if len(q) >= 8 else np.array([0.0, 0.0, 0.0, float(np.mean(signed))])
    half = len(q) // 2
    up, down = (q[:half], signed[:half]), (q[half:], signed[half:])
    grid = np.linspace(q.min(), q.max(), 20)
    hysteresis = 0.0
    if half > 3 and len(q) - half > 3:
        i_up = np.interp(grid, *sorted_pair(up))
        i_down = np.interp(grid, *sorted_pair(down))
        hysteresis = float(np.mean(np.abs(i_up - i_down)))
    return sign, coef, hysteresis, int(moving.sum())


def sorted_pair(pair):
    q, i = pair
    order = np.argsort(q)
    return q[order], i[order]


def write_yaml(path, entries):
    import yaml
    existing = {}
    if path.exists():
        with open(path, encoding='utf-8') as f:
            existing = (yaml.safe_load(f) or {}).get('calibration', {}).get('joints', {}) or {}
    existing.update(entries)
    doc = {'calibration': {
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'frame': 'command-frame degrees, signed joint current in amps',
        'joints': {name: existing[name] for name in JOINT_NAMES if name in existing}}}
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# Written by retargetor/dex-retargetor-cpp/python/admittance_calibrate.py; used by teleop/calibration/acc/admittance/admittance.yaml.\n')
        yaml.safe_dump(doc, f, sort_keys=False)


def watch(hand, config, rate, stop, sign):
    ctl = load_admittance(config)
    if ctl is None:
        sys.exit(f'{config}: admittance is disabled')
    ctl.readback_sign = list(sign)
    if ctl.calibration.needs_idle:
        print('no full calibration: sampling idle current for 0.5 s, keep the hand still')
        ctl.set_idle_current([hand.read_currents() for _ in range(25)])
    print('external torque estimate (N m) per joint; push a finger closed then open, the sign must flip')
    print('  t      ' + ' '.join(f'{n:>7}' for n in SHORT))
    period, start = 1.0 / rate, time.monotonic()
    while not stop:
        now = time.monotonic()
        raw, motor = hand.read_positions_and_currents()
        _, tau, _ = ctl.external_torque(raw, motor, now)
        print(f'  {now - start:5.1f}  ' + ' '.join(f'{v:+7.3f}' for v in tau))
        delay = period - (time.monotonic() - now)
        if delay > 0:
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--joints', nargs='*', default=None, help='joint names to sweep (default: all)')
    parser.add_argument('--steps', type=int, default=40, help='setpoints per direction')
    parser.add_argument('--hold', type=float, default=0.5, help='seconds per setpoint')
    parser.add_argument('--fraction', type=float, default=0.9, help='part of the joint range to sweep')
    parser.add_argument('--speed', type=int, default=20, help='finger speed percent during the sweep')
    parser.add_argument('--curl', type=float, default=0.0,
                        help='deg of mcp/pip curl on the other fingers while a finger abduction sweeps (0 = flat)')
    parser.add_argument('--rate', type=float, default=50.0, help='sample rate, Hz')
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'position_control' / 'hand_config_aac_l_admittance_rs485.json'))
    parser.add_argument('--hand_serial', default='AAC-001')
    parser.add_argument('--watch', action='store_true', help='print tau_ext live instead of calibrating')
    parser.add_argument('--params', default=str(DEFAULT_CONFIG), help='admittance YAML for --watch')
    args = parser.parse_args()
    joints = list(JOINT_NAMES) if not args.joints else args.joints
    unknown = set(joints) - set(JOINT_NAMES)
    if unknown:
        parser.error(f'unknown joints: {", ".join(sorted(unknown))}')
    if not 0 < args.fraction <= 1:
        parser.error('--fraction must be in (0, 1]')

    role = json.loads(Path(args.config).read_text())['hand_roles'][args.hand_serial]
    hand = AACHand(role['serial_device'], role['device_address'], 0.1, bus=role.get('bus', 'canfd'))
    hand.open()
    rb_sign = readback_sign(role['role'], hand.register_map)
    print(f"{role['role']}, firmware map {hand.register_map}, bus {hand.bus}; readback sign {rb_sign}")
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    try:
        if hand.sdk.read_control_mode() != 1:
            print('hand is not in position mode: zeroing torque and switching (keep the hand clear)')
            hand.write_motor_torques([0.0] * 16)
            hand.write_positions([min(max(v, lo), hi) for v, (lo, hi) in zip(hand.read_positions(), ANGLE_LIMITS.values())])
        hand.set_position_mode()
        if args.watch:
            watch(hand, args.params, args.rate, stop, rb_sign)
            return
        hand.set_speeds([args.speed] * 5)
        print('moving to the home pose')
        hand.write_positions(home_pose())
        time.sleep(1.5)
        entries = {}
        for name in joints:
            if stop:
                break
            j = JOINT_NAMES.index(name)
            print(f'sweeping {name} ({2 * args.steps} setpoints, {2 * args.steps * args.hold:.0f} s); do not touch the hand')
            if j in FINGER_ABDUCTION and args.curl > 0:
                print(f'  other fingers curled to {args.curl:.0f} deg so the {FINGER_ABDUCTION[j]} cannot touch them')
            t, q, cur, dq = sweep_joint(hand, j, args.steps, args.hold, args.fraction, args.rate, stop, args.curl, rb_sign)
            if len(t) < 8:
                print(f'  {name}: too few samples, skipped')
                continue
            sign, coef, hysteresis, n_moving = fit_joint(q, cur, dq)
            entries[name] = {'current_sign': int(sign), 'i_free_coef_a': [float(c) for c in coef],
                             'hysteresis_a': round(hysteresis, 5),
                             'q_range_deg': [round(float(q.min()), 2), round(float(q.max()), 2)]}
            fit = np.polyval(coef, q)
            print(f'  sign {sign:+d} ({n_moving} moving samples), I_free {1000 * fit.min():+.0f}..{1000 * fit.max():+.0f} mA, '
                  f'hysteresis {1000 * hysteresis:.1f} mA, residual {1000 * np.std(sign * cur - fit):.1f} mA')
            if n_moving < 20:
                print('  warning: few moving samples, the sign is uncertain; sweep faster (--hold 0.2) or check the hand')
        if entries:
            write_yaml(Path(args.output), entries)
            print(f'wrote {len(entries)} joints to {args.output}')
    finally:
        try:
            hand.write_positions(home_pose())
            time.sleep(1.0)
        finally:
            hand.close()
        print('stopped; hand back at the home pose in position mode')


if __name__ == '__main__':
    main()
