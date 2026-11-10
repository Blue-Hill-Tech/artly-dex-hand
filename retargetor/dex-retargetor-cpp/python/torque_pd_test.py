"""Torque-mode PD test on a D21 without the glove: move chosen joints along a sine or step and report how they track.

Same law and motor mapping as the teleop controller (teleop/hand_control/hand_controller_aac.py), run over RS-485 with the pipelined
write+read exchange. Gentle by default: 120 mA torque cap, +-10 deg around the current pose at 0.5 Hz, one finger, and
an abort to zero torque if any joint moves faster than --max-speed. Position mode is always restored at the measured
pose, also on Ctrl-C or an error.

    # index finger, gentle defaults
    hand/.hand/bin/python dex-retargetor-cpp/python/torque_pd_test.py --joints index
    # all fingers (thumb held), your gains, 150 mA cap, 6 s
    hand/.hand/bin/python dex-retargetor-cpp/python/torque_pd_test.py --joints fingers --kp 12 --kd 0.3 --window 0.06 --tau-max 150 --seconds 6
    # one 20 deg step on the index PIP instead of a sine
    hand/.hand/bin/python dex-retargetor-cpp/python/torque_pd_test.py --joints index_pip --profile step --amp 20
    # constant torque on one motor (open loop), e.g. 60 mA on motor 7 for 1 s: shows the breakaway current
    hand/.hand/bin/python dex-retargetor-cpp/python/torque_pd_test.py --probe 7 60 --seconds 1

Joints: thumb_abduction thumb_rotation thumb_cmc thumb_mcp, then index/middle/ring/pinky _abduction _mcp _pip; groups:
thumb, index, middle, ring, pinky, fingers (all finger joints), all. The report per joint: mean and max |error| after the
first second, 95th-percentile speed, torque sign reversals per second, mean |motor torque|.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
from hand.backends.aac import ANGLE_LIMITS, AACHand, JOINT_NAMES  # noqa: E402
from teleop.hand_control.hand_controller_aac import joint_to_motor_torques, pair_polarity, pd_extras, readback_sign  # noqa: E402

DEFAULT_PORT = '/dev/serial/by-id/usb-wch.cn_USB_Single_Serial_BC4C8140ABCDE959-if00'
GROUPS = {
    'thumb': [0, 1, 2, 3], 'index': [4, 5, 6], 'middle': [7, 8, 9], 'ring': [10, 11, 12], 'pinky': [13, 14, 15],
    'fingers': list(range(4, 16)), 'all': list(range(16)),
}


def parse_joints(spec):
    joints = []
    for item in spec.split(','):
        item = item.strip()
        if item in GROUPS:
            joints += GROUPS[item]
        elif item in JOINT_NAMES:
            joints.append(JOINT_NAMES.index(item))
        else:
            raise SystemExit(f'unknown joint or group {item!r}; joints: {" ".join(JOINT_NAMES)}; groups: {" ".join(GROUPS)}')
    return sorted(set(joints))


def clip_pose(pose):
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(pose, ANGLE_LIMITS.values())]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--rs485', default=DEFAULT_PORT, metavar='PORT')
    p.add_argument('--hand-id', default='LeftHand', help='LeftHand or RightHand (readback sign on old firmware)')
    p.add_argument('--joints', default='index', help='joint names and/or groups, comma separated (default index)')
    p.add_argument('--profile', choices=('sine', 'step'), default='sine')
    p.add_argument('--amp', type=float, default=10.0, help='deg: sine amplitude, or step size (default 10)')
    p.add_argument('--freq', type=float, default=0.5, help='sine frequency in Hz (default 0.5)')
    p.add_argument('--seconds', type=float, default=5.0)
    p.add_argument('--kp', type=float, default=12.0, help='mA per deg')
    p.add_argument('--kd', type=float, default=0.3, help='mA per deg/s')
    p.add_argument('--window', type=float, default=0.06, help='velocity estimate window in s')
    p.add_argument('--deadband', type=float, default=1.0, help='deg')
    p.add_argument('--tau-max', type=float, default=120.0, help='mA per motor (default 120; the controller config uses 150)')
    p.add_argument('--friction', type=float, default=0.0, help='mA kick while a joint is stuck outside the deadband')
    p.add_argument('--friction-ramp', type=float, default=0.0,
                   help='deg: SMOOTH friction compensation fric*tanh(error/ramp) instead of the switched kick (0 = off)')
    p.add_argument('--friction-table', type=Path, help='JSON {joint_name+/-: breakaway mA} for per-joint friction')
    p.add_argument('--friction-scale', type=float, default=0.8, help='friction = scale * breakaway from the table')
    p.add_argument('--dither-ma', type=float, default=0.0, help='mA: alternating torque on every motor (0 = off)')
    p.add_argument('--dither-hz', type=float, default=50.0)
    p.add_argument('--ki', type=float, default=0.0, help='mA per deg*s: leaky integral')
    p.add_argument('--i-leak', type=float, default=2.0, help='1/s')
    p.add_argument('--i-max', type=float, default=60.0, help='mA')
    p.add_argument('--kd-fast', type=float, default=0.0, help='extra damping above --v-fast deg/s')
    p.add_argument('--v-fast', type=float, default=100.0)
    p.add_argument('--max-speed', type=float, default=400.0, help='deg/s: abort to zero torque above this (default 400)')
    p.add_argument('--probe', nargs=2, metavar=('MOTOR', 'MA'), type=float, help='open loop: constant MA on MOTOR (1-16)')
    p.add_argument('--log', type=Path, help='write every tick as JSON lines')
    args = p.parse_args()
    if args.tau_max <= 0 or args.tau_max > 300:
        raise SystemExit('--tau-max must be in (0, 300] mA for this test')
    joints = parse_joints(args.joints)
    fric = [args.friction] * 16
    if args.friction_table:
        table = json.loads(args.friction_table.read_text())
        for j, name in enumerate(JOINT_NAMES):
            vals = [table.get(name + s) for s in ('+', '-')]
            vals = [v for v in vals if v]
            fric[j] = args.friction_scale * (sum(vals) / len(vals)) if vals else args.friction_scale * 65.0

    hand = AACHand(args.rs485, 1, 0.1, bus='rs485')
    start_raw = hand.open()
    sign = readback_sign(args.hand_id, hand.register_map)
    to_cmd = lambda raw: [s * v for s, v in zip(sign, raw)]
    print(f'opened {hand.register_map} firmware over RS-485, pipelining {"on" if hand.can_pipeline() else "off"}; '
          f'tau_max {args.tau_max:.0f} mA, abort above {args.max_speed:.0f} deg/s')
    log = args.log.open('w') if args.log else None
    rec = []
    aborted = None
    try:
        hand.write_positions(clip_pose(start_raw))
        hand.write_motor_torques([0.0] * 16)
        hand.set_torque_mode()
        pose = to_cmd(hand.read_positions())
        base = list(pose)
        hist = {j: [] for j in range(16)}
        integral = [0.0] * 16
        t0 = time.monotonic()
        last_pose, last_t = list(pose), t0
        print(f'torque mode; moving {", ".join(JOINT_NAMES[j] for j in joints)} for {args.seconds:.0f} s (Ctrl-C to stop)')
        prev_tick = t0
        while time.monotonic() - t0 < args.seconds:
            now = time.monotonic()
            elapsed = now - t0
            dt_tick = min(now - prev_tick, 0.05)
            prev_tick = now
            if args.probe:
                motor, ma = int(args.probe[0]), float(args.probe[1])
                cmd = [0.0] * 16
                cmd[motor - 1] = ma
                target = list(base)
            else:
                target = list(base)
                for j in joints:
                    lo, hi = ANGLE_LIMITS[j]
                    if args.profile == 'sine':
                        target[j] = base[j] + args.amp * math.sin(2 * math.pi * args.freq * elapsed)
                    else:
                        target[j] = base[j] + (args.amp if (elapsed // 1.5) % 2 == 0 else 0.0)
                    target[j] = min(max(target[j], lo + 2), hi - 2)
                tau, err, vel = [0.0] * 16, [0.0] * 16, [0.0] * 16
                for j in range(16):
                    q = pose[j]
                    hist[j].append((now, q))
                    hist[j] = [h for h in hist[j] if now - h[0] <= args.window]
                    span = now - hist[j][0][0]
                    vel[j] = (q - hist[j][0][1]) / span if span > 0 else 0.0
                    e = target[j] - q
                    err[j] = e
                    e = 0.0 if abs(e) < args.deadband else e
                    tau[j] = args.kp * e - args.kd * vel[j]
                    if args.friction_ramp > 0 and fric[j]:
                        tau[j] += fric[j] * math.tanh(err[j] / args.friction_ramp)   # smooth, no switching
                    if args.ki > 0:
                        integral[j] += (args.ki * e - args.i_leak * integral[j]) * dt_tick
                        integral[j] = max(-args.i_max, min(args.i_max, integral[j]))
                        tau[j] += integral[j]
                if args.friction_ramp <= 0:
                    tau = pd_extras(tau, err, vel, friction_ma=args.friction, kd_fast=args.kd_fast, v_fast_dps=args.v_fast,
                                    deadband_deg=args.deadband)
                cmd = joint_to_motor_torques(tau, args.tau_max, pair_polarity(args.hand_id))
                if args.dither_ma > 0:
                    d = args.dither_ma * (1.0 if math.sin(2 * math.pi * args.dither_hz * elapsed) >= 0 else -1.0)
                    cmd = [max(-args.tau_max, min(args.tau_max, c + d)) for c in cmd]
            raw, _ = hand.write_torques_and_read_positions(cmd)
            pose = to_cmd(raw)
            dt = now - last_t
            speed = max(abs(a - b) / dt for a, b in zip(pose, last_pose)) if dt > 0.002 else 0.0   # one full tick
            last_pose, last_t = list(pose), now
            rec.append((elapsed, target, list(pose), list(cmd)))
            if log:
                log.write(json.dumps({'t': elapsed, 'target': target, 'measured': pose, 'motor_ma': cmd}) + '\n')
            if speed > args.max_speed:
                aborted = f'joint speed {speed:.0f} deg/s above {args.max_speed:.0f}'
                break
    except KeyboardInterrupt:
        aborted = 'Ctrl-C'
    finally:
        try:
            hand.write_motor_torques([0.0] * 16)
            end = hand.read_positions()
            hand.write_positions(clip_pose(end))
            hand.set_position_mode()
            print(f'restored position mode at the measured pose (control word {hand.sdk.read_control_mode()})')
        finally:
            hand.close()
            if log:
                log.close()
    if aborted:
        print(f'STOPPED: {aborted}')
    if len(rec) < 10:
        return
    t = np.array([r[0] for r in rec])
    T, M, C = (np.array([r[k] for r in rec]) for k in (1, 2, 3))
    keep = t > min(1.0, t[-1] / 2)
    v = np.abs(np.diff(M, axis=0) / np.diff(t)[:, None])
    print(f'{len(rec) / t[-1]:.0f} ticks/s, {hand.pipeline_fallbacks} sequential fallbacks')
    print(f'{"joint":16s} {"|err| mean":>10s} {"max":>6s} {"vel p95":>8s} {"rev/s":>6s} {"|tau| mA":>8s}')
    show = joints if not args.probe else [j for j in range(16) if np.ptp(M[:, j]) > 0.5]
    for j in show:
        e = T[keep, j] - M[keep, j]
        rev = np.sum(np.diff(np.sign(C[keep, j])) != 0) / max(t[keep][-1] - t[keep][0], 0.1)
        print(f'{JOINT_NAMES[j]:16s} {np.abs(e).mean():10.1f} {np.abs(e).max():6.1f} {np.percentile(v[keep[1:], j], 95):8.0f} '
              f'{rev:6.1f} {np.abs(C[keep, j]).mean():8.0f}')


if __name__ == '__main__':
    main()
