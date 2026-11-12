"""Friction compensation, step 1 of 3 (run from the repo root; see friction_tune.py, friction_apply.py).

Fine breakaway measurement: per joint, per direction, at several angles along the joint's range.

For each joint: home the whole hand to teleop/calibration/acc/friction/aac_friction_home_left.json (position mode, slow), then move only this
joint (position mode) to each test angle, switch to torque mode, and raise the joint torque in STEP mA every HOLD s
until the joint moves MOVE deg. Both directions at each angle (skipped where the limit is closer than 6 deg).
Other joints: zero torque. After each joint the hand is homed again. Output: a JSON with every measurement and a
controller-compatible table (per joint and direction: mean over the angles) in teleop/calibration/acc/friction/aac_breakaway_left_fine.json.

    hand/.hand/bin/python friction_breakaway.py [--side right] thumb_abduction index_pip ...   (default: all 16 joints)
"""
import json
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
from hand.backends.aac import AACHand, ANGLE_LIMITS, JOINT_NAMES  # noqa: E402
from teleop.hand_control.hand_controller_aac import joint_to_motor_torques, load_aac_config, pair_polarity  # noqa: E402
from friction_side import SIDE  # noqa: E402  (--side left|right)

CONFIG = str(SIDE.config)
HOME = SIDE.home
OUT = SIDE.fine
POLARITY = pair_polarity(SIDE.role)
STEP, HOLD, MOVE, START = 5.0, 0.5, 1.5, 20.0
MAX = float(__import__('os').environ.get('AAC_BREAKAWAY_MAX_MA', 200.0))   # mA cap per joint test
FRACTIONS = (0.15, 0.5, 0.85)            # test angles along the joint's range
DENSE = {'thumb_rotation': (0.07, 0.25, 0.5, 0.75, 0.93)}   # joints whose friction changes fast along the range
# Finger abductions can collide with the neighbouring finger: test only +-ABD_SPAN deg around the HOME angle (the home
# pose has the fingers spread), move only ABD_MOVE deg, and stop if the joint drifts more than ABD_GUARD from home.
FINGER_ABD = {'index_abduction', 'middle_abduction', 'ring_abduction', 'pinky_abduction'}
ABD_SPAN, ABD_MOVE, ABD_GUARD = 4.0, 1.0, 8.0


def clip(p):
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(p, ANGLE_LIMITS.values())]


def home_pose():
    pose = json.loads(HOME.read_text())['pose_deg']
    return clip([float(pose[n]) for n in JOINT_NAMES])


def open_hand():
    s = load_aac_config(SimpleNamespace(hand_serial=SIDE.serial), CONFIG)
    h = AACHand(s['serial_device'], s['device_address'], s['connection_timeout_s'], bus=s['bus'])
    h.open()
    return h


def go_to(h, pose, timeout=4.0):
    h.set_position_mode()
    h.set_speeds([20] * 5)
    h.write_positions(pose)
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        time.sleep(0.15)
        p = h.read_positions()
        if max(abs(a - b) for a, b in zip(p, pose)) < 1.5:
            return p
    return h.read_positions()


def breakaway(h, j, direction, move=MOVE, guard=None):
    """Raise the joint torque until the joint moves `move` deg; returns (mA, start angle) or (None, start).
    guard = (home angle, max drift): stop with None if the joint gets further than that from home."""
    pose, _ = h.write_torques_and_read_positions([0.0] * 16)
    q0 = pose[j]
    ma = START
    while ma <= MAX:
        tau = [0.0] * 16
        tau[j] = direction * ma
        cmd = joint_to_motor_torques(tau, MAX, POLARITY)
        t0 = time.monotonic()
        while time.monotonic() - t0 < HOLD:
            pose, _ = h.write_torques_and_read_positions(cmd)
            if guard and abs(pose[j] - guard[0]) > guard[1]:
                h.write_torques_and_read_positions([0.0] * 16)
                print(f"    guard: {JOINT_NAMES[j]} drifted to {pose[j]:.1f} deg, stopped")
                return None, q0
            if abs(pose[j] - q0) > move:
                h.write_torques_and_read_positions([0.0] * 16)
                return ma, q0
        ma += STEP
    h.write_torques_and_read_positions([0.0] * 16)
    return None, q0


def main(joints):
    results = json.loads(OUT.read_text()) if OUT.exists() else {'measurements': []}
    for name in joints:
        j = JOINT_NAMES.index(name)
        lo, hi = ANGLE_LIMITS[j]
        h = open_hand()
        try:
            home = home_pose()
            go_to(h, home)
            results['measurements'] = [m for m in results['measurements'] if m['joint'] != name]   # remeasure
            finger_abd = name in FINGER_ABD
            if finger_abd:
                angles = [home[j] - ABD_SPAN, home[j], home[j] + ABD_SPAN]
            else:
                angles = [lo + frac * (hi - lo) for frac in DENSE.get(name, FRACTIONS)]
            for angle in angles:
                angle = min(max(angle, lo), hi)
                pose = list(home)
                pose[j] = angle
                go_to(h, pose)
                h.write_positions(clip(h.read_positions()))
                h.write_motor_torques([0.0] * 16)
                h.set_torque_mode()
                for d in (+1, -1):
                    q = h.read_positions()[j]
                    if (d > 0 and q > hi - 6) or (d < 0 and q < lo + 6):
                        continue
                    if finger_abd and ((d > 0 and q > home[j] + ABD_SPAN + 1) or (d < 0 and q < home[j] - ABD_SPAN - 1)):
                        continue
                    ma, q0 = breakaway(h, j, d, move=ABD_MOVE if finger_abd else MOVE,
                                       guard=(home[j], ABD_GUARD) if finger_abd else None)
                    results['measurements'].append(dict(joint=name, direction='+' if d > 0 else '-', angle_deg=round(q0, 1),
                                                        breakaway_ma=ma))
                    print(f"{name:16s} {'+' if d > 0 else '-'} at {q0:6.1f} deg: {ma if ma else '>' + str(int(MAX))} mA")
                    sys.stdout.flush()
                    # back to the test angle for the other direction, in position mode
                    h.write_motor_torques([0.0] * 16)
                    h.write_positions(clip(h.read_positions()))
                    h.set_position_mode()
                    go_to(h, pose)
                    h.write_positions(clip(h.read_positions()))
                    h.write_motor_torques([0.0] * 16)
                    h.set_torque_mode()
                h.write_motor_torques([0.0] * 16)
                h.write_positions(clip(h.read_positions()))
                h.set_position_mode()
            go_to(h, home)
        finally:
            try:
                h.write_motor_torques([0.0] * 16)
                h.write_positions(clip(h.read_positions()))
                h.set_position_mode()
            finally:
                h.close()
        # controller-compatible table: mean over the angles per joint and direction
        table = {}
        for m in results['measurements']:
            if m['breakaway_ma']:
                table.setdefault(m['joint'] + m['direction'], []).append(m['breakaway_ma'])
        results['table'] = {k: round(sum(v) / len(v), 1) for k, v in table.items()}
        OUT.write_text(json.dumps(results, indent=1) + '\n')
        if SIDE.side != 'left':
            # the controller's coarse friction_table for this side: the measured means, unmeasured -> null
            SIDE.coarse.write_text(json.dumps({n + d: results['table'].get(n + d) for n in JOINT_NAMES for d in '+-'}, indent=1) + '\n')
    print(f'saved {OUT}')



def _terminate(signum, frame):
    """A kill (e.g. a timeout) must still run the cleanup: leave torque mode at the measured pose."""
    raise KeyboardInterrupt(f"signal {signum}")


signal.signal(signal.SIGTERM, _terminate)

if __name__ == '__main__':
    main(sys.argv[1:] or list(JOINT_NAMES))
