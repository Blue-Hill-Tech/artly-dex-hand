"""Bench probe for the undocumented AAC D21 torque mode.

Answers, one experiment each:
  status   read-only: control mode, enable, positions, currents, torque registers
  free     torque mode with zero torque; push the fingers by hand to feel backdrivability
  step     +T / -T on one motor: which joint moves, which way, how much current
  map      `step` on all 16 motors: torque index -> joint mapping and sign table
  sweep    increasing torque on one motor: breakaway threshold and current-vs-register slope
  timeout  write a torque once, then stop writing: does the firmware keep pushing?
  switch   return to position mode with and without copying the measured pose first

Every command that enters torque mode writes zero torque first and, on exit, error or
Ctrl+C, writes zero torque, copies the measured pose into the position targets, restores
position mode, and reports how far the hand jumped. Stop hand.hand_controller first: only
one process may own the CAN adapter.

    hand/.hand/bin/python dex-retargetor-cpp/python/torque_probe.py status
    hand/.hand/bin/python dex-retargetor-cpp/python/torque_probe.py step --motor 6 --torque 30
"""
import argparse
import csv
import json
import signal
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
from hand.backends.aac import AACHand, ANGLE_LIMITS, JOINT_NAMES  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from d21_torque import write_torques, zero_torques  # noqa: E402

LIMITS = np.array([ANGLE_LIMITS[i] for i in range(16)], dtype=float)
MID_POSE = LIMITS.mean(axis=1)
FLAT_POSE = np.zeros(16)  # every joint at 0 deg: fingers straight, no abduction, thumb parked
# A small curl on every joint. Moving flat -> curl -> flat proves each motor drives: a dead motor
# keeps reporting its last value, so "already at flat" alone would pass without any motion.
CHECK_POSE = np.clip(np.full(16, 10.0), LIMITS[:, 0], LIMITS[:, 1])
# Finger abduction reads back with the opposite sign to the command (teleop telemetry: commanded +20
# -> measured -20), so a non-zero abduction target can never "arrive". Abduction shares its motor pair
# with MCP flexion, and the MCP curl already drives both motors, so leave abduction at 0.
FINGER_ABDUCTION = [4, 7, 10, 13]
CHECK_POSE[FINGER_ABDUCTION] = 0.0
# Joints each motor drives (spec section 3.5); pairs drive two joints differentially.
MOTOR_JOINTS = {1: [0, 1], 2: [0, 1], 3: [2], 4: [3], 5: [4, 5], 6: [4, 5], 7: [6], 8: [7, 8], 9: [7, 8],
                10: [9], 11: [10, 11], 12: [10, 11], 13: [12], 14: [13, 14], 15: [13, 14], 16: [15]}


def test_pose(motors):
    """Flat hand with only the joints under test at mid-range, so a push has room both ways.

    With every finger at mid-range the thumb presses on the index finger and preloads both;
    keeping everything else flat avoids that collision.
    """
    pose = FLAT_POSE.copy()
    for motor, _ in motors:
        for joint in MOTOR_JOINTS[motor]:
            pose[joint] = MID_POSE[joint]
    return pose


class Probe:
    def __init__(self, args):
        cfg = json.loads(Path(args.config).read_text())
        role = cfg['hand_roles'][args.hand_serial]
        self.hand = AACHand(role['serial_device'], role['device_address'], 0.1)
        self.hand.open()
        self.sdk = self.hand.sdk
        self.in_torque = False
        self.rows = []
        self.log = args.log
        self.check_after = False

    # ---------------------------------------------------------------- I/O
    def sample(self, phase, torques):
        q = self.sdk.read_joint_positions()
        current = self.sdk.read_motor_currents()
        row = dict(t=time.monotonic(), phase=phase, q=q, i=current, tau=list(torques))
        self.rows.append(row)
        return row

    def phase(self, label, torques, duration, refresh=True):
        """Hold `torques` for `duration` s while sampling; refresh=False writes only once."""
        torques = [int(v) for v in torques]
        if self.in_torque:
            write_torques(self.sdk, torques)
        rows, start = [], time.monotonic()
        while time.monotonic() - start < duration:
            if refresh and self.in_torque:
                write_torques(self.sdk, torques)
            rows.append(self.sample(label, torques))
        return rows

    # -------------------------------------------------------------- modes
    def go_to(self, pose, timeout=6.0):
        """Position-mode move at the configured finger speed. Returns the joints that did not arrive."""
        pose = np.clip(pose, LIMITS[:, 0], LIMITS[:, 1])
        self.sdk.write_joint_position_commands(pose.tolist())
        start = time.monotonic()
        while time.monotonic() - start < timeout:
            q = np.array(self.sdk.read_joint_positions())
            if np.abs(q - pose).max() < 2.0:
                return []
            time.sleep(0.05)
        return [f'{JOINT_NAMES[j]} ({q[j]:.1f} vs {pose[j]:.1f})' for j in np.flatnonzero(np.abs(q - pose) >= 2.0)]

    def verify_motion(self):
        """Flat -> 10 deg curl on every joint -> flat. Returns joints that failed to follow."""
        stuck = self.go_to(CHECK_POSE)
        return stuck + self.go_to(FLAT_POSE)

    def enter_torque(self):
        # Vendor order (spec examples 9-10, and every script that moved the hand): switch to torque
        # mode first, then write torques. Torque writes are only valid in mode 2.
        self.sdk.set_control_mode(2)
        self.in_torque = True
        mode = self.sdk.read_control_mode()
        if mode != 2:
            raise RuntimeError(f'control mode reads {mode} after requesting torque mode')
        zero_torques(self.sdk)
        print('  torque mode on (zero torque)')

    def restore(self, copy_pose=True, observe=1.0):
        """Zero torque, optionally copy measured pose into position targets, position mode; report jump."""
        zero_torques(self.sdk)
        before = np.array(self.sdk.read_joint_positions())
        if copy_pose:
            self.sdk.write_joint_position_commands(before.tolist())
        self.sdk.set_control_mode(1)
        self.in_torque = False
        rows = self.phase('restore' + ('' if copy_pose else '_nocopy'), [0] * 16, observe)
        q = np.array([r['q'] for r in rows])
        jump = np.abs(q - before).max(axis=0)
        j = int(np.argmax(jump))
        print(f'  position mode restored ({"pose copied" if copy_pose else "pose NOT copied"}); '
              f'max jump {jump[j]:.2f} deg on {JOINT_NAMES[j]}; mode reads {self.sdk.read_control_mode()}')
        return jump

    def leave_torque_mode(self):
        """Zero torque and position mode, retried; each step is attempted even if an earlier one fails."""
        for attempt in range(3):
            try:
                self.restore()
                return True
            except Exception as error:
                print(f'  restore attempt {attempt + 1} failed: {error}')
                for step in (lambda: zero_torques(self.sdk), lambda: self.sdk.set_control_mode(1)):
                    try:
                        step()
                    except Exception:
                        pass
                time.sleep(0.2)
        return False

    def close(self):
        # Cleanup must not be cut short: a second Ctrl+C here could leave the hand in torque mode.
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            restored = True
            if self.in_torque:
                print('  cleaning up (Ctrl+C is ignored until this finishes)')
                restored = self.leave_torque_mode()
            if self.check_after:
                # After torque-mode sessions motors have stopped driving even in position mode
                # (mode still reads 1) until a power cycle. Returning to flat proves every joint drives.
                stuck = None
                try:
                    print('  exit check: curling every joint 10 deg and back to flat in position mode')
                    stuck = self.verify_motion()
                except Exception as error:
                    print(f'  exit check could not run: {error}')
                try:
                    mode = self.sdk.read_control_mode()
                except Exception:
                    mode = 'unreadable'
                if restored and stuck == [] and mode == 1:
                    print('  exit check passed: position mode, hand flat, every joint follows commands')
                else:
                    print(f'\n  !!! hand NOT confirmed safe: control mode {mode}, '
                          f'{"restore failed, " if not restored else ""}'
                          f'joints not following: {stuck if stuck is not None else "unknown"}')
                    print('  !!! Power-cycle the hand (its supply, not just USB), then run poscheck.')
        finally:
            signal.signal(signal.SIGINT, previous)
            self.hand.close()
            if self.rows and self.log:
                with open(self.log, 'w', newline='') as f:
                    w = csv.writer(f)
                    w.writerow(['t', 'phase'] + [f'q_{n}' for n in JOINT_NAMES] +
                               [f'i_{k}' for k in range(1, 17)] + [f'tau_{k}' for k in range(1, 17)])
                    t0 = self.rows[0]['t']
                    for r in self.rows:
                        w.writerow([f"{r['t'] - t0:.4f}", r['phase']] + r['q'] + r['i'] + r['tau'])
                print(f'  log: {self.log} ({len(self.rows)} samples)')


def speeds_percent(sdk):
    # The vendor SDK writes percent*20 but reads back register/10, so read the registers directly.
    return [round(r / 20.0, 1) for r in sdk._read_16bit_holding(16, 5, 'finger speeds')]


def summary(rows, baseline=None):
    q = np.array([r['q'] for r in rows])
    i = np.array([r['i'] for r in rows])
    di = i.mean(axis=0) - (baseline if baseline is not None else 0)
    return q[-1] - q[0], di, len(rows) / max(rows[-1]['t'] - rows[0]['t'], 1e-9)


def parse_motors(spec):
    """'7' -> [(7, 1)]; '5,6' -> both +T (MCP flexion); '5,-6' -> opposite signs (abduction).

    Spec section 3.5: motor pairs 1-2, 5-6, 8-9, 11-12, 14-15 drive two joints differentially and
    must be commanded together; 3, 4, 7, 10, 13, 16 each drive a single joint.
    """
    motors = []
    for item in str(spec).split(','):
        value = int(item)
        if not 1 <= abs(value) <= 16:
            raise ValueError('motor numbers must be 1-16')
        motors.append((abs(value), 1 if value > 0 else -1))
    return motors


def one_motor(motors, torque):
    tau = [0] * 16
    for motor, sign in motors:
        tau[motor - 1] = sign * torque
    return tau


def report_step(label, rows, baseline):
    dq, di, hz = summary(rows, baseline)
    j, k = int(np.argmax(np.abs(dq))), int(np.argmax(np.abs(di)))
    moved = ', '.join(f'{JOINT_NAMES[n]} {dq[n]:+.1f}' for n in np.argsort(-np.abs(dq))[:3] if abs(dq[n]) >= 0.5)
    print(f'  {label:>10}: moved [{moved or "nothing >= 0.5 deg"}]   '
          f'current change max on index {k + 1} ({di[k]:+.3f} A)   {hz:.0f} Hz')
    return j, dq[j], k, di[k]


# ------------------------------------------------------------------ commands
def cmd_status(p, args):
    print(f'control mode {p.sdk.read_control_mode()} (1 position, 2 torque), hand enable {p.sdk.read_hand_enable()}')
    q, cur, tau = p.sdk.read_joint_positions(), p.sdk.read_motor_currents(), p.sdk.read_all_torques()
    for n in range(16):
        print(f'  {n:2d} {JOINT_NAMES[n]:<18} q {q[n]:7.2f} deg   i[{n + 1:2d}] {cur[n]:+.3f} A   torque reg[{n + 1:2d}] {tau[n]:+5d}')


def cmd_poscheck(p, args):
    """Position mode only: move one joint 10 deg and report target, commanded register and measured angle."""
    j = args.joint
    print(f'  control mode {p.sdk.read_control_mode()}, hand enable {p.sdk.read_hand_enable()}, '
          f'finger speeds {speeds_percent(p.sdk)} %')
    q = np.array(p.sdk.read_joint_positions())
    lo, hi = LIMITS[j]
    target = q.copy()
    target[j] = q[j] - 10 if q[j] - 10 >= lo else q[j] + 10
    target = np.clip(target, LIMITS[:, 0], LIMITS[:, 1])
    print(f'  {JOINT_NAMES[j]}: measured {q[j]:.2f} -> target {target[j]:.2f}')
    p.sdk.write_joint_position_commands(target.tolist())
    start = time.monotonic()
    while time.monotonic() - start < 3.0:
        time.sleep(0.5)
        print(f'  t={time.monotonic() - start:3.1f}s  commanded {p.sdk.read_commanded_joint_positions()[j]:7.2f}  '
              f'measured {p.sdk.read_joint_positions()[j]:7.2f}')


def cmd_trigger(p, args):
    """Isolate what breaks the motor drivers: one action, then the exit check tests position control.

    regs: write zero to the torque registers while in POSITION mode (no mode switch).
    mode: switch to torque mode and back, writing nothing else.
    """
    if args.variant == 'regs':
        print('  writing zero to torque registers 68-83 while in position mode')
        zero_torques(p.sdk)
        p.phase('after_regs', [0] * 16, 1.0)
    else:
        print('  switching to torque mode (no torque writes), holding 1 s, switching back')
        p.sdk.set_control_mode(2)
        p.in_torque = True
        print(f'  mode reads {p.sdk.read_control_mode()}')
        p.phase('mode2', [0] * 16, 1.0, refresh=False)
        p.sdk.set_control_mode(1)
        p.in_torque = False
        p.phase('mode1', [0] * 16, 1.0)
    rows = [r for r in p.rows]
    frozen = [k + 1 for k in range(16) if np.ptp(np.array([r['i'][k] for r in rows])) == 0]
    print(f'  current channels that did not change during the test: {frozen or "none"}')


def cmd_free(p, args):
    p.enter_torque()
    print(f'  zero torque for {args.duration:.0f} s: push the fingers by hand and feel the resistance')
    rows = p.phase('free', [0] * 16, args.duration)
    q = np.array([r['q'] for r in rows])
    span = q.max(axis=0) - q.min(axis=0)
    for n in np.argsort(-span)[:6]:
        print(f'  {JOINT_NAMES[n]:<18} range of motion seen {span[n]:6.1f} deg')


def run_step(p, motors, torque, duration, settle, one_way=False):
    label = ','.join(str(m * s) for m, s in motors)
    base = p.phase('baseline', [0] * 16, settle)
    _, baseline_i, _ = summary(base)
    plus = p.phase(f'm{label}+', one_motor(motors, torque), duration)
    p.phase('settle', [0] * 16, settle)
    if one_way:
        return report_step(f'+{torque}', plus, baseline_i), None
    minus = p.phase(f'm{label}-', one_motor(motors, -torque), duration)
    p.phase('settle', [0] * 16, settle)
    return report_step(f'+{torque}', plus, baseline_i), report_step(f'-{torque}', minus, baseline_i)


def cmd_step(p, args):
    p.enter_torque()
    direction = '+' if args.one_way else '+/-'
    print(f'  motors {args.motor}, torque {direction}{args.torque}, {args.duration} s')
    print('  positions do not update in torque mode; the jump reported on restore is the real motion')
    run_step(p, parse_motors(args.motor), args.torque, args.duration, args.settle, args.one_way)


def cmd_map(p, args):
    p.enter_torque()
    table = []
    for motor in range(1, 17):
        print(f'motor {motor:2d} (expected {JOINT_NAMES[motor - 1]})')
        (j, dq, k, di), _ = run_step(p, [(motor, 1)], args.torque, args.duration, args.settle)
        table.append((motor, j, dq, k, di))
    print('\nmotor -> joint moved by +T (sign), current index:')
    for motor, j, dq, k, di in table:
        flag = '' if j == motor - 1 and abs(dq) >= 0.5 else '   <-- CHECK'
        print(f'  {motor:2d} -> {JOINT_NAMES[j]:<18} {dq:+6.1f} deg   current idx {k + 1:2d} {di:+.3f} A{flag}')


def cmd_sweep(p, args):
    p.enter_torque()
    levels = [int(v) for v in args.levels.split(',')]
    base = p.phase('baseline', [0] * 16, args.settle)
    _, baseline_i, _ = summary(base)
    motors = parse_motors(args.motor)
    j = motors[0][0] - 1
    points = []
    for level in levels:
        for sign in (1, -1):
            rows = p.phase(f'm{args.motor}{sign * level:+d}', one_motor(motors, sign * level), args.duration)
            dq, di, _ = summary(rows, baseline_i)
            points.append((sign * level, dq[j], di[j]))
            print(f'  torque {sign * level:+5d}: {JOINT_NAMES[j]} moved {dq[j]:+6.2f} deg, current {di[j]:+.3f} A')
            p.phase('settle', [0] * 16, args.settle)
    moving = [abs(t) for t, dq, _ in points if abs(dq) >= 1.0]
    print(f'  breakaway: smallest |torque| that moved >= 1 deg: {min(moving) if moving else "none in range"}')
    t, _, i = np.array(points).T
    slope = np.polyfit(t, i, 1)[0] if len(t) > 1 else float('nan')
    print(f'  current per register unit: {slope * 1000:.3f} mA/unit (linear fit over all levels)')


def cmd_timeout(p, args):
    p.enter_torque()
    base = p.phase('baseline', [0] * 16, args.settle)
    _, baseline_i, _ = summary(base)
    motors = parse_motors(args.motor)
    k = motors[0][0] - 1
    print(f'  writing torque {args.torque} on motors {args.motor} ONCE, then only reading for {args.hold} s')
    rows = p.phase('hold_once', one_motor(motors, args.torque), args.hold, refresh=False)
    t0 = rows[0]['t']
    for mark in (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0):
        if mark > args.hold:
            break
        r = min(rows, key=lambda r: abs(r['t'] - t0 - mark))
        print(f'  t={mark:4.2f}s  {JOINT_NAMES[k]} {r["q"][k] - rows[0]["q"][k]:+6.2f} deg   '
              f'current {r["i"][k] - baseline_i[k]:+.3f} A')
    print(f'  torque register now reads {p.sdk.read_all_torques()[k]:+d}')
    print('  if current falls to ~0 while the torque register still holds the value -> firmware timeout;')
    print('  if current stays and the finger keeps moving -> NO timeout, the PC loop must zero it itself.')
    p.phase('settle', [0] * 16, args.settle)


def cmd_switch(p, args):
    for copy_pose in (True, False):
        p.enter_torque()
        print(f'  displacing motors {args.motor} with torque {args.torque} for {args.duration} s')
        p.phase('displace', one_motor(parse_motors(args.motor), args.torque), args.duration)
        p.phase('settle', [0] * 16, args.settle)
        p.restore(copy_pose=copy_pose)
        time.sleep(0.5)
    print('  a large jump without the copy means the firmware resumes the OLD position target;')
    print('  the copy (what the C++ loop does) should keep the jump near zero.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command',
                        choices=('status', 'poscheck', 'trigger', 'free', 'step', 'map', 'sweep', 'timeout', 'switch'))
    parser.add_argument('--variant', choices=('regs', 'mode'), default='regs',
                        help='trigger: regs = zero torque registers in position mode; mode = switch modes only')
    parser.add_argument('--joint', type=int, default=6, help='poscheck: joint index 0-15 (default 6 = index_pip)')
    parser.add_argument('--speed', type=float, default=20.0, help='finger speed %% for probe moves (default 20)')
    parser.add_argument('--config', default=str(ROOT / 'teleop' / 'configure' / 'acc' / 'torque_control' / 'hand_config_aac_l_torque_canfd.json'))
    parser.add_argument('--hand-serial', default='AAC-001')
    parser.add_argument('--motor', default='7',
                        help="motor(s): '7' single (index PIP), '5,6' pair same direction (index MCP flexion), "
                             "'5,-6' pair opposite (index abduction)")
    parser.add_argument('--one-way', action='store_true', help='step: push +T only, so the restore jump shows the motion')
    parser.add_argument('--torque', type=int, default=30)
    parser.add_argument('--levels', default='10,20,30,40,60,80,100')
    parser.add_argument('--duration', type=float, default=0.3, help='seconds per torque phase')
    parser.add_argument('--settle', type=float, default=0.5)
    parser.add_argument('--hold', type=float, default=2.0, help='timeout test observation time')
    parser.add_argument('--pose', choices=('flat', 'current'), default='flat',
                        help='flat (default): move the hand flat, then only the tested joints to mid-range; '
                             'current: start where it is')
    parser.add_argument('--max-torque', type=int, default=100, help='safety cap on |torque|')
    parser.add_argument('--log', default=None, help='CSV log path (default /tmp/aac_torque_<command>_<time>.csv)')
    parser.add_argument('--yes', action='store_true', help='skip the confirmation prompt')
    args = parser.parse_args()
    try:
        parse_motors(args.motor)
    except ValueError as error:
        parser.error(f'--motor: {error}')
    levels = [abs(int(v)) for v in args.levels.split(',')] if args.command == 'sweep' else [abs(args.torque)]
    if args.max_torque > 500:
        parser.error('--max-torque is limited to 500 (the vendor example uses 500)')
    if max(levels) > args.max_torque:
        parser.error(f'torque {max(levels)} exceeds --max-torque {args.max_torque}')
    # `free` applies zero torque, so its duration is only an observation window.
    if args.command == 'free':
        if args.duration > 60.0:
            parser.error('free --duration is capped at 60 s')
    elif args.duration > 1.0 or args.hold > 5.0:
        parser.error('--duration is capped at 1 s per torque phase and --hold at 5 s')
    if not 0 <= args.joint <= 15:
        parser.error('--joint must be 0-15')
    if not 1 <= args.speed <= 100:
        parser.error('--speed must be 1-100')
    if args.log is None and args.command not in ('status', 'poscheck'):
        args.log = f'/tmp/aac_torque_{args.command}_{time.strftime("%H%M%S")}.csv'

    if args.command not in ('status', 'poscheck') and not args.yes:
        print(f'{args.command}: this ENTERS TORQUE MODE and may move the hand. Keep it clear of objects,')
        print('fingers away from the joints, and a hand on Ctrl+C (which restores position mode).')
        if input('type yes to continue: ').strip() != 'yes':
            return 1
    p = Probe(args)
    try:
        if args.command != 'status':
            # A power cycle resets finger speed to 100%; keep probe moves gentle.
            p.sdk.write_joint_speed_commands([args.speed] * 5)
        if args.command in ('status', 'poscheck'):
            globals()[f'cmd_{args.command}'](p, args)
            return 0
        print('  start check: curling every joint 10 deg and back to flat in position mode')
        stuck = p.verify_motion()
        if stuck:
            print(f'  joints {stuck} did not follow position commands; not entering torque mode.')
            print('  Power-cycle the hand (its supply, not just USB), then run poscheck.')
            return 1
        poses = []
        if args.pose == 'flat':
            # map and free cover the whole hand, so they stay flat; single-target tests raise only their joints.
            if args.command in ('step', 'sweep', 'timeout', 'switch'):
                pose = test_pose(parse_motors(args.motor))
                raised = [JOINT_NAMES[j] for j in np.flatnonzero(pose != FLAT_POSE)]
                if raised:
                    poses.append((', '.join(raised) + ' to mid-range', pose))
        for label, pose in poses:
            print(f'  moving {label} in position mode')
            stuck = p.go_to(pose)
            if stuck:
                print(f'  joints {stuck} did not follow position commands; not entering torque mode.')
                print('  Power-cycle the hand (its supply, not just USB), then run poscheck.')
                return 1
        p.check_after = True
        globals()[f'cmd_{args.command}'](p, args)
        return 0
    except KeyboardInterrupt:
        print('\n  interrupted')
        return 130
    finally:
        p.close()


if __name__ == '__main__':
    sys.exit(main())
