"""Auto-calibration of the TORQUE DEMO (demo/impedance_ellipses.py) for one hand. Nothing it writes is used by teleop.

Two measurements, both automatic, on the hand:

1. Breakaway currents where the demo works. Per joint and direction, with the hand at the demo's ellipse-centre pose
   (demo/parameters/ellipses_trajectory.yaml: the spread, the MCP flexion centre, the thumb centre), at the angles the ellipse visits
   (its low, centre and high point; a holding joint at its one angle): the joint's torque is raised STEP mA every
   HOLD s until the joint moves MOVE deg, other joints at zero torque (the method of friction_breakaway.py). Finger
   abductions are tested only a little around their spread and stop if they drift toward a neighbour. Results:
   demo/calibration/torque_demo_breakaway_<side>_fine.json (every measurement, angle-dependent) and
   demo/calibration/torque_demo_breakaway_<side>.json (per joint and direction means).

2. The demo's learned feed-forward: N learning passes of impedance_ellipses (its --learn), which record the current
   each joint needed along its ellipse and feed it forward on the next pass; saved to
   demo/calibration/impedance_feedforward_<side>.json, which later torque demo runs load frozen.

Between the two it writes the demo's config, demo/configure/hand_config_aac_<l|r>_torque_canfd_demo.json: the hand's teleop
torque config with torque.friction_table / breakaway_fine pointed at the files of step 1 (an existing demo config
keeps its own gains; only those two entries change). The GUI's Torque demo runs with that config when it exists; the
teleop config, and so Start the motor, are never touched.

    hand/.hand/bin/python demo/tools/calibrate_torque_demo.py --config teleop/configure/acc/torque_control/hand_config_aac_l_torque_canfd.json \\
        --hand_serial AAC-001                      # left hand, both steps (about 10 minutes)
    ... --skip-breakaway                           # only the learning passes
    ... --joints middle_mcp pinky_mcp              # breakaway on these joints only (others keep earlier values)

Ctrl-C (or SIGTERM) at any point: zero torque, position mode at the measured pose, an eased move back to rest, and
what was measured so far is kept. Stop the GUI's motor and demos first: only one process may own the adapter.
"""
import argparse
import json
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]   # agi-os (this file: demo/tools/)
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hand.backends.aac import AACHand, ANGLE_LIMITS, JOINT_NAMES  # noqa: E402
from teleop.hand_control.hand_controller_aac import joint_to_motor_torques, load_aac_config, pair_polarity  # noqa: E402
from finger_circles import REST, Streamer, TipKinematics  # noqa: E402
from mcp_ellipses import DEFAULT_PARAMS, build_plan, load_params  # noqa: E402

DEMO = ROOT / "demo"
STEP, HOLD, MOVE, START = 5.0, 0.5, 1.5, 20.0      # mA per step, s per step, deg of motion that counts, first mA
ABD_SPAN, ABD_MOVE, ABD_GUARD = 3.0, 1.0, 8.0     # finger abductions: test +-span around the spread, small motion
LIMIT_MARGIN_DEG = 6.0                             # no test toward a joint stop closer than this
FINGER_ABD = {"index_abduction", "middle_abduction", "ring_abduction", "pinky_abduction"}
FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def clip(pose):
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(pose, ANGLE_LIMITS.values())]


def side_of(settings):
    return "Right" if settings["hand_id"].lower().startswith("right") else "Left"


def demo_config_path(side):
    return DEMO / "configure" / f"hand_config_aac_{side[0].lower()}_torque_canfd_demo.json"


def breakaway_paths(side):
    return (DEMO / "calibration" / f"torque_demo_breakaway_{side.lower()}.json",
            DEMO / "calibration" / f"torque_demo_breakaway_{side.lower()}_fine.json")


def test_angles(plan, centre, fingers, samples=36):
    """joint index -> angles to test: the low, centre and high point of the joint's ellipse, or its one angle."""
    angles = {j: [centre[j]] for j in range(16)}
    for finger in fingers:
        values = {}
        for k in range(samples):
            for j, v in plan[finger][0](2 * 3.141592653589793 * k / samples):
                values.setdefault(j, []).append(v)
        for j, v in values.items():
            angles[j] = sorted({round(min(v), 1), round(centre[j], 1), round(max(v), 1)})
    return angles


class Calibrator:
    def __init__(self, hand, polarity, max_ma, log=print, sleep=time.sleep, clock=time.monotonic):
        self.hand, self.polarity, self.max_ma = hand, polarity, max_ma
        self.log, self.sleep, self.clock = log, sleep, clock

    def go_to(self, pose, timeout=4.0):
        self.hand.set_position_mode()
        self.hand.set_speeds([20] * 5)
        self.hand.write_positions(pose)
        t0 = self.clock()
        while self.clock() - t0 < timeout:
            self.sleep(0.15)
            p = self.hand.read_positions()
            if max(abs(a - b) for a, b in zip(p, pose)) < 1.5:
                return p
        return self.hand.read_positions()

    def hold_here(self):
        """Zero torque, position mode at the measured pose."""
        self.hand.write_motor_torques([0.0] * 16)
        self.hand.write_positions(clip(self.hand.read_positions()))
        self.hand.set_position_mode()

    def breakaway(self, j, direction, move, guard=None):
        """Raise joint j's torque until it moves `move` deg; (mA, start angle), mA None if it never moved."""
        pose, _ = self.hand.write_torques_and_read_positions([0.0] * 16)
        q0 = pose[j]
        ma = START
        while ma <= self.max_ma:
            tau = [0.0] * 16
            tau[j] = direction * ma
            command = joint_to_motor_torques(tau, self.max_ma, self.polarity)
            t0 = self.clock()
            while self.clock() - t0 < HOLD:
                pose, _ = self.hand.write_torques_and_read_positions(command)
                if guard and abs(pose[j] - guard[0]) > guard[1]:
                    self.hand.write_torques_and_read_positions([0.0] * 16)
                    self.log(f"    guard: {JOINT_NAMES[j]} drifted to {pose[j]:.1f} deg, stopped")
                    return None, q0
                if abs(pose[j] - q0) > move:
                    self.hand.write_torques_and_read_positions([0.0] * 16)
                    return ma, q0
            ma += STEP
        self.hand.write_torques_and_read_positions([0.0] * 16)
        return None, q0

    def measure(self, joints, centre, angles, results):
        """Breakaway of each joint, both directions, at its test angles, from the demo's centre pose."""
        for name in joints:
            j = JOINT_NAMES.index(name)
            lo, hi = ANGLE_LIMITS[j]
            finger_abd = name in FINGER_ABD
            results["measurements"] = [m for m in results["measurements"] if m["joint"] != name]   # remeasure
            tests = ([centre[j] - ABD_SPAN, centre[j], centre[j] + ABD_SPAN] if finger_abd else angles[j])
            for angle in tests:
                angle = min(max(angle, lo), hi)
                pose = list(centre)
                pose[j] = angle
                self.go_to(pose)
                self.hand.write_positions(clip(self.hand.read_positions()))
                self.hand.write_motor_torques([0.0] * 16)
                self.hand.set_torque_mode()
                for d in (+1, -1):
                    q = self.hand.read_positions()[j]
                    if (d > 0 and q > hi - LIMIT_MARGIN_DEG) or (d < 0 and q < lo + LIMIT_MARGIN_DEG):
                        continue
                    if finger_abd and ((d > 0 and q > centre[j] + ABD_SPAN + 1) or (d < 0 and q < centre[j] - ABD_SPAN - 1)):
                        continue
                    ma, q0 = self.breakaway(j, d, ABD_MOVE if finger_abd else MOVE,
                                            guard=(centre[j], ABD_GUARD) if finger_abd else None)
                    results["measurements"].append(dict(joint=name, direction="+" if d > 0 else "-",
                                                        angle_deg=round(q0, 1), breakaway_ma=ma))
                    self.log(f"  {name:16s} {'+' if d > 0 else '-'} at {q0:6.1f} deg: "
                             f"{ma if ma else '>' + str(int(self.max_ma))} mA")
                    self.hold_here()                 # back to the test angle for the other direction
                    self.go_to(pose)
                    self.hand.write_positions(clip(self.hand.read_positions()))
                    self.hand.write_motor_torques([0.0] * 16)
                    self.hand.set_torque_mode()
                self.hold_here()
            self.go_to(centre)
        return results


def tables(results):
    """(coarse {joint+/-: mean mA or None}, results with its 'table')."""
    grouped = {}
    for m in results["measurements"]:
        if m["breakaway_ma"]:
            grouped.setdefault(m["joint"] + m["direction"], []).append(m["breakaway_ma"])
    results["table"] = {k: round(sum(v) / len(v), 1) for k, v in grouped.items()}
    coarse = {n + d: results["table"].get(n + d) for n in JOINT_NAMES for d in "+-"}
    return coarse, results


def write_demo_config(teleop_config, side):
    """The demo's config: the teleop torque config (or the existing demo config, keeping its gains) with the
    friction tables pointed at the demo's measured breakaway files. Paths are relative to teleop/ (how the
    controller resolves them)."""
    path = demo_config_path(side)
    base = json.loads((path if path.exists() else Path(teleop_config)).read_text())
    coarse, fine = breakaway_paths(side)
    torque = base.setdefault("torque", {})
    torque["friction_table"] = f"demo/calibration/{coarse.name}"     # repo-relative (the controller looks there too)
    torque["breakaway_fine"] = f"demo/calibration/{fine.name}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, indent=2) + "\n")
    return path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(ROOT / "teleop" / "configure" / "acc" / "torque_control" / "hand_config_aac_r_torque_canfd.json"),
                   help="the hand's TELEOP torque config (read only; the demo config is derived from it)")
    p.add_argument("--hand_serial", default="AAC-002", help="hand_roles key (AAC-002 right, AAC-001 left)")
    p.add_argument("--params", default=str(DEFAULT_PARAMS), help="the demo's ellipses YAML")
    p.add_argument("--joints", nargs="+", choices=JOINT_NAMES, metavar="JOINT", default=list(JOINT_NAMES),
                   help="breakaway: only these joints (others keep earlier measurements)")
    p.add_argument("--skip-breakaway", action="store_true", help="only the learning passes")
    p.add_argument("--learn", type=int, default=5, help="learning passes of the torque demo (0 = none)")
    p.add_argument("--keep-feedforward", action="store_true",
                   help="continue the saved feed-forward table instead of learning from zero")
    p.add_argument("--max-ma", type=float, default=200.0, help="cap on the breakaway test current")
    p.add_argument("--settle", type=float, default=2.0, help="seconds of each position-mode easing move")
    p.add_argument("--no-tip-stop", action="store_true",
                   help="learning passes without the fingertip-distance safety stop (impedance_ellipses --no-tip-stop)")
    args = p.parse_args(argv)
    if args.learn < 0 or args.max_ma <= 0 or args.settle <= 0:
        p.error("--learn must not be negative; --max-ma and --settle must be positive")
    return args


def _terminate(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")


def run(args, hand=None, log=print, sleep=time.sleep, clock=time.monotonic, learner=None):
    """Both steps; returns a summary dict. `hand` and `learner` can be stand-ins (tests)."""
    settings = load_aac_config(args, args.config)
    side = side_of(settings)
    params = load_params(args.params)
    plan, centre = build_plan(TipKinematics(side), list(FINGERS), params)
    coarse_path, fine_path = breakaway_paths(side)
    summary = dict(side=side, breakaway=None, config=None, learning=None, outcome="done")
    log(f"[calibrate] {settings['hand_id']} ({args.hand_serial}): torque demo calibration")
    if not args.skip_breakaway:
        results = json.loads(fine_path.read_text()) if fine_path.exists() else {"measurements": []}
        if hand is None:
            hand = AACHand(settings["serial_device"], settings["device_address"], settings["connection_timeout_s"],
                           bus=settings["bus"])
        hand.open()
        calibrator = Calibrator(hand, pair_polarity(settings["hand_id"]), args.max_ma, log=log, sleep=sleep, clock=clock)
        streamer = Streamer(hand, 100.0, 20.0, read_every=5, sleep=sleep, clock=clock, log=log)
        log(f"[calibrate] breakaway of {len(args.joints)} joint(s) at the demo's ellipse poses "
            f"(up to {args.max_ma:g} mA, {STEP:g} mA every {HOLD:g} s)")
        try:
            streamer.start(args.settle)                          # rest, position mode
            streamer.move_to(centre, args.settle)
            calibrator.measure(args.joints, centre, test_angles(plan, centre, FINGERS), results)
        except KeyboardInterrupt:
            summary["outcome"] = "stopped (Ctrl-C) during the breakaway measurement"
        finally:
            try:
                calibrator.hold_here()
                streamer.pose = clip(hand.read_positions())
                streamer.finish(args.settle)
                hand.set_position_mode()
            finally:
                hand.close()
            coarse, results = tables(results)
            fine_path.parent.mkdir(parents=True, exist_ok=True)
            fine_path.write_text(json.dumps(results, indent=1) + "\n")
            coarse_path.write_text(json.dumps(coarse, indent=1) + "\n")
            measured = sum(1 for v in coarse.values() if v)
            log(f"[calibrate] breakaway saved: {coarse_path.name} ({measured} of 32 joint directions measured), "
                f"{fine_path.name} ({len(results['measurements'])} measurements)")
            summary["breakaway"] = dict(coarse=str(coarse_path), fine=str(fine_path), measured=measured)
        if summary["outcome"] != "done":
            log(f"[calibrate] {summary['outcome']}")
            return summary
    if not coarse_path.exists():
        log("[calibrate] no breakaway table for this hand yet: the demo config keeps the teleop friction tables")
    config = write_demo_config(args.config, side) if coarse_path.exists() else None
    summary["config"] = str(config) if config else None
    if config:
        log(f"[calibrate] demo config written: {config}")
    if args.learn > 0:
        import impedance_ellipses
        learn = learner or impedance_ellipses.run
        demo_args = ["--config", str(config or args.config), "--hand_serial", args.hand_serial, "--params", args.params,
                     "--learn", str(args.learn), "--loops", "1", "--settle", str(args.settle)]
        if not args.keep_feedforward:
            demo_args.append("--reset-feedforward")
        if getattr(args, "no_tip_stop", False):
            demo_args.append("--no-tip-stop")
        log(f"[calibrate] learning: {args.learn} learning pass(es) of the torque demo, then one check pass")
        errors = []

        def capture(line):
            log(line)
            if "tip error rms" in line:
                parts = line.split()
                errors.append((parts[0], float(parts[parts.index("rms") + 1])))
        passes, outcome = learn(impedance_ellipses.parse_args(demo_args), log=capture)
        per_pass = {}
        for finger, value in errors:
            per_pass.setdefault(finger, []).append(value)
        summary["learning"] = dict(passes=passes, outcome=outcome,
                                   tip_error_rms_mm={f: dict(first=v[0], last=v[-1]) for f, v in per_pass.items()})
        if per_pass:
            log("[calibrate] fingertip error rms, first pass -> last pass: "
                + ", ".join(f"{f} {v[0]:.1f} -> {v[-1]:.1f} mm" for f, v in per_pass.items()))
        if outcome != "done":
            summary["outcome"] = f"learning {outcome}"
    log(f"[calibrate] {summary['outcome']}")
    return summary


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    result = run(parse_args())
    sys.exit(0 if result["outcome"] == "done" else 1)
