"""Demo: spread the fingers, then every fingertip draws its own circle, all at once, in position mode.

1. The hand eases from wherever it is to the rest pose (flexions open, abductions centred).
2. The four fingers ease to the SPREAD pose (index and pinky toward their side limits, middle and ring part way,
   --spread) and on to the start of their circles.
3. Every fingertip draws a circle of --radius mm in the plane of its finger (the plane it flexes in): MCP and PIP
   are solved from the hand's URDF (2-joint inverse kinematics on the fingertip) so the tip really goes round, at
   about half curl (--centre, default MCP 45 / PIP 55 deg; 20 mm fits between MCP 21..71 and PIP 22..78 deg). A small
   abduction swing (--side) is added in quadrature for depth. The thumb circumducts with thumb_abduction and
   thumb_rotation. Every finger has its own phase offset and direction (--pattern, or --phases/--spin), so they all
   move differently. The phase speeds up over the first half turn and slows over the last, so the fingers leave and
   return to their start points with zero speed.
4. The hand eases back to rest. Ctrl-C does the same from wherever it is.

Why the circle is in the finger's plane: sideways the fingers have almost no room. At rest their MCP housings sit
~1.5 mm apart and a finger abducting alone touches its neighbour at ~4 deg (URDF collision meshes, D21_Right; the
bench showed the same 2026-10-30), so with four fingers in the +-20 deg range only a few degrees of swing each are
possible, invisible at the tip. Flexion has no such limit.

Collision guard (URDF scan, 2026-11-05): neighbours swinging opposite ways come within spread gap - 2 * side of each
other; that must stay >= NEIGHBOUR_MIN_DEG (the default 10 deg gaps with +-3 deg keep 1.9 mm), and the outer fingers
must stay --margin inside the +-20 deg stops. parse_args refuses settings that break either rule. The thumb is 50+ mm
from the fingers along its loop (60/25) even with the index spread toward it.

The firmware's own position loop moves the motors (no torque mode); the script only streams targets, so the finger
speed is left at 100 % and the stream sets the pace: targets go out as fast as the bus answers (--rate 1000 asked;
~500-1000 Hz on CAN FD), positions are read back every --read-every targets.

    hand/.hand/bin/python demo/finger_circles.py                            # right hand, CAN FD, 2 turns
    hand/.hand/bin/python demo/finger_circles.py --loops 0                  # repeat until Ctrl-C
    hand/.hand/bin/python demo/finger_circles.py --pattern wave --turns 3 --period 2 --radius 15
    hand/.hand/bin/python demo/finger_circles.py --phases 0 0 90 180 270 --spin ccw cw cw ccw ccw
    hand/.hand/bin/python demo/finger_circles.py --config teleop/configure/acc/torque_control/hand_config_aac_l_torque_canfd.json \\
        --hand_serial AAC-001                                               # left hand

Only the hand_roles entry and connection_timeout_s of the config are used (the torque block is ignored).
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hand.aac.model import HandModel  # noqa: E402
from hand.backends.aac import AACHand, ANGLE_LIMITS, JOINT_NAMES  # noqa: E402
from teleop.hand_control.hand_controller_aac import load_aac_config  # noqa: E402
from teleop.manus.stream.hand_stream_aac import joint_map  # noqa: E402

DEFAULT_CONFIG = ROOT / "teleop" / "configure" / "acc" / "torque_control" / "hand_config_aac_r_torque_canfd.json"
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
FOUR = FINGERS[1:]
# Spread pose, deg of abduction for index, middle, ring, pinky (+ is toward the thumb on every finger): 10 deg
# between neighbours; with +-3 deg swings the outer fingers reach 18, 2 deg from their stops, and neighbours
# swinging opposite ways keep 4 deg (3 x 10 + 2 x 3 + 2 x 2 = the 40 deg range).
DEFAULT_SPREAD = (15.0, 5.0, -5.0, -15.0)
NEIGHBOUR_MIN_DEG = 4.0   # closest relative abduction allowed between neighbours once spread (URDF scan: 1.9 mm left)
DEFAULT_CENTRE = (45.0, 55.0)   # MCP, PIP deg at the circle's centre
# Per-finger (phase offset deg, direction) for thumb, index, middle, ring, pinky. Direction -1 runs the phase
# backwards, so the finger goes round the other way.
PATTERNS = {"same": ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1)),
            "alternate": ((90, 1), (0, 1), (0, -1), (0, 1), (0, -1)),      # neighbours go round opposite ways
            "wave": ((45, 1), (0, 1), (90, 1), (180, 1), (270, 1))}        # a ripple from index to pinky
SPIN = {"ccw": 1, "cw": -1}
# Rest: flexions open (lower limit), abductions and thumb rotation centred.
REST = [0.0 if lo < 0 else float(lo) for lo, hi in (ANGLE_LIMITS[i] for i in range(16))]
LIMITS = [ANGLE_LIMITS[i] for i in range(16)]


def ease(fraction):
    """Cosine ease: 0 -> 1 with zero speed at both ends."""
    return 0.5 - 0.5 * math.cos(math.pi * min(max(fraction, 0.0), 1.0))


def spread_pose(spread):
    pose = list(REST)
    for finger, angle in zip(FOUR, spread):
        pose[JOINT_NAMES.index(f"{finger}_abduction")] = float(angle)
    return pose


def check_spread(spread, side, margin_deg):
    """Refuse a spread + side swing that lets neighbours or the stops be hit."""
    for finger, angle in zip(FOUR, spread):
        lo, hi = LIMITS[JOINT_NAMES.index(f"{finger}_abduction")]
        if not (lo + margin_deg <= angle - side and angle + side <= hi - margin_deg):
            raise ValueError(f"{finger} spread {angle:g} +- {side:g} deg leaves the {margin_deg:g} deg margin to its "
                             f"[{lo}, {hi}] stops")
    for (a, pa), (b, pb) in zip(zip(FOUR, spread), zip(FOUR[1:], spread[1:])):
        if pa - pb - 2 * side < NEIGHBOUR_MIN_DEG:
            raise ValueError(f"{a} ({pa:g}) and {b} ({pb:g}) swinging +-{side:g} deg come within "
                             f"{pa - pb - 2 * side:g} deg of each other; keep {NEIGHBOUR_MIN_DEG:g} deg "
                             f"(spread them further apart or lower --side)")


class TipKinematics:
    """Fingertip positions (mm) from the hand's URDF, for one hand side."""

    def __init__(self, side):
        self.model = HandModel(side)
        self.map = joint_map(side)      # SDK degrees -> URDF joint degrees (sign per joint)

    def tip(self, finger, pose):
        values = dict(zip(JOINT_NAMES, pose))
        model = [values[sdk] * sign for _, sdk, sign in self.map]
        return self.model.transforms(model)[f"{finger}_tip"][:3, 3] * 1000.0


def tip_circle_table(kin, finger, base, centre_deg, radius_mm, margin_deg, samples=72):
    """(samples + 1) x 2 array of (MCP, PIP) degrees, one row per phase step of a full turn, whose fingertip follows
    a circle of radius_mm in the finger's flexion plane around the tip position at centre_deg, with the rest of the
    hand at `base`. Gauss-Newton on the URDF fingertip, continued from one step to the next. Raises ValueError when
    the circle is not reachable within the joint limits less the margin."""
    mi, pi = JOINT_NAMES.index(f"{finger}_mcp"), JOINT_NAMES.index(f"{finger}_pip")
    lo = np.array([LIMITS[mi][0], LIMITS[pi][0]], dtype=float)
    hi = np.array([LIMITS[mi][1], LIMITS[pi][1]], dtype=float)

    def tip(q):
        pose = list(base)
        pose[mi], pose[pi] = float(min(max(q[0], lo[0]), hi[0])), float(min(max(q[1], lo[1]), hi[1]))
        return kin.tip(finger, pose)

    def jacobian(q, h=0.5):
        t0 = tip(q)
        return np.stack([(tip(q + [h, 0]) - t0) / h, (tip(q + [0, h]) - t0) / h], axis=1), t0

    q0 = np.array(centre_deg, dtype=float)
    j0, centre = jacobian(q0)
    u = j0[:, 0] / np.linalg.norm(j0[:, 0])                 # the direction the tip moves when the MCP flexes
    normal = np.cross(j0[:, 0], j0[:, 1])
    normal /= np.linalg.norm(normal)                       # normal of the finger's plane
    v = np.cross(normal, u)
    rows, q = [], q0.copy()
    for k in range(samples + 1):
        phase = 2 * math.pi * k / samples
        target = centre + radius_mm * (math.cos(phase) * u + math.sin(phase) * v)
        for _ in range(40):
            jac, at = jacobian(q)
            residual = at - target
            if np.linalg.norm(residual) < 0.05:
                break
            step = np.linalg.solve(jac.T @ jac + 0.01 * np.eye(2), -jac.T @ residual)
            q = np.clip(q + np.clip(step, -8.0, 8.0), lo, hi)
        miss = np.linalg.norm(tip(q) - target)
        if miss > 0.5:
            raise ValueError(f"{finger}: a {radius_mm:g} mm circle around MCP {centre_deg[0]:g} / PIP "
                             f"{centre_deg[1]:g} deg is out of reach (misses by {miss:.1f} mm at phase "
                             f"{math.degrees(phase):.0f} deg); lower --radius or move --centre")
        rows.append(q.copy())
    table = np.array(rows)
    if np.abs(table[0] - table[-1]).max() > 0.5:
        raise ValueError(f"{finger}: the circle does not close (ambiguous kinematics); change --centre")
    table[-1] = table[0]
    for name, column, (jlo, jhi) in (("mcp", 0, LIMITS[mi]), ("pip", 1, LIMITS[pi])):
        if table[:, column].min() < jlo + margin_deg or table[:, column].max() > jhi - margin_deg:
            raise ValueError(f"{finger}_{name} needs {table[:, column].min():.0f}..{table[:, column].max():.0f} deg "
                             f"for a {radius_mm:g} mm circle, outside [{jlo + margin_deg:g}, {jhi - margin_deg:g}]; "
                             f"lower --radius or move --centre")
    return table


def interpolate(table, phase):
    """(MCP, PIP) at any phase (radians) from a tip_circle_table, linear between its rows."""
    n = len(table) - 1
    x = (phase / (2 * math.pi)) % 1.0 * n
    k = int(x)
    f = x - k
    return table[k] * (1 - f) + table[min(k + 1, n)] * f


def build_plan(kin, fingers, base, circle_centre, radius_mm, side_deg, thumb_radii, margin_deg, samples=72):
    """finger -> function(phase) giving [(joint index, degrees), ...] for that finger's loop."""
    plan = {}
    for finger in fingers:
        if finger == "thumb":
            ti, ri = JOINT_NAMES.index("thumb_abduction"), JOINT_NAMES.index("thumb_rotation")
            flex, rot = thumb_radii

            def thumb(phase, ti=ti, ri=ri, flex=flex, rot=rot):
                return [(ti, REST[ti] + flex * (1 - math.cos(phase)) / 2), (ri, REST[ri] + rot * math.sin(phase))]
            plan[finger] = thumb
        else:
            table = tip_circle_table(kin, finger, base, circle_centre, radius_mm, margin_deg, samples)
            mi, pi, ai = (JOINT_NAMES.index(f"{finger}_{j}") for j in ("mcp", "pip", "abduction"))

            def loop(phase, table=table, mi=mi, pi=pi, ai=ai, centre=base[ai]):
                mcp, pip = interpolate(table, phase)
                return [(mi, float(mcp)), (pi, float(pip)), (ai, centre + side_deg * math.sin(phase))]
            plan[finger] = loop
    return plan


def apply_plan(base, plan, phase, motion):
    """Pose: `base` with every planned finger at its own phase (offset deg, direction) of the common `phase`."""
    pose = list(base)
    for finger, joints in plan.items():
        offset, direction = motion.get(finger, (0.0, 1))
        for j, value in joints(direction * phase + math.radians(offset)):
            pose[j] = value
    return pose


def phase_profile(turns, period_s):
    """phase(t) over [0, turns * period_s]: 2 pi per period at full speed, with the speed eased from 0 over the first
    half turn and back to 0 over the last (cosine ramps), so the loops start and stop without a jerk."""
    duration = turns * period_s
    ramp_s = min(period_s / 2, duration / 2)

    def travelled(x):                       # integral of ease over [0, x], x in [0, 1]
        return 0.5 * x - math.sin(math.pi * x) / (2 * math.pi)

    total = duration - ramp_s                 # integral of the speed shape over the whole duration
    scale = 2 * math.pi * turns / total

    def phase(t):
        t = min(max(t, 0.0), duration)
        if t < ramp_s:
            s = ramp_s * travelled(t / ramp_s)
        elif t <= duration - ramp_s:
            s = 0.5 * ramp_s + (t - ramp_s)
        else:
            s = total - ramp_s * travelled((duration - t) / ramp_s)
        return scale * s

    return phase, duration


def loops(base, plan, motion, turns, period_s):
    """Pose at time t in [0, turns * period_s] with every planned finger going round; starts and ends at the start
    pose apply_plan(base, plan, 0, motion)."""
    phase, duration = phase_profile(turns, period_s)

    def pose_at(t):
        if t <= 0 or t >= duration:
            return apply_plan(base, plan, 0.0, motion)     # exactly the start pose (sin(2 pi turns) is not quite 0)
        return apply_plan(base, plan, phase(t), motion)

    return pose_at, duration


def ramp(start, end, duration_s):
    """Pose at time t in [0, duration_s] easing from `start` to `end` (16 joints each)."""
    def pose(t):
        f = ease(t / duration_s if duration_s > 0 else 1.0)
        return [a + (b - a) * f for a, b in zip(start, end)]
    return pose


class Streamer:
    """Streams position targets to the hand at a fixed rate and reports how well each motion was followed."""

    def __init__(self, hand, rate_hz, speed_percent, read_every=1, sleep=time.sleep, clock=time.monotonic,
                 log=print):
        self.hand, self.dt, self.sleep, self.clock, self.log = hand, 1.0 / rate_hz, sleep, clock, log
        self.speed, self.read_every = speed_percent, read_every
        self.rate_hz = None       # achieved by the last stream
        self.pose = None          # last commanded pose
        self.measured = None      # last read pose

    def start(self, settle_s):
        self.hand.set_position_mode()
        self.hand.set_speeds([self.speed] * 5)
        self.measured = self.hand.read_positions()
        self.pose = list(self.measured)
        self.move_to(REST, settle_s)

    def stream(self, pose_at, duration_s):
        """Write pose_at(t) every dt until duration_s, reading the pose back every read_every ticks (and on the
        last one); returns per-joint (min, max) measured and the worst lag seen at a read."""
        extremes = [[math.inf, -math.inf] for _ in range(16)]
        worst = 0.0
        t0 = self.clock()
        t, ticks = 0.0, 0
        while True:
            pose = [min(max(v, lo), hi) for v, (lo, hi) in zip(pose_at(t), LIMITS)]
            self.hand.write_positions(pose)
            self.pose = pose
            ticks += 1
            done = t >= duration_s
            if done or ticks % self.read_every == 0:
                self.measured = self.hand.read_positions()
                for q, e in zip(self.measured, extremes):
                    e[0], e[1] = min(e[0], q), max(e[1], q)
                worst = max(worst, max(abs(q - c) for q, c in zip(self.measured, pose)))
            self.hand.check_health()
            if done:
                elapsed = self.clock() - t0
                self.rate_hz = ticks / elapsed if elapsed > 0 else math.inf
                return extremes, worst
            self.sleep(self.dt)
            t = min(self.clock() - t0, duration_s)

    def move_to(self, target, duration_s):
        self.stream(ramp(self.pose, target, duration_s), duration_s)

    def loops(self, base, plan, motion, turns, period_s):
        pose_at, duration = loops(base, plan, motion, turns, period_s)
        extremes, worst = self.stream(pose_at, duration)
        driven = set()
        for finger in (f for f in FINGERS if f in plan):
            joints = sorted({j for j, _ in plan[finger](0.0)})
            driven |= set(joints)
            self.log(f"  {finger:6s} " + "   ".join(f"{JOINT_NAMES[j]:16s} {extremes[j][0]:6.1f} .. "
                                                    f"{extremes[j][1]:5.1f} deg" for j in joints))
        others = max([e[1] - e[0] for j, e in enumerate(extremes) if j not in driven], default=0.0)
        self.log(f"  lag {worst:.1f} deg, other joints moved {others:.1f} deg, streamed at {self.rate_hz:.0f} Hz")

    def finish(self, settle_s):
        if self.pose is not None and max(abs(a - b) for a, b in zip(self.pose, REST)) > 0.01:
            self.move_to(REST, settle_s)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="hand config with the hand_roles entry")
    p.add_argument("--hand_serial", default="AAC-002", help="hand_roles key (AAC-002 right, AAC-001 left)")
    p.add_argument("--fingers", nargs="+", choices=FINGERS, metavar="FINGER", default=list(FINGERS),
                   help="only these fingers loop (the others still spread and hold)")
    p.add_argument("--turns", type=int, default=2, help="loops per pass")
    p.add_argument("--period", type=float, default=3.0, help="seconds per loop")
    p.add_argument("--loops", type=int, default=1, help="passes; 0 = until Ctrl-C")
    p.add_argument("--radius", type=float, default=20.0, help="fingertip circle radius, mm, in the finger's plane")
    p.add_argument("--centre", type=float, nargs=2, default=list(DEFAULT_CENTRE), metavar="DEG",
                   help="MCP and PIP angles at the circle's centre")
    p.add_argument("--spread", type=float, nargs=4, default=list(DEFAULT_SPREAD), metavar="DEG",
                   help="spread pose: index middle ring pinky abduction, deg (+ toward the thumb)")
    p.add_argument("--side", type=float, default=3.0, help="abduction swing about the spread added to each "
                   f"finger's circle, +- deg; neighbours must keep {NEIGHBOUR_MIN_DEG:g} deg between them")
    p.add_argument("--thumb-flex", type=float, default=60.0, help="thumb loop height: thumb_abduction sweep, deg")
    p.add_argument("--thumb-side", type=float, default=25.0, help="thumb loop half-width: thumb_rotation, +- deg")
    p.add_argument("--pattern", choices=sorted(PATTERNS), default="alternate",
                   help="per-finger (phase offset, direction) for thumb..pinky: "
                        + "; ".join(f"{k} {v}" for k, v in PATTERNS.items()))
    p.add_argument("--phases", type=float, nargs=5, metavar="DEG",
                   help="custom phase offsets for thumb index middle ring pinky (overrides the pattern's)")
    p.add_argument("--spin", choices=sorted(SPIN), nargs=5, metavar="ccw|cw",
                   help="custom directions for thumb index middle ring pinky (overrides the pattern's)")
    p.add_argument("--speed", type=float, default=100.0, help="firmware finger speed percent (position mode)")
    p.add_argument("--margin", type=float, default=2.0, help="degrees kept clear of each joint's hard stop")
    p.add_argument("--rate", type=float, default=1000.0, help="target stream rate, Hz (each write waits for the "
                   "hand's reply, so the achieved rate is printed per pass)")
    p.add_argument("--read-every", type=int, default=10, help="read the pose back every N targets")
    p.add_argument("--settle", type=float, default=2.0, help="seconds of each easing move (rest, spread, back)")
    args = p.parse_args(argv)
    for name in ("period", "rate", "settle", "radius", "thumb_flex", "thumb_side"):
        if getattr(args, name) <= 0:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.side < 0:
        p.error("--side must not be negative")
    if not 0 < args.speed <= 100:
        p.error("--speed must be in (0, 100]")
    if args.turns < 1 or args.loops < 0 or args.margin < 0 or args.read_every < 1:
        p.error("--turns and --read-every must be at least 1; --loops and --margin must not be negative")
    args.fingers = [f for f in FINGERS if f in args.fingers]
    offsets = [o for o, _ in PATTERNS[args.pattern]] if args.phases is None else args.phases
    spins = [d for _, d in PATTERNS[args.pattern]] if args.spin is None else [SPIN[s] for s in args.spin]
    args.motion = {f: (float(o) % 360.0, d) for f, o, d in zip(FINGERS, offsets, spins) if f in args.fingers}
    try:
        check_spread(args.spread, args.side if any(f in args.fingers for f in FOUR) else 0.0, args.margin)
    except ValueError as error:
        p.error(str(error))
    ri = JOINT_NAMES.index("thumb_rotation")
    args.thumb_radii = (min(args.thumb_flex, LIMITS[JOINT_NAMES.index("thumb_abduction")][1] - args.margin),
                        min(args.thumb_side, LIMITS[ri][1] - args.margin))
    args.base = spread_pose(args.spread)
    return args


def run(args, hand=None, sleep=time.sleep, clock=time.monotonic, log=print, ik_samples=72):
    settings = load_aac_config(args, args.config)
    side = "Right" if settings["hand_id"].lower().startswith("right") else "Left"
    plan = build_plan(TipKinematics(side), args.fingers, args.base, tuple(args.centre), args.radius, args.side,
                      args.thumb_radii, args.margin, ik_samples)
    start = apply_plan(args.base, plan, 0.0, args.motion)
    if hand is None:
        hand = AACHand(settings["serial_device"], settings["device_address"], settings["connection_timeout_s"],
                       bus=settings["bus"])
    log(f"[demo] {settings['hand_id']} ({args.hand_serial}) over {settings['bus']} at {settings['serial_device']}")
    hand.open()
    streamer = Streamer(hand, args.rate, args.speed, read_every=args.read_every, sleep=sleep, clock=clock, log=log)
    passes = 0
    try:
        log(f"[demo] easing to rest, then to the spread {args.spread} and the circle starts ({args.settle:g} s "
            f"each), then {', '.join(args.fingers)}: {args.turns} loop(s) of {args.period:g} s, "
            f"{args.radius:g} mm circles, {'until Ctrl-C' if args.loops == 0 else f'{args.loops} pass(es)'}; "
            + ", ".join(f"{f} {o:g} deg {'ccw' if d > 0 else 'cw'}" for f, (o, d) in args.motion.items()))
        streamer.start(args.settle)
        streamer.move_to(args.base, args.settle)
        streamer.move_to(start, args.settle)
        while args.loops == 0 or passes < args.loops:
            streamer.loops(args.base, plan, args.motion, args.turns, args.period)
            passes += 1
    except KeyboardInterrupt:
        log("\n[demo] stopped: easing back to rest")
    finally:
        try:
            streamer.finish(args.settle)
            hand.set_position_mode()
        finally:
            hand.close()
    log(f"[demo] done: {passes} pass(es)")
    return passes


if __name__ == "__main__":
    run(parse_args())
