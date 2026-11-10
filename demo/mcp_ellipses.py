"""Demo: every fingertip follows an ellipse on its MCP joint only, each at its own frequency, in position mode.

The ellipses come from demo/parameters/ellipses_trajectory.yaml (--params): per finger the semi-axes a_mm (along flexion) and b_mm
(sideways), the frequency hz, the direction spin and a starting phase_deg; plus the spread, the flexion centre, the
pass duration and the ramp. Millimetres are converted to joint degrees from the hand's URDF (the joint amplitude whose
tip travel from the centre pose is that many mm): flexion = centre + a * sin(phase), abduction = spread + b * cos(phase), so the tip goes
round an ellipse a_mm by b_mm. The PIP joints stay open. The thumb does the same on its base with thumb_abduction (a)
and thumb_cmc (b); thumb_rotation and thumb_mcp hold still. Not rotation: with the thumb straight, rotation moves the
tip only 11 deg away from the direction abduction does (URDF, D21_Right), so the pair traced a thin sliver, while cmc
moves it at 78 deg and the pair traces the ellipse asked for (a 45 / b 30 -> 90 x 57 mm, within 1 mm). The amplitudes ease in over ramp_s at the start of a pass and out at its end, and the hand eases
rest -> spread -> centres before and back to rest after (Ctrl-C too).

Sideways the fingers have little room: their MCP housings sit ~1.5 mm apart at rest and a finger abducting alone
touches its neighbour at ~4 deg (URDF collision meshes, D21_Right; the bench showed the same 2026-10-30). Spread
10 deg apart they can swing about +-3 deg each (4 mm at the tip) and still clear each other in the worst case; the
demo does NOT check this, so watch the fingers when raising b_mm and stop (Ctrl-C) if neighbours touch. Joint stops
are not checked either: a target past a joint's range is clipped to the range (the SDK refuses anything outside it),
so that part of the ellipse runs flat along the stop. Bigger circles need the PIP as well: demo/finger_circles.py.

    hand/.hand/bin/python demo/mcp_ellipses.py                              # right hand, CAN FD, one 20 s pass
    hand/.hand/bin/python demo/mcp_ellipses.py --loops 0                    # until Ctrl-C
    hand/.hand/bin/python demo/mcp_ellipses.py --params my_ellipses.yaml --fingers index middle
    hand/.hand/bin/python demo/mcp_ellipses.py --config teleop/configure/acc/torque_control/hand_config_aac_l_torque_canfd.json \\
        --hand_serial AAC-001                                               # left hand
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hand.backends.aac import AACHand, JOINT_NAMES  # noqa: E402
from teleop.hand_control.hand_controller_aac import load_aac_config  # noqa: E402
from finger_circles import (DEFAULT_CONFIG, FINGERS, FOUR, LIMITS, REST, SPIN, Streamer, TipKinematics,  # noqa: E402
                            ease, spread_pose)

DEFAULT_PARAMS = ROOT / "demo" / "parameters" / "ellipses_trajectory.yaml"
AXES = {"thumb": ("thumb_abduction", "thumb_cmc")}     # not thumb_rotation: nearly parallel to abduction at the tip
AXES.update({f: (f"{f}_mcp", f"{f}_abduction") for f in FOUR})       # (a axis, b axis) per finger
FINGER_KEYS = {"a_mm", "b_mm", "hz", "spin", "phase_deg"}
TOP_KEYS = {"duration_s", "ramp_s", "spread_deg", "flex_centre_deg", "thumb_centre_deg", "fingers"}


def number(value, name, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number")
    if minimum is not None and (value < minimum or strict and value <= minimum):
        raise ValueError(f"{name} must be {'above' if strict else 'at least'} {minimum:g}")
    return float(value)


def load_params(path):
    """The YAML as a dict with checked types; every finger present with every key."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or set(raw) != TOP_KEYS:
        raise ValueError(f"{path} must contain exactly: {', '.join(sorted(TOP_KEYS))}")
    p = {"duration_s": number(raw["duration_s"], "duration_s", 0, strict=True),
         "ramp_s": number(raw["ramp_s"], "ramp_s", 0),
         "flex_centre_deg": number(raw["flex_centre_deg"], "flex_centre_deg")}
    for key, n in (("spread_deg", 4), ("thumb_centre_deg", 3)):
        values = raw[key]
        if not isinstance(values, list) or len(values) != n:
            raise ValueError(f"{key} must be a list of {n} numbers")
        p[key] = [number(v, f"{key}[{i}]") for i, v in enumerate(values)]
    fingers = raw["fingers"]
    if not isinstance(fingers, dict) or set(fingers) != set(FINGERS):
        raise ValueError(f"fingers must have exactly: {', '.join(FINGERS)}")
    p["fingers"] = {}
    for finger, block in fingers.items():
        if not isinstance(block, dict) or set(block) != FINGER_KEYS:
            raise ValueError(f"fingers.{finger} must have exactly: {', '.join(sorted(FINGER_KEYS))}")
        if block["spin"] not in SPIN:
            raise ValueError(f"fingers.{finger}.spin must be ccw or cw")
        p["fingers"][finger] = {"a_mm": number(block["a_mm"], f"fingers.{finger}.a_mm", 0),
                                "b_mm": number(block["b_mm"], f"fingers.{finger}.b_mm", 0),
                                "hz": number(block["hz"], f"fingers.{finger}.hz", 0, strict=True),
                                "spin": SPIN[block["spin"]],
                                "phase_deg": number(block["phase_deg"], f"fingers.{finger}.phase_deg")}
    return p


def axis_degrees(kin, finger, pose, joint, mm):
    """The joint amplitude (deg) that moves the fingertip `mm` from its position at `pose` along `joint`, averaged
    over the + and - directions (bisection on the URDF; the tip's travel per degree is not constant)."""
    j = JOINT_NAMES.index(joint)
    t0 = kin.tip(finger, pose)

    def travel(deg):
        total = 0.0
        for sign in (1, -1):
            bumped = list(pose)
            bumped[j] = min(max(pose[j] + sign * deg, LIMITS[j][0]), LIMITS[j][1])
            total += np.linalg.norm(kin.tip(finger, bumped) - t0)
        return total / 2

    if mm <= 0:
        return 0.0
    lo, hi = 0.0, float(LIMITS[j][1] - LIMITS[j][0])
    if travel(hi) < mm:
        return hi                            # more than the joint can give: its full range (clipped when streamed)
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if travel(mid) < mm else (lo, mid)
    return (lo + hi) / 2


def build_plan(kin, fingers, params):
    """(plan, centre): plan maps finger -> (function(phase) -> [(joint, deg), ...], info with the degrees used);
    centre is the 16-joint pose the ellipses are drawn about. Amplitudes are not limited: targets past a joint's
    range are clipped to it when streamed."""
    centre = spread_pose(params["spread_deg"])
    for joint, angle in zip(("thumb_abduction", "thumb_rotation", "thumb_cmc"), params["thumb_centre_deg"]):
        centre[JOINT_NAMES.index(joint)] = angle
    for finger in FOUR:
        centre[JOINT_NAMES.index(f"{finger}_mcp")] = params["flex_centre_deg"]
    plan = {}
    for finger in fingers:
        cfg = params["fingers"][finger]
        ai, bi = (JOINT_NAMES.index(j) for j in AXES[finger])
        a_deg = axis_degrees(kin, finger, centre, AXES[finger][0], cfg["a_mm"])
        b_deg = axis_degrees(kin, finger, centre, AXES[finger][1], cfg["b_mm"])

        def ellipse(phase, ai=ai, bi=bi, ca=centre[ai], cb=centre[bi], a=a_deg, b=b_deg):
            return [(ai, ca + a * math.sin(phase)), (bi, cb + b * math.cos(phase))]
        plan[finger] = (ellipse, {"a_deg": a_deg, "b_deg": b_deg, "a_mm": cfg["a_mm"], "b_mm": cfg["b_mm"]})
    return plan, centre


def ellipses(centre, plan, params, fingers):
    """Pose at time t in [0, duration_s]: every planned finger goes round its ellipse at its own frequency, direction
    and starting phase, with the amplitudes eased in over ramp_s at the start and out over ramp_s at the end."""
    duration = params["duration_s"]
    ramp_s = min(params["ramp_s"], duration / 2)
    motion = {f: (params["fingers"][f]["spin"] * 2 * math.pi * params["fingers"][f]["hz"],
                  math.radians(params["fingers"][f]["phase_deg"])) for f in fingers}
    start = list(centre)
    for finger, (joints, _) in plan.items():
        for j, value in joints(motion[finger][1]):
            start[j] = value

    def pose_at(t):
        if ramp_s <= 0 and 0 <= t <= duration:
            envelope = 1.0
        elif t <= 0 or t >= duration:
            return list(centre)
        else:
            envelope = ease(t / ramp_s) * ease((duration - t) / ramp_s)
        pose = list(centre)
        for finger, (joints, _) in plan.items():
            rate, offset = motion[finger]
            for j, value in joints(rate * t + offset):
                pose[j] = centre[j] + envelope * (value - centre[j])
        return pose

    return pose_at, duration, start


class EllipseStreamer(Streamer):
    def ellipses(self, centre, plan, params, fingers):
        pose_at, duration, _ = ellipses(centre, plan, params, fingers)
        extremes, worst = self.stream(pose_at, duration)
        driven = set()
        for finger in (f for f in FINGERS if f in plan):
            joints = [JOINT_NAMES.index(j) for j in AXES[finger]]
            driven |= set(joints)
            self.log(f"  {finger:6s} {params['fingers'][finger]['hz']:.2f} Hz " + "   ".join(
                f"{JOINT_NAMES[j]:16s} {extremes[j][0]:6.1f} .. {extremes[j][1]:5.1f} deg" for j in joints))
        others = max([e[1] - e[0] for j, e in enumerate(extremes) if j not in driven], default=0.0)
        self.log(f"  lag {worst:.1f} deg, other joints moved {others:.1f} deg, streamed at {self.rate_hz:.0f} Hz")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="hand config with the hand_roles entry")
    p.add_argument("--hand_serial", default="AAC-002", help="hand_roles key (AAC-002 right, AAC-001 left)")
    p.add_argument("--params", default=str(DEFAULT_PARAMS), help="YAML with the ellipses (see the default file)")
    p.add_argument("--fingers", nargs="+", choices=FINGERS, metavar="FINGER", default=list(FINGERS),
                   help="only these fingers move (the others still spread and hold their centre)")
    p.add_argument("--loops", type=int, default=1, help="passes; 0 = until Ctrl-C")
    p.add_argument("--speed", type=float, default=100.0, help="firmware finger speed percent (position mode)")
    p.add_argument("--rate", type=float, default=1000.0, help="target stream rate, Hz (achieved rate is printed)")
    p.add_argument("--read-every", type=int, default=10, help="read the pose back every N targets")
    p.add_argument("--settle", type=float, default=2.0, help="seconds of each easing move (rest, spread, back)")
    args = p.parse_args(argv)
    for name in ("rate", "settle"):
        if getattr(args, name) <= 0:
            p.error(f"--{name} must be positive")
    if not 0 < args.speed <= 100:
        p.error("--speed must be in (0, 100]")
    if args.loops < 0 or args.read_every < 1:
        p.error("--read-every must be at least 1; --loops must not be negative")
    args.fingers = [f for f in FINGERS if f in args.fingers]
    try:
        args.ellipses = load_params(args.params)
    except (OSError, yaml.YAMLError, ValueError) as error:
        p.error(f"{args.params}: {error}")
    return args


def run(args, hand=None, sleep=time.sleep, clock=time.monotonic, log=print):
    settings = load_aac_config(args, args.config)
    side = "Right" if settings["hand_id"].lower().startswith("right") else "Left"
    params = args.ellipses
    plan, centre = build_plan(TipKinematics(side), args.fingers, params)
    _, _, start = ellipses(centre, plan, params, args.fingers)
    if hand is None:
        hand = AACHand(settings["serial_device"], settings["device_address"], settings["connection_timeout_s"],
                       bus=settings["bus"])
    log(f"[demo] {settings['hand_id']} ({args.hand_serial}) over {settings['bus']} at {settings['serial_device']}")
    for finger, (_, info) in plan.items():
        cfg = params["fingers"][finger]
        a_axis, b_axis = AXES[finger]
        log(f"[demo] {finger}: {info['a_mm']:g} x {info['b_mm']:g} mm ellipse = {a_axis} +-{info['a_deg']:.1f} deg, "
            f"{b_axis} +-{info['b_deg']:.1f} deg, {cfg['hz']:.2f} Hz {'ccw' if cfg['spin'] > 0 else 'cw'}, "
            f"from {cfg['phase_deg']:g} deg")
    hand.open()
    streamer = EllipseStreamer(hand, args.rate, args.speed, read_every=args.read_every, sleep=sleep, clock=clock,
                               log=log)
    passes = 0
    try:
        log(f"[demo] easing to rest, to the spread {params['spread_deg']} and the ellipse centres ({args.settle:g} s "
            f"each), then {params['duration_s']:g} s passes, "
            f"{'until Ctrl-C' if args.loops == 0 else f'{args.loops} pass(es)'}")
        streamer.start(args.settle)
        streamer.move_to(spread_pose(params["spread_deg"]), args.settle)
        streamer.move_to(centre if params["ramp_s"] > 0 else start, args.settle)
        while args.loops == 0 or passes < args.loops:
            streamer.ellipses(centre, plan, params, args.fingers)
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
