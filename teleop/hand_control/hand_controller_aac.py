"""Single-owner AAC live position transport. No hardware is opened on import."""

import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import uuid

import zmq

from hand.backends.aac import (
    AACHand, ANGLE_LIMITS, JOINT_NAMES, cleanup, finite_number, positive_number,
    validate_positions, validate_speeds,
)
from hand.config import load_hand_config, resolve_hand_role, resolve_zmq_transport
from teleop.hand_control.admittance import load_admittance

REPO_ROOT = Path(__file__).resolve().parents[2]            # agi-os (this file: teleop/hand_control/)
TELEOP_DIR = REPO_ROOT / "teleop"
# Where a file named in a hand config is looked for, in order, when the name is relative: the config's own folder,
# then the repo root, teleop/ (the configs' old home: "../demo/x.json" style paths keep working), and the folders the
# calibration data lives in, so a bare name such as "aac_breakaway_left.json" is found where it is kept.
DATA_DIRS = (REPO_ROOT, TELEOP_DIR, TELEOP_DIR / "calibration" / "acc" / "friction",
             TELEOP_DIR / "calibration" / "acc" / "admittance")


def resolve_data_path(name, config_path=None, what="file"):
    """The existing file a config names (see DATA_DIRS); FileNotFoundError listing where it was looked for."""
    if Path(name).is_absolute():
        return str(name)
    bases = ([Path(config_path).resolve().parent] if config_path else []) + list(DATA_DIRS)
    for base in bases:
        candidate = (base / name).resolve()
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(f"{what} {name!r} not found in " + ", ".join(str(b) for b in bases))


FUTURE_LEAD_S = 0.05
MAX_DRAIN = 32
COMPLIANCE_KEYS = {"min_idle_current_a", "idle_sigma", "gain_dps_per_a", "return_rate", "offset_max_deg",
                   "error_clamp_deg", "velocity_gate_dps", "deviation_min_deg", "velocity_window_s", "max_dt_s",
                   "rest_bias_tau_s", "direction_from_current", "sign_min_samples", "sign_min_agreement",
                   "target_motion_gate_dps", "target_holdoff_s"}
COMPLIANCE_IDLE_SAMPLES = 25
COMMAND_RESOLUTION_DEG = 0.05  # old firmware (v1) register step is 0.035-0.05 deg; smaller changes are not re-sent.
# The device's own command_resolution_deg (v2 firmware: 0.0025 deg) takes precedence when it has one.
# The LEFT D21 reads finger abduction (joints 4, 7, 10, 13) back with the opposite sign to the command.
READBACK_SIGN = [1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1]


def readback_sign(hand_id, register_map=None):
    """Per-hand, per-firmware table turning the raw position readback into the command frame.

    Old firmware (v1), left hand: abduction mirrored (READBACK_SIGN). Right hand: every joint reads back with the
    command's sign (teleop logs 2026-10-30: left run, index abduction commanded +17.6 read -18.2; right run,
    commanded +16.1 read +16.1). Applying the left table to the right hand made every loop see a ~32 deg abduction
    error. New firmware (v2): no mirror on either hand (left hand over RS-485, 2026-11-02: index abduction
    commanded +15 read +15); applying the v1 left table drove every finger abduction into its limit in torque mode
    (log manus_torque_3). The controller re-evaluates this after open() once the device knows its firmware.
    """
    if register_map == "v2" or "Right" in str(hand_id):
        return [1] * 16
    return list(READBACK_SIGN)
TRACKING_KEYS = {"kp", "ki", "kd", "leak", "deadband_deg", "max_deg", "target_motion_gate_dps", "target_holdoff_s",
                 "velocity_window_s"}


class TrackingCorrection:
    """Closed loop on the firmware's tracking error: command = target + kp*e - kd*qdot + I, e = target - measured.

    The hand's own position loop stops short of its target; this pushes the command past the
    target by the observed gap. I is a leaky integrator so a steady gap is removed while noise
    (below deadband_deg) and the lag of a commanded move (target_motion_gate_dps, target_holdoff_s)
    are not integrated. The correction is bounded by max_deg and the joint limits; against an
    object it pushes harder, up to that bound, which is the opposite of compliance.

    PD: ki 0 and kp/kd > 0. qdot is the measured joint velocity over velocity_window_s (feedback only
    changes at 50 Hz, so a per-tick difference would be zero or a spike); kd*qdot damps overshoot of
    the kp push. Like kp, it is off while the target moves.

    The integrator is off unless ki is given (default 0); leak, deadband_deg and max_deg act on I only.
    """

    def __init__(self, kp=0.0, ki=0.0, kd=0.0, leak=0.5, deadband_deg=0.3, max_deg=10.0,
                 target_motion_gate_dps=10.0, target_holdoff_s=0.2, velocity_window_s=0.1):
        for name, value in (("kp", kp), ("ki", ki), ("kd", kd), ("leak", leak), ("deadband_deg", deadband_deg),
                            ("target_holdoff_s", target_holdoff_s)):
            if finite_number(value, f"tracking.{name}") < 0:
                raise ValueError(f"tracking.{name} must be non-negative")
        self.kp, self.ki, self.kd = float(kp), float(ki), float(kd)
        self.leak, self.deadband = float(leak), float(deadband_deg)
        self.velocity_window = positive_number(velocity_window_s, "tracking.velocity_window_s")
        self.max_deg = positive_number(max_deg, "tracking.max_deg")
        self.gate = positive_number(target_motion_gate_dps, "tracking.target_motion_gate_dps")
        self.holdoff = float(target_holdoff_s)
        self.readback_sign = list(READBACK_SIGN)  # the controller replaces this with readback_sign(hand_id)
        self.reset()

    def reset(self):
        self.integral = [0.0] * 16
        self.last_time = None
        self.last_target = None
        self.moved_at = [float("-inf")] * 16
        self.history = []  # (time, measured) within velocity_window_s
        self.velocity = [0.0] * 16

    def step(self, target, measured_raw, now):
        """Returns (command, correction) in command-frame degrees, clipped to the joint limits."""
        measured = [s * m for s, m in zip(self.readback_sign, measured_raw)]
        dt = min(max(now - self.last_time, 0.0), 0.1) if self.last_time is not None else 0.0
        self.history = [h for h in self.history if now - h[0] <= self.velocity_window] + [(now, measured)]
        span = now - self.history[0][0]
        self.velocity = ([(m - m0) / span for m, m0 in zip(measured, self.history[0][1])] if span > 0
                         else [0.0] * 16)
        command, correction = [], []
        for j in range(16):
            if self.last_target is None or (dt > 0 and abs(target[j] - self.last_target[j]) / dt >= self.gate):
                self.moved_at[j] = now
            moving = now - self.moved_at[j] < self.holdoff
            error = target[j] - measured[j]
            if not moving and abs(error) > self.deadband:
                self.integral[j] += self.ki * error * dt
            self.integral[j] -= self.leak * self.integral[j] * dt
            self.integral[j] = min(max(self.integral[j], -self.max_deg), self.max_deg)
            delta = (0.0 if moving else self.kp * error - self.kd * self.velocity[j]) + self.integral[j]
            lo, hi = ANGLE_LIMITS[j]
            command.append(min(max(target[j] + delta, lo), hi))
            correction.append(command[-1] - target[j])
        self.last_time, self.last_target = now, list(target)
        return command, correction


TORQUE_KEYS = {"kp", "kd", "tau_max", "deadband_deg", "velocity_window_s", "stale_feedback_s", "open_loop", "center",
               "center_lag_s", "law", "force_close_n", "force_open_n", "curl_full_deg", "thumb_curl_full_deg",
               "nm_per_ma", "friction_ma", "pipeline", "kd_fast", "v_fast_dps", "v_stick_dps",
               "friction_ramp_deg", "friction_table", "friction_scale", "dither_ma", "dither_hz", "joints",
               "firmware_velocity", "friction_ff_scale", "friction_ff_ramp_dps", "target_velocity_window_s",
               "breakaway_fine"}
TORQUE_JOINT_KEYS = {"kp", "kd", "friction_scale", "dither_ma", "friction_ff_scale",   # per-joint overrides
                     "boost_ma", "boost_above_deg", "boost_max_s", "boost_cooldown_s", "boost_min_error_deg",
                     "target_max_deg"}
BOOST_KEYS = {"boost_ma", "boost_above_deg", "boost_max_s", "boost_cooldown_s", "boost_min_error_deg"}
TORQUE_LAWS = ("pd", "force")
# Open-loop torque: no usable pose feedback in torque mode, so the PD runs against a fixed per-joint
# "center" pose: tau = kp*(q_glove - q_center). center "start" = the pose measured just before entering
# torque mode (the last live reading; the firmware freezes it at the switch), "mid" = joint mid-range,
# "previous" = the glove target from center_lag_s ago (0 = the previous frame, i.e. the retargeter's
# q_prev regularisation point), which makes the torque follow recent glove motion rather than a pose.
TORQUE_OPEN_LOOP_CENTER = [(lo + hi) / 2 for lo, hi in ANGLE_LIMITS.values()]
TORQUE_CENTERS = ("start", "mid", "previous")
# Pipelined torque loop: the pose fetched with the previous torque write is used if it is at most this old,
# otherwise the tick reads a fresh pose first (after idle, a slow tick, or a pause in the glove stream).
PIPELINE_POSE_MAX_AGE_S = 0.02
STALE_FEEDBACK_S = 1.0  # torque mode: joint positions that never change while pushing mean the PD is blind
# Joint torques -> motor torques (spec section 3.5): each finger's abduction and flexion joints share a
# differential motor pair; same sign on both motors = flexion (thumb: abduction), opposite = abduction
# (thumb: rotation). {same-sign joint, opposite-sign joint, motor a, motor b}, 0-based.
TORQUE_PAIRS = ((0, 1, 0, 1), (5, 4, 4, 5), (8, 7, 7, 8), (11, 10, 10, 11), (14, 13, 13, 14))
TORQUE_SINGLES = ((2, 2), (3, 3), (6, 6), (9, 9), (12, 12), (15, 15))  # (joint, motor)
# Polarity of the opposite-sign half: +joint (abduction toward the thumb, thumb rotation +) needs motor a NEGATIVE
# and motor b POSITIVE. Bench on the v2 firmware (left hand, RS-485, 2026-11-02): thumb m1- m2+ -> rotation +25 deg,
# index m5- m6+ -> abduction +18, pinky m14- m15+ -> abduction +25; the other polarity pinned the thumb at -35 deg
# for a whole session (log manus_torque_1). Same-sign half confirmed: + = flexion (thumb: abduction).
TORQUE_PAIR_OPPOSITE_SIGN = -1
# Per hand, per pair (thumb, index, middle, ring, pinky). The right hand (v2, bench 2026-11-03, from the home pose):
# thumb rotation as on the left, but every finger abduction moved opposite to the left-hand polarity (index -120 mA
# -> +1.5 deg, middle +90 -> -4.2, ring +90 -> -5.1, pinky +90 -> -6.2), so its finger pairs use +1.
PAIR_POLARITY = {"Left": (-1, -1, -1, -1, -1), "Right": (-1, 1, 1, 1, 1)}


def pair_polarity(hand_id):
    """Opposite-sign polarity of the five motor pairs for this hand (thumb, index, middle, ring, pinky)."""
    return PAIR_POLARITY["Right" if "Right" in str(hand_id) else "Left"]


def per_joint(options, key, default):
    """16 values of a torque setting: the global value, overridden per joint by torque.joints[<name>][key]."""
    overrides = options.get("joints") or {}
    return [float(overrides.get(name, {}).get(key, default)) for name in JOINT_NAMES]


def target_maxima(options):
    """Per-joint upper target limits (deg) from torque.joints.<joint>.target_max_deg: 16 values, None = no limit
    (key absent or null). None overall when no joint has one."""
    joints = (options or {}).get("joints") or {}
    maxima = [joints.get(name, {}).get("target_max_deg") for name in JOINT_NAMES]
    return maxima if any(m is not None for m in maxima) else None


def limit_targets(target, maxima):
    """Clamp each joint target to its maximum: a joint with a mechanical stop or a high-friction zone deep in its
    curl is never asked to go there (right D21 ring MCP stops at ~80 deg while GeoRT asks up to 85, log
    right_ring_issue_1, 2026-11-05; left index PIP jammed against its ~89 deg stop when asked for 93.7)."""
    if maxima is None:
        return list(target)
    return [t if m is None else min(t, m) for t, m in zip(target, maxima)]


def load_friction_table(path, scale, default_ma=65.0):
    """Per-joint friction feed-forward (mA) = scale * mean measured breakaway current of the joint, from a JSON file
    {"<joint_name>+": mA, "<joint_name>-": mA} (null = not measured; missing joints use default_ma)."""
    table = json.loads(Path(path).read_text())
    scales = scale if isinstance(scale, (list, tuple)) else [scale] * 16
    out = []
    for name, s in zip(JOINT_NAMES, scales):
        values = [table.get(name + sign) for sign in ("+", "-")]
        values = [float(v) for v in values if v]
        out.append(s * (sum(values) / len(values) if values else default_ma))
    return out


def load_breakaway(path, default_ma=65.0):
    """Measured breakaway current per joint and direction: (plus[16], minus[16]) mA from the friction table file.
    A direction that was not measured (null) uses the other one, a joint with neither uses default_ma."""
    table = json.loads(Path(path).read_text())
    plus, minus = [], []
    for name in JOINT_NAMES:
        p, m = table.get(name + "+"), table.get(name + "-")
        p = float(p) if p else (float(m) if m else default_ma)
        m = float(m) if m else p
        plus.append(p)
        minus.append(m)
    return plus, minus


class TargetVelocity:
    """Speed of the glove target (deg/s): change over the last window_s of targets. Targets arrive at ~100 Hz and are
    held between frames, so the window spans a few frames; reset() after idle so a new session starts at rest."""

    def __init__(self, window_s=0.03):
        self.window_s = positive_number(window_s, "torque.target_velocity_window_s")
        self.reset()

    def reset(self):
        self.history = []

    def update(self, now, target):
        self.history = [h for h in self.history if now - h[0] <= self.window_s] + [(now, list(target))]
        t0, q0 = self.history[0]
        span = now - t0
        return [(a - b) / span for a, b in zip(target, q0)] if span > 0 else [0.0] * len(target)


class BreakawayModel:
    """Breakaway current per joint and direction as a function of the joint angle.

    Built from the fine table (teleop/calibration/acc/friction/aac_breakaway_left_fine.json: measurements at several angles per joint and
    direction) with linear interpolation between the measured angles and the nearest value beyond them; or from a
    coarse table ({joint+/-: mA}) as constants. Thumb rotation needs this: its breakaway runs from 20 to 125 mA
    along the range, so a single mean over- and under-compensates by turns (bench 2026-11-03)."""

    def __init__(self, path, default_ma=65.0, max_ma=200.0):
        data = json.loads(Path(path).read_text())
        self.curves = {}                       # (joint index, +1/-1) -> (angles[], mA[]) sorted by angle
        if "measurements" in data:
            for m in data["measurements"]:
                if m.get("breakaway_ma") and m["joint"] in JOINT_NAMES:
                    key = (JOINT_NAMES.index(m["joint"]), 1 if m["direction"] == "+" else -1)
                    self.curves.setdefault(key, []).append((float(m["angle_deg"]), min(float(m["breakaway_ma"]), max_ma)))
            for key, pts in self.curves.items():
                pts.sort()
                self.curves[key] = ([a for a, _ in pts], [c for _, c in pts])
        else:
            for j, name in enumerate(JOINT_NAMES):
                for sign, suffix in ((1, "+"), (-1, "-")):
                    v = data.get(name + suffix)
                    if v:
                        self.curves[(j, sign)] = ([0.0], [float(v)])
        self.default_ma = default_ma

    def current(self, j, angle, direction):
        """mA to break joint j loose at this angle, moving in direction (+1/-1)."""
        key = (j, 1 if direction >= 0 else -1)
        curve = self.curves.get(key) or self.curves.get((j, -key[1]))
        if curve is None:
            return self.default_ma
        angles, values = curve
        if angle <= angles[0]:
            return values[0]
        if angle >= angles[-1]:
            return values[-1]
        for k in range(1, len(angles)):
            if angle <= angles[k]:
                w = (angle - angles[k - 1]) / (angles[k] - angles[k - 1])
                return values[k - 1] + w * (values[k] - values[k - 1])
        return values[-1]

    def both(self, j, angle):
        return self.current(j, angle, 1), self.current(j, angle, -1)


def friction_feedforward(tau, target_velocity, plus, minus, scale, ramp_dps):
    """Friction compensation driven by the TARGET's motion: scale * breakaway * tanh(v_target / ramp), with the
    breakaway of the direction the target moves in. The push arrives as soon as the glove starts moving, instead of
    after an error has built up, which is what made the joints stick and then catch up (log manus_torque_11: joints
    still 22-37 % of the time while their target moved at 10-80 deg/s)."""
    out = []
    for t, v, p, m, s in zip(tau, target_velocity, plus, minus, scale):
        if s:
            t += s * (p if v > 0 else m) * math.tanh(v / ramp_dps)
        out.append(t)
    return out


def smooth_friction(tau, error, friction_ma, ramp_deg):
    """Friction compensation that ramps with the error, fric * tanh(error / ramp), instead of switching at a deadband.

    Bench 2026-11-02 (v2 left, all 16 joints following 0.5 Hz sines, 120 mA cap): with dither this kept every joint
    within ~2-3 deg at slip speeds under ~90 deg/s; the switched kick drove the motor pairs into bang-bang."""
    return [t + f * math.tanh(e / ramp_deg) for t, e, f in zip(tau, error, friction_ma)]


def motor_active(joint_active):
    """Per-motor flag from per-joint flags through the pair/single mapping of joint_to_motor_torques."""
    out = [False] * 16
    for same, opposite, a, b in TORQUE_PAIRS:
        out[a] = out[b] = bool(joint_active[same] or joint_active[opposite])
    for joint, m in TORQUE_SINGLES:
        out[m] = bool(joint_active[joint])
    return out


def dither(command, amplitude_ma, hz, now, limit, active=None):
    """Square-wave torque of +-amplitude on the motors that need it. The D21 joints stick until 60-100 mA and then
    slip almost freely; a small alternating torque keeps them on the edge of moving so a low kp follows the target
    without the stick-and-release cycle (bench 2026-11-02: 30 mA at 50 Hz cut slip speeds from ~150 to ~50 deg/s).

    Not applied to a motor already at the cap (the dither would only reduce a saturated push: log manus_torque_6,
    index PIP pinned at -120 could not extend) nor, when `active` is given, to motors whose joints are inside the
    deadband (no error to close: no need to buzz the hand at rest)."""
    d = amplitude_ma if math.sin(2 * math.pi * hz * now) >= 0 else -amplitude_ma
    out = []
    for m, c in enumerate(command):
        if abs(c) >= limit or (active is not None and not active[m]):
            out.append(c)
        else:
            out.append(max(-limit, min(limit, c + d)))
    return out


class OpeningBoost:
    """Extra current to free a joint that is stuck deep in its curl (torque.joints.<joint>.boost_*).

    The right D21's ring PIP needs ~210-280 mA to break loose once curled past ~70 deg (breakaway table, bench
    2026-11-03), above the 200 mA cap: in teleop it stayed at 76-82 deg for up to 4.6 s with the controller pushing
    it open at -200 mA (log right_ring_issue, 2026-11-05). For a joint with boost_ma, the motor command is set to
    -boost_ma when ALL hold:
      - the joint is curled past boost_above_deg,
      - its target lies more than boost_min_error_deg below it (the push is toward OPENING; curling is never boosted,
        so the boost cannot drive a joint into its end stop),
      - it is stuck: |speed| below v_stick_dps.
    It stops as soon as the joint moves (speed above v_stick_dps), and lasts at most boost_max_s per stuck episode;
    after that the joint gets no boost for boost_cooldown_s, so a truly jammed joint cannot hold the motor at
    boost_ma. Only joints driven by one motor alone (the PIPs, thumb CMC and thumb MCP); the motor-pair joints
    (abduction, MCP) are rejected at config load."""

    DEFAULTS = dict(boost_above_deg=70.0, boost_max_s=1.0, boost_cooldown_s=2.0, boost_min_error_deg=3.0)

    def __init__(self, options):
        self.joints = {}
        single = dict(TORQUE_SINGLES)
        for name, block in (options.get("joints") or {}).items():
            if "boost_ma" not in block:
                continue
            j = JOINT_NAMES.index(name)
            self.joints[j] = dict(self.DEFAULTS, **{k: float(v) for k, v in block.items() if k in BOOST_KEYS},
                                  motor=single[j], since=None, cool_until=-math.inf, count=0, active=False)

    def __bool__(self):
        return bool(self.joints)

    def apply(self, command, q, error, qdot, now, v_stick_dps):
        """Overwrite the boosted motors' commands in place; returns the command list."""
        for j, s in self.joints.items():
            want = (q[j] > s["boost_above_deg"] and error[j] < -s["boost_min_error_deg"]
                    and abs(float(qdot[j])) < v_stick_dps)
            s["active"] = False
            if not want:
                s["since"] = None
                continue
            if now < s["cool_until"]:
                continue
            if s["since"] is None:
                s["since"] = now
                s["count"] += 1
            if now - s["since"] > s["boost_max_s"]:
                s["since"], s["cool_until"] = None, now + s["boost_cooldown_s"]
                continue
            command[s["motor"]] = -s["boost_ma"]
            s["active"] = True
        return command

    def summary(self):
        return {JOINT_NAMES[j]: dict(active=s["active"], episodes=s["count"]) for j, s in self.joints.items()}

    def describe(self):
        return ", ".join(f"{JOINT_NAMES[j]} -{s['boost_ma']:g} mA when stuck past {s['boost_above_deg']:g} deg and "
                         f"opening (max {s['boost_max_s']:g} s, then {s['boost_cooldown_s']:g} s cool-down)"
                         for j, s in self.joints.items())


def pd_extras(tau, error, qdot, friction_ma=0.0, kd_fast=0.0, v_fast_dps=100.0, deadband_deg=0.0, v_stick_dps=20.0):
    """Stiction-aware additions to the joint PD (bench, v2 left hand, 2026-11-02).

    The D21 joints stick until ~60-100 mA and then slip with almost no inertia, so a plain PD either sits short of
    the target or overshoots at hundreds of deg/s. friction_ma is a constant push in the direction of the error
    (outside the deadband) while the joint is stuck (|qdot| < v_stick_dps) so a small kp still breaks it loose; once it
    moves the kick is dropped, otherwise it becomes a bang-bang push that flips sign across the deadband (seen with
    all 16 joints moving: every motor pair saturated, ~20 sign flips/s). kd_fast is extra damping applied only
    while |qdot| > v_fast_dps, i.e. during a slip, so it brakes the overshoot without adding noise torque at rest."""
    out = []
    for t, e, v in zip(tau, error, qdot):
        if friction_ma and abs(e) > deadband_deg and abs(v) < v_stick_dps:
            t += friction_ma if e > 0 else -friction_ma
        if kd_fast and abs(v) > v_fast_dps:
            t -= kd_fast * v
        out.append(t)
    return out


def joint_to_motor_torques(tau, limit, polarity=None):
    """Joint torques -> 16 motor torques. polarity: per pair (thumb, index, middle, ring, pinky), the sign of the
    opposite-sign half (pair_polarity(hand_id)); default the left hand's."""
    polarity = PAIR_POLARITY["Left"] if polarity is None else polarity
    motor = [0.0] * 16
    for k, (same, opposite, a, b) in enumerate(TORQUE_PAIRS):
        motor[a] = tau[same] + polarity[k] * tau[opposite]
        motor[b] = tau[same] - polarity[k] * tau[opposite]
    for joint, m in TORQUE_SINGLES:
        motor[m] = tau[joint]
    return [max(-limit, min(limit, v)) for v in motor]


def load_torque_controller(options):
    """Joint PD (impedance) from dex-retargetor-cpp: tau = kp*(q_target - q) - kd*qdot, in degrees;
    torque units are register units (1 unit = 1 mA = about 1 mN*m at the joint, bench-measured)."""
    sys.path.insert(0, str(REPO_ROOT / "retargetor" / "dex-retargetor-cpp" / "python"))
    import dexrt
    kp, kd, tau_max = (float(options.get(k, d)) for k, d in (("kp", 3.0), ("kd", 0.02), ("tau_max", 200.0)))
    kp, kd = per_joint(options, "kp", kp), per_joint(options, "kd", kd)
    return dexrt.Impedance(kp=kp, kd=kd, tau_max=[tau_max] * 16, target_timeout_s=0.0,
                           velocity_window_s=float(options.get("velocity_window_s", 0.1)),
                           deadband=float(options.get("deadband_deg", 1.0)))


class FingertipForceLaw:
    """Open-loop fingertip force from the retargeter solution: tau = J(q_sol)^T F per finger.

    The glove's curl (sum of the finger's flexion angles in q_sol) sets the force magnitude,
    F = force_close_n * u - force_open_n * (1 - u), u = curl / curl_full in [0, 1]; the direction is
    where the fingertip moves when that finger curls, from the fingertip Jacobian at q_sol. Joint
    torques (N*m) go only to the finger's flexion joints and become register units (mA) through
    nm_per_ma, plus a friction feed-forward. No position feedback is used.
    """
    # (tip link, flexion joint indices in the 16 independent joints = SDK joint order for the left hand)
    FINGERS = (("thumb_tip", (2, 3)), ("index_tip", (5, 6)), ("middle_tip", (8, 9)),
               ("ring_tip", (11, 12)), ("pinky_tip", (14, 15)))

    def __init__(self, options, side="Left"):
        sys.path.insert(0, str(REPO_ROOT / "retargetor" / "dex-retargetor-cpp" / "python"))
        import dexrt
        import numpy as np
        self.np = np
        urdf = REPO_ROOT / "hand" / "aac" / "description" / f"D21_{side}.urdf"
        self.model = dexrt.Retargeter.for_aac(urdf)
        get = lambda k, d: float(options.get(k, d))
        self.close_n, self.open_n = get("force_close_n", 3.0), get("force_open_n", 1.0)
        self.curl_full, self.thumb_curl_full = get("curl_full_deg", 120.0), get("thumb_curl_full_deg", 80.0)
        self.nm_per_ma, self.friction_ma = get("nm_per_ma", 0.001), get("friction_ma", 50.0)
        if self.nm_per_ma <= 0 or self.curl_full <= 0 or self.thumb_curl_full <= 0:
            raise ValueError("torque.nm_per_ma and the curl ranges must be positive")
        self.last = None  # (per-finger force N, joint command mA) for telemetry/tests

    def joint_commands(self, target_deg):
        np = self.np
        lo, hi = self.model.bounds
        q = np.clip(np.deg2rad(np.asarray(target_deg, dtype=float)), lo, hi)
        command = np.zeros(16)
        forces = []
        for tip, joints in self.FINGERS:
            curl = float(np.sum(np.asarray(target_deg, dtype=float)[list(joints)]))
            full = self.thumb_curl_full if tip == "thumb_tip" else self.curl_full
            u = min(max(curl / full, 0.0), 1.0)
            magnitude = self.close_n * u - self.open_n * (1.0 - u)
            J = self.model.link_jacobian(q, tip)[:, list(joints)]          # 3 x 2, m/rad
            direction = J @ np.ones(len(joints))
            norm = np.linalg.norm(direction)
            forces.append(magnitude)
            if norm < 1e-9:
                continue
            tau = J.T @ (magnitude * direction / norm)                      # N*m on the flexion joints
            for j, t in zip(joints, tau):
                ma = t / self.nm_per_ma
                command[j] = ma + (np.sign(ma) * self.friction_ma if abs(ma) > 1e-6 else 0.0)
        self.last = (forces, command.tolist())
        return command.tolist()


def load_force_law(options, side="Left"):
    return FingertipForceLaw(options, side)


def torque_options(cfg):
    """Optional top-level "torque" block: drive the hand in torque mode with a joint PD on the glove target."""
    block = cfg.get("torque")
    if block is None:
        return None
    if not isinstance(block, dict) or not isinstance(block.get("enabled"), bool):
        raise ValueError("torque must be an object with a boolean 'enabled'")
    options = {k: v for k, v in block.items() if k != "enabled"}
    unknown = set(options) - TORQUE_KEYS
    if unknown:
        raise ValueError("Unknown torque keys: " + ", ".join(sorted(unknown)))
    # The gains chosen in the GUI's Dev Mode page: it starts the controller with AAC_TORQUE_GAINS='{"kp": .., "kd": ..}',
    # which replaces the block's global kp / kd (per-joint torque.joints overrides keep their own values).
    gains = os.environ.get("AAC_TORQUE_GAINS", "").strip()
    if gains:
        try:
            gains = json.loads(gains)
        except ValueError as error:
            raise ValueError(f"AAC_TORQUE_GAINS is not JSON: {error}") from None
        if not isinstance(gains, dict) or set(gains) - {"kp", "kd"} or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in gains.values()):
            raise ValueError('AAC_TORQUE_GAINS must be {"kp": number, "kd": number}')
        options.update({k: float(v) for k, v in gains.items()})
        print("[aac] torque gains from AAC_TORQUE_GAINS: " + ", ".join(f"{k} {v:g}" for k, v in gains.items()), flush=True)
    if not block["enabled"]:
        return None
    open_loop = options.pop("open_loop", False)
    if not isinstance(open_loop, bool):
        raise ValueError("torque.open_loop must be a boolean")
    # RS-485 only: write this tick's torque and read the next pose in one pipelined exchange (see
    # AACHand.write_torques_and_read_positions); the PD then acts on the pose from the previous exchange.
    pipeline = options.pop("pipeline", False)
    if not isinstance(pipeline, bool):
        raise ValueError("torque.pipeline must be a boolean")
    # Damp with the hand's own joint velocity (input 16-31, v2 firmware, 1 kHz loop) instead of differentiating the
    # position over velocity_window_s, which lags by about half the window (~30 ms at 0.06 s).
    firmware_velocity = options.pop("firmware_velocity", False)
    if not isinstance(firmware_velocity, bool):
        raise ValueError("torque.firmware_velocity must be a boolean")
    law = options.pop("law", "pd")
    if law not in TORQUE_LAWS:
        raise ValueError("torque.law must be one of " + ", ".join(TORQUE_LAWS))
    center = options.pop("center", "start")
    if center not in TORQUE_CENTERS:
        raise ValueError("torque.center must be one of " + ", ".join(TORQUE_CENTERS))
    friction_table = options.pop("friction_table", None)
    breakaway_fine = options.pop("breakaway_fine", None)
    joints = options.pop("joints", None)
    options = {k: finite_number(v, f"torque.{k}") for k, v in options.items()}
    options["friction_table"] = friction_table
    if breakaway_fine is not None:
        if not isinstance(breakaway_fine, str) or not breakaway_fine:
            raise ValueError("torque.breakaway_fine must be a file path")
        breakaway_fine = resolve_data_path(breakaway_fine, what="torque.breakaway_fine")
        BreakawayModel(breakaway_fine)      # validate before any hardware is opened
    options["breakaway_fine"] = breakaway_fine
    if joints is not None:
        if not isinstance(joints, dict):
            raise ValueError("torque.joints must be an object {joint_name: {kp, kd, friction_scale, dither_ma}}")
        for name, block in joints.items():
            if name not in JOINT_NAMES or not isinstance(block, dict) or set(block) - TORQUE_JOINT_KEYS:
                raise ValueError(f"torque.joints[{name!r}]: unknown joint or keys; allowed {sorted(TORQUE_JOINT_KEYS)}")
            for k, v in block.items():
                if k == "target_max_deg":
                    if v is None:          # null: no limit for this joint
                        continue
                    lo, hi = ANGLE_LIMITS[JOINT_NAMES.index(name)]
                    if not lo < finite_number(v, f"torque.joints.{name}.{k}") <= hi:
                        raise ValueError(f"torque.joints.{name}.target_max_deg must be in ({lo}, {hi}] deg or null")
                    continue
                if finite_number(v, f"torque.joints.{name}.{k}") < 0:
                    raise ValueError(f"torque.joints.{name}.{k} must be non-negative")
            boost = set(block) & BOOST_KEYS
            if boost:
                j = JOINT_NAMES.index(name)
                if "boost_ma" not in block:
                    raise ValueError(f"torque.joints.{name}: {sorted(boost)} need boost_ma")
                if j not in dict(TORQUE_SINGLES):
                    raise ValueError(f"torque.joints.{name}.boost_ma: only joints driven by one motor can be boosted "
                                     f"(PIPs, thumb_cmc, thumb_mcp); {name} shares a motor pair")
                cap = 500.0 if dict(TORQUE_SINGLES)[j] < 4 else 1000.0     # thumb motors 1-4: 500 mA on v2
                tau_max = float(options.get("tau_max", 200.0))
                if not tau_max < float(block["boost_ma"]) <= cap:
                    raise ValueError(f"torque.joints.{name}.boost_ma must be above tau_max ({tau_max:g}) and at most "
                                     f"{cap:g} mA")
                if float(block.get("boost_max_s", 1.0)) <= 0:
                    raise ValueError(f"torque.joints.{name}.boost_max_s must be positive")
        options["joints"] = {name: {k: None if v is None else float(v) for k, v in block.items()}
                             for name, block in joints.items()}
    options["open_loop"] = open_loop
    options["pipeline"] = pipeline
    options["firmware_velocity"] = firmware_velocity
    options["center"] = center
    options["law"] = law
    if options.get("center_lag_s", 0.0) < 0:
        raise ValueError("torque.center_lag_s must be non-negative")
    if not 0 < options.get("tau_max", 200.0) <= 1000:
        raise ValueError("torque.tau_max must be in (0, 1000] register units (1 unit = 1 mA)")
    if options.get("kp", 3.0) < 0 or options.get("kd", 0.02) < 0:
        raise ValueError("torque.kp and torque.kd must be non-negative")
    if (options.get("friction_ma", 0.0) < 0 or options.get("kd_fast", 0.0) < 0 or options.get("v_fast_dps", 100.0) <= 0
            or options.get("v_stick_dps", 20.0) <= 0):
        raise ValueError("torque.friction_ma and torque.kd_fast must be non-negative, torque.v_fast_dps and "
                         "torque.v_stick_dps positive")
    if (options.get("dither_ma", 0.0) < 0 or options.get("dither_hz", 50.0) <= 0 or options.get("friction_ramp_deg", 4.0) <= 0
            or options.get("friction_scale", 0.4) < 0):
        raise ValueError("torque.dither_ma and torque.friction_scale must be non-negative, torque.dither_hz and "
                         "torque.friction_ramp_deg positive")
    if options.get("friction_ff_scale", 0.0) < 0 or options.get("friction_ff_ramp_dps", 10.0) <= 0 \
            or options.get("target_velocity_window_s", 0.03) <= 0:
        raise ValueError("torque.friction_ff_scale must be non-negative, torque.friction_ff_ramp_dps and "
                         "torque.target_velocity_window_s positive")
    if options.get("friction_ff_scale", 0.0) and options.get("friction_table") is None:
        raise ValueError("torque.friction_ff_scale needs torque.friction_table (the measured breakaway currents)")
    if options.get("friction_table") is not None:
        table = options["friction_table"]
        if not isinstance(table, str) or not table:
            raise ValueError("torque.friction_table must be a file path")
        table = resolve_data_path(table, what="torque.friction_table")
        load_friction_table(table, float(options.get("friction_scale", 0.4)))   # validate before any hardware is opened
        options["friction_table"] = table
    return options


def tracking_options(cfg):
    """Optional top-level "tracking" block (closed-loop tracking correction); see TrackingCorrection."""
    block = cfg.get("tracking")
    if block is None:
        return None
    if not isinstance(block, dict) or not isinstance(block.get("enabled"), bool):
        raise ValueError("tracking must be an object with a boolean 'enabled'")
    options = {k: v for k, v in block.items() if k != "enabled"}
    unknown = set(options) - TRACKING_KEYS
    if unknown:
        raise ValueError("Unknown tracking keys: " + ", ".join(sorted(unknown)))
    if not block["enabled"]:
        return None
    TrackingCorrection(**options)  # validate before any hardware is opened
    return options


def load_compliance(options):
    """Position-mode compliance from dex-retargetor-cpp (needs its build/libdexrt.so)."""
    sys.path.insert(0, str(REPO_ROOT / "retargetor" / "dex-retargetor-cpp" / "python"))
    import dexrt
    return dexrt.Compliance(**options)


def admittance_options(cfg, config_path):
    """Optional top-level "admittance" block: {"enabled": bool, "params": "<yaml>"} (teleop/hand_control/admittance.py).

    Returns the resolved YAML path (relative to the hand config) or None. The YAML is loaded here so a
    bad file fails before any hardware is opened; the YAML's own "enabled: false" also disables it.
    """
    block = cfg.get("admittance")
    if block is None:
        return None
    if not isinstance(block, dict) or not isinstance(block.get("enabled"), bool):
        raise ValueError("admittance must be an object with a boolean 'enabled'")
    unknown = set(block) - {"enabled", "params"}
    if unknown:
        raise ValueError("Unknown admittance keys: " + ", ".join(sorted(unknown)))
    if not block["enabled"]:
        return None
    params = block.get("params", "admittance.yaml")
    if not isinstance(params, str) or not params:
        raise ValueError("admittance.params must be a YAML file path")
    try:
        path = Path(resolve_data_path(params, config_path, what="admittance.params"))
    except FileNotFoundError as error:
        raise ValueError(str(error)) from None
    return str(path) if load_admittance(path) is not None else None


def compliance_options(cfg):
    """Optional top-level "compliance" block; absent or enabled=false keeps plain position control."""
    block = cfg.get("compliance")
    if block is None:
        return None
    if not isinstance(block, dict) or not isinstance(block.get("enabled"), bool):
        raise ValueError("compliance must be an object with a boolean 'enabled'")
    options = {k: v for k, v in block.items() if k != "enabled"}
    unknown = set(options) - COMPLIANCE_KEYS
    if unknown:
        raise ValueError("Unknown compliance keys: " + ", ".join(sorted(unknown)))
    return {k: finite_number(v, f"compliance.{k}") for k, v in options.items()} if block["enabled"] else None


def load_aac_config(args, path):
    cfg = load_hand_config(path)
    key = getattr(args, "hand_serial", "") or "AAC-001"
    role = resolve_hand_role(cfg, key)
    if not isinstance(role, dict) or not role:
        raise ValueError(f"Missing AAC hand_roles entry for {key}")
    hand_id = role.get("role")
    if not isinstance(hand_id, str) or not hand_id.strip():
        raise ValueError("AAC role must be a nonempty logical hand_id")
    for name in ("hand_id", "hand_role"):
        override = getattr(args, name, None)
        if override and override != hand_id:
            raise ValueError(f"--{name} must match configured role {hand_id}")
    for name in ("serial_device", "device_address"):
        if name not in role:
            raise ValueError(f"AAC role requires {name}")
    # The adapter chosen in the GUI's Dev Mode page: it starts the controller and the demos with AAC_SERIAL_DEVICE set,
    # which replaces the config's serial_device (the rest of the config is unchanged).
    chosen = os.environ.get("AAC_SERIAL_DEVICE", "").strip()
    if chosen:
        role = dict(role, serial_device=chosen)
    settings = cfg.get("aac_settings")
    required = {"connection_timeout_s", "control_rate_hz", "feedback_rate_hz",
                "command_timeout_s", "finger_speeds_percent"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError("aac_settings requires exactly: " + ", ".join(sorted(required)))
    settings = dict(settings)
    for name in required - {"finger_speeds_percent"}:
        settings[name] = positive_number(settings[name], name)
    settings["finger_speeds_percent"] = validate_speeds(settings["finger_speeds_percent"])
    if getattr(args, "dt", None) is not None:
        settings["control_rate_hz"] = 1 / positive_number(args.dt, "--dt")
    if getattr(args, "telem_rate", None) is not None:
        settings["feedback_rate_hz"] = positive_number(args.telem_rate, "--telem_rate")
    if getattr(args, "command_timeout", None) is not None:
        settings["command_timeout_s"] = positive_number(args.command_timeout, "--command_timeout")
    if settings["feedback_rate_hz"] > settings["control_rate_hz"]:
        raise ValueError("feedback_rate_hz must not exceed control_rate_hz")
    # Bus: "canfd" (default) or "rs485" per role; the RS-485 baud rate follows the firmware (v2 4 Mbaud).
    bus = role.get("bus", "canfd")
    # Validate connection fields before creating sockets or opening the port.
    AACHand(role["serial_device"], role["device_address"], settings["connection_timeout_s"], bus=bus)
    settings.update(hand_id=hand_id, serial_device=role["serial_device"],
                    device_address=role["device_address"], bus=bus)
    # Sockets: the top-level transport, overridden per role by an optional role["transport"]["zmq"] block so one
    # config can hold both hands (run the controller once per role with --hand_serial).
    transport = dict(resolve_zmq_transport(cfg), **resolve_zmq_transport(role))
    for dest, source, port in (("stream_addr", "hand_stream_pull", 6300),
                               ("control_addr", "hand_control_rep", 6301),
                               ("telemetry_addr", "hand_telem_pub", 6302)):
        endpoint = transport.get(source, f"tcp://0.0.0.0:{port}")
        if not isinstance(endpoint, str) or not endpoint:
            raise ValueError(f"{source} must be a ZMQ endpoint")
        settings[dest] = endpoint
    compliance = compliance_options(cfg)
    if compliance is not None:
        settings["compliance"] = compliance
    tracking = tracking_options(cfg)
    if tracking is not None:
        if compliance is not None:
            raise ValueError("compliance and tracking cannot both be enabled")
        settings["tracking"] = tracking
    torque = torque_options(cfg)
    if torque is not None:
        if compliance is not None or tracking is not None:
            raise ValueError("torque cannot be combined with compliance or tracking")
        settings["torque"] = torque
    admittance = admittance_options(cfg, path)
    if admittance is not None:
        if compliance is not None or tracking is not None or torque is not None:
            raise ValueError("admittance cannot be combined with compliance, tracking or torque")
        settings["admittance"] = admittance
    return settings


class LoopStats:
    """Achieved control-loop rate over the last window_s: published in telemetry ("loop") every control tick, so
    a config's control_rate_hz can be checked against what the bus actually delivers.

    hz: control ticks per second achieved. busy_ms: time spent inside a tick (bus I/O + law), median and max.
    interval_ms_max: longest gap between two tick starts (jitter). overrun: share of ticks whose work took longer
    than the period, i.e. the configured rate is not reachable. Totals since start: ticks, overruns."""

    def __init__(self, period, window_s=1.0):
        from collections import deque
        self.period, self.window_s = float(period), float(window_s)
        self.samples = deque()   # (tick start, busy seconds)
        self.ticks = self.overruns = 0

    def record(self, start, end):
        busy = end - start
        self.samples.append((start, busy))
        self.ticks += 1
        self.overruns += busy > self.period
        while self.samples and self.samples[0][0] < start - self.window_s:
            self.samples.popleft()

    def summary(self):
        if len(self.samples) < 2:
            return {"target_hz": round(1 / self.period, 1), "hz": None, "ticks": self.ticks, "overruns": self.overruns}
        starts = [s for s, _ in self.samples]
        busy = sorted(b for _, b in self.samples)
        gaps = [b - a for a, b in zip(starts, starts[1:])]
        return {"target_hz": round(1 / self.period, 1),
                "hz": round((len(starts) - 1) / (starts[-1] - starts[0]), 1) if starts[-1] > starts[0] else None,
                "busy_ms_p50": round(1000 * busy[len(busy) // 2], 2), "busy_ms_max": round(1000 * busy[-1], 2),
                "interval_ms_max": round(1000 * max(gaps), 2),
                "overrun": round(sum(b > self.period for b in busy) / len(busy), 3),
                "ticks": self.ticks, "overruns": self.overruns}


class AACController:
    def __init__(self, device, *, hand_id, stream_addr, control_addr, telemetry_addr,
                 control_rate_hz, feedback_rate_hz, command_timeout_s,
                 finger_speeds_percent, context=None, clock=time, compliance=None,
                 compliance_factory=load_compliance, tracking=None, torque=None,
                 torque_factory=load_torque_controller, force_law_factory=load_force_law,
                 admittance=None, admittance_factory=load_admittance):
        self.device = device
        # Current-based admittance (position mode): q_cmd = target + offset driven by the motor currents.
        self.admittance_path = admittance
        self.admittance_factory = admittance_factory
        self.admittance = None
        self.admittance_offset = None
        self.external_torque = None
        # Torque mode: a joint PD on the glove target writes motor torques every control tick.
        self.torque_options = torque
        self.torque_factory = torque_factory
        self.force_law_factory = force_law_factory
        self.force_law = None
        self.torque = None
        self.torque_command = None
        self.torque_zeroed = True
        self.torque_last_pose = None
        self.torque_pose_changed_at = None
        self.torque_frozen_events = 0
        self.loop_pose_at = None       # monotonic time a closed loop last refreshed self.measured itself
        self.friction_table = None     # per-joint smooth friction feed-forward (mA) when torque.friction_table is set
        if torque is not None and torque.get("friction_table"):
            self.friction_table = load_friction_table(torque["friction_table"],
                                                      per_joint(torque, "friction_scale", float(torque.get("friction_scale", 0.4))))
        # Target-velocity friction feed-forward (torque.friction_ff_scale, per joint under torque.joints).
        self.ff_scale = None
        self.target_velocity = None
        if torque is not None and torque.get("friction_table"):
            ff = per_joint(torque, "friction_ff_scale", float(torque.get("friction_ff_scale", 0.0)))
            if any(ff):
                self.ff_scale = ff
                self.breakaway = load_breakaway(torque["friction_table"])
                self.target_velocity = TargetVelocity(float(torque.get("target_velocity_window_s", 0.03)))
        # Angle-dependent breakaway (fine table) for the feed-forward and the error-based term, when available.
        self.breakaway_model = None
        if torque is not None and torque.get("breakaway_fine"):
            self.breakaway_model = BreakawayModel(torque["breakaway_fine"])
        # Opening boost for joints that stick deep in their curl (torque.joints.<joint>.boost_ma); empty = off.
        self.boost = OpeningBoost(torque) if torque is not None else None
        # Per-joint upper target limits (torque.joints.<joint>.target_max_deg; null or absent = none).
        self.target_max = target_maxima(torque) if torque is not None else None
        self.pipeline = False          # set in open() when torque.pipeline is on and the device supports it
        self.pipeline_pose = None      # (monotonic time, pose, velocity or None) fetched with the last torque write
        self.firmware_velocity = False  # set in open() when torque.firmware_velocity is on and the device supports it
        self.torque_center = None  # open loop: command-frame pose the torque is referenced to
        self.torque_target_history = []  # (monotonic time, target) for center "previous"
        # Closed-loop tracking correction (position mode): command = target + correction(target - measured).
        self.tracking = TrackingCorrection(**tracking) if tracking is not None else None
        self.tracking_correction = None
        # Position-mode compliance: when configured, every control tick re-reads the hand and
        # writes the compliant command instead of the raw target.
        self.compliance_options = compliance
        self.compliance_factory = compliance_factory
        self.compliance = None
        self.compliance_offset = None
        self.hand_id = hand_id
        self.readback_sign = readback_sign(hand_id)  # raw readback -> command frame; differs per hand
        self.pair_polarity = pair_polarity(hand_id)   # torque mode: motor-pair polarity, differs per hand
        if self.tracking is not None:
            self.tracking.readback_sign = list(self.readback_sign)
        self.addresses = (stream_addr, control_addr, telemetry_addr)
        self.period = 1 / positive_number(control_rate_hz, "control_rate_hz")
        self.loop_stats = LoopStats(self.period)
        self.feedback_period = 1 / positive_number(feedback_rate_hz, "feedback_rate_hz")
        self.timeout = positive_number(command_timeout_s, "command_timeout_s")
        self.speeds = validate_speeds(finger_speeds_percent)
        self.context = context
        self.owns_context = context is None
        self.clock = clock
        self.sockets = []
        self.execution_enabled = False
        self.execution_id = None
        self.stopping = False
        self.target = None
        self.target_pending = False
        self.execute_deadline = None
        self.accept_after = float("-inf")
        self.measured = self.measured_t = None
        self.commanded = self.commanded_t = None
        self.next_control = self.next_feedback = 0
        self.rejected_targets = 0
        self.last_rejection = None

    def open(self):
        self.measured = self.device.open()
        self.measured_t = self.clock.time()
        self.set_readback_sign(readback_sign(self.hand_id, getattr(self.device, "register_map", None)))
        self.device.set_position_mode()
        self.device.set_speeds(self.speeds)
        if self.torque_options is not None:
            # Vendor order: mode 2 first, then torque writes (only valid in torque mode).
            self.torque = self.torque_factory(self.torque_options)
            if self.torque_options.get("law") == "force":
                side = "Right" if "Right" in self.hand_id else "Left"
                self.force_law = self.force_law_factory(self.torque_options, side)
            # self.measured is the live pose read in position mode a moment ago (open() reads before switching).
            self.torque_center = ([s * m for s, m in zip(self.readback_sign, self.measured)]
                                  if self.torque_options.get("center", "start") == "start" else list(TORQUE_OPEN_LOOP_CENTER))
            self.device.set_torque_mode()
            self.device.write_motor_torques([0] * 16)
            self.torque_zeroed = True
            if self.torque_options.get("firmware_velocity"):
                can = getattr(self.device, "can_read_velocity", None)
                self.firmware_velocity = bool(can and can())
                print("[aac] torque damping: firmware joint velocity (input 16-31)" if self.firmware_velocity else
                      "[aac] torque.firmware_velocity requested but this firmware has no signed velocity readback; "
                      "differentiating the position instead")
            velocity_source = ("firmware joint velocity" if self.firmware_velocity else
                               f"position difference over {self.torque_options.get('velocity_window_s', 0.1)} s")
            if self.torque_options.get("pipeline"):
                can = getattr(self.device, "can_pipeline", None)
                self.pipeline = bool(can and can())
                if not self.pipeline:
                    print("[aac] torque.pipeline requested but this hand/bus cannot pipeline (needs RS-485 and the v2 "
                          "firmware); using the sequential exchange")
            # Tick order. pipeline true: the PD uses the pose and velocity fetched together with the PREVIOUS torque
            # write (one round trip per tick, ~360 Hz, data ~3 ms older). pipeline false: read the pose and velocity
            # first, then compute the PD from them, then write (two round trips, ~230 Hz, freshest data).
            print(f"[aac] torque loop: {'pipelined: write torque + read state in one round trip, PD from the previous read' if self.pipeline else 'sequential: read state, then compute the PD, then write torque'}; "
                  f"damping speed from {velocity_source}")
            print(f"[aac] TORQUE MODE{' (FINGERTIP FORCE LAW, J^T F from the solved pose)' if self.force_law is not None else ' (OPEN LOOP, center = ' + self.torque_options.get('center', 'start') + ' pose)' if self.torque_options.get('open_loop') else ''}: "
                  f"kp {self.torque_options.get('kp', 3.0)} mA/deg, "
                  f"kd {self.torque_options.get('kd', 0.02)} mA/(deg/s), tau_max {self.torque_options.get('tau_max', 200.0)} mA, "
                  f"friction {self.torque_options.get('friction_ma', 0.0)} mA"
                  + (f" (table x{self.torque_options.get('friction_scale', 0.4)}, ramp {self.torque_options.get('friction_ramp_deg', 4.0)} deg)"
                     if self.friction_table is not None else "")
                  + f", dither {self.torque_options.get('dither_ma', 0.0)} mA at {self.torque_options.get('dither_hz', 50.0)} Hz; "
                  "zero torque until execute")
            if self.boost:
                print(f"[aac] OPENING BOOST: {self.boost.describe()}")
            if self.target_max is not None:
                print("[aac] TARGET LIMITS: " + ", ".join(f"{n} <= {m:g} deg" for n, m in zip(JOINT_NAMES, self.target_max)
                                                         if m is not None))
        if self.admittance_path is not None:
            self.admittance = self.admittance_factory(self.admittance_path)
            self.admittance.readback_sign = list(self.readback_sign)
            if self.admittance.calibration.needs_idle:
                # No free-motion current fit: use the idle current of the still, unloaded hand.
                self.admittance.set_idle_current([self.device.read_currents() for _ in range(COMPLIANCE_IDLE_SAMPLES)])
                print("[aac] admittance on without a full calibration: I_free = idle current, current signs +1 "
                      "(run retargetor/dex-retargetor-cpp/python/admittance_calibrate.py)")
            else:
                print(f"[aac] admittance on; calibrated joints: {self.admittance.describe()['calibrated_joints']}")
        if self.compliance_options is not None:
            self.compliance = self.compliance_factory(self.compliance_options)
            # The hand is still and unloaded at startup: learn each motor's idle current.
            for _ in range(COMPLIANCE_IDLE_SAMPLES):
                self.compliance.add_idle_sample(self.device.read_currents())
            idle = self.compliance.finish_idle()
            print(f"[aac] compliance on; idle motor current (A): {[round(float(v), 3) for v in idle]}")
        if self.context is None:
            self.context = zmq.Context()
        for kind, addr in zip((zmq.PULL, zmq.REP, zmq.PUB), self.addresses):
            sock = self.context.socket(kind)
            self.sockets.append(sock)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.RCVHWM, MAX_DRAIN)
            sock.setsockopt(zmq.SNDHWM, MAX_DRAIN)
            sock.setsockopt(zmq.SNDTIMEO, 100)
            sock.bind(addr)
        self.stream, self.control, self.telemetry = self.sockets
        self.poller = zmq.Poller()
        self.poller.register(self.stream, zmq.POLLIN)
        self.poller.register(self.control, zmq.POLLIN)
        self.device.check_health()
        self.idle()
        print(f"[aac] ready: {self.hand_id}, software-idle; no motor enable-state change")

    def accept_target(self, msg):
        self.watchdog()
        fields = {"type", "hand_id", "execution_id", "t", "angle_unit", "angle"}
        if not isinstance(msg, dict) or set(msg) != fields:
            raise ValueError("AAC hand_stream requires exactly type, hand_id, execution_id, t, angle_unit, angle")
        if msg["type"] != "hand_stream" or msg["hand_id"] != self.hand_id:
            raise ValueError("Unsupported stream type or wrong hand_id")
        if msg["angle_unit"] != "deg":
            raise ValueError("AAC angle_unit must be deg")
        if not isinstance(msg["execution_id"], str) or not msg["execution_id"]:
            raise ValueError("AAC execution_id must be a nonempty string from the execute reply")
        stamp = finite_number(msg["t"], "t")
        angle = validate_positions(msg["angle"])
        now = self.clock.time()
        age = now - stamp
        if age < -FUTURE_LEAD_S:
            raise ValueError("AAC target timestamp is more than 0.05 s in the future")
        if not self.execution_enabled:
            return self.reject_target("software idle")
        if msg["execution_id"] != self.execution_id:
            return self.reject_target("wrong execution session")
        if age >= self.timeout:
            return self.reject_target("stale target")
        if stamp <= self.accept_after:
            return self.reject_target("target predates execution boundary")
        if self.target is not None and stamp <= self.target[0]:
            return self.reject_target("out-of-order target")
        # Preserve source age; rejected packets never refresh either deadline.
        expires = self.clock.monotonic() + self.timeout - max(0, age)
        self.target = (stamp, angle, expires)
        self.target_pending = True
        return True

    def reject_target(self, reason):
        self.rejected_targets = min(self.rejected_targets + 1, 2**31 - 1)
        self.last_rejection = reason
        return False

    def fresh(self):
        return (self.target is not None
                and self.clock.monotonic() < self.target[2]
                and self.clock.time() - self.target[0] < self.timeout
                and self.target[0] - self.clock.time() <= FUTURE_LEAD_S)

    def idle(self):
        self.execution_enabled = False
        self.execution_id = None
        self.target = None
        self.target_pending = False
        self.execute_deadline = None
        self.accept_after = max(self.accept_after, self.clock.time())
        if self.compliance is not None:
            self.compliance.reset()  # a new session starts from the raw target again
            self.compliance_offset = None
        if self.admittance is not None:
            self.admittance.reset()
            self.admittance_offset = None
            self.external_torque = None
        if self.tracking is not None:
            self.tracking.reset()
            self.tracking_correction = None
        if self.torque is not None:
            self.torque.reset()
            self.torque_command = None
            self.torque_target_history = []
            self.zero_torque()

    def set_readback_sign(self, sign):
        """Raw readback -> command frame, for every loop that reads the pose."""
        self.readback_sign = list(sign)
        if self.tracking is not None:
            self.tracking.readback_sign = list(sign)
        if self.admittance is not None:
            self.admittance.readback_sign = list(sign)

    def zero_torque(self):
        self.pipeline_pose = None
        if getattr(self, "target_velocity", None) is not None:
            self.target_velocity.reset()
        if self.torque is not None and not self.torque_zeroed:
            self.device.write_motor_torques([0] * 16)
            self.torque_zeroed = True

    def torque_write(self):
        """Every control tick while executing: PD on the measured pose, write motor torques."""
        if not (self.execution_enabled and self.target is not None and self.fresh()):
            self.zero_torque()
            return
        if self.force_law is not None:
            # Fingertip force law: open loop, needs no position read (positions freeze in torque mode).
            joint = self.force_law.joint_commands(self.target[1])
            command = joint_to_motor_torques(joint, float(self.torque_options.get("tau_max", 200.0)), self.pair_polarity)
            self.device.write_motor_torques(command)
            self.torque_zeroed = all(v == 0 for v in command)
            self.torque_command = command
            self.commanded = self.target[1]
            self.commanded_t = self.clock.time()
            self.target_pending = False
            return
        piped = self.pipeline_pose
        velocity = None
        if self.pipeline and piped is not None and self.clock.monotonic() - piped[0] <= PIPELINE_POSE_MAX_AGE_S:
            measured, velocity = piped[1], piped[2]   # fetched together with the previous torque write
        elif self.firmware_velocity:
            measured, velocity = self.device.read_state()
            self.measured, self.measured_t = measured, self.clock.time()
        else:
            measured = self.device.read_positions()
            self.measured, self.measured_t = measured, self.clock.time()
        self.loop_pose_at = self.clock.monotonic()
        self.watchdog()  # The read is blocking; the target can expire meanwhile.
        if self.stopping or not (self.execution_enabled and self.target is not None and self.fresh()):
            self.zero_torque()
            return
        # The firmware has been seen to freeze joint positions in torque mode. A PD on a frozen pose
        # is a constant push, not control: stop pushing if the pose never changes while we push.
        now = self.clock.monotonic()
        if measured != self.torque_last_pose or self.torque_pose_changed_at is None:
            self.torque_last_pose, self.torque_pose_changed_at = list(measured), now
        open_loop = bool(self.torque_options.get("open_loop", False))
        stale_s = 0.0 if open_loop else float(self.torque_options.get("stale_feedback_s", STALE_FEEDBACK_S))
        if stale_s > 0 and not self.torque_zeroed and now - self.torque_pose_changed_at > stale_s:
            print(f"[aac] torque mode: joint positions unchanged for {stale_s:.1f} s while pushing; "
                  "feedback is frozen, going idle (zero torque)")
            self.idle()
            self.last_rejection = "frozen position feedback in torque mode"
            self.torque_frozen_events += 1
            return
        # Command frame: finger abduction reads back mirrored. Open loop: fixed mid-range instead.
        if open_loop and self.torque_options.get("center") == "previous":
            # Reference = the glove target from center_lag_s ago (previous frame when 0).
            lag = float(self.torque_options.get("center_lag_s", 0.0))
            history = self.torque_target_history
            previous = history[0][1] if history else self.target[1]
            for stamp, target in history:
                if stamp <= now - lag:
                    previous = target
            history.append((now, list(self.target[1])))
            while len(history) > 1 and history[1][0] <= now - lag:
                history.pop(0)
            q = previous
        else:
            q = self.torque_center if open_loop else [s * m for s, m in zip(self.readback_sign, measured)]
        options = self.torque_options
        # Target smoothing belongs to the sender's one-euro filter (teleop/manus/parameters/teleop_filter.yaml), not here.
        target = limit_targets(self.target[1], self.target_max)
        self.torque.set_target(target, now)
        qdot_in = None
        if self.firmware_velocity and velocity is not None and not open_loop:
            qdot_in = [s * v for s, v in zip(self.readback_sign, velocity)]   # same frame as q
        tau, qdot, _ = self.torque.step(q, now, qdot=qdot_in)
        error = [a - b for a, b in zip(target, q)]
        tau = [float(v) for v in tau]
        if self.friction_table is not None:
            table = self.friction_table
            if self.breakaway_model is not None:
                # scale the per-joint share by the breakaway at the current angle (mean of the two directions)
                shares = per_joint(options, "friction_scale", float(options.get("friction_scale", 0.4)))
                table = [s * 0.5 * sum(self.breakaway_model.both(j, qj)) for j, (s, qj) in enumerate(zip(shares, q))]
            tau = smooth_friction(tau, error, table, float(options.get("friction_ramp_deg", 4.0)))
        if self.ff_scale is not None:
            v_target = self.target_velocity.update(now, target)
            plus, minus = self.breakaway
            if self.breakaway_model is not None:
                plus = [self.breakaway_model.current(j, qj, 1) for j, qj in enumerate(q)]
                minus = [self.breakaway_model.current(j, qj, -1) for j, qj in enumerate(q)]
            tau = friction_feedforward(tau, v_target, plus, minus, self.ff_scale,
                                       float(options.get("friction_ff_ramp_dps", 10.0)))
        elif options.get("friction_ma", 0.0) or options.get("kd_fast", 0.0):
            tau = pd_extras(tau, error, [float(v) for v in qdot],
                            friction_ma=float(options.get("friction_ma", 0.0)), kd_fast=float(options.get("kd_fast", 0.0)),
                            v_fast_dps=float(options.get("v_fast_dps", 100.0)), deadband_deg=float(options.get("deadband_deg", 1.0)),
                            v_stick_dps=float(options.get("v_stick_dps", 20.0)))
        limit = float(options.get("tau_max", 200.0))
        command = joint_to_motor_torques(tau, limit, self.pair_polarity)
        if options.get("dither_ma", 0.0) > 0:
            # Only where it helps: a joint that is stuck (|qdot| below v_stick_dps) with an error to close. A joint
            # that is moving, or one at rest on target, gets no dither (glove sessions: the hand buzzed all the time).
            deadband = float(options.get("deadband_deg", 1.0))
            v_stick = float(options.get("v_stick_dps", 20.0))
            amplitude = per_joint(options, "dither_ma", float(options["dither_ma"]))
            stuck = [abs(e) > deadband and abs(float(v)) < v_stick and a > 0 for e, v, a in zip(error, qdot, amplitude)]
            command = dither(command, float(options["dither_ma"]), float(options.get("dither_hz", 50.0)), now, limit,
                             active=motor_active(stuck))
        if self.boost:
            command = self.boost.apply(list(command), q, error, qdot, now, float(options.get("v_stick_dps", 20.0)))
        if self.pipeline:
            if self.firmware_velocity:
                pose, vel, _ = self.device.write_torques_and_read_state(command)
            else:
                (pose, _), vel = self.device.write_torques_and_read_positions(command), None
            self.pipeline_pose = (self.clock.monotonic(), pose, vel)
            self.measured, self.measured_t = pose, self.clock.time()
            self.loop_pose_at = self.clock.monotonic()
        else:
            self.device.write_motor_torques(command)
        self.torque_zeroed = all(v == 0 for v in command)
        self.torque_command = command
        self.commanded = target            # what the PD actually tracked: the sender's target after the limits
        self.commanded_t = self.clock.time()
        self.target_pending = False

    def closed_loop_write(self, compute):
        """Every control tick while executing: re-read the hand, write compute(measured) if it changed."""
        if not (self.execution_enabled and self.target is not None and self.fresh()):
            return
        if (self.compliance or self.admittance) is not None:
            read_both = getattr(self.device, "read_positions_and_currents", None)
            if read_both is not None:
                measured, currents = read_both()
            else:
                measured, currents = self.device.read_positions(), self.device.read_currents()
        else:
            measured, currents = self.device.read_positions(), None
        self.measured, self.measured_t = measured, self.clock.time()
        self.loop_pose_at = self.clock.monotonic()
        self.watchdog()  # Reads are blocking; the target can expire meanwhile.
        if self.stopping or not (self.execution_enabled and self.target is not None and self.fresh()):
            return
        command = validate_positions(compute(measured, currents))
        if (self.target_pending or self.commanded is None
                or max(abs(a - b) for a, b in zip(command, self.commanded)) >= self.command_resolution()):
            self.device.write_positions(command)
            self.commanded = command
            self.commanded_t = self.clock.time()
        self.target_pending = False

    def command_resolution(self):
        """Smallest change worth re-sending: one position register step of the hand's firmware."""
        resolution = getattr(self.device, "command_resolution_deg", None)
        return resolution if isinstance(resolution, float) and resolution > 0 else COMMAND_RESOLUTION_DEG

    def compliant_command(self, measured, currents):
        out = self.compliance.step(self.target[1], measured, currents, self.clock.monotonic())
        self.compliance_offset = [float(v) for v in out["offset"]]
        return [float(v) for v in out["q_cmd"]]

    def admittance_command(self, measured, currents):
        out = self.admittance.step(self.target[1], measured, currents, self.clock.monotonic())
        self.admittance_offset = [float(v) for v in out["offset_deg"]]
        self.external_torque = [float(v) for v in out["tau_ext"]]
        return [float(v) for v in out["q_cmd"]]

    def tracking_command(self, measured, currents):
        command, correction = self.tracking.step(self.target[1], measured, self.clock.monotonic())
        self.tracking_correction = correction
        return command

    def request(self, req):
        # Validate RPC structure before touching execution state or the device.
        if not isinstance(req, dict):
            return {"ok": False, "err": "Request must be an object"}
        cmd = req.get("cmd")
        allowed = {"cmd", "hand_id", "mode"} if cmd == "set_mode" else {"cmd", "hand_id"}
        if set(req) - allowed or req.get("hand_id", self.hand_id) != self.hand_id:
            return {"ok": False, "err": "Unsupported fields or wrong hand_id"}
        if cmd not in ("noop", "shutdown", "set_mode"):
            return {"ok": False, "err": "Unsupported cmd"}
        if cmd == "noop":
            return {"ok": True}
        if cmd == "shutdown":
            self.idle()
            self.stopping = True
            return {"ok": True}
        mode = req.get("mode")
        if mode not in ("execute", "idle"):
            return {"ok": False, "err": "mode must be execute or idle"}
        if mode == "idle":
            self.idle()
        else:
            self.watchdog()
            if self.stopping:
                return {"ok": False, "err": "Controller is shutting down"}
            if not self.execution_enabled:
                self.idle()  # New authorization requires a new source timestamp.
                # Timestamps alone cannot isolate sessions with tolerated clock skew.
                self.execution_id = uuid.uuid4().hex
                self.execution_enabled = True
                self.execute_deadline = self.clock.monotonic() + self.timeout
        return {"ok": True, "execution_enabled": self.execution_enabled,
                "execution_id": self.execution_id}

    def watchdog(self):
        if self.execution_enabled:
            expired = (not self.fresh() if self.target is not None else
                       self.clock.monotonic() >= self.execute_deadline)
            if expired:
                self.idle()

    def tick(self):
        if self.stopping:
            return
        self.device.check_health()
        self.watchdog()
        if self.clock.monotonic() >= self.next_feedback:
            # Telemetry pose at feedback_rate_hz. The closed loops (torque, tracking, compliance, admittance) read
            # the pose themselves every control tick; a separate read then only costs a bus round trip. Plain
            # position mode never sets loop_pose_at, so its feedback read (and the checks around it) is unchanged.
            if self.loop_pose_at is None or self.clock.monotonic() - self.loop_pose_at >= self.feedback_period:
                self.measured = self.device.read_positions()
                self.measured_t = self.clock.time()
            self.next_feedback = self.clock.monotonic() + self.feedback_period
        self.watchdog()  # Reads are blocking; cached targets can expire meanwhile.
        if self.stopping:
            return
        if self.clock.monotonic() >= self.next_control:
            tick_start = self.clock.monotonic()
            self.device.check_health()
            self.watchdog()
            if self.stopping:
                return
            if self.torque is not None:
                self.torque_write()
            elif self.compliance is not None:
                self.closed_loop_write(self.compliant_command)
            elif self.admittance is not None:
                self.closed_loop_write(self.admittance_command)
            elif self.tracking is not None:
                self.closed_loop_write(self.tracking_command)
            elif (self.execution_enabled and self.target_pending and self.fresh()
                    and not self.stopping):
                angle = self.target[1]
                self.device.write_positions(angle)
                self.commanded = angle
                self.commanded_t = self.clock.time()
                self.target_pending = False
            self.watchdog()  # A write may also consume the remaining lifetime.
            self.loop_stats.record(tick_start, self.clock.monotonic())
            telemetry = {
                "type": "hand_telemetry", "hand_id": self.hand_id, "angle_unit": "deg",
                "joint_names": JOINT_NAMES, "t": self.clock.time(),
                "measured_angle": self.measured, "measured_t": self.measured_t,
                "readback_sign": self.readback_sign,   # raw measured_angle * sign = command frame
                "commanded_angle": self.commanded, "commanded_t": self.commanded_t,
                "execution_enabled": self.execution_enabled,
                "execution_id": self.execution_id,
                "rejected_targets": self.rejected_targets, "last_rejection": self.last_rejection,
                "loop": self.loop_stats.summary(),
            }
            if self.boost:
                telemetry["boost"] = self.boost.summary()
            if self.compliance is not None:
                telemetry["compliance_offset"] = self.compliance_offset
            if self.admittance is not None:
                telemetry["admittance_offset"] = self.admittance_offset
                telemetry["external_torque_nm"] = self.external_torque
            if self.tracking is not None:
                telemetry["tracking_correction"] = self.tracking_correction
            if self.torque is not None:
                telemetry["torque_command"] = self.torque_command
            self.telemetry.send_json(telemetry)
            if self.loop_pose_at is None:
                # Position mode: schedule from completion, slow I/O must not cause catch-up bursts.
                self.next_control = self.clock.monotonic() + self.period
            else:
                # Closed loops block on bus I/O every tick: schedule from the tick's start, but never earlier than
                # now, so a tick whose I/O outlasts the period is followed by the next one at once (no dead time)
                # and still no burst. Scheduling from completion added the period to every tick's I/O (RS-485
                # torque loop: 5.8 ms per tick, 172 Hz, for 2.8 ms of work and a 2.9 ms period).
                self.next_control = max(tick_start + self.period, self.clock.monotonic())

    def step(self, wait_ms=20):
        events = dict(self.poller.poll(wait_ms))
        self.device.check_health()
        self.watchdog()
        if self.control in events:
            try:
                req = self.control.recv_json()
            except (ValueError, UnicodeError) as error:
                reply = {"ok": False, "err": f"Invalid JSON: {error}"}
            else:
                reply = self.request(req)
            self.control.send_json(reply)
        if self.stopping:
            return
        for _ in range(MAX_DRAIN):
            try:
                msg = self.stream.recv_json(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            self.accept_target(msg)
        self.tick()

    def leave_torque_mode(self):
        """Zero torque, hold the measured pose as the position target, position mode. Prevents the
        snap to a stale position target that a bare mode switch causes."""
        self.device.write_motor_torques([0] * 16)
        self.torque_zeroed = True
        pose = [min(max(v, lo), hi) for v, (lo, hi) in zip(self.device.read_positions(), ANGLE_LIMITS.values())]
        self.device.write_positions(pose)
        self.device.set_position_mode()
        print("[aac] torque zeroed, position mode restored at the measured pose")
        if self.pipeline:
            print(f"[aac] pipelined RS-485 ticks: {getattr(self.device, 'pipeline_ok', 0)}, "
                  f"sequential fallbacks: {getattr(self.device, 'pipeline_fallbacks', 0)}")

    def close(self, primary=None):
        s = self.loop_stats.summary()
        if s["ticks"]:
            print(f"[aac] control loop: {s['ticks']} ticks, last second {s['hz']} Hz of {s['target_hz']} Hz configured, "
                  f"{s['overruns']} ticks longer than the period")
        if self.boost:
            print("[aac] opening boost episodes: " + ", ".join(f"{name} {v['episodes']}" for name, v in self.boost.summary().items()))
        self.stopping = True
        self.idle()
        actions = [self.device.close]
        if self.torque is not None:
            actions.insert(0, self.leave_torque_mode)
        actions.extend(sock.close for sock in self.sockets)
        if self.owns_context and self.context is not None:
            actions.append(self.context.term)
        cleanup(actions, primary=primary)

    def start(self):
        old_handlers = {}
        def stop_signal(signum, frame):
            # Do not mutate target/deadline state midway through a freshness check.
            # The loop observes this intent before writing, then close() idles.
            self.stopping = True
        try:
            for signum in (signal.SIGINT, signal.SIGTERM):
                old_handlers[signum] = signal.signal(signum, stop_signal)
            self.open()
            while not self.stopping:
                deadline = min(self.next_control, self.next_feedback)
                wait_ms = min(20, max(0, int(1000 * (deadline - self.clock.monotonic()))))
                self.step(wait_ms)
        finally:
            try:
                self.close(primary=sys.exc_info()[1])
            finally:
                for signum, handler in old_handlers.items():
                    signal.signal(signum, handler)
