"""Demo: every fingertip tracks its ellipse under Cartesian impedance control (torque mode).

Same ellipses as demo/mcp_ellipses.py (demo/parameters/ellipses_trajectory.yaml: a_mm, b_mm, hz, spin, phase per finger; MCP flexion +
abduction on the fingers, thumb abduction + cmc), but instead of streaming joint positions to the firmware, the
fingertip is pulled toward its moving target point by a spring and a damper:

    F = K (x_d - x) - D v                x, v: fingertip position and velocity from the URDF and the measured joints
    tau = J^T F / nm_per_ma              J: fingertip Jacobian of the finger's two ellipse joints (analytic, URDF)

x_d is the fingertip at the planned joint pose q_d(t) of mcp_ellipses. The damper is the teleop controller's form
(kp (q_d - q) - kd qdot: it brakes the joint's own motion, no target velocity); damp_target_velocity: true in the
gains file changes it to D (v_d - v) with v_d = J(q_d) qdot_d, the planned tip velocity. Joints not on an ellipse
(PIPs, thumb rotation and MCP) hold their pose with the torque config's joint PD (kp, kd per joint). The torque
config's friction compensation is added (breakaway-scaled smooth term on the joint error, and the feed-forward on
the target's joint velocity, smoothed over the config's target_velocity_window_s exactly as in teleop, since the
calibrated friction_ff_scale was tuned with that smoothing), and the config's dither (dither_ma at dither_hz on stuck
joints, as in teleop) when dither: true in the gains file. Joint torques become motor torques through the motor pairs
(joint_to_motor_torques with the hand's pair polarity) and are capped at tau_max. Gains and safety limits are in
demo/parameters/impedance_ellipses.yaml.

Sequence: position mode rest -> spread -> ellipse centres (eased), then torque mode with the torque faded in over
gain_ramp_s, the passes (each starts and ends at the centres), then zero torque and position mode at the measured
pose, and an eased move back to rest. Safety: the demo writes zero torque and returns to position mode where the hand
stands when a fingertip is more than max_tip_error_mm from its target, when a joint runs faster than max_speed_dps
for max_speed_ticks ticks, on Ctrl-C and on any error. Neighbouring fingers are not checked (see demo/parameters/ellipses_trajectory.yaml).

    hand/.hand/bin/python demo/impedance_ellipses.py                       # right hand, CAN FD
    hand/.hand/bin/python demo/impedance_ellipses.py --fingers index --log /tmp/impedance_index.json
    hand/.hand/bin/python demo/impedance_ellipses.py --gains my_gains.yaml --params my_ellipses.yaml --loops 0
    hand/.hand/bin/python demo/impedance_ellipses.py --learn 5 --log /tmp/impedance_learn.json   # see below

Learned feed-forward (--learn N): the joints stick until the spring has built 0.4-0.8 deg of error and then jump
(runs 1-5, 2026-11-05), because no friction model fires the breakaway current at the right moment. The ellipses are
periodic, so the current each joint needs is a function of its finger's phase: N learning passes record, per finger
phase bin, the spring + damper torque the law had to add, and feed learn_gain x that mean forward on the next pass
(iterative learning control). The table (demo/calibration/impedance_feedforward_<side>.json, or --feedforward) is tied to the
ellipse settings it was learned on and is reloaded, frozen, by later runs; --reset-feedforward starts over. Learning
excludes the amplitude ramps at the ends of a pass and any pass that ends in a safety stop; values are smoothed over
neighbouring bins, kept to the first learn_harmonics harmonics of the turn (the friction profile along a periodic
trajectory is smooth; what a pass records beyond that is the stick-slip jitter of that pass, which fed forward makes
the next pass worse: left pinky, 2026-11-06) and capped at learn_max_ma. A saved table is filtered the same way when
it is loaded.
"""
import argparse
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "demo"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from hand.aac.model import HandModel  # noqa: E402
from hand.backends.aac import AACHand, ANGLE_LIMITS, JOINT_NAMES  # noqa: E402
from teleop.hand_control.hand_controller_aac import (BreakawayModel, TargetVelocity, dither, friction_feedforward,  # noqa: E402
                                        joint_to_motor_torques, load_aac_config, load_breakaway, load_friction_table,
                                        motor_active, pair_polarity, per_joint, readback_sign, smooth_friction)
from teleop.manus.stream.hand_stream_aac import joint_map  # noqa: E402
from finger_circles import DEFAULT_CONFIG, FINGERS, LIMITS, REST, Streamer, spread_pose  # noqa: E402
from mcp_ellipses import AXES, DEFAULT_PARAMS, build_plan, ellipses, load_params  # noqa: E402
from finger_circles import TipKinematics  # noqa: E402

DEFAULT_GAINS = ROOT / "demo" / "parameters" / "impedance_ellipses.yaml"
GAIN_KEYS = {"stiffness_n_per_m", "damping_ns_per_m", "damp_target_velocity", "fingers", "friction", "friction_ff_gain",
             "friction_error_gain",
             "dither", "dither_moving_ma", "dither_moving_ticks", "learn_gain", "learn_max_ma", "learn_bins",
             "learn_harmonics",
             "nm_per_ma", "gain_ramp_s", "velocity_window_s", "max_tip_error_mm", "max_speed_dps", "max_speed_ticks"}
FINGER_GAIN_KEYS = {"stiffness_n_per_m", "damping_ns_per_m"}
COUPLED = {"thumb_joint5": ("thumb_joint4", 0.85)}
COUPLED.update({f"{f}_joint4": (f"{f}_joint3", 0.8) for f in FINGERS[1:]})


def _number(value, name, minimum=0.0, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a number")
    if value < minimum or strict and value <= minimum:
        raise ValueError(f"{name} must be {'above' if strict else 'at least'} {minimum:g}")
    return float(value)


def load_gains(path):
    with open(path) as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict) or set(raw) != GAIN_KEYS:
        raise ValueError(f"{path} must contain exactly: {', '.join(sorted(GAIN_KEYS))}")
    g = {k: _number(raw[k], k) for k in ("stiffness_n_per_m", "damping_ns_per_m", "gain_ramp_s", "friction_ff_gain",
                                         "friction_error_gain")}
    for k in ("nm_per_ma", "velocity_window_s", "max_tip_error_mm", "max_speed_dps"):
        g[k] = _number(raw[k], k, strict=True)
    g["dither_moving_ma"] = _number(raw["dither_moving_ma"], "dither_moving_ma")
    if g["dither_moving_ma"] > 100:
        raise ValueError("dither_moving_ma must be at most 100")
    for key, minimum in (("max_speed_ticks", 1), ("learn_bins", 8), ("dither_moving_ticks", 1), ("learn_harmonics", 0)):
        value = raw[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be a whole number of at least {minimum}")
        g[key] = value
    g["learn_gain"] = _number(raw["learn_gain"], "learn_gain")
    if g["learn_gain"] > 1:
        raise ValueError("learn_gain must be at most 1")
    g["learn_max_ma"] = _number(raw["learn_max_ma"], "learn_max_ma", strict=True)
    for key in ("friction", "dither", "damp_target_velocity"):
        if not isinstance(raw[key], bool):
            raise ValueError(f"{key} must be true or false")
        g[key] = raw[key]
    fingers = raw["fingers"] or {}
    if not isinstance(fingers, dict) or not set(fingers) <= set(FINGERS):
        raise ValueError(f"fingers may only name: {', '.join(FINGERS)}")
    g["fingers"] = {}
    for finger in FINGERS:
        block = fingers.get(finger) or {}
        if not isinstance(block, dict) or not set(block) <= FINGER_GAIN_KEYS:
            raise ValueError(f"fingers.{finger} may only set: {', '.join(sorted(FINGER_GAIN_KEYS))}")
        g["fingers"][finger] = {k: _number(block.get(k, g[k]), f"fingers.{finger}.{k}") for k in FINGER_GAIN_KEYS}
    return g


def _rotation(axis, angle):
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1 - c
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


class FingerChains:
    """Fingertip position (m) and the Jacobian of the finger's two ellipse joints (m per rad of the SDK joint), for
    one hand side, straight from the URDF chain: only the joints between the palm and the tip are evaluated, and the
    Jacobian is analytic (axis x (tip - joint)). Joint angles are SDK degrees in the command frame."""

    def __init__(self, side):
        model = HandModel(side)
        by_child = {child: (name, parent, fixed, axis) for name, parent, child, fixed, axis in model.joints}
        self.sign = {m: s for m, _, s in joint_map(side)}                    # URDF joint -> sign of the SDK angle
        self.sdk_of = {m: sdk for m, sdk, _ in joint_map(side)}
        self.chains = {}
        for finger in FINGERS:
            link, chain = f"{finger}_tip", []
            while link in by_child:
                name, parent, fixed, axis = by_child[link]
                chain.append((name, fixed, axis))
                link = parent
            self.chains[finger] = list(reversed(chain))
        self.axis_model = {}                                                 # finger -> URDF joints of its 2 axes
        for finger, (a, b) in AXES.items():
            self.axis_model[finger] = tuple(next(m for m, s in self.sdk_of.items() if s == joint) for joint in (a, b))

    def angles(self, pose):
        values = dict(zip(JOINT_NAMES, pose))
        out = {m: math.radians(values[sdk]) * self.sign[m] for m, sdk in self.sdk_of.items()}
        for dependent, (source, factor) in COUPLED.items():
            out[dependent] = factor * out[source]
        return out

    def tip(self, finger, pose):
        """(tip position (3,), Jacobian (3, 2) in m per rad of the two SDK ellipse joints)."""
        angles = self.angles(pose)
        R, p = np.eye(3), np.zeros(3)
        frames = {}
        for name, fixed, axis in self.chains[finger]:
            p = p + R @ fixed[:3, 3]
            R = R @ fixed[:3, :3]
            if axis is not None:
                frames[name] = (R @ axis, p.copy())
                R = R @ _rotation(axis, angles[name])
        J = np.empty((3, 2))
        for k, m in enumerate(self.axis_model[finger]):
            w, origin = frames[m]
            J[:, k] = np.cross(w, p - origin) * self.sign[m]
        return p, J


class VelocityEstimator:
    """Joint speed (deg/s): change of the measured angle over the last window_s."""

    def __init__(self, window_s):
        self.window_s = window_s
        self.history = deque()

    def update(self, now, q):
        self.history.append((now, list(q)))
        while len(self.history) > 2 and now - self.history[1][0] >= self.window_s:
            self.history.popleft()
        t0, q0 = self.history[0]
        span = now - t0
        return [(a - b) / span for a, b in zip(q, q0)] if span > 0 else [0.0] * len(q)


class ImpedanceLaw:
    """Joint torques (mA, command frame) for the fingertip impedance plus the joint hold PD and friction terms."""

    def __init__(self, chains, fingers, gains, torque, side):
        self.chains, self.fingers, self.gains = chains, list(fingers), gains
        self.kp = per_joint(torque, "kp", float(torque.get("kp", 8.0)))
        self.kd = per_joint(torque, "kd", float(torque.get("kd", 0.3)))
        self.axes = {f: tuple(JOINT_NAMES.index(j) for j in AXES[f]) for f in self.fingers}
        self.friction = None
        if gains["friction"] and torque.get("friction_table"):
            # The torque config's teleop amounts, times this demo's multipliers (impedance_ellipses.yaml).
            scale = [gains["friction_error_gain"] * s
                     for s in per_joint(torque, "friction_scale", float(torque.get("friction_scale", 0.4)))]
            self.friction_share = scale
            self.friction_table = load_friction_table(torque["friction_table"], scale)
            self.model = BreakawayModel(torque["breakaway_fine"]) if torque.get("breakaway_fine") else None
            self.ff_scale = [gains["friction_ff_gain"] * s
                             for s in per_joint(torque, "friction_ff_scale", float(torque.get("friction_ff_scale", 0.0)))]
            self.breakaway = load_breakaway(torque["friction_table"])
            self.friction_ramp_deg = float(torque.get("friction_ramp_deg", 4.0))
            self.ff_ramp_dps = float(torque.get("friction_ff_ramp_dps", 10.0))
            self.friction = True
        self.last_tip_error_mm = {}

    def torques(self, q, qdot, qd, qd_dot, ff_velocity=None):
        """ff_velocity: the target speed the friction feed-forward follows (default qd_dot). The runner passes the
        teleop controller's smoothed target speed, which is what the calibrated friction_ff_scale was tuned with.
        self.last_base keeps the spring + damper (+ hold PD) part without the friction terms: the learning signal."""
        tau = [kp * (d - a) - kd * v for kp, kd, d, a, v in zip(self.kp, self.kd, qd, q, qdot)]   # hold PD
        deg = math.pi / 180
        for finger in self.fingers:
            i, j = self.axes[finger]
            x, J = self.chains.tip(finger, q)
            xd, Jd = self.chains.tip(finger, qd)
            v = J @ np.array([qdot[i], qdot[j]]) * deg
            # Damper as in the teleop controller: D * (-v), braking the tip's own motion only. With
            # damp_target_velocity: true the planned tip velocity is subtracted (D * (v_d - v)), so there is no drag
            # at the planned speed (less lag, needs the trajectory's velocity: not available to a glove session).
            vd = Jd @ np.array([qd_dot[i], qd_dot[j]]) * deg if self.gains["damp_target_velocity"] else np.zeros(3)
            gains = self.gains["fingers"][finger]
            F = gains["stiffness_n_per_m"] * (xd - x) + gains["damping_ns_per_m"] * (vd - v)
            t = J.T @ F / self.gains["nm_per_ma"]
            tau[i], tau[j] = float(t[0]), float(t[1])
            self.last_tip_error_mm[finger] = float(np.linalg.norm(xd - x) * 1000.0)
        self.last_base = list(tau)
        if self.friction:
            error = [d - a for d, a in zip(qd, q)]
            table = self.friction_table
            if self.model is not None:
                table = [s * 0.5 * sum(self.model.both(k, a)) for k, (s, a) in enumerate(zip(self.friction_share, q))]
            tau = smooth_friction(tau, error, table, self.friction_ramp_deg)
            plus, minus = self.breakaway
            if self.model is not None:
                plus = [self.model.current(k, a, 1) for k, a in enumerate(q)]
                minus = [self.model.current(k, a, -1) for k, a in enumerate(q)]
            tau = friction_feedforward(tau, qd_dot if ff_velocity is None else ff_velocity, plus, minus, self.ff_scale,
                                       self.ff_ramp_dps)
        return tau


class FeedforwardTable:
    """Learned feed-forward torque (mA) per tracked finger, per ellipse joint, per phase bin of that finger's loop:
    the current the joint needs along THIS trajectory (friction, static load), found by iterative learning on the
    hand (--learn). After each learning pass, bin k gains learn_gain x the mean spring + damper torque the law had to
    add there, smoothed over the neighbouring bins and capped at max_ma. Tied to the ellipse settings it was learned
    on (the signature); a different trajectory needs a new table."""

    def __init__(self, fingers, bins, max_ma, signature, harmonics=4):
        self.bins, self.max_ma, self.signature, self.passes = bins, max_ma, signature, 0
        self.harmonics = int(harmonics)          # 0 = keep everything (the raw binned means)
        self.table = {f: {JOINT_NAMES.index(n): np.zeros(bins) for n in AXES[f]} for f in fingers}
        self.others = {}          # fingers of the file not tracked now: kept as they are when saving
        self.other_signatures = {}

    @staticmethod
    def signature_of(params, fingers):
        return {"fingers": {f: dict(params["fingers"][f]) for f in fingers}, "spread_deg": list(params["spread_deg"]),
                "flex_centre_deg": params["flex_centre_deg"], "thumb_centre_deg": list(params["thumb_centre_deg"])}

    def lookup(self, finger, phase):
        """{joint index: mA} at this phase (radians): linear between bins, circular."""
        x = (phase / (2 * math.pi)) % 1.0 * self.bins
        k, f = int(x), x - int(x)
        return {j: float(v[k % self.bins] * (1 - f) + v[(k + 1) % self.bins] * f) for j, v in self.table[finger].items()}

    def learn(self, samples, gain):
        """samples: {finger: {joint: [(phase, residual mA), ...]}} of one pass; adds gain x the binned mean."""
        for finger, per_joint in samples.items():
            for j, pairs in per_joint.items():
                if not pairs:
                    continue
                sums, counts = np.zeros(self.bins), np.zeros(self.bins)
                for phase, residual in pairs:
                    k = int((phase / (2 * math.pi)) % 1.0 * self.bins) % self.bins
                    sums[k] += residual
                    counts[k] += 1
                mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
                update = self.smooth((np.roll(mean, 1) + mean + np.roll(mean, -1)) / 3.0)
                self.table[finger][j] = np.clip(self.table[finger][j] + gain * update, -self.max_ma, self.max_ma)
        self.passes += 1

    def smooth(self, values):
        """`values` over one turn with only its harmonics up to self.harmonics kept (0 = unchanged)."""
        if self.harmonics <= 0:
            return np.asarray(values, dtype=float)
        spectrum = np.fft.rfft(np.asarray(values, dtype=float))
        spectrum[self.harmonics + 1:] = 0.0
        return np.fft.irfft(spectrum, n=len(values))

    def smooth_all(self):
        for per_joint in self.table.values():
            for j in per_joint:
                per_joint[j] = np.clip(self.smooth(per_joint[j]), -self.max_ma, self.max_ma)

    def peak_ma(self):
        return max((float(np.abs(v).max()) for t in self.table.values() for v in t.values()), default=0.0)

    def save(self, path):
        fingers = dict(self.others)
        fingers.update({f: {JOINT_NAMES[j]: [round(float(v), 2) for v in arr] for j, arr in t.items()}
                        for f, t in self.table.items()})
        signature = dict(self.signature)
        signature["fingers"] = dict(self.other_signatures, **self.signature["fingers"])
        Path(path).write_text(json.dumps({"signature": signature, "bins": self.bins, "max_ma": self.max_ma,
                                          "passes": self.passes, "fingers": fingers}, indent=1))

    @classmethod
    def load(cls, path, fingers, bins, max_ma, signature, harmonics=4):
        """The table for `fingers` from a file; the file's other fingers are carried along. Refuses a file learned
        on other ellipse settings (for the tracked fingers) or with another bin count."""
        data = json.loads(Path(path).read_text())
        saved = data.get("signature") or {}
        for key in ("spread_deg", "flex_centre_deg", "thumb_centre_deg"):
            if saved.get(key) != signature[key]:
                raise ValueError(f"{path}: learned for other ellipse settings ({key}); use --reset-feedforward")
        for f in fingers:
            if f in saved.get("fingers", {}) and saved["fingers"][f] != signature["fingers"][f]:
                raise ValueError(f"{path}: learned for another {f} ellipse; use --reset-feedforward")
        if data.get("bins") != bins:
            raise ValueError(f"{path}: {data.get('bins')} phase bins, the gains file asks {bins}; use --reset-feedforward")
        table = cls(fingers, bins, max_ma, signature, harmonics)
        table.passes = int(data.get("passes", 0))
        for f, block in (data.get("fingers") or {}).items():
            if f in table.table:
                for j in table.table[f]:
                    values = block.get(JOINT_NAMES[j])
                    if values is not None and len(values) == bins:
                        table.table[f][j] = np.clip(np.array(values, dtype=float), -max_ma, max_ma)
            else:
                table.others[f] = block
        table.other_signatures = {f: v for f, v in saved.get("fingers", {}).items() if f not in fingers}
        table.smooth_all()                       # an older, noisier table is cleaned the same way
        return table


class SafetyStop(Exception):
    pass


class ImpedanceRunner:
    """The torque-mode part: one tick = joint torques -> motor torques -> write, read the pose back."""

    def __init__(self, hand, law, gains, tau_max, polarity, sign, clock=time.monotonic, log=print,
                 target_velocity_window_s=0.15, torque=None):
        self.hand, self.law, self.gains, self.tau_max = hand, law, gains, tau_max
        self.polarity, self.sign, self.clock, self.log = polarity, sign, clock, log
        self.velocity = VelocityEstimator(gains["velocity_window_s"])
        # The friction feed-forward follows the target's speed over the same window as in teleop
        # (torque.target_velocity_window_s), not the exact planned speed: the calibrated friction_ff_scale was tuned
        # with that smoothing (~half a window behind), and the undelayed push made the abductions lead their targets
        # by 74-236 ms (run impedance_ellipse_1).
        self.target_velocity = TargetVelocity(target_velocity_window_s)
        # Dither exactly as the teleop controller applies it (torque.dither_ma at torque.dither_hz): a square wave on
        # the motors of joints that are stuck (|error| > deadband_deg and |speed| < v_stick_dps) and whose per-joint
        # dither_ma is not 0; never on a motor already at tau_max. Keeps a stuck joint on the edge of moving, so the
        # spring moves it before a large error builds up (bench 2026-11-02: slip speeds ~150 -> ~50 deg/s).
        torque = torque or {}
        self.dither = None
        if gains.get("dither") and float(torque.get("dither_ma", 0.0)) > 0:
            self.dither = {"ma": float(torque["dither_ma"]), "hz": float(torque.get("dither_hz", 50.0)),
                           "per_joint": per_joint(torque, "dither_ma", float(torque["dither_ma"])),
                           "deadband": float(torque.get("deadband_deg", 1.0)),
                           "v_stick": float(torque.get("v_stick_dps", 20.0))}
        # Moving-target dither (gains dither_moving_ma, dither_moving_ticks): +-amplitude per JOINT on every ellipse
        # joint whose target is moving, flipping every dither_moving_ticks control ticks. Unlike the teleop dither it
        # is on before a joint sticks, reaches the abductions (per joint, not per motor), and is a clean square wave
        # at the loop's own rate. It keeps the joints off static friction so they do not move in a staircase.
        self.dither_joints = sorted({JOINT_NAMES.index(j) for f in law.fingers for j in AXES[f]})
        self.ticks = 0
        self.fast = [0] * 16
        self.rows = []
        self.q = None
        self.torque_on = False

    def enter(self):
        self.q = [s * v for s, v in zip(self.sign, self.hand.read_positions())]
        self.hand.set_torque_mode()
        self.torque_on = True
        self.started = self.clock()

    def run_pass(self, pose_at, duration, label, table=None, phases=None, learn=False, ramp_s=0.0):
        """Track pose_at(t) for duration seconds. table/phases: the learned feed-forward and each finger's phase as a
        function of t (applied at full strength throughout: friction does not shrink with the amplitude ramps).
        learn: also collect, per finger and joint, the (phase, spring + damper torque) samples outside the ramps.
        Returns (tip errors mm per finger, samples)."""
        t0 = self.clock()
        errors = {f: [] for f in self.law.fingers}
        samples = {f: {j: [] for j in table.table[f]} for f in table.table} if (table is not None and learn) else {}
        self.target_velocity.reset()
        while True:
            now = self.clock()
            t = min(now - t0, duration)
            qd = pose_at(t)
            h = 1e-3                          # planned joint velocity: central difference (one-sided at the ends)
            t_hi, t_lo = min(t + h, duration), max(t - h, 0.0)
            qd_dot = [(a - b) / (t_hi - t_lo) for a, b in zip(pose_at(t_hi), pose_at(t_lo))]
            qdot = self.velocity.update(now, self.q)
            tau = self.law.torques(self.q, qdot, qd, qd_dot, ff_velocity=self.target_velocity.update(now, qd))
            ramp = min(1.0, (now - self.started) / self.gains["gain_ramp_s"]) if self.gains["gain_ramp_s"] > 0 else 1.0
            ff = [0.0] * 16
            if table is not None:
                in_body = ramp >= 1.0 and ramp_s < t < duration - ramp_s
                for finger in table.table:
                    phase = phases[finger](t)
                    for j, value in table.lookup(finger, phase).items():
                        ff[j] = value
                        if learn and in_body:
                            samples[finger][j].append((phase, self.law.last_base[j]))
                tau = [a + b for a, b in zip(tau, ff)]
            amp = self.gains.get("dither_moving_ma", 0.0)
            if amp > 0 and ramp >= 1.0:
                sign = 1.0 if (self.ticks // self.gains["dither_moving_ticks"]) % 2 == 0 else -1.0
                tau = list(tau)
                for j in self.dither_joints:
                    if abs(qd_dot[j]) > 1.0:
                        tau[j] += sign * amp
            self.ticks += 1
            command = joint_to_motor_torques([ramp * v for v in tau], self.tau_max, self.polarity)
            if self.dither is not None and ramp >= 1.0:
                dz = self.dither
                stuck = [abs(d - a) > dz["deadband"] and abs(v) < dz["v_stick"] and amp > 0
                         for d, a, v, amp in zip(qd, self.q, qdot, dz["per_joint"])]
                command = dither(command, dz["ma"], dz["hz"], now, self.tau_max, active=motor_active(stuck))
            raw, _ = self.hand.write_torques_and_read_positions(command)
            self.q = [s * v for s, v in zip(self.sign, raw)]
            for finger, e in self.law.last_tip_error_mm.items():
                errors[finger].append(e)
            self.rows.append({"pass": label, "t": round(t, 5), "q": [round(v, 3) for v in self.q],
                              "qd": [round(v, 3) for v in qd], "tip_error_mm": {f: round(e, 2) for f, e in
                                                                               self.law.last_tip_error_mm.items()},
                              "command_ma": [round(v, 1) for v in command], "ff_ma": [round(v, 1) for v in ff]})
            self.check(qdot)
            if t >= duration:
                return errors, samples

    def check(self, qdot):
        worst = max(self.law.last_tip_error_mm.items(), key=lambda kv: kv[1], default=(None, 0.0))
        if worst[1] > self.gains["max_tip_error_mm"]:
            raise SafetyStop(f"{worst[0]} tip {worst[1]:.1f} mm from its target (limit {self.gains['max_tip_error_mm']:g})")
        for j, v in enumerate(qdot):
            self.fast[j] = self.fast[j] + 1 if abs(v) > self.gains["max_speed_dps"] else 0
            if self.fast[j] >= self.gains["max_speed_ticks"]:
                raise SafetyStop(f"{JOINT_NAMES[j]} at {v:.0f} deg/s for {self.fast[j]} ticks "
                                 f"(limit {self.gains['max_speed_dps']:g})")

    def leave(self):
        """Zero torque, then position mode holding the measured pose (no snap to an old position target)."""
        if not self.torque_on:
            return None
        self.torque_on = False
        self.hand.write_motor_torques([0.0] * 16)
        pose = [min(max(s * v, lo), hi) for s, v, (lo, hi) in zip(self.sign, self.hand.read_positions(),
                                                                    ANGLE_LIMITS.values())]
        self.hand.write_positions(pose)
        self.hand.set_position_mode()
        return pose


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--config", default=str(DEFAULT_CONFIG), help="hand config with hand_roles and the torque block")
    p.add_argument("--hand_serial", default="AAC-002", help="hand_roles key (AAC-002 right, AAC-001 left)")
    p.add_argument("--params", default=str(DEFAULT_PARAMS), help="ellipses YAML (shared with mcp_ellipses.py)")
    p.add_argument("--gains", default=str(DEFAULT_GAINS), help="impedance and safety YAML")
    p.add_argument("--fingers", nargs="+", choices=FINGERS, metavar="FINGER", default=list(FINGERS),
                   help="only these fingers track an ellipse (the others hold their pose)")
    p.add_argument("--loops", type=int, default=1, help="passes with the feed-forward frozen; 0 = until Ctrl-C")
    p.add_argument("--learn", type=int, default=0, help="learning passes first (each updates and saves the table)")
    p.add_argument("--feedforward", help="learned feed-forward JSON (default demo/calibration/impedance_feedforward_<side>.json)")
    p.add_argument("--reset-feedforward", action="store_true", help="ignore the saved table and learn from zero")
    p.add_argument("--settle", type=float, default=2.0, help="seconds of each position-mode easing move")
    p.add_argument("--log", help="write every control tick (pose, target, tip errors, motor commands) to this JSON")
    p.add_argument("--no-tip-stop", action="store_true",
                   help="no safety stop for a fingertip far from its target (max_tip_error_mm); the joint over-speed "
                        "stop (max_speed_dps) stays on")
    args = p.parse_args(argv)
    if args.loops < 0 or args.learn < 0 or args.settle <= 0:
        p.error("--loops and --learn must not be negative and --settle must be positive")
    args.fingers = [f for f in FINGERS if f in args.fingers]
    try:
        args.ellipses = load_params(args.params)
        args.impedance = load_gains(args.gains)
    except (OSError, yaml.YAMLError, ValueError) as error:
        p.error(str(error))
    if args.no_tip_stop:
        args.impedance["max_tip_error_mm"] = math.inf      # never exceeded: check() only keeps the over-speed stop
    return args


def run(args, hand=None, sleep=time.sleep, clock=time.monotonic, log=print):
    settings = load_aac_config(args, args.config)
    torque = settings.get("torque")
    if torque is None:
        raise ValueError(f"{args.config} has no torque block (needed for tau_max, the hold gains and friction)")
    side = "Right" if settings["hand_id"].lower().startswith("right") else "Left"
    params, gains = args.ellipses, args.impedance
    plan, centre = build_plan(TipKinematics(side), args.fingers, params)
    pose_at, duration, _ = ellipses(centre, plan, params, args.fingers)
    law = ImpedanceLaw(FingerChains(side), args.fingers, gains, torque, side)
    if not math.isfinite(gains["max_tip_error_mm"]):
        log("[demo] fingertip-distance safety stop OFF (--no-tip-stop); the joint over-speed stop stays on")
    phases = {f: (lambda t, f=f: params["fingers"][f]["spin"] * 2 * math.pi * params["fingers"][f]["hz"] * t
                  + math.radians(params["fingers"][f]["phase_deg"])) for f in args.fingers}
    ramp_s = min(params["ramp_s"], duration / 2)
    ff_path = Path(args.feedforward) if args.feedforward else ROOT / "demo" / "calibration" / f"impedance_feedforward_{side.lower()}.json"
    signature = FeedforwardTable.signature_of(params, args.fingers)
    table = None
    if ff_path.exists() and not args.reset_feedforward:
        table = FeedforwardTable.load(ff_path, args.fingers, gains["learn_bins"], gains["learn_max_ma"], signature,
                                      gains["learn_harmonics"])
        log(f"[demo] learned feed-forward: {ff_path} ({table.passes} learning passes so far, peak {table.peak_ma():.0f} mA"
            + (f", smoothed to {gains['learn_harmonics']} harmonics" if gains["learn_harmonics"] else "") + ")")
    elif args.learn > 0:
        table = FeedforwardTable(args.fingers, gains["learn_bins"], gains["learn_max_ma"], signature,
                                 gains["learn_harmonics"])
        log(f"[demo] learned feed-forward: starting from zero, will be saved to {ff_path}")
    if hand is None:
        hand = AACHand(settings["serial_device"], settings["device_address"], settings["connection_timeout_s"],
                       bus=settings["bus"])
    log(f"[demo] {settings['hand_id']} ({args.hand_serial}) over {settings['bus']}: impedance K "
        + ", ".join(f"{f} {gains['fingers'][f]['stiffness_n_per_m']:g}" for f in args.fingers)
        + f" N/m, D {gains['damping_ns_per_m']:g} N s/m, tau_max {torque.get('tau_max', 200.0):g} mA, friction "
        f"{'on' if law.friction else 'off'}, dither "
        + (f"{torque.get('dither_ma', 0.0):g} mA at {torque.get('dither_hz', 50.0):g} Hz on stuck joints"
           if gains["dither"] and float(torque.get("dither_ma", 0.0)) > 0 else "off"))
    hand.open()
    sign = readback_sign(settings["hand_id"], getattr(hand, "register_map", None))
    streamer = Streamer(hand, 100.0, 100.0, read_every=5, sleep=sleep, clock=clock, log=log)
    runner = ImpedanceRunner(hand, law, gains, float(torque.get("tau_max", 200.0)), pair_polarity(settings["hand_id"]),
                             sign, clock=clock, log=log,
                             target_velocity_window_s=float(torque.get("target_velocity_window_s", 0.03)),
                             torque=torque)
    passes, outcome = 0, "done"
    try:
        log(f"[demo] position mode: easing to rest, the spread and the ellipse centres ({args.settle:g} s each)")
        streamer.start(args.settle)
        streamer.move_to(spread_pose(params["spread_deg"]), args.settle)
        streamer.move_to(centre, args.settle)
        log(f"[demo] torque mode: {', '.join(args.fingers)} on their ellipses, {duration:g} s passes "
            f"({'until Ctrl-C' if args.loops == 0 else f'{args.loops} pass(es)'}), torque fading in over "
            f"{gains['gain_ramp_s']:g} s")
        runner.enter()
        while args.loops == 0 or passes < args.learn + args.loops:
            learning = passes < args.learn
            tick0 = len(runner.rows)
            started = clock()
            errors, samples = runner.run_pass(pose_at, duration, passes, table=table, phases=phases,
                                              learn=learning, ramp_s=ramp_s)
            rate = (len(runner.rows) - tick0) / max(clock() - started, 1e-9)
            log(f"  pass {passes + 1}" + (f" (learning {passes + 1}/{args.learn})" if learning else ""))
            for finger in args.fingers:
                e = np.array(errors[finger])
                log(f"  {finger:6s} tip error rms {np.sqrt(np.mean(e ** 2)):5.1f} mm, max {e.max():5.1f} mm")
            log(f"  control rate {rate:.0f} Hz")
            if learning:
                table.learn(samples, gains["learn_gain"])
                table.save(ff_path)
                log(f"  feed-forward updated and saved (peak {table.peak_ma():.0f} mA)")
            passes += 1
    except KeyboardInterrupt:
        outcome = "stopped (Ctrl-C)"
    except SafetyStop as stop:
        outcome = f"SAFETY STOP: {stop}"
    finally:
        try:
            pose = runner.leave()
            log(f"[demo] {outcome}: torque zeroed, position mode at the measured pose" if pose is not None
                else f"[demo] {outcome}")
            if pose is not None:
                streamer.pose = pose
            streamer.finish(args.settle)
            hand.set_position_mode()
        finally:
            hand.close()
            if args.log:
                Path(args.log).write_text(json.dumps({"joint_names": list(JOINT_NAMES), "fingers": args.fingers,
                                                      "params": params, "gains": gains, "outcome": outcome,
                                                      "ticks": runner.rows}))
                log(f"[demo] log: {args.log} ({len(runner.rows)} ticks)")
    log(f"[demo] {passes} pass(es)")
    return passes, outcome


if __name__ == "__main__":
    run(parse_args())
