"""Current-based admittance on the D21's own position loop (no firmware change, no force sensor).

After Shi et al., *Minimalist Compliance Control*, RSS 2026 (arXiv 2603.00913). Per joint, in SI:

    I_joint   = sign * (motor currents decomposed onto the joint)      spec section 3.5 motor pairs
    tau_m     = k_t * (I_joint - I_free(q))                            servo torque above the free-motion baseline
    d         = sign(tau_m * dq) while |dq| > eps_v, else previous     is the motor driving or being backdriven
    tau_ext   = -r * eta^(+-1) * k_t * (I_joint - I_free(q))           external torque estimate (eq. 6-8)
    tau_f     = alpha * tau_ext + (1 - alpha) * tau_f                  EMA, then deadband tau_db
    M dd(off) + D d(off) + K off = tau_bar                             virtual spring-mass-damper on the offset
    q_cmd     = q_des + off                                            the servo keeps closing its own loop

The offset is anchored to the task target, so a released finger springs back to q_des; K = 0 is
hand guiding. All parameters come from a YAML file (teleop/calibration/acc/admittance/admittance.yaml), the free-motion current
I_free(q) and current signs from teleop/calibration/acc/admittance/admittance_calibration.yaml written by
retargetor/dex-retargetor-cpp/python/admittance_calibrate.py; without that file I_free is the idle current
sampled at startup and every sign is +1.

Angles cross this module in command-frame degrees (SDK joint order); the dynamics run in radians.
"""
import math
from pathlib import Path

import numpy as np

from hand.backends.aac import ANGLE_LIMITS, JOINT_NAMES

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "calibration" / "acc" / "admittance" / "admittance.yaml"
# Finger abduction (4, 7, 10, 13) reads back mirrored on the LEFT D21; the right hand reads back in the command
# frame. This is the left table; the controller sets Admittance.readback_sign per hand (hand_controller_aac.readback_sign).
READBACK_SIGN = np.array([1, 1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1], dtype=float)
# (same-sign joint, opposite-sign joint, motor a, motor b), 0-based: a motor pair's common-mode
# current is the flexion (thumb: abduction) joint, the differential part the abduction (thumb: rotation).
MOTOR_PAIRS = ((0, 1, 0, 1), (5, 4, 4, 5), (8, 7, 7, 8), (11, 10, 10, 11), (14, 13, 13, 14))
MOTOR_SINGLES = ((2, 2), (3, 3), (6, 6), (9, 9), (12, 12), (15, 15))  # (joint, motor)
LOWER = np.array([ANGLE_LIMITS[j][0] for j in range(16)], dtype=float)
UPPER = np.array([ANGLE_LIMITS[j][1] for j in range(16)], dtype=float)

# Per-joint parameters (overridable under "joints:"), with their defaults and validity.
PER_JOINT = {
    "kt_nm_per_a": (1.0, "positive"),      # joint torque per amp of joint current
    "gear_ratio": (1.0, "positive"),       # 1 when current is already referred to the joint
    "efficiency": (0.8, "unit"),           # eta in (0, 1]
    "M": (0.002, "positive"),              # virtual inertia, kg m^2
    "D": (None, "nonnegative"),            # virtual damping, N m s/rad; null = damping_ratio * 2 sqrt(K M)
    "damping_ratio": (1.0, "positive"),
    "K": (1.5, "nonnegative"),             # virtual stiffness, N m/rad; 0 = hand guiding
    "alpha": (0.3, "unit"),                # EMA weight on the torque estimate
    "tau_db_nm": (0.02, "nonnegative"),    # torque deadband; calibration hysteresis is a good start
    "eps_v_rad_s": (0.02, "nonnegative"),  # debounce for the drive-direction decision
    "dq_max_rad": (0.5, "positive"),       # offset clamp
    "offset_rate_max_rad_s": (None, "nonnegative"),  # limit on |d(off)/dt|; null = none (Kikuuwe-style contact limiter)
}
GLOBAL = {
    "velocity_window_s": (0.1, "positive"),       # feedback changes at 50 Hz: difference over a window
    "max_dt_s": (0.1, "positive"),
    "target_motion_gate_dps": (10.0, "positive"),  # a target moving faster than this is a commanded move...
    "target_holdoff_s": (0.2, "nonnegative"),      # ...whose current is not read as contact for this long
}
KEYS = set(PER_JOINT) | set(GLOBAL) | {"enabled", "calibration", "joints"}


def joint_currents(motor_a):
    """16 motor currents -> 16 joint currents (amps), spec section 3.5 pair decomposition."""
    motor = np.asarray(motor_a, dtype=float)
    joint = np.zeros(16)
    for same, opposite, a, b in MOTOR_PAIRS:
        joint[same] = (motor[a] + motor[b]) / 2
        joint[opposite] = (motor[a] - motor[b]) / 2
    for j, m in MOTOR_SINGLES:
        joint[j] = motor[m]
    return joint


class Calibration:
    """Per-joint current sign and free-motion current I_free(q) (polynomial in command-frame degrees)."""

    def __init__(self, sign=None, coef=None):
        self.sign = np.ones(16) if sign is None else np.asarray(sign, dtype=float)
        self.coef = [None] * 16 if coef is None else list(coef)  # highest power first, or None
        self.idle = np.zeros(16)  # fallback for joints without a polynomial

    @classmethod
    def load(cls, path):
        import yaml
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
        block = doc.get("calibration")
        if not isinstance(block, dict) or not isinstance(block.get("joints"), dict):
            raise ValueError(f'{path}: expected "calibration: {{joints: {{...}}}}"')
        cal = cls()
        for name, entry in block["joints"].items():
            if name not in JOINT_NAMES:
                raise ValueError(f'{path}: unknown joint "{name}"')
            if not isinstance(entry, dict):
                raise ValueError(f"{path}: calibration.joints.{name} must be a mapping")
            j = JOINT_NAMES.index(name)
            sign = entry.get("current_sign", 1)
            if sign not in (1, -1):
                raise ValueError(f"{path}: {name}.current_sign must be 1 or -1")
            cal.sign[j] = sign
            coef = entry.get("i_free_coef_a")
            if coef is not None:
                coef = [_number(c, f"{path}: {name}.i_free_coef_a") for c in coef]
                if not coef:
                    raise ValueError(f"{path}: {name}.i_free_coef_a must not be empty")
                cal.coef[j] = np.array(coef)
        return cal

    @property
    def needs_idle(self):
        return any(c is None for c in self.coef)

    def i_free(self, q_deg):
        return np.array([np.polyval(c, q) if c is not None else idle
                         for c, q, idle in zip(self.coef, q_deg, self.idle)])


class Admittance:
    def __init__(self, per_joint, global_params, calibration=None):
        """per_joint: {name: 16-vector}, global_params: {name: float}; see PER_JOINT / GLOBAL."""
        p = {k: np.asarray(v, dtype=float) for k, v in per_joint.items()}
        self.readback_sign = READBACK_SIGN.copy()  # left table; the controller overrides it for the right hand
        self.kt, self.ratio, self.eta = p["kt_nm_per_a"], p["gear_ratio"], p["efficiency"]
        self.M, self.K, self.alpha = p["M"], p["K"], p["alpha"]
        self.D = np.where(np.isnan(p["D"]), p["damping_ratio"] * 2 * np.sqrt(p["K"] * p["M"]), p["D"])
        self.tau_db, self.eps_v, self.dq_max = p["tau_db_nm"], p["eps_v_rad_s"], p["dq_max_rad"]
        self.rate_max = np.where(np.isnan(p["offset_rate_max_rad_s"]), np.inf, p["offset_rate_max_rad_s"])
        g = global_params
        self.window, self.max_dt = g["velocity_window_s"], g["max_dt_s"]
        self.gate, self.holdoff = g["target_motion_gate_dps"], g["target_holdoff_s"]
        self.calibration = calibration if calibration is not None else Calibration()
        self.reset()

    def reset(self):
        self.offset = np.zeros(16)        # rad
        self.offset_rate = np.zeros(16)   # rad/s
        self.tau_f = np.zeros(16)
        self.direction = np.ones(16)
        self.last_time = None
        self.last_target = None
        self.moved_at = np.full(16, -np.inf)
        self.history = []                 # (time, q_deg) within the velocity window
        self.velocity = np.zeros(16)      # rad/s

    def set_idle_current(self, motor_samples_a):
        """Fallback I_free: the mean joint current of samples taken while the hand holds still, unloaded."""
        samples = np.array([joint_currents(m) for m in motor_samples_a])
        self.calibration.idle = self.calibration.sign * samples.mean(axis=0)

    def describe(self):
        return {"type": "admittance", "K": self.K.tolist(), "D": self.D.tolist(), "M": self.M.tolist(),
                "kt_nm_per_a": self.kt.tolist(), "current_sign": self.calibration.sign.tolist(),
                "calibrated_joints": [n for n, c in zip(JOINT_NAMES, self.calibration.coef) if c is not None]}

    def external_torque(self, measured_raw_deg, motor_currents_a, now):
        """Returns (q_deg command frame, tau_ext N m, joint current A); updates the velocity estimate."""
        q = np.asarray(self.readback_sign, dtype=float) * np.asarray(measured_raw_deg, dtype=float)
        self.history = [h for h in self.history if now - h[0] <= self.window] + [(now, q)]
        span = now - self.history[0][0]
        self.velocity = np.deg2rad(q - self.history[0][1]) / span if span > 0 else np.zeros(16)
        current = self.calibration.sign * joint_currents(motor_currents_a)
        excess = current - self.calibration.i_free(q)
        tau_m = self.kt * excess
        moving = np.abs(self.velocity) > self.eps_v
        self.direction = np.where(moving, np.sign(tau_m * self.velocity), self.direction)
        self.direction[self.direction == 0] = 1.0
        c = self.ratio * np.where(self.direction > 0, self.eta, 1.0 / self.eta) * self.kt
        return q, -c * excess, current

    def step(self, target_deg, measured_raw_deg, motor_currents_a, now):
        """One control tick; returns a dict with q_cmd (command-frame degrees) and diagnostics."""
        target = np.asarray(target_deg, dtype=float)
        q, tau_ext, current = self.external_torque(measured_raw_deg, motor_currents_a, now)
        dt = min(max(now - self.last_time, 0.0), self.max_dt) if self.last_time is not None else 0.0
        if self.last_target is None:
            self.moved_at[:] = now
        elif dt > 0:
            self.moved_at[np.abs(target - self.last_target) / dt >= self.gate] = now
        commanded_move = now - self.moved_at < self.holdoff
        self.tau_f = self.alpha * tau_ext + (1 - self.alpha) * self.tau_f
        tau_bar = np.where(np.abs(self.tau_f) < self.tau_db, 0.0, self.tau_f - self.tau_db * np.sign(self.tau_f))
        tau_bar = np.where(commanded_move, 0.0, tau_bar)  # free motion current is not contact
        if dt > 0:  # semi-implicit Euler
            acc = (tau_bar - self.D * self.offset_rate - self.K * self.offset) / self.M
            self.offset_rate = np.clip(self.offset_rate + dt * acc, -self.rate_max, self.rate_max)
            self.offset = np.clip(self.offset + dt * self.offset_rate, -self.dq_max, self.dq_max)
            # Anti-windup: the offset never reaches past what the joint limits let the command do.
            self.offset = np.clip(self.offset, np.deg2rad(LOWER - target), np.deg2rad(UPPER - target))
            self.offset_rate = np.where(np.abs(self.offset) >= self.dq_max, 0.0, self.offset_rate)
        self.last_time, self.last_target = now, target.copy()
        offset_deg = np.rad2deg(self.offset)
        q_cmd = np.clip(target + offset_deg, LOWER, UPPER)
        return {"q_cmd": q_cmd, "offset_deg": offset_deg, "tau_ext": tau_ext, "tau_bar": tau_bar,
                "joint_current_a": current, "q_meas": q, "velocity_rad_s": self.velocity}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _checked(value, name, kind):
    if value is None:
        if kind == "nonnegative":
            return math.nan
        raise ValueError(f"{name} must not be null")
    v = _number(value, name)
    if kind == "positive" and v <= 0:
        raise ValueError(f"{name} must be positive")
    if kind == "nonnegative" and v < 0:
        raise ValueError(f"{name} must be non-negative")
    if kind == "unit" and not 0 < v <= 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return v


def load_admittance(path):
    """Build the controller from YAML; returns None when the file disables it. See admittance.yaml."""
    import yaml
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    block = doc.get("admittance")
    if not isinstance(block, dict):
        raise ValueError(f'{path}: missing "admittance" section')
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError(f"{path}: admittance.enabled must be true or false")
    if not enabled:
        return None
    unknown = set(block) - KEYS
    if unknown:
        raise ValueError(f'{path}: unknown admittance keys: {", ".join(sorted(unknown))}')
    per_joint = {k: np.full(16, _checked(block.get(k, default), f"{path}: {k}", kind))
                 for k, (default, kind) in PER_JOINT.items()}
    global_params = {k: _checked(block.get(k, default), f"{path}: {k}", kind) for k, (default, kind) in GLOBAL.items()}
    overrides = block.get("joints") or {}
    if not isinstance(overrides, dict):
        raise ValueError(f"{path}: admittance.joints must be a mapping of joint name -> settings")
    for joint, settings in overrides.items():
        if joint not in JOINT_NAMES:
            raise ValueError(f'{path}: unknown joint "{joint}" (expected one of {", ".join(JOINT_NAMES)})')
        if not isinstance(settings, dict):
            raise ValueError(f"{path}: admittance.joints.{joint} must be a mapping")
        for k, v in settings.items():
            if k not in PER_JOINT:
                raise ValueError(f"{path}: joints.{joint}: unknown key {k}")
            per_joint[k][JOINT_NAMES.index(joint)] = _checked(v, f"{path}: joints.{joint}.{k}", PER_JOINT[k][1])
    calibration = None
    cal_path = block.get("calibration")
    if cal_path is not None:
        if not isinstance(cal_path, str) or not cal_path:
            raise ValueError(f"{path}: admittance.calibration must be a file path")
        cal_file = Path(cal_path) if Path(cal_path).is_absolute() else path.parent / cal_path
        if cal_file.exists():
            calibration = Calibration.load(cal_file)
    return Admittance(per_joint, global_params, calibration)
