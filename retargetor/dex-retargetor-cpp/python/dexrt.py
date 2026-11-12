"""ctypes bindings for libdexrt: IRLS/OSQP vector retargeting and joint impedance.

Build the library first (see README.md). The shared object is located through
DEXRT_LIB or <this file>/../build/libdexrt.so.
"""
import ctypes
import os
import socket
import struct
from pathlib import Path

import numpy as np

MODEL_JOINTS = [f'thumb_joint{i}' for i in range(1, 5)] + [
    f'{finger}_joint{i}' for finger in ('index', 'middle', 'ring', 'pinky') for i in range(1, 4)]
COUPLINGS = [('thumb_joint5', 'thumb_joint4', 0.85)] + [
    (f'{f}_joint4', f'{f}_joint3', 0.8) for f in ('index', 'middle', 'ring', 'pinky')]
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
VECTOR_TABLE = [('thumb_pitch' if f == 'thumb' else f'{f}_mcp',
                 f'{f}_tip' if tip else ('thumb_mcp' if f == 'thumb' else f'{f}_pip'))
                for tip in (True, False) for f in FINGERS]
TIP_LINKS = [f'{f}_tip' for f in FINGERS]
FINGER_BASES = [f'{f}_mcp' for f in FINGERS[1:]]


def dexpilot_table(tips=TIP_LINKS, finger_bases=FINGER_BASES, shape_weight=15.0,
                   thumb_proximal=('thumb_pitch', 'thumb_mcp'), thumb_proximal_weight=1.0):
    """DexPilot vector layout (dex-retargeting DexPilotOptimizer, 5 fingers) as (origin, task, weight,
    projection, gate_a, gate_b) rows: thumb->finger tips (primary, projected by their own length),
    finger->finger tips (secondary, gated by the two thumb pairs), each finger's knuckle->tip shape
    vector (DexPilot's wrist->tip, anchored at a point both hands have), and the thumb's base->knuckle
    vector so the D21's 4-joint thumb posture stays determined. Vector = task - origin."""
    n = len(tips)
    rows = []
    for i in range(1, n):
        rows.append((tips[i], tips[0], 1.0, 1, -1, -1))
    for i in range(1, n - 1):
        for j in range(i + 1, n):
            rows.append((tips[j], tips[i], 1.0, 2, i - 1, j - 1))
    for base, tip in zip(finger_bases, tips[1:]):
        rows.append((base, tip, float(shape_weight), 0, -1, -1))
    if thumb_proximal is not None:
        rows.append((thumb_proximal[0], thumb_proximal[1], float(thumb_proximal_weight), 0, -1, -1))
    return rows


DEXPILOT_TABLE = dexpilot_table()
# Vendor SDK joint ranges in degrees, same order as MODEL_JOINTS (hand/backends/aac.py ANGLE_LIMITS).
SDK_LIMITS_DEG = [(0, 100), (-35, 35), (0, 50), (0, 70)] + [(-20, 20), (0, 90), (0, 100)] * 4


class _Options(ctypes.Structure):
    _fields_ = [('huber_delta', ctypes.c_double), ('norm_delta', ctypes.c_double), ('scaling', ctypes.c_double),
                ('max_steps', ctypes.c_int), ('step_limit', ctypes.c_double), ('levenberg', ctypes.c_double),
                ('stop_step', ctypes.c_double), ('eps_abs', ctypes.c_double), ('eps_rel', ctypes.c_double),
                ('max_iter', ctypes.c_int), ('polish', ctypes.c_int),
                ('project_dist', ctypes.c_double), ('escape_dist', ctypes.c_double),
                ('eta_primary', ctypes.c_double), ('eta_secondary', ctypes.c_double),
                ('projected_weight_primary', ctypes.c_double), ('projected_weight_secondary', ctypes.c_double)]


class _ComplianceConfig(ctypes.Structure):
    _fields_ = [('readback_sign', ctypes.c_double * 16), ('lower_deg', ctypes.c_double * 16),
                ('upper_deg', ctypes.c_double * 16), ('min_idle_current_a', ctypes.c_double),
                ('idle_sigma', ctypes.c_double), ('gain_dps_per_a', ctypes.c_double), ('return_rate', ctypes.c_double),
                ('offset_max_deg', ctypes.c_double), ('error_clamp_deg', ctypes.c_double),
                ('velocity_gate_dps', ctypes.c_double), ('deviation_min_deg', ctypes.c_double),
                ('velocity_window_s', ctypes.c_double), ('max_dt_s', ctypes.c_double),
                ('rest_bias_tau_s', ctypes.c_double), ('direction_from_current', ctypes.c_int),
                ('sign_min_samples', ctypes.c_int), ('sign_min_agreement', ctypes.c_double),
                ('target_motion_gate_dps', ctypes.c_double), ('target_holdoff_s', ctypes.c_double)]


class _SolveInfo(ctypes.Structure):
    _fields_ = [('cost', ctypes.c_double), ('steps', ctypes.c_int), ('osqp_iterations', ctypes.c_int),
                ('osqp_status', ctypes.c_int), ('converged', ctypes.c_int), ('solve_seconds', ctypes.c_double)]


_D = ctypes.POINTER(ctypes.c_double)
_I = ctypes.POINTER(ctypes.c_int)
_S = ctypes.POINTER(ctypes.c_char_p)
_lib = None


def load_library(path=None):
    global _lib
    if _lib is not None and path is None:
        return _lib
    candidates = [path, os.environ.get('DEXRT_LIB'), Path(__file__).resolve().parent.parent / 'build' / 'libdexrt.so']
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            lib = ctypes.CDLL(str(candidate))
            break
    else:
        raise FileNotFoundError('libdexrt.so not found; build dex-retargetor-cpp or set DEXRT_LIB')
    lib.dexrt_default_options.argtypes = [ctypes.POINTER(_Options)]
    lib.dexrt_version.restype = ctypes.c_char_p
    lib.dexrt_retargeter_create.restype = ctypes.c_void_p
    lib.dexrt_retargeter_create.argtypes = [ctypes.c_char_p, _S, ctypes.c_int, _S, _S, _D, ctypes.c_int, _S, _S,
                                            ctypes.c_int, _D, _I, _I, _I, _D, _D, _D, _D, ctypes.POINTER(_Options),
                                            ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_retargeter_set_scaling.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_retargeter_scaling.restype = ctypes.c_double
    lib.dexrt_retargeter_scaling.argtypes = [ctypes.c_void_p]
    lib.dexrt_retargeter_projected.argtypes = [ctypes.c_void_p, _I]
    lib.dexrt_retargeter_set_joint_targets.argtypes = [ctypes.c_void_p, _D, _D, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_retargeter_destroy.argtypes = [ctypes.c_void_p]
    for name in ('dof', 'full_dof', 'vector_count'):
        getattr(lib, f'dexrt_retargeter_{name}').argtypes = [ctypes.c_void_p]
    lib.dexrt_retargeter_bounds.argtypes = [ctypes.c_void_p, _D, _D]
    lib.dexrt_retargeter_last_q.argtypes = [ctypes.c_void_p, _D]
    lib.dexrt_retargeter_reset.argtypes = [ctypes.c_void_p, _D, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_retargeter_solve.argtypes = [ctypes.c_void_p, _D, _D, _D, ctypes.POINTER(_SolveInfo), ctypes.c_char_p,
                                           ctypes.c_size_t]
    lib.dexrt_retargeter_full_qpos.argtypes = [ctypes.c_void_p, _D, _D, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_retargeter_robot_vectors.argtypes = [ctypes.c_void_p, _D, _D, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_retargeter_link_jacobian.argtypes = [ctypes.c_void_p, _D, ctypes.c_char_p, _D, ctypes.c_char_p,
                                                   ctypes.c_size_t]
    lib.dexrt_retargeter_link_position.argtypes = [ctypes.c_void_p, _D, ctypes.c_char_p, _D, ctypes.c_char_p,
                                                   ctypes.c_size_t]
    lib.dexrt_impedance_create.restype = ctypes.c_void_p
    lib.dexrt_impedance_create.argtypes = [ctypes.c_int, _D, _D, _D, ctypes.c_double, ctypes.c_int, ctypes.c_double,
                                           ctypes.c_double, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_impedance_destroy.argtypes = [ctypes.c_void_p]
    lib.dexrt_impedance_set_target.argtypes = [ctypes.c_void_p, _D, ctypes.c_double, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_impedance_set_gains.argtypes = [ctypes.c_void_p, _D, _D, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_impedance_step.argtypes = [ctypes.c_void_p, _D, _D, ctypes.c_double, _D, _D, ctypes.POINTER(ctypes.c_int),
                                         ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_impedance_reset.argtypes = [ctypes.c_void_p]
    lib.dexrt_compliance_default_config.argtypes = [ctypes.POINTER(_ComplianceConfig)]
    lib.dexrt_compliance_create.restype = ctypes.c_void_p
    lib.dexrt_compliance_create.argtypes = [ctypes.POINTER(_ComplianceConfig), ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_compliance_destroy.argtypes = [ctypes.c_void_p]
    lib.dexrt_compliance_add_idle_sample.argtypes = [ctypes.c_void_p, _D, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_compliance_finish_idle.argtypes = [ctypes.c_void_p, _D, ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_compliance_step.argtypes = [ctypes.c_void_p, _D, _D, _D, ctypes.c_double, _D, _D, _D, _D, _D, _D, _D, _D,
                                          ctypes.c_char_p, ctypes.c_size_t]
    lib.dexrt_compliance_reset.argtypes = [ctypes.c_void_p]
    lib.dexrt_compliance_drive_signs.argtypes = [ctypes.c_void_p, _D]
    _lib = lib
    return lib


def _array(values, n=None):
    arr = np.ascontiguousarray(values, dtype=np.float64).reshape(-1)
    if n is not None and arr.size != n:
        raise ValueError(f'expected {n} values, got {arr.size}')
    return arr


def _ptr(arr):
    return arr.ctypes.data_as(_D)


def _strings(items):
    return (ctypes.c_char_p * len(items))(*[s.encode() for s in items])


def _err():
    return ctypes.create_string_buffer(512)


def _check(status, err):
    if status != 0:
        raise RuntimeError(err.value.decode(errors='replace') or 'dexrt call failed')


def default_options():
    opts = _Options()
    load_library().dexrt_default_options(ctypes.byref(opts))
    return {name: getattr(opts, name) for name, _ in _Options._fields_}


class Retargeter:
    """Vector retargeting: human finger vectors (URDF world frame, metres) -> 16 joint angles (rad).

    vectors: (origin_link, task_link) pairs, or (origin, task, weight, projection, gate_a, gate_b) rows
    for DexPilot projection (see dexpilot_table), optionally followed by origin_offset, task_offset
    (3 metres each, in the link frames) for points fixed on a link."""

    def __init__(self, urdf_path, independent_joints, couplings, vectors, lower=None, upper=None, **options):
        self._lib = load_library()
        opts = _Options()
        self._lib.dexrt_default_options(ctypes.byref(opts))
        for key, value in options.items():
            if not hasattr(opts, key):
                raise TypeError(f'unknown option {key}')
            setattr(opts, key, type(getattr(opts, key))(value))
        self.joint_names = list(independent_joints)
        zero = (0.0, 0.0, 0.0)
        self.vectors = [tuple(v) + ((1.0, 0, -1, -1, zero, zero)[len(v) - 2:] if len(v) < 8 else ()) for v in vectors]
        if any(len(v) != 8 for v in self.vectors):
            raise ValueError('vectors must be (origin, task), (origin, task, weight, projection, gate_a, gate_b) '
                             'or that plus (origin_offset, task_offset)')
        lower_arr = _array(lower, len(self.joint_names)) if lower is not None else None
        upper_arr = _array(upper, len(self.joint_names)) if upper is not None else None
        m = len(self.vectors)
        weights = _array([v[2] for v in self.vectors])
        projection = (ctypes.c_int * m)(*[int(v[3]) for v in self.vectors])
        gate_a = (ctypes.c_int * m)(*[int(v[4]) for v in self.vectors])
        gate_b = (ctypes.c_int * m)(*[int(v[5]) for v in self.vectors])
        err = _err()
        self._handle = self._lib.dexrt_retargeter_create(
            str(urdf_path).encode(), _strings(self.joint_names), len(self.joint_names),
            _strings([c[0] for c in couplings]), _strings([c[1] for c in couplings]),
            _ptr(_array([c[2] for c in couplings])), len(couplings),
            _strings([v[0] for v in self.vectors]), _strings([v[1] for v in self.vectors]), m,
            _ptr(weights), projection, gate_a, gate_b,
            _ptr(_array([v[6] for v in self.vectors], 3 * m)), _ptr(_array([v[7] for v in self.vectors], 3 * m)),
            _ptr(lower_arr) if lower_arr is not None else None, _ptr(upper_arr) if upper_arr is not None else None,
            ctypes.byref(opts), err, len(err))
        if not self._handle:
            raise RuntimeError(err.value.decode(errors='replace'))
        self.dof = self._lib.dexrt_retargeter_dof(self._handle)
        self.full_dof = self._lib.dexrt_retargeter_full_dof(self._handle)
        self.vector_count = self._lib.dexrt_retargeter_vector_count(self._handle)

    @classmethod
    def for_aac(cls, urdf_path, sdk_limits_deg=SDK_LIMITS_DEG, **options):
        limits = np.deg2rad(np.asarray(sdk_limits_deg, dtype=float))
        return cls(urdf_path, MODEL_JOINTS, COUPLINGS, VECTOR_TABLE, limits[:, 0], limits[:, 1], **options)

    @classmethod
    def for_aac_dexpilot(cls, urdf_path, sdk_limits_deg=SDK_LIMITS_DEG, table=DEXPILOT_TABLE, **options):
        """DexPilot vectors (thumb/finger tip pairs with contact projection, finger knuckle->tips, thumb proximal)."""
        options.setdefault('huber_delta', 0.03)
        limits = np.deg2rad(np.asarray(sdk_limits_deg, dtype=float))
        return cls(urdf_path, MODEL_JOINTS, COUPLINGS, table, limits[:, 0], limits[:, 1], **options)

    @property
    def scaling(self):
        return self._lib.dexrt_retargeter_scaling(self._handle)

    def set_scaling(self, scaling):
        err = _err()
        _check(self._lib.dexrt_retargeter_set_scaling(self._handle, float(scaling), err, len(err)), err)

    def set_joint_targets(self, target, weight):
        """Per-joint quadratic targets, weight 0 = none: adds weight*(q - target)^2 to the objective."""
        err = _err()
        t, w = _array(target, self.dof), _array(weight, self.dof)
        _check(self._lib.dexrt_retargeter_set_joint_targets(self._handle, _ptr(t), _ptr(w), err, len(err)), err)

    def clear_joint_targets(self):
        err = _err()
        _check(self._lib.dexrt_retargeter_set_joint_targets(self._handle, None, None, err, len(err)), err)

    @property
    def projected(self):
        """Per vector: True where the last solve projected the target to contact (DexPilot)."""
        out = (ctypes.c_int * self.vector_count)()
        self._lib.dexrt_retargeter_projected(self._handle, out)
        return np.array(out, dtype=bool)

    def close(self):
        if getattr(self, '_handle', None):
            self._lib.dexrt_retargeter_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    @property
    def bounds(self):
        lower, upper = np.empty(self.dof), np.empty(self.dof)
        self._lib.dexrt_retargeter_bounds(self._handle, _ptr(lower), _ptr(upper))
        return lower, upper

    @property
    def last_q(self):
        q = np.empty(self.dof)
        self._lib.dexrt_retargeter_last_q(self._handle, _ptr(q))
        return q

    def reset(self, q=None):
        err = _err()
        arr = _array(q, self.dof) if q is not None else None
        _check(self._lib.dexrt_retargeter_reset(self._handle, _ptr(arr) if arr is not None else None, err, len(err)), err)

    def solve(self, vectors):
        """Returns (q_rad[16], errors_m[10], info dict)."""
        target = _array(vectors, 3 * self.vector_count)
        q = np.empty(self.dof)
        errors = np.empty(self.vector_count)
        info = _SolveInfo()
        err = _err()
        _check(self._lib.dexrt_retargeter_solve(self._handle, _ptr(target), _ptr(q), _ptr(errors), ctypes.byref(info),
                                                err, len(err)), err)
        return q, errors, {name: getattr(info, name) for name, _ in _SolveInfo._fields_}

    def full_qpos(self, q):
        out = np.empty(self.full_dof)
        err = _err()
        _check(self._lib.dexrt_retargeter_full_qpos(self._handle, _ptr(_array(q, self.dof)), _ptr(out), err, len(err)), err)
        return out

    def robot_vectors(self, q):
        out = np.empty(3 * self.vector_count)
        err = _err()
        _check(self._lib.dexrt_retargeter_robot_vectors(self._handle, _ptr(_array(q, self.dof)), _ptr(out), err, len(err)),
               err)
        return out.reshape(-1, 3)

    def link_jacobian(self, q, link):
        """3 x dof Jacobian of the link origin w.r.t. the independent joints (m/rad), couplings folded in."""
        out = np.empty(3 * self.dof)
        err = _err()
        _check(self._lib.dexrt_retargeter_link_jacobian(self._handle, _ptr(_array(q, self.dof)), link.encode(), _ptr(out),
                                                        err, len(err)), err)
        return out.reshape(3, self.dof)

    def link_position(self, q, link):
        out = np.empty(3)
        err = _err()
        _check(self._lib.dexrt_retargeter_link_position(self._handle, _ptr(_array(q, self.dof)), link.encode(), _ptr(out),
                                                        err, len(err)), err)
        return out


class Impedance:
    """tau = kp*(q_target - q) - kd*qdot, clamped. Unit-agnostic; keep q, kp, kd consistent."""

    def __init__(self, kp, kd, tau_max, target_timeout_s=0.5, zero_torque_on_timeout=False,
                 velocity_window_s=0.01, deadband=0.0):
        self._lib = load_library()
        kp, kd, tau_max = _array(kp), _array(kd), _array(tau_max)
        self.n = kp.size
        err = _err()
        self._handle = self._lib.dexrt_impedance_create(self.n, _ptr(kp), _ptr(kd), _ptr(tau_max), target_timeout_s,
                                                        int(zero_torque_on_timeout), velocity_window_s, deadband,
                                                        err, len(err))
        if not self._handle:
            raise RuntimeError(err.value.decode(errors='replace'))

    def close(self):
        if getattr(self, '_handle', None):
            self._lib.dexrt_impedance_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def set_target(self, q_target, now):
        err = _err()
        _check(self._lib.dexrt_impedance_set_target(self._handle, _ptr(_array(q_target, self.n)), now, err, len(err)), err)

    def set_gains(self, kp, kd):
        err = _err()
        _check(self._lib.dexrt_impedance_set_gains(self._handle, _ptr(_array(kp, self.n)), _ptr(_array(kd, self.n)),
                                                   err, len(err)), err)

    def step(self, q, now, qdot=None):
        """Returns (tau, qdot_used, flags dict)."""
        q = _array(q, self.n)
        qdot_arr = _array(qdot, self.n) if qdot is not None else None
        tau, used = np.empty(self.n), np.empty(self.n)
        flags = ctypes.c_int(0)
        err = _err()
        _check(self._lib.dexrt_impedance_step(self._handle, _ptr(q), _ptr(qdot_arr) if qdot_arr is not None else None,
                                              now, _ptr(tau), _ptr(used), ctypes.byref(flags), err, len(err)), err)
        return tau, used, dict(has_target=bool(flags.value & 1), target_fresh=bool(flags.value & 2),
                               saturated=bool(flags.value & 4))

    def reset(self):
        self._lib.dexrt_impedance_reset(self._handle)


def default_compliance_config():
    cfg = _ComplianceConfig()
    load_library().dexrt_compliance_default_config(ctypes.byref(cfg))
    return {name: (list(getattr(cfg, name)) if name in ('readback_sign', 'lower_deg', 'upper_deg')
                   else getattr(cfg, name)) for name, _ in _ComplianceConfig._fields_}


class Compliance:
    """Position-mode compliance for the AAC D21 (see include/dexrt/compliance.hpp).

    All joint arrays are 16 values in SDK joint order, degrees, command frame; currents are the
    16 motor currents in amps.
    """
    ARRAYS = ('readback_sign', 'lower_deg', 'upper_deg')

    def __init__(self, **overrides):
        self._lib = load_library()
        cfg = _ComplianceConfig()
        self._lib.dexrt_compliance_default_config(ctypes.byref(cfg))
        fields = dict(_ComplianceConfig._fields_)
        for key, value in overrides.items():
            if key not in fields:
                raise TypeError(f'unknown compliance option {key}')
            if key in self.ARRAYS:
                setattr(cfg, key, (ctypes.c_double * 16)(*_array(value, 16)))
            elif fields[key] is ctypes.c_int:
                setattr(cfg, key, int(value))
            else:
                setattr(cfg, key, float(value))
        err = _err()
        self._handle = self._lib.dexrt_compliance_create(ctypes.byref(cfg), err, len(err))
        if not self._handle:
            raise RuntimeError(err.value.decode(errors='replace'))

    def close(self):
        if getattr(self, '_handle', None):
            self._lib.dexrt_compliance_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def add_idle_sample(self, currents):
        err = _err()
        _check(self._lib.dexrt_compliance_add_idle_sample(self._handle, _ptr(_array(currents, 16)), err, len(err)), err)

    def finish_idle(self):
        idle, err = np.empty(16), _err()
        _check(self._lib.dexrt_compliance_finish_idle(self._handle, _ptr(idle), err, len(err)), err)
        return idle

    def step(self, q_target, q_read, currents, now):
        """Returns dict with q_cmd, offset, effort, q_meas, velocity, yielding, deviation, source (16 each)."""
        keys = ('q_cmd', 'offset', 'effort', 'q_meas', 'velocity', 'yielding', 'deviation', 'source')
        out = {k: np.empty(16) for k in keys}
        err = _err()
        _check(self._lib.dexrt_compliance_step(
            self._handle, _ptr(_array(q_target, 16)), _ptr(_array(q_read, 16)), _ptr(_array(currents, 16)), now,
            *[_ptr(out[k]) for k in keys], err, len(err)), err)
        out['yielding'] = out['yielding'] > 0.5
        return out

    def reset(self):
        self._lib.dexrt_compliance_reset(self._handle)

    def drive_signs(self):
        """16x2 learned drive signs per joint and motor slot; 0 = not yet learned."""
        out = np.empty(32)
        self._lib.dexrt_compliance_drive_signs(self._handle, _ptr(out))
        return out.reshape(16, 2)


class UdpTargetSender:
    """Sends 16 joint targets in DEGREES to dexrt_impedance_loop (little-endian float64 datagrams)."""

    def __init__(self, port=6400, host='127.0.0.1'):
        self.address = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def send(self, degrees):
        values = _array(degrees, 16)
        if not np.isfinite(values).all():
            raise ValueError('target must be finite')
        self.sock.sendto(struct.pack('<16d', *values), self.address)

    def close(self):
        self.sock.close()
