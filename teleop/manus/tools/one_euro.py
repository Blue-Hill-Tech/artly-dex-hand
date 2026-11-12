"""One-Euro filter for the retargeted joint targets, configured from YAML.

One-Euro (Casiez, Roussel & Vogel, CHI 2012) is a low-pass filter whose cutoff rises with speed:
    cutoff = min_cutoff_hz + beta * |filtered derivative|
Slow motion and holding still are smoothed hard (jitter removed); fast motion raises the cutoff so
the target lags little. Units here: degrees and degrees/second, so beta is in Hz per (deg/s).
"""
import math
from pathlib import Path

import numpy as np

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / 'parameters' / 'teleop_filter.yaml'
KEYS = {'min_cutoff_hz', 'beta', 'd_cutoff_hz'}


def _alpha(cutoff_hz, dt):
    tau = 1.0 / (2.0 * math.pi * cutoff_hz)
    return 1.0 / (1.0 + tau / dt)


class OneEuroFilter:
    """Per-channel One-Euro filter on a vector; min_cutoff_hz and beta may be per channel arrays."""

    def __init__(self, size, min_cutoff_hz=1.0, beta=0.02, d_cutoff_hz=1.0):
        self.size = int(size)
        self.min_cutoff = np.broadcast_to(np.asarray(min_cutoff_hz, dtype=float), (self.size,)).copy()
        self.beta = np.broadcast_to(np.asarray(beta, dtype=float), (self.size,)).copy()
        self.d_cutoff = float(d_cutoff_hz)
        if (self.min_cutoff <= 0).any() or (self.beta < 0).any() or self.d_cutoff <= 0:
            raise ValueError('One-Euro: min_cutoff_hz and d_cutoff_hz must be positive, beta non-negative')
        self.reset()

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, value, stamp):
        value = np.asarray(value, dtype=float)
        if value.shape != (self.size,) or not np.isfinite(value).all():
            raise ValueError(f'One-Euro: expected {self.size} finite values')
        if self.x is None or self.t is None or stamp <= self.t:
            # First sample, or a clock that did not advance: pass through and restart the derivative.
            self.x, self.dx, self.t = value.copy(), np.zeros(self.size), float(stamp)
            return value.copy()
        dt = float(stamp) - self.t
        a_d = _alpha(self.d_cutoff, dt)
        self.dx = a_d * (value - self.x) / dt + (1.0 - a_d) * self.dx
        cutoff = self.min_cutoff + self.beta * np.abs(self.dx)
        tau = 1.0 / (2.0 * np.pi * cutoff)
        a = 1.0 / (1.0 + tau / dt)
        self.x = a * value + (1.0 - a) * self.x
        self.t = float(stamp)
        return self.x.copy()

    def describe(self):
        return dict(type='one_euro', min_cutoff_hz=self.min_cutoff.tolist(), beta=self.beta.tolist(),
                    d_cutoff_hz=self.d_cutoff)


SIDES = ('left', 'right')


def load_filter(path, joint_names, side=None):
    """Build the filter from YAML; returns None when the file disables it.

    Format (see teleop_filter.yaml):
        one_euro:
          enabled: true
          min_cutoff_hz: 1.5
          beta: 0.05
          d_cutoff_hz: 1.0
          joints:                 # optional per-joint overrides, by SDK joint name (both hands)
            thumb_rotation: {min_cutoff_hz: 1.0, beta: 0.02}
          left:                   # optional per-hand block: overrides the settings above for that hand only.
            joints:               #   It may set enabled, min_cutoff_hz, beta, d_cutoff_hz and joints; its joints are
              thumb_mcp: {beta: 0.0}   #   merged into the shared ones per joint and per key.
          right:
            min_cutoff_hz: 0.8

    side: 'Left' / 'Right' (any case) applies that hand's block; None uses the shared settings only.
    """
    import yaml
    with open(path, encoding='utf-8') as f:
        doc = yaml.safe_load(f) or {}
    block = doc.get('one_euro')
    if not isinstance(block, dict):
        raise ValueError(f'{path}: missing "one_euro" section')
    unknown = set(block) - KEYS - {'enabled', 'joints'} - set(SIDES)
    if unknown:
        raise ValueError(f'{path}: unknown one_euro keys: {", ".join(sorted(unknown))}')
    for hand in SIDES:
        sub = block.get(hand)
        if sub is None:
            continue
        if not isinstance(sub, dict):
            raise ValueError(f'{path}: one_euro.{hand} must be a mapping')
        bad = set(sub) - KEYS - {'enabled', 'joints'}
        if bad:
            raise ValueError(f'{path}: unknown one_euro.{hand} keys: {", ".join(sorted(bad))}')
        joints = sub.get('joints') or {}
        if not isinstance(joints, dict):
            raise ValueError(f'{path}: one_euro.{hand}.joints must be a mapping of joint name -> settings')
        for joint, values in joints.items():         # checked for both hands, whichever one is loaded
            if joint not in joint_names:
                raise ValueError(f'{path}: unknown joint "{joint}" in one_euro.{hand}.joints')
            if not isinstance(values, dict) or set(values) - {'min_cutoff_hz', 'beta'}:
                raise ValueError(f'{path}: one_euro.{hand}.joints.{joint} may only set min_cutoff_hz and beta')
    if side is not None and str(side).lower() not in SIDES:
        raise ValueError(f'side must be Left or Right, got {side!r}')
    hand_block = (block.get(str(side).lower()) or {}) if side is not None else {}
    settings = {k: v for k, v in block.items() if k not in SIDES}
    settings.update({k: v for k, v in hand_block.items() if k != 'joints'})

    enabled = settings.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError(f'{path}: one_euro.enabled must be true or false')
    if not enabled:
        return None

    def number(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'{path}: {name} must be a finite number')
        return float(value)

    names = list(joint_names)
    min_cutoff = np.full(len(names), number(settings.get('min_cutoff_hz', 1.0), 'min_cutoff_hz'))
    beta = np.full(len(names), number(settings.get('beta', 0.02), 'beta'))
    for where, overrides in (('joints', block.get('joints')),
                             (f'{str(side).lower()}.joints', hand_block.get('joints'))):
        overrides = overrides or {}
        if not isinstance(overrides, dict):
            raise ValueError(f'{path}: one_euro.{where} must be a mapping of joint name -> settings')
        for joint, values in overrides.items():
            if joint not in names:
                raise ValueError(f'{path}: unknown joint "{joint}" in one_euro.{where} '
                                 f'(expected one of {", ".join(names)})')
            if not isinstance(values, dict) or set(values) - {'min_cutoff_hz', 'beta'}:
                raise ValueError(f'{path}: one_euro.{where}.{joint} may only set min_cutoff_hz and beta')
            i = names.index(joint)
            if 'min_cutoff_hz' in values:
                min_cutoff[i] = number(values['min_cutoff_hz'], f'{where}.{joint}.min_cutoff_hz')
            if 'beta' in values:
                beta[i] = number(values['beta'], f'{where}.{joint}.beta')
    return OneEuroFilter(len(names), min_cutoff, beta, number(settings.get('d_cutoff_hz', 1.0), 'd_cutoff_hz'))
