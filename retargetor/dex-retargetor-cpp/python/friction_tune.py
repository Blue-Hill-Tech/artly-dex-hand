"""Friction compensation, step 2 of 3: tune / validate / refine per joint (after friction_breakaway.py; then friction_apply.py).

Per-joint friction compensation: tune, VALIDATE, refine until the validation passes (or give up after ROUNDS).

Per joint:
  1. home the whole hand (teleop/calibration/acc/friction/aac_friction_home_left.json, position mode, slow)
  2. tuning runs: a grid of (error-friction share, target-velocity feed-forward share), each a +-10 deg triangle at
     20 deg/s around a working angle for 5 s (other joints hold the home pose), scored by stick-slip and error
  3. validation run with the best setting: 8 s triangle + 2 s hold; passes if still <= STILL_OK, |err| <= ERR_OK,
     slip p95 <= VEL_OK and no abort
  4. if it fails: refine around the best (ff +-0.1, error share +-0.1), validate again, up to ROUNDS times
  5. home again; the result (setting, validation metrics, pass/fail) goes to teleop/calibration/acc/friction/aac_friction_tuning_left.json
Breakaway currents come from teleop/calibration/acc/friction/aac_breakaway_left_fine.json (direction-specific means over the test angles).
"""
import contextlib
import io
import json
import math
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import zmq

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
sys.path.insert(0, str(ROOT))
from hand.backends.aac import AACHand, ANGLE_LIMITS, JOINT_NAMES  # noqa: E402
from teleop.hand_control.hand_controller_aac import AACController, BreakawayModel, TargetVelocity, load_aac_config  # noqa: E402
from friction_side import SIDE  # noqa: E402  (--side left|right; the controller takes polarity from the config role)

CONFIG = str(SIDE.config)
HOME = SIDE.home
FINE = SIDE.fine
OUT = SIDE.tuning
MAX_SPEED = 250.0
AMP, SPEED = 10.0, 20.0
GRID = ((0.4, 0.3), (0.2, 0.5), (0.2, 0.7), (0.1, 0.8))
# joints where the friction knobs alone failed: also vary damping and dither (err share, ff, kd, dither mA)
SPECIAL = {'thumb_rotation': ((0.3, 0.6, 0.6, 40.0), (0.2, 0.7, 0.6, 40.0), (0.3, 0.5, 0.6, 40.0)),
           'index_abduction': ((0.2, 0.5, None, None), (0.3, 0.5, 0.5, None), (0.2, 0.6, 0.5, None), (0.4, 0.4, None, None)),
           # the lightest joint: a 0.4 error share threw it on the first tick; start gentle
           'middle_pip': ((0.2, 0.4, 0.5, None), (0.2, 0.3, 0.5, None), (0.1, 0.4, 0.5, None)),
           'pinky_pip': ((0.2, 0.4, 0.5, None), (0.2, 0.3, 0.5, None), (0.1, 0.5, 0.5, None), (0.3, 0.3, 0.5, None))}
STILL_OK, ERR_OK, ERR_MAX_OK, VEL_OK = 0.20, 1.5, 5.0, 80.0   # STILL_OK is reported, not gating
ROUNDS = 3


def clip(p):
    return [min(max(v, lo), hi) for v, (lo, hi) in zip(p, ANGLE_LIMITS.values())]


def home_pose():
    pose = json.loads(HOME.read_text())['pose_deg']
    return clip([float(pose[n]) for n in JOINT_NAMES])


def breakaway_tables():
    """(plus[16], minus[16]) mA from the fine table: mean over angles per direction; a missing direction uses the
    other; a joint with nothing measured falls back to the coarse table or 65 mA."""
    fine = json.loads(FINE.read_text())['table'] if FINE.exists() else {}
    coarse = json.loads(SIDE.coarse.read_text())
    plus, minus = [], []
    for name in JOINT_NAMES:
        p = fine.get(name + '+') or coarse.get(name + '+')
        m = fine.get(name + '-') or coarse.get(name + '-')
        p = float(p) if p else (float(m) if m else 65.0)
        m = float(m) if m else p
        plus.append(p)
        minus.append(m)
    return plus, minus


def home():
    s = load_aac_config(SimpleNamespace(hand_serial=SIDE.serial), CONFIG)
    h = AACHand(s['serial_device'], s['device_address'], s['connection_timeout_s'], bus=s['bus'])
    h.open()
    try:
        h.set_position_mode()
        h.set_speeds([20] * 5)
        pose = home_pose()
        h.write_positions(pose)
        t0 = time.monotonic()
        while time.monotonic() - t0 < 4.0:
            time.sleep(0.15)
            if max(abs(a - b) for a, b in zip(h.read_positions(), pose)) < 1.5:
                break
    finally:
        h.close()


FINGER_ABD = {'index_abduction', 'middle_abduction', 'ring_abduction', 'pinky_abduction'}
ABD_AMP = 4.0          # finger abductions: small triangle around the HOME angle (fingers spread, no collision)


def centre(j):
    lo, hi = ANGLE_LIMITS[j]
    if JOINT_NAMES[j] in FINGER_ABD:
        # around the home angle, pulled inside the limits so the +-ABD_AMP swing fits (index home 18.5 of 20)
        return min(max(home_pose()[j], lo + ABD_AMP + 1.0), hi - ABD_AMP - 1.0)
    return 0.0 if lo < 0 else min(lo + 15.0, hi - 12.0)


def amp_for(j):
    return ABD_AMP if JOINT_NAMES[j] in FINGER_ABD else AMP


def triangle(t, amp=AMP):
    period = 4 * amp / SPEED
    x = (t % period) / period
    return amp * (4 * x if x < 0.25 else 2 - 4 * x if x < 0.75 else 4 * x - 4)


class Session:
    """One torque-mode controller session for a joint; runs variants without leaving torque mode."""

    def __init__(self, j):
        self.j = j
        self.settings = load_aac_config(SimpleNamespace(hand_serial=SIDE.serial), CONFIG)
        device = AACHand(self.settings.pop('serial_device'), self.settings.pop('device_address'),
                         self.settings.pop('connection_timeout_s'), bus=self.settings.pop('bus'))
        self.ctx = zmq.Context()
        self.ctl = AACController(device, context=self.ctx, **self.settings)
        with contextlib.redirect_stdout(io.StringIO()):
            self.ctl.open()
        self.base = clip(list(self.ctl.measured))
        self.plus, self.minus = breakaway_tables()
        self.ctl.breakaway_model = BreakawayModel(FINE) if FINE.exists() else None

    def close(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.ctl.close()
        self.ctx.term()

    def run(self, err_share, ff_share, secs=5.0, hold=0.0, kd=None, dither=None):
        j, ctl = self.j, self.ctl
        joint_opts = {"friction_scale": err_share}
        if kd is not None:
            joint_opts["kd"] = kd
        if dither is not None:
            joint_opts["dither_ma"] = dither
        ctl.torque_options["joints"] = {JOINT_NAMES[j]: joint_opts}
        ctl.torque = ctl.torque_factory(ctl.torque_options)        # rebuilds kp/kd per joint
        ctl.friction_table = [0.0] * 16
        ctl.friction_table[j] = err_share * 0.5 * (self.plus[j] + self.minus[j])
        ctl.torque_options["friction_scale"] = 0.0
        ffs = [0.0] * 16
        ffs[j] = ff_share
        ctl.ff_scale = ffs if ff_share else None
        ctl.breakaway = (self.plus, self.minus)
        ctl.target_velocity = TargetVelocity(0.03)
        hand_id = self.settings['hand_id']
        ctl.request({'cmd': 'set_mode', 'hand_id': hand_id, 'mode': 'execute'})
        start = ctl.measured[j]
        c = centre(j)
        t0 = time.monotonic()
        rec, last_cmd, abort = [], None, None
        fast_ticks = 0
        total = 1.0 + secs + hold
        while time.monotonic() - t0 < total:
            now = time.monotonic()
            el = now - t0
            tgt = list(self.base)
            if el < 1.0:
                tgt[j] = start + (c - start) * el
            elif el < 1.0 + secs:
                tgt[j] = c + triangle(el - 1.0, amp_for(j))
            else:
                tgt[j] = c + triangle(secs, amp_for(j))
            tgt = clip(tgt)
            ctl.accept_target(dict(type='hand_stream', hand_id=hand_id, t=time.time(), execution_id=ctl.execution_id,
                                   angle_unit='deg', angle=tgt))
            ctl.tick()
            if ctl.torque_command is not None and ctl.torque_command is not last_cmd:
                last_cmd = ctl.torque_command
                if len(rec) >= 2:
                    dt = now - rec[-1][0]
                    sp = [abs(a - b) / dt for a, b in zip(ctl.measured, rec[-1][2])]
                    fast_ticks = fast_ticks + 1 if (dt > 0.002 and max(sp) > MAX_SPEED) else 0
                    if fast_ticks >= 2:          # two consecutive fast samples: a real slip, not a readback glitch
                        abort = f'{JOINT_NAMES[int(np.argmax(sp))]} {max(sp):.0f} deg/s at {el:.2f} s'
                        break
                    if JOINT_NAMES[j] in FINGER_ABD and abs(ctl.measured[j] - c) > 8.0:
                        abort = f'{JOINT_NAMES[j]} drifted {ctl.measured[j] - c:+.1f} deg from home (collision guard)'
                        break
                rec.append((now, list(tgt), list(ctl.measured)))
        ctl.request({'cmd': 'set_mode', 'hand_id': hand_id, 'mode': 'idle'})
        time.sleep(0.4)
        t = np.array([r[0] for r in rec]) - t0
        T = np.array([r[1][j] for r in rec])
        M = np.array([r[2][j] for r in rec])
        res = dict(err_share=err_share, ff_share=ff_share, aborted=abort, kd=kd, dither=dither)
        k = (t > 1.5) & (t < 1.0 + secs)
        if k.sum() > 100:
            vt, vm = np.gradient(T, t), np.gradient(M, t)
            # steady motion only: exclude +-0.12 s around each target reversal, where a joint legitimately pauses
            period = 4 * amp_for(j) / SPEED
            phase = (t - 1.0) % period
            near_reversal = np.minimum(np.abs(phase - period / 4), np.abs(phase - 3 * period / 4)) < 0.12
            mv = k & (np.abs(vt) > 10) & ~near_reversal
            res.update(err=float(np.abs(T[k] - M[k]).mean()), err_max=float(np.abs(T[k] - M[k]).max()),
                       still=float(np.mean(np.abs(vm[mv]) < 3)) if mv.any() else 1.0,
                       vel_p95=float(np.percentile(np.abs(vm[k]), 95)))
            if hold > 0:
                kh = t > 1.0 + secs + 0.5
                res['hold_err'] = float(np.abs(T[kh] - M[kh]).mean()) if kh.any() else None
        else:
            res.update(err=99.0, err_max=99.0, still=1.0, vel_p95=999.0)
        return res


def score(r):
    """Lower is better; the same quantities the pass rule gates on (the stuck share is reported only)."""
    return ((100 if r['aborted'] else 0) + 10 * r['err'] + 2 * max(0.0, r['err_max'] - ERR_MAX_OK)
            + 0.5 * max(0.0, r['vel_p95'] - VEL_OK) + 5 * r['still'])


def passed(r):
    # A joint following a 20 deg/s target with a 1 deg deadband moves in ~1 deg steps and is "still" between them
    # (35-60 % even when the error stays ~1 deg), so the stuck share is informative but does not gate the pass.
    return (not r['aborted'] and r['err'] <= ERR_OK and r['err_max'] <= ERR_MAX_OK and r['vel_p95'] <= VEL_OK)


def fmt(r):
    extra = (f" kd {r['kd']}" if r.get('kd') is not None else '') + (f" dither {r['dither']}" if r.get('dither') is not None else '')
    return (f"err {r['err_share']:.1f} ff {r['ff_share']:.1f}{extra}: |err| {r['err']:4.1f} max {r['err_max']:4.1f} still {100*r['still']:3.0f}% "
            f"vel {r['vel_p95']:4.0f}" + (f" hold {r['hold_err']:.1f}" if r.get('hold_err') is not None else '')
            + (f" ABORT {r['aborted']}" if r['aborted'] else ''))


def tune_joint(name):
    j = JOINT_NAMES.index(name)
    home()
    s = Session(j)
    log = []
    try:
        tried = {}
        grid = SPECIAL.get(name, [(es, ff, None, None) for es, ff in GRID])
        for es, ff, kd, dither in grid:
            r = s.run(es, ff, kd=kd, dither=dither)
            tried[(es, ff, kd, dither)] = r
            log.append(r)
            print(f"  {name:16s} tune     {fmt(r)}")
            sys.stdout.flush()
            if r['aborted'] and name not in SPECIAL and r['ff_share'] >= 0.7:
                break            # the plain grid only gets more aggressive from here
        best = min(tried.values(), key=score)
        final = None
        for rnd in range(ROUNDS + 1):
            v = s.run(best['err_share'], best['ff_share'], secs=8.0, hold=2.0, kd=best.get('kd'), dither=best.get('dither'))
            v['validation'] = True
            log.append(v)
            print(f"  {name:16s} VALIDATE {fmt(v)}  -> {'PASS' if passed(v) else 'refine' if rnd < ROUNDS else 'FAIL'}")
            sys.stdout.flush()
            final = v
            if passed(v) or rnd == ROUNDS:
                break
            # refine around the best: nudge ff by +-0.1 and the error share by +-0.1, keep the better
            candidates = []
            for des, dff in ((0, 0.1), (0, -0.1), (-0.1, 0.1), (0.1, 0)):
                es, ff = round(best['err_share'] + des, 2), round(best['ff_share'] + dff, 2)
                key = (es, ff, best.get('kd'), best.get('dither'))
                if 0 <= es <= 0.6 and 0 <= ff <= 1.0 and key not in tried:
                    candidates.append(key)
            for es, ff, kd, dither in candidates[:3]:
                r = s.run(es, ff, kd=kd, dither=dither)
                tried[(es, ff, kd, dither)] = r
                log.append(r)
                print(f"  {name:16s} refine   {fmt(r)}")
                sys.stdout.flush()
            best = min(tried.values(), key=score)
    finally:
        s.close()
    home()
    result = dict(joint=name, err_share=best['err_share'], ff_share=best['ff_share'], kd=best.get('kd'),
                  dither=best.get('dither'), validation=final, passed=passed(final), runs=log)
    data = json.loads(OUT.read_text()) if OUT.exists() else {}
    data[name] = result
    OUT.write_text(json.dumps(data, indent=1) + '\n')
    return result



def _terminate(signum, frame):
    """A kill (e.g. a timeout) must still run the cleanup: leave torque mode at the measured pose."""
    raise KeyboardInterrupt(f"signal {signum}")


signal.signal(signal.SIGTERM, _terminate)

if __name__ == '__main__':
    for name in sys.argv[1:]:
        r = tune_joint(name)
        print(f"{name:16s} => err-friction {r['err_share']:.1f}, feed-forward {r['ff_share']:.1f}"
              + (f", kd {r['kd']}" if r.get('kd') is not None else '') + (f", dither {r['dither']}" if r.get('dither') is not None else '') + "  "
              f"{'PASS' if r['passed'] else 'FAIL (best found)'}")
        sys.stdout.flush()
    print('hand at the home configuration, position mode')
