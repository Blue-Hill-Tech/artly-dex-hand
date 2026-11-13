"""Friction calibration in parallel: the joints of a group are measured and tuned at the same time
(friction_calibrate.py; the default, --sequential goes back to one joint at a time).

GROUPS: each thumb joint alone; the four finger MCPs together; the four finger PIPs together; the abductions in two
pairs, index + ring and middle + pinky, so two neighbouring fingers never spread at the same time. No two joints of a
group share a motor pair (a finger's abduction and MCP do, which is why they are never in one group).

The step scripts (retargetor/dex-retargetor-cpp/python/friction_breakaway.py, friction_tune.py) stay as they are:
their settings, helpers and data files are used, only the loops over joints are done here for a whole group.

Breakaway (breakaway_group): every joint of the group goes to its own test angle (k-th of its own list), then the
torque rises on all of them at once (STEP mA every HOLD s); a joint that moves gets zero torque and its current is its
reading, the others keep rising. Same skips, collision guard (abductions) and cap as the step script. Fine
(friction_fine.py): every reading 3 times with the head start, the median kept, as there.

Tune (tune_group): the joints run their triangles at the same time, each with its own setting, speed and centre;
the grid, the validation and the refinement go in lockstep but every joint is scored, refined and passed on its own
and gets its own result entry, as tune_joint / fine_tune_joint write it. Fine: friction_fine's shorter rounds (after a
refinement only the worst condition is run again, every condition once it passes there; a joint's rounds end early
when it cannot pass), fine_rounds. A run that aborts because a joint slipped
counts as aborted for that joint only: the others run it again without it.
"""
import json
import statistics
import sys
import time
from itertools import zip_longest

import numpy as np

GROUPS = (("thumb_abduction",), ("thumb_rotation",), ("thumb_cmc",), ("thumb_mcp",),
          ("index_mcp", "middle_mcp", "ring_mcp", "pinky_mcp"),
          ("index_pip", "middle_pip", "ring_pip", "pinky_pip"),
          ("index_abduction", "ring_abduction"),
          ("middle_abduction", "pinky_abduction"))

clock, sleep = time.monotonic, time.sleep        # replaced by the tests (a simulated hand)


def groups_for(joints):
    """The groups for these joints (each group keeps only the asked joints; a joint in no group runs alone)."""
    asked = list(joints)
    out = []
    for group in GROUPS:
        part = tuple(n for n in group if n in asked)
        if part:
            out.append(part)
    grouped = {n for g in out for n in g}
    out += [(n,) for n in asked if n not in grouped]
    return out


def _say(text):
    print(text, flush=True)


# --- step 1: breakaway ----------------------------------------------------------------------------------------------
def ramp(fb, h, joints, direction, start=None):
    """Raise the torque of all `joints` at once until each moves; {j: (mA or None, start angle)}.
    start: {j: first mA} (fine's head start), default fb.START. Per joint: fb.ABD_MOVE + its guard for the finger
    abductions, else fb.MOVE; None when it hit the cap or its guard."""
    pose, _ = h.write_torques_and_read_positions([0.0] * 16)
    q0 = {j: pose[j] for j in joints}
    ma = {j: (start or {}).get(j, fb.START) for j in joints}
    abd = {j: fb.JOINT_NAMES[j] in fb.FINGER_ABD for j in joints}
    move = {j: fb.ABD_MOVE if abd[j] else fb.MOVE for j in joints}
    home = fb.home_pose()
    result = {}
    active = [j for j in joints if ma[j] <= fb.MAX]
    result.update({j: None for j in joints if j not in active})
    while active:
        def command():
            tau = [0.0] * 16
            for j in active:
                tau[j] = direction * ma[j]
            return fb.joint_to_motor_torques(tau, fb.MAX, fb.POLARITY)
        cmd = command()
        t0 = clock()
        while active and clock() - t0 < fb.HOLD:
            pose, _ = h.write_torques_and_read_positions(cmd)
            done = []
            for j in active:
                if abd[j] and abs(pose[j] - home[j]) > fb.ABD_GUARD:
                    _say(f"    guard: {fb.JOINT_NAMES[j]} drifted to {pose[j]:.1f} deg, stopped")
                    result[j] = None
                    done.append(j)
                elif abs(pose[j] - q0[j]) > move[j]:
                    result[j] = ma[j]
                    done.append(j)
            if done:
                active = [j for j in active if j not in done]
                cmd = command()                          # the moved joints: zero torque, the others keep theirs
        for j in list(active):
            ma[j] += fb.STEP
            if ma[j] > fb.MAX:
                result[j] = None
                active.remove(j)
    h.write_torques_and_read_positions([0.0] * 16)
    return {j: (result[j], q0[j]) for j in joints}


def _torque_mode_at(fb, h, pose):
    """Position mode, to `pose`, then torque mode there (zero torque)."""
    h.write_motor_torques([0.0] * 16)
    h.write_positions(fb.clip(h.read_positions()))
    h.set_position_mode()
    fb.go_to(h, fb.clip(pose))
    h.write_positions(fb.clip(h.read_positions()))
    h.write_motor_torques([0.0] * 16)
    h.set_torque_mode()


def _fine_ramp(fb, h, joints, direction, pose):
    """friction_fine's measurement for a group: friction_fine.REPEATS readings per joint (back to `pose` between
    them), each ramp from friction_fine.HEAD_START x what is known (the coarse table, then the first reading); a joint
    that moved at its head start is measured again from fb.START. {j: (median mA or None, start angle)}."""
    import friction_fine as ff
    try:
        known = json.loads(fb.SIDE.coarse.read_text())
    except (OSError, ValueError):
        known = {}
    sign = "+" if direction > 0 else "-"

    def head_start(reference):
        if not reference:
            return fb.START
        return max(fb.START, ff.STEP_MA * round(ff.HEAD_START * float(reference) / ff.STEP_MA))

    readings = {j: [] for j in joints}
    q0, out = {}, {}
    live = list(joints)
    for k in range(ff.REPEATS):
        if not live:
            break
        if k:
            _torque_mode_at(fb, h, pose)
        start = {j: head_start(readings[j][0] if readings[j] else known.get(fb.JOINT_NAMES[j] + sign)) for j in live}
        got = ramp(fb, h, live, direction, start)
        for j, (_, q) in got.items():
            q0.setdefault(j, q)
        again = [j for j in live if got[j][0] is not None and start[j] > fb.START and got[j][0] <= start[j]]
        if again:                                        # moved at the head start: the friction is below it
            _torque_mode_at(fb, h, pose)
            got.update(ramp(fb, h, again, direction))
        for j in list(live):
            if got[j][0] is None:                        # cap or guard: no reading for this joint
                out[j] = (None, q0[j])
                live.remove(j)
            else:
                readings[j].append(got[j][0])
    for j in live:
        value = statistics.median(readings[j])
        _say(f"    {fb.JOINT_NAMES[j]:16s} readings {', '.join(f'{r:g}' for r in readings[j])} mA -> median {value:g}")
        out[j] = (value, q0[j])
    return out


def breakaway_group(fb, names, fine=False):
    """friction_breakaway.main for a whole group at once; writes the same data files."""
    results = json.loads(fb.OUT.read_text()) if fb.OUT.exists() else {"measurements": []}
    idx = [fb.JOINT_NAMES.index(n) for n in names]
    h = fb.open_hand()
    try:
        home = fb.home_pose()
        fb.go_to(h, home)
        results["measurements"] = [m for m in results["measurements"] if m["joint"] not in names]   # remeasure
        angle_lists = []
        for j in idx:
            lo, hi = fb.ANGLE_LIMITS[j]
            if fb.JOINT_NAMES[j] in fb.FINGER_ABD:
                angles = [home[j] - fb.ABD_SPAN, home[j], home[j] + fb.ABD_SPAN]
            else:
                angles = [lo + frac * (hi - lo) for frac in fb.DENSE.get(fb.JOINT_NAMES[j], fb.FRACTIONS)]
            angle_lists.append([min(max(a, lo), hi) for a in angles])
        for angles in zip_longest(*angle_lists):
            pose = list(home)
            here = []                                    # the joints with a k-th test angle (the others stay home)
            for j, angle in zip(idx, angles):
                if angle is not None:
                    pose[j] = angle
                    here.append(j)
            fb.go_to(h, pose)
            h.write_positions(fb.clip(h.read_positions()))
            h.write_motor_torques([0.0] * 16)
            h.set_torque_mode()
            for d in (+1, -1):
                q = h.read_positions()
                todo = []
                for j in here:
                    lo, hi = fb.ANGLE_LIMITS[j]
                    if (d > 0 and q[j] > hi - 6) or (d < 0 and q[j] < lo + 6):
                        continue
                    if fb.JOINT_NAMES[j] in fb.FINGER_ABD and (
                            (d > 0 and q[j] > home[j] + fb.ABD_SPAN + 1) or (d < 0 and q[j] < home[j] - fb.ABD_SPAN - 1)):
                        continue
                    todo.append(j)
                if todo:
                    got = _fine_ramp(fb, h, todo, d, pose) if fine else ramp(fb, h, todo, d)
                    for j in todo:
                        ma, q0 = got[j]
                        name = fb.JOINT_NAMES[j]
                        results["measurements"].append(dict(joint=name, direction="+" if d > 0 else "-",
                                                            angle_deg=round(q0, 1), breakaway_ma=ma))
                        _say(f"{name:16s} {'+' if d > 0 else '-'} at {q0:6.1f} deg: "
                             f"{ma if ma else '>' + str(int(fb.MAX))} mA")
                _torque_mode_at(fb, h, pose)             # back to the test angles for the other direction
            h.write_motor_torques([0.0] * 16)
            h.write_positions(fb.clip(h.read_positions()))
            h.set_position_mode()
        fb.go_to(h, home)
    finally:
        try:
            h.write_motor_torques([0.0] * 16)
            h.write_positions(fb.clip(h.read_positions()))
            h.set_position_mode()
        finally:
            h.close()
    table = {}
    for m in results["measurements"]:
        if m["breakaway_ma"]:
            table.setdefault(m["joint"] + m["direction"], []).append(m["breakaway_ma"])
    results["table"] = {k: round(sum(v) / len(v), 1) for k, v in table.items()}
    fb.OUT.write_text(json.dumps(results, indent=1) + "\n")
    if fb.SIDE.side != "left":
        fb.SIDE.coarse.write_text(json.dumps({n + d: results["table"].get(n + d) for n in fb.JOINT_NAMES for d in "+-"},
                                             indent=1) + "\n")
    _say(f"saved {fb.OUT}")


# --- step 2: tune ---------------------------------------------------------------------------------------------------
def triangle(t, amp, speed):
    period = 4 * amp / speed
    x = (t % period) / period
    return amp * (4 * x if x < 0.25 else 2 - 4 * x if x < 0.75 else 4 * x - 4)


class GroupSession:
    """friction_tune.Session for several joints: one torque-mode controller session, each run moves every joint of
    `settings` on its own triangle with its own friction setting."""

    def __init__(self, ft):
        self.ft = ft
        self._open()

    def _open(self):                                  # (the tests replace this with a simulated controller)
        import contextlib
        import io
        from types import SimpleNamespace
        import zmq
        ft = self.ft
        self.settings = ft.load_aac_config(SimpleNamespace(hand_serial=ft.SIDE.serial), ft.CONFIG)
        device = ft.AACHand(self.settings.pop("serial_device"), self.settings.pop("device_address"),
                            self.settings.pop("connection_timeout_s"), bus=self.settings.pop("bus"))
        self.ctx = zmq.Context()
        self.ctl = ft.AACController(device, context=self.ctx, **self.settings)
        with contextlib.redirect_stdout(io.StringIO()):
            self.ctl.open()
        self.base = ft.clip(list(self.ctl.measured))
        self.plus, self.minus = ft.breakaway_tables()
        self.ctl.breakaway_model = ft.BreakawayModel(ft.FINE) if ft.FINE.exists() else None

    def close(self):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            self.ctl.close()
        if getattr(self, "ctx", None) is not None:
            self.ctx.term()

    def run(self, settings, secs=5.0, hold=0.0, settling=()):
        """settings: {j: (err_share, ff_share, kd, dither, speed, centre)}. {j: result} as Session.run's, plus
        speed and centre. A joint that slips aborts the run for itself; the others run it again without it (the
        slipped joint, back at its hold angle, is not taken for a new slip during that run's first second)."""
        res = self._run_once(settings, secs, hold, settling)
        culprit = res.pop("_culprit", None)
        if culprit is not None and culprit in settings and len(settings) > 1:
            rest = {j: s for j, s in settings.items() if j != culprit}
            _say(f"    {self.ft.JOINT_NAMES[culprit]} aborted the group run: again without it for "
                 + ", ".join(self.ft.JOINT_NAMES[j] for j in rest))
            res.update(self.run(rest, secs, hold, settling=(culprit,)))
        return res

    def _run_once(self, settings, secs, hold, settling=()):
        ft, ctl = self.ft, self.ctl
        joints = list(settings)
        options = {}
        ctl.friction_table = [0.0] * 16
        ffs = [0.0] * 16
        for j, (es, ff, kd, dither, _speed, _centre) in settings.items():
            opts = {"friction_scale": es}
            if kd is not None:
                opts["kd"] = kd
            if dither is not None:
                opts["dither_ma"] = dither
            options[ft.JOINT_NAMES[j]] = opts
            ctl.friction_table[j] = es * 0.5 * (self.plus[j] + self.minus[j])
            ffs[j] = ff
        ctl.torque_options["joints"] = options
        ctl.torque = ctl.torque_factory(ctl.torque_options)        # rebuilds kp/kd per joint
        ctl.torque_options["friction_scale"] = 0.0
        ctl.ff_scale = ffs if any(ffs) else None
        ctl.breakaway = (self.plus, self.minus)
        ctl.target_velocity = ft.TargetVelocity(0.03)
        hand_id = self.settings["hand_id"]
        ctl.request({"cmd": "set_mode", "hand_id": hand_id, "mode": "execute"})
        start = {j: ctl.measured[j] for j in joints}
        amp = {j: ft.amp_for(j) for j in joints}
        speed = {j: settings[j][4] for j in joints}
        centre = {j: settings[j][5] for j in joints}
        t0 = clock()
        rec, last_cmd = [], None
        abort, culprit, fast_ticks = {}, None, 0
        total = 1.0 + secs + hold
        while clock() - t0 < total:
            now = clock()
            el = now - t0
            tgt = list(self.base)
            for j in joints:
                if el < 1.0:
                    tgt[j] = start[j] + (centre[j] - start[j]) * el
                elif el < 1.0 + secs:
                    tgt[j] = centre[j] + triangle(el - 1.0, amp[j], speed[j])
                else:
                    tgt[j] = centre[j] + triangle(secs, amp[j], speed[j])
            tgt = ft.clip(tgt)
            ctl.accept_target(dict(type="hand_stream", hand_id=hand_id, t=time.time(), execution_id=ctl.execution_id,
                                   angle_unit="deg", angle=tgt))
            ctl.tick()
            if ctl.torque_command is not None and ctl.torque_command is not last_cmd:
                last_cmd = ctl.torque_command
                if len(rec) >= 2:
                    dt = now - rec[-1][0]
                    sp = [abs(a - b) / dt for a, b in zip(ctl.measured, rec[-1][2])]
                    if el < 1.0:                 # a joint that just slipped returns to its hold angle meanwhile
                        for j in settling:
                            sp[j] = 0.0
                    fast_ticks = fast_ticks + 1 if (dt > 0.002 and max(sp) > ft.MAX_SPEED) else 0
                    if fast_ticks >= 2:          # two consecutive fast samples: a real slip, not a readback glitch
                        culprit = int(np.argmax(sp))
                        text = f"{ft.JOINT_NAMES[culprit]} {max(sp):.0f} deg/s at {el:.2f} s"
                        abort = {j: text for j in joints} if culprit not in joints else {culprit: text}
                        break
                    drifted = [j for j in joints if ft.JOINT_NAMES[j] in ft.FINGER_ABD
                               and abs(ctl.measured[j] - centre[j]) > 8.0]
                    if drifted:
                        culprit = drifted[0]
                        abort = {culprit: f"{ft.JOINT_NAMES[culprit]} drifted {ctl.measured[culprit] - centre[culprit]:+.1f} "
                                          f"deg from home (collision guard)"}
                        break
                rec.append((now, list(tgt), list(ctl.measured)))
        ctl.request({"cmd": "set_mode", "hand_id": hand_id, "mode": "idle"})
        sleep(0.4)
        out = {}
        t = np.array([r[0] for r in rec]) - t0 if rec else np.zeros(0)
        for j in joints:
            es, ff, kd, dither, _s, _c = settings[j]
            res = dict(err_share=es, ff_share=ff, aborted=abort.get(j), kd=kd, dither=dither, speed=speed[j],
                       centre=centre[j])
            T = np.array([r[1][j] for r in rec])
            M = np.array([r[2][j] for r in rec])
            k = (t > 1.5) & (t < 1.0 + secs)
            if k.sum() > 100:
                vt, vm = np.gradient(T, t), np.gradient(M, t)
                period = 4 * amp[j] / speed[j]
                phase = (t - 1.0) % period
                near_reversal = np.minimum(np.abs(phase - period / 4), np.abs(phase - 3 * period / 4)) < 0.12
                mv = k & (np.abs(vt) > 10) & ~near_reversal
                res.update(err=float(np.abs(T[k] - M[k]).mean()), err_max=float(np.abs(T[k] - M[k]).max()),
                           still=float(np.mean(np.abs(vm[mv]) < 3)) if mv.any() else 1.0,
                           vel_p95=float(np.percentile(np.abs(vm[k]), 95)))
                if hold > 0:
                    kh = t > 1.0 + secs + 0.5
                    res["hold_err"] = float(np.abs(T[kh] - M[kh]).mean()) if kh.any() else None
            else:
                res.update(err=99.0, err_max=99.0, still=1.0, vel_p95=999.0)
            out[j] = res
        if culprit is not None and culprit in joints:
            out["_culprit"] = culprit
        return out


def fine_rounds(ft, s, idx, name_of, conditions, best, logs):
    """The fine validation and refinement of a group (friction_fine's rules, joints in lockstep): every condition is
    validated once; a joint that fails is refined at its own worst condition, then only that condition is run again
    (RE-CHECK) and every condition only once it passes there. A joint's rounds end early when a refinement found no
    better setting or improved too little to pass in the rounds left (friction_fine.give_up). `best` is updated in
    place; returns {j: the validation result to report}."""
    import friction_fine as ffine
    latest, final = {}, {}

    def verdict(j, text):
        f = final[j]
        _say(f"  {name_of[j]:16s} worst: |err| {f['err']:.1f} max {f['err_max']:.1f} vel {f['vel_p95']:.0f}"
             f"{' ABORT' if f['aborted'] else ''}  -> {text}")

    def validate(joints):
        runs = {j: {} for j in joints}
        for c in range(max(len(conditions[j]) for j in joints)):
            settings = {j: best[j] + conditions[j][c] for j in joints if c < len(conditions[j])}
            for j, v in s.run(settings, secs=8.0, hold=2.0).items():
                runs[j][(v["speed"], v["centre"])] = v
                logs[j].append(dict(v, validation=True))
                _say(f"  {name_of[j]:16s} VALIDATE {v['speed']:4.0f} deg/s @ {v['centre']:5.1f} {ffine.shares(best[j])}: "
                     f"{ft.fmt(v)}")
        for j in joints:
            latest[j] = runs[j]
            final[j] = ffine.final_of(latest[j], best[j])

    validate(idx)
    left = {j: ffine.ROUNDS for j in idx}
    pending = []
    for j in idx:
        ok = ffine.fully_passed(ft, final[j], conditions[j])
        verdict(j, "PASS" if ok else "refine" if left[j] else "FAIL")
        if not ok and left[j]:
            pending.append(j)
    while pending:
        # each joint at its own worst condition: the current best there against small nudges
        focus, before, here, candidates = {}, {}, {}, {}
        for j in pending:
            bad = ffine.worst_run(ft, latest[j], best[j])
            focus[j] = (bad["speed"], bad["centre"])
            before[j] = ffine.excess(ft, bad)
            here[j] = {best[j]: bad}
            candidates[j] = ffine.candidates_around(best[j], here[j])
        for i in range(3):
            settings = {j: candidates[j][i] + focus[j] for j in pending if i < len(candidates[j])}
            if not settings:
                break
            for j, r in s.run(settings).items():
                here[j][tuple(settings[j][:4])] = r
                logs[j].append(r)
                _say(f"  {name_of[j]:16s} refine   {focus[j][0]:4.0f} deg/s @ {focus[j][1]:5.1f} "
                     f"{ffine.shares(tuple(settings[j][:4]))}: {ft.fmt(r)}")
        recheck = []
        for j in list(pending):
            left[j] -= 1
            better = min(here[j], key=lambda key, pool=here[j]: ft.score(pool[key]))
            if better == best[j]:
                _say(f"  {name_of[j]:16s} -> FAIL: no setting one step away was better at {focus[j][0]:g} deg/s @ "
                     f"{focus[j][1]:g}; rounds ended early")
                pending.remove(j)
            else:
                best[j] = better
                recheck.append(j)
        if not recheck:
            continue
        # only that condition again; every condition once it passes there
        again = []
        for j, check in s.run({j: best[j] + focus[j] for j in recheck}, secs=8.0, hold=2.0).items():
            logs[j].append(dict(check, validation=True, recheck=True))
            latest[j][focus[j]] = check
            _say(f"  {name_of[j]:16s} RE-CHECK {focus[j][0]:4.0f} deg/s @ {focus[j][1]:5.1f} {ffine.shares(best[j])}: "
                 f"{ft.fmt(check)}")
            if ft.passed(check):
                again.append(j)
                continue
            final[j] = ffine.final_of(latest[j], best[j])
            stop, why = ffine.give_up(before[j], ffine.excess(ft, check), left[j])
            if stop or left[j] == 0:
                verdict(j, "FAIL: " + (why + "; rounds ended early" if stop else "no rounds left"))
                pending.remove(j)
            else:
                verdict(j, "refine")
        if again:
            validate(again)
            for j in again:
                ok = ffine.fully_passed(ft, final[j], conditions[j])
                verdict(j, "PASS" if ok else "refine" if left[j] else "FAIL")
                if ok or left[j] == 0:
                    pending.remove(j)
    return final


def tune_group(ft, names, fine=False):
    """friction_tune.tune_joint (or friction_fine.fine_tune_joint) for a whole group at once; writes the same result
    entry per joint. Returns {name: result}."""
    import friction_fine as ffine
    if fine:
        ft.ERR_OK = ffine.ERR_OK
    idx = [ft.JOINT_NAMES.index(n) for n in names]
    name_of = {j: ft.JOINT_NAMES[j] for j in idx}

    def normal(j):                                  # the grid's and the normal validation's condition
        return ft.SPEED, ft.centre(j)

    conditions = {j: ([(v, c) for v in ffine.SPEEDS for c in ffine.positions(ft, j)] if fine else [normal(j)])
                  for j in idx}
    if fine:
        for j in idx:
            _say(f"  {name_of[j]:16s} validation at {', '.join(f'{v:g}' for v in ffine.SPEEDS)} deg/s around "
                 f"{', '.join(f'{c:g}' for c in ffine.positions(ft, j))} deg; pass: mean error <= {ffine.ERR_OK:g} deg")
    ft.home()
    s = GroupSession(ft)
    logs = {j: [] for j in idx}
    final, best = {}, {}
    try:
        # 1. the first grid, in lockstep (a joint whose grid is shorter or that gave up stays at home meanwhile)
        tried = {j: {} for j in idx}
        grids = {j: list(ft.SPECIAL.get(name_of[j], [(es, ff, None, None) for es, ff in ft.GRID])) for j in idx}
        gave_up = set()
        for k in range(max(len(g) for g in grids.values())):
            settings = {j: tuple(grids[j][k]) + normal(j) for j in idx if k < len(grids[j]) and j not in gave_up}
            if not settings:
                break
            for j, r in s.run(settings).items():
                tried[j][tuple(grids[j][k])] = r
                logs[j].append(r)
                _say(f"  {name_of[j]:16s} tune     {ft.fmt(r)}")
                if r["aborted"] and name_of[j] not in ft.SPECIAL and r["ff_share"] >= 0.7:
                    gave_up.add(j)                       # the plain grid only gets more aggressive from here
        best = {j: min(tried[j], key=lambda key, j=j: ft.score(tried[j][key])) for j in idx}
        if fine:
            # 2. fine: validate every condition, refine at the worst, re-check it, stop early when it cannot pass
            final = fine_rounds(ft, s, idx, name_of, conditions, best, logs)
        else:
            # 2. normal: validate, refine the joints that did not pass, until all passed or the rounds are up
            pending = list(idx)
            for rnd in range(ft.ROUNDS + 1):
                settings = {j: best[j] + normal(j) for j in pending}
                refine = []
                for j, v in s.run(settings, secs=8.0, hold=2.0).items():
                    logs[j].append(dict(v, validation=True))
                    final[j] = v
                    ok = ft.passed(v)
                    _say(f"  {name_of[j]:16s} VALIDATE: {ft.fmt(v)}  -> "
                         f"{'PASS' if ok else 'refine' if rnd < ft.ROUNDS else 'FAIL'}")
                    if not ok and rnd < ft.ROUNDS:
                        refine.append(j)
                pending = refine
                if not pending:
                    break
                # refine around the best: nudge ff by +-0.1 and the error share by +-0.1, keep the better
                candidates = {}
                for j in pending:
                    es0, ff0, kd, dither = best[j]
                    cands = [(round(es0 + de, 2), round(ff0 + df, 2), kd, dither)
                             for de, df in ((0, 0.1), (0, -0.1), (-0.1, 0.1), (0.1, 0))]
                    candidates[j] = [c for c in cands if 0 <= c[0] <= 0.6 and 0 <= c[1] <= 1.0 and c not in tried[j]][:3]
                for i in range(3):
                    settings = {j: candidates[j][i] + normal(j) for j in pending if i < len(candidates[j])}
                    if not settings:
                        break
                    for j, r in s.run(settings).items():
                        tried[j][tuple(settings[j][:4])] = r
                        logs[j].append(r)
                        _say(f"  {name_of[j]:16s} refine  : {ft.fmt(r)}")
                for j in pending:
                    best[j] = min(tried[j], key=lambda key, j=j: ft.score(tried[j][key]))
    finally:
        s.close()
    ft.home()
    data = json.loads(ft.OUT.read_text()) if ft.OUT.exists() else {}
    out = {}
    for j in idx:
        es, ff, kd, dither = best[j]
        f = final.get(j)
        result = dict(joint=name_of[j], err_share=es, ff_share=ff, kd=kd, dither=dither, validation=f,
                      passed=(ffine.fully_passed(ft, f, conditions[j]) if fine else bool(f is not None and ft.passed(f))),
                      runs=logs[j])
        if fine:
            result.update(fine=True, validated_conditions=len(f["conditions"]) if f else 0, conditions=len(conditions[j]))
        data[name_of[j]] = result
        out[name_of[j]] = result
    ft.OUT.write_text(json.dumps(data, indent=1) + "\n")
    sys.stdout.flush()
    return out
