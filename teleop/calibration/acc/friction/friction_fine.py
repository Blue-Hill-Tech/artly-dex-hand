"""The FINE friction calibration (friction_calibrate.py --fine; the GUI's Dev Mode "Friction calibration: Fine").

The step scripts (retargetor/dex-retargetor-cpp/python/friction_breakaway.py, friction_tune.py) stay as they are;
this module changes their settings and wraps their measurement, the way friction_canfd.py swaps their config.

Step 1, breakaway (fine_breakaway):
  - 10 test angles along each joint's range (5 % ... 95 %) instead of 3 (thumb rotation: 5). The finger abductions
    keep their 3 angles around the home spread (more would bring them toward the neighbouring finger).
  - the current rises in 2 mA steps (not 5) until the joint moves 0.8 deg (not 1.5): closer to the true breakaway.
  - every reading 3 times, the median kept. Head start: a ramp starts at 60 % of what is already known (the earlier
    coarse table for the first reading, the first reading for the repeats), so the finer steps do not take 2.5x as
    long; a joint that moves at the head start itself is measured again from the normal start (20 mA).

Step 2, tune (fine_tune_joint):
  - the same first grid, at 20 deg/s around the joint's working angle.
  - validation at 3 speeds (10, 20, 40 deg/s) x 2 positions (30 % and 70 % of the range; finger abductions: their
    home spread only): the worst of the 6 runs decides, so a setting must work slow, fast and at both ends.
  - passes only with a mean error <= 1.0 deg (normal: 1.5); peak error, slip speed and aborts as before.
  - refine in steps of 0.05 (normal 0.1), up to 5 rounds (normal 3), each round at the condition that failed worst.
  - after a refinement only that condition is run again (RE-CHECK); all conditions are validated again only once it
    passes there, so a PASS always stands on a full validation of the final setting.
  - the rounds end early when a refinement found no better setting, or improved the worst condition by too little
    to pass in the rounds left (give_up): such rounds only repeated the same failure (2026-11-12, right thumb:
    6 full validations each, worst error 2.4 -> 2.2 -> 2.4 deg against the 1.0 limit, 9 min per joint).
"""
import json
import statistics
import sys

ANGLES = tuple(round(0.05 + 0.1 * i, 2) for i in range(10))    # 0.05 ... 0.95 of the range
STEP_MA, MOVE_DEG, REPEATS, HEAD_START = 2.0, 0.8, 3, 0.6
SPEEDS = (10.0, 20.0, 40.0)                                      # deg/s, validation
POSITIONS = (0.3, 0.7)                                           # of the joint's range, validation
REFINE, ROUNDS, ERR_OK = 0.05, 5, 1.0


def fine_breakaway(fb):
    """Switch friction_breakaway (the imported module) to the fine measurement."""
    fb.FRACTIONS = ANGLES
    fb.DENSE = {}
    fb.STEP = STEP_MA
    fb.MOVE = MOVE_DEG
    fb.ABD_MOVE = min(fb.ABD_MOVE, MOVE_DEG)
    single, normal_start = fb.breakaway, fb.START
    try:
        known = json.loads(fb.SIDE.coarse.read_text())
    except (OSError, ValueError):
        known = {}

    def measure(h, j, direction, move, guard, start):
        fb.START = start
        try:
            return single(h, j, direction, move=move, guard=guard)
        finally:
            fb.START = normal_start

    def back_to(h, j, angle):
        """Torque off, position mode, joint j back to `angle` (the others where they are), torque mode again."""
        h.write_motor_torques([0.0] * 16)
        h.write_positions(fb.clip(h.read_positions()))
        h.set_position_mode()
        pose = fb.clip(h.read_positions())
        pose[j] = angle
        fb.go_to(h, fb.clip(pose))
        h.write_positions(fb.clip(h.read_positions()))
        h.write_motor_torques([0.0] * 16)
        h.set_torque_mode()

    def head_start(reference):
        if not reference:
            return normal_start
        return max(normal_start, STEP_MA * round(HEAD_START * float(reference) / STEP_MA))

    def repeated(h, j, direction, move=fb.MOVE, guard=None):
        hint = known.get(fb.JOINT_NAMES[j] + ("+" if direction > 0 else "-"))
        readings, q0 = [], None
        for k in range(REPEATS):
            if k:
                back_to(h, j, q0)
            start = head_start(readings[0] if readings else hint)
            ma, q = measure(h, j, direction, move, guard, start)
            q0 = q if q0 is None else q0
            if ma is not None and start > normal_start and ma <= start:
                # it moved at the head start: the friction is below it, measure from the normal start
                back_to(h, j, q0)
                ma, _ = measure(h, j, direction, move, guard, normal_start)
            if ma is None:            # did not move up to the cap, or the abduction guard stopped it
                return None, q0
            readings.append(ma)
        value = statistics.median(readings)
        print(f"    readings {', '.join(f'{r:g}' for r in readings)} mA -> median {value:g}", flush=True)
        return value, q0

    fb.breakaway = repeated


def positions(ft, j):
    """Validation centres of joint j: 30 % and 70 % of its range, kept far enough in for the triangle; the finger
    abductions only around their home spread (the step script's own centre)."""
    if ft.JOINT_NAMES[j] in ft.FINGER_ABD:
        return [ft.centre(j)]
    lo, hi = ft.ANGLE_LIMITS[j]
    margin = ft.amp_for(j) + 1.0
    out = []
    for f in POSITIONS:
        c = round(min(max(lo + f * (hi - lo), lo + margin), hi - margin), 1)
        if c not in out:
            out.append(c)
    return out


def run_at(ft, s, setting, speed, centre, **kw):
    """One friction_tune Session run of `setting` at `speed` deg/s around `centre` deg."""
    es, ff, kd, dither = setting
    saved_speed, saved_centre = ft.SPEED, ft.centre
    ft.SPEED, ft.centre = speed, (lambda j: centre)
    try:
        r = s.run(es, ff, kd=kd, dither=dither, **kw)
    finally:
        ft.SPEED, ft.centre = saved_speed, saved_centre
    r.update(speed=speed, centre=centre)
    return r


def shares(setting):
    """The exact shares (the step script's fmt rounds them to 0.1, too coarse for 0.05 refinement steps)."""
    es, ff, kd, dither = setting
    return f"[err {es:.2f} ff {ff:.2f}" + (f" kd {kd:g}" if kd is not None else "") + \
        (f" dither {dither:g}" if dither is not None else "") + "]"


def worst_of(runs):
    """The validation result: the worst value of each metric over the conditions."""
    aborted = next((r["aborted"] for r in runs if r["aborted"]), None)
    worst = dict(runs[0], aborted=aborted, err=max(r["err"] for r in runs), err_max=max(r["err_max"] for r in runs),
                 vel_p95=max(r["vel_p95"] for r in runs), still=max(r["still"] for r in runs))
    holds = [r["hold_err"] for r in runs if r.get("hold_err") is not None]
    worst["hold_err"] = max(holds) if holds else None
    worst["conditions"] = [{k: r.get(k) for k in ("speed", "centre", "err", "err_max", "vel_p95", "still", "aborted")}
                           for r in runs]
    return worst


def setting_of(r):
    """The setting (error share, feed-forward share, kd, dither) a run was made with."""
    return (r["err_share"], r["ff_share"], r.get("kd"), r.get("dither"))


def excess(ft, r):
    """How far a run is outside the pass limits; 0 when it passes. Each limit's overshoot as a fraction of the limit
    (mean error, peak error, slip speed), plus 1 for an abort."""
    return (max(0.0, r["err"] - ft.ERR_OK) / ft.ERR_OK + max(0.0, r["err_max"] - ft.ERR_MAX_OK) / ft.ERR_MAX_OK
            + max(0.0, r["vel_p95"] - ft.VEL_OK) / ft.VEL_OK + (1.0 if r["aborted"] else 0.0))


def give_up(before, after, rounds_left):
    """After a refinement and the re-check of the worst condition: (stop, why). before / after: excess() there with
    the setting before and after the refinement. Stops when it did not get better, or when it got better by so
    little that as much again in every round left would still not pass."""
    gain = before - after
    if gain <= 0.0:
        return True, "the refinement did not improve the worst condition"
    if after > gain * rounds_left:
        return True, f"improving too slowly to pass in the {rounds_left} round(s) left"
    return False, ""


def candidates_around(best, tried):
    """Up to 3 settings one REFINE step from `best` that are not in `tried`."""
    es0, ff0, kd, dither = best
    nudges = ((0, REFINE), (0, -REFINE), (-REFINE, REFINE), (REFINE, 0))
    out = [(round(es0 + de, 2), round(ff0 + df, 2), kd, dither) for de, df in nudges]
    return [c for c in out if 0 <= c[0] <= 0.6 and 0 <= c[1] <= 1.0 and c not in tried][:3]


def worst_run(ft, latest, best):
    """The run to refine at: the worst-scoring one among those measured with `best` (after a failed re-check that is
    the re-checked condition: the other conditions' runs are still the previous setting's)."""
    return max((r for r in latest.values() if setting_of(r) == best), key=ft.score)


def final_of(latest, best):
    """The validation result to report for `best`: the worst over the conditions measured with it (all of them after
    a full validation; only the re-checked one when the rounds ended on a failed re-check)."""
    return worst_of([r for r in latest.values() if setting_of(r) == best])


def fully_passed(ft, final, conditions):
    """Passed, and on a validation of every condition."""
    return bool(final is not None and ft.passed(final) and len(final["conditions"]) == len(conditions))


def fine_tune_joint(ft, name):
    """friction_tune.tune_joint with the fine validation and refinement. Writes the same result entry."""
    ft.ERR_OK = ERR_OK
    j = ft.JOINT_NAMES.index(name)
    centres = positions(ft, j)
    conditions = [(v, c) for v in SPEEDS for c in centres]
    print(f"  {name:16s} validation at {', '.join(f'{v:g}' for v in SPEEDS)} deg/s around "
          f"{', '.join(f'{c:g}' for c in centres)} deg; pass: mean error <= {ERR_OK:g} deg", flush=True)
    ft.home()
    s = ft.Session(j)
    log = []
    try:
        tried = {}
        grid = ft.SPECIAL.get(name, [(es, ff, None, None) for es, ff in ft.GRID])
        for setting in grid:                          # the first grid at the normal speed and working angle
            r = s.run(setting[0], setting[1], kd=setting[2], dither=setting[3])
            tried[setting] = r
            log.append(r)
            print(f"  {name:16s} tune     {ft.fmt(r)}", flush=True)
            if r["aborted"] and name not in ft.SPECIAL and r["ff_share"] >= 0.7:
                break
        best = min(tried, key=lambda k: ft.score(tried[k]))

        def validate_all():
            """Every condition with `best`; returns {condition: run}."""
            runs = {}
            for speed, centre in conditions:
                v = run_at(ft, s, best, speed, centre, secs=8.0, hold=2.0)
                runs[(speed, centre)] = v
                log.append(dict(v, validation=True))
                print(f"  {name:16s} VALIDATE {speed:4.0f} deg/s @ {centre:5.1f} {shares(best)}: {ft.fmt(v)}", flush=True)
            return runs

        def report(final, verdict):
            print(f"  {name:16s} worst: |err| {final['err']:.1f} max {final['err_max']:.1f} vel {final['vel_p95']:.0f}"
                  f"{' ABORT' if final['aborted'] else ''}  -> {verdict}", flush=True)

        latest = validate_all()
        final = final_of(latest, best)
        rounds_left = ROUNDS
        report(final, "PASS" if fully_passed(ft, final, conditions) else "refine" if rounds_left else "FAIL")
        while not fully_passed(ft, final, conditions) and rounds_left > 0:
            # refine at the condition that failed worst: the current best there against small nudges
            bad = worst_run(ft, latest, best)
            condition = (bad["speed"], bad["centre"])
            before = excess(ft, bad)
            here = {best: bad}
            for c in candidates_around(best, here):
                r = run_at(ft, s, c, *condition)
                here[c] = r
                log.append(r)
                print(f"  {name:16s} refine   {condition[0]:4.0f} deg/s @ {condition[1]:5.1f} {shares(c)}: {ft.fmt(r)}",
                      flush=True)
            rounds_left -= 1
            new = min(here, key=lambda k: ft.score(here[k]))
            if new == best:
                print(f"  {name:16s} -> FAIL: no setting one step away was better at "
                      f"{condition[0]:g} deg/s @ {condition[1]:g}; rounds ended early", flush=True)
                break
            best = new
            # only that condition again; every condition once it passes there
            check = run_at(ft, s, best, *condition, secs=8.0, hold=2.0)
            log.append(dict(check, validation=True, recheck=True))
            latest[condition] = check
            print(f"  {name:16s} RE-CHECK {condition[0]:4.0f} deg/s @ {condition[1]:5.1f} {shares(best)}: {ft.fmt(check)}",
                  flush=True)
            if ft.passed(check):
                latest = validate_all()
                final = final_of(latest, best)
                report(final, "PASS" if fully_passed(ft, final, conditions) else "refine" if rounds_left else "FAIL")
                continue
            final = final_of(latest, best)
            stop, why = give_up(before, excess(ft, check), rounds_left)
            if stop or rounds_left == 0:
                report(final, "FAIL: " + (why if stop else "no rounds left") + ("; rounds ended early" if stop else ""))
                break
            report(final, "refine")
    finally:
        s.close()
    ft.home()
    result = dict(joint=name, err_share=best[0], ff_share=best[1], kd=best[2], dither=best[3], validation=final,
                  passed=fully_passed(ft, final, conditions), validated_conditions=len(final["conditions"]),
                  conditions=len(conditions), runs=log, fine=True)
    data = json.loads(ft.OUT.read_text()) if ft.OUT.exists() else {}
    data[name] = result
    ft.OUT.write_text(json.dumps(data, indent=1) + "\n")
    sys.stdout.flush()
    return result
