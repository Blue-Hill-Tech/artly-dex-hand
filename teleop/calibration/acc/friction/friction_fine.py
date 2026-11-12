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
        final = None
        for rnd in range(ROUNDS + 1):
            runs = []
            for speed, centre in conditions:
                v = run_at(ft, s, best, speed, centre, secs=8.0, hold=2.0)
                runs.append(v)
                log.append(dict(v, validation=True))
                print(f"  {name:16s} VALIDATE {speed:4.0f} deg/s @ {centre:5.1f} {shares(best)}: {ft.fmt(v)}", flush=True)
            final = worst_of(runs)
            ok = ft.passed(final)
            print(f"  {name:16s} worst: |err| {final['err']:.1f} max {final['err_max']:.1f} vel {final['vel_p95']:.0f}"
                  f"{' ABORT' if final['aborted'] else ''}  -> {'PASS' if ok else 'refine' if rnd < ROUNDS else 'FAIL'}",
                  flush=True)
            if ok or rnd == ROUNDS:
                break
            # refine at the condition that failed worst: the current best there against small nudges
            bad = max(runs, key=ft.score)
            speed, centre = bad["speed"], bad["centre"]
            here = {best: bad}
            es0, ff0, kd, dither = best
            nudges = ((0, REFINE), (0, -REFINE), (-REFINE, REFINE), (REFINE, 0))
            candidates = [(round(es0 + de, 2), round(ff0 + df, 2), kd, dither) for de, df in nudges]
            candidates = [c for c in candidates if 0 <= c[0] <= 0.6 and 0 <= c[1] <= 1.0 and c not in here][:3]
            for c in candidates:
                r = run_at(ft, s, c, speed, centre)
                here[c] = r
                log.append(r)
                print(f"  {name:16s} refine   {speed:4.0f} deg/s @ {centre:5.1f} {shares(c)}: {ft.fmt(r)}", flush=True)
            best = min(here, key=lambda k: ft.score(here[k]))
    finally:
        s.close()
    ft.home()
    result = dict(joint=name, err_share=best[0], ff_share=best[1], kd=best[2], dither=best[3], validation=final,
                  passed=ft.passed(final), runs=log, fine=True)
    data = json.loads(ft.OUT.read_text()) if ft.OUT.exists() else {}
    data[name] = result
    ft.OUT.write_text(json.dumps(data, indent=1) + "\n")
    sys.stdout.flush()
    return result
