#!/usr/bin/env python3
"""Friction calibration for torque control, one hand, all three steps in one run (CAN FD, v2 firmware).

    1. breakaway  per joint and direction, at several angles: the current at which the joint starts to move
                  (retargetor/dex-retargetor-cpp/python/friction_breakaway.py; each current step is held
                  BREAKAWAY_HOLD_S = 0.1 s here, not the script's own 0.5 s)
    2. tune       per joint: the friction-compensation shares (and kd / dither) with the least stick-slip and
                  error, validated and refined (friction_tune.py)
    3. apply      the results into the hand's CAN FD torque config, hand_config_aac_<l|r>_torque_canfd.json
                  (friction_apply.py); review with git diff

The data files are the friction calibration's own (teleop/calibration/acc/friction/aac_*_<side>*.json): each joint
measured here replaces that joint's earlier result, the other joints keep theirs. Steps 1 and 2 move the hand
(position homing, then torque mode; the joints of a group in parallel, friction_group.py): nothing else may use the hand (stop its motor in the GUI).
Ctrl-C (the GUI's Stop) zeroes the torque, leaves the hand in position mode where it stands, and skips step 3; the
progress is saved as a checkpoint (friction_checkpoint_<side>.json): --resume continues with the joint that was
interrupted (the joints already finished are not redone).
The GUI's "Calibrate friction" button (the hand's "..." panel) runs this script with --config / --hand_serial.

    hand/.hand/bin/python teleop/calibration/acc/friction/friction_calibrate.py --side right             # all 16 joints
    hand/.hand/bin/python teleop/calibration/acc/friction/friction_calibrate.py --side right index_pip middle_mcp
    ... --skip-breakaway        # re-tune with the existing breakaway measurements
    ... --no-apply              # measure and tune only; leave the config as it is
    ... --resume                # continue a stopped run (Ctrl-C / the GUI's Stop) from its checkpoint
    ... --restart               # discard a stopped run and start a new one
    ... --checkpoint            # show what a stopped run has done
    ... --fine                  # the finer, much slower calibration (friction_fine.py): 10 angles, 2 mA steps,
                                #   3 readings each; validation at 3 speeds x 2 positions, pass at <= 1.0 deg
    ... --sequential            # one joint at a time (default: in parallel groups, friction_group.py: the thumb
                                #   joints alone, the 4 MCPs, the 4 PIPs, index + ring and middle + pinky abduction)
    ... --config teleop/configure/acc/torque_control/hand_config_aac_r_torque_canfd.json --hand_serial AAC-002
"""
import argparse
import json
import runpy
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from friction_canfd import REPO, SCRIPTS, use_canfd  # noqa: E402

SIDES = {"LeftHand": "left", "RightHand": "right"}


BREAKAWAY_HOLD_S = 0.1   # breakaway: seconds each current step is held while watching for motion (normal and fine)


def side_of(config, serial):
    """The side (left / right) of `serial`'s hand in `config` (its role)."""
    path = Path(config) if Path(config).is_absolute() else REPO / config
    roles = json.loads(path.read_text())["hand_roles"]
    role = roles[serial or next(iter(roles))].get("role")
    if role not in SIDES:
        raise SystemExit(f"{path.name}: hand role {role!r} is neither LeftHand nor RightHand")
    return SIDES[role]


def minutes(seconds):
    return f"{seconds / 60:.1f} min"


# --- checkpoint: a stopped (or failed) run can be continued -----------------------------------------------------------
# friction_checkpoint_<side>.json next to this script: the run's settings and which joints have finished each step.
# Written when the run starts and after every finished joint, so a stop (Ctrl-C / the GUI's Stop), an unplugged
# adapter or a crash loses at most the joint in progress (its earlier result stays in the data files until it is
# measured again). Deleted when the run completes. --resume continues it; --restart discards it.
CHECKPOINT_VERSION = 1


def checkpoint_path(side):
    return HERE / f"friction_checkpoint_{side}.json"


def load_checkpoint(side):
    """The saved checkpoint of `side`, or None."""
    path = checkpoint_path(side)
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {path.name}: {exc} (delete it, or start over with --restart)")
    if data.get("version") != CHECKPOINT_VERSION:
        raise SystemExit(f"{path.name} is from another version of this script; start over with --restart")
    return data


def describe(ck):
    """One line: what a checkpoint has done and what is left."""
    joints, done = ck["joints"], ck["done"]
    steps = [] if ck["skip_breakaway"] else [f"breakaway {len(done['breakaway'])}/{len(joints)}"]
    steps.append(f"tune {len(done['tune'])}/{len(joints)}")
    where = f", stopped during {ck['stopped_during']}" if ck.get("stopped_during") else ""
    return (f"{'FINE' if ck['fine'] else 'normal'}{', parallel' if ck.get('grouped') else ''} calibration of "
            f"{len(joints)} joint(s): {', '.join(steps)} done"
            f"{where}, {minutes(ck.get('elapsed_s', 0.0))} so far")


def save_checkpoint(side, ck):
    ck["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    path = checkpoint_path(side)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(ck, indent=1) + "\n")
    tmp.replace(path)                                     # atomic: a kill mid-write leaves the previous checkpoint


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("joints", nargs="*", help="joints to calibrate (default: all 16)")
    parser.add_argument("--side", choices=("left", "right"), help="the hand (default: from --config, else left)")
    parser.add_argument("--config", help="CAN FD torque config (default: the side's hand_config_aac_<l|r>_torque_canfd.json)")
    parser.add_argument("--hand_serial", help="the hand in --config (default: its first)")
    parser.add_argument("--skip-breakaway", action="store_true", help="step 2 and 3 only, with the saved breakaway")
    parser.add_argument("--no-apply", action="store_true", help="steps 1 and 2 only: do not write the config")
    parser.add_argument("--fine", action="store_true", help="the finer, much slower calibration (friction_fine.py)")
    parser.add_argument("--sequential", action="store_true",
                        help="one joint at a time (default: the joints of a group in parallel, friction_group.py)")
    again = parser.add_mutually_exclusive_group()
    again.add_argument("--resume", action="store_true",
                       help="continue the stopped run from its checkpoint (its joints and settings)")
    again.add_argument("--restart", action="store_true", help="discard a saved checkpoint and start a new run")
    parser.add_argument("--checkpoint", action="store_true", help="only show the saved checkpoint, then exit")
    args = parser.parse_args(argv)

    side = args.side or (side_of(args.config, args.hand_serial) if args.config else "left")
    if args.config and side_of(args.config, args.hand_serial) != side:
        raise SystemExit(f"--side {side} does not match the hand in {args.config}")
    ck = load_checkpoint(side)
    if args.checkpoint:
        print(f"[friction] {side} hand: " + (describe(ck) + f" (saved {ck['updated_at']}; continue with --resume)"
                                             if ck else "no checkpoint"))
        return 0
    if args.resume and ck is None:
        raise SystemExit(f"[friction] no checkpoint for the {side} hand to continue ({checkpoint_path(side).name})")
    if ck is not None and not (args.resume or args.restart):
        raise SystemExit(f"[friction] the {side} hand has a stopped calibration: {describe(ck)}.\n"
                         "Continue it with --resume, or discard it and start a new run with --restart.")
    if args.resume:
        if args.config and Path(args.config) != Path(ck["config"]) or args.hand_serial and args.hand_serial != ck["serial"]:
            raise SystemExit(f"[friction] the checkpoint is for {ck['config']} / {ck['serial']}, not this hand")
        if args.joints or args.fine != ck["fine"] or args.skip_breakaway or args.no_apply \
                or args.sequential != (not ck.get("grouped", False)):
            print("[friction] --resume: continuing with the checkpoint's own joints and settings", flush=True)
        config, serial = ck["config"], ck["serial"]
    else:
        config, serial = args.config, args.hand_serial

    sys.argv = [sys.argv[0], "--side", side]           # friction_side reads the side from here on import
    s, role = use_canfd(config, serial)
    import friction_breakaway                          # both read the CAN FD SIDE now
    import friction_tune
    friction_breakaway.HOLD = BREAKAWAY_HOLD_S         # each current step held this long (the script's own: 0.5 s)
    names = list(friction_breakaway.JOINT_NAMES)
    if ck is None or args.restart:
        joints = args.joints or names
        unknown = [j for j in joints if j not in names]
        if unknown:
            raise SystemExit(f"unknown joint(s) {', '.join(unknown)}; joints are: {' '.join(names)}")
        ck = dict(version=CHECKPOINT_VERSION, side=side, config=str(s.config.relative_to(REPO)), serial=s.serial,
                  joints=joints, fine=args.fine, grouped=not args.sequential, skip_breakaway=args.skip_breakaway,
                  no_apply=args.no_apply,
                  started_at=time.strftime("%Y-%m-%d %H:%M:%S"), elapsed_s=0.0,
                  done={"breakaway": [], "tune": []}, passed={})
    joints, fine = ck["joints"], ck["fine"]
    grouped = ck.get("grouped", False)                 # (a checkpoint from before the groups: one joint at a time)
    import friction_group
    units = friction_group.groups_for(joints) if grouped else [(n,) for n in joints]
    tune = friction_tune.tune_joint
    if fine:
        import friction_fine
        friction_fine.fine_breakaway(friction_breakaway)
        tune = lambda name: friction_fine.fine_tune_joint(friction_tune, name)   # noqa: E731

    print(f"[friction] {s.side} hand ({s.role}) over CAN FD, config {s.config.relative_to(REPO)}, hand {s.serial}, "
          f"adapter {role['serial_device'].rsplit('/', 1)[-1]}", flush=True)
    print(f"[friction] breakaway ramp: {friction_breakaway.STEP:g} mA every {friction_breakaway.HOLD:g} s "
          f"({friction_breakaway.STEP / friction_breakaway.HOLD:g} mA/s), from {friction_breakaway.START:g} mA, "
          f"up to {friction_breakaway.MAX:g} mA", flush=True)
    if args.resume:
        print(f"[friction] CONTINUING: {describe(ck)}", flush=True)
    if grouped:
        print("[friction] in parallel: " + " | ".join(" + ".join(g) for g in units), flush=True)
    print(f"[friction] {'FINE calibration, ' if fine else ''}{len(joints)} joint(s): {' '.join(joints)}. "
          "Keep clear of the fingers; Ctrl-C stops (progress is kept: continue with --resume).", flush=True)
    start, before = time.monotonic(), float(ck.get("elapsed_s", 0.0))
    ck.pop("stopped_during", None)
    current = None

    def finished(step, name, passed=None):
        ck["done"][step].append(name)
        if passed is not None:
            ck["passed"][name] = bool(passed)
        ck["elapsed_s"] = before + time.monotonic() - start
        save_checkpoint(side, ck)

    def stamp():
        return minutes(before + time.monotonic() - start)

    save_checkpoint(side, ck)
    try:
        # a group of several joints runs them in parallel (friction_group.py); a single joint the step scripts' way
        if not ck["skip_breakaway"]:
            for k, group in enumerate(units, 1):
                todo = [n for n in group if n not in ck["done"]["breakaway"]]
                if not todo:
                    continue
                current = f"breakaway {' + '.join(todo)}"
                print(f"\n[friction] step 1/3 breakaway: {' + '.join(todo)} ({k}/{len(units)}, {stamp()})", flush=True)
                if len(todo) > 1:
                    friction_group.breakaway_group(friction_breakaway, todo, fine=fine)
                else:
                    friction_breakaway.main(todo)
                for name in todo:
                    finished("breakaway", name)
        for k, group in enumerate(units, 1):
            todo = [n for n in group if n not in ck["done"]["tune"]]
            if not todo:
                continue
            current = f"tune {' + '.join(todo)}"
            print(f"\n[friction] step 2/3 tune: {' + '.join(todo)} ({k}/{len(units)}, {stamp()})", flush=True)
            if len(todo) > 1:
                results = friction_group.tune_group(friction_tune, todo, fine=fine)
                for name in todo:
                    finished("tune", name, results[name].get("passed"))
            else:
                finished("tune", todo[0], tune(todo[0]).get("passed"))
        current = None
    except BaseException as exc:                       # Ctrl-C / Stop, or an error: keep the progress
        ck["elapsed_s"] = before + time.monotonic() - start
        ck["stopped_during"] = current
        save_checkpoint(side, ck)
        if isinstance(exc, KeyboardInterrupt):
            print(f"\n[friction] stopped during {current}, {stamp()} in all: torque zeroed, hand in position mode. "
                  f"The config was NOT changed. Progress saved ({describe(ck)}): continue with --resume "
                  "(or the GUI's Calibrate friction > Continue).", flush=True)
            return 130
        print(f"\n[friction] failed during {current}: progress saved, continue with --resume once fixed", flush=True)
        raise
    failed = [n for n in joints if not ck["passed"].get(n)]
    print(f"\n[friction] measured and tuned in {stamp()}; validation passed for "
          f"{len(joints) - len(failed)}/{len(joints)} joint(s)" + (f" (best found for {' '.join(failed)})" if failed else ""),
          flush=True)
    if ck["no_apply"]:
        checkpoint_path(side).unlink(missing_ok=True)
        print("[friction] --no-apply: config unchanged. Apply later with: friction_canfd.py apply"
              + ("" if side == "left" else " --side right"), flush=True)
        return 0
    print("\n[friction] step 3/3 apply", flush=True)
    runpy.run_path(str(SCRIPTS / "friction_apply.py"), run_name="__main__")
    checkpoint_path(side).unlink(missing_ok=True)
    print(f"[friction] done. Review: git diff {s.config.relative_to(REPO)}. Restart the hand's motor to use it.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
