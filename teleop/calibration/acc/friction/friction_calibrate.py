#!/usr/bin/env python3
"""Friction calibration for torque control, one hand, all three steps in one run (CAN FD, v2 firmware).

    1. breakaway  per joint and direction, at several angles: the current at which the joint starts to move
                  (retargetor/dex-retargetor-cpp/python/friction_breakaway.py)
    2. tune       per joint: the friction-compensation shares (and kd / dither) with the least stick-slip and
                  error, validated and refined (friction_tune.py)
    3. apply      the results into the hand's CAN FD torque config, hand_config_aac_<l|r>_torque_canfd.json
                  (friction_apply.py); review with git diff

The data files are the friction calibration's own (teleop/calibration/acc/friction/aac_*_<side>*.json): each joint
measured here replaces that joint's earlier result, the other joints keep theirs. Steps 1 and 2 move the hand
(position homing, then torque mode, one joint at a time): nothing else may use the hand (stop its motor in the GUI).
Ctrl-C (the GUI's Stop) zeroes the torque, leaves the hand in position mode where it stands, and skips step 3.
The GUI's "Calibrate friction" button (the hand's "..." panel) runs this script with --config / --hand_serial.

    hand/.hand/bin/python teleop/calibration/acc/friction/friction_calibrate.py --side right             # all 16 joints
    hand/.hand/bin/python teleop/calibration/acc/friction/friction_calibrate.py --side right index_pip middle_mcp
    ... --skip-breakaway        # re-tune with the existing breakaway measurements
    ... --no-apply              # measure and tune only; leave the config as it is
    ... --fine                  # the finer, much slower calibration (friction_fine.py): 10 angles, 2 mA steps,
                                #   3 readings each; validation at 3 speeds x 2 positions, pass at <= 1.0 deg
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
    args = parser.parse_args(argv)

    side = args.side or (side_of(args.config, args.hand_serial) if args.config else "left")
    if args.config and side_of(args.config, args.hand_serial) != side:
        raise SystemExit(f"--side {side} does not match the hand in {args.config}")
    sys.argv = [sys.argv[0], "--side", side]           # friction_side reads the side from here on import
    s, role = use_canfd(args.config, args.hand_serial)
    import friction_breakaway                          # both read the CAN FD SIDE now
    import friction_tune
    tune = friction_tune.tune_joint
    if args.fine:
        import friction_fine
        friction_fine.fine_breakaway(friction_breakaway)
        tune = lambda name: friction_fine.fine_tune_joint(friction_tune, name)   # noqa: E731

    names = list(friction_breakaway.JOINT_NAMES)
    joints = args.joints or names
    unknown = [j for j in joints if j not in names]
    if unknown:
        raise SystemExit(f"unknown joint(s) {', '.join(unknown)}; joints are: {' '.join(names)}")
    print(f"[friction] {s.side} hand ({s.role}) over CAN FD, config {s.config.relative_to(REPO)}, hand {s.serial}, "
          f"adapter {role['serial_device'].rsplit('/', 1)[-1]}", flush=True)
    print(f"[friction] {'FINE calibration, ' if args.fine else ''}{len(joints)} joint(s): {' '.join(joints)}. "
          "Keep clear of the fingers; Ctrl-C stops.", flush=True)
    start = time.monotonic()
    try:
        if not args.skip_breakaway:
            for k, name in enumerate(joints, 1):
                print(f"\n[friction] step 1/3 breakaway: {name} ({k}/{len(joints)}, {minutes(time.monotonic() - start)})",
                      flush=True)
                friction_breakaway.main([name])
        results = {}
        for k, name in enumerate(joints, 1):
            print(f"\n[friction] step 2/3 tune: {name} ({k}/{len(joints)}, {minutes(time.monotonic() - start)})",
                  flush=True)
            results[name] = tune(name)
    except KeyboardInterrupt:
        print(f"\n[friction] stopped after {minutes(time.monotonic() - start)}: torque zeroed, hand in position mode. "
              "Joints finished so far are saved; the config was NOT changed.", flush=True)
        return 130
    failed = [n for n, r in results.items() if not r.get("passed")]
    print(f"\n[friction] measured and tuned in {minutes(time.monotonic() - start)}; validation passed for "
          f"{len(results) - len(failed)}/{len(results)} joint(s)" + (f" (best found for {' '.join(failed)})" if failed else ""),
          flush=True)
    if args.no_apply:
        print("[friction] --no-apply: config unchanged. Apply later with: friction_canfd.py apply"
              + ("" if side == "left" else " --side right"), flush=True)
        return 0
    print("\n[friction] step 3/3 apply", flush=True)
    runpy.run_path(str(SCRIPTS / "friction_apply.py"), run_name="__main__")
    print(f"[friction] done. Review: git diff {s.config.relative_to(REPO)}. Restart the hand's motor to use it.",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
