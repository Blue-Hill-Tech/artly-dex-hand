#!/usr/bin/env python3
"""Run the AAC friction calibration (retargetor/dex-retargetor-cpp/python/friction_*.py) against the CAN FD torque config.

The three friction scripts are wired to the RS-485 torque configs (friction_side.py picks
hand_config_aac_<l|r>_torque_rs485.json). This launcher runs them unchanged, but with the side's CAN FD config
(hand_config_aac_<l|r>_torque_canfd.json) in its place: the hand is opened over the CAN FD adapter (the v2 firmware is
detected on open), and step 3 writes the tuning into the CAN FD config. The calibration data files are the same ones
(teleop/calibration/acc/friction/aac_*_<side>*.json).

    hand/.hand/bin/python teleop/calibration/acc/friction/friction_canfd.py show [--side right]   # what it would use; no hardware
    hand/.hand/bin/python teleop/calibration/acc/friction/friction_canfd.py breakaway [--side right] [joint ...]   # step 1
    hand/.hand/bin/python teleop/calibration/acc/friction/friction_canfd.py tune      [--side right] [joint ...]   # step 2
    hand/.hand/bin/python teleop/calibration/acc/friction/friction_canfd.py apply     [--side right]               # step 3

All three steps in one go: friction_calibrate.py (also the GUI's "Calibrate friction" button).

Steps 1 and 2 move the hand (position homing, then torque mode, one joint at a time): nothing else may use the hand
(close the GUI hand page / hand controller). Step 3 only edits the CAN FD config; review it with git diff.
"""
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]
SCRIPTS = REPO / "retargetor" / "dex-retargetor-cpp" / "python"
STEPS = {"breakaway": "friction_breakaway.py", "tune": "friction_tune.py", "apply": "friction_apply.py"}


def canfd_config(side):
    return REPO / "teleop" / "configure" / "acc" / "torque_control" / f"hand_config_aac_{side[0]}_torque_canfd.json"


def use_canfd(config=None, serial=None):
    """Import friction_side (it takes --side out of sys.argv) and point its SIDE at a CAN FD torque config: `config`
    (default: the side's hand_config_aac_<l|r>_torque_canfd.json) and its hand `serial` (default: its first hand).
    The step scripts read SIDE when they are imported, so call this before importing them. Returns (SIDE, role)."""
    for path in (str(REPO), str(SCRIPTS)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import friction_side                              # builds SIDE from the RS-485 config; replaced below

    side = friction_side.SIDE
    config = Path(config) if config else canfd_config(side.side)
    config = config if config.is_absolute() else REPO / config
    roles = json.loads(config.read_text())["hand_roles"]
    serial = serial or next(iter(roles))
    if serial not in roles:
        raise SystemExit(f"{config.name} has no hand {serial!r} (it has {', '.join(roles)})")
    role = roles[serial]
    if role.get("role") != side.role:
        raise SystemExit(f"{config.name} is for {role.get('role')}, not {side.role}")
    if role.get("bus", "canfd") != "canfd":
        raise SystemExit(f"{config.name} does not use the CAN FD bus (bus = {role.get('bus')!r})")
    friction_side.SIDE = SimpleNamespace(**{**vars(side), "config": config, "serial": serial})
    return friction_side.SIDE, role


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in (*STEPS, "show"):
        raise SystemExit(__doc__)
    step = sys.argv[1]
    script = SCRIPTS / STEPS.get(step, "friction_apply.py")
    sys.argv = [str(script), *sys.argv[2:]]          # what the step script would have seen (friction_side pops --side)
    s, role = use_canfd()
    config = s.config
    print(f"[friction_canfd] {step}: {s.side} hand ({s.role}) over CAN FD, config {config.relative_to(REPO)}, "
          f"adapter {role['serial_device'].rsplit('/', 1)[-1]}")
    if step == "show":
        for name in ("home", "fine", "coarse", "tuning"):
            path = getattr(s, name)
            print(f"  {name:7s} {path.relative_to(REPO)}{'' if path.exists() else '  (missing)'}")
        print(f"  adapter present: {Path(role['serial_device']).exists()}")
        return
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
