"""Which hand the friction scripts work on: `--side left|right` (default left), taken out of sys.argv.

Each side has its own torque config, home pose, breakaway tables and tuning result in teleop/. The motor-pair
polarity follows the config's hand role (teleop.hand_control.hand_controller_aac.pair_polarity).
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]   # the agi-os repo (retargetor/dex-retargetor-cpp/python/..)
TELEOP = ROOT / 'teleop'


def pop_side(argv=None):
    argv = sys.argv if argv is None else argv
    side = 'left'
    for flag in ('--side', '-s'):
        if flag in argv:
            i = argv.index(flag)
            side = argv[i + 1]
            del argv[i:i + 2]
    for a in list(argv):
        if a.startswith('--side='):
            side = a.split('=', 1)[1]
            argv.remove(a)
    if side not in ('left', 'right'):
        raise SystemExit(f'--side must be left or right, not {side!r}')
    return side


def side_files(side):
    letter = side[0]
    config = TELEOP / 'configure' / 'acc' / 'torque_control' / f'hand_config_aac_{letter}_torque_rs485.json'
    friction = TELEOP / 'calibration' / 'acc' / 'friction'      # the calibration data folder
    serial, entry = next(iter(json.loads(config.read_text())['hand_roles'].items()))
    return SimpleNamespace(
        side=side, role=entry['role'], serial=serial, config=config,
        home=friction / f'aac_friction_home_{side}.json',
        fine=friction / f'aac_breakaway_{side}_fine.json',
        coarse=friction / f'aac_breakaway_{side}.json',
        tuning=friction / f'aac_friction_tuning_{side}.json')


SIDE = side_files(pop_side())
