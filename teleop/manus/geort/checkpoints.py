"""Which trained GeoRT run drives a hand: one place for the sender (--solver geort) and the GUI.

The runs the teleop uses live in one folder per hand: retargetor/GeoRT_checkpoint/left_hand/<run>/ and
right_hand/<run>/ (config.json + last.pth; GeoRT's trainer writes them to retargetor/GeoRT/checkpoint/, copy a run into
its hand's folder to use it). A hand only ever reads its own folder; a run whose config.json says it was trained for the
other hand is listed as such and never picked. Per (source, side) one run can be pinned; otherwise the newest run of
that hand is taken. The pins are retargetor/GeoRT_checkpoint/selection.json,
which the GUI's Dev Mode page edits and saves (or edit it by hand). Standard library only (the GUI imports it).
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]             # agi-os (this file: teleop/manus/geort/)
GEORT_CHECKPOINTS = ROOT / 'retargetor' / 'GeoRT_checkpoint'
SELECTION_FILE = GEORT_CHECKPOINTS / 'selection.json'
CAMERA_SOURCES = ('camera', 'wilor', 'wilor_mini')      # webcam trackers share the "camera" pin
SOURCES = ('glove', 'camera')                           # the pin groups: the MANUS glove, every webcam tracker
SIDES = ('Left', 'Right')
HAND_FOLDERS = {'Left': 'left_hand', 'Right': 'right_hand'}
# The pins used when selection.json is missing (what was pinned before the file existed).
BUILTIN_PINS = {
    ('glove', 'Right'): 'aac_d21_right_2026-11-04_18-36-38_d21_right_4_w40_p5_d2',
    ('camera', 'Right'): 'aac_d21_right_2026-11-05_10-33-15_d21_right_5_w40_p5_d2',
}
RUN_NAME = re.compile(r'aac_d21_(left|right)_(\d{4}-\d\d-\d\d)_(\d\d)-(\d\d)-(\d\d)_?(.*)$')


def load_selection(path=SELECTION_FILE):
    """{(source, side): run name} from selection.json ({"glove": {"Left": null, "Right": "<run>"}, "camera": ...}; null
    or a missing entry = the newest run), or BUILTIN_PINS when the file does not exist."""
    path = Path(path)
    if not path.exists():
        return dict(BUILTIN_PINS)
    data = json.loads(path.read_text())
    pins = {}
    for source in SOURCES:
        for side in SIDES:
            run = (data.get(source) or {}).get(side)
            if run:
                pins[(source, side)] = str(run)
    return pins


def save_selection(pins, path=SELECTION_FILE):
    """Write the pins ({(source, side): run name or None}) as the defaults; returns the file."""
    path = Path(path)
    record = {'_about': 'The GeoRT run each hand uses, per tracking source ("glove": the MANUS glove, "camera": every '
                        'webcam tracker): a run folder name in left_hand/ or right_hand/; null = the newest run there. Edited by the GUI '
                        '(Dev Mode, GeoRT models, Save) or by hand; read by teleop/manus/geort/checkpoints.py.'}
    for source in SOURCES:
        record[source] = {side: pins.get((source, side)) for side in SIDES}
    path.write_text(json.dumps(record, indent=1) + '\n')
    return path


# The pins in effect: the saved selection. One dict object for the whole process (the GUI and its tests update it in
# place: the tracking service applies the GUI's current choice with set_pins).
CHECKPOINTS = load_selection()


def set_pins(pins):
    """Replace the pins in effect (this process only) with {(source, side): run name or None}."""
    CHECKPOINTS.clear()
    CHECKPOINTS.update({key: run for key, run in pins.items() if run})


def complete(run):
    return (run / 'config.json').exists() and (run / 'last.pth').exists()


def hand_folder(side, root=None):
    """retargetor/GeoRT_checkpoint/left_hand or right_hand: the only folder this hand's runs are read from."""
    return Path(root or GEORT_CHECKPOINTS) / HAND_FOLDERS[side.capitalize()]


def _trained_at(run):
    """When the run was trained: the date and time in its name (aac_d21_<side>_<date>_<time>_...), else the time of
    its last.pth."""
    match = RUN_NAME.match(run.name)
    if match:
        return f'{match[2]} {match[3]}:{match[4]}:{match[5]}'
    import time
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime((run / 'last.pth').stat().st_mtime))


def list_runs(side, root=None):
    """The complete runs in this hand's folder, oldest first: [{name, label, training, wrong_hand}]. label is
    "<date> <time>  <tag>"; training the settings saved with the run (config.json "training") or {}; wrong_hand is
    True for a run whose config.json names the other hand (listed, never picked as newest, refused when loaded)."""
    runs = []
    folder = hand_folder(side, root)
    for run in sorted(folder.iterdir()) if folder.is_dir() else []:
        if not run.is_dir() or not complete(run):
            continue
        try:
            config = json.loads((run / 'config.json').read_text())
        except (OSError, ValueError):
            config = {}
        match = RUN_NAME.match(run.name)
        tag = (match[6] if match else run.name)
        when = _trained_at(run)
        runs.append(dict(name=run.name, label=f'{when[:16]}' + (f'  {tag}' if tag else ''), when=when,
                         training=config.get('training') or {},
                         wrong_hand=config.get('name') not in (None, f'aac_d21_{side.lower()}')))
    return sorted(runs, key=lambda r: (r['when'], r['name']))


def newest_run(side, root=None):
    runs = [r for r in list_runs(side, root) if not r['wrong_hand']]
    return runs[-1]['name'] if runs else None


def checkpoint_for(side, source='glove'):
    """The GeoRT run for this side and tracking source ("glove"/"manus", or a webcam tracker), from this hand's folder
    (retargetor/GeoRT_checkpoint/left_hand or right_hand): the pinned one (CHECKPOINTS), otherwise the newest."""
    folder = hand_folder(side)
    pinned = CHECKPOINTS.get(('camera' if source in CAMERA_SOURCES else 'glove', side.capitalize()))
    if pinned is not None:
        run = folder / pinned
        if not complete(run):
            raise FileNotFoundError(f'pinned GeoRT checkpoint for the {side.lower()} hand ({source}) not found: {run} '
                                    f'(needs config.json + last.pth)')
        return run
    newest = newest_run(side)
    if newest is None:
        raise FileNotFoundError(f'no GeoRT checkpoint for the {side.lower()} hand in {folder} '
                                f'(a run folder there with config.json + last.pth)')
    return folder / newest
