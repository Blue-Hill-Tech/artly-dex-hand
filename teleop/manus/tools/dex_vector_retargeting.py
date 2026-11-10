"""Installed dex-retargeting full-hand vector baseline; no robot transport."""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np
from scipy.spatial.transform import Rotation
from dex_retargeting.retargeting_config import RetargetingConfig
from dex_retargeting.optimizer import VectorOptimizer
from dex_retargeting.robot_wrapper import RobotWrapper
from dex_retargeting.seq_retarget import SeqRetargeting
from dex_retargeting.optimizer_utils import LPFilter
from dex_retargeting.kinematics_adaptor import MimicJointKinematicAdaptor

from hand.aac.model import HandModel, MODEL_JOINTS
from hand.backends.aac import ANGLE_LIMITS
from teleop.manus.tools.skeleton_source import STALE_S, palm_positions, palm_rotation_of


SCHEMA = 'manus-aac-dex-vector-v2'
FINGERS = ('thumb', 'index', 'middle', 'ring', 'pinky')
CAPTURE_S = 1.0
MIN_CAPTURE_FRAMES = 30
LENGTH_EPS_M = 1e-6
BASIS_MIN_SINE = 1e-3
VECTOR_TABLE = [dict(human_chain=f'Finger{f.title()}',
    human_origin='Metacarpal' if f == 'thumb' else 'Proximal',
    human_task='Tip' if tip else 'Proximal' if f == 'thumb' else 'Intermediate',
    robot_origin='thumb_pitch' if f == 'thumb' else f'{f}_mcp',
    robot_task=f'{f}_tip' if tip else 'thumb_mcp' if f == 'thumb' else f'{f}_pip')
    for tip in (True, False) for f in FINGERS]
ORIGIN_LINKS = [row['robot_origin'] for row in VECTOR_TABLE]
TASK_LINKS = [row['robot_task'] for row in VECTOR_TABLE]
# Hand, ten tasks, five origins. Metadata IDs are never array offsets.
HUMAN_LANDMARKS = [('Hand', None)] + [(r['human_chain'], r['human_task']) for r in VECTOR_TABLE] + list(
    dict.fromkeys((r['human_chain'], r['human_origin']) for r in VECTOR_TABLE))
HUMAN_INDICES = np.array([[HUMAN_LANDMARKS.index((r['human_chain'], r[key])) for r in VECTOR_TABLE]
                          for key in ('human_origin', 'human_task')])
# Where the two thumb vectors start (retarget_tuning.yaml vector.thumb_anchor). 'base': each hand's own thumb
# base (glove CMC, robot thumb_pitch), so the target is the thumb's angle and shape from its base; the two bases
# sit differently relative to the knuckles, so the target does not land on the glove thumb tip. 'index_knuckle':
# both start at the index MCP, a point the hands share, so the robot thumb tip is placed where the glove thumb
# tip is relative to the fingers. Landmarks and row labels are unchanged either way.
THUMB_ANCHORS = dict(base=None, **{f'{f}_knuckle': ((f'Finger{f.title()}', 'Proximal'), f'{f}_mcp')
                                   for f in ('index', 'middle', 'ring', 'pinky')})


PIP_TIP_FINGERS = ('index', 'middle', 'ring', 'pinky')


def vector_table_for(thumb_anchor='base', pip_tip=()):
    """(vector_table, origin_links, task_links, human_indices) for a thumb anchor name, plus one extra
    "<finger>-pip-tip" row (glove Intermediate -> Tip, robot <finger>_pip -> <finger>_tip) per finger in pip_tip,
    appended after the ten standard rows."""
    if thumb_anchor not in THUMB_ANCHORS:
        raise ValueError(f'thumb_anchor must be one of {sorted(THUMB_ANCHORS)}, got {thumb_anchor!r}')
    table = [dict(row, human_origin_chain=row['human_chain']) for row in VECTOR_TABLE]
    anchor = THUMB_ANCHORS[thumb_anchor]
    if anchor is not None:
        (chain, joint), link = anchor
        for row in table:
            if row['human_chain'] == 'FingerThumb':
                row.update(human_origin_chain=chain, human_origin=joint, robot_origin=link)
    for finger in pip_tip:
        chain = f'Finger{finger.title()}'
        table.append(dict(human_chain=chain, human_origin_chain=chain, human_origin='Intermediate', human_task='Tip',
                          robot_origin=f'{finger}_pip', robot_task=f'{finger}_tip', label=f'{finger}-pip-tip'))
    indices = np.array([[HUMAN_LANDMARKS.index((r['human_origin_chain'], r['human_origin'])) for r in table],
                        [HUMAN_LANDMARKS.index((r['human_chain'], r['human_task'])) for r in table]])
    return table, [r['robot_origin'] for r in table], [r['robot_task'] for r in table], indices


# The D21 URDF puts each finger's flexion-joint frames (<f>_mcp, <f>_pip, <f>_dip) at the side of the finger, 8.4 /
# 5.7 / 7.1 mm from the centreline that the abduction joint (<f>_pitch) and the tip sphere lie on; the thumb's
# thumb_mcp frame sits ~11 mm beside the line of its tip. The glove's joints are at the finger centres, so vectors
# measured between link origins carry a sideways offset the solver then "matches" with abduction. Sliding a point
# along its own joint axis leaves its motion unchanged, so each point is moved along the link's axis to the
# reference's position: knuckle and middle points to the abduction joint, thumb_mcp to the thumb tip.
CENTRELINE_REFERENCES = {**{f'{f}_{link}': f'{f}_pitch' for f in FINGERS[1:] for link in ('mcp', 'pip')},
                         'thumb_mcp': 'thumb_tip'}


def centreline_offsets(model, references=CENTRELINE_REFERENCES):
    """{link: offset (link frame, m)} putting the link's point on the finger centreline (see CENTRELINE_REFERENCES)."""
    poses = model.transforms([0.]*16)
    axes = {child: axis for _, _, child, _, axis in model.joints if axis is not None}
    out = {}
    for link, reference in references.items():
        if link not in poses or reference not in poses or link not in axes:
            raise ValueError(f'centreline offsets: missing {link} / {reference} in the hand model')
        rotation, position = poses[link][:3, :3], poses[link][:3, 3]
        axis = np.asarray(axes[link], dtype=float)
        axis = axis/np.linalg.norm(axis)
        along = float((poses[reference][:3, 3]-position) @ (rotation @ axis))
        out[link] = along*axis
    return out
# DexPilot layout (Handa et al. 2019; dex-retargeting DexPilotOptimizer): vectors between fingertips, with the
# thumb->finger pairs projected to contact when the human pair closes. DexPilot's wrist->tip shape vectors are
# anchored here at each finger's own knuckle (MCP -> tip), a point both hands have exactly, instead of a wrist
# that the glove and the URDF place differently. The thumb is placed by its pair vectors plus a low-weight
# base->knuckle vector (the D21 thumb has 4 joints for a 3-dof tip). Human indices refer to HUMAN_LANDMARKS:
# 1..5 = fingertips (thumb..pinky), 6 = thumb proximal, 11 = thumb metacarpal, 12..15 = finger MCPs.
DEXPILOT_WEIGHTS = dict(thumb_pairs=1.0, finger_pairs=1.0, finger_shape=15.0, thumb_proximal=1.0)
DEXPILOT_THUMB_PROXIMAL_WEIGHT = DEXPILOT_WEIGHTS['thumb_proximal']
# DexPilot solver options (dex-retargetor-cpp names) and the YAML keys (retarget_tuning.yaml) they come from.
DEXPILOT_OPTIONS = dict(huber_delta=0.03, norm_delta=0.004, project_dist=0.03, escape_dist=0.05, eta_primary=1e-4,
                        eta_secondary=3e-2, projected_weight_primary=200., projected_weight_secondary=400.)
DEXPILOT_OPTION_KEYS = dict(huber_delta='huber_delta_m', norm_delta='norm_delta', project_dist='project_dist_m',
                            escape_dist='escape_dist_m', eta_primary='eta_primary_m', eta_secondary='eta_secondary_m',
                            projected_weight_primary='projected_weight_primary',
                            projected_weight_secondary='projected_weight_secondary')
VECTOR_TUNING = dict(huber_delta_m=0.02, norm_delta=0.004, low_pass_alpha=0.1)
DEFAULT_TUNING_PATH = Path(__file__).resolve().parents[1] / 'parameters' / 'retarget_tuning.yaml'


def _dexpilot_table(weights=None):
    w = dict(DEXPILOT_WEIGHTS, **(weights or {}))
    tips = [f'{f}_tip' for f in FINGERS]
    rows = []
    for i in range(1, 5):   # thumb <- finger i: origin = finger tip, task = thumb tip
        rows.append(dict(human_chain='Pair', human_origin=1 + i, human_task=1, robot_origin=tips[i], robot_task=tips[0],
                         weight=float(w['thumb_pairs']), projection=1, gate_a=-1, gate_b=-1, label=f'thumb-{FINGERS[i]}'))
    for i in range(1, 4):
        for j in range(i + 1, 5):
            rows.append(dict(human_chain='Pair', human_origin=1 + j, human_task=1 + i, robot_origin=tips[j],
                             robot_task=tips[i], weight=float(w['finger_pairs']), projection=2, gate_a=i - 1, gate_b=j - 1,
                             label=f'{FINGERS[i]}-{FINGERS[j]}'))
    for i in range(1, 5):
        f = FINGERS[i]
        rows.append(dict(human_chain='Finger' + f.title(), human_origin=11 + i, human_task=1 + i, robot_origin=f'{f}_mcp',
                         robot_task=tips[i], weight=float(w['finger_shape']), projection=0, gate_a=-1, gate_b=-1,
                         label=f'{f}-tip'))
    rows.append(dict(human_chain='FingerThumb', human_origin=11, human_task=6, robot_origin='thumb_pitch',
                     robot_task='thumb_mcp', weight=float(w['thumb_proximal']), projection=0, gate_a=-1, gate_b=-1,
                     label='thumb-proximal'))
    return rows


DEXPILOT_TABLE = _dexpilot_table()


def _positive(section, key, value, allow_zero=False):
    try:
        value = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'retarget tuning {section}.{key} must be a number, got {value!r}')
    if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f'retarget tuning {section}.{key} must be positive, got {value!r}')
    return value


def _thumb_lock_values(prefix, raw, defaults):
    """enabled / angle_deg / weight of a thumb rotation lock block, validated, missing keys from defaults."""
    lock = dict(enabled=raw.get('enabled', defaults['enabled']), angle_deg=raw.get('angle_deg', defaults['angle_deg']),
                weight=_positive(prefix, 'weight', raw.get('weight', defaults['weight'])))
    if type(lock['enabled']) is not bool:
        raise ValueError(f'retarget tuning {prefix}.enabled must be true or false')
    try:
        lock['angle_deg'] = float(lock['angle_deg'])
    except (TypeError, ValueError):
        raise ValueError(f'retarget tuning {prefix}.angle_deg must be a number') from None
    if not np.isfinite(lock['angle_deg']) or not -35 <= lock['angle_deg'] <= 35:
        raise ValueError(f'retarget tuning {prefix}.angle_deg must be within the rotation range, -35..35')
    return lock


def load_tuning(source=None):
    """Retargeting tunables (see retarget_tuning.yaml): a path, an already-loaded dict, or None for the default file.

    Returns dict(vector=..., dexpilot=..., path=...) with every value validated and defaults filled in."""
    if isinstance(source, dict) and 'path' in source and 'vector' in source and 'dexpilot' in source:
        return source
    path = None
    if isinstance(source, dict):
        raw = source
    else:
        path = DEFAULT_TUNING_PATH if source is None else Path(source)
        if not path.exists():
            if source is not None:
                raise FileNotFoundError(f'retarget tuning not found: {path}')
            raw = {}
        else:
            import yaml
            with open(path) as handle:
                raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError('retarget tuning must be a mapping with "vector" and "dexpilot" sections')
    unknown = set(raw) - {'vector', 'dexpilot', 'glove'}
    if unknown:
        raise ValueError(f'retarget tuning: unknown sections {sorted(unknown)}')
    vector_raw, dexpilot_raw = dict(raw.get('vector') or {}), dict(raw.get('dexpilot') or {})
    vector = {k: _positive('vector', k, vector_raw.pop(k, v), allow_zero=(k == 'norm_delta')) for k, v in VECTOR_TUNING.items()}
    if not 0 < vector['low_pass_alpha'] <= 1:
        raise ValueError('retarget tuning vector.low_pass_alpha must be in (0, 1]')
    vector['thumb_anchor'] = str(vector_raw.pop('thumb_anchor', 'base'))
    pip_raw = dict(vector_raw.pop('pip_tip_rows', None) or {})
    unknown = set(pip_raw) - set(PIP_TIP_FINGERS)
    if unknown:
        raise ValueError(f'retarget tuning vector.pip_tip_rows: unknown fingers {sorted(unknown)}; expected {list(PIP_TIP_FINGERS)}')
    vector['pip_tip_rows'] = {}
    for finger in PIP_TIP_FINGERS:
        row = pip_raw.get(finger) or {}
        if not isinstance(row, dict) or set(row) - {'weight', 'scale'}:
            raise ValueError(f'retarget tuning vector.pip_tip_rows.{finger} must be a mapping with weight and scale')
        vector['pip_tip_rows'][finger] = dict(
            weight=_positive(f'vector.pip_tip_rows.{finger}', 'weight', row.get('weight', 0.0), allow_zero=True),
            scale=_positive(f'vector.pip_tip_rows.{finger}', 'scale', row.get('scale', 1.0)))
    lock_raw = dict(vector_raw.pop('thumb_rotation_lock', None) or {})
    unknown = set(lock_raw) - {'enabled', 'angle_deg', 'weight', 'left', 'right'}
    if unknown:
        raise ValueError(f'retarget tuning vector.thumb_rotation_lock: unknown keys {sorted(unknown)}')
    lock = _thumb_lock_values('vector.thumb_rotation_lock', lock_raw, dict(enabled=False, angle_deg=0.0, weight=50.0))
    for side in ('left', 'right'):   # per-hand overrides of any of the three values; the base block is the default
        side_raw = lock_raw.get(side)
        if side_raw is None:
            side_raw = {}
        if not isinstance(side_raw, dict):
            raise ValueError(f'retarget tuning vector.thumb_rotation_lock.{side} must be a mapping')
        unknown = set(side_raw) - {'enabled', 'angle_deg', 'weight'}
        if unknown:
            raise ValueError(f'retarget tuning vector.thumb_rotation_lock.{side}: unknown keys {sorted(unknown)}')
        lock[side] = _thumb_lock_values(f'vector.thumb_rotation_lock.{side}', side_raw, lock)
    vector['thumb_rotation_lock'] = lock
    soft_raw = dict(vector_raw.pop('mcp_soft_zero', None) or {})
    unknown = set(soft_raw) - set(PIP_TIP_FINGERS)
    if unknown:
        raise ValueError(f'retarget tuning vector.mcp_soft_zero: unknown fingers {sorted(unknown)}; expected {list(PIP_TIP_FINGERS)}')
    vector['mcp_soft_zero'] = {finger: _positive('vector.mcp_soft_zero', finger, soft_raw.get(finger, 0.0), allow_zero=True)
                               for finger in PIP_TIP_FINGERS}
    if vector['thumb_anchor'] not in THUMB_ANCHORS:
        raise ValueError(f"retarget tuning vector.thumb_anchor must be one of {sorted(THUMB_ANCHORS)}, "
                         f"got {vector['thumb_anchor']!r}")
    labels =[f"{r['human_chain'][6:].lower()}-{'tip' if r['robot_task'].endswith('_tip') else 'mid'}" for r in VECTOR_TABLE]
    rows = dict(vector_raw.pop('row_weights', None) or {})
    unknown = set(rows) - set(labels)
    if unknown:
        raise ValueError(f'retarget tuning vector.row_weights: unknown vectors {sorted(unknown)}; expected {labels}')
    vector['row_weights'] = [_positive('vector.row_weights', label, rows.get(label, 1.0), allow_zero=True) for label in labels]
    scales = dict(vector_raw.pop('row_scales', None) or {})
    # per hand: vector.row_scales.left / .right; rows given directly under row_scales (older files) are the default
    # for a hand whose block leaves them out, else 1.0
    by_side_raw = {side: scales.pop(side, None) for side in ('left', 'right')}
    unknown = set(scales) - set(labels)
    if unknown:
        raise ValueError(f'retarget tuning vector.row_scales: unknown vectors {sorted(unknown)}; expected {labels}')
    vector['row_scales'] = [_positive('vector.row_scales', label, scales.get(label, 1.0)) for label in labels]
    vector['row_scales_by_side'] = {}
    for side, raw_side in by_side_raw.items():
        raw_side = dict(raw_side or {})
        unknown = set(raw_side) - set(labels)
        if unknown:
            raise ValueError(f'retarget tuning vector.row_scales.{side}: unknown vectors {sorted(unknown)}; expected {labels}')
        vector['row_scales_by_side'][side] = [_positive(f'vector.row_scales.{side}', label, raw_side[label]) if label in raw_side
                                              else base for label, base in zip(labels, vector['row_scales'])]
    orientation = dict(vector_raw.pop('thumb_orientation', None) or {})
    unknown = set(orientation) - {'proximal', 'distal', 'proximal_amplify', 'distal_amplify',
                                  'roll_enabled', 'proximal_roll_deg', 'distal_roll_deg'}
    if unknown:
        raise ValueError(f'retarget tuning vector.thumb_orientation: unknown keys {sorted(unknown)}')
    roll_enabled = orientation.get('roll_enabled', False)
    if type(roll_enabled) is not bool:
        raise ValueError('retarget tuning vector.thumb_orientation.roll_enabled must be true or false')
    rolls = {}
    for key in ('proximal_roll_deg', 'distal_roll_deg'):
        try:
            value = float(orientation.get(key, 0.0))
        except (TypeError, ValueError):
            raise ValueError(f'retarget tuning vector.thumb_orientation.{key} must be a number') from None
        if not np.isfinite(value) or abs(value) > 180:
            raise ValueError(f'retarget tuning vector.thumb_orientation.{key} must be within +-180 deg')
        rolls[key] = value
    vector['thumb_orientation'] = {key: _positive('vector.thumb_orientation', key, orientation.get(key, 0.0), allow_zero=True)
                                   for key in ('proximal', 'distal')}
    vector['thumb_orientation'].update({f'{key}_amplify': _positive('vector.thumb_orientation', f'{key}_amplify',
                                                                    orientation.get(f'{key}_amplify', 1.0))
                                        for key in ('proximal', 'distal')})
    vector['thumb_orientation'].update(roll_enabled=roll_enabled, **rolls)
    if vector_raw:
        raise ValueError(f'retarget tuning vector: unknown keys {sorted(vector_raw)}')
    options = {name: _positive('dexpilot', key, dexpilot_raw.pop(key, DEXPILOT_OPTIONS[name]), allow_zero=(name == 'norm_delta'))
               for name, key in DEXPILOT_OPTION_KEYS.items()}
    if options['escape_dist'] <= options['project_dist']:
        raise ValueError('retarget tuning dexpilot.escape_dist_m must exceed project_dist_m')
    weights_raw = dict(dexpilot_raw.pop('weights', None) or {})
    unknown = set(weights_raw) - set(DEXPILOT_WEIGHTS)
    if unknown:
        raise ValueError(f'retarget tuning dexpilot.weights: unknown keys {sorted(unknown)}')
    weights = {k: _positive('dexpilot.weights', k, weights_raw.get(k, v), allow_zero=True) for k, v in DEXPILOT_WEIGHTS.items()}
    if dexpilot_raw:
        raise ValueError(f'retarget tuning dexpilot: unknown keys {sorted(dexpilot_raw)}')
    glove = load_glove_tuning(dict(raw.get('glove') or {}))
    return dict(vector=vector, dexpilot=dict(options=options, weights=weights), glove=glove,
                path=None if path is None else str(path))


THUMB_TIP_OFFSET = dict(enabled=False, toward_index_deg=0.0, toward_index_mm=0.0, along_mm=0.0)


def load_glove_tuning(raw):
    """The glove section: corrections to the glove skeleton before any solver sees it."""
    unknown = set(raw) - {'thumb_tip_offset', 'knuckle_slant_deg'}
    if unknown:
        raise ValueError(f'retarget tuning glove: unknown keys {sorted(unknown)}')
    slant = raw.get('knuckle_slant_deg')
    if slant is not None:
        try:
            slant = float(slant)
        except (TypeError, ValueError):
            raise ValueError('retarget tuning glove.knuckle_slant_deg must be a number or null') from None
        if not np.isfinite(slant) or abs(slant) > 45:
            raise ValueError('retarget tuning glove.knuckle_slant_deg must be within +-45 deg')
    offset_raw = dict(raw.get('thumb_tip_offset') or {})
    unknown = set(offset_raw) - set(THUMB_TIP_OFFSET)
    if unknown:
        raise ValueError(f'retarget tuning glove.thumb_tip_offset: unknown keys {sorted(unknown)}')
    offset = dict(THUMB_TIP_OFFSET, **offset_raw)
    if type(offset['enabled']) is not bool:
        raise ValueError('retarget tuning glove.thumb_tip_offset.enabled must be true or false')
    for key, limit, unit in (('toward_index_deg', 90, 'deg'), ('toward_index_mm', 40, 'mm'), ('along_mm', 40, 'mm')):
        try:
            value = float(offset[key])
        except (TypeError, ValueError):
            raise ValueError(f'retarget tuning glove.thumb_tip_offset.{key} must be a number') from None
        if not np.isfinite(value) or abs(value) > limit:
            raise ValueError(f'retarget tuning glove.thumb_tip_offset.{key} must be within +-{limit} {unit}')
        offset[key] = value
    return dict(thumb_tip_offset=offset, knuckle_slant_deg=slant)


def thumb_tip_shift(tip, distal, index_knuckle, toward_index_mm, along_mm, toward_index_deg=0.0):
    """Shift (m, same frame as the inputs) for the glove thumb tip point, in a frame following the thumb:
    along = the distal bone (thumb distal joint -> tip), toward_index = from the tip toward the index knuckle with its
    along part removed (the side of the thumb tip that meets the fingers in a pinch).

    First the distal bone is rotated about the distal joint by toward_index_deg, about the axis along x toward_index,
    so the tip swings sideways toward the index (an orientation offset of the last bone; its length is kept). Then
    toward_index_mm / along_mm are added in that rotated frame. Zero if degenerate."""
    tip, distal = np.asarray(tip, dtype=float), np.asarray(distal, dtype=float)
    bone = tip - distal
    length = np.linalg.norm(bone)
    if length < LENGTH_EPS_M:
        return np.zeros(3)
    along = bone/length
    side = np.asarray(index_knuckle, dtype=float) - tip
    side = side - (side @ along)*along
    width = np.linalg.norm(side)
    if width < LENGTH_EPS_M:
        return along*along_mm/1000
    side = side/width
    angle = np.deg2rad(toward_index_deg)
    # Rotation about along x side by +angle takes along toward side: along' = cos along + sin side, side' = -sin along + cos side.
    along_r = np.cos(angle)*along + np.sin(angle)*side
    side_r = -np.sin(angle)*along + np.cos(angle)*side
    rotated_tip = distal + length*along_r
    return rotated_tip - tip + (side_r*toward_index_mm + along_r*along_mm)/1000
DEXPILOT_HUMAN_INDICES = np.array([[r['human_origin'] for r in DEXPILOT_TABLE], [r['human_task'] for r in DEXPILOT_TABLE]])
DEXPILOT_DISPLAY_ROWS = [0, 10, 11, 12, 13]  # viewer markers: thumb-index pair, then the four finger tips
COUPLINGS = [('thumb_joint5', 'thumb_joint4', .85)] + [
    (f'{f}_joint4', f'{f}_joint3', .8) for f in FINGERS[1:]]
BOUND_TOL_RAD = 2e-7  # Float32 package state rounding only (~0.000012 degrees).
POOR_FIT_M = .010
ALIGN_MOTION_M = .005
ALIGN_MOTION_DEG = 10.
STATUS = {1: 'SUCCESS', 2: 'STOPVAL_REACHED', 3: 'FTOL_REACHED', 4: 'XTOL_REACHED',
          5: 'MAXEVAL_REACHED', 6: 'MAXTIME_REACHED'}
ALIGNMENT_SCHEMA = 'manus-aac-alignment-v2'  # v2: palm basis takes the finger direction first (basis()); v1 files must be re-aligned


def validate_alignment(record, adapter, glove_id=None):
    """Validate saved data without reading its historical source log or clock."""
    try:
        expected = dict(schema=ALIGNMENT_SCHEMA, mapping_schema=SCHEMA, method='vector',
                        side=adapter.side, dex_retargeting_version=version('dex-retargeting'), scaling_factor=1.)
        if record['schema'] != ALIGNMENT_SCHEMA:
            raise ValueError(f"Alignment file schema {record['schema']!r} is not {ALIGNMENT_SCHEMA!r}: press A and S again")
        if any(record[key] != value for key, value in expected.items()) or type(record['scaling_factor']) not in (int, float):
            raise ValueError('Calibration schema/method/side/version/scale mismatch')
        gid = record['glove_id']
        if type(gid) is not int or not 0 < gid <= 0xffffffff or (glove_id is not None and gid != glove_id):
            raise ValueError('Calibration glove identity mismatch')
        digest = hashlib.sha256(adapter.model.path.read_bytes()).hexdigest()
        if record['robot_model']['sha256'] != digest:
            raise ValueError('Calibration model SHA-256 mismatch')
        a = np.asarray(record['alignment']['A'], dtype=float)
        if (a.shape != (3, 3) or not np.isfinite(a).all()
                or not np.allclose(a.T @ a, np.eye(3), rtol=0, atol=1e-6)
                or not np.isclose(np.linalg.det(a), 1., rtol=0, atol=1e-6)):
            raise ValueError('Calibration A must be a finite proper rotation')
        return dict(A=a.tolist(), C=(adapter.palm_root[:3, :3] @ a).tolist(), scale=1.,
                    source=record.get('source', {}))
    except (KeyError, TypeError) as error:
        raise ValueError('Malformed alignment calibration') from error


def load_alignment(path, adapter, glove_id=None):
    record = json.loads(Path(path).read_text())
    validate_alignment(record, adapter, glove_id)
    return record


def save_alignment(path, engine, glove_id):
    if engine.alignment is None or engine.collection is not None or engine.metadata is None:
        raise ValueError('No valid current alignment to save')
    record = dict(schema=ALIGNMENT_SCHEMA, mapping_schema=SCHEMA, method='vector', side=engine.side,
        glove_id=glove_id, dex_retargeting_version=version('dex-retargeting'), scaling_factor=1.,
        robot_model=dict(path_relative_to_repo=str(engine.adapter.model.path.relative_to(Path(__file__).resolve().parents[3])),
                         sha256=hashlib.sha256(engine.adapter.model.path.read_bytes()).hexdigest()),
        alignment=dict(convention='Column vectors: v_aac_palm = A @ v_manus_palm; C = R_root_from_aac_palm @ A',
                       A=engine.alignment['A']), source=engine.alignment['source'])
    validate_alignment(record, engine.adapter, glove_id)
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name+'.', delete=False) as output:
            temporary = Path(output.name)
            json.dump(record, output, indent=2, allow_nan=False)
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return record


class GeometryError(ValueError):
    """Expected invalid calibration geometry, not a programming error."""


def basis(mcp, pip):
    """Palm frame from four knuckles and four middle joints (index..pinky): columns x = across the palm toward the
    index, y = along the fingers, z = x cross y (the palm normal, mirrored between hands).

    y is taken first, from the mean knuckle->middle-joint direction, and the knuckle line is then made orthogonal
    to it. On a human hand the knuckle line is slanted (the index knuckle sits 15-25 mm further out than the
    pinky's); taking it as x first tilted the whole glove hand about the palm normal by that slant (~15-19 deg
    measured, more on the right glove), so every finger leaned toward the thumb after alignment. The D21's knuckle
    line is perpendicular to its fingers, so its basis is the same either way."""
    mcp, pip = np.asarray(mcp, dtype=float), np.asarray(pip, dtype=float)
    for points in (mcp, pip):
        if points.shape != (4, 3) or not np.isfinite(points).all():
            raise GeometryError('Expected finite geometry of shape (4, 3)')
    width, forward = mcp[0]-mcp[3], np.mean(pip-mcp, axis=0)
    w, f = np.linalg.norm(width), np.linalg.norm(forward)
    if min(w, f) < LENGTH_EPS_M:
        raise GeometryError(f'Zero/short palm width or forward vector: width_m={w:g}, forward_m={f:g}')
    y = forward/f
    orthogonal = width - y*np.dot(y, width)
    length = np.linalg.norm(orthogonal)
    sine = length/w
    if length < LENGTH_EPS_M or sine < BASIS_MIN_SINE:
        raise GeometryError(f'Nearly parallel palm width/forward vectors: width_m={w:g}, forward_m={f:g}, sine={sine:g}')
    x = orthogonal/length
    return np.column_stack((x, y, np.cross(x, y))), dict(width_m=float(w), forward_m=float(f), sine=float(sine))


def knuckle_slant_deg(mcp, y):
    """Angle of the index->pinky knuckle line out of the plane perpendicular to the finger direction y, in degrees:
    positive when the index knuckle sits further along the fingers than the pinky's. A relaxed hand with the fingers
    straight along the forearm measures about +14..+20 on the MANUS skeleton; since the alignment takes the finger
    direction as its axis, a very different value means the fingers were angled sideways during the capture, which
    turns the whole glove hand about the palm normal by that difference (both hands should match within a few
    degrees)."""
    width = np.asarray(mcp[0], dtype=float) - np.asarray(mcp[3], dtype=float)
    return float(np.degrees(np.arcsin(np.clip(np.dot(np.asarray(y, dtype=float), width)/np.linalg.norm(width), -1., 1.))))


def set_knuckle_slant(b, mcp, target_deg):
    """Turn the palm basis b (columns x, y, z from basis()) about its palm normal z so that the knuckle line's slant
    out of the plane perpendicular to y equals target_deg. The knuckle line is a fixed feature of the glove's hand
    model, so with a target every alignment lands on the same in-plane rotation however the fingers were held during
    the capture (glove.knuckle_slant_deg in retarget_tuning.yaml; the left hand's alignments measured +5.8)."""
    b = np.asarray(b, dtype=float)
    theta = np.deg2rad(knuckle_slant_deg(mcp, b[:, 1]) - float(target_deg))
    x, y, z = b[:, 0], b[:, 1], b[:, 2]
    return np.column_stack((np.cos(theta)*x + np.sin(theta)*y, -np.sin(theta)*x + np.cos(theta)*y, z))


def palm_point(poses, link):
    palm = poses['palm_down']
    return palm[:3, :3].T @ (poses[link][:3, 3]-palm[:3, 3])


def robot_landmarks(poses):
    return (np.array([palm_point(poses, f'{finger}_mcp') for finger in FINGERS[1:]]),
            np.array([palm_point(poses, f'{finger}_pip') for finger in FINGERS[1:]]))


def json_value(value):
    """Keep failure evidence writable even when the failing numerical input is NaN."""
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_value(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def checked_bounds(values, limits):
    q = np.asarray(values, dtype=float)
    if q.shape != (len(limits),) or not np.isfinite(q).all():
        raise ValueError('Nonfinite/malformed package joint output')
    if np.any(q < limits[:, 0]-BOUND_TOL_RAD) or np.any(q > limits[:, 1]+BOUND_TOL_RAD):
        raise ValueError('Package output outside effective joint limits')
    return np.clip(q, limits[:, 0], limits[:, 1])  # Only tolerated float32 roundoff.


def model_degrees(names, qpos, bounds):
    if len(names) != len(set(names)) or len(names) != len(qpos):
        raise ValueError('Invalid package joint name/order table')
    selected = np.asarray(qpos)[[names.index(n) for n in MODEL_JOINTS]]
    return np.rad2deg(checked_bounds(selected, bounds)).tolist()


class StrictVectorOptimizer(VectorOptimizer):
    """Only replace the package's exception-swallowing solve entrypoint."""
    def retarget(self, ref_value, fixed_qpos, last_qpos):
        self.attempt = dict(status=None, evaluations=None, previous_rad=np.asarray(last_qpos).tolist())
        if len(fixed_qpos) != len(self.idx_pin2fixed):
            raise ValueError('Unexpected fixed joint count')
        objective = self.get_objective_function(ref_value, fixed_qpos, np.asarray(last_qpos, dtype=np.float32))
        self.opt.set_min_objective(objective)
        # No catch: a RuntimeError must escape, even after a prior successful call.
        result = self.opt.optimize(last_qpos)
        status = int(self.opt.last_optimize_result())
        self.attempt.update(status=status, termination=STATUS.get(status, 'FAILURE'),
                            evaluations=int(self.opt.get_numevals()), cost=float(self.opt.last_optimum_value()),
                            candidate_rad=np.asarray(result).tolist())
        if status not in (1, 2, 3, 4):
            raise RuntimeError(f'NLopt did not converge: {self.attempt["termination"]} ({status})')
        limits = np.column_stack((self.opt.get_lower_bounds(), self.opt.get_upper_bounds()))
        result = checked_bounds(result, limits)
        if not np.isfinite(self.attempt['cost']):
            raise ValueError('Nonfinite NLopt objective value')
        return result.astype(np.float32)  # Preserve package sequencing precision.


class DexAdapter:
    CENTRELINE_POINTS = False  # the package solver cannot offset points from the link origins
    JOINT_TARGETS = False      # nor hold single joints (vector.joint_bias, thumb rotation lock)

    def __init__(self, side='Left', tuning=None):
        if version('dex-retargeting') != '0.5.0':
            raise RuntimeError('This adapter requires the inspected dex-retargeting 0.5.0 API')
        self.side, self.model = side, HandModel(side)
        self.tuning = load_tuning(tuning)  # retarget_tuning.yaml: loss shape, smoothing and per-vector weights
        # Optional extra PIP -> tip rows (vector.pip_tip_rows; weight 0 = off), appended after the ten standard rows.
        self.pip_tip = [(f, row) for f, row in self.tuning['vector']['pip_tip_rows'].items() if row['weight'] > 0]
        self.base_row_weights = list(self.tuning['vector']['row_weights']) + [row['weight'] for _, row in self.pip_tip]
        self.row_weights = list(self.base_row_weights)
        # Per-vector human->robot length scale, times --scale.
        by_side = self.tuning['vector'].get('row_scales_by_side') or {}   # vector.row_scales.left / .right override the base
        self.row_scales = list(by_side.get(self.side.lower()) or self.tuning['vector']['row_scales']) + [row['scale'] for _, row in self.pip_tip]
        self.output_filter = None  # optional One-Euro filter on the solved joints (teleop/manus/tools/one_euro.py)
        # Thumb offset: the D21 thumb at CMC 0 points straight out sideways, a direction a human thumb
        # cannot reach, so a 1:1 direction mapping leaves the robot's outward CMC range unused and pins
        # the CMC at its inward limit. Positive degrees rotate the two thumb vectors toward the index
        # side (outward), in the palm plane, before the solve.
        self.thumb_offset_deg = 0.0
        self.method = 'vector'
        self.thumb_anchor = self.tuning['vector']['thumb_anchor']
        self.base_table, self.base_origin_links, self.base_task_links, self.human_indices = vector_table_for(
            self.thumb_anchor, [f for f, _ in self.pip_tip])
        self.vector_table = self.base_table
        self.origin_links, self.task_links = self.base_origin_links, self.base_task_links
        self.display_rows = list(range(5))  # tip vectors shown by the viewer
        # Robot points on the finger centrelines (see centreline_offsets); only the dexrt solvers take offsets, the
        # dex-retargeting package works on link origins, so there the vendor frames stay in use.
        self.link_offsets = centreline_offsets(self.model) if self.CENTRELINE_POINTS else {}
        self.origin_offsets = [self.link_offsets.get(link) for link in self.origin_links]
        # Per row: point fixed on the task link (link frame, m), or None for the link origin.
        self.task_offsets = [self.link_offsets.get(link) for link in self.task_links]
        self.orientation_bones = []  # (key, glove joint, robot link, weight) per thumb bone with orientation rows
        self.orientation_weight = dict(proximal=0.0, distal=0.0)
        # Multiplies the glove bone's rotation since the reference before it is applied to the robot link's axes
        # (vector.thumb_orientation.<bone>_amplify; 1 = off): the glove under-reports thumb bone rotation.
        orientation = self.tuning['vector']['thumb_orientation']
        self.orientation_amplify = dict(proximal=orientation['proximal_amplify'], distal=orientation['distal_amplify'])
        # Fixed roll of each thumb bone's orientation target about the bone's own long axis (base -> tip), as if the
        # glove sensor were remounted turned by that angle: the robot link is asked to roll by it relative to the
        # reference pose. Right-hand rule about the base->tip direction; off unless roll_enabled.
        self.orientation_roll = (dict(proximal=orientation['proximal_roll_deg'], distal=orientation['distal_roll_deg'])
                                 if orientation['roll_enabled'] else dict(proximal=0.0, distal=0.0))
        self.vector_labels = [r.get('label') or f"{r['human_chain'][6:].lower()}-{'tip' if r['robot_task'].endswith('_tip') else 'mid'}"
                              for r in self.base_table]
        self.scale = 1.0
        # Thumb shaping after the solve: the CMC (motor 3) angle times a gain, and an optional hold on the
        # rotation joint (motors 1 and 2 then move together: abduction only). Both default to off.
        self.thumb_cmc_gain = 1.0
        # The glove under-reports thumb CMC motion, so the swing of the human thumb metacarpal away from its
        # relaxed direction is multiplied by this before the vectors are built (vector method and DexPilot).
        self.thumb_cmc_amplify = 1.0
        # Optional soft cap on the amplified swing: swing' = cap * tanh(gain * swing / cap). Small swings still get
        # the full gain; large ones approach the cap instead of pushing the thumb past what the robot can reach.
        self.thumb_cmc_amplify_max_deg = None
        # Hold the thumb rotation joint (motor pair 1-2 then move together: abduction only) at a fixed angle, from
        # vector.thumb_rotation_lock in the tuning YAML; --thumb-rotation-lock overrides it. dexrt solvers only.
        lock = self.tuning['vector']['thumb_rotation_lock']
        lock = lock.get(self.side.lower(), lock)   # per-hand block (vector.thumb_rotation_lock.left / .right)
        self.thumb_rotation_lock_weight = float(lock['weight'])
        self.set_thumb_rotation_lock(lock['angle_deg'] if lock['enabled'] else None)
        # Soft pull of each finger's knuckle (joint2) toward 0 (vector.mcp_soft_zero; weight 0 = off): with the -mid
        # rows off, this makes the solver take a fingertip bend at the middle joint instead of the knuckle. dexrt only.
        # YAML weight unit: 1.0 means 10 deg of knuckle costs as much as 10 mm of fingertip error (the solver's own
        # joint term is in rad^2 against a tip term of 1/(m*delta) per m^2, where a raw 0.1 already pins a joint).
        unit = (0.01**2/(10*self.tuning['vector']['huber_delta_m']))/np.deg2rad(10.0)**2
        self.joint_bias_unit = unit
        self.joint_bias = {f'{finger}_joint2': (0.0, weight*unit)
                           for finger, weight in self.tuning['vector']['mcp_soft_zero'].items() if weight > 0}
        # Glove-side correction of the thumb tip point (retarget_tuning.yaml glove.thumb_tip_offset).
        self.thumb_tip_offset = dict(self.tuning['glove']['thumb_tip_offset'])
        self.config = RetargetingConfig(type='vector', urdf_path=str(self.model.path),
            target_joint_names=list(MODEL_JOINTS), target_origin_link_names=list(self.base_origin_links),
            target_task_link_names=list(self.base_task_links), target_link_human_indices=self.human_indices.copy(),
            add_dummy_free_joint=False)
        if self.config.scaling_factor != 1.:
            raise RuntimeError('Expected default package scaling_factor=1.0')
        vector = self.tuning['vector']
        self.config.huber_delta, self.config.normal_delta = vector['huber_delta_m'], vector['norm_delta']
        self.config.low_pass_alpha = vector['low_pass_alpha']
        # No URDF rewrite is needed: Pinocchio reads the original directly, no visuals.
        self.robot = RobotWrapper(self.config.urdf_path)
        self.pose_indices = {link: self.robot.get_link_index(link)
                             for link in dict.fromkeys(('palm_down', *ORIGIN_LINKS, *TASK_LINKS,
                                                        *[r['robot_origin'] for r in DEXPILOT_TABLE],
                                                        *[r['robot_task'] for r in DEXPILOT_TABLE],
                                                        'thumb_ip'))}  # thumb tip orientation rows
        self.optimizer = StrictVectorOptimizer(self.robot, self.config.target_joint_names,
            self.config.target_origin_link_names, self.config.target_task_link_names,
            self.config.target_link_human_indices, huber_delta=self.config.huber_delta,
            norm_delta=self.config.normal_delta, scaling=self.config.scaling_factor)
        dependent, sources, factors = zip(*COUPLINGS)
        self.adaptor = MimicJointKinematicAdaptor(self.robot, list(MODEL_JOINTS), list(sources),
                                                list(dependent), list(factors), [0.]*5)
        self.optimizer.set_kinematic_adaptor(self.adaptor)
        if self.robot.dof != 21 or len(self.optimizer.idx_pin2fixed) or self.optimizer.has_free_joint:
            raise RuntimeError('AAC requires 16 independent variables, 5 coupled joints and a fixed root')
        urdf = self.robot.joint_limits
        limits = urdf[self.optimizer.idx_pin2target].copy()
        sdk = np.deg2rad([ANGLE_LIMITS[i] for i in range(16)])
        limits[:, 0] = np.maximum(limits[:, 0], sdk[:, 0])
        limits[:, 1] = np.minimum(limits[:, 1], sdk[:, 1])
        for dependent, source, factor in COUPLINGS:
            i = MODEL_JOINTS.index(source)
            derived = urdf[self.robot.get_joint_index(dependent)]/factor
            limits[i, 0] = max(limits[i, 0], derived[0])
            limits[i, 1] = min(limits[i, 1], derived[1])
        if not np.isfinite(limits).all() or np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError('Invalid effective AAC joint limits')
        self.bounds = limits
        self.sequence = SeqRetargeting(self.optimizer, has_joint_limits=self.config.has_joint_limits,
                                      lp_filter=LPFilter(self.config.low_pass_alpha))
        # SeqRetargeting initially installs expanded URDF bounds; replace before use.
        self.sequence.joint_limits = limits.copy()
        self.optimizer.set_joint_limit(limits, epsilon=0.)
        self.reset()
        self.robot.compute_forward_kinematics(np.zeros(self.robot.dof))
        self.palm_root = self.robot.get_link_pose(self.robot.get_link_index('palm_down')).copy()
        self.robot_basis = basis(*robot_landmarks(self.model.transforms([0.]*16)))[0]

    def reset(self):
        self.sequence.reset()
        self.sequence.filter.reset()  # Package SeqRetargeting.reset does not do this.
        self.optimizer.attempt = dict(status=None, evaluations=None)
        if getattr(self, 'output_filter', None) is not None:
            self.output_filter.reset()

    @staticmethod
    def link_point(poses, link, offset=None):
        pose = poses[link]
        return pose[:3, 3] if offset is None else pose[:3, 3] + pose[:3, :3] @ np.asarray(offset, dtype=float)

    def point(self, poses, link):
        """A link's centreline point (link origin plus its centreline offset, if any), root frame."""
        return self.link_point(poses, link, self.link_offsets.get(link))

    def task_points(self, poses):
        """Task point of every row: the task link's centreline point, or a point fixed on it (orientation rows)."""
        return np.array([self.link_point(poses, link, off) for link, off in zip(self.task_links, self.task_offsets)])

    def origin_points(self, poses):
        return np.array([self.link_point(poses, link, off) for link, off in zip(self.origin_links, self.origin_offsets)])

    def warmup(self):
        """Initialize numerical kernels before acquisition, retaining no pose history."""
        poses = self.package_poses(self.full_qpos(self.bounds.mean(axis=1)))
        vectors = self.task_points(poses) - self.origin_points(poses)
        self.solve(vectors)
        self.reset()

    def full_qpos(self, independent):
        q = np.zeros(self.robot.dof)
        q[self.optimizer.idx_pin2target] = checked_bounds(independent, self.bounds)
        return self.adaptor.forward_qpos(q)

    def package_poses(self, q):
        checked_bounds(q, self.robot.joint_limits)
        self.robot.compute_forward_kinematics(q)
        return {link: self.robot.get_link_pose(index).copy() for link, index in self.pose_indices.items()}

    def _retarget(self, vectors):
        """Package solve. Returns (raw, displayed) full package qpos; displayed is low-pass filtered."""
        displayed = self.sequence.retarget(vectors)
        checked_bounds(displayed, self.robot.joint_limits)
        return self.full_qpos(self.sequence.last_qpos), displayed

    def offset_thumb(self, vectors):
        """Rotate the thumb tip and proximal vectors about the palm normal by thumb_offset_deg (outward +)."""
        if not self.thumb_offset_deg:
            return vectors
        # robot_basis columns: x = index->pinky width (toward the index), y = toward the fingertips, z = x cross y.
        # A rotation about z by -theta moves a y-pointing vector toward +x, i.e. the thumb swings outward.
        rotation = Rotation.from_rotvec(-np.deg2rad(self.thumb_offset_deg)*self.robot_basis[:, 2]).as_matrix()
        out = vectors.copy()
        for row, spec in enumerate(self.vector_table):
            if spec['human_chain'] == 'FingerThumb':
                out[row] = rotation @ vectors[row]
        return out

    def set_thumb_rotation_lock(self, angle_deg):
        """Hold thumb_joint2 at angle_deg given in the LEFT hand's convention (tuning YAML / --thumb-rotation-lock).

        The vendor's right URDF keeps the left's thumb_joint2 axis on mirrored geometry, so the same number turns
        the thumb the opposite way (+ tips the right thumb toward the palm, the left thumb away); the right hand
        therefore holds -angle_deg so both hands hold mirror poses. None disables the lock."""
        if angle_deg is None:
            self.thumb_rotation_lock_deg = None
        else:
            angle = float(angle_deg)
            self.thumb_rotation_lock_deg = angle if self.side == 'Left' else -angle

    @property
    def display_labels(self):
        return [self.vector_labels[i] for i in self.display_rows]

    def set_scale(self, scale):
        """Human -> robot length scale: every target vector is the human vector times this (1.0 = metres kept)."""
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError('scale must be a positive number')
        self.scale = scale
        self.optimizer.scaling = scale  # package VectorOptimizer multiplies its targets by this every solve

    def solved_targets(self, vectors):
        return vectors*self.scale

    def scale_rows(self, vectors):
        """Per-vector scale from the tuning (vector.row_scales), applied before the solver's global scale; rows beyond
        the table (orientation rows) are left alone."""
        scales = np.ones(len(vectors))
        n = min(len(self.row_scales), len(vectors))
        scales[:n] = self.row_scales[:n]
        return vectors*scales[:, None]

    def apply_joint_targets(self, targets_deg):
        """Install per-joint targets {model joint name: degrees} for the next solve (dexrt solvers only)."""
        if targets_deg:
            raise RuntimeError('joint targets need --solver dexrt (dex-retargetor-cpp)')

    def shape_thumb(self, degrees):
        """Post-solve thumb shaping in model-joint degrees: CMC gain, clipped to the joint's limits."""
        if self.thumb_cmc_gain == 1.0:
            return list(degrees)
        out = list(degrees)
        i = MODEL_JOINTS.index('thumb_joint3')
        lo, hi = np.rad2deg(self.bounds[i])
        out[i] = float(np.clip(out[i]*self.thumb_cmc_gain, lo, hi))
        return out

    def solve(self, vectors, stamp=None, joint_targets_deg=None):
        if self.joint_bias and self.JOINT_TARGETS:
            joint_targets_deg = dict(self.joint_bias, **(joint_targets_deg or {}))
        if self.thumb_rotation_lock_deg is not None and not (joint_targets_deg and 'thumb_joint2' in joint_targets_deg):
            joint_targets_deg = dict(joint_targets_deg or {},
                                     thumb_joint2=(self.thumb_rotation_lock_deg, self.thumb_rotation_lock_weight))
        self.apply_joint_targets(joint_targets_deg)
        vectors = np.asarray(vectors, dtype=float)
        if vectors.shape != (len(self.vector_table), 3) or not np.isfinite(vectors).all():
            raise ValueError(f'Vector retargeting requires {len(self.vector_table)} finite vectors in meters')
        input_vectors = vectors
        vectors = self.scale_rows(self.offset_thumb(vectors))
        previous = self.sequence.last_qpos.copy()
        previous_filter = None if self.sequence.filter.y is None else self.sequence.filter.y.copy()
        started = time.perf_counter()
        raw, displayed = self._retarget(vectors)
        if self.output_filter is not None:
            # One-Euro on the 16 independent joints (degrees, model order), then the coupled joints
            # are rebuilt from them so the 0.8/0.85 couplings stay exact.
            degrees = np.asarray(model_degrees(self.sequence.joint_names, displayed, self.bounds))
            filtered = self.output_filter(degrees, started if stamp is None else stamp)
            displayed = self.full_qpos(np.clip(np.deg2rad(filtered), self.bounds[:, 0], self.bounds[:, 1]))
        solve_s = time.perf_counter()-started
        raw_degrees = self.shape_thumb(model_degrees(self.sequence.joint_names, raw, self.bounds))
        display_degrees = self.shape_thumb(model_degrees(self.sequence.joint_names, displayed, self.bounds))
        if self.thumb_cmc_gain != 1.0:  # keep the package-order arrays consistent with the shaped degrees
            raw = self.full_qpos(np.deg2rad(raw_degrees))
            displayed = self.full_qpos(np.deg2rad(display_degrees))
        raw_poses = self.package_poses(raw)
        # Residuals for display use the exact native degree array sent to HandVisual.
        display_poses = self.model.transforms(display_degrees)
        origins = self.origin_points(display_poses)
        raw_origins = self.origin_points(raw_poses)
        solved_targets = self.solved_targets(vectors)  # what the solver was actually asked for
        target = origins+solved_targets
        raw_target = raw_origins+solved_targets
        achieved = self.task_points(display_poses)
        raw_achieved = self.task_points(raw_poses)
        errors = np.linalg.norm(achieved-target, axis=1)
        angles = np.deg2rad(display_degrees)
        return dict(input_vectors_root_m=input_vectors.tolist(), solved_vectors_root_m=vectors.tolist(),
                    thumb_offset_deg=self.thumb_offset_deg, previous_rad=previous.tolist(),
                    previous_filter_rad=None if previous_filter is None else previous_filter.tolist(),
                    raw_package_rad=raw.tolist(), display_package_rad=displayed.tolist(),
                    raw_degrees=raw_degrees, display_degrees=display_degrees,
                    raw_named_deg=dict(zip(MODEL_JOINTS, raw_degrees)), display_named_deg=dict(zip(MODEL_JOINTS, display_degrees)),
                    origin_root_m=origins.tolist(), target_root_m=target.tolist(), achieved_root_m=achieved.tolist(),
                    raw_origin_root_m=raw_origins.tolist(), raw_target_root_m=raw_target.tolist(), raw_achieved_root_m=raw_achieved.tolist(),
                    raw_errors_m=np.linalg.norm(raw_achieved-raw_target, axis=1).tolist(), display_errors_m=errors.tolist(),
                    bound_flags=np.any(np.abs(angles[:, None]-self.bounds) <= BOUND_TOL_RAD, axis=1).tolist(),
                    method=self.method, scale=self.scale, projected=self.projected_flags(),
                    joint_targets_deg=joint_targets_deg or None,
                    fit_status='poor fit' if max(errors) > POOR_FIT_M else 'valid fit',
                    solver=dict(self.optimizer.attempt), solve_s=solve_s)

    def projected_flags(self):
        return None

    def configuration(self):
        opt = self.optimizer.opt
        return json_value(dict(schema=SCHEMA, method=self.method, scale=self.scale, side=self.side,
            output_filter=None if self.output_filter is None else self.output_filter.describe(),
            thumb_offset_deg=self.thumb_offset_deg, thumb_cmc_gain=self.thumb_cmc_gain,
            thumb_cmc_amplify=self.thumb_cmc_amplify, thumb_cmc_amplify_max_deg=self.thumb_cmc_amplify_max_deg,
            thumb_orientation_weight=self.orientation_weight, thumb_orientation_amplify=self.orientation_amplify,
            thumb_orientation_roll_deg=self.orientation_roll,
            thumb_tip_offset=self.thumb_tip_offset,
            thumb_rotation_lock_deg=self.thumb_rotation_lock_deg, mcp_soft_zero=self.tuning['vector']['mcp_soft_zero'],
            versions={p: version(p) for p in ('dex-retargeting', 'pin', 'nlopt', 'numpy', 'torch')},
            urdf_path=str(self.model.path), urdf_sha256=hashlib.sha256(self.model.path.read_bytes()).hexdigest(),
            settings=asdict(self.config), vector_table=self.vector_table, thumb_anchor=self.thumb_anchor, model_joint_names=MODEL_JOINTS,
            tuning=self.tuning, row_weights=self.row_weights, row_scales=self.row_scales,
            centreline_offsets_m={k: v.tolist() for k, v in self.link_offsets.items()},
            human_local_array=[dict(index=i, chain=c, joint=j) for i, (c, j) in enumerate(HUMAN_LANDMARKS)],
            package_joint_names=self.sequence.joint_names, effective_bounds_rad=self.bounds,
            effective_bounds_deg=np.rad2deg(self.bounds), couplings=COUPLINGS, boundary_tolerance_rad=BOUND_TOL_RAD,
            poor_fit_m=POOR_FIT_M, palm_root=self.palm_root, units='meters / radians internally; native URDF degrees for display',
            nlopt=dict(algorithm=opt.get_algorithm_name(), ftol_abs=opt.get_ftol_abs(), ftol_rel=opt.get_ftol_rel(),
                       xtol_abs=opt.get_xtol_abs(), xtol_rel=opt.get_xtol_rel(), maxeval=opt.get_maxeval(), maxtime=opt.get_maxtime())))


class DexrtAdapter(DexAdapter):
    CENTRELINE_POINTS = True
    JOINT_TARGETS = True
    """Same objective, bounds and geometry, solved by dex-retargetor-cpp (IRLS Huber on OSQP).

    No output low-pass filter: raw and displayed angles are identical.
    """
    def __init__(self, side='Left', tuning=None):
        super().__init__(side, tuning)
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'retargetor' / 'dex-retargetor-cpp' / 'python'))
        import dexrt
        self.dexrt = dexrt
        self.solver = self.build_solver()
        self.reset()

    def set_scale(self, scale):
        super().set_scale(scale)
        self.solver.set_scaling(self.scale)

    def apply_joint_targets(self, targets_deg):
        if not targets_deg:
            self.solver.clear_joint_targets()
            return
        target, weight = np.zeros(16), np.zeros(16)
        for name, (deg, w) in targets_deg.items():
            i = MODEL_JOINTS.index(name)
            target[i], weight[i] = np.deg2rad(deg), w
        self.solver.set_joint_targets(target, weight)

    @staticmethod
    def offset_tuple(offset):
        return (0., 0., 0.) if offset is None else tuple(float(v) for v in offset)

    # The C++ solver needs every weight > 0; a row weight of 0 in the YAML means "off", sent as this negligible weight
    # (1e-9 of a normal row: it cannot move the solution, the row stays in the table so logs keep their layout).
    OFF_WEIGHT = 1e-9

    def build_solver(self):
        rows = [(o, t, max(float(w), self.OFF_WEIGHT), 0, -1, -1, self.offset_tuple(oo), self.offset_tuple(to))
                for o, t, w, oo, to in zip(self.origin_links, self.task_links, self.row_weights,
                                           self.origin_offsets, self.task_offsets)]
        return self.dexrt.Retargeter(self.model.path, MODEL_JOINTS, COUPLINGS, rows,
                                     self.bounds[:, 0], self.bounds[:, 1], huber_delta=self.config.huber_delta,
                                     norm_delta=self.config.normal_delta, scaling=self.scale)

    # Thumb orientation (vector method): two axes fixed on the robot's thumb_mcp link (the proximal phalanx),
    # 3 cm along its local x and y, rotated by FK, are matched to the glove's thumb proximal bone rotation.
    # Thumb orientation (vector method): two axes fixed on a robot thumb link, 3 cm along its local x and y,
    # rotated by FK, are matched to a glove thumb bone's rotation. glove joint -> robot link (both the bone that
    # starts there): proximal phalanx and distal phalanx (the tip segment; on the robot it follows MCP at 0.85x).
    ORIENTATION_BONES = (('proximal', 'Proximal', 'thumb_mcp'), ('distal', 'Distal', 'thumb_ip'))
    ORIENTATION_AXES = (np.array([0.03, 0., 0.]), np.array([0., 0.03, 0.]))

    def enable_thumb_orientation(self, proximal=0.0, distal=0.0):
        """Append two orientation rows per thumb bone with weight > 0 and rebuild the solver."""
        if self.method != 'vector':
            raise ValueError('thumb orientation is only for the vector method')
        weights = dict(proximal=float(proximal), distal=float(distal))
        if any(w < 0 for w in weights.values()):
            raise ValueError('orientation weights must be >= 0')
        n = len(self.base_table)
        self.orientation_bones = [(key, joint, link, weights[key]) for key, joint, link in self.ORIENTATION_BONES
                                  if weights[key] > 0]
        self.orientation_weight = weights
        rows = [(joint, link, axis, w) for _, joint, link, w in self.orientation_bones for axis in self.ORIENTATION_AXES]
        self.vector_table = list(self.base_table) + [
            dict(human_chain='ThumbOrientation', human_origin=joint, human_task=joint, robot_origin=link,
                 robot_task=link, axis=axis.tolist()) for joint, link, axis, _ in rows]
        self.origin_links = list(self.base_origin_links) + [link for _, link, _, _ in rows]
        self.task_links = list(self.base_task_links) + [link for _, link, _, _ in rows]
        self.origin_offsets = [self.link_offsets.get(link) for link in self.base_origin_links] + [None] * len(rows)
        self.task_offsets = [self.link_offsets.get(link) for link in self.base_task_links] + [axis for _, _, axis, _ in rows]
        self.row_weights = list(self.base_row_weights) + [w for _, _, _, w in rows]
        self.vector_labels = self.vector_labels[:n] + [f"{link.replace('thumb_', 'thumb-')}-axis-{'xy'[k % 2]}"
                                                       for k, (_, link, _, _) in enumerate(rows)]
        self.solver = self.build_solver()
        self.reset()

    def orientation_reference(self, position_vectors, iterations=30):
        """Robot thumb axes (root frame) after solving this frame from positions alone: the orientation rows'
        targets are set to the robot's own axes each iteration, so they never pull. Two rows per enabled bone."""
        position_vectors = self.offset_thumb_rows(np.asarray(position_vectors, dtype=float))
        q = self.solver.last_q
        for _ in range(iterations):
            axes = self.solver.robot_vectors(q)[len(self.base_table):]
            q, _, _ = self.solver.solve(np.vstack([position_vectors, axes/self.scale]))
        return self.solver.robot_vectors(q)[len(self.base_table):]

    def offset_thumb_rows(self, position_vectors):
        padded = np.vstack([position_vectors, np.zeros((len(self.vector_table)-len(position_vectors), 3))])
        return self.offset_thumb(padded)[:len(position_vectors)]

    def reset(self):
        super().reset()
        if getattr(self, 'solver', None) is not None:
            self.solver.reset()

    def _retarget(self, vectors):
        previous = self.solver.last_q
        q, _, info = self.solver.solve(vectors)
        status = 1 if info['converged'] else 5
        self.optimizer.attempt = dict(status=status, termination=STATUS[status], evaluations=info['osqp_iterations'],
                                      cost=info['cost'], steps=info['steps'], previous_rad=previous.tolist(),
                                      candidate_rad=q.tolist(), solver='dex-retargetor-cpp')
        if not info['converged']:
            raise RuntimeError('OSQP did not converge')
        q = checked_bounds(q, self.bounds)
        self.sequence.last_qpos = q  # keeps previous_rad and failure evidence meaningful
        raw = self.full_qpos(q)
        return raw, raw

    def configuration(self):
        result = super().configuration()
        result['solver'] = dict(name='dex-retargetor-cpp', method='IRLS Huber weights, box-constrained QP on OSQP',
                                low_pass_filter=None, huber_delta=self.config.huber_delta,
                                norm_delta=self.config.normal_delta)
        return result


class DexPilotAdapter(DexrtAdapter):
    """DexPilot retargeting on dex-retargetor-cpp: fingertip-to-fingertip vectors with contact projection.

    The thumb is placed relative to the other fingertips instead of by its direction from its own base,
    which the D21's thumb geometry cannot reproduce. Targets are scaled by `scale` (robot palm width /
    human palm width, set from the live skeleton) except projected pairs, which snap to contact.
    """
    DEXPILOT_OPTIONS = DEXPILOT_OPTIONS

    def __init__(self, side='Left', tuning=None, **options):
        tuning = load_tuning(tuning)
        self.dexpilot_options = dict(tuning['dexpilot']['options'], **options)
        self.dexpilot_table = _dexpilot_table(tuning['dexpilot']['weights'])
        super().__init__(side, tuning)
        self.method = 'dexpilot'

    def build_solver(self):
        table = self.dexpilot_table
        self.vector_table, self.human_indices = table, DEXPILOT_HUMAN_INDICES
        self.origin_links = [r['robot_origin'] for r in table]
        self.task_links = [r['robot_task'] for r in table]
        self.display_rows = DEXPILOT_DISPLAY_ROWS
        self.vector_labels = [r['label'] for r in table]
        self.row_weights = [r['weight'] for r in table]
        self.row_scales = [1.0]*len(table)  # the YAML row_scales name vector-method rows; DexPilot scales by palm width
        self.origin_offsets = [self.link_offsets.get(link) for link in self.origin_links]
        self.task_offsets = [self.link_offsets.get(link) for link in self.task_links]
        rows = [(r['robot_origin'], r['robot_task'], r['weight'], r['projection'], r['gate_a'], r['gate_b'],
                 self.offset_tuple(oo), self.offset_tuple(to))
                for r, oo, to in zip(table, self.origin_offsets, self.task_offsets)]
        return self.dexrt.Retargeter(self.model.path, MODEL_JOINTS, COUPLINGS, rows, self.bounds[:, 0], self.bounds[:, 1],
                                     scaling=self.scale, **self.dexpilot_options)

    def solved_targets(self, vectors):
        """Scaled, or snapped to the contact distance where the solver projected the pair."""
        projected = self.solver.projected
        out = vectors*self.scale
        for row, spec in enumerate(self.vector_table):
            if projected[row]:
                eta = self.dexpilot_options['eta_primary' if spec['projection'] == 1 else 'eta_secondary']
                out[row] = vectors[row]/(np.linalg.norm(vectors[row])+1e-6)*eta
        return out

    def projected_flags(self):
        return self.solver.projected.tolist()

    def configuration(self):
        result = super().configuration()
        result['solver'].update(method='DexPilot: fingertip pair vectors with contact projection, IRLS Huber on OSQP',
                                dexpilot=self.dexpilot_options, vector_labels=[r['label'] for r in self.dexpilot_table],
                                dexpilot_weights={r['label']: r['weight'] for r in self.dexpilot_table})
        return result


def select_landmarks(metadata, side):
    nodes = {n['id']: n for n in metadata}
    if len(nodes) != len(metadata):
        raise ValueError('Duplicate node IDs')
    def choose(chain, joint=None):
        ids = [n['id'] for n in metadata if n['side']==side and n['chain']==chain and (joint is None or n['joint']==joint)]
        if len(ids) != 1:
            raise ValueError(f'Missing/ambiguous {side} {chain}/{joint}: candidates={ids}; metadata={metadata}')
        return ids[0]
    def descends(child, ancestor, chain):
        seen = set()
        while child != ancestor:
            if child not in nodes or child in seen or nodes[child]['side'] != side or nodes[child]['chain'] != chain:
                raise ValueError(f'Invalid {chain} ancestry to {ancestor}: node={child}; metadata={metadata}')
            seen.add(child)
            child = nodes[child]['parent']
    local_ids = [choose(chain, joint) for chain, joint in HUMAN_LANDMARKS]
    palm, tasks, origins = local_ids[0], local_ids[1:11], local_ids[11:]
    for i, finger in enumerate(FINGERS):
        chain = 'Finger'+finger.title()
        descends(tasks[i], tasks[i+5], chain)
        descends(tasks[i+5], origins[i], chain)
        descends(origins[i], palm, chain)
    return dict(palm=palm, tasks=tasks, origins=origins, mcp=origins[1:], pip=tasks[6:], local_ids=local_ids)


ALIGN_PROMPT = 'Press A: straighten and bring together non-thumb fingers'


class VectorDiagnostic:
    """Per-run alignment and freshness; consumes copied distinct skeleton frames only."""
    def __init__(self, adapter):
        self.adapter, self.side = adapter, adapter.side
        self.metadata = self.selected = self.alignment = self.collection = None
        # Camera source: a tracking gap does not change the hand's own geometry, so the alignment survives it;
        # a webcam also gives fewer frames per alignment capture than the glove (~25 vs ~100 per second).
        self.keep_alignment_on_gap = False
        self.min_capture_frames = MIN_CAPTURE_FRAMES
        self.last_frame_id = self.last_received = self.result = self.geometry = None
        self.reason = ALIGN_PROMPT
        self.count = self.skipped = 0
        self.events = []
        self.orientation_ref = None  # (glove thumb proximal rotation in the root frame, robot thumb axes) at reference
        self.cmc_reference = None   # relaxed thumb metacarpal direction (palm frame), for --thumb-cmc-amplify
        self.cmc_deviation_deg = None
        self.auto_scale = True      # DexPilot: scale = robot palm width / human palm width, from the skeleton
        self.scale_set = False
        self.robot_palm_width_m = basis(*robot_landmarks(adapter.model.transforms([0.]*16)))[1]['width_m']

    def apply_auto_scale(self, xyz):
        """Set the DexPilot length scale from this frame's knuckle width (the MANUS skeleton's bone lengths are
        fixed per user, so one frame is enough)."""
        mcp = np.array([xyz[i] for i in self.selected['mcp']])
        width = float(np.linalg.norm(mcp[0]-mcp[3]))
        if width < LENGTH_EPS_M:
            return
        scale = self.robot_palm_width_m/width
        self.adapter.set_scale(scale)
        self.scale_set = True
        self.events.append(dict(outcome='scale', scale=scale, human_palm_width_m=width, robot_palm_width_m=self.robot_palm_width_m))
        print(f'DexPilot scale {scale:.3f} (robot palm width {self.robot_palm_width_m*1000:.1f} mm / human {width*1000:.1f} mm)')

    def set_fixed_scale(self, scale):
        self.auto_scale = False
        self.adapter.set_scale(scale)
        self.scale_set = True

    def invalidate(self, reason, clear_alignment=True):
        if self.collection is not None:
            self.events.append(dict(outcome='failure', reason=reason, count=len(self.collection['samples']),
                                    started=self.collection['started']))
        if self.result is not None or self.reason != reason:
            self.events.append(dict(outcome='invalid', reason=reason))
        self.collection = self.result = None
        if clear_alignment:
            self.alignment = None
        self.adapter.reset()
        self.reason = reason

    def set_metadata(self, metadata):
        metadata = sorted(metadata, key=lambda n: n['id'])
        if metadata == self.metadata:
            return False
        self.invalidate('Hierarchy changed; press A again')
        self.geometry = None
        self.selected = select_landmarks(metadata, self.side)
        self.metadata = metadata
        return True

    def install_alignment(self, record, glove_id):
        if self.metadata is None or self.selected is None or self.collection is not None:
            raise ValueError('Install alignment after metadata initialization, outside capture')
        alignment = validate_alignment(record, self.adapter, glove_id)
        self.invalidate('Loaded alignment; waiting for a fresh skeleton')
        self.alignment = alignment
        self.events.append(dict(outcome='loaded', **alignment))

    def amplify_thumb_cmc(self, local):
        """Multiply the thumb metacarpal's swing about the CMC (node 1) by adapter.thumb_cmc_amplify.

        local: HUMAN_LANDMARKS positions in the palm frame (1 = thumb tip, 6 = thumb proximal, 11 = thumb
        metacarpal). The swing is the rotation taking the relaxed metacarpal direction to the current one;
        the rest of that rotation, (gain - 1) times its angle about the same axis, is applied to the whole
        thumb chain about node 1, so the thumb keeps its shape and only the CMC motion grows. The relaxed
        direction is the first live frame after each alignment: hold the thumb relaxed then."""
        swing = self.thumb_cmc_rotation(local)
        return local if swing is None else self.apply_thumb_swing(local, swing, rows=(6, 1))

    def thumb_cmc_rotation(self, local):
        """The extra rotation (pivot, 3x3) that amplifies this frame's thumb CMC swing, or None when there is
        nothing to amplify. Also records the swing in cmc_deviation_deg and captures the reference."""
        meta = local[11]
        bone = local[6] - meta
        length = np.linalg.norm(bone)
        if length < LENGTH_EPS_M:
            return None
        direction = bone/length
        if self.cmc_reference is None:
            self.cmc_reference = direction
            self.events.append(dict(outcome='thumb_cmc_reference', direction=direction.tolist()))
            print('Thumb CMC reference captured (thumb relaxed); press A to retake it')
        axis = np.cross(self.cmc_reference, direction)
        sine, cosine = float(np.linalg.norm(axis)), float(np.clip(self.cmc_reference @ direction, -1., 1.))
        angle = np.arctan2(sine, cosine)
        self.cmc_deviation_deg = float(np.degrees(angle))
        if sine < 1e-9:
            return None
        gain, cap = self.adapter.thumb_cmc_amplify, self.adapter.thumb_cmc_amplify_max_deg
        target = gain*angle if cap is None else np.deg2rad(cap)*np.tanh(gain*angle/np.deg2rad(cap))
        return meta, Rotation.from_rotvec(axis/sine*(target-angle)).as_matrix()

    @staticmethod
    def apply_thumb_swing(points, swing, rows):
        """Rotate the given rows (thumb nodes beyond the CMC) about the pivot so the whole chain keeps its shape."""
        meta, extra = swing
        out = points.copy()
        for row in rows:
            out[row] = meta + extra @ (points[row]-meta)
        return out

    def thumb_node(self, joint):
        """Glove node id of a thumb joint ('Proximal' is a selected landmark; others come from the metadata)."""
        if joint == 'Proximal':
            return self.selected['local_ids'][6]
        ids = [n['id'] for n in self.metadata
               if n['side'] == self.side and n['chain'] == 'FingerThumb' and n['joint'] == joint]
        if len(ids) != 1:
            raise ValueError(f'Missing/ambiguous thumb {joint} node: {ids}')
        return ids[0]

    def thumb_orientation_targets(self, hand, position_vectors):
        """Targets for the orientation rows: each robot thumb link's axes rotated as the matching glove thumb bone
        has rotated since the reference (first live frame after each alignment; hold the thumb relaxed).

        R_glove(t) = C R_palm^T R_bone (root frame); target_k = R_glove(t) R_glove(ref)^T axis_k(ref)."""
        c = np.asarray(self.alignment['C'])
        r_glove = [c @ palm_rotation_of(hand, self.selected['palm'], self.thumb_node(joint))
                   for _, joint, _, _ in self.adapter.orientation_bones]
        if self.orientation_ref is None:
            self.orientation_ref = (r_glove, self.adapter.orientation_reference(position_vectors))
            self.events.append(dict(outcome='thumb_orientation_reference',
                                    bones=[key for key, _, _, _ in self.adapter.orientation_bones]))
            print('Thumb orientation reference captured (thumb relaxed); press A to retake it')
        r_ref, axes_ref = self.orientation_ref
        axes_ref = np.asarray(axes_ref)
        amplify = getattr(self.adapter, 'orientation_amplify', {})
        roll = getattr(self.adapter, 'orientation_roll', {})
        rows = []
        for b, ((key, _, _, _), r_now, r0) in enumerate(zip(self.adapter.orientation_bones, r_glove, r_ref)):
            relative = r_now @ r0.T  # the bone's rotation since the reference, root frame
            gain = amplify.get(key, 1.0)
            if gain != 1.0:  # same axis, gain x the angle (kept below a half turn so the axis cannot flip)
                rotvec = Rotation.from_matrix(relative).as_rotvec()
                angle = float(np.linalg.norm(rotvec))
                if angle > 1e-9:
                    relative = Rotation.from_rotvec(rotvec/angle*min(gain*angle, np.pi - 1e-6)).as_matrix()
            roll_deg = roll.get(key, 0.0)
            if roll_deg:
                axis = self.thumb_bone_axis(key, c)
                if axis is not None:
                    relative = Rotation.from_rotvec(axis*np.deg2rad(roll_deg)).as_matrix() @ relative
            rows.append((relative @ axes_ref[2*b:2*b+2].T).T)
        return np.vstack(rows)

    def thumb_bone_axis(self, bone, c):
        """Unit direction (root frame) of a glove thumb bone, base -> tip, from this frame's palm-frame positions:
        proximal = Proximal joint -> Distal joint, distal = Distal joint -> tip. None if unavailable."""
        xyz = self.geometry
        if xyz is None:
            return None
        try:
            if bone == 'distal':
                start, end = xyz[self.thumb_node('Distal')], xyz[self.selected['local_ids'][1]]
            else:
                start, end = xyz[self.thumb_node('Proximal')], xyz[self.thumb_node('Distal')]
        except (KeyError, ValueError):
            return None
        direction = np.asarray(c) @ (np.asarray(end, dtype=float) - np.asarray(start, dtype=float))
        length = np.linalg.norm(direction)
        return None if length < LENGTH_EPS_M else direction/length

    def thumb_tip_offset_shift(self, xyz, local, swing):
        """Palm-frame shift for the glove thumb tip (glove.thumb_tip_offset), computed on the thumb after any CMC
        amplify; kept in self.thumb_tip_shift for the viewer. Zero when disabled or the distal node is missing."""
        options = getattr(self.adapter, 'thumb_tip_offset', None) or {}
        self.thumb_tip_shift = np.zeros(3)
        if not options.get('enabled'):
            return self.thumb_tip_shift
        try:
            distal = np.asarray(xyz[self.thumb_node('Distal')], dtype=float)
        except (ValueError, KeyError):
            return self.thumb_tip_shift
        if swing is not None:
            distal = self.apply_thumb_swing(distal[None], swing, rows=(0,))[0]
        self.thumb_tip_shift = thumb_tip_shift(local[1], distal, local[12], options.get('toward_index_mm', 0.0),
                                               options.get('along_mm', 0.0), options.get('toward_index_deg', 0.0))
        return self.thumb_tip_shift

    def start_alignment(self, now):
        self.orientation_ref = None  # retaken on the first live frame after this alignment
        self.cmc_reference = None  # retaken on the first live frame after this alignment
        self.invalidate('Collecting alignment; hold still one second')
        self.collection = dict(started=now, samples=[], times=[])
        self.events.append(dict(outcome='start', started=now, capture_s=CAPTURE_S, min_frames=MIN_CAPTURE_FRAMES,
                                motion_m=ALIGN_MOTION_M, motion_deg=ALIGN_MOTION_DEG, scale=1.))

    def check_freshness(self, hand, now):
        if hand is None or not 0 <= now-hand['received'] < STALE_S:
            if self.keep_alignment_on_gap and self.alignment is not None:
                self.invalidate('Hand not tracked; waiting for it to reappear', clear_alignment=False)
            else:
                self.invalidate('Stale/missing skeleton; press A again')
            self.geometry = None
            return False
        if self.collection is not None and now-self.collection['started'] >= CAPTURE_S and hand['frame_id']==self.last_frame_id:
            self.finish_alignment()
        return True

    def finish_alignment(self):
        c = self.collection
        if len(c['samples']) < self.min_capture_frames:
            self.invalidate(f'Alignment needs at least {self.min_capture_frames} distinct frames; press A again')
            return
        samples = np.asarray(c['samples'])
        mean = samples.mean(axis=0)
        try:
            bg = basis(mean[:4], mean[4:])[0]
            rotations = np.array([basis(s[:4], s[4:])[0] for s in samples])
            angular = float(np.rad2deg(np.max(Rotation.from_matrix(bg.T @ rotations).magnitude())))
            motion = float(np.max(np.linalg.norm(samples-mean, axis=2)))
            if motion > ALIGN_MOTION_M or angular > ALIGN_MOTION_DEG:
                raise GeometryError(f'Moving alignment: {motion*1000:.2f} mm / {angular:.2f} deg')
            raw_slant = knuckle_slant_deg(mean[:4], bg[:, 1])
            tuning = getattr(self.adapter, 'tuning', None) or {}
            target_slant = (tuning.get('glove') or {}).get('knuckle_slant_deg')
            if target_slant is not None:
                bg = set_knuckle_slant(bg, mean[:4], target_slant)
            slant = knuckle_slant_deg(mean[:4], bg[:, 1])
            a = self.adapter.robot_basis @ bg.T
            c_root = self.adapter.palm_root[:3, :3] @ a
            if not np.allclose(c_root.T @ c_root, np.eye(3), atol=1e-9) or not np.isclose(np.linalg.det(c_root), 1.):
                raise GeometryError('Invalid alignment rotation')
        except GeometryError as error:
            self.invalidate(str(error)+'; press A again')
            return
        self.alignment = dict(A=a.tolist(), C=c_root.tolist(), scale=1., motion_m=motion, motion_deg=angular,
            source=dict(captured_unix_s=time.time(), alignment_event_monotonic_s=c['started'],
                        capture_frame_count=len(c['samples']), capture_span_s=c['times'][-1]-c['times'][0],
                        motion_m=motion, motion_deg=angular, knuckle_slant_deg=slant, knuckle_slant_raw_deg=raw_slant,
                        knuckle_slant_target_deg=target_slant))
        if target_slant is None:
            print(f'Aligned: knuckle line slant {slant:+.1f} deg (fingers straight along the forearm give about +14..+20; '
                  f'a much smaller value means the fingers were angled toward the pinky during the capture, which turns the '
                  f'whole glove hand about the palm normal; align both hands to within a few degrees of each other)')
        else:
            print(f'Aligned: glove hand turned about the palm normal by {raw_slant - slant:+.1f} deg so the knuckle line '
                  f'slant is {slant:+.1f} deg (captured {raw_slant:+.1f}; glove.knuckle_slant_deg {target_slant:+.1f})')
        self.events.append(dict(outcome='success', started=c['started'], count=len(c['samples']),
                                received_times=c['times'], span_s=c['times'][-1]-c['times'][0], **self.alignment))
        self.collection = None
        self.reason = 'Aligned; waiting for a fresh skeleton'

    def snapshot(self, now=None):
        return dict(schema=SCHEMA, method='vector', scale=1., reason=self.reason,
                    received=self.last_received, frame_id=self.last_frame_id,
                    state='collecting alignment' if self.collection is not None else 'live' if self.alignment is not None else 'awaiting alignment',
                    source_age_s=None if now is None or self.last_received is None else now-self.last_received,
                    progress=None if self.collection is None else dict(count=len(self.collection['samples']),
                        elapsed_s=0 if now is None else now-self.collection['started']),
                    alignment=self.alignment, result=self.result, processed_count=self.count, skipped_total=self.skipped)

    def process(self, hand, now):
        started = time.perf_counter()
        if not self.check_freshness(hand, now):
            return self.snapshot(now)
        if hand['frame_id'] == self.last_frame_id:
            return self.snapshot(now)
        interval = None if self.last_received is None else hand['received']-self.last_received
        skipped = 0 if self.last_frame_id is None else max(0, hand['frame_id']-self.last_frame_id-1)
        if interval is not None and (interval <= 0 or interval >= STALE_S or hand['frame_id'] < self.last_frame_id):
            if self.keep_alignment_on_gap and self.alignment is not None:
                self.invalidate('Hand tracking resumed', clear_alignment=False)
                self.reason = None
            else:
                self.invalidate('Discontinuous skeleton; press A again')
        self.last_frame_id, self.last_received = hand['frame_id'], hand['received']
        self.count += 1
        self.skipped += skipped
        self.result = None
        if {n['id'] for n in hand['nodes']} != {n['id'] for n in self.metadata}:
            self.invalidate('Skeleton differs from metadata; press A again')
            self.geometry = None
        else:
            try:
                _, self.geometry = palm_positions(hand, self.selected['palm'])
            except ValueError as error:
                self.invalidate('Invalid palm/positions: '+str(error))
                self.geometry = None
            else:
                xyz = self.geometry
                if self.alignment is None and self.collection is None and self.reason != ALIGN_PROMPT:
                    # A valid frame clears an earlier frame's error; without an alignment the blocker is now A.
                    self.reason = ALIGN_PROMPT
                if self.collection is not None:
                    c = self.collection
                    if c['started'] <= hand['received'] <= c['started']+CAPTURE_S:
                        c['samples'].append(np.array([xyz[i] for i in self.selected['mcp']+self.selected['pip']]))
                        c['times'].append(hand['received'])
                    if now-c['started'] >= CAPTURE_S:
                        self.finish_alignment()
                if self.alignment is not None:
                    local = np.array([xyz[i] for i in self.selected['local_ids']])
                    amplify = self.adapter.thumb_cmc_amplify != 1.0
                    swing = self.thumb_cmc_rotation(local) if amplify else None
                    if swing is not None:
                        local = self.apply_thumb_swing(local, swing, rows=(6, 1))
                    shift = self.thumb_tip_offset_shift(xyz, local, swing)
                    local[1] = local[1] + shift  # row 1 = thumb tip
                    if self.adapter.method == 'dexpilot' and self.auto_scale and not self.scale_set:
                        self.apply_auto_scale(xyz)
                    human_origins, human_tasks = local[self.adapter.human_indices[0]], local[self.adapter.human_indices[1]]
                    vectors = (human_tasks-human_origins) @ np.asarray(self.alignment['C']).T
                    if self.adapter.orientation_bones:
                        targets = self.thumb_orientation_targets(hand, vectors)
                        vectors = np.vstack([vectors, targets/self.adapter.scale])  # the solver re-applies the scale
                    observe = getattr(self.adapter, 'observe_hand', None)  # GeoRT: the solve uses the raw skeleton
                    if observe is not None:
                        observe(hand, self.metadata)
                    preprocess_s = time.perf_counter()-started
                    self.result = self.adapter.solve(vectors, stamp=hand['received'])
                    self.result['human_origin_palm_m'] = human_origins.tolist()
                    self.result['human_task_palm_m'] = human_tasks.tolist()
                    self.result['preprocess_s'] = preprocess_s
                    if amplify:
                        self.result['thumb_cmc_deviation_deg'] = self.cmc_deviation_deg
                    self.reason = None
        sample = self.snapshot(now)
        sample.update(frame_id=hand['frame_id'], received=hand['received'], processed=now,
                      inter_frame_s=interval, skipped_frame_ids=skipped, math_s=time.perf_counter()-started)
        return sample
