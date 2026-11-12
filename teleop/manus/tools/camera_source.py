"""Webcam hand tracking as a drop-in replacement for the MANUS glove source (skeleton_source.Source).

MediaPipe's hand landmarker estimates 21 3D "world" landmarks per hand (metres, about the hand's centre) from one RGB
image. This module turns them into the same skeleton the glove delivers: a MANUS-style node hierarchy (metadata with
chain / joint names), node positions, and node rotations (quaternion xyzw), so the rest of the sender (alignment with
A / S, the D21 retargeter, the one-euro filter, the controller session) runs unchanged.

    hand/.hand/bin/python teleop/manus/stream/hand_stream_client.py --model aac --glove-side left --solver dexrt --source camera

Mapping (MediaPipe index -> MANUS node): wrist 0 -> Hand; thumb 1 CMC, 2 MCP, 3 IP, 4 tip -> Metacarpal, Proximal, Distal,
Tip; each finger MCP, PIP, DIP, tip -> Proximal, Intermediate, Distal, Tip, with its Metacarpal node synthesised a
quarter of the way from the wrist to the MCP (the glove has one; MediaPipe does not). Rotations: each node's frame has
its x axis along its bone (node -> child), its y axis from the palm normal; the palm (Hand) node uses wrist -> middle MCP
and the index -> pinky knuckle line. Only rotation CHANGES are used downstream (alignment, thumb orientation reference),
so any consistent frame convention works; the geometry must not be mirrored, which is why the image is NOT flipped.

Handedness (checked against a MANUS left-hand log, 2026-11-03): with an ordinary, unmirrored webcam MediaPipe labels
the real hand ("Left" for the left hand) and its landmarks have the real hand's geometry, so both are used as they are.
A webcam that mirrors its image (--camera-mirror) shows a left hand as a right one: the label is swapped and the
landmarks are reflected back in x. Mirrored geometry is fatal (curl lands on the back of the hand, no alignment can fix it).
"""

import math
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]   # agi-os
CAMERA_GLOVE_ID = {'Left': 0xCA4E0001, 'Right': 0xCA4E0002}   # fixed "glove ids" so alignment files can name them
MODEL_CANDIDATES = (
    Path(__file__).resolve().parents[1] / 'hand_landmarker.task',          # teleop/manus/
    ROOT / 'teleop' / 'models' / 'hand_landmarker.task',
    ROOT.parent / 'agi-os' / 'teleop' / 'models' / 'hand_landmarker.task',
)
FINGERS = ('Index', 'Middle', 'Ring', 'Pinky')
METACARPAL_FRACTION = 0.25   # synthesised finger Metacarpal node: this far from the wrist toward the MCP
# The 21-point hand as a tree: (parent, child) in MediaPipe order; every finger chain hangs off the wrist.
BONE_TREE = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
             (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
GLOVE_HAND_LENGTHS = Path(__file__).resolve().parents[1] / 'alignment' / 'glove_hand_lengths.json'


def glove_bone_lengths(side, path=GLOVE_HAND_LENGTHS):
    """{(parent, child): metres} of the glove hand GeoRT was trained on (teleop/manus/alignment/glove_hand_lengths.json)."""
    import json
    record = json.loads(Path(path).read_text())[side.capitalize()]['bones_mm']
    lengths = {tuple(int(i) for i in key.split('-')): float(mm) / 1000.0 for key, mm in record.items()}
    missing = [bone for bone in BONE_TREE if bone not in lengths]
    if missing:
        raise ValueError(f'{path}: no length for bones {missing}')
    return lengths


def fit_bone_lengths(points21, lengths):
    """The hand with every bone resized to `lengths[(parent, child)]` along its measured direction, from the wrist
    out: joint angles are unchanged, only the hand's size and proportions become the reference hand's. A bone with
    no direction (parent and child coincide) keeps its measured position."""
    p = np.asarray(points21, dtype=float)
    out = p.copy()
    for parent, child in BONE_TREE:
        direction = _unit(p[child] - p[parent])
        if direction is None:
            out[child] = out[parent] + (p[child] - p[parent])
        else:
            out[child] = out[parent] + direction * lengths[(parent, child)]
    return out


def node_layout():
    """[(node id, parent id, chain, joint, mediapipe index or ('meta', finger mcp index))] in MANUS order."""
    layout = [(0, 0, 'Hand', 'Invalid', 0)]
    layout += [(1, 0, 'FingerThumb', 'Metacarpal', 1), (2, 1, 'FingerThumb', 'Proximal', 2),
               (3, 2, 'FingerThumb', 'Distal', 3), (4, 3, 'FingerThumb', 'Tip', 4)]
    for k, finger in enumerate(FINGERS):
        base, mcp = 5 + 5 * k, 5 + 4 * k
        chain = 'Finger' + finger
        layout += [(base, 0, chain, 'Metacarpal', ('meta', mcp)), (base + 1, base, chain, 'Proximal', mcp),
                   (base + 2, base + 1, chain, 'Intermediate', mcp + 1), (base + 3, base + 2, chain, 'Distal', mcp + 2),
                   (base + 4, base + 3, chain, 'Tip', mcp + 3)]
    return layout


LAYOUT = node_layout()
CHILD = {}
for _id, _parent, _chain, _joint, _src in LAYOUT:
    if _id != _parent and _chain != 'Hand':
        CHILD.setdefault(_parent, _id)


def metadata(side):
    """MANUS-style node metadata for one side (what select_landmarks reads)."""
    return [dict(id=i, parent=p, side=side, chain=c, joint=j) for i, p, c, j, _ in LAYOUT]


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else None


def _frame(x_axis, up_hint):
    """Right-handed rotation matrix with columns (x, y, z), x = x_axis, y as close to up_hint x x as possible."""
    x = _unit(np.asarray(x_axis, dtype=float))
    if x is None:
        return np.eye(3)
    y = _unit(np.cross(up_hint, x))
    if y is None:                                        # bone parallel to the hint: any perpendicular
        y = _unit(np.cross([0.0, 0.0, 1.0], x)) if abs(x[2]) < 0.9 else _unit(np.cross([0.0, 1.0, 0.0], x))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def quaternion_xyzw(r):
    """Rotation matrix -> quaternion (x, y, z, w)."""
    m = np.asarray(r, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    return (float(x), float(y), float(z), float(w))


def skeleton_from_landmarks(points21, glove_id, frame_id, received):
    """21 MediaPipe world landmarks (metres) -> a glove-style hand dict {glove_id, nodes, frame_id, received}."""
    p = np.asarray(points21, dtype=float)
    if p.shape != (21, 3) or not np.isfinite(p).all():
        raise ValueError('expected 21 finite 3D landmarks')
    pos = {}
    for i, _, _, _, src in LAYOUT:
        if isinstance(src, tuple):
            pos[i] = p[0] + METACARPAL_FRACTION * (p[src[1]] - p[0])
        else:
            pos[i] = p[src]
    wrist, index_mcp, middle_mcp, pinky_mcp = p[0], p[5], p[9], p[17]
    normal = _unit(np.cross(index_mcp - wrist, pinky_mcp - wrist))
    normal = normal if normal is not None else np.array([0.0, 0.0, 1.0])
    palm = _frame(middle_mcp - wrist, normal)
    # palm: x toward the fingers, y along the knuckle line (index -> pinky), z = x x y
    across = _unit((pinky_mcp - index_mcp) - np.dot(pinky_mcp - index_mcp, palm[:, 0]) * palm[:, 0])
    if across is not None:
        palm = np.column_stack([palm[:, 0], across, np.cross(palm[:, 0], across)])
    rot = {0: palm}
    for i, parent, _, joint, _ in LAYOUT:
        if i == 0:
            continue
        child = CHILD.get(i)
        bone = pos[child] - pos[i] if child is not None else pos[i] - pos[parent]
        rot[i] = _frame(bone, normal)
    nodes = [dict(id=i, position=tuple(float(v) for v in pos[i]), rotation=quaternion_xyzw(rot[i])) for i, *_ in LAYOUT]
    return dict(glove_id=glove_id, nodes=nodes, frame_id=frame_id, received=received)


FINGER_CHAINS = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]   # MCP, PIP, DIP, tip
PALM_POINTS = (0, 1, 5, 9, 13, 17)        # wrist, thumb CMC, the four knuckles: rigid in a real hand


class PalmStabilizer:
    """Keep the palm rigid and its orientation steady.

    The palm frame came from three noisy points (wrist, index and pinky knuckles), so the knuckles moved 0.67 mm per
    frame (p95 3.3 mm, ~0.4 deg of frame wobble) relative to it, against 0.00 with the glove (logs 22 vs 14): the whole
    hand tilted in the viewer and every finger vector shook with it. Here the frame uses all four knuckles (forward =
    wrist -> their mean, across = their line), its orientation is low-passed (tau_frame), and the palm points are
    replaced by a slowly averaged rigid template in that frame (tau_shape); each finger chain moves with its knuckle,
    the thumb with its CMC."""

    # Defaults after "stabilise it more, I don't want it to move": the frame holds still for any tilt within
    # deadband_deg of where it is and then follows only the excess, smoothed with tau_frame.
    def __init__(self, tau_frame=0.15, tau_shape=2.0, deadband_deg=4.0):
        self.tau_frame, self.tau_shape = tau_frame, tau_shape
        self.deadband = math.radians(deadband_deg)
        self.frame = None
        self.template = None
        self.last = None

    @staticmethod
    def raw_frame(p):
        mcp = p[[5, 9, 13, 17]]
        fwd = _unit(mcp.mean(axis=0) - p[0])
        across = (mcp[3] - mcp[0]) + (mcp[2] - mcp[1])     # index -> pinky, two pairs averaged
        if fwd is None:
            return None
        y = _unit(across - np.dot(across, fwd) * fwd)
        if y is None:
            return None
        return np.column_stack([fwd, y, np.cross(fwd, y)])

    def __call__(self, points21, now):
        p = np.array(points21, dtype=float)
        r = self.raw_frame(p)
        if r is None:
            return p
        dt = None if self.last is None else now - self.last
        self.last = now
        if self.frame is None or dt is None or dt <= 0 or dt > 0.5:
            self.frame, self.template = r, None                  # start over after a tracking gap
        else:
            # rotation from the held frame to the measured one, as an axis-angle
            rel = self.frame.T @ r
            angle = math.acos(max(-1.0, min(1.0, (np.trace(rel) - 1.0) / 2.0)))
            if angle > self.deadband:
                # move toward the measurement by the part beyond the deadband, smoothed
                a = (1.0 - math.exp(-dt / self.tau_frame)) * (angle - self.deadband) / angle
                axis = np.array([rel[2, 1] - rel[1, 2], rel[0, 2] - rel[2, 0], rel[1, 0] - rel[0, 1]])
                axis = _unit(axis) if angle < math.pi - 1e-6 else None
                if axis is not None:
                    step = a * angle
                    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
                    delta = np.eye(3) + math.sin(step) * k + (1 - math.cos(step)) * (k @ k)
                    self.frame = self.frame @ delta
        f = self.frame
        local = (p[list(PALM_POINTS)] - p[0]) @ f              # palm points in the smoothed palm frame
        if self.template is None:
            self.template = local
        else:
            b = 1.0 - math.exp(-dt / self.tau_shape)
            self.template = self.template + b * (local - self.template)
        out = p.copy()
        placed = {i: p[0] + f @ self.template[k] for k, i in enumerate(PALM_POINTS)}
        for i, q in placed.items():
            out[i] = q
        for chain in FINGER_CHAINS:                             # fingers follow their knuckle
            shift = placed[chain[0]] - p[chain[0]]
            out[list(chain[1:])] = p[list(chain[1:])] + shift
        shift = placed[1] - p[1]                                # thumb follows its CMC
        out[[2, 3, 4]] = p[[2, 3, 4]] + shift
        return out


class DepthFix:
    """Restore finger curl that a single camera loses in depth.

    With the palm toward the camera, curling moves the fingertips toward the lens: MediaPipe shortens the finger bones
    but barely moves them out of the palm plane (log manus_torque_19: middle MCP->tip 101 -> 31 mm, correlation of the
    palm-normal component with the curl only 0.09). Bones have a fixed length, so each finger segment is rebuilt with
    its reference length: the in-plane part as measured, the missing part along the palm normal toward the palm side
    (fingers flex toward the palm). The reference is the longest length seen recently (decaying slowly, so it adapts if
    the estimate's scale drifts). The thumb is left as measured: it tracks well and moves out of the palm plane."""

    # Foreshortening below START is treated as length noise; the correction ramps in fully by START + RAMP. Without this,
    # sqrt(ref^2 - visible^2) turned 1-2 % length noise on a straight finger into 15-20 % of the bone length of
    # out-of-plane motion: straight fingers shook in the viewer though the 2D camera points were steady.
    START, RAMP = 0.04, 0.08

    def __init__(self, side, decay=0.9995):
        # palm side of (index MCP - wrist) x (pinky MCP - wrist): -n for a left hand, +n for a right hand (the MANUS
        # left-hand logs bend against n in every frame)
        self.palm_sign = -1.0 if side == 'Left' else 1.0
        self.decay = decay
        self.ref = {}

    def palm_normal(self, p):
        n = _unit(np.cross(p[5] - p[0], p[17] - p[0]))
        return None if n is None else self.palm_sign * n

    def __call__(self, points21):
        p = np.array(points21, dtype=float)
        n = self.palm_normal(p)
        if n is None:
            return p
        out = p.copy()
        # palm axes: forward (wrist -> middle MCP, in the palm plane) and across (the knuckle line)
        fwd = _unit((p[9] - p[0]) - np.dot(p[9] - p[0], n) * n)
        across = None if fwd is None else np.cross(n, fwd)
        for chain in FINGER_CHAINS:
            hinge = None                       # the finger's bending plane: spanned by its in-plane heading and n
            for k, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
                v = p[b] - p[a]
                length = float(np.linalg.norm(v))
                key = (a, b)
                ref = max(length, self.ref.get(key, 0.0) * self.decay)
                self.ref[key] = ref
                along = float(np.dot(v, n))
                in_plane = v - along * n
                if k == 0 and across is not None and ref > 0:
                    # Abduction is only observable from the part of the bone still in the palm plane: scale the
                    # sideways component by that fraction (cos of the flexion), so a finger curled toward the lens
                    # no longer reports large, noisy spread (log manus_torque_21: abduction swung 30-40 deg with curl).
                    side = float(np.dot(in_plane, across))
                    fraction = min(1.0, float(np.linalg.norm(in_plane)) / ref)
                    in_plane = in_plane - side * (1.0 - fraction) * across
                    heading = _unit(in_plane)
                    hinge = None if heading is None else _unit(np.cross(heading, n))   # normal of the bending plane
                elif hinge is not None:
                    # PIP and DIP are hinges: the rest of the finger stays in the plane of the first bone
                    in_plane = in_plane - np.dot(in_plane, hinge) * hinge
                in_len = float(np.linalg.norm(in_plane))
                shortening = 1.0 - in_len / ref if ref > 0 else 0.0
                weight = min(1.0, max(0.0, (shortening - self.START) / self.RAMP))
                missing = weight * math.sqrt(max(ref * ref - in_len * in_len, 0.0))
                # out of the plane toward the palm; below the threshold keep the measured component
                if weight == 0.0:
                    normal = along
                else:
                    normal = missing if along > -missing else along
                out[b] = out[a] + in_plane + normal * n
        return out


def find_model(path=None):
    candidates = [Path(path)] if path else list(MODEL_CANDIDATES)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError('MediaPipe hand_landmarker.task not found; looked in: '
                            + ', '.join(str(c) for c in candidates) + '. Pass --camera-model PATH.')


class _NoSdk:
    """Stands in for the MANUS SDK calls the sender loop makes on its source (initialize / shut down)."""

    class lib:
        @staticmethod
        def CoreSdk_InitializeIntegrated():
            return 0

        @staticmethod
        def CoreSdk_ShutDown():
            return 0

    @staticmethod
    def check(code, label):
        return code


FLAT_POSE_FILE = Path(__file__).resolve().parents[1] / 'alignment' / 'camera_flat_{side}.json'
FLAT_FINGERS = dict(index=(5, 6, 7, 8), middle=(9, 10, 11, 12), ring=(13, 14, 15, 16), pinky=(17, 18, 19, 20))
FLAT_CAPTURE_FRAMES = 15      # ~0.5 s at 30 fps
FLAT_CAPTURE_TIMEOUT_S = 3.0  # no hand seen for this long: the capture is dropped
FLAT_MAX_BEND_DEG = 45.0      # a larger bend in the captured pose: the hand was not flat, capture refused
FLAT_FULL_BEND_DEG = 90.0     # from here on a joint's bend is passed through unchanged


class FlatPose:
    """Removes the bend MediaPipe reports for a FLAT hand, per finger joint.

    The glove skeleton GeoRT was trained on has the fingers of a flat hand straight along the wrist -> middle-knuckle
    axis (fingertips within ~1 cm of that plane); MediaPipe's world hand has them bent toward the palm at every
    joint (fingertips 2-4 cm palm-side, more for some hands), which GeoRT reads as curled fingers. One capture of the
    user's flat hand gives each joint's flexion bend there (MCP against that axis, PIP, DIP; degrees, in the plane
    across the knuckle line); afterwards a joint's measured bend m becomes (m - flat) * 90 / (90 - flat) below 90 deg
    (flat -> 0, 90 -> 90, so the full curl is kept) and m itself above. Bone lengths and the sideways (abduction)
    part of every bone are unchanged; the thumb and the palm are left as measured."""

    def __init__(self, side, bends=None):
        self.side = side.capitalize()
        self.sign = 1.0 if self.side == 'Right' else -1.0   # canonical x is palm-side on a right hand, back-side on a left
        self.bends = {name: [float(v) for v in (bends or {}).get(name, (0.0, 0.0, 0.0))] for name in FLAT_FINGERS}

    @property
    def active(self):
        return any(abs(v) > 1e-9 for values in self.bends.values() for v in values)

    @staticmethod
    def _frame(p):
        """GeoRT's canonical hand axes as columns (geort_record.hand_to_canonical), or None for a degenerate hand."""
        z, y_aux = _unit(p[9] - p[0]), _unit(p[5] - p[13])
        if z is None or y_aux is None:
            return None
        x = _unit(np.cross(y_aux, z))
        if x is None:
            return None
        return np.column_stack([x, np.cross(z, x), z])

    def _bends(self, local, chain):
        """Flexion bend (deg, + toward the palm) at the finger's MCP, PIP and DIP: the turn about the knuckle line
        (canonical y) from the previous bone, the first from the wrist -> middle-knuckle axis."""
        out, previous = [], np.array([0.0, 1.0])            # (x toward the palm, z) of the reference direction
        for a, b in zip(chain[:-1], chain[1:]):
            d = local[b] - local[a]
            v = np.array([self.sign * d[0], d[2]])
            if np.linalg.norm(v) < 1e-9:
                out.append(0.0)
                continue
            out.append(float(np.degrees(np.arctan2(previous[1] * v[0] - previous[0] * v[1], previous @ v))))
            previous = v
        return out

    def measure(self, points21):
        """{finger: [MCP, PIP, DIP bend in deg]} of this frame, or None for a degenerate hand."""
        p = np.asarray(points21, dtype=float)
        frame = self._frame(p)
        if frame is None:
            return None
        local = (p - p[0]) @ frame
        return {name: self._bends(local, chain) for name, chain in FLAT_FINGERS.items()}

    @staticmethod
    def corrected(bend, flat):
        if bend >= FLAT_FULL_BEND_DEG or flat >= FLAT_FULL_BEND_DEG:
            return bend
        if bend <= flat:
            return bend - flat
        return (bend - flat) * FLAT_FULL_BEND_DEG / (FLAT_FULL_BEND_DEG - flat)

    def __call__(self, points21):
        if not self.active:
            return points21
        p = np.asarray(points21, dtype=float)
        frame = self._frame(p)
        if frame is None:
            return points21
        local = (p - p[0]) @ frame
        out = local.copy()
        for name, chain in FLAT_FINGERS.items():
            turn = 0.0                                       # deg taken out of the flexion so far along the chain
            for (a, b), bend, flat in zip(zip(chain[:-1], chain[1:]), self._bends(local, chain), self.bends[name]):
                turn += bend - self.corrected(bend, flat)
                angle = np.radians(self.sign * turn)         # about canonical y: lowers the palm-side pitch by `turn`
                cos, sin = np.cos(angle), np.sin(angle)
                d = local[b] - local[a]
                out[b] = out[a] + np.array([d[0] * cos - d[2] * sin, d[1], d[0] * sin + d[2] * cos])
        return p[0] + out @ frame.T

    @classmethod
    def load(cls, side, path=None):
        """The saved flat pose of this side (teleop/manus/alignment/camera_flat_<side>.json), or no correction."""
        import json
        path = Path(path or str(FLAT_POSE_FILE).format(side=side.lower()))
        if not path.exists():
            return cls(side)
        return cls(side, json.loads(path.read_text())['bends_deg'])

    def save(self, path=None, frames=0):
        import json
        path = Path(path or str(FLAT_POSE_FILE).format(side=self.side.lower()))
        record = dict(_about='Bend (deg, toward the palm) MediaPipe reported at each finger joint [MCP, PIP, DIP] while '
                             'the hand was held flat: teleop/manus/tools/camera_source.py FlatPose removes it. Captured '
                             'by Align in the GUI camera mode; delete the file for no correction.',
                      side=self.side, frames=frames, measured=time.strftime('%Y-%m-%d %H:%M:%S'),
                      bends_deg={name: [round(v, 2) for v in values] for name, values in self.bends.items()})
        path.write_text(json.dumps(record, indent=1) + '\n')
        return path


class CameraSource:
    """Same interface as skeleton_source.Source: connect(), consume() -> (frames, devices), metadata(glove_id)."""

    def __init__(self, sides, camera_index=None, width=640, height=480, fps=30.0, model=None, mirror=False,
                 any_hand=False, preview=False, min_confidence=0.5, depth_fix=False, palm=None, fit_to_glove=True,
                 flat_pose=True):
        self.backend = _NoSdk()
        self.keeps_alignment = True     # the sender keeps the alignment across tracking gaps for this source
        self.sides = list(sides)
        self.camera_index, self.width, self.height, self.fps = camera_index, width, height, fps
        self.model = find_model(model)
        # One side tracked: take the most confident detected hand whatever MediaPipe's label says (labels depend on
        # whether the webcam mirrors its image, and a missed label meant no frames at all: log manus_torque_18).
        # Two sides need the labels to tell the hands apart.
        self.any_hand = any_hand or len(self.sides) == 1
        self.mirror, self.preview = mirror, preview
        self.label_warned = False
        # Off by default since 2026-11-06: on a recording of known poses (palm toward the camera) it made the joint
        # angles worse than MediaPipe's own 3D (mean error 31 vs 25 deg): it pushes every finger bone out of the palm,
        # so in a fist or hook the curled middle/end bones were flipped straight (fist DIP 87 -> 13 deg). It had been
        # tuned on logs recorded with the mirrored geometry. Opt in with --camera-depth-fix.
        self.depth_fix = {s: DepthFix(s) for s in self.sides} if depth_fix else None
        self.palm = {s: PalmStabilizer(**(palm or {})) for s in self.sides}
        # GeoRT maps fingertip POSITIONS and was trained on the glove's hand; MediaPipe's world hand is smaller, so a
        # flat hand's fingertips sat where the glove's are with the fingers curled. Every bone is resized to the
        # glove hand's length along its measured direction (as the WiLoR source does); --camera-no-hand-fit turns it off.
        self.bone_lengths = {s: glove_bone_lengths(s) for s in self.sides} if fit_to_glove else None
        # The user's flat hand as MediaPipe sees it (FlatPose), captured by capture_flat(); --camera-no-flat-pose: off
        self.flat = {s: FlatPose.load(s) for s in self.sides} if flat_pose else None
        self.flat_capture = {}          # side -> dict(samples, deadline) while a capture runs
        self.flat_state = {}            # side -> 'capturing' | 'saved' | 'failed: why'
        self.min_confidence = min_confidence
        self.lock = threading.Lock()
        self.latest = {}
        self.error = None
        self.frame_id = 0
        self.stop = threading.Event()
        self.thread = None
        self.stats = dict(frames=0, detections=0)

    # -- interface used by preview_dex_retargeting.run --
    def connect(self):
        import cv2
        from teleop.manus.tools.camera_device import open_camera
        # camera_index None: AAC_CAMERA_INDEX, else the first camera that gives a colour image (camera_device.py)
        self.capture, self.camera_index = open_camera(self.camera_index, self.width, self.height, self.fps)
        from mediapipe.tasks.python import BaseOptions, vision
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model)), running_mode=vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence, min_tracking_confidence=self.min_confidence)
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        print(f'camera source: /dev/video{self.camera_index} {self.width}x{self.height}, model {self.model}; '
              f'hands: {", ".join(self.sides)}{" (any detected hand)" if self.any_hand else ""}')
        self.thread = threading.Thread(target=self._loop, name='camera-hand-tracker', daemon=True)
        self.thread.start()

    def consume(self):
        with self.lock:
            frames, self.latest = self.latest, {}
            error = self.error
        if error is not None:
            raise error
        devices = {CAMERA_GLOVE_ID[s]: dict(side=s, family='Camera', device_class='Camera') for s in self.sides}
        return frames, devices

    def capture_flat(self, side):
        """Measure this side's FLAT hand over the next frames (hold it flat and still) and remove that bend from then
        on; saved to teleop/manus/alignment/camera_flat_<side>.json. False when the correction is off."""
        if self.flat is None or side not in self.flat:
            return False
        with self.lock:
            self.flat_capture[side] = dict(samples=[], deadline=time.perf_counter() + FLAT_CAPTURE_TIMEOUT_S)
            self.flat_state[side] = 'capturing'
        return True

    def flat_status(self, side):
        """None (never captured in this run), 'capturing', 'saved', or 'failed: <why>'."""
        with self.lock:
            return self.flat_state.get(side)

    def _flat(self, side, points, now):
        """One tracked frame through the flat-pose capture (if one runs) and the correction."""
        with self.lock:
            capture = self.flat_capture.get(side)
        if capture is not None:
            sample = self.flat[side].measure(points)
            if sample is not None:
                capture['samples'].append(sample)
            state = None
            if len(capture['samples']) >= FLAT_CAPTURE_FRAMES:
                bends = {name: np.median([s[name] for s in capture['samples']], axis=0).tolist() for name in FLAT_FINGERS}
                worst = max(abs(v) for values in bends.values() for v in values)
                if worst > FLAT_MAX_BEND_DEG:
                    state = f'failed: the hand was not flat (a joint bent {worst:.0f} deg)'
                else:
                    self.flat[side] = FlatPose(side, bends)
                    path = self.flat[side].save(frames=len(capture['samples']))
                    print(f'camera source: {side} flat hand saved to {path.name}: '
                          + ', '.join(f"{n} {'/'.join(f'{v:.0f}' for v in b)}" for n, b in bends.items()) + ' deg')
                    state = 'saved'
            elif now > capture['deadline']:
                state = 'failed: no hand seen'
            if state is not None:
                with self.lock:
                    self.flat_capture.pop(side, None)
                    self.flat_state[side] = state
        return self.flat[side](points)

    def metadata(self, glove_id):
        side = next(s for s, gid in CAMERA_GLOVE_ID.items() if gid == glove_id)
        return metadata(side)

    def close(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        landmarker = getattr(self, 'landmarker', None)
        if landmarker is not None:
            landmarker.close()
            self.landmarker = None
        capture = getattr(self, 'capture', None)
        if capture is not None:
            capture.release()
        if self.preview:
            import cv2
            cv2.destroyAllWindows()

    # -- worker --
    def side_of(self, label):
        """True hand side from MediaPipe's label (the image's hand; swapped for a mirroring camera)."""
        if not self.mirror:
            return label
        return 'Left' if label == 'Right' else 'Right'

    def _loop(self):
        import cv2
        import mediapipe as mp
        try:
            t0 = time.perf_counter()
            while not self.stop.is_set():
                ok, bgr = self.capture.read()
                received = time.perf_counter()
                if not ok:
                    time.sleep(0.005)
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = self.landmarker.detect_for_video(image, int((received - t0) * 1000))
                self.stats['frames'] += 1
                hands = self._hands(result)
                if hands:
                    self.stats['detections'] += 1
                    self.frame_id += 1
                    frame = {}
                    for side, points in hands.items():
                        gid = CAMERA_GLOVE_ID[side]
                        points = self.palm[side](points, received)
                        if self.depth_fix is not None:
                            points = self.depth_fix[side](points)
                        if self.bone_lengths is not None:
                            points = fit_bone_lengths(points, self.bone_lengths[side])   # the glove's hand size
                        if self.flat is not None:
                            points = self._flat(side, points, received)
                        frame[gid] = skeleton_from_landmarks(points, gid, self.frame_id, received)
                    with self.lock:
                        self.latest.update(frame)
                if self.preview:
                    self._draw(cv2, bgr, result)
        except Exception as error:   # surfaced on the next consume()
            with self.lock:
                self.error = error

    def _hands(self, result):
        """{side: 21x3 world landmarks} for the requested sides in this result."""
        found = {}
        detections = []
        for k, world in enumerate(result.hand_world_landmarks or []):
            category = result.handedness[k][0]
            # A mirroring camera shows a mirror-image hand, so MediaPipe's 3D landmarks are a reflection of the real
            # hand: flip x (the image's horizontal axis) back, or the curl lands on the back of the hand.
            sx = -1.0 if self.mirror else 1.0
            detections.append((float(category.score), self.side_of(category.category_name),
                               [(sx * lm.x, lm.y, lm.z) for lm in world]))
        if self.any_hand and len(self.sides) == 1 and detections:
            score, label_side, points = max(detections, key=lambda d: d[0])
            if label_side != self.sides[0] and not self.label_warned:
                self.label_warned = True
                print(f'camera source: the tracked hand is labelled {label_side} but the '
                      f'sender drives the {self.sides[0]} hand; using it anyway. If it really is your '
                      f'{label_side.lower()} hand, the D21 will move wrongly: show the other hand, or if this webcam '
                      f'mirrors its image, run with --camera-mirror.')
            return {self.sides[0]: points}
        for score, side, points in sorted(detections, key=lambda d: -d[0]):
            if side in self.sides and side not in found:
                found[side] = points
        return found

    def _draw(self, cv2, bgr, result):
        h, w = bgr.shape[:2]
        for k, lms in enumerate(result.hand_landmarks or []):
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
            for a, b in ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
                         (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)):
                cv2.line(bgr, pts[a], pts[b], (0, 200, 255), 2)
            label = self.side_of(result.handedness[k][0].category_name)
            cv2.putText(bgr, label, pts[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('camera hand tracking', bgr if self.mirror else cv2.flip(bgr, 1))   # always shown like a mirror
        cv2.waitKey(1)


def load_camera_source(args, sides):
    return CameraSource(sides, camera_index=getattr(args, 'camera_index', None), width=getattr(args, 'camera_width', 640),
                        height=getattr(args, 'camera_height', 480), fps=getattr(args, 'camera_fps', 30.0),
                        model=getattr(args, 'camera_model', None), mirror=getattr(args, 'camera_mirror', False),
                        any_hand=getattr(args, 'camera_any_hand', False), preview=getattr(args, 'camera_preview', False),
                        depth_fix=getattr(args, 'camera_depth_fix', False) and not getattr(args, 'camera_no_depth_fix', False),
                        fit_to_glove=not getattr(args, 'camera_no_hand_fit', False),
                        flat_pose=not getattr(args, 'camera_no_flat_pose', False),
                        palm=dict(tau_frame=getattr(args, 'camera_palm_tau', 0.15),
                                  deadband_deg=getattr(args, 'camera_palm_deadband', 4.0)))
