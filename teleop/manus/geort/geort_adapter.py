"""GeoRT learned retargeting as a sender solver (--solver geort).

GeoRT (retargetor/GeoRT, set up by ./setup_envs.sh geort) trains one small MLP per finger that maps the human fingertip position, in its canonical
hand frame, straight to that finger's joint angles. Training data come from teleop/manus/geort/geort_record.py, so this
module builds the network input the same way (MANUS raw skeleton -> 21 MediaPipe-order keypoints ->
geort_record.hand_to_canonical) and loads the checkpoint without importing the GeoRT package (no SAPIEN needed):

    hand/.hand/bin/python teleop/manus/stream/hand_stream_client.py --model aac --glove-side left --solver geort ...
    # default checkpoint: retargetor/GeoRT_checkpoint/<side>_hand/, the run teleop/manus/geort/checkpoints.py picks for the side
    # and source (pinned, else the newest aac_d21_<side>_*), the same one the GUI uses
    # --geort-checkpoint DIR picks another training run, --geort-epoch N one epoch of it (epoch_N.pth)

The output is URDF radians in the SDK model order (GeoRT config joint_order == hand.aac.model.MODEL_JOINTS), clipped to
the adapter's effective bounds (URDF n SDK n coupling); the coupled DIP/IP joints follow as usual. Everything after the
solve is shared with the other solvers: One-Euro filter, thumb CMC gain, the viewer and its residuals (the vector
targets are still computed from the saved alignment, for display and logging only), joint_map(side) to the controller.
The glove-side tweaks that act on the vectors (row_scales, thumb offset, thumb CMC amplify, thumb tip offset, thumb
rotation lock, thumb orientation rows) do not apply: GeoRT sees the skeleton as recorded.
"""
import json
from pathlib import Path

import numpy as np

from hand.aac.model import MODEL_JOINTS
from teleop.manus.tools.dex_vector_retargeting import DexAdapter, checked_bounds
from teleop.manus.geort.geort_record import frame_keypoints, mediapipe_node_ids
from teleop.manus.geort.checkpoints import checkpoint_for


def default_checkpoint(side, source='glove'):
    """retargetor/GeoRT_checkpoint/: the pinned run for (source, side), else the newest (checkpoints.py)."""
    return checkpoint_for(side, source)


def _ik_net(n_joint, hidden=128):
    """geort/model.py get_finger_ik, unchanged (the checkpoint's state_dict must load strictly)."""
    from torch import nn
    return nn.Sequential(nn.Linear(3, hidden), nn.LeakyReLU(), nn.BatchNorm1d(hidden),
                         nn.Linear(hidden, hidden), nn.LeakyReLU(), nn.BatchNorm1d(hidden),
                         nn.Linear(hidden, n_joint), nn.Tanh())


class GeortIK:
    """A trained GeoRT IK model: 21 canonical keypoints (m) -> 16 joint angles (rad, config joint_order)."""

    def __init__(self, checkpoint, epoch=None, side=None):
        import torch
        from torch import nn
        self.torch = torch
        path = Path(checkpoint).expanduser()
        directory = path.parent if path.is_file() else path
        self.config = json.loads((directory / 'config.json').read_text())
        if side is not None and self.config.get('name') != f'aac_d21_{side.lower()}':
            raise ValueError(f"GeoRT checkpoint {directory} is for {self.config.get('name')!r}, "
                             f'not the {side} D21 (aac_d21_{side.lower()})')
        order = self.config['joint_order']
        if order != list(MODEL_JOINTS):
            raise ValueError(f'GeoRT joint_order {order} is not the SDK model order {MODEL_JOINTS}')
        fingers = self.config['fingertip_link']
        self.human_ids = [f['human_hand_id'] for f in fingers]
        self.finger_names = [f['name'] for f in fingers]
        self.keypoint_joints = [[order.index(j) for j in f['joint']] for f in fingers]
        self.lower = np.asarray(self.config['joint']['lower'], dtype=float)
        self.upper = np.asarray(self.config['joint']['upper'], dtype=float)
        self.weights = path if path.is_file() else directory / ('last.pth' if epoch is None else f'epoch_{epoch}.pth')
        if not self.weights.exists():
            raise FileNotFoundError(f'GeoRT weights not found: {self.weights}')

        class Model(nn.Module):
            def __init__(self, joints):
                super().__init__()
                self.nets = nn.ModuleList([_ik_net(len(j)) for j in joints])

        self.model = Model(self.keypoint_joints)
        self.model.load_state_dict(torch.load(self.weights, map_location='cpu', weights_only=True))
        self.model.eval()
        self.directory = directory

    def __call__(self, keypoints):
        """keypoints (21, 3) canonical frame -> (16,) radians, joint_order (= MODEL_JOINTS)."""
        points = np.asarray(keypoints, dtype=np.float32)[self.human_ids]
        torch = self.torch
        out = np.zeros(len(self.lower))
        with torch.no_grad():
            for net, joints, point in zip(self.model.nets, self.keypoint_joints, points):
                out[joints] = net(torch.from_numpy(point).reshape(1, 3)).numpy()[0]
        return (out / 2 + 0.5) * (self.upper - self.lower) + self.lower   # geort/formatter.py unnormalize


class GeortAdapter(DexAdapter):
    """DexAdapter whose solve step is the GeoRT network on this frame's keypoints (observe_hand) instead of the
    vector optimizer. The vectors still pass through solve() so the residuals and the viewer keep working."""

    def __init__(self, side='Left', tuning=None, checkpoint=None, epoch=None, source='glove'):
        super().__init__(side, tuning=tuning)
        self.method = 'geort'
        self.thumb_rotation_lock_deg = None   # a joint hold needs the dexrt QP; GeoRT outputs the angles directly
        self.ik = GeortIK(checkpoint or default_checkpoint(side, source), epoch=epoch, side=side)
        self.keypoints = None
        self._node_ids = self._node_metadata = None
        self.clipped = []

    def observe_hand(self, hand, metadata):
        """Called by the engine with each new skeleton frame, before solve()."""
        if metadata is not self._node_metadata:
            self._node_ids = mediapipe_node_ids(metadata, self.side)
            self._node_metadata = metadata
        self.keypoints = frame_keypoints(hand, self._node_ids)

    def warmup(self):
        self.ik(np.zeros((21, 3)))
        self.reset()

    def _retarget(self, vectors):
        if self.keypoints is None:
            raise RuntimeError('GeoRT needs the skeleton keypoints (observe_hand) before each solve')
        q = self.ik(self.keypoints)
        if not np.isfinite(q).all():
            raise ValueError('GeoRT produced non-finite joint angles')
        clipped = np.clip(q, self.bounds[:, 0], self.bounds[:, 1])
        self.clipped = [MODEL_JOINTS[i] for i in np.flatnonzero(np.abs(clipped - q) > 1e-9)]
        previous = self.sequence.last_qpos
        self.optimizer.attempt = dict(status=1, termination='GeoRT', evaluations=1, solver='geort',
                                      previous_rad=np.asarray(previous).tolist(), candidate_rad=q.tolist(),
                                      clipped=self.clipped)
        q = checked_bounds(clipped, self.bounds)
        self.sequence.last_qpos = q
        raw = self.full_qpos(q)
        return raw, raw

    def configuration(self):
        result = super().configuration()
        result['solver'] = dict(name='geort', checkpoint=str(self.ik.directory), weights=str(self.ik.weights),
                                fingers=self.ik.finger_names, human_ids=self.ik.human_ids,
                                joint_limits_rad=dict(lower=self.ik.lower.tolist(), upper=self.ik.upper.tolist()),
                                input='MANUS raw skeleton -> MediaPipe 21 keypoints -> GeoRT canonical frame')
        return result
