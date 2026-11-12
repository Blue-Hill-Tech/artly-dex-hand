"""WiLoR hand tracking on one image (runs in WiLoR's own environment, WiLoR/.venv, with CUDA torch).

WiLoR (WiLoR/, CVPR 2025) detects the hands with its YOLO detector, then fits the MANO hand model to each crop: a
hand with fixed bone lengths and anatomically valid joints, so finger curl toward the camera is not lost the way it
is in MediaPipe's 3D guess. MANO is a right hand: left hands are run mirrored and the result is mirrored back in x.

Output per hand: side ("Left"/"Right", from the detector), score, joints (21 x 3, metres, wrist at the origin, the
camera's axes: x right, y down, z away from the camera; real hand geometry, not mirrored), image (21 x 2 pixels).
The 21 joints are in OpenPose order, which is MediaPipe's: wrist, thumb CMC/MCP/IP/tip, then each finger
MCP/PIP/DIP/tip, so the rest of the pipeline (camera_source.skeleton_from_landmarks) takes them unchanged.

Needs WiLoR/pretrained_models/{detector.pt, wilor_final.ckpt} and WiLoR/mano_data/MANO_RIGHT.pkl (MANO licence). The
official download (mano_v1_2.zip) may simply be unzipped into WiLoR/MANO/: its models/MANO_RIGHT.pkl is linked into
mano_data/ on first use (link_mano).

WilorMiniHands: the same through WiLoR-mini (github.com/warmshao/WiLoR-mini @ ebec42f, installed --no-deps into
WiLoR/.venv so torch stays CUDA 2.14): a repackaged inference-only WiLoR with the same weights. It reads its files from
<WiLoR>/wilor_mini/pretrained_models, linked here to the WiLoR ones where those exist; whatever is missing (the
detector, the 2.6 GB checkpoint, MANO_RIGHT.pkl) it downloads itself from its author's Hugging Face page on first use.
That MANO_RIGHT.pkl is a third-party copy of the licensed MANO model (registration, non-commercial use); the download
is on by default at the user's request (2026-11-09), allow_mano_download=False turns it off.
"""
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np

WILOR_ROOT = Path(__file__).resolve().parents[3] / "WiLoR"   # <repo>/WiLoR


@contextmanager
def _inside(path):
    """WiLoR loads its MANO files by relative paths (./mano_data/...)."""
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


@contextmanager
def _full_torch_load(torch):
    """ultralytics 8.1 (WiLoR's pinned version) loads its YOLO file with torch.load's old default; torch >= 2.6 only
    unpickles plain tensors by default and refuses it. Allowed here only while WiLoR's own detector.pt (the official
    WiLoR release) is loaded."""
    original = torch.load

    def load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original(*args, **kwargs)
    torch.load = load
    try:
        yield
    finally:
        torch.load = original


# Where the official MANO download (mano_v1_2.zip from mano.is.tue.mpg.de) ends up when unzipped into WiLoR/: WiLoR
# itself reads WiLoR/mano_data/MANO_RIGHT.pkl, so a file found here is linked there (link_mano).
MANO_DOWNLOAD_PLACES = ("MANO/mano_v1_2/models", "MANO/models", "mano_v1_2/models", "MANO", "models")


def link_mano(root=WILOR_ROOT):
    """WiLoR/mano_data/MANO_RIGHT.pkl, linked from the unzipped official download when it is somewhere else in WiLoR/
    (MANO_DOWNLOAD_PLACES). Returns its path, or None when there is no MANO_RIGHT.pkl."""
    root = Path(root)
    target = root / "mano_data" / "MANO_RIGHT.pkl"
    if target.is_symlink() and not target.exists():
        target.unlink()                                  # the file it pointed to moved
    if target.exists():
        return target
    for place in MANO_DOWNLOAD_PLACES:
        found = root / place / "MANO_RIGHT.pkl"
        if found.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(found.resolve())
            print(f"wilor: using the MANO model {found} (linked as {target})", flush=True)
            return target
    return None


def missing_files(root=WILOR_ROOT):
    root = Path(root)
    link_mano(root)
    needed = [root / "pretrained_models" / "detector.pt", root / "pretrained_models" / "wilor_final.ckpt",
              root / "mano_data" / "MANO_RIGHT.pkl"]
    return [p for p in needed if not p.exists()]


def best_per_side(right, scores):
    """Indices of the best-scoring detection of each side (the model costs ~20 ms per hand; other people's hands are
    not wanted)."""
    return [max(np.flatnonzero(right == cls), key=lambda i: scores[i]) for cls in (0.0, 1.0) if np.any(right == cls)]


class WilorHands:
    def __init__(self, root=WILOR_ROOT, device=None, fast=True, detection_confidence=0.3, rescale_factor=2.0):
        import torch
        missing = missing_files(root)
        if missing:
            raise FileNotFoundError("WiLoR files missing: " + ", ".join(str(p) for p in missing))
        self.root = Path(root)
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fast = fast and self.device.type == "cuda"
        self.detection_confidence = detection_confidence
        self.rescale_factor = rescale_factor
        import sys
        if str(WILOR_ROOT) not in sys.path:             # WiLoR's code; `root` only holds its weights and MANO
            sys.path.insert(0, str(WILOR_ROOT))
        from ultralytics import YOLO
        from wilor.models import load_wilor
        with _inside(self.root):
            self.model, self.cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                              cfg_path="./pretrained_models/model_config.yaml")
            with _full_torch_load(torch):
                self.detector = YOLO("./pretrained_models/detector.pt")
        if self.fast:                       # WiLoR's --fast: half precision and a pruned backbone (~1.6x faster)
            torch.set_float32_matmul_precision("high")
            self.model = self.model.half()
            self.model.backbone.skip_blocks = True
        self.model = self.model.to(self.device).eval()
        self.detector = self.detector.to(self.device)

    def __call__(self, bgr):
        """All hands in one BGR image: [dict(side, score, joints (21, 3), image (21, 2))]."""
        torch = self.torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full
        detections = self.detector(bgr, conf=self.detection_confidence, verbose=False)[0]
        if len(detections) == 0:
            return []
        boxes = detections.boxes.xyxy.detach().cpu().numpy().reshape(-1, 4)
        right = detections.boxes.cls.detach().cpu().numpy().reshape(-1)
        scores = detections.boxes.conf.detach().cpu().numpy().reshape(-1)
        keep = best_per_side(right, scores)
        boxes, right, scores = boxes[keep], right[keep], scores[keep]
        dataset = ViTDetDataset(self.cfg, bgr, boxes, right, rescale_factor=self.rescale_factor, fp16=self.fast)
        batch = torch.utils.data.default_collate([dataset[i] for i in range(len(dataset))])
        batch = recursive_to(batch, self.device)
        with torch.no_grad():
            out = self.model(batch)
        sign = 2 * batch["right"] - 1                                   # +1 right, -1 left (mirrored back)
        cam = out["pred_cam"].float().clone()
        cam[:, 1] = sign * cam[:, 1]
        img_size = batch["img_size"].float()
        focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t = cam_crop_to_full(cam, batch["box_center"].float(), batch["box_size"].float(), img_size, focal)
        joints = out["pred_keypoints_3d"].float().clone()
        joints[:, :, 0] = sign[:, None] * joints[:, :, 0]
        hands = []
        for n in range(joints.shape[0]):
            j = joints[n].detach().cpu().numpy().astype(float)
            t = cam_t[n].detach().cpu().numpy().astype(float)
            camera = j + t                                              # in front of the camera
            w, h = img_size[n].detach().cpu().numpy()
            f = float(focal)
            image = np.column_stack([f * camera[:, 0] / camera[:, 2] + w / 2, f * camera[:, 1] / camera[:, 2] + h / 2])
            hands.append(dict(side="Right" if right[n] > 0.5 else "Left", score=float(scores[n]),
                              joints=j - j[0], image=image))
        return hands


class WilorMiniHands:
    """WiLoR through the WiLoR-mini package; same call and output as WilorHands."""

    def __init__(self, root=WILOR_ROOT, device=None, fast=True, detection_confidence=0.3, rescale_factor=2.5,
                 allow_mano_download=True):
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
        self.root = Path(root)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.fast = fast and self.device.type == "cuda"
        self.detection_confidence, self.rescale_factor = detection_confidence, rescale_factor
        folder = self.root / "wilor_mini"
        self._link_files(folder, allow_mano_download)
        with _full_torch_load(torch):                   # its checkpoint and YOLO file both need the old torch.load
            self.pipe = WiLorHandPose3dEstimationPipeline(device=self.device, verbose=False,
                                                          dtype=torch.float16 if self.fast else torch.float32,
                                                          wilor_pretrained_dir=str(folder))

    def _link_files(self, folder, allow_mano_download):
        """WiLoR-mini's files are the WiLoR ones where present: link them instead of downloading 2.6 GB again (what is
        not linked, WiLoR-mini downloads)."""
        target = folder / "pretrained_models"
        target.mkdir(parents=True, exist_ok=True)
        sources = {"detector.pt": self.root / "pretrained_models" / "detector.pt",
                   "wilor_final.ckpt": self.root / "pretrained_models" / "wilor_final.ckpt",
                   "mano_mean_params.npz": self.root / "mano_data" / "mano_mean_params.npz",
                   "MANO_RIGHT.pkl": self.root / "mano_data" / "MANO_RIGHT.pkl"}
        for name, source in sources.items():
            link = target / name
            if link.is_symlink() and not link.exists():
                link.unlink()                            # its file moved: link again
            if source.exists() and not link.exists():
                link.symlink_to(source.resolve())
        if not (target / "MANO_RIGHT.pkl").exists() and not allow_mano_download:
            raise FileNotFoundError(
                f"WiLoR files missing: {sources['MANO_RIGHT.pkl']} (download it from mano.is.tue.mpg.de, or drop "
                f"--no-mano-download to let WiLoR-mini fetch its copy)")

    def __call__(self, bgr):
        """All hands in one BGR image (WiLoR-mini's model takes BGR): [dict(side, score, joints, image)]."""
        detections = self.pipe.hand_detector(bgr, conf=self.detection_confidence, verbose=False)[0]
        if len(detections) == 0:
            return []
        boxes = detections.boxes.xyxy.detach().cpu().numpy().reshape(-1, 4)
        right = detections.boxes.cls.detach().cpu().numpy().reshape(-1)
        scores = detections.boxes.conf.detach().cpu().numpy().reshape(-1)
        keep = best_per_side(right, scores)
        boxes, right, scores = boxes[keep], right[keep], scores[keep]
        results = self.pipe.predict_with_bboxes(bgr, boxes, right, rescale_factor=self.rescale_factor)
        hands = []
        for result, score in zip(results, scores):
            preds = result["wilor_preds"]                # left hands already mirrored back by WiLoR-mini
            joints = np.asarray(preds["pred_keypoints_3d"][0], dtype=float)
            hands.append(dict(side="Right" if result["is_right"] > 0.5 else "Left", score=float(score),
                              joints=joints - joints[0], image=np.asarray(preds["pred_keypoints_2d"][0], dtype=float)))
        return hands


TRACKERS = {"wilor": WilorHands, "wilor_mini": WilorMiniHands}
