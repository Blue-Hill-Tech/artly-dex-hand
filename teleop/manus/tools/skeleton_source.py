"""Latest MANUS skeleton acquisition and palm coordinates; no SDK load on import."""

import importlib
from pathlib import Path
import sys

import numpy as np


STALE_S = 0.35


def rotation(q):
    value = np.asarray(q, dtype=float)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError("quaternion xyzw: expected 4 finite values")
    # Scale first so large finite inputs normalize without overflow.
    scale = np.max(np.abs(value))
    if scale == 0:
        raise ValueError("quaternion xyzw: zero norm")
    value = value / scale
    x, y, z, w = value / np.linalg.norm(value)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def palm_rotation_of(hand, palm_id, node_id):
    """Rotation of one node relative to the palm node (R_palm^T R_node), the frame palm_positions uses."""
    nodes = {node["id"]: node for node in hand["nodes"]}
    if palm_id not in nodes or node_id not in nodes:
        raise ValueError("Missing palm or requested skeleton node")
    return rotation(nodes[palm_id]["rotation"]).T @ rotation(nodes[node_id]["rotation"])


def palm_positions(hand, palm_id):
    """Same-frame positions using only the palm quaternion (SDK order: xyzw)."""
    nodes = {node["id"]: node for node in hand["nodes"]}
    if len(nodes) != len(hand["nodes"]):
        raise ValueError("Duplicate skeleton node IDs")
    if palm_id not in nodes:
        raise ValueError("Missing selected palm node")
    palm = nodes[palm_id]
    rp = rotation(palm["rotation"])
    positions = np.array([node["position"] for node in hand["nodes"]], dtype=float)
    if positions.shape != (len(nodes), 3) or not np.isfinite(positions).all():
        raise ValueError("Nonfinite/malformed skeleton positions")
    xyz = (positions - np.asarray(palm["position"])) @ rp
    return rp, dict(zip(nodes, xyz))


def select_glove(devices, side, requested=None, locked=None):
    """Lock once to a physical landscape ID; disappearance never selects another."""
    chosen = locked if locked is not None else requested
    if chosen is not None:
        if chosen not in devices:
            return None
        if devices[chosen]["side"] != side:
            raise ValueError(f"Glove {chosen:08X} side does not match {side}")
        return chosen
    matches = [gid for gid, item in devices.items() if item["side"] == side]
    if len(matches) > 1:
        raise ValueError(f"Multiple {side} gloves: {[f'{gid:08X}' for gid in matches]}; use --glove-id")
    return matches[0] if matches else None


class Source:
    """Copied latest-per-glove skeletons and callback errors, without ergonomics."""
    def __init__(self, backend, enums):
        self.backend, self.enums = backend, enums
        self.stream = backend.GloveStream()
        self.devices = {}

    def on_landscape(self, landscape):
        try:
            devices = {}
            for glove in landscape.gloveDevices.gloves[0:landscape.gloveDevices.gloveCount]:
                devices[int(glove.id)] = dict(side=self.enums.Side(glove.side).name,
                    family=self.enums.DeviceFamilyType(glove.familyType).name,
                    device_class=self.enums.DeviceClassType(glove.classType).name)
            with self.stream.lock:
                self.devices = devices
        except Exception as error:
            with self.stream.lock:
                self.stream.error = error

    def consume(self):
        with self.stream.lock:
            frames, self.stream.latest = self.stream.latest, {}
            devices, error = self.devices.copy(), self.stream.error
        if error is not None:
            raise error.with_traceback(error.__traceback__)
        return frames, devices

    def metadata(self, glove_id):
        ffi, lib, check, enums = self.backend.ffi, self.backend.lib, self.backend.check, self.enums
        count = ffi.new("uint32_t *")
        check(lib.CoreSdk_GetRawSkeletonNodeCount(glove_id, count), "Get node count")
        nodes = ffi.new("NodeInfo[]", count[0])
        check(lib.CoreSdk_GetRawSkeletonNodeInfoArray(glove_id, nodes, count[0]), "Get node metadata")
        return [dict(id=int(n.nodeId), parent=int(n.parentId), side=enums.Side(n.side).name,
                     chain=enums.ChainType(n.chainType).name, joint=enums.FingerJointType(n.fingerJointType).name)
                for n in nodes]

    def connect(self):
        b, e = self.backend, self.enums
        for method, callback, label in (
            (self.stream.sdk.register_landscape_callback, self.on_landscape, "landscape"),
            (self.stream.sdk.register_raw_skeleton_callback, self.stream.on_skeletons, "skeleton")):
            b.check(method(callback), f"Register {label} callback")
        coordinates = b.ffi.new("CoordinateSystemVUH *")
        coordinates.handedness, coordinates.up = e.Side.Right, e.AxisPolarity.PositiveZ
        coordinates.view, coordinates.unitScale = e.AxisView.XFromViewer, 1.0
        b.check(b.lib.CoreSdk_InitializeCoordinateSystemWithVUH(coordinates[0], True), "Set global coordinates")
        host = b.ffi.new("ManusHost *")
        b.check(b.lib.CoreSdk_ConnectToHost(host[0]), "Connect integrated SDK")
        b.check(b.lib.CoreSdk_SetRawSkeletonHandMotion(e.HandMotion.IMU), "Set IMU hand motion")


def load_source():
    # Math/import/help never load the native SDK or any visualization backend.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "stream"))   # stream_gloves (teleop/manus/stream/)
    backend = importlib.import_module("stream_gloves")
    enums = importlib.import_module("manus_sdk.generated._enums")
    return Source(backend, enums)
