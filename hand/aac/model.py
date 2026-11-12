"""AAC URDF forward kinematics and mesh display; no hardware connection."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation

from hand.backends.aac import validate_positions


MODEL_DIR = Path(__file__).with_name("description")
MODEL_JOINTS = [f"thumb_joint{i}" for i in range(1, 5)] + [
    f"{finger}_joint{i}" for finger in ("index", "middle", "ring", "pinky")
    for i in range(1, 4)
]


def origin(element):
    transform = np.eye(4)
    if element is not None:
        transform[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        transform[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return transform


class HandModel:
    """Forward kinematics for the supplied URDF, without physics simulation."""
    def __init__(self, side="Left"):
        if side not in ("Left", "Right"):
            raise ValueError("Model side must be Left or Right")
        self.path = MODEL_DIR / f"D21_{side}.urdf"
        self.root = ET.parse(self.path).getroot()
        self.joints = []
        for joint in self.root.findall("joint"):
            axis = None
            if joint.get("type") != "fixed":
                if joint.get("type") != "revolute":
                    raise ValueError(f"Unsupported URDF joint: {joint.attrib}")
                axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
                axis /= np.linalg.norm(axis)
            self.joints.append((joint.get("name"), joint.find("parent").get("link"),
                                joint.find("child").get("link"), origin(joint.find("origin")), axis))

    def transforms(self, degrees):
        angles = dict(zip(MODEL_JOINTS, np.deg2rad(validate_positions(degrees))))
        # Vendor model couplings; these distal joints are not independent commands.
        angles["thumb_joint5"] = 0.85 * angles["thumb_joint4"]
        for finger in ("index", "middle", "ring", "pinky"):
            angles[f"{finger}_joint4"] = 0.8 * angles[f"{finger}_joint3"]
        poses = {"world": np.eye(4)}
        # This supplied URDF lists parents before children.
        for name, parent, child, fixed, axis in self.joints:
            rotation = np.eye(4)
            if axis is not None:
                rotation[:3, :3] = Rotation.from_rotvec(axis * angles[name]).as_matrix()
            poses[child] = poses[parent] @ fixed @ rotation
        return poses


class HandVisual:
    """Reusable mesh renderer inside an existing VisPy view/backend."""
    def __init__(self, viewport, side="Left", degrees=None):
        import trimesh
        from vispy import scene
        from vispy.visuals.transforms import MatrixTransform

        self.model = HandModel(side)
        self.visuals = []
        bounds = []
        degrees = [0] * 16 if degrees is None else degrees
        poses = self.model.transforms(degrees)
        for link in self.model.root.findall("link"):
            for visual in link.findall("visual"):
                mesh_element = visual.find("geometry/mesh")
                mesh = trimesh.load_mesh(self.model.path.parent / mesh_element.get("filename"), process=False)
                vertices = mesh.vertices * np.fromstring(mesh_element.get("scale", "1 1 1"), sep=" ")
                color = np.fromstring(visual.find("material/color").get("rgba"), sep=" ")
                node = scene.visuals.Mesh(vertices=vertices, faces=mesh.faces, color=color,
                                          shading="smooth", parent=viewport.scene)
                node.transform = MatrixTransform()
                local = origin(visual.find("origin"))
                self.visuals.append((link.get("name"), node, local))
                transform = poses[link.get("name")] @ local
                bounds.append(vertices @ transform[:3, :3].T + transform[:3, 3])
        self.update(degrees)
        vertices = np.concatenate(bounds)
        lo, hi = vertices.min(axis=0), vertices.max(axis=0)
        viewport.camera.set_range(x=(lo[0], hi[0]), y=(lo[1], hi[1]), z=(lo[2], hi[2]))

    def update(self, degrees):
        poses = self.model.transforms(degrees)
        for link, node, local in self.visuals:
            # VisPy maps row vectors; URDF transforms above use column vectors.
            node.transform.matrix = (poses[link] @ local).T

    def set_visible(self, visible):
        for _, node, _ in self.visuals:
            node.visible = visible

    def detach(self):
        for _, node, _ in self.visuals:
            node.parent = None
