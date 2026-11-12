"""3D view of one D21 hand (URDF meshes, VisPy), embeddable as a Qt widget.

Poses are given in SDK degrees (command frame, hand/backends/aac.JOINT_NAMES order) and mapped to the URDF joints with
the per-side signs of teleop/manus/hand_stream_aac.joint_map, so the same numbers the controller uses pose the model.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]       # agi-os (this file: gui/tools/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vispy import app, scene  # noqa: E402

app.use_app("pyqt5")

from hand.aac.model import HandVisual  # noqa: E402
from hand.backends.aac import JOINT_NAMES  # noqa: E402
from teleop.manus.stream.hand_stream_aac import joint_map  # noqa: E402

BACKGROUND = "#FAF8EF"
HAND_COLOR = (0.60, 0.61, 0.64, 1.0)       # metal grey: darker than the URDF's 0.8 grey, which vanishes on cream
# Start views: palm facing the screen. Computed from the URDF: the fingertips curl toward +y on the left hand and -y
# on the right (the palm normal), and these azimuths look straight into it; 8 deg of elevation keeps some depth.
CAMERA = {"Left": dict(azimuth=180, elevation=8), "Right": dict(azimuth=0, elevation=8)}
# Zoom of the start view after centring on the body: 1.0 left the hand at 40 % of its view; 0.72 leaves the thumb tip
# 4-6 % of the view width from the edge at 1200x800 and 1920x1080 windows (measured on renders, 2026-11-06).
BODY_ZOOM = 0.72
SKELETON_COLOR = (0.91, 0.47, 0.10, 0.95)   # the glove skeleton over the model: orange, readable on cream and grey
SKELETON_JOINT_PX = 7
# The hand without a glove: see-through. Back faces are culled while it is, so each part shows only its outer skin
# (otherwise the inside of every mesh shows through as patches).
GHOST_OPACITY = 0.4
SKELETON_BONE_PX = 3


class HandView:
    """One hand in its own canvas. `widget` is the Qt widget to put in a layout."""

    def __init__(self, side, background=BACKGROUND, color=HAND_COLOR):
        self.side = side
        self.canvas = scene.SceneCanvas(bgcolor=background, keys=None, show=False)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=35, up="+z", **CAMERA[side])
        self.model = HandVisual(self.view, side)                 # frames the camera on the whole model
        self._centre_on_body()
        self._home = self.view.camera.get_state()                # the framed view, for recenter()
        self.color = tuple(color[:3])
        self.opacity = 1.0
        for _, node, _ in self.model.visuals:
            node.color = color
            shading = getattr(node, "shading_filter", None)
            if shading is not None:                    # soft studio light: bright ambient, gentle highlight
                shading.ambient_light = (1, 1, 1, 0.5)
                shading.diffuse_light = (1, 1, 1, 0.5)
                shading.specular_light = (1, 1, 1, 0.1)   # facing the light head-on, more glare whites the tips out
                shading.shininess = 40
        self.map = joint_map(side)
        # The glove skeleton (shown while the glove is live), drawn over the model.
        self.skeleton_bones = scene.visuals.Line(parent=self.view.scene, connect="segments", color=SKELETON_COLOR,
                                                 width=SKELETON_BONE_PX, antialias=True)
        self.skeleton_joints = scene.visuals.Markers(parent=self.view.scene)
        for visual in (self.skeleton_bones, self.skeleton_joints):
            visual.set_gl_state("translucent", depth_test=False)
            visual.visible = False
        self.skeleton_enabled = True     # the user's show/hide choice
        self._skeleton_live = False      # whether there is a skeleton to show
        # Headlight: the light comes from the viewer whichever way the hand is turned (a fixed light leaves the
        # far side dark when the user orbits).
        self.view.scene.transform.changed.connect(self._follow_light)
        self._follow_light()
        self.pose([0.0] * 16)

    def _follow_light(self, event=None):
        direction = self.view.camera.transform.map((0, 0, -1, 0))[:3]   # along the view direction
        for _, node, _ in self.model.visuals:
            shading = getattr(node, "shading_filter", None)
            if shading is not None:
                shading.light_dir = direction

    def _centre_on_body(self):
        """Centre the view on the hand body (palm + fingers, not the thumb, which sticks out to one side), so the
        hand sits in the middle of its view, under its caption; zoom out just enough that the thumb still fits."""
        body, every = [], []
        for link, node, _ in self.model.visuals:
            vertices = node.mesh_data.get_vertices()
            world = (np.c_[vertices, np.ones(len(vertices))] @ node.transform.matrix)[:, :3]
            every.append(world)
            if not link.startswith("thumb"):
                body.append(world)
        body, every = np.concatenate(body), np.concatenate(every)
        camera = self.view.camera
        old_centre = np.asarray(camera.center, dtype=float)
        centre = (body.min(0) + body.max(0)) / 2
        centre[2] = old_centre[2]                                 # keep the height framing of the whole model
        shift = np.linalg.norm((centre - old_centre)[:2])
        half = (every.max(0) - every.min(0))[:2].max() / 2       # half the model's horizontal size
        camera.center = tuple(centre)
        camera.scale_factor *= ((half + shift) / half if half > 0 else 1.0) * BODY_ZOOM

    @property
    def widget(self):
        return self.canvas.native

    def recenter(self):
        """Back to the start-up view: angle, centre and zoom."""
        self.view.camera.set_state(self._home)
        self.view.camera.view_changed()
        self._follow_light()
        self.canvas.update()

    def show_skeleton(self, overlay):
        """overlay: (joint points (N, 3), bone segment ends (2M, 3)) in the model's frame, or None to hide."""
        self._skeleton_live = overlay is not None and len(overlay[0]) > 0
        if self._skeleton_live and self.skeleton_enabled:
            points, segments = overlay
            self.skeleton_joints.set_data(np.asarray(points, dtype=np.float32), face_color=SKELETON_COLOR,
                                          edge_color=(1, 1, 1, 0.9), edge_width=1, size=SKELETON_JOINT_PX)
            self.skeleton_bones.set_data(np.asarray(segments, dtype=np.float32))
        self._apply_skeleton_visibility()

    def set_skeleton_enabled(self, enabled):
        """Show or hide the glove skeleton (it is drawn only while the glove is live)."""
        self.skeleton_enabled = bool(enabled)
        self._apply_skeleton_visibility()

    def _apply_skeleton_visibility(self):
        visible = self._skeleton_live and self.skeleton_enabled
        self.skeleton_joints.visible = self.skeleton_bones.visible = visible
        self.canvas.update()

    def set_opacity(self, opacity):
        """1 = solid hand; less = see-through (GHOST_OPACITY while no glove is found)."""
        opacity = float(min(max(opacity, 0.0), 1.0))
        for _, node, _ in self.model.visuals:
            node.color = self.color + (opacity,)
            node.set_gl_state("translucent", depth_test=True, cull_face=opacity < 1.0)   # solid: VisPy's default
        self.opacity = opacity
        self.canvas.update()

    def pose(self, sdk_degrees):
        """Show the hand at these SDK angles (16, JOINT_NAMES order)."""
        values = dict(zip(JOINT_NAMES, sdk_degrees))
        self.model.update([values[sdk] * sign for _, sdk, sign in self.map])
        self.canvas.update()
