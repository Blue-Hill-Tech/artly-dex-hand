"""The MANUS glove connection of the whole GUI: searches for gloves from start-up and shares their frames.

The integrated MANUS SDK is one per process, so a single thread owns it (initialise, connect, consume, shut down) and
the per-side GloveWorkers read from it through view(), which behaves like skeleton_source.Source (consume() returns the
frames that are new for that reader, metadata() passes through). `gloves` tells the GUI which side has a glove: a glove
counts as found while the MANUS landscape lists it and its frames keep arriving.

mode "camera": the webcam hand tracker (teleop/manus/tools/camera_source.py, both sides, told apart by MediaPipe's hand
labels) in place of the SDK; a side counts as found while its hand is tracked. mode "wilor" / "wilor_mini": the WiLoR
service (teleop/manus/wilor/wilor_source.py) with WiLoR's code or the WiLoR-mini package; its model takes a while to load,
`connected` says when it is up.
"""
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtCore import QThread, pyqtSignal

ROOT = Path(__file__).resolve().parents[2]       # agi-os (this file: gui/tools/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SIDES = ("Left", "Right")
PRESENT_TIMEOUT_S = 1.0   # a glove without frames for this long counts as gone (frames come at ~60-120 Hz)
CHECK_HZ = 10             # how often presence is re-evaluated


def present_gloves(devices, seen, now, timeout=PRESENT_TIMEOUT_S):
    """{side: glove id or None}: per side, a glove the landscape lists whose last frame is recent."""
    found = dict.fromkeys(SIDES)
    for glove_id, device in sorted(devices.items()):
        side = device.get("side")
        if side in found and found[side] is None and now - seen.get(glove_id, -1e9) < timeout:
            found[side] = glove_id
    return found


class HubView:
    """One reader's view of the hub, shaped like skeleton_source.Source for GloveWorker."""

    def __init__(self, hub):
        self.hub = hub
        self._delivered = {}                       # glove id -> the frame last handed to this reader

    def consume(self):
        frames, devices = self.hub.snapshot()
        new = {gid: frame for gid, frame in frames.items() if self._delivered.get(gid) is not frame}
        self._delivered.update(new)
        return new, devices

    def metadata(self, glove_id):
        return self.hub.metadata(glove_id)

    def capture_flat(self, side):
        """Camera (MediaPipe) source: measure the flat hand now (camera_source.FlatPose); False for other sources."""
        source = self.hub._source
        return bool(source is not None and hasattr(source, "capture_flat") and source.capture_flat(side))

    def flat_status(self, side):
        source = self.hub._source
        return source.flat_status(side) if source is not None and hasattr(source, "flat_status") else None


class GloveHub(QThread):
    gloves = pyqtSignal(object)   # {"Left": glove id or None, "Right": ...}, on start and on every change
    failed = pyqtSignal(str)      # the SDK could not be started or stopped working
    connected = pyqtSignal()      # the source is up (the search itself starts now)

    def __init__(self, parent=None, mode="glove", wilor_fast=True):
        super().__init__(parent)
        self.wilor_fast = bool(wilor_fast)               # WiLoR's fast mode on the GPU (Dev Mode)
        self.mode = mode
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frames, self._devices, self._seen = {}, {}, {}
        self._source = None
        self._starting = None

    # --- for the GUI thread and the glove workers ---------------------------------------------------------------
    def stop(self):
        self._stop.set()
        starting = self._starting
        if starting is not None and isinstance(getattr(starting, "stop", None), threading.Event):
            starting.stop.set()                          # a source still connecting (WiLoR loading) gives up

    def view(self):
        return HubView(self)

    def snapshot(self):
        """The newest frame of every glove and the landscape's devices (copies)."""
        with self._lock:
            return dict(self._frames), dict(self._devices)

    def metadata(self, glove_id):
        source = self._source
        if source is None:
            raise RuntimeError("the MANUS SDK is not connected yet")
        return source.metadata(glove_id)

    # --- the thread ---------------------------------------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")

    def _run(self):
        if self.mode == "camera":
            from teleop.manus.tools.camera_source import load_camera_source
            source = load_camera_source(SimpleNamespace(), list(SIDES))   # no SDK: init / shut down do nothing
        elif self.mode in ("wilor", "wilor_mini"):
            from teleop.manus.wilor.wilor_source import load_wilor_source
            source = load_wilor_source(list(SIDES), tracker=self.mode, fast=self.wilor_fast)
        else:
            from teleop.manus.tools.skeleton_source import load_source
            source = load_source()
        initialized = False
        reported, last_check = None, 0.0
        try:
            source.backend.check(source.backend.lib.CoreSdk_InitializeIntegrated(), "Initialize integrated SDK")
            initialized = True
            self._starting = source
            source.connect()
            self._starting = None
            self._source = source
            self.connected.emit()
            while not self._stop.is_set():
                frames, devices = source.consume()
                now = time.perf_counter()
                with self._lock:
                    self._frames.update(frames)
                    self._devices = devices
                    for glove_id in frames:
                        self._seen[glove_id] = now
                    seen = dict(self._seen)
                if reported is None or now - last_check >= 1.0 / CHECK_HZ:
                    last_check = now
                    found = present_gloves(devices, seen, now)
                    if found != reported:
                        reported = found
                        self.gloves.emit(dict(found))
                if not frames:
                    time.sleep(0.001)
        finally:
            self._source = None
            if initialized:
                try:
                    source.backend.check(source.backend.lib.CoreSdk_ShutDown(), "Shut down SDK")
                except Exception:
                    pass
            if hasattr(source, "close"):
                source.close()
            self.gloves.emit(dict.fromkeys(SIDES))
