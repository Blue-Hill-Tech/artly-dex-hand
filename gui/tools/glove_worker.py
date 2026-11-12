"""Glove -> GeoRT -> hand model (and, once the motor runs, -> the hand controller), in a background thread.

The same pipeline as the teleop sender (teleop/manus/stream/hand_stream_aac.py --solver geort), built from its own parts so
nothing differs: the MANUS integrated SDK source, build_adapter (GeoRT checkpoint, the saved glove alignment
teleop/manus/alignment/aac_alignment_<side>.json, the One-Euro filter of teleop/manus/parameters/teleop_filter.yaml), VectorDiagnostic for
the solve, and AACSession for the controller: its guards (fresh glove frame, valid alignment and solve, fresh measured
feedback) and protocol (execute/idle, timestamped targets) are the sender's own.

The GUI talks to it only through Qt signals (pose, status) and three thread-safe requests: attach_motor(), detach_motor()
and stop(). The worker owns the adapter and the session; nothing of them is touched from the GUI thread. Its glove
frames come from the GUI's GloveHub (gui/tools/glove_hub.py, which owns the one MANUS SDK of the process) when one is given,
otherwise the worker starts and shuts down the SDK itself.

mode "camera": the same pipeline on the webcam hand tracker (teleop/manus/tools/camera_source.py) with the sender's camera
settings (--source camera): its own alignment file aac_alignment_<side>_camera.json, retarget_tuning_camera.yaml,
teleop_filter_camera.yaml, alignment kept across tracking gaps, 15-frame capture. Without a saved camera alignment the
worker still runs and align() captures one (hold the hand flat and still for a second); it is saved to the camera file
only, never to the glove's.
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

ALIGNMENT = ROOT / "teleop" / "manus" / "alignment" / "aac_alignment_{side}{suffix}.json"   # suffix "_camera" for the camera
CAMERA_TUNING = ROOT / "teleop" / "manus" / "parameters" / "retarget_tuning_camera.yaml"
CAMERA_CAPTURE_FRAMES = 15   # the sender's alignment capture for the camera (~30 fps), instead of the glove's 30
MODES = ("glove", "camera", "wilor", "wilor_mini")
CAMERA_MODES = ("camera", "wilor", "wilor_mini")   # webcam trackers: camera settings, own alignment file, align()
ALIGNMENT_SUFFIX = {"glove": "", "camera": "_camera", "wilor": "_wilor", "wilor_mini": "_wilor_mini"}
POSE_HZ = 60              # model updates sent to the GUI (the glove runs ~60-120 Hz)
STATUS_HZ = 4
SEND_RATE_HZ = 100        # target submissions to the controller (the sender's --send-rate)
EXECUTE_RETRY_S = 0.5     # how often to ask the controller to execute until its guards pass
ATTACH_RETRY_S = 0.5      # how often to try the controller's sockets while it starts


# The GeoRT run per side and source (pins and the newest-run rule): teleop/manus/geort/checkpoints.py, shared
# with the sender.
from teleop.manus.geort.checkpoints import CHECKPOINTS, GEORT_CHECKPOINTS, checkpoint_for  # noqa: E402,F401


class GloveWorker(QThread):
    pose = pyqtSignal(list)        # 16 SDK degrees (command frame), the retargeted glove pose
    skeleton = pyqtSignal(object)  # (joint points (N, 3), bone segment ends (2M, 3)) on the robot model, or None
    status = pyqtSignal(str)       # one line for the GUI
    failed = pyqtSignal(str)       # the thread stopped on an error
    alignment = pyqtSignal(str)    # "none" (needs align()), "collecting", "ok"; on every change

    def __init__(self, side, parent=None, hub=None, mode="glove"):
        super().__init__(parent)
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.side = side.capitalize()
        self.hub = hub
        self.mode = mode
        self._align = False
        self.want_skeleton = True     # compute the skeleton overlay for the display (skipped when it is hidden)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._motor_request = None    # ("attach", sockets dict) or ("detach", None), consumed by the thread
        self.motor_attached = False

    # --- requests from the GUI thread -------------------------------------------------------------------------
    def stop(self):
        self._stop.set()

    def attach_motor(self, stream, control, telemetry, hand_id):
        with self._lock:
            self._motor_request = ("attach", dict(stream=stream, control=control, telemetry=telemetry, hand_id=hand_id))

    def detach_motor(self):
        with self._lock:
            self._motor_request = ("detach", None)

    def align(self):
        """Capture a new alignment now (camera mode): the hand flat and still for a second; saved when it succeeds."""
        with self._lock:
            self._align = True

    def alignment_path(self):
        return Path(str(ALIGNMENT).format(side=self.side.lower(), suffix=ALIGNMENT_SUFFIX[self.mode]))

    # --- the thread ---------------------------------------------------------------------------------------------
    def run(self):
        try:
            self._run()
        except Exception as error:                    # reported to the GUI; the thread ends
            self.failed.emit(f"{type(error).__name__}: {error}")

    def _build(self):
        from teleop.manus.tools.dex_vector_retargeting import VectorDiagnostic, load_tuning
        from teleop.manus.tools.preview_dex_retargeting import build_adapter
        checkpoint = checkpoint_for(self.side, self.mode)
        self.status.emit(f"Loading GeoRT ({checkpoint.name}) ...")
        args = SimpleNamespace(solver="geort", geort_checkpoint=checkpoint, geort_epoch=None, method="vector",
                               thumb_offset_deg=0.0, thumb_cmc_amplify=1.0, thumb_cmc_amplify_max_deg=None,
                               thumb_cmc_gain=1.0, thumb_rotation_lock=None, no_filter=False, filter_config=None,
                               source="camera" if self.mode in CAMERA_MODES else "manus")   # picks the camera filter
        alignment = self.alignment_path()
        spec = dict(side=self.side, hand_id=f"{self.side}Hand", glove_id=None, log=None,
                    calibration=alignment if alignment.exists() else None, calibration_explicit=False)
        camera = self.mode in CAMERA_MODES
        tuning = load_tuning(CAMERA_TUNING if camera and CAMERA_TUNING.exists() else None)   # as the sender does
        adapter, calibration = build_adapter(args, tuning, spec)
        if calibration is None and not camera:
            raise RuntimeError(f"no usable saved glove alignment at {alignment}: align and save it in the viewer first")
        engine = VectorDiagnostic(adapter)
        if camera:
            engine.min_capture_frames = CAMERA_CAPTURE_FRAMES
        return adapter, calibration, engine

    def _run(self):
        from teleop.manus.tools.preview_dex_retargeting import Evidence
        from teleop.manus.tools.skeleton_source import load_source, select_glove
        from teleop.manus.tools.dex_vector_retargeting import save_alignment
        from teleop.manus.stream.hand_stream_aac import convert_angles
        from teleop.manus.tools.preview_dex_retargeting import glove_overlay
        adapter, calibration, engine = self._build()
        # A saved alignment stays valid through tracking gaps: by default the engine drops it after any gap > 0.35 s
        # (the viewer then needs A again, and the GUI has no A), which happened right after connecting while the
        # MANUS SDK settles. Keep it across gaps, and re-install the saved one whenever it is lost anyway (a changed
        # skeleton hierarchy also clears it).
        engine.keep_alignment_on_gap = True
        overlay_engine = self._overlay_view(engine, adapter)
        evidence = Evidence(None)
        if self.hub is not None:
            source = self.hub.view()
        elif self.mode == "camera":
            from teleop.manus.tools.camera_source import load_camera_source
            source = load_camera_source(SimpleNamespace(), [self.side])
        elif self.mode in ("wilor", "wilor_mini"):
            from teleop.manus.wilor.wilor_source import load_wilor_source
            source = load_wilor_source([self.side], tracker=self.mode)
        else:
            source = load_source()
        reported_alignment = None
        session = None
        initialized = False
        locked = hand = None
        execute_wanted = False
        pending_attach, attach_at = None, 0.0
        last_execute_try = last_pose = last_status = 0.0
        flat_wait = False                # MediaPipe: Align waits for the flat-hand capture
        try:
            if self.hub is None:                           # no shared connection: this worker owns the SDK
                self.status.emit("Connecting to the camera ..." if self.mode in CAMERA_MODES else
                                 "Connecting to the MANUS glove ...")
                source.backend.check(source.backend.lib.CoreSdk_InitializeIntegrated(), "Initialize integrated SDK")
                initialized = True
                source.connect()
            while not self._stop.is_set():
                # Motor requests from the GUI.
                with self._lock:
                    request, self._motor_request = self._motor_request, None
                if request is not None:
                    kind, sockets = request
                    if session is not None:
                        try:
                            session.close()                      # idles the controller first
                        except Exception:
                            pass                                 # the controller may already be gone
                        session, self.motor_attached = None, False
                    pending_attach = sockets if kind == "attach" else None
                    attach_at = 0.0
                if pending_attach is not None and time.perf_counter() >= attach_at:
                    session = self._attach(adapter, evidence, pending_attach)
                    if session is None:                          # controller not up yet: try again shortly
                        attach_at = time.perf_counter() + ATTACH_RETRY_S
                    else:
                        pending_attach = None
                        self.motor_attached, execute_wanted = True, True
                # Newest glove frame for this side.
                frames, devices = source.consume()
                chosen = select_glove(devices, self.side, None, locked)
                if locked is None and chosen is not None:
                    locked = chosen
                new = frames.get(locked) if chosen is not None else None
                stamp = session.frame_timestamp(new) if session is not None and new is not None else None
                if new is not None:
                    try:
                        engine.set_metadata(source.metadata(locked))
                    except Exception:                            # the glove went away just after this frame
                        new, stamp = None, None
                if new is not None:
                    if calibration is not None and engine.alignment is None and engine.collection is None:
                        engine.install_alignment(calibration, locked)     # first frame, or after it was lost
                    if new["frame_id"] == engine.last_frame_id:
                        new, stamp = None, None
                    else:
                        hand = new
                now = time.perf_counter()
                with self._lock:
                    align, self._align = self._align, False
                if align and locked is not None:
                    if session is not None and session.execution_id is not None:
                        session.idle("Alignment requested")      # as the sender: no targets while aligning
                        execute_wanted = True                    # executes again once the new alignment is live
                    # MediaPipe: the same flat hand first gives the flat-pose correction (camera_source.FlatPose),
                    # then the alignment is captured from the corrected skeleton.
                    capture = getattr(source, "capture_flat", None)
                    flat_wait = bool(capture is not None and capture(self.side))
                    if not flat_wait:
                        engine.start_alignment(now)
                if flat_wait:
                    flat = source.flat_status(self.side)
                    if flat != "capturing":
                        flat_wait = False
                        if flat is not None and flat.startswith("failed"):
                            self.status.emit("Flat hand not captured (" + flat[8:] + "): hold it flat and click Align")
                        else:
                            engine.start_alignment(now)
                collecting = engine.collection is not None
                engine.check_freshness(None if chosen is None else hand, now)
                sample = engine.process(new, now) if new is not None else engine.snapshot(now)
                if collecting and engine.collection is None and engine.alignment is not None:
                    path = self.alignment_path()
                    calibration = save_alignment(path, engine, locked)  # the new alignment, also for re-installs
                    self.status.emit(f"Alignment saved to {path.name}")
                state = "collecting" if flat_wait or engine.collection is not None else "ok" if engine.alignment is not None \
                    else "ok" if calibration is not None else "none"
                if state != reported_alignment:
                    reported_alignment = state
                    self.alignment.emit(state)
                # The controller session (only once the motor is started).
                if session is not None:
                    ask = execute_wanted and session.execution_id is None and now - last_execute_try >= EXECUTE_RETRY_S
                    if ask:
                        last_execute_try = now
                    sample["session"] = session.update(sample, stamp, execute=ask)
                    if session.execution_id is not None:
                        execute_wanted = False
                # The model.
                result = sample.get("result")
                if result is not None and now - last_pose >= 1.0 / POSE_HZ:
                    last_pose = now
                    try:
                        self.pose.emit(list(convert_angles(result["display_degrees"], adapter.bounds, to_sdk=True,
                                                           side=adapter.side)))
                    except ValueError:
                        pass
                    # The glove skeleton on the robot model, as in the viewer: rotated by the alignment, knuckles
                    # centred on the robot's (the same frame as the model's meshes).
                    live = sample.get("reason") is None
                    self.skeleton.emit(glove_overlay(overlay_engine, result["display_degrees"])
                                       if live and self.want_skeleton else None)
                elif result is None and now - last_pose >= 1.0 / POSE_HZ:
                    last_pose = now
                    self.skeleton.emit(None)
                if now - last_status >= 1.0 / STATUS_HZ:
                    last_status = now
                    self.status.emit(self._describe(locked, sample, session, pending_attach is not None))
                if new is None:
                    time.sleep(0.001)
        finally:
            if session is not None:
                try:
                    session.close()
                finally:
                    self.motor_attached = False
            if initialized:
                try:
                    source.backend.check(source.backend.lib.CoreSdk_ShutDown(), "Shut down SDK")
                except Exception:
                    pass
            if self.hub is None and hasattr(source, "close"):
                source.close()
            evidence.close()
            self.skeleton.emit(None)
            self.status.emit("Glove disconnected")

    @staticmethod
    def _overlay_view(engine, adapter):
        """The engine as glove_overlay sees it, but with the robot knuckle points on the finger centrelines.

        glove_overlay centres the glove's knuckles on the robot's (adapter.point of <f>_mcp). The D21 URDF puts the
        <f>_mcp frames 8.4 mm beside the finger centrelines (dex_vector_retargeting.CENTRELINE_REFERENCES), and the
        GeoRT adapter inherits CENTRELINE_POINTS = False, so its point() is the raw frame origin: the whole skeleton
        was drawn 8.4 mm to the side of the fingers. This view gives the overlay the corrected points; GeoRT's solve
        and the controller targets do not use it."""
        from types import SimpleNamespace
        from teleop.manus.tools.dex_vector_retargeting import centreline_offsets
        offsets = centreline_offsets(adapter.model)

        class View:
            def __getattr__(self, name):                 # geometry, alignment, metadata, ... read live
                return getattr(engine, name)

        view = View()
        view.adapter = SimpleNamespace(palm_root=adapter.palm_root, model=adapter.model,
                                       point=lambda poses, link: adapter.link_point(poses, link, offsets.get(link)))
        return view

    def _attach(self, adapter, evidence, sockets):
        """The sender session on the controller's sockets, or None while the controller does not answer yet."""
        from teleop.manus.stream.hand_stream_aac import AACConnection, AACSession
        connection = AACConnection(sockets["stream"], sockets["control"], sockets["telemetry"], sockets["hand_id"])
        session = AACSession(adapter, evidence, connection, hand_id=sockets["hand_id"], send_rate_hz=SEND_RATE_HZ)
        try:
            session.start()                              # asks the controller for idle: it must answer
        except Exception:
            connection.close()
            return None
        return session

    def _describe(self, locked, sample, session, pending_attach=False):
        if locked is None:
            return f"Waiting for the {self.side.lower()} " + ("hand in the camera ..." if self.mode in CAMERA_MODES
                                                               else "MANUS glove ...")
        glove = (f"Camera: {sample.get('state')}" if self.mode in CAMERA_MODES
                 else f"Glove {locked:08X}: {sample.get('state')}")
        if sample.get("reason"):
            glove += f" ({sample['reason']})"
        if session is None:
            return glove + (" - motor: waiting for the controller to start" if pending_attach else " - model only")
        if session.execution_id is not None:
            return glove + f" - sending to the hand ({session.sent} targets)"
        return glove + f" - motor waiting: {session.reason}"
