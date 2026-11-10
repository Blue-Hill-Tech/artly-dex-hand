"""The tracking service: glove / camera -> GeoRT -> hand controller, as its own process, apart from the GUI.

The GUI only draws. Everything on the control path runs here: the tracking source (the MANUS SDK, the webcam with
MediaPipe), the per-side GloveWorkers (GeoRT retargeting, the saved alignment, the controller
session at 100 Hz) and the search / auto-connect logic. Nothing the GUI does (rendering, a garbage-collection pause,
a window drag) can delay a target to the hand, because it happens in another process.

Started by the GUI (gui/tools/tracking_link.py), or by hand:

    hand/.hand/bin/python gui/tools/tracking_service.py --source glove|camera

Two local ZMQ sockets, bound on free ports and announced on stdout once the service is up:

    TRACKING READY state=tcp://127.0.0.1:NNNNN commands=tcp://127.0.0.1:MMMMM

state (PUB, PUBLISH_HZ): the whole state as one JSON snapshot each time (a late reader just takes the newest):
    {type: "tracking", t, source, phase: "starting"|"searching"|"ready"|"failed", error,
     found: {Left: glove id | null, Right: ...},
     sides: {Left: {connected, pose: [16 SDK degrees] | null, skeleton: {points, segments} | null, status,
                    alignment: "none"|"collecting"|"ok"|null, motor: bool, error}, Right: {...}}}
commands (PULL): JSON, one per message:
    {cmd: "connect", side} / {cmd: "disconnect", side}            a side's retargeting (auto-connect on/off)
    {cmd: "attach_motor", side, stream, control, telemetry, hand_id} / {cmd: "detach_motor", side}
    {cmd: "align", side}                                          camera modes: capture a new alignment
    {cmd: "skeleton", side, enabled}                              compute the overlay for the GUI or not
    {cmd: "stop"}                                                 idle the controllers and exit (SIGTERM does the same)
"""
import argparse
import gc
import json
import os
import signal
import sys
import time
from pathlib import Path

from PyQt5.QtCore import QCoreApplication, QObject, QTimer, pyqtSignal

PARENT_CHECK_MS = 1000     # how often the service checks that the GUI that started it still runs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                             # agi-os (this file: gui/tools/)
for path in (HERE, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

SIDES = ("Left", "Right")
SOURCES = ("glove", "camera")
SEARCH_GRACE_S = 2.5          # "searching" this long after the source is up, then a missing hand is "not found"
PUBLISH_HZ = 60
COMMAND_POLL_MS = 5
# Garbage collection: a full (generation-2) pass over everything this process holds (torch, numpy, Qt, the models)
# took 70-90 ms in the GUI process and froze the control thread with it. Here the long-lived objects are moved out
# of the collector's way every GC_FREEZE_S (gc.freeze: cheap), so the passes that remain only look at recent
# garbage, and full passes are made rarer.
GC_FREEZE_S = 5.0
GC_THRESHOLDS = (2000, 20, 50)


class TrackingService(QObject):
    """The search, the per-side workers and the state they report; driven by commands, publishes snapshots."""

    changed = pyqtSignal()        # something in the snapshot changed (the publisher may send one early)

    def __init__(self, source, parent=None):
        super().__init__(parent)
        if source not in SOURCES:
            raise ValueError(f"source must be one of {SOURCES}")
        self.source = source
        self.hub = None
        self.workers = {}                                        # side -> GloveWorker
        self.found = dict.fromkeys(SIDES)
        self.phase, self.error = "starting", None
        self.auto = dict.fromkeys(SIDES, True)                   # connect a side as soon as its hand is found
        self.motor_sockets = {}                                  # side -> sockets while its motor runs
        self.want_skeleton = dict.fromkeys(SIDES, True)
        self.sides = {side: self._blank_side() for side in SIDES}
        self._skeleton_cache = {}                                # side -> (object, lists)

    @staticmethod
    def _blank_side():
        return dict(connected=False, pose=None, skeleton=None, status="", alignment=None, motor=False, error=None)

    # --- the search ---------------------------------------------------------------------------------------------
    def start(self):
        from glove_hub import GloveHub
        hub = self.hub = GloveHub(self, mode=self.source)
        hub.gloves.connect(lambda found, hub=hub: self._gloves_changed(hub, found))
        hub.failed.connect(lambda message, hub=hub: self._search_failed(hub, message))
        hub.connected.connect(lambda hub=hub: self._source_up(hub))
        hub.start()

    def _source_up(self, hub):
        if hub is not self.hub:
            return
        self.phase = "searching"
        QTimer.singleShot(int(SEARCH_GRACE_S * 1000), lambda: self._search_settled(hub))
        self.changed.emit()

    def _search_settled(self, hub):
        if hub is self.hub and self.phase == "searching":
            self.phase = "ready"
            self.changed.emit()

    def _search_failed(self, hub, message):
        if hub is not self.hub:
            return
        self.phase, self.error = "failed", message
        print(f"[tracking] {self.source} search failed: {message}", flush=True)
        self.changed.emit()

    def _gloves_changed(self, hub, found):
        if hub is not self.hub:
            return
        self.found = dict(found)
        for side in SIDES:
            if self.found.get(side) is not None and self.auto[side] and not self._running(side):
                self._start_side(side)
        self.changed.emit()

    # --- the sides ----------------------------------------------------------------------------------------------
    def _running(self, side):
        worker = self.workers.get(side)
        return worker is not None and worker.isRunning()

    def connect_side(self, side):
        self.auto[side] = True
        if not self._running(side):
            self._start_side(side)

    def disconnect_side(self, side):
        self.auto[side] = False                                  # stays off until connect is asked again
        if self._running(side):
            self.workers[side].stop()

    def _start_side(self, side):
        from glove_worker import GloveWorker
        hub = self.hub if self.hub is not None and self.hub.isRunning() else None
        worker = GloveWorker(side, self, hub=hub, mode=self.source)
        worker.want_skeleton = self.want_skeleton[side]
        state = self.sides[side] = self._blank_side()
        state["connected"] = True
        worker.pose.connect(lambda pose, s=state: s.__setitem__("pose", pose))
        worker.skeleton.connect(lambda overlay, side=side: self._skeleton(side, overlay))
        worker.status.connect(lambda text, s=state: s.__setitem__("status", text))
        worker.alignment.connect(lambda text, s=state: (s.__setitem__("alignment", text), self.changed.emit()))
        worker.failed.connect(lambda message, side=side, worker=worker: self._side_failed(side, worker, message))
        worker.finished.connect(lambda side=side, worker=worker: self._side_finished(side, worker))
        self.workers[side] = worker
        worker.start()
        if side in self.motor_sockets:                           # its motor already runs: drive it
            worker.attach_motor(**self.motor_sockets[side])
        self.changed.emit()

    def _skeleton(self, side, overlay):
        state = self.sides[side]
        if overlay is None:
            state["skeleton"] = None
            return
        cached = self._skeleton_cache.get(side)
        if cached is None or cached[0] is not overlay:
            points, segments = overlay
            lists = dict(points=[[float(v) for v in p] for p in points],
                         segments=[[float(v) for v in p] for p in segments])
            self._skeleton_cache[side] = (overlay, lists)
            cached = self._skeleton_cache[side]
        state["skeleton"] = cached[1]

    def _side_failed(self, side, worker, message):
        if self.workers.get(side) is worker:
            self.sides[side]["error"] = message
            print(f"[tracking] {side}: {message}", flush=True)

    def _side_finished(self, side, worker):
        if self.workers.get(side) is not worker:
            return
        state = self.sides[side]
        state.update(connected=False, pose=None, skeleton=None, motor=False, alignment=None)
        self.changed.emit()

    def attach_motor(self, side, sockets):
        self.motor_sockets[side] = dict(sockets)
        if self._running(side):
            self.workers[side].attach_motor(**sockets)

    def detach_motor(self, side):
        self.motor_sockets.pop(side, None)
        if self._running(side):
            self.workers[side].detach_motor()

    def align(self, side):
        if self._running(side):
            self.workers[side].align()

    def set_skeleton(self, side, enabled):
        self.want_skeleton[side] = bool(enabled)
        if self._running(side):
            self.workers[side].want_skeleton = bool(enabled)

    # --- commands and state ---------------------------------------------------------------------------------------
    def handle(self, command):
        cmd, side = command.get("cmd"), command.get("side")
        if side is not None and side not in SIDES:
            print(f"[tracking] ignored command for side {side!r}", flush=True)
            return
        if cmd == "connect":
            self.connect_side(side)
        elif cmd == "disconnect":
            self.disconnect_side(side)
        elif cmd == "attach_motor":
            self.attach_motor(side, {k: command[k] for k in ("stream", "control", "telemetry", "hand_id")})
        elif cmd == "detach_motor":
            self.detach_motor(side)
        elif cmd == "align":
            self.align(side)
        elif cmd == "skeleton":
            self.set_skeleton(side, command.get("enabled", True))
        elif cmd == "stop":
            QCoreApplication.instance().quit()
        else:
            print(f"[tracking] unknown command {command}", flush=True)

    def snapshot(self):
        sides = {}
        for side in SIDES:
            state = dict(self.sides[side])
            worker = self.workers.get(side)
            state["motor"] = bool(worker is not None and worker.isRunning() and worker.motor_attached)
            sides[side] = state
        return dict(type="tracking", t=time.time(), source=self.source, phase=self.phase, error=self.error,
                    found=dict(self.found), sides=sides)

    def shutdown(self):
        """Idle every controller session, stop the workers, then the source."""
        for worker in self.workers.values():
            if worker.isRunning():
                worker.detach_motor()
        for worker in self.workers.values():
            worker.stop()
        for side, worker in self.workers.items():
            worker.wait(5000)
            # the worker's finished signal cannot arrive any more once the event loop has ended: record it here
            self.sides[side].update(connected=False, pose=None, skeleton=None, motor=False, alignment=None)
        if self.hub is not None:
            self.hub.stop()
            self.hub.wait(4000)


def install_fake_source(sides):
    """--fake-sides: a synthetic MANUS glove per side (the test double of hand/tests), for GUI runs without hardware.
    The fake gloves carry the ids of the saved glove alignments, so the retargeter accepts them."""
    import json as _json
    from unittest import mock
    sys.path.insert(0, str(ROOT / "hand" / "tests"))
    from test_both_hands import FakeSource, skeleton
    import teleop.manus.tools.skeleton_source as skeleton_source
    source = FakeSource(period=1 / 100)
    source.gloves, source.skeletons = {}, {}
    for k, side in enumerate(sides, 1):
        alignment = ROOT / "teleop" / "manus" / "alignment" / f"aac_alignment_{side.lower()}.json"
        gid = _json.loads(alignment.read_text())["glove_id"]
        source.gloves[gid] = side
        source.skeletons[gid] = skeleton(side, 100 * k)
    patch = mock.patch.object(skeleton_source, "load_source", return_value=source)
    patch.start()
    return patch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=SOURCES, default="glove")
    parser.add_argument("--state", default="tcp://127.0.0.1:*", help="PUB address to bind (default: a free port)")
    parser.add_argument("--commands", default="tcp://127.0.0.1:*", help="PULL address to bind (default: a free port)")
    parser.add_argument("--publish-hz", type=float, default=PUBLISH_HZ)
    parser.add_argument("--checkpoints", default=None,
                        help='JSON {"glove/Right": "<run>" or null, ...}: the GeoRT runs to use instead of the saved '
                             'selection (the GUI\'s Dev Mode choice; null = the newest run)')
    parser.add_argument("--fake-sides", default=None,
                        help="comma-separated sides tracked by a synthetic glove instead of hardware (tests, demos)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    import zmq
    if args.checkpoints:
        from teleop.manus.geort.checkpoints import set_pins
        pins = {tuple(key.split("/", 1)): run for key, run in json.loads(args.checkpoints).items()}
        set_pins(pins)
        print("[tracking] GeoRT runs: " + ", ".join(f"{s} {side.lower()}: {run or 'newest'}"
                                                    for (s, side), run in sorted(pins.items())), flush=True)
    app = QCoreApplication(sys.argv if argv is None else [sys.argv[0]] + list(argv))
    if args.fake_sides:
        install_fake_source([s.strip().capitalize() for s in args.fake_sides.split(",") if s.strip()])
        if args.source != "glove":
            print("[tracking] --fake-sides: the synthetic glove stands in for the glove source", flush=True)
            args.source = "glove"
    context = zmq.Context()
    state = context.socket(zmq.PUB)
    state.setsockopt(zmq.SNDHWM, 4)
    state.bind(args.state)
    commands = context.socket(zmq.PULL)
    commands.bind(args.commands)
    service = TrackingService(args.source)

    def publish():
        state.send_string(json.dumps(service.snapshot()))

    def poll_commands():
        while True:
            try:
                message = commands.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                service.handle(json.loads(message))
            except Exception as error:                   # a bad command must not take the service down
                print(f"[tracking] command failed: {type(error).__name__}: {error}", flush=True)

    publisher = QTimer()
    publisher.setInterval(int(1000 / args.publish_hz))
    publisher.timeout.connect(publish)
    poller = QTimer()
    poller.setInterval(COMMAND_POLL_MS)
    poller.timeout.connect(poll_commands)
    service.changed.connect(publish)                     # discrete changes go out at once, too
    gc.set_threshold(*GC_THRESHOLDS)
    freezer = QTimer()
    freezer.setInterval(int(GC_FREEZE_S * 1000))
    freezer.timeout.connect(gc.freeze)
    freezer.start()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: app.quit())
    # The GUI that started this service is gone (killed, crashed, or a test cut off): shut down the normal way rather
    # than keep holding the glove / camera, which a new GUI then cannot open. A parent that exits makes this process a
    # child of init / the user's systemd, so its parent id changes.
    parent = os.getppid()

    def check_parent():
        if os.getppid() != parent:
            print("[tracking] the GUI that started this service is gone: shutting down", flush=True)
            app.quit()
    orphan_watch = QTimer()
    orphan_watch.setInterval(PARENT_CHECK_MS)
    orphan_watch.timeout.connect(check_parent)
    orphan_watch.start()
    service.start()
    publisher.start()
    poller.start()
    print(f"TRACKING READY state={state.getsockopt_string(zmq.LAST_ENDPOINT)} "
          f"commands={commands.getsockopt_string(zmq.LAST_ENDPOINT)}", flush=True)
    code = app.exec_()
    publisher.stop()
    poller.stop()
    service.shutdown()
    publish()                                            # the final snapshot: everything disconnected
    state.close(0)
    commands.close(0)
    context.term()
    return code


if __name__ == "__main__":
    sys.exit(main())
