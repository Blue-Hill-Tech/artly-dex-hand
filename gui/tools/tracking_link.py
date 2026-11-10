"""The GUI's side of the tracking service (gui/tools/tracking_service.py): start it as a child process, read its state
snapshots, send it commands.

TrackingLink starts the service with the GUI's own interpreter, waits for its "TRACKING READY ..." line, connects a
SUB socket (conflated: the GUI always sees the newest snapshot, never a backlog) read by a small thread that emits
`state`, and a PUSH socket for commands. Commands sent before the service is ready are queued.
"""
import json
import sys
import threading
from pathlib import Path

from PyQt5.QtCore import QObject, QProcess, QThread, pyqtSignal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                             # agi-os (this file: gui/tools/)
SERVICE = HERE / "tracking_service.py"
READY_PREFIX = "TRACKING READY "
STOP_WAIT_MS = 6000           # the service idles the controller sessions and shuts the source down first


class StateReader(QThread):
    state = pyqtSignal(dict)

    def __init__(self, address, parent=None):
        super().__init__(parent)
        self.address = address
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def run(self):
        import zmq
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.setsockopt_string(zmq.SUBSCRIBE, "")
        socket.connect(self.address)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._stop.is_set():
                if dict(poller.poll(20)):
                    try:
                        self.state.emit(json.loads(socket.recv_string()))
                    except ValueError:
                        pass
        finally:
            socket.close(0)
            context.term()


class TrackingLink(QObject):
    state = pyqtSignal(dict)      # the newest snapshot (see tracking_service.py for its shape)
    ready = pyqtSignal()          # the service is up (sockets connected)
    stopped = pyqtSignal(int)     # the service exited (code)
    output = pyqtSignal(str)      # its other output lines

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setWorkingDirectory(str(ROOT))
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(self._finished)
        self.reader = None
        self.context = self.push = None
        self.queued = []
        self.source = None
        self._buffer = ""

    def running(self):
        return self.process.state() != QProcess.NotRunning

    def command(self, source, fake_sides=None, checkpoints=None):
        """checkpoints: {(source group, side): GeoRT run name or None (newest)} to use instead of the saved selection."""
        arguments = [sys.executable, "-u", str(SERVICE), "--source", source]
        if fake_sides:
            arguments += ["--fake-sides", ",".join(fake_sides)]
        if checkpoints is not None:
            arguments += ["--checkpoints", json.dumps({f"{group}/{side}": run for (group, side), run in checkpoints.items()})]
        return arguments

    def start(self, source, fake_sides=None, checkpoints=None):
        self.source = source
        program, *arguments = self.command(source, fake_sides, checkpoints)
        self.process.start(program, arguments)

    def send(self, **command):
        """A command to the service (queued until it is ready)."""
        if self.push is None:
            self.queued.append(command)
            return
        self.push.send_string(json.dumps(command))

    def stop(self, wait=False):
        """SIGTERM: the service idles the controllers, stops its workers and source, publishes a last snapshot and
        exits; killed if it has not after STOP_WAIT_MS."""
        if not self.running():
            return
        self.process.terminate()
        if wait and not self.process.waitForFinished(STOP_WAIT_MS):
            self.process.kill()
            self.process.waitForFinished(1000)

    # --- internals ------------------------------------------------------------------------------------------------
    def _read(self):
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            line = line.rstrip()
            if line.startswith(READY_PREFIX):
                self._connect(line[len(READY_PREFIX):])
            elif line.strip():
                self.output.emit(f"[tracking] {line}")

    def _connect(self, announce):
        import zmq
        addresses = dict(part.split("=", 1) for part in announce.split())
        self.context = zmq.Context()
        self.push = self.context.socket(zmq.PUSH)
        self.push.setsockopt(zmq.LINGER, 500)
        self.push.connect(addresses["commands"])
        self.reader = StateReader(addresses["state"], self)
        self.reader.state.connect(self.state)
        self.reader.start()
        for command in self.queued:
            self.push.send_string(json.dumps(command))
        self.queued.clear()
        self.ready.emit()

    def _finished(self, code, _status):
        if self.reader is not None:
            self.reader.stop()
            self.reader.wait(2000)
            self.reader = None
        if self.push is not None:
            self.push.close(0)
            self.context.term()
            self.push = self.context = None
        self.stopped.emit(code)
