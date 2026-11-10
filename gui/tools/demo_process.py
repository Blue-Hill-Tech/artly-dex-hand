"""The hand demos as child processes of the GUI: the position demo (demo/mcp_ellipses.py, fingertip ellipses on the
MCP joints through the firmware's position loop) and the torque demo (demo/impedance_ellipses.py, the same ellipses
under Cartesian impedance in torque mode). Both read their ellipses from demo/parameters/ellipses_trajectory.yaml.

Each runs for one hand with that hand's config (the same file the motor uses, gui/motor_process.CONFIGS) and loops
until stopped. Stop sends SIGINT, the demos' Ctrl-C: mcp_ellipses eases back to rest; impedance_ellipses writes zero
torque, returns to position mode where the hand stands and eases back to rest. A demo still running after
STOP_WAIT_MS is killed (only on window close, when the GUI waits). A demo opens the hand's adapter itself, so it
cannot run together with that hand's motor (the controller) or its other demo.
"""
import os
import signal
import sys
from pathlib import Path

from PyQt5.QtCore import QObject, QProcess, pyqtSignal

ROOT = Path(__file__).resolve().parents[2]       # agi-os (this file: gui/tools/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# kind -> (script, button label, label while it runs, how messages name it)
DEMOS = {"position": ("demo/mcp_ellipses.py", "Position demo", "Stop the position demo", "the position demo"),
         "torque": ("demo/impedance_ellipses.py", "Torque demo", "Stop the torque demo", "the torque demo"),
         "calibrate": ("demo/tools/calibrate_torque_demo.py", "Calibrate torque demo", "Stop calibrating", "the calibration")}
LOOPING = ("position", "torque")     # run until stopped; the calibration ends by itself
TIP_STOP_KINDS = ("torque", "calibrate")   # have the fingertip-distance safety stop (Dev Mode can turn it off)
STOP_WAIT_MS = 15000          # easing back to rest takes a few seconds (the impedance demo leaves torque mode first)


def torque_demo_config(side):
    """The torque demo's config: the calibrated demo config (demo/tools/calibrate_torque_demo.py) when it exists, else
    the hand's teleop torque config. Teleop (Start the motor) always uses the teleop config."""
    from motor_process import CONFIGS
    calibrated = ROOT / "demo" / "configure" / f"hand_config_aac_{side[0].lower()}_torque_canfd_demo.json"
    return f"demo/configure/{calibrated.name}" if calibrated.exists() else CONFIGS[side][0]


class DemoProcess(QObject):
    output = pyqtSignal(str)
    stopped = pyqtSignal(int)          # exit code

    def __init__(self, side, kind, parent=None, adapter=None, tip_stop=True):
        super().__init__(parent)
        if kind not in DEMOS:
            raise ValueError(f"demo kind must be one of {sorted(DEMOS)}")
        self.side, self.kind, self.adapter = side, kind, adapter
        self.tip_stop = tip_stop                           # Dev Mode: the torque demo's fingertip-distance safety stop
        self.label, self.running_label, self.noun = DEMOS[kind][1], DEMOS[kind][2], DEMOS[kind][3]
        self.stop_requested = False
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setWorkingDirectory(str(ROOT))
        from motor_process import set_adapter
        set_adapter(self.process, adapter)                 # Dev Mode's adapter for this hand (AAC_SERIAL_DEVICE)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(lambda code, _status: self.stopped.emit(code))
        self._buffer = ""

    def running(self):
        return self.process.state() != QProcess.NotRunning

    def command(self):
        from motor_process import CONFIGS
        config, serial = CONFIGS[self.side]
        script = DEMOS[self.kind][0]
        if self.kind == "torque":
            config = torque_demo_config(self.side)
        command = [sys.executable, "-u", script, "--config", config, "--hand_serial", serial]
        if self.kind in LOOPING:
            command += ["--loops", "0"]
        if self.kind in TIP_STOP_KINDS and not self.tip_stop:
            command.append("--no-tip-stop")
        return command

    def start(self):
        program, *arguments = self.command()
        self.output.emit(f"[{self.side} {self.kind} demo] {' '.join(self.command())}"
                         + (f"  (adapter {self.adapter})" if self.adapter else ""))
        self.process.start(program, arguments)

    def stop(self, wait=False):
        """SIGINT (the demo's Ctrl-C: back to rest, out of torque mode); killed if it has not ended after
        STOP_WAIT_MS when waiting."""
        if not self.running():
            return
        self.stop_requested = True
        pid = int(self.process.processId())
        if pid > 0:
            try:
                os.kill(pid, signal.SIGINT)
            except ProcessLookupError:
                return
        if wait and not self.process.waitForFinished(STOP_WAIT_MS):
            self.process.kill()
            self.process.waitForFinished(1000)

    def _read(self):
        self._buffer += bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        *lines, self._buffer = self._buffer.split("\n")
        for line in lines:
            if line.strip():
                self.output.emit(f"[{self.side} {self.kind} demo] {line.rstrip()}")
