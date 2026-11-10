"""The hand controller (hand.hand_controller, teleop mode, AAC torque over CAN FD) as a child process of the GUI.

Start: the same command as in a terminal, with the side's torque CAN FD config. Stop: SIGTERM, which the controller
handles like Ctrl-C (zero torque, position mode at the measured pose, close the bus). Its output goes to the GUI's
terminal, prefixed with the side.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

from PyQt5.QtCore import QObject, QProcess, QProcessEnvironment, pyqtSignal

ROOT = Path(__file__).resolve().parents[2]       # agi-os (this file: gui/tools/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIGS = {"Left": ("teleop/configure/acc/torque_control/hand_config_aac_l_torque_canfd.json", "AAC-001"),
           "Right": ("teleop/configure/acc/torque_control/hand_config_aac_r_torque_canfd.json", "AAC-002")}
STOP_WAIT_MS = 4000       # time the controller gets to leave torque mode before it is killed
ADAPTER_DIR = "/dev/serial/by-id"   # CAN FD / RS-485 adapters, named after their USB serial number


def connect_address(bind_address, host="127.0.0.1"):
    """The controller binds tcp://0.0.0.0:PORT; a client connects to tcp://HOST:PORT."""
    if bind_address.startswith("tcp://0.0.0.0:") or bind_address.startswith("tcp://*:"):
        return f"tcp://{host}:{bind_address.rsplit(':', 1)[1]}"
    return bind_address


def default_adapter(side):
    """The adapter path in the side's config (what the hand uses unless Dev Mode picks another)."""
    import os
    from teleop.hand_control.hand_controller_aac import load_aac_config
    config, serial = CONFIGS[side]
    saved = os.environ.pop("AAC_SERIAL_DEVICE", None)         # the config's own value, not an override
    try:
        return load_aac_config(SimpleNamespace(hand_serial=serial, telem_rate=None), str(ROOT / config))["serial_device"]
    finally:
        if saved is not None:
            os.environ["AAC_SERIAL_DEVICE"] = saved


def adapter_config_files(side):
    """The configs that name this hand's adapter: its teleop torque config (motor, position demo, calibration) and,
    when it exists, the calibrated torque-demo config (demo/configure/hand_config_aac_<l|r>_torque_canfd_demo.json)."""
    files = [ROOT / CONFIGS[side][0]]
    demo = ROOT / "demo" / "configure" / f"hand_config_aac_{side[0].lower()}_torque_canfd_demo.json"
    if demo.exists():
        files.append(demo)
    return files


def save_default_adapter(side, path):
    """Make `path` this hand's default adapter: its serial_device in every config of adapter_config_files(side) (only
    that value changes; the files keep their layout). Returns the files changed, relative to the repo."""
    import json
    import re
    serial = CONFIGS[side][1]
    changed = []
    for config in adapter_config_files(side):
        text = config.read_text()
        data = json.loads(text)
        role = data.get("hand_roles", {}).get(serial)
        if not isinstance(role, dict) or "serial_device" not in role:
            raise ValueError(f"{config.relative_to(ROOT)} has no hand_roles.{serial}.serial_device")
        if role["serial_device"] == path:
            continue
        pattern = re.compile(r'("serial_device"\s*:\s*)' + re.escape(json.dumps(role["serial_device"])))
        new_text, count = pattern.subn(lambda m: m.group(1) + json.dumps(path), text)
        if count != 1:
            raise ValueError(f"{config.relative_to(ROOT)}: expected one serial_device line, found {count}")
        data["hand_roles"][serial]["serial_device"] = path
        if json.loads(new_text) != data:                     # only that one value may have changed
            raise ValueError(f"{config.relative_to(ROOT)}: editing serial_device changed more than it should")
        config.write_text(new_text)
        changed.append(str(config.relative_to(ROOT)))
    return changed


def config_gains(side):
    """The hand's teleop torque gains from its config: dict(kp, kd, joints={joint: {kp/kd}}) (joints: the per-joint
    overrides, which keep their own values whatever the global gains are)."""
    import json
    torque = json.loads((ROOT / CONFIGS[side][0]).read_text()).get("torque") or {}
    joints = {name: {k: v for k, v in values.items() if k in ("kp", "kd")}
              for name, values in (torque.get("joints") or {}).items() if "kp" in values or "kd" in values}
    return dict(kp=float(torque.get("kp", 3.0)), kd=float(torque.get("kd", 0.02)), joints=joints)


def save_default_gains(side, kp, kd):
    """Make kp / kd the hand's default teleop torque gains: torque.kp and torque.kd in its teleop config (only those two
    numbers change; the file keeps its layout). Returns the file changed, relative to the repo, or None if unchanged."""
    import json
    import re
    config = ROOT / CONFIGS[side][0]
    text = config.read_text()
    data = json.loads(text)
    torque = data.get("torque")
    if not isinstance(torque, dict) or "kp" not in torque or "kd" not in torque:
        raise ValueError(f"{config.relative_to(ROOT)} has no torque.kp / torque.kd")
    if (float(torque["kp"]), float(torque["kd"])) == (float(kp), float(kd)):
        return None
    for key, value in (("kp", kp), ("kd", kd)):
        # the torque block's own key: one indent level deeper than "torque" (per-joint overrides sit deeper still)
        torque_indent = re.search(r'^( *)"torque"\s*:', text, re.M)[1]
        pattern = re.compile(r'^(' + torque_indent + r'  "' + key + r'"\s*:\s*)(-?[0-9.eE+-]+)', re.M)
        text, count = pattern.subn(lambda m: m.group(1) + json.dumps(float(value)), text)
        if count != 1:
            raise ValueError(f"{config.relative_to(ROOT)}: expected one torque.{key} line, found {count}")
    torque.update(kp=float(kp), kd=float(kd))
    if json.loads(text) != data:                             # only those two numbers may have changed
        raise ValueError(f"{config.relative_to(ROOT)}: editing the gains changed more than it should")
    config.write_text(text)
    return str(config.relative_to(ROOT))


def list_adapters(directory=ADAPTER_DIR):
    """Every adapter plugged in now: full paths in /dev/serial/by-id (empty when there is none)."""
    import os
    return [os.path.join(directory, name) for name in sorted(os.listdir(directory))] if os.path.isdir(directory) else []


def adapter_for(side, chosen=None):
    """(the adapter this hand uses: `chosen` (Dev Mode) or the one in the side's config, whether that exact adapter is
    plugged in, the adapters that are). Exact: the backend would fall back to the one other adapter of the same model,
    which, with one adapter per hand, can be the OTHER hand's; the GUI refuses that and asks for this hand's adapter."""
    import os
    path = chosen or default_adapter(side)
    directory = os.path.dirname(path)
    attached = sorted(os.listdir(directory)) if os.path.isdir(directory) else []
    return path, os.path.exists(path), attached


def adapter_name(path):
    """The short name of an adapter: its serial number (usb-..._<serial>-if00), or the file name."""
    name = Path(path).name
    stem = name.rpartition("-")[0] or name
    return stem.rpartition("_")[2] or name


def sockets_for(side):
    """The client sockets and hand id of the side's controller, read from its config."""
    from teleop.hand_control.hand_controller_aac import load_aac_config
    config, serial = CONFIGS[side]
    settings = load_aac_config(SimpleNamespace(hand_serial=serial), str(ROOT / config))
    return dict(stream=connect_address(settings["stream_addr"]), control=connect_address(settings["control_addr"]),
                telemetry=connect_address(settings["telemetry_addr"]), hand_id=settings["hand_id"])


def set_adapter(process, adapter, gains=None):
    """Start `process` with AAC_SERIAL_DEVICE = adapter (load_aac_config uses it instead of the config's), or without
    it (the config's adapter) when adapter is None; and with AAC_TORQUE_GAINS = gains ({"kp", "kd"}: the motor's
    torque gains instead of the config's global ones) when given."""
    import json
    environment = QProcessEnvironment.systemEnvironment()
    environment.remove("AAC_SERIAL_DEVICE")
    environment.remove("AAC_TORQUE_GAINS")
    if adapter:
        environment.insert("AAC_SERIAL_DEVICE", adapter)
    if gains:
        environment.insert("AAC_TORQUE_GAINS", json.dumps({k: float(v) for k, v in gains.items()}))
    process.setProcessEnvironment(environment)


class MotorProcess(QObject):
    output = pyqtSignal(str)
    stopped = pyqtSignal(int)          # exit code

    def __init__(self, side, parent=None, adapter=None, gains=None):
        """gains: {"kp": .., "kd": ..} for this run (Dev Mode), instead of the config's global torque gains."""
        super().__init__(parent)
        self.side, self.adapter, self.gains = side, adapter, gains
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.setWorkingDirectory(str(ROOT))
        set_adapter(self.process, adapter, gains)
        self.process.readyReadStandardOutput.connect(self._read)
        self.process.finished.connect(lambda code, _status: self.stopped.emit(code))

    def running(self):
        return self.process.state() != QProcess.NotRunning

    def command(self):
        config, serial = CONFIGS[self.side]
        return [sys.executable, "-u", "-m", "hand.hand_controller", "--mode", "teleop", "--model", "aac",
                "--config", config, "--hand_serial", serial]

    def start(self):
        program, *arguments = self.command()
        self.output.emit(f"[{self.side} motor] {' '.join(self.command())}"
                         + (f"  (adapter {self.adapter})" if self.adapter else "")
                         + (f"  (kp {self.gains['kp']:g}, kd {self.gains['kd']:g})" if self.gains else ""))
        self.process.start(program, arguments)

    def stop(self, wait=False):
        if not self.running():
            return
        self.process.terminate()                       # SIGTERM: the controller leaves torque mode and exits
        if wait and not self.process.waitForFinished(STOP_WAIT_MS):
            self.process.kill()
            self.process.waitForFinished(1000)

    def _read(self):
        text = bytes(self.process.readAllStandardOutput()).decode(errors="replace")
        for line in text.splitlines():
            if line.strip():
                self.output.emit(f"[{self.side} motor] {line}")
