"""Whether the MANUS glove can work on this computer, without loading the SDK (the GUI asks before offering Glove mode).

The glove needs the MANUS SDK beside agi-os (../MANUS_Core_3.2.0_SDK_Linux/Python, the folder stream_gloves.py loads),
with a compiled module for this Python and machine, and on Linux the dongle's udev rule (without it only root may open
the dongle; ./setup_envs.sh hand installs it).
"""
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]                       # agi-os (this file: teleop/manus/stream/)
SDK_PYTHON = ROOT.parent / "MANUS_Core_3.2.0_SDK_Linux" / "Python"   # the same folder as stream_gloves.SDK_PYTHON
UDEV_DIRS = (Path("/etc/udev/rules.d"), Path("/lib/udev/rules.d"), Path("/usr/lib/udev/rules.d"))
MANUS_VENDOR = 'idVendor}=="3325"'


def _udev_rule_present():
    for folder in UDEV_DIRS:
        try:
            for rule in folder.glob("*.rules"):
                if MANUS_VENDOR in rule.read_text(errors="ignore"):
                    return True
        except OSError:
            continue
    return False


def glove_unavailable(sdk_python=SDK_PYTHON):
    """Why the glove cannot work here (a sentence for the user), or None when nothing is missing."""
    sdk_python = Path(sdk_python)
    if not (sdk_python / "manus_sdk" / "__init__.py").exists():
        return (f"The MANUS SDK is not installed. Copy the MANUS_Core_3.2.0_SDK_Linux folder into "
                f"{sdk_python.parent.parent}, then restart.")
    if sys.platform.startswith("linux"):
        arch = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "aarch64", "arm64": "aarch64"}.get(
            platform.machine().lower(), platform.machine().lower())
        tag = f"cpython-{sys.version_info.major}{sys.version_info.minor}"
        if not list((sdk_python / "manus_sdk" / "linux" / arch).glob(f"_manus_sdk.{tag}-*.so")):
            return (f"The MANUS SDK has no build for Python {sys.version_info.major}.{sys.version_info.minor} on "
                    f"{arch}.")
        if any(folder.is_dir() for folder in UDEV_DIRS) and not _udev_rule_present():
            return ("The MANUS dongle is not allowed to be opened (its USB rule is missing). Run ./setup_envs.sh hand, "
                    "then unplug and replug the dongle.")
    return None
