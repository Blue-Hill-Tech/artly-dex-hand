import argparse
import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hand.engines.teleop import (
    run_inspire_teleop as engine_run_inspire_teleop,
    run_realhand_teleop as engine_run_realhand_teleop,
    run_aac_teleop as engine_run_aac_teleop,
)
from hand.engines.recording import (
    run_inspire_record as engine_run_inspire_record,
    run_realhand_record as engine_run_realhand_record,
)


def _default_config_path(mode: str, model: str) -> str:
    if model == "aac":
        if mode != "teleop":
            raise ValueError("AAC recording is unsupported")
        # the left hand's torque-mode CAN FD config (the one the demo GUI runs; the plain position configs are gone)
        subdir, name = os.path.join("teleop", "configure", "acc", "torque_control"), "hand_config_aac_l_torque_canfd.json"
    elif model == "inspire":
        subdir = os.path.join("teleop", "configure", "inspire") if mode == "teleop" else "recording"
        name = "hand_config.json"
    else:
        subdir = os.path.join("teleop", "configure", "rh") if mode == "teleop" else "recording"
        name = "hand_config_rh_r.json" if mode == "teleop" else "hand_config_rh.json"
    return os.path.join(PROJECT_ROOT, subdir, name)


def _run_inspire_teleop(args: argparse.Namespace) -> None:
    cfg_path = args.config or _default_config_path("teleop", "inspire")
    engine_run_inspire_teleop(args, cfg_path)


def _run_realhand_teleop(args: argparse.Namespace) -> None:
    cfg_path = args.config or _default_config_path("teleop", "realhand")
    engine_run_realhand_teleop(args, cfg_path)


def _run_inspire_record(args: argparse.Namespace) -> None:
    cfg_path = args.config or _default_config_path("record", "inspire")
    engine_run_inspire_record(args, cfg_path)


def _run_realhand_record(args: argparse.Namespace) -> None:
    cfg_path = args.config or _default_config_path("record", "realhand")
    engine_run_realhand_record(args, cfg_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified hand controller/recorder front-end")
    parser.add_argument("--mode", choices=["teleop", "record"], required=True, help="Operating mode")
    parser.add_argument("--model", choices=["inspire", "realhand", "aac"], required=True, help="Hand model/backend")
    parser.add_argument("--config", type=str, default="", help="Path to hand config JSON (defaults by mode/model)")
    parser.add_argument("--hand_serial", type=str, default="", help="Hand serial for role/config lookup")

    parser.add_argument("--hand_id", type=str, default="", help="Logical hand id (e.g., RightHand)")
    parser.add_argument("--dt", type=float, default=None, help="Teleop control period (s)")
    parser.add_argument("--telem_rate", type=int, default=None, help="Telemetry rate (Hz)")
    parser.add_argument("--idle_to_default_s", type=float, default=None, help="Idle time before default pose (teleop)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs for teleop")
    parser.add_argument("--quiet", action="store_true", help="Suppress periodic state display in teleop")
    parser.add_argument("--display_rate", type=float, default=None, help="Teleop display refresh rate (Hz)")
    parser.add_argument("--exec_speed", type=float, default=None, help="Teleop execution speed scale")

    parser.add_argument("--output", type=str, default="", help="Recording output root directory")
    parser.add_argument("--rate", type=float, default=None, help="Recording control loop frequency (Hz)")
    parser.add_argument("--zmq_port", type=int, default=None, help="Recording ZMQ control port")
    parser.add_argument("--grpc_port", type=int, default=None, help="Recording GRPC port")
    parser.add_argument("--telemetry", type=str, default="", help="Recording GUI telemetry PUB endpoint")
    parser.add_argument("--no_zmq", action="store_true", help="Disable ZMQ control in recording mode")
    parser.add_argument("--no_grpc", action="store_true", help="Disable GRPC glove interface in recording mode")
    parser.add_argument("--no_console", action="store_true", help="Disable console control in recording mode")
    parser.add_argument("--no_status_display", action="store_true", help="Disable terminal status visualization in recording mode")
    parser.add_argument("--no_motor_protection", action="store_true", help="Disable motor protection in recording mode")
    parser.add_argument("--show_tactile", action="store_true", help="Enable tactile display for Inspire recording")

    parser.add_argument("--force_threshold", type=float, default=None, help="Deprecated global force threshold")
    parser.add_argument("--idle_threshold", type=float, default=None, help="Idle force threshold (0-1)")
    parser.add_argument("--active_threshold", type=float, default=None, help="Active force threshold (0-1)")
    parser.add_argument("--command_timeout", type=float, default=None, help="Glove command timeout (s)")
    parser.add_argument("--smoothing", type=float, default=None, help="Command smoothing factor (0-1)")
    parser.add_argument("--max_change", type=float, default=None, help="Max angle change per cycle")
    parser.add_argument("--recovery_rate", type=float, default=None, help="Protection recovery rate (0-1)")
    parser.add_argument("--hand_role", type=str, default="", help="Hand role label (RightHand/LeftHand)")

    args = parser.parse_args()

    if args.model == "aac":
        if args.mode != "teleop":
            parser.error("AAC recording is unsupported; use --mode teleop")
        supported = {"--mode", "--model", "--config", "--hand_serial", "--hand_id",
                     "--hand_role", "--dt", "--telem_rate", "--command_timeout"}
        supplied = {arg.split("=", 1)[0] for arg in sys.argv[1:] if arg.startswith("--")}
        unsupported = supplied - supported
        if unsupported:
            parser.error("AAC does not support: " + ", ".join(sorted(unsupported)))
        engine_run_aac_teleop(args, args.config or _default_config_path("teleop", "aac"))
        return

    if args.mode == "teleop":
        if args.model == "inspire":
            _run_inspire_teleop(args)
        elif args.model == "realhand":
            _run_realhand_teleop(args)
    else:
        if args.model == "inspire":
            _run_inspire_record(args)
        elif args.model == "realhand":
            _run_realhand_record(args)


if __name__ == "__main__":
    main()
