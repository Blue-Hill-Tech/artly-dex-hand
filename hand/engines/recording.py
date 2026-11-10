import os
import sys
from pathlib import Path

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hand.config import load_hand_config

__all__ = [
    "InspireRecorder",
    "InspireRecorderConfig",
    "RealHandRecorder",
    "RealHandRecorderConfig",
    "run_inspire_record",
    "run_realhand_record",
]


def __getattr__(name: str):
    if name in {"InspireRecorder", "InspireRecorderConfig"}:
        from recording.hand_recorder import HandRecorder as InspireRecorder
        from recording.hand_recorder import HandRecorderConfig as InspireRecorderConfig

        return {
            "InspireRecorder": InspireRecorder,
            "InspireRecorderConfig": InspireRecorderConfig,
        }[name]
    if name in {"RealHandRecorder", "RealHandRecorderConfig"}:
        from recording.hand_recorder_rh import HandRecorder as RealHandRecorder
        from recording.hand_recorder_rh import HandRecorderConfig as RealHandRecorderConfig

        return {
            "RealHandRecorder": RealHandRecorder,
            "RealHandRecorderConfig": RealHandRecorderConfig,
        }[name]
    raise AttributeError(name)


def run_inspire_record(args, cfg_path: str) -> None:
    """Instantiate and run Inspire hand recorder using parsed CLI args.

    Mirrors the configuration logic from hand/hand_controller.py.
    """
    from recording.hand_recorder import HandRecorder as InspireRecorder
    from recording.hand_recorder import HandRecorderConfig as InspireRecorderConfig

    output_dir = Path(args.output) if getattr(args, "output", None) else Path("./recording/recordings")
    serial = args.hand_serial or "FTP-001"

    if args.idle_threshold is not None and not (0.0 <= args.idle_threshold <= 1.0):
        raise SystemExit("--idle-threshold must be between 0.0 and 1.0")
    if args.active_threshold is not None and not (0.0 <= args.active_threshold <= 1.0):
        raise SystemExit("--active-threshold must be between 0.0 and 1.0")
    if args.command_timeout is not None and args.command_timeout <= 0:
        raise SystemExit("--command-timeout must be positive")

    cfg = load_hand_config(cfg_path)
    settings = cfg.get("recording_settings", {}) if isinstance(cfg, dict) else {}
    default_rate = settings.get("sampling_rate_hz") if isinstance(settings, dict) else None
    if getattr(args, "rate", None) is not None:
        sampling_rate = float(args.rate)
    elif isinstance(default_rate, (int, float)):
        sampling_rate = float(default_rate)
    else:
        sampling_rate = 120.0
    zmq_port = int(args.zmq_port) if getattr(args, "zmq_port", None) is not None else 5557
    grpc_port = int(args.grpc_port) if getattr(args, "grpc_port", None) is not None else 50089

    idle_thr = args.idle_threshold if args.idle_threshold is not None else 0.2
    active_thr = args.active_threshold if args.active_threshold is not None else 0.7
    cmd_timeout = args.command_timeout if args.command_timeout is not None else 0.5
    smoothing = args.smoothing if args.smoothing is not None else 0.3
    max_change = args.max_change if args.max_change is not None else 50.0
    recovery = args.recovery_rate if args.recovery_rate is not None else 0.95

    cfg = InspireRecorderConfig(
        output_directory=output_dir,
        hand_ip="192.168.11.210",
        hand_port=6000,
        sampling_rate_hz=sampling_rate,
        zmq_port=zmq_port,
        grpc_port=grpc_port,
        telemetry_addr=getattr(args, "telemetry", "") or "",
        enable_zmq=not args.no_zmq,
        enable_grpc=not args.no_grpc,
        enable_console=not args.no_console,
        enable_status_display=not getattr(args, "no_status_display", False),
        enable_motor_protection=not args.no_motor_protection,
        enable_tactile_display=args.show_tactile,
        force_threshold=args.force_threshold if args.force_threshold is not None else 0.5,
        idle_force_threshold=idle_thr,
        active_force_threshold=active_thr,
        command_timeout_sec=cmd_timeout,
        command_smoothing_alpha=smoothing,
        max_angle_change_per_cycle=max_change,
        protection_backoff_recovery_rate=recovery,
        hand_config_file=str(cfg_path),
        hand_role=args.hand_role or "RightHand",
        hand_serial=serial,
    )
    recorder = InspireRecorder(cfg)
    recorder.run()


def run_realhand_record(args, cfg_path: str) -> None:
    """Instantiate and run RealHand L6 recorder using parsed CLI args."""
    from recording.hand_recorder_rh import HandRecorder as RealHandRecorder
    from recording.hand_recorder_rh import HandRecorderConfig as RealHandRecorderConfig

    output_dir = Path(args.output) if getattr(args, "output", None) else Path("./recording/recordings")

    if args.idle_threshold is not None and not (0.0 <= args.idle_threshold <= 1.0):
        raise SystemExit("--idle-threshold must be between 0.0 and 1.0")
    if args.active_threshold is not None and not (0.0 <= args.active_threshold <= 1.0):
        raise SystemExit("--active-threshold must be between 0.0 and 1.0")
    if args.command_timeout is not None and args.command_timeout <= 0:
        raise SystemExit("--command-timeout must be positive")

    cfg = load_hand_config(cfg_path)
    settings = cfg.get("recording_settings", {}) if isinstance(cfg, dict) else {}
    default_rate = settings.get("sampling_rate_hz") if isinstance(settings, dict) else None
    if getattr(args, "rate", None) is not None:
        sampling_rate = float(args.rate)
    elif isinstance(default_rate, (int, float)):
        sampling_rate = float(default_rate)
    else:
        sampling_rate = 120.0
    zmq_port = int(args.zmq_port) if getattr(args, "zmq_port", None) is not None else 5557
    grpc_port = int(args.grpc_port) if getattr(args, "grpc_port", None) is not None else 50089

    idle_thr = args.idle_threshold if args.idle_threshold is not None else 0.2
    active_thr = args.active_threshold if args.active_threshold is not None else 0.7
    cmd_timeout = args.command_timeout if args.command_timeout is not None else 0.5
    smoothing = args.smoothing if args.smoothing is not None else 0.3
    max_change = args.max_change if args.max_change is not None else 50.0
    recovery = args.recovery_rate if args.recovery_rate is not None else 0.95

    cfg = RealHandRecorderConfig(
        output_directory=output_dir,
        can="can0",
        hand_type="right",
        sampling_rate_hz=sampling_rate,
        zmq_port=zmq_port,
        grpc_port=grpc_port,
        telemetry_addr=getattr(args, "telemetry", "") or "",
        enable_zmq=not args.no_zmq,
        enable_grpc=not args.no_grpc,
        enable_console=not args.no_console,
        enable_status_display=not getattr(args, "no_status_display", False),
        enable_motor_protection=not args.no_motor_protection,
        enable_tactile_display=False,
        force_threshold=args.force_threshold if args.force_threshold is not None else 0.5,
        idle_force_threshold=idle_thr,
        active_force_threshold=active_thr,
        command_timeout_sec=cmd_timeout,
        command_smoothing_alpha=smoothing,
        max_angle_change_per_cycle=max_change,
        protection_backoff_recovery_rate=recovery,
        hand_config_file=str(cfg_path),
        hand_role=args.hand_role or "RightHand",
        hand_serial=args.hand_serial or "L6-001",
    )
    recorder = RealHandRecorder(cfg)
    recorder.run()
