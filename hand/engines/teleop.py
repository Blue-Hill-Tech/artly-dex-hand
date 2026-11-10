import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from hand.config import load_hand_config, resolve_hand_role, resolve_zmq_transport

__all__ = [
    "InspireTeleopController",
    "RealHandTeleopController",
    "load_inspire_teleop_config",
    "load_realhand_teleop_config",
    "run_inspire_teleop",
    "run_realhand_teleop",
    "run_aac_teleop",
]


def __getattr__(name):
    if name in {"InspireTeleopController", "load_inspire_teleop_config"}:
        from teleop.hand_control.hand_controller import HandController, load_config
        return HandController if name == "InspireTeleopController" else load_config
    if name in {"RealHandTeleopController", "load_realhand_teleop_config"}:
        from teleop.hand_control.hand_controller_rh import HandController, load_config
        return HandController if name == "RealHandTeleopController" else load_config
    raise AttributeError(name)


def run_inspire_teleop(args, cfg_path: str) -> None:
    """Instantiate and start Inspire teleop controller using parsed CLI args.

    This mirrors the logic previously embedded in hand/hand_controller.py, but uses
    shared helpers from hand.config to interpret the JSON config.
    """
    from teleop.hand_control.hand_controller import HandController as InspireTeleopController

    cfg = load_hand_config(cfg_path)
    serial = getattr(args, "hand_serial", None) or "FTP-001"
    role_info = resolve_hand_role(cfg, serial)
    transport = resolve_zmq_transport(cfg)

    role_hand_id = role_info.get("role") if isinstance(role_info, dict) else None

    hand_ip = (role_info.get("ip") if isinstance(role_info, dict) else "192.168.11.210") or "192.168.11.210"
    hand_port = int(role_info.get("port") if isinstance(role_info, dict) else 6000)

    stream_addr = transport.get("hand_stream_pull", "tcp://0.0.0.0:6300")
    control_addr = transport.get("hand_control_rep", "tcp://0.0.0.0:6301")
    telemetry_addr = transport.get("hand_telem_pub", "tcp://0.0.0.0:6302")

    dt = float(args.dt) if getattr(args, "dt", None) is not None else 1.0 / 120.0
    telem_rate = int(args.telem_rate) if getattr(args, "telem_rate", None) is not None else 60
    cli_hand_role = getattr(args, "hand_role", None) or None
    hand_id = (getattr(args, "hand_id", None) or cli_hand_role or role_hand_id or "RightHand")

    ctl = InspireTeleopController(
        hand_ip=hand_ip,
        hand_port=hand_port,
        stream_addr=stream_addr,
        control_addr=control_addr,
        telemetry_addr=telemetry_addr,
        dt=dt,
        telemetry_rate_hz=telem_rate,
        queue_horizon_s=6.0,
        timeout_s=6.0,
        hand_id=hand_id,
    )
    if getattr(args, "idle_to_default_s", None) is not None:
        try:
            ctl.idle_to_default_s = float(args.idle_to_default_s)
        except Exception:
            pass
    try:
        ctl.verbose = bool(getattr(args, "verbose", False))
        ctl.quiet = bool(getattr(args, "quiet", False))
        if getattr(args, "display_rate", None) is not None:
            ctl.display_rate_hz = float(args.display_rate)
        if getattr(args, "exec_speed", None) is not None and args.exec_speed > 0:
            ctl.exec_speed = float(args.exec_speed)
    except Exception:
        pass
    ctl.start()


def run_realhand_teleop(args, cfg_path: str) -> None:
    """Instantiate and start RealHand L6 teleop controller using parsed CLI args."""
    from teleop.hand_control.hand_controller_rh import HandController as RealHandTeleopController

    cfg = load_hand_config(cfg_path)
    serial = getattr(args, "hand_serial", None) or "L6-001"
    role_info = resolve_hand_role(cfg, serial)
    transport = resolve_zmq_transport(cfg)

    if not role_info:
        keys = []
        try:
            roles = cfg.get("hand_roles", {}) if isinstance(cfg, dict) else {}
            if isinstance(roles, dict):
                keys = list(roles.keys())
        except Exception:
            keys = []
        print(f"[hand] warning: no hand_roles entry found for serial={serial} in config={cfg_path}; available_serials={keys}")

    role_hand_id = role_info.get("role") if isinstance(role_info, dict) else None
    role_hand_type = role_info.get("hand_type") if isinstance(role_info, dict) else None
    cli_hand_role = getattr(args, "hand_role", None) or None

    can_if = (role_info.get("can") if isinstance(role_info, dict) else "can0") or "can0"
    hand_port = 0

    stream_addr = transport.get("hand_stream_pull", "tcp://0.0.0.0:6300")
    control_addr = transport.get("hand_control_rep", "tcp://0.0.0.0:6301")
    telemetry_addr = transport.get("hand_telem_pub", "tcp://0.0.0.0:6302")

    dt = float(args.dt) if getattr(args, "dt", None) is not None else 1.0 / 120.0
    telem_rate = int(args.telem_rate) if getattr(args, "telem_rate", None) is not None else 60
    hand_id = (getattr(args, "hand_id", None) or cli_hand_role or role_hand_id or "RightHand")

    ctl = RealHandTeleopController(
        hand_ip=can_if,
        hand_port=hand_port,
        stream_addr=stream_addr,
        control_addr=control_addr,
        telemetry_addr=telemetry_addr,
        dt=dt,
        telemetry_rate_hz=telem_rate,
        queue_horizon_s=6.0,
        timeout_s=6.0,
        hand_id=hand_id,
    )
    effective_hand_type = role_hand_type
    if isinstance(hand_id, str):
        inferred_from_id = None
        if hand_id.lower().startswith("left"):
            inferred_from_id = "left"
        elif hand_id.lower().startswith("right"):
            inferred_from_id = "right"
        if cli_hand_role and inferred_from_id and role_hand_type and str(role_hand_type).lower() != inferred_from_id:
            print(f"[hand] warning: config hand_type={role_hand_type} disagrees with --hand_role={cli_hand_role}; using {inferred_from_id}")
        if not effective_hand_type:
            effective_hand_type = inferred_from_id
        elif cli_hand_role and inferred_from_id:
            effective_hand_type = inferred_from_id

    if effective_hand_type:
        try:
            ctl.hand_type = str(effective_hand_type)
        except Exception:
            pass

    limits_cfg = {}
    if isinstance(cfg, dict):
        global_limits = cfg.get("angle_limits", {})
        if isinstance(global_limits, dict):
            limits_cfg.update(global_limits)
    if isinstance(role_info, dict):
        role_limits = role_info.get("angle_limits", {})
        if isinstance(role_limits, dict):
            limits_cfg.update(role_limits)
    try:
        thumb_cmc_max = limits_cfg.get("thumb_cmc_max", None)
        if thumb_cmc_max is not None:
            ctl.angle_limits[5] = (
                float(ctl.angle_limits[5][0]),
                max(float(ctl.angle_limits[5][0]), min(1000.0, float(thumb_cmc_max))),
            )
    except Exception:
        pass

    if bool(getattr(args, "verbose", False)):
        print(f"[hand] resolved: serial={serial} hand_id={hand_id} hand_type={getattr(ctl, 'hand_type', None)} can={can_if} config={cfg_path}")
    if getattr(args, "idle_to_default_s", None) is not None:
        try:
            ctl.idle_to_default_s = float(args.idle_to_default_s)
        except Exception:
            pass
    try:
        ctl.verbose = bool(getattr(args, "verbose", False))
        ctl.quiet = bool(getattr(args, "quiet", False))
        if getattr(args, "display_rate", None) is not None:
            ctl.display_rate_hz = float(args.display_rate)
        if getattr(args, "exec_speed", None) is not None and args.exec_speed > 0:
            ctl.exec_speed = float(args.exec_speed)
    except Exception:
        pass
    ctl.start()


def run_aac_teleop(args, cfg_path: str) -> None:
    from hand.backends.aac import AACHand
    from teleop.hand_control.hand_controller_aac import AACController, load_aac_config

    settings = load_aac_config(args, cfg_path)
    device = AACHand(settings.pop("serial_device"), settings.pop("device_address"),
                     settings.pop("connection_timeout_s"), bus=settings.pop("bus", "canfd"))
    AACController(device, **settings).start()
