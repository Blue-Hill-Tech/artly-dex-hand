import json
import os
from typing import Any, Dict, Optional


def load_hand_config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(f"Hand config not found: {path}") from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON in hand config: {path}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to load hand config: {path}: {e}") from e


def resolve_hand_role(cfg: Dict[str, Any], serial: str) -> Dict[str, Any]:
    roles = cfg.get("hand_roles", {}) if isinstance(cfg, dict) else {}
    if not isinstance(roles, dict):
        return {}
    return roles.get(serial, {}) if serial in roles else {}


def resolve_zmq_transport(cfg: Dict[str, Any]) -> Dict[str, str]:
    if not isinstance(cfg, dict):
        return {}
    t = cfg.get("transport", {})
    if not isinstance(t, dict):
        return {}
    z = t.get("zmq", {})
    return z if isinstance(z, dict) else {}
