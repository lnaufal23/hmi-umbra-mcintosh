"""Read data and output locations from config.yaml.

The file is searched in this order: the path in the environment variable
HMI_UMBRA_CONFIG, then config.yaml at the repository root. Entries under
hosts.<hostname> override the common entries under paths, so one file serves
several machines.
"""
import os
import socket
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]


def load():
    path = Path(os.environ.get("HMI_UMBRA_CONFIG", _ROOT / "config.yaml"))
    with open(path) as f:
        return yaml.safe_load(f)


def get(key):
    """Return the path for <key>, with ~ expanded; host-specific entries take precedence."""
    cfg = load()
    host = (cfg.get("hosts") or {}).get(socket.gethostname(), {}) or {}
    value = host[key] if key in host else cfg["paths"][key]
    return os.path.expanduser(str(value))
