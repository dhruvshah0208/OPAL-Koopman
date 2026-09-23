"""JSON config loading shared by the scripts.

Every script takes an optional JSON config path. Keys listed in the script's
``_DEFAULTS`` override the defaults; an ``"explanation"`` block is documentation
only and ignored. Relative paths inside configs are resolved against the project
root by :func:`src_omega_partition.paths.resolve`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def load_json_config(config_path: str | Path, *, required: tuple[str, ...] = ()) -> dict[str, Any]:
    with open(config_path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object.")
    for key in required:
        if key not in raw:
            raise ValueError(f"Config must include '{key}'.")
    return raw


def config_path_from_argv(script_name: str, default: Path | None) -> Path | None:
    """Parse ``[CONFIG.json]`` from ``sys.argv``; return ``default`` when omitted."""
    if len(sys.argv) > 2:
        raise SystemExit(f"usage: {script_name} [CONFIG.json]")
    cfg = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else default
    if cfg is not None and not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")
    return cfg
