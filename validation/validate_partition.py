"""
Validate a saved graph-merged partition on labeled validation snapshots.

Anchor points (one per class) establish which partition ID corresponds to each
semantic class. Validation accuracy is measured by whether each snapshot lands
in the same partition as its class anchor.

Run:
    python validation/validate_partition.py
    python validation/validate_partition.py configs/duffing/validate_partition.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_VALIDATION_DIR = Path(__file__).resolve().parent
_ROOT = _VALIDATION_DIR.parent
_KOOPMAN_ROOT = _ROOT.parent
_SRC_DIR = _ROOT / "src_deep_koopman"
_PRUNING_ROOT = _KOOPMAN_ROOT / "MPV-SPV-pruning-code"
for _path in (_ROOT, _SRC_DIR, _PRUNING_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src_basin_classification import load_partition_inference  # noqa: E402

if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))
from partition_classification import (  # noqa: E402
    apply_anchor_config,
    classify_with_anchors,
    print_anchor_assignments,
    print_classification_table,
    resolve_path,
)

_DEFAULTS = {
    "partition_path": _ROOT / "data/duffing/partition_output.npz",
    "validation_data_path": _ROOT / "data/duffing/duffing_validation_data.npz",
    "target_equilibria": [[-1.0, 0.0], [1.0, 0.0]],
    "target_data_paths": None,
    "exclude_dropped_predictions": True,
}


def _settings(config_path: Path | None) -> dict[str, Any]:
    s = dict(_DEFAULTS)
    if config_path is None:
        return s

    with open(config_path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object.")

    for key in _DEFAULTS:
        if key in raw:
            s[key] = raw[key]

    apply_anchor_config(raw, s)

    s["partition_path"] = Path(s["partition_path"])
    s["validation_data_path"] = Path(s["validation_data_path"])
    s["exclude_dropped_predictions"] = bool(s["exclude_dropped_predictions"])
    return s


def _load_validation_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    if "X" not in data:
        raise KeyError(f"{path} does not contain 'X'.")
    if "labels" not in data:
        raise KeyError(f"{path} does not contain 'labels'.")
    X = np.asarray(data["X"])
    labels = np.asarray(data["labels"]).astype(np.int64).ravel()
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"X and labels size mismatch: {X.shape[0]} vs {labels.shape[0]}.")
    return X, labels


def main(config_path: Path | None = None) -> None:
    cfg = _settings(config_path)
    partition_path = resolve_path(cfg["partition_path"])
    validation_data_path = resolve_path(cfg["validation_data_path"])

    if not partition_path.is_file():
        raise FileNotFoundError(f"Partition archive not found: {partition_path}")
    if not validation_data_path.is_file():
        raise FileNotFoundError(f"Validation data file not found: {validation_data_path}")

    X, labels = _load_validation_data(validation_data_path)
    partition = load_partition_inference(partition_path)

    result = classify_with_anchors(cfg=cfg, X=X, gt_labels=labels, partition=partition)

    print(f"Partition: {partition_path}")
    print(f"Validation data: {validation_data_path}")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes (anchors): {result.n_classes}")
    print_anchor_assignments(result.anchors, result.anchor_partitions, pred_raw=result.pred_raw)
    print()
    print_classification_table(result.table_rows)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: validate_partition.py [CONFIG.json]")
    cfg = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else None
    if cfg is not None and not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")
    main(cfg)
