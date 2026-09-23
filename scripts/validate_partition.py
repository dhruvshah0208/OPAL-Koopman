"""
Validate a saved graph-merged partition on labeled validation snapshots.

Anchor points (one per class) establish which partition ID corresponds to each
semantic class. Validation accuracy is measured by whether each snapshot lands
in the same partition as its class anchor.

Run:
    python scripts/validate_partition.py configs/<system>/validate_partition.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from src_omega_partition.anchors import (  # noqa: E402
    apply_anchor_config,
    classify_with_anchors,
    print_anchor_assignments,
    print_classification_table,
)
from src_omega_partition.config import config_path_from_argv, load_json_config  # noqa: E402
from src_omega_partition.partition import (  # noqa: E402
    compute_relative_boundary_distances,
    load_partition_inference,
)
from src_omega_partition.paths import PROJECT_ROOT, resolve  # noqa: E402
from src_omega_partition.plotting import (  # noqa: E402
    plot_boundary_distance_histogram,
    plot_misclassified_points,
)

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs/duffing/validate_partition.json"

_DEFAULTS = {
    "partition_path": PROJECT_ROOT / "data/duffing/partition_output.npz",
    "validation_data_path": PROJECT_ROOT / "data/duffing/duffing_validation_data.npz",
    "target_equilibria": [[-1.0, 0.0], [1.0, 0.0]],
    "target_data_paths": None,
    "exclude_dropped_predictions": True,
    "show_plots": False,
}


def load_settings(config_path: Path | None) -> dict[str, Any]:
    s = dict(_DEFAULTS)
    if config_path is None:
        return s

    raw = load_json_config(config_path)
    for key in _DEFAULTS:
        if key in raw:
            s[key] = raw[key]

    apply_anchor_config(raw, s)

    s["partition_path"] = Path(s["partition_path"])
    s["validation_data_path"] = Path(s["validation_data_path"])
    s["exclude_dropped_predictions"] = bool(s["exclude_dropped_predictions"])
    s["show_plots"] = bool(s["show_plots"])
    return s


def load_validation_data(path: Path) -> tuple[np.ndarray, np.ndarray]:
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
    cfg = load_settings(config_path)
    partition_path = resolve(cfg["partition_path"])
    validation_data_path = resolve(cfg["validation_data_path"])

    if not partition_path.is_file():
        raise FileNotFoundError(f"Partition archive not found: {partition_path}")
    if not validation_data_path.is_file():
        raise FileNotFoundError(f"Validation data file not found: {validation_data_path}")

    X, labels = load_validation_data(validation_data_path)
    partition = load_partition_inference(partition_path)

    result = classify_with_anchors(cfg=cfg, X=X, gt_labels=labels, partition=partition)

    print(f"Partition: {partition_path}")
    print(f"Validation data: {validation_data_path}")
    print(f"Samples: {X.shape[0]}")
    print(f"Classes (anchors): {result.n_classes}")
    print(f"Robustness margin from partition archive: tau_robust={partition['tau_robust']}")
    print_anchor_assignments(result.anchors, result.anchor_partitions, pred_raw=result.pred_raw)
    print()
    print_classification_table(result.table_rows)

    dropped = result.pred_mapped < 0
    misclassified = (~dropped) & (result.pred_mapped != result.gt_labels)
    rel_dist, _ = compute_relative_boundary_distances(result.F, partition)
    print()
    _print_boundary_distance_summary(rel_dist, misclassified, result.correct_mask)

    if cfg["show_plots"]:
        if X.shape[1] in (1, 2):
            plot_misclassified_points(
                X,
                result.gt_labels,
                result.pred_mapped,
                anchors=result.anchors,
            )
        plot_boundary_distance_histogram(
            rel_dist,
            misclassified,
            baseline_mask=result.correct_mask,
            tau_robust=partition["tau_robust"],
        )


def _print_boundary_distance_summary(
    rel_dist: np.ndarray, misclassified: np.ndarray, correct: np.ndarray
) -> None:
    print("Relative distance to nearest final-partition boundary (min over coordinates):")
    for name, mask in (("misclassified", misclassified), ("correct", correct)):
        vals = rel_dist[mask & np.isfinite(rel_dist)]
        if vals.size == 0:
            print(f"  {name:>13}: n=0")
            continue
        q = np.percentile(vals, [5, 25, 50, 75, 95])
        print(
            f"  {name:>13}: n={vals.size}  median={q[2]:.4g}  "
            f"p5={q[0]:.3g} p25={q[1]:.3g} p75={q[3]:.3g} p95={q[4]:.3g}  "
            f"frac<0.05={np.mean(vals < 0.05):.3f}"
        )


if __name__ == "__main__":
    main(config_path_from_argv("validate_partition.py", DEFAULT_CONFIG_PATH))
