"""Target-anchored classification: map partition ids to semantic classes via anchor states.

Anchors are either explicit target equilibria or the terminal snapshots of
reference trajectories (limit cycles). Each anchor is classified by the saved
partition; the resulting partition-id -> class map is applied to validation data
and compared with ground-truth labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .dictionary import build_rbf_polynomial_dictionary
from .invariants import evaluate_invariant_embedding
from .partition import predict_partition_labels_from_embedding
from .paths import resolve
from .tables import print_table


@dataclass
class TableRow:
    method: str
    mapping: str
    metric: str
    correct: int
    total: int
    rate: float


@dataclass
class PartitionClassificationResult:
    anchors: np.ndarray
    n_classes: int
    gt_labels: np.ndarray
    pred_raw: np.ndarray
    pred_mapped: np.ndarray
    partition_to_class: dict[int, int]
    anchor_partitions: np.ndarray
    correct_mask: np.ndarray
    table_rows: list[TableRow]
    F: np.ndarray


def apply_anchor_config(raw: dict[str, Any], settings: dict[str, Any]) -> None:
    """Set target_equilibria / target_data_paths with mutual exclusivity."""
    if "target_data_paths" in raw:
        settings["target_data_paths"] = raw["target_data_paths"]
        settings["target_equilibria"] = None
    elif "target_equilibria" in raw:
        settings["target_equilibria"] = raw["target_equilibria"]
        settings["target_data_paths"] = None


def validate_target_equilibria(raw: Any, *, state_dim: int | None = None) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2:
        raise ValueError("target_equilibria must be a 2-D list/array of shape (n_classes, n_state).")
    if arr.shape[0] < 2:
        raise ValueError("target_equilibria must contain at least two classes.")
    if state_dim is not None and arr.shape[1] != state_dim:
        raise ValueError(
            f"target_equilibria has state dimension {arr.shape[1]}, expected {state_dim}."
        )
    return arr


def validate_target_data_paths(raw: Any) -> list[Path]:
    if raw is None:
        return []
    if isinstance(raw, str):
        paths = [raw]
    elif isinstance(raw, list):
        if not raw:
            raise ValueError("target_data_paths must not be an empty list.")
        if not all(isinstance(item, str) for item in raw):
            raise ValueError("target_data_paths list entries must be strings.")
        paths = raw
    else:
        raise ValueError("target_data_paths must be a path string or a list of path strings.")

    resolved = [resolve(item) for item in paths]
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"Target data file not found: {path}")
    if len(resolved) < 2:
        raise ValueError("target_data_paths must specify at least two files (one per class).")
    return resolved


def load_final_snapshot(path: Path) -> np.ndarray:
    data = np.load(path)
    if "trajectories" in data:
        trajs = np.asarray(data["trajectories"], dtype=float)
        if trajs.ndim != 3:
            raise ValueError(f"{path} trajectories must have shape (n_traj, T, n_state).")
        if trajs.shape[0] < 1:
            raise ValueError(f"{path} must contain at least one trajectory.")
        if trajs.shape[0] > 1:
            print(
                f"  Warning: {path} has {trajs.shape[0]} trajectories; "
                "using the first for the class anchor."
            )
        return trajs[0, -1, :]
    if "X" in data:
        X = np.asarray(data["X"], dtype=float)
        if X.ndim != 2 or X.shape[0] < 1:
            raise ValueError(f"{path} X must be a non-empty 2-D array.")
        return X[-1, :]
    raise ValueError(f"{path} must contain 'trajectories' or 'X'.")


def build_anchor_points(cfg: dict[str, Any], *, state_dim: int) -> np.ndarray:
    has_equilibria = cfg.get("target_equilibria") is not None
    has_paths = cfg.get("target_data_paths") is not None
    if has_equilibria and has_paths:
        raise ValueError("Config must specify exactly one of target_equilibria or target_data_paths.")
    if not has_equilibria and not has_paths:
        raise ValueError("Config must specify exactly one of target_equilibria or target_data_paths.")

    if has_equilibria:
        anchors = validate_target_equilibria(cfg["target_equilibria"], state_dim=state_dim)
    else:
        paths = validate_target_data_paths(cfg["target_data_paths"])
        snapshots = [load_final_snapshot(path) for path in paths]
        anchors = np.stack(snapshots, axis=0)
        if anchors.shape[1] != state_dim:
            raise ValueError(
                f"Target snapshots have state dimension {anchors.shape[1]}, expected {state_dim}."
            )
    return anchors


def normalize_labels(labels: np.ndarray, n_classes: int) -> np.ndarray:
    unique_labels = np.unique(labels)
    if unique_labels.shape[0] != n_classes:
        raise ValueError(
            f"Expected {n_classes} unique ground-truth classes (matching anchors); "
            f"found labels: {unique_labels.tolist()}"
        )
    if np.array_equal(unique_labels, np.arange(n_classes)):
        return labels.astype(np.int64)
    remap = {int(label): idx for idx, label in enumerate(unique_labels)}
    return np.array([remap[int(y)] for y in labels], dtype=np.int64)


def partition_to_class_from_anchors(anchor_partitions: np.ndarray) -> dict[int, int]:
    partition_to_class: dict[int, int] = {}
    for class_idx, partition in enumerate(anchor_partitions):
        partition = int(partition)
        if partition < 0:
            raise ValueError(
                f"Class {class_idx} anchor was dropped by the partition (label = -1)."
            )
        if partition in partition_to_class:
            other = partition_to_class[partition]
            raise ValueError(
                f"Classes {other} and {class_idx} share anchor partition {partition}; "
                "anchors must define a unique class per partition."
            )
        partition_to_class[partition] = class_idx
    return partition_to_class


def map_partitions_to_classes(
    y_pred_raw: np.ndarray,
    partition_to_class: dict[int, int],
) -> np.ndarray:
    mapped = np.full(y_pred_raw.shape[0], -1, dtype=np.int64)
    for partition, class_idx in partition_to_class.items():
        mapped[y_pred_raw == partition] = class_idx
    return mapped


def embed_states(X: np.ndarray, partition: dict[str, Any], Phi) -> np.ndarray:
    """Evaluate the invariant embedding F(X) for the saved partition."""
    return evaluate_invariant_embedding(Phi.evaluate(X), partition["coeffs_inv"])


def predict_from_embedding(F: np.ndarray, partition: dict[str, Any]) -> np.ndarray:
    return predict_partition_labels_from_embedding(F, partition)


def predict_partitions(
    X: np.ndarray,
    partition: dict[str, Any],
    Phi,
) -> np.ndarray:
    return predict_from_embedding(embed_states(X, partition, Phi), partition)


def build_partition_dictionary(partition: dict[str, Any]):
    return build_rbf_polynomial_dictionary(
        partition["n"],
        partition["centers_global"],
        partition["gamma"],
        partition["d_poly"],
    )


def evaluate_anchor_predictions(
    *,
    method: str,
    y_true: np.ndarray,
    y_pred_mapped: np.ndarray,
    partition_to_class: dict[int, int],
) -> list[TableRow]:
    total_data = int(y_true.shape[0])
    undropped_mask = y_pred_mapped >= 0
    n_undropped = int(np.sum(undropped_mask))
    n_dropped = total_data - n_undropped

    correct_undropped = int(np.sum(y_true[undropped_mask] == y_pred_mapped[undropped_mask]))
    correct_total = correct_undropped

    mapping_str = ", ".join(f"{k}->{v}" for k, v in sorted(partition_to_class.items()))
    mapping = f"anchor ({mapping_str})"

    return [
        TableRow(
            method=method,
            mapping=mapping,
            metric="accuracy (total data)",
            correct=correct_total,
            total=total_data,
            rate=(correct_total / total_data if total_data > 0 else 0.0),
        ),
        TableRow(
            method=method,
            mapping=mapping,
            metric="accuracy (un-dropped data)",
            correct=correct_undropped,
            total=n_undropped,
            rate=(correct_undropped / n_undropped if n_undropped > 0 else 0.0),
        ),
        TableRow(
            method=method,
            mapping=mapping,
            metric="dropped predictions",
            correct=n_dropped,
            total=total_data,
            rate=(n_dropped / total_data if total_data > 0 else 0.0),
        ),
    ]


def print_classification_table(rows: list[TableRow]) -> None:
    headers = ("Method", "Anchor mapping", "Metric", "Correct", "Total", "Rate")
    body = [
        (
            r.method,
            r.mapping,
            r.metric,
            str(r.correct),
            str(r.total),
            f"{100.0 * r.rate:.2f}%",
        )
        for r in rows
    ]
    print_table(headers, body)


def print_anchor_assignments(
    anchors: np.ndarray,
    anchor_partitions: np.ndarray,
    *,
    pred_raw: np.ndarray | None = None,
) -> None:
    print("Anchor assignments:")
    for class_idx, (point, partition_label) in enumerate(zip(anchors, anchor_partitions)):
        point_str = ", ".join(f"{v:.6g}" for v in point)
        print(f"  class {class_idx}: state=({point_str}) -> partition {int(partition_label)}")
    if pred_raw is not None:
        unique_pred = np.unique(pred_raw[pred_raw >= 0]).tolist()
        print(f"Predicted partition labels (excluding dropped): {unique_pred}")


def classify_with_anchors(
    *,
    cfg: dict[str, Any],
    X: np.ndarray,
    gt_labels: np.ndarray,
    partition: dict[str, Any],
    method: str = "GraphMergedPartition",
) -> PartitionClassificationResult:
    anchors = build_anchor_points(cfg, state_dim=X.shape[1])
    Phi_part = build_partition_dictionary(partition)
    return classify_embedding_with_anchors(
        F=embed_states(X, partition, Phi_part),
        F_anchors=embed_states(anchors, partition, Phi_part),
        anchors=anchors,
        gt_labels=gt_labels,
        partition=partition,
        method=method,
    )


def classify_embedding_with_anchors(
    *,
    F: np.ndarray,
    F_anchors: np.ndarray,
    anchors: np.ndarray,
    gt_labels: np.ndarray,
    partition: dict[str, Any],
    method: str = "GraphMergedPartition",
) -> PartitionClassificationResult:
    """Same as ``classify_with_anchors`` but on precomputed embeddings F and F_anchors."""
    n_classes = anchors.shape[0]
    gt_labels = normalize_labels(gt_labels, n_classes)

    anchor_partitions = predict_from_embedding(F_anchors, partition)
    partition_to_class = partition_to_class_from_anchors(anchor_partitions)

    pred_raw = predict_from_embedding(F, partition)
    pred_mapped = map_partitions_to_classes(pred_raw, partition_to_class)
    correct_mask = (pred_mapped >= 0) & (pred_mapped == gt_labels)

    table_rows = evaluate_anchor_predictions(
        method=method,
        y_true=gt_labels,
        y_pred_mapped=pred_mapped,
        partition_to_class=partition_to_class,
    )

    return PartitionClassificationResult(
        anchors=anchors,
        n_classes=n_classes,
        gt_labels=gt_labels,
        pred_raw=pred_raw,
        pred_mapped=pred_mapped,
        partition_to_class=partition_to_class,
        anchor_partitions=anchor_partitions,
        correct_mask=correct_mask,
        table_rows=table_rows,
        F=F,
    )
