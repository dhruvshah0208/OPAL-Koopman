"""
Validate saved Koopman classifiers on labeled validation snapshots.

Run:
    python validation/validate_classifier.py
    python validation/validate_classifier.py validation/validate_classifier.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_VALIDATION_DIR = Path(__file__).resolve().parent
_ROOT = _VALIDATION_DIR.parent
_SRC_DIR = _ROOT / "src_deep_koopman"
for _path in (_ROOT, _SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src_basin_classification import apply_eigenvector, load_dictionary_from_classifiers  # noqa: E402

_DEFAULTS = {
    "classifiers_path": _ROOT / "data/duffing/classifiers.npz",
    "validation_data_path": _ROOT / "data/duffing/duffing_validation_data.npz",
    "allow_label_flip_for_binary": True,
}


@dataclass
class AccuracyResult:
    classifier: str
    mapping: str
    correct: int
    total: int
    accuracy: float


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

    s["classifiers_path"] = Path(s["classifiers_path"])
    s["validation_data_path"] = Path(s["validation_data_path"])
    s["allow_label_flip_for_binary"] = bool(s["allow_label_flip_for_binary"])
    return s


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_ROOT / path)


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


def _predict_binary(Phi_X: np.ndarray, v: np.ndarray, threshold: float) -> np.ndarray:
    phi = apply_eigenvector(Phi_X, v)
    return (phi > threshold).astype(np.int64)


def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, float]:
    correct = int(np.sum(y_true == y_pred))
    total = int(y_true.shape[0])
    return correct, total, (correct / total if total > 0 else 0.0)


def _evaluate_classifier(
    *,
    classifier_name: str,
    y_true: np.ndarray,
    y_pred_raw: np.ndarray,
    allow_flip: bool,
) -> list[AccuracyResult]:
    results: list[AccuracyResult] = []
    c, n, a = _accuracy(y_true, y_pred_raw)
    results.append(
        AccuracyResult(
            classifier=classifier_name,
            mapping="identity (0->0,1->1)",
            correct=c,
            total=n,
            accuracy=a,
        )
    )
    if allow_flip:
        flipped = 1 - y_pred_raw
        c_flip, n_flip, a_flip = _accuracy(y_true, flipped)
        results.append(
            AccuracyResult(
                classifier=classifier_name,
                mapping="flipped (0->1,1->0)",
                correct=c_flip,
                total=n_flip,
                accuracy=a_flip,
            )
        )
    return results


def _print_table(rows: list[AccuracyResult]) -> None:
    headers = ("Classifier", "Mapping", "Correct", "Total", "Accuracy")
    body = [
        (
            r.classifier,
            r.mapping,
            str(r.correct),
            str(r.total),
            f"{100.0 * r.accuracy:.2f}%",
        )
        for r in rows
    ]
    widths = [
        max(len(headers[i]), *(len(row[i]) for row in body))
        for i in range(len(headers))
    ]

    def _line(ch: str = "-") -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def _fmt(row: tuple[str, ...]) -> str:
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(row))) + " |"

    print(_line("-"))
    print(_fmt(headers))
    print(_line("="))
    for row in body:
        print(_fmt(row))
    print(_line("-"))


def main(config_path: Path | None = None) -> None:
    cfg = _settings(config_path)
    classifiers_path = _resolve(cfg["classifiers_path"])
    validation_data_path = _resolve(cfg["validation_data_path"])

    if not classifiers_path.is_file():
        raise FileNotFoundError(f"Classifier archive not found: {classifiers_path}")
    if not validation_data_path.is_file():
        raise FileNotFoundError(f"Validation data file not found: {validation_data_path}")

    X, labels = _load_validation_data(validation_data_path)
    classifiers = np.load(classifiers_path)
    for key in ("v_edmd", "v_inv", "threshold_edmd", "threshold_inv"):
        if key not in classifiers:
            raise KeyError(f"{classifiers_path} does not contain '{key}'.")

    Phi = load_dictionary_from_classifiers(classifiers_path)
    Phi_X = Phi.evaluate(X)

    pred_edmd = _predict_binary(Phi_X, classifiers["v_edmd"], float(classifiers["threshold_edmd"]))
    pred_inv = _predict_binary(Phi_X, classifiers["v_inv"], float(classifiers["threshold_inv"]))

    unique_labels = np.unique(labels)
    if unique_labels.shape[0] != 2:
        raise ValueError(
            "This validator currently supports binary labels only; "
            f"found labels: {unique_labels.tolist()}"
        )
    if not np.array_equal(unique_labels, np.array([0, 1])):
        # Re-index labels to 0/1 deterministically if user labels are e.g. [-1, 1].
        remap = {int(unique_labels[0]): 0, int(unique_labels[1]): 1}
        labels = np.array([remap[int(y)] for y in labels], dtype=np.int64)

    rows = []
    rows.extend(
        _evaluate_classifier(
            classifier_name="EDMD",
            y_true=labels,
            y_pred_raw=pred_edmd,
            allow_flip=cfg["allow_label_flip_for_binary"],
        )
    )
    rows.extend(
        _evaluate_classifier(
            classifier_name="ApproxInvariant",
            y_true=labels,
            y_pred_raw=pred_inv,
            allow_flip=cfg["allow_label_flip_for_binary"],
        )
    )

    print(f"Classifiers: {classifiers_path}")
    print(f"Validation data: {validation_data_path}")
    print(f"Samples: {X.shape[0]}")
    print()
    _print_table(rows)


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: validate_classifier.py [CONFIG.json]")
    cfg = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else None
    if cfg is not None and not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")
    main(cfg)