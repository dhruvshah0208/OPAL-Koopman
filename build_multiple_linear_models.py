'''
Build switched (multiple local linear) Koopman models on a graph-merged partition.

For each partition component, fit a local EDMD model on snapshot pairs from
data_path, save switched_models.npz and multiple_linear_models_plot_data.npz,
plot partition scatter and phase trajectories, then evaluate one-step and
rollout RMSE on validation_data_path using target-anchored classification
(metrics computed only on correctly classified snapshots / trajectories with
correctly classified initial conditions).

Paper figures can be regenerated from the saved plot data with generate_plots.py.

Run with a config from configs/<system>/build_multiple_linear_models.json.
'''

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_VALIDATION_DIR = _ROOT / "validation"
_KOOPMAN_ROOT = _ROOT.parent
_SRC_DEEP_KOOPMAN = _ROOT / "src_deep_koopman"
_PRUNING_ROOT = _KOOPMAN_ROOT / "MPV-SPV-pruning-code"
for _path in (_ROOT, _SRC_DEEP_KOOPMAN, _PRUNING_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))
if str(_VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(_VALIDATION_DIR))

from partition_classification import (
    apply_anchor_config,
    classify_with_anchors,
    print_anchor_assignments,
    print_classification_table,
)
from src_basin_classification import (
    build_rbf_polynomial_dictionary,
    classify_partition_labels,
    fit_switched_edmd_models,
    load_partition_inference,
    sample_edmd_centers,
    switched_koopman_rollout,
)

DEFAULT_CONFIG_PATH = _ROOT / "configs/duffing/build_multiple_linear_models.json"

_DEFAULTS = {
    "partition_path": _ROOT / "data/duffing/partition_output.npz",
    "data_path": _ROOT / "data/duffing/duffing_data.npz",
    "validation_data_path": None,
    "centers_global_data_path": _ROOT / "data/duffing/duffing_data.npz",
    "output_path": _ROOT / "data/duffing/switched_models.npz",
    "target_equilibria": [[-1.0, 0.0], [1.0, 0.0]],
    "target_data_paths": None,
    "system_name": "duffing_eqn",
    "edmd_gamma": 30.0,
    "edmd_d_poly": 1,
    "number_centers_edmd": -1,
    "min_samples_per_cell": 100,
    "prediction_horizon": 50,
    "n_rollout_trajectories": 5,
    "rollout_rel_rmse_times": [0, 50, 100, 200, 300],
    "num_phase_plot": 5,
    "ICs_phase_plot": [],
    "reclassify_each_step": True,
    "random_state": 42,
    "show_plots": True,
    "plots_output_dir": _ROOT / "figures/duffing/multiple_linear_models",
    "plot_data_output_path": _ROOT / "data/duffing/multiple_linear_models_plot_data.npz",
    "docs_figures_relpath": "duffing/multiple_linear_models",
}


@dataclass
class CellResult:
    component: int
    n_samples: int
    lifted_rel_err: float
    state_rmse: float
    state_rel_err: float


def _settings(config_path: Path | None) -> dict:
    s = dict(_DEFAULTS)
    if config_path is None:
        return s

    with open(config_path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object.")
    if "data_path" not in raw:
        raise ValueError("Config must include 'data_path' (path to .npz from generate_data.py).")
    if "partition_path" not in raw:
        raise ValueError("Config must include 'partition_path' (path to partition_output.npz).")
    if "centers_global_data_path" not in raw:
        raise ValueError(
            "Config must include 'centers_global_data_path' "
            "(path to .npz containing centers_global from generate_data.py)."
        )

    for key in _DEFAULTS:
        if key in raw:
            s[key] = raw[key]

    if "ICs_phase_plot" in raw:
        s["ICs_phase_plot"] = raw["ICs_phase_plot"]
    if s["ICs_phase_plot"] is None:
        s["ICs_phase_plot"] = []

    apply_anchor_config(raw, s)

    s["partition_path"] = Path(s["partition_path"])
    s["data_path"] = Path(s["data_path"])
    if "validation_data_path" in raw:
        s["validation_data_path"] = Path(raw["validation_data_path"])
    elif s.get("validation_data_path") is None:
        s["validation_data_path"] = s["data_path"]
    else:
        s["validation_data_path"] = Path(s["validation_data_path"])
    s["centers_global_data_path"] = Path(s["centers_global_data_path"])
    s["output_path"] = Path(s["output_path"])
    s["system_name"] = str(s["system_name"])
    s["edmd_gamma"] = float(s["edmd_gamma"])
    s["edmd_d_poly"] = int(s["edmd_d_poly"])
    s["number_centers_edmd"] = int(s["number_centers_edmd"])
    s["min_samples_per_cell"] = int(s["min_samples_per_cell"])
    s["prediction_horizon"] = int(s["prediction_horizon"])
    s["n_rollout_trajectories"] = int(s["n_rollout_trajectories"])
    if "rollout_rel_rmse_times" in raw:
        s["rollout_rel_rmse_times"] = [int(t) for t in raw["rollout_rel_rmse_times"]]
    else:
        s["rollout_rel_rmse_times"] = [int(t) for t in s["rollout_rel_rmse_times"]]
    s["num_phase_plot"] = int(s["num_phase_plot"])
    s["reclassify_each_step"] = bool(s["reclassify_each_step"])
    s["random_state"] = int(s["random_state"])
    s["show_plots"] = bool(s["show_plots"])
    s["plots_output_dir"] = Path(s["plots_output_dir"])
    if "plot_data_output_path" in raw:
        s["plot_data_output_path"] = Path(raw["plot_data_output_path"])
    elif "plot_data_output_path" in s:
        s["plot_data_output_path"] = Path(s["plot_data_output_path"])
    else:
        s["plot_data_output_path"] = _resolve(s["output_path"]).with_name(
            "multiple_linear_models_plot_data.npz"
        )
    if "docs_figures_relpath" in raw:
        s["docs_figures_relpath"] = str(raw["docs_figures_relpath"])
    else:
        s["docs_figures_relpath"] = _docs_figures_relpath_from_cfg(s)
    if s.get("validation_data_path") is None:
        s["validation_data_path"] = s["data_path"]
    return s


def _docs_figures_relpath_from_cfg(cfg: dict) -> str:
    """Map plots_output_dir like figures/duffing/multiple_linear_models -> duffing/multiple_linear_models."""
    plots_dir = Path(cfg["plots_output_dir"])
    parts = plots_dir.parts
    if parts and parts[0] == "figures":
        return str(Path(*parts[1:]))
    return str(plots_dir)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (_ROOT / path)


def _load_centers_global(path: Path) -> np.ndarray:
    path = _resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"Centers data file not found: {path}")
    data = np.load(path)
    if "centers_global" not in data:
        raise KeyError(
            f"{path} has no 'centers_global' array. "
            "Regenerate the dataset with generate_data.py."
        )
    return np.asarray(data["centers_global"])


def _finalize_figure(fig, *, save_path: Path | None, show: bool) -> None:
    import matplotlib.pyplot as plt

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _pack_rollouts(rollouts: list | None) -> tuple[np.ndarray, np.ndarray]:
    """Stack rollout tuples into (n_traj, horizon, n_state) arrays."""
    if not rollouts:
        return np.empty((0, 0, 0), dtype=np.float64), np.empty((0, 0, 0), dtype=np.float64)
    true_list = [np.asarray(item[0], dtype=np.float64) for item in rollouts]
    pred_list = [np.asarray(item[1], dtype=np.float64) for item in rollouts]
    n_traj = len(true_list)
    horizon = true_list[0].shape[0]
    n_state = true_list[0].shape[1]
    true_stack = np.full((n_traj, horizon, n_state), np.nan, dtype=np.float64)
    pred_stack = np.full((n_traj, horizon, n_state), np.nan, dtype=np.float64)
    for i, (true_traj, pred_traj) in enumerate(zip(true_list, pred_list, strict=True)):
        true_stack[i] = true_traj
        pred_stack[i] = pred_traj
    return true_stack, pred_stack


def rollouts_from_arrays(
    true_stack: np.ndarray,
    pred_stack: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Rebuild rollout tuples expected by the plotting helpers."""
    true_stack = np.asarray(true_stack, dtype=np.float64)
    pred_stack = np.asarray(pred_stack, dtype=np.float64)
    if true_stack.size == 0 or true_stack.shape[0] == 0:
        return []
    rollouts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(true_stack.shape[0]):
        step_labels = np.full(true_stack.shape[1] - 1, -1, dtype=np.int64)
        rollouts.append((true_stack[i], pred_stack[i], step_labels))
    return rollouts


def save_plot_data(
    output_path: Path | str,
    *,
    X: np.ndarray,
    labels: np.ndarray,
    phase_rollouts: list | None,
    system_name: str,
    docs_figures_relpath: str,
    evaluation: dict | None = None,
) -> None:
    """Persist arrays used to render switched-model paper figures."""
    output_path = _resolve(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation = evaluation or {}
    true_stack, pred_stack = _pack_rollouts(phase_rollouts)
    rel_times = np.asarray(evaluation.get("rel_rmse_times", []), dtype=np.int64)
    rel_values = np.asarray(evaluation.get("rel_rmse_values", []), dtype=np.float64)

    np.savez(
        output_path,
        partition_X=np.asarray(X, dtype=np.float64),
        partition_labels=np.asarray(labels, dtype=np.int64).ravel(),
        phase_true_trajs=true_stack,
        phase_pred_trajs=pred_stack,
        system_name=np.array(system_name),
        n_state=np.int64(np.asarray(X).shape[1]),
        docs_figures_relpath=np.array(docs_figures_relpath),
        rel_rmse_times=rel_times,
        rel_rmse_values=rel_values,
        partition_accuracy=np.float64(evaluation.get("partition_accuracy", np.nan)),
    )
    print(f"Plot data saved to {output_path}")


def load_plot_data(path: Path | str) -> dict:
    """Load arrays written by save_plot_data."""
    path = _resolve(Path(path))
    if not path.is_file():
        raise FileNotFoundError(f"Plot data file not found: {path}")
    data = np.load(path, allow_pickle=False)
    required = (
        "partition_X",
        "partition_labels",
        "phase_true_trajs",
        "phase_pred_trajs",
        "system_name",
        "n_state",
        "docs_figures_relpath",
    )
    for key in required:
        if key not in data:
            raise KeyError(f"{path} does not contain '{key}'.")

    rel_times = np.asarray(data["rel_rmse_times"], dtype=np.int64) if "rel_rmse_times" in data else np.array([], dtype=np.int64)
    rel_values = (
        np.asarray(data["rel_rmse_values"], dtype=np.float64)
        if "rel_rmse_values" in data
        else np.array([], dtype=np.float64)
    )
    partition_accuracy = float(data["partition_accuracy"]) if "partition_accuracy" in data else float("nan")

    return {
        "partition_X": np.asarray(data["partition_X"], dtype=np.float64),
        "partition_labels": np.asarray(data["partition_labels"], dtype=np.int64).ravel(),
        "phase_true_trajs": np.asarray(data["phase_true_trajs"], dtype=np.float64),
        "phase_pred_trajs": np.asarray(data["phase_pred_trajs"], dtype=np.float64),
        "system_name": str(data["system_name"]),
        "n_state": int(data["n_state"]),
        "docs_figures_relpath": str(data["docs_figures_relpath"]),
        "rel_rmse_times": rel_times,
        "rel_rmse_values": rel_values,
        "partition_accuracy": partition_accuracy,
        "source_path": path,
    }


def _plot_state_partition_2d(
    X,
    labels,
    *,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X)
    labels = np.asarray(labels).ravel()
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("plot_state_partition_2d requires X with shape (N, 2).")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of rows.")

    partition_cmap = ListedColormap(["red", "blue", "green", "orange"])
    partition_cmap.set_under("lightgray")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        s=4,
        alpha=0.45,
        edgecolors="none",
        cmap=partition_cmap,
        vmin=0,
        vmax=len(partition_cmap.colors) - 1,
    )
    fig.colorbar(scatter, ax=ax, label="partition label")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title or "State-space partition")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)
    return fig


def _plot_state_partition_1d(
    X,
    labels,
    *,
    title: str | None = None,
    save_path: Path | None = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X)
    labels = np.asarray(labels).ravel()
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("plot_state_partition_1d requires X with shape (N, 1).")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of rows.")

    partition_cmap = ListedColormap(["red", "blue", "green", "orange"])
    partition_cmap.set_under("lightgray")

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.05, 0.05, size=X.shape[0])

    fig, ax = plt.subplots(figsize=(8, 3))
    scatter = ax.scatter(
        X[:, 0],
        jitter,
        c=labels,
        s=4,
        alpha=0.45,
        edgecolors="none",
        cmap=partition_cmap,
        vmin=0,
        vmax=len(partition_cmap.colors) - 1,
    )
    fig.colorbar(scatter, ax=ax, label="partition label")
    ax.set_xlabel(r"$x_1$")
    ax.set_yticks([])
    ax.set_title(title or "State-space partition")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)
    return fig


def _save_models(
    output_path: Path,
    *,
    models: dict,
    cfg: dict,
    centers_global: np.ndarray,
    edmd_centers: np.ndarray,
    n: int,
) -> None:
    output_path = _resolve(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = sorted(models["K_by_label"].keys())
    component_labels = np.array(labels, dtype=np.int64)
    if labels:
        K_stack = np.stack([models["K_by_label"][label] for label in labels], axis=0)
        R_stack = np.stack([models["R_by_label"][label] for label in labels], axis=0)
        sample_counts = np.array(
            [models["sample_counts"][label] for label in labels], dtype=np.int64
        )
    else:
        K_stack = np.empty((0, models["n_dict"], models["n_dict"]), dtype=np.float64)
        R_stack = np.empty((0, models["n_dict"], models["n_state"]), dtype=np.float64)
        sample_counts = np.empty((0,), dtype=np.int64)

    skipped_items = sorted(models["skipped"].items())
    skipped_labels = np.array([label for label, _ in skipped_items], dtype=np.int64)
    skipped_counts = np.array([count for _, count in skipped_items], dtype=np.int64)

    np.savez(
        output_path,
        component_labels=component_labels,
        K_stack=K_stack,
        R_stack=R_stack,
        sample_counts=sample_counts,
        skipped_labels=skipped_labels,
        skipped_counts=skipped_counts,
        centers_global=np.asarray(centers_global),
        edmd_centers=np.asarray(edmd_centers),
        number_centers_edmd=np.int64(cfg["number_centers_edmd"]),
        edmd_gamma=np.float64(cfg["edmd_gamma"]),
        edmd_d_poly=np.int64(cfg["edmd_d_poly"]),
        n=np.int64(n),
        n_dict=np.int64(models["n_dict"]),
        n_state=np.int64(models["n_state"]),
        partition_path=np.array(str(_resolve(cfg["partition_path"]))),
        system_name=np.array(cfg["system_name"]),
    )
    print(f"Switched models saved to {output_path}")


def _print_training_diagnostics(models: dict, labels: np.ndarray) -> None:
    n_total = int(labels.shape[0])
    n_dropped = int(np.sum(labels < 0))
    fitted = models["sample_counts"]
    skipped = models["skipped"]

    print()
    print("Training diagnostics")
    print(f"  Total snapshot pairs: {n_total}")
    if n_total > 0:
        print(
            f"  Dropped (unclassified, label = -1): {n_dropped} "
            f"({100.0 * n_dropped / n_total:.2f}%)"
        )
    print(f"  Fitted components: {len(fitted)}")
    for label in sorted(fitted):
        print(f"    component {label}: {fitted[label]} snapshot pairs")
    if skipped:
        print(f"  Skipped components (< min_samples): {len(skipped)}")
        for label in sorted(skipped):
            print(f"    component {label}: {skipped[label]} snapshot pairs")


def _load_eval_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    path = _resolve(path)
    data = np.load(path)
    if "X" not in data or "Y" not in data:
        raise KeyError(f"{path} must contain 'X' and 'Y' snapshot pairs.")
    if "labels" not in data:
        raise KeyError(f"{path} must contain 'labels' for classification-gated evaluation.")
    X = np.asarray(data["X"], dtype=np.float64)
    Y = np.asarray(data["Y"], dtype=np.float64)
    labels = np.asarray(data["labels"], dtype=np.int64).ravel()
    if X.shape[0] != labels.shape[0]:
        raise ValueError(f"X and labels size mismatch: {X.shape[0]} vs {labels.shape[0]}.")
    trajectories = np.asarray(data["trajectories"], dtype=np.float64) if "trajectories" in data else None
    return X, Y, trajectories, labels


def _one_step_cell_results(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    models: dict,
    Phi,
    *,
    sample_mask: np.ndarray | None = None,
) -> list[CellResult]:
    eps = 1e-12
    results: list[CellResult] = []
    for label in sorted(models["K_by_label"].keys()):
        mask = labels == label
        if sample_mask is not None:
            mask = mask & sample_mask
        count = int(np.sum(mask))
        if count == 0:
            continue
        K = models["K_by_label"][label]
        R = models["R_by_label"][label]
        Phi_X = Phi.evaluate(X[mask])
        Phi_Y = Phi.evaluate(Y[mask])

        lifted_pred = Phi_X @ K
        lifted_num = np.linalg.norm(Phi_Y - lifted_pred, axis=1)
        lifted_den = np.linalg.norm(Phi_X, axis=1) + eps
        lifted_rel = float(np.mean(lifted_num / lifted_den))

        state_pred = lifted_pred @ R
        state_err = np.linalg.norm(Y[mask] - state_pred, axis=1)
        state_rmse = float(np.sqrt(np.mean(state_err**2)))
        state_den = np.linalg.norm(Y[mask], axis=1) + eps
        state_rel = float(np.mean(state_err / state_den))

        results.append(
            CellResult(
                component=label,
                n_samples=count,
                lifted_rel_err=lifted_rel,
                state_rmse=state_rmse,
                state_rel_err=state_rel,
            )
        )
    return results


def _relative_state_error(true_state: np.ndarray, pred_state: np.ndarray, *, eps: float = 1e-12) -> float:
    """Relative L2 state error ||pred - true|| / ||true||."""
    true_state = np.asarray(true_state, dtype=np.float64).ravel()
    pred_state = np.asarray(pred_state, dtype=np.float64).ravel()
    if not np.all(np.isfinite(pred_state)):
        return float("nan")
    denom = float(np.linalg.norm(true_state))
    if denom < eps:
        return float("nan")
    return float(np.linalg.norm(pred_state - true_state) / denom)


def _initial_state_correct_mask(
    trajectories: np.ndarray,
    correct_mask: np.ndarray,
) -> np.ndarray:
    n_traj = trajectories.shape[0]
    if n_traj == 0:
        return np.zeros(0, dtype=bool)
    snapshots_per_traj = correct_mask.shape[0] // n_traj
    if snapshots_per_traj * n_traj != correct_mask.shape[0]:
        raise ValueError(
            f"Cannot align {correct_mask.shape[0]} snapshots with {n_traj} trajectories."
        )
    return correct_mask.reshape(n_traj, snapshots_per_traj)[:, 0]


def _rollout_metrics(
    trajectories: np.ndarray,
    models: dict,
    partition: dict,
    Phi,
    cfg: dict,
    *,
    ic_correct_mask: np.ndarray | None = None,
) -> tuple[dict[str, float], list]:
    max_horizon = trajectories.shape[1] - 1
    horizon = min(cfg["prediction_horizon"], max_horizon)
    n_traj = min(cfg["n_rollout_trajectories"], trajectories.shape[0])
    rel_times = sorted({int(t) for t in cfg.get("rollout_rel_rmse_times", [0, 50, 100, 200, 300])})
    rel_times = [t for t in rel_times if 0 <= t <= horizon]
    if not rel_times:
        rel_times = [0]

    rng = np.random.default_rng(cfg["random_state"])
    traj_indices = rng.choice(trajectories.shape[0], size=n_traj, replace=False)

    rmse_per_traj: list[float] = []
    rel_err_by_time: dict[int, list[float]] = {t: [] for t in rel_times}
    rollouts: list = []
    dropped_steps = 0
    total_steps = 0
    n_skipped_ic = 0
    for idx in traj_indices:
        if ic_correct_mask is not None and not ic_correct_mask[idx]:
            n_skipped_ic += 1
            continue
        true_traj = trajectories[idx, : horizon + 1, :]
        pred_traj, step_labels = switched_koopman_rollout(
            true_traj[0],
            horizon,
            models,
            partition,
            Phi,
            reclassify=cfg["reclassify_each_step"],
        )
        rollouts.append((true_traj, pred_traj, step_labels))
        valid = np.all(np.isfinite(pred_traj), axis=1)
        if int(np.sum(valid)) > 1:
            err = np.linalg.norm(pred_traj[valid] - true_traj[valid], axis=1)
            rmse = float(np.sqrt(np.mean(err**2)))
            rmse_per_traj.append(rmse)
        for t in rel_times:
            if t < true_traj.shape[0] and np.all(np.isfinite(pred_traj[t])):
                rel_err_by_time[t].append(
                    _relative_state_error(true_traj[t], pred_traj[t])
                )
        dropped_steps += int(np.sum(step_labels < 0))
        total_steps += int(step_labels.shape[0])

    rel_rmse_at_times = {
        t: float(np.mean(vals)) if vals else float("nan") for t, vals in rel_err_by_time.items()
    }
    metrics = {
        "horizon": float(horizon),
        "n_traj": float(n_traj),
        "mean_rmse": float(np.mean(rmse_per_traj)) if rmse_per_traj else float("nan"),
        "max_rmse": float(np.max(rmse_per_traj)) if rmse_per_traj else float("nan"),
        "n_valid_traj": float(len(rmse_per_traj)),
        "n_skipped_ic": float(n_skipped_ic),
        "dropped_fraction": (dropped_steps / total_steps) if total_steps else float("nan"),
        "rel_rmse_times": rel_times,
        "rel_rmse_at_times": rel_rmse_at_times,
    }
    return metrics, rollouts


def _print_rollout_rel_rmse_times(metrics: dict[str, float]) -> None:
    rel_times = metrics.get("rel_rmse_times", [])
    rel_values = metrics.get("rel_rmse_at_times", {})
    if not rel_times:
        return
    headers = ("Time step", *(str(t) for t in rel_times))
    values = tuple(
        f"{float(rel_values.get(t, float('nan'))):.6g}" if t in rel_values else "nan"
        for t in rel_times
    )
    body = (("Mean rel.\ error", *values),)
    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

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


def _print_cell_table(rows: list[CellResult]) -> None:
    headers = ("Component", "Samples", "Lifted rel err", "State RMSE", "State rel err")
    body = [
        (
            str(r.component),
            str(r.n_samples),
            f"{r.lifted_rel_err:.4e}",
            f"{r.state_rmse:.4e}",
            f"{r.state_rel_err:.4e}",
        )
        for r in rows
    ]
    if not body:
        print("No components with evaluation samples.")
        return

    widths = [max(len(headers[i]), *(len(row[i]) for row in body)) for i in range(len(headers))]

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


def _run_classification_gated_evaluation(
    cfg: dict,
    *,
    models: dict,
    partition: dict,
    Phi_edmd,
) -> dict:
    validation_data_path = _resolve(cfg["validation_data_path"])
    if not validation_data_path.is_file():
        raise FileNotFoundError(f"Validation data file not found: {validation_data_path}")

    result = {
        "partition_accuracy": float("nan"),
        "rel_rmse_times": [],
        "rel_rmse_values": [],
        "rollout_metrics": {},
    }

    X_val, Y_val, traj_val, gt_labels = _load_eval_data(validation_data_path)
    classification = classify_with_anchors(
        cfg=cfg,
        X=X_val,
        gt_labels=gt_labels,
        partition=partition,
    )
    correct_mask = classification.correct_mask
    n_correct = int(np.sum(correct_mask))
    n_total = int(correct_mask.shape[0])

    print()
    print(f"--- Classification-gated evaluation ---")
    print(f"Validation data: {validation_data_path}")
    print(f"Samples: {X_val.shape[0]}")
    print(f"Fitted components: {sorted(models['K_by_label'].keys())}")
    print()
    print("--- Partition classification ---")
    print(f"Classes (anchors): {classification.n_classes}")
    print_anchor_assignments(
        classification.anchors,
        classification.anchor_partitions,
        pred_raw=classification.pred_raw,
    )
    print()
    print_classification_table(classification.table_rows)
    print(
        f"Correctly classified: {n_correct} / {n_total} "
        f"({100.0 * n_correct / n_total if n_total else 0.0:.2f}%)"
    )
    result["partition_accuracy"] = (
        100.0 * n_correct / n_total if n_total else float("nan")
    )

    component_labels = classify_partition_labels(X_val, partition)
    cell_rows = _one_step_cell_results(
        X_val, Y_val, component_labels, models, Phi_edmd, sample_mask=correct_mask
    )
    print()
    print("--- One-step EDMD (correctly classified snapshots) ---")
    _print_cell_table(cell_rows)

    filtered_rollouts = None
    if traj_val is not None:
        if not models["K_by_label"]:
            print("\nNo fitted components; skipping rollout diagnostics.")
        else:
            ic_correct_mask = _initial_state_correct_mask(traj_val, correct_mask)
            metrics, filtered_rollouts = _rollout_metrics(
                traj_val,
                models,
                partition,
                Phi_edmd,
                cfg,
                ic_correct_mask=ic_correct_mask,
            )
            result["rollout_metrics"] = metrics
            rel_times = metrics.get("rel_rmse_times", [])
            rel_values = metrics.get("rel_rmse_at_times", {})
            result["rel_rmse_times"] = list(rel_times)
            result["rel_rmse_values"] = [
                float(rel_values[t]) for t in rel_times if t in rel_values
            ]
            print()
            print("--- T-step rollout (correct IC only) ---")
            print(
                f"Horizon = {int(metrics['horizon'])}, "
                f"{int(metrics['n_traj'])} trajectories sampled, "
                f"reclassify_each_step = {cfg['reclassify_each_step']}"
            )
            print(
                f"  Valid rollouts (correct IC): {int(metrics['n_valid_traj'])}/"
                f"{int(metrics['n_traj'])} "
                f"(skipped {int(metrics['n_skipped_ic'])} misclassified ICs)"
            )
            print(f"  Mean rollout RMSE: {metrics['mean_rmse']:.6g}")
            print(f"  Max rollout RMSE:  {metrics['max_rmse']:.6g}")
            print(f"  Dropped rollout steps: {100.0 * metrics['dropped_fraction']:.2f}%")
            print()
            print("--- Rollout relative error at selected time steps (correct IC only) ---")
            _print_rollout_rel_rmse_times(metrics)
    else:
        print("\nNo 'trajectories' array in validation data; skipping rollout metrics.")

    return result


def _collect_rollouts(
    trajectories: np.ndarray,
    models: dict,
    partition: dict,
    Phi_edmd,
    cfg: dict,
    *,
    n_traj: int,
    rng: np.random.Generator,
) -> tuple[list, int]:
    max_horizon = trajectories.shape[1] - 1
    horizon = min(cfg["prediction_horizon"], max_horizon)
    n_traj = min(n_traj, trajectories.shape[0])
    traj_indices = rng.choice(trajectories.shape[0], size=n_traj, replace=False)

    rollouts: list = []
    for idx in traj_indices:
        true_traj = trajectories[idx, : horizon + 1, :]
        pred_traj, step_labels = switched_koopman_rollout(
            true_traj[0],
            horizon,
            models,
            partition,
            Phi_edmd,
            reclassify=cfg["reclassify_each_step"],
        )
        rollouts.append((true_traj, pred_traj, step_labels))
    return rollouts, horizon


def _parse_ics_phase_plot(raw, state_dim: int) -> np.ndarray:
    """Parse ICs_phase_plot config into an (n_ic, n_state) array; empty if unset."""
    if raw is None or raw == []:
        return np.empty((0, state_dim), dtype=np.float64)
    ics = np.asarray(raw, dtype=np.float64)
    if ics.ndim == 1:
        if state_dim == 1:
            ics = ics.reshape(-1, 1)
        else:
            ics = ics.reshape(1, -1)
    if ics.ndim != 2:
        raise ValueError(
            "ICs_phase_plot must be a list of state vectors, e.g. [[x1, x2], [x1, x2], ...]."
        )
    if ics.shape[1] != state_dim:
        raise ValueError(
            f"ICs_phase_plot state dimension {ics.shape[1]} does not match data ({state_dim})."
        )
    return ics


def _simulate_true_trajectories(
    system_name: str,
    initial_conditions: np.ndarray,
    horizon: int,
) -> np.ndarray:
    try:
        from src_deep_koopman.dynamical_systems import SystemsAutonomous
    except ImportError:  # pragma: no cover
        from dynamical_systems import SystemsAutonomous  # type: ignore[import-not-found]

    system = SystemsAutonomous(system_name)
    n_steps = horizon + 1
    x0 = np.asarray(initial_conditions, dtype=np.float64)
    if x0.ndim == 1:
        x0 = x0.reshape(1, -1)
    return system.simulate_trajectory_true(x0=x0, time_horizon=n_steps)


def _collect_phase_plot_rollouts(
    trajectories: np.ndarray | None,
    models: dict,
    partition: dict,
    Phi_edmd,
    cfg: dict,
    *,
    n_state: int,
) -> tuple[list | None, int]:
    if not models["K_by_label"]:
        return None, 0

    ics = _parse_ics_phase_plot(cfg["ICs_phase_plot"], n_state)
    if ics.size > 0:
        horizon = int(cfg["prediction_horizon"])
        print(f"\nPhase plots: simulating {ics.shape[0]} trajectory(s) from ICs_phase_plot")
        try:
            true_trajs = _simulate_true_trajectories(cfg["system_name"], ics, horizon)
        except ModuleNotFoundError as exc:
            print(
                f"Warning: could not simulate true trajectories ({exc}). "
                "Falling back to sampled validation trajectories for phase plots."
            )
            if trajectories is None:
                return None, 0
            phase_rng = np.random.default_rng(cfg["random_state"] + 1)
            n_traj = cfg["num_phase_plot"] if cfg["num_phase_plot"] > 0 else ics.shape[0]
            return _collect_rollouts(
                trajectories,
                models,
                partition,
                Phi_edmd,
                cfg,
                n_traj=n_traj,
                rng=phase_rng,
            )
        rollouts: list = []
        for i in range(true_trajs.shape[0]):
            true_traj = true_trajs[i]
            pred_traj, step_labels = switched_koopman_rollout(
                true_traj[0],
                horizon,
                models,
                partition,
                Phi_edmd,
                reclassify=cfg["reclassify_each_step"],
            )
            rollouts.append((true_traj, pred_traj, step_labels))
        return rollouts, horizon

    if trajectories is None or cfg["num_phase_plot"] <= 0:
        return None, 0

    phase_rng = np.random.default_rng(cfg["random_state"] + 1)
    return _collect_rollouts(
        trajectories,
        models,
        partition,
        Phi_edmd,
        cfg,
        n_traj=cfg["num_phase_plot"],
        rng=phase_rng,
    )


def _plot_rollouts(
    rollouts: list,
    system_name: str,
    *,
    save_path: Path | None = None,
    show: bool = False,
):
    import matplotlib.pyplot as plt

    if not rollouts:
        return None

    n_state = rollouts[0][0].shape[1]
    horizon = rollouts[0][0].shape[0]
    n_traj = len(rollouts)
    fig, axes = plt.subplots(n_state, 1, figsize=(8, 3.0 * n_state), squeeze=False)
    t = np.arange(horizon)

    for j in range(n_state):
        ax = axes[j, 0]
        errors = np.full((n_traj, horizon), np.nan, dtype=np.float64)
        rel_errors = np.full((n_traj, horizon), np.nan, dtype=np.float64)
        for i, (true_traj, pred_traj, _) in enumerate(rollouts):
            valid = np.isfinite(pred_traj[:, j])
            errors[i, valid] = pred_traj[valid, j] - true_traj[valid, j]
            rel_errors[i, valid] = errors[i, valid] / np.linalg.norm(true_traj[valid, j])
                
        mean_err = np.nanmean(rel_errors, axis=0)
        std_err = np.nanstd(rel_errors, axis=0)
        has_data = np.isfinite(mean_err)
        if not np.any(has_data):
            continue

        t_valid = t[has_data]
        mean_valid = mean_err[has_data]
        std_valid = std_err[has_data]
        ax.plot(t_valid, mean_valid, color="crimson", lw=1.5, label="mean relative error")
        ax.fill_between(
            t_valid,
            mean_valid - std_valid,
            mean_valid + std_valid,
            color="crimson",
            alpha=0.25,
            label=r"$\pm 1$ std",
        )
        ax.axhline(0.0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_ylabel(rf"relative error in $x_{j + 1}$")
        ax.set_title(rf"Rollout relative error: $x_{j + 1}$ (pred $-$ true)")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("time step")
    fig.suptitle(
        f"Switched-model rollout errors ({system_name}, {n_traj} trajectories)",
        y=1.01,
    )
    plt.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)
    return fig


def _plot_phase_trajectories(
    rollouts: list,
    system_name: str,
    *,
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot true vs switched-EDMD state trajectories, one subplot per state variable."""
    import matplotlib.pyplot as plt

    if not rollouts:
        return None

    n_state = rollouts[0][0].shape[1]
    horizon = rollouts[0][0].shape[0]
    n_traj = len(rollouts)
    fig, axes = plt.subplots(n_state, 1, figsize=(8, 3.0 * n_state), squeeze=False)
    t = np.arange(horizon)

    for j in range(n_state):
        ax = axes[j, 0]
        for i, (true_traj, pred_traj, _) in enumerate(rollouts):
            valid = np.isfinite(pred_traj[:, j])
            if not np.any(valid):
                continue
            t_valid = t[valid]
            ax.plot(
                t_valid,
                true_traj[valid, j],
                color="black",
                lw=1.2,
                alpha=0.65,
                label="true" if i == 0 else None,
            )
            ax.plot(
                t_valid,
                pred_traj[valid, j],
                color="crimson",
                lw=1.2,
                ls="--",
                alpha=0.85,
                label="EDMD" if i == 0 else None,
            )
        ax.set_ylabel(rf"$x_{j + 1}$")
        ax.set_title(rf"State $x_{j + 1}$")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("time step")
    fig.suptitle(
        f"True vs switched-EDMD trajectories ({system_name}, {n_traj} ICs)",
        y=1.01,
    )
    plt.tight_layout()
    _finalize_figure(fig, save_path=save_path, show=show)
    return fig


def main(config_path: Path | None = None) -> None:
    cfg = _settings(config_path)

    partition_path = _resolve(cfg["partition_path"])
    if not partition_path.is_file():
        raise FileNotFoundError(f"Partition archive not found: {partition_path}")
    data_path = _resolve(cfg["data_path"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    centers_global_path = _resolve(cfg["centers_global_data_path"])

    partition = load_partition_inference(partition_path)
    print(f" Partition loaded from {partition_path}")

    data = np.load(data_path)
    if "X" not in data or "Y" not in data:
        raise KeyError(f"{data_path} must contain 'X' and 'Y' snapshot pairs.")
    X = data["X"]
    Y = data["Y"]
    trajectories = data["trajectories"] if "trajectories" in data else None
    n = X.shape[1]

    centers_global = _load_centers_global(centers_global_path)

    print(f" Data loaded from {data_path}")
    print(f" Total snapshot pairs: {X.shape[0]}")
    print(f" RBF centers loaded from {centers_global_path}: {centers_global.shape}")

    if partition["n"] != n:
        raise ValueError(
            f"Partition state dimension ({partition['n']}) does not match data ({n})."
        )

    labels = classify_partition_labels(X, partition)
    unique_components = np.unique(labels[labels >= 0])
    print(f" Partition components present in data: {unique_components.tolist()}")

    edmd_centers = sample_edmd_centers(
        centers_global,
        cfg["number_centers_edmd"],
        random_state=cfg["random_state"],
    )
    Phi_edmd = build_rbf_polynomial_dictionary(
        n, edmd_centers, cfg["edmd_gamma"], cfg["edmd_d_poly"]
    )
    print(f" Number of EDMD dictionary elements = {Phi_edmd.s}")

    models = fit_switched_edmd_models(
        X, Y, labels, Phi_edmd, min_samples=cfg["min_samples_per_cell"]
    )

    _print_training_diagnostics(models, labels)
    _save_models(
        cfg["output_path"],
        models=models,
        cfg=cfg,
        centers_global=centers_global,
        edmd_centers=edmd_centers,
        n=n,
    )

    plots_dir = _resolve(cfg["plots_output_dir"]) if cfg["plots_output_dir"] else None
    show = cfg["show_plots"]
    phase_rollouts = None
    if cfg["show_plots"] or cfg["plots_output_dir"]:
        if n == 2:
            _plot_state_partition_2d(
                X,
                labels,
                title="Training data colored by partition component",
                save_path=plots_dir / "partition_scatter.png" if plots_dir else None,
                show=show,
            )
        elif n == 1:
            _plot_state_partition_1d(
                X,
                labels,
                title="Training data colored by partition component",
                save_path=plots_dir / "partition_scatter.png" if plots_dir else None,
                show=show,
            )

        phase_rollouts, _ = _collect_phase_plot_rollouts(
            trajectories,
            models,
            partition,
            Phi_edmd,
            cfg,
            n_state=n,
        )
        if phase_rollouts:
            _plot_phase_trajectories(
                phase_rollouts,
                cfg["system_name"],
                save_path=plots_dir / "phase_trajectories.png" if plots_dir else None,
                show=show,
            )
        elif cfg["ICs_phase_plot"] or cfg["num_phase_plot"] > 0:
            print("\nPhase plots skipped: no valid switched-model rollouts for the configured ICs.")

    evaluation = _run_classification_gated_evaluation(
        cfg,
        models=models,
        partition=partition,
        Phi_edmd=Phi_edmd,
    )

    save_plot_data(
        cfg["plot_data_output_path"],
        X=X,
        labels=labels,
        phase_rollouts=phase_rollouts,
        system_name=cfg["system_name"],
        docs_figures_relpath=cfg["docs_figures_relpath"],
        evaluation=evaluation,
    )


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: build_multiple_linear_models.py [CONFIG.json]")
    cfg_path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    main(cfg_path)
