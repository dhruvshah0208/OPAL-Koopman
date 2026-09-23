"""
Build switched (multiple local linear) Koopman models on a graph-merged partition
and compare them against a single vanilla EDMD model.

For each partition component, fit a local EDMD model on the snapshot pairs from
``data_path`` that the partition assigns to it (pairs the partition drops, label -1,
e.g. by the ``tau_robust`` margin, are excluded). Fit one global (vanilla) EDMD model
on *all* snapshot pairs with the *same* dictionary (the ``vanilla_*`` keys can override
the dictionary, but by default it is shared). Save ``switched_models.npz``,
``vanilla_edmd_model.npz`` and ``multiple_linear_models_plot_data.npz``, plot the
partition scatter, the true / switched / vanilla state trajectories and the
rollout error curves, then evaluate one-step and rollout errors for both models on
``validation_data_path`` using target-anchored classification (metrics are
computed only on correctly classified snapshots / trajectories with correctly
classified initial conditions, and on the same snapshots and trajectories for both
models).

Set ``run_gated_evaluation`` to false to stop after the ``ICs_phase_plot`` rollouts
(their rollout errors are still printed and plotted), which is much faster when tuning
the dictionary parameters.

Paper figures can be regenerated from the saved plot data with ``scripts/generate_plots.py``.

Run:
    python scripts/build_multiple_linear_models.py configs/<system>/build_multiple_linear_models.json
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root

from src_omega_partition.anchors import (  # noqa: E402
    apply_anchor_config,
    classify_with_anchors,
    print_anchor_assignments,
    print_classification_table,
)
from src_omega_partition.config import config_path_from_argv, load_json_config  # noqa: E402
from src_omega_partition.dictionary import build_rbf_polynomial_dictionary, sample_edmd_centers  # noqa: E402
from src_omega_partition.partition import classify_partition_labels, load_partition_inference  # noqa: E402
from src_omega_partition.paths import PROJECT_ROOT, resolve  # noqa: E402
from src_omega_partition.plotting import (  # noqa: E402
    finalize_figure,
    plot_rollout_phase_portrait,
    plot_state_partition_1d,
    plot_state_partition_2d,
)
from src_omega_partition.switched_models import (  # noqa: E402
    fit_switched_edmd_models,
    fit_vanilla_edmd_model,
    switched_koopman_rollout,
    vanilla_koopman_rollout,
)
from src_omega_partition.tables import print_table  # noqa: E402

_ROOT = PROJECT_ROOT
DEFAULT_CONFIG_PATH = _ROOT / "configs/duffing/build_multiple_linear_models.json"

SWITCHED_LABEL = "switched EDMD"
VANILLA_LABEL = "vanilla EDMD"

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
    "compare_vanilla_edmd": True,
    # None for all three -> vanilla EDMD reuses the switched dictionary object exactly.
    "vanilla_edmd_gamma": None,
    "vanilla_edmd_d_poly": None,
    "vanilla_number_centers_edmd": None,
    "vanilla_output_path": None,  # None -> output_path with name vanilla_edmd_model.npz
    "min_samples_per_cell": 100,
    "prediction_horizon": 50,
    "n_rollout_trajectories": 5,
    "rollout_rel_rmse_times": [0, 50, 100, 200, 300],
    "num_phase_plot": 5,
    "ICs_phase_plot": [],
    "reclassify_each_step": True,
    "run_gated_evaluation": True,
    "random_state": 42,
    "show_plots": True,
    "plots_output_dir": _ROOT / "figures/duffing/multiple_linear_models",
    # plot_data_output_path and paper_figures_relpath default to values derived from
    # output_path and plots_output_dir (see load_settings), so they follow the system.
}


@dataclass
class CellResult:
    component: str
    n_samples: int
    lifted_rel_err: float
    state_rmse: float
    state_rel_err: float


# A rollout function maps (x0, horizon) -> (pred_traj, step_labels).
RolloutFn = Callable[[np.ndarray, int], tuple[np.ndarray, np.ndarray]]


def load_settings(config_path: Path | None) -> dict:
    s = dict(_DEFAULTS)
    if config_path is None:
        return _finalizeload_settings(s, raw={})

    raw = load_json_config(
        config_path, required=("data_path", "partition_path", "centers_global_data_path")
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
    s["run_gated_evaluation"] = bool(s["run_gated_evaluation"])
    s["random_state"] = int(s["random_state"])
    s["show_plots"] = bool(s["show_plots"])
    s["plots_output_dir"] = Path(s["plots_output_dir"])
    if "plot_data_output_path" in raw:
        s["plot_data_output_path"] = Path(raw["plot_data_output_path"])
    else:
        s["plot_data_output_path"] = resolve(s["output_path"]).with_name(
            "multiple_linear_models_plot_data.npz"
        )
    if "paper_figures_relpath" in raw:
        s["paper_figures_relpath"] = str(raw["paper_figures_relpath"])
    else:
        s["paper_figures_relpath"] = _paper_figures_relpath_from_cfg(s)
    if s.get("validation_data_path") is None:
        s["validation_data_path"] = s["data_path"]
    return _finalizeload_settings(s, raw=raw)


def _finalizeload_settings(s: dict, *, raw: dict) -> dict:
    """Resolve the vanilla-EDMD keys, which default to the switched-EDMD values."""
    s["compare_vanilla_edmd"] = bool(s["compare_vanilla_edmd"])
    s["vanilla_shares_dictionary"] = all(
        s[k] is None
        for k in ("vanilla_edmd_gamma", "vanilla_edmd_d_poly", "vanilla_number_centers_edmd")
    )
    s["vanilla_edmd_gamma"] = (
        float(s["edmd_gamma"]) if s["vanilla_edmd_gamma"] is None else float(s["vanilla_edmd_gamma"])
    )
    s["vanilla_edmd_d_poly"] = (
        int(s["edmd_d_poly"]) if s["vanilla_edmd_d_poly"] is None else int(s["vanilla_edmd_d_poly"])
    )
    s["vanilla_number_centers_edmd"] = (
        int(s["number_centers_edmd"])
        if s["vanilla_number_centers_edmd"] is None
        else int(s["vanilla_number_centers_edmd"])
    )
    if s["vanilla_output_path"] is None:
        s["vanilla_output_path"] = Path(s["output_path"]).with_name("vanilla_edmd_model.npz")
    else:
        s["vanilla_output_path"] = Path(s["vanilla_output_path"])
    return s


def _paper_figures_relpath_from_cfg(cfg: dict) -> str:
    """Map plots_output_dir like figures/duffing/multiple_linear_models -> duffing/multiple_linear_models."""
    plots_dir = Path(cfg["plots_output_dir"])
    parts = plots_dir.parts
    if parts and parts[0] == "figures":
        return str(Path(*parts[1:]))
    return str(plots_dir)


def _load_centers_global(path: Path) -> np.ndarray:
    path = resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"Centers data file not found: {path}")
    data = np.load(path)
    if "centers_global" not in data:
        raise KeyError(
            f"{path} has no 'centers_global' array. "
            "Regenerate the dataset with generate_data.py."
        )
    return np.asarray(data["centers_global"])


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
    if true_stack.size == 0 or pred_stack.size == 0 or true_stack.shape[0] == 0:
        return []
    if pred_stack.shape != true_stack.shape:
        raise ValueError(
            f"Predicted stack shape {pred_stack.shape} does not match true stack {true_stack.shape}."
        )
    rollouts: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(true_stack.shape[0]):
        step_labels = np.full(true_stack.shape[1] - 1, -1, dtype=np.int64)
        rollouts.append((true_stack[i], pred_stack[i], step_labels))
    return rollouts


def _empty_curves() -> np.ndarray:
    return np.empty((0, 0), dtype=np.float64)


def save_plot_data(
    output_path: Path | str,
    *,
    X: np.ndarray,
    labels: np.ndarray,
    phase_rollouts: list | None,
    system_name: str,
    paper_figures_relpath: str,
    evaluation: dict | None = None,
    phase_vanilla_rollouts: list | None = None,
) -> None:
    """Persist arrays used to render switched-model paper figures.

    Vanilla-EDMD arrays (``phase_vanilla_trajs``, ``vanilla_rel_rmse_values`` and the
    per-trajectory rollout error curves) are written empty when no comparison was run.
    """
    output_path = resolve(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation = evaluation or {}
    true_stack, pred_stack = _pack_rollouts(phase_rollouts)
    _, vanilla_stack = _pack_rollouts(phase_vanilla_rollouts)
    rel_times = np.asarray(evaluation.get("rel_rmse_times", []), dtype=np.int64)
    rel_values = np.asarray(evaluation.get("rel_rmse_values", []), dtype=np.float64)
    vanilla_rel_values = np.asarray(evaluation.get("vanilla_rel_rmse_values", []), dtype=np.float64)
    curves = evaluation.get("error_curves", {})

    np.savez(
        output_path,
        partition_X=np.asarray(X, dtype=np.float64),
        partition_labels=np.asarray(labels, dtype=np.int64).ravel(),
        phase_true_trajs=true_stack,
        phase_pred_trajs=pred_stack,
        phase_vanilla_trajs=vanilla_stack,
        system_name=np.array(system_name),
        n_state=np.int64(np.asarray(X).shape[1]),
        paper_figures_relpath=np.array(paper_figures_relpath),
        rel_rmse_times=rel_times,
        rel_rmse_values=rel_values,
        vanilla_rel_rmse_values=vanilla_rel_values,
        partition_accuracy=np.float64(evaluation.get("partition_accuracy", np.nan)),
        rollout_rel_err_switched=np.asarray(curves.get("rel_switched", _empty_curves()), dtype=np.float64),
        rollout_rel_err_vanilla=np.asarray(curves.get("rel_vanilla", _empty_curves()), dtype=np.float64),
        rollout_abs_err_switched=np.asarray(curves.get("abs_switched", _empty_curves()), dtype=np.float64),
        rollout_abs_err_vanilla=np.asarray(curves.get("abs_vanilla", _empty_curves()), dtype=np.float64),
    )
    print(f"Plot data saved to {output_path}")


def load_plot_data(path: Path | str) -> dict:
    """Load arrays written by save_plot_data (vanilla-EDMD keys are optional)."""
    path = resolve(Path(path))
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
        "paper_figures_relpath",
    )
    for key in required:
        if key not in data:
            raise KeyError(f"{path} does not contain '{key}'.")

    def _optional(key: str, dtype, empty: np.ndarray) -> np.ndarray:
        return np.asarray(data[key], dtype=dtype) if key in data else empty

    rel_times = _optional("rel_rmse_times", np.int64, np.array([], dtype=np.int64))
    rel_values = _optional("rel_rmse_values", np.float64, np.array([], dtype=np.float64))
    vanilla_rel_values = _optional("vanilla_rel_rmse_values", np.float64, np.array([], dtype=np.float64))
    partition_accuracy = float(data["partition_accuracy"]) if "partition_accuracy" in data else float("nan")

    return {
        "partition_X": np.asarray(data["partition_X"], dtype=np.float64),
        "partition_labels": np.asarray(data["partition_labels"], dtype=np.int64).ravel(),
        "phase_true_trajs": np.asarray(data["phase_true_trajs"], dtype=np.float64),
        "phase_pred_trajs": np.asarray(data["phase_pred_trajs"], dtype=np.float64),
        "phase_vanilla_trajs": _optional("phase_vanilla_trajs", np.float64, np.empty((0, 0, 0))),
        "system_name": str(data["system_name"]),
        "n_state": int(data["n_state"]),
        "paper_figures_relpath": str(data["paper_figures_relpath"]),
        "rel_rmse_times": rel_times,
        "rel_rmse_values": rel_values,
        "vanilla_rel_rmse_values": vanilla_rel_values,
        "partition_accuracy": partition_accuracy,
        "rollout_rel_err_switched": _optional("rollout_rel_err_switched", np.float64, _empty_curves()),
        "rollout_rel_err_vanilla": _optional("rollout_rel_err_vanilla", np.float64, _empty_curves()),
        "rollout_abs_err_switched": _optional("rollout_abs_err_switched", np.float64, _empty_curves()),
        "rollout_abs_err_vanilla": _optional("rollout_abs_err_vanilla", np.float64, _empty_curves()),
        "source_path": path,
    }


def _save_models(
    output_path: Path,
    *,
    models: dict,
    cfg: dict,
    centers_global: np.ndarray,
    edmd_centers: np.ndarray,
    n: int,
) -> None:
    output_path = resolve(Path(output_path))
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
        partition_path=np.array(str(resolve(cfg["partition_path"]))),
        system_name=np.array(cfg["system_name"]),
    )
    print(f"Switched models saved to {output_path}")


def _save_vanilla_model(
    output_path: Path,
    *,
    model: dict,
    cfg: dict,
    centers_global: np.ndarray,
    edmd_centers: np.ndarray,
    n: int,
) -> None:
    output_path = resolve(Path(output_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        K=np.asarray(model["K"]),
        R=np.asarray(model["R"]),
        n_samples=np.int64(model["n_samples"]),
        centers_global=np.asarray(centers_global),
        edmd_centers=np.asarray(edmd_centers),
        number_centers_edmd=np.int64(cfg["vanilla_number_centers_edmd"]),
        edmd_gamma=np.float64(cfg["vanilla_edmd_gamma"]),
        edmd_d_poly=np.int64(cfg["vanilla_edmd_d_poly"]),
        n=np.int64(n),
        n_dict=np.int64(model["n_dict"]),
        n_state=np.int64(model["n_state"]),
        system_name=np.array(cfg["system_name"]),
    )
    print(f"Vanilla EDMD model saved to {output_path}")


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
    path = resolve(path)
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


# --------------------------------------------------------------------------- #
# One-step metrics
# --------------------------------------------------------------------------- #


def _one_step_sample_errors(
    X: np.ndarray, Y: np.ndarray, K: np.ndarray, R: np.ndarray, Phi
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-sample (lifted relative error, state error norm, state relative error)."""
    eps = 1e-12
    Phi_X = Phi.evaluate(X)
    Phi_Y = Phi.evaluate(Y)
    lifted_pred = Phi_X @ K
    lifted_rel = np.linalg.norm(Phi_Y - lifted_pred, axis=1) / (np.linalg.norm(Phi_X, axis=1) + eps)
    state_err = np.linalg.norm(Y - lifted_pred @ R, axis=1)
    state_rel = state_err / (np.linalg.norm(Y, axis=1) + eps)
    return lifted_rel, state_err, state_rel


def _cell_result(
    name: str, lifted_rel: np.ndarray, state_err: np.ndarray, state_rel: np.ndarray
) -> CellResult:
    return CellResult(
        component=name,
        n_samples=int(state_err.shape[0]),
        lifted_rel_err=float(np.mean(lifted_rel)),
        state_rmse=float(np.sqrt(np.mean(state_err**2))),
        state_rel_err=float(np.mean(state_rel)),
    )


def _one_step_cell_results(
    X: np.ndarray,
    Y: np.ndarray,
    labels: np.ndarray,
    component_labels: list[int],
    Phi,
    *,
    operator_for_label: Callable[[int], tuple[np.ndarray, np.ndarray]],
    sample_mask: np.ndarray | None = None,
) -> list[CellResult]:
    """One-step errors per partition component plus an ``all`` row over their union.

    ``operator_for_label`` returns the (K, R) pair used on a component; for the
    switched model that is the local operator, for vanilla EDMD the global one.
    """
    results: list[CellResult] = []
    pooled: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for label in component_labels:
        mask = labels == label
        if sample_mask is not None:
            mask = mask & sample_mask
        if int(np.sum(mask)) == 0:
            continue
        K, R = operator_for_label(label)
        errs = _one_step_sample_errors(X[mask], Y[mask], K, R, Phi)
        pooled.append(errs)
        results.append(_cell_result(str(label), *errs))
    if len(results) > 1:
        results.append(
            _cell_result("all", *(np.concatenate([e[i] for e in pooled]) for i in range(3)))
        )
    return results


def _print_cell_table(rows: list[CellResult]) -> None:
    headers = ("Component", "Samples", "Lifted rel err", "State RMSE", "State rel err")
    body = [
        (
            r.component,
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
    print_table(headers, body)


def _print_cell_comparison_table(switched: list[CellResult], vanilla: list[CellResult]) -> None:
    """Side-by-side one-step errors; rows are matched by component name."""
    vanilla_by_name = {r.component: r for r in vanilla}
    headers = (
        "Component",
        "Samples",
        "Lifted rel (sw)",
        "Lifted rel (van)",
        "State RMSE (sw)",
        "State RMSE (van)",
        "State rel (sw)",
        "State rel (van)",
    )
    body = []
    for r in switched:
        v = vanilla_by_name.get(r.component)
        body.append(
            (
                r.component,
                str(r.n_samples),
                f"{r.lifted_rel_err:.4e}",
                f"{v.lifted_rel_err:.4e}" if v else "nan",
                f"{r.state_rmse:.4e}",
                f"{v.state_rmse:.4e}" if v else "nan",
                f"{r.state_rel_err:.4e}",
                f"{v.state_rel_err:.4e}" if v else "nan",
            )
        )
    if not body:
        print("No components with evaluation samples.")
        return
    print_table(headers, body)


# --------------------------------------------------------------------------- #
# Rollout metrics
# --------------------------------------------------------------------------- #


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


def _error_curves(true_traj: np.ndarray, pred_traj: np.ndarray, *, eps: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
    """Per-time-step (relative, absolute) state error along one rollout; NaN where invalid."""
    valid = np.all(np.isfinite(pred_traj), axis=1)
    abs_err = np.full(true_traj.shape[0], np.nan, dtype=np.float64)
    abs_err[valid] = np.linalg.norm(pred_traj[valid] - true_traj[valid], axis=1)
    denom = np.linalg.norm(true_traj, axis=1)
    rel_err = np.where(denom >= eps, abs_err / np.maximum(denom, eps), np.nan)
    return rel_err, abs_err


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


def _switched_rollout_fn(models: dict, partition: dict, Phi, cfg: dict) -> RolloutFn:
    def _run(x0: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        return switched_koopman_rollout(
            x0, horizon, models, partition, Phi, reclassify=cfg["reclassify_each_step"]
        )

    return _run


def _vanilla_rollout_fn(model: dict, Phi) -> RolloutFn:
    def _run(x0: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        pred = vanilla_koopman_rollout(x0, horizon, model, Phi)
        return pred, np.zeros(horizon, dtype=np.int64)

    return _run


def _select_rollout_indices(
    trajectories: np.ndarray, cfg: dict, *, ic_correct_mask: np.ndarray | None
) -> tuple[np.ndarray, int, int]:
    """Sample rollout trajectories once so every model is evaluated on the same set."""
    max_horizon = trajectories.shape[1] - 1
    horizon = min(cfg["prediction_horizon"], max_horizon)
    n_traj = min(cfg["n_rollout_trajectories"], trajectories.shape[0])
    rng = np.random.default_rng(cfg["random_state"])
    traj_indices = rng.choice(trajectories.shape[0], size=n_traj, replace=False)
    n_skipped_ic = 0
    if ic_correct_mask is not None:
        keep = ic_correct_mask[traj_indices]
        n_skipped_ic = int(np.sum(~keep))
        traj_indices = traj_indices[keep]
    return traj_indices, horizon, n_skipped_ic


def _metrics_from_rollouts(rollouts: list, horizon: int, cfg: dict) -> dict:
    """Summarize (true, pred, step_labels) rollout tuples: RMSE per trajectory, mean
    relative error at ``rollout_rel_rmse_times`` and per-trajectory error curves."""
    rel_times = sorted({int(t) for t in cfg.get("rollout_rel_rmse_times", [0, 50, 100, 200, 300])})
    rel_times = [t for t in rel_times if 0 <= t <= horizon]
    if not rel_times:
        rel_times = [0]

    rmse_per_traj: list[float] = []
    rel_err_by_time: dict[int, list[float]] = {t: [] for t in rel_times}
    rel_curves: list[np.ndarray] = []
    abs_curves: list[np.ndarray] = []
    dropped_steps = 0
    total_steps = 0
    for true_traj, pred_traj, step_labels in rollouts:
        valid = np.all(np.isfinite(pred_traj), axis=1)
        if int(np.sum(valid)) > 1:
            err = np.linalg.norm(pred_traj[valid] - true_traj[valid], axis=1)
            rmse_per_traj.append(float(np.sqrt(np.mean(err**2))))
        for t in rel_times:
            if t < true_traj.shape[0] and np.all(np.isfinite(pred_traj[t])):
                rel_err_by_time[t].append(_relative_state_error(true_traj[t], pred_traj[t]))
        rel_curve, abs_curve = _error_curves(true_traj, pred_traj)
        rel_curves.append(rel_curve)
        abs_curves.append(abs_curve)
        dropped_steps += int(np.sum(step_labels < 0))
        total_steps += int(step_labels.shape[0])

    rel_rmse_at_times = {
        t: float(np.mean(vals)) if vals else float("nan") for t, vals in rel_err_by_time.items()
    }
    return {
        "horizon": float(horizon),
        "n_traj": float(len(rollouts)),
        "mean_rmse": float(np.mean(rmse_per_traj)) if rmse_per_traj else float("nan"),
        "max_rmse": float(np.max(rmse_per_traj)) if rmse_per_traj else float("nan"),
        "n_valid_traj": float(len(rmse_per_traj)),
        "n_skipped_ic": 0.0,
        "dropped_fraction": (dropped_steps / total_steps) if total_steps else float("nan"),
        "rel_rmse_times": rel_times,
        "rel_rmse_at_times": rel_rmse_at_times,
        "rel_err_curves": np.stack(rel_curves) if rel_curves else _empty_curves(),
        "abs_err_curves": np.stack(abs_curves) if abs_curves else _empty_curves(),
    }


def _rollout_metrics(
    trajectories: np.ndarray,
    traj_indices: np.ndarray,
    horizon: int,
    rollout_fn: RolloutFn,
    cfg: dict,
    *,
    n_sampled: int,
    n_skipped_ic: int,
) -> tuple[dict, list]:
    rollouts = _rollouts_from_trajectories(trajectories[traj_indices], horizon, rollout_fn)
    metrics = _metrics_from_rollouts(rollouts, horizon, cfg)
    metrics["n_traj"] = float(n_sampled)
    metrics["n_skipped_ic"] = float(n_skipped_ic)
    return metrics, rollouts


def _print_rollout_rel_rmse_times(metrics: dict, vanilla_metrics: dict | None = None) -> None:
    rel_times = metrics.get("rel_rmse_times", [])
    if not rel_times:
        return

    def _row(name: str, m: dict) -> tuple[str, ...]:
        vals = m.get("rel_rmse_at_times", {})
        return (name, *(f"{float(vals.get(t, float('nan'))):.6g}" for t in rel_times))

    headers = ("Time step", *(str(t) for t in rel_times))
    rows = [_row(f"Mean rel. error ({SWITCHED_LABEL})", metrics)]
    if vanilla_metrics is not None:
        rows.append(_row(f"Mean rel. error ({VANILLA_LABEL})", vanilla_metrics))
    print_table(headers, rows)


def _print_rollout_summary(
    metrics: dict, vanilla_metrics: dict | None, cfg: dict, *, gated: bool = True
) -> None:
    print(
        f"Horizon = {int(metrics['horizon'])}, "
        f"{int(metrics['n_traj'])} trajectories sampled, "
        f"reclassify_each_step = {cfg['reclassify_each_step']}"
    )
    if gated:
        print(
            f"  Evaluated rollouts (correct IC): "
            f"{int(metrics['n_traj']) - int(metrics['n_skipped_ic'])}/{int(metrics['n_traj'])} "
            f"(skipped {int(metrics['n_skipped_ic'])} misclassified ICs"
            + ("; same set for both models)" if vanilla_metrics is not None else ")")
        )
    headers = ("Metric", SWITCHED_LABEL) + ((VANILLA_LABEL,) if vanilla_metrics else ())

    def _row(name: str, key: str, fmt: str) -> tuple[str, ...]:
        cells = [name, format(metrics[key], fmt)]
        if vanilla_metrics is not None:
            cells.append(format(vanilla_metrics[key], fmt))
        return tuple(cells)

    print_table(
        headers,
        [
            _row("Valid rollouts", "n_valid_traj", ".0f"),
            _row("Mean rollout RMSE", "mean_rmse", ".6g"),
            _row("Max rollout RMSE", "max_rmse", ".6g"),
            _row("Dropped rollout steps (frac)", "dropped_fraction", ".4f"),
        ],
    )


def run_classification_gated_evaluation(
    cfg: dict,
    *,
    models: dict,
    partition: dict,
    Phi_edmd,
    vanilla_model: dict | None = None,
    Phi_vanilla=None,
) -> dict:
    validation_data_path = resolve(cfg["validation_data_path"])
    if not validation_data_path.is_file():
        raise FileNotFoundError(f"Validation data file not found: {validation_data_path}")

    result = {
        "partition_accuracy": float("nan"),
        "rel_rmse_times": [],
        "rel_rmse_values": [],
        "vanilla_rel_rmse_values": [],
        "rollout_metrics": {},
        "vanilla_rollout_metrics": {},
        "error_curves": {},
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
    print("--- Classification-gated evaluation ---")
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
    fitted_labels = sorted(models["K_by_label"].keys())
    cell_rows = _one_step_cell_results(
        X_val,
        Y_val,
        component_labels,
        fitted_labels,
        Phi_edmd,
        operator_for_label=lambda l: (models["K_by_label"][l], models["R_by_label"][l]),
        sample_mask=correct_mask,
    )
    print()
    if vanilla_model is None:
        print("--- One-step EDMD (correctly classified snapshots) ---")
        _print_cell_table(cell_rows)
    else:
        vanilla_rows = _one_step_cell_results(
            X_val,
            Y_val,
            component_labels,
            fitted_labels,
            Phi_vanilla,
            operator_for_label=lambda _l: (vanilla_model["K"], vanilla_model["R"]),
            sample_mask=correct_mask,
        )
        print("--- One-step EDMD, switched vs vanilla (correctly classified snapshots) ---")
        _print_cell_comparison_table(cell_rows, vanilla_rows)

    if traj_val is None:
        print("\nNo 'trajectories' array in validation data; skipping rollout metrics.")
        return result
    if not models["K_by_label"]:
        print("\nNo fitted components; skipping rollout diagnostics.")
        return result

    ic_correct_mask = _initial_state_correct_mask(traj_val, correct_mask)
    traj_indices, horizon, n_skipped_ic = _select_rollout_indices(
        traj_val, cfg, ic_correct_mask=ic_correct_mask
    )
    n_sampled = traj_indices.shape[0] + n_skipped_ic
    metrics, _ = _rollout_metrics(
        traj_val,
        traj_indices,
        horizon,
        _switched_rollout_fn(models, partition, Phi_edmd, cfg),
        cfg,
        n_sampled=n_sampled,
        n_skipped_ic=n_skipped_ic,
    )
    result["rollout_metrics"] = metrics
    rel_times = metrics.get("rel_rmse_times", [])
    result["rel_rmse_times"] = list(rel_times)
    result["rel_rmse_values"] = [float(metrics["rel_rmse_at_times"][t]) for t in rel_times]
    result["error_curves"] = {
        "rel_switched": metrics["rel_err_curves"],
        "abs_switched": metrics["abs_err_curves"],
    }

    vanilla_metrics = None
    if vanilla_model is not None:
        vanilla_metrics, _ = _rollout_metrics(
            traj_val,
            traj_indices,
            horizon,
            _vanilla_rollout_fn(vanilla_model, Phi_vanilla),
            cfg,
            n_sampled=n_sampled,
            n_skipped_ic=n_skipped_ic,
        )
        result["vanilla_rollout_metrics"] = vanilla_metrics
        result["vanilla_rel_rmse_values"] = [
            float(vanilla_metrics["rel_rmse_at_times"][t]) for t in rel_times
        ]
        result["error_curves"]["rel_vanilla"] = vanilla_metrics["rel_err_curves"]
        result["error_curves"]["abs_vanilla"] = vanilla_metrics["abs_err_curves"]

    print()
    print("--- T-step rollout (correct IC only) ---")
    _print_rollout_summary(metrics, vanilla_metrics, cfg)
    print()
    print("--- Rollout relative error at selected time steps (correct IC only) ---")
    _print_rollout_rel_rmse_times(metrics, vanilla_metrics)
    return result


# --------------------------------------------------------------------------- #
# Phase-plot rollouts
# --------------------------------------------------------------------------- #


def _rollouts_from_trajectories(
    true_trajs: np.ndarray, horizon: int, rollout_fn: RolloutFn
) -> list:
    rollouts: list = []
    for i in range(true_trajs.shape[0]):
        true_traj = true_trajs[i, : horizon + 1, :]
        pred_traj, step_labels = rollout_fn(true_traj[0], horizon)
        rollouts.append((true_traj, pred_traj, step_labels))
    return rollouts


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
    n_steps = horizon + 1
    x0 = np.asarray(initial_conditions, dtype=np.float64)
    if x0.ndim == 1:
        x0 = x0.reshape(1, -1)
    from src_omega_partition.dynamical_systems import SystemsAutonomous

    system = SystemsAutonomous(system_name)
    return system.simulate_trajectory_true(x0=x0, time_horizon=n_steps)


def _phase_plot_true_trajectories(
    trajectories: np.ndarray | None, cfg: dict, *, n_state: int
) -> tuple[np.ndarray | None, int]:
    """True trajectories for the phase plots: simulated from ``ICs_phase_plot`` when
    given, otherwise ``num_phase_plot`` randomly sampled validation trajectories."""
    ics = _parse_ics_phase_plot(cfg["ICs_phase_plot"], n_state)
    if ics.size > 0:
        horizon = int(cfg["prediction_horizon"])
        print(f"\nPhase plots: simulating {ics.shape[0]} trajectory(s) from ICs_phase_plot")
        try:
            return _simulate_true_trajectories(cfg["system_name"], ics, horizon), horizon
        except (ModuleNotFoundError, ValueError) as exc:
            print(
                f"Warning: could not simulate true trajectories ({exc}). "
                "Falling back to sampled validation trajectories for phase plots."
            )
            n_traj = cfg["num_phase_plot"] if cfg["num_phase_plot"] > 0 else ics.shape[0]
    else:
        n_traj = cfg["num_phase_plot"]

    if trajectories is None or n_traj <= 0:
        return None, 0
    horizon = min(cfg["prediction_horizon"], trajectories.shape[1] - 1)
    n_traj = min(n_traj, trajectories.shape[0])
    phase_rng = np.random.default_rng(cfg["random_state"] + 1)
    idx = phase_rng.choice(trajectories.shape[0], size=n_traj, replace=False)
    return trajectories[idx], horizon


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #


def true_trajectory_ylim(rollouts: list, j: int, *, pad_frac: float = 0.25, min_pad: float = 0.1) -> tuple[float, float]:
    """y-limits for state ``j`` from the true trajectories only, padded by ``pad_frac``
    of their range, so a diverging prediction cannot rescale the panel."""
    vals = np.concatenate([np.asarray(item[0])[:, j] for item in rollouts])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(pad_frac * (hi - lo), min_pad)
    return (lo - pad, hi + pad)


def plot_phase_trajectories(
    rollouts: list,
    system_name: str,
    *,
    vanilla_rollouts: list | None = None,
    save_path: Path | None = None,
    show: bool = False,
):
    """Plot true vs switched-EDMD (and optionally vanilla-EDMD) state trajectories,
    one subplot per state variable."""
    import matplotlib.pyplot as plt

    if not rollouts:
        return None

    n_state = rollouts[0][0].shape[1]
    horizon = rollouts[0][0].shape[0]
    n_traj = len(rollouts)
    fig, axes = plt.subplots(n_state, 1, figsize=(8, 3.0 * n_state), squeeze=False)
    t = np.arange(horizon)

    series = [(rollouts, "black", "-", 0.65, "true", 0), (rollouts, "crimson", "--", 0.85, SWITCHED_LABEL, 1)]
    if vanilla_rollouts:
        series.append((vanilla_rollouts, "steelblue", ":", 0.9, VANILLA_LABEL, 1))

    for j in range(n_state):
        ax = axes[j, 0]
        for rolls, color, ls, alpha, label, which in series:
            for i, item in enumerate(rolls):
                traj = item[which][:, j]
                valid = np.isfinite(traj)
                if not np.any(valid):
                    continue
                ax.plot(t[valid], traj[valid], color=color, lw=1.2, ls=ls, alpha=alpha,
                        label=label if i == 0 else None)
        ax.set_ylim(*true_trajectory_ylim(rollouts, j))
        ax.set_ylabel(rf"$x_{j + 1}$")
        ax.set_title(rf"State $x_{j + 1}$")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)

    axes[-1, 0].set_xlabel("time step")
    title = f"True vs {SWITCHED_LABEL}"
    if vanilla_rollouts:
        title += f" vs {VANILLA_LABEL}"
    fig.suptitle(f"{title} trajectories ({system_name}, {n_traj} ICs)", y=1.01)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def plot_rollout_errors(
    rel_switched: np.ndarray,
    abs_switched: np.ndarray,
    *,
    rel_vanilla: np.ndarray | None = None,
    abs_vanilla: np.ndarray | None = None,
    system_name: str = "",
    subtitle: str = "correct ICs only",
    save_path: Path | None = None,
    show: bool = False,
):
    """Rollout error vs time step: mean over trajectories (thick) with the
    per-trajectory curves (thin). Left: relative state error, right: absolute."""
    import matplotlib.pyplot as plt

    rel_switched = np.asarray(rel_switched, dtype=np.float64)
    if rel_switched.size == 0:
        return None

    curves = [(rel_switched, abs_switched, "crimson", SWITCHED_LABEL)]
    if rel_vanilla is not None and np.asarray(rel_vanilla).size > 0:
        curves.append((np.asarray(rel_vanilla), np.asarray(abs_vanilla), "steelblue", VANILLA_LABEL))

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for ax, which, ylabel in ((axes[0], 0, r"$\|\hat{x}_t - x_t\| / \|x_t\|$"),
                              (axes[1], 1, r"$\|\hat{x}_t - x_t\|$")):
        for rel, abs_, color, label in curves:
            arr = (rel, abs_)[which]
            t = np.arange(arr.shape[1])
            for row in arr:
                ax.plot(t, row, color=color, lw=0.5, alpha=0.15)
            with np.errstate(all="ignore"):
                mean = np.nanmean(arr, axis=0)
            ax.plot(t, mean, color=color, lw=2.0, label=f"{label} (mean, {arr.shape[0]} traj.)")
        ax.set_yscale("log")
        ax.set_xlabel("time step")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3, which="both")
        ax.legend(loc="best")
    axes[0].set_title("Relative state error")
    axes[1].set_title("Absolute state error")
    fig.suptitle(f"Rollout error vs time ({system_name}, {subtitle})", y=1.02)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def _build_dictionary(centers_global: np.ndarray, n: int, *, n_centers: int, gamma: float,
                      d_poly: int, random_state: int) -> tuple[np.ndarray, object]:
    centers = sample_edmd_centers(centers_global, n_centers, random_state=random_state)
    Phi = build_rbf_polynomial_dictionary(n, centers, gamma, d_poly)
    return centers, Phi


def load_fit_data(cfg: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, int]:
    """Snapshot pairs (and trajectories, if stored) from ``data_path`` plus the RBF
    center pool from ``centers_global_data_path``. Returns (X, Y, trajectories, centers_global, n)."""
    data_path = resolve(cfg["data_path"])
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    data = np.load(data_path)
    if "X" not in data or "Y" not in data:
        raise KeyError(f"{data_path} must contain 'X' and 'Y' snapshot pairs.")
    X, Y = data["X"], data["Y"]
    trajectories = data["trajectories"] if "trajectories" in data else None
    centers_global = _load_centers_global(resolve(cfg["centers_global_data_path"]))
    print(f" Data loaded from {data_path}")
    print(f" Total snapshot pairs: {X.shape[0]}")
    print(f" RBF centers loaded from {resolve(cfg['centers_global_data_path'])}: {centers_global.shape}")
    return X, Y, trajectories, centers_global, int(X.shape[1])


def fit_models_from_config(cfg: dict, partition: dict, X: np.ndarray, Y: np.ndarray,
                           centers_global: np.ndarray) -> dict:
    """
    Fit the switched local EDMD models (pairs the partition drops are excluded) and,
    if ``compare_vanilla_edmd``, the vanilla EDMD model on all pairs with the same
    dictionary (or the ``vanilla_*`` override). Nothing is written to disk.

    Returns a dict with ``labels``, ``edmd_centers``, ``Phi_edmd``, ``models`` and
    ``vanilla_centers`` / ``Phi_vanilla`` / ``vanilla_model`` (``None`` without comparison).
    """
    n = X.shape[1]
    if partition["n"] != n:
        raise ValueError(
            f"Partition state dimension ({partition['n']}) does not match data ({n})."
        )
    labels = classify_partition_labels(X, partition)
    unique_components = np.unique(labels[labels >= 0])
    print(f" Partition components present in data: {unique_components.tolist()}")

    edmd_centers, Phi_edmd = _build_dictionary(
        centers_global, n,
        n_centers=cfg["number_centers_edmd"], gamma=cfg["edmd_gamma"],
        d_poly=cfg["edmd_d_poly"], random_state=cfg["random_state"],
    )
    print(f" Switched EDMD dictionary: gamma = {cfg['edmd_gamma']}, d_poly = {cfg['edmd_d_poly']}, "
          f"{edmd_centers.shape[0]} centers, {Phi_edmd.s} elements")

    models = fit_switched_edmd_models(
        X, Y, labels, Phi_edmd, min_samples=cfg["min_samples_per_cell"]
    )
    _print_training_diagnostics(models, labels)
    n_used = int(sum(models["sample_counts"].values()))
    print(
        f"  Switched EDMD fitted on {n_used} of {X.shape[0]} snapshot pairs "
        f"({X.shape[0] - n_used} excluded: partition-dropped or skipped components)"
    )

    fit = {
        "labels": labels, "edmd_centers": edmd_centers, "Phi_edmd": Phi_edmd, "models": models,
        "vanilla_centers": None, "Phi_vanilla": None, "vanilla_model": None,
    }
    if not cfg["compare_vanilla_edmd"]:
        return fit

    if cfg["vanilla_shares_dictionary"]:
        vanilla_centers, Phi_vanilla = edmd_centers, Phi_edmd
        print("\n Vanilla EDMD dictionary: same object as the switched EDMD dictionary")
    else:
        vanilla_centers, Phi_vanilla = _build_dictionary(
            centers_global, n,
            n_centers=cfg["vanilla_number_centers_edmd"], gamma=cfg["vanilla_edmd_gamma"],
            d_poly=cfg["vanilla_edmd_d_poly"], random_state=cfg["random_state"],
        )
        print(f"\n Vanilla EDMD dictionary (override): gamma = {cfg['vanilla_edmd_gamma']}, "
              f"d_poly = {cfg['vanilla_edmd_d_poly']}, {vanilla_centers.shape[0]} centers, "
              f"{Phi_vanilla.s} elements")
    # The global model uses every snapshot pair, including those the partition drops.
    vanilla_model = fit_vanilla_edmd_model(X, Y, Phi_vanilla)
    print(
        f" Vanilla EDMD fitted on all {vanilla_model['n_samples']} of {X.shape[0]} "
        "snapshot pairs (nothing dropped)"
    )
    fit.update(vanilla_centers=vanilla_centers, Phi_vanilla=Phi_vanilla, vanilla_model=vanilla_model)
    return fit


def phase_plot_rollouts(cfg: dict, fit: dict, partition: dict, trajectories: np.ndarray | None,
                        n_state: int) -> tuple[list | None, list | None]:
    """Switched (and vanilla) rollouts from ``ICs_phase_plot`` (simulated) or, if that is
    empty, from ``num_phase_plot`` sampled trajectories. Returns (switched, vanilla) rollout lists."""
    if not fit["models"]["K_by_label"]:
        return None, None
    phase_true, horizon = _phase_plot_true_trajectories(trajectories, cfg, n_state=n_state)
    if phase_true is None:
        return None, None
    switched = _rollouts_from_trajectories(
        phase_true, horizon, _switched_rollout_fn(fit["models"], partition, fit["Phi_edmd"], cfg)
    )
    vanilla = None
    if fit["vanilla_model"] is not None:
        vanilla = _rollouts_from_trajectories(
            phase_true, horizon, _vanilla_rollout_fn(fit["vanilla_model"], fit["Phi_vanilla"])
        )
    return switched, vanilla


def main(config_path: Path | None = None) -> None:
    cfg = load_settings(config_path)

    partition_path = resolve(cfg["partition_path"])
    if not partition_path.is_file():
        raise FileNotFoundError(f"Partition archive not found: {partition_path}")
    partition = load_partition_inference(partition_path)
    print(f" Partition loaded from {partition_path}")

    X, Y, trajectories, centers_global, n = load_fit_data(cfg)
    fit = fit_models_from_config(cfg, partition, X, Y, centers_global)
    models, Phi_edmd, labels = fit["models"], fit["Phi_edmd"], fit["labels"]
    vanilla_model, Phi_vanilla = fit["vanilla_model"], fit["Phi_vanilla"]

    _save_models(
        cfg["output_path"],
        models=models,
        cfg=cfg,
        centers_global=centers_global,
        edmd_centers=fit["edmd_centers"],
        n=n,
    )
    if vanilla_model is not None:
        _save_vanilla_model(
            cfg["vanilla_output_path"],
            model=vanilla_model,
            cfg=cfg,
            centers_global=centers_global,
            edmd_centers=fit["vanilla_centers"],
            n=n,
        )

    plots_dir = resolve(cfg["plots_output_dir"]) if cfg["plots_output_dir"] else None
    show = cfg["show_plots"]
    phase_rollouts = None
    phase_vanilla_rollouts = None
    if cfg["show_plots"] or cfg["plots_output_dir"]:
        if n == 2:
            plot_state_partition_2d(
                X,
                labels,
                title="Training data colored by partition component",
                save_path=plots_dir / "partition_scatter.png" if plots_dir else None,
                show=show,
            )
        elif n == 1:
            plot_state_partition_1d(
                X,
                labels,
                title="Training data colored by partition component",
                save_path=plots_dir / "partition_scatter.png" if plots_dir else None,
                show=show,
            )

        phase_rollouts, phase_vanilla_rollouts = phase_plot_rollouts(cfg, fit, partition, trajectories, n)
        if phase_rollouts:
            plot_phase_trajectories(
                phase_rollouts,
                cfg["system_name"],
                vanilla_rollouts=phase_vanilla_rollouts,
                save_path=plots_dir / "phase_trajectories.png" if plots_dir else None,
                show=show,
            )
            if n == 2:
                plot_rollout_phase_portrait(
                    X,
                    labels,
                    phase_rollouts,
                    phase_vanilla_rollouts,
                    title=f"Rollouts over the partition ({cfg['system_name']})",
                    switched_label=SWITCHED_LABEL,
                    vanilla_label=VANILLA_LABEL,
                    save_path=plots_dir / "phase_portrait.png" if plots_dir else None,
                    show=show,
                )
        elif cfg["ICs_phase_plot"] or cfg["num_phase_plot"] > 0:
            print("\nPhase plots skipped: no valid switched-model rollouts for the configured ICs.")

    if phase_rollouts:
        # Cheap diagnostics on the phase-plot ICs; the main use is parameter tuning
        # with run_gated_evaluation = false.
        ic_horizon = phase_rollouts[0][0].shape[0] - 1
        ic_metrics = _metrics_from_rollouts(phase_rollouts, ic_horizon, cfg)
        ic_vanilla_metrics = (
            _metrics_from_rollouts(phase_vanilla_rollouts, ic_horizon, cfg)
            if phase_vanilla_rollouts
            else None
        )
        print()
        print(f"--- Phase-plot IC rollouts ({len(phase_rollouts)} ICs, horizon = {ic_horizon}) ---")
        _print_rollout_summary(ic_metrics, ic_vanilla_metrics, cfg, gated=False)
        print()
        print("--- Phase-plot IC rollout relative error at selected time steps ---")
        _print_rollout_rel_rmse_times(ic_metrics, ic_vanilla_metrics)

    if cfg["run_gated_evaluation"]:
        evaluation = run_classification_gated_evaluation(
            cfg,
            models=models,
            partition=partition,
            Phi_edmd=Phi_edmd,
            vanilla_model=vanilla_model,
            Phi_vanilla=Phi_vanilla,
        )
        curves = evaluation.get("error_curves", {})
        error_plot_title = "correct ICs only"
    else:
        print("\nrun_gated_evaluation = false: skipping the classification-gated evaluation.")
        evaluation = {}
        curves = {}
        if phase_rollouts:
            curves = {
                "rel_switched": ic_metrics["rel_err_curves"],
                "abs_switched": ic_metrics["abs_err_curves"],
            }
            if ic_vanilla_metrics is not None:
                curves["rel_vanilla"] = ic_vanilla_metrics["rel_err_curves"]
                curves["abs_vanilla"] = ic_vanilla_metrics["abs_err_curves"]
        error_plot_title = "phase-plot ICs"

    if (cfg["show_plots"] or cfg["plots_output_dir"]) and "rel_switched" in curves:
        plot_rollout_errors(
            curves["rel_switched"],
            curves["abs_switched"],
            rel_vanilla=curves.get("rel_vanilla"),
            abs_vanilla=curves.get("abs_vanilla"),
            system_name=cfg["system_name"],
            subtitle=error_plot_title,
            save_path=plots_dir / "rollout_errors.png" if plots_dir else None,
            show=show,
        )

    save_plot_data(
        cfg["plot_data_output_path"],
        X=X,
        labels=labels,
        phase_rollouts=phase_rollouts,
        phase_vanilla_rollouts=phase_vanilla_rollouts,
        system_name=cfg["system_name"],
        paper_figures_relpath=cfg["paper_figures_relpath"],
        evaluation=evaluation,
    )


if __name__ == "__main__":
    main(config_path_from_argv("build_multiple_linear_models.py", DEFAULT_CONFIG_PATH))
