"""
Generate trajectory-level validation labels from equilibrium or limit-cycle proximity.

Run:
    python data/generate_validation_data.py
    python data/generate_validation_data.py path/to/config.json

Each trajectory is simulated for the configured horizon, then labeled by the
nearest target equilibrium or limit-cycle reference trajectory to its terminal state;
that label is repeated across all snapshots of the trajectory.

Set ``target_limit_cycle_data_path`` (string or list of paths to .npz files with a
``trajectories`` array) to label by shortest distance to reference limit cycles.
Otherwise labels use ``target_equilibria``.

Set ``show_worst`` to true to display (not save) a plot of the validation trajectory whose
terminal state has the largest distance to its assigned target.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DATA_SUBFOLDER = "duffing"
SYSTEM_NAME = "duffing_eqn"
FILE_NAME = "duffing_validation_data.npz"
DEFAULT_DATASET = {"num_traj": 200, "traj_length": 101}
DEFAULT_TARGET_EQUILIBRIA = [[-1.0, 0.0], [1.0, 0.0]]

_DATA_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _DATA_DIR.parent
_SRC_DIR = _ROOT_DIR / "src_deep_koopman"
for _path in (_ROOT_DIR, _SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src_deep_koopman.dynamical_systems import SystemsAutonomous  # noqa: E402


def _apply_dataset_overrides(dataset: dict[str, int], raw: dict[str, Any]) -> None:
    nested = raw.get("dataset")
    if isinstance(nested, dict):
        if "num_traj" in nested:
            dataset["num_traj"] = int(nested["num_traj"])
        if "traj_length" in nested:
            dataset["traj_length"] = int(nested["traj_length"])
    if "num_traj" in raw:
        dataset["num_traj"] = int(raw["num_traj"])
    if "traj_length" in raw:
        dataset["traj_length"] = int(raw["traj_length"])


def _validate_dataset(dataset: dict[str, Any]) -> dict[str, int]:
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be a JSON object with num_traj and traj_length.")
    try:
        num_traj = int(dataset["num_traj"])
        traj_length = int(dataset["traj_length"])
    except KeyError as exc:
        raise ValueError("dataset must include 'num_traj' and 'traj_length'.") from exc
    if num_traj < 1:
        raise ValueError("num_traj must be a positive integer.")
    if traj_length < 2:
        raise ValueError("traj_length must be at least 2.")
    return {"num_traj": num_traj, "traj_length": traj_length}


def _validate_target_equilibria(raw: Any, state_dim: int) -> np.ndarray:
    arr = np.asarray(raw, dtype=float)
    if arr.ndim != 2:
        raise ValueError("target_equilibria must be a 2-D list/array of shape (n_classes, n_state).")
    if arr.shape[0] < 2:
        raise ValueError("target_equilibria must contain at least two equilibria/classes.")
    if arr.shape[1] != state_dim:
        raise ValueError(
            f"target_equilibria has state dimension {arr.shape[1]}, expected {state_dim}."
        )
    return arr


def _resolve_data_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.is_absolute():
        p = _ROOT_DIR / p
    return p


def _validate_target_limit_cycle_data_path(raw: Any) -> list[Path]:
    if raw is None:
        return []
    if isinstance(raw, str):
        paths = [raw]
    elif isinstance(raw, list):
        if not raw:
            raise ValueError("target_limit_cycle_data_path must not be an empty list.")
        if not all(isinstance(item, str) for item in raw):
            raise ValueError("target_limit_cycle_data_path list entries must be strings.")
        paths = raw
    else:
        raise ValueError(
            "target_limit_cycle_data_path must be a path string or a list of path strings."
        )
    resolved = [_resolve_data_path(item) for item in paths]
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"Limit-cycle data file not found: {path}")
    if len(resolved) < 2:
        raise ValueError("target_limit_cycle_data_path must specify at least two reference trajectories.")
    return resolved


def _load_limit_cycle_trajectories(paths: list[Path], *, state_dim: int) -> list[np.ndarray]:
    """One reference polyline per path (first trajectory if the file stores several)."""
    limit_cycles: list[np.ndarray] = []
    for path in paths:
        data = np.load(path)
        if "trajectories" not in data:
            raise ValueError(f"{path} has no 'trajectories' array.")
        trajs = np.asarray(data["trajectories"], dtype=float)
        if trajs.ndim != 3:
            raise ValueError(f"{path} trajectories must have shape (n_traj, T, n_state).")
        if trajs.shape[0] < 1:
            raise ValueError(f"{path} must contain at least one trajectory.")
        if trajs.shape[2] != state_dim:
            raise ValueError(
                f"{path} trajectories have state dimension {trajs.shape[2]}, expected {state_dim}."
            )
        if trajs.shape[0] > 1:
            print(
                f"  Warning: {path} has {trajs.shape[0]} trajectories; using the first for labeling."
            )
        limit_cycles.append(trajs[0])
    return limit_cycles


def _min_distance_to_trajectory(points: np.ndarray, trajectory: np.ndarray) -> np.ndarray:
    """Shortest Euclidean distance from each point to samples along one trajectory."""
    dists = np.linalg.norm(points[:, None, :] - trajectory[None, :, :], axis=2)
    return np.min(dists, axis=1)


def _build_trajectories(X: np.ndarray, Y: np.ndarray, *, num_traj: int) -> np.ndarray:
    if X.shape != Y.shape:
        raise ValueError(f"X and Y must have matching shape; got {X.shape} and {Y.shape}.")
    if len(X) % num_traj != 0:
        raise ValueError(
            f"Cannot split {len(X)} snapshot pairs evenly across {num_traj} trajectories."
        )
    snapshots_per_traj = len(X) // num_traj
    n_state = X.shape[1]
    X_traj = X.reshape(num_traj, snapshots_per_traj, n_state)
    Y_traj = Y.reshape(num_traj, snapshots_per_traj, n_state)
    return np.concatenate([X_traj, Y_traj[:, -1:, :]], axis=1)


def _snapshots_from_trajectories(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n_state = trajectories.shape[2]
    X = trajectories[:, :-1, :].reshape(-1, n_state)
    Y = trajectories[:, 1:, :].reshape(-1, n_state)
    return X, Y


def _terminal_labels_equilibria(
    trajectories: np.ndarray,
    target_equilibria: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label trajectories by nearest equilibrium; return all terminal distances."""
    terminal_states = trajectories[:, -1, :]
    dists = np.linalg.norm(
        terminal_states[:, None, :] - target_equilibria[None, :, :],
        axis=2,
    )
    labels_per_trajectory = np.argmin(dists, axis=1).astype(np.int64)
    best_final_distances = np.min(dists, axis=1)
    return labels_per_trajectory, dists, best_final_distances


def _terminal_labels_limit_cycles(
    trajectories: np.ndarray,
    limit_cycle_trajectories: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Label trajectories by nearest limit-cycle reference; return all terminal distances."""
    terminal_states = trajectories[:, -1, :]
    dists = np.stack(
        [
            _min_distance_to_trajectory(terminal_states, ref_traj)
            for ref_traj in limit_cycle_trajectories
        ],
        axis=1,
    )
    labels_per_trajectory = np.argmin(dists, axis=1).astype(np.int64)
    best_final_distances = np.min(dists, axis=1)
    return labels_per_trajectory, dists, best_final_distances


def _state_axis_label(state_idx: int) -> str:
    return rf"$x_{state_idx + 1}$"


def _show_worst_trajectory_plot(
    trajectories: np.ndarray,
    *,
    worst_idx: int,
    worst_distance: float,
    assigned_label: int,
    system_name: str,
    use_limit_cycles: bool,
    limit_cycle_trajectories: list[np.ndarray] | None,
    target_equilibria: np.ndarray | None,
) -> None:
    """Display (do not save) the validation trajectory with largest terminal distance."""
    worst_traj = trajectories[worst_idx]
    terminal = worst_traj[-1]
    n_state = worst_traj.shape[1]
    target_name = "limit cycle" if use_limit_cycles else "equilibrium"

    if n_state == 2:
        fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
        ax.plot(
            worst_traj[:, 0],
            worst_traj[:, 1],
            color="tab:blue",
            linewidth=1.5,
            label="worst validation trajectory",
        )
        ax.plot(
            terminal[0],
            terminal[1],
            "o",
            color="tab:red",
            markersize=8,
            label=f"terminal (d={worst_distance:.4g})",
        )
        if use_limit_cycles and limit_cycle_trajectories is not None:
            ref = limit_cycle_trajectories[assigned_label]
            ax.plot(
                ref[:, 0],
                ref[:, 1],
                "--",
                color="tab:green",
                linewidth=1.2,
                alpha=0.85,
                label=f"nearest {target_name} (class {assigned_label})",
            )
        elif target_equilibria is not None:
            for class_idx, eq in enumerate(target_equilibria):
                is_nearest = class_idx == assigned_label
                ax.plot(
                    eq[0],
                    eq[1],
                    "s",
                    color="tab:green" if is_nearest else "0.5",
                    markersize=9 if is_nearest else 7,
                    label=(
                        f"nearest {target_name} (class {assigned_label})"
                        if is_nearest
                        else f"{target_name} {class_idx}"
                    ),
                )
        ax.set_xlabel(_state_axis_label(0))
        ax.set_ylabel(_state_axis_label(1))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.legend(loc="best")
    else:
        time = np.arange(worst_traj.shape[0])
        fig, axes = plt.subplots(
            n_state,
            1,
            figsize=(10, max(3, 2.4 * n_state)),
            sharex=True,
            squeeze=False,
            layout="constrained",
        )
        for state_idx, ax in enumerate(axes[:, 0]):
            ax.plot(time, worst_traj[:, state_idx], color="tab:blue", linewidth=1.5)
            ax.axhline(terminal[state_idx], color="tab:red", linestyle=":", linewidth=1)
            ax.set_ylabel(_state_axis_label(state_idx))
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        axes[-1, 0].set_xlabel("time step")
        fig.legend(
            ["worst validation trajectory", f"terminal (d={worst_distance:.4g})"],
            loc="upper right",
        )

    fig.suptitle(
        f"{system_name}: worst-distance validation trajectory "
        f"(index {worst_idx}, d={worst_distance:.6g}, class {assigned_label})"
    )
    plt.show()


def _settings(config_path: Path | None) -> dict[str, Any]:
    s: dict[str, Any] = {
        "data_subfolder": DATA_SUBFOLDER,
        "system_name": SYSTEM_NAME,
        "file_name": FILE_NAME,
        "dataset": dict(DEFAULT_DATASET),
        "target_equilibria": DEFAULT_TARGET_EQUILIBRIA,
        "seed": 42,
        "verbose": False,
        "show_worst": False,
    }
    if config_path is None:
        return s

    with open(config_path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError("Config must be a JSON object.")

    for key in (
        "data_subfolder",
        "system_name",
        "file_name",
        "target_equilibria",
        "target_limit_cycle_data_path",
        "seed",
        "verbose",
        "show_worst",
    ):
        if key in raw:
            s[key] = raw[key]
    _apply_dataset_overrides(s["dataset"], raw)
    return s


def main(config_path: Path | None = None) -> None:
    cfg = _settings(config_path)
    dataset = _validate_dataset(cfg["dataset"])
    num_traj, traj_length = dataset["num_traj"], dataset["traj_length"]

    file_name = str(cfg["file_name"])
    if not file_name.endswith(".npz"):
        raise ValueError("file_name must end with .npz")

    out_dir = _DATA_DIR / str(cfg["data_subfolder"])
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / file_name
    meta_path = out_dir / file_name.replace(".npz", ".meta.json")

    rng = np.random.default_rng(cfg["seed"]) if cfg["seed"] is not None else None
    system = SystemsAutonomous(str(cfg["system_name"]), rng=rng, verbose=bool(cfg["verbose"]))

    X_raw, Y_raw = system.gather_traj_data(
        num_traj=num_traj,
        traj_length=traj_length,
        plot_results=False,
        show_progress=True,
    )
    trajectories = _build_trajectories(X_raw, Y_raw, num_traj=num_traj)
    X, Y = _snapshots_from_trajectories(trajectories)

    state_dim = X.shape[1]
    limit_cycle_paths = _validate_target_limit_cycle_data_path(
        cfg.get("target_limit_cycle_data_path")
    )
    use_limit_cycles = len(limit_cycle_paths) > 0

    target_equilibria: np.ndarray | None = None
    limit_cycle_trajectories: list[np.ndarray] | None = None
    if use_limit_cycles:
        limit_cycle_trajectories = _load_limit_cycle_trajectories(
            limit_cycle_paths, state_dim=state_dim
        )
        labels_per_trajectory, terminal_distances, best_final_distances = (
            _terminal_labels_limit_cycles(trajectories, limit_cycle_trajectories)
        )
        n_classes = len(limit_cycle_trajectories)
        label_mode = "limit_cycle"
    else:
        if "target_equilibria" not in cfg:
            raise ValueError(
                "Config must include target_equilibria or target_limit_cycle_data_path."
            )
        target_equilibria = _validate_target_equilibria(cfg["target_equilibria"], state_dim=state_dim)
        labels_per_trajectory, terminal_distances, best_final_distances = (
            _terminal_labels_equilibria(trajectories, target_equilibria)
        )
        n_classes = target_equilibria.shape[0]
        label_mode = "equilibrium"

    max_best_final_distance = float(np.max(best_final_distances))
    labels = np.repeat(labels_per_trajectory, trajectories.shape[1] - 1)

    save_kwargs: dict[str, Any] = {
        "trajectories": trajectories,
        "X": X,
        "Y": Y,
        "labels": labels,
        "labels_per_trajectory": labels_per_trajectory,
        "terminal_distances": terminal_distances,
    }
    if target_equilibria is not None:
        save_kwargs["target_equilibria"] = target_equilibria
    if limit_cycle_trajectories is not None:
        save_kwargs["target_limit_cycle_trajectories"] = np.stack(
            limit_cycle_trajectories, axis=0
        )
    np.savez(npz_path, **save_kwargs)

    class_counts = {
        str(class_idx): int(np.sum(labels_per_trajectory == class_idx))
        for class_idx in range(n_classes)
    }
    meta: dict[str, Any] = {
        "system_name": str(cfg["system_name"]),
        "data_subfolder": str(cfg["data_subfolder"]),
        "file_name": file_name,
        "num_traj": num_traj,
        "traj_length": traj_length,
        "label_mode": label_mode,
        "class_counts_per_trajectory": class_counts,
        "max_best_final_distance": max_best_final_distance,
    }
    if "target_equilibria" in cfg:
        meta["target_equilibria"] = cfg["target_equilibria"]
    if use_limit_cycles:
        meta["target_limit_cycle_data_path"] = [
            str(p.relative_to(_ROOT_DIR)) if p.is_relative_to(_ROOT_DIR) else str(p)
            for p in limit_cycle_paths
        ]
    if cfg["seed"] is not None:
        meta["seed"] = int(cfg["seed"])
    if cfg["verbose"]:
        meta["verbose"] = True

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    target_name = "limit cycles" if use_limit_cycles else "equilibria"
    print(
        f"Max best final distance to {target_name} (over trajectories): "
        f"{max_best_final_distance:.6g}"
    )
    print(f"Validation data saved to {npz_path}")
    print(f"  trajectories: {trajectories.shape}")
    print(f"  X: {X.shape}, Y: {Y.shape}")
    print(f"  labels (snapshot): {labels.shape}")
    print(f"  labels_per_trajectory: {labels_per_trajectory.shape}")
    print(f"Metadata saved to {meta_path}")
    print(f"Class counts (per trajectory): {class_counts}")

    if bool(cfg.get("show_worst")):
        worst_idx = int(np.argmax(best_final_distances))
        _show_worst_trajectory_plot(
            trajectories,
            worst_idx=worst_idx,
            worst_distance=float(best_final_distances[worst_idx]),
            assigned_label=int(labels_per_trajectory[worst_idx]),
            system_name=str(cfg["system_name"]),
            use_limit_cycles=use_limit_cycles,
            limit_cycle_trajectories=limit_cycle_trajectories,
            target_equilibria=target_equilibria,
        )


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: generate_validation_data.py [CONFIG.json]")
    cfg_path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else None
    if cfg_path is not None and not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    main(cfg_path)