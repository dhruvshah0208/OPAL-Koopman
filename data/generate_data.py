'''

Generate snapshot data from a dynamical system and save .npz + metadata under data/<data_subfolder>/.

Simulates trajectories (``num_traj`` × ``traj_length``), stores ``trajectories``, one-step
snapshot pairs ``X`` / ``Y``, and optional RBF centers ``centers_global`` (K-means or random
subsample, same as ``build_classifier.py``).

Run: ``python generate_data.py`` (uses the module defaults below) or
``python generate_data.py path/to/config.json`` (JSON keys override those defaults).

Recognized JSON keys: data_subfolder, system_name, file_name, num_traj, traj_length,
cut_initial_steps (optional, default 0), initial_conditions (optional list of state vectors),
compute_k_means, n_global_centers, random_center_seed, seed (optional), verbose (optional bool),
show_phase_space (optional bool, default false). When true, display a phase-space plot of all
trajectories (2-D state only; not saved).
If ``initial_conditions`` is set, one trajectory is simulated per listed state and ``num_traj``
is ignored. Otherwise ``num_traj`` random initial conditions are drawn (default count from
``DEFAULT_DATASET`` when omitted). After simulation, the first ``cut_initial_steps`` time
points are removed from every trajectory so the saved data reflects long-term dynamics.
A nested ``dataset`` object with ``num_traj`` and ``traj_length`` is also accepted.
'''

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

DATA_SUBFOLDER = "pendulum"
SYSTEM_NAME = "pendulum"
FILE_NAME = "pendulum_data.npz"
DEFAULT_DATASET = {"num_traj": 400, "traj_length": 500}
DEFAULT_N_GLOBAL_CENTERS = 500
DEFAULT_RANDOM_CENTER_SEED = 42

_DATA_DIR = Path(__file__).resolve().parent
_ROOT_DIR = _DATA_DIR.parent
_SRC_DIR = _ROOT_DIR / "src_deep_koopman"
for _path in (_ROOT_DIR, _SRC_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src_deep_koopman.dynamical_systems import SystemsAutonomous  # noqa: E402
from src_basin_classification import select_rbf_centers  # noqa: E402


def _state_label(system_name: str, state_idx: int) -> str:
    if system_name == "pendulum":
        labels = (r"$\theta$ (rad)", r"$\dot{\theta}$")
        if state_idx < len(labels):
            return labels[state_idx]
    return rf"$x_{state_idx + 1}$"


def _build_trajectories(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    num_traj: int,
) -> np.ndarray:
    if len(X) != len(Y):
        raise ValueError(f"X and Y must have the same number of rows; got {len(X)} and {len(Y)}.")
    if len(X) % num_traj != 0:
        raise ValueError(
            f"Cannot split {len(X)} snapshot pairs evenly across {num_traj} trajectories."
        )
    snapshots_per_traj = len(X) // num_traj
    n_state = X.shape[1]
    X_traj = X.reshape(num_traj, snapshots_per_traj, n_state)
    Y_traj = Y.reshape(num_traj, snapshots_per_traj, n_state)
    return np.concatenate([X_traj, Y_traj[:, -1:, :]], axis=1)


def _cut_trajectories(trajectories: np.ndarray, cut_initial_steps: int) -> np.ndarray:
    if cut_initial_steps < 0:
        raise ValueError("cut_initial_steps must be non-negative.")
    if cut_initial_steps == 0:
        return trajectories
    if trajectories.ndim != 3:
        raise ValueError(f"Expected trajectories.ndim == 3; got {trajectories.ndim}.")
    if cut_initial_steps >= trajectories.shape[1]:
        raise ValueError(
            f"cut_initial_steps ({cut_initial_steps}) must be smaller than the "
            f"trajectory length ({trajectories.shape[1]})."
        )
    return trajectories[:, cut_initial_steps:, :]


def _validate_initial_conditions(
    raw: Any,
    *,
    state_dim: int,
) -> np.ndarray:
    if not isinstance(raw, list) or len(raw) == 0:
        raise ValueError("initial_conditions must be a non-empty list of state vectors.")
    initial_conditions = np.asarray(raw, dtype=float)
    if initial_conditions.ndim == 1:
        if state_dim != 1:
            raise ValueError(
                "initial_conditions must be a list of vectors with one entry per state dimension."
            )
        initial_conditions = initial_conditions.reshape(1, -1)
    if initial_conditions.ndim != 2:
        raise ValueError(
            "initial_conditions must be a list of vectors, e.g. [[x1, x2], [x1, x2], ...]."
        )
    if initial_conditions.shape[1] != state_dim:
        raise ValueError(
            f"Each initial condition must have length {state_dim}; "
            f"got shape {initial_conditions.shape}."
        )
    return initial_conditions


def _simulate_trajectories(
    system: SystemsAutonomous,
    *,
    initial_conditions: np.ndarray,
    traj_length: int,
    show_progress: bool,
) -> np.ndarray:
    num_traj = initial_conditions.shape[0]
    traj_data: list[np.ndarray] = []
    x0_iter: Any = initial_conditions
    if show_progress:
        for i, x0 in enumerate(initial_conditions, 1):
            pct = i / num_traj
            width = 40
            filled = int(width * pct)
            bar = "=" * filled + "-" * (width - filled)
            end = "\n" if i == num_traj else ""
            print(
                f"\rSimulating trajectories: |{bar}| {i}/{num_traj} ({100 * pct:.1f}%)",
                end=end,
                flush=True,
            )
            x_traj = system.simulate_trajectory_true(
                x0=x0.reshape(1, -1),
                time_horizon=traj_length,
            )[0]
            traj_data.append(x_traj)
    else:
        for x0 in x0_iter:
            x_traj = system.simulate_trajectory_true(
                x0=x0.reshape(1, -1),
                time_horizon=traj_length,
            )[0]
            traj_data.append(x_traj)
    return np.stack(traj_data, axis=0)


def _snapshots_from_trajectories(trajectories: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One-step pairs from full trajectories of shape (N, T, n_state)."""
    if trajectories.ndim != 3:
        raise ValueError(f"Expected trajectories.ndim == 3; got {trajectories.ndim}.")
    n_state = trajectories.shape[2]
    X = trajectories[:, :-1, :].reshape(-1, n_state)
    Y = trajectories[:, 1:, :].reshape(-1, n_state)
    return X, Y


def _gather_trajectories(
    system: SystemsAutonomous,
    *,
    num_traj: int,
    traj_length: int,
    initial_conditions: np.ndarray | None,
    show_progress: bool,
) -> np.ndarray:
    if initial_conditions is not None:
        return _simulate_trajectories(
            system,
            initial_conditions=initial_conditions,
            traj_length=traj_length,
            show_progress=show_progress,
        )
    X, Y = system.gather_traj_data(
        num_traj=num_traj,
        traj_length=traj_length,
        plot_results=False,
        show_progress=show_progress,
    )
    return _build_trajectories(X, Y, num_traj=num_traj)


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
        raise ValueError(
            "traj_length must be at least 2 to create one-step snapshot pairs."
        )
    return {"num_traj": num_traj, "traj_length": traj_length}


def _plot_time_domain_trajectories(
    trajectories: np.ndarray,
    *,
    system_name: str,
    save_path: Path,
) -> None:
    num_traj, num_time, n_state = trajectories.shape
    time = np.arange(num_time)
    alpha = 0.8 if num_traj <= 20 else 0.15

    fig, axes = plt.subplots(
        n_state,
        1,
        figsize=(10, max(3, 2.4 * n_state)),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    for state_idx, ax in enumerate(axes[:, 0]):
        ax.plot(
            time,
            trajectories[:, :, state_idx].T,
            color="tab:blue",
            linewidth=0.6,
            alpha=alpha,
        )
        ax.set_ylabel(_state_label(system_name, state_idx))
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    axes[-1, 0].set_xlabel("time step")
    fig.suptitle(f"{system_name} trajectories over time ({num_traj} trajectories)")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _show_phase_space_trajectories(
    trajectories: np.ndarray,
    *,
    system_name: str,
) -> None:
    """Display phase-space plot of all trajectories (not saved). Requires 2-D state."""
    num_traj, _, n_state = trajectories.shape
    if n_state != 2:
        raise ValueError(
            f"show_phase_space requires a 2-D state; got state dimension {n_state}."
        )

    alpha = 0.8 if num_traj <= 20 else 0.15
    fig, ax = plt.subplots(figsize=(8, 6), layout="constrained")
    for traj_idx in range(num_traj):
        ax.plot(
            trajectories[traj_idx, :, 0],
            trajectories[traj_idx, :, 1],
            color="tab:blue",
            linewidth=0.6,
            alpha=alpha,
        )
    ax.set_xlabel(_state_label(system_name, 0))
    ax.set_ylabel(_state_label(system_name, 1))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    fig.suptitle(f"{system_name} phase space ({num_traj} trajectories)")
    plt.show()


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


def _settings(config_path: Path | None) -> dict:
    s: dict[str, Any] = {
        "data_subfolder": DATA_SUBFOLDER,
        "system_name": SYSTEM_NAME,
        "file_name": FILE_NAME,
        "dataset": dict(DEFAULT_DATASET),
        "compute_k_means": True,
        "n_global_centers": DEFAULT_N_GLOBAL_CENTERS,
        "random_center_seed": DEFAULT_RANDOM_CENTER_SEED,
        "seed": None,
        "verbose": False,
        "cut_initial_steps": 0,
        "initial_conditions": None,
        "show_phase_space": False,
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
        "seed",
        "verbose",
        "compute_k_means",
        "n_global_centers",
        "random_center_seed",
        "show_phase_space",
    ):
        if key in raw:
            s[key] = raw[key]
    _apply_dataset_overrides(s["dataset"], raw)
    if "cut_initial_steps" in raw:
        s["cut_initial_steps"] = int(raw["cut_initial_steps"])
    if "initial_conditions" in raw:
        s["initial_conditions"] = raw["initial_conditions"]
    return s


def main(config_path: Path | None = None) -> None:
    s = _settings(config_path)
    dataset = _validate_dataset(s["dataset"])
    num_traj, traj_length = dataset["num_traj"], dataset["traj_length"]
    cut_initial_steps = int(s["cut_initial_steps"])
    compute_k_means = bool(s["compute_k_means"])
    n_global_centers = int(s["n_global_centers"])
    random_center_seed = int(s["random_center_seed"])

    fn = s["file_name"]
    if not str(fn).endswith(".npz"):
        raise ValueError("file_name must end with .npz")

    out_dir = _DATA_DIR / s["data_subfolder"]
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / fn
    meta_path = out_dir / str(fn).replace(".npz", ".meta.json")
    plot_stem = Path(fn).stem
    plots_dir = _DATA_DIR / "dataset_plots"

    rng = np.random.default_rng(s["seed"]) if s["seed"] is not None else None
    system = SystemsAutonomous(s["system_name"], rng=rng, verbose=s["verbose"])

    initial_conditions: np.ndarray | None = None
    if s["initial_conditions"] is not None:
        initial_conditions = _validate_initial_conditions(
            s["initial_conditions"],
            state_dim=system.n,
        )
        num_traj = int(initial_conditions.shape[0])

    if cut_initial_steps >= traj_length:
        raise ValueError(
            f"cut_initial_steps ({cut_initial_steps}) must be smaller than traj_length "
            f"({traj_length})."
        )
    saved_traj_length = traj_length - cut_initial_steps
    if saved_traj_length < 2:
        raise ValueError(
            "After cutting initial steps, at least 2 time points must remain to form "
            "one-step snapshot pairs."
        )

    print(f"System: {s['system_name']}")
    if initial_conditions is not None:
        print(
            f"Dataset: {num_traj} trajectories from specified initial conditions, "
            f"simulated for {traj_length} steps"
        )
    else:
        print(f"Dataset: {num_traj} trajectories, simulated for {traj_length} steps")
    if cut_initial_steps > 0:
        print(
            f"  Discarding first {cut_initial_steps} steps; "
            f"saving {saved_traj_length} steps per trajectory"
        )
    if getattr(system, "dt", None) is not None:
        print(f"  Simulated physical horizon: {traj_length * system.dt:.6g} (dt = {system.dt})")
        if cut_initial_steps > 0:
            print(f"  Saved physical horizon: {saved_traj_length * system.dt:.6g}")

    trajectories = _gather_trajectories(
        system,
        num_traj=num_traj,
        traj_length=traj_length,
        initial_conditions=initial_conditions,
        show_progress=True,
    )
    if cut_initial_steps > 0:
        trajectories = _cut_trajectories(trajectories, cut_initial_steps)

    X, Y = _snapshots_from_trajectories(trajectories)

    n_centers = min(n_global_centers, len(X))
    if n_centers < n_global_centers:
        print(
            f"  RBF centers: using {n_centers} "
            f"(requested {n_global_centers}, {len(X)} snapshot pairs available)"
        )

    centers_global = select_rbf_centers(
        X,
        n_centers,
        use_kmeans=compute_k_means,
        random_state=random_center_seed,
        verbose=s["verbose"],
    )

    np.savez(
        npz_path,
        trajectories=trajectories,
        X=X,
        Y=Y,
        centers_global=centers_global,
    )

    plot_path = plots_dir / f"{plot_stem}_time_trajectories.png"
    _plot_time_domain_trajectories(
        trajectories,
        system_name=s["system_name"],
        save_path=plot_path,
    )

    meta: dict[str, Any] = {
        "system_name": s["system_name"],
        "num_traj": num_traj,
        "traj_length": saved_traj_length,
        "simulated_traj_length": traj_length,
        "data_subfolder": s["data_subfolder"],
        "file_name": fn,
        "compute_k_means": compute_k_means,
        "n_global_centers": n_centers,
        "random_center_seed": random_center_seed,
    }
    if n_centers < n_global_centers:
        meta["n_global_centers_requested"] = n_global_centers
    if cut_initial_steps > 0:
        meta["cut_initial_steps"] = cut_initial_steps
    if initial_conditions is not None:
        meta["initial_conditions"] = initial_conditions.tolist()
    if s["seed"] is not None:
        meta["seed"] = s["seed"]
    if s["verbose"]:
        meta["verbose"] = True

    with open(meta_path, "w") as f:
        json.dump(meta, f)

    print(f"Data saved to {npz_path}")
    print(f"  trajectories {trajectories.shape}")
    print(f"  X {X.shape}, Y {Y.shape}")
    print(f"  centers_global {centers_global.shape}")
    print(f"Metadata saved to {meta_path}")
    print(f"Time-domain plot: {plot_path}")

    if bool(s.get("show_phase_space")):
        _show_phase_space_trajectories(trajectories, system_name=s["system_name"])


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: generate_data.py [CONFIG.json]")
    cfg = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else None
    if cfg is not None and not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")
    main(cfg)
