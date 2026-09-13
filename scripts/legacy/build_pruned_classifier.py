"""
LEGACY: earlier single-eigenfunction basin classifier (not the paper pipeline).
Kept for reference; the paper uses scripts/build_partition.py instead.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root

import run_pruning  # noqa: E402  (sibling legacy script)
from src.helper_functions import compute_lifting_matrices  # noqa: E402

from omega_partition.legacy import (  # noqa: E402
    apply_eigenvector,
    basin_mask_from_eigenfunction,
    filter_snapshots_by_mask,
    make_phase_space_grid,
    sorted_koopman_eigenmodes,
)
from omega_partition.paths import PROJECT_ROOT  # noqa: E402

_ROOT = PROJECT_ROOT

DEFAULT_CONFIG_PATH = _ROOT / "configs/duffing/build_pruned_classifier.json"

_DEFAULTS = {
    "data_path": _ROOT / "data/duffing/duffing_data.npz",
    "pruned_data_path": _ROOT / "data/duffing/pruning_output.npz",
    "dim_S": 10,
    "system_name": "duffing_eqn",
    "idx_EDMD": 0,
    "koopman_eigenvalue_target": 1.0,
    "epsilon_fix": 0.0,
    "grid_n": 300,
    "classifier_output_path": _ROOT / "data/duffing/pruned_classifiers.npz",
}


def _phase_space_limits(X: np.ndarray, system_name: str) -> tuple[float, float, float, float]:
    try:
        from src_deep_koopman.dynamical_systems import SystemsAutonomous  # noqa: E402

        system = SystemsAutonomous(name=system_name)
        return (
            float(system.X[0].min()),
            float(system.X[0].max()),
            float(system.X[1].min()),
            float(system.X[1].max()),
        )
    except ImportError:
        margin = 0.05
        x0, x1 = float(X[:, 0].min()), float(X[:, 0].max())
        y0, y1 = float(X[:, 1].min()), float(X[:, 1].max())
        dx = x1 - x0
        dy = y1 - y0
        return x0 - margin * dx, x1 + margin * dx, y0 - margin * dy, y1 + margin * dy


def _normalization_scale(phi: np.ndarray) -> float:
    norm = np.linalg.norm(phi)
    return 1.0 / norm if norm > 0 else 1.0


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
    if "dim_S" not in raw:
        raise ValueError("Config must include 'dim_S' (pruned subspace dimension).")
    if "pruned_data_path" not in raw:
        raise ValueError(
            "Config must include 'pruned_data_path' (path to .npz from run_pruning.py)."
        )
    for key in _DEFAULTS:
        if key in raw:
            s[key] = raw[key]
    s["data_path"] = Path(s["data_path"])
    s["pruned_data_path"] = Path(s["pruned_data_path"])
    s["classifier_output_path"] = Path(s["classifier_output_path"])
    s["system_name"] = str(s["system_name"])
    s["dim_S"] = int(s["dim_S"])
    s["idx_EDMD"] = int(s["idx_EDMD"])
    s["koopman_eigenvalue_target"] = float(s["koopman_eigenvalue_target"])
    s["epsilon_fix"] = float(s["epsilon_fix"])
    s["grid_n"] = int(s["grid_n"])
    if s["dim_S"] <= 0 and s["dim_S"] != -1:
        raise ValueError("dim_S must be a positive integer or -1 for the full dictionary.")
    if s["idx_EDMD"] < 0:
        raise ValueError("idx_EDMD must be a non-negative integer.")
    if s["grid_n"] < 2:
        raise ValueError("grid_n must be at least 2.")
    return s


def _save_pruned_classifiers(
    output_path: Path,
    *,
    v_edmd: np.ndarray,
    threshold_edmd: float,
    D_pruning: np.ndarray,
    dim_S: int,
    ipt: float,
    centers_global: np.ndarray,
    gamma: float,
    d_poly: int,
    n: int,
    idx_edmd: int,
    eig_val_edmd,
    system_name: str,
    koopman_eigenvalue_target: float,
) -> None:
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = _ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        v_edmd=np.asarray(v_edmd),
        threshold_edmd=np.float64(threshold_edmd),
        D_pruning=np.asarray(D_pruning),
        dim_S=np.int64(dim_S),
        ipt=np.float64(ipt),
        centers_global=np.asarray(centers_global),
        gamma=np.float64(gamma),
        d_poly=np.int64(d_poly),
        n=np.int64(n),
        idx_EDMD=np.int64(idx_edmd),
        eig_val_edmd=np.asarray(eig_val_edmd),
        system_name=np.array(system_name),
        koopman_eigenvalue_target=np.float64(koopman_eigenvalue_target),
    )
    print(f"Pruned classifiers saved to {output_path}")


def _plot_pruned_classifier_dashboard(
    *,
    X_grid,
    V_grid,
    phi_edmd_grid,
    X,
    mask_edmd,
    phi_real_edmd,
    threshold_edmd,
    idx_edmd,
    eig_val_edmd,
    dim_S,
    ipt,
    n_basin_edmd,
    x_ticks,
    y_ticks,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 4), gridspec_kw={"width_ratios": [2.2, 0.9, 2.0, 0.9]})
    phi_grid = np.asarray(phi_edmd_grid).reshape(X_grid.shape)

    im = axes[0].contourf(X_grid, V_grid, phi_grid, levels=60, cmap="RdBu_r")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    axes[0].set_title(f"Pruned EDMD eigenfunction (dim_S={dim_S}, IPT={ipt:.4f})")
    axes[0].set_xlabel(r"$x_1$")
    axes[0].set_ylabel(r"$x_2$")
    axes[0].set_xticks(x_ticks)
    axes[0].set_yticks(y_ticks)

    axes[1].hist(phi_real_edmd, bins=100, density=True, alpha=0.75, color="tab:orange")
    axes[1].axvline(threshold_edmd, color="black", linestyle="--", linewidth=1.5)
    axes[1].set_title("Training histogram")
    axes[1].set_xlabel("Re(phi)")
    axes[1].set_ylabel("Density")
    axes[1].grid(True, alpha=0.3)

    colors = np.where(mask_edmd, "tab:blue", "tab:gray")
    axes[2].scatter(X[:, 0], X[:, 1], c=colors, s=4, alpha=0.35, edgecolors="none")
    axes[2].set_title(f"Basin mask ({n_basin_edmd} points in basin)")
    axes[2].set_xlabel(r"$x_1$")
    axes[2].set_ylabel(r"$x_2$")
    axes[2].set_xticks(x_ticks)
    axes[2].set_yticks(y_ticks)
    axes[2].set_aspect("equal", adjustable="box")

    axes[3].hist(phi_real_edmd, bins=100, density=True, alpha=0.75, color="tab:green")
    axes[3].axvline(threshold_edmd, color="black", linestyle="--", linewidth=1.5)
    axes[3].set_title(f"idx={idx_edmd}, lambda={eig_val_edmd:.4f}")
    axes[3].set_xlabel("Re(phi)")
    axes[3].set_ylabel("Density")
    axes[3].grid(True, alpha=0.3)

    fig.suptitle("Pruned Koopman classifier (EDMD on dim_S subspace)", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main(config_path: Path | None = None) -> None:
    cfg = _settings(config_path)

    data_path = Path(cfg["data_path"])
    if not data_path.is_absolute():
        data_path = _ROOT / data_path
    if not data_path.is_file():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = np.load(data_path)
    if "centers_global" not in data:
        raise KeyError(
            f"{data_path} has no 'centers_global' array. "
            "Regenerate the dataset with generate_data.py (see compute_k_means in its config)."
        )
    X = data["X"]
    Y = data["Y"]
    centers_global = data["centers_global"]
    n = X.shape[1]

    print(" Data loaded from ", data_path)
    print(" Total data points: ", X.shape[0])
    print(f" RBF centers loaded from file: {centers_global.shape}")

    pruning_data = run_pruning._load_pruning_output(cfg["pruned_data_path"])
    print(f" Pruning data loaded from {cfg['pruned_data_path']}")

    Phi_X = pruning_data["Phi_X"]
    Phi_Y = pruning_data["Phi_Y"]
    gamma = pruning_data["gamma"]
    d_poly = pruning_data["d_poly"]
    print(f" Dictionary from pruning output: d_poly={d_poly}, gamma={gamma}")
    print(f"Number of Dictionary elements = {Phi_X.shape[1]}")

    Phi = run_pruning._build_dictionary(n, centers_global, gamma, d_poly)
    dim_S = cfg["dim_S"]
    if dim_S != -1 and dim_S not in pruning_data["S_saved"]:
        available = sorted(pruning_data["S_saved"].keys())
        raise ValueError(
            f"dim_S={dim_S} is not in pruning output. Available subspace dimensions: {available}"
        )

    info_og = {
        "S_saved": pruning_data["S_saved"],
        "sin_decreasing": pruning_data["sin_decreasing"],
        "Phi_X": Phi_X,
        "Phi_Y": Phi_Y,
        "X": X,
        "Phi": Phi,
    }
    if dim_S == -1:
        ipt = float("nan")
        print("Using full dictionary (dim_S = -1)")
    else:
        ipt = pruning_data["sin_decreasing"][dim_S]
        print(f"Using dim_S = {dim_S} with IPT = {ipt:.6f}")

    info = compute_lifting_matrices(dim_S, info_og)
    D_pruning = info["D_pruning"]
    K_EDMD = info["K_EDMD"]

    Phi_X_pruned = Phi_X @ D_pruning
    print(f"Pruned dictionary dimension: {Phi_X_pruned.shape[1]}")

    w_global, v_global = sorted_koopman_eigenmodes(
        K_EDMD, target=cfg["koopman_eigenvalue_target"]
    )

    print("Top 5 EDMD eigenvalues:")
    print(w_global[:5])

    idx_EDMD = cfg["idx_EDMD"]
    if idx_EDMD >= v_global.shape[1]:
        raise ValueError(
            f"idx_EDMD={idx_EDMD} is out of range; "
            f"pruned Koopman operator has {v_global.shape[1]} eigenmodes."
        )

    eig_val_edmd = w_global[idx_EDMD]
    print(f"Selected EDMD eigenvalue at idx_EDMD={idx_EDMD}: {eig_val_edmd}")

    if n == 2:
        x0, x1, y0, y1 = _phase_space_limits(X, cfg["system_name"])
        xlim = (x0, x1)
        ylim = (y0, y1)
        X_grid, V_grid, grid_points = make_phase_space_grid(
            xlim=xlim, ylim=ylim, n=cfg["grid_n"]
        )
        Phi_grid_pruned = Phi.evaluate(grid_points) @ D_pruning
    else:
        X_grid = V_grid = Phi_grid_pruned = None

    v_edmd_raw = v_global[:, idx_EDMD]
    phi_edmd_data = apply_eigenvector(Phi_X_pruned, v_edmd_raw)
    k_edmd = _normalization_scale(phi_edmd_data)
    v_edmd = k_edmd * v_edmd_raw
    phi_edmd_data = k_edmd * phi_edmd_data

    phi_edmd_grid = apply_eigenvector(Phi_grid_pruned, v_edmd) if n == 2 else None

    mask_edmd, threshold_edmd, phi_real_edmd = basin_mask_from_eigenfunction(
        phi_edmd_data, epsilon_fix=cfg["epsilon_fix"]
    )
    X_basin_edmd, _ = filter_snapshots_by_mask(X, Y, mask_edmd)

    effective_dim_S = Phi_X_pruned.shape[1] if dim_S == -1 else dim_S

    if n == 2:
        x_ticks = (float(xlim[0]), 0.0, float(xlim[1]))
        y_ticks = (float(ylim[0]), 0.0, float(ylim[1]))
        _plot_pruned_classifier_dashboard(
            X_grid=X_grid,
            V_grid=V_grid,
            phi_edmd_grid=phi_edmd_grid,
            X=X,
            mask_edmd=mask_edmd,
            phi_real_edmd=phi_real_edmd,
            threshold_edmd=threshold_edmd,
            idx_edmd=idx_EDMD,
            eig_val_edmd=eig_val_edmd,
            dim_S=effective_dim_S,
            ipt=ipt,
            n_basin_edmd=X_basin_edmd.shape[0],
            x_ticks=x_ticks,
            y_ticks=y_ticks,
        )
    else:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(6, 4))
        plt.hist(phi_real_edmd, bins=100, density=True, alpha=0.75, color="tab:orange")
        plt.axvline(threshold_edmd, color="black", linestyle="--", linewidth=1.5)
        plt.title(f"Pruned EDMD eigenfunction histogram (dim_S={effective_dim_S}, idx={idx_EDMD})")
        plt.xlabel("Re(phi)")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    _save_pruned_classifiers(
        cfg["classifier_output_path"],
        v_edmd=v_edmd,
        threshold_edmd=threshold_edmd,
        D_pruning=D_pruning,
        dim_S=effective_dim_S,
        ipt=ipt,
        centers_global=centers_global,
        gamma=gamma,
        d_poly=d_poly,
        n=n,
        idx_edmd=idx_EDMD,
        eig_val_edmd=eig_val_edmd,
        system_name=cfg["system_name"],
        koopman_eigenvalue_target=cfg["koopman_eigenvalue_target"],
    )


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: build_pruned_classifier.py [CONFIG.json]")
    cfg_path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    main(cfg_path)
