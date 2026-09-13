import json
import sys
from pathlib import Path



import numpy as np
from src_deep_koopman.basis_functions import *
from src_deep_koopman.dynamical_systems import SystemsAutonomous
from src_basin_classification import (
    apply_eigenvector,
    basin_mask_from_eigenfunction,
    compute_approximate_invariant_observables,
    filter_snapshots_by_mask,
    fit_edmd_operator,
    make_phase_space_grid,
    plot_combined_classifier_dashboard,
    sorted_koopman_eigenmodes,
)

_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = _ROOT / "data/duffing/duffing_data.npz"

_DEFAULTS = {
    "data_path": DEFAULT_DATA_PATH,
    "system_name": "duffing_eqn",
    "d_poly": 1,
    "gamma": 10.0,
    "epsilon": 1e-5,
    "idx_EDMD": 1,
    "idx_approximate_invariant": 1,
    "koopman_eigenvalue_target": 1.0,
    "epsilon_fix": 0.0,
    "epsilon_fix_new": 0.0,
    "grid_n": 300,
    "classifier_output_path": _ROOT / "data/duffing/classifiers.npz",
}



def _normalization_scale(phi: np.ndarray) -> float:
    N = phi.shape[0]
    norm = np.linalg.norm(phi) / N
    return 1.0 / norm if norm > 0 else 1.0


def _save_classifiers(
    output_path: Path,
    *,
    v_edmd: np.ndarray,
    v_inv: np.ndarray,
    threshold_edmd: float,
    threshold_inv: float,
    centers_global: np.ndarray,
    gamma: float,
    d_poly: int,
    n: int,
    idx_edmd: int,
    idx_approximate_invariant: int,
    eig_val_edmd,
    eig_val_inv: float,
    system_name: str,
) -> None:
    """Save classifier coefficients and everything needed to rebuild Phi."""
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = _ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        v_edmd=np.asarray(v_edmd),
        v_inv=np.asarray(v_inv),
        threshold_edmd=np.float64(threshold_edmd),
        threshold_inv=np.float64(threshold_inv),
        centers_global=np.asarray(centers_global),
        gamma=np.float64(gamma),
        d_poly=np.int64(d_poly),
        n=np.int64(n),
        idx_EDMD=np.int64(idx_edmd),
        idx_approximate_invariant=np.int64(idx_approximate_invariant),
        eig_val_edmd=np.asarray(eig_val_edmd),
        eig_val_inv=np.float64(eig_val_inv),
        system_name=np.array(system_name),
    )
    print(f"Classifiers saved to {output_path}")


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
    # Legacy config key names
    if "eigenfunction_index" in raw and "idx_EDMD" not in raw:
        raw["idx_EDMD"] = raw["eigenfunction_index"]
    if "idx_new" in raw and "idx_approximate_invariant" not in raw:
        raw["idx_approximate_invariant"] = raw["idx_new"]
    for key in _DEFAULTS:
        if key in raw:
            s[key] = raw[key]
    s["data_path"] = Path(s["data_path"])
    s["classifier_output_path"] = Path(s["classifier_output_path"])
    s["system_name"] = str(s["system_name"])
    s["d_poly"] = int(s["d_poly"])
    s["gamma"] = float(s["gamma"])
    s["epsilon"] = float(s["epsilon"])
    s["idx_EDMD"] = int(s["idx_EDMD"])
    s["idx_approximate_invariant"] = int(s["idx_approximate_invariant"])
    s["koopman_eigenvalue_target"] = float(s["koopman_eigenvalue_target"])
    s["epsilon_fix"] = float(s["epsilon_fix"])
    s["epsilon_fix_new"] = float(s["epsilon_fix_new"])
    s["grid_n"] = int(s["grid_n"])
    if s["idx_EDMD"] < 0:
        raise ValueError("idx_EDMD must be a non-negative integer.")
    if s["idx_approximate_invariant"] < 0:
        raise ValueError("idx_approximate_invariant must be a non-negative integer.")
    if s["grid_n"] < 2:
        raise ValueError("grid_n must be at least 2.")
    return s


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

    d_poly = cfg["d_poly"]
    gamma = cfg["gamma"]

    Phi_RBF = RadialBasis(n, centers=centers_global, gamma=gamma)

    if d_poly == -1:
        Phi = Phi_RBF
    else:
        poly = PolynomialBasis(n=n, d=d_poly)
        Phi = CombinedDictionary(Phi_RBF, poly)

    n_dict = Phi.s
    print(f"Number of Dictionary elements = {n_dict}")

    Phi_X = Phi.evaluate(X)
    Phi_Y = Phi.evaluate(Y)

    lambda_vals, lambda_vecs = compute_approximate_invariant_observables(
        Phi_X, Phi_Y, epsilon=cfg["epsilon"]
    )

    print(f"Dimension of the approximate invariant space = {len(lambda_vals)}")
    print(lambda_vals)

    idx_approximate_invariant = cfg["idx_approximate_invariant"]
    if idx_approximate_invariant >= lambda_vecs.shape[1]:
        raise ValueError(
            f"idx_approximate_invariant={idx_approximate_invariant} is out of range; "
            f"approximate invariant space has {lambda_vecs.shape[1]} modes."
        )
    print(f"approximate invariant eigenvalue at idx_approximate_invariant = {idx_approximate_invariant} is {lambda_vals[idx_approximate_invariant]}")

    K_global = fit_edmd_operator(Phi_X, Phi_Y)
    w_global, v_global = sorted_koopman_eigenmodes(
        K_global, target=cfg["koopman_eigenvalue_target"]
    )

    idx_EDMD = cfg["idx_EDMD"]
    if idx_EDMD >= v_global.shape[1]:
        raise ValueError(
            f"idx_EDMD={idx_EDMD} is out of range; "
            f"Koopman operator has {v_global.shape[1]} eigenmodes."
        )

    if n == 2:
        system = SystemsAutonomous(name=cfg["system_name"])
        xlim = system.X[0].min(), system.X[0].max()
        ylim = system.X[1].min(), system.X[1].max()

        X_grid, V_grid, grid_points = make_phase_space_grid(
            xlim=xlim, ylim=ylim, n=cfg["grid_n"]
        )
        Phi_grid = Phi.evaluate(grid_points)
    else:
        X_grid = V_grid = Phi_grid = None

    eig_val_edmd = w_global[idx_EDMD]

    
    v_edmd_raw = v_global[:, idx_EDMD]
    phi_edmd_data = apply_eigenvector(Phi_X, v_edmd_raw)
    k_edmd = _normalization_scale(phi_edmd_data)
    v_edmd = k_edmd * v_edmd_raw
    phi_edmd_data = k_edmd * phi_edmd_data

    phi_edmd_grid = apply_eigenvector(Phi_grid, v_edmd) if n == 2 else None

    mask_edmd, threshold_edmd, phi_real_edmd = basin_mask_from_eigenfunction(
        phi_edmd_data, epsilon_fix=cfg["epsilon_fix"]
    )
    X_basin_edmd, _ = filter_snapshots_by_mask(X, Y, mask_edmd)

    eig_val_inv = lambda_vals[idx_approximate_invariant]

    
    v_inv_raw = lambda_vecs[:, idx_approximate_invariant]
    phi_inv_data = apply_eigenvector(Phi_X, v_inv_raw)
    k_inv = _normalization_scale(phi_inv_data)
    v_inv = k_inv * v_inv_raw
    phi_inv_data = k_inv * phi_inv_data

    phi_inv_grid = apply_eigenvector(Phi_grid, v_inv) if n == 2 else None
    
    mask_inv, threshold_inv, phi_real_inv = basin_mask_from_eigenfunction(
        phi_inv_data, epsilon_fix=cfg["epsilon_fix_new"]
    )
    

    X_basin_inv, _ = filter_snapshots_by_mask(X, Y, mask_inv)

    if n == 2:
        x_ticks = (float(xlim[0]), 0.0, float(xlim[1]))
        y_ticks = (float(ylim[0]), 0.0, float(ylim[1]))
        _ = plot_combined_classifier_dashboard(
            X_grid,
            V_grid,
            phi_edmd_grid=phi_edmd_grid,
            X=X,
            mask_edmd=mask_edmd,
            phi_real_edmd=phi_real_edmd,
            threshold_edmd=threshold_edmd,
            idx_edmd=idx_EDMD,
            eig_val_edmd=eig_val_edmd,
            phi_inv_grid=phi_inv_grid,
            mask_inv=mask_inv,
            phi_real_inv=phi_real_inv,
            threshold_inv=threshold_inv,
            idx_approximate_invariant=idx_approximate_invariant,
            eig_val_approximate_invariant=eig_val_inv,
            n_basin_edmd=X_basin_edmd.shape[0],
            n_basin_inv=X_basin_inv.shape[0],
            x_ticks=x_ticks,
            y_ticks=y_ticks,
        )
    else:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(phi_real_edmd, bins=100, density=True, alpha=0.75, color="tab:orange")
        axes[0].axvline(threshold_edmd, color="black", linestyle="--", linewidth=1.5)
        axes[0].set_title(f"EDMD eigenfunction histogram (idx={idx_EDMD})")
        axes[0].set_xlabel("Re(phi)")
        axes[0].set_ylabel("Density")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(phi_real_inv, bins=100, density=True, alpha=0.75, color="tab:green")
        axes[1].axvline(threshold_inv, color="black", linestyle="--", linewidth=1.5)
        axes[1].set_title(
            "Approx. invariant histogram "
            f"(idx={idx_approximate_invariant})"
        )
        axes[1].set_xlabel("Re(phi)")
        axes[1].set_ylabel("Density")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(f"Classifier histograms ({n}D system)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    _save_classifiers(
        cfg["classifier_output_path"],
        v_edmd=v_edmd,
        v_inv=v_inv,
        threshold_edmd=threshold_edmd,
        threshold_inv=threshold_inv,
        centers_global=centers_global,
        gamma=gamma,
        d_poly=d_poly,
        n=n,
        idx_edmd=idx_EDMD,
        idx_approximate_invariant=idx_approximate_invariant,
        eig_val_edmd=eig_val_edmd,
        eig_val_inv=eig_val_inv,
        system_name=cfg["system_name"],
    )

if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: build_classifier.py [CONFIG.json]")
    cfg_path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else None
    if cfg_path is not None and not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    main(cfg_path)
