import json
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_KOOPMAN_ROOT = _ROOT.parent
_SRC_DEEP_KOOPMAN = _ROOT / "src_deep_koopman"
_PRUNING_ROOT = _KOOPMAN_ROOT / "MPV-SPV-pruning-code"
for _path in (_ROOT, _SRC_DEEP_KOOPMAN, _PRUNING_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src.pruning_algorithms import MPV_SPV_rank_one  # noqa: E402

try:
    from src_deep_koopman.basis_functions import (  # noqa: E402
        CombinedDictionary,
        PolynomialBasis,
        RadialBasis,
    )

    def _build_dictionary(n, centers_global, gamma, d_poly):
        Phi_RBF = RadialBasis(n, centers=centers_global, gamma=gamma)
        if d_poly == -1:
            return Phi_RBF
        poly = PolynomialBasis(n=n, d=d_poly)
        return CombinedDictionary(Phi_RBF, poly)

except ImportError:
    from src.basis_functions import CombinedDictionary, PolynomialBasis, RadialBasis  # noqa: E402

    def _build_dictionary(n, centers_global, gamma, d_poly):
        Phi_RBF = RadialBasis(n, centers=centers_global, gamma=gamma)
        if d_poly == -1:
            return Phi_RBF
        poly = PolynomialBasis(n=n, d=d_poly)
        return CombinedDictionary(poly, Phi_RBF)

DEFAULT_CONFIG_PATH = _ROOT / "configs/duffing/run_pruning.json"

_DEFAULTS = {
    "data_path": _ROOT / "data/duffing/duffing_data.npz",
    "data_file_path": _ROOT / "data/duffing/pruning_output.npz",
    "d_poly": 1,
    "gamma": 20.0,
    "epsilon_pre_pruning": 1e-4,
    "epsilon": 1e-5,
    "verbose": True,
}


def _load_pruning_output(output_path: Path) -> dict:
    """Load pruning_output.npz written by _save_pruning_output."""
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = _ROOT / output_path
    if not output_path.is_file():
        raise FileNotFoundError(f"Pruning output not found: {output_path}")

    data = np.load(output_path)
    dims = data["subspace_dims"].astype(int)
    S_saved = {int(dim): data[f"S_saved_{int(dim)}"] for dim in dims}
    sin_decreasing = {
        int(dim): float(val) for dim, val in zip(dims, data["sin_decreasing"], strict=True)
    }
    result = {
        "S_saved": S_saved,
        "sin_decreasing": sin_decreasing,
        "S_final": data["S_final"],
        "subspace_dims": dims,
    }
    for key in ("Phi_X", "Phi_Y", "gamma", "d_poly"):
        if key not in data:
            raise KeyError(
                f"{output_path} has no '{key}' array. "
                "Regenerate the pruning output with run_pruning.py."
            )
        result[key] = data[key]
    result["gamma"] = float(result["gamma"])
    result["d_poly"] = int(result["d_poly"])
    return result


def _save_pruning_output(
    output_path: Path,
    info_out: dict,
    S_final: np.ndarray,
    *,
    Phi_X: np.ndarray,
    Phi_Y: np.ndarray,
    gamma: float,
    d_poly: int,
) -> None:
    """Save MPV_SPV_rank_one output to .npz (dict keys are not natively supported)."""
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = _ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not isinstance(info_out, dict):
        raise TypeError(f"info_out must be a dict, got {type(info_out).__name__}.")

    S_saved = info_out.get("S_saved")
    sin_decreasing = info_out.get("sin_decreasing")
    if not isinstance(S_saved, dict) or not isinstance(sin_decreasing, dict):
        raise KeyError("info_out must contain 'S_saved' and 'sin_decreasing' dictionaries.")

    dims = np.array(sorted(S_saved.keys()), dtype=np.int64)
    sin_vals = np.array([sin_decreasing[int(dim)] for dim in dims], dtype=np.float64)

    save_kwargs = {
        "subspace_dims": dims,
        "sin_decreasing": sin_vals,
        "S_final": np.asarray(S_final),
        "Phi_X": np.asarray(Phi_X),
        "Phi_Y": np.asarray(Phi_Y),
        "gamma": np.float64(gamma),
        "d_poly": np.int64(d_poly),
    }
    for dim in dims:
        save_kwargs[f"S_saved_{int(dim)}"] = np.asarray(S_saved[int(dim)])

    np.savez(output_path, **save_kwargs)
    print(f"Pruning output saved to {output_path}")


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
    for key in _DEFAULTS:
        if key in raw:
            s[key] = raw[key]
    s["data_path"] = Path(s["data_path"])
    s["data_file_path"] = Path(s["data_file_path"])
    s["d_poly"] = int(s["d_poly"])
    s["gamma"] = float(s["gamma"])
    s["epsilon"] = float(s["epsilon"])
    s["verbose"] = bool(s["verbose"])
    if "epsilon_pre_pruning" in raw:
        s["epsilon_pre_pruning"] = float(s["epsilon_pre_pruning"])
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
    Phi = _build_dictionary(n, centers_global, gamma, d_poly)

    n_dict = Phi.s
    print(f"Number of Dictionary elements = {n_dict}")

    Phi_X = Phi.evaluate(X)
    Phi_Y = Phi.evaluate(Y)

    algo_info = None
    if "epsilon_pre_pruning" in cfg:
        algo_info = {"epsilon_pre_pruning": float(cfg["epsilon_pre_pruning"])}

    t0 = time.time()

    S_final, info_out = MPV_SPV_rank_one(
        Phi_X,
        Phi_Y,
        epsilon=cfg["epsilon"],
        verbose=cfg["verbose"],
        info=algo_info,
        skip_all_plots=True,
    )

    print(f"MPV_SPV_rank_one time: {time.time() - t0:.4f} s")
    print(f"Final pruned subspace dimension: {S_final.shape[1]}")

    _save_pruning_output(
        cfg["data_file_path"],
        info_out,
        S_final,
        Phi_X=Phi_X,
        Phi_Y=Phi_Y,
        gamma=gamma,
        d_poly=d_poly,
    )


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: run_pruning.py [CONFIG.json]")
    cfg_path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    main(cfg_path)
