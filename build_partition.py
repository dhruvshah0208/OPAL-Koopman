import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent
_KOOPMAN_ROOT = _ROOT.parent
_SRC_DEEP_KOOPMAN = _ROOT / "src_deep_koopman"
_PRUNING_ROOT = _KOOPMAN_ROOT / "MPV-SPV-pruning-code"
for _path in (_ROOT, _SRC_DEEP_KOOPMAN, _PRUNING_ROOT):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from src_basin_classification import (
    build_rbf_polynomial_dictionary,
    compute_approximate_invariant_observables,
    evaluate_invariant_embedding,
    fit_graph_merged_partition,
    plot_inter_centroid_distances,
    plot_invariant_embedding_2d,
    plot_invariant_scalar_histograms,
    plot_state_partition_2d,
    select_invariant_modes,
)

DEFAULT_CONFIG_PATH = _ROOT / "configs/duffing/build_partition.json"

_DEFAULTS = {
    "data_path": _ROOT / "data/duffing/duffing_data.npz",
    "output_path": _ROOT / "data/duffing/partition_output.npz",
    "system_name": "duffing_eqn",
    "gamma": 1.0,
    "d_poly": -1,
    "epsilon": 1e-7,
    "reg": 1e-6,
    "max_invariant_modes": 5,
    "skip_first_mode": True,
    "scalar_cluster_count": 10,
    "tau_occ": 0.002,
    "tau_tr": 0.05,
    "tau_val": 0.1,
    "random_state": 42,
    "show_plots": True,
}


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
    s["output_path"] = Path(s["output_path"])
    s["system_name"] = str(s["system_name"])
    s["gamma"] = float(s["gamma"])
    s["d_poly"] = int(s["d_poly"])
    s["epsilon"] = float(s["epsilon"])
    s["reg"] = float(s["reg"])
    s["max_invariant_modes"] = int(s["max_invariant_modes"])
    s["skip_first_mode"] = bool(s["skip_first_mode"])
    s["scalar_cluster_count"] = raw.get("scalar_cluster_count", s["scalar_cluster_count"])
    s["tau_occ"] = float(s["tau_occ"])
    s["tau_tr"] = float(s["tau_tr"])
    s["tau_val"] = float(s["tau_val"])
    s["random_state"] = int(s["random_state"])
    s["show_plots"] = bool(s["show_plots"])
    return s


def _stack_cluster_centers(cluster_models) -> np.ndarray:
    """Stack KMeans cluster centers as (r, k_max, 1)."""
    centers = []
    for model in cluster_models:
        centers.append(np.asarray(model.cluster_centers_, dtype=np.float64))
    max_k = max(center.shape[0] for center in centers)
    stacked = np.full((len(centers), max_k, 1), np.nan, dtype=np.float64)
    for j, center in enumerate(centers):
        stacked[j, : center.shape[0], 0] = center[:, 0]
    return stacked


def _print_diagnostics(result: dict, n_samples: int, cfg: dict) -> None:
    n_occupied = int(result["n_occupied_fine_cells"])
    retained = result["retained_cell_ids"]
    n_retained = len(retained)
    n_transition_merged = len(result["transition_components"])
    n_final = len(result["components"])
    dropped_frac = float(np.mean(result["final_labels_X"] < 0))

    retained_mask = (result["final_labels_X"] >= 0) & (result["final_labels_Y"] >= 0)
    retained_total = int(np.sum(retained_mask))
    if retained_total > 0:
        consistency = float(
            np.mean(
                result["final_labels_X"][retained_mask]
                == result["final_labels_Y"][retained_mask]
            )
        )
    else:
        consistency = float("nan")

    tau_occ = float(cfg["tau_occ"])
    tau_tr = float(cfg["tau_tr"])
    tau_val = float(cfg["tau_val"])
    scalar_cluster_count = cfg["scalar_cluster_count"]
    n_edges_tr = len(result["merge_edges_transition"])
    n_edges_val = len(result["merge_edges_value"])

    print(f"Partition thresholds: tau_occ={tau_occ}, tau_tr={tau_tr}, tau_val={tau_val}")
    print(f"Scalar cluster count: {scalar_cluster_count}")
    print(f"Occupied fine cells (before occupancy filter): {n_occupied}")
    print(f"Retained fine cells (weight >= tau_occ={tau_occ}): {n_retained}")
    print(f"Transition-merged cells (after tau_tr): {n_transition_merged}")
    print(f"Final merged cells (after tau_val): {n_final}")
    print(
        f"Merge edges: {n_edges_tr} transition edges on fine cells (tau_tr={tau_tr}), "
        f"{n_edges_val} centroid edges on transition-merged nodes (tau_val={tau_val})"
    )
    distances = np.asarray(result["centroid_pair_distances"], dtype=np.float64)
    if distances.size > 0:
        below_tau = int(np.sum(distances <= tau_val))
        print(
            f"Inter-centroid distances (transition-merged nodes): n_pairs={distances.size}, "
            f"min={distances.min():.6g}, max={distances.max():.6g}, "
            f"median={np.median(distances):.6g}, "
            f"pairs <= tau_val: {below_tau}/{distances.size}"
        )
    else:
        print("Inter-centroid distances (transition-merged nodes): n_pairs=0")
    transition_scores = np.asarray(result["transition_pair_scores"], dtype=np.float64)
    if transition_scores.size > 0:
        above_tau = int(np.sum(transition_scores >= tau_tr))
        print(
            f"Inter-node transition scores: n_pairs={transition_scores.size}, "
            f"min={transition_scores.min():.6g}, max={transition_scores.max():.6g}, "
            f"median={np.median(transition_scores):.6g}, "
            f"pairs >= tau_tr: {above_tau}/{transition_scores.size}"
        )
    else:
        print("Inter-node transition scores: n_pairs=0")
    print(
        f"Retained transition consistency: "
        f"{consistency * 100.0:.2f}% over {retained_total} transitions"
    )
    print(f"Fraction of X points dropped (label = -1): {dropped_frac * 100.0:.2f}%")
    print(f"Total samples: {n_samples}")


def _save_partition(
    output_path: Path,
    *,
    result: dict,
    coeffs_inv: np.ndarray,
    eigvals_inv: np.ndarray,
    F_X: np.ndarray,
    F_Y: np.ndarray,
    centers_global: np.ndarray,
    gamma: float,
    d_poly: int,
    n: int,
    system_name: str,
    cfg: dict,
) -> None:
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = _ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    retained_cell_ids = np.asarray(result["retained_cell_ids"], dtype=np.int64)
    cell_weights_items = sorted(result["cell_weights"].items())
    cell_weight_ids = np.array([cell_id for cell_id, _ in cell_weights_items], dtype=np.int64)
    cell_weight_vals = np.array([weight for _, weight in cell_weights_items], dtype=np.float64)

    centroid_ids = np.array(sorted(result["cell_centroids"].keys()), dtype=np.int64)
    centroid_vals = np.stack(
        [result["cell_centroids"][int(cell_id)] for cell_id in centroid_ids], axis=0
    )

    label_to_id = result["label_to_id"]
    sorted_items = sorted(label_to_id.items(), key=lambda item: item[1])
    multi_index_keys = np.array([tuple(key) for key, _ in sorted_items], dtype=object)
    multi_index_cell_ids = np.array([cell_id for _, cell_id in sorted_items], dtype=np.int64)
    cell_to_component_items = sorted(result["cell_to_component"].items())
    cell_to_component_ids = np.array(
        [cell_id for cell_id, _ in cell_to_component_items], dtype=np.int64
    )
    cell_to_component_labels = np.array(
        [label for _, label in cell_to_component_items], dtype=np.int64
    )

    np.savez(
        output_path,
        coeffs_inv=np.asarray(coeffs_inv),
        eigvals_inv=np.asarray(eigvals_inv),
        F_X=np.asarray(F_X),
        F_Y=np.asarray(F_Y),
        final_labels_X=np.asarray(result["final_labels_X"], dtype=np.int64),
        final_labels_Y=np.asarray(result["final_labels_Y"], dtype=np.int64),
        cell_ids_X=np.asarray(result["cell_ids_X"], dtype=np.int64),
        cell_ids_Y=np.asarray(result["cell_ids_Y"], dtype=np.int64),
        retained_cell_ids=retained_cell_ids,
        cell_weight_ids=cell_weight_ids,
        cell_weight_vals=cell_weight_vals,
        cell_centroid_ids=centroid_ids,
        cell_centroid_vals=centroid_vals,
        cluster_centers=_stack_cluster_centers(result["cluster_models"]),
        multi_index_keys=multi_index_keys,
        multi_index_cell_ids=multi_index_cell_ids,
        cell_to_component_ids=cell_to_component_ids,
        cell_to_component_labels=cell_to_component_labels,
        transition_components=np.asarray(result["transition_components"], dtype=object),
        merge_edges_transition=np.asarray(result["merge_edges_transition"], dtype=object),
        merge_edges_value=np.asarray(result["merge_edges_value"], dtype=object),
        centroid_pair_distances=np.asarray(result["centroid_pair_distances"], dtype=np.float64),
        centroid_pair_cell_ids=np.asarray(result["centroid_pair_cell_ids"], dtype=np.int64),
        transition_pair_scores=np.asarray(result["transition_pair_scores"], dtype=np.float64),
        transition_pair_cell_ids=np.asarray(result["transition_pair_cell_ids"], dtype=np.int64),
        components=np.asarray(result["components"], dtype=object),
        centers_global=np.asarray(centers_global),
        gamma=np.float64(gamma),
        d_poly=np.int64(d_poly),
        n=np.int64(n),
        system_name=np.array(system_name),
        epsilon=np.float64(cfg["epsilon"]),
        reg=np.float64(cfg["reg"]),
        max_invariant_modes=np.int64(cfg["max_invariant_modes"]),
        skip_first_mode=np.bool_(cfg["skip_first_mode"]),
        scalar_cluster_count=np.asarray(cfg["scalar_cluster_count"], dtype=object),
        tau_occ=np.float64(cfg["tau_occ"]),
        tau_tr=np.float64(cfg["tau_tr"]),
        tau_val=np.float64(cfg["tau_val"]),
        random_state=np.int64(cfg["random_state"]),
        n_occupied_fine_cells=np.int64(result["n_occupied_fine_cells"]),
        n_retained_fine_cells=np.int64(len(retained_cell_ids)),
        n_transition_merged_cells=np.int64(len(result["transition_components"])),
        n_final_cells=np.int64(len(result["components"])),
    )
    print(f"Partition output saved to {output_path}")


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
            "Regenerate the dataset with generate_data.py."
        )
    X = data["X"]
    Y = data["Y"]
    centers_global = data["centers_global"]
    n = X.shape[1]

    print(" Data loaded from ", data_path)
    print(" Total data points: ", X.shape[0])
    print(f" RBF centers loaded from file: {centers_global.shape}")

    Phi = build_rbf_polynomial_dictionary(n, centers_global, cfg["gamma"], cfg["d_poly"])
    print(f"Number of Dictionary elements = {Phi.s}")

    Phi_X = Phi.evaluate(X)
    Phi_Y = Phi.evaluate(Y)

    lambda_vals, lambda_vecs = compute_approximate_invariant_observables(
        Phi_X, Phi_Y, epsilon=cfg["epsilon"], reg=cfg["reg"]
    )
    eigvals_inv, coeffs_inv = select_invariant_modes(
        lambda_vals,
        lambda_vecs,
        epsilon=cfg["epsilon"],
        max_modes=cfg["max_invariant_modes"],
        skip_first_mode=cfg["skip_first_mode"],
    )
    print(f"Selected invariant modes: {eigvals_inv.size}")
    print("Selected generalized eigenvalues:")
    print(eigvals_inv)

    F_X = evaluate_invariant_embedding(Phi_X, coeffs_inv)
    F_Y = evaluate_invariant_embedding(Phi_Y, coeffs_inv)
    if np.max(np.abs(np.imag(Phi_X @ coeffs_inv))) > 1e-8:
        print("Warning: invariant embedding has non-negligible imaginary part.")

    result = fit_graph_merged_partition(
        F_X,
        F_Y,
        n_clusters=cfg["scalar_cluster_count"],
        tau_occ=cfg["tau_occ"],
        tau_tr=cfg["tau_tr"],
        tau_val=cfg["tau_val"],
        random_state=cfg["random_state"],
    )

    _print_diagnostics(result, X.shape[0], cfg)
    _save_partition(
        cfg["output_path"],
        result=result,
        coeffs_inv=coeffs_inv,
        eigvals_inv=eigvals_inv,
        F_X=F_X,
        F_Y=F_Y,
        centers_global=centers_global,
        gamma=cfg["gamma"],
        d_poly=cfg["d_poly"],
        n=n,
        system_name=cfg["system_name"],
        cfg=cfg,
    )

    if cfg["show_plots"]:
        plot_invariant_scalar_histograms(
            F_X,
            eigvals_inv,
            title="GEP invariant scalar observables",
        )
        if n == 2:
            plot_invariant_embedding_2d(
                F_X,
                labels=result["final_labels_X"],
                title="Invariant embedding (colored by final partition)",
            )
            plot_state_partition_2d(
                X,
                result["final_labels_X"],
                title="Graph-merged omega-limit partition",
            )
            plot_inter_centroid_distances(
                result["centroid_pair_distances"],
                cfg["tau_val"],
            )


if __name__ == "__main__":
    if len(sys.argv) > 2:
        raise SystemExit("usage: build_partition.py [CONFIG.json]")
    cfg_path = Path(sys.argv[1]).resolve() if len(sys.argv) == 2 else DEFAULT_CONFIG_PATH
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    main(cfg_path)
