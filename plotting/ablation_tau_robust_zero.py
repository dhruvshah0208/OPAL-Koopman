"""Ablation: rebuild the partitions with ``tau_robust = 0`` (no robustness margin) and re-fit /
re-evaluate the regional and vanilla EDMD models exactly as notebooks/paper_plots.ipynb does.

Nothing is plotted or written to data/<system>/; the numbers go to
data/paper_plots/ablation_tau_robust_zero.json, which make_results_table.py reads.

    python plotting/ablation_tau_robust_zero.py
"""

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from _common import DATA_DIR, PROJECT_ROOT

sys.path[:0] = [str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts")]
import build_partition  # noqa: E402
import validate_partition  # noqa: E402
import build_multiple_linear_models as bmlm  # noqa: E402
from src_omega_partition.anchors import classify_with_anchors  # noqa: E402
from src_omega_partition.paths import resolve  # noqa: E402

OUT = DATA_DIR / "ablation_tau_robust_zero.json"
BEST_EDMD = dict(edmd_gamma=1.0, edmd_d_poly=6, number_centers_edmd=100)   # as in paper_plots.ipynb
SYSTEMS = ("duffing", "bistable_oscillations")


def spectral_radius(K):
    return float(np.max(np.abs(np.linalg.eigvals(K))))


def rel_at_times(m):
    return {str(t): float(m["rel_rmse_at_times"][t]) for t in m["rel_rmse_times"] if t > 0}


def run(system, tau_robust):
    """Partition -> validation classification -> EDMD fits -> gated and ungated rollouts."""
    with contextlib.redirect_stdout(io.StringIO()):
        pcfg = build_partition.load_settings(PROJECT_ROOT / f"configs/{system}/build_partition.json")
        if tau_robust is not None:
            pcfg["tau_robust"] = tau_robust
        part = build_partition.build_partition_from_config(pcfg)
        partition = part["inference"]

        vcfg = validate_partition.load_settings(PROJECT_ROOT / f"configs/{system}/validate_partition.json")
        X_val, labels_val = validate_partition.load_validation_data(resolve(vcfg["validation_data_path"]))
        res = classify_with_anchors(cfg=vcfg, X=X_val, gt_labels=labels_val, partition=partition)

        mcfg = bmlm.load_settings(PROJECT_ROOT / f"configs/{system}/build_multiple_linear_models.json")
        mcfg.update(BEST_EDMD)
        X, Y, traj, centers_global, n = bmlm.load_fit_data(mcfg)
        fit = bmlm.fit_models_from_config(mcfg, partition, X, Y, centers_global)

        ic_ok = bmlm._initial_state_correct_mask(traj, res.correct_mask)
        idx_all, horizon, _ = bmlm._select_rollout_indices(traj, mcfg, ic_correct_mask=None)
        idx_ok = idx_all[ic_ok[idx_all]]
        sw_fn = bmlm._switched_rollout_fn(fit["models"], partition, fit["Phi_edmd"], mcfg)
        van_fn = bmlm._vanilla_rollout_fn(fit["vanilla_model"], fit["Phi_vanilla"])
        metrics = {}
        for name, idx in (("gated", idx_ok), ("ungated", idx_all)):
            m_sw, _ = bmlm._rollout_metrics(traj, idx, horizon, sw_fn, mcfg, n_sampled=len(idx), n_skipped_ic=0)
            m_van, _ = bmlm._rollout_metrics(traj, idx, horizon, van_fn, mcfg, n_sampled=len(idx), n_skipped_ic=0)
            final_sw = m_sw["rel_err_curves"][:, -1]
            metrics[name] = dict(
                n=int(len(idx)), n_misclassified_ic=int(np.sum(~ic_ok[idx])),
                regional=rel_at_times(m_sw), regional_rmse=m_sw["mean_rmse"],
                regional_median_final=float(np.nanmedian(final_sw)),
                regional_diverged=int(np.sum(~(final_sw < 1.0))),   # final rel. error >= 1 or NaN
                vanilla=rel_at_times(m_van), vanilla_rmse=m_van["mean_rmse"],
            )

    pred, gt = res.pred_mapped, res.gt_labels
    dropped = pred < 0
    wrong = (~dropped) & (pred != gt)
    labels = fit["labels"]
    return dict(
        tau_robust=float(pcfg["tau_robust"]),
        n_validation=int(gt.size), n_correct=int(np.sum((~dropped) & ~wrong)), n_wrong=int(wrong.sum()),
        n_dropped=int(dropped.sum()), train_dropped_frac=float(np.mean(part["result"]["final_labels_X"] < 0)),
        fit_pairs_regional=int(np.sum(labels >= 0)), fit_pairs_total=int(labels.size),
        fit_pairs_misclassified=int(np.sum((labels >= 0) & wrong)),   # fit data == validation data
        spectral_radius={str(l): spectral_radius(K) for l, K in fit["models"]["K_by_label"].items()},
        spectral_radius_vanilla=spectral_radius(fit["vanilla_model"]["K"]),
        times=[int(t) for t in mcfg["rollout_rel_rmse_times"] if t > 0],
        **metrics,
    )


if __name__ == "__main__":
    results = {}
    for system in SYSTEMS:
        results[system] = {"config": run(system, None), "zero": run(system, 0.0)}
        for key, r in results[system].items():
            print(f"{system} tau_robust={r['tau_robust']}: misclassified {r['n_wrong']}, dropped {r['n_dropped']}, "
                  f"rho(K_l)={ {k: round(v, 4) for k, v in r['spectral_radius'].items()} }, "
                  f"gated regional rel@{r['times'][-1]} = {r['gated']['regional'][str(r['times'][-1])]:.3g}, "
                  f"diverged {r['gated']['regional_diverged']}/{r['gated']['n']}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"Written {OUT}")
