"""Write plotting/results_table.md: experimental setup, parameters and result tables.

Everything is read from the sources that produced the cached arrays so the document stays in
sync: configs/<system>/*.json, data/<system>/*.meta.json, the BEST_EDMD / ICS lines of
notebooks/paper_plots.ipynb, and data/paper_plots/<system>/*.npz written by that notebook.

    python plotting/make_results_table.py
"""

import json
from datetime import datetime
from math import comb
from pathlib import Path

import numpy as np

from _common import DATA_DIR, PROJECT_ROOT, SWITCHED_LABEL, VANILLA_LABEL

OUT = Path(__file__).resolve().parent / "results_table.md"
NOTEBOOK = PROJECT_ROOT / "notebooks/paper_plots.ipynb"

SYSTEMS = {
    "duffing": dict(
        title="Duffing oscillator",
        equations=r"$\dot x_1 = x_2,\ \dot x_2 = -0.5\,x_2 + x_1 - x_1^3$ (RK4, $\Delta t = 0.05$)",
        attractors="stable equilibria $(\\pm 1, 0)$, saddle at the origin",
        ic_domain="uniform in $[-2, 2]^2$",
        train_cfg="generate_data",
        val_cfg="generate_validation_data",
        label_rule="by the equilibrium closest to the final state",
    ),
    "bistable_oscillations": dict(
        title="Bistable oscillations",
        equations=r"$\dot r = (r - 2) - (r - 2)^3,\ \dot\theta = 1$ in Cartesian coordinates (exact flow, $\Delta t = 0.05$)",
        attractors="stable limit cycles $r = 1$ and $r = 3$, unstable cycle $r = 2$",
        ic_domain="uniform in $r \\in [0, 4]$, $\\theta \\in [0, 2\\pi]$",
        train_cfg="generate_data",
        val_cfg="generate_validation_data",
        label_rule="by the reference limit cycle closest to the final state",
    ),
}


def cfg(system, stage):
    d = json.load(open(PROJECT_ROOT / f"configs/{system}/{stage}.json"))
    d.pop("explanation", None)
    return d


def meta(path):
    return json.load(open(PROJECT_ROOT / path))


def load(system, name):
    return dict(np.load(DATA_DIR / system / f"{name}.npz", allow_pickle=False))


def notebook_overrides(system):
    """BEST_EDMD and ICS as set in the notebook's per-system header cell."""
    nb = json.load(open(NOTEBOOK))
    for c in nb["cells"]:
        src = "".join(c["source"])
        if c["cell_type"] == "code" and f'SYSTEM, NAME = "{system}"' in src:
            ns = {"np": np}
            for line in src.splitlines():
                if line.startswith(("BEST_EDMD =", "ICS =")):
                    exec(line.split("#")[0], ns)
            return ns["BEST_EDMD"], np.asarray(ns["ICS"], dtype=float)
    raise KeyError(f"{system} not found in {NOTEBOOK}")


def rollout_rmse(abs_curves):
    with np.errstate(all="ignore"):
        return float(np.nanmean(np.sqrt(np.nanmean(abs_curves**2, axis=1))))


def fmt(x):
    return f"{x:.2e}".replace("e-0", "e-").replace("e+0", "e+")


def fmt_ic(x):
    return "(" + ", ".join(f"{v:.3g}" for v in x) + ")"


def md_table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|",
            *("| " + " | ".join(str(v) for v in r) + " |" for r in rows)]


def relpath(p):
    return str(Path(p).relative_to(PROJECT_ROOT)) if str(p).startswith(str(PROJECT_ROOT)) else str(p)


# ----------------------------------------------------------------------------- collect
info = {}
for system, S in SYSTEMS.items():
    train, val = cfg(system, S["train_cfg"]), cfg(system, S["val_cfg"])
    part, valp, mlm = cfg(system, "build_partition"), cfg(system, "validate_partition"), cfg(system, "build_multiple_linear_models")
    best, ics = notebook_overrides(system)
    mlm = {**mlm, **best}
    train_meta = meta(f"data/{system}/{Path(train['file_name']).stem}.meta.json")
    val_meta = meta(f"data/{system}/{Path(val['file_name']).stem}.meta.json")
    val_arr, err, roll, partc = (load(system, n) for n in ("validation", "rollout_errors", "rollouts", "partition"))
    if "target_equilibria" in valp and valp.get("target_equilibria") is not None:
        anchors = "target equilibria " + ", ".join(fmt_ic(a) for a in valp["target_equilibria"])
    else:
        anchors = "final snapshots of the reference trajectories " + ", ".join(
            f"`{Path(p).name}`" for p in valp["target_data_paths"])
    info[system] = dict(S=S, train=train, val=val, part=part, mlm=mlm, best=best, ics=ics, anchors=anchors,
                        train_meta=train_meta, val_meta=val_meta, val_arr=val_arr, err=err, roll=roll, partc=partc)

cache_time = max(p.stat().st_mtime for p in DATA_DIR.glob("*/*.npz"))

# ----------------------------------------------------------------------------- write
L = [
    "# Experimental setup and results",
    "",
    "All numbers and figures come from `notebooks/paper_plots.ipynb`, which runs the pipeline functions",
    "(`scripts/build_partition.py`, `scripts/validate_partition.py`, `scripts/build_multiple_linear_models.py`)",
    "with the configs in `configs/<system>/` and caches every array in `data/paper_plots/<system>/*.npz`",
    f"(cache last written {datetime.fromtimestamp(cache_time):%Y-%m-%d %H:%M}). The figures in `plotting/` and the",
    "tables below are regenerated from that cache by `plotting/plot_all.py` and `plotting/make_results_table.py`.",
    "",
    "## 1. Systems and data",
    "",
]
rows = []
for system, I in info.items():
    S, tr, va, tm, vm = I["S"], I["train"], I["val"], I["train_meta"], I["val_meta"]
    n_pairs_train = tr["num_traj"] * (tr["traj_length"] - 1)
    n_pairs_val = va["num_traj"] * (va["traj_length"] - 1)
    rows.append((S["title"], S["equations"], S["attractors"]))
L += md_table(["System", "Dynamics", "Attractors"], rows) + [""]
rows = []
for system, I in info.items():
    S, tr, va, tm, vm = I["S"], I["train"], I["val"], I["train_meta"], I["val_meta"]
    rows.append((S["title"],
                 f"{tr['num_traj']} trajectories × {tr['traj_length']} steps, ICs {S['ic_domain']} (seed {tr['seed']}) → {tr['num_traj'] * (tr['traj_length'] - 1)} snapshot pairs",
                 f"{tm['n_global_centers']} k-means centers of the training snapshots (seed {tm['random_center_seed']})",
                 f"{va['num_traj']} trajectories × {va['traj_length']} steps (seed {va['seed']}) → {va['num_traj'] * (va['traj_length'] - 1)} snapshot pairs; "
                 f"labeled {S['label_rule']} (classes {vm['class_counts_per_trajectory']})"))
L += md_table(["System", "Training data (partition)", "RBF center pool", "Validation data (labels, EDMD fit, rollouts)"], rows)
L += ["", "Both EDMD models are fitted on the *validation* snapshot pairs (`data_path` = `validation_data_path`); the",
      "partition is fitted on the training data. The RBF center pool is shared by the partition dictionary (all",
      "centers) and the EDMD dictionary (random subsample).", "",
      "## 2. Partition (Algorithm 1)", "",
      "Dictionary: Gaussian RBFs $e^{-\\gamma\\|x - c\\|^2}$ on the full center pool plus monomials up to degree `d_poly`.",
      "Generalized eigenvalue problem with tolerance `epsilon` and regularization `reg`; at most `max_invariant_modes`",
      "invariant modes; per-coordinate k-means with `scalar_cluster_count` clusters (the first mode is the constant",
      "function); occupancy filter `tau_occ`, transition merge `tau_tr`, value merge `tau_val`, robustness margin",
      "`tau_robust` (points closer than this relative distance to a final boundary are dropped, at training and at",
      "inference). Configuration files: `configs/<system>/build_partition.json`.", ""]
keys = ["gamma", "d_poly", "epsilon", "reg", "max_invariant_modes", "scalar_cluster_count", "tau_occ", "tau_tr", "tau_val", "tau_robust", "random_state"]
rows = [(f"`{k}`", *(str(I["part"][k]) for I in info.values())) for k in keys]
n_cells = {s: int(np.max(I["partc"]["labels"]) + 1) for s, I in info.items()}
rows.append(("resulting cells", *(str(n_cells[s]) for s in info)))
rows.append(("training points dropped", *(f"{100 * np.mean(I['partc']['labels'] < 0):.2f}%" for I in info.values())))
L += md_table(["Parameter", *(I["S"]["title"] for I in info.values())], rows)
L += ["", "### Validation of the partition", "",
      "Each class anchor is classified by the partition; this maps partition ids to classes, and every labeled",
      "validation snapshot is scored against its ground-truth attractor (`configs/<system>/validate_partition.json`).", ""]
rows = [(I["S"]["title"], I["anchors"]) for I in info.values()]
L += md_table(["System", "Anchors"], rows)
L += ["", "## 3. Regional and vanilla EDMD models", "",
      "Regional (switched) EDMD: one least-squares EDMD operator $K_\\ell$ and reconstruction map $R_\\ell$ per partition",
      "cell, fitted on the snapshot pairs whose state the partition assigns to that cell (dropped pairs are excluded).",
      "Vanilla EDMD: one global $(K, R)$ fitted on **all** snapshot pairs with the **same** dictionary object. Rollouts",
      "lift the initial state once, $z_0 = \\Psi(x_0)$, propagate $z_{t+1} = z_t K$ (the cell of $x_0$ selects $K_\\ell$;",
      "no reclassification along the rollout) and decode $x_t = z_t R$ for storage only. Configuration files:",
      "`configs/<system>/build_multiple_linear_models.json`; the notebook overrides the dictionary with the best",
      "values from the sweeps in the walkthrough notebooks (`BEST_EDMD`).", ""]
keys = ["edmd_gamma", "edmd_d_poly", "number_centers_edmd", "min_samples_per_cell", "reclassify_each_step",
        "prediction_horizon", "n_rollout_trajectories", "rollout_rel_rmse_times", "random_state"]
rows = [(f"`{k}`", *(str(I["mlm"][k]) for I in info.values())) for k in keys]
rows.append(("dictionary size $s$", *(str(I["mlm"]["number_centers_edmd"] + comb(I["mlm"]["edmd_d_poly"] + 2, 2)) for I in info.values())))
rows.append(("fit pairs (regional / vanilla)", *(f"{int(np.sum(I['roll']['labels'] >= 0))} / {I['roll']['labels'].size}" for I in info.values())))
rows.append(("phase-plot initial conditions", *(", ".join(fmt_ic(x) for x in I["ics"]) for I in info.values())))
L += md_table(["Parameter", *(I["S"]["title"] for I in info.values())], rows)
L += ["", "`number_centers_edmd` RBF centers are subsampled from the center pool with `random_state`; the dictionary size",
      "counts them plus the $\\binom{d + 2}{2}$ monomials up to degree $d$ in two variables. The gated evaluation samples",
      "`n_rollout_trajectories` validation trajectories with `random_state`, keeps those whose initial condition is",
      "correctly classified, and rolls both models out for `prediction_horizon` steps from the same initial conditions.", "",
      "## 4. Figures in `plotting/`", "",
      ]
rows = [
    ("`<system>_partition.png`", "`plot_partition.py`", "`partition.npz`", "training snapshots colored by partition cell (gray = dropped by `tau_robust`)"),
    ("`<system>_validation_phase_portrait.png`", "`plot_validation.py`", "`validation.npz`", "validation snapshots: correct (colored by class), misclassified (× colored by true class), dropped (gray ×), class anchors (★)"),
    ("`<system>_rollout_trajectories.png`", "`plot_rollouts.py`", "`rollouts.npz`", "state time series from the two phase-plot initial conditions: true, regional EDMD, vanilla EDMD (y-limits from the true trajectories)"),
    ("`<system>_rollout_phase_portrait.png`", "`plot_rollouts.py`", "`rollouts.npz`", "the same rollouts in the phase plane over the partition cells"),
    ("`<system>_rollout_errors.png`", "`plot_rollout_errors.py`", "`rollout_errors.npz`", "relative state error vs time step for the gated validation rollouts: thin lines per trajectory, thick lines the mean"),
]
L += md_table(["File", "Script", "Cache", "Content"], rows)
L += ["", "## 5. Results", "",
      "### Partition classification on the labeled validation snapshots", "",
      "*All* counts dropped points (robustness margin) as errors; *un-dropped* scores only the classified points.", ""]
rows = []
for I in info.values():
    gt, pred = I["val_arr"]["gt_labels"], I["val_arr"]["pred_labels"]
    dropped = pred < 0
    n, n_dropped = gt.size, int(dropped.sum())
    n_correct = int(np.sum((~dropped) & (pred == gt)))
    rows.append((I["S"]["title"], str(n), f"{100 * n_correct / n:.2f}%", f"{100 * n_correct / (n - n_dropped):.2f}%",
                 f"{100 * n_dropped / n:.2f}% ({n_dropped})"))
L += md_table(["System", "Snapshots", "Accuracy (all)", "Accuracy (un-dropped)", "Dropped"], rows)
L += ["", "### Rollout error on validation trajectories", "",
      "Mean relative state error $\\|\\hat{x}_t - x_t\\| / \\|x_t\\|$ over the gated trajectories at the given time step,",
      "and the mean over trajectories of the RMSE along the full horizon.", ""]
rows = []
times = None
for I in info.values():
    err = I["err"]
    times = [int(t) for t in err["rel_rmse_times"] if int(t) > 0]
    keep = [i for i, t in enumerate(err["rel_rmse_times"]) if int(t) > 0]
    title = I["S"]["title"]
    for label, key, vals in ((SWITCHED_LABEL, "switched", err["rel_rmse_values"]),
                             (VANILLA_LABEL, "vanilla", err["vanilla_rel_rmse_values"])):
        rows.append((title, label, *(fmt(vals[i]) for i in keep), fmt(rollout_rmse(err[f"abs_{key}"])),
                     str(err[f"rel_{key}"].shape[0])))
        title = ""
L += md_table(["System", "Model", *(f"$t={t}$" for t in times), "Mean rollout RMSE", "Trajectories"], rows) + [""]

# ----------------------------------------------------------------------------- ablation
ABLATION = DATA_DIR / "ablation_tau_robust_zero.json"
if ABLATION.is_file():
    ab = json.load(open(ABLATION))

    def fmt_big(x):
        return "diverged" if not np.isfinite(x) or x > 1e3 else fmt(x)

    L += ["## 6. Ablation: no robustness margin (`tau_robust = 0`)", "",
          "Same pipeline and parameters as above, but the partitions are rebuilt with `tau_robust = 0`",
          "(`plotting/ablation_tau_robust_zero.py`, results in `data/paper_plots/ablation_tau_robust_zero.json`).",
          "Nothing is dropped, so every validation snapshot is classified — including the ones near the boundary",
          "that the margin would have left undecided — and every snapshot pair enters the regional fits. Because",
          "the EDMD models are fitted on the validation pairs, the misclassified pairs are exactly the pairs that",
          "end up in the wrong cell's least-squares problem.", "",
          "### Partition classification", ""]
    rows = []
    for system, I in info.items():
        for key, tag in (("config", "with margin"), ("zero", "`tau_robust = 0`")):
            r = ab[system][key]
            n, nc, nw, nd = r["n_validation"], r["n_correct"], r["n_wrong"], r["n_dropped"]
            rows.append((I["S"]["title"] if key == "config" else "", f"{tag} ({r['tau_robust']:g})", str(n),
                         f"{100 * nc / n:.2f}%", f"{100 * nc / (n - nd):.3f}%", f"{100 * nd / n:.2f}% ({nd})", str(nw),
                         f"{r['fit_pairs_misclassified']} / {r['fit_pairs_regional']}"))
    L += md_table(["System", "Setting", "Snapshots", "Accuracy (all)", "Accuracy (un-dropped)", "Dropped",
                   "Misclassified", "Misclassified pairs in the regional fits"], rows)
    L += ["", "### Regional operators", "",
          "Largest eigenvalue modulus $\\rho(K_\\ell)$ of each local operator (the constant observable always",
          "contributes an eigenvalue of exactly 1, so $\\rho = 1$ means every other mode contracts or rotates).", ""]
    rows = []
    for system, I in info.items():
        for key, tag in (("config", "with margin"), ("zero", "`tau_robust = 0`")):
            r = ab[system][key]
            rows.append((I["S"]["title"] if key == "config" else "", f"{tag} ({r['tau_robust']:g})",
                         *(f"{v:.4f}" for v in r["spectral_radius"].values()), f"{r['spectral_radius_vanilla']:.4f}"))
    L += md_table(["System", "Setting", "$\\rho(K_0)$", "$\\rho(K_1)$", "$\\rho(K)$ vanilla"], rows)
    L += ["", "### Rollout error", "",
          "*Gated*: trajectories whose initial condition is correctly classified (as in Section 5). *Ungated*: all",
          "sampled trajectories, including those whose initial condition is misclassified and therefore propagated",
          "with the wrong local model. \"Diverged\" counts rollouts whose relative error at the end of the horizon is",
          "$\\ge 1$ (or not finite) for the regional model; when it is non-zero the mean is dominated by those",
          "rollouts and the median of the final relative error over all rollouts is reported as well. The ungated",
          "rows are omitted where none of the sampled initial conditions is misclassified (identical to gated).", ""]
    rows = []
    for system, I in info.items():
        r0 = ab[system]["config"]; r1 = ab[system]["zero"]; times = r1["times"]
        for key, tag in (("config", "with margin"), ("zero", "`tau_robust = 0`")):
            r = ab[system][key]
            for gate in ("gated", "ungated"):
                g = r[gate]
                if gate == "ungated" and (key == "config" or g["n_misclassified_ic"] == 0):
                    continue   # identical to the gated rows when no sampled IC is misclassified
                for model, name in (("regional", SWITCHED_LABEL), ("vanilla", VANILLA_LABEL)):
                    vals = [g[model][str(t)] for t in times]
                    div = f"{g['regional_diverged']}/{g['n']}" if model == "regional" else "—"
                    med = fmt(g["regional_median_final"]) if model == "regional" else "—"
                    rows.append((I["S"]["title"] if (key == "config" and model == "regional") else "",
                                 f"{tag} ({r['tau_robust']:g})" if model == "regional" else "",
                                 f"{gate} ({g['n']} traj., {g['n_misclassified_ic']} misclassified ICs)" if model == "regional" else "",
                                 name, *(fmt_big(v) for v in vals), fmt_big(g[f"{model}_rmse"]), div, med))
    L += md_table(["System", "Setting", "Evaluation", "Model", *(f"$t={t}$" for t in times), "Mean rollout RMSE",
                   "Diverged", "Median final rel. err."], rows)
    L += [""]
    d0, d1 = ab["duffing"]["config"], ab["duffing"]["zero"]
    b0, b1 = ab["bistable_oscillations"]["config"], ab["bistable_oscillations"]["zero"]
    L += ["**Reading.** For the Duffing oscillator the margin is what keeps the regional models stable: without it,",
          f"{d1['fit_pairs_misclassified']} snapshot pairs ({100 * d1['fit_pairs_misclassified'] / d1['fit_pairs_regional']:.2f}% of the fit data) that",
          "belong to the other basin enter the fit of one cell, its operator acquires an unstable mode",
          f"($\\rho(K_0) = {d1['spectral_radius']['0']:.2f}$ instead of $1$), and every rollout that starts in that cell diverges —",
          f"{d1['gated']['regional_diverged']} of the {d1['gated']['n']} gated rollouts, even though their initial conditions are correctly",
          f"classified; the remaining {d1['gated']['n'] - d1['gated']['regional_diverged']} rollouts, which start in the other cell, stay bounded.",
          "The vanilla model is unaffected, so without the margin the regional model is *worse* than the vanilla",
          "one on average — a single unstable direction, learned from 0.14% of the data, is enough. For the bistable",
          f"oscillations the {b1['fit_pairs_misclassified']} misclassified pairs do not destabilize the operators, but the regional error",
          f"still grows ({fmt(b0['gated']['regional'][str(b0['times'][-1])])} to {fmt(b1['gated']['regional'][str(b1['times'][-1])])} at $t = {b1['times'][-1]}$,",
          f"mean RMSE {fmt(b0['gated']['regional_rmse'])} to {fmt(b1['gated']['regional_rmse'])}), and the {b1['ungated']['n_misclassified_ic']} rollouts",
          "from misclassified initial conditions are propagated with the wrong cycle's model and end with a relative",
          "error of order 1 (ungated mean at $t = 300$: "
          f"{fmt(b1['ungated']['regional'][str(b1['times'][-1])])} vs {fmt(b1['gated']['regional'][str(b1['times'][-1])])} gated).", ""]
OUT.write_text("\n".join(L))
print(f"Written {OUT}")
