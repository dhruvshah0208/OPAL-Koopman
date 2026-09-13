# Switched_Linear_Models

Code for "Learning Omega-Limit Partitions from Approximate Koopman Invariants"
(`docs/acc_27_multiple-linear-models.tex`). Recovers the omega-limit partition of a
dynamical system from approximate Koopman invariant observables (Algorithm 1 in the paper),
then fits one local EDMD model per partition cell.

## Environment

Always run code with the `macbook_env` conda environment:

```bash
conda activate macbook_env
```

(Python 3.11.4, numpy 2.2, scipy 1.13, scikit-learn 1.7.) The scripts depend on two
packages that live outside this folder and are importable only inside that env:

- `src` (basis functions `RadialBasis`, `PolynomialBasis`, `CombinedDictionary`, pruning)
  -> editable install of `../MPV-SPV-pruning-code`.
- `src_deep_koopman` (`SystemsAutonomous` dynamical-system simulator)
  -> editable install of `~/Documents/PhD-code-files/Benchmarking_Deep_Koopman`.

When running headless (no display), set `MPLBACKEND=Agg`; several scripts call
`plt.show()` when `show_plots` is true.

## Pipeline (run from the project root, each script takes one JSON config)

```bash
python data/generate_data.py            configs/<sys>/generate_<sys>_data.json   # training snapshots + K-means RBF centers
python data/generate_validation_data.py configs/<sys>/generate_validation_data.json  # labeled validation trajectories
python build_partition.py               configs/<sys>/build_partition.json       # GEP -> cluster -> merge  (Algorithm 1)
python validation/validate_partition.py configs/<sys>/validate_partition.json    # anchor-based partition accuracy
python build_multiple_linear_models.py  configs/<sys>/build_multiple_linear_models.json  # local EDMD per cell + rollout metrics + figures
python generate_plots.py                                                         # paper figures from saved plot_data.npz
```

`<sys>` is `duffing` or `bistable_oscillations` (paper examples) or `bistable` (1-D toy).
Config paths are relative to the project root. Datasets are already generated under `data/`;
steps 3-5 can be run without regenerating.

## Gotchas

- `bistable_oscillations` is NOT registered in the currently installed `src_deep_koopman`
  `systems_dict`, so `generate_data.py` cannot regenerate that dataset; the saved `.npz`
  files still work for the partition/model steps.
- The first GEP mode is the constant function when the dictionary contains the constant
  (`d_poly >= 0`). Asking k-means to split it (`scalar_cluster_count[0] > 1`) splits on
  floating-point noise and is not reproducible across machines. Use `1` for that mode
  (or `skip_first_mode: true`). Duffing with `[1, 4]` reproduces Table I exactly.
- RBF convention in code is `exp(-gamma * ||x - c||^2)` (gamma = precision); the paper's
  `exp(-r^2 / 2 gamma^2)` text is inconsistent with the code.
- `build_classifier.py`, `build_pruned_classifier.py`, `run_pruning.py`,
  `validation/validate_classifier.py` are the older single-eigenfunction "Tier 1/2"
  classifier approach, not used in the paper.
