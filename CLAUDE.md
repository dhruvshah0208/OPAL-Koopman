# Switched_Linear_Models

Code for "Learning Omega-Limit Partitions from Approximate Koopman Invariants"
(`docs/acc_27_multiple-linear-models.tex`). See `README.md` for the layout and the
per-stage run commands; this file holds what is not obvious from the code.

## Environment

Always run code with the `macbook_env` conda environment:

```bash
conda activate macbook_env
```

(Python 3.11.4, numpy 2.2, scipy 1.13, scikit-learn 1.7.) Two dependencies live outside
this repo and are importable only in that env: `src` (editable install of
`../MPV-SPV-pruning-code`) and `src_deep_koopman` (editable install of
`~/Documents/PhD-code-files/Benchmarking_Deep_Koopman`). `omega_partition/paths.py`
appends the `src_deep_koopman` package directory to `sys.path` because its
`dynamical_systems` module does `from utils import ts`; import `omega_partition`
before `src_deep_koopman`. Use `MPLBACKEND=Agg` when running headless.

## Conventions

- Scripts live in `scripts/` and bootstrap `sys.path` with the project root so that
  `omega_partition` imports without installation. Each script takes one JSON config;
  relative paths in configs resolve against the project root (`omega_partition.paths.resolve`).
- Library code goes in `omega_partition/`; scripts should only orchestrate, print, and save.
- Do not write pipeline outputs into `data/` or `figures/` when experimenting — point the
  config's output paths elsewhere; the tracked `.meta.json` files describe how the
  datasets were generated.

## Gotchas

- The first GEP mode is the constant function when the dictionary contains the constant
  (`d_poly >= 0`); its `scalar_cluster_count` entry must be `1`. Duffing with `[1, 4]`
  reproduces Table I exactly (the paper text still says (2, 4)).
- The 1-D `bistable` toy has four GEP eigenvalues at ~1e-9 (degenerate): its fine-cell
  layout changes with BLAS floating-point noise (e.g. whether torch is loaded in the
  process). Final results (2 cells, ~99.8%) are stable; the intermediate counts are not.
- `bistable_oscillations` is not in the installed `src_deep_koopman` `systems_dict`;
  `generate_data.py` cannot regenerate it and `build_multiple_linear_models.py` falls
  back to validation trajectories for phase plots.
- RBF convention in code is `exp(-gamma * ||x - c||^2)` (gamma = precision); the paper's
  `exp(-r^2 / 2 gamma^2)` text is inconsistent with the code.
- Both `build_multiple_linear_models.json` configs fit the local EDMD models on the
  validation set (`data_path` == `validation_data_path`); the partition itself is fit on
  the training set.
- `scripts/legacy/` is the older single-eigenfunction classifier and pruning path; not
  used in the paper.
