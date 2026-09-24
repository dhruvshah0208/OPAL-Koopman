# Regional-EDMD

Code for "Learning Omega-Limit Partitions from Approximate Koopman Invariants"
(the paper source is not in this repository). See `README.md` for the layout and the
per-stage run commands; this file holds what is not obvious from the code.

## Environment

Always run code with the `macbook_env` conda environment:

```bash
conda activate macbook_env
```

(Python 3.11.4, numpy 2.2, scipy 1.13, scikit-learn 1.7.) Both external research
modules are vendored directly in `src_omega_partition/`: the simulator `SystemsAutonomous`
(`dynamical_systems.py`: `duffing_eqn` and `bistable` copied verbatim from
`src_deep_koopman`, plus `bistable_oscillations`) and the EDMD dictionary classes (`basis_functions.py`, four classes
copied verbatim from `src/basis_functions.py` of `../MPV-SPV-pruning-code`). Import them
from there, not from `src` / `src_deep_koopman`. Use `MPLBACKEND=Agg` when running headless.

## Conventions

- Scripts live in `scripts/` and bootstrap `sys.path` with the project root so that
  `src_omega_partition` imports without installation. Each script takes one JSON config;
  relative paths in configs resolve against the project root (`src_omega_partition.paths.resolve`).
- Library code goes in `src_omega_partition/`; scripts should only orchestrate, print, and save.
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
- `bistable_oscillations_dynamics` must stay the exact flow map (closed-form radial flow,
  theta advances by exactly dt = 0.05): it matches the stored datasets to ~1e-7, while an
  Euler step drifts by ~0.7. Its IC box is [-4, 4]^2 (the stored data used seed 42/41).
- The stored `duffing` datasets were generated with `dt = 5e-2` in
  `duffing_equation_dynamics`; keep the vendored copy at that `dt` or `ICs_phase_plot` "true"
  trajectories are on the wrong time scale (metrics on stored validation trajectories are
  unaffected).
- Rollouts lift once and propagate in the lifted space (`z_{t+1} = z_t K`, decode only for
  storage). The original code and the paper text (`\Psi(x_{t+1}) \approx \Psi(x_t) K`,
  `x_{t+1} \approx \Psi(x_{t+1}) R`) re-lifted the decoded state every step, which is a
  chained one-step predictor and gives ~100x smaller errors; Table I in the paper draft was
  produced that way and must be regenerated.
- The best EDMD dictionary from the sweeps in the walkthrough notebooks is
  `edmd_gamma = 1.0`, `edmd_d_poly = 6`, `number_centers_edmd = 100` for both paper
  systems (now in the configs); `number_centers_edmd = 0` gives a polynomial-only dictionary.
- RBF convention in code is `exp(-gamma * ||x - c||^2)` (gamma = precision); the paper's
  `exp(-r^2 / 2 gamma^2)` text is inconsistent with the code.
- Both `build_multiple_linear_models.json` configs fit the regional EDMD models on the
  validation set (`data_path` == `validation_data_path`); the partition itself is fit on
  the training set.
