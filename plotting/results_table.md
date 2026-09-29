# Experimental setup and results

All numbers and figures come from `notebooks/paper_plots.ipynb`, which runs the pipeline functions
(`scripts/build_partition.py`, `scripts/validate_partition.py`, `scripts/build_multiple_linear_models.py`)
with the configs in `configs/<system>/` and caches every array in `data/paper_plots/<system>/*.npz`
(cache last written 2026-09-23 16:37). The figures in `plotting/` and the
tables below are regenerated from that cache by `plotting/plot_all.py` and `plotting/make_results_table.py`.

## 1. Systems and data

| System | Dynamics | Attractors |
|---|---|---|
| Duffing oscillator | $\dot x_1 = x_2,\ \dot x_2 = -0.5\,x_2 + x_1 - x_1^3$ (RK4, $\Delta t = 0.05$) | stable equilibria $(\pm 1, 0)$, saddle at the origin |
| Bistable oscillations | $\dot r = (r - 2) - (r - 2)^3,\ \dot\theta = 1$ in Cartesian coordinates (exact flow, $\Delta t = 0.05$) | stable limit cycles $r = 1$ and $r = 3$, unstable cycle $r = 2$ |

| System | Training data (partition) | RBF center pool | Validation data (labels, EDMD fit, rollouts) |
|---|---|---|---|
| Duffing oscillator | 500 trajectories × 101 steps, ICs uniform in $[-2, 2]^2$ (seed 42) → 50000 snapshot pairs | 500 k-means centers of the training snapshots (seed 42) | 800 trajectories × 500 steps (seed 42) → 399200 snapshot pairs; labeled by the equilibrium closest to the final state (classes {'0': 396, '1': 404}) |
| Bistable oscillations | 500 trajectories × 51 steps, ICs uniform in $r \in [0, 4]$, $\theta \in [0, 2\pi]$ (seed 42) → 25000 snapshot pairs | 500 k-means centers of the training snapshots (seed 42) | 500 trajectories × 301 steps (seed 41) → 150000 snapshot pairs; labeled by the reference limit cycle closest to the final state (classes {'0': 94, '1': 406}) |

Both EDMD models are fitted on the *validation* snapshot pairs (`data_path` = `validation_data_path`); the
partition is fitted on the training data. The RBF center pool is shared by the partition dictionary (all
centers) and the EDMD dictionary (random subsample).

## 2. Partition (Algorithm 1)

Dictionary: Gaussian RBFs $e^{-\gamma\|x - c\|^2}$ on the full center pool plus monomials up to degree `d_poly`.
Generalized eigenvalue problem with tolerance `epsilon` and regularization `reg`; at most `max_invariant_modes`
invariant modes; per-coordinate k-means with `scalar_cluster_count` clusters (the first mode is the constant
function); occupancy filter `tau_occ`, transition merge `tau_tr`, value merge `tau_val`, robustness margin
`tau_robust` (points closer than this relative distance to a final boundary are dropped, at training and at
inference). Configuration files: `configs/<system>/build_partition.json`.

| Parameter | Duffing oscillator | Bistable oscillations |
|---|---|---|
| `gamma` | 30.0 | 5.0 |
| `d_poly` | 0 | 3 |
| `epsilon` | 0.0001 | 0.0001 |
| `reg` | 1e-08 | 1e-10 |
| `max_invariant_modes` | 2 | 4 |
| `scalar_cluster_count` | [2, 4] | [1, 4] |
| `tau_occ` | 0.002 | 0.0008 |
| `tau_tr` | 0.005 | 0.1 |
| `tau_val` | 0.0 | 0.006 |
| `tau_robust` | 0.05 | 0.2 |
| `random_state` | 42 | 42 |
| resulting cells | 2 | 2 |
| training points dropped | 1.80% | 0.84% |

### Validation of the partition

Each class anchor is classified by the partition; this maps partition ids to classes, and every labeled
validation snapshot is scored against its ground-truth attractor (`configs/<system>/validate_partition.json`).

| System | Anchors |
|---|---|
| Duffing oscillator | target equilibria (-1, 0), (1, 0) |
| Bistable oscillations | final snapshots of the reference trajectories `bistable_oscillations_target_data1.npz`, `bistable_oscillations_target_data2.npz` |

## 3. Regional and vanilla EDMD models

Regional EDMD: one least-squares EDMD operator $K_\ell$ and reconstruction map $R_\ell$ per partition
cell, fitted on the snapshot pairs whose state the partition assigns to that cell (dropped pairs are excluded).
Vanilla EDMD: one global $(K, R)$ fitted on **all** snapshot pairs with the **same** dictionary object. Rollouts
lift the initial state once, $z_0 = \Psi(x_0)$, propagate $z_{t+1} = z_t K$ (the cells are forward invariant, so
the cell of $x_0$ selects $K_\ell$ for the whole rollout) and decode $x_t = z_t R$ for storage only. Configuration files:
`configs/<system>/build_multiple_linear_models.json`; the notebook overrides the dictionary with the best
values from the sweeps in the walkthrough notebooks (`BEST_EDMD`).

| Parameter | Duffing oscillator | Bistable oscillations |
|---|---|---|
| `edmd_gamma` | 1.0 | 1.0 |
| `edmd_d_poly` | 6 | 6 |
| `number_centers_edmd` | 100 | 100 |
| `min_samples_per_cell` | 1 | 1 |
| `prediction_horizon` | 400 | 300 |
| `n_rollout_trajectories` | 50 | 100 |
| `rollout_rel_rmse_times` | [0, 50, 100, 200, 300] | [0, 50, 100, 200, 300] |
| `random_state` | 42 | 2 |
| dictionary size $s$ | 128 | 128 |
| fit pairs (regional / vanilla) | 397508 / 399200 | 149653 / 150000 |
| phase-plot initial conditions | (-1.5, 0.3), (1.3, 0.4) | (1.06, 1.06), (-1.25, -2.17) |

`number_centers_edmd` RBF centers are subsampled from the center pool with `random_state`; the dictionary size
counts them plus the $\binom{d + 2}{2}$ monomials up to degree $d$ in two variables. The gated evaluation samples
`n_rollout_trajectories` validation trajectories with `random_state`, keeps those whose initial condition is
correctly classified, and rolls both models out for `prediction_horizon` steps from the same initial conditions.

## 4. Figures in `plotting/`

| File | Script | Cache | Content |
|---|---|---|---|
| `<system>_partition.png` | `plot_partition.py` | `partition.npz` | training snapshots colored by partition cell (gray = dropped by `tau_robust`) |
| `<system>_validation_phase_portrait.png` | `plot_validation.py` | `validation.npz` | validation snapshots: correct (colored by class), misclassified (× colored by true class), dropped (gray ×), class anchors (★) |
| `<system>_rollout_trajectories.png` | `plot_rollouts.py` | `rollouts.npz` | state time series from the two phase-plot initial conditions: true, regional EDMD, vanilla EDMD (y-limits from the true trajectories) |
| `<system>_rollout_phase_portrait.png` | `plot_rollouts.py` | `rollouts.npz` | the same rollouts in the phase plane over the partition cells |
| `<system>_rollout_errors.png` | `plot_rollout_errors.py` | `rollout_errors.npz` | relative state error vs time step for the gated validation rollouts: thin lines per trajectory, thick lines the mean |

## 5. Results

### Partition classification on the labeled validation snapshots

*All* counts dropped points (robustness margin) as errors; *un-dropped* scores only the classified points.

| System | Snapshots | Accuracy (all) | Accuracy (un-dropped) | Dropped |
|---|---|---|---|---|
| Duffing oscillator | 399200 | 99.57% | 100.00% | 0.42% (1692) |
| Bistable oscillations | 150000 | 99.77% | 100.00% | 0.23% (347) |

### Rollout error on validation trajectories

Mean relative state error $\|\hat{x}_t - x_t\| / \|x_t\|$ over the gated trajectories at the given time step,
and the mean over trajectories of the RMSE along the full horizon.

| System | Model | $t=50$ | $t=100$ | $t=200$ | $t=300$ | Mean rollout RMSE | Trajectories |
|---|---|---|---|---|---|---|---|
| Duffing oscillator | regional EDMD | 2.26e-2 | 1.06e-1 | 3.95e-2 | 1.64e-2 | 4.66e-2 | 47 |
|  | vanilla EDMD | 1.00e-1 | 3.37e-1 | 2.53e-1 | 2.55e-1 | 2.53e-1 | 47 |
| Bistable oscillations | regional EDMD | 9.16e-3 | 5.50e-3 | 4.34e-3 | 4.52e-3 | 1.10e-2 | 97 |
|  | vanilla EDMD | 7.13e-2 | 1.13e-1 | 1.24e-1 | 1.18e-1 | 2.14e-1 | 97 |

## 6. Ablation: no robustness margin (`tau_robust = 0`)

Same pipeline and parameters as above, but the partitions are rebuilt with `tau_robust = 0`
(`plotting/ablation_tau_robust_zero.py`, results in `data/paper_plots/ablation_tau_robust_zero.json`).
Nothing is dropped, so every validation snapshot is classified — including the ones near the boundary
that the margin would have left undecided — and every snapshot pair enters the regional fits. Because
the EDMD models are fitted on the validation pairs, the misclassified pairs are exactly the pairs that
end up in the wrong cell's least-squares problem.

### Partition classification

| System | Setting | Snapshots | Accuracy (all) | Accuracy (un-dropped) | Dropped | Misclassified | Misclassified pairs in the regional fits |
|---|---|---|---|---|---|---|---|
| Duffing oscillator | with margin (0.05) | 399200 | 99.57% | 99.997% | 0.42% (1692) | 10 | 10 / 397508 |
|  | `tau_robust = 0` (0) | 399200 | 99.86% | 99.862% | 0.00% (0) | 549 | 549 / 399200 |
| Bistable oscillations | with margin (0.2) | 150000 | 99.77% | 100.000% | 0.23% (347) | 0 | 0 / 149653 |
|  | `tau_robust = 0` (0) | 150000 | 99.90% | 99.900% | 0.00% (0) | 150 | 150 / 150000 |

### Regional operators

Largest eigenvalue modulus $\rho(K_\ell)$ of each regional operator (the constant observable always
contributes an eigenvalue of exactly 1, so $\rho = 1$ means every other mode contracts or rotates).

| System | Setting | $\rho(K_0)$ | $\rho(K_1)$ | $\rho(K)$ vanilla |
|---|---|---|---|---|
| Duffing oscillator | with margin (0.05) | 1.0000 | 1.0000 | 1.0000 |
|  | `tau_robust = 0` (0) | 1.7647 | 1.0000 | 1.0000 |
| Bistable oscillations | with margin (0.2) | 1.0002 | 1.0003 | 1.0003 |
|  | `tau_robust = 0` (0) | 1.0002 | 1.0003 | 1.0003 |

### Rollout error

*Gated*: trajectories whose initial condition is correctly classified (as in Section 5). *Ungated*: all
sampled trajectories, including those whose initial condition is misclassified and therefore propagated
with the wrong regional model. "Diverged" counts rollouts whose relative error at the end of the horizon is
$\ge 1$ (or not finite) for the regional model; when it is non-zero the mean is dominated by those
rollouts and the median of the final relative error over all rollouts is reported as well. The ungated
rows are omitted where none of the sampled initial conditions is misclassified (identical to gated).

| System | Setting | Evaluation | Model | $t=50$ | $t=100$ | $t=200$ | $t=300$ | Mean rollout RMSE | Diverged | Median final rel. err. |
|---|---|---|---|---|---|---|---|---|---|---|
| Duffing oscillator | with margin (0.05) | gated (47 traj., 0 misclassified ICs) | regional EDMD | 2.26e-2 | 1.06e-1 | 3.95e-2 | 1.64e-2 | 4.66e-2 | 0/47 | 8.43e-3 |
|  |  |  | vanilla EDMD | 1.00e-1 | 3.37e-1 | 2.53e-1 | 2.55e-1 | 2.53e-1 | — | — |
|  | `tau_robust = 0` (0) | gated (50 traj., 0 misclassified ICs) | regional EDMD | diverged | diverged | diverged | diverged | diverged | 24/50 | 3.49e-2 |
|  |  |  | vanilla EDMD | 1.02e-1 | 3.42e-1 | 2.64e-1 | 2.69e-1 | 2.64e-1 | — | — |
| Bistable oscillations | with margin (0.2) | gated (97 traj., 0 misclassified ICs) | regional EDMD | 9.16e-3 | 5.50e-3 | 4.34e-3 | 4.52e-3 | 1.10e-2 | 0/97 | 8.10e-4 |
|  |  |  | vanilla EDMD | 7.13e-2 | 1.13e-1 | 1.24e-1 | 1.18e-1 | 2.14e-1 | — | — |
|  | `tau_robust = 0` (0) | gated (98 traj., 0 misclassified ICs) | regional EDMD | 1.88e-2 | 1.81e-2 | 6.48e-3 | 6.11e-3 | 3.11e-2 | 0/98 | 2.42e-3 |
|  |  |  | vanilla EDMD | 7.08e-2 | 1.15e-1 | 1.26e-1 | 1.20e-1 | 2.20e-1 | — | — |
|  | `tau_robust = 0` (0) | ungated (100 traj., 2 misclassified ICs) | regional EDMD | 2.15e-2 | 4.61e-2 | 4.62e-2 | 4.63e-2 | 6.29e-2 | 2/100 | 2.43e-3 |
|  |  |  | vanilla EDMD | 7.33e-2 | 1.29e-1 | 1.40e-1 | 1.36e-1 | 2.31e-1 | — | — |

**Reading.** For the Duffing oscillator the margin is what keeps the regional models stable: without it,
549 snapshot pairs (0.14% of the fit data) that
belong to the other basin enter the fit of one cell, its operator acquires an unstable mode
($\rho(K_0) = 1.76$ instead of $1$), and every rollout that starts in that cell diverges —
24 of the 50 gated rollouts, even though their initial conditions are correctly
classified; the remaining 26 rollouts, which start in the other cell, stay bounded.
The vanilla model is unaffected, so without the margin the regional model is *worse* than the vanilla
one on average — a single unstable direction, learned from 0.14% of the data, is enough. For the bistable
oscillations the 150 misclassified pairs do not destabilize the operators, but the regional error
still grows (4.52e-3 to 6.11e-3 at $t = 300$,
mean RMSE 1.10e-2 to 3.11e-2), and the 2 rollouts
from misclassified initial conditions are propagated with the wrong cycle's model and end with a relative
error of order 1 (ungated mean at $t = 300$: 4.63e-2 vs 6.11e-3 gated).
