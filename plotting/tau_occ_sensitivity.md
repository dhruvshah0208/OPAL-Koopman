# Sensitivity of Algorithm 1 to the occupancy threshold `tau_occ`

*Experiment run 2026-09-17 on the committed datasets, with every other parameter fixed to the
paper values in `configs/<system>/build_partition.json`.*

## Question

Can `tau_occ` be set to zero for all examples? If yes, what is it for?

## Answer

**No.** `tau_occ` is irrelevant for Duffing, mildly harmful for the 1-D `bistable` toy, and
**essential** for `bistable_oscillations`: with `tau_occ = 0` the two limit cycles collapse into a
single component and the partition fails.

Its role is not cosmetic. The transition merge rule of Algorithm 1 compares an empirical
*probability* `p_ab = n_ab / n_a` against `tau_tr`. For a fine cell with very few points the
denominator is tiny, so one or two stray transitions produce a large `p_ab`. A low-occupancy cell
that straddles the separatrix and contains points from both basins therefore transitions into
**both** attractors and becomes a bridge that merges them. `tau_occ` removes such cells before the
transition graph is built. Equivalently, it is a minimum-count guard on the denominator of
`p_ab`: an edge needs at least `tau_occ * N * tau_tr` transitions.

| System | N | tau_occ (paper) | tau_tr | Min. cell size | Min. transitions for an edge |
|---|---|---|---|---|---|
| duffing | 50 000 | 0.002 | 0.005 | 100 | 0.5 (one suffices) |
| bistable | 10 000 | 0.002 | 0.05 | 20 | 1 |
| bistable_oscillations | 25 000 | 0.0008 | 0.1 | 20 | 2 |

## How the sweep was run

For each system the dictionary, GEP, and invariant embedding were computed once. Then
`fit_graph_merged_partition` was re-run for a grid of `tau_occ` values (`scalar_cluster_count`,
`tau_tr`, `tau_val`, `random_state` at the paper values). Each resulting partition was applied to
the validation set and scored with the anchor-based classifier used by `validate_partition.py`
(`target_equilibria` for duffing/bistable, terminal snapshots of the reference limit cycles for
bistable_oscillations). "val. dropped" is the fraction of validation points that land in a
dropped or unseen fine cell (label −1). "FAILS" means the anchors could not be mapped to distinct
components, i.e. the partition is unusable.

## Results

### duffing (N = 50 000, paper `tau_occ = 0.002`)

With `(k1, k2) = (1, 4)` only 4 fine cells exist and the smallest holds ~500 points, so no cell
is ever dropped in the range tested.

| tau_occ | min count | retained / fine | transition-merged | final | val. acc. | val. dropped |
|---|---|---|---|---|---|---|
| 0 through 0.1 (all values) | 0 – 5000 | 4 / 4 | 2 | 2 | 99.86 % | 0.00 % |

### bistable (1-D, N = 10 000, paper `tau_occ = 0.002`)

`tau_occ = 0` gives the best result. Raising it only drops data and, in one band, produces a
spurious third component by breaking a chain of small cells.

| tau_occ | min count | retained / fine | transition-merged | final | val. acc. | val. dropped |
|---|---|---|---|---|---|---|
| 0 | 0 | 38 / 38 | 2 | 2 | **99.98 %** | 0.01 % |
| 1e-5 – 1e-4 | 1 | 38 / 38 | 2 | 2 | 99.98 % | 0.01 % |
| 2e-4 | 2 | 29 / 38 | 2 | 2 | 99.96 % | 0.03 % |
| 4e-4 | 4 | 27 / 38 | 2 | 2 | 99.95 % | 0.04 % |
| 8e-4 – 1e-3 | 8 – 10 | 19 / 38 | 3 | 3 | 99.83 % | 0.17 % |
| **0.002 (paper)** | 20 | 15 / 38 | 2 | 2 | 99.77 % | 0.23 % |
| 0.005 | 50 | 11 / 38 | 2 | 2 | 99.56 % | 0.44 % |
| 0.01 | 100 | 9 / 38 | 2 | 2 | 99.30 % | 0.70 % |
| 0.02 | 200 | 8 / 38 | 2 | 2 | 98.94 % | 1.06 % |
| 0.05 | 500 | 5 / 38 | 3 | 3 | 94.23 % | 5.77 % |
| 0.1 | 1000 | 2 / 38 | 2 | 2 | 91.85 % | 8.15 % |

Why `tau_occ = 0` is safe here: the separatrix of the 1-D system is a single point (x = 0), so
none of the 23 low-occupancy cells straddle it. Every one of them transitions into cells of a
single component, so none can act as a bridge. (The 1-D toy also has four degenerate GEP
eigenvalues at ~1e-9, so the *intermediate* cell counts are sensitive to floating-point noise;
the final 2-cell result is stable.)

### bistable_oscillations (N = 25 000, paper `tau_occ = 8e-4`)

| tau_occ | min count | retained / fine | transition-merged | final | val. acc. | val. dropped |
|---|---|---|---|---|---|---|
| **0** | 0 | 32 / 32 | 1 | **1 — FAILS** | — | — |
| 1e-5 – 5e-5 | 1 – 2 | 32 / 32 | 1 | 1 — FAILS | — | — |
| 1e-4 | 3 | 31 / 32 | 1 | 1 — FAILS | — | — |
| 2e-4 | 5 | 30 / 32 | 1 | 1 — FAILS | — | — |
| 4e-4 | 10 | 29 / 32 | 1 | 1 — FAILS | — | — |
| **8e-4 (paper)** – 1e-3 | 20 – 25 | 27 / 32 | 2 | 2 | **99.80 %** | 0.05 % |
| 0.002 | 50 | 23 / 32 | 4 | 2 | 99.70 % | 0.17 % |
| 0.005 | 125 | 18 / 32 | 4 | 2 | 99.40 % | 0.48 % |
| 0.01 | 250 | 11 / 32 | 5 | 2 | 98.44 % | 1.52 % |
| 0.02 | 500 | 6 / 32 | 3 | 2 | 97.35 % | 2.65 % |
| 0.05 | 1250 | 5 / 32 | 3 | 2 | 96.79 % | 3.21 % |
| 0.1 | 2500 | 3 / 32 | 1 | 1 — FAILS (anchor dropped) | — | — |

Both anchors (the r = 1 and r = 3 limit cycles) land in the same component for every
`tau_occ < 6.8e-4`, so the classifier cannot be built at all.

## Anatomy of the failure in bistable_oscillations

At `tau_occ = 0` the five cells below the paper threshold are:

| cell | count | weight | radii of its points (min / median / max) | out-transitions `p_ab` |
|---|---|---|---|---|
| **28** | 17 | 6.8e-4 | 1.62 / 1.70 / 3.83 | 5: 0.118 (inner comp.), **18: 0.176 (outer comp.)**, 1: 0.06, 22: 0.06, 0: 0.06, 31: 0.06 |
| 8 | 14 | 5.6e-4 | 3.30 / 3.53 / 4.60 | 9: 0.64, 1: 0.07 |
| 29 | 5 | 2.0e-4 | 0.78 / 0.83 / 0.87 | 21: 0.20 |
| **30** | 3 | 1.2e-4 | 1.37 / 1.39 / 3.86 | **22: 0.33 (outer comp.), 0: 0.33 (inner comp.)** |
| 31 | 2 | 0.8e-4 | 2.04 / 3.72 / 5.40 | 4: 1.0 |

Cells 28 and 30 are the bridges. Their points span radii on both sides of the unstable cycle at
r = 2, so they are scraps of invariant-value space that the k-means product labelling assigned to
neither basin cleanly. Because `n_28 = 17` and `n_30 = 3`, two or three transitions are enough to
exceed `tau_tr = 0.1` into both components. Ablation (recomputing the merge graph with individual
cells removed from the retained set):

| retained set | final components |
|---|---|
| all 32 cells | 1 |
| all minus cell 8, 28, 29, 30, or 31 (each alone) | 1 |
| all minus cells 28 **and** 30 | **2** |

Removing only these two cells is necessary and sufficient. Cells 8, 29, 31 are harmless: each
transitions into a single component only.

## Observations for the paper

1. The paper value `tau_occ = 8e-4` (20 points) sits just above the 17-point bridge cell
   (weight 6.8e-4). It works, but it is at the edge of the safe window, which is roughly
   `[8e-4, 0.05]`; values around `1e-3` – `2e-3` give the best accuracy (99.7–99.8 %) with a
   comfortable margin.
2. The cost of `tau_occ` is monotone: every dropped fine cell becomes label −1 at inference, so
   the "dropped" fraction on new data grows with `tau_occ`, and eventually an anchor itself is
   dropped (bistable_oscillations at 0.1).
3. Whether `tau_occ` is needed depends on the geometry of the separatrix relative to the fine
   cells. For Duffing the chosen `(k1, k2)` produces no small cells at all; for the 1-D toy the
   separatrix is a point that no cell straddles; for the bistable oscillator the separatrix is a
   closed curve and a few tiny cells straddle it.
4. A count-based transition rule (for example additionally requiring `n_ab >= n_min`, or
   normalising by the larger of the two cells) would remove the bridge mechanism at its source
   and would likely allow `tau_occ = 0` everywhere. This was not implemented or tested.

## Reproducing

The sweep script (`sweep_tau_occ.py`) is not part of the repository; it computes the embedding
once per system, calls `src_omega_partition.partition.fit_graph_merged_partition` for each `tau_occ`
in `[0, 1e-5, 5e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]`, and
scores the result with `src_omega_partition.anchors` on the validation data. It reads only the
committed `data/<system>/*.npz` files and writes nothing into `data/` or `figures/`.
