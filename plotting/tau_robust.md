# The robustness margin `tau_robust`

`tau_robust` is a post-processing step of Algorithm 1. After the fine cells have been merged into
the final $\omega$-limit cells, it declares a thin band around every boundary between two final
cells to be *undecided* and drops the points in that band (label $-1$) instead of classifying
them. It is set in `configs/<system>/build_partition.json`, stored in `partition_output.npz`, and
applied identically wherever the partition is used: to the training snapshots when the partition is
built, to the validation snapshots in `validate_partition.py`, and to the snapshot pairs and the
initial conditions in `build_multiple_linear_models.py`. `tau_robust = 0` disables it.

## 1. Why it exists

The partition is learned from data, so its boundaries are only approximately the true basin
boundaries (the stable manifold of the saddle for Duffing, the unstable cycle $r = 2$ for the
bistable oscillations). The walkthrough notebooks show where the resulting errors sit: with
`tau_robust = 0` the Duffing partition classifies 99.86% of the validation snapshots correctly, and
essentially every misclassified point lies within a thin band around the learned boundary
(`plot_boundary_distance_histogram`, `plot_misclassified_points`). Points deep inside a cell are
never wrong; points a hair's breadth from the boundary are wrong at roughly the rate one would
expect from the boundary's own positioning error.

Two consumers of the partition care about this:

* **The regional EDMD models.** Each regional model is fitted on the snapshot pairs its cell keeps.
  A misclassified pair near the boundary is a pair whose dynamics belong to the *other* attractor,
  i.e. an outlier in the least-squares fit of that cell's operator. Rollouts are also gated by the
  classification of the initial condition: a misclassified initial condition is propagated with the
  wrong regional model and converges to the wrong attractor.
* **The classification claim itself.** For the paper we want to say that, *when the partition
  decides*, it decides correctly. Dropping the ambiguous band trades a small fraction of
  *undecided* points for 100% accuracy on the points that are decided.

That is exactly what the sweeps show (from `duffing_walkthrough.ipynb` and
`bistable_oscillations_walkthrough.ipynb`, validation snapshots):

| `tau_robust` | Duffing: accuracy on un-dropped | Duffing: dropped | Bistable: accuracy on un-dropped | Bistable: dropped |
|---|---|---|---|---|
| 0.00 | 99.862% | 0.00% | 99.900% | 0.00% |
| 0.02 | 99.945% | 0.16% | 99.915% | 0.03% |
| 0.05 | 99.997% | 0.42% | 99.942% | 0.08% |
| 0.10 | 100.000% | 1.00% | 99.971% | 0.13% |
| 0.20 | 100.000% | 2.31% | 100.000% | 0.23% |
| 0.30 | 100.000% | 4.02% | 100.000% | 0.30% |

The configs use `tau_robust = 0.05` for Duffing and `0.2` for the bistable oscillations. For the
bistable oscillations 0.2 is the smallest grid value with no misclassification left; for Duffing
0.05 leaves 10 misclassified snapshots out of 397,508 classified (99.997%, which rounds to 100.00%
in `results_table.md`), and 0.1 removes those too at the cost of a 1% drop rate. The price of the
chosen values is 0.42% and 0.23% of the validation snapshots left undecided.

## 2. What is a "boundary" and what is "close"

The partition lives in the invariant-embedding space, not in state space. Every state $x$ is
mapped to $F(x) = (f_1(x), \ldots, f_r(x))$, the values of the approximate Koopman-invariant
observables. Along each coordinate $j$ the values $f_j$ were clustered with 1-D $k$-means into
$k_j$ scalar clusters; the *fine cell* of a point is the tuple of its $r$ scalar-cluster labels, and
the merge steps assign every retained fine cell to a final cell (component). Because the clustering
is one-dimensional per coordinate, the decision surfaces are simple: along coordinate $j$, two
adjacent scalar clusters are separated by the **midpoint between their sorted $k$-means centers**.
The whole partition is therefore a grid of axis-aligned boxes in embedding space, and a final cell is
a union of such boxes.

Not every box edge is a boundary of the *final* partition, though — many separate two fine cells
that were merged into the same component. The margin only looks at edges where the final label
actually changes. Concretely, for a point in fine cell $c$ with final label $\ell$, and for each
coordinate $j$:

1. Walk from $c$ to its neighbouring scalar cluster along $j$ (the other coordinates fixed), then
   the next one, and so on, as long as the neighbour still has final label $\ell$. Stop at the first
   neighbour with a different final label — including an unretained or otherwise dropped cell — or at
   the last cluster.
2. Do the same in the other direction.
3. The result is the **merged interval** $[\mathrm{lo}_j, \mathrm{hi}_j]$ of coordinate $j$ over
   which the final label is $\ell$ (with the other coordinates fixed at the point's values). An end
   of the interval is a *boundary* if the walk stopped because the label changed; if it stopped at
   the outermost cluster, the end is not a boundary, and its position is capped by the range of the
   training embedding (`embedding_min` / `embedding_max`) only to give the interval a length.

The **relative boundary distance** of the point along coordinate $j$ is

$$
d_j(x) = \frac{\min\big(f_j(x) - \mathrm{lo}_j,\ \mathrm{hi}_j - f_j(x)\big)\ \text{(over boundary ends only)}}{\mathrm{hi}_j - \mathrm{lo}_j},
$$

i.e. the distance to the nearest *label-changing* edge, measured in units of the length of the
region the point sits in along that coordinate. If neither end is a boundary, $d_j = \infty$. The
overall relative distance is the worst coordinate,

$$
d(x) = \min_j d_j(x), \qquad\text{and the point is dropped iff } d(x) < \texttt{tau\_robust}.
$$

So `tau_robust = 0.05` means: a point is undecided if, along some invariant coordinate, it lies in
the outer 5% of its region next to a boundary where the label flips. Because the distance is
*relative* to the region length, the same value is meaningful for coordinates with very different
scales (the invariant observables are not normalized), and it adapts automatically if the merge
steps produce wide or narrow regions. Dividing by the region length is also why a coordinate that
does not separate cells at all (e.g. the constant first mode, $k_1 = 1$) never contributes: its
interval has no label-changing end, so $d_j = \infty$.

Implementation: `src_omega_partition.partition.compute_relative_boundary_distances` computes
$d(x)$ (and the per-coordinate $d_j$), `apply_robustness_margin` sets the labels below the threshold
to $-1$, and `predict_graph_merged_labels` calls both whenever `tau_robust > 0`.

## 3. Where it is applied

* **Fitting** (`fit_graph_merged_partition`): after merging, the margin is applied to the training
  snapshots `X` *and* to their successors `Y` (`final_labels_X`, `final_labels_Y`); the number of
  dropped training points is reported (`n_robust_dropped_X`) and shown in gray in the partition
  scatter plots. The merge steps themselves (occupancy, transition and value merging) run *before*
  the margin and are not affected by it.
* **Inference** (`classify_partition_labels`, `predict_graph_merged_labels`): the same
  `tau_robust`, cluster centers and training embedding range are read from the archive, so a new
  state is dropped under exactly the rule used at training time.
* **Validation** (`validate_partition.py`): dropped validation snapshots are counted separately
  ("dropped predictions") and excluded from the "accuracy (un-dropped data)" line; the histogram of
  relative boundary distances shows the margin as a vertical line, with the remaining
  misclassifications (if any) just outside it.
* **Regional EDMD** (`build_multiple_linear_models.py`): dropped snapshot pairs are excluded from
  every regional fit (the vanilla EDMD baseline, which does not use the partition, is fitted on all
  pairs); rollouts whose initial condition is dropped or misclassified are skipped in the gated
  evaluation. Rollouts classify only the initial condition (the cells are forward invariant), so
  the margin never interrupts a rollout.

## 4. Choosing the value

`tau_robust` is the last parameter to set, after the partition has the right number of cells.
Sweep it on the validation data and pick the smallest value at which the un-dropped accuracy
reaches 100%; anything larger only increases the undecided fraction without changing what is
decided (the sweep tables above are flat beyond that point). How large that value is depends on
how well the learned boundary matches the true one — Duffing's boundary (a curved stable manifold
approximated by a grid of boxes) needs 5%, the circular $r = 2$ boundary of the bistable
oscillations is captured less tightly by the clustered invariant coordinates and needs 20% — and on
how densely the data sample the boundary region. The margin does not repair a wrong partition: if
two attractors share a final cell, no band around the *existing* boundaries separates them, so the
sweep of `tau_robust` should only be run once `tau_tr` / `tau_val` give the expected number of
cells.
