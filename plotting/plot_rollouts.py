"""Rollouts from the chosen initial conditions: true vs regional vs vanilla EDMD.

Writes two figures: the state time series (one panel per state) and the phase-plane view over
the partition. Input: data/paper_plots/<system>/rollouts.npz
(true_trajs, regional_trajs, vanilla_trajs: (n_ic, horizon+1, n_state); X, labels: fit data
colored by partition cell; system_name)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from _common import (CELL_FILL_COLORS, CELL_POINT_ALPHA, CELL_POINT_SIZE, COLOR_REGIONAL, COLOR_TRUE, COLOR_VANILLA,
                     LW_REGIONAL, LW_TRUE, LW_VANILLA, REGIONAL_LABEL, VANILLA_LABEL, finish, load, parse_args)


def true_ylim(true_trajs, j, *, pad_frac=0.25, min_pad=0.1):
    """y-limits from the true trajectories only, so a diverging prediction leaves the frame."""
    vals = true_trajs[:, :, j][np.isfinite(true_trajs[:, :, j])]
    lo, hi = float(vals.min()), float(vals.max())
    pad = max(pad_frac * (hi - lo), min_pad)
    return lo - pad, hi + pad


def plot_time_series(true_trajs, regional_trajs, vanilla_trajs):
    n_ic, horizon, n_state = true_trajs.shape
    t = np.arange(horizon)
    series = [(true_trajs, COLOR_TRUE, "-", 0.7, "true"), (regional_trajs, COLOR_REGIONAL, "--", 0.9, REGIONAL_LABEL)]
    if vanilla_trajs.size:
        series.append((vanilla_trajs, COLOR_VANILLA, "-.", 0.9, VANILLA_LABEL))

    fig, axes = plt.subplots(n_state, 1, figsize=(8, 3.0 * n_state), squeeze=False)
    for j in range(n_state):
        ax = axes[j, 0]
        for trajs, color, ls, alpha, label in series:
            for i in range(n_ic):
                valid = np.isfinite(trajs[i, :, j])
                ax.plot(t[valid], trajs[i, valid, j], color=color, lw=1.8, ls=ls, alpha=alpha,
                        label=label if i == 0 else None)
        ax.set_ylim(*true_ylim(true_trajs, j))
        ax.set_ylabel(rf"$x_{j + 1}$")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
    axes[-1, 0].set_xlabel("time step")
    fig.tight_layout()
    return fig


def plot_phase_portrait(X, labels, true_trajs, regional_trajs, vanilla_trajs):
    cmap = ListedColormap(CELL_FILL_COLORS)
    dropped = labels < 0

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[dropped, 0], X[dropped, 1], s=CELL_POINT_SIZE, c="darkgray", edgecolors="none")
    ax.scatter(X[~dropped, 0], X[~dropped, 1], s=CELL_POINT_SIZE, c=labels[~dropped], cmap=cmap, vmin=0, vmax=cmap.N - 1,
               alpha=CELL_POINT_ALPHA, edgecolors="none")
    # (trajectories, color, linestyle, linewidth, label, zorder): vanilla is drawn on top of the
    # true curve, regional on top of both, so all three stay visible where they overlap.
    series = [(true_trajs, COLOR_TRUE, "-", LW_TRUE, "true", 3)]
    if vanilla_trajs.size:
        series.append((vanilla_trajs, COLOR_VANILLA, "-.", LW_VANILLA, VANILLA_LABEL, 4))
    series.append((regional_trajs, COLOR_REGIONAL, "--", LW_REGIONAL, REGIONAL_LABEL, 5))
    for trajs, color, ls, lw, label, z in series:
        for i in range(trajs.shape[0]):
            ax.plot(trajs[i, :, 0], trajs[i, :, 1], color=color, ls=ls, lw=lw, zorder=z, label=label if i == 0 else None)
    ax.plot(true_trajs[:, 0, 0], true_trajs[:, 0, 1], "o", color=COLOR_TRUE, ms=7, zorder=6)   # initial conditions

    lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    pad = 0.1 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    args = parse_args(__doc__)
    d = load(args, "rollouts")
    fig = plot_time_series(d["true_trajs"], d["regional_trajs"], d["vanilla_trajs"])
    finish(fig, args, "rollout_trajectories")
    fig = plot_phase_portrait(d["X"], d["labels"], d["true_trajs"], d["regional_trajs"], d["vanilla_trajs"])
    finish(fig, args, "rollout_phase_portrait")
