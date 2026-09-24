"""Relative rollout error vs time on validation trajectories (correctly classified ICs),
regional vs vanilla EDMD.

Thin lines: individual trajectories; thick: mean over trajectories (NaN-aware); log scale.
Input: data/paper_plots/<system>/rollout_errors.npz
(rel_regional, rel_vanilla: (n_traj, horizon+1); the abs_* arrays are ignored here)
"""

import numpy as np
import matplotlib.pyplot as plt

from _common import COLOR_REGIONAL, COLOR_VANILLA, REGIONAL_LABEL, VANILLA_LABEL, finish, load, parse_args


def plot_relative_error(rel_regional, rel_vanilla=None):
    curves = [(rel_regional, COLOR_REGIONAL, REGIONAL_LABEL)]
    if rel_vanilla is not None and rel_vanilla.size:
        curves.append((rel_vanilla, COLOR_VANILLA, VANILLA_LABEL))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for arr, color, label in curves:
        t = np.arange(arr.shape[1])
        for row in arr:
            ax.plot(t, row, color=color, lw=0.5, alpha=0.15)
        with np.errstate(all="ignore"):
            ax.plot(t, np.nanmean(arr, axis=0), color=color, lw=2.4, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("time step")
    ax.set_ylabel(r"$\|\hat{x}_t - x_t\| / \|x_t\|$")
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    args = parse_args(__doc__)
    d = load(args, "rollout_errors")
    fig = plot_relative_error(d["rel_regional"], d.get("rel_vanilla"))
    finish(fig, args, "rollout_errors")
