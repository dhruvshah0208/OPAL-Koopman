"""Training snapshots colored by omega-limit partition cell (gray = dropped).

Input: data/paper_plots/<system>/partition.npz  (X, labels, system_name)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from _common import CLASS_COLORS, finish, load, parse_args


def plot_partition(X, labels):
    cmap = ListedColormap(CLASS_COLORS)
    cmap.set_under("lightgray")   # label -1 = dropped

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(X[:, 0], X[:, 1], c=labels, s=4, alpha=0.45, edgecolors="none",
                    cmap=cmap, vmin=0, vmax=len(CLASS_COLORS) - 1)
    fig.colorbar(sc, ax=ax, label="partition label")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    args = parse_args(__doc__)
    d = load(args, "partition")
    fig = plot_partition(d["X"], d["labels"])
    finish(fig, args, "partition")
