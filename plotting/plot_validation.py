"""Validation phase portrait: correctly classified, misclassified and dropped snapshots + anchors.

Input: data/paper_plots/<system>/validation.npz  (X, gt_labels, pred_labels, anchors, system_name)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from _common import CLASS_COLORS, finish, load, parse_args


def plot_validation(X, gt_labels, pred_labels, anchors):
    cmap = ListedColormap(CLASS_COLORS)
    vmax = len(CLASS_COLORS) - 1
    dropped = pred_labels < 0
    wrong = (~dropped) & (pred_labels != gt_labels)
    correct = (~dropped) & (~wrong)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[correct, 0], X[correct, 1], c=gt_labels[correct], s=4, alpha=0.2, edgecolors="none",
               cmap=cmap, vmin=0, vmax=vmax, label=f"correct ({int(correct.sum())})")
    if wrong.any():
        ax.scatter(X[wrong, 0], X[wrong, 1], c=gt_labels[wrong], s=28, marker="x", linewidths=1.2,
                   cmap=cmap, vmin=0, vmax=vmax, label=f"misclassified ({int(wrong.sum())}, colored by true class)")
    if dropped.any():
        ax.scatter(X[dropped, 0], X[dropped, 1], c="dimgray", s=20, marker="x", linewidths=0.8,
                   label=f"dropped ({int(dropped.sum())})")
    ax.scatter(anchors[:, 0], anchors[:, 1], c=np.arange(anchors.shape[0]), s=160, marker="*",
               edgecolors="black", linewidths=0.8, cmap=cmap, vmin=0, vmax=vmax, zorder=5, label="class anchors")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    args = parse_args(__doc__)
    d = load(args, "validation")
    fig = plot_validation(d["X"], d["gt_labels"], d["pred_labels"], d["anchors"])
    finish(fig, args, "validation_phase_portrait")
