"""Plot helpers for the partition pipeline."""

from pathlib import Path

import numpy as np


def finalize_figure(fig, *, save_path: Path | None = None, show: bool = True) -> None:
    """Save ``fig`` to ``save_path`` (if given), then show it or close it."""
    import matplotlib.pyplot as plt

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_invariant_embedding_2d(
    F_X, labels=None, title="Invariant embedding", *, save_path=None, show=True
):
    """If r >= 2, scatter F[:,0] vs F[:,1]; if r == 1, plot histogram."""
    import matplotlib.pyplot as plt

    F_X = np.asarray(F_X)
    if F_X.ndim != 2:
        raise ValueError("F_X must be 2-D with shape (N, r).")

    fig, ax = plt.subplots(figsize=(7, 5))
    if F_X.shape[1] >= 2:
        if labels is None:
            ax.scatter(F_X[:, 0], F_X[:, 1], s=4, alpha=0.35, edgecolors="none")
        else:
            labels = np.asarray(labels).ravel()
            scatter = ax.scatter(
                F_X[:, 0], F_X[:, 1], c=labels, s=4, alpha=0.5, edgecolors="none", cmap="tab10"
            )
            fig.colorbar(scatter, ax=ax, label="label")
        ax.set_xlabel(r"$f_1(x)$")
        ax.set_ylabel(r"$f_2(x)$")
    else:
        ax.hist(F_X[:, 0], bins=60, color="steelblue", edgecolor="black", alpha=0.85)
        ax.set_xlabel(r"$f_1(x)$")
        ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def plot_invariant_scalar_histograms(
    F_X,
    eigvals=None,
    *,
    bins=60,
    title="Invariant scalar observables",
    figsize=None,
    save_path=None,
    show=True,
):
    """One 1D histogram per scalar invariant coordinate f_j(x), j = 1, ..., r."""
    import matplotlib.pyplot as plt

    F_X = np.asarray(F_X, dtype=np.float64)
    if F_X.ndim != 2:
        raise ValueError("F_X must be 2-D with shape (N, r).")

    r = F_X.shape[1]
    if figsize is None:
        figsize = (max(4.0 * r, 5.0), 4.0)

    fig, axes = plt.subplots(1, r, figsize=figsize, squeeze=False)
    eigvals = None if eigvals is None else np.asarray(eigvals).ravel()

    for j in range(r):
        ax = axes[0, j]
        ax.hist(F_X[:, j], bins=bins, color="steelblue", edgecolor="black", alpha=0.85)
        if eigvals is not None and j < eigvals.size:
            ax.set_title(rf"$f_{{{j + 1}}}(x)$, $\lambda$ = {eigvals[j]:.4g}")
        else:
            ax.set_title(rf"$f_{{{j + 1}}}(x)$")
        ax.set_xlabel(rf"$f_{{{j + 1}}}(x)$")
        ax.set_ylabel("Count")
        ax.grid(alpha=0.3)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def plot_state_partition_2d(X, labels, title=None, *, save_path=None, show=True):
    """Scatter 2D state-space points colored by final partition labels."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X)
    labels = np.asarray(labels).ravel()
    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("plot_state_partition_2d requires X with shape (N, 2).")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of rows.")

    partition_cmap = ListedColormap(["red", "blue", "green", "orange"])
    partition_cmap.set_under("lightgray")

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        s=4,
        alpha=0.45,
        edgecolors="none",
        cmap=partition_cmap,
        vmin=0,
        vmax=len(partition_cmap.colors) - 1,
    )
    fig.colorbar(scatter, ax=ax, label="partition label")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title or "State-space partition")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def plot_inter_centroid_distances(
    distances,
    tau_val,
    *,
    title="Inter-centroid distances vs tau_val",
    figsize=(10, 3),
    save_path=None,
    show=True,
):
    """
    Plot pairwise centroid distances as thin vertical lines and tau_val as a thick line.
    """
    import matplotlib.pyplot as plt

    distances = np.asarray(distances, dtype=np.float64).ravel()
    tau_val = float(tau_val)

    fig, ax = plt.subplots(figsize=figsize)
    ymin, ymax = 0.0, 1.0
    for dist in distances:
        ax.vlines(dist, ymin, ymax, colors="steelblue", linewidth=0.9, alpha=0.75)
    ax.vlines(
        tau_val,
        ymin,
        ymax,
        colors="crimson",
        linewidth=3.5,
        label=rf"$\tau_{{val}}$ = {tau_val:.4g}",
    )

    if distances.size > 0:
        below = int(np.sum(distances <= tau_val))
        ax.set_title(
            f"{title} ({below}/{distances.size} pairs merge at "
            rf"$\tau_{{val}}$ = {tau_val:.4g})"
        )
        ax.set_xlim(0.0, max(float(np.max(distances)), tau_val) * 1.05)
    else:
        ax.set_title(f"{title} (no retained centroid pairs)")
        ax.set_xlim(0.0, max(tau_val, 1.0) * 1.05)

    ax.set_xlabel(r"$\|m_\alpha - m_\beta\|_2$")
    ax.set_yticks([])
    ax.legend(loc="upper right")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def plot_cell_weights(
    cell_weights,
    tau_occ,
    *,
    title="Fine-cell weights vs tau_occ",
    figsize=(10, 4),
    save_path=None,
    show=True,
):
    """
    Bar-plot the empirical weight |C_alpha|/N of every occupied fine cell (sorted
    descending) on a log axis, with tau_occ as a horizontal line. Bars at or above
    tau_occ are the retained cells; bars below are dropped by the occupancy filter.
    """
    import matplotlib.pyplot as plt

    tau_occ = float(tau_occ)
    items = sorted(cell_weights.items(), key=lambda kv: (-float(kv[1]), int(kv[0])))
    ids = [int(cell_id) for cell_id, _ in items]
    weights = np.array([float(w) for _, w in items], dtype=np.float64)
    retained = weights >= tau_occ

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(weights.size)
    colors = np.where(retained, "steelblue", "lightgray")
    ax.bar(x, weights, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(
        tau_occ,
        color="crimson",
        linewidth=2.5,
        label=rf"$\tau_{{occ}}$ = {tau_occ:.4g}",
    )

    if weights.size > 0:
        ax.set_yscale("log")
        positive = weights[weights > 0]
        lo = min(float(np.min(positive)) if positive.size else tau_occ, tau_occ)
        hi = max(float(np.max(weights)), tau_occ)
        ax.set_ylim(lo / 3.0, hi * 3.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(cell_id) for cell_id in ids], rotation=90, fontsize=7)
        ax.set_title(
            f"{title} ({int(np.sum(retained))}/{weights.size} cells retained at "
            rf"$\tau_{{occ}}$ = {tau_occ:.4g})"
        )
    else:
        ax.set_title(f"{title} (no occupied fine cells)")

    ax.set_xlabel("fine cell id (sorted by weight)")
    ax.set_ylabel(r"$|C_\alpha| / N$")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def plot_state_partition_1d(X, labels, title=None, *, save_path=None, show=True):
    """Scatter 1-D state-space points (with vertical jitter) colored by partition labels."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X)
    labels = np.asarray(labels).ravel()
    if X.ndim != 2 or X.shape[1] != 1:
        raise ValueError("plot_state_partition_1d requires X with shape (N, 1).")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of rows.")

    partition_cmap = ListedColormap(["red", "blue", "green", "orange"])
    partition_cmap.set_under("lightgray")

    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.05, 0.05, size=X.shape[0])

    fig, ax = plt.subplots(figsize=(8, 3))
    scatter = ax.scatter(
        X[:, 0],
        jitter,
        c=labels,
        s=4,
        alpha=0.45,
        edgecolors="none",
        cmap=partition_cmap,
        vmin=0,
        vmax=len(partition_cmap.colors) - 1,
    )
    fig.colorbar(scatter, ax=ax, label="partition label")
    ax.set_xlabel(r"$x_1$")
    ax.set_yticks([])
    ax.set_title(title or "State-space partition")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def plot_misclassified_points(
    X,
    gt_labels,
    pred_labels,
    *,
    anchors=None,
    title="Validation misclassifications",
    save_path=None,
    show=True,
):
    """
    Scatter validation states (1-D or 2-D) showing where classification fails.

    Correct points are drawn faintly, colored by ground-truth class. Points whose
    mapped prediction disagrees with the ground truth are drawn as bold crosses;
    points the partition dropped (``pred_labels < 0``) as gray crosses. Anchors
    (one per class) are marked with stars when given.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X)
    gt_labels = np.asarray(gt_labels).ravel()
    pred_labels = np.asarray(pred_labels).ravel()
    if X.ndim != 2 or X.shape[1] not in (1, 2):
        raise ValueError("plot_misclassified_points requires X with shape (N, 1) or (N, 2).")
    if not (X.shape[0] == gt_labels.shape[0] == pred_labels.shape[0]):
        raise ValueError("X, gt_labels and pred_labels must have the same number of rows.")

    dropped = pred_labels < 0
    wrong = (~dropped) & (pred_labels != gt_labels)
    correct = (~dropped) & (~wrong)

    if X.shape[1] == 2:
        xs, ys = X[:, 0], X[:, 1]
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        rng = np.random.default_rng(0)
        xs, ys = X[:, 0], rng.uniform(-0.05, 0.05, size=X.shape[0])
        fig, ax = plt.subplots(figsize=(8, 3))

    class_cmap = ListedColormap(["red", "blue", "green", "orange"])
    n_classes = int(gt_labels.max()) + 1 if gt_labels.size else 1
    ax.scatter(
        xs[correct],
        ys[correct],
        c=gt_labels[correct],
        s=4,
        alpha=0.2,
        edgecolors="none",
        cmap=class_cmap,
        vmin=0,
        vmax=len(class_cmap.colors) - 1,
        label=f"correct ({int(correct.sum())})",
    )
    if np.any(wrong):
        ax.scatter(
            xs[wrong],
            ys[wrong],
            c=gt_labels[wrong],
            s=28,
            marker="x",
            linewidths=1.2,
            cmap=class_cmap,
            vmin=0,
            vmax=len(class_cmap.colors) - 1,
            label=f"misclassified ({int(wrong.sum())}, colored by true class)",
        )
    if np.any(dropped):
        ax.scatter(
            xs[dropped],
            ys[dropped],
            c="dimgray",
            s=20,
            marker="x",
            linewidths=0.8,
            label=f"dropped ({int(dropped.sum())})",
        )
    if anchors is not None:
        anchors = np.asarray(anchors, dtype=float)
        ay = anchors[:, 1] if X.shape[1] == 2 else np.zeros(anchors.shape[0])
        ax.scatter(
            anchors[:, 0],
            ay,
            c=np.arange(anchors.shape[0]),
            s=160,
            marker="*",
            edgecolors="black",
            linewidths=0.8,
            cmap=class_cmap,
            vmin=0,
            vmax=len(class_cmap.colors) - 1,
            zorder=5,
            label="class anchors",
        )

    ax.set_xlabel(r"$x_1$")
    if X.shape[1] == 2:
        ax.set_ylabel(r"$x_2$")
    else:
        ax.set_yticks([])
    ax.set_title(f"{title} ({n_classes} classes)")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig



def plot_boundary_distance_histogram(
    rel_dist,
    misclassified_mask,
    *,
    baseline_mask=None,
    tau_robust=None,
    bins=50,
    title="Relative distance to nearest partition boundary",
    save_path=None,
    show=True,
):
    """
    Histogram of the relative boundary distance (see
    ``compute_relative_boundary_distances``) over misclassified samples, with the
    correctly classified samples overlaid as a baseline density when given.
    Non-finite distances (no boundary along any coordinate) are excluded. A
    positive ``tau_robust`` is drawn as a vertical line.
    """
    import matplotlib.pyplot as plt

    rel_dist = np.asarray(rel_dist, dtype=np.float64).ravel()
    misclassified_mask = np.asarray(misclassified_mask, dtype=bool).ravel()
    finite = np.isfinite(rel_dist)
    wrong = rel_dist[misclassified_mask & finite]
    base = None
    if baseline_mask is not None:
        baseline_mask = np.asarray(baseline_mask, dtype=bool).ravel()
        base = rel_dist[baseline_mask & finite]

    pooled = wrong if base is None or base.size == 0 else np.concatenate([wrong, base])
    upper = float(np.max(pooled)) if pooled.size else 0.5
    bin_edges = np.linspace(0.0, max(upper, 1e-12), bins + 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    if base is not None and base.size:
        ax.hist(
            base,
            bins=bin_edges,
            density=True,
            color="steelblue",
            alpha=0.35,
            label=f"correct (n={base.size}, median={np.median(base):.3g})",
        )
    if wrong.size:
        ax.hist(
            wrong,
            bins=bin_edges,
            density=True,
            color="crimson",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.5,
            label=f"misclassified (n={wrong.size}, median={np.median(wrong):.3g})",
        )
    if tau_robust is not None and float(tau_robust) > 0.0:
        ax.axvline(
            float(tau_robust),
            color="black",
            linestyle="--",
            linewidth=1.5,
            label=rf"$\tau_{{robust}}$ = {float(tau_robust):.4g}",
        )
    ax.set_xlabel(r"$\min_j\, \mathrm{dist}_j / \mathrm{length}_j$")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig


def plot_rollout_phase_portrait(
    X,
    labels,
    rollouts,
    vanilla_rollouts=None,
    *,
    title="Rollouts over the partition",
    regional_label="regional EDMD",
    vanilla_label="vanilla EDMD",
    save_path=None,
    show=True,
):
    """
    2-D state-space view: training snapshots colored by partition cell (dropped in
    gray) with true / regional (/ vanilla) rollout trajectories drawn on top.

    ``rollouts`` and ``vanilla_rollouts`` are lists of ``(true_traj, pred_traj, ...)``
    tuples as produced by the rollout helpers in ``build_multiple_linear_models``.
    Axis limits follow the data, so a diverging prediction leaves the frame.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    X = np.asarray(X)
    if X.shape[1] != 2:
        raise ValueError("plot_rollout_phase_portrait needs 2-D states.")
    labels = np.asarray(labels).ravel()
    dropped = labels < 0
    cmap = ListedColormap(["#f4a6a6", "#a6c8f4", "#a6e3b5", "#f7d59c"])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(X[dropped, 0], X[dropped, 1], s=2, c="lightgray", edgecolors="none")
    ax.scatter(X[~dropped, 0], X[~dropped, 1], s=2, c=labels[~dropped], cmap=cmap,
               vmin=0, vmax=cmap.N - 1, alpha=0.5, edgecolors="none")
    series = [(rollouts, 0, "black", "-", 1.6, "true"), (rollouts, 1, "crimson", "--", 1.3, regional_label)]
    if vanilla_rollouts:
        series.append((vanilla_rollouts, 1, "steelblue", ":", 1.3, vanilla_label))
    for rolls, which, color, ls, lw, label in series:
        for i, item in enumerate(rolls):
            traj = np.asarray(item[which])
            ax.plot(traj[:, 0], traj[:, 1], color=color, ls=ls, lw=lw, label=label if i == 0 else None)
    for item in rollouts:
        ax.plot(item[0][0, 0], item[0][0, 1], "o", color="black", ms=6)

    lo, hi = np.nanmin(X, axis=0), np.nanmax(X, axis=0)
    pad = 0.1 * (hi - lo)
    ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
    ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()
    finalize_figure(fig, save_path=save_path, show=show)
    return fig
