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
