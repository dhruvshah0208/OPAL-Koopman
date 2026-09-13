"""Helpers for the earlier single-eigenfunction basin classifier (scripts/legacy).

Not used by the paper pipeline; kept so that ``build_classifier.py``,
``build_pruned_classifier.py`` and ``validate_classifier.py`` keep working.
"""

import numpy as np

def sorted_koopman_eigenmodes(K, target=1.0):
    """Eigen-decompose K and sort modes by distance of eigenvalue to `target`."""
    eigenvalues, eigenvectors = np.linalg.eig(K)
    idx = np.argsort(np.abs(eigenvalues - target))
    return eigenvalues[idx], eigenvectors[:, idx]


def apply_eigenvector(Phi_lifted, v):
    """Real-valued observable samples Psi(x) = Re(Phi_lifted @ v)."""
    return np.real(Phi_lifted @ v)


# ---------------------------------------------------------------------------
# Phase-space grid and basin separation
# ---------------------------------------------------------------------------

def make_phase_space_grid(xlim=(-2, 2), ylim=(-2, 2), n=300):
    """Uniform 2-D grid for contour plots of eigenfunctions."""
    x_range = np.linspace(xlim[0], xlim[1], n)
    y_range = np.linspace(ylim[0], ylim[1], n)
    X_grid, V_grid = np.meshgrid(x_range, y_range)
    grid_points = np.column_stack([X_grid.ravel(), V_grid.ravel()])
    return X_grid, V_grid, grid_points


def basin_mask_from_eigenfunction(phi_indicator, threshold=None, epsilon_fix=0.0):
    """
    Threshold Re(phi) to obtain a basin mask.

    If threshold is None, uses mean(Re(phi)) + epsilon_fix.
    """
    phi_real = np.real(phi_indicator)
    if threshold is None:
        threshold = np.mean(phi_real) + epsilon_fix
    return phi_real > threshold, threshold, phi_real


def filter_snapshots_by_mask(X, Y, mask):
    """Keep snapshot pairs that lie in the selected basin."""
    return X[mask], Y[mask]


def _draw_eigenfunction_contour_histogram(
    ax,
    ax_hist,
    X_grid,
    V_grid,
    phi_values,
    *,
    title,
    fig,
    x_ticks=(-2, 0, 2),
    y_ticks=(-2, 0, 2),
    compact=False,
):
    phi_grid = np.real(phi_values).reshape(X_grid.shape)
    phi_flat = np.real(phi_values).ravel()
    title_fs = 10 if compact else 14
    label_fs = 9 if compact else 12
    tick_fs = 8 if compact else 10

    contour = ax.contourf(X_grid, V_grid, phi_grid, levels=100, cmap="jet")
    ax.set_title(title, fontsize=title_fs)
    ax.set_xlabel("$x_1$", fontsize=label_fs)
    ax.set_ylabel("$x_2$", fontsize=label_fs)
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="both", which="major", labelsize=tick_fs)
    ax.tick_params(which="both", direction="in", top=True, right=True)

    cbar = fig.colorbar(
        contour, ax=ax, fraction=0.05, pad=0.03, aspect=12 if compact else 20
    )
    cbar.set_ticks(np.linspace(phi_grid.min(), phi_grid.max(), 5))
    cbar.ax.tick_params(labelsize=7 if compact else 9)

    hist_title_fs = 9 if compact else 11
    ax_hist.hist(phi_flat, bins=40, color="steelblue", edgecolor="black", alpha=0.85)
    ax_hist.set_title("Value Histogram", fontsize=hist_title_fs)
    ax_hist.set_xlabel("Re($\\phi$)", fontsize=label_fs - 1)
    ax_hist.set_ylabel("Count", fontsize=label_fs - 1)
    ax_hist.tick_params(axis="both", which="major", labelsize=tick_fs - 1)
    ax_hist.tick_params(which="both", direction="in", top=True, right=True)


def _draw_basin_separation(
    ax_scatter,
    ax_hist,
    X,
    mask,
    phi_real,
    threshold,
    *,
    title,
    compact=False,
):
    title_fs = 10 if compact else 14
    label_fs = 9 if compact else 12
    ax_scatter.set_title(title, fontsize=title_fs)
    ax_scatter.scatter(X[mask, 0], X[mask, 1], c="red", s=1, alpha=0.5)
    ax_scatter.scatter(X[~mask, 0], X[~mask, 1], c="blue", s=1, alpha=0.5)
    ax_scatter.set_xlabel("$x_1$", fontsize=label_fs)
    ax_scatter.set_ylabel("$x_2$", fontsize=label_fs)
    ax_scatter.grid(alpha=0.3)

    ax_hist.hist(phi_real.ravel(), bins=40, color="steelblue", edgecolor="black", alpha=0.85)
    ax_hist.axvline(
        threshold,
        color="red",
        linestyle="-",
        linewidth=2,
        label=f"Threshold = {threshold:.3f}",
    )
    ax_hist.set_title("Histogram of Re($\\phi$)", fontsize=9 if compact else 11)
    ax_hist.set_xlabel("Re($\\phi$)", fontsize=label_fs - 1)
    ax_hist.set_ylabel("Count", fontsize=label_fs - 1)
    ax_hist.legend(fontsize=6 if compact else 8)
    ax_hist.grid(alpha=0.3)


def plot_eigenfunction_contour_histogram(
    X_grid,
    V_grid,
    phi_values,
    title="Real Part of Eigenfunction",
    x_ticks=(-2, 0, 2),
    y_ticks=(-2, 0, 2),
    figsize=(14, 6),
):
    """Contour of eigenfunction on a grid plus histogram of grid values."""
    import matplotlib.pyplot as plt

    fig, (ax, ax_hist) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [3, 1]}
    )
    _draw_eigenfunction_contour_histogram(
        ax, ax_hist, X_grid, V_grid, phi_values, title=title, fig=fig,
        x_ticks=x_ticks, y_ticks=y_ticks,
    )
    plt.tight_layout()
    plt.show()
    return fig


def plot_basin_separation(
    X,
    mask,
    phi_real,
    threshold,
    title="Tier 1: Learned Basin Separation",
    figsize=(12, 5),
):
    """Scatter plot of basin labels with eigenfunction histogram and threshold."""
    import matplotlib.pyplot as plt

    fig, (ax_scatter, ax_hist) = plt.subplots(
        1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2, 1]}
    )
    _draw_basin_separation(ax_scatter, ax_hist, X, mask, phi_real, threshold, title=title)
    plt.tight_layout()
    plt.show()
    return fig


def plot_combined_classifier_dashboard(
    X_grid,
    V_grid,
    *,
    phi_edmd_grid,
    X,
    mask_edmd,
    phi_real_edmd,
    threshold_edmd,
    idx_edmd,
    eig_val_edmd,
    phi_inv_grid,
    mask_inv,
    phi_real_inv,
    threshold_inv,
    idx_approximate_invariant,
    eig_val_approximate_invariant,
    n_basin_edmd,
    n_basin_inv,
    x_ticks=(-2, 0, 2),
    y_ticks=(-2, 0, 2),
    figsize=(15, 6),
):
    """
    Single 2×4 figure: row 0 = EDMD (idx), row 1 = approximate invariant observable (idx).

    Columns: [contour | grid histogram | basin scatter | threshold histogram].
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        4,
        figsize=figsize,
        gridspec_kw={
            "width_ratios": [2.2, 0.9, 2.0, 0.9],
            "hspace": 0.42,
            "wspace": 0.32,
        },
    )

    mu_edmd = eig_val_edmd
    if np.iscomplexobj(mu_edmd):
        mu_str = f"{np.real(mu_edmd):.4f}{np.imag(mu_edmd):+.4f}j"
    else:
        mu_str = f"{float(np.real(mu_edmd)):.4f}"

    _draw_eigenfunction_contour_histogram(
        axes[0, 0],
        axes[0, 1],
        X_grid,
        V_grid,
        phi_edmd_grid,
        title=f"EDMD (idx={idx_edmd}), $\\mu$ = {mu_str}",
        fig=fig,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        compact=True,
    )
    _draw_basin_separation(
        axes[0, 2],
        axes[0, 3],
        X,
        mask_edmd,
        phi_real_edmd,
        threshold_edmd,
        title=f"EDMD basin (idx={idx_edmd}), {n_basin_edmd} pts",
        compact=True,
    )

    lam_str = f"{float(np.real(eig_val_approximate_invariant)):.4e}"
    _draw_eigenfunction_contour_histogram(
        axes[1, 0],
        axes[1, 1],
        X_grid,
        V_grid,
        phi_inv_grid,
        title=f"Inv. Observable (idx={idx_approximate_invariant}), $\\lambda$ = {lam_str}",
        fig=fig,
        x_ticks=x_ticks,
        y_ticks=y_ticks,
        compact=True,
    )
    _draw_basin_separation(
        axes[1, 2],
        axes[1, 3],
        X,
        mask_inv,
        phi_real_inv,
        threshold_inv,
        title=(
            f"Inv. basin (idx={idx_approximate_invariant}), {n_basin_inv} pts"
        ),
        compact=True,
    )

    fig.suptitle("Classifier summary", fontsize=13, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    return fig


def plot_basin_scatter_simple(X, mask, title="Learned Basin Separation", xlabel="x", ylabel="y"):
    """Single-panel basin scatter (no histogram)."""
    import matplotlib.pyplot as plt

    plt.title(title)
    plt.scatter(X[mask, 0], X[mask, 1], c="red", s=1, alpha=0.5)
    plt.scatter(X[~mask, 0], X[~mask, 1], c="blue", s=1, alpha=0.5)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid()
    plt.tight_layout()
    plt.show()


