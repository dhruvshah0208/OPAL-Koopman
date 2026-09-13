'''
Helper functions for Koopman / EDMD experiments in this project.
'''

import time
from pathlib import Path

import numpy as np
from scipy.linalg import eigh as scipy_generalized_eigh
from sklearn.cluster import KMeans

from src.basis_functions import CombinedDictionary, PolynomialBasis, ThinPlateSpline

try:
    from src_deep_koopman.basis_functions import RadialBasis
except ImportError:  # pragma: no cover
    from src.basis_functions import RadialBasis  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Approximate invariant observables (generalized eigenvalue problem)
# ---------------------------------------------------------------------------

def compute_approximate_invariant_observables(Phi_X, Phi_Y, epsilon=0.01, reg=1e-6):
    """
    Find dictionary coefficients w that are approximately invariant under the snapshot pair.

    Minimizes ||(Phi_Y - Phi_X) w||^2 / ||Phi_X w||^2, i.e. solves M w = lambda N w with
    M = (Phi_Y - Phi_X)^T (Phi_Y - Phi_X) and N = Phi_X^T Phi_X.
    """
    Phi_X = np.asarray(Phi_X)
    Phi_Y = np.asarray(Phi_Y)

    if Phi_X.ndim != 2 or Phi_Y.ndim != 2:
        raise ValueError("Phi_X and Phi_Y must be 2-D arrays of shape (n_samples, n_features)")
    if Phi_X.shape != Phi_Y.shape:
        raise ValueError("Phi_X and Phi_Y must have the same shape")

    n_features = Phi_X.shape[1]
    diff = Phi_Y - Phi_X
    gram_M = diff.T @ diff
    #gram_N = Phi_X.T @ Phi_X
    gram_N = Phi_Y.T @ Phi_Y

    gram_M = 0.5 * (gram_M + gram_M.T)
    gram_N = 0.5 * (gram_N + gram_N.T)

    if reg > 0:
        scale = np.abs(np.trace(gram_N) / n_features)
        gram_N = gram_N + reg * scale * np.eye(n_features)

    lambda_vals, lambda_vecs = scipy_generalized_eigh(
        gram_M, gram_N, check_finite=False
    )

    mask = lambda_vals < epsilon
    return lambda_vals[mask], lambda_vecs[:, mask]


# ---------------------------------------------------------------------------
# RBF centers and dictionary
# ---------------------------------------------------------------------------

def select_rbf_centers(X, n_centers, use_kmeans=True, random_state=42, verbose=True):
    """Pick RBF centers via K-means or a random subsample of data points."""
    X = np.asarray(X)
    n_samples = X.shape[0]
    if n_samples < 1:
        raise ValueError("Need at least one data point to select RBF centers.")
    if n_centers > n_samples:
        if verbose:
            print(
                f"Requested {n_centers} centers but only {n_samples} data points; "
                f"using {n_samples} centers."
            )
        n_centers = n_samples

    if use_kmeans:
        t0 = time.time()
        kmeans = KMeans(n_clusters=n_centers, random_state=random_state, n_init=10)
        kmeans.fit(X)
        centers = kmeans.cluster_centers_
        if verbose:
            print(f"Center selection: K-means ({n_centers} centers)")
            print(f"Time to compute K-means solution = {time.time() - t0:.3f} seconds")
    else:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n_samples, size=n_centers, replace=False)
        centers = X[idx]
        if verbose:
            print(
                f"Center selection: random subsample "
                f"({n_centers} centers from {n_samples} points)"
            )

    if verbose:
        print(f"centers shape: {centers.shape}")
    return centers


def sample_edmd_centers(centers_pool, n_centers, random_state=42, verbose=True):
    """
    Randomly subsample RBF centers from an existing pool (e.g. centers_global).

    If ``n_centers`` is non-positive or exceeds the pool size, all pool centers
    are returned unchanged.
    """
    centers_pool = np.asarray(centers_pool, dtype=np.float64)
    if centers_pool.ndim != 2:
        raise ValueError("centers_pool must have shape (n_centers, n_state).")
    n_available = centers_pool.shape[0]
    if n_available < 1:
        raise ValueError("centers_pool must contain at least one center.")

    if n_centers <= 0 or n_centers >= n_available:
        if verbose:
            if n_centers > n_available:
                print(
                    f"Requested {n_centers} EDMD centers but pool has {n_available}; "
                    f"using all pool centers."
                )
            else:
                print(f"Using all {n_available} centers from pool.")
        return centers_pool.copy()

    rng = np.random.default_rng(random_state)
    idx = rng.choice(n_available, size=n_centers, replace=False)
    sampled = centers_pool[idx]
    if verbose:
        print(
            f"EDMD center selection: random subsample "
            f"({n_centers} of {n_available} pool centers)"
        )
        print(f"EDMD centers shape: {sampled.shape}")
    return sampled


def build_thin_plate_dictionary(state_dim, centers, d_poly=1):
    """Polynomial + thin-plate spline dictionary used throughout the notebook."""
    poly = PolynomialBasis(n=state_dim, d=d_poly)
    spline_rbfs = ThinPlateSpline(n=state_dim, centers=centers)
    if d_poly >= 0:
        return CombinedDictionary(poly, spline_rbfs)
    elif d_poly == -1:
        return spline_rbfs
    else:
        raise ValueError(f"Invalid polynomial degree: {d_poly}")


def build_rbf_polynomial_dictionary(
    n: int,
    centers: np.ndarray,
    gamma: float,
    d_poly: int = 1,
):
    """RBF (+ optional polynomial) dictionary used by build_classifier.py."""
    Phi_RBF = RadialBasis(n, centers=centers, gamma=gamma)
    if d_poly == -1:
        return Phi_RBF
    poly = PolynomialBasis(n=n, d=d_poly)
    return CombinedDictionary(Phi_RBF, poly)


def load_dictionary_from_classifiers(
    archive_path: Path | str,
    base_dir: Path | str | None = None,
):
    """
    Rebuild Phi from a classifiers.npz written by build_classifier._save_classifiers.

    Relative paths are resolved against ``base_dir`` if given, otherwise the cwd.
    """
    path = Path(archive_path)
    if not path.is_absolute():
        root = Path(base_dir) if base_dir is not None else Path.cwd()
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"Classifier archive not found: {path}")

    data = np.load(path)
    return build_rbf_polynomial_dictionary(
        int(data["n"]),
        data["centers_global"],
        float(data["gamma"]),
        int(data["d_poly"]),
    )


# ---------------------------------------------------------------------------
# EDMD / Koopman eigenanalysis
# ---------------------------------------------------------------------------

def fit_edmd_operator(Phi_X, Phi_Y):
    """Least-squares EDMD operator K such that Phi_Y approx Phi_X @ K."""
    return np.linalg.lstsq(Phi_X, Phi_Y, rcond=None)[0]


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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Graph-merged partition (Algorithm 1)
# ---------------------------------------------------------------------------

def select_invariant_modes(
    lambda_vals,
    lambda_vecs,
    *,
    epsilon,
    max_modes,
    skip_first_mode=False,
):
    """
    Select approximate invariant modes with generalized eigenvalue < epsilon.

    Returns (eigvals_selected, coeffs_selected) where coeffs_selected has shape (s, r).
    """
    lambda_vals = np.asarray(lambda_vals)
    lambda_vecs = np.asarray(lambda_vecs)
    if lambda_vecs.ndim != 2:
        raise ValueError("lambda_vecs must be 2-D with shape (s, n_modes).")

    mask = lambda_vals < float(epsilon)
    eigvals = lambda_vals[mask]
    coeffs = lambda_vecs[:, mask]

    if eigvals.size == 0:
        raise ValueError(f"No invariant modes with eigenvalue < {epsilon}.")

    order = np.argsort(eigvals)
    eigvals = eigvals[order]
    coeffs = coeffs[:, order]

    if skip_first_mode and eigvals.size > 1:
        eigvals = eigvals[1:]
        coeffs = coeffs[:, 1:]

    if max_modes is not None and int(max_modes) > 0:
        r = min(int(max_modes), eigvals.size)
        eigvals = eigvals[:r]
        coeffs = coeffs[:, :r]

    return eigvals, coeffs


def evaluate_invariant_embedding(Phi, coeffs):
    """
    Parameters
    ----------
    Phi : ndarray, shape (N, s)
        Lifted data matrix.
    coeffs : ndarray, shape (s, r)
        Selected invariant observable coefficient vectors.

    Returns
    -------
    F : ndarray, shape (N, r)
        Real-valued invariant-observable embedding.
    """
    Phi = np.asarray(Phi)
    coeffs = np.asarray(coeffs)
    if Phi.ndim != 2 or coeffs.ndim != 2:
        raise ValueError("Phi must be (N, s) and coeffs must be (s, r).")
    if Phi.shape[1] != coeffs.shape[0]:
        raise ValueError(
            f"Phi has {Phi.shape[1]} features but coeffs has {coeffs.shape[0]} rows."
        )
    return np.real(Phi @ coeffs)


def cluster_scalar_observables(F, n_clusters, random_state=42, method="kmeans"):
    """
    Cluster each scalar coordinate F[:, j] independently.

    Parameters
    ----------
    F : ndarray, shape (N, r)
    n_clusters : int or list[int]
        If int, use the same number of clusters for each coordinate.
        If list, must have length r.

    Returns
    -------
    scalar_labels : ndarray, shape (N, r)
    cluster_models : list
        Fitted clustering models, one per coordinate.
    """
    F = np.asarray(F)
    if F.ndim != 2:
        raise ValueError("F must be 2-D with shape (N, r).")
    n_samples, r = F.shape
    if n_samples < 1:
        raise ValueError("F must contain at least one sample.")

    if np.isscalar(n_clusters):
        cluster_counts = [int(n_clusters)] * r
    else:
        cluster_counts = [int(k) for k in n_clusters]
        if len(cluster_counts) != r:
            raise ValueError(
                f"n_clusters list length {len(cluster_counts)} must match r={r}."
            )

    if method != "kmeans":
        raise ValueError(f"Unsupported clustering method: {method!r}.")

    scalar_labels = np.zeros((n_samples, r), dtype=np.int64)
    cluster_models = []
    for j in range(r):
        k_j = min(cluster_counts[j], n_samples)
        if k_j < 1:
            raise ValueError(f"Coordinate {j} requires at least one cluster.")
        model = KMeans(n_clusters=k_j, random_state=random_state, n_init=10)
        col = F[:, j : j + 1]
        scalar_labels[:, j] = model.fit_predict(col)
        cluster_models.append(model)
    return scalar_labels, cluster_models


def assign_scalar_clusters(F, cluster_models):
    """
    Assign new embedding values to previously fitted scalar cluster models.

    Returns scalar labels of shape (N, r).
    """
    F = np.asarray(F)
    if F.ndim != 2:
        raise ValueError("F must be 2-D with shape (N, r).")
    if len(cluster_models) != F.shape[1]:
        raise ValueError("Number of cluster models must match embedding dimension.")

    scalar_labels = np.zeros((F.shape[0], F.shape[1]), dtype=np.int64)
    for j, model in enumerate(cluster_models):
        scalar_labels[:, j] = model.predict(F[:, j : j + 1])
    return scalar_labels


def assign_scalar_clusters_from_centers(F, cluster_centers):
    """
    Assign scalar cluster labels using saved KMeans centers of shape (r, k_max, 1).

    Rows/columns containing NaN are ignored when choosing the nearest center.
    """
    F = np.asarray(F, dtype=np.float64)
    cluster_centers = np.asarray(cluster_centers, dtype=np.float64)
    if F.ndim != 2:
        raise ValueError("F must be 2-D with shape (N, r).")
    if cluster_centers.ndim != 3 or cluster_centers.shape[2] != 1:
        raise ValueError("cluster_centers must have shape (r, k_max, 1).")
    if F.shape[1] != cluster_centers.shape[0]:
        raise ValueError("F embedding dimension must match cluster_centers.shape[0].")

    scalar_labels = np.zeros((F.shape[0], cluster_centers.shape[0]), dtype=np.int64)
    for j in range(cluster_centers.shape[0]):
        valid = np.isfinite(cluster_centers[j, :, 0])
        centers = cluster_centers[j, valid, 0]
        if centers.size == 0:
            raise ValueError(f"Coordinate {j} has no valid cluster centers.")
        values = F[:, j : j + 1]
        dists = np.abs(values - centers[None, :])
        scalar_labels[:, j] = np.argmin(dists, axis=1)
    return scalar_labels


def build_cell_to_component(components):
    """Build fine-cell-id -> merged-component-label mapping from component lists."""
    cell_to_component = {}
    for component_label, component_nodes in enumerate(components):
        for cell_id in component_nodes:
            cell_to_component[int(cell_id)] = int(component_label)
    return cell_to_component


def label_to_id_from_archive(multi_index_keys, multi_index_cell_ids):
    """Reconstruct label_to_id dict saved in partition_output.npz."""
    label_to_id = {}
    for key, cell_id in zip(multi_index_keys, multi_index_cell_ids, strict=True):
        if isinstance(key, tuple):
            normalized = tuple(int(v) for v in key)
        else:
            normalized = tuple(int(v) for v in np.asarray(key).ravel())
        label_to_id[normalized] = int(cell_id)
    return label_to_id


def encode_multi_index_labels_fixed(scalar_labels, label_to_id):
    """
    Map scalar label rows to known fine cell ids only.

    Unseen multi-index tuples receive cell id -1.
    """
    scalar_labels = np.asarray(scalar_labels, dtype=np.int64)
    cell_ids = np.full(scalar_labels.shape[0], -1, dtype=np.int64)
    for i, row in enumerate(scalar_labels):
        key = tuple(int(v) for v in row)
        if key in label_to_id:
            cell_ids[i] = label_to_id[key]
    return cell_ids


def predict_graph_merged_labels(
    F,
    *,
    label_to_id,
    cell_to_component,
    retained_cell_ids,
    cluster_centers,
):
    """
    Apply a fitted graph-merged partition to new embedding values F.

    Returns final partition labels; unknown or dropped points receive -1.
    """
    scalar_labels = assign_scalar_clusters_from_centers(F, cluster_centers)
    cell_ids = encode_multi_index_labels_fixed(scalar_labels, label_to_id)
    return _assign_final_labels(cell_ids, cell_to_component, retained_cell_ids)


def encode_multi_index_labels(scalar_labels, existing_label_to_id=None):
    """
    Convert rows of scalar labels into occupied multi-index cell ids.

    If existing_label_to_id is provided, use it and assign unseen labels new ids.

    Returns
    -------
    tuple_labels : list[tuple]
    label_to_id : dict[tuple, int]
    cell_ids : ndarray, shape (N,)
    """
    scalar_labels = np.asarray(scalar_labels, dtype=np.int64)
    if scalar_labels.ndim != 2:
        raise ValueError("scalar_labels must be 2-D with shape (N, r).")

    label_to_id = {} if existing_label_to_id is None else dict(existing_label_to_id)
    next_id = (max(label_to_id.values()) + 1) if label_to_id else 0

    tuple_labels = []
    cell_ids = np.empty(scalar_labels.shape[0], dtype=np.int64)
    for i, row in enumerate(scalar_labels):
        key = tuple(int(v) for v in row)
        if key not in label_to_id:
            label_to_id[key] = next_id
            next_id += 1
        tuple_labels.append(key)
        cell_ids[i] = label_to_id[key]
    return tuple_labels, label_to_id, cell_ids


def compute_fine_cells(cell_ids):
    """Return dict mapping cell_id -> array of point indices."""
    cell_ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    cells = {}
    for idx, cell_id in enumerate(cell_ids):
        cell_id = int(cell_id)
        if cell_id not in cells:
            cells[cell_id] = []
        cells[cell_id].append(idx)
    return {cell_id: np.asarray(indices, dtype=np.int64) for cell_id, indices in cells.items()}


def compute_cell_weights_and_centroids(cell_ids, F):
    """
    Compute empirical weight |C_alpha|/N and centroid m_alpha for each fine cell.

    Returns
    -------
    unique_cell_ids : ndarray
    cell_weights : dict[int, float]
    cell_centroids : dict[int, ndarray]
    """
    cell_ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    F = np.asarray(F)
    if F.shape[0] != cell_ids.shape[0]:
        raise ValueError("cell_ids and F must have the same number of rows.")

    n_total = cell_ids.shape[0]
    fine_cells = compute_fine_cells(cell_ids)
    unique_cell_ids = np.array(sorted(fine_cells.keys()), dtype=np.int64)
    cell_weights = {}
    cell_centroids = {}
    for cell_id, indices in fine_cells.items():
        cell_weights[int(cell_id)] = indices.size / n_total
        cell_centroids[int(cell_id)] = np.mean(F[indices], axis=0)
    return unique_cell_ids, cell_weights, cell_centroids


def filter_low_occupancy_cells(cell_weights, tau_occ):
    """Return retained cell ids satisfying weight >= tau_occ."""
    tau_occ = float(tau_occ)
    return {int(cell_id) for cell_id, weight in cell_weights.items() if weight >= tau_occ}


def compute_transition_counts(cell_ids_X, cell_ids_Y, retained_cell_ids):
    """
    Compute transition counts n_{alpha beta} and probabilities p_{alpha beta}.

    Ignore transitions where either endpoint is not in retained_cell_ids.

    Returns
    -------
    transition_counts : dict[(int, int), int]
    outgoing_counts : dict[int, int]
    transition_probs : dict[(int, int), float]
    """
    cell_ids_X = np.asarray(cell_ids_X, dtype=np.int64).ravel()
    cell_ids_Y = np.asarray(cell_ids_Y, dtype=np.int64).ravel()
    if cell_ids_X.shape != cell_ids_Y.shape:
        raise ValueError("cell_ids_X and cell_ids_Y must have the same shape.")

    retained = {int(cell_id) for cell_id in retained_cell_ids}
    transition_counts = {}
    outgoing_counts = {cell_id: 0 for cell_id in retained}

    for alpha, beta in zip(cell_ids_X, cell_ids_Y, strict=True):
        alpha = int(alpha)
        beta = int(beta)
        if alpha not in retained or beta not in retained:
            continue
        outgoing_counts[alpha] = outgoing_counts.get(alpha, 0) + 1
        key = (alpha, beta)
        transition_counts[key] = transition_counts.get(key, 0) + 1

    transition_probs = {}
    for (alpha, beta), count in transition_counts.items():
        denom = outgoing_counts.get(alpha, 0)
        if denom > 0:
            transition_probs[(alpha, beta)] = count / denom
    return transition_counts, outgoing_counts, transition_probs


def union_find_connected_components(nodes, edges):
    """
    Compute connected components without adding a networkx dependency.

    Returns components as a list of lists of node ids.
    """
    parent = {int(node): int(node) for node in nodes}

    def find(node):
        node = int(node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in edges:
        if int(a) != int(b):
            union(a, b)

    groups = {}
    for node in parent:
        root = find(node)
        groups.setdefault(root, []).append(int(node))
    return [sorted(group) for group in groups.values()]


def compute_inter_node_transition_scores(transition_probs, retained_cell_ids=None):
    """
    Pairwise symmetric transition scores max(p_ab, p_ba) between fine cells.

    Returns
    -------
    scores : ndarray, shape (n_pairs,)
    pair_cell_ids : ndarray, shape (n_pairs, 2)
        Each row is (alpha, beta) with alpha < beta.
    """
    prob_lookup = {tuple(map(int, key)): float(val) for key, val in transition_probs.items()}
    if retained_cell_ids is not None:
        nodes = sorted(int(cell_id) for cell_id in retained_cell_ids)
    else:
        nodes = sorted({node for edge in prob_lookup for node in edge})

    pairs = []
    scores = []
    for i, alpha in enumerate(nodes):
        for beta in nodes[i + 1 :]:
            p_ab = prob_lookup.get((alpha, beta), 0.0)
            p_ba = prob_lookup.get((beta, alpha), 0.0)
            pairs.append((alpha, beta))
            scores.append(float(max(p_ab, p_ba)))
    if not scores:
        return np.array([], dtype=np.float64), np.zeros((0, 2), dtype=np.int64)
    return np.asarray(scores, dtype=np.float64), np.asarray(pairs, dtype=np.int64)


def build_merge_edges_from_transitions(
    transition_probs, tau_tr, mode="sym_max", retained_cell_ids=None
):
    """
    Add undirected edge (alpha, beta) when max(p_ab, p_ba) >= tau_tr.

    Ignore self-edges.
    """
    tau_tr = float(tau_tr)
    if mode != "sym_max":
        raise ValueError(f"Unsupported transition merge mode: {mode!r}.")

    scores, pair_cell_ids = compute_inter_node_transition_scores(
        transition_probs, retained_cell_ids=retained_cell_ids
    )
    edges = [
        (int(alpha), int(beta))
        for (alpha, beta), score in zip(pair_cell_ids, scores, strict=True)
        if score >= tau_tr
    ]
    return sorted(edges)


def compute_inter_centroid_distances(cell_centroids):
    """
    Pairwise L2 distances between fine-cell centroids in embedding space.

    Returns
    -------
    distances : ndarray, shape (n_pairs,)
    pair_cell_ids : ndarray, shape (n_pairs, 2)
        Each row is (alpha, beta) with alpha < beta.
    """
    nodes = sorted(int(cell_id) for cell_id in cell_centroids.keys())
    pairs = []
    dists = []
    for i, alpha in enumerate(nodes):
        m_alpha = np.asarray(cell_centroids[alpha], dtype=float)
        for beta in nodes[i + 1 :]:
            m_beta = np.asarray(cell_centroids[beta], dtype=float)
            pairs.append((alpha, beta))
            dists.append(float(np.linalg.norm(m_alpha - m_beta)))
    if not dists:
        return np.array([], dtype=np.float64), np.zeros((0, 2), dtype=np.int64)
    return np.asarray(dists, dtype=np.float64), np.asarray(pairs, dtype=np.int64)


def build_merge_edges_from_centroids(cell_centroids, tau_val):
    """
    Add undirected edge (alpha, beta) when ||m_alpha - m_beta||_2 <= tau_val.
    """
    tau_val = float(tau_val)
    distances, pair_cell_ids = compute_inter_centroid_distances(cell_centroids)
    edges = [
        (int(alpha), int(beta))
        for (alpha, beta), dist in zip(pair_cell_ids, distances, strict=True)
        if dist <= tau_val
    ]
    return sorted(edges)


def compute_merged_component_centroids(components, fine_cells, F):
    """
    Empirical embedding centroid for each merged component.

    Parameters
    ----------
    components : list[list[int]]
        Each inner list is a group of fine cell ids merged together.
    fine_cells : dict[int, ndarray]
        Fine cell id -> point indices in F.
    F : ndarray, shape (N, r)
        Invariant embedding values.

    Returns
    -------
    merged_centroids : dict[int, ndarray]
        Component label -> centroid vector.
    """
    F = np.asarray(F)
    merged_centroids = {}
    for comp_id, nodes in enumerate(components):
        node_ids = [int(node) for node in nodes]
        indices = np.concatenate(
            [fine_cells[node_id] for node_id in node_ids if node_id in fine_cells]
        )
        if indices.size == 0:
            raise ValueError(f"Component {comp_id} has no points in fine_cells.")
        merged_centroids[int(comp_id)] = np.mean(F[indices], axis=0)
    return merged_centroids


def merge_partition_graph_sequential(
    retained_cell_ids,
    transition_probs,
    fine_cells,
    F,
    tau_tr,
    tau_val,
):
    """
    Two-stage graph merge over retained fine cells.

    1. Merge fine cells using transition probabilities (tau_tr).
    2. Compute centroids of the transition-merged components.
    3. Merge those components using centroid distances (tau_val).
    """
    retained_cell_ids = sorted(int(cell_id) for cell_id in retained_cell_ids)

    transition_pair_scores, transition_pair_cell_ids = compute_inter_node_transition_scores(
        transition_probs, retained_cell_ids=retained_cell_ids
    )
    merge_edges_transition = build_merge_edges_from_transitions(
        transition_probs, tau_tr, mode="sym_max", retained_cell_ids=retained_cell_ids
    )

    transition_components = union_find_connected_components(
        retained_cell_ids, merge_edges_transition
    )
    transition_component_ids = list(range(len(transition_components)))
    merged_centroids = compute_merged_component_centroids(
        transition_components, fine_cells, F
    )

    centroid_pair_distances, centroid_pair_cell_ids = compute_inter_centroid_distances(
        merged_centroids
    )
    merge_edges_value = build_merge_edges_from_centroids(merged_centroids, tau_val)

    value_components = union_find_connected_components(
        transition_component_ids, merge_edges_value
    )

    components = []
    cell_to_component = {}
    for final_label, coarse_group in enumerate(value_components):
        fine_nodes = sorted(
            {int(cell_id) for coarse_id in coarse_group for cell_id in transition_components[coarse_id]}
        )
        components.append(fine_nodes)
        for cell_id in fine_nodes:
            cell_to_component[int(cell_id)] = final_label

    return {
        "transition_components": transition_components,
        "merged_centroids": merged_centroids,
        "transition_pair_scores": transition_pair_scores,
        "transition_pair_cell_ids": transition_pair_cell_ids,
        "merge_edges_transition": merge_edges_transition,
        "merge_edges_value": merge_edges_value,
        "centroid_pair_distances": centroid_pair_distances,
        "centroid_pair_cell_ids": centroid_pair_cell_ids,
        "components": components,
        "cell_to_component": cell_to_component,
    }


def _assign_final_labels(cell_ids, cell_to_component, retained_cell_ids):
    """Map fine cell ids to merged component labels; dropped cells get -1."""
    cell_ids = np.asarray(cell_ids, dtype=np.int64).ravel()
    retained = {int(cell_id) for cell_id in retained_cell_ids}
    final_labels = np.full(cell_ids.shape[0], -1, dtype=np.int64)
    for idx, cell_id in enumerate(cell_ids):
        cell_id = int(cell_id)
        if cell_id in retained and cell_id in cell_to_component:
            final_labels[idx] = cell_to_component[cell_id]
    return final_labels


def fit_graph_merged_partition(
    F_X,
    F_Y,
    n_clusters,
    tau_occ,
    tau_tr,
    tau_val,
    random_state=42,
):
    """
    Fit the graph-merged approximate omega-limit partition.

    Returns a dictionary containing scalar labels, fine cell ids, retained cells,
    centroids, transition statistics, sequential merge graph edges, connected
    components, and final labels for X and Y.

    Merge is two-stage: transition edges on fine cells, then centroid edges on
    the transition-merged graph.
    """
    F_X = np.asarray(F_X)
    F_Y = np.asarray(F_Y)
    if F_X.ndim != 2 or F_Y.ndim != 2:
        raise ValueError("F_X and F_Y must be 2-D arrays.")
    if F_X.shape[1] != F_Y.shape[1]:
        raise ValueError("F_X and F_Y must have the same embedding dimension.")
    if F_X.shape[0] != F_Y.shape[0]:
        raise ValueError("F_X and F_Y must have the same number of samples.")

    scalar_labels_X, cluster_models = cluster_scalar_observables(
        F_X, n_clusters, random_state=random_state
    )
    scalar_labels_Y = assign_scalar_clusters(F_Y, cluster_models)

    _, label_to_id, cell_ids_X = encode_multi_index_labels(scalar_labels_X)
    _, _, cell_ids_Y = encode_multi_index_labels(scalar_labels_Y, label_to_id)

    fine_cells = compute_fine_cells(cell_ids_X)
    _, cell_weights, cell_centroids = compute_cell_weights_and_centroids(cell_ids_X, F_X)
    retained_cell_ids = filter_low_occupancy_cells(cell_weights, tau_occ)

    transition_counts, outgoing_counts, transition_probs = compute_transition_counts(
        cell_ids_X, cell_ids_Y, retained_cell_ids
    )

    merge_result = merge_partition_graph_sequential(
        retained_cell_ids,
        transition_probs,
        fine_cells,
        F_X,
        tau_tr,
        tau_val,
    )
    transition_components = merge_result["transition_components"]
    merged_centroids = merge_result["merged_centroids"]
    transition_pair_scores = merge_result["transition_pair_scores"]
    transition_pair_cell_ids = merge_result["transition_pair_cell_ids"]
    merge_edges_transition = merge_result["merge_edges_transition"]
    merge_edges_value = merge_result["merge_edges_value"]
    centroid_pair_distances = merge_result["centroid_pair_distances"]
    centroid_pair_cell_ids = merge_result["centroid_pair_cell_ids"]
    components = merge_result["components"]
    cell_to_component = merge_result["cell_to_component"]

    final_labels_X = _assign_final_labels(cell_ids_X, cell_to_component, retained_cell_ids)
    final_labels_Y = _assign_final_labels(cell_ids_Y, cell_to_component, retained_cell_ids)

    return {
        "scalar_labels_X": scalar_labels_X,
        "scalar_labels_Y": scalar_labels_Y,
        "cell_ids_X": cell_ids_X,
        "cell_ids_Y": cell_ids_Y,
        "retained_cell_ids": sorted(int(cell_id) for cell_id in retained_cell_ids),
        "cell_weights": cell_weights,
        "cell_centroids": cell_centroids,
        "transition_counts": transition_counts,
        "outgoing_counts": outgoing_counts,
        "transition_probs": transition_probs,
        "transition_components": transition_components,
        "merged_centroids": merged_centroids,
        "merge_edges_transition": merge_edges_transition,
        "merge_edges_value": merge_edges_value,
        "centroid_pair_distances": centroid_pair_distances,
        "centroid_pair_cell_ids": centroid_pair_cell_ids,
        "transition_pair_scores": transition_pair_scores,
        "transition_pair_cell_ids": transition_pair_cell_ids,
        "components": components,
        "cell_to_component": cell_to_component,
        "final_labels_X": final_labels_X,
        "final_labels_Y": final_labels_Y,
        "cluster_models": cluster_models,
        "label_to_id": label_to_id,
        "fine_cells": fine_cells,
        "n_occupied_fine_cells": len(fine_cells),
    }


def plot_invariant_embedding_2d(F_X, labels=None, title="Invariant embedding"):
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
    plt.show()
    return fig


def plot_invariant_scalar_histograms(
    F_X,
    eigvals=None,
    *,
    bins=60,
    title="Invariant scalar observables",
    figsize=None,
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
    plt.show()
    return fig


def plot_state_partition_2d(X, labels, title=None):
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
    plt.show()
    return fig


def plot_inter_centroid_distances(
    distances,
    tau_val,
    *,
    title="Inter-centroid distances vs tau_val",
    figsize=(10, 3),
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
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Switched / multiple local linear (Koopman) models
# ---------------------------------------------------------------------------

def load_partition_inference(path):
    """
    Load the parts of a partition_output.npz archive needed for inference.

    Returns the invariant-embedding coefficients, dictionary parameters, and
    the merge-graph metadata required to classify new states into the final
    (component-level) partition.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    required = (
        "coeffs_inv",
        "centers_global",
        "gamma",
        "d_poly",
        "n",
        "cluster_centers",
        "retained_cell_ids",
    )
    for key in required:
        if key not in data:
            raise KeyError(f"{path} does not contain '{key}'.")

    if "multi_index_keys" in data and "multi_index_cell_ids" in data:
        label_to_id = label_to_id_from_archive(
            data["multi_index_keys"], data["multi_index_cell_ids"]
        )
    else:
        raise KeyError(
            f"{path} has no multi_index_keys/multi_index_cell_ids. "
            "Re-run build_partition.py to regenerate partition_output.npz."
        )

    if "cell_to_component_ids" in data and "cell_to_component_labels" in data:
        cell_to_component = {
            int(cell_id): int(label)
            for cell_id, label in zip(
                data["cell_to_component_ids"],
                data["cell_to_component_labels"],
                strict=True,
            )
        }
    elif "components" in data:
        cell_to_component = build_cell_to_component(list(data["components"]))
    else:
        raise KeyError(f"{path} does not contain cell-to-component mapping metadata.")

    return {
        "coeffs_inv": np.asarray(data["coeffs_inv"]),
        "centers_global": np.asarray(data["centers_global"]),
        "gamma": float(data["gamma"]),
        "d_poly": int(data["d_poly"]),
        "n": int(data["n"]),
        "cluster_centers": np.asarray(data["cluster_centers"]),
        "retained_cell_ids": np.asarray(data["retained_cell_ids"], dtype=np.int64),
        "label_to_id": label_to_id,
        "cell_to_component": cell_to_component,
    }


def classify_partition_labels(X, partition):
    """
    Assign final (component-level) partition labels to states X.

    Builds the partition dictionary, evaluates the invariant embedding, and
    applies the saved graph-merged partition. Unknown or dropped states get -1.
    Accepts a single state of shape (n,) or a batch of shape (N, n).
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    Phi = build_rbf_polynomial_dictionary(
        partition["n"],
        partition["centers_global"],
        partition["gamma"],
        partition["d_poly"],
    )
    F = evaluate_invariant_embedding(Phi.evaluate(X), partition["coeffs_inv"])
    return predict_graph_merged_labels(
        F,
        label_to_id=partition["label_to_id"],
        cell_to_component=partition["cell_to_component"],
        retained_cell_ids=partition["retained_cell_ids"],
        cluster_centers=partition["cluster_centers"],
    )


def fit_reconstruction_matrix(Phi_X, X):
    """Least-squares reconstruction R such that X approx Phi_X @ R, shape (s, n_state)."""
    return np.linalg.lstsq(np.asarray(Phi_X), np.asarray(X), rcond=None)[0]


def fit_switched_edmd_models(X, Y, labels, Phi, *, min_samples=100):
    """
    Fit a local EDMD model on each partition component.

    For every component label l >= 0 with at least ``min_samples`` snapshot
    pairs, fit a lifted-space propagator and a state-reconstruction map:

        Phi(Y) approx Phi(X) @ K_l      (one-step propagation)
        X      approx Phi(X) @ R_l      (lifted-state -> state reconstruction)

    Parameters
    ----------
    X, Y : ndarray, shape (N, n_state)
        Snapshot pairs.
    labels : ndarray, shape (N,)
        Component labels for each X row (from ``classify_partition_labels``).
    Phi : dictionary object with an ``evaluate`` method and ``s`` attribute.
    min_samples : int
        Components with fewer snapshot pairs are skipped.

    Returns
    -------
    dict with keys: ``K_by_label``, ``R_by_label`` (label -> ndarray),
    ``sample_counts`` (label -> int), ``skipped`` (label -> int),
    ``n_dict``, ``n_state``.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64).ravel()
    if X.shape != Y.shape:
        raise ValueError("X and Y must have the same shape.")
    if X.shape[0] != labels.shape[0]:
        raise ValueError("X and labels must have the same number of rows.")

    K_by_label = {}
    R_by_label = {}
    sample_counts = {}
    skipped = {}
    for label in sorted(int(v) for v in np.unique(labels) if v >= 0):
        mask = labels == label
        count = int(np.sum(mask))
        if count < min_samples:
            skipped[label] = count
            continue
        Phi_X = Phi.evaluate(X[mask])
        Phi_Y = Phi.evaluate(Y[mask])
        K_by_label[label] = fit_edmd_operator(Phi_X, Phi_Y)
        R_by_label[label] = fit_reconstruction_matrix(Phi_X, X[mask])
        sample_counts[label] = count

    return {
        "K_by_label": K_by_label,
        "R_by_label": R_by_label,
        "sample_counts": sample_counts,
        "skipped": skipped,
        "n_dict": int(Phi.s),
        "n_state": int(X.shape[1]),
    }


def switched_koopman_rollout(x0, horizon, models, partition, Phi, *, reclassify=False):
    """
    Roll a single initial state forward ``horizon`` steps with the switched model.

    At each step the current state is classified into a partition component and
    the corresponding local model (K_l, R_l) advances it one step in lifted space
    before reconstructing the next state. If a state cannot be classified
    (label -1) or its component has no fitted model, the rollout stops and the
    remaining states are left as NaN.

    Parameters
    ----------
    x0 : ndarray, shape (n_state,)
    horizon : int
        Number of one-step predictions to make.
    models : dict
        Output of ``fit_switched_edmd_models`` (or loaded equivalent).
    partition : dict
        Output of ``load_partition_inference``.
    Phi : EDMD dictionary object used to fit ``models``.
    reclassify : bool
        If True, re-classify the state at every step. If False, classify only
        the initial state and reuse that local model for the whole rollout.

    Returns
    -------
    traj : ndarray, shape (horizon + 1, n_state)
    step_labels : ndarray, shape (horizon,)
        Component label used to propagate from each step (-1 after an early stop).
    """
    K_by_label = models["K_by_label"]
    R_by_label = models["R_by_label"]
    x0 = np.asarray(x0, dtype=np.float64).ravel()
    n_state = x0.shape[0]

    traj = np.full((horizon + 1, n_state), np.nan, dtype=np.float64)
    step_labels = np.full(horizon, -1, dtype=np.int64)
    traj[0] = x0

    fixed_label = None
    label = int(classify_partition_labels(x0, partition)[0])
    
    if label < 0:
        # If the initial state is not classified, return a trajectory of NaNs
        return traj, step_labels
    
    for t in range(horizon):
        x_t = traj[t]
        step_labels[t] = label
        phi_next = Phi.evaluate(x_t) @ K_by_label[label]
        traj[t + 1] = (phi_next @ R_by_label[label]).ravel()

    return traj, step_labels
