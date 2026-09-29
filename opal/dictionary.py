"""RBF center selection and dictionary construction."""

import time

import numpy as np
from sklearn.cluster import KMeans

from opal.basis_functions import CombinedDictionary, PolynomialBasis, RadialBasis, ThinPlateSpline

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

    ``n_centers == 0`` returns an empty ``(0, n_state)`` array (no RBFs: the
    dictionary becomes polynomial-only). If ``n_centers`` is negative or exceeds
    the pool size, all pool centers are returned unchanged.
    """
    centers_pool = np.asarray(centers_pool, dtype=np.float64)
    if centers_pool.ndim != 2:
        raise ValueError("centers_pool must have shape (n_centers, n_state).")
    n_available = centers_pool.shape[0]
    if n_available < 1:
        raise ValueError("centers_pool must contain at least one center.")

    if n_centers == 0:
        if verbose:
            print("number_centers_edmd = 0: no RBF centers (polynomial-only dictionary).")
        return np.empty((0, centers_pool.shape[1]), dtype=np.float64)

    if n_centers < 0 or n_centers >= n_available:
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
    """RBF (+ optional polynomial) dictionary used by the partition and EDMD models.

    ``centers`` with zero rows gives a polynomial-only dictionary; ``d_poly == -1``
    gives an RBF-only dictionary. Both at once is an error.
    """
    centers = np.asarray(centers, dtype=np.float64)
    if centers.ndim != 2:
        raise ValueError("centers must have shape (n_centers, n_state).")
    if centers.shape[0] == 0:
        if d_poly == -1:
            raise ValueError("Dictionary would be empty: no RBF centers and d_poly = -1.")
        return PolynomialBasis(n=n, d=d_poly)
    Phi_RBF = RadialBasis(n, centers=centers, gamma=gamma)
    if d_poly == -1:
        return Phi_RBF
    poly = PolynomialBasis(n=n, d=d_poly)
    return CombinedDictionary(Phi_RBF, poly)
