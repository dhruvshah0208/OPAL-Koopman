"""Approximate Koopman invariant observables via the generalized eigenvalue problem."""

import numpy as np
from scipy.linalg import eigh as scipy_generalized_eigh

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

