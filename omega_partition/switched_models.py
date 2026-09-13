"""Multiple local linear (switched) EDMD models, one per partition component."""

import numpy as np

from .partition import classify_partition_labels

def fit_edmd_operator(Phi_X, Phi_Y):
    """Least-squares EDMD operator K such that Phi_Y approx Phi_X @ K."""
    return np.linalg.lstsq(Phi_X, Phi_Y, rcond=None)[0]


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

    label = int(classify_partition_labels(x0, partition)[0])
    if label < 0 or label not in K_by_label:
        # Unclassified initial state: leave the whole prediction as NaN.
        return traj, step_labels

    for t in range(horizon):
        x_t = traj[t]
        if reclassify and t > 0:
            label = int(classify_partition_labels(x_t, partition)[0])
            if label < 0 or label not in K_by_label:
                break
        step_labels[t] = label
        phi_next = Phi.evaluate(x_t) @ K_by_label[label]
        traj[t + 1] = (phi_next @ R_by_label[label]).ravel()

    return traj, step_labels
