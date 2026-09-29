"""Graph-merged approximate omega-limit partition (Algorithm 1).

Scalar k-means on each invariant coordinate -> occupied multi-index fine cells ->
occupancy filter -> transition-based merge -> value-space merge -> connected
components. Also loads a saved partition archive and classifies new states.
"""

from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

from .dictionary import build_rbf_polynomial_dictionary
from .invariants import evaluate_invariant_embedding

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
    tau_robust=0.0,
    embedding_min=None,
    embedding_max=None,
):
    """
    Apply a fitted graph-merged partition to new embedding values F.

    Returns final partition labels; unknown or dropped points receive -1. When
    ``tau_robust > 0``, points whose relative distance to the nearest final
    boundary (``compute_relative_boundary_distances``) is below ``tau_robust``
    are dropped as well; this needs ``embedding_min``/``embedding_max``.
    """
    scalar_labels = assign_scalar_clusters_from_centers(F, cluster_centers)
    cell_ids = encode_multi_index_labels_fixed(scalar_labels, label_to_id)
    labels = _assign_final_labels(cell_ids, cell_to_component, retained_cell_ids)
    if float(tau_robust) > 0.0:
        rel_dist, _ = compute_relative_boundary_distances(
            F,
            {
                "cluster_centers": cluster_centers,
                "label_to_id": label_to_id,
                "cell_to_component": cell_to_component,
                "retained_cell_ids": retained_cell_ids,
                "embedding_min": embedding_min,
                "embedding_max": embedding_max,
            },
        )
        labels = apply_robustness_margin(labels, rel_dist, tau_robust)
    return labels


def predict_partition_labels_from_embedding(F, partition):
    """Apply a loaded/fitted partition dict (see ``load_partition_inference``) to F."""
    return predict_graph_merged_labels(
        F,
        label_to_id=partition["label_to_id"],
        cell_to_component=partition["cell_to_component"],
        retained_cell_ids=partition["retained_cell_ids"],
        cluster_centers=partition["cluster_centers"],
        tau_robust=partition.get("tau_robust", 0.0),
        embedding_min=partition.get("embedding_min"),
        embedding_max=partition.get("embedding_max"),
    )


def apply_robustness_margin(labels, rel_dist, tau_robust):
    """Set labels to -1 where the relative boundary distance is below tau_robust."""
    labels = np.asarray(labels, dtype=np.int64).copy()
    rel_dist = np.asarray(rel_dist, dtype=np.float64).ravel()
    tau_robust = float(tau_robust)
    if tau_robust > 0.0:
        labels[rel_dist < tau_robust] = -1
    return labels


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
    tau_robust=0.0,
):
    """
    Fit the graph-merged approximate omega-limit partition.

    Returns a dictionary containing scalar labels, fine cell ids, retained cells,
    centroids, transition statistics, sequential merge graph edges, connected
    components, and final labels for X and Y.

    Merge is two-stage: transition edges on fine cells, then centroid edges on
    the transition-merged graph. Afterwards, if ``tau_robust > 0``, points whose
    relative distance to the nearest final boundary is below ``tau_robust`` are
    dropped (label -1); the same margin is applied at inference.
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

    # Robustness margin: drop points too close (relatively) to a final boundary.
    embedding_min = np.min(F_X, axis=0)
    embedding_max = np.max(F_X, axis=0)
    inference = {
        "cluster_centers": _stack_cluster_centers(cluster_models),
        "label_to_id": label_to_id,
        "cell_to_component": cell_to_component,
        "retained_cell_ids": np.array(sorted(retained_cell_ids), dtype=np.int64),
        "embedding_min": embedding_min,
        "embedding_max": embedding_max,
    }
    rel_boundary_dist_X, _ = compute_relative_boundary_distances(F_X, inference)
    rel_boundary_dist_Y, _ = compute_relative_boundary_distances(F_Y, inference)
    n_robust_dropped_X = int(np.sum((final_labels_X >= 0) & (rel_boundary_dist_X < tau_robust)))
    final_labels_X = apply_robustness_margin(final_labels_X, rel_boundary_dist_X, tau_robust)
    final_labels_Y = apply_robustness_margin(final_labels_Y, rel_boundary_dist_Y, tau_robust)

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
        "embedding_min": embedding_min,
        "embedding_max": embedding_max,
        "rel_boundary_dist_X": rel_boundary_dist_X,
        "rel_boundary_dist_Y": rel_boundary_dist_Y,
        "tau_robust": float(tau_robust),
        "n_robust_dropped_X": n_robust_dropped_X,
    }


def _stack_cluster_centers(cluster_models):
    """Stack per-coordinate KMeans centers into (r, k_max, 1), NaN-padded."""
    k_max = max(int(model.cluster_centers_.shape[0]) for model in cluster_models)
    stacked = np.full((len(cluster_models), k_max, 1), np.nan, dtype=np.float64)
    for j, model in enumerate(cluster_models):
        center = np.asarray(model.cluster_centers_, dtype=np.float64)
        stacked[j, : center.shape[0], 0] = center[:, 0]
    return stacked


def build_partition_inference(result, *, coeffs_inv, centers_global, gamma, d_poly, n):
    """
    Build the inference dict (same keys as ``load_partition_inference``) directly
    from a ``fit_graph_merged_partition`` result, without saving an archive.
    """
    return {
        "coeffs_inv": np.asarray(coeffs_inv),
        "centers_global": np.asarray(centers_global),
        "embedding_min": np.asarray(result["embedding_min"], dtype=np.float64),
        "embedding_max": np.asarray(result["embedding_max"], dtype=np.float64),
        "tau_robust": float(result["tau_robust"]),
        "gamma": float(gamma),
        "d_poly": int(d_poly),
        "n": int(n),
        "cluster_centers": _stack_cluster_centers(result["cluster_models"]),
        "retained_cell_ids": np.asarray(result["retained_cell_ids"], dtype=np.int64),
        "label_to_id": dict(result["label_to_id"]),
        "cell_to_component": dict(result["cell_to_component"]),
    }


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

    if "F_X" in data:
        F_X = np.asarray(data["F_X"], dtype=np.float64)
        embedding_min = np.min(F_X, axis=0)
        embedding_max = np.max(F_X, axis=0)
    else:
        embedding_min = None
        embedding_max = None

    return {
        "coeffs_inv": np.asarray(data["coeffs_inv"]),
        "centers_global": np.asarray(data["centers_global"]),
        "embedding_min": embedding_min,
        "embedding_max": embedding_max,
        "tau_robust": float(data["tau_robust"]) if "tau_robust" in data else 0.0,
        "gamma": float(data["gamma"]),
        "d_poly": int(data["d_poly"]),
        "n": int(data["n"]),
        "cluster_centers": np.asarray(data["cluster_centers"]),
        "retained_cell_ids": np.asarray(data["retained_cell_ids"], dtype=np.int64),
        "label_to_id": label_to_id,
        "cell_to_component": cell_to_component,
    }


def compute_relative_boundary_distances(F, partition):
    """
    Relative distance of each embedding value to the nearest final-partition boundary.

    Along each invariant coordinate j, scalar clusters are separated by midpoints
    between adjacent sorted KMeans centers. Starting from a sample's fine cell, the
    interval along j is extended across neighbouring clusters (other coordinates
    fixed) as long as the final component label is unchanged, so only midpoints
    where the final label changes (or the point would be dropped) count as
    boundaries. Open sides of outermost clusters are capped by the training-set
    embedding range (``partition["embedding_min"/"embedding_max"]``) for the
    region length but are not treated as boundaries.

    Returns
    -------
    rel_dist : ndarray, shape (N,)
        min_j dist_j / length_j; +inf where no coordinate has a boundary.
    per_coord : ndarray, shape (N, r)
        Per-coordinate relative distances (inf where coordinate j has no boundary).
    """
    F = np.asarray(F, dtype=np.float64)
    cluster_centers = np.asarray(partition["cluster_centers"], dtype=np.float64)
    label_to_id = partition["label_to_id"]
    cell_to_component = partition["cell_to_component"]
    retained = {int(cell_id) for cell_id in partition["retained_cell_ids"]}
    embedding_min = partition.get("embedding_min")
    embedding_max = partition.get("embedding_max")
    if embedding_min is None or embedding_max is None:
        raise ValueError("partition must carry embedding_min/embedding_max (needs F_X in archive).")

    n_samples, r = F.shape
    scalar_labels = assign_scalar_clusters_from_centers(F, cluster_centers)

    # Per coordinate: cluster label -> rank in sorted-center order, and the K+1 edges
    # [data_min, midpoints..., data_max] bounding the K sorted clusters.
    orders, ranks, edges = [], [], []
    for j in range(r):
        valid = np.isfinite(cluster_centers[j, :, 0])
        centers = cluster_centers[j, valid, 0]
        order = np.argsort(centers)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        sorted_centers = centers[order]
        mids = 0.5 * (sorted_centers[:-1] + sorted_centers[1:])
        lo = min(float(embedding_min[j]), float(sorted_centers[0]))
        hi = max(float(embedding_max[j]), float(sorted_centers[-1]))
        orders.append(order)
        ranks.append(rank)
        edges.append(np.concatenate([[lo], mids, [hi]]))

    def final_label(row):
        cell_id = label_to_id.get(tuple(int(v) for v in row), -1)
        if cell_id in retained and cell_id in cell_to_component:
            return cell_to_component[cell_id]
        return -1

    # Merged interval per unique multi-index (cheap: few distinct cells).
    unique_rows, inverse = np.unique(scalar_labels, axis=0, return_inverse=True)
    inverse = inverse.ravel()
    n_unique = unique_rows.shape[0]
    lo_edge = np.empty((n_unique, r))
    hi_edge = np.empty((n_unique, r))
    lo_is_boundary = np.zeros((n_unique, r), dtype=bool)
    hi_is_boundary = np.zeros((n_unique, r), dtype=bool)
    for u, row in enumerate(unique_rows):
        own_label = final_label(row)
        for j in range(r):
            n_clusters = orders[j].size
            p = int(ranks[j][row[j]])
            neighbour = row.copy()

            q = p
            while q - 1 >= 0:
                neighbour[j] = orders[j][q - 1]
                if final_label(neighbour) != own_label:
                    break
                q -= 1
            lo_edge[u, j] = edges[j][q]
            lo_is_boundary[u, j] = q > 0

            q = p
            while q + 1 < n_clusters:
                neighbour[j] = orders[j][q + 1]
                if final_label(neighbour) != own_label:
                    break
                q += 1
            hi_edge[u, j] = edges[j][q + 1]
            hi_is_boundary[u, j] = q + 1 < n_clusters

    lo_s = lo_edge[inverse]
    hi_s = hi_edge[inverse]
    length = hi_s - lo_s
    d_lo = np.where(lo_is_boundary[inverse], F - lo_s, np.inf)
    d_hi = np.where(hi_is_boundary[inverse], hi_s - F, np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        per_coord = np.minimum(d_lo, d_hi) / length
    per_coord = np.where(np.isfinite(np.minimum(d_lo, d_hi)), per_coord, np.inf)
    rel_dist = np.min(per_coord, axis=1) if r > 0 else np.full(n_samples, np.inf)
    return rel_dist, per_coord


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
    return predict_partition_labels_from_embedding(F, partition)


