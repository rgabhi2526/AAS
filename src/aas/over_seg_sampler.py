"""
Sample pairs to resolve over-segmentation errors (U_os).

Over-segmentation: one individual split across multiple uncertain regions.
Strategy: find the medoid of each region, then pair medoids from different
regions whose cosine similarity >= s_min.
"""
import numpy as np
from typing import List, Tuple


def compute_medoid(indices: List[int], features: np.ndarray) -> int:
    """
    Find the medoid of a set of samples.

    The medoid minimises average cosine distance to all other samples in the set.
    Since features are L2-normalised, cosine distance = 1 − dot product.

    Args:
        indices: sample indices (into the full features array)
        features: (n, d) L2-normalised feature matrix

    Returns:
        Index (in the global features array) of the medoid
    """
    sub_feats = features[indices]               # (m, d)
    sim_matrix = sub_feats @ sub_feats.T        # (m, m)
    avg_dist = (1 - sim_matrix).mean(axis=1)    # average cosine distance per sample
    local_idx = int(np.argmin(avg_dist))
    return indices[local_idx]


def sample_over_seg_pairs(
    regions: List[List[int]],
    features: np.ndarray,
    k_max: int = 5,
    s_min: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Construct U_os: medoid pairs across different uncertain regions with
    high similarity (likely same individual, over-segmented).

    For each region r_k with medoid m_k, pair m_k with up to k_max other
    medoids m_k' whose cosine similarity >= s_min.

    Args:
        regions: list of uncertainty regions (each a list of sample indices)
        features: (n, d) L2-normalised features
        k_max: max nearest-medoid neighbours per medoid
        s_min: minimum cosine similarity threshold

    Returns:
        pairs: list of unique (i, j) pairs (i < j) for U_os
    """
    if len(regions) < 2:
        return []

    medoids = [compute_medoid(region, features) for region in regions]
    medoid_feats = features[medoids]                # (M, d)
    sim_matrix = medoid_feats @ medoid_feats.T      # (M, M)
    np.fill_diagonal(sim_matrix, -2.0)              # exclude self-similarity

    pairs: set = set()
    for k, m_k in enumerate(medoids):
        sorted_neighbours = np.argsort(sim_matrix[k])[::-1]
        count = 0
        for neighbour_k in sorted_neighbours:
            if count >= k_max:
                break
            sim = sim_matrix[k, neighbour_k]
            if sim < s_min:
                break   # descending order — no further neighbours qualify
            m_neighbour = medoids[neighbour_k]
            pairs.add(tuple(sorted((m_k, m_neighbour))))
            count += 1

    return list(pairs)
