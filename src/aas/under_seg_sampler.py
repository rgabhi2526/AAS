"""
Sample pairs to resolve under-segmentation errors (U_us).

Under-segmentation: multiple individuals merged into one cluster.
Strategy: within each uncertain region, find pairs where methods A and B
disagree (symmetric difference), then filter to the single most-informative
(closest) pair per distinct cluster boundary (P_cand).
"""
import numpy as np
from collections import defaultdict
from typing import List, Set, Tuple


def sample_under_seg_pairs(
    regions: List[List[int]],
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    features: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Construct U_us: non-redundant inconsistent pairs within uncertainty regions.

    For each region S_k:
      I_tilde_k  = (intra-cluster pairs from A) △ (intra-cluster pairs from B)
      P_cand_k   = closest cross-cluster pair for each distinct cluster boundary
      I_k        = I_tilde_k ∩ P_cand_k

    U_us = ∪ I_k  over all regions.

    Args:
        regions:  list of uncertainty regions (sample indices)
        labels_a: (n,) DBSCAN cluster labels (-1 = outlier)
        labels_b: (n,) FINCH cluster labels
        features: (n, d) L2-normalised features

    Returns:
        pairs: list of unique (i, j) pairs (i < j) for U_us
    """
    all_pairs: Set[Tuple[int, int]] = set()

    for region in regions:
        a_groups: dict = defaultdict(set)
        b_groups: dict = defaultdict(set)
        for idx in region:
            if labels_a[idx] != -1:
                a_groups[labels_a[idx]].add(idx)
            b_groups[labels_b[idx]].add(idx)

        a_plus = _intra_cluster_pairs(a_groups)
        b_plus = _intra_cluster_pairs(b_groups)
        i_tilde = a_plus.symmetric_difference(b_plus)

        if not i_tilde:
            continue

        all_group_sets = list(a_groups.values()) + list(b_groups.values())
        p_cand = _closest_inter_cluster_pairs(all_group_sets, features)

        all_pairs |= (i_tilde & p_cand)

    return list(all_pairs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _intra_cluster_pairs(groups: dict) -> Set[Tuple[int, int]]:
    """All ordered (i < j) pairs whose members share a cluster."""
    pairs: Set[Tuple[int, int]] = set()
    for members in groups.values():
        members_list = sorted(members)
        for i in range(len(members_list)):
            for j in range(i + 1, len(members_list)):
                pairs.add((members_list[i], members_list[j]))
    return pairs


def _closest_inter_cluster_pairs(
    groups: List[Set[int]],
    features: np.ndarray,
) -> Set[Tuple[int, int]]:
    """
    For each pair of groups, return the single closest cross-group pair
    (highest cosine similarity). This forms P_cand.
    """
    pairs: Set[Tuple[int, int]] = set()
    groups_list = [sorted(g) for g in groups if g]

    for i in range(len(groups_list)):
        for j in range(i + 1, len(groups_list)):
            g1, g2 = groups_list[i], groups_list[j]
            f1 = features[g1]           # (n1, d)
            f2 = features[g2]           # (n2, d)
            sim_matrix = f1 @ f2.T      # (n1, n2)
            best = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            best_i = int(g1[best[0]])
            best_j = int(g2[best[1]])
            pairs.add(tuple(sorted((best_i, best_j))))

    return pairs
