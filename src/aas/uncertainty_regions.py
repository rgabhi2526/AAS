"""
Identify regions of uncertainty from disagreements between two clusterings.

A 'region of uncertainty' is the transitive closure of cluster pairs
that partially overlap (0 < IoU < 1) between methods A (DBSCAN) and B (FINCH).
"""
import numpy as np
from collections import defaultdict
from typing import List, Set


def compute_iou(set_a: Set[int], set_b: Set[int]) -> float:
    """Intersection-over-Union between two sets of sample indices."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def find_uncertainty_regions(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> List[List[int]]:
    """
    Find uncertainty regions from disagreements between clusterings A and B.

    Args:
        labels_a: (n,) cluster labels from DBSCAN  (-1 = outlier, excluded)
        labels_b: (n,) cluster labels from FINCH   (all >= 0)

    Returns:
        regions: list of regions; each region is a sorted list of sample indices.
                 Empty list when the two clusterings fully agree.
    """
    # Build cluster membership sets (exclude DBSCAN outliers)
    clusters_a: dict = defaultdict(set)
    for i, lbl in enumerate(labels_a):
        if lbl != -1:
            clusters_a[lbl].add(i)

    clusters_b: dict = defaultdict(set)
    for i, lbl in enumerate(labels_b):
        clusters_b[lbl].add(i)

    # Build adjacency graph of partially overlapping clusters
    adj: dict = defaultdict(set)
    for lbl_a, set_a in clusters_a.items():
        for lbl_b, set_b in clusters_b.items():
            iou = compute_iou(set_a, set_b)
            if 0 < iou < 1:
                node_a = ('A', lbl_a)
                node_b = ('B', lbl_b)
                adj[node_a].add(node_b)
                adj[node_b].add(node_a)

    # Connected components via BFS = transitive closure = uncertainty regions
    visited: Set = set()
    regions: List[List[int]] = []

    for start in list(adj.keys()):
        if start in visited:
            continue
        component: set = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component.add(node)
            queue.extend(adj[node] - visited)

        sample_indices: set = set()
        for (source, lbl) in component:
            if source == 'A':
                sample_indices |= clusters_a[lbl]
            else:
                sample_indices |= clusters_b[lbl]

        if sample_indices:
            regions.append(sorted(sample_indices))

    return regions
