"""
NP3: Non-Parametric, Plug-and-Play constrained cluster refinement.

Given initial cluster labels and pairwise must-link (ML) / cannot-link (CL)
constraints from the oracle, NP3 produces refined labels satisfying all constraints.

Algorithm (paper Section 3, Figure 2):
  1. Satisfy all ML constraints by merging clusters (union-find).
  2. For each resulting impure cluster (has internal CL violations):
     a. Find ML groups (transitive closure of ML constraints within the cluster).
     b. Build conflict graph: nodes = ML groups, edges = CL between groups.
     c. Greedy-colour the conflict graph.
     d. Assign cluster labels by colour; Hungarian-match unconstrained points.
"""
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple


def refine_labels(
    labels: np.ndarray,
    must_links: List[Tuple[int, int]],
    cannot_links: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Refine cluster labels to satisfy all pairwise constraints.

    Args:
        labels:       (n,) initial cluster label array (e.g. SpCL pseudo-labels)
        must_links:   list of (i, j) — samples i and j must share a cluster
        cannot_links: list of (i, j) — samples i and j must be in different clusters

    Returns:
        refined_labels: (n,) label array satisfying all provided constraints
    """
    if not must_links and not cannot_links:
        return labels.copy()

    labels = _merge_must_links(labels.copy(), must_links)
    labels = _resolve_cannot_links(labels, must_links, cannot_links)
    return labels


# ---------------------------------------------------------------------------
# Step 1: Merge clusters to satisfy must-link constraints
# ---------------------------------------------------------------------------

def _merge_must_links(
    labels: np.ndarray,
    must_links: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Union-find: samples connected by ML constraints receive the same cluster label.
    The cluster label chosen is that of the component root in the original labels.
    """
    n = len(labels)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]   # path compression
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (i, j) in must_links:
        union(i, j)

    root_to_label: dict = {}
    new_labels = labels.copy()
    for idx in range(n):
        root = find(idx)
        if root not in root_to_label:
            root_to_label[root] = labels[root]
        new_labels[idx] = root_to_label[root]

    return new_labels


# ---------------------------------------------------------------------------
# Step 2: Resolve cannot-link violations via graph colouring
# ---------------------------------------------------------------------------

def _resolve_cannot_links(
    labels: np.ndarray,
    must_links: List[Tuple[int, int]],
    cannot_links: List[Tuple[int, int]],
) -> np.ndarray:
    """
    For each cluster with internal CL violations, split it using graph colouring.
    """
    cl_set = {tuple(sorted(p)) for p in cannot_links}
    ml_set = {tuple(sorted(p)) for p in must_links}

    label_to_members: dict = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_to_members[int(lbl)].append(idx)

    new_labels = labels.copy()
    next_label = int(labels.max()) + 1

    for cluster_lbl, members in label_to_members.items():
        members_set = set(members)

        # Find CL constraints internal to this cluster
        internal_cls = [
            (i, j) for (i, j) in cl_set
            if i in members_set and j in members_set
        ]
        if not internal_cls:
            continue   # cluster is pure

        # Build ML groups within this cluster (transitive closure)
        ml_graph = nx.Graph()
        ml_graph.add_nodes_from(members)
        for (i, j) in ml_set:
            if i in members_set and j in members_set:
                ml_graph.add_edge(i, j)

        ml_groups = list(nx.connected_components(ml_graph))
        group_of = {
            sample: g_idx
            for g_idx, group in enumerate(ml_groups)
            for sample in group
        }

        # Build conflict graph: nodes = ML groups, edges = CL between groups
        conflict = nx.Graph()
        conflict.add_nodes_from(range(len(ml_groups)))
        for (i, j) in internal_cls:
            gi, gj = group_of[i], group_of[j]
            if gi != gj:
                conflict.add_edge(gi, gj)

        # Greedy colouring
        coloring = nx.coloring.greedy_color(conflict, strategy='largest_first')
        max_color = max(coloring.values()) if coloring else 0

        # colour 0 → keep existing cluster label; other colours → new labels
        color_to_label = {0: cluster_lbl}
        for color in range(1, max_color + 1):
            color_to_label[color] = next_label
            next_label += 1

        for g_idx, color in coloring.items():
            lbl = color_to_label[color]
            for sample in ml_groups[g_idx]:
                new_labels[sample] = lbl

    return new_labels
