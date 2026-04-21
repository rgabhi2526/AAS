"""
NP3: Non-Parametric, Plug-and-Play constrained cluster refinement.

Given initial cluster labels and pairwise must-link (ML) / cannot-link (CL)
constraints from the oracle, NP3 produces refined labels satisfying all constraints.

Algorithm (paper Section 3, Figure 2):
  1. Satisfy all ML constraints by merging clusters (union-find).
  2. For each resulting impure cluster (has internal CL violations):
     a. Find ML groups (transitive closure of ML constraints within the cluster).
     b. Build conflict graph: nodes = ML groups, edges = CL between groups.
     c. Find CL groups (connected components of conflict graph).
     d. Color the CL group with highest chromatic number.
     e. Hungarian-match remaining CL groups to existing color labels.
     f. Assign unconstrained points to closest existing sub-cluster.
"""
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple, Optional

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def refine_labels(
    labels: np.ndarray,
    must_links: List[Tuple[int, int]],
    cannot_links: List[Tuple[int, int]],
    features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Refine cluster labels to satisfy all pairwise constraints.

    Args:
        labels:       (n,) initial cluster label array (e.g. SpCL pseudo-labels)
        must_links:   list of (i, j) — samples i and j must share a cluster
        cannot_links: list of (i, j) — samples i and j must be in different clusters
        features:     (n, d) feature matrix for Hungarian matching.  If None,
                      falls back to greedy coloring without Hungarian matching.

    Returns:
        refined_labels: (n,) label array satisfying all provided constraints
    """
    if not must_links and not cannot_links:
        return labels.copy()

    labels = _merge_must_links(labels.copy(), must_links)
    labels = _resolve_cannot_links(labels, must_links, cannot_links, features)
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
# Step 2: Resolve cannot-link violations
# ---------------------------------------------------------------------------

def _resolve_cannot_links(
    labels: np.ndarray,
    must_links: List[Tuple[int, int]],
    cannot_links: List[Tuple[int, int]],
    features: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    For each cluster with internal CL violations, split it.

    If *features* is provided (and scipy is installed), the full paper algorithm
    is used: CL groups → color hardest → Hungarian-match rest → assign
    unconstrained to closest sub-cluster.

    Otherwise, falls back to simple greedy graph coloring (legacy behaviour).
    """
    use_hungarian = True
    if features is None:
        print("[NP3] WARNING: features not provided — using greedy coloring "
              "fallback (Hungarian matching disabled)")
        use_hungarian = False
    elif not _HAS_SCIPY:
        print("[NP3] WARNING: scipy not installed — using greedy coloring "
              "fallback (pip install scipy for Hungarian matching)")
        use_hungarian = False

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

        # ---- Build ML groups within this cluster (transitive closure) ----
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

        # ---- Build conflict graph: nodes = ML groups, edges = CL --------
        conflict = nx.Graph()
        conflict.add_nodes_from(range(len(ml_groups)))
        for (i, j) in internal_cls:
            gi, gj = group_of[i], group_of[j]
            if gi != gj:
                conflict.add_edge(gi, gj)

        # ---- Dispatch to full or fallback algorithm ----------------------
        if use_hungarian:
            next_label = _resolve_cluster_hungarian(
                conflict, ml_groups, features, new_labels,
                cluster_lbl, next_label,
            )
        else:
            next_label = _resolve_cluster_greedy(
                conflict, ml_groups, new_labels,
                cluster_lbl, next_label,
            )

    return new_labels


# ---------------------------------------------------------------------------
# Fallback: greedy colouring (original behaviour)
# ---------------------------------------------------------------------------

def _resolve_cluster_greedy(conflict, ml_groups, new_labels, cluster_lbl,
                            next_label):
    """Simple greedy graph coloring — no Hungarian matching."""
    coloring = nx.coloring.greedy_color(conflict, strategy='largest_first')
    max_color = max(coloring.values()) if coloring else 0

    color_to_label = {0: cluster_lbl}
    for color in range(1, max_color + 1):
        color_to_label[color] = next_label
        next_label += 1

    for g_idx, color in coloring.items():
        lbl = color_to_label[color]
        for sample in ml_groups[g_idx]:
            new_labels[sample] = lbl

    return next_label


# ---------------------------------------------------------------------------
# Full NP3: CL groups + Hungarian matching (paper algorithm)
# ---------------------------------------------------------------------------

def _resolve_cluster_hungarian(conflict, ml_groups, features, new_labels,
                               cluster_lbl, next_label):
    """
    Full NP3 algorithm for one impure cluster:
      1. Find CL groups (connected components of conflict graph).
      2. Color the CL group with highest chromatic number → palette.
      3. Hungarian-match remaining CL groups to the palette.
      4. Assign unconstrained ML groups to closest sub-cluster.
    """
    from scipy.optimize import linear_sum_assignment

    # ---- 1. Find CL groups (connected components) -----------------------
    cl_components = list(nx.connected_components(conflict))

    constrained_ccs = []    # CL groups with actual conflicts (edges)
    unconstrained_gidxs = []  # ML group indices with no CL edges

    for cc in cl_components:
        subg = conflict.subgraph(cc)
        if subg.number_of_edges() > 0:
            constrained_ccs.append(cc)
        else:
            # Isolated node(s) — no CL constraints
            unconstrained_gidxs.extend(cc)

    if not constrained_ccs:
        # No actual conflicts — nothing to split
        return next_label

    # ---- 2. Find the CL group with highest chromatic number ---------------
    best_chi = 0
    hardest_cc = None
    hardest_coloring = None
    cc_colorings = {}  # cc_id → (coloring_dict, chromatic_number)

    for cc in constrained_ccs:
        subg = conflict.subgraph(cc)
        local_coloring = nx.coloring.greedy_color(subg, strategy='largest_first')
        chi = (max(local_coloring.values()) + 1) if local_coloring else 1
        cc_colorings[id(cc)] = (local_coloring, chi)
        if chi > best_chi:
            best_chi = chi
            hardest_cc = cc
            hardest_coloring = local_coloring

    n_global_colors = best_chi

    # ---- 3. Establish colour palette from the hardest CL group -----------
    color_to_label = {0: cluster_lbl}
    for c in range(1, n_global_colors):
        color_to_label[c] = next_label
        next_label += 1

    # Assign labels for the hardest group
    for g_idx, color in hardest_coloring.items():
        for sample in ml_groups[g_idx]:
            new_labels[sample] = color_to_label[color]

    # Compute centroids per global colour (L2-normalised for cosine)
    color_centroids = _compute_color_centroids(
        hardest_coloring, ml_groups, features, n_global_colors
    )

    # ---- 4. Hungarian-match remaining constrained CL groups ---------------
    for cc in constrained_ccs:
        if cc is hardest_cc:
            continue

        local_coloring, local_chi = cc_colorings[id(cc)]

        # Group samples by their local colour
        local_color_samples: dict = defaultdict(list)
        for g_idx, lc in local_coloring.items():
            local_color_samples[lc].extend(ml_groups[g_idx])

        local_colors = sorted(local_color_samples.keys())

        if local_chi == 1:
            # Only one colour — assign entire group to closest global colour
            all_samples = []
            for s in local_color_samples.values():
                all_samples.extend(s)
            centroid = _centroid(features, all_samples)
            best_gc = _closest_color(centroid, color_centroids)
            for sample in all_samples:
                new_labels[sample] = color_to_label[best_gc]
            continue

        # Extend global palette if local chi exceeds it (greedy approx edge case)
        effective_n_cols = max(n_global_colors, local_chi)
        for c in range(n_global_colors, effective_n_cols):
            if c not in color_to_label:
                color_to_label[c] = next_label
                next_label += 1

        # Build cost matrix: rows = local colours, cols = global colours
        cost = np.full((len(local_colors), effective_n_cols), 2.0)
        for ri, lc in enumerate(local_colors):
            samples = local_color_samples[lc]
            centroid = _centroid(features, samples)
            for gc in range(effective_n_cols):
                if gc in color_centroids:
                    cost[ri, gc] = 1.0 - float(np.dot(centroid, color_centroids[gc]))
                # else: keep default 2.0 (new colour — acceptable but not preferred)

        # Solve optimal assignment
        row_ind, col_ind = linear_sum_assignment(cost)

        for ri, ci in zip(row_ind, col_ind):
            lc = local_colors[ri]
            lbl = color_to_label[ci]
            for sample in local_color_samples[lc]:
                new_labels[sample] = lbl

        # Update centroids for any newly used colours
        for ri, ci in zip(row_ind, col_ind):
            if ci not in color_centroids:
                lc = local_colors[ri]
                color_centroids[ci] = _centroid(features, local_color_samples[lc])

    # ---- 5. Assign unconstrained ML groups to closest sub-cluster --------
    for g_idx in unconstrained_gidxs:
        samples = list(ml_groups[g_idx])
        centroid = _centroid(features, samples)
        best_gc = _closest_color(centroid, color_centroids)
        for sample in samples:
            new_labels[sample] = color_to_label[best_gc]

    return next_label


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _centroid(features: np.ndarray, sample_indices) -> np.ndarray:
    """Compute L2-normalised centroid of sample features."""
    indices = list(sample_indices)
    centroid = features[indices].mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm > 0:
        centroid = centroid / norm
    return centroid


def _compute_color_centroids(coloring, ml_groups, features, n_colors):
    """Compute L2-normalised centroid for each colour class."""
    centroids = {}
    for c in range(n_colors):
        samples = []
        for g_idx, assigned_color in coloring.items():
            if assigned_color == c:
                samples.extend(ml_groups[g_idx])
        if samples:
            centroids[c] = _centroid(features, samples)
    return centroids


def _closest_color(centroid: np.ndarray, color_centroids: dict) -> int:
    """Find the global colour whose centroid is most similar (cosine)."""
    best_gc = 0
    best_sim = -2.0
    for gc, gc_centroid in color_centroids.items():
        sim = float(np.dot(centroid, gc_centroid))
        if sim > best_sim:
            best_sim = sim
            best_gc = gc
    return best_gc
