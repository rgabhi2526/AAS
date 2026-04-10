"""
Main AAS sampler — orchestrates all sub-steps to produce B annotatable pairs.

Steps:
  1. Cluster features with DBSCAN (A) and FINCH (B).
  2. Find uncertainty regions from cluster disagreements.
  3. Build U_os (over-segmentation) and U_us (under-segmentation) pools.
  4. Sample B pairs from U = U_os ∪ U_us via marginal distribution P(Y).
"""
import numpy as np
from typing import List, Tuple

from src.clustering.dbscan import fit as dbscan_fit
from src.clustering.finch import fit as finch_fit
from src.aas.uncertainty_regions import find_uncertainty_regions
from src.aas.over_seg_sampler import sample_over_seg_pairs
from src.aas.under_seg_sampler import sample_under_seg_pairs


def run_aas(
    features: np.ndarray,
    budget: int,
    epsilon: float = 0.6,
    k_max: int = 5,
    s_min: float = 0.3,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 4,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Run the full AAS pipeline and return `budget` pairs for oracle annotation.

    Args:
        features:           (n, d) L2-normalised features
        budget:             number of pairs to sample (B)
        epsilon:            weight balancing U_os vs U_us
                            (0 = all U_us, 1 = all U_os)
        k_max:              max nearest-medoid neighbours for U_os
        s_min:              min cosine similarity threshold for U_os
        dbscan_eps:         DBSCAN neighbourhood radius (cosine distance)
        dbscan_min_samples: DBSCAN minimum core points
        seed:               RNG seed

    Returns:
        sampled_pairs: list of (i, j) sample index pairs
        labels_a:      (n,) DBSCAN cluster labels (passed to NP3)
        labels_b:      (n,) FINCH cluster labels  (passed to NP3)
    """
    rng = np.random.default_rng(seed)

    labels_a = dbscan_fit(features, eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_b = finch_fit(features, random_state=seed)

    regions = find_uncertainty_regions(labels_a, labels_b)

    if not regions:
        # No disagreements — fallback to base USL training without AL
        return [], labels_a, labels_b

    u_os = sample_over_seg_pairs(regions, features, k_max=k_max, s_min=s_min)
    u_us = sample_under_seg_pairs(regions, labels_a, labels_b, features)

    sampled = _marginal_sample(u_os, u_us, features, budget, epsilon, rng)

    return sampled, labels_a, labels_b


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _marginal_sample(
    u_os: List[Tuple[int, int]],
    u_us: List[Tuple[int, int]],
    features: np.ndarray,
    budget: int,
    epsilon: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """
    Sample B pairs from U = U_os ∪ U_us using the marginal distribution P(Y).

    P(Y) is a weighted combination (Eq. 6 in the paper):
      P(Y) = ε · P(Y | Y ∈ U_os) + (1−ε) · P(Y | Y ∈ U_us)

    For U_os pairs: P(Y | U_os) ∝ cosine similarity (higher sim → more likely).
    For U_us pairs: P(Y | U_us) ∝ uniform (equal weight per pair).
    """
    os_set = set(u_os)
    us_set = set(u_us)
    total_pool = list(os_set | us_set)

    if not total_pool:
        return []

    # If pool is smaller than budget, return everything
    if len(total_pool) <= budget:
        return total_pool

    n_us = max(len(u_us), 1)
    probs = np.zeros(len(total_pool), dtype=np.float64)

    for idx, pair in enumerate(total_pool):
        in_os = pair in os_set
        in_us = pair in us_set
        sim = float(features[pair[0]] @ features[pair[1]])

        if in_os and not in_us:
            probs[idx] = epsilon * max(sim, 0.0)
        elif in_us and not in_os:
            probs[idx] = (1 - epsilon) / n_us
        else:
            # Pair appears in both pools — blend both components
            p_os = epsilon * max(sim, 0.0)
            p_us = (1 - epsilon) / n_us
            probs[idx] = 0.5 * p_os + 0.5 * p_us

    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum()

    chosen = rng.choice(len(total_pool), size=budget, replace=False, p=probs)
    return [total_pool[i] for i in chosen]
