"""
Re-ID evaluation metrics.

Implements: mAP, mINP, BAKS, AUCROC, Top-{1,3,5,10}
as used in the AAS paper (Table 1).
"""
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional


def compute_metrics(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_is_known: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute all Re-ID metrics for one dataset split.

    Args:
        query_feats:    (nq, d) L2-normalised query features
        gallery_feats:  (ng, d) L2-normalised gallery features
        query_labels:   (nq,) integer identity labels
        gallery_labels: (ng,) integer identity labels
        query_is_known: (nq,) bool array — True if query ID appears in gallery.
                        Required for BAKS and AUCROC. Pass None to skip.

    Returns:
        dict with keys: mAP, mINP, top1, top3, top5, top10
                        + BAKS and AUCROC if query_is_known is provided
    """
    sim_matrix = query_feats @ gallery_feats.T      # (nq, ng)

    map_scores, minp_scores = [], []
    top_k_hits: Dict[int, list] = {1: [], 3: [], 5: [], 10: []}

    for q_idx in range(len(query_labels)):
        sims = sim_matrix[q_idx]
        sorted_idx = np.argsort(sims)[::-1]
        matches = (gallery_labels[sorted_idx] == query_labels[q_idx]).astype(int)

        map_scores.append(_average_precision(matches))
        minp_scores.append(_mean_inverse_negative_penalty(matches))

        for k in top_k_hits:
            top_k_hits[k].append(int(matches[:k].sum() > 0))

    results: Dict[str, float] = {
        'mAP':   float(np.mean(map_scores)),
        'mINP':  float(np.mean(minp_scores)),
        'top1':  float(np.mean(top_k_hits[1])),
        'top3':  float(np.mean(top_k_hits[3])),
        'top5':  float(np.mean(top_k_hits[5])),
        'top10': float(np.mean(top_k_hits[10])),
    }

    if query_is_known is not None:
        known_idx = np.where(query_is_known)[0]
        results['BAKS'] = (
            float(np.mean([map_scores[i] for i in known_idx]))
            if len(known_idx) > 0 else 0.0
        )
        max_sims = sim_matrix.max(axis=1)
        results['AUCROC'] = float(roc_auc_score(query_is_known.astype(int), max_sims))

    return results


def _average_precision(matches: np.ndarray) -> float:
    """Compute AP from a binary relevance array (1 = match, 0 = non-match)."""
    n_pos = matches.sum()
    if n_pos == 0:
        return 0.0
    hits, ap = 0, 0.0
    for rank, m in enumerate(matches, 1):
        if m:
            hits += 1
            ap += hits / rank
    return ap / n_pos


def _mean_inverse_negative_penalty(matches: np.ndarray) -> float:
    """mINP = 1 / (1-indexed rank of the last positive match)."""
    pos_ranks = np.where(matches)[0]
    if len(pos_ranks) == 0:
        return 0.0
    return 1.0 / (pos_ranks[-1] + 1)
