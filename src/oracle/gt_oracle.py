"""
Ground-truth oracle simulator for pairwise annotations.

Replaces a human annotator by using GT identity labels to deterministically
answer must-link / cannot-link queries. Used exclusively during reproduction.
"""
import numpy as np
from typing import List, Tuple


class GTOracle:
    """
    Simulates a human annotator using ground-truth identity labels.

    For each queried pair (i, j):
      - gt_labels[i] == gt_labels[j]  →  must-link   (same individual)
      - gt_labels[i] != gt_labels[j]  →  cannot-link (different individuals)

    Args:
        gt_labels: (n,) integer array of ground-truth identity labels
    """

    def __init__(self, gt_labels: np.ndarray):
        self.gt_labels = np.asarray(gt_labels, dtype=int)

    def query(
        self,
        pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Get must-link and cannot-link annotations for a list of pairs.

        Args:
            pairs: list of (i, j) sample index pairs

        Returns:
            must_links:   pairs whose samples share an identity
            cannot_links: pairs whose samples differ in identity
        """
        must_links: List[Tuple[int, int]] = []
        cannot_links: List[Tuple[int, int]] = []

        for (i, j) in pairs:
            if self.gt_labels[i] == self.gt_labels[j]:
                must_links.append((i, j))
            else:
                cannot_links.append((i, j))

        return must_links, cannot_links
