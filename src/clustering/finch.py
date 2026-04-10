"""FINCH clustering wrapper — returns first-partition labels."""
import numpy as np
from finch import FINCH


def fit(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Run FINCH on L2-normalised features and return the first partition.

    FINCH assigns every point to a cluster (no outliers), making it
    complementary to DBSCAN for AAS's uncertainty-region detection.

    Args:
        features:     (n, d) L2-normalised float32 array
        random_state: seed for reproducibility

    Returns:
        labels: (n,) int array; all values >= 0
    """
    c, num_clust, req_c = FINCH(
        features,
        distance='cosine',
        verbose=False,
        random_state=random_state,
    )
    return c[:, 0]   # first partition
