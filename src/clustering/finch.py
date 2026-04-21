"""FINCH clustering wrapper — returns first-partition labels."""
import numpy as np
from finch import FINCH


def fit(features: np.ndarray, random_state: int = 42, partition: int = 0) -> np.ndarray:
    """
    Run FINCH on L2-normalised features and return the requested partition.

    FINCH assigns every point to a cluster (no outliers), making it
    complementary to DBSCAN for AAS's uncertainty-region detection.

    Args:
        features:     (n, d) L2-normalised float32 array
        random_state: seed for reproducibility
        partition:    which FINCH partition to return (0 = finest, default)

    Returns:
        labels: (n,) int array; all values >= 0
    """
    c, num_clust, req_c = FINCH(
        features,
        distance='cosine',
        verbose=False,
        random_state=random_state,
    )
    # Clamp partition index to available range
    partition = min(partition, c.shape[1] - 1)
    return c[:, partition]
