"""DBSCAN clustering wrapper using cosine distance on L2-normalised features."""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


def fit(features: np.ndarray, eps: float = 0.5, min_samples: int = 4) -> np.ndarray:
    """
    Run DBSCAN on L2-normalised features using cosine distance.

    Args:
        features:    (n, d) L2-normalised float32 array
        eps:         max cosine distance for a point to be in a neighbourhood
        min_samples: min points to form a core point

    Returns:
        labels: (n,) int array; -1 indicates noise/outlier
    """
    dist_matrix = cosine_distances(features).astype(np.float64)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
    return db.fit_predict(dist_matrix)
