import numpy as np
import pytest
from src.clustering.dbscan import fit as dbscan_fit
from src.clustering.finch import fit as finch_fit


def make_clustered_features(n_clusters=3, n_per_cluster=20, d=128, seed=0):
    """Create L2-normalised features with clear cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    features = []
    for c in centers:
        pts = c + rng.standard_normal((n_per_cluster, d)) * 0.05
        features.append(pts)
    features = np.concatenate(features, axis=0).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features


# ---- DBSCAN ----

def test_dbscan_output_length():
    feats = make_clustered_features()
    labels = dbscan_fit(feats)
    assert labels.shape == (60,)


def test_dbscan_finds_clusters():
    feats = make_clustered_features(n_clusters=3)
    labels = dbscan_fit(feats, eps=0.3, min_samples=3)
    unique = set(labels[labels != -1])
    assert len(unique) >= 2


def test_dbscan_outlier_label_is_minus_one():
    feats = make_clustered_features()
    labels = dbscan_fit(feats)
    assert set(np.unique(labels)).issubset(set(range(-1, 100)))


# ---- FINCH ----

def test_finch_output_length():
    feats = make_clustered_features()
    labels = finch_fit(feats)
    assert labels.shape == (60,)


def test_finch_no_outliers():
    feats = make_clustered_features()
    labels = finch_fit(feats)
    assert (labels >= 0).all()


def test_finch_multiple_clusters():
    feats = make_clustered_features(n_clusters=3)
    labels = finch_fit(feats)
    assert len(np.unique(labels)) >= 2
