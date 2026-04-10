import numpy as np
import pytest
from src.aas.over_seg_sampler import compute_medoid, sample_over_seg_pairs


def make_features(n=10, d=4, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, d)).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f


def test_medoid_is_in_region():
    features = make_features(10)
    indices = list(range(10))
    m = compute_medoid(indices, features)
    assert m in indices


def test_medoid_single_element():
    features = make_features(5)
    m = compute_medoid([3], features)
    assert m == 3


def test_no_pairs_when_single_region():
    features = make_features(10)
    pairs = sample_over_seg_pairs([list(range(10))], features)
    assert pairs == []


def test_no_pairs_when_no_regions():
    features = make_features(10)
    pairs = sample_over_seg_pairs([], features)
    assert pairs == []


def test_pairs_below_s_min_excluded():
    # Two regions with orthogonal (sim≈0) medoids
    features = make_features(20)
    features[:10] = np.array([1, 0, 0, 0], dtype=np.float32)
    features[10:] = np.array([0, 1, 0, 0], dtype=np.float32)
    regions = [list(range(10)), list(range(10, 20))]
    pairs = sample_over_seg_pairs(regions, features, k_max=5, s_min=0.9)
    assert pairs == []


def test_pairs_formed_for_similar_regions():
    d = 128
    base = np.ones(d, dtype=np.float32)
    base /= np.linalg.norm(base)
    features = np.tile(base, (20, 1))
    rng = np.random.default_rng(0)
    features += rng.standard_normal((20, d)).astype(np.float32) * 0.001
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    regions = [list(range(10)), list(range(10, 20))]
    pairs = sample_over_seg_pairs(regions, features, k_max=1, s_min=0.9)
    assert len(pairs) >= 1


def test_pairs_are_sorted_tuples():
    features = make_features(20)
    regions = [list(range(10)), list(range(10, 20))]
    pairs = sample_over_seg_pairs(regions, features, k_max=5, s_min=-1.0)
    for (i, j) in pairs:
        assert i < j
