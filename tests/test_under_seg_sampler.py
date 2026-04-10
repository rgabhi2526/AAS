import numpy as np
import pytest
from src.aas.under_seg_sampler import sample_under_seg_pairs


def make_features(n=8, d=4, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, d)).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f


def test_returns_list():
    features = make_features(8)
    labels_a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    regions = [list(range(8))]
    pairs = sample_under_seg_pairs(regions, labels_a, labels_b, features)
    assert isinstance(pairs, list)


def test_perfect_agreement_gives_no_pairs():
    features = make_features(6)
    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([0, 0, 1, 1, 2, 2])
    regions = [list(range(6))]
    pairs = sample_under_seg_pairs(regions, labels_a, labels_b, features)
    assert pairs == []


def test_empty_regions_gives_no_pairs():
    features = make_features(8)
    labels_a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    pairs = sample_under_seg_pairs([], labels_a, labels_b, features)
    assert pairs == []


def test_pairs_are_sorted_tuples():
    features = make_features(8)
    labels_a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    regions = [list(range(8))]
    pairs = sample_under_seg_pairs(regions, labels_a, labels_b, features)
    for (i, j) in pairs:
        assert i < j, "pairs should be sorted (i < j)"


def test_no_self_pairs():
    features = make_features(8)
    labels_a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    regions = [list(range(8))]
    pairs = sample_under_seg_pairs(regions, labels_a, labels_b, features)
    for (i, j) in pairs:
        assert i != j
