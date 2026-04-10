import numpy as np
import pytest
from src.eval.metrics import compute_metrics, _average_precision, _mean_inverse_negative_penalty


# ---- AP ----

def test_ap_perfect_retrieval():
    matches = np.array([1, 1, 1, 0, 0])
    assert abs(_average_precision(matches) - 1.0) < 1e-6


def test_ap_no_positives():
    assert _average_precision(np.array([0, 0, 0])) == 0.0


def test_ap_single_positive_at_rank1():
    matches = np.array([1, 0, 0, 0])
    assert abs(_average_precision(matches) - 1.0) < 1e-6


def test_ap_single_positive_at_rank2():
    # AP = (1/2) / 1 = 0.5
    matches = np.array([0, 1, 0, 0])
    assert abs(_average_precision(matches) - 0.5) < 1e-6


# ---- mINP ----

def test_minp_last_positive_rank1():
    matches = np.array([1, 0, 0, 0])
    assert abs(_mean_inverse_negative_penalty(matches) - 1.0) < 1e-6


def test_minp_last_positive_rank4():
    matches = np.array([0, 0, 0, 1])
    assert abs(_mean_inverse_negative_penalty(matches) - 0.25) < 1e-6


def test_minp_no_positives():
    assert _mean_inverse_negative_penalty(np.array([0, 0, 0])) == 0.0


# ---- compute_metrics ----

def make_feats(n, d=64, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, d)).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f


def test_metrics_keys_present():
    qf = make_feats(5)
    gf = make_feats(20, seed=1)
    ql = np.array([0, 1, 2, 3, 4])
    gl = np.arange(20) % 5
    m = compute_metrics(qf, gf, ql, gl)
    for key in ['mAP', 'mINP', 'top1', 'top3', 'top5', 'top10']:
        assert key in m


def test_metrics_range():
    qf = make_feats(5)
    gf = make_feats(20, seed=1)
    ql = np.array([0, 1, 2, 3, 4])
    gl = np.arange(20) % 5
    m = compute_metrics(qf, gf, ql, gl)
    for v in m.values():
        assert 0.0 <= v <= 1.0


def test_baks_aucroc_present_when_known_mask_given():
    qf = make_feats(6)
    gf = make_feats(10, seed=1)
    ql = np.array([0, 1, 2, 3, 4, 5])
    gl = np.arange(10) % 5   # IDs 5 not in gallery
    known = np.array([True, True, True, True, True, False])
    m = compute_metrics(qf, gf, ql, gl, query_is_known=known)
    assert 'BAKS' in m
    assert 'AUCROC' in m


def test_perfect_retrieval_top1():
    # Each query is its own gallery item (same feature vector)
    n = 5
    feats = make_feats(n)
    m = compute_metrics(feats, feats, np.arange(n), np.arange(n))
    assert abs(m['top1'] - 1.0) < 1e-6
    assert abs(m['mAP'] - 1.0) < 1e-6
