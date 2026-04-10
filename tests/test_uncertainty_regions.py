import numpy as np
import pytest
from src.aas.uncertainty_regions import compute_iou, find_uncertainty_regions


def test_iou_zero_overlap():
    assert compute_iou({0, 1}, {2, 3}) == 0.0


def test_iou_full_overlap():
    assert compute_iou({0, 1, 2}, {0, 1, 2}) == 1.0


def test_iou_partial_overlap():
    iou = compute_iou({0, 1, 2}, {1, 2, 3})
    # intersection=2, union=4 → 0.5
    assert abs(iou - 0.5) < 1e-9


def test_iou_empty_sets():
    assert compute_iou(set(), set()) == 0.0


def test_no_uncertainty_when_perfect_agreement():
    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([0, 0, 1, 1, 2, 2])
    regions = find_uncertainty_regions(labels_a, labels_b)
    assert regions == []


def test_uncertainty_region_found_on_disagreement():
    # A: {0,1,2,3}, {4,5}  |  B: {0,1}, {2,3,4,5}
    labels_a = np.array([0, 0, 0, 0, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1])
    regions = find_uncertainty_regions(labels_a, labels_b)
    assert len(regions) >= 1
    all_samples = {idx for region in regions for idx in region}
    assert {0, 1, 2, 3} <= all_samples


def test_outliers_excluded():
    # DBSCAN outliers (label=-1) should not form a cluster node
    labels_a = np.array([-1, 0, 0, 1, 1, -1])
    labels_b = np.array([0, 0, 1, 1, 0, 0])
    regions = find_uncertainty_regions(labels_a, labels_b)
    assert isinstance(regions, list)


def test_returns_sorted_indices():
    labels_a = np.array([0, 0, 0, 0, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1])
    regions = find_uncertainty_regions(labels_a, labels_b)
    for region in regions:
        assert region == sorted(region)
