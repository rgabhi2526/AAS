import numpy as np
import pytest
from src.aas.np3 import refine_labels


def test_no_constraints_unchanged():
    labels = np.array([0, 1, 2])
    refined = refine_labels(labels, [], [])
    np.testing.assert_array_equal(refined, labels)


def test_output_length_unchanged():
    labels = np.array([0, 0, 1, 1, 2])
    refined = refine_labels(labels, [(0, 1)], [(0, 2)])
    assert len(refined) == len(labels)


def test_must_link_satisfied():
    labels = np.array([0, 1, 2, 3])
    refined = refine_labels(labels, [(0, 1)], [])
    assert refined[0] == refined[1]


def test_cannot_link_satisfied_within_cluster():
    # All 4 in same cluster initially; CL should split 0 and 1
    labels = np.array([0, 0, 0, 0])
    refined = refine_labels(labels, [], [(0, 1)])
    assert refined[0] != refined[1]


def test_ml_and_cl_combined():
    # ML: 0↔1 must be together; CL: 0↔2 must be apart
    labels = np.array([0, 0, 0, 1])
    refined = refine_labels(labels, [(0, 1)], [(0, 2)])
    assert refined[0] == refined[1], "ML: 0 and 1 in same cluster"
    assert refined[0] != refined[2], "CL: 0 and 2 in different clusters"


def test_multiple_must_links_transitive():
    # 0↔1 and 1↔2 → all three must be together
    labels = np.array([0, 1, 2, 3])
    refined = refine_labels(labels, [(0, 1), (1, 2)], [])
    assert refined[0] == refined[1] == refined[2]


def test_cannot_link_across_already_separated_clusters():
    # CL on samples already in different clusters — should be a no-op
    labels = np.array([0, 0, 1, 1])
    refined = refine_labels(labels, [], [(0, 2)])
    # 0 and 2 are already in different clusters; labels should not change
    assert refined[0] == refined[1]   # 0 and 1 still together
    assert refined[2] == refined[3]   # 2 and 3 still together
    assert refined[0] != refined[2]   # still in different clusters


def test_output_is_numpy_array():
    labels = np.array([0, 0, 1])
    refined = refine_labels(labels, [], [])
    assert isinstance(refined, np.ndarray)
