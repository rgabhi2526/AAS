import numpy as np
import pytest
from src.oracle.gt_oracle import GTOracle


def test_same_identity_is_must_link():
    oracle = GTOracle(np.array([0, 0, 1, 1]))
    ml, cl = oracle.query([(0, 1)])
    assert (0, 1) in ml
    assert cl == []


def test_different_identity_is_cannot_link():
    oracle = GTOracle(np.array([0, 0, 1, 1]))
    ml, cl = oracle.query([(0, 2)])
    assert ml == []
    assert (0, 2) in cl


def test_mixed_pairs():
    oracle = GTOracle(np.array([0, 0, 1, 2]))
    ml, cl = oracle.query([(0, 1), (0, 2), (1, 3)])
    assert (0, 1) in ml
    assert (0, 2) in cl
    assert (1, 3) in cl


def test_empty_query():
    oracle = GTOracle(np.array([0, 1, 2]))
    ml, cl = oracle.query([])
    assert ml == [] and cl == []


def test_all_same_identity():
    oracle = GTOracle(np.array([5, 5, 5, 5]))
    ml, cl = oracle.query([(0, 1), (1, 2), (2, 3)])
    assert len(ml) == 3
    assert cl == []


def test_all_different_identity():
    oracle = GTOracle(np.array([0, 1, 2, 3]))
    ml, cl = oracle.query([(0, 1), (0, 2), (1, 3)])
    assert ml == []
    assert len(cl) == 3
