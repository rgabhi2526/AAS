import numpy as np
import pandas as pd
import pytest
from src.data.dataset import WildlifeSubsetDataset


def make_dummy_df(n=10):
    return pd.DataFrame({
        'image_id': range(n),
        'path': [f'img_{i}.jpg' for i in range(n)],
        'identity': [i % 3 for i in range(n)],
        'split': ['train'] * n,
    })


def test_dataset_len():
    df = make_dummy_df(10)
    ds = WildlifeSubsetDataset(df, root='/tmp', transform=None)
    assert len(ds) == 10


def test_dataset_identities_length():
    df = make_dummy_df(10)
    ds = WildlifeSubsetDataset(df, root='/tmp', transform=None)
    assert len(ds.identities) == 10


def test_dataset_identities_values():
    df = make_dummy_df(10)
    ds = WildlifeSubsetDataset(df, root='/tmp', transform=None)
    assert set(ds.identities) == {0, 1, 2}


def test_dataset_identities_dtype():
    df = make_dummy_df(10)
    ds = WildlifeSubsetDataset(df, root='/tmp', transform=None)
    assert ds.identities.dtype == int
