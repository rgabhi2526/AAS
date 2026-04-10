import numpy as np
import pandas as pd
import pytest
from src.data.splits import make_splits


def make_df(n_train_ids=10, n_test_ids=5, images_per_id=8):
    rows = []
    for identity in range(n_train_ids):
        for _ in range(images_per_id):
            rows.append({'image_id': len(rows), 'path': f'img_{len(rows)}.jpg',
                         'identity': identity, 'split': 'train'})
    for identity in range(n_train_ids, n_train_ids + n_test_ids):
        for _ in range(images_per_id):
            rows.append({'image_id': len(rows), 'path': f'img_{len(rows)}.jpg',
                         'identity': identity, 'split': 'test'})
    return pd.DataFrame(rows)


def test_query_is_full_test_set():
    df = make_df()
    _, query_df, _ = make_splits(df, seed=0)
    assert set(query_df['split'].unique()) == {'test'}
    assert len(query_df) == 5 * 8


def test_held_out_fraction():
    df = make_df(n_train_ids=10)
    _, _, held_out_df = make_splits(df, held_out_fraction=0.2, seed=0)
    assert len(held_out_df['identity'].unique()) == 2  # 20% of 10


def test_gallery_max_exemplars():
    df = make_df(n_train_ids=10, images_per_id=8)
    gallery_df, _, _ = make_splits(df, max_exemplars=5, seed=0)
    for _, group in gallery_df.groupby('identity'):
        assert len(group) <= 5


def test_no_overlap_gallery_held_out():
    df = make_df()
    gallery_df, _, held_out_df = make_splits(df, seed=0)
    gallery_ids = set(gallery_df['identity'].unique())
    held_out_ids = set(held_out_df['identity'].unique())
    assert gallery_ids.isdisjoint(held_out_ids)


def test_gallery_ids_subset_of_train_ids():
    df = make_df()
    gallery_df, _, _ = make_splits(df, seed=0)
    train_ids = set(df[df['split'] == 'train']['identity'].unique())
    gallery_ids = set(gallery_df['identity'].unique())
    assert gallery_ids <= train_ids


def test_random_exemplar_fallback():
    # Without embeddings, should still produce valid gallery
    df = make_df(n_train_ids=5, images_per_id=10)
    gallery_df, _, _ = make_splits(df, max_exemplars=3, embeddings=None, seed=0)
    for _, group in gallery_df.groupby('identity'):
        assert len(group) <= 3
