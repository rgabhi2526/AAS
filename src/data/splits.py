"""Gallery / query / held-out split logic following the AAS paper's protocol."""
import numpy as np
import pandas as pd
from typing import Optional, Tuple


def make_splits(
    df: pd.DataFrame,
    held_out_fraction: float = 0.2,
    max_exemplars: int = 5,
    embeddings: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a single-dataset DataFrame into gallery, query, and held-out sets.

    Protocol (exactly as AAS paper):
    - Query:     full test split (open-set; may contain unseen individuals)
    - Held-out:  20% of training identities, absent from gallery
    - Gallery:   up to 5 exemplars per identity from the remaining 80%,
                 chosen by cosine similarity to the identity centroid in
                 embedding space (requires `embeddings`).
                 Falls back to random selection when embeddings is None.

    Args:
        df:                 DataFrame with ['image_id', 'path', 'identity', 'split']
        held_out_fraction:  fraction of training identities to withhold
        max_exemplars:      max gallery images per identity
        embeddings:         (n_train, d) L2-normalised features aligned with the
                            train subset of `df` (row order must match).
                            Pass None for random exemplar selection.
        seed:               RNG seed (add run index here for 4-run averaging)

    Returns:
        gallery_df, query_df, held_out_df  — each a reset-index DataFrame
    """
    rng = np.random.default_rng(seed)

    train_df = df[df['split'] == 'train'].copy().reset_index(drop=True)
    query_df = df[df['split'] == 'test'].copy().reset_index(drop=True)

    train_ids = train_df['identity'].unique()
    n_held_out = max(1, int(len(train_ids) * held_out_fraction))

    held_out_ids = rng.choice(train_ids, size=n_held_out, replace=False)
    gallery_ids = np.setdiff1d(train_ids, held_out_ids)

    held_out_df = train_df[train_df['identity'].isin(held_out_ids)].reset_index(drop=True)
    gallery_pool_df = train_df[train_df['identity'].isin(gallery_ids)].reset_index(drop=True)

    if embeddings is not None:
        gallery_df = _select_by_centroid_similarity(gallery_pool_df, embeddings, max_exemplars)
    else:
        gallery_df = _random_exemplars(gallery_pool_df, max_exemplars, seed)

    return gallery_df, query_df, held_out_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_by_centroid_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    max_exemplars: int,
) -> pd.DataFrame:
    """
    For each identity, pick up to max_exemplars images closest to the
    L2-normalised centroid in the given embedding space.

    `embeddings` must be row-aligned with `df` (same reset index).
    """
    selected = []
    for _, group in df.groupby('identity'):
        idxs = group.index.tolist()
        embs = embeddings[idxs]                              # (m, d)
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs_norm = embs / (norms + 1e-8)
        centroid = embs_norm.mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-8
        sims = embs_norm @ centroid                          # (m,)
        top_k = np.argsort(sims)[::-1][:max_exemplars]
        selected.append(group.iloc[top_k])
    return pd.concat(selected).reset_index(drop=True)


def _random_exemplars(df: pd.DataFrame, max_exemplars: int, seed: int) -> pd.DataFrame:
    """Random fallback: sample up to max_exemplars per identity."""
    selected = []
    for _, group in df.groupby('identity'):
        selected.append(group.sample(min(max_exemplars, len(group)), random_state=seed))
    return pd.concat(selected).reset_index(drop=True)
