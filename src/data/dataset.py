"""Unified PyTorch dataset wrapping WildlifeReID-10k subsets."""
import os
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class WildlifeSubsetDataset(Dataset):
    """
    PyTorch Dataset for a single-species subset of WildlifeReID-10k.

    Each __getitem__ returns: (image_tensor, identity_label_int, sample_index)

    Args:
        df:        DataFrame with at minimum columns ['path', 'identity'].
                   'identity' may be a string (e.g. 'BelugaID_5') or an int.
                   String identities are encoded to consecutive integers
                   automatically; the mapping is stored in `self.identity_map`.
        root:      Root directory prepended to each 'path' value.
        transform: torchvision transform applied to PIL images.
    """

    # Datasets with at most this many images get automatic in-memory caching
    # to avoid repeated disk I/O.  Above this threshold caching is skipped
    # to prevent OOM (e.g. 140k images on Colab).
    _CACHE_THRESHOLD: int = 10_000

    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        transform: Optional[Callable] = None,
        cache_images: Optional[bool] = None,
    ):
        self.df = df.reset_index(drop=True).copy()
        self.root = root
        self.transform = transform
        self.identity_map: dict = {}  # int_label -> original_string (if encoded)

        # Auto-gate caching: on for small datasets, off for large ones
        if cache_images is None:
            cache_images = len(self.df) <= self._CACHE_THRESHOLD
        self._cache_images = cache_images
        self._image_cache: dict = {}

        # Encode string identities to consecutive integers
        if not pd.api.types.is_integer_dtype(self.df["identity"]):
            cats = pd.Categorical(self.df["identity"])
            self.df["identity"] = cats.codes
            self.identity_map = dict(enumerate(cats.categories))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        if self._cache_images and idx in self._image_cache:
            image = self._image_cache[idx].copy()
        else:
            img_path = os.path.join(self.root, row["path"])
            image = Image.open(img_path).convert("RGB")
            if self._cache_images:
                self._image_cache[idx] = image.copy()
        if self.transform:
            image = self.transform(image)
        return image, int(row["identity"]), idx

    @property
    def identities(self) -> np.ndarray:
        """(n,) integer array of identity labels."""
        return self.df["identity"].values.astype(int)
