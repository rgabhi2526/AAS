"""Unified PyTorch dataset wrapping WildlifeReID-10k subsets."""
import os
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


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
        preload:   If True, load all images into RAM during __init__
                   (before DataLoader forks workers, so they inherit via
                   copy-on-write with zero extra memory).
                   If None (default), auto-enable for datasets ≤ 10 000 images.
        shared_images:
                   Optional pre-loaded list of PIL images (same length as df).
                   When provided, skips disk I/O entirely — use this to share
                   one set of preloaded images across eval/train datasets that
                   differ only by transform.
    """

    _PRELOAD_THRESHOLD: int = 10_000

    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        transform: Optional[Callable] = None,
        preload: Optional[bool] = None,
        shared_images: Optional[List[Image.Image]] = None,
    ):
        df = df.reset_index(drop=True).copy()
        self.root = root
        self.transform = transform
        self.identity_map: dict = {}  # int_label -> original_string (if encoded)

        # Encode string identities to consecutive integers
        if not pd.api.types.is_integer_dtype(df["identity"]):
            cats = pd.Categorical(df["identity"])
            df["identity"] = cats.codes
            self.identity_map = dict(enumerate(cats.categories))

        # ── Pre-extract arrays (eliminates df.iloc overhead in __getitem__) ──
        self._paths: list = [
            os.path.join(root, p) for p in df["path"].values
        ]
        self._labels: np.ndarray = df["identity"].values.astype(np.int64)
        self._len: int = len(df)

        # Keep df for any external inspection
        self.df = df

        # ── Preload / share images ──────────────────────────────────────────
        if shared_images is not None:
            # Reuse an existing preloaded list (zero extra RAM or I/O)
            assert len(shared_images) == self._len, (
                f"shared_images length {len(shared_images)} != dataset length {self._len}"
            )
            self._images = shared_images
            self._preloaded = True
        else:
            if preload is None:
                preload = self._len <= self._PRELOAD_THRESHOLD
            self._preloaded = preload
            self._images: list = []

            if self._preloaded:
                # Resize during preload to avoid storing full-resolution images
                # in RAM (a 3000×2000 photo = 18 MB; at 256×128 = 98 KB).
                # Both train and val transforms start with Resize((256, 128)),
                # so the transform's Resize becomes a no-op.
                _target = (256, 128)  # (height, width) — Re-ID standard
                self._images = [
                    Image.open(p).convert("RGB").resize(
                        (_target[1], _target[0]),  # PIL takes (w, h)
                        Image.BILINEAR,
                    )
                    for p in tqdm(self._paths, desc="Preloading images",
                                  disable=self._len < 50)
                ]

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        if self._preloaded:
            # Copy-on-write read from preloaded list — no disk I/O,
            # no .copy() needed because transforms create new tensors.
            image = self._images[idx]
        else:
            image = Image.open(self._paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, int(self._labels[idx]), idx

    @property
    def identities(self) -> np.ndarray:
        """(n,) integer array of identity labels."""
        return self._labels
