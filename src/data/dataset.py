"""Unified PyTorch dataset wrapping WildlifeReID-10k subsets."""
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable


class WildlifeSubsetDataset(Dataset):
    """
    PyTorch Dataset for a single-species subset of WildlifeReID-10k.

    Each __getitem__ returns: (image_tensor, identity_label, sample_index)

    Args:
        df:        DataFrame with columns ['image_id', 'path', 'identity']
        root:      root directory prepended to each 'path' value
        transform: torchvision transform applied to PIL images
    """

    def __init__(
        self,
        df: pd.DataFrame,
        root: str,
        transform: Optional[Callable] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root, row['path'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(row['identity']), idx

    @property
    def identities(self) -> np.ndarray:
        """(n,) integer array of identity labels."""
        return self.df['identity'].values.astype(int)
