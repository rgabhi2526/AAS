"""Download WildlifeReID-10k (requires ~/.kaggle/kaggle.json credentials)."""
import os
import pandas as pd
from wildlife_datasets.datasets import WildlifeReID10k

# 13 datasets used in AAS paper (Table 1).
# NOTE: After downloading, verify these match the 'dataset' column in metadata.csv.
# Update names if they differ — run: python3 src/data/download.py data/raw
PAPER_DATASETS_13 = [
    'BelugaID',
    'CowDataset',
    'FriesianCattle2015',
    'GiraffeZebraID',
    'HyenaID2022',
    'IPanda50',
    'LeopardID2022',
    'MacaqueFaces',
    'NyalaData',
    'SealID',
    'WhaleSharkID',
    'HappyWhale',
    'ELPephants',
]


def download(root: str) -> WildlifeReID10k:
    """
    Download WildlifeReID-10k to `root` directory.
    Requires Kaggle credentials at ~/.kaggle/kaggle.json.

    Args:
        root: directory where data will be saved (e.g. 'data/raw')

    Returns:
        WildlifeReID10k dataset object with .df DataFrame
    """
    os.makedirs(root, exist_ok=True)
    dataset = WildlifeReID10k(root)
    return dataset


def load_metadata(root: str) -> pd.DataFrame:
    """Load and return the full metadata DataFrame without checking file existence."""
    dataset = WildlifeReID10k(root, check_files=False)
    return dataset.df


def filter_paper_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata to only the 13 datasets used in the AAS paper."""
    if 'dataset' not in df.columns:
        raise ValueError(
            "metadata.csv missing 'dataset' column. "
            "Check actual column name and update PAPER_DATASETS_13."
        )
    filtered = df[df['dataset'].isin(PAPER_DATASETS_13)].copy()
    missing = set(PAPER_DATASETS_13) - set(filtered['dataset'].unique())
    if missing:
        print(f"WARNING: datasets not found in metadata: {missing}")
    return filtered


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    print(f"Downloading WildlifeReID-10k to {root} ...")
    ds = download(root)
    print(f"Total records: {len(ds.df)}")
    print(f"Columns: {list(ds.df.columns)}")
    if 'dataset' in ds.df.columns:
        print(f"\nDatasets present:")
        print(ds.df['dataset'].value_counts().to_string())
