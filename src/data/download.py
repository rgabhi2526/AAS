"""Download WildlifeReID-10k via Kaggle API (requires ~/.kaggle/kaggle.json)."""
import json
import os
import zipfile

import pandas as pd
import requests

KAGGLE_API_URL = (
    "https://www.kaggle.com/api/v1/datasets/download"
    "/wildlifedatasets/wildlifereid-10k"
)


def load_dataset_list(dataset_txt: str) -> list:
    """Read dataset names from a text file (one per line, # = comment)."""
    with open(dataset_txt) as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


def download(root: str, dataset_txt: str | None = None) -> pd.DataFrame:
    """
    Download WildlifeReID-10k to `root` via the Kaggle API.
    Requires Kaggle credentials at ~/.kaggle/kaggle.json.

    Args:
        root:        Directory where data will be saved (e.g. 'data/raw').
        dataset_txt: Optional path to dataset.txt; if given, the returned
                     DataFrame is filtered to those datasets only.

    Returns:
        metadata DataFrame (optionally filtered).
    """
    os.makedirs(root, exist_ok=True)

    zip_path = os.path.join(root, "wildlifereid-10k.zip")
    if not os.path.exists(zip_path):
        creds_path = os.path.expanduser("~/.kaggle/kaggle.json")
        with open(creds_path) as f:
            creds = json.load(f)

        print("Downloading WildlifeReID-10k from Kaggle …")
        resp = requests.get(
            KAGGLE_API_URL,
            auth=(creds["username"], creds["key"]),
            stream=True,
        )
        resp.raise_for_status()
        with open(zip_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                fh.write(chunk)
        print("Download complete. Extracting …")
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(root)
        print("Extraction complete.")
    else:
        print(f"Archive already present at {zip_path}, skipping download.")

    return load_metadata(root, dataset_txt=dataset_txt)


def load_metadata(root: str, dataset_txt: str | None = None) -> pd.DataFrame:
    """
    Load metadata.csv from `root` without re-downloading.

    Args:
        root:        Directory that contains metadata.csv.
        dataset_txt: Optional path to dataset.txt for filtering.

    Returns:
        metadata DataFrame (optionally filtered).
    """
    metadata_path = os.path.join(root, "metadata.csv")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"metadata.csv not found in {root}. "
            "Run download() first or point root at the extracted directory."
        )
    df = pd.read_csv(metadata_path)

    if dataset_txt is not None:
        datasets = load_dataset_list(dataset_txt)
        df = _filter_datasets(df, datasets)

    return df


def _filter_datasets(df: pd.DataFrame, datasets: list) -> pd.DataFrame:
    """Keep only rows whose 'dataset' column is in `datasets`."""
    if "dataset" not in df.columns:
        raise ValueError("metadata.csv is missing a 'dataset' column.")
    filtered = df[df["dataset"].isin(datasets)].copy()
    missing = set(datasets) - set(filtered["dataset"].unique())
    if missing:
        print(f"WARNING: datasets not found in metadata: {sorted(missing)}")
    return filtered


if __name__ == "__main__":
    import sys

    root = sys.argv[1] if len(sys.argv) > 1 else "data/raw"
    txt = sys.argv[2] if len(sys.argv) > 2 else "dataset.txt"
    df = download(root, dataset_txt=txt)
    print(f"Total records after filtering: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\nPer-dataset counts:")
    print(df["dataset"].value_counts().to_string())
