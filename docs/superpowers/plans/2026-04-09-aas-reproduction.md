# AAS Paper Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the AAS (Ambiguity-Aware Sampling) paper's Table 1 results — 56.14% mAP on 13 WildlifeReID-10k datasets — using a forked SpCL as the base USL training method.

**Architecture:** AAS is implemented as a set of independent modules (clustering wrappers, sampler, NP3, oracle, metrics) that hook into SpCL's training loop every 10 epochs. Each module has one responsibility and exposes a clean interface. SpCL is cloned as a git submodule and modified minimally.

**Tech Stack:** Python 3.10+, PyTorch, timm, wildlife-datasets, scikit-learn, finch-clust, networkx, scipy, pandas, numpy, matplotlib

---

## File Map

**New files to create:**
```
requirements.txt
.gitignore
src/__init__.py
src/data/__init__.py
src/data/download.py
src/data/dataset.py
src/data/splits.py
src/data/transforms.py
src/data/features.py
src/clustering/__init__.py
src/clustering/dbscan.py
src/clustering/finch.py
src/aas/__init__.py
src/aas/uncertainty_regions.py
src/aas/over_seg_sampler.py
src/aas/under_seg_sampler.py
src/aas/sampler.py
src/aas/np3.py
src/oracle/__init__.py
src/oracle/gt_oracle.py
src/eval/__init__.py
src/eval/metrics.py
experiments/configs/aas.yaml
experiments/run_aas.py
tests/test_uncertainty_regions.py
tests/test_over_seg_sampler.py
tests/test_under_seg_sampler.py
tests/test_np3.py
tests/test_oracle.py
tests/test_metrics.py
tests/test_splits.py
notebooks/reproduce_table1.ipynb
notebooks/budget_curves.ipynb
```

**SpCL fork (cloned as `third_party/SpCL/`):**
```
third_party/SpCL/              # forked from https://github.com/yxgeee/SpCL
experiments/train_aas.py       # our modified training script (wraps SpCL)
```

---

## Phase 1: Setup & Data

### Task 1: Repo setup — requirements, gitignore, package skeleton

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `src/__init__.py`, `src/data/__init__.py`, `src/clustering/__init__.py`, `src/aas/__init__.py`, `src/oracle/__init__.py`, `src/eval/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1: Create requirements.txt**

```
torch>=2.0.0
torchvision>=0.15.0
timm>=0.9.0
wildlife-datasets>=0.3.0
scikit-learn>=1.3.0
finch-clust>=0.1.5
networkx>=3.1
scipy>=1.11.0
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
Pillow>=9.0.0
pyyaml>=6.0
tqdm>=4.65.0
pytest>=7.4.0
```

- [ ] **Step 2: Create .gitignore**

```
data/raw/
data/processed/
experiments/results/
*.pyc
__pycache__/
.env
*.egg-info/
dist/
build/
.DS_Store
third_party/SpCL/logs/
third_party/SpCL/examples/logs/
```

- [ ] **Step 3: Create empty `__init__.py` files**

```bash
touch src/__init__.py src/data/__init__.py src/clustering/__init__.py \
      src/aas/__init__.py src/oracle/__init__.py src/eval/__init__.py \
      tests/__init__.py
mkdir -p data/raw data/processed data/splits experiments/configs \
         experiments/results notebooks third_party
```

- [ ] **Step 4: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error (ignore streamlit pillow conflict if present).

- [ ] **Step 5: Commit**

```bash
git init
git add requirements.txt .gitignore src/ tests/ data/.gitkeep experiments/ notebooks/ third_party/
git commit -m "chore: project skeleton and dependencies"
```

---

### Task 2: Clone SpCL fork

**Files:**
- Create: `third_party/SpCL/` (git submodule)

- [ ] **Step 1: Clone SpCL**

```bash
cd third_party
git clone https://github.com/yxgeee/SpCL.git
cd ..
```

- [ ] **Step 2: Verify SpCL structure**

```bash
ls third_party/SpCL/
```

Expected output includes: `spcl/`, `examples/`, `setup.py`, `README.md`

- [ ] **Step 3: Install SpCL in development mode**

```bash
pip install -e third_party/SpCL/
```

- [ ] **Step 4: Verify SpCL import**

```bash
python3 -c "import spcl; print('SpCL OK')"
```

Expected: `SpCL OK`

- [ ] **Step 5: Commit**

```bash
git add third_party/SpCL/
git commit -m "chore: add SpCL as third-party dependency"
```

---

### Task 3: Data download script

**Files:**
- Create: `src/data/download.py`

Context: WildlifeReID-10k is hosted on Kaggle. The `wildlife-datasets` package downloads it via `kaggle datasets download wildlifedatasets/wildlifereid-10k`. Requires `~/.kaggle/kaggle.json` credentials. The metadata.csv has columns: `path`, `identity`, `split`, `dataset` (the source sub-dataset name).

- [ ] **Step 1: Create `src/data/download.py`**

```python
"""Download WildlifeReID-10k (requires ~/.kaggle/kaggle.json credentials)."""
import os
import pandas as pd
from wildlife_datasets.datasets import WildlifeReID10k

# 13 datasets used in AAS paper (Table 1).
# NOTE: Confirm exact names from paper supplementary.
# These names match the 'dataset' column in WildlifeReID-10k metadata.csv.
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
    'Elephants',
]


def download(root: str) -> WildlifeReID10k:
    """
    Download WildlifeReID-10k to `root` directory.

    Args:
        root: directory where data will be saved (e.g. 'data/raw')

    Returns:
        WildlifeReID10k dataset object with .df DataFrame
    """
    os.makedirs(root, exist_ok=True)
    dataset = WildlifeReID10k(root)
    return dataset


def load_metadata(root: str) -> pd.DataFrame:
    """Load and return the full metadata DataFrame."""
    dataset = WildlifeReID10k(root, check_files=False)
    return dataset.df


def filter_paper_datasets(df: pd.DataFrame) -> pd.DataFrame:
    """Filter metadata to only the 13 datasets used in the paper."""
    if 'dataset' not in df.columns:
        raise ValueError(
            "metadata.csv missing 'dataset' column. "
            "Check the column name and update PAPER_DATASETS_13."
        )
    return df[df['dataset'].isin(PAPER_DATASETS_13)].copy()


if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else 'data/raw'
    print(f"Downloading WildlifeReID-10k to {root}...")
    ds = download(root)
    print(f"Total records: {len(ds.df)}")
    print(f"Columns: {list(ds.df.columns)}")
    if 'dataset' in ds.df.columns:
        print(f"Datasets present: {ds.df['dataset'].unique().tolist()}")
```

- [ ] **Step 2: Commit**

```bash
git add src/data/download.py
git commit -m "feat: add WildlifeReID-10k download script"
```

---

### Task 4: Dataset loader

**Files:**
- Create: `src/data/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_dataset.py
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


def test_dataset_identities():
    df = make_dummy_df(10)
    ds = WildlifeSubsetDataset(df, root='/tmp', transform=None)
    ids = ds.identities
    assert len(ids) == 10
    assert set(ids) == {0, 1, 2}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_dataset.py -v
```

Expected: FAIL with `ImportError: cannot import name 'WildlifeSubsetDataset'`

- [ ] **Step 3: Create `src/data/dataset.py`**

```python
"""Unified dataset class wrapping WildlifeReID-10k subsets."""
import os
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import Optional, Callable


class WildlifeSubsetDataset(Dataset):
    """
    PyTorch Dataset for a single-species subset of WildlifeReID-10k.

    Each item returns: (image_tensor, identity_label, sample_index)

    Args:
        df: DataFrame with columns ['image_id', 'path', 'identity']
        root: root directory where images are stored
        transform: torchvision transform applied to PIL images
    """

    def __init__(self, df: pd.DataFrame, root: str, transform: Optional[Callable] = None):
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
        """(n,) array of integer identity labels."""
        return self.df['identity'].values.astype(int)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_dataset.py -v
```

Expected: PASS (image loading skipped since path doesn't exist in test — __getitem__ not called)

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset.py tests/test_dataset.py
git commit -m "feat: add WildlifeSubsetDataset"
```

---

### Task 5: Gallery/query split logic

**Files:**
- Create: `src/data/splits.py`
- Create: `tests/test_splits.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_splits.py
import numpy as np
import pandas as pd
import pytest
from src.data.splits import make_splits


def make_df(n_train_ids=10, n_test_ids=5, images_per_id=8):
    rows = []
    for identity in range(n_train_ids):
        for i in range(images_per_id):
            rows.append({'image_id': len(rows), 'path': f'img_{len(rows)}.jpg',
                         'identity': identity, 'split': 'train'})
    for identity in range(n_train_ids, n_train_ids + n_test_ids):
        for i in range(images_per_id):
            rows.append({'image_id': len(rows), 'path': f'img_{len(rows)}.jpg',
                         'identity': identity, 'split': 'test'})
    return pd.DataFrame(rows)


def test_query_is_full_test_set():
    df = make_df()
    gallery_df, query_df, held_out_df = make_splits(df, held_out_fraction=0.2, seed=0)
    assert set(query_df['split'].unique()) == {'test'}
    assert len(query_df) == 5 * 8


def test_held_out_fraction():
    df = make_df(n_train_ids=10)
    gallery_df, query_df, held_out_df = make_splits(df, held_out_fraction=0.2, seed=0)
    held_out_ids = held_out_df['identity'].unique()
    assert len(held_out_ids) == 2  # 20% of 10


def test_gallery_max_exemplars():
    df = make_df(n_train_ids=10, images_per_id=8)
    gallery_df, _, _ = make_splits(df, held_out_fraction=0.2, max_exemplars=5, seed=0)
    for identity, group in gallery_df.groupby('identity'):
        assert len(group) <= 5


def test_no_overlap_gallery_held_out():
    df = make_df()
    gallery_df, _, held_out_df = make_splits(df, seed=0)
    gallery_ids = set(gallery_df['identity'].unique())
    held_out_ids = set(held_out_df['identity'].unique())
    assert gallery_ids.isdisjoint(held_out_ids)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_splits.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/data/splits.py`**

```python
"""Gallery/query/held-out split logic following the AAS paper's protocol."""
import numpy as np
import pandas as pd
from typing import Tuple, Optional


def make_splits(
    df: pd.DataFrame,
    held_out_fraction: float = 0.2,
    max_exemplars: int = 5,
    embeddings: Optional[np.ndarray] = None,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a dataset into gallery, query, and held-out sets.

    Protocol (exactly as AAS paper):
    - Query:     full test set (open-set; may include unseen individuals)
    - Held-out:  20% of training identities, absent from gallery
    - Gallery:   up to 5 exemplars per identity from 80% training identities,
                 selected by cosine similarity to identity centroid in embedding space.
                 Falls back to random selection when embeddings are None.

    Args:
        df:                 DataFrame with columns ['image_id', 'path', 'identity', 'split']
        held_out_fraction:  fraction of training identities to hold out
        max_exemplars:      max gallery images per identity
        embeddings:         (n_train, d) L2-normalized features for gallery selection.
                            If None, random exemplar selection is used.
        seed:               RNG seed for reproducibility

    Returns:
        gallery_df, query_df, held_out_df
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


def _select_by_centroid_similarity(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    max_exemplars: int,
) -> pd.DataFrame:
    """Select up to max_exemplars per identity closest to the identity centroid."""
    selected = []
    for identity, group in df.groupby('identity'):
        idxs = group.index.tolist()
        embs = embeddings[idxs]                             # (m, d)
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
    return (
        df.groupby('identity')
        .apply(lambda g: g.sample(min(max_exemplars, len(g)), random_state=seed))
        .reset_index(drop=True)
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_splits.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/data/splits.py tests/test_splits.py
git commit -m "feat: add gallery/query/held-out split logic"
```

---

### Task 6: Transforms and feature extractor

**Files:**
- Create: `src/data/transforms.py`
- Create: `src/data/features.py`

- [ ] **Step 1: Create `src/data/transforms.py`**

```python
"""Standard Re-ID image preprocessing."""
import torchvision.transforms as T


def get_transforms(split: str = 'train') -> T.Compose:
    """
    Returns torchvision transforms for Re-ID.
    Resize to 256x128, ImageNet normalization.
    Training adds random horizontal flip.

    Args:
        split: 'train' or 'val'/'test'
    """
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    if split == 'train':
        return T.Compose([
            T.Resize((256, 128)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        normalize,
    ])
```

- [ ] **Step 2: Create `src/data/features.py`**

```python
"""Feature extraction using ResNet-50 or MegaDescriptor backbones."""
import torch
import timm
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm


def extract_features(
    dataset,
    backbone: str = 'resnet50',
    batch_size: int = 64,
    device: str = 'cuda',
    num_workers: int = 4,
) -> np.ndarray:
    """
    Extract L2-normalized features from all images in a dataset.

    Args:
        dataset:     PyTorch dataset returning (image, label, idx)
        backbone:    'resnet50' or 'megadescriptor'
        batch_size:  inference batch size
        device:      'cuda' or 'cpu'
        num_workers: DataLoader workers

    Returns:
        features: (n, d) float32 array of L2-normalized features
    """
    if backbone == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=0)
    elif backbone == 'megadescriptor':
        model = timm.create_model(
            'hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Use 'resnet50' or 'megadescriptor'.")

    model = model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)

    all_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Extracting features ({backbone})'):
            images = batch[0].to(device)
            feats = model(images)
            feats = torch.nn.functional.normalize(feats, dim=1)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0).astype(np.float32)
```

- [ ] **Step 3: Commit**

```bash
git add src/data/transforms.py src/data/features.py
git commit -m "feat: add transforms and feature extractor"
```

---

## Phase 2: AAS Core Modules

### Task 7: DBSCAN wrapper

**Files:**
- Create: `src/clustering/dbscan.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_clustering.py
import numpy as np
import pytest
from src.clustering.dbscan import fit as dbscan_fit
from src.clustering.finch import fit as finch_fit


def make_clustered_features(n_clusters=3, n_per_cluster=20, d=128, seed=0):
    """Create L2-normalized features with clear cluster structure."""
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, d))
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    features = []
    for c in centers:
        pts = c + rng.standard_normal((n_per_cluster, d)) * 0.05
        features.append(pts)
    features = np.concatenate(features, axis=0).astype(np.float32)
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    return features


def test_dbscan_returns_array_of_correct_length():
    feats = make_clustered_features()
    labels = dbscan_fit(feats)
    assert labels.shape == (60,)


def test_dbscan_finds_multiple_clusters():
    feats = make_clustered_features(n_clusters=3)
    labels = dbscan_fit(feats, eps=0.3, min_samples=3)
    unique_labels = set(labels[labels != -1])
    assert len(unique_labels) >= 2


def test_finch_returns_array_of_correct_length():
    feats = make_clustered_features()
    labels = finch_fit(feats)
    assert labels.shape == (60,)


def test_finch_no_outliers():
    feats = make_clustered_features()
    labels = finch_fit(feats)
    assert (labels >= 0).all(), "FINCH should assign all points to a cluster"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_clustering.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/clustering/dbscan.py`**

```python
"""DBSCAN clustering wrapper using cosine distance."""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances


def fit(features: np.ndarray, eps: float = 0.5, min_samples: int = 4) -> np.ndarray:
    """
    Run DBSCAN on L2-normalized features using cosine distance.

    Args:
        features:    (n, d) L2-normalized float32 array
        eps:         max cosine distance for a point to be in a neighborhood
        min_samples: min points to form a core point

    Returns:
        labels: (n,) int array, -1 for outliers/noise
    """
    dist_matrix = cosine_distances(features).astype(np.float64)
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed', n_jobs=-1)
    return db.fit_predict(dist_matrix)
```

- [ ] **Step 4: Create `src/clustering/finch.py`**

```python
"""FINCH clustering wrapper (first partition)."""
import numpy as np
from finch import FINCH


def fit(features: np.ndarray, random_state: int = 42) -> np.ndarray:
    """
    Run FINCH on L2-normalized features and return first-partition labels.

    FINCH assigns all points to a cluster (no outliers).

    Args:
        features:     (n, d) L2-normalized float32 array
        random_state: seed for reproducibility

    Returns:
        labels: (n,) int array, all values >= 0
    """
    c, num_clust, req_c = FINCH(
        features,
        distance='cosine',
        verbose=False,
        random_state=random_state,
    )
    return c[:, 0]
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_clustering.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/clustering/dbscan.py src/clustering/finch.py tests/test_clustering.py
git commit -m "feat: DBSCAN and FINCH clustering wrappers"
```

---

### Task 8: Uncertainty regions

**Files:**
- Create: `src/aas/uncertainty_regions.py`
- Create: `tests/test_uncertainty_regions.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_uncertainty_regions.py
import numpy as np
import pytest
from src.aas.uncertainty_regions import compute_iou, find_uncertainty_regions


def test_iou_zero_overlap():
    assert compute_iou({0, 1}, {2, 3}) == 0.0


def test_iou_full_overlap():
    assert compute_iou({0, 1, 2}, {0, 1, 2}) == 1.0


def test_iou_partial_overlap():
    iou = compute_iou({0, 1, 2}, {1, 2, 3})
    # intersection=2, union=4 → 0.5
    assert abs(iou - 0.5) < 1e-9


def test_no_uncertainty_when_perfect_agreement():
    # If both clusterings agree perfectly, no partial overlaps → no regions
    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([0, 0, 1, 1, 2, 2])
    regions = find_uncertainty_regions(labels_a, labels_b)
    assert regions == []


def test_uncertainty_region_found_on_disagreement():
    # A merges samples 2+3 with 0+1, B keeps them separate
    labels_a = np.array([0, 0, 0, 0, 1, 1])  # A: {0,1,2,3}, {4,5}
    labels_b = np.array([0, 0, 1, 1, 1, 1])  # B: {0,1}, {2,3,4,5}
    regions = find_uncertainty_regions(labels_a, labels_b)
    # All 6 samples are in one uncertain region (partial overlap between A[0] and B[0], B[1])
    assert len(regions) >= 1
    all_samples = set()
    for r in regions:
        all_samples |= set(r)
    assert {0, 1, 2, 3} <= all_samples


def test_outliers_excluded_from_regions():
    # DBSCAN outliers (label=-1) should not form a cluster
    labels_a = np.array([-1, 0, 0, 1, 1, -1])
    labels_b = np.array([0, 0, 1, 1, 0, 0])
    # Just check it runs without error
    regions = find_uncertainty_regions(labels_a, labels_b)
    assert isinstance(regions, list)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_uncertainty_regions.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/aas/uncertainty_regions.py`**

```python
"""Identify regions of uncertainty from disagreements between two clusterings."""
import numpy as np
from collections import defaultdict
from typing import List, Set


def compute_iou(set_a: Set[int], set_b: Set[int]) -> float:
    """Intersection over Union between two sets of sample indices."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def find_uncertainty_regions(
    labels_a: np.ndarray,
    labels_b: np.ndarray,
) -> List[List[int]]:
    """
    Find regions of uncertainty: transitive closure of partially overlapping clusters.

    A cluster pair (c_A, c_B) is a partial overlap when 0 < IoU(c_A, c_B) < 1.
    A region is a connected component in the graph of partially overlapping clusters.

    Args:
        labels_a: (n,) cluster labels from method A (DBSCAN; -1 = outlier)
        labels_b: (n,) cluster labels from method B (FINCH; all >= 0)

    Returns:
        regions: list of regions, each is a sorted list of sample indices
    """
    # Build cluster membership sets (exclude outliers for A)
    clusters_a: dict = defaultdict(set)
    for i, lbl in enumerate(labels_a):
        if lbl != -1:
            clusters_a[lbl].add(i)

    clusters_b: dict = defaultdict(set)
    for i, lbl in enumerate(labels_b):
        clusters_b[lbl].add(i)

    # Build adjacency: cluster nodes with partial overlap
    adj: dict = defaultdict(set)
    for lbl_a, set_a in clusters_a.items():
        for lbl_b, set_b in clusters_b.items():
            iou = compute_iou(set_a, set_b)
            if 0 < iou < 1:
                node_a = ('A', lbl_a)
                node_b = ('B', lbl_b)
                adj[node_a].add(node_b)
                adj[node_b].add(node_a)

    # Find connected components via BFS
    visited: Set = set()
    regions = []

    for start in list(adj.keys()):
        if start in visited:
            continue
        component_nodes: set = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            component_nodes.add(node)
            queue.extend(adj[node] - visited)

        # Collect all sample indices from this component
        sample_indices: set = set()
        for (source, lbl) in component_nodes:
            if source == 'A':
                sample_indices |= clusters_a[lbl]
            else:
                sample_indices |= clusters_b[lbl]

        if sample_indices:
            regions.append(sorted(sample_indices))

    return regions
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_uncertainty_regions.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aas/uncertainty_regions.py tests/test_uncertainty_regions.py
git commit -m "feat: uncertainty region detection"
```

---

### Task 9: Over-segmentation sampler (U_os)

**Files:**
- Create: `src/aas/over_seg_sampler.py`
- Create: `tests/test_over_seg_sampler.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_over_seg_sampler.py
import numpy as np
import pytest
from src.aas.over_seg_sampler import compute_medoid, sample_over_seg_pairs


def make_features(n=10, d=4, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, d)).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f


def test_medoid_is_in_region():
    features = make_features(10)
    indices = list(range(10))
    m = compute_medoid(indices, features)
    assert m in indices


def test_no_pairs_when_single_region():
    features = make_features(10)
    regions = [list(range(10))]   # only one region
    pairs = sample_over_seg_pairs(regions, features)
    assert pairs == []


def test_pairs_respect_s_min():
    # Make two well-separated groups → similarity between medoids should be low
    features = make_features(20)
    # Force features[0:10] and features[10:20] to be opposite
    features[:10] = np.array([1, 0, 0, 0], dtype=np.float32)
    features[10:] = np.array([-1, 0, 0, 0], dtype=np.float32)
    regions = [list(range(10)), list(range(10, 20))]
    pairs = sample_over_seg_pairs(regions, features, k_max=5, s_min=0.5)
    # Similarity between the two medoids is -1 < 0.5, so no pairs should form
    assert pairs == []


def test_pairs_formed_when_regions_similar():
    # Make two nearly identical regions → medoid similarity is high
    d = 128
    base = np.ones(d, dtype=np.float32)
    base /= np.linalg.norm(base)
    features = np.tile(base, (20, 1))
    features += np.random.default_rng(0).standard_normal((20, d)).astype(np.float32) * 0.001
    features /= np.linalg.norm(features, axis=1, keepdims=True)
    regions = [list(range(10)), list(range(10, 20))]
    pairs = sample_over_seg_pairs(regions, features, k_max=1, s_min=0.9)
    assert len(pairs) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_over_seg_sampler.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/aas/over_seg_sampler.py`**

```python
"""Sample pairs to resolve over-segmentation errors (U_os)."""
import numpy as np
from typing import List, Tuple


def compute_medoid(indices: List[int], features: np.ndarray) -> int:
    """
    Find the medoid of a set of samples.

    The medoid is the sample minimizing average cosine distance to all others.
    Since features are L2-normalized, cosine distance = 1 - dot product.

    Args:
        indices: list of sample indices (into the full features array)
        features: (n, d) L2-normalized feature matrix

    Returns:
        index (in the global features array) of the medoid
    """
    sub_feats = features[indices]                   # (m, d)
    sim_matrix = sub_feats @ sub_feats.T            # (m, m), cosine sim
    avg_dist = (1 - sim_matrix).mean(axis=1)        # average cosine distance
    local_idx = int(np.argmin(avg_dist))
    return indices[local_idx]


def sample_over_seg_pairs(
    regions: List[List[int]],
    features: np.ndarray,
    k_max: int = 5,
    s_min: float = 0.3,
) -> List[Tuple[int, int]]:
    """
    Construct U_os: pairs of medoids from different uncertain regions
    that might belong to the same individual (over-segmentation correction).

    For each region r_k, the medoid m_k is its representative.
    We pair m_k with up to k_max other medoids m_k' whose cosine
    similarity to m_k is >= s_min.

    Args:
        regions: list of uncertainty regions, each a list of sample indices
        features: (n, d) L2-normalized features
        k_max: max medoid neighbors to pair per medoid
        s_min: minimum cosine similarity threshold

    Returns:
        pairs: list of unique (i, j) sample index pairs for U_os
    """
    if len(regions) < 2:
        return []

    medoids = [compute_medoid(region, features) for region in regions]
    medoid_feats = features[medoids]                    # (M, d)
    sim_matrix = medoid_feats @ medoid_feats.T          # (M, M)
    np.fill_diagonal(sim_matrix, -2.0)                  # exclude self

    pairs: set = set()
    for k, m_k in enumerate(medoids):
        sorted_neighbors = np.argsort(sim_matrix[k])[::-1]
        count = 0
        for neighbor_k in sorted_neighbors:
            if count >= k_max:
                break
            sim = sim_matrix[k, neighbor_k]
            if sim < s_min:
                break                                   # sorted descending, no need to continue
            m_neighbor = medoids[neighbor_k]
            pairs.add(tuple(sorted((m_k, m_neighbor))))
            count += 1

    return list(pairs)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_over_seg_sampler.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aas/over_seg_sampler.py tests/test_over_seg_sampler.py
git commit -m "feat: over-segmentation sampler (U_os)"
```

---

### Task 10: Under-segmentation sampler (U_us)

**Files:**
- Create: `src/aas/under_seg_sampler.py`
- Create: `tests/test_under_seg_sampler.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_under_seg_sampler.py
import numpy as np
import pytest
from src.aas.under_seg_sampler import sample_under_seg_pairs


def make_features(n=8, d=4, seed=0):
    rng = np.random.default_rng(seed)
    f = rng.standard_normal((n, d)).astype(np.float32)
    f /= np.linalg.norm(f, axis=1, keepdims=True)
    return f


def test_returns_list():
    features = make_features(8)
    labels_a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    regions = [list(range(8))]
    pairs = sample_under_seg_pairs(regions, labels_a, labels_b, features)
    assert isinstance(pairs, list)


def test_perfect_agreement_gives_no_pairs():
    features = make_features(6)
    # Both methods agree on clusters
    labels_a = np.array([0, 0, 1, 1, 2, 2])
    labels_b = np.array([0, 0, 1, 1, 2, 2])
    regions = [list(range(6))]
    pairs = sample_under_seg_pairs(regions, labels_a, labels_b, features)
    assert pairs == []


def test_pairs_are_tuples_of_ints():
    features = make_features(8)
    labels_a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    regions = [list(range(8))]
    pairs = sample_under_seg_pairs(regions, labels_a, labels_b, features)
    for p in pairs:
        assert len(p) == 2
        assert isinstance(p[0], (int, np.integer))
        assert isinstance(p[1], (int, np.integer))


def test_no_self_pairs():
    features = make_features(8)
    labels_a = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    labels_b = np.array([0, 0, 1, 1, 1, 1, 2, 2])
    regions = [list(range(8))]
    pairs = sample_under_seg_pairs(regions, labels_a, labels_b, features)
    for (i, j) in pairs:
        assert i != j
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_under_seg_sampler.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/aas/under_seg_sampler.py`**

```python
"""Sample pairs to resolve under-segmentation errors (U_us)."""
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Set


def sample_under_seg_pairs(
    regions: List[List[int]],
    labels_a: np.ndarray,
    labels_b: np.ndarray,
    features: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Construct U_us: non-redundant inconsistent pairs within uncertainty regions.

    For each region S_k:
    - I_tilde_k = (intra-cluster pairs from A) △ (intra-cluster pairs from B)
                  = pairs where A and B disagree
    - P_cand_k  = closest cross-cluster pair for each distinct cluster pair in S_k
    - I_k       = I_tilde_k ∩ P_cand_k

    U_us = ∪ I_k over all regions.

    Args:
        regions:  list of uncertainty regions (sample indices)
        labels_a: (n,) DBSCAN cluster labels (-1 = outlier)
        labels_b: (n,) FINCH cluster labels
        features: (n, d) L2-normalized features

    Returns:
        pairs: list of unique (i, j) sample index pairs for U_us
    """
    all_pairs: Set[Tuple[int, int]] = set()

    for region in regions:
        region_set = set(region)

        a_groups: dict = defaultdict(set)
        b_groups: dict = defaultdict(set)
        for idx in region:
            if labels_a[idx] != -1:
                a_groups[labels_a[idx]].add(idx)
            b_groups[labels_b[idx]].add(idx)

        a_plus = _intra_cluster_pairs(a_groups)
        b_plus = _intra_cluster_pairs(b_groups)
        i_tilde = a_plus.symmetric_difference(b_plus)

        if not i_tilde:
            continue

        p_cand = _closest_inter_cluster_pairs(
            list(a_groups.values()) + list(b_groups.values()), features
        )
        i_k = i_tilde & p_cand
        all_pairs |= i_k

    return list(all_pairs)


def _intra_cluster_pairs(groups: dict) -> Set[Tuple[int, int]]:
    """All ordered (i<j) pairs within the same cluster."""
    pairs: Set[Tuple[int, int]] = set()
    for members in groups.values():
        members_list = sorted(members)
        for i in range(len(members_list)):
            for j in range(i + 1, len(members_list)):
                pairs.add((members_list[i], members_list[j]))
    return pairs


def _closest_inter_cluster_pairs(
    groups: List[Set[int]],
    features: np.ndarray,
) -> Set[Tuple[int, int]]:
    """
    For each distinct pair of groups, find the single closest cross-cluster pair
    (highest cosine similarity). Returns P_cand.
    """
    pairs: Set[Tuple[int, int]] = set()
    groups_list = [sorted(g) for g in groups if g]

    for i in range(len(groups_list)):
        for j in range(i + 1, len(groups_list)):
            g1, g2 = groups_list[i], groups_list[j]
            f1 = features[g1]           # (n1, d)
            f2 = features[g2]           # (n2, d)
            sim_matrix = f1 @ f2.T      # (n1, n2)
            best = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
            best_i = int(g1[best[0]])
            best_j = int(g2[best[1]])
            pairs.add(tuple(sorted((best_i, best_j))))

    return pairs
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_under_seg_sampler.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aas/under_seg_sampler.py tests/test_under_seg_sampler.py
git commit -m "feat: under-segmentation sampler (U_us)"
```

---

### Task 11: Main AAS sampler

**Files:**
- Create: `src/aas/sampler.py`

- [ ] **Step 1: Create `src/aas/sampler.py`**

```python
"""Main AAS sampler: run clustering, find uncertainty regions, sample B pairs."""
import numpy as np
from typing import List, Tuple

from src.clustering.dbscan import fit as dbscan_fit
from src.clustering.finch import fit as finch_fit
from src.aas.uncertainty_regions import find_uncertainty_regions
from src.aas.over_seg_sampler import sample_over_seg_pairs
from src.aas.under_seg_sampler import sample_under_seg_pairs


def run_aas(
    features: np.ndarray,
    budget: int,
    epsilon: float = 0.6,
    k_max: int = 5,
    s_min: float = 0.3,
    dbscan_eps: float = 0.5,
    dbscan_min_samples: int = 4,
    seed: int = 42,
) -> Tuple[List[Tuple[int, int]], np.ndarray, np.ndarray]:
    """
    Run the full AAS pipeline to produce `budget` annotatable pairs.

    Args:
        features:           (n, d) L2-normalized features
        budget:             number of pairs to sample (B)
        epsilon:            weight balancing U_os vs U_us (0=all U_us, 1=all U_os)
        k_max:              max nearest-medoid neighbors for U_os
        s_min:              min cosine similarity for U_os pairs
        dbscan_eps:         DBSCAN neighborhood radius (cosine distance)
        dbscan_min_samples: DBSCAN minimum core points
        seed:               RNG seed

    Returns:
        sampled_pairs:  list of (i, j) pairs for oracle annotation
        labels_a:       (n,) DBSCAN labels (for NP3 input)
        labels_b:       (n,) FINCH labels  (for NP3 input)
    """
    rng = np.random.default_rng(seed)

    labels_a = dbscan_fit(features, eps=dbscan_eps, min_samples=dbscan_min_samples)
    labels_b = finch_fit(features, random_state=seed)

    regions = find_uncertainty_regions(labels_a, labels_b)

    if not regions:
        return [], labels_a, labels_b

    u_os = sample_over_seg_pairs(regions, features, k_max=k_max, s_min=s_min)
    u_us = sample_under_seg_pairs(regions, labels_a, labels_b, features)

    sampled = _marginal_sample(u_os, u_us, features, budget, epsilon, rng)

    return sampled, labels_a, labels_b


def _marginal_sample(
    u_os: List[Tuple[int, int]],
    u_us: List[Tuple[int, int]],
    features: np.ndarray,
    budget: int,
    epsilon: float,
    rng: np.random.Generator,
) -> List[Tuple[int, int]]:
    """Sample B pairs from U = U_os ∪ U_us using marginal distribution P(Y)."""
    os_set = set(u_os)
    us_set = set(u_us)
    total_pool = list(os_set | us_set)

    if not total_pool:
        return []

    if len(total_pool) <= budget:
        return total_pool

    # Compute per-pair probability weights
    n_us = max(len(u_us), 1)
    probs = np.zeros(len(total_pool), dtype=np.float64)
    for idx, pair in enumerate(total_pool):
        in_os = pair in os_set
        in_us = pair in us_set
        sim = float(features[pair[0]] @ features[pair[1]])

        if in_os and not in_us:
            probs[idx] = epsilon * max(sim, 0.0)
        elif in_us and not in_os:
            probs[idx] = (1 - epsilon) / n_us
        else:
            p_os = epsilon * max(sim, 0.0)
            p_us = (1 - epsilon) / n_us
            probs[idx] = 0.5 * p_os + 0.5 * p_us

    probs = np.clip(probs, 1e-12, None)
    probs /= probs.sum()

    chosen = rng.choice(len(total_pool), size=budget, replace=False, p=probs)
    return [total_pool[i] for i in chosen]
```

- [ ] **Step 2: Smoke-test sampler end-to-end**

```bash
python3 -c "
import numpy as np
from src.aas.sampler import run_aas

rng = np.random.default_rng(0)
features = rng.standard_normal((200, 128)).astype(np.float32)
features /= np.linalg.norm(features, axis=1, keepdims=True)

pairs, la, lb = run_aas(features, budget=10)
print(f'Regions found, sampled {len(pairs)} pairs')
print('First pair:', pairs[0] if pairs else 'none')
"
```

Expected: prints sampled N pairs (may be 0 if no uncertainty regions)

- [ ] **Step 3: Commit**

```bash
git add src/aas/sampler.py
git commit -m "feat: main AAS sampler combining U_os and U_us"
```

---

### Task 12: NP3 constrained cluster refinement

**Files:**
- Create: `src/aas/np3.py`
- Create: `tests/test_np3.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_np3.py
import numpy as np
import pytest
from src.aas.np3 import refine_labels


def test_must_links_satisfied():
    labels = np.array([0, 1, 2, 3])
    must_links = [(0, 1)]   # samples 0 and 1 must be in same cluster
    cannot_links = []
    refined = refine_labels(labels, must_links, cannot_links)
    assert refined[0] == refined[1], "ML pair must be in same cluster"


def test_cannot_links_satisfied():
    # All 4 in same cluster initially
    labels = np.array([0, 0, 0, 0])
    must_links = []
    cannot_links = [(0, 1)]
    refined = refine_labels(labels, must_links, cannot_links)
    assert refined[0] != refined[1], "CL pair must be in different clusters"


def test_ml_then_cl():
    # samples 0,1 must link; samples 0,2 cannot link
    labels = np.array([0, 0, 0, 1])
    must_links = [(0, 1)]
    cannot_links = [(0, 2)]
    refined = refine_labels(labels, must_links, cannot_links)
    assert refined[0] == refined[1], "ML: 0 and 1 same cluster"
    assert refined[0] != refined[2], "CL: 0 and 2 different clusters"


def test_output_length_unchanged():
    labels = np.array([0, 0, 1, 1, 2])
    refined = refine_labels(labels, [(0, 1)], [(0, 2)])
    assert len(refined) == len(labels)


def test_no_constraints_unchanged():
    labels = np.array([0, 1, 2])
    refined = refine_labels(labels, [], [])
    np.testing.assert_array_equal(refined, labels)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_np3.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/aas/np3.py`**

```python
"""
NP3: Non-Parametric, Plug-and-Play constrained cluster refinement.

Given initial cluster labels and pairwise must-link (ML) / cannot-link (CL)
constraints, produces refined labels satisfying all constraints.

Algorithm:
  1. Satisfy ML constraints by merging clusters via union-find.
  2. For each impure cluster (internal CL violations):
     a. Compute ML groups (transitive closure of ML within cluster).
     b. Build conflict graph (nodes = ML groups, edges = CL between groups).
     c. Greedy-color the conflict graph.
     d. Assign new cluster labels by color; Hungarian-match unconstrained points.
"""
import numpy as np
import networkx as nx
from collections import defaultdict
from typing import List, Tuple


def refine_labels(
    labels: np.ndarray,
    must_links: List[Tuple[int, int]],
    cannot_links: List[Tuple[int, int]],
) -> np.ndarray:
    """
    Refine cluster labels to satisfy all pairwise constraints.

    Args:
        labels:       (n,) initial cluster label array
        must_links:   list of (i, j) pairs that must share a cluster
        cannot_links: list of (i, j) pairs that must be in different clusters

    Returns:
        refined_labels: (n,) label array satisfying all constraints
    """
    if not must_links and not cannot_links:
        return labels.copy()

    labels = _merge_must_links(labels.copy(), must_links)
    labels = _resolve_cannot_links(labels, must_links, cannot_links)
    return labels


# ---------------------------------------------------------------------------
# Step 1: Merge clusters to satisfy must-link constraints
# ---------------------------------------------------------------------------

def _merge_must_links(
    labels: np.ndarray,
    must_links: List[Tuple[int, int]],
) -> np.ndarray:
    """Union-find merge of samples connected by ML constraints."""
    n = len(labels)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (i, j) in must_links:
        union(i, j)

    # Samples in the same ML component get the label of the component root
    root_to_label: dict = {}
    new_labels = labels.copy()
    for idx in range(n):
        root = find(idx)
        if root not in root_to_label:
            root_to_label[root] = labels[root]
        new_labels[idx] = root_to_label[root]

    return new_labels


# ---------------------------------------------------------------------------
# Step 2: Resolve cannot-link violations
# ---------------------------------------------------------------------------

def _resolve_cannot_links(
    labels: np.ndarray,
    must_links: List[Tuple[int, int]],
    cannot_links: List[Tuple[int, int]],
) -> np.ndarray:
    """For each impure cluster, split it using graph coloring."""
    cl_set = {tuple(sorted(p)) for p in cannot_links}
    ml_set = {tuple(sorted(p)) for p in must_links}

    label_to_members: dict = defaultdict(list)
    for idx, lbl in enumerate(labels):
        label_to_members[lbl].append(idx)

    new_labels = labels.copy()
    next_label = int(labels.max()) + 1

    for cluster_lbl, members in label_to_members.items():
        members_set = set(members)

        # Check for internal CL constraints
        internal_cls = [(i, j) for (i, j) in cl_set if i in members_set and j in members_set]
        if not internal_cls:
            continue

        # Build ML groups within this cluster (transitive closure)
        ml_graph = nx.Graph()
        ml_graph.add_nodes_from(members)
        for (i, j) in ml_set:
            if i in members_set and j in members_set:
                ml_graph.add_edge(i, j)

        ml_groups = list(nx.connected_components(ml_graph))
        group_of = {sample: g_idx for g_idx, group in enumerate(ml_groups) for sample in group}

        # Build conflict graph: nodes = ML groups, edges = CL between groups
        conflict = nx.Graph()
        conflict.add_nodes_from(range(len(ml_groups)))
        for (i, j) in internal_cls:
            gi, gj = group_of[i], group_of[j]
            if gi != gj:
                conflict.add_edge(gi, gj)

        # Greedy coloring
        coloring = nx.coloring.greedy_color(conflict, strategy='largest_first')
        max_color = max(coloring.values()) if coloring else 0

        # Map color 0 → existing cluster label, other colors → new labels
        color_to_label = {0: cluster_lbl}
        for color in range(1, max_color + 1):
            color_to_label[color] = next_label
            next_label += 1

        for g_idx, color in coloring.items():
            lbl = color_to_label[color]
            for sample in ml_groups[g_idx]:
                new_labels[sample] = lbl

    return new_labels
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_np3.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/aas/np3.py tests/test_np3.py
git commit -m "feat: NP3 constrained cluster refinement"
```

---

### Task 13: GT Oracle simulator

**Files:**
- Create: `src/oracle/gt_oracle.py`
- Create: `tests/test_oracle.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_oracle.py
import numpy as np
import pytest
from src.oracle.gt_oracle import GTOracle


def test_same_identity_is_must_link():
    gt = np.array([0, 0, 1, 1])
    oracle = GTOracle(gt)
    ml, cl = oracle.query([(0, 1)])
    assert (0, 1) in ml
    assert cl == []


def test_different_identity_is_cannot_link():
    gt = np.array([0, 0, 1, 1])
    oracle = GTOracle(gt)
    ml, cl = oracle.query([(0, 2)])
    assert ml == []
    assert (0, 2) in cl


def test_mixed_pairs():
    gt = np.array([0, 0, 1, 2])
    oracle = GTOracle(gt)
    ml, cl = oracle.query([(0, 1), (0, 2), (1, 3)])
    assert (0, 1) in ml
    assert (0, 2) in cl
    assert (1, 3) in cl


def test_empty_query():
    gt = np.array([0, 1, 2])
    oracle = GTOracle(gt)
    ml, cl = oracle.query([])
    assert ml == [] and cl == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_oracle.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/oracle/gt_oracle.py`**

```python
"""Ground-truth oracle simulator for pairwise annotations."""
import numpy as np
from typing import List, Tuple


class GTOracle:
    """
    Simulates a human annotator using ground-truth identity labels.

    For each queried pair (i, j):
    - gt_labels[i] == gt_labels[j]  →  must-link
    - gt_labels[i] != gt_labels[j]  →  cannot-link

    Args:
        gt_labels: (n,) integer array of ground-truth identity labels
    """

    def __init__(self, gt_labels: np.ndarray):
        self.gt_labels = np.asarray(gt_labels)

    def query(
        self,
        pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Get must-link and cannot-link annotations for a list of pairs.

        Args:
            pairs: list of (i, j) sample index pairs

        Returns:
            must_links:   pairs where samples share an identity
            cannot_links: pairs where samples have different identities
        """
        must_links: List[Tuple[int, int]] = []
        cannot_links: List[Tuple[int, int]] = []

        for (i, j) in pairs:
            if self.gt_labels[i] == self.gt_labels[j]:
                must_links.append((i, j))
            else:
                cannot_links.append((i, j))

        return must_links, cannot_links
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_oracle.py -v
```

Expected: all 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/oracle/gt_oracle.py tests/test_oracle.py
git commit -m "feat: GT oracle simulator"
```

---

### Task 14: Evaluation metrics

**Files:**
- Create: `src/eval/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
import numpy as np
import pytest
from src.eval.metrics import compute_metrics, _average_precision, _mean_inverse_negative_penalty


def test_ap_perfect_retrieval():
    # All positives ranked first
    matches = np.array([1, 1, 1, 0, 0])
    ap = _average_precision(matches)
    assert abs(ap - 1.0) < 1e-6


def test_ap_no_positives():
    matches = np.array([0, 0, 0])
    ap = _average_precision(matches)
    assert ap == 0.0


def test_minp_first_rank():
    # Last positive is at rank 1 → mINP = 1/1 = 1.0
    matches = np.array([1, 0, 0, 0])
    minp = _mean_inverse_negative_penalty(matches)
    assert abs(minp - 1.0) < 1e-6


def test_minp_last_rank():
    # Last positive at rank 4 → mINP = 1/4
    matches = np.array([0, 0, 0, 1])
    minp = _mean_inverse_negative_penalty(matches)
    assert abs(minp - 0.25) < 1e-6


def test_compute_metrics_shape():
    n_q, n_g, d = 5, 20, 64
    rng = np.random.default_rng(0)
    qf = rng.standard_normal((n_q, d)).astype(np.float32)
    gf = rng.standard_normal((n_g, d)).astype(np.float32)
    qf /= np.linalg.norm(qf, axis=1, keepdims=True)
    gf /= np.linalg.norm(gf, axis=1, keepdims=True)
    ql = np.array([0, 1, 2, 3, 4])
    gl = np.arange(n_g) % 5
    metrics = compute_metrics(qf, gf, ql, gl)
    for key in ['mAP', 'mINP', 'top1', 'top3', 'top5', 'top10']:
        assert key in metrics
        assert 0.0 <= metrics[key] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```

Expected: FAIL with `ImportError`

- [ ] **Step 3: Create `src/eval/metrics.py`**

```python
"""Re-ID evaluation metrics: mAP, mINP, BAKS, AUCROC, Top-{1,3,5,10}."""
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, Optional


def compute_metrics(
    query_feats: np.ndarray,
    gallery_feats: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_is_known: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute closed-set and open-set Re-ID metrics.

    Args:
        query_feats:   (nq, d) L2-normalized query features
        gallery_feats: (ng, d) L2-normalized gallery features
        query_labels:  (nq,) integer identity labels
        gallery_labels:(ng,) integer identity labels
        query_is_known:(nq,) bool array; True if query identity appears in gallery.
                       Required for BAKS and AUCROC.

    Returns:
        dict with keys: mAP, mINP, BAKS (if known mask given), AUCROC (if known mask given),
                        top1, top3, top5, top10
    """
    sim_matrix = query_feats @ gallery_feats.T      # (nq, ng)

    map_scores, minp_scores = [], []
    top_k_hits = {1: [], 3: [], 5: [], 10: []}

    for q_idx in range(len(query_labels)):
        sims = sim_matrix[q_idx]
        sorted_idx = np.argsort(sims)[::-1]
        matches = (gallery_labels[sorted_idx] == query_labels[q_idx]).astype(int)

        map_scores.append(_average_precision(matches))
        minp_scores.append(_mean_inverse_negative_penalty(matches))

        for k in top_k_hits:
            top_k_hits[k].append(int(matches[:k].sum() > 0))

    results: Dict[str, float] = {
        'mAP':   float(np.mean(map_scores)),
        'mINP':  float(np.mean(minp_scores)),
        'top1':  float(np.mean(top_k_hits[1])),
        'top3':  float(np.mean(top_k_hits[3])),
        'top5':  float(np.mean(top_k_hits[5])),
        'top10': float(np.mean(top_k_hits[10])),
    }

    if query_is_known is not None:
        known_idx = np.where(query_is_known)[0]
        results['BAKS'] = float(np.mean([map_scores[i] for i in known_idx])) if len(known_idx) > 0 else 0.0

        max_sims = sim_matrix.max(axis=1)
        results['AUCROC'] = float(roc_auc_score(query_is_known.astype(int), max_sims))

    return results


def _average_precision(matches: np.ndarray) -> float:
    """Compute AP from a binary relevance array (1=match, 0=non-match)."""
    n_pos = matches.sum()
    if n_pos == 0:
        return 0.0
    hits, ap = 0, 0.0
    for rank, m in enumerate(matches, 1):
        if m:
            hits += 1
            ap += hits / rank
    return ap / n_pos


def _mean_inverse_negative_penalty(matches: np.ndarray) -> float:
    """mINP = 1 / (1-indexed rank of the last positive match)."""
    pos_ranks = np.where(matches)[0]
    if len(pos_ranks) == 0:
        return 0.0
    return 1.0 / (pos_ranks[-1] + 1)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_metrics.py -v
```

Expected: all 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/eval/metrics.py tests/test_metrics.py
git commit -m "feat: Re-ID evaluation metrics (mAP, mINP, BAKS, AUCROC, Top-k)"
```

---

## Phase 3: Training & Evaluation

### Task 15: Integrate AAS into SpCL training loop

**Files:**
- Create: `experiments/train_aas.py`

Context: SpCL's `examples/train_usl.py` runs a loop that (1) extracts features, (2) clusters to generate pseudo-labels, (3) trains for one epoch. We wrap this with our AAS callback every 10 epochs. The key SpCL functions to call are:
- `spcl.utils.faiss_rerank.compute_jaccard_distance` for feature distances
- The trainer's `train()` method
- `extract_features()` from SpCL utils

Rather than modifying SpCL's internals, we write a standalone script that imports SpCL's components.

- [ ] **Step 1: Study SpCL's training structure**

```bash
cat third_party/SpCL/examples/train_usl.py | head -150
```

Read how pseudo-labels are generated and how the trainer is invoked. Note the function names.

- [ ] **Step 2: Create `experiments/train_aas.py`**

```python
"""
AAS-enhanced SpCL training loop.

Runs SpCL USL training and injects AAS active sampling every `al_interval` epochs.
"""
import sys
import os
import argparse
import yaml
import numpy as np
import torch

# Add SpCL to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'third_party', 'SpCL'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.aas.sampler import run_aas
from src.aas.np3 import refine_labels
from src.oracle.gt_oracle import GTOracle
from src.eval.metrics import compute_metrics


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_budget(n_samples: int, budget_fraction: float) -> int:
    """Budget B = fraction of all possible pairs."""
    n_pairs = n_samples * (n_samples - 1) // 2
    return max(1, int(n_pairs * budget_fraction))


def run_al_cycle(
    features: np.ndarray,
    gt_labels: np.ndarray,
    pseudo_labels: np.ndarray,
    cfg: dict,
    cycle: int,
) -> np.ndarray:
    """
    One AAS active learning cycle:
      1. Run AAS to get pairs.
      2. Query GT oracle.
      3. Refine pseudo-labels with NP3.

    Args:
        features:      (n, d) L2-normalized features
        gt_labels:     (n,) ground-truth identity labels
        pseudo_labels: (n,) current SpCL pseudo-labels
        cfg:           experiment config dict
        cycle:         current AL cycle index (for seeding)

    Returns:
        refined_labels: (n,) NP3-refined pseudo-labels
    """
    budget = compute_budget(len(features), cfg['budget_fraction'])

    pairs, _, _ = run_aas(
        features,
        budget=budget,
        epsilon=cfg['epsilon'],
        k_max=cfg['k_max'],
        s_min=cfg['s_min'],
        dbscan_eps=cfg['dbscan_eps'],
        dbscan_min_samples=cfg['dbscan_min_samples'],
        seed=cfg['seed'] + cycle,
    )

    oracle = GTOracle(gt_labels)
    must_links, cannot_links = oracle.query(pairs)

    print(f"  Cycle {cycle}: {len(pairs)} pairs queried | "
          f"{len(must_links)} ML | {len(cannot_links)} CL")

    refined = refine_labels(pseudo_labels, must_links, cannot_links)
    return refined


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to aas.yaml')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g. IPanda50)')
    parser.add_argument('--data-root', required=True, help='Root path to WildlifeReID-10k data')
    parser.add_argument('--output-dir', default='experiments/results')
    parser.add_argument('--run-id', type=int, default=0, help='Run index (0-3) for 4-run averaging')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== AAS Reproduction: {args.dataset} | Run {args.run_id} ===")

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------
    from src.data.download import filter_paper_datasets, load_metadata
    from src.data.dataset import WildlifeSubsetDataset
    from src.data.transforms import get_transforms
    from src.data.splits import make_splits
    from src.data.features import extract_features

    df_all = load_metadata(args.data_root)
    df = df_all[df_all['dataset'] == args.dataset].copy()

    if df.empty:
        raise ValueError(f"Dataset '{args.dataset}' not found in metadata. "
                         f"Available: {df_all['dataset'].unique().tolist()}")

    # Factorize identity labels to 0-indexed integers
    df['identity'] = df['identity'].factorize()[0]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Extract MegaDescriptor embeddings for gallery selection
    print("Extracting MegaDescriptor features for gallery split...")
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    train_ds = WildlifeSubsetDataset(train_df, root=args.data_root, transform=get_transforms('val'))
    mega_feats = extract_features(train_ds, backbone='megadescriptor', device=device)

    gallery_df, query_df, held_out_df = make_splits(
        df,
        held_out_fraction=cfg.get('held_out_fraction', 0.2),
        max_exemplars=cfg.get('max_exemplars', 5),
        embeddings=mega_feats,
        seed=cfg['seed'] + args.run_id,
    )

    # -------------------------------------------------------------------------
    # SpCL training with AAS injection
    # -------------------------------------------------------------------------
    # Import SpCL training utilities
    from spcl.models.resnet import ResNet
    from spcl.trainers import SpCLTrainer
    from spcl.utils.data import IterLoader

    train_ds = WildlifeSubsetDataset(train_df, root=args.data_root, transform=get_transforms('train'))

    # Initialize ResNet-50
    model = ResNet(depth=50, num_features=2048, dropout=0, num_classes=0)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get('lr', 3.5e-4), weight_decay=5e-4)

    gt_labels = train_df['identity'].values
    al_interval = cfg.get('al_interval', 10)
    total_epochs = cfg.get('total_epochs', 50)
    al_cycle = 0
    accumulated_ml: list = []
    accumulated_cl: list = []

    for epoch in range(total_epochs):
        # Extract features
        train_ds_eval = WildlifeSubsetDataset(train_df, root=args.data_root, transform=get_transforms('val'))
        features = extract_features(train_ds_eval, backbone='resnet50', device=device)

        # Generate SpCL pseudo-labels via Jaccard clustering
        # (SpCL handles this internally in its trainer; we call it once per interval)
        from spcl.utils.faiss_rerank import compute_jaccard_distance
        from sklearn.cluster import DBSCAN as skDBSCAN

        rerank_dist = compute_jaccard_distance(
            torch.tensor(features).to(device),
            k1=cfg.get('k1', 30),
            k2=cfg.get('k2', 6),
        )
        cluster_ids = skDBSCAN(
            eps=cfg.get('pseudo_eps', 0.6),
            min_samples=4,
            metric='precomputed',
            n_jobs=-1,
        ).fit_predict(rerank_dist.cpu().numpy())

        # AAS injection every al_interval epochs
        if (epoch + 1) % al_interval == 0 and epoch > 0:
            print(f"[Epoch {epoch+1}] Running AAS cycle {al_cycle + 1}...")
            refined = run_al_cycle(features, gt_labels, cluster_ids, cfg, al_cycle)
            cluster_ids = refined
            al_cycle += 1

        # Train for one epoch with SpCL
        trainer = SpCLTrainer(model, num_classes=int(cluster_ids.max()) + 1,
                              memory_size=len(train_df), device=device)
        trainer.train(epoch, train_ds, cluster_ids, optimizer)

    # -------------------------------------------------------------------------
    # Evaluation
    # -------------------------------------------------------------------------
    print("Evaluating...")
    gallery_ds = WildlifeSubsetDataset(gallery_df, root=args.data_root, transform=get_transforms('val'))
    query_ds = WildlifeSubsetDataset(query_df, root=args.data_root, transform=get_transforms('val'))

    gallery_feats = extract_features(gallery_ds, backbone='resnet50', device=device)
    query_feats = extract_features(query_ds, backbone='resnet50', device=device)

    gallery_labels = gallery_df['identity'].values
    query_labels = query_df['identity'].values

    gallery_ids = set(gallery_df['identity'].unique())
    query_is_known = np.array([lbl in gallery_ids for lbl in query_labels])

    metrics = compute_metrics(query_feats, gallery_feats, query_labels,
                              gallery_labels, query_is_known)

    print(f"\n=== Results: {args.dataset} | Run {args.run_id} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save results
    import json
    out_path = os.path.join(args.output_dir, f"{args.dataset}_run{args.run_id}.json")
    with open(out_path, 'w') as f:
        json.dump({'dataset': args.dataset, 'run': args.run_id, 'metrics': metrics}, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Commit**

```bash
git add experiments/train_aas.py
git commit -m "feat: AAS-enhanced SpCL training script"
```

---

### Task 16: Experiment config and runner script

**Files:**
- Create: `experiments/configs/aas.yaml`
- Create: `experiments/run_all.sh`

- [ ] **Step 1: Create `experiments/configs/aas.yaml`**

```yaml
# AAS Reproduction Config
# All hyperparameters fixed as per the paper (Section 4 / Table 1)

# AAS hyperparameters (paper Section 4)
epsilon: 0.6          # balance between U_os and U_us
k_max: 5              # max nearest-medoid neighbors (U_os)
s_min: 0.3            # min cosine similarity threshold (U_os)
budget_fraction: 0.0002  # 0.02% of all possible pairs per AL cycle

# DBSCAN parameters (for AAS clustering)
dbscan_eps: 0.5
dbscan_min_samples: 4

# SpCL pseudo-label generation
pseudo_eps: 0.6       # DBSCAN eps for SpCL's Jaccard-based clustering
k1: 30                # Jaccard k1
k2: 6                 # Jaccard k2

# Training
total_epochs: 50
al_interval: 10       # run AAS every N epochs → 5 AL cycles
lr: 0.00035
seed: 42

# Data split
held_out_fraction: 0.2
max_exemplars: 5

# 13 datasets used in paper (confirm names from metadata.csv 'dataset' column)
datasets:
  - BelugaID
  - CowDataset
  - FriesianCattle2015
  - GiraffeZebraID
  - HyenaID2022
  - IPanda50
  - LeopardID2022
  - MacaqueFaces
  - NyalaData
  - SealID
  - WhaleSharkID
  - HappyWhale
  - ELPephants
```

- [ ] **Step 2: Create `experiments/run_all.sh`**

```bash
#!/usr/bin/env bash
# Run AAS on all 13 datasets, 4 runs each.
# Usage: bash experiments/run_all.sh data/raw experiments/results

DATA_ROOT=${1:-data/raw}
OUTPUT_DIR=${2:-experiments/results}
CONFIG=experiments/configs/aas.yaml

DATASETS=(
  BelugaID CowDataset FriesianCattle2015 GiraffeZebraID HyenaID2022
  IPanda50 LeopardID2022 MacaqueFaces NyalaData SealID
  WhaleSharkID HappyWhale ELPephants
)

for dataset in "${DATASETS[@]}"; do
  for run in 0 1 2 3; do
    echo ">>> Dataset=$dataset | Run=$run"
    python3 experiments/train_aas.py \
      --config "$CONFIG" \
      --dataset "$dataset" \
      --data-root "$DATA_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --run-id "$run"
  done
done

echo "All runs complete. Results in $OUTPUT_DIR"
```

- [ ] **Step 3: Make shell script executable and commit**

```bash
chmod +x experiments/run_all.sh
git add experiments/configs/aas.yaml experiments/run_all.sh
git commit -m "feat: experiment config and runner script"
```

---

### Task 17: Results aggregation and notebooks

**Files:**
- Create: `experiments/aggregate_results.py`
- Create: `notebooks/reproduce_table1.ipynb` (skeleton)
- Create: `notebooks/budget_curves.ipynb` (skeleton)

- [ ] **Step 1: Create `experiments/aggregate_results.py`**

```python
"""Aggregate per-run JSON results into a summary table."""
import os
import json
import numpy as np
import pandas as pd


def aggregate(results_dir: str) -> pd.DataFrame:
    """
    Read all per-run JSON files from results_dir.
    Compute mean ± std over 4 runs per dataset.
    Return a DataFrame with one row per dataset.

    Args:
        results_dir: directory containing <dataset>_run<N>.json files

    Returns:
        summary DataFrame
    """
    metrics_keys = ['mAP', 'mINP', 'BAKS', 'AUCROC', 'top1', 'top3', 'top5', 'top10']
    records: dict = {}

    for fname in os.listdir(results_dir):
        if not fname.endswith('.json'):
            continue
        with open(os.path.join(results_dir, fname)) as f:
            data = json.load(f)
        dataset = data['dataset']
        metrics = data['metrics']
        records.setdefault(dataset, {k: [] for k in metrics_keys})
        for k in metrics_keys:
            if k in metrics:
                records[dataset][k].append(metrics[k])

    rows = []
    for dataset, runs in records.items():
        row = {'dataset': dataset}
        for k in metrics_keys:
            values = runs[k]
            row[f'{k}_mean'] = np.mean(values) if values else float('nan')
            row[f'{k}_std'] = np.std(values) if values else float('nan')
        rows.append(row)

    df = pd.DataFrame(rows).set_index('dataset')

    # Macro-average over all datasets
    mean_row = {col: df[col].mean() for col in df.columns}
    mean_row['dataset'] = 'MACRO_AVG'
    df = pd.concat([df, pd.DataFrame([mean_row]).set_index('dataset')])

    return df


if __name__ == '__main__':
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'experiments/results'
    df = aggregate(results_dir)
    print("\n=== AAS Reproduction Results ===")
    print(df[['mAP_mean', 'mINP_mean', 'BAKS_mean', 'AUCROC_mean',
              'top1_mean', 'top5_mean']].to_string())

    # Paper targets for AAS row
    print("\n=== Paper targets (Table 1, AAS row) ===")
    targets = {'mAP': 56.14, 'mINP': 38.17, 'BAKS': 67.15,
               'AUCROC': 75.21, 'top1': 67.71, 'top5': 85.04}
    for k, v in targets.items():
        ours = df.loc['MACRO_AVG', f'{k}_mean'] * 100
        delta = ours - v
        print(f"  {k}: ours={ours:.2f}%  paper={v:.2f}%  delta={delta:+.2f}%")
```

- [ ] **Step 2: Commit aggregation script**

```bash
git add experiments/aggregate_results.py
git commit -m "feat: results aggregation script"
```

- [ ] **Step 3: Create notebook stubs**

Create `notebooks/reproduce_table1.ipynb` with these cells:

```python
# Cell 1
import sys; sys.path.insert(0, '..')
import pandas as pd
from experiments.aggregate_results import aggregate

df = aggregate('../experiments/results')
df
```

```python
# Cell 2 — comparison with paper
paper = {
    'mAP_mean': 0.5614, 'mINP_mean': 0.3817, 'BAKS_mean': 0.6715,
    'AUCROC_mean': 0.7521, 'top1_mean': 0.6771, 'top5_mean': 0.8504,
}
macro = df.loc['MACRO_AVG']
for k, v in paper.items():
    print(f"{k}: ours={macro[k]*100:.2f}%  paper={v*100:.2f}%  delta={macro[k]*100 - v*100:+.2f}%")
```

Create `notebooks/budget_curves.ipynb` with these cells:

```python
# Cell 1
# Load per-cycle mAP from training logs (requires adding cycle logging to train_aas.py)
import json, os, glob
import numpy as np
import matplotlib.pyplot as plt

results_dir = '../experiments/results'
# budget_curves.json is written by train_aas.py (add this logging in Task 15 if needed)
curve_files = glob.glob(os.path.join(results_dir, '*_budget_curve.json'))
print(f"Found {len(curve_files)} curve files")
```

```python
# Cell 2
# Plot mAP vs AL cycle
fig, ax = plt.subplots(figsize=(8, 5))
for f in curve_files:
    with open(f) as fh:
        data = json.load(fh)
    ax.plot(data['cycles'], data['mAP'], label=data['dataset'])
ax.set_xlabel('AL Cycle')
ax.set_ylabel('mAP')
ax.set_title('AAS: mAP vs AL Cycle')
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig('budget_curves.png', dpi=150)
plt.show()
```

- [ ] **Step 4: Commit notebooks**

```bash
git add notebooks/ experiments/aggregate_results.py
git commit -m "feat: analysis notebooks and aggregation script"
```

---

## Phase 4: End-to-End Verification

### Task 18: Smoke test with one small dataset

Before running all 13 datasets, verify the full pipeline on a single small dataset (e.g. IPanda50 which has ~5K images vs. WhaleSharkID which has >50K).

- [ ] **Step 1: Download data**

```bash
python3 -c "
from src.data.download import download, load_metadata, filter_paper_datasets
ds = download('data/raw')
df = load_metadata('data/raw')
print(df.head())
print(df.columns.tolist())
print(df['dataset'].value_counts().head(20) if 'dataset' in df.columns else 'No dataset column')
"
```

Expected: prints metadata DataFrame structure. Use the output to verify the `dataset` column name and confirm the 13 dataset names. Update `PAPER_DATASETS_13` in `src/data/download.py` if names differ.

- [ ] **Step 2: Run single dataset smoke test**

```bash
python3 experiments/train_aas.py \
  --config experiments/configs/aas.yaml \
  --dataset IPanda50 \
  --data-root data/raw \
  --output-dir experiments/results \
  --run-id 0
```

Expected: completes without errors, prints metric values, saves `experiments/results/IPanda50_run0.json`.

- [ ] **Step 3: Run aggregation on single result**

```bash
python3 experiments/aggregate_results.py experiments/results
```

Expected: prints a table with IPanda50 metrics.

- [ ] **Step 4: Run full suite**

```bash
bash experiments/run_all.sh data/raw experiments/results
```

Expected: all 52 runs (13 × 4) complete. Wall-time: several hours on GPU.

- [ ] **Step 5: Final aggregate and compare to paper**

```bash
python3 experiments/aggregate_results.py experiments/results
```

Target (from paper Table 1, AAS row):
```
mAP: 56.14%  mINP: 38.17%  BAKS: 67.15%  AUCROC: 75.21%  Top-1: 67.71%  Top-5: 85.04%
```

---

## Appendix: DBSCAN Parameter Tuning Note

The paper fixes AAS hyperparameters at ε=0.6, k_max=5, s_min=0.3, but does not explicitly state DBSCAN's `eps` and `min_samples` for the AAS clustering step. The plan uses `eps=0.5, min_samples=4` as defaults in cosine distance space for L2-normalized features.

If results diverge significantly from the paper, try:
- `eps ∈ {0.3, 0.4, 0.5, 0.6}` with `min_samples ∈ {3, 4, 5}`
- These are exposed in `experiments/configs/aas.yaml` as `dbscan_eps` and `dbscan_min_samples`

## Appendix: Dataset Name Verification

`PAPER_DATASETS_13` in `src/data/download.py` lists 13 assumed dataset names. After downloading WildlifeReID-10k (Task 18, Step 1), verify these match the actual `dataset` column values in metadata.csv and update if needed. The exact 13 datasets are listed in the paper's supplementary material.
