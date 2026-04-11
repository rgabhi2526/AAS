# AAS Paper Reproduction — Design Spec

**Date:** 2026-04-09
**Paper:** "Active Learning for Animal Re-Identification with Ambiguity-Aware Sampling" (Sani, Khurana, Anand — IIIT Delhi, AAAI 2026)
**Goal:** Faithfully reproduce the AAS results claimed in Table 1 of the paper on 13 WildlifeReID-10k datasets.

---

## 1. Overall Architecture

Linear pipeline across 3 phases:

```
Phase 1: Data Infrastructure
  └── WildlifeReID-10k: download + dataset loader (13 datasets)

Phase 2: Core AAS Implementation (forked SpCL)
  ├── Clustering module: DBSCAN + FINCH wrappers
  ├── AAS sampler: uncertainty regions, U_os, U_us, marginal distribution
  ├── NP3 algorithm: constrained cluster refinement
  └── Oracle simulator: GT-based pairwise annotation

Phase 3: Reproduction Experiments
  └── Run AAS on 13 datasets → compare against paper's claimed numbers
```

**Repository structure:**
```
ALS/
├── CLAUDE.md
├── data/                        # gitignored raw/processed images
├── src/
│   ├── data/                    # dataset loaders, split logic
│   ├── clustering/              # DBSCAN + FINCH wrappers
│   ├── aas/                     # AAS sampler + NP3
│   ├── oracle/                  # GT-based oracle simulator
│   └── eval/                    # metrics + reporting
├── experiments/
│   ├── configs/                 # YAML configs
│   ├── run_aas.py               # main experiment runner
│   └── results/                 # CSV/JSON output (gitignored large files)
├── notebooks/
│   ├── reproduce_table1.ipynb
│   └── budget_curves.ipynb
├── docs/superpowers/specs/
└── requirements.txt
```

---

## 2. Data Infrastructure

**Source:** WildlifeReID-10k (Adam et al. 2025) via `wildlife-datasets` pip package. 13 datasets, none used to train the foundation models.

**Experimental protocol (exactly as paper):**
- **Gallery:** up to 5 exemplar images per individual, from 80% of training individuals, selected by MegaDescriptor cosine similarity
- **Query:** full test set (open-set — may contain unseen individuals)
- **Held-out:** 20% of training individuals (absent from gallery, enables open-set evaluation)

**Modules:**
- `src/data/download.py` — fetch all 13 datasets via wildlife-datasets toolkit
- `src/data/dataset.py` — unified `WildlifeDataset` class: `(image, identity_id, dataset_name)`
- `src/data/splits.py` — gallery/query/held-out split logic
- `src/data/transforms.py` — resize 256×128, ImageNet normalization

**Gitignore:** `data/raw/`, `data/processed/`. Only split index files (JSON/CSV) committed.

---

## 3. Core AAS Implementation

### 3.1 Clustering Module (`src/clustering/`)

- `dbscan.py` — sklearn DBSCAN wrapper. Interface: `fit(features: np.ndarray) -> np.ndarray` (cluster IDs, -1 = outlier)
- `finch.py` — official FINCH wrapper, first partition level. Same interface.

### 3.2 AAS Sampler (`src/aas/`)

- `uncertainty_regions.py` — partial IoU between DBSCAN and FINCH clusters → transitive closure → regions S = {S₁, ..., S_M}
- `over_seg_sampler.py` — per region: find medoid, query k_max nearest medoid neighbors with similarity ≥ s_min → U_os
- `under_seg_sampler.py` — symmetric difference of cluster pairs per region, filtered by closest inter-cluster pairs (P_cand) → U_us
- `sampler.py` — combines U_os + U_us, marginal distribution P(Y) with weight ε, samples B pairs per AL cycle

### 3.3 NP3 Algorithm (`src/aas/np3.py`)

Input: cluster labels + must-link (ML) / cannot-link (CL) constraint pairs
1. Merge clusters to satisfy all ML constraints
2. For each impure cluster: build conflict graph of ML groups → graph coloring → Hungarian matching for unconstrained points
Output: refined cluster labels satisfying all constraints

### 3.4 Oracle Simulator (`src/oracle/gt_oracle.py`)

Given sampled pair (x_u, x_v) and GT identity labels:
- Same ID → must-link
- Different ID → cannot-link

Simulates the human annotator deterministically from ground truth.

---

## 4. Experiments & Evaluation

### 4.1 What we run

- ResNet-50 (ImageNet pretrained) as backbone — no fine-tuning, just feature extraction starting point
- SpCL base training with AAS active sampling
- 5 AL cycles, active sampling every 10 epochs, 50 total training epochs
- Budget: 0.02% of all pairwise combinations per AL cycle

### 4.2 Target numbers (from paper Table 1, AAS row)

| Metric | Paper value |
|--------|------------|
| mAP | 56.14% |
| mINP | 38.17% |
| BAKS | 67.15% |
| AUCROC | 75.21% |
| Top-1 | 67.71% |
| Top-3 | 79.08% |
| Top-5 | 85.04% |
| Top-10 | 91.18% |
| Budget used | 0.033% |

Results averaged over 4 runs per dataset, then macro-averaged over 13 datasets.

### 4.3 Experiment runner

- `experiments/configs/aas.yaml` — all hyperparameters
- `experiments/run_aas.py` — single entry point
- `experiments/results/` — per-dataset CSVs + aggregate JSON

### 4.4 Fixed hyperparameters (from paper)

| Parameter | Value |
|-----------|-------|
| ε (over/under-seg balance) | 0.6 |
| k_max (nearest medoid neighbors) | 5 |
| s_min (similarity threshold) | 0.3 |
| AL cycles | 5 |
| Epochs per cycle | 10 |
| Budget per cycle | 0.02% of pairs |
| Backbone | ResNet-50 |
| Base USL method | SpCL |

### 4.5 Analysis notebooks

- `notebooks/reproduce_table1.ipynb` — our numbers vs. paper's AAS row
- `notebooks/budget_curves.ipynb` — mAP vs. AL cycle + actual budget utilized per cycle

---

## 5. Key Dependencies

- PyTorch + torchvision
- `wildlife-datasets` (WildlifeReID-10k toolkit)
- `scikit-learn` (DBSCAN)
- `finch-clust` (FINCH clustering)
- SpCL (forked, base USL training)
- `networkx` (conflict graph construction in NP3)
- `scipy` (Hungarian algorithm)
- `numpy`, `pandas`, `matplotlib`

---

## 6. Out of Scope

- TFC tiger dataset application
- Comparison against baseline/USL/AL methods
- Person Re-ID datasets (Market-1501, Person-X)
- Any new model architecture
