# ALS — Active Learning for Animal Re-Identification

## Project Overview

This project reproduces the results of the paper:
**"Active Learning for Animal Re-Identification with Ambiguity-Aware Sampling"**
(Sani, Khurana, Anand — IIIT Delhi, AAAI 2026)

**Goal:** Reproduce the AAS method's claimed Table 1 results on 13 WildlifeReID-10k datasets.

The paper's reference PDF is at: `active_learning .pdf`
The design spec is at: `docs/superpowers/specs/2026-04-09-aas-reproduction-design.md`

---

## Repository Structure

```
ALS/
├── CLAUDE.md                        # this file
├── active_learning .pdf             # reference paper
├── data/                            # datasets (large files gitignored)
│   ├── raw/                         # downloaded wildlife datasets
│   ├── processed/                   # cropped/resized images
│   └── splits/                      # gallery/query/held-out index files (committed)
├── src/
│   ├── data/
│   │   ├── download.py              # fetch 13 WildlifeReID-10k datasets
│   │   ├── dataset.py               # unified WildlifeDataset class
│   │   ├── splits.py                # gallery/query/held-out split logic
│   │   └── transforms.py           # image preprocessing (256x128, ImageNet norm)
│   ├── clustering/
│   │   ├── dbscan.py               # sklearn DBSCAN wrapper
│   │   └── finch.py                # FINCH wrapper (first partition)
│   ├── aas/
│   │   ├── uncertainty_regions.py  # partial IoU → transitive closure → regions S
│   │   ├── over_seg_sampler.py     # U_os: medoid nearest-neighbor pairs
│   │   ├── under_seg_sampler.py    # U_us: symmetric diff filtered by P_cand
│   │   ├── sampler.py              # marginal distribution P(Y), samples B pairs
│   │   └── np3.py                  # NP3: constrained cluster refinement
│   ├── oracle/
│   │   └── gt_oracle.py            # GT-based must-link/cannot-link simulator
│   └── eval/
│       └── metrics.py              # mAP, mINP, BAKS, AUCROC, Top-k
├── experiments/
│   ├── configs/
│   │   └── aas.yaml                # all hyperparameters
│   ├── run_aas.py                  # main experiment entry point
│   └── results/                    # output CSVs + JSON (gitignored large files)
├── notebooks/
│   ├── reproduce_table1.ipynb      # our numbers vs. paper's Table 1 AAS row
│   └── budget_curves.ipynb         # mAP vs. AL cycle + budget utilization
├── docs/superpowers/specs/
│   └── 2026-04-09-aas-reproduction-design.md
└── requirements.txt
```

---

## Key Concepts

### AAS (Ambiguity-Aware Sampling)
An active learning strategy that identifies the most informative image pairs to annotate by finding **disagreements between two complementary clustering algorithms** (DBSCAN and FINCH).

- **Regions of uncertainty (S):** Sets of images whose cluster assignments are inconsistent across the two methods (partial IoU between clusters)
- **U_os (over-segmentation pool):** Pairs of medoids from different uncertain regions that may belong to the same individual
- **U_us (under-segmentation pool):** Inconsistent pairs within uncertain regions that may need to be separated
- **NP3:** Post-hoc algorithm that refines cluster labels using the oracle's must-link / cannot-link feedback

### Fixed Hyperparameters (from paper)
| Parameter | Value |
|-----------|-------|
| ε (over/under-seg balance) | 0.6 |
| k_max | 5 |
| s_min | 0.3 |
| AL cycles | 5 |
| Epochs total | 50 (active sampling every 10) |
| Budget per cycle | 0.02% of all pairs |
| Backbone | ResNet-50 |
| Base USL | SpCL |

### Target Results (Table 1, AAS row — averaged over 13 datasets, 4 runs)
| mAP | mINP | BAKS | AUCROC | Top-1 | Top-5 | Budget |
|-----|------|------|--------|-------|-------|--------|
| 56.14% | 38.17% | 67.15% | 75.21% | 67.71% | 85.04% | 0.033% |

---

## Experimental Protocol (exactly as paper)

1. For each of the 13 datasets:
   - Gallery: up to 5 exemplars per individual from 80% of training IDs (selected by MegaDescriptor similarity)
   - Query: full test set
   - Held-out: 20% of training IDs (absent from gallery — open-set condition)
2. Train SpCL base model
3. Every 10 epochs: run AAS → query oracle → refine pseudo-labels via NP3 → continue training
4. Evaluate after 5 AL cycles
5. Average over 4 independent runs

---

## Development Notes

- Keep `src/` modules independent and unit-testable — each module has one clear purpose
- Clustering wrappers share a common interface: `fit(features: np.ndarray) -> np.ndarray`
- Oracle simulator is deterministic (uses GT labels) — no human-in-the-loop needed for reproduction
- `data/raw/` and `data/processed/` are gitignored; only split index files are committed
- All hyperparameters live in `experiments/configs/aas.yaml` — do not hardcode them in src/

---

## Dependencies

- PyTorch + torchvision
- `wildlife-datasets` (WildlifeReID-10k official toolkit)
- `scikit-learn` (DBSCAN)
- `finch-clust` (FINCH clustering)
- SpCL (forked repo, base USL training pipeline)
- `networkx` (conflict graph in NP3)
- `scipy` (Hungarian algorithm)
- `numpy`, `pandas`, `matplotlib`

---

## Out of Scope

- TFC tiger dataset
- Comparison against baseline / USL / other AL methods
- Person Re-ID datasets (Market-1501, Person-X)
- New model architectures

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
