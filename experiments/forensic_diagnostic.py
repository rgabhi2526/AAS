"""
Over-Segmentation Forensic Diagnostic — Experiments 1, 2, 3, 5
================================================================
Paste this entire script into a Colab cell AFTER a training run completes
(or after loading a checkpoint). It uses the model and data already in memory.

If running standalone, set the paths below and it will load a checkpoint.

Outputs a detailed report to stdout.
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

# ── Config: adjust these if running standalone ─────────────────────────────
CONFIG_PATH   = "experiments/configs/aas.yaml"
DATASET       = "CowDataset"
DATA_ROOT     = "data/raw"
RUN_ID        = 0
CKPT_PATH     = None  # Set to checkpoint path if loading from disk

# ── Make imports work ──────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath("__file__"))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
    sys.path.insert(0, os.path.join(_ROOT, 'third_party', 'SpCL'))

import yaml
from sklearn.cluster import DBSCAN as skDBSCAN
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter, defaultdict

from src.data.download import load_metadata
from src.data.dataset import WildlifeSubsetDataset
from src.data.transforms import get_transforms
from src.data.features import extract_features as extract_features_timm, get_device

try:
    from third_party.SpCL.spcl.utils.faiss_rerank import compute_jaccard_distance
except ImportError:
    print("WARNING: SpCL not importable — Jaccard distance experiments will fail")

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  LOAD DATA + FEATURES                                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def load_data_and_features():
    """Load dataset and extract features (or use checkpoint)."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    df_all = load_metadata(DATA_ROOT, dataset_txt=cfg.get('dataset_txt', 'dataset.txt'))
    df = df_all[df_all['dataset'] == DATASET].copy()
    df['identity'] = df['identity'].factorize()[0]
    train_df = df[df['split'] == 'train'].reset_index(drop=True)

    gt_labels = train_df['identity'].values
    n_gt = len(np.unique(gt_labels))

    print(f"\n{'='*70}")
    print(f"  FORENSIC DIAGNOSTIC: {DATASET}")
    print(f"  Samples: {len(train_df)} | GT identities: {n_gt}")
    print(f"{'='*70}\n")

    # Extract fresh features with pretrained ResNet-50
    device = get_device()
    val_transform = get_transforms('val')
    ds = WildlifeSubsetDataset(train_df, root=DATA_ROOT, transform=val_transform)

    print("Extracting ResNet-50 features...")
    features = extract_features_timm(ds, backbone='resnet50',
                                      device=device, num_workers=2)
    features = features / np.linalg.norm(features, axis=1, keepdims=True)  # ensure L2-norm

    return train_df, gt_labels, features, cfg


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 1: Forensic Snapshot — Per-Stage Cluster Count               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def experiment_1_forensic_snapshot(features, gt_labels, cfg):
    """Trace cluster count through every pipeline stage."""
    print("\n" + "="*70)
    print("  EXPERIMENT 1: Forensic Snapshot — Where Do Clusters Come From?")
    print("="*70)

    n = len(features)
    feat_tensor = torch.tensor(features, dtype=torch.float32)

    # Stage 1: Jaccard distance
    print("\n[Stage 1] Computing Jaccard distance...")
    rerank_dist = compute_jaccard_distance(
        F.normalize(feat_tensor, dim=1).cuda(),
        k1=cfg.get('k1', 30),
        k2=cfg.get('k2', 6),
    )

    # Stage 2: DBSCAN on Jaccard
    print("\n[Stage 2] DBSCAN clustering...")
    pseudo_eps = cfg.get('pseudo_eps', 0.6)
    pseudo_labels = skDBSCAN(
        eps=pseudo_eps, min_samples=4, metric='precomputed', n_jobs=-1
    ).fit_predict(rerank_dist)

    n_dbscan = int(pseudo_labels.max() + 1) if (pseudo_labels >= 0).any() else 0
    n_outliers = int((pseudo_labels < 0).sum())
    n_valid = int((pseudo_labels >= 0).sum())

    print(f"  DBSCAN: {n_dbscan} clusters, {n_valid} valid, {n_outliers} outliers")
    print(f"  ARI vs GT: {adjusted_rand_score(gt_labels, pseudo_labels):.4f}")
    print(f"  NMI vs GT: {normalized_mutual_info_score(gt_labels, pseudo_labels):.4f}")

    # Stage 3: ML merge (simulate with empty constraints for baseline)
    print(f"\n[Stage 3] ML merge (0 constraints — baseline)")
    print(f"  Clusters unchanged: {n_dbscan}")

    # Stage 4: NP3 (no constraints — baseline)
    print(f"\n[Stage 4] NP3 (0 constraints — baseline)")
    print(f"  Clusters unchanged: {n_dbscan}")

    # Stage 5: Outlier relabeling (current approach)
    labels_singleton = pseudo_labels.copy()
    outlier_mask = labels_singleton < 0
    labels_singleton[outlier_mask] = np.arange(
        n_dbscan, n_dbscan + outlier_mask.sum(), dtype=labels_singleton.dtype
    )
    n_memory_classes = int(labels_singleton.max() + 1)

    print(f"\n[Stage 5] Outlier relabeling (current: each outlier = singleton)")
    print(f"  Classes in memory: {n_memory_classes}")
    print(f"  = {n_dbscan} DBSCAN clusters + {n_outliers} singleton outliers")
    print(f"  INFLATION: {n_memory_classes - n_dbscan} classes added by outliers")

    # Stage 5 alt: Assign outliers to nearest cluster
    labels_nearest = pseudo_labels.copy()
    if n_outliers > 0 and n_dbscan > 0:
        # Compute cluster centroids
        centroids = {}
        for c in range(n_dbscan):
            mask = pseudo_labels == c
            if mask.any():
                centroid = features[mask].mean(axis=0)
                centroid /= np.linalg.norm(centroid)
                centroids[c] = centroid

        # Assign each outlier to nearest centroid
        outlier_indices = np.where(pseudo_labels < 0)[0]
        for idx in outlier_indices:
            best_c, best_sim = 0, -2.0
            for c, centroid in centroids.items():
                sim = float(features[idx] @ centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_c = c
            labels_nearest[idx] = best_c

    n_memory_nearest = int(labels_nearest.max() + 1) if (labels_nearest >= 0).any() else 0

    print(f"\n[Stage 5 ALT] Outlier → nearest cluster")
    print(f"  Classes in memory: {n_memory_nearest}")
    print(f"  ARI vs GT (singleton): {adjusted_rand_score(gt_labels, labels_singleton):.4f}")
    print(f"  ARI vs GT (nearest):   {adjusted_rand_score(gt_labels, labels_nearest):.4f}")
    print(f"  NMI vs GT (singleton): {normalized_mutual_info_score(gt_labels, labels_singleton):.4f}")
    print(f"  NMI vs GT (nearest):   {normalized_mutual_info_score(gt_labels, labels_nearest):.4f}")

    # Summary
    print(f"\n{'─'*70}")
    print(f"  SUMMARY:")
    print(f"    GT identities:           {len(np.unique(gt_labels))}")
    print(f"    DBSCAN clusters:         {n_dbscan}")
    print(f"    Memory classes (current): {n_memory_classes}")
    print(f"    Memory classes (nearest): {n_memory_nearest}")
    print(f"    Over-segmentation ratio:  {n_memory_classes / len(np.unique(gt_labels)):.1f}x (current)")
    print(f"                              {n_memory_nearest / len(np.unique(gt_labels)):.1f}x (nearest)")
    print(f"{'─'*70}")

    return pseudo_labels, rerank_dist


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 2: DBSCAN Sensitivity Analysis                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def experiment_2_dbscan_sweep(features, gt_labels, rerank_dist):
    """Sweep DBSCAN eps and report cluster count + quality metrics."""
    print("\n" + "="*70)
    print("  EXPERIMENT 2: DBSCAN eps Sensitivity")
    print("="*70)

    eps_values = [0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]

    print(f"\n  {'eps':>5s} | {'clusters':>8s} | {'outliers':>8s} | {'valid':>6s} | {'ARI':>6s} | {'NMI':>6s}")
    print(f"  {'─'*5}-+-{'─'*8}-+-{'─'*8}-+-{'─'*6}-+-{'─'*6}-+-{'─'*6}")

    best_ari = -1
    best_eps = 0

    for eps in eps_values:
        labels = skDBSCAN(
            eps=eps, min_samples=4, metric='precomputed', n_jobs=-1
        ).fit_predict(rerank_dist)

        n_c = int(labels.max() + 1) if (labels >= 0).any() else 0
        n_out = int((labels < 0).sum())
        n_val = int((labels >= 0).sum())
        ari = adjusted_rand_score(gt_labels, labels)
        nmi = normalized_mutual_info_score(gt_labels, labels)

        marker = " ◀ current" if eps == 0.6 else ""
        if ari > best_ari:
            best_ari = ari
            best_eps = eps
            if eps != 0.6:
                marker += " ★ best ARI"

        print(f"  {eps:5.2f} | {n_c:8d} | {n_out:8d} | {n_val:6d} | {ari:6.3f} | {nmi:6.3f}{marker}")

    print(f"\n  Best ARI: eps={best_eps:.2f} (ARI={best_ari:.3f})")

    # Also try min_samples sweep at current eps
    print(f"\n  min_samples sweep (eps=0.6 fixed):")
    print(f"  {'ms':>4s} | {'clusters':>8s} | {'outliers':>8s} | {'ARI':>6s}")
    print(f"  {'─'*4}-+-{'─'*8}-+-{'─'*8}-+-{'─'*6}")

    for ms in [2, 3, 4, 5, 6, 8, 10]:
        labels = skDBSCAN(
            eps=0.6, min_samples=ms, metric='precomputed', n_jobs=-1
        ).fit_predict(rerank_dist)
        n_c = int(labels.max() + 1) if (labels >= 0).any() else 0
        n_out = int((labels < 0).sum())
        ari = adjusted_rand_score(gt_labels, labels)
        marker = " ◀ current" if ms == 4 else ""
        print(f"  {ms:4d} | {n_c:8d} | {n_out:8d} | {ari:6.3f}{marker}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 3: Intra-Identity Feature Consistency                         ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def experiment_3_identity_consistency(features, gt_labels, pseudo_labels):
    """For each GT identity, measure embedding consistency."""
    print("\n" + "="*70)
    print("  EXPERIMENT 3: Intra-Identity Feature Consistency")
    print("="*70)

    unique_ids = np.unique(gt_labels)
    n_ids = len(unique_ids)

    print(f"\n  {'ID':>4s} | {'n':>5s} | {'intra_sim':>10s} | {'min_sim':>8s} | {'std_sim':>8s} | "
          f"{'DBSCAN_clusters':>15s} | {'fragments':>10s}")
    print(f"  {'─'*4}-+-{'─'*5}-+-{'─'*10}-+-{'─'*8}-+-{'─'*8}-+-{'─'*15}-+-{'─'*10}")

    all_intra = []
    all_inter = []
    fragmented_ids = []

    for gid in unique_ids:
        mask = gt_labels == gid
        id_feats = features[mask]
        n_samples = mask.sum()

        # Intra-identity cosine similarity
        if n_samples > 1:
            sim_matrix = id_feats @ id_feats.T
            # Upper triangle (excluding diagonal)
            triu_idx = np.triu_indices(n_samples, k=1)
            pairwise_sims = sim_matrix[triu_idx]
            mean_sim = pairwise_sims.mean()
            min_sim = pairwise_sims.min()
            std_sim = pairwise_sims.std()
            all_intra.extend(pairwise_sims.tolist())
        else:
            mean_sim = 1.0
            min_sim = 1.0
            std_sim = 0.0

        # How many DBSCAN clusters does this GT identity span?
        id_pseudo = pseudo_labels[mask]
        unique_clusters = np.unique(id_pseudo[id_pseudo >= 0])
        n_fragments = len(unique_clusters)
        n_outlier_in_id = (id_pseudo < 0).sum()

        if n_fragments > 1:
            fragmented_ids.append((gid, n_fragments, n_samples))

        frag_str = f"{n_fragments} (+{n_outlier_in_id} outliers)" if n_outlier_in_id > 0 else str(n_fragments)

        print(f"  {gid:4d} | {n_samples:5d} | {mean_sim:10.4f} | {min_sim:8.4f} | "
              f"{std_sim:8.4f} | {frag_str:>15s} | "
              f"{'⚠ SPLIT' if n_fragments > 1 else '✓'}")

    # Inter-identity similarities (sample 500 pairs to keep it fast)
    rng = np.random.default_rng(42)
    for _ in range(min(500, n_ids * (n_ids - 1) // 2)):
        i, j = rng.choice(len(features), 2, replace=False)
        if gt_labels[i] != gt_labels[j]:
            all_inter.append(float(features[i] @ features[j]))

    all_intra = np.array(all_intra)
    all_inter = np.array(all_inter)

    print(f"\n  {'─'*70}")
    print(f"  SUMMARY:")
    print(f"    Intra-identity sim: mean={all_intra.mean():.4f}, "
          f"min={all_intra.min():.4f}, std={all_intra.std():.4f}")
    print(f"    Inter-identity sim: mean={all_inter.mean():.4f}, "
          f"max={all_inter.max():.4f}, std={all_inter.std():.4f}")
    print(f"    Overlap zone:       intra_min ({all_intra.min():.4f}) vs "
          f"inter_max ({all_inter.max():.4f})")
    if all_intra.min() < all_inter.max():
        print(f"    ⚠ OVERLAP DETECTED: some same-identity pairs are LESS similar "
              f"than different-identity pairs")
    print(f"    Fragmented identities: {len(fragmented_ids)}/{n_ids}")
    for gid, n_frag, n_samp in fragmented_ids:
        print(f"      ID {gid}: split into {n_frag} DBSCAN clusters ({n_samp} samples)")
    print(f"  {'─'*70}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  EXPERIMENT 5: Constraint Coverage Analysis                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def experiment_5_constraint_coverage(gt_labels, cfg):
    """Analyze whether constraint budget can cover the needed merges."""
    print("\n" + "="*70)
    print("  EXPERIMENT 5: Constraint Coverage Analysis")
    print("="*70)

    n = len(gt_labels)
    unique_ids = np.unique(gt_labels)
    n_ids = len(unique_ids)

    total_pairs = n * (n - 1) // 2
    budget_frac = cfg.get('budget_fraction', 0.002)
    budget_per_cycle = int(total_pairs * budget_frac)
    al_interval = cfg.get('al_interval', 10)
    total_epochs = cfg.get('total_epochs', 50)
    n_cycles = total_epochs // al_interval + 1

    # Count same-identity pairs (ML needed) vs different-identity pairs (CL)
    id_counts = Counter(gt_labels)
    ml_pairs_needed = sum(c * (c - 1) // 2 for c in id_counts.values())
    cl_pairs_total = total_pairs - ml_pairs_needed

    # From the log: actual constraint counts
    actual_ml = 645
    actual_cl = 5577
    actual_total = actual_ml + actual_cl

    # ML per identity
    ml_per_id = actual_ml / n_ids if n_ids > 0 else 0

    # Unique samples covered by ML constraints (estimate)
    # Each ML constraint connects 2 samples. With 645 MLs, maximum coverage
    # is 645 * 2 = 1290 sample mentions, but with overlap ~645 unique samples
    # out of 1019 total

    print(f"\n  Total samples:          {n}")
    print(f"  GT identities:          {n_ids}")
    print(f"  Total possible pairs:   {total_pairs:,}")
    print(f"  Same-identity pairs:    {ml_pairs_needed:,} ({ml_pairs_needed/total_pairs*100:.1f}%)")
    print(f"  Diff-identity pairs:    {cl_pairs_total:,} ({cl_pairs_total/total_pairs*100:.1f}%)")
    print(f"\n  Budget per cycle:       {budget_per_cycle}")
    print(f"  Cycles run:             {n_cycles}")
    print(f"  Total queried:          {actual_total}")
    print(f"  Coverage:               {actual_total/total_pairs*100:.2f}%")
    print(f"\n  Actual ML constraints:  {actual_ml}")
    print(f"  Actual CL constraints:  {actual_cl}")
    print(f"  ML/CL ratio:            {actual_ml/max(actual_cl,1):.3f}")
    print(f"  ML per GT identity:     {ml_per_id:.1f}")

    # Key insight: ML constraints needed to merge fragments
    print(f"\n  {'─'*70}")
    print(f"  KEY QUESTION: Are 645 ML constraints enough to merge {n_ids} identities?")
    print(f"    If DBSCAN splits each identity into ~3 fragments on average,")
    print(f"    we need at least {n_ids * 2} ML constraints ({n_ids} ids × 2 bridge links).")
    print(f"    We have {actual_ml} → {'SUFFICIENT' if actual_ml >= n_ids * 2 else 'INSUFFICIENT'}")
    print(f"    BUT: are they spread evenly across identities?")

    # Samples per identity
    print(f"\n  Samples per GT identity:")
    for gid in unique_ids:
        count = id_counts[gid]
        print(f"    ID {gid:3d}: {count:5d} samples ({count/n*100:.1f}%)")
    print(f"  {'─'*70}")


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  RUN ALL EXPERIMENTS                                                      ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    train_df, gt_labels, features, cfg = load_data_and_features()

    pseudo_labels, rerank_dist = experiment_1_forensic_snapshot(features, gt_labels, cfg)
    experiment_2_dbscan_sweep(features, gt_labels, rerank_dist)
    experiment_3_identity_consistency(features, gt_labels, pseudo_labels)
    experiment_5_constraint_coverage(gt_labels, cfg)

    print("\n\n" + "="*70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("  Copy the output above and share it for analysis.")
    print("="*70)
