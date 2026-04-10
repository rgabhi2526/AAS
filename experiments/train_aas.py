"""
AAS-enhanced SpCL training script.

Runs SpCL USL training and injects one AAS cycle every `al_interval` epochs.

Usage (Colab / GPU machine):
    python3 experiments/train_aas.py \\
        --config experiments/configs/aas.yaml \\
        --dataset IPanda50 \\
        --data-root data/raw \\
        --output-dir experiments/results \\
        --run-id 0
"""
import sys
import os
import argparse
import json
import yaml
import numpy as np
import torch

# Make project root importable
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, 'third_party', 'SpCL'))

from src.aas.sampler import run_aas
from src.aas.np3 import refine_labels
from src.oracle.gt_oracle import GTOracle
from src.eval.metrics import compute_metrics
from src.data.download import load_metadata
from src.data.dataset import WildlifeSubsetDataset
from src.data.transforms import get_transforms
from src.data.splits import make_splits
from src.data.features import extract_features, get_device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def compute_budget(n_samples: int, budget_fraction: float) -> int:
    """B = budget_fraction × (n choose 2)."""
    n_pairs = n_samples * (n_samples - 1) // 2
    return max(1, int(n_pairs * budget_fraction))


def run_al_cycle(
    features: np.ndarray,
    gt_labels: np.ndarray,
    pseudo_labels: np.ndarray,
    cfg: dict,
    cycle: int,
) -> tuple:
    """
    One AAS active-learning cycle:
      1. Run AAS  →  sample B pairs
      2. Query GT oracle  →  ML / CL constraints
      3. Refine pseudo-labels with NP3

    Returns:
        refined_labels: (n,) np.ndarray
        n_pairs:        actual number of pairs queried
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

    print(f"  [Cycle {cycle}] budget={budget} | queried={len(pairs)} | "
          f"ML={len(must_links)} | CL={len(cannot_links)}")

    refined = refine_labels(pseudo_labels, must_links, cannot_links)
    return refined, len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='AAS Re-ID reproduction')
    parser.add_argument('--config',     required=True)
    parser.add_argument('--dataset',    required=True,
                        help='Dataset name matching metadata.csv dataset column')
    parser.add_argument('--data-root',  required=True,
                        help='Root dir of WildlifeReID-10k (contains metadata.csv)')
    parser.add_argument('--output-dir', default='experiments/results')
    parser.add_argument('--run-id',     type=int, default=0,
                        help='Run index 0-3 (used for seed offset and output filename)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    device = get_device()
    print(f"\n=== AAS: {args.dataset} | Run {args.run_id} | Device: {device} ===\n")

    # ── Data ─────────────────────────────────────────────────────────────────
    df_all = load_metadata(args.data_root)
    df = df_all[df_all['dataset'] == args.dataset].copy()

    if df.empty:
        available = df_all['dataset'].unique().tolist() if 'dataset' in df_all.columns else 'unknown column'
        raise ValueError(
            f"Dataset '{args.dataset}' not found.\n"
            f"Available: {available}\n"
            f"Update PAPER_DATASETS_13 in src/data/download.py if needed."
        )

    # Factorize identity labels to contiguous 0-indexed integers
    df['identity'] = df['identity'].factorize()[0]

    train_df = df[df['split'] == 'train'].reset_index(drop=True)

    # ── Gallery split (needs MegaDescriptor embeddings) ──────────────────────
    print("Extracting MegaDescriptor features for gallery selection...")
    train_ds_val = WildlifeSubsetDataset(train_df, root=args.data_root,
                                          transform=get_transforms('val'))
    mega_feats = extract_features(train_ds_val, backbone='megadescriptor',
                                  device=device, num_workers=2)

    gallery_df, query_df, held_out_df = make_splits(
        df,
        held_out_fraction=cfg.get('held_out_fraction', 0.2),
        max_exemplars=cfg.get('max_exemplars', 5),
        embeddings=mega_feats,
        seed=cfg['seed'] + args.run_id,
    )
    print(f"Split — train: {len(train_df)} | gallery: {len(gallery_df)} | "
          f"query: {len(query_df)} | held-out: {len(held_out_df)}")

    # ── SpCL model + training ────────────────────────────────────────────────
    # NOTE: SpCL must be cloned to third_party/SpCL/
    # Run: cd third_party && git clone https://github.com/yxgeee/SpCL.git
    try:
        from spcl.models.resnet import ResNet
        from spcl.trainers import SpCLTrainer
        from spcl.utils.faiss_rerank import compute_jaccard_distance
    except ImportError as e:
        raise ImportError(
            "SpCL not found. Clone it first:\n"
            "  cd third_party && git clone https://github.com/yxgeee/SpCL.git\n"
            "  pip install -e third_party/SpCL/"
        ) from e

    from sklearn.cluster import DBSCAN as skDBSCAN

    model = ResNet(depth=50, num_features=2048, dropout=0, num_classes=0)
    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get('lr', 3.5e-4),
        weight_decay=cfg.get('weight_decay', 5e-4),
    )

    gt_labels = train_df['identity'].values
    al_interval = cfg.get('al_interval', 10)
    total_epochs = cfg.get('total_epochs', 50)
    al_cycle = 0
    total_pairs_used = 0
    cycle_metrics = []

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(total_epochs):
        print(f"\n[Epoch {epoch + 1}/{total_epochs}]")

        # Extract features with current model
        train_ds_eval = WildlifeSubsetDataset(
            train_df, root=args.data_root, transform=get_transforms('val')
        )
        features = extract_features(train_ds_eval, backbone='resnet50',
                                    device=device, num_workers=2)

        # Generate SpCL pseudo-labels via Jaccard-distance DBSCAN
        rerank_dist = compute_jaccard_distance(
            torch.tensor(features).to(device),
            k1=cfg.get('k1', 30),
            k2=cfg.get('k2', 6),
        )
        pseudo_labels = skDBSCAN(
            eps=cfg.get('pseudo_eps', 0.6),
            min_samples=4,
            metric='precomputed',
            n_jobs=-1,
        ).fit_predict(rerank_dist.cpu().numpy())

        # ── AAS injection every al_interval epochs ─────────────────────────
        if epoch > 0 and (epoch + 1) % al_interval == 0:
            print(f"  Running AAS cycle {al_cycle + 1}...")
            pseudo_labels, n_pairs = run_al_cycle(
                features, gt_labels, pseudo_labels, cfg, al_cycle
            )
            total_pairs_used += n_pairs
            al_cycle += 1

        # Train one epoch with SpCL
        n_classes = int((pseudo_labels[pseudo_labels >= 0]).max() + 1) if (pseudo_labels >= 0).any() else 1
        trainer = SpCLTrainer(
            model,
            num_classes=n_classes,
            memory_size=len(train_df),
            device=device,
        )
        train_ds_train = WildlifeSubsetDataset(
            train_df, root=args.data_root, transform=get_transforms('train')
        )
        trainer.train(epoch, train_ds_train, pseudo_labels, optimizer)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating...")
    gallery_ds = WildlifeSubsetDataset(gallery_df, root=args.data_root,
                                        transform=get_transforms('val'))
    query_ds   = WildlifeSubsetDataset(query_df,   root=args.data_root,
                                        transform=get_transforms('val'))

    gallery_feats = extract_features(gallery_ds, backbone='resnet50', device=device)
    query_feats   = extract_features(query_ds,   backbone='resnet50', device=device)

    gallery_ids_set = set(gallery_df['identity'].unique())
    query_is_known  = np.array([lbl in gallery_ids_set for lbl in query_df['identity'].values])

    metrics = compute_metrics(
        query_feats, gallery_feats,
        query_df['identity'].values, gallery_df['identity'].values,
        query_is_known=query_is_known,
    )

    n_all_pairs = len(train_df) * (len(train_df) - 1) // 2
    actual_budget_pct = (total_pairs_used / n_all_pairs * 100) if n_all_pairs > 0 else 0

    print(f"\n=== Results: {args.dataset} | Run {args.run_id} ===")
    for k, v in metrics.items():
        print(f"  {k:8s}: {v * 100:.2f}%")
    print(f"  Budget used: {actual_budget_pct:.4f}%  ({total_pairs_used} pairs)")

    # Save
    out = {
        'dataset':        args.dataset,
        'run':            args.run_id,
        'metrics':        metrics,
        'total_pairs':    total_pairs_used,
        'budget_pct':     actual_budget_pct,
    }
    out_path = os.path.join(args.output_dir, f"{args.dataset}_run{args.run_id}.json")
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == '__main__':
    main()
