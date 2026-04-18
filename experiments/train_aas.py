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
from src.data.features import extract_features as extract_features_timm, get_device


# ---------------------------------------------------------------------------
# NaN Diagnostic Logger
# ---------------------------------------------------------------------------

def _diag(name: str, t, log_file: str = "experiments/results/nan_diag.log"):
    """Log tensor/array stats for NaN debugging."""
    if isinstance(t, np.ndarray):
        t = torch.tensor(t)
    if not isinstance(t, torch.Tensor):
        line = f"[DIAG] {name}: type={type(t).__name__} value={t}"
    elif t.numel() == 0:
        line = f"[DIAG] {name}: EMPTY shape={tuple(t.shape)}"
    else:
        tf = t.float().detach().cpu()
        line = (
            f"[DIAG] {name}: shape={tuple(t.shape)} dtype={t.dtype} "
            f"min={tf.min().item():.6f} max={tf.max().item():.6f} "
            f"mean={tf.mean().item():.6f} std={tf.std().item():.6f} "
            f"nan={torch.isnan(tf).sum().item()} inf={torch.isinf(tf).sum().item()}"
        )
        if tf.ndim == 2:
            norms = tf.norm(dim=1)
            line += (
                f" | row_norms: min={norms.min().item():.4f} "
                f"max={norms.max().item():.4f} mean={norms.mean().item():.4f}"
            )
    print(line)
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(line + "\n")

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


@torch.no_grad()
def extract_features_model(model, loader):
    """Extract L2-normalised features using the SpCL training model.

    Args:
        model:  the encoder (moved to eval mode internally).
        loader: a pre-built DataLoader (reuse across calls to avoid
                repeated worker spawn/teardown).
    """
    model.eval()
    all_feats = []
    for batch in loader:
        imgs = batch[0].cuda()
        feats = model(imgs)
        feats = torch.nn.functional.normalize(feats, dim=1)
        all_feats.append(feats.cpu())
    model.train()
    return torch.cat(all_feats, dim=0)


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
    parser.add_argument('--dataset-txt', default='dataset.txt',
                        help='path to dataset.txt listing datasets to use')
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
    df_all = load_metadata(args.data_root, dataset_txt=args.dataset_txt)
    df = df_all[df_all['dataset'] == args.dataset].copy()

    if df.empty:
        available = df_all['dataset'].unique().tolist() if 'dataset' in df_all.columns else 'unknown column'
        raise ValueError(
            f"Dataset '{args.dataset}' not found.\n"
            f"Available: {available}\n"
            f"Check dataset.txt at {args.dataset_txt}."
        )

    # Identity encoding is handled inside WildlifeSubsetDataset; factorize
    # here so split logic downstream sees consistent integer labels.
    df['identity'] = df['identity'].factorize()[0]

    train_df = df[df['split'] == 'train'].reset_index(drop=True)

    # ── Shared config values ──────────────────────────────────────────────────
    batch_size  = cfg.get('batch_size', 64)
    train_iters = cfg.get('train_iters', 400)
    _nw = cfg.get('num_workers', 4)
    val_transform  = get_transforms('val')
    train_transform = get_transforms('train')

    # ── Single eval dataset (reused for gallery split, init, and per-epoch) ──
    train_ds_eval = WildlifeSubsetDataset(train_df, root=args.data_root,
                                          transform=val_transform)

    # ── Extract pretrained features ONCE (used for gallery split + memory init)
    print("Extracting pretrained features (gallery selection + memory init)...")
    pretrained_feats = extract_features_timm(train_ds_eval, backbone='resnet50',
                                             device=device, num_workers=_nw)

    gallery_df, query_df, held_out_df = make_splits(
        df,
        held_out_fraction=cfg.get('held_out_fraction', 0.2),
        max_exemplars=cfg.get('max_exemplars', 5),
        embeddings=pretrained_feats,
        seed=cfg['seed'] + args.run_id,
    )
    print(f"Split — train: {len(train_df)} | gallery: {len(gallery_df)} | "
          f"query: {len(query_df)} | held-out: {len(held_out_df)}")

    # ── SpCL model + training ────────────────────────────────────────────────
    # NOTE: SpCL must be cloned to third_party/SpCL/
    # Run: cd third_party && git clone https://github.com/yxgeee/SpCL.git
    # third_party/SpCL is already on sys.path (line 25), so import directly.
    try:
        from third_party.SpCL.spcl.models.resnet import ResNet
        from third_party.SpCL.spcl.models.hm import HybridMemory
        from third_party.SpCL.spcl.trainers import SpCLTrainer_USL
        from third_party.SpCL.spcl.utils.data import IterLoader
        from third_party.SpCL.spcl.utils.faiss_rerank import compute_jaccard_distance
    except ImportError as e:
        raise ImportError(
            "SpCL not found. Clone it first:\n"
            "  cd third_party && git clone https://github.com/yxgeee/SpCL.git\n"
            "  pip install -e third_party/SpCL/"
        ) from e

    from sklearn.cluster import DBSCAN as skDBSCAN
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, RandomSampler

    # WildlifeSubsetDataset returns (img, label, idx) — SpCLTrainer_USL._parse_data
    # expects 5-tuple (imgs, _, pids, _, indexes).  Wrap with a collate adapter.
    class _SpCLBatchAdapter(torch.utils.data.Dataset):
        """Wraps WildlifeSubsetDataset to return the 5-tuple SpCL expects."""
        def __init__(self, base_ds):
            self._ds = base_ds
        def __len__(self):
            return len(self._ds)
        def __getitem__(self, idx):
            img, label, sample_idx = self._ds[idx]
            # SpCL _parse_data: imgs, _, pids, _, indexes
            return img, 0, label, 0, sample_idx

    NUM_FEATURES = 2048
    model = ResNet(depth=50, num_features=NUM_FEATURES, dropout=0, num_classes=0)
    model = model.to(device)

    memory = HybridMemory(
        NUM_FEATURES,
        len(train_df),
        temp=cfg.get('temp', 0.05),
        momentum=cfg.get('momentum', 0.2),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.get('lr', 3.5e-4),
        weight_decay=cfg.get('weight_decay', 5e-4),
    )

    trainer = SpCLTrainer_USL(model, memory)

    gt_labels = train_df['identity'].values
    al_interval = cfg.get('al_interval', 10)
    total_epochs = cfg.get('total_epochs', 50)
    al_cycle = 0
    total_pairs_used = 0
    cycle_metrics = []

    # ── Reset diagnostic log ────────────────────────────────────────────────
    diag_path = "experiments/results/nan_diag.log"
    os.makedirs(os.path.dirname(diag_path), exist_ok=True)
    with open(diag_path, "w") as f:
        f.write(f"=== NaN diagnostic: {args.dataset} run={args.run_id} ===\n")

    # ── Initialise memory with pretrained features (prevents NaN at epoch 0) ──
    # Reuse the features already extracted above (no second timm model / pass).
    memory.features = F.normalize(
        torch.tensor(pretrained_feats, dtype=torch.float32), dim=1
    ).to(device)
    print(f"Memory initialised: {memory.features.shape}")
    _diag("memory.features_pretrained_init", memory.features)

    # ── Build DataLoaders once (reuse across all epochs) ─────────────────────
    # Eval loader: used for per-epoch feature extraction (50× reused)
    eval_loader = DataLoader(
        train_ds_eval,
        batch_size=batch_size,
        shuffle=False,
        num_workers=_nw,
        pin_memory=True,
        persistent_workers=(_nw > 0),
    )

    # Training loader: share preloaded images from eval dataset (no 2nd disk read)
    train_ds_train = WildlifeSubsetDataset(
        train_df, root=args.data_root,
        transform=train_transform,
        shared_images=train_ds_eval._images if train_ds_eval._preloaded else None,
    )
    adapted_ds = _SpCLBatchAdapter(train_ds_train)
    train_loader = IterLoader(
        DataLoader(
            adapted_ds,
            batch_size=batch_size,
            num_workers=_nw,
            sampler=RandomSampler(adapted_ds, replacement=True,
                                  num_samples=batch_size * train_iters),
            pin_memory=True,
            drop_last=True,
            persistent_workers=(_nw > 0),
        ),
        length=train_iters,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(total_epochs):
        print(f"\n[Epoch {epoch + 1}/{total_epochs}]")

        # Extract features with the *training* model (reuses eval_loader)
        feat_tensor = extract_features_model(model, eval_loader)
        features = feat_tensor.numpy()
        _diag(f"epoch{epoch}_feat_tensor", feat_tensor)

        # Sync hybrid memory features (already L2-normalised)
        memory.features = feat_tensor.cuda()
        _diag(f"epoch{epoch}_memory.features", memory.features)

        # Generate SpCL pseudo-labels via Jaccard-distance DBSCAN
        rerank_dist = compute_jaccard_distance(
            memory.features.clone(),
            k1=cfg.get('k1', 30),
            k2=cfg.get('k2', 6),
        )
        pseudo_labels = skDBSCAN(
            eps=cfg.get('pseudo_eps', 0.6),
            min_samples=4,
            metric='precomputed',
            n_jobs=-1,
        ).fit_predict(rerank_dist)

        _n_valid   = int((pseudo_labels >= 0).sum())
        _n_outlier = int((pseudo_labels < 0).sum())
        _n_clusters = int(pseudo_labels.max() + 1) if _n_valid > 0 else 0
        print(f"  DBSCAN: {_n_clusters} clusters, {_n_valid} valid, {_n_outlier} outliers "
              f"(total={len(pseudo_labels)})")
        _diag(f"epoch{epoch}_pseudo_labels", pseudo_labels)

        # ── AAS injection every al_interval epochs ─────────────────────────
        if epoch > 0 and (epoch + 1) % al_interval == 0:
            print(f"  Running AAS cycle {al_cycle + 1}...")
            pseudo_labels, n_pairs = run_al_cycle(
                features, gt_labels, pseudo_labels, cfg, al_cycle
            )
            total_pairs_used += n_pairs
            al_cycle += 1

        # Update memory labels (assign outliers to unique classes — vectorised)
        n_valid = int((pseudo_labels >= 0).sum())
        n_clusters = int(pseudo_labels.max() + 1) if n_valid > 0 else 0
        labels_for_memory = pseudo_labels.copy()
        outlier_mask = labels_for_memory < 0
        labels_for_memory[outlier_mask] = np.arange(
            n_clusters, n_clusters + outlier_mask.sum(), dtype=labels_for_memory.dtype
        )
        memory.labels = torch.tensor(labels_for_memory, dtype=torch.long).to(device)
        _diag(f"epoch{epoch}_memory.labels", memory.labels)

        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=cfg.get('print_freq', 50),
                      train_iters=train_iters)
        _diag(f"epoch{epoch}_memory.features_post_train", memory.features)

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating...")
    gallery_ds = WildlifeSubsetDataset(gallery_df, root=args.data_root,
                                        transform=val_transform)
    query_ds   = WildlifeSubsetDataset(query_df,   root=args.data_root,
                                        transform=val_transform)

    gallery_loader = DataLoader(gallery_ds, batch_size=batch_size, shuffle=False,
                                num_workers=_nw, pin_memory=True)
    query_loader   = DataLoader(query_ds,   batch_size=batch_size, shuffle=False,
                                num_workers=_nw, pin_memory=True)

    gallery_feats = extract_features_model(model, gallery_loader).numpy()
    query_feats   = extract_features_model(model, query_loader).numpy()

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
