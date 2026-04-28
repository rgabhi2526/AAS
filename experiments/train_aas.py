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
import collections
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


def self_paced_pseudo_labels(
    rerank_dist: np.ndarray,
    eps: float,
    eps_gap: float = 0.02,
    indep_thres: float = None,
    indep_thres_pct: float = 0.9,
) -> tuple:
    """SpCL's three-DBSCAN self-paced pseudo-label generation.

    Runs DBSCAN at three epsilon values (tight, normal, loose) to measure
    cluster reliability.  Unreliable samples are demoted to singletons.
    This is the "self-paced" curriculum: only confident cluster assignments
    contribute to contrastive learning; uncertain samples are maintained as
    individual instances until features improve enough for DBSCAN to assign
    them reliably.

    Reference: Ge et al., "Self-paced Contrastive Learning with Hybrid
    Memory for Domain Adaptive Object Re-ID", NeurIPS 2020.

    Args:
        rerank_dist:      (n, n) precomputed Jaccard distance matrix.
        eps:              DBSCAN epsilon for the normal clustering.
        eps_gap:          offset for tight (eps-gap) and loose (eps+gap).
        indep_thres:      R_indep threshold; None = auto-compute (epoch 0).
        indep_thres_pct:  percentile for auto-threshold (default 0.9).

    Returns:
        pseudo_labels:  (n,) numpy int64 array.  Reliable cluster members
                        keep IDs 0..K-1; unreliable samples get unique
                        singleton IDs ≥ K.
        indep_thres:    the threshold used (cache for subsequent epochs).
        stats:          dict with diagnostic counts.
    """
    from sklearn.cluster import DBSCAN as skDBSCAN

    eps_tight = eps - eps_gap
    eps_loose = eps + eps_gap

    # --- Three DBSCAN runs ------------------------------------------------
    pl       = skDBSCAN(eps=eps,       min_samples=4, metric='precomputed', n_jobs=-1).fit_predict(rerank_dist)
    pl_tight = skDBSCAN(eps=eps_tight, min_samples=4, metric='precomputed', n_jobs=-1).fit_predict(rerank_dist)
    pl_loose = skDBSCAN(eps=eps_loose, min_samples=4, metric='precomputed', n_jobs=-1).fit_predict(rerank_dist)

    num_ids       = len(set(pl))       - (1 if -1 in pl else 0)
    num_ids_tight = len(set(pl_tight)) - (1 if -1 in pl_tight else 0)
    num_ids_loose = len(set(pl_loose)) - (1 if -1 in pl_loose else 0)

    n_outliers_raw = int((pl == -1).sum())

    # --- Assign outliers to unique singletons (first pass) ----------------
    def _assign_singletons(cluster_ids, n_clusters):
        labels = []
        outliers = 0
        for cid in cluster_ids:
            if cid != -1:
                labels.append(cid)
            else:
                labels.append(n_clusters + outliers)
                outliers += 1
        return torch.tensor(labels, dtype=torch.long)

    pseudo_labels       = _assign_singletons(pl, num_ids)
    pseudo_labels_tight = _assign_singletons(pl_tight, num_ids_tight)
    pseudo_labels_loose = _assign_singletons(pl_loose, num_ids_loose)

    # --- Compute R_indep and R_comp ---------------------------------------
    N = pseudo_labels.size(0)
    label_sim       = pseudo_labels.expand(N, N).eq(pseudo_labels.expand(N, N).t()).float()
    label_sim_tight = pseudo_labels_tight.expand(N, N).eq(pseudo_labels_tight.expand(N, N).t()).float()
    label_sim_loose = pseudo_labels_loose.expand(N, N).eq(pseudo_labels_loose.expand(N, N).t()).float()

    R_comp  = 1 - torch.min(label_sim, label_sim_tight).sum(-1) / torch.max(label_sim, label_sim_tight).sum(-1)
    R_indep = 1 - torch.min(label_sim, label_sim_loose).sum(-1) / torch.max(label_sim, label_sim_loose).sum(-1)
    assert (R_comp.min() >= 0) and (R_comp.max() <= 1)
    assert (R_indep.min() >= 0) and (R_indep.max() <= 1)

    # --- Cluster-level reliability scores ---------------------------------
    cluster_R_comp  = collections.defaultdict(list)
    cluster_R_indep = collections.defaultdict(list)
    cluster_img_num = collections.defaultdict(int)
    for i, (comp, indep, label) in enumerate(zip(R_comp, R_indep, pseudo_labels)):
        cluster_R_comp[label.item()].append(comp.item())
        cluster_R_indep[label.item()].append(indep.item())
        cluster_img_num[label.item()] += 1

    cluster_R_comp  = [min(cluster_R_comp[i])  for i in sorted(cluster_R_comp.keys())]
    cluster_R_indep = [min(cluster_R_indep[i]) for i in sorted(cluster_R_indep.keys())]
    cluster_R_indep_noins = [
        iou for iou, num in zip(cluster_R_indep, sorted(cluster_img_num.keys()))
        if cluster_img_num[num] > 1
    ]

    # --- Threshold (auto-compute at epoch 0, reuse later) -----------------
    if indep_thres is None:
        if len(cluster_R_indep_noins) > 0:
            idx = min(len(cluster_R_indep_noins) - 1,
                      int(np.round(len(cluster_R_indep_noins) * indep_thres_pct)))
            indep_thres = np.sort(cluster_R_indep_noins)[idx]
        else:
            indep_thres = 0.5  # fallback

    # --- Self-paced filtering (second pass) -------------------------------
    n_demoted = 0
    filtered_labels = pseudo_labels.clone()
    for i, label in enumerate(pseudo_labels):
        _indep = cluster_R_indep[label.item()]
        _comp  = R_comp[i].item()
        if _indep <= indep_thres and _comp <= cluster_R_comp[label.item()]:
            pass  # reliable — keep label
        else:
            filtered_labels[i] = len(cluster_R_indep) + n_demoted
            n_demoted += 1

    # --- Statistics -------------------------------------------------------
    index2label = collections.defaultdict(int)
    for lbl in filtered_labels:
        index2label[lbl.item()] += 1
    counts = np.fromiter(index2label.values(), dtype=float)
    n_clusters_final = int((counts > 1).sum())
    n_singletons     = int((counts == 1).sum())

    stats = {
        'n_clusters_dbscan': num_ids,
        'n_clusters_tight':  num_ids_tight,
        'n_clusters_loose':  num_ids_loose,
        'n_outliers_raw':    n_outliers_raw,
        'n_demoted':         n_demoted,
        'n_clusters_final':  n_clusters_final,
        'n_singletons':      n_singletons,
        'n_total_classes':   n_clusters_final + n_singletons,
        'indep_thres':       indep_thres,
    }

    return filtered_labels.numpy(), indep_thres, stats


def log_feature_quality(
    features: np.ndarray,
    gt_labels: np.ndarray,
    pseudo_labels: np.ndarray,
    epoch: int,
) -> None:
    """Log per-epoch intra/inter identity cosine similarity (Fix 3).

    Tracks whether training is improving or degrading the feature space.
    Key diagnostic: if intra-identity sim increases and inter-identity sim
    decreases over epochs, features are consolidating (good). If the gap
    shrinks or intra drops, features are degenerating.
    """
    unique_ids = np.unique(gt_labels)
    intra_sims = []
    inter_sims = []

    # Intra-identity: pairwise cosine sim within each GT identity
    for gid in unique_ids:
        mask = gt_labels == gid
        id_feats = features[mask]
        n = id_feats.shape[0]
        if n > 1:
            sim_matrix = id_feats @ id_feats.T
            triu_idx = np.triu_indices(n, k=1)
            intra_sims.extend(sim_matrix[triu_idx].tolist())

    # Inter-identity: sample random cross-identity pairs (cap at 1000)
    rng = np.random.default_rng(epoch)
    for _ in range(min(1000, len(features) * 2)):
        i, j = rng.choice(len(features), 2, replace=False)
        if gt_labels[i] != gt_labels[j]:
            inter_sims.append(float(features[i] @ features[j]))

    intra = np.array(intra_sims) if intra_sims else np.array([0.0])
    inter = np.array(inter_sims) if inter_sims else np.array([0.0])
    gap = intra.mean() - inter.mean()

    n_pseudo = int(pseudo_labels.max() + 1) if (pseudo_labels >= 0).any() else 0

    print(f"  [FeatureQ] epoch={epoch} | "
          f"intra: mean={intra.mean():.4f} min={intra.min():.4f} | "
          f"inter: mean={inter.mean():.4f} max={inter.max():.4f} | "
          f"gap={gap:.4f} | pseudo_clusters={n_pseudo}")


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
    all_must_links: list,
    all_cannot_links: list,
) -> tuple:
    """
    One AAS active-learning cycle:
      1. Run AAS  →  sample B pairs (constrained by existing ML/CL)
      2. Query GT oracle  →  ML / CL constraints
      3. Accumulate constraints with all previous cycles
      4. Refine pseudo-labels with NP3 using ALL accumulated constraints

    The `all_must_links` and `all_cannot_links` lists are mutated in-place
    (extended with this cycle's new constraints).

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
        existing_ml=all_must_links,
        existing_cl=all_cannot_links,
        finch_partition=cfg.get('finch_partition', 0),
    )

    oracle = GTOracle(gt_labels)
    new_ml, new_cl = oracle.query(pairs)

    # Accumulate constraints (mutates the lists held by caller)
    all_must_links.extend(new_ml)
    all_cannot_links.extend(new_cl)

    print(f"  [Cycle {cycle}] budget={budget} | queried={len(pairs)} | "
          f"new_ML={len(new_ml)} | new_CL={len(new_cl)} | "
          f"total_ML={len(all_must_links)} | total_CL={len(all_cannot_links)}")

    # Refine with ALL accumulated constraints
    refined = refine_labels(pseudo_labels, all_must_links, all_cannot_links,
                            features=features)
    return refined, len(pairs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _ckpt_path(output_dir, dataset, run_id, tag='latest'):
    """Return checkpoint file path for given tag ('latest' or 'best')."""
    return os.path.join(output_dir, f"{dataset}_run{run_id}_ckpt_{tag}.pth")


def save_checkpoint(path, epoch, model, optimizer, memory, al_cycle,
                    total_pairs_used, best_loss, consec_low, **extra):
    """Save training state to disk."""
    ckpt = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'memory_features': memory.features.cpu(),
        'memory_labels': memory.labels.cpu(),
        'al_cycle': al_cycle,
        'total_pairs_used': total_pairs_used,
        'best_loss': best_loss,
        'consec_low': consec_low,
    }
    ckpt.update(extra)
    torch.save(ckpt, path)


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
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from the latest checkpoint')
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

    # ── Single eval dataset (reused for memory init and per-epoch extraction) ──
    train_ds_eval = WildlifeSubsetDataset(train_df, root=args.data_root,
                                          transform=val_transform)

    # ── Gallery exemplar selection with MegaDescriptor (paper protocol) ───────
    # Paper: "up to five exemplars per ID were selected ... using MegaDescriptor
    # based similarity." Uses 384×384 input, one-time extraction then discarded.
    print("Extracting MegaDescriptor features (gallery exemplar selection)...")
    mega_transform = get_transforms('megadescriptor')
    mega_ds = WildlifeSubsetDataset(train_df, root=args.data_root,
                                    transform=mega_transform)
    gallery_feats = extract_features_timm(mega_ds, backbone='megadescriptor',
                                          device=device, num_workers=_nw)
    del mega_ds  # free memory

    gallery_df, query_df, held_out_df = make_splits(
        df,
        held_out_fraction=cfg.get('held_out_fraction', 0.2),
        max_exemplars=cfg.get('max_exemplars', 5),
        embeddings=gallery_feats,
        seed=cfg['seed'] + args.run_id,
    )
    del gallery_feats  # no longer needed
    print(f"Split — train: {len(train_df)} | gallery: {len(gallery_df)} | "
          f"query: {len(query_df)} | held-out: {len(held_out_df)}")

    # ── ResNet-50 features for SpCL memory init ──────────────────────────────
    print("Extracting ResNet-50 features (memory init)...")
    pretrained_feats = extract_features_timm(train_ds_eval, backbone='resnet50',
                                             device=device, num_workers=_nw)

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

    import torch.nn.functional as F
    from torch.utils.data import DataLoader, RandomSampler

    # WildlifeSubsetDataset returns (img, label, idx) — SpCLTrainer_USL._parse_data
    # expects 5-tuple (imgs, _, pids, _, indexes).  Wrap with a collate adapter.
    class _SpCLBatchAdapter(torch.utils.data.Dataset):
        """Wraps WildlifeSubsetDataset to return the 5-tuple SpCL expects."""
        def __init__(self, base_ds, pseudo_labels=None):
            self._ds = base_ds
            self.pseudo_labels = pseudo_labels

        def __len__(self):
            return len(self._ds)

        def __getitem__(self, idx):
            img, gt_label, sample_idx = self._ds[idx]
            # Use pseudo-label if provided, otherwise gt_label
            pid = self.pseudo_labels[idx] if self.pseudo_labels is not None else gt_label
            return img, 0, pid, 0, sample_idx

    from torch.utils.data.sampler import Sampler

    class PKLabelSampler(Sampler):
        """SpCL-style PK sampler ensuring K instances per P identities in a batch."""
        def __init__(self, pseudo_labels, num_instances=4):
            self.pseudo_labels = pseudo_labels
            self.num_instances = num_instances
            self.index_dic = collections.defaultdict(list)
            for index, pid in enumerate(pseudo_labels):
                self.index_dic[pid].append(index)
            self.pids = list(self.index_dic.keys())
            self.num_samples = len(self.pids)

        def __len__(self):
            return self.num_samples * self.num_instances

        def __iter__(self):
            indices = torch.randperm(self.num_samples).tolist()
            ret = []
            for i in indices:
                pid = self.pids[i]
                t = self.index_dic[pid]
                if len(t) >= self.num_instances:
                    t = np.random.choice(t, size=self.num_instances, replace=False)
                else:
                    t = np.random.choice(t, size=self.num_instances, replace=True)
                ret.extend(t)
            return iter(ret)

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

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.get('step_size', 20), gamma=0.1
    )

    trainer = SpCLTrainer_USL(model, memory)

    gt_labels = train_df['identity'].values
    al_interval = cfg.get('al_interval', 10)
    total_epochs = cfg.get('total_epochs', 50)
    al_cycle = 0
    total_pairs_used = 0
    cycle_metrics = []
    start_epoch = 0
    best_loss = float('inf')
    consec_low = 0  # consecutive epochs below early-stop threshold
    indep_thres = None  # SpCL self-paced R_indep threshold (auto-computed at epoch 0)

    # Accumulated constraints across all AAS cycles (Fix 1)
    all_must_links = []
    all_cannot_links = []

    # ── Early stopping config ────────────────────────────────────────────────
    es_threshold = cfg.get('early_stop_threshold', 0.005)
    es_patience  = cfg.get('early_stop_patience', 5)
    es_min_epoch = cfg.get('early_stop_min_epoch', 15)

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

    # ── Resume from checkpoint (if requested) ────────────────────────────────
    # ── Per-window early stopping config ──────────────────────────────────
    ws_threshold = cfg.get('window_stop_threshold', 0.01)
    ws_patience  = cfg.get('window_stop_patience', 2)
    window_counter  = 0
    skip_to_next_aas = False

    ckpt_latest = _ckpt_path(args.output_dir, args.dataset, args.run_id, 'latest')
    if args.resume and os.path.isfile(ckpt_latest):
        print(f"\nResuming from checkpoint: {ckpt_latest}")
        ckpt = torch.load(ckpt_latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        memory.features = ckpt['memory_features'].to(device)
        memory.labels = ckpt['memory_labels'].to(device)
        start_epoch = ckpt['epoch'] + 1
        al_cycle = ckpt.get('al_cycle', 0)
        total_pairs_used = ckpt.get('total_pairs_used', 0)
        best_loss = ckpt.get('best_loss', float('inf'))
        consec_low = ckpt.get('consec_low', 0)
        window_counter = ckpt.get('window_counter', 0)
        all_must_links = ckpt.get('all_must_links', [])
        all_cannot_links = ckpt.get('all_cannot_links', [])
        indep_thres = ckpt.get('indep_thres', None)
        print(f"  Resuming from epoch {start_epoch}, al_cycle={al_cycle}, "
              f"best_loss={best_loss:.6f}")
        del ckpt
    elif args.resume:
        print(f"WARNING: --resume set but no checkpoint found at {ckpt_latest}. "
              f"Starting from scratch.")

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
    def build_train_loader(p_labels):
        """Rebuild DataLoader every epoch to sample from new pseudo_labels."""
        adapted_ds = _SpCLBatchAdapter(train_ds_train, pseudo_labels=p_labels)
        sampler = PKLabelSampler(p_labels, num_instances=4)
        return IterLoader(
            DataLoader(
                adapted_ds,
                batch_size=batch_size,
                num_workers=_nw,
                sampler=sampler,
                pin_memory=True,
                drop_last=True,
                persistent_workers=(_nw > 0),
            ),
            length=train_iters,
        )

    # Initialise train_loader with None (will be built inside epoch loop)
    train_loader = None

    # ── Training loop ─────────────────────────────────────────────────────────
    # Patch the trainer to capture the epoch-avg loss for early stopping.
    # SpCLTrainer_USL.train() prints it but doesn't return it.
    _original_train = trainer.train.__func__
    _last_epoch_loss = [float('inf')]  # mutable container for closure

    def _train_with_loss_capture(self, epoch, data_loader, optimizer,
                                 print_freq=10, train_iters=400):
        """Wrapper that captures the epoch-average loss."""
        import time
        self.encoder.train()
        from third_party.SpCL.spcl.utils.meters import AverageMeter
        losses = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        end = time.time()
        for i in range(train_iters):
            inputs = data_loader.next()
            data_time.update(time.time() - end)
            inputs, _, indexes = self._parse_data(inputs)
            f_out = self._forward(inputs)
            loss = self.memory(f_out, indexes)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.update(loss.item())
            batch_time.update(time.time() - end)
            end = time.time()
            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))
        _last_epoch_loss[0] = losses.avg

    import types
    trainer.train = types.MethodType(_train_with_loss_capture, trainer)

    for epoch in range(start_epoch, total_epochs):
        # ── Per-window skip: if window converged, jump to next AAS epoch ──
        is_aas_epoch = (epoch == 0) or (epoch > 0 and (epoch + 1) % al_interval == 0)
        if skip_to_next_aas and not is_aas_epoch:
            print(f"\n[Epoch {epoch + 1}/{total_epochs}] ⏭ skipped (window converged)")
            continue
        if is_aas_epoch:
            # Reset window state — new AAS cycle will inject fresh labels
            skip_to_next_aas = False
            window_counter = 0

        print(f"\n[Epoch {epoch + 1}/{total_epochs}]")

        # Extract features with the *training* model (reuses eval_loader)
        feat_tensor = extract_features_model(model, eval_loader)
        features = feat_tensor.numpy()
        _diag(f"epoch{epoch}_feat_tensor", feat_tensor)

        # Sync hybrid memory features (already L2-normalised)
        memory.features = feat_tensor.cuda()
        _diag(f"epoch{epoch}_memory.features", memory.features)

        # SpCL self-paced pseudo-label generation (three-DBSCAN + reliability filter)
        rerank_dist = compute_jaccard_distance(
            memory.features.clone(),
            k1=cfg.get('k1', 30),
            k2=cfg.get('k2', 6),
        )
        pseudo_labels, indep_thres, sp_stats = self_paced_pseudo_labels(
            rerank_dist,
            eps=cfg.get('pseudo_eps', 0.6),
            eps_gap=cfg.get('eps_gap', 0.02),
            indep_thres=indep_thres if epoch > 0 else None,
            indep_thres_pct=cfg.get('indep_thres_pct', 0.9),
        )
        print(f"  Self-paced: {sp_stats['n_clusters_dbscan']} DBSCAN clusters, "
              f"{sp_stats['n_outliers_raw']} raw outliers → "
              f"{sp_stats['n_clusters_final']} reliable clusters + "
              f"{sp_stats['n_singletons']} singletons = "
              f"{sp_stats['n_total_classes']} total classes "
              f"(R_indep_thres={sp_stats['indep_thres']:.4f})")
        _diag(f"epoch{epoch}_pseudo_labels", pseudo_labels)

        # Apply accumulated ML constraints (can rescue singletons back into clusters)
        if all_must_links:
            from src.aas.np3 import _merge_must_links
            pre_merge = int(pseudo_labels.max() + 1)
            pseudo_labels = _merge_must_links(pseudo_labels.copy(), all_must_links)
            post_merge = int(pseudo_labels.max() + 1)
            if pre_merge != post_merge:
                print(f"  ML merge: {pre_merge} → {post_merge} classes "
                      f"({pre_merge - post_merge} merged)")

        # ── AAS injection every al_interval epochs (and epoch 0) ───────────
        aas_ran = False
        if epoch == 0 or (epoch > 0 and (epoch + 1) % al_interval == 0):
            print(f"  Running AAS cycle {al_cycle + 1}...")
            pseudo_labels, n_pairs = run_al_cycle(
                features, gt_labels, pseudo_labels, cfg, al_cycle,
                all_must_links, all_cannot_links,
            )
            total_pairs_used += n_pairs
            al_cycle += 1
            aas_ran = True

        # Synchronous UMAP visualization
        try:
            from src.visualization.umap_vis import plot_epoch_umap
            viz_dir = os.path.join(args.output_dir, args.dataset, str(args.run_id), 'visualizations')
            os.makedirs(viz_dir, exist_ok=True)
            _viz_gt = train_df['identity'].values
            plot_epoch_umap(
                epoch=epoch,
                features=features,
                pseudo_labels=pseudo_labels,
                gt_labels=_viz_gt,
                output_dir=viz_dir
            )
        except Exception as e:
            print(f"  [Viz] Failed to generate UMAP: {e}")

        # Set memory labels (self-paced: reliable clusters + singletons)
        memory.labels = torch.tensor(pseudo_labels, dtype=torch.long).to(device)
        _diag(f"epoch{epoch}_memory.labels", memory.labels)

        # Feature quality monitor (Fix 3: track intra/inter identity similarity)
        log_feature_quality(features, gt_labels, pseudo_labels, epoch)

        # ── Rebuild Dataloader with new pseudo-labels (PK sampling) ────────
        train_loader = build_train_loader(pseudo_labels)

        train_loader.new_epoch()
        trainer.train(epoch, train_loader, optimizer,
                      print_freq=cfg.get('print_freq', 50),
                      train_iters=train_iters)
        _diag(f"epoch{epoch}_memory.features_post_train", memory.features)

        lr_scheduler.step()

        # ── Checkpoint (latest + best) ─────────────────────────────────────
        epoch_loss = _last_epoch_loss[0]
        save_checkpoint(
            _ckpt_path(args.output_dir, args.dataset, args.run_id, 'latest'),
            epoch, model, optimizer, memory,
            al_cycle, total_pairs_used, best_loss, consec_low,
            window_counter=window_counter,
            all_must_links=all_must_links,
            all_cannot_links=all_cannot_links,
            indep_thres=indep_thres,
        )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(
                _ckpt_path(args.output_dir, args.dataset, args.run_id, 'best'),
                epoch, model, optimizer, memory,
                al_cycle, total_pairs_used, best_loss, consec_low,
                window_counter=window_counter,
                all_must_links=all_must_links,
                all_cannot_links=all_cannot_links,
                indep_thres=indep_thres,
            )
            print(f"  ★ New best loss: {best_loss:.6f} (saved best checkpoint)")

        # ── Per-window convergence check ────────────────────────────────────
        if not aas_ran and epoch_loss < ws_threshold:
            window_counter += 1
            if window_counter >= ws_patience:
                skip_to_next_aas = True
                print(f"  ⏭ Window converged (loss < {ws_threshold} for "
                      f"{ws_patience} consecutive epochs) — skipping to next AAS cycle")
        elif not aas_ran:
            window_counter = 0

        # ── Global early stopping ──────────────────────────────────────────
        if aas_ran:
            # AAS injection spikes loss; reset counter
            consec_low = 0
        elif epoch >= es_min_epoch and epoch_loss < es_threshold:
            consec_low += 1
            print(f"  Early-stop counter: {consec_low}/{es_patience} "
                  f"(loss={epoch_loss:.6f} < {es_threshold})")
        else:
            consec_low = 0

        if consec_low >= es_patience:
            print(f"\n✓ Early stopping at epoch {epoch + 1} "
                  f"(loss < {es_threshold} for {es_patience} consecutive epochs)")
            break

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
