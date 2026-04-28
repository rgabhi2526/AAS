import os
os.environ['MPLBACKEND'] = 'Agg'
import matplotlib
try:
    matplotlib.use('Agg', force=True)
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import umap
except ImportError:
    umap = None
    print("Warning: umap-learn not installed. Please run `pip install umap-learn`. Visualization will be skipped.")

_LEGEND_THRESHOLD = 30  # show per-class legend only if n_unique <= this


def _add_legend_or_colorbar(ax, fig, scatter, labels, unique_vals, cmap, title):
    """Add a legend (few classes) or colorbar (many classes)."""
    n = len(unique_vals)
    if n <= _LEGEND_THRESHOLD:
        handles = [
            mpatches.Patch(color=cmap(i / max(n - 1, 1)), label=str(v))
            for i, v in enumerate(unique_vals)
        ]
        ax.legend(
            handles=handles,
            title=title,
            loc='upper right',
            fontsize=7,
            title_fontsize=8,
            ncol=max(1, n // 15),
            markerscale=1.2,
            framealpha=0.6,
        )
    else:
        cbar = fig.colorbar(scatter, ax=ax, pad=0.02, shrink=0.85)
        cbar.set_label(title, fontsize=9)
        cbar.ax.tick_params(labelsize=7)


def plot_epoch_umap(epoch: int, features: np.ndarray, pseudo_labels: np.ndarray,
                    gt_labels: np.ndarray, output_dir: str):
    """
    Generate side-by-side UMAP projections of features, colored by GT and Pseudo-labels.

    Args:
        epoch: Current epoch number.
        features: (N, 2048) L2-normalized feature array.
        pseudo_labels: (N,) Array of pseudo-cluster assignments.
        gt_labels: (N,) Array of ground truth identities.
        output_dir: Directory to save the PNG plot.
    """
    if umap is None:
        return

    print(f"  [Viz] Generating UMAP plot for epoch {epoch}...")

    reducer = umap.UMAP(random_state=42, n_components=2)
    embedding = reducer.fit_transform(features)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.suptitle(f"Epoch {epoch} Feature Space UMAP", fontsize=16)

    # ── Left: Ground Truth ──────────────────────────────────────────────────
    unique_gt = np.unique(gt_labels)
    n_gt = len(unique_gt)
    cmap_gt = plt.cm.get_cmap('tab20' if n_gt <= 20 else 'nipy_spectral', n_gt)
    gt_to_idx = {v: i for i, v in enumerate(unique_gt)}
    c_gt = np.array([gt_to_idx[l] for l in gt_labels], dtype=float)

    sc1 = ax1.scatter(
        embedding[:, 0], embedding[:, 1],
        c=c_gt, cmap=cmap_gt, vmin=0, vmax=n_gt - 1,
        alpha=0.7, s=15,
    )
    ax1.set_title(f"Ground Truth Identities (n={n_gt})", fontsize=12)
    ax1.axis('off')
    _add_legend_or_colorbar(ax1, fig, sc1, gt_labels, unique_gt, cmap_gt, "Identity")

    # ── Right: Pseudo-labels ────────────────────────────────────────────────
    valid_pl = np.unique(pseudo_labels[pseudo_labels >= 0])
    n_pl = len(valid_pl)
    cmap_pl = plt.cm.get_cmap('nipy_spectral', max(n_pl, 1))
    pl_to_idx = {v: i for i, v in enumerate(valid_pl)}

    outlier_mask = pseudo_labels < 0
    n_outliers = int(outlier_mask.sum())

    # Plot outliers first (grey, behind clusters)
    if outlier_mask.any():
        ax2.scatter(
            embedding[outlier_mask, 0], embedding[outlier_mask, 1],
            color='lightgray', alpha=0.3, s=10, zorder=1, label='Outliers',
        )

    # Plot valid clusters
    c_pl = np.array([pl_to_idx[l] for l in pseudo_labels if l >= 0], dtype=float)
    valid_emb = embedding[~outlier_mask]
    sc2 = ax2.scatter(
        valid_emb[:, 0], valid_emb[:, 1],
        c=c_pl, cmap=cmap_pl, vmin=0, vmax=max(n_pl - 1, 1),
        alpha=0.7, s=15, zorder=2,
    )

    ax2.set_title(f"Pseudo-labels (Clusters: {n_pl}, Outliers: {n_outliers})", fontsize=12)
    ax2.axis('off')

    if n_pl <= _LEGEND_THRESHOLD:
        handles = [
            mpatches.Patch(color=cmap_pl(i / max(n_pl - 1, 1)), label=str(v))
            for i, v in enumerate(valid_pl)
        ]
        if outlier_mask.any():
            handles.insert(0, mpatches.Patch(color='lightgray', label='Outliers'))
        ax2.legend(
            handles=handles,
            title="Cluster",
            loc='upper right',
            fontsize=7,
            title_fontsize=8,
            ncol=max(1, (n_pl + 1) // 15),
            markerscale=1.2,
            framealpha=0.6,
        )
    else:
        cbar = fig.colorbar(sc2, ax=ax2, pad=0.02, shrink=0.85)
        cbar.set_label("Cluster ID", fontsize=9)
        cbar.ax.tick_params(labelsize=7)
        if outlier_mask.any():
            ax2.scatter([], [], color='lightgray', label=f'Outliers ({n_outliers})', s=15)
            ax2.legend(loc='lower right', fontsize=8, framealpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(output_dir, f"epoch_{epoch:02d}_umap.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Viz] Saved UMAP plot to {out_path}")
