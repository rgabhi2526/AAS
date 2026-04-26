import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

try:
    import umap
except ImportError:
    umap = None
    print("Warning: umap-learn not installed. Please run `pip install umap-learn`. Visualization will be skipped.")

def plot_epoch_umap(epoch: int, features: np.ndarray, pseudo_labels: np.ndarray, gt_labels: np.ndarray, output_dir: str):
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
    
    # Compute UMAP projection
    reducer = umap.UMAP(random_state=42, n_components=2)
    embedding = reducer.fit_transform(features)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f"Epoch {epoch} Feature Space UMAP", fontsize=16)
    
    # Left subplot: Ground Truth
    unique_gt = np.unique(gt_labels)
    cmap_gt = plt.cm.get_cmap('tab20', len(unique_gt))
    for i, gt in enumerate(unique_gt):
        mask = (gt_labels == gt)
        ax1.scatter(embedding[mask, 0], embedding[mask, 1], 
                    color=cmap_gt(i), label=str(gt), alpha=0.7, s=15)
    ax1.set_title(f"Ground Truth Identities (n={len(unique_gt)})")
    ax1.axis('off')
    
    # Right subplot: Pseudo-labels
    unique_pl = np.unique(pseudo_labels)
    # Mask outliers (-1)
    valid_pl = unique_pl[unique_pl >= 0]
    cmap_pl = plt.cm.get_cmap('nipy_spectral', len(valid_pl))
    
    # Plot outliers in grey
    outlier_mask = (pseudo_labels < 0)
    if outlier_mask.any():
        ax2.scatter(embedding[outlier_mask, 0], embedding[outlier_mask, 1], 
                    color='gray', label='Outliers', alpha=0.3, s=10)
        
    # Plot valid clusters
    for i, pl in enumerate(valid_pl):
        mask = (pseudo_labels == pl)
        ax2.scatter(embedding[mask, 0], embedding[mask, 1], 
                    color=cmap_pl(i), label=str(pl), alpha=0.7, s=15)
    
    n_outliers = outlier_mask.sum()
    ax2.set_title(f"Pseudo-labels (Clusters: {len(valid_pl)}, Outliers: {n_outliers})")
    ax2.axis('off')
    
    # Save plot
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"epoch_{epoch:02d}_umap.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [Viz] Saved UMAP plot to {out_path}")
