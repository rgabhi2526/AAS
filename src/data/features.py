"""Feature extraction using ResNet-50 or MegaDescriptor backbones."""
import torch
import timm
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional


def get_device() -> str:
    """Return best available device: cuda > mps (Apple Silicon) > cpu."""
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def extract_features(
    dataset,
    backbone: str = 'resnet50',
    batch_size: int = 64,
    device: Optional[str] = None,
    num_workers: int = 0,
) -> np.ndarray:
    """
    Extract L2-normalised features from every image in a dataset.

    GPU/MPS recommended for large datasets; falls back to CPU automatically.

    Args:
        dataset:     PyTorch dataset returning (image_tensor, label, idx)
        backbone:    'resnet50'       — ImageNet-pretrained ResNet-50 (2048-d)
                     'megadescriptor' — BVRA/MegaDescriptor-L-384 from HF hub
        batch_size:  inference batch size (reduce if OOM)
        device:      'cuda' | 'mps' | 'cpu' | None (auto-detect)
        num_workers: DataLoader workers (0 = main process, safe on Mac/Colab)

    Returns:
        features: (n, d) float32 array of L2-normalised feature vectors
    """
    if device is None:
        device = get_device()

    if backbone == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=0)
    elif backbone == 'megadescriptor':
        model = timm.create_model(
            'hf-hub:BVRA/MegaDescriptor-L-384', pretrained=True, num_classes=0
        )
    else:
        raise ValueError(f"Unknown backbone: {backbone!r}. Use 'resnet50' or 'megadescriptor'.")

    model = model.to(device).eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device == 'cuda'),
    )

    all_feats = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Extracting [{backbone}] on {device}'):
            images = batch[0].to(device)
            feats = model(images)
            feats = torch.nn.functional.normalize(feats, dim=1)
            all_feats.append(feats.cpu().numpy())

    return np.concatenate(all_feats, axis=0).astype(np.float32)
