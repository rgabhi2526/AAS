"""Standard Re-ID image preprocessing transforms."""
import torchvision.transforms as T


def get_transforms(split: str = 'train') -> T.Compose:
    """
    Return torchvision transforms for Re-ID.

    Both splits resize to 256×128 and apply ImageNet normalisation.
    Training additionally adds a random horizontal flip.

    Args:
        split: 'train'  →  resize + flip + normalise
               anything else  →  resize + normalise

    Returns:
        A torchvision Compose transform
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
