"""Standard Re-ID image preprocessing transforms."""
import torchvision.transforms as T


def get_transforms(split: str = 'train') -> T.Compose:
    """
    Return torchvision transforms for Re-ID.

    Both splits resize to 256×128 and apply ImageNet normalisation.
    Training additionally adds a random horizontal flip.

    The 'megadescriptor' split uses 384×384 (MegaDescriptor-L-384 input size)
    for gallery exemplar selection as per the paper protocol.

    Args:
        split: 'train'           →  256×128 + flip + normalise
               'megadescriptor'  →  384×384 + normalise (for gallery exemplar selection)
               anything else     →  256×128 + normalise

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
    if split == 'megadescriptor':
        return T.Compose([
            T.Resize((384, 384)),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        normalize,
    ])
