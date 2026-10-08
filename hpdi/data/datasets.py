"""BMP datasets preserving the original independently sorted pairing order."""

from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


def image_paths(root):
    """Return sorted BMP paths, with early errors for missing or empty folders."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError("Image directory does not exist: {}".format(root))
    paths = sorted(root.glob("*.bmp"))
    if not paths:
        raise ValueError("No BMP images found in {}".format(root))
    return paths


def read_image(path, transform=None):
    """Close file handles without converting image mode or changing pixels."""
    with Image.open(path) as image:
        image = image.copy()
    return transform(image) if transform is not None else image


class PairedDataset(Dataset):
    """Pair measurements and labels by independently sorted positions."""

    def __init__(self, root1, root2, transform=None, target_transform=None):
        inputs, targets = image_paths(root1), image_paths(root2)
        if len(inputs) != len(targets):
            raise ValueError(
                "Mismatch between number of images ({}) and labels ({})".format(
                    len(inputs), len(targets)
                )
            )
        self.imgs = list(zip(inputs, targets))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        input_path, target_path = self.imgs[index]
        return (
            read_image(input_path, self.transform),
            read_image(target_path, self.target_transform),
        )

    def __len__(self):
        return len(self.imgs)


class FusionDataset(Dataset):
    """Pair RAW, CTF, and labels using the original sorting convention."""

    def __init__(
        self,
        root1,
        root2,
        root3,
        transform_diffuse=None,
        transform_generation=None,
        target_transform=None,
    ):
        raw, ctf, targets = image_paths(root1), image_paths(root2), image_paths(root3)
        if len(raw) != len(targets) or len(ctf) != len(targets):
            raise ValueError("Mismatch between number of RAW/CTF images and labels")
        self.imgs = list(zip(raw, ctf, targets))
        self.transform_diffuse = transform_diffuse
        self.transform_generation = transform_generation
        self.target_transform = target_transform

    def __getitem__(self, index):
        raw_path, ctf_path, target_path = self.imgs[index]
        return (
            read_image(raw_path, self.transform_diffuse),
            read_image(ctf_path, self.transform_generation),
            read_image(target_path, self.target_transform),
        )

    def __len__(self):
        return len(self.imgs)


class ImageDataset(Dataset):
    """Unlabelled RAW measurements for FISTA preprocessing."""

    def __init__(self, root, transform=None):
        self.paths = image_paths(root)
        self.transform = transform

    def __getitem__(self, index):
        return read_image(self.paths[index], self.transform)

    def __len__(self):
        return len(self.paths)
