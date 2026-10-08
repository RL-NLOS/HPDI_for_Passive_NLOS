"""Shared loaders; keep ToTensor, shuffle, and drop_last settings unchanged."""

from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from hpdi.runtime import worker_init

from .datasets import FusionDataset, PairedDataset


def make_dataset(options, split, fusion=False):
    transform = ToTensor()
    labels = options.dataset_dir / "label" / (split + "_label")
    if fusion:
        return FusionDataset(
            options.dataset_dir / options.raw_type / split,
            options.dataset_dir / options.ctf_type / split,
            labels,
            transform_diffuse=transform,
            transform_generation=transform,
            target_transform=transform,
        )
    return PairedDataset(
        options.dataset_dir / options.input_type / split,
        labels,
        transform=transform,
        target_transform=transform,
    )


def make_loader(dataset, options, shuffle=False, batch_size=None):
    return DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=batch_size or options.batch_size,
        num_workers=options.num_workers,
        drop_last=False,
        pin_memory=options.pin_memory,
        worker_init_fn=worker_init(options.seed),
    )


def training_loaders(options, fusion=False):
    return {
        split: make_loader(
            make_dataset(options, split, fusion),
            options,
            shuffle=split == "train",
            batch_size=options.batch_size if split == "train" else options.val_batch_size,
        )
        for split in ("train", "val")
    }
