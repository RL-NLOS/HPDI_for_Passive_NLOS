"""Device setup, reproducibility, configuration records, and tensor loading."""

import hashlib
import inspect
import json
import os
import platform
import random
from functools import partial
from pathlib import Path

import numpy as np
import torch


def resolve_device(name):
    """Select a device explicitly without rewriting CUDA visibility on import."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Supported devices: auto, cpu, cuda, cuda:N")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --device cpu")
        index = device.index if device.index is not None else torch.cuda.current_device()
        if index >= torch.cuda.device_count():
            raise ValueError("CUDA device index {} is unavailable".format(index))
        torch.cuda.set_device(index)
    return device


def seed_everything(seed):
    """Apply the original RNG and cuDNN settings when a seed is supplied."""
    if seed is None:
        return
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id, seed):
    """Match the original NumPy worker seed; callable is Windows-pickleable."""
    np.random.seed(int(seed) + worker_id)


def worker_init(seed):
    return partial(seed_worker, seed=seed) if seed is not None else None


def file_sha256(path):
    """Identify the exact input file without reading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_options(options, directory, stem, device, extra=None):
    """Record resolved CLI parameters and runtime versions for reproduction."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "options": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(options).items()
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
        },
    }
    if extra is not None:
        payload.update(extra)
    with (directory / (stem + "_options.json")).open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_tensor(path, device):
    """Load existing tensor/state_dict files on old and current PyTorch versions."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("Tensor/checkpoint does not exist: {}".format(path))
    kwargs = {"map_location": device}
    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = True
    return torch.load(str(path), **kwargs)
