"""Original rounded, uint8-image PSNR convention."""

import math

import numpy as np
import torch


def to_uint8(tensor):
    """Round exactly as in the original test script without modifying outputs."""
    return (
        tensor.detach()
        .cpu()
        .clone()
        .squeeze()
        .mul_(255)
        .add_(0.5)
        .clamp_(0, 255)
        .to(torch.uint8)
        .numpy()
    )


def psnr(image, target):
    """PSNR on 0..255 arrays; cap the near-zero-MSE case at 100 dB."""
    mse = np.mean((image / 255.0 - target / 255.0) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * math.log10(1.0 / math.sqrt(mse))
