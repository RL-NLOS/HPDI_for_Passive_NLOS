"""Generate CTF inputs for the RN with FISTA before network training."""

import time

import torch
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from hpdi.algorithms import FISTASolver
from hpdi.data.datasets import ImageDataset
from hpdi.data.loaders import make_loader
from hpdi.runtime import file_sha256, load_tensor, resolve_device, save_options, seed_everything


def save_ctf_image(output, path):
    """Keep grayscale inputs single-channel with the original pixel rounding."""
    if output.shape[0] == 1:
        pixels = output[0].mul(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
        Image.fromarray(pixels).save(str(path))
    else:
        save_image(output, str(path))


def reconstruct(options):
    """Prepare one or all dataset splits, reusing a single fixed-matrix solver."""
    device = resolve_device(options.device)
    seed_everything(options.seed)
    splits = ("train", "val", "test") if options.split == "all" else (options.split,)
    datasets = []
    for split in splits:
        input_dir = options.input_dir / split if options.split == "all" else options.input_dir
        output_dir = options.output_dir / split if options.split == "all" else options.output_dir
        if input_dir.resolve() == output_dir.resolve():
            raise ValueError("FISTA output must use a different directory from the RAW input")
        # Validate all input splits before writing any preprocessing outputs.
        dataset = ImageDataset(input_dir, transform=ToTensor())
        datasets.append((split, dataset, output_dir))
    matrix = load_tensor(options.matrix_path, device)
    if not isinstance(matrix, torch.Tensor) or matrix.ndim != 2:
        raise ValueError("The matrix PT file must contain a 2-D tensor")
    if options.height * options.width != matrix.shape[1]:
        raise ValueError("--height * --width must equal the matrix column count")
    matrix_metadata = {
        "sha256": file_sha256(options.matrix_path),
        "shape": list(matrix.shape),
        "source_dtype": str(matrix.dtype),
        "scale": options.matrix_scale,
    }
    # Preserve the original float32 matrix conversion and scale factor.
    matrix = options.matrix_scale * matrix.float()
    solver = FISTASolver(matrix, lipschitz=options.lipschitz)
    matrix_metadata["compute_dtype"] = str(matrix.dtype)
    matrix_metadata["lipschitz"] = solver.lipschitz.item()
    since, total = time.perf_counter(), 0
    for split, dataset, output_dir in datasets:
        loader = make_loader(dataset, options)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_options(options, output_dir, "fista", device, extra={"matrix": matrix_metadata})
        count = 0
        for measurements in tqdm(loader, desc="FISTA " + split, unit="batch"):
            outputs = solver.solve_images(
                measurements,
                height=options.height,
                width=options.width,
                lambd=options.lambd,
                max_iter=options.max_iter,
                tol=options.tol,
                stopping_rule=options.stopping_rule,
            )
            for output in outputs:
                # Keep sample names so RAW, CTF, and labels retain their correspondence.
                save_ctf_image(output, output_dir / dataset.paths[count].name)
                count += 1
        print("Prepared {} CTF images in {}".format(count, output_dir))
        total += count
    elapsed = time.perf_counter() - since
    print("FISTA preprocessing complete: {} images in {:.3f}s".format(total, elapsed))
    return total
