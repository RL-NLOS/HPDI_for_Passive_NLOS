"""Separate and joint inference with complete per-sample output saving."""

import json
import time

import torch
from torchvision.utils import save_image

from hpdi.data.loaders import make_dataset, make_loader
from hpdi.metrics import psnr, to_uint8
from hpdi.models import FusionNet, ImplicitReconstructionNet, RefinementNet
from hpdi.runtime import load_tensor, resolve_device, save_options, seed_everything
from hpdi.training import forward_batch


def evaluate(options, fusion=False):
    device = resolve_device(options.device)
    seed_everything(options.seed)
    dataset = make_dataset(options, "test", fusion=fusion)
    dataloader = make_loader(dataset, options)
    if fusion:
        models = (
            ImplicitReconstructionNet(options.channels, options.channels).to(device),
            RefinementNet(options.channels, options.channels).to(device),
            FusionNet(options.channels).to(device),
        )
        checkpoints = (options.raw_checkpoint, options.ctf_checkpoint, options.fusion_checkpoint)
    else:
        model_class = (
            RefinementNet if options.network == "rn" else ImplicitReconstructionNet
        )
        models = (model_class(options.channels, options.channels).to(device),)
        checkpoints = (options.checkpoint,)
    for model, checkpoint in zip(models, checkpoints):
        model.load_state_dict(load_tensor(checkpoint, device))
        model.eval()
    options.output_dir.mkdir(parents=True, exist_ok=True)
    save_options(options, options.output_dir, "evaluation", device)
    total_psnr, elapsed, count = 0.0, 0.0, 0
    compute_psnr = not fusion or options.compute_psnr
    with torch.no_grad():
        for batch in dataloader:
            # Synchronize to report completed inference time on CUDA.
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            since = time.perf_counter()
            outputs, labels = forward_batch(models, batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed += time.perf_counter() - since
            for output, target in zip(outputs, labels):
                if compute_psnr:
                    total_psnr += psnr(to_uint8(output), to_uint8(target))
                sample_name = dataset.imgs[count][0].name
                save_image(output, str(options.output_dir / sample_name))
                count += 1
    result = {"samples": count, "seconds_per_frame": elapsed / count}
    if compute_psnr:
        result["psnr"] = total_psnr / count
        print("PSNR: {:.6f}".format(result["psnr"]))
    with (options.output_dir / "evaluation_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print("Seconds/frame: {:.8f}".format(result["seconds_per_frame"]))
    return result
