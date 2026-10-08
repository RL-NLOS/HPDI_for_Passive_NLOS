"""Separate training followed by joint training of both branches and fusion."""

import copy
import time

import torch
from torch.optim import lr_scheduler
from tqdm import tqdm

from hpdi.data.loaders import training_loaders
from hpdi.models import FusionNet, ImplicitReconstructionNet, RefinementNet
from hpdi.options.base import joint_checkpoint_paths
from hpdi.runtime import load_tensor, resolve_device, save_options, seed_everything


def make_optimizer(model, options, lr):
    return torch.optim.Adam(
        model.parameters(),
        lr=lr,
        betas=tuple(options.betas),
        weight_decay=options.weight_decay,
    )


def make_scheduler(optimizer, options):
    if options.scheduler == "step":
        return lr_scheduler.StepLR(
            optimizer,
            step_size=options.scheduler_step_size,
            gamma=options.scheduler_gamma,
        )
    return lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=options.epochs,
        eta_min=options.min_lr,
    )


def forward_batch(models, batch, device):
    """Use a branch reconstruction, or fuse IRN features followed by RN features."""
    if len(models) == 1:
        inputs, labels = batch
        outputs, _ = models[0](inputs.to(device))
    else:
        raw, ctf, labels = batch
        _, features_irn = models[0](raw.to(device))
        _, features_rn = models[1](ctf.to(device))
        outputs = models[2](features_irn, features_rn)
    return outputs, labels.to(device)


def fit(models, optimizers, scheduler, dataloaders, options, device):
    """Preserve BCE, epoch averaging, scheduler timing, and best-val checkpoints."""
    since = time.time()
    criterion = torch.nn.BCELoss()
    best_loss = float("inf")
    best_weights = [copy.deepcopy(model.state_dict()) for model in models]
    options.model_dir.mkdir(parents=True, exist_ok=True)
    log_path = options.model_dir / (options.name + "_loss_log.txt")
    with log_path.open("w", encoding="utf-8") as log:
        log.write("Epoch\tTrain_Loss\tVal_Loss\n")
        for epoch in range(options.epochs):
            losses = {}
            print("Epoch {}/{}".format(epoch + 1, options.epochs))
            for phase in ("train", "val"):
                training = phase == "train"
                for model in models:
                    model.train(training)
                epoch_loss = 0.0
                with torch.set_grad_enabled(training):
                    for batch in tqdm(dataloaders[phase], desc=phase, unit="batch"):
                        if training:
                            for optimizer in optimizers:
                                optimizer.zero_grad()
                        outputs, labels = forward_batch(models, batch, device)
                        loss = criterion(outputs, labels)
                        if training:
                            loss.backward()
                            for optimizer in optimizers:
                                optimizer.step()
                        epoch_loss += loss.item() * labels.size(0)
                losses[phase] = epoch_loss / len(dataloaders[phase].dataset)
            print("Train: {:.8f}  Val: {:.8f}".format(losses["train"], losses["val"]))
            log.write("{}\t{:.8f}\t{:.8f}\n".format(epoch + 1, losses["train"], losses["val"]))
            log.flush()
            scheduler.step()
            if losses["val"] < best_loss:
                best_loss = losses["val"]
                best_weights = [copy.deepcopy(model.state_dict()) for model in models]
    checkpoints = (
        (options.model_dir / (options.name + ".pkl"),)
        if len(models) == 1
        else joint_checkpoint_paths(options.model_dir, options.name)
    )
    for model, weights, checkpoint in zip(models, best_weights, checkpoints):
        model.load_state_dict(weights)
        torch.save(model.state_dict(), str(checkpoint))
        print("Saved {}".format(checkpoint))
    elapsed = time.time() - since
    print("Training complete in {:.0f}m {:.0f}s".format(elapsed // 60, elapsed % 60))
    return models


def train_separate(options):
    """Train the IRN on RAW measurements or the RN on FISTA reconstructions."""
    device = resolve_device(options.device)
    seed_everything(options.seed)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    dataloaders = training_loaders(options)
    model_class = (
        RefinementNet if options.network == "rn" else ImplicitReconstructionNet
    )
    model = model_class(options.channels, options.channels).to(device)
    optimizer = make_optimizer(model, options, options.lr)
    scheduler = make_scheduler(optimizer, options)
    save_options(options, options.model_dir, options.name, device)
    return fit((model,), (optimizer,), scheduler, dataloaders, options, device)[0]


def train_joint(options):
    """Initialize both separately trained branches and jointly update all networks."""
    device = resolve_device(options.device)
    seed_everything(options.seed)
    # Check required stage-1 weights before creating the training job.
    for checkpoint in (options.raw_checkpoint, options.ctf_checkpoint):
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "Run separate training first; checkpoint missing: {}".format(checkpoint)
            )
    dataloaders = training_loaders(options, fusion=True)
    models = (
        ImplicitReconstructionNet(options.channels, options.channels).to(device),
        RefinementNet(options.channels, options.channels).to(device),
        FusionNet(options.channels).to(device),
    )
    models[0].load_state_dict(load_tensor(options.raw_checkpoint, device))
    models[1].load_state_dict(load_tensor(options.ctf_checkpoint, device))
    optimizers = tuple(
        make_optimizer(model, options, options.lr if index == 2 else options.branch_lr)
        for index, model in enumerate(models)
    )
    # Preserve the original joint schedule: only the fusion optimizer is scheduled.
    scheduler = make_scheduler(optimizers[2], options)
    save_options(options, options.model_dir, options.name, device)
    return fit(models, optimizers, scheduler, dataloaders, options, device)[2]
