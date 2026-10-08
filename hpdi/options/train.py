"""Separate and joint training parameters."""

from pathlib import Path

from .base import beta, nonnegative_float, positive_float, positive_int


def add_training_options(parser, fusion=False):
    group = parser.add_argument_group("Training")
    group.add_argument(
        "--epochs",
        type=positive_int,
        default=10 if fusion else None,
        help="training epochs; default: RN 30, IRN 100, joint 10",
    )
    group.add_argument("--val-batch-size", type=positive_int, help="defaults to --batch-size")
    group.add_argument("--name", help="checkpoint stem and log prefix")
    group.add_argument(
        "--lr",
        type=positive_float,
        default=1e-4,
        help="reconstruction or fusion Adam learning rate",
    )
    group.add_argument(
        "--betas",
        type=beta,
        nargs=2,
        default=(0.9, 0.999),
        metavar=("BETA1", "BETA2"),
        help="Adam coefficients",
    )
    group.add_argument(
        "--weight-decay", type=nonnegative_float, default=1e-4, help="Adam weight decay"
    )
    group.add_argument(
        "--scheduler",
        choices=("step", "cosine"),
        help="default: step for RN/joint, cosine for IRN",
    )
    group.add_argument(
        "--scheduler-step-size",
        type=positive_int,
        help="StepLR interval: 2/3 epochs (separate), 10 (joint)",
    )
    group.add_argument(
        "--scheduler-gamma", type=positive_float, default=0.5, help="StepLR multiplier"
    )
    group.add_argument(
        "--min-lr", type=nonnegative_float, default=1e-6, help="CosineAnnealingLR minimum"
    )
    if fusion:
        group.add_argument("--raw-checkpoint", type=Path, help="RAW weights from separate training")
        group.add_argument("--ctf-checkpoint", type=Path, help="CTF weights from separate training")
        group.add_argument(
            "--branch-lr",
            type=positive_float,
            default=1e-6,
            help="Adam learning rate for both branches",
        )
