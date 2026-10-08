"""Shared CLI conventions and portable experiment paths."""

import argparse
import math
from pathlib import Path


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return value


def nonnegative_float(value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be finite and nonnegative")
    return value


def positive_float(value):
    value = nonnegative_float(value)
    if value == 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return value


def beta(value):
    value = nonnegative_float(value)
    if value >= 1:
        raise argparse.ArgumentTypeError("Adam beta must be in [0, 1)")
    return value


def branch_checkpoint_name(input_type):
    """Name a separately trained branch by its input type."""
    return "{}_best".format(input_type.lower())


def joint_checkpoint_paths(model_dir, name="joint"):
    """Return meaningful names for the jointly trained RAW, CTF, and fusion models."""
    return tuple(
        Path(model_dir) / "{}_{}_best.pkl".format(name, role)
        for role in ("raw", "ctf", "fusion")
    )


def build_parser(command):
    """Build a parser without importing models or initializing a device."""
    if command not in {"train", "test", "fusion-train", "fusion-test", "fista"}:
        raise ValueError("Unknown command: {}".format(command))
    training = command in {"train", "fusion-train"}
    fusion = command.startswith("fusion-")
    parser = argparse.ArgumentParser(
        description="HPDI {}".format(command),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    data = parser.add_argument_group("Data and output paths")
    data.add_argument(
        "--data-root", type=Path, default=Path("data/HPDI"), help="parent of dataset folders"
    )
    data.add_argument(
        "--dataset-name",
        default="SHAPES_img",
        help="dataset folder name",
    )
    data.add_argument("--model-dir", type=Path, help="checkpoint directory; default: DATASET/model")
    if not fusion:
        data.add_argument(
            "--input-type",
            default="ctf" if command == "test" else "raw",
            help="measurement subfolder",
        )
    else:
        data.add_argument("--raw-type", default="raw", help="RAW measurement subfolder")
        data.add_argument(
            "--ctf-type",
            default="ctf",
            help="CTF measurement subfolder",
        )
    if command in {"train", "test"}:
        network = parser.add_argument_group("Network")
        network.add_argument(
            "--network",
            choices=("irn", "rn"),
            help="paper module; default: IRN for RAW, RN for CTF",
        )
    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or cuda:N; respects CUDA_VISIBLE_DEVICES"
    )
    runtime.add_argument(
        "--seed",
        type=nonnegative_int,
        default=150 if training else None,
        help="random seed; evaluation unseeded unless supplied",
    )
    runtime.add_argument(
        "--num-workers", type=nonnegative_int, default=0, help="DataLoader workers"
    )
    runtime.add_argument(
        "--batch-size",
        type=positive_int,
        default=16 if training else 1,
        help="training or inference batch size",
    )
    runtime.add_argument(
        "--no-pin-memory", action="store_true", help="disable DataLoader pinned memory"
    )
    if command != "fista":
        runtime.add_argument(
            "--channels",
            type=int,
            choices=(1, 3),
            help="3 for Anime_img/SHAPES_img/SuperModel_img, otherwise 1",
        )
    if training:
        from .train import add_training_options

        add_training_options(parser, fusion=fusion)
    elif command == "fista":
        from .fista import add_fista_options

        add_fista_options(parser)
    else:
        from .test import add_test_options

        add_test_options(parser, fusion=fusion)
    return parser


def parse_options(command, argv=None):
    """Resolve paths and defaults consistently for each entry point."""
    args = build_parser(command).parse_args(argv)
    args.command = command
    args.dataset_dir = args.data_root / args.dataset_name
    args.model_dir = args.model_dir or args.dataset_dir / "model"
    args.pin_memory = not args.no_pin_memory
    if command in {"train", "test"}:
        args.network = args.network or ("rn" if args.input_type == "ctf" else "irn")
    if command != "fista" and args.channels is None:
        args.channels = 3 if args.dataset_name in {"Anime_img", "SHAPES_img", "SuperModel_img"} else 1
    if command in {"train", "fusion-train"}:
        args.val_batch_size = args.val_batch_size or args.batch_size
        if command == "train":
            if args.epochs is None:
                args.epochs = 30 if args.network == "rn" else 100
            args.name = args.name or branch_checkpoint_name(args.input_type)
            args.scheduler = args.scheduler or ("step" if args.network == "rn" else "cosine")
            if args.scheduler_step_size is None:
                args.scheduler_step_size = max(1, args.epochs * 2 // 3)
        else:
            args.name = args.name or "joint"
            args.scheduler = args.scheduler or "step"
            args.scheduler_step_size = args.scheduler_step_size or 10
            args.raw_checkpoint = args.raw_checkpoint or args.model_dir / (
                branch_checkpoint_name(args.raw_type) + ".pkl"
            )
            args.ctf_checkpoint = args.ctf_checkpoint or args.model_dir / (
                branch_checkpoint_name(args.ctf_type) + ".pkl"
            )
    elif command in {"test", "fusion-test"}:
        args.output_dir = args.output_dir or args.dataset_dir / (
            "output_fusion" if command == "fusion-test" else "output_" + args.input_type
        )
        if command == "test":
            args.checkpoint = args.checkpoint or args.model_dir / (
                branch_checkpoint_name(args.input_type) + ".pkl"
            )
        else:
            fields = ("raw_checkpoint", "ctf_checkpoint", "fusion_checkpoint")
            for field, checkpoint in zip(fields, joint_checkpoint_paths(args.model_dir)):
                if getattr(args, field) is None:
                    setattr(args, field, checkpoint)
    else:
        input_root = args.dataset_dir / args.input_type
        output_root = args.dataset_dir / args.ctf_type
        if args.split == "all":
            args.input_dir = args.input_dir or input_root
            args.output_dir = args.output_dir or output_root
        else:
            args.input_dir = args.input_dir or input_root / args.split
            args.output_dir = args.output_dir or output_root / args.split
    return args
