"""Evaluation parameters."""

from pathlib import Path


def add_test_options(parser, fusion=False):
    group = parser.add_argument_group("Evaluation")
    group.add_argument("--output-dir", type=Path, help="output directory; images retain their input filenames")
    if fusion:
        group.add_argument(
            "--raw-checkpoint", type=Path, help="jointly trained RAW branch checkpoint"
        )
        group.add_argument(
            "--ctf-checkpoint", type=Path, help="jointly trained CTF branch checkpoint"
        )
        group.add_argument(
            "--fusion-checkpoint", type=Path, help="jointly trained fusion checkpoint"
        )
        group.add_argument(
            "--compute-psnr",
            action="store_true",
            help="also report PSNR; original fusion test only measured time",
        )
    else:
        group.add_argument(
            "--checkpoint", type=Path, help="separately trained reconstruction checkpoint"
        )
