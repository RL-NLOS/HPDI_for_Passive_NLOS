"""FISTA preprocessing parameters for the CTF branch inputs."""

from pathlib import Path

from .base import nonnegative_float, positive_float, positive_int

DEFAULT_MATRIX_PATH = (
    Path(__file__).resolve().parents[1] / "transport_matrix" / "real_trans_matrix.pt"
)


def add_fista_options(parser):
    group = parser.add_argument_group("FISTA reconstruction")
    group.add_argument(
        "--matrix-path",
        type=Path,
        default=DEFAULT_MATRIX_PATH,
        help="PT file containing a 2-D transport matrix; defaults to the bundled matrix",
    )
    group.add_argument(
        "--input-dir", type=Path, help="RAW folder; parent of split folders for --split all"
    )
    group.add_argument(
        "--output-dir", type=Path, help="CTF folder; parent of split folders for --split all"
    )
    group.add_argument("--ctf-type", default="ctf", help="CTF output subfolder in the dataset")
    group.add_argument(
        "--split",
        choices=("train", "val", "test", "all"),
        default="all",
        help="use all to prepare train, val, and test CTF inputs in one run",
    )
    group.add_argument(
        "--lambda",
        dest="lambd",
        type=nonnegative_float,
        default=1e-4,
        help="L1 regularization strength",
    )
    group.add_argument("--max-iter", type=positive_int, default=200, help="iterations per channel")
    group.add_argument(
        "--tol", type=nonnegative_float, default=1e-6, help="absolute stopping threshold"
    )
    group.add_argument(
        "--matrix-scale", type=positive_float, default=1.0, help="matrix scale factor"
    )
    group.add_argument("--height", type=positive_int, default=32, help="reconstructed height")
    group.add_argument("--width", type=positive_int, default=32, help="reconstructed width")
    group.add_argument(
        "--lipschitz", type=positive_float, help="optional known upper bound for ||A.T A||_2"
    )
    group.add_argument(
        "--stopping-rule",
        choices=("legacy", "latest"),
        default="legacy",
        help="legacy preserves original early return; latest returns the new iterate",
    )
