"""Thin entry points; --help works even before installing scientific packages."""

from hpdi.options import parse_options


def train_main(argv=None):
    options = parse_options("train", argv)
    from hpdi.training import train_separate

    train_separate(options)


def test_main(argv=None):
    options = parse_options("test", argv)
    from hpdi.evaluation import evaluate

    evaluate(options)


def fusion_train_main(argv=None):
    options = parse_options("fusion-train", argv)
    from hpdi.training import train_joint

    train_joint(options)


def fusion_test_main(argv=None):
    options = parse_options("fusion-test", argv)
    from hpdi.evaluation import evaluate

    evaluate(options, fusion=True)


def fista_main(argv=None):
    options = parse_options("fista", argv)
    from hpdi.reconstruction import reconstruct

    reconstruct(options)
