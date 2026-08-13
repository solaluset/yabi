import argparse
import os
import sys

import pwcp

from . import config
from .hooks import YabiHooks
from .version import __version__

parser = argparse.ArgumentParser(
    (
        "python -m " + __package__
        if sys.argv[0] == "-m"
        else os.path.basename(sys.argv[0])
    ),
    description="Yet Another Bython (braced Python) Implementation",
)
parser.add_argument(
    "--version", action="version", version=__package__ + " " + __version__
)
parser.add_argument("-m", action="store_true", help="run target as module")
parser.add_argument(
    "-c", action="store_true", help="run target as command line"
)
parser.add_argument(
    "--prefer-python",
    dest="prefer_python",
    action="store_true",
    help="prefer .py files over .by when importing",
)
parser.add_argument(
    "--save-files",
    dest="save_files",
    action="store_true",
    help="save .by files to .py after preprocessing",
)
parser.add_argument(
    "--enable-preprocessing",
    dest="enable_preprocessing",
    action="store_true",
    help="enable C-style preprocessing by default",
)
parser.add_argument("target", nargs=argparse.OPTIONAL)
parser.add_argument("args", nargs=argparse.REMAINDER)


def main(args=sys.argv[1:]):
    args = parser.parse_args(args)

    config.SAVE_FILES = args.save_files
    config.ENABLE_PREPROCESSING = args.enable_preprocessing
    del args.enable_preprocessing

    pwcp.add_file_extension(
        config.EXTENSION, preprocess=config.ENABLE_PREPROCESSING
    )
    pwcp.add_hook(YabiHooks())

    if not args.target:
        args.m = True
        args.target = f"{__package__}.console"

    pwcp.main_with_params(
        **vars(args), skip_unknown_sources=not config.ENABLE_PREPROCESSING
    )
