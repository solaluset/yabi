import os
import sys
import argparse
from io import TextIOBase
from importlib import metadata

import pwcp
from pwcp.preprocessor import PyPreprocessor, preprocess

from . import config
from .parser import to_pure_python


__version__ = metadata.version(__package__)

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
    "--prefer-py",
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
parser.add_argument("target", nargs=argparse.OPTIONAL)
parser.add_argument("args", nargs=argparse.ZERO_OR_MORE)


def preproc(src, p=None):
    if p is None:
        if not isinstance(src, TextIOBase) or src.name.endswith(
            config.EXTENSION
        ):
            p = PyPreprocessor(disabled=True)
    res, deps = preprocess(src, p)
    return to_pure_python(res), deps


def main(args=sys.argv[1:]):
    args = parser.parse_args(args)

    pwcp.config.FILE_EXTENSIONS.append(config.EXTENSION)
    pwcp.preprocessor.preprocess = preproc
    config.SAVE_FILES = args.save_files

    if not args.target:
        args.m = True
        args.target = f"{__package__}.console"

    pwcp.main_with_params(**vars(args))
