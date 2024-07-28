from __future__ import annotations
import sys
from io import TextIOBase

import pwcp
from pwcp import main as p_main
from pwcp.preprocessor import PyPreprocessor, preprocess

from . import config
from .parser import to_bython, to_pure_python


def preproc(src, p=None):
    if p is None:
        if not isinstance(src, TextIOBase) or src.name.endswith(
            config.EXTENSION
        ):
            p = PyPreprocessor(disabled=True)
    res, deps = preprocess(src, p)
    return to_pure_python(res), deps


def main(args=sys.argv[1:]):
    pwcp.config.FILE_EXTENSIONS.append(config.EXTENSION)
    pwcp.preprocessor.preprocess = preproc
    config.SAVE_FILES = "--save-files" in args
    if all(a.startswith("--") for a in args):
        args.extend(("-m", f"{__package__}.console"))
    p_main(args)


def convert_main():
    if len(sys.argv) < 2:
        print("Missing target file.", file=sys.stderr)
        return
    print(to_bython(open(sys.argv[1]).read()), end="")
