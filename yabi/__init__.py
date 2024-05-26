from __future__ import annotations
import sys
from io import TextIOBase

import pwcp
from pwcp import main as p_main
from pwcp.preprocessor import PyPreprocessor, preprocess

from .parser import to_bython, to_pure_python


EXTENSION = ".by"


def preproc(src, p=None):
    if p is None:
        if not isinstance(src, TextIOBase) or src.name.endswith(EXTENSION):
            p = PyPreprocessor(disabled=True)
    res = preprocess(src, p)
    if isinstance(src, TextIOBase):
        res = to_pure_python(res)
    return res


def main(args=None):
    pwcp.config.FILE_EXTENSIONS.append(EXTENSION)
    pwcp.preprocessor.preprocess = preproc
    args = (args,) if args is not None else ()
    if args or len(sys.argv) > 1:
        p_main(*args)
    else:
        p_main(["-m", "yabi.console"])


def convert_main():
    if len(sys.argv) < 2:
        print("Missing target file.", file=sys.stderr)
        return
    print(to_bython(open(sys.argv[1]).read()), end="")
