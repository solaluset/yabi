from io import TextIOBase

import pwcp
from pwcp import main as p_main
from pwcp.preprocessor import PyPreprocessor, preprocess

from .parser import to_pure_python


EXTENSION = ".by"


def preproc(src, p=None):
    if p is None:
        if not isinstance(src, TextIOBase) or src.name.endswith(EXTENSION):
            p = PyPreprocessor(disabled=True)
    res = preprocess(src, p)
    if isinstance(src, TextIOBase):
        res = to_pure_python(res + "\n\n")
    return res


def main(args=None):
    pwcp.config.FILE_EXTENSIONS.append(EXTENSION)
    pwcp.preprocessor.preprocess = preproc
    args = (args,) if args is not None else ()
    p_main(*args)
