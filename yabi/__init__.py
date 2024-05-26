from __future__ import annotations
import sys
from io import TextIOBase
from code import InteractiveConsole

import pwcp
from pwcp import main as p_main
from pwcp.preprocessor import PyPreprocessor, PreprocessorError, preprocess

from .parser import UNCLOSED_BLOCK_ERROR, to_bython, _to_pure_python_inner, to_pure_python


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
        try:
            import readline
        except ImportError:
            pass
        pwcp.hooks.install()
        YabiConsole().interact()


class YabiConsole(InteractiveConsole):
    def __init__(self):
        super().__init__()
        self.preprocessor = PyPreprocessor(disabled=True)
        # compile is an attribute, not a function
        self.compile, self._compiler = self._compiler, self.compile

    def runsource(self, source, filename="<input>", symbol="single") -> bool:
        try:
            parsed, first_block_is_braced = _to_pure_python_inner(preproc(source, self.preprocessor))
        except PreprocessorError:
            self.showtraceback()
            return False
        except SyntaxError as e:
            if e.args[0] == UNCLOSED_BLOCK_ERROR:
                return True
            self.showtraceback()
            return False
        if not first_block_is_braced and not source.endswith("\n"):
            parsed = parsed.rstrip("\n")
        return super().runsource(parsed, filename, symbol)

    def _compiler(self, source, filename, symbol):
        try:
            return self._compiler(source, filename, symbol)
        except PreprocessorError:
            self.showtraceback()
            return self._compiler("", filename, symbol)
        except SyntaxError:
            return self._compiler(source, filename, "exec")


def convert_main():
    if len(sys.argv) < 2:
        print("Missing target file.", file=sys.stderr)
        return
    print(to_bython(open(sys.argv[1]).read()), end="")
