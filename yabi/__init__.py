import sys
from io import TextIOBase
from codeop import CommandCompiler
from traceback import print_exc

import pwcp
from pwcp import main as p_main
from pwcp.preprocessor import PyPreprocessor, preprocess

from .parser import UNCLOSED_BLOCK_ERROR, to_bython, to_pure_python


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
        console()


def console():
    try:
        import readline
    except ImportError:
        pass
    compiler = CommandCompiler()
    namespace = {}
    while True:
        try:
            code = input(">>> ")
        except EOFError:
            print()
            break
        code, parsed = _read_braced(code)
        compiled = None
        while True:
            try:
                compiled = compiler(parsed, "<console>", "single")
            except SyntaxError:
                try:
                    compiled = compiler(parsed, "<console>", "exec")
                except SyntaxError:
                    print_exc()
                    break
            if compiled is not None:
                break
            try:
                code += "\n" + input("... ")
            except EOFError:
                break
            code, parsed = _read_braced(code)
        try:
            exec(compiled or "", namespace)
        except BaseException:
            print_exc()


def _read_braced(code):
    while True:
        try:
            parsed = to_pure_python(code)
        except SyntaxError as e:
            if e.args[0] != UNCLOSED_BLOCK_ERROR:
                print_exc()
                return "", ""
            try:
                code += "\n" + input("... ")
            except EOFError:
                return "", ""
        else:
            if not code.endswith("\n"):
                parsed = parsed.rstrip("\n")
            return code, parsed


def convert_main():
    if len(sys.argv) < 2:
        print("Missing target file.", file=sys.stderr)
        return
    print(to_bython(open(sys.argv[1]).read()), end="")
