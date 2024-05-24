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
        res = to_pure_python(res + "\n\n")
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
    while True:
        try:
            code = input(">>> ")
        except EOFError:
            print()
            break
        while True:
            try:
                code = to_pure_python(code + "\n\n")
                break
            except SyntaxError as e:
                if e.args[0] != UNCLOSED_BLOCK_ERROR:
                    print_exc()
                    code = ""
                    break
                try:
                    code += "\n" + input("... ")
                except EOFError:
                    break
        compiled = None
        while True:
            try:
                compiled = compiler(code, "<console>", "single")
            except SyntaxError:
                try:
                    compiled = compiler(code, "<console>", "exec")
                except SyntaxError:
                    print_exc()
                    break
            if compiled is not None:
                break
            try:
                code += "\n" + input("... ")
            except EOFError:
                break
        try:
            exec(compiled or "")
        except BaseException:
            print_exc()


def convert_main():
    if len(sys.argv) < 2:
        print("Missing target file.", file=sys.stderr)
        return
    print(to_bython(open(sys.argv[1]).read()), end="")
