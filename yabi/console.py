import sys
import ast
from code import InteractiveConsole

from pwcp.preprocessor import PyPreprocessor, PreprocessorError

from . import preproc
from .parser import UNCLOSED_BLOCK_ERROR, to_pure_python


class YabiConsole(InteractiveConsole):
    def __init__(self):
        super().__init__()
        self.preprocessor = PyPreprocessor(disabled=True)

    def runsource(self, source, filename="<input>", symbol="single") -> bool:
        try:
            parsed, module = to_pure_python(preproc(source, self.preprocessor))
        except PreprocessorError:
            self.showtraceback()
            return False
        except SyntaxError as e:
            if e.args[0] == UNCLOSED_BLOCK_ERROR or e.args[0].startswith("Unterminated"):
                return True
            self.showtraceback()
            return False
        if not source.endswith("\n"):
            parsed = parsed.rstrip("\n")
        if module:
            sys.modules[module.__name__] = module

        try:
            code = self._compiler(parsed, filename, symbol)
        except (OverflowError, SyntaxError, ValueError):
            self.showsyntaxerror(filename)
            return False

        if code is None:
            return True

        if symbol == "single":
            tree = ast.parse(parsed, filename, "exec")
            for node in tree.body:
                code = self.compile.compiler(ast.Interactive([node]), filename, symbol)
                self.runcode(code)
        else:
            self.runcode(code)
        return False

    def _compiler(self, source, filename, symbol):
        try:
            return self.compile(source, filename, symbol)
        except PreprocessorError:
            self.showtraceback()
            return self.compile("", filename, symbol)
        except SyntaxError:
            return self.compile(source, filename, "exec")


if __name__ == "__main__":
    try:
        import readline
    except ImportError:
        pass
    # use default handler instead of PWCP's one
    sys.excepthook = sys.__excepthook__
    YabiConsole().interact()
