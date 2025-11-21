import sys
import ast
from code import InteractiveConsole

from pwcp.preprocessor import PyPreprocessor, PreprocessorError, preprocess

from . import config
from .parser import UNCLOSED_BLOCK_ERROR, to_pure_python


class YabiConsole(InteractiveConsole):
    def __init__(self):
        super().__init__()
        self.preprocessor = PyPreprocessor(
            disabled=not config.ENABLE_PREPROCESSING
        )

    def runsource(self, source, filename="<input>", symbol="single") -> bool:
        try:
            parsed = to_pure_python(
                preprocess(source, filename, self.preprocessor)[0]
            )
        except PreprocessorError:
            self.showtraceback()
            return False
        except SyntaxError as e:
            if e.args[0] == UNCLOSED_BLOCK_ERROR or e.args[0].startswith(
                "Unterminated"
            ):
                return True
            self.showtraceback()
            return False
        if not source.endswith("\n"):
            parsed = parsed.rstrip("\n")

        try:
            code = self._compiler(parsed, filename, symbol, source)
        except (OverflowError, SyntaxError, ValueError):
            self.showsyntaxerror(filename)
            return False

        if code is None:
            return True

        if symbol == "single":
            tree = ast.parse(parsed, filename, "exec")
            for node in tree.body:
                code = self.compile.compiler(
                    ast.Interactive([node]), filename, symbol
                )
                if not self.runcode(code):
                    break
        else:
            self.runcode(code)
        return False

    def runcode(self, code) -> bool:
        try:
            exec(code, self.locals)
            return True
        except SystemExit:
            raise
        except BaseException:
            self.showtraceback()
            return False

    def _compiler(self, source, filename, symbol, original_source):
        if original_source.endswith("\\"):
            return None
        try:
            return self.compile(source, filename, symbol)
        except PreprocessorError:
            self.showtraceback()
            return self.compile("", filename, symbol)
        except SyntaxError:
            if "\n" in original_source and not source.endswith("\n"):
                return None
            return self.compile(source, filename, "exec")


if __name__ == "__main__":
    try:
        import readline  # noqa: F401
    except ImportError:
        pass
    # use default handler instead of PWCP's one
    sys.excepthook = sys.__excepthook__
    YabiConsole().interact()
