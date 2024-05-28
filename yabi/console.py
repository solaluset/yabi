import sys
from code import InteractiveConsole

from pwcp.preprocessor import PyPreprocessor, PreprocessorError

from . import preproc
from .parser import UNCLOSED_BLOCK_ERROR, to_pure_python


class YabiConsole(InteractiveConsole):
    def __init__(self):
        super().__init__()
        self.preprocessor = PyPreprocessor(disabled=True)
        # compile is an attribute, not a function
        self.compile, self._compiler = self._compiler, self.compile

    def runsource(self, source, filename="<input>", symbol="single") -> bool:
        try:
            parsed = to_pure_python(preproc(source, self.preprocessor))
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
        return super().runsource(parsed, filename, symbol)

    def _compiler(self, source, filename, symbol):
        try:
            return self._compiler(source, filename, symbol)
        except PreprocessorError:
            self.showtraceback()
            return self._compiler("", filename, symbol)
        except SyntaxError:
            return self._compiler(source, filename, "exec")


if __name__ == "__main__":
    try:
        import readline
    except ImportError:
        pass
    # use default handler instead of PWCP's one
    sys.excepthook = sys.__excepthook__
    YabiConsole().interact()
