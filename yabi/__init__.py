__all__ = (
    "__version__",
    "main",
    "convert_main",
    "to_bython",
    "to_pure_python",
)


import sys

from .runner import __version__, main
from .parser import to_bython, to_pure_python


def convert_main():
    if len(sys.argv) < 2:
        print("Missing target file.", file=sys.stderr)
        return
    print(to_bython(open(sys.argv[1]).read()), end="")
