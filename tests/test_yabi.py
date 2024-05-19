import os
import sys
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from yabi import main


def test_regular():
    with patch("sys.stdout", new=StringIO()):
        main(["tests/hello.by"])
        assert sys.stdout.getvalue() == "Hello\n"


def test_mixed():
    with patch("sys.stdout", new=StringIO()):
        main(["tests/mixed.by"])
        assert sys.stdout.getvalue() == "3\n"


def test_dict():
    with patch("sys.stdout", new=StringIO()):
        main(["tests/dict.by"])
        assert sys.stdout.getvalue() == "1\n2\n3\nok\n"


def test_except():
    with patch("sys.stdout", new=StringIO()):
        main(["tests/except.by"])
        assert sys.stdout.getvalue() == "e\ndivision by zero\ne\ne\ndivision by zero\n"


def test_inline_if():
    with patch("sys.stdout", new=StringIO()):
        main(["tests/inline_if.by"])
        assert sys.stdout.getvalue() == "a\n"
