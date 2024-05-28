import os
import sys
from io import StringIO
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from yabi import main
from yabi.console import YabiConsole


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


def test_async():
    with patch("sys.stdout", new=StringIO()):
        main(["tests/async.by"])
        assert sys.stdout.getvalue() == "hello\n"


def test_match():
    with patch("sys.stdout", new=StringIO()):
        main(["tests/match.by"])
        assert sys.stdout.getvalue() == "1\n2\n"


def test_console():
    code = """
for i in range(
    10
) {
    if i % 2 { print(i) }
}

if True:
    print(1)
    print(2)
    print(3)
    """.strip() + "\n\n"
    with patch("sys.stdout", new=StringIO()), patch("sys.stdin", new=StringIO(code)):
        YabiConsole().interact()
        assert sys.stdout.getvalue() == """
>>> ... ... ... ... ... 1
3
5
7
9
>>> ... ... ... ... 1
2
3
>>>
        """.strip() + " "
