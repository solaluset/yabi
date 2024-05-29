import os
import sys
from io import StringIO
from unittest.mock import patch

from pytest import mark

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from yabi import main, to_bython
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


@mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_match():
    with patch("sys.stdout", new=StringIO()):
        main(["tests/match.by"])
        assert sys.stdout.getvalue() == "1\n2\n"


@mark.skipif(sys.implementation.name == "pypy" and sys.version_info < (3, 9), reason="pypy's console is weird on older versions")
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
    if sys.implementation.name == "pypy":
        code = code.replace("(", "{", 1)
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


def test_to_bython():
    assert to_bython("""
for i in {1, 2, 3}:
    if i % 2 == 1:
        print("Yes")
    else:
        print("No")
    """.strip()) == """
for (i in {1, 2, 3}) {
    if i % 2 == 1 {
        print("Yes")
    }
    else {
        print("No")
    }
}
    """.strip() + "\n"
