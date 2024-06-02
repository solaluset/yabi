import os
import sys
from io import StringIO
from unittest.mock import patch

from pytest import mark

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from yabi import main, to_bython
from yabi.console import YabiConsole


def check_file(file, expexted_output):
    with patch("sys.stdout", new=StringIO()):
        main([file])
        assert sys.stdout.getvalue() == expexted_output


def check_console(code, expexted_output):
    with patch("sys.stdout", new=StringIO()), patch("sys.stdin", new=StringIO(code)):
        YabiConsole().interact()
        assert sys.stdout.getvalue() == expexted_output


def test_regular():
    check_file("tests/hello.by", "Hello\n")


def test_mixed():
    check_file("tests/mixed.by", "3\n")


def test_dict():
    check_file("tests/dict.by", "1\n2\n3\nok\n")


def test_except():
    check_file("tests/except.by", "e\ndivision by zero\ne\ne\ndivision by zero\n")


def test_inline_if():
    check_file("tests/inline_if.by", "a\n")


def test_async():
    check_file("tests/async.by", "hello\n")


def test_nl_after_kw():
    check_file("tests/nl_after_kw.by", "1\n")


def test_lambda():
    check_file("tests/lambda.by", "Hello\n")


@mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_match():
    check_file("tests/match.by", "1\n2\n")


@mark.skipif(sys.implementation.name == "pypy" and sys.version_info < (3, 10), reason="pypy's console is weird on older versions")
def test_console():
    code = """
from __future__ import annotations
a: lol
1; 2; 3
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
    expexted_output = """
>>> >>> >>> 1
2
3
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
    check_console(code, expexted_output)


def test_console_linecont():
    code = """
if \\
  1: pass
    """.strip() + "\n\n"
    expexted_output = ">>> ... ... >>> "
    check_console(code, expexted_output)


def test_to_bython():
    assert to_bython("""
for i in {1, 2, 3}:
    if i % 2 == 1:
        print("Yes")
    else:
        print("No")
    """.strip())[0] == """
for (i in {1, 2, 3}) {
    if i % 2 == 1 {
        print("Yes")
    }
    else {
        print("No")
    }
}
    """.strip() + "\n"
