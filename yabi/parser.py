from __future__ import annotations
import os
from io import StringIO
from hashlib import md5
from random import choices
from types import ModuleType
from string import ascii_letters
from typing import Generator, Iterable
from tokenize import NAME, OP, generate_tokens

from pypp.parser import default_lexer

from . import config


KEYWORDS = {
    "async",
    "class",
    "def",
    "if",
    "elif",
    "else",
    "try",
    "except",
    "finally",
    "for",
    "while",
    "with",
}
SOFT_KEYWORDS = {
    "match",
    "case",
}
BRACES = {
    "(": ")",
    "[": "]",
    "{": "}",
}
INDENT_SIZE = 4
UNCLOSED_BLOCK_ERROR = "there is an unclosed block"
DIR_NAME = "_yabi_lambdas"
LAMBDA_MODULE_HEAD = """
__all__ = []

import ast
try:
    from sys import _getframe
except ImportError:
    def _getframe():
        try:
            raise Exception
        except Exception as e:
            return e.__traceback__.tb_frame.f_back


def _make_arg(key):
    arg = ast.arg(key)
    arg.lineno = arg.col_offset = 0
    return arg


def _make_return(value):
    ret = ast.Return(value)
    ret.lineno = ret.col_offset = 0
    return ret


def _make_func(code, locals_):
    tree = ast.parse(code)
    body = tree.body[0].body[0].body
    if isinstance(body[-1], ast.Expr):
        body[-1] = _make_return(body[-1].value)
    tree.body[0].args.args = [_make_arg(key) for key in locals_]
    tree.body[0].args.kwarg = _make_arg(".")

    ns = {}
    exec(compile(tree, "<yabi-lambda>", "exec"), None, ns)
    return ns["yabi_lambda_wrapper"]
"""
LAMBDA_WRAPPER = """
__all__.append("{name}")

{name}_func = None

def {name}():
    global {name}_func

    locals_ = _getframe().f_back.f_locals
    if {name}_func is not None:
        return {name}_func(**locals_)

    {name}_func = _make_func({code}, locals_)
    return {name}_func(**locals_)
"""


def tokenize(text: str) -> Generator[str, None, None]:
    lexer = default_lexer()
    lexer.input(text)
    while token := lexer.token():
        yield token.value


def get_token_type(token: str) -> int:
    tok = next(generate_tokens(StringIO(token).readline))
    return tok.type


def expand_semicolons(tokens: Iterable[str]) -> Generator[str, None, None]:
    after_nl = True
    after_semicolon = False
    for tok in tokens:
        if tok == "\n":
            after_nl = True
            after_semicolon = False
        elif after_nl:
            after_nl = False
            if tok.isspace():
                indent = tok
            else:
                indent = ""
        if tok == ";":
            after_semicolon = True
            continue
        elif after_semicolon:
            after_semicolon = False
            yield "\n"
            if indent:
                yield indent
            if tok.isspace():
                continue
        yield tok


def _has_opening_brace(tokens: list[str]) -> bool:
    stack = []
    for tok in tokens:
        if tok in BRACES:
            stack.append(tok)
        elif tok in BRACES.values():
            stack.pop()
        if stack == ["{"]:
            return True
    return False


class Block:
    def __init__(self):
        self.head = []
        self.body = []
        self.finished = False

    def _child_unfinished(self) -> bool:
        return (
            bool(self.body)
            and isinstance(self.body[-1], Block)
            and not self.body[-1].finished
        )

    def append(self, part: str | Block):
        if self._child_unfinished():
            self.body[-1].append(part)
        else:
            self.body.append(part)

    def head_append(self, part: str):
        if self._child_unfinished():
            self.body[-1].head_append(part)
        else:
            self.head.append(part)

    def finish(self):
        if self._child_unfinished():
            self.body[-1].finish()
        else:
            if self.finished:
                raise SyntaxError("the block was already closed")
            self.finished = True

    def reindent(self, indent: str):
        after_nl = True
        braces_opened = 0
        i = 0
        while i < len(self.body):
            if self.body[i] in BRACES:
                braces_opened += 1
            elif self.body[i] in BRACES.values():
                braces_opened -= 1
            if isinstance(self.body[i], Block):
                if (
                    i != 0
                    and not isinstance(self.body[i - 1], Block)
                    and self.body[i - 1] != "\n"
                    and self.body[i - 1].isspace()
                ):
                    del self.body[i - 1]
                    i -= 1
                if i + 1 == len(self.body) or self.body[i + 1] != "\n":
                    self.body.insert(i + 1, "\n")
            elif self.body[i] == "\n":
                after_nl = True
            elif after_nl:
                after_nl = False
                if not braces_opened:
                    if self.body[i].isspace():
                        self.body[i] = indent
                    else:
                        self.body.insert(i, indent)
                        i += 1
            i += 1

    def unparse(self, pure_python=True, depth=0) -> str:
        outer_indent = (depth - 1) * INDENT_SIZE * " "
        inner_indent = depth * INDENT_SIZE * " "
        if self.head:
            result, *head = self.head
            if head:
                while head and head[0].isspace():
                    del head[0]
                while head and head[-1].isspace():
                    del head[-1]
                if pure_python:
                    if (
                        (head and head[0] == "(" and head[-1] == ")")
                        and (result != "except" or "as" in head)
                        and result not in {"if", "elif", "while"}
                    ):
                        head = head[1:-1]
                    head = [i for i in head if i != "\n"]
                elif _has_opening_brace(head):
                    head.insert(0, "(")
                    head.append(")")
            result = (outer_indent + result + " " + "".join(head)).rstrip()
            if pure_python:
                result += ":"
            else:
                result += " {"
        else:
            result = ""
        self.reindent(inner_indent)
        body = "".join(
            child.unparse(pure_python, depth + 1) if isinstance(child, Block) else child
            for child in self.body
        )
        if not pure_python:
            if depth != 0:
                body = body.rstrip(" ")
                stripped = body.rstrip("\n")
                nls = len(body) - len(stripped) - 1
                body = stripped + "\n" + outer_indent + "}" + "\n" * nls
        elif not body or body.isspace():
            body = inner_indent + "pass"
        if result:
            result += "\n"
        return result + body.lstrip("\n")


def _is_op(token: str) -> bool:
    if token in BRACES or token in BRACES.values():
        return False
    return get_token_type(token) == OP


def _get_head_terminator(tokens: list[tokens], start: int, keywords: set, only_colon: bool = False) -> str | None:
    if tokens[start] not in keywords:
        return None
    tokens_range = range(start + 1, len(tokens))
    try:
        first_token = next(tokens[i] for i in tokens_range if not tokens[i].isspace())
    except StopIteration:
        return None
    if first_token == ":":
        return ":"
    if _is_op(first_token):
        return None
    brace_stack = []
    after_nl = False
    potential_block = False
    prev_tok = None
    for i in tokens_range:
        tok = tokens[i]
        if tok == "\n" and not brace_stack:
            after_nl = True
        if tok.isspace():
            continue
        if tok in BRACES:
            brace_stack.append(tok)
        elif tok in BRACES.values():
            try:
                if BRACES[brace_stack.pop()] != tok:
                    return None
            except IndexError:
                return None
        if not brace_stack:
            if tok == ":":
                return ":"
            if potential_block:
                next_tok = next((tokens[j] for j in range(i + 1, len(tokens)) if not tokens[j].isspace()), None)
                if not (next_tok == ":" if only_colon else _is_op(next_tok)):
                    return "{"
                return ":"
        if brace_stack == ["{"] and not _is_op(prev_tok):
            potential_block = True
        if after_nl:
            after_nl = False
            if not potential_block and get_token_type(tok) == NAME:
                return None
        prev_tok = tok
    return "{" if potential_block else None


def _gen_lambda_name() -> str:
    return "_yabi_lambda_" + "".join(choices(ascii_letters, k=16))


def _parse_long_lambda(tokens: list[str], i: int, result: Block, lambda_module_code: str, async_lambda: bool, in_head: bool) -> tuple[int, str]:
    i += 1
    brace_stack = []
    body = Block()
    in_body = False
    while i < len(tokens):
        tok = tokens[i]
        if tok in BRACES:
            brace_stack.append(tok)
        elif tok in BRACES.values():
            brace_stack.pop()
        if not in_body and brace_stack == ["{"]:
            in_body = True
            i += 1
            continue
        if in_body:
            if not brace_stack:
                break
            body.append(tok)
        else:
            body.head_append(tok)
        i += 1
    if brace_stack:
        raise SyntaxError(UNCLOSED_BLOCK_ERROR)
    head = "".join(body.head).strip()
    if not head.startswith("(") or not head.endswith(")"):
        head = "(" + head + ")"
    name = _gen_lambda_name()
    if in_head:
        result.head_append(name + "()")
    else:
        result.append(name + "()")
    body.head = list(tokenize(("async " if async_lambda else "") + "def _yabi_lambda" + head))
    parsed_body, module_code = parse(body.body)
    body.body = parsed_body.body
    code = f"def yabi_lambda_wrapper():\n" + body.unparse(depth=2) + "\n" + " " * INDENT_SIZE + "return _yabi_lambda\n"
    code = LAMBDA_WRAPPER.format(name=name, code=repr(code))
    if not lambda_module_code:
        lambda_module_code = LAMBDA_MODULE_HEAD
    lambda_module_code += code
    lambda_module_code += module_code.replace(LAMBDA_MODULE_HEAD, "")
    return i + 1, lambda_module_code


def parse(tokens: Iterable[str]) -> tuple[Block, str]:
    brace_stack = []
    indent_stack = [""]
    result = Block()
    in_head = after_colon = finish_on_nl = False
    capture_indent = after_indent = False
    after_nl = True
    accept_keyword = False
    async_lambda = False
    seen_lambdas = 0
    lambda_module_code = ""
    tokens = list(tokens)
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        block_started = False
        if after_colon:
            if tok in {"\n", "#"}:
                after_colon = False
                capture_indent = True
            elif not tok.isspace():
                after_colon = False
                finish_on_nl = True
        if after_indent:
            after_indent = False
            if tok in {"\n", "#"}:
                if len(indent_stack) > 1:
                    indent_stack.pop()
                capture_indent = True
        if after_nl:
            after_nl = False
            accept_keyword = True
            if tok not in {"\n", "#"}:
                indent = tok if tok.isspace() else ""
                while indent_stack[-1] is not None and not indent.startswith(
                    indent_stack[-1]
                ):
                    result.append(indent_stack.pop())
                    result.finish()
                if capture_indent:
                    capture_indent = False
                    after_indent = True
                    if indent:
                        indent_stack.append(indent)
        if tok == "\n":
            after_nl = True
            if finish_on_nl:
                finish_on_nl = False
                result.finish()
        elif tok == "async" and next((tokens[j] for j in range(i + 1, len(tokens)) if not tokens[j].isspace()), None) == "lambda":
            async_lambda = True
            i += 1
            continue
        elif accept_keyword:
            if not tok.isspace():
                accept_keyword = False
            if not in_head:
                head_term = _get_head_terminator(tokens, i, KEYWORDS) or _get_head_terminator(tokens, i, SOFT_KEYWORDS)
                if head_term and all(b[1] for b in brace_stack):
                    result.append(Block())
                    in_head = True
        terminator = _get_head_terminator(tokens, i, {"lambda"}, True)
        if terminator == "{" or async_lambda:
            if tok.isspace():
                i += 1
                continue
            if terminator != "{":
                raise SyntaxError("async lambda must use braces")
            i, lambda_module_code = _parse_long_lambda(tokens, i, result, lambda_module_code, async_lambda, in_head)
            async_lambda = False
            continue
        if in_head:
            if tok == "lambda":
                seen_lambdas += 1
                result.head_append(tok)
            elif tok == ":" and seen_lambdas:
                seen_lambdas -= 1
                result.head_append(tok)
            elif (
                tok == head_term
                and all(b[1] for b in brace_stack)
                and not seen_lambdas
            ):
                in_head = False
                block_started = True
                after_colon = tok == ":"
            else:
                result.head_append(tok)
        skip = in_head or block_started
        if tok in BRACES:
            brace_stack.append((tok, block_started))
            if block_started:
                indent_stack.append(None)
                accept_keyword = True
        elif tok in BRACES.values():
            try:
                brace, block_finished = brace_stack.pop()
            except IndexError as e:
                raise SyntaxError(f"unmatched '{tok}'") from e
            accept_keyword = block_finished
            if BRACES[brace] != tok:
                raise SyntaxError(
                    f"closing parenthesis '{tok}' does not match"
                    f" opening parenthesis '{brace}'"
                )
            if block_finished:
                result.finish()
                skip = True
                if indent_stack.pop() is not None:
                    raise SyntaxError("indented block was not properly closed")
        if not skip:
            result.append(tok)
        i += 1
    for i in indent_stack:
        if i is None:
            raise SyntaxError(UNCLOSED_BLOCK_ERROR)
        result.finish()
    if not result.finished:
        raise SyntaxError(UNCLOSED_BLOCK_ERROR)
    return result, lambda_module_code


def _insert_import(body: list[Block | str], import_: str):
    after_nl = True
    braces_seen = 0
    for i, tok in enumerate(body):
        if tok == "\n":
            after_nl = True
        if isinstance(tok, str) and tok.isspace():
            continue
        if after_nl:
            after_nl = False
            if not braces_seen and tok not in {"#", "import", "from"}:
                body.insert(i, import_)
                break
        if tok in BRACES:
            braces_seen += 1
        elif tok in BRACES.values():
            braces_seen -= 1


def _transform(code: str, python: bool) -> tuple[str, ModuleType | None]:
    result, module_code = parse(expand_semicolons(tokenize(code + "\n")))
    if module_code:
        code_hash = md5(code.encode()).hexdigest()
        basename = "l_" + code_hash
        module = ModuleType(DIR_NAME + "." + basename)
        exec(module_code, vars(module))
        if config.SAVE_FILES:
            if not os.path.isdir(DIR_NAME):
                os.mkdir(DIR_NAME)
            with open(os.path.join(DIR_NAME, basename) + ".py", "w") as f:
                f.write(module_code)
        _insert_import(result.body, f"from {module.__name__} import *\n")
    else:
        module = None
    return result.unparse(python), module


def to_pure_python(code: str) -> tuple[str, ModuleType | None]:
    return _transform(code, True)


def to_bython(code: str) -> tuple[str, ModuleType | None]:
    return _transform(code, False)
