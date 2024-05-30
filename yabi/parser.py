from __future__ import annotations
from io import StringIO
from random import choices
from types import ModuleType
from string import ascii_letters
from typing import Generator, Iterable
from tokenize import NAME, OP, generate_tokens

from pypp.parser import default_lexer


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


def _get_head_terminator(tokens: list[tokens], start: int, keywords: set) -> str | None:
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
                if not _is_op(next_tok):
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


def _gen_lambda_name():
    return "_yabi_lambda_" + "".join(choices(ascii_letters, k=16))


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
"""
LAMBDA_WRAPPER = """
__all__.append("{name}")
def {name}():
    tree = ast.parse({code})
    locals = _getframe().f_back.f_locals
    tree.body[0].args.args = [_make_arg(key) for key in locals]
    ns = {{}}
    exec(compile(tree, "<yabi-lambda>", "exec"), ns)
    return ns["yabi_lambda_wrapper"](**locals)
"""


def _parse_long_lambda(tokens: list[str], i: int, result: Block, lambda_module) -> int:
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
    result.append(name + "()")
    body.head = list(tokenize("def _yabi_lambda" + head))
    parsed_body, module = parse(body.body)
    body.body = parsed_body.body
    code = f"def yabi_lambda_wrapper():\n" + body.unparse(depth=2) + "\n" + " " * INDENT_SIZE + "return _yabi_lambda\n"
    code = LAMBDA_WRAPPER.format(name=name, code=repr(code))
    if not lambda_module:
        lambda_module = ModuleType("yabi_lambdas")
        exec(LAMBDA_MODULE_HEAD, vars(lambda_module))
    if module:
        for attr in module.__all__:
            lambda_module.__all__.append(attr)
            setattr(lambda_module, attr, getattr(module, attr))
    exec(code, vars(lambda_module))
    return i + 1, lambda_module


def parse(tokens: Iterable[str]) -> Block:
    brace_stack = []
    indent_stack = [""]
    result = Block()
    in_head = after_colon = finish_on_nl = False
    capture_indent = after_indent = False
    after_nl = True
    accept_keyword = False
    seen_lambdas = 0
    lambda_module = None
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
        elif accept_keyword:
            if not tok.isspace():
                accept_keyword = False
            if not in_head:
                head_term = _get_head_terminator(tokens, i, KEYWORDS) or _get_head_terminator(tokens, i, SOFT_KEYWORDS)
                if head_term and all(b[1] for b in brace_stack):
                    result.append(Block())
                    in_head = True
        if _get_head_terminator(tokens, i, {"lambda"}) == "{":
            i, lambda_module = _parse_long_lambda(tokens, i, result, lambda_module)
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
            except IndexError:
                raise SyntaxError(f"unmatched '{tok}'")
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
    return result, lambda_module


def _transform(code: str, python: bool) -> str:
    result, module = parse(expand_semicolons(tokenize(code + "\n")))
    if module:
        result.body.insert(0, "from yabi_lambdas import *\n")
    return result.unparse(python), module


def to_pure_python(code: str) -> str:
    return _transform(code, True)


def to_bython(code: str) -> str:
    return _transform(code, False)
