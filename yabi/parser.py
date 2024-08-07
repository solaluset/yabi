from __future__ import annotations
import ast
import random
from io import StringIO
from builtins import compile
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


def strip_spaces(tokens: list[str]) -> None:
    while tokens and tokens[0].isspace():
        del tokens[0]
    while tokens and tokens[-1].isspace():
        del tokens[-1]


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

    def insert_lambda(self, part: str | Block, brace_stack):
        if self._child_unfinished():
            if not self.body[-1].body:
                self.body.insert(-1, part)
            else:
                self.body[-1].insert_lambda(part, brace_stack)
        else:
            i = len(self.body)
            if not brace_stack:
                i -= 1
            else:
                while brace_stack:
                    i -= 1
                    if self.body[i] == brace_stack[-1]:
                        brace_stack.pop()
            while i > 0 and self.body[i] != "\n":
                i -= 1
            self.body.insert(i, part)
            self.body.insert(i, "\n")

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
                strip_spaces(head)
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
            (
                child.unparse(pure_python, depth + 1)
                if isinstance(child, Block)
                else child
            )
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
            stripped_body = body.lstrip(" ")
            if stripped_body.startswith("#"):
                return result + "  " + stripped_body
            result += "\n"
        return result + body.lstrip("\n")


def _is_op(token: str) -> bool:
    if token in BRACES or token in BRACES.values():
        return False
    return get_token_type(token) == OP


def _get_head_terminator(
    tokens: list[str], start: int, keywords: set, only_colon: bool = False
) -> str | None:
    if tokens[start] not in keywords:
        return None
    tokens_range = range(start + 1, len(tokens))
    try:
        first_token = next(
            tokens[i] for i in tokens_range if not tokens[i].isspace()
        )
    except StopIteration:
        return None
    if first_token == ":":
        return ":"
    if first_token.endswith("="):
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
                next_tok = next(
                    (
                        tokens[j]
                        for j in range(i + 1, len(tokens))
                        if not tokens[j].isspace()
                    ),
                    None,
                )
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


def _insert_into_line(line: str, index: int, part: str):
    line = list(line)
    line.insert(index, part)
    return "".join(line)


def _add_return(code: str) -> str:
    # same as `ast.parse`, but with non-overriden compile
    tree = compile(code, "<string>", "exec", ast.PyCF_ONLY_AST)
    last_node = tree.body[0].body[-1]
    if not isinstance(last_node, ast.Expr):
        return code
    code = code.splitlines()
    code[last_node.end_lineno - 1] = _insert_into_line(
        code[last_node.end_lineno - 1],
        last_node.end_col_offset + 1,
        ")",
    )
    code[last_node.lineno - 1] = _insert_into_line(
        code[last_node.lineno - 1],
        last_node.col_offset,
        "return (",
    )
    return "\n".join(code)


def _gen_lambda_name() -> str:
    return "_yabi_lambda_" + "".join(random.choices(ascii_letters, k=16))


class Parser:
    def __init__(self, tokens: Iterable[str]):
        self.brace_stack = []
        self.indent_stack = [""]
        self.result = None
        self.in_head = False
        self.after_colon = False
        self.finish_on_nl = False
        self.capture_indent = False
        self.after_indent = False
        self.after_nl = True
        self.accept_keyword = False
        self.async_lambda = False
        self.seen_lambdas = 0
        self.block_started = False
        self.skip = False
        self.next_indent = None
        self.head_term = None
        self.tokens = [tok for tok in tokens if tok]
        self.i = 0

    def parse(self):
        if self.result is not None:
            return self.result
        self.result = Block()
        self._parse()
        return self.result

    def _parse(self):
        while self.i < len(self.tokens):
            tok = self.tokens[self.i]
            if self.next_indent is not None:
                if not tok.isspace():
                    self.i -= 1
                tok = self.next_indent
                self.next_indent = None
            self.block_started = False
            if tok == "#":
                while (
                    self.i < len(self.tokens) and self.tokens[self.i] != "\n"
                ):
                    self.result.append(self.tokens[self.i])
                    self.i += 1
                continue
            if self.after_colon:
                self._parse_after_colon(tok)
            elif self.after_indent:
                self._parse_after_indent(tok)
            elif self.after_nl:
                self._parse_after_nl(tok)
            if tok == ";" and not self.finish_on_nl:
                tok = "\n"
                self.next_indent = self.indent_stack[-1]
            if tok == "\n":
                self.after_nl = True
                if self.finish_on_nl:
                    self.finish_on_nl = False
                    self.result.finish()
            elif tok == "async" and self._next_nonspace(self.i) == "lambda":
                self.async_lambda = True
                self.i += 1
                continue
            elif self.accept_keyword:
                self._parse_accept_keyword(tok)
            if self._parse_long_lambda(tok):
                continue
            if self.in_head:
                self._parse_in_head(tok)

            self.skip = self.in_head or self.block_started

            if tok in BRACES:
                self._parse_tok_in_braces(tok)
            elif tok in BRACES.values():
                self._parse_tok_in_braces_values(tok)

            if not self.skip:
                self.result.append(tok)
            self.i += 1

        for indent in self.indent_stack:
            if indent is None:
                raise SyntaxError(UNCLOSED_BLOCK_ERROR)
            self.result.finish()
        if not self.result.finished:
            raise SyntaxError(UNCLOSED_BLOCK_ERROR)

    def _next_nonspace(self, i: int) -> str | None:
        return next(
            (
                self.tokens[j]
                for j in range(i + 1, len(self.tokens))
                if not self.tokens[j].isspace()
            ),
            None,
        )

    def _parse_after_colon(self, tok: str):
        if tok == "\n":
            self.after_colon = False
            self.capture_indent = True
        elif not tok.isspace():
            self.after_colon = False
            self.finish_on_nl = True

    def _parse_after_indent(self, tok: str):
        self.after_indent = False
        if tok == "\n":
            if len(self.indent_stack) > 1:
                self.indent_stack.pop()
            self.capture_indent = True

    def _parse_after_nl(self, tok: str):
        self.after_nl = False
        if all(b[1] for b in self.brace_stack):
            self.accept_keyword = True
            if tok != "\n":
                indent = tok if tok.isspace() else ""
                while self.indent_stack[
                    -1
                ] is not None and not indent.startswith(self.indent_stack[-1]):
                    self.result.append(self.indent_stack.pop())
                    self.result.finish()
                if self.capture_indent:
                    self.capture_indent = False
                    self.after_indent = True
                    if indent:
                        self.indent_stack.append(indent)

    def _parse_accept_keyword(self, tok: str):
        if not tok.isspace():
            self.accept_keyword = False
        if not self.in_head:
            self.head_term = _get_head_terminator(
                self.tokens, self.i, KEYWORDS
            ) or _get_head_terminator(self.tokens, self.i, SOFT_KEYWORDS)
            if self.head_term and all(b[1] for b in self.brace_stack):
                self.result.append(Block())
                self.in_head = True

    def _parse_long_lambda(self, tok: str) -> bool:
        terminator = _get_head_terminator(
            self.tokens, self.i, {"lambda"}, True
        )
        if terminator == "{" or self.async_lambda:
            if tok.isspace():
                self.i += 1
                return True
            if terminator != "{":
                raise SyntaxError("async lambda must use braces")
            lambda_body = self._fully_parse_long_lambda()
            brace_stack_copy = []
            for brace, is_block in reversed(self.brace_stack):
                if is_block:
                    break
                brace_stack_copy.append(brace)
            brace_stack_copy.reverse()
            self.result.insert_lambda(lambda_body, brace_stack_copy)
            self.async_lambda = False
            return True
        return False

    def _fully_parse_long_lambda(self) -> Block:
        self.i += 1
        brace_stack = []
        head = []
        body_start = None
        in_body = False
        while self.i < len(self.tokens):
            tok = self.tokens[self.i]
            if tok in BRACES:
                brace_stack.append(tok)
            elif tok in BRACES.values():
                brace_stack.pop()
            if not in_body and brace_stack == ["{"]:
                in_body = True
                body_start = self.i
            if in_body:
                if not brace_stack:
                    break
            else:
                head.append(tok)
            self.i += 1
        if brace_stack:
            raise SyntaxError(UNCLOSED_BLOCK_ERROR)

        name = _gen_lambda_name()
        if self.in_head:
            self.result.head_append(name)
        else:
            self.result.append(name)

        strip_spaces(head)
        if not head or not head[0] == "(" or not head[-1] == ")":
            head.insert(0, "(")
            head.append(")")
        head = ["def", " ", name] + head
        if self.async_lambda:
            head = ["async", " "] + head

        # reparse because inner Bython code was not processed
        body = parse(head + self.tokens[body_start : self.i + 1]).body[0]
        # reparse again because we need to add return
        code = _add_return(body.unparse(depth=1))
        body = parse(tokenize(code)).body[0]
        self.i += 1
        return body

    def _parse_in_head(self, tok: str):
        if tok == "lambda":
            self.seen_lambdas += 1
            self.result.head_append(tok)
        elif tok == ":" and self.seen_lambdas:
            self.seen_lambdas -= 1
            self.result.head_append(tok)
        elif (
            tok == self.head_term
            and all(b[1] for b in self.brace_stack)
            and not self.seen_lambdas
        ):
            self.in_head = False
            self.block_started = True
            self.after_colon = tok == ":"
        else:
            self.result.head_append(tok)

    def _parse_tok_in_braces(self, tok: str):
        self.brace_stack.append((tok, self.block_started))
        if self.block_started:
            self.indent_stack.append(None)
            self.accept_keyword = True

    def _parse_tok_in_braces_values(self, tok: str):
        try:
            brace, block_finished = self.brace_stack.pop()
        except IndexError as e:
            raise SyntaxError(f"unmatched '{tok}'") from e
        self.accept_keyword = block_finished
        if BRACES[brace] != tok:
            raise SyntaxError(
                f"closing parenthesis '{tok}' does not match"
                f" opening parenthesis '{brace}'"
            )
        if block_finished:
            self.result.finish()
            self.skip = True
            if self.indent_stack.pop() is not None:
                raise SyntaxError("indented block was not properly closed")


def parse(tokens: Iterable[str]) -> Block:
    parser = Parser(tokens)
    return parser.parse()


def _transform(code: str, python: bool) -> str:
    result = parse(tokenize(code + "\n"))
    return result.unparse(python)


def to_pure_python(code: str) -> str:
    return _transform(code, True)


def to_bython(code: str) -> str:
    return _transform(code, False)
