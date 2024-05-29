from __future__ import annotations
from io import StringIO
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
        i = 0
        while i < len(self.body):
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
                if self.body[i].isspace():
                    self.body[i] = indent
                else:
                    self.body.insert(i, indent)
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
        return result + "\n" + body.lstrip("\n")


def _is_op(token: str) -> bool:
    if token in BRACES or token in BRACES.values():
        return False
    return get_token_type(token) == OP


def _get_head_terminator(tokens: list[tokens], start: int, soft: bool) -> str | None:
    if tokens[start] not in (SOFT_KEYWORDS if soft else KEYWORDS):
        return None
    tokens_range = range(start + 1, len(tokens))
    try:
        first_token = next(tokens[i] for i in tokens_range if not tokens[i].isspace())
    except StopIteration:
        return None
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


def parse(tokens: Iterable[str]) -> Block:
    brace_stack = []
    indent_stack = [""]
    result = Block()
    in_head = after_colon = finish_on_nl = False
    capture_indent = after_indent = False
    after_nl = True
    accept_keyword = False
    seen_lambdas = 0
    tokens = list(tokens)
    for i, tok in enumerate(tokens):
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
                head_term = _get_head_terminator(tokens, i, False) or _get_head_terminator(tokens, i, True)
                if head_term and all(b[1] for b in brace_stack):
                    result.append(Block())
                    in_head = True
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
    for i in indent_stack:
        if i is None:
            raise SyntaxError(UNCLOSED_BLOCK_ERROR)
        result.finish()
    if not result.finished:
        raise SyntaxError(UNCLOSED_BLOCK_ERROR)
    return result


def _transform(code: str, python: bool) -> str:
    result = parse(expand_semicolons(tokenize(code + "\n")))
    return result.unparse(python)


def to_pure_python(code: str) -> str:
    return _transform(code, True)


def to_bython(code: str) -> str:
    return _transform(code, False)
