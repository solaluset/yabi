from __future__ import annotations
from typing import Generator, Iterable

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
BRACES = {
    "(": ")",
    "[": "]",
    "{": "}",
}
INDENT_SIZE = 4


def tokenize(text: str) -> Generator[str, None, None]:
    lexer = default_lexer()
    lexer.input(text)
    while token := lexer.token():
        yield token.value


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
            yield indent
            if tok.isspace():
                continue
        yield tok


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
        if self.head:
            result, *head = self.head
            if head:
                while head and head[0].isspace():
                    del head[0]
                while head and head[-1].isspace():
                    del head[-1]
                if pure_python and head and head[0] == "(" and head[-1] == ")":
                    if result != "except" or "as" in head:
                        head = head[1:-1]
                indent = " " * (depth - 1) * INDENT_SIZE
                result = (indent + result + " " + "".join(head)).rstrip()
            if pure_python:
                result += ":\n"
            else:
                result += " {\n"
        else:
            result = ""
        indent = " " * (depth) * INDENT_SIZE
        self.reindent(indent)
        body = "".join(
            child.unparse(pure_python, depth + 1) if isinstance(child, Block) else child
            for child in self.body
        )
        if not pure_python:
            if depth != 0:
                body += "\n}"
        elif not body or body.isspace():
            body = indent + "pass"
        return result + body


def parse(tokens: Iterable[str]) -> Block:
    brace_stack = []
    indent_stack = [""]
    result = Block()
    in_head = after_colon = finish_on_nl = False
    capture_indent = after_nl = after_indent = False
    after_async = False
    seen_lambdas = 0
    for tok in tokens:
        block_started = False
        if after_colon:
            if tok in {"\n", "#"}:
                after_colon = False
                capture_indent = True
            elif not tok.isspace():
                finish_on_nl = True
        if after_indent:
            after_indent = False
            if tok in {"\n", "#"}:
                indent_stack.pop()
                capture_indent = True
        if after_nl:
            after_nl = False
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
        elif tok in KEYWORDS and all(b[1] for b in brace_stack):
            if not after_async:
                result.append(Block())
                in_head = True
                after_async = tok == "async"
            else:
                after_async = False
        if in_head:
            if tok == "lambda":
                seen_lambdas += 1
                result.head_append(tok)
            elif tok == ":" and seen_lambdas:
                seen_lambdas -= 1
                result.head_append(tok)
            elif (
                tok in {"{", ":"}
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
            brace_stack.append((BRACES[tok], block_started))
            if block_started:
                indent_stack.append(None)
        elif tok in BRACES.values():
            brace, block_finished = brace_stack.pop()
            if brace != tok:
                raise SyntaxError(
                    f"closing parenthesis '{tok}' does not match"
                    f"opening parenthesis '{brace}'"
                )
            if block_finished:
                result.finish()
                skip = True
                assert (
                    indent_stack.pop() is None
                ), "indented block was not properly closed"
        if not skip:
            result.append(tok)
    for i in indent_stack:
        assert i is not None, "there is an unclosed block"
        result.finish()
    if not result.finished:
        raise SyntaxError("there is an unclosed block")
    return result


def to_pure_python(code: str):
    return parse(expand_semicolons(tokenize(code))).unparse()
