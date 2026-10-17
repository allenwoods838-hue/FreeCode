"""Streaming Markdown renderer + lightweight syntax highlighter."""

from __future__ import annotations

import re

from . import ansi
from .ansi import dim, italic, paint, raw, term_width

# --------------------------------------------------------------- inline ----

_INLINE_RE = re.compile(
    r"(?P<code>`[^`]+`)"
    r"|(?P<bold>\*\*[^*]+\*\*|__[^_]+__)"
    r"|(?P<italic>(?<!\*)\*[^*\n]+\*(?!\*)|(?<!_)_[^_\n]+_(?!_))"
    r"|(?P<link>\[[^\]]+\]\([^)\s]+\))"
    r"|(?P<url>https?://[^\s)`]+)"
)


def render_inline(text: str) -> str:
    """Render inline markdown (code, bold, italic, links) with colour."""
    if not ansi.ENABLED:
        return text
    out: list[str] = []
    pos = 0
    for m in _INLINE_RE.finditer(text):
        out.append(_plain_with_style(text[pos : m.start()]))
        kind = m.lastgroup
        token = m.group(0)
        if kind == "code":
            out.append(paint(token[1:-1], "md_code_inline"))
        elif kind == "bold":
            out.append(raw("1", token[2:-2]))
        elif kind == "italic":
            out.append(italic(token[1:-1]))
        elif kind == "link":
            label = token[1 : token.index("](")]
            url = token[token.index("](") + 2 : -1]
            out.append(paint(label, "md_link") + " " + dim(f"({url})"))
        elif kind == "url":
            out.append(paint(token, "md_link"))
        pos = m.end()
    out.append(_plain_with_style(text[pos:]))
    return "".join(out)


def _plain_with_style(text: str) -> str:
    """Escape nothing; just leave plain text as-is."""
    return text


# ---------------------------------------------------------------- block ----

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_RE = re.compile(r"^(\s*)([-*+]|\d+[.)])\s+(.*)$")
_QUOTE_RE = re.compile(r"^\s*>\s?(.*)$")
_HR_RE = re.compile(r"^\s*([-*_])(\s*\1){2,}\s*$")


def render_line(line: str) -> str:
    """Render one complete Markdown line (block level)."""
    if not line:
        return ""
    m = _HEADING_RE.match(line)
    if m:
        level = len(m.group(1))
        colour = "md_heading" if level <= 2 else "accent"
        prefix = dim("› ") if level > 2 else ""
        return prefix + raw("1", paint(render_inline(m.group(2)), colour))
    if _HR_RE.match(line):
        return dim("─" * max(10, min(term_width(), 72)))
    m = _QUOTE_RE.match(line)
    if m:
        return dim("│ ") + paint(render_inline(m.group(1)), "md_quote")
    m = _LIST_RE.match(line)
    if m:
        indent, marker, rest = m.groups()
        bullet = marker if marker[0] not in "-*+" else "•"
        return indent + paint(bullet, "md_list") + " " + render_inline(rest)
    return render_inline(line)


# --------------------------------------------------- syntax highlighting ---

_KEYWORDS = {
    "python": {
        "and", "as", "assert", "async", "await", "break", "class", "continue",
        "def", "del", "elif", "else", "except", "finally", "for", "from",
        "global", "if", "import", "in", "is", "lambda", "nonlocal", "not",
        "or", "pass", "raise", "return", "try", "while", "with", "yield",
        "match", "case", "True", "False", "None", "self", "cls", "print",
    },
    "javascript": {
        "async", "await", "break", "case", "catch", "class", "const",
        "continue", "default", "delete", "do", "else", "export", "extends",
        "finally", "for", "function", "if", "import", "in", "instanceof",
        "let", "new", "of", "return", "super", "switch", "this", "throw",
        "try", "typeof", "var", "void", "while", "with", "yield", "static",
        "true", "false", "null", "undefined", "console", "require", "module",
    },
    "shell": {
        "if", "then", "else", "elif", "fi", "for", "while", "do", "done",
        "case", "esac", "function", "in", "export", "local", "return",
        "source", "alias", "set", "unset", "echo", "cd", "exit",
    },
    "json": {"true", "false", "null"},
    "rust": {
        "as", "async", "await", "break", "const", "continue", "crate", "dyn",
        "else", "enum", "extern", "fn", "for", "if", "impl", "in", "let",
        "loop", "match", "mod", "move", "mut", "pub", "ref", "return",
        "self", "Self", "static", "struct", "super", "trait", "type", "use",
        "where", "while", "true", "false", "None", "Some", "Ok", "Err",
    },
    "go": {
        "break", "case", "chan", "const", "continue", "default", "defer",
        "else", "fallthrough", "for", "func", "go", "goto", "if", "import",
        "interface", "map", "package", "range", "return", "select", "struct",
        "switch", "type", "var", "nil", "true", "false",
    },
}
_KEYWORDS["ts"] = _KEYWORDS["javascript"]
_KEYWORDS["typescript"] = _KEYWORDS["javascript"]
_KEYWORDS["js"] = _KEYWORDS["javascript"]
_KEYWORDS["jsx"] = _KEYWORDS["javascript"]
_KEYWORDS["tsx"] = _KEYWORDS["javascript"]
_KEYWORDS["bash"] = _KEYWORDS["shell"]
_KEYWORDS["sh"] = _KEYWORDS["shell"]
_KEYWORDS["zsh"] = _KEYWORDS["shell"]
_KEYWORDS["py"] = _KEYWORDS["python"]

_BUILTINS = {
    "python": {
        "len", "range", "open", "str", "int", "float", "list", "dict", "set",
        "tuple", "bool", "type", "isinstance", "super", "enumerate", "zip",
        "map", "filter", "sorted", "sum", "min", "max", "abs", "any", "all",
        "input", "print", "raise", "isinstance", "json", "os", "sys", "re",
    },
}

_STRING_RE = {
    "python": r"('''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'[^\n']*'|\"[^\n\"]*\")",
    "generic": r"('[^\n']*'|\"[^\n\"]*\"|`[^`]*`)",
}
_COMMENT_RE = {
    "python": r"(#[^\n]*)",
    "javascript": r"(//[^\n]*|/\*[\s\S]*?\*/)",
    "typescript": r"(//[^\n]*|/\*[\s\S]*?\*/)",
    "shell": r"(#[^\n]*)",
    "go": r"(//[^\n]*|/\*[\s\S]*?\*/)",
    "rust": r"(//[^\n]*|/\*[\s\S]*?\*/)",
    "json": r"(?!)",
    "html": r"(<!--[\s\S]*?-->)",
    "css": r"(/\*[\s\S]*?\*/)",
}


def _lang_key(lang: str) -> str:
    lang = (lang or "").lower().strip()
    return lang if lang in _KEYWORDS or lang in _COMMENT_RE else "generic"


def highlight(code: str, lang: str = "") -> str:
    """Colour *code* for *lang*. Falls back to a generic token pass."""
    if not ansi.ENABLED:
        return code
    key = _lang_key(lang)
    comment = _COMMENT_RE.get(key, r"(?!)")
    strings = _STRING_RE.get("python" if key == "python" else "generic")
    keywords = _KEYWORDS.get(key, set())
    builtins = _BUILTINS.get(key, set())

    pattern = re.compile(
        f"(?P<comment>{comment})|(?P<string>{strings})"
        r"|(?P<decorator>@@?[\w.]+)"
        r"|(?P<number>\b\d[\d_]*(?:\.\d+)?\b)"
        r"|(?P<word>[A-Za-z_][\w]*)"
    )

    out: list[str] = []
    pos = 0
    for m in pattern.finditer(code):
        out.append(code[pos : m.start()])
        kind = m.lastgroup or ""
        token = m.group(0)
        if kind == "comment":
            out.append(paint(token, "syn_comment"))
        elif kind == "string":
            out.append(paint(token, "syn_string"))
        elif kind == "decorator":
            out.append(paint(token, "syn_decorator"))
        elif kind == "number":
            out.append(paint(token, "syn_number"))
        else:
            nxt = code[m.end() : m.end() + 1]
            if token in keywords:
                out.append(paint(token, "syn_keyword"))
            elif token in builtins:
                out.append(paint(token, "syn_builtin"))
            elif nxt == "(":
                out.append(paint(token, "syn_func"))
            elif key in ("javascript", "html", "css") and nxt == ":":
                out.append(paint(token, "syn_attr"))
            else:
                out.append(token)
        pos = m.end()
    out.append(code[pos:])
    return "".join(out)


def render_code_block(code: str, lang: str = "") -> str:
    """A fenced code block with a dim left gutter."""
    body = highlight(code.rstrip("\n"), lang)
    lines = body.split("\n") if body else [""]
    gutter = paint("│", "border")
    return "\n".join(f"{gutter} {line}" for line in lines)


# ------------------------------------------------------- streaming output ---

_LINE_TOKEN_RE = re.compile(
    r"(?P<code>`[^`\n]+`)"
    r"|(?P<bold>\*\*[^*\n]+\*\*)"
    r"|(?P<link>\[[^\]]+\]\([^)\s]+\))"
)


class StreamWriter:
    """Feed raw model deltas, get back coloured text ready to print.

    Markdown block elements (headings, lists, quotes, code fences) are
    rendered as soon as a full line arrives, so output stays live while the
    model streams.
    """

    def __init__(self) -> None:
        self._line = ""
        self._in_code = False
        self._code_lang = ""
        self._code_buf: list[str] = []
        self._at_start = True  # last emit ended a line
        self._last_para = False
        self._last_char = ""
        self.text = ""  # raw accumulated text (no ANSI)

    # -- public ------------------------------------------------------------

    def feed(self, delta: str) -> str:
        """Consume *delta*, return printable chunk (may be '')."""
        if not delta:
            return ""
        self.text += delta
        self._line += delta
        out: list[str] = []
        while "\n" in self._line:
            line, self._line = self._line.split("\n", 1)
            out.append(self._render_complete(line))
        return "".join(out)

    def flush(self) -> str:
        """Render any trailing partial line."""
        if not self._line:
            return ""
        line, self._line = self._line, ""
        return self._render_complete(line, final=True)

    # -- internals ---------------------------------------------------------

    def _emit(self, text: str, block: bool = False) -> str:
        """Emit *text*; block elements get a blank line before them."""
        if not text:
            return ""
        prefix = ""
        if block and not self._at_start:
            prefix = "\n"
        self._at_start = text.endswith("\n")
        return prefix + text

    def _render_complete(self, line: str, final: bool = False) -> str:
        if self._in_code:
            if line.strip().startswith("```"):
                self._in_code = False
                block = "\n".join(self._code_buf)
                self._code_buf = []
                self._last_para = False
                self._last_char = ""
                body = render_code_block(block, self._code_lang)
                return self._emit(body + "\n", block=True) if block.strip() else ""
            self._code_buf.append(line)
            return ""  # revealed when the fence closes
        if line.strip().startswith("```"):
            fence = line.strip()[3:]
            self._in_code = True
            self._code_lang = fence
            self._code_buf = []
            return ""
        if not line.strip():
            self._last_para = False
            if self._at_start or final:
                return ""
            self._at_start = True
            self._last_char = ""
            return "\n"
        if _needs_own_line(line):
            self._last_para = False
            self._last_char = line.rstrip()[-1:]
            return self._emit(render_line(line) + "\n", block=True)
        # paragraph line: keep source line breaks, never glue words together
        sep = " " if (self._last_para and self._last_char
                      and not self._last_char.isspace()) else ""
        self._last_para = True
        self._last_char = line.rstrip()[-1:]
        return self._emit(sep + render_line(line) + ("\n" if final else ""))


def _needs_own_line(line: str) -> bool:
    return bool(
        _HEADING_RE.match(line)
        or _HR_RE.match(line)
        or _LIST_RE.match(line)
        or _QUOTE_RE.match(line)
    )


def inline_hint(text: str, width: int | None = None) -> str:
    """Single-line coloured rendering used for prompts/echoes."""
    width = width or term_width()
    rendered = render_inline(text)
    return ansi.truncate(rendered, width)
