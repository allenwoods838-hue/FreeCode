"""ANSI colour helpers, terminal width and wrapping. Zero dependencies."""

from __future__ import annotations

import os
import re
import shutil
import sys
import unicodedata

# ---------------------------------------------------------------- palette ---


def _enabled() -> bool:
    if os.environ.get("NO_COLOR") is not None:
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    try:
        return sys.stdout.isatty()
    except Exception:
        return False


ENABLED = _enabled()

# name -> SGR parameters
_PALETTE: dict[str, str] = {
    # structure
    "accent": "38;2;120;190;255",      # soft blue
    "user": "1;38;2;255;190;120",      # warm amber (prompt)
    "assistant": "38;2;230;230;235",
    "dim": "2",
    "faint": "38;2;120;120;130",
    "error": "1;38;2;255;105;105",
    "warn": "1;38;2;255;200;80",
    "ok": "38;2;130;220;150",
    "tool": "38;2;200;150;255",
    "border": "38;2;70;70;80",
    "highlight": "7",
    # markdown
    "md_heading": "1;38;2;130;200;255",
    "md_quote": "38;2;150;190;150",
    "md_list": "38;2;255;170;100",
    "md_link": "38;2;120;190;255;4",
    "md_code_inline": "38;2;255;220;140",
    # syntax highlighting
    "syn_comment": "2;38;2;110;120;115",
    "syn_string": "38;2;190;230;160",
    "syn_keyword": "1;38;2;255;130;170",
    "syn_number": "38;2;255;190;140",
    "syn_func": "38;2;130;190;255",
    "syn_builtin": "38;2;200;170;255",
    "syn_decorator": "38;2;255;215;120",
    "syn_tag": "38;2;255;130;170",
    "syn_attr": "38;2;255;190;140",
}


def paint(text: str, *styles: str) -> str:
    """Colour *text* with one or more palette names / raw SGR params."""
    if not ENABLED or not text:
        return text
    codes = [_PALETTE.get(s, s) for s in styles if s]
    if not codes:
        return text
    return "\x1b[" + ";".join(codes) + "m" + text + "\x1b[0m"


def raw(sgr: str, text: str) -> str:
    if not ENABLED:
        return text
    return f"\x1b[{sgr}m{text}\x1b[0m"


_BOLD = "1"
_DIM = "2"
_ITALIC = "3"
_UNDERLINE = "4"


def bold(t: str) -> str:
    return raw(_BOLD, t)


def dim(t: str) -> str:
    return raw(_DIM, t)


def italic(t: str) -> str:
    return raw(_ITALIC, t)


def underline(t: str) -> str:
    return raw(_UNDERLINE, t)


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


def supports_truecolor() -> bool:
    if not ENABLED:
        return False
    ct = os.environ.get("COLORTERM", "")
    term = os.environ.get("TERM", "")
    return "truecolor" in ct or "direct" in ct or term.endswith("-direct")


# ----------------------------------------------------------------- layout ---

_WIDE = {"W", "F"}


def display_width(text: str) -> int:
    """Terminal columns used by *text* (ANSI ignored, CJK counted double)."""
    width = 0
    for ch in strip_ansi(text):
        if unicodedata.combining(ch):
            continue
        width += 2 if unicodedata.east_asian_width(ch) in _WIDE else 1
    return width


def term_width(fallback: int = 100) -> int:
    try:
        w = shutil.get_terminal_size((fallback, 24)).columns
    except Exception:
        w = fallback
    return max(40, min(w, 160))


def wrap(text: str, width: int | None = None, indent: str = "") -> str:
    """Wrap plain text to *width* columns, honouring ANSI sequences."""
    width = width or term_width()
    avail = max(10, width - display_width(indent))
    out: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            out.append("")
            continue
        line = ""
        line_w = 0
        for token in _split_keep_ansi(paragraph):
            tw = display_width(token)
            if line_w + tw > avail and line.strip():
                out.append(indent + line.rstrip())
                line, line_w = "", 0
            if tw > avail:  # hard-break very long tokens (e.g. urls)
                for ch in token:
                    cw = display_width(ch)
                    if line_w + cw > avail and line.strip():
                        out.append(indent + line.rstrip())
                        line, line_w = "", 0
                    line += ch
                    line_w += cw
                continue
            line += token
            line_w += tw
        if line.strip() or not out:
            out.append(indent + line.rstrip())
    return "\n".join(out)


def _split_keep_ansi(text: str) -> list[str]:
    """Split on spaces but never break ANSI escape sequences apart."""
    parts: list[str] = []
    buf = ""
    i = 0
    while i < len(text):
        if text[i] == "\x1b":
            m = ANSI_RE.match(text, i)
            if m:
                buf += m.group(0)
                i = m.end()
                continue
        ch = text[i]
        if ch == " ":
            parts.append(buf + " ")
            buf = ""
        else:
            buf += ch
        i += 1
    if buf:
        parts.append(buf)
    return parts


def truncate(text: str, limit: int) -> str:
    """Trim *text* to *limit* display columns, keeping ANSI intact."""
    if display_width(text) <= limit:
        return text
    out = ""
    w = 0
    i = 0
    while i < len(text):
        if text[i] == "\x1b":
            m = ANSI_RE.match(text, i)
            if m:
                out += m.group(0)
                i = m.end()
                continue
        ch = text[i]
        cw = 2 if unicodedata.east_asian_width(ch) in _WIDE else 1
        if w + cw > limit - 1:
            break
        out += ch
        w += cw
        i += 1
    return out + "…" + "\x1b[0m" if ENABLED else out + "…"


def ellipsize(text: str, limit: int) -> str:
    """Trim plain text to *limit* characters (no ANSI handling)."""
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)] + "…"
