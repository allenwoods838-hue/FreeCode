"""Tools the model can call: files, shell, search. Stdlib only."""

from __future__ import annotations

import fnmatch
import os
import re
import shutil
import subprocess
from pathlib import Path

MAX_RESULT_CHARS = 30_000
MAX_FILE_READ_CHARS = 60_000
DEFAULT_READ_LIMIT = 2000

IGNORE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv",
    "env", ".tox", ".mypy_cache", ".pytest_cache", "dist", "build", ".next",
    ".cache", "target", ".idea", ".vscode", "site-packages", ".eggs",
}


# ----------------------------------------------------------------- schemas --

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read a text file. Returns numbered lines. Use offset/limit for "
                "large files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path (absolute or relative to cwd)"},
                    "offset": {"type": "integer", "description": "1-based line to start from", "default": 1},
                    "limit": {"type": "integer", "description": "Max lines to return", "default": DEFAULT_READ_LIMIT},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a new file or fully overwrite an existing one.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": (
                "Replace an exact string in a file. old_string must appear exactly "
                "once unless replace_all is true. Include surrounding context to be unique."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean", "default": False},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a shell command in the project directory. Output is truncated.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout": {"type": "integer", "description": "Seconds", "default": 120},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files by glob pattern (e.g. **/*.py). Returns sorted paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string", "description": "Directory to search (default: cwd)"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search file contents with a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Regular expression"},
                    "path": {"type": "string", "description": "File or directory (default: cwd)"},
                    "include": {"type": "string", "description": "Glob filter, e.g. *.py"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List a directory's entries (non-recursive).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "default": "."}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "think",
            "description": "Write down reasoning or a plan. Never returns anything; use it to think.",
            "parameters": {
                "type": "object",
                "properties": {"thoughts": {"type": "string"}},
                "required": ["thoughts"],
            },
        },
    },
]

TOOL_NAMES = [t["function"]["name"] for t in TOOLS]
# tools that never need approval
SAFE_TOOLS = {"read_file", "glob", "grep", "list_dir", "think"}
# tools that mutate state / run commands
RISKY_TOOLS = {"bash", "write_file", "edit_file"}


# ---------------------------------------------------------------- executor --


class ToolError(Exception):
    pass


class BackupStore:
    """Keeps pre-edit copies of files so /undo can restore them."""

    def __init__(self) -> None:
        self._stack: list[tuple[Path, str | None]] = []

    def snapshot(self, path: Path) -> None:
        if path.exists() and path.is_file():
            try:
                self._stack.append((path, path.read_text(encoding="utf-8", errors="replace")))
            except OSError:
                self._stack.append((path, None))
        else:
            self._stack.append((path, None))

    def undo(self) -> bool:
        if not self._stack:
            return False
        path, content = self._stack.pop()
        if content is None:
            if path.exists():
                path.unlink()
        else:
            path.write_text(content, encoding="utf-8")
        return True

    @property
    def count(self) -> int:
        return len(self._stack)


class Executor:
    def __init__(self, cwd: str | None = None, bash_timeout: int = 120,
                 backups: BackupStore | None = None) -> None:
        self.cwd = Path(cwd or os.getcwd()).resolve()
        self.bash_timeout = bash_timeout
        self.backups = backups or BackupStore()
        self.last_command: str | None = None

    # ------------------------------------------------------------ dispatch

    def run(self, name: str, args: dict) -> str:
        handler = getattr(self, f"_{name}", None)
        if handler is None:
            raise ToolError(f"Unknown tool: {name}")
        try:
            return handler(**args)
        except ToolError:
            raise
        except TypeError as exc:
            raise ToolError(f"Bad arguments for {name}: {exc}") from None
        except Exception as exc:  # noqa: BLE001 - report, don't crash the loop
            raise ToolError(f"{type(exc).__name__}: {exc}") from None

    def resolve(self, path: str) -> Path:
        p = Path(str(path)).expanduser()
        if not p.is_absolute():
            p = self.cwd / p
        return Path(os.path.normpath(str(p)))

    # ---------------------------------------------------------------- tools

    def _read_file(self, path: str, offset: int = 1, limit: int = DEFAULT_READ_LIMIT) -> str:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"File not found: {target}")
        if target.is_dir():
            raise ToolError(f"Is a directory: {target} (use list_dir)")
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ToolError(f"Cannot read {target}: {exc}") from None
        lines = text.split("\n")
        offset = max(1, int(offset or 1))
        limit = max(1, min(int(limit or DEFAULT_READ_LIMIT), 5000))
        window = lines[offset - 1 : offset - 1 + limit]
        width = len(str(offset + len(window) - 1))
        out = [
            f"{str(i).rjust(width)}→{line}"
            for i, line in enumerate(window, start=offset)
        ]
        body = "\n".join(out)
        if len(body) > MAX_FILE_READ_CHARS:
            body = body[:MAX_FILE_READ_CHARS] + "\n… (truncated)"
        if offset + len(window) - 1 < len(lines):
            body += f"\n… more lines ({len(lines) - offset - len(window) + 1} remaining)"
        return body or "(empty file)"

    def _write_file(self, path: str, content: str) -> str:
        target = self.resolve(path)
        if target.is_dir():
            raise ToolError(f"Is a directory: {target}")
        existed = target.exists()
        self.backups.snapshot(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        added = content.count("\n") + (1 if content else 0)
        verb = "Updated" if existed else "Created"
        return f"{verb} {target} ({added} lines, {len(content)} bytes)"

    def _edit_file(self, path: str, old_string: str, new_string: str,
                   replace_all: bool = False) -> str:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"File not found: {target}")
        try:
            text = target.read_text(encoding="utf-8")
        except OSError as exc:
            raise ToolError(f"Cannot read {target}: {exc}") from None
        if old_string == new_string:
            raise ToolError("old_string and new_string are identical")
        count = text.count(old_string)
        if count == 0:
            fuzzy = self._find_near(text, old_string)
            hint = f"\nClosest match found:\n{fuzzy}" if fuzzy else ""
            raise ToolError(f"old_string not found in {target}{hint}")
        if count > 1 and not replace_all:
            raise ToolError(
                f"old_string appears {count} times in {target}. "
                "Add surrounding context to make it unique, or set replace_all=true."
            )
        self.backups.snapshot(target)
        new_text = text.replace(old_string, new_string) if replace_all else \
            text.replace(old_string, new_string, 1)
        target.write_text(new_text, encoding="utf-8")
        n = count if replace_all else 1
        return f"Edited {target} ({n} replacement{'s' if n != 1 else ''})"

    @staticmethod
    def _find_near(text: str, needle: str) -> str:
        """Best-effort hint when an exact match fails (indentation issues etc.)."""
        target = needle.strip()
        if not target:
            return ""
        for i, line in enumerate(text.split("\n"), start=1):
            if target in line:
                return f"  line {i}: {line.strip()[:120]}"
        first = target.split("\n")[0][:60]
        if first:
            for i, line in enumerate(text.split("\n"), start=1):
                if first in line:
                    return f"  line {i}: {line.strip()[:120]}"
        return ""

    def _bash(self, command: str, timeout: int = 0) -> str:
        timeout = int(timeout or self.bash_timeout)
        timeout = max(5, min(timeout, 600))
        self.last_command = command
        env = dict(os.environ)
        env.setdefault("PAGER", "cat")
        env["TERM"] = env.get("TERM", "xterm-256color")
        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=str(self.cwd),
                capture_output=True,
                text=True,
                errors="replace",
                timeout=timeout,
                env=env,
                stdin=subprocess.DEVNULL,
            )
        except subprocess.TimeoutExpired:
            raise ToolError(f"Command timed out after {timeout}s: {command}") from None
        parts = []
        if proc.stdout:
            parts.append(proc.stdout)
        if proc.stderr:
            parts.append(proc.stderr if not proc.stdout else "--- stderr ---\n" + proc.stderr)
        if proc.returncode != 0:
            parts.append(f"(exit code {proc.returncode})")
        output = "\n".join(parts).strip() or "(no output)"
        if len(output) > MAX_RESULT_CHARS:
            half = MAX_RESULT_CHARS // 2
            output = output[:half] + "\n… (output truncated) …\n" + output[-half:]
        return output

    def _glob(self, pattern: str, path: str = ".") -> str:
        root = self.resolve(path)
        if not root.exists():
            raise ToolError(f"Directory not found: {root}")
        matches: list[str] = []
        # translate ** so pathlib handles it
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS and not d.startswith(".git")]
            rel_dir = os.path.relpath(dirpath, root)
            for name in filenames:
                rel = name if rel_dir == "." else f"{rel_dir}/{name}"
                if fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(name, pattern):
                    matches.append(str(Path(dirpath) / name))
                elif "**" in pattern and fnmatch.fnmatch(rel, pattern.replace("**/", "")):
                    matches.append(str(Path(dirpath) / name))
            if len(matches) > 500:
                break
        if not matches:
            return f"No files matched {pattern} under {root}"
        matches.sort(key=lambda p: (p.count(os.sep), p))
        shown = matches[:200]
        body = "\n".join(shown)
        if len(matches) > len(shown):
            body += f"\n… and {len(matches) - len(shown)} more"
        return body

    def _grep(self, pattern: str, path: str = ".", include: str = "") -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"Invalid regex {pattern!r}: {exc}") from None
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"Not found: {target}")
        hits: list[str] = []
        files_scanned = 0
        files = [target] if target.is_file() else self._walk(target, include)
        for file in files:
            files_scanned += 1
            if files_scanned > 4000:
                break
            try:
                if file.stat().st_size > 2_000_000:
                    continue
                text = file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for i, line in enumerate(text.split("\n"), start=1):
                if regex.search(line):
                    rel = os.path.relpath(file, self.cwd)
                    hits.append(f"{rel}:{i}: {line.strip()[:200]}")
                    if len(hits) >= 200:
                        break
            if len(hits) >= 200:
                break
        if not hits:
            return f"No matches for /{pattern}/ in {os.path.relpath(target, self.cwd) or '.'}"
        body = "\n".join(hits)
        if len(hits) == 200:
            body += "\n… (match limit reached)"
        return body

    def _walk(self, root: Path, include: str) -> list[Path]:
        out: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
            for name in filenames:
                if include and not fnmatch.fnmatch(name, include):
                    continue
                out.append(Path(dirpath) / name)
        return out

    def _list_dir(self, path: str = ".") -> str:
        target = self.resolve(path)
        if not target.exists():
            raise ToolError(f"Not found: {target}")
        if not target.is_dir():
            return self._read_file(str(target))
        entries = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        if not entries:
            return "(empty)"
        rows = []
        for entry in entries[:300]:
            name = entry.name + ("/" if entry.is_dir() else "")
            size = ""
            if entry.is_file():
                try:
                    size = f"{entry.stat().st_size}B"
                except OSError:
                    size = "?"
            rows.append(f"{name:<40} {size}")
        if shutil.which("git") and (target / ".git").exists():
            rows.append("\n(git repository)")
        return "\n".join(rows)

    def _think(self, thoughts: str) -> str:
        return "(noted)"


# --------------------------------------------------------------- commands --

BUILTIN_COMMANDS: dict[str, str] = {
    "help": "Show this help",
    "login": "Save an API key (hidden input) — /login openrouter|openai|anthropic",
    "logout": "Forget the stored API key",
    "clear": "Clear the conversation (keeps the session file)",
    "model": "Show or set the model (/model <name>)",
    "models": "List models available on the current server",
    "config": "Show configuration (/config key value to change it)",
    "compact": "Summarise history to free context",
    "undo": "Revert the last file edit/write",
    "sessions": "List saved sessions",
    "resume": "Resume a session (/resume <id>)",
    "cwd": "Print the working directory",
    "paste": "Multi-line input mode (finish with a lone '.')",
    "think": "Toggle showing the model's <think> blocks",
    "plan": "Toggle plan mode (read-only research, proposes a plan, no edits)",
    "exit": "Quit (Ctrl-D also works)",
}
