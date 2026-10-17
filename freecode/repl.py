"""Interactive REPL: streaming output, tool cards, slash commands."""

from __future__ import annotations

import getpass
import json
import os
import sys
import threading
import time
from pathlib import Path

from . import ansi, markdown
from .agent import Agent, AgentCallbacks, TurnReport
from .config import Config, list_sessions, load_session, new_session_id, save_session
from .llm import LLMError
from .tools import BUILTIN_COMMANDS, BackupStore, Executor

BANNER_VERSION = "1.0.0"
MAX_RESULT_LINES = 40
MAX_RESULT_CHARS = 4000

_TOOL_ICON = {
    "bash": "⚙",
    "read_file": "📖",
    "write_file": "✏️ ",
    "edit_file": "✂️ ",
    "glob": "🔍",
    "grep": "🔎",
    "list_dir": "📁",
    "think": "💭",
}


class Spinner:
    """Tiny status line shown while waiting for the model."""

    FRAMES = "✳ ✳ ".split()

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._status = ""
        self._active = False
        self._thread: threading.Thread | None = None

    def start(self, status: str) -> None:
        if not self.enabled:
            return
        self._status = status
        self._active = True
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def update(self, status: str) -> None:
        self._status = status

    def pause(self) -> None:
        self._active = False
        self._clear()

    def resume(self) -> None:
        self._active = True

    def stop(self) -> None:
        self._active = False
        self._clear()
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=0.3)
            self._thread = None
        self._stop.clear()

    def _loop(self) -> None:
        i = 0
        while not self._stop.is_set():
            if self._active and ansi.ENABLED:
                with self._lock:
                    frame = self.FRAMES[i % len(self.FRAMES)]
                    i += 1
                    line = ansi.paint(frame, "accent") + " " + ansi.dim(self._status)
                    line = ansi.truncate(line, ansi.term_width() - 1)
                    sys.stdout.write("\r\x1b[2K" + line)
                    sys.stdout.flush()
            self._stop.wait(0.14)

    def _clear(self) -> None:
        if ansi.ENABLED:
            with self._lock:
                sys.stdout.write("\r\x1b[2K")
                sys.stdout.flush()


class REPL:
    def __init__(self, config: Config, cwd: str | None = None,
                 yolo: bool = False, quiet_banner: bool = False) -> None:
        self.config = config
        self.cwd = Path(cwd or os.getcwd()).resolve()
        if yolo:
            config.allow_bash = True
            config.allow_write = True
        self.backups = BackupStore()
        self.executor = Executor(str(self.cwd), config.bash_timeout, self.backups)
        self.session_id = new_session_id()
        self.spinner = Spinner()
        self._stream: markdown.StreamWriter | None = None
        self._at_line_start = True
        self._interrupted = False
        self._echo = True
        self.agent = Agent(config, self.executor, self._callbacks())
        self._quiet_banner = quiet_banner

    # ------------------------------------------------------------ callbacks

    def _callbacks(self) -> AgentCallbacks:
        return AgentCallbacks(
            on_text=self._on_text,
            on_think=self._on_think,
            on_flush=self._on_flush,
            on_tool_start=self._on_tool_start,
            on_tool_end=self._on_tool_end,
            on_tool_error=self._on_tool_error,
            on_retry=self._on_retry,
            on_model_switch=self._on_model_switch,
            confirm=self._confirm,
            interrupted=lambda: self._interrupted,
        )

    def _on_text(self, chunk: str) -> None:
        self.spinner.pause()
        if self._stream is None:
            self._stream = markdown.StreamWriter()
            self._newline()
        out = self._stream.feed(chunk)
        if out:
            self._write(out)

    def _on_think(self, chunk: str) -> None:
        if self._stream is not None:  # interleave safely with normal text
            self._flush_stream()
        self.spinner.pause()
        if not self._thinking:
            self._thinking = True
            self._newline()
            self._write("  " + ansi.dim("· thinking " + "·" * 46) + "\n")
        self._write(ansi.paint(chunk, "faint"))

    _thinking = False

    def _on_flush(self) -> None:
        self._flush_stream()
        if self._thinking:
            self._thinking = False
        self._newline()
        self._stream = None

    def _flush_stream(self) -> None:
        if self._stream is not None:
            out = self._stream.flush()
            if out:
                self._write(out)
            self._stream = None

    def _on_tool_start(self, name: str, args: dict) -> None:
        self._flush_stream()
        self.spinner.pause()
        icon = _TOOL_ICON.get(name, "🔧")
        self._newline()
        self._write(f"  {icon} {ansi.paint(name, 'tool')} {ansi.dim(self._arg_summary(name, args))}\n")

    def _on_tool_end(self, name: str, result: str, secs: float) -> None:
        preview = self._result_preview(name, result)
        if preview:
            for line in preview.split("\n"):
                self._write("    " + ansi.paint(line, "faint") + "\n")
        if secs > 2:
            self._write("    " + ansi.dim(f"({secs:.1f}s)") + "\n")

    def _on_tool_error(self, name: str, error: str) -> None:
        for line in error.split("\n")[:12]:
            self._write("    " + ansi.paint(line, "error") + "\n")

    def _on_retry(self, attempt: int, total: int, delay: float, reason: str) -> None:
        self.spinner.pause()
        self._newline()
        self._write("  " + ansi.dim(
            f"↻ {reason} — retrying in {delay:.0f}s "
            f"(attempt {attempt}/{total})") + "\n")
        self.spinner.resume()

    def _on_model_switch(self, old: str, new: str, reason: str) -> None:
        self.spinner.pause()
        self._newline()
        self._write("  " + ansi.paint(
            f"↻ {reason} — switching model: {old} → {new}", "warn") + "\n")
        self.spinner.resume()

    def _confirm(self, name: str, summary: str) -> str:
        self._flush_stream()
        self.spinner.pause()
        icon = _TOOL_ICON.get(name, "🔧")
        question = (
            f"\n  {icon} {ansi.paint('Allow ' + name + '?', 'warn')} "
            f"{ansi.dim(summary)}\n"
            f"  {ansi.dim('[y]es  [n]o  [a]lways for this session')} "
            f"{ansi.paint('›', 'user')} "
        )
        try:
            answer = input(question).strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "no"
        self._write("\n")
        return {"y": "yes", "yes": "yes", "": "no", "n": "no",
                "a": "always", "always": "always"}.get(answer, "no")

    # ----------------------------------------------------------------- io

    def _write(self, text: str) -> None:
        if not text:
            return
        sys.stdout.write(text)
        sys.stdout.flush()
        self._at_line_start = text.endswith("\n")

    def _newline(self) -> None:
        if not self._at_line_start:
            self._write("\n")

    # ------------------------------------------------------------- commands

    def run(self) -> int:
        self._print_banner()
        while True:
            try:
                user_input = self._read_input()
            except KeyboardInterrupt:
                self._write("\n" + ansi.dim("(input cancelled)") + "\n")
                continue
            except EOFError:
                self._write("\n")
                break
            if user_input is None:
                continue
            if not user_input.strip():
                continue
            if user_input.startswith("/"):
                if self._handle_command(user_input):
                    break
                continue
            self._ask(user_input)
            self._save()
        self._save()
        return 0

    def _ask(self, user_text: str) -> None:
        self._interrupted = False
        self._flush_stream()
        self._stream = None
        self._thinking = False
        self._newline()
        self._write(f"{ansi.paint('›', 'user')} {user_text}\n\n")
        self.spinner.start(f"thinking with {self.config.model} …")
        try:
            report: TurnReport = self.agent.ask(user_text)
        except KeyboardInterrupt:
            self._interrupted = True
            self.spinner.stop()
            self._flush_stream()
            self._newline()
            self._write(ansi.paint("✗ interrupted", "warn") + "\n")
            return
        except LLMError as exc:
            self.spinner.stop()
            self._flush_stream()
            self._newline()
            self._write(ansi.paint(f"✗ {exc}", "error") + "\n")
            return
        finally:
            self.spinner.stop()
        self._flush_stream()
        self._newline()
        self._status_line(report)

    def _status_line(self, report: TurnReport) -> None:
        ctx = self.agent.tokens()
        budget = max(1, self.config.context_chars // 4)
        pct = min(99, int(ctx / budget * 100))
        parts = [
            f"{report.seconds:.1f}s",
            f"{report.tool_calls} tool call{'s' if report.tool_calls != 1 else ''}",
            f"~{self._fmt_tokens(ctx)} tokens ctx ({pct}%)",
        ]
        if report.interrupted:
            parts.append("interrupted")
        if report.switches:
            parts.append(f"model switch{'es' if report.switches != 1 else ''} ×{report.switches}")
        self._write("  " + ansi.dim(" · ".join(parts)) + "\n\n")

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        return f"{n / 1000:.1f}k" if n >= 1000 else str(n)

    # ---------------------------------------------------------------- input

    def _read_input(self) -> str | None:
        cwd = ansi.paint(self.cwd.name or str(self.cwd), "accent")
        badge = ansi.paint("plan ", "warn") if self.config.plan_mode else ""
        prompt = f"{badge}{cwd} {ansi.paint('❯', 'user')} "
        line = input(prompt)
        if line.strip() == "/paste":
            return self._read_paste()
        buf = line
        while self._needs_continuation(buf):
            more = input(ansi.dim("… "))
            if more.strip() == "" and buf.strip().endswith("\\"):
                buf = buf.rstrip()[:-1]
                break
            buf = buf + "\n" + more
        return buf

    @staticmethod
    def _needs_continuation(buf: str) -> bool:
        stripped = buf.rstrip()
        if stripped.endswith("\\"):
            return True
        opens = sum(stripped.count(c) for c in "([{")
        closes = sum(stripped.count(c) for c in ")]}")
        return opens > closes

    @staticmethod
    def _read_paste() -> str:
        print(ansi.dim("paste mode — finish with a lone '.'"))
        lines: list[str] = []
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == ".":
                break
            lines.append(line)
        return "\n".join(lines)

    # --------------------------------------------------------------- render

    @staticmethod
    def _arg_summary(name: str, args: dict) -> str:
        if name == "bash":
            return "$ " + str(args.get("command", ""))[:160]
        if name == "read_file":
            extra = ""
            if args.get("offset"):
                extra += f" from line {args['offset']}"
            return f"{args.get('path', '?')}{extra}"
        if name in ("write_file", "edit_file"):
            return str(args.get("path", "?"))
        if name == "grep":
            return f"/{args.get('pattern', '')}/ {args.get('path', '.')}".strip()
        if name == "glob":
            return str(args.get("pattern", ""))
        if name == "think":
            return ansi.ellipsize(str(args.get("thoughts", "")), 120)
        return ansi.ellipsize(json.dumps(args), 120)

    @staticmethod
    def _result_preview(name: str, result: str) -> str:
        lines = result.split("\n")
        clipped = len(lines) > MAX_RESULT_LINES
        text = "\n".join(lines[:MAX_RESULT_LINES])
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS]
            clipped = True
        if clipped:
            text += f"\n… ({len(lines) - MAX_RESULT_LINES} more lines)"
        return text

    # ------------------------------------------------------------- commands

    def _handle_command(self, line: str) -> bool:
        """Returns True to quit."""
        parts = line[1:].strip().split(None, 1)
        name = parts[0].lower() if parts else ""
        arg = parts[1].strip() if len(parts) > 1 else ""

        if name in ("exit", "quit", "q"):
            return True
        if name == "help":
            self._cmd_help()
        elif name == "login":
            self._cmd_login(arg)
        elif name == "logout":
            self._cmd_logout()
        elif name == "clear":
            self.agent.messages.clear()
            self._write(ansi.dim("context cleared (session file kept)\n"))
        elif name == "model":
            self._cmd_model(arg)
        elif name == "models":
            self._cmd_models()
        elif name == "config":
            self._cmd_config(arg)
        elif name == "compact":
            self.spinner.start("compacting context …")
            try:
                message = self.agent.compact(silent=False)
            finally:
                self.spinner.stop()
            self._newline()
            self._write(ansi.paint(message, "ok") + "\n")
            self._save()
        elif name == "undo":
            if self.backups.undo():
                self._write(ansi.paint("reverted last file change", "ok") + "\n")
            else:
                self._write(ansi.dim("nothing to undo") + "\n")
        elif name == "sessions":
            self._cmd_sessions()
        elif name == "resume":
            self._cmd_resume(arg)
        elif name == "cwd":
            self._write(str(self.cwd) + "\n")
        elif name == "paste":
            text = self._read_paste()
            if text.strip():
                self._ask(text)
                self._save()
        elif name == "think":
            self.config.show_thoughts = not self.config.show_thoughts
            state = "on" if self.config.show_thoughts else "off"
            self._write(f"showing thinking blocks: {state}\n")
            self.config.save()
        elif name == "plan":
            self._cmd_plan(arg)
        elif name == "usage":
            self._write(
                f"prompt ~{self._fmt_tokens(self.agent.usage.prompt_tokens)} · "
                f"completion ~{self._fmt_tokens(self.agent.usage.completion_tokens)} · "
                f"context ~{self._fmt_tokens(self.agent.tokens())}\n"
            )
        else:
            self._write(ansi.paint(f"unknown command /{name}", "error")
                        + ansi.dim(" — try /help") + "\n")
        return False

    def _cmd_help(self) -> None:
        self._newline()
        for cmd, desc in BUILTIN_COMMANDS.items():
            self._write(f"  {ansi.paint('/' + cmd.ljust(10), 'accent')} {desc}\n")
        self._write("\n" + ansi.dim("  Multi-line input: end a line with \\ , or unbalanced brackets continue."))
        self._write("\n" + ansi.dim("  Ctrl-C cancels a reply · Ctrl-D quits.") + "\n")

    def _cmd_plan(self, arg: str) -> None:
        """Toggle plan mode: read-only research that ends with a proposed plan."""
        arg = arg.lower()
        if arg in ("on", "true", "1", "yes"):
            state = True
        elif arg in ("off", "false", "0", "no"):
            state = False
        else:
            state = not self.config.plan_mode
        self.config.plan_mode = state
        self.agent.plan_mode = state
        self.config.save()
        if state:
            self._write(ansi.paint("plan mode on", "ok") + ansi.dim(
                " — read-only: the agent explores the code and proposes a plan;\n"
                "    file writes/edits are blocked, shell commands need a per-command yes\n"))
        else:
            self._write(ansi.paint("plan mode off", "ok") + ansi.dim(
                " — the agent may edit files and run commands again\n"))
            self._offer_plan_approval()

    def _offer_plan_approval(self) -> None:
        """After leaving plan mode, offer to execute the plan on screen."""
        last_plan = next(
            (m.get("content", "") for m in reversed(self.agent.messages)
             if m.get("role") == "assistant"
             and isinstance(m.get("content"), str) and m["content"].strip()),
            "",
        )
        if not last_plan:
            return
        try:
            answer = input("  " + ansi.dim("proceed with the last plan? ")
                           + f"{ansi.paint('›', 'user')} {ansi.dim('[y/N] ')}").strip().lower()
        except (EOFError, KeyboardInterrupt):
            self._write("\n")
            return
        self._write("\n")
        if answer in ("y", "yes"):
            self._ask("Proceed with the plan you just outlined. Carry out the steps now.")
            self._save()

    def _cmd_login(self, arg: str) -> None:
        """Store an API key without echoing it, and without it being visible."""
        provider = arg or self._guess_provider()
        hints = {
            "openrouter": "openrouter.com/keys",
            "openai": "platform.openai.com/api-keys",
            "anthropic": "console.anthropic.com/settings/keys",
        }
        base_urls = {
            "openrouter": "https://openrouter.ai/api/v1",
            "openai": "https://api.openai.com/v1",
            "anthropic": "https://api.anthropic.com/v1/",
        }
        if provider not in hints:
            self._write(ansi.paint(f"unknown provider '{provider}'", "error")
                        + ansi.dim(" — try /login openrouter|openai|anthropic") + "\n")
            return

        self._write(f"  {ansi.dim('get a key at')} {ansi.paint(hints[provider], 'md_link')}\n")
        try:
            key = getpass.getpass(f"  {ansi.paint('API key', 'warn')} {ansi.dim('(hidden)')} › ")
        except (EOFError, KeyboardInterrupt):
            self._write("\n" + ansi.dim("cancelled") + "\n")
            return
        key = key.strip()
        if not key:
            self._write(ansi.dim("no key entered — nothing saved\n"))
            return

        self.config.provider = provider
        self.config.api_key = key
        # Point at the right endpoint unless the user set a custom one.
        current = self.config.base_url.lower()
        is_default_local = (not current or "localhost" in current or "127.0.0.1" in current)
        if is_default_local:
            self.config.base_url = base_urls[provider]
        if provider == "openrouter" and (not self.config.model or ":free" not in self.config.model):
            self.config.model = "cohere/north-mini-code:free"
        self.config.save()
        self._reconnect_client()
        self._write(ansi.paint(f"logged in to {provider} as {self._mask(key)}", "ok") + "\n")
        self._write(ansi.dim(f"  saved to {self.config_path()} (owner-only, never in the project)\n"))

    def _cmd_logout(self) -> None:
        if not self.config.api_key:
            self._write(ansi.dim("no key stored — nothing to remove\n"))
            return
        self.config.api_key = ""
        self.config.save()
        self._reconnect_client()
        self._write(ansi.paint("key removed", "ok") + ansi.dim(" (env vars, if any, still apply)\n"))

    def _reconnect_client(self) -> None:
        self.agent.client = self.agent.make_client()

    @staticmethod
    def _mask(key: str) -> str:
        return f"{key[:8]}…{key[-4:]}" if len(key) > 14 else "••••"

    @staticmethod
    def config_path() -> str:
        from .config import CONFIG_PATH
        return str(CONFIG_PATH)

    def _guess_provider(self) -> str:
        url = self.config.base_url.lower()
        if "openrouter" in url:
            return "openrouter"
        if "openai" in url:
            return "openai"
        if "anthropic" in url:
            return "anthropic"
        if url.startswith("http://localhost"):
            return "openrouter"
        return "openrouter"

    def _cmd_model(self, arg: str) -> None:
        if not arg:
            self._write(f"{self.config.model} @ {self.config.base_url}\n")
            return
        self.config.model = arg
        self.agent.client.model = arg
        self.config.save()
        self._write(ansi.paint(f"model → {arg}", "ok") + "\n")

    def _cmd_models(self) -> None:
        self.spinner.start(f"querying {self.config.base_url} …")
        try:
            models = self.agent.client.list_models()
        except LLMError as exc:
            self.spinner.stop()
            self._write(ansi.paint(str(exc), "error") + "\n")
            return
        self.spinner.stop()
        self._newline()
        if not models:
            self._write(ansi.dim("no models reported\n"))
            return
        for model in models[:60]:
            marker = " ●" if model == self.config.model else "  "
            self._write(f"{marker} {model}\n")
        if len(models) > 60:
            self._write(ansi.dim(f"  … {len(models) - 60} more\n"))

    def _cmd_config(self, arg: str) -> None:
        if not arg:
            data = {
                "provider": self.config.provider,
                "base_url": self.config.base_url,
                "api_key": (self.config.api_key[:6] + "…") if self.config.api_key else "(none)",
                "model": self.config.model,
                "temperature": self.config.temperature,
                "allow_bash": self.config.allow_bash,
                "allow_write": self.config.allow_write,
                "always_allow": self.config.always_allow,
                "show_thoughts": self.config.show_thoughts,
                "auto_compact": self.config.auto_compact,
                "plan_mode": self.config.plan_mode,
                "max_retries": self.config.max_retries,
                "fallback_models": ", ".join(self.config.fallback_models[:4])
                                   + (" …" if len(self.config.fallback_models) > 4 else ""),
                "cwd": str(self.cwd),
            }
            for key, value in data.items():
                self._write(f"  {key.ljust(14)} {ansi.paint(str(value), 'accent')}\n")
            return
        pieces = arg.split(None, 1)
        key = pieces[0]
        value = pieces[1] if len(pieces) > 1 else ""
        if not hasattr(self.config, key) or key in ("system_prompt",):
            self._write(ansi.paint(f"unknown or protected key: {key}", "error") + "\n")
            return
        current = getattr(self.config, key)
        if key == "api_key" and value:
            self._write(ansi.paint("refusing to echo a key in plain text.", "warn") + "\n")
            self._write(ansi.dim("  use /login — it hides your input and stores it owner-only\n"))
            return
        if isinstance(current, bool):
            parsed: object = value.lower() in ("1", "true", "yes", "on")
        elif isinstance(current, int):
            try:
                parsed = int(value)
            except ValueError:
                self._write(ansi.paint(f"{key} needs an integer", "error") + "\n")
                return
        elif isinstance(current, float):
            try:
                parsed = float(value)
            except ValueError:
                self._write(ansi.paint(f"{key} needs a number", "error") + "\n")
                return
        elif isinstance(current, list):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                parsed = [v.strip() for v in value.split(",") if v.strip()]
            if not isinstance(parsed, list):
                self._write(ansi.paint(f"{key} needs a list, e.g. [\"bash\"]", "error") + "\n")
                return
        else:
            parsed = value
        setattr(self.config, key, parsed)
        if key in ("base_url", "api_key", "model", "temperature", "max_tokens",
                   "max_retries"):
            self.agent.client = self.agent.make_client()
        self.config.save()
        self._write(ansi.paint(f"{key} = {parsed}", "ok") + "\n")

    def _cmd_sessions(self) -> None:
        sessions = list_sessions()
        if not sessions:
            self._write(ansi.dim("no saved sessions\n"))
            return
        for session in sessions[:20]:
            when = time.strftime("%m-%d %H:%M", time.localtime(session["when"]))
            self._write(
                f"  {ansi.paint(session['id'], 'accent')}  {ansi.dim(when)}  "
                f"{session['turns']} turns  {session['model']}\n"
                f"      {ansi.dim(session['preview'])}\n"
            )

    def _cmd_resume(self, arg: str) -> None:
        if not arg:
            self._write(ansi.dim("usage: /resume <session-id>  (see /sessions)\n"))
            return
        data = load_session(arg)
        if data is None:
            self._write(ansi.paint(f"no session {arg}", "error") + "\n")
            return
        self.session_id = data["id"]
        self.agent.messages = data.get("messages", [])
        meta = data.get("meta", {})
        if meta.get("model"):
            self.config.model = meta["model"]
            self.agent.client.model = meta["model"]
        self._write(ansi.paint(f"resumed {self.session_id} "
                               f"({len(self.agent.messages)} messages)", "ok") + "\n")

    # ------------------------------------------------------------- sessions

    def _save(self) -> None:
        try:
            save_session(self.session_id, self.agent.messages, {
                "model": self.config.model,
                "cwd": str(self.cwd),
                "saved_at": time.time(),
            })
        except OSError:
            pass

    # ---------------------------------------------------------------- banner

    def _print_banner(self) -> None:
        if self._quiet_banner:
            return
        art = ansi.paint(" freecode ", "md_heading") + ansi.dim(f"v{BANNER_VERSION}")
        self._write(f"\n  {art}  {ansi.dim('free · local · open coding agent')}\n")
        provider = ansi.paint(self.config.model, "accent") + ansi.dim(f" @ {self.config.base_url}")
        self._write(f"  {provider}\n")
        self._write(f"  {ansi.dim('cwd')} {self.cwd}\n")

        # No provider detected: local Ollama endpoint, no API key. Most likely
        # a first run on a new machine — tell the user how to hook one up.
        if self.config.base_url.startswith("http://localhost") and not self.config.api_key:
            self._write("\n  " + ansi.paint("No model backend configured yet.", "warn") + "\n")
            self._write("  " + ansi.dim("Pick one (each person supplies their own key):") + "\n\n")
            self._write("    " + ansi.paint("free + fully local", "accent") + ansi.dim("  — no key needed") + "\n")
            self._write("      " + ansi.dim("install ollama.com, then:  ollama pull qwen3-coder:8b") + "\n\n")
            self._write("    " + ansi.paint("free via OpenRouter", "accent") + ansi.dim("  — needs an account") + "\n")
            self._write("      " + ansi.dim("export OPENROUTER_API_KEY=sk-or-…   (get one at openrouter.com/keys)") + "\n\n")
            self._write("  " + ansi.dim("Your key stays on your machine — run /login to save one, or just") + "\n")
            self._write("  " + ansi.dim("export OPENROUTER_API_KEY=… for the session. It never enters the project.") + "\n")
        self._write("\n")
