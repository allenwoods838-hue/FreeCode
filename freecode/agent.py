"""The agent loop: message history, streaming, tool dispatch, context budget."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

from .config import FALLBACK_SKIP, Config
from .llm import Client, LLMError, LLMInterrupted, LLMRateLimited, Usage, estimate_tokens
from .tools import RISKY_TOOLS, SAFE_TOOLS, TOOLS, Executor, ToolError

THINK_OPEN = "<" + "think" + ">"
THINK_CLOSE = "<" + "/" + "think" + ">"


def _model_rank(model: str) -> tuple:
    """Score discovered free models so coding-capable ones fall back first."""
    name = model.lower()
    score = 0
    if "code" in name:
        score -= 20
    if any(k in name for k in ("deepseek", "qwen", "kimi", "glm",
                               "laguna", "inkling", "nemotron")):
        score -= 5
    if any(k in name for k in ("safety", "guard", "lfm", "ling-")):
        score += 50
    return (score, name)

PLAN_MODE_NOTICE = """\

PLAN MODE is active: you are in a read-only research phase.
- Explore with read_file, glob, grep and list_dir. bash only if the user approves it.
- write_file and edit_file are blocked. Do not try to modify anything.
- Do not implement yet. Finish with a concrete, numbered plan covering:
  the goal, the exact files and functions to change, the steps in order,
  how to verify (commands/tests), and any risks or open questions.
- The user will review the plan and ask you to proceed.
"""


@dataclass
class AgentCallbacks:
    """Hooks the UI provides so the agent never prints directly."""

    on_text: Callable[[str], None] = lambda chunk: None       # streamed deltas
    on_think: Callable[[str], None] = lambda chunk: None      # thinking deltas
    on_flush: Callable[[], None] = lambda: None               # end of message
    on_tool_start: Callable[[str, dict], None] = lambda name, args: None
    on_tool_end: Callable[[str, str, float], None] = lambda name, result, secs: None
    on_tool_error: Callable[[str, str], None] = lambda name, error: None
    on_assistant_message: Callable[[dict], None] = lambda message: None
    on_retry: Callable[[int, int, float, str], None] = \
        lambda attempt, total, delay, reason: None
    on_model_switch: Callable[[str, str, str], None] = \
        lambda old, new, reason: None
    confirm: Callable[[str, str], str] = lambda name, summary: "yes"  # yes|no|always
    interrupted: Callable[[], bool] = lambda: False


@dataclass
class TurnReport:
    interrupted: bool = False
    tool_calls: int = 0
    switches: int = 0
    seconds: float = 0.0
    usage: Usage = field(default_factory=Usage)


class Agent:
    def __init__(self, config: Config, executor: Executor,
                 callbacks: AgentCallbacks | None = None) -> None:
        self.config = config
        self.executor = executor
        self.cb = callbacks or AgentCallbacks()
        self.client = self.make_client()
        self.messages: list[dict] = []
        self.plan_mode = bool(config.plan_mode)
        self._tried_models = {config.model}
        self.usage = Usage()
        self._thought_buf: list[str] = []
        self._in_thought = False

    def make_client(self) -> Client:
        """Build the HTTP client, wiring the retry hook into the UI."""
        client = Client(
            base_url=self.config.base_url,
            api_key=self.config.api_key,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
        )
        client.on_retry = lambda attempt, total, delay, reason: \
            self.cb.on_retry(attempt, total, delay, reason)
        return client

    # ---------------------------------------------------------------- loop

    def ask(self, user_text: str) -> TurnReport:
        """Run one user turn, looping through tool calls until done."""
        report = TurnReport()
        started = time.time()
        self.messages.append({"role": "user", "content": user_text})
        self._maybe_auto_compact()

        dupes: dict[str, int] = {}
        nudged = False
        for _ in range(max(1, self.config.max_turns)):
            if self.cb.interrupted():
                report.interrupted = True
                break
            try:
                message, turn_usage = self._stream_once(report)
            except LLMRateLimited as exc:
                if self._fallback_model(exc):
                    report.switches += 1
                    continue  # replay this iteration on the next free model
                raise  # no fallback left — surface the limit to the user
            self.usage.add(turn_usage)
            report.usage = turn_usage
            self.cb.on_flush()
            self.messages.append(message)

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                break
            report.tool_calls += len(tool_calls)

            key = json.dumps([[c.get("function", {}).get("name"),
                               c.get("function", {}).get("arguments")]
                              for c in tool_calls])
            dupes[key] = dupes.get(key, 0) + 1

            for call in tool_calls:
                result = self._dispatch(call)
                self.messages.append({
                    "role": "tool",
                    "tool_call_id": result["id"],
                    "content": result["content"],
                })
            if dupes[key] >= 3:
                if nudged:
                    break  # model is stuck in a loop; stop here
                nudged = True
                self.messages.append({
                    "role": "user",
                    "content": "You repeated the exact same tool call three times. "
                               "Stop calling tools and answer in text, describing "
                               "what happened and what you recommend.",
                })
            if self.cb.interrupted():
                report.interrupted = True
                break
        else:
            # out of turns: keep the transcript valid for the next request
            self.messages.append({
                "role": "user",
                "content": "(stop: tool-call limit reached for this turn. "
                           "Summarise progress in two sentences.)",
            })

        report.seconds = time.time() - started
        self._trim_context()
        return report

    # ------------------------------------------------------------ streaming

    def _stream_once(self, report: TurnReport) -> tuple[dict, Usage]:
        """Single completion; returns (assistant message, usage delta)."""
        system = self.config.system_prompt
        if self.plan_mode:
            system += PLAN_MODE_NOTICE
        payload = [{"role": "system", "content": system}]
        payload += self.messages

        text_parts: list[str] = []
        tool_acc: dict[int, dict] = {}
        usage = Usage()
        self._thought_buf = []
        self._in_thought = False

        try:
            for event in self.client.stream(payload, tools=TOOLS):
                usage = event.usage or usage
                if event.text:
                    self._handle_delta(event.text, text_parts)
                for call in event.tool_calls:
                    tool_acc[len(tool_acc) + 1] = call
        except LLMInterrupted:
            report.interrupted = True
            self._in_thought = False
            content = "".join(text_parts).strip()
            return {"role": "assistant", "content": content or "(interrupted)"}, usage

        self._in_thought = False
        message: dict = {"role": "assistant", "content": "".join(text_parts)}
        if tool_acc:
            message["tool_calls"] = [
                {
                    "id": call.get("id") or f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"] or "{}",
                    },
                }
                for i, call in enumerate(tool_acc.values(), start=1)
            ]
        self.cb.on_assistant_message(message)
        return message, usage

    def _handle_delta(self, text: str, parts: list[str]) -> None:
        """Route deltas: thinking blocks are shown dimmed, the rest streams."""
        consumed = text
        while consumed:
            if self._in_thought:
                end = consumed.find(THINK_CLOSE)
                if end == -1:
                    self._thought_buf.append(consumed)
                    if self.config.show_thoughts:
                        self.cb.on_think(consumed)
                    return
                self._thought_buf.append(consumed[:end])
                if self.config.show_thoughts:
                    self.cb.on_think(consumed[:end])
                self._in_thought = False
                consumed = consumed[end + len(THINK_CLOSE):]
                continue
            start = consumed.find(THINK_OPEN)
            if start == -1:
                parts.append(consumed)
                self.cb.on_text(consumed)
                return
            parts.append(consumed[:start])
            if self.config.show_thoughts:
                self.cb.on_text(consumed[:start])
            consumed = consumed[start + len(THINK_OPEN):]
            self._in_thought = True

    # ---------------------------------------------------------------- tools

    def _dispatch(self, call: dict) -> dict:
        name = call.get("function", {}).get("name", "")
        raw_args = call.get("function", {}).get("arguments") or "{}"
        call_id = call.get("id") or "call_0"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            if not isinstance(args, dict):
                raise ValueError("arguments must be an object")
        except (json.JSONDecodeError, ValueError) as exc:
            self.cb.on_tool_error(name, f"invalid JSON arguments: {exc}")
            return {"id": call_id, "content": f"Error: invalid JSON arguments ({exc})"}

        self.cb.on_tool_start(name, args)

        if self.plan_mode and name in ("write_file", "edit_file"):
            self.cb.on_tool_error(name, "blocked: plan mode is active")
            return {"id": call_id,
                    "content": "Plan mode is active: file writes and edits are "
                               "blocked. Research with read-only tools and "
                               "present a plan for the user to approve."}

        if name in RISKY_TOOLS and not self._approved(name, args):
            return {"id": call_id,
                    "content": "User declined this action. Ask them what they prefer."}

        started = time.time()
        try:
            result = self.executor.run(name, args)
        except ToolError as exc:
            self.cb.on_tool_error(name, str(exc))
            return {"id": call_id, "content": f"Error: {exc}"}
        except Exception as exc:  # noqa: BLE001 - never crash the loop
            self.cb.on_tool_error(name, f"{type(exc).__name__}: {exc}")
            return {"id": call_id, "content": f"Error: {type(exc).__name__}: {exc}"}

        self.cb.on_tool_end(name, result, time.time() - started)
        return {"id": call_id, "content": result or "(no output)"}

    def _approved(self, name: str, args: dict) -> bool:
        if self.plan_mode:
            # Writes/edit are already pre-blocked in _dispatch; bash always
            # needs a fresh yes here, even under -y / always_allow, so plan
            # mode stays strictly read-only unless the user says otherwise.
            return self.cb.confirm(name, self._summary(name, args)) == "yes"
        if name in SAFE_TOOLS or self.config.allow(name):
            return True
        if name == "bash" and self.config.allow_bash:
            return True
        if name in ("write_file", "edit_file") and self.config.allow_write:
            return True
        answer = self.cb.confirm(name, self._summary(name, args))
        if answer == "always":
            self.config.grant(name)
            return True
        return answer == "yes"

    @staticmethod
    def _summary(name: str, args: dict) -> str:
        if name == "bash":
            return str(args.get("command", ""))[:300]
        if name in ("write_file", "edit_file"):
            return str(args.get("path", ""))
        return json.dumps(args)[:300]

    # --------------------------------------------------------- model fallback

    def _fallback_model(self, exc: LLMRateLimited) -> bool:
        """Switch to the next best free model after a rate/credit limit."""
        self._tried_models.add(self.client.model)
        nxt = self._next_model()
        if not nxt:
            return False
        old = self.client.model
        self._tried_models.add(nxt)
        self.client.model = nxt
        self.config.model = nxt  # session-only; not written back to disk
        self.cb.on_model_switch(
            old, nxt, "out of credits" if exc.status == 402 else "rate limited")
        return True

    def _next_model(self) -> str | None:
        """Best untried fallback model the endpoint actually serves."""
        try:
            served = self.client.list_models()
        except Exception:  # noqa: BLE001 - a dead /models must not kill the turn
            return None
        available = [m for m in served
                     if m not in self._tried_models and m not in FALLBACK_SKIP]
        if not available:
            return None
        for wanted in self.config.fallback_models:
            if wanted in available:
                return wanted
        # auto-discovery: only :free models, so we never spend real credits
        free = sorted((m for m in available if m.endswith(":free")),
                      key=_model_rank)
        return free[0] if free else None

    # -------------------------------------------------------------- context

    def tokens(self) -> int:
        return estimate_tokens(self.messages)

    def _maybe_auto_compact(self) -> None:
        if not self.config.auto_compact:
            return
        if self.tokens() * 4 < self.config.context_chars:
            return
        self.compact(silent=True)

    def _trim_context(self) -> None:
        """Shrink old tool payloads when the transcript grows too large."""
        if self.tokens() * 4 <= self.config.context_chars:
            return
        for message in self.messages[1:-8]:
            if message.get("role") == "tool" and isinstance(message.get("content"), str):
                if len(message["content"]) > 800:
                    message["content"] = message["content"][:400] + "\n… (older output trimmed)"

    def compact(self, silent: bool = False) -> str:
        """Summarise the transcript into one checkpoint message."""
        if len(self.messages) < 4:
            return "Nothing to compact yet."
        transcript: list[str] = []
        for message in self.messages:
            role = message.get("role")
            content = message.get("content")
            if role in ("user", "assistant") and isinstance(content, str) and content.strip():
                transcript.append(f"{role.upper()}: {content[:1500]}")
            if role == "assistant":
                for call in message.get("tool_calls") or []:
                    fn = call.get("function", {})
                    transcript.append(f"TOOL: {fn.get('name')} {str(fn.get('arguments'))[:200]}")
        text = "\n".join(transcript)
        if len(text) > 60_000:
            text = text[:30_000] + "\n…\n" + text[-25_000:]

        prompt = (
            "Summarise this coding session so work can continue seamlessly. "
            "Include: the goal, files touched and their current state, decisions "
            "made, errors hit, and the exact next steps.\n\n" + text
        )
        usage = Usage()
        chunks: list[str] = []
        try:
            for event in self.client.stream([{"role": "user", "content": prompt}], tools=None):
                usage = event.usage or usage
                chunks.append(event.text)
                if event.text and not silent:
                    self.cb.on_text(event.text)
        except LLMError as exc:
            return f"Compact failed: {exc}"
        summary = "".join(chunks).strip() or "(empty summary)"
        self.usage.add(usage)
        first_user = next(
            (m["content"] for m in self.messages if m.get("role") == "user"), ""
        )
        self.messages = [
            {"role": "user", "content": f"[session start]\n{first_user}"},
            {"role": "assistant",
             "content": f"[context compacted]\n\nSummary of work so far:\n{summary}"},
        ]
        return f"Compacted → ~{estimate_tokens(self.messages)} tokens of context."
