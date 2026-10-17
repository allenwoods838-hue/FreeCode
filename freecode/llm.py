"""Minimal OpenAI-compatible chat client (stdlib only, SSE streaming).

Works with Ollama, LM Studio, vLLM, OpenRouter, OpenAI, and any other
server that speaks the /chat/completions protocol.
"""

from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Iterator

# transient statuses worth retrying with backoff
RETRYABLE_STATUS = {429, 503, 529}
# statuses that mean "this model is out of budget" — retrying won't help,
# but switching to another (free) model may
LIMIT_STATUS = {402, 429}


class LLMError(RuntimeError):
    pass


class LLMInterrupted(LLMError):
    """Raised when the user hits Ctrl-C mid stream."""


class LLMRateLimited(LLMError):
    """429 (rate limited) or 402 (out of credits) after retries.

    The caller may switch to another model and replay the request.
    """

    def __init__(self, message: str, status: int = 429) -> None:
        super().__init__(message)
        self.status = status


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens


@dataclass
class StreamEvent:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # complete calls
    usage: Usage | None = None


class Client:
    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        temperature: float = 0.2,
        max_tokens: int = 0,
        timeout: int = 600,
        max_retries: int = 2,
        on_retry: Callable[[int, int, float, str], None] | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self.on_retry = on_retry
        self.extra_headers = extra_headers or {}

    # ------------------------------------------------------------- request

    def stream(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
    ) -> Iterator[StreamEvent]:
        """Stream a completion; yields text deltas and completed tool calls."""
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": self.temperature,
        }
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        body = json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=body,
            headers=self._headers(),
            method="POST",
        )
        try:
            response = self._open(request)
        except LLMRateLimited:
            raise
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:2000]
            if tools and exc.code == 400 and "tool" in detail.lower():
                # provider may not support tool calling; degrade gracefully
                yield StreamEvent(text="")
                yield from self.stream(messages, tools=None)
                return
            raise LLMError(self._pretty_http_error(exc.code, detail)) from None
        except urllib.error.URLError as exc:
            raise LLMError(self._pretty_url_error(exc)) from None
        except (TimeoutError, socket.timeout):
            raise LLMError(f"Request to {self.base_url} timed out after {self.timeout}s") from None

        try:
            yield from self._read_sse(response)
        except (TimeoutError, socket.timeout):
            raise LLMError(f"Stream from {self.base_url} timed out") from None
        except BrokenPipeError as exc:
            raise LLMInterrupted() from exc
        finally:
            try:
                response.close()
            except Exception:
                pass

    # ------------------------------------------------------------- helpers

    def _open(self, request: urllib.request.Request):
        """urlopen with retry + backoff on transient errors (429/503/529).

        Honours the provider's Retry-After header, capped at 60s. When a 429
        survives the retries (or the provider says 402 — out of credits),
        raises LLMRateLimited so the caller can switch to another model.
        """
        attempt = 0
        while True:
            attempt += 1
            try:
                return urllib.request.urlopen(request, timeout=self.timeout)
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_STATUS and attempt <= self.max_retries:
                    delay = self._backoff(exc, attempt)
                    if self.on_retry:
                        reason = "rate limited" if exc.code == 429 else "server busy"
                        try:
                            self.on_retry(attempt, self.max_retries, delay, reason)
                        except Exception:  # noqa: BLE001 - UI hook must never kill the request
                            pass
                    time.sleep(delay)
                    continue
                if exc.code in LIMIT_STATUS:
                    detail = exc.read().decode("utf-8", "replace")[:2000]
                    raise LLMRateLimited(
                        self._pretty_http_error(exc.code, detail), status=exc.code
                    ) from None
                raise

    @staticmethod
    def _backoff(exc: urllib.error.HTTPError, attempt: int) -> float:
        """Exponential backoff, stretched by the Retry-After header if present."""
        delay = min(2.0 * (2 ** (attempt - 1)), 60.0)
        header = ""
        try:
            header = (exc.headers.get("Retry-After", "") or "") if exc.headers else ""
        except AttributeError:
            pass
        if header:
            try:  # only the seconds form; HTTP-dates are ignored
                delay = max(delay, min(float(header), 60.0))
            except ValueError:
                pass
        return delay

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "freecode/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _read_sse(self, response) -> Iterator[StreamEvent]:
        pending: dict[int, dict] = {}
        usage = Usage()
        buffer = b""
        done = False
        while not done:
            chunk = response.read(8192)
            if not chunk:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                line = raw_line.decode("utf-8", "replace").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                event = self._consume(obj, pending, usage)
                if event is not None:
                    yield event
        # flush leftovers: uncompleted tool calls + a trailing usage chunk
        if pending:
            yield StreamEvent(tool_calls=[pending[i] for i in sorted(pending)], usage=usage)
        if usage.prompt_tokens or usage.completion_tokens or usage.total_tokens:
            yield StreamEvent(usage=usage)

    def _consume(self, obj: dict, pending: dict[int, dict], usage: Usage) -> StreamEvent | None:
        if obj.get("usage"):
            u = obj["usage"]
            usage.prompt_tokens = u.get("prompt_tokens", 0)
            usage.completion_tokens = u.get("completion_tokens", 0)
            usage.total_tokens = u.get("total_tokens", 0)
        choices = obj.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        event = StreamEvent()
        text = delta.get("content")
        if isinstance(text, str) and text:
            event.text = text
        finished: list[dict] = []
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            slot = pending.setdefault(
                idx,
                {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
            )
            if tc.get("id"):
                slot["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                slot["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                slot["function"]["arguments"] += fn["arguments"]
        if choices[0].get("finish_reason") == "tool_calls" and pending:
            finished = [pending[i] for i in sorted(pending)]
            pending.clear()
        if finished:
            event.tool_calls = finished
        if event.text or finished:
            return event
        return None

    @staticmethod
    def _pretty_http_error(code: int, detail: str) -> str:
        hint = ""
        try:
            parsed = json.loads(detail)
            detail = parsed.get("error", {}).get("message", detail)
        except (json.JSONDecodeError, AttributeError, TypeError):
            pass
        if code == 401:
            hint = " (bad or missing API key — set FREECODE_API_KEY)"
        elif code == 402:
            hint = " (out of credits — freecode will try a free model)"
        elif code == 404:
            hint = " (model not found? try /models, then /model <name>)"
        elif code == 429:
            hint = " (rate limited)"
        return f"HTTP {code}: {detail[:600]}{hint}"

    @staticmethod
    def _pretty_url_error(exc: urllib.error.URLError) -> str:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (ConnectionRefusedError, OSError)):
            return (
                "Could not reach the LLM server. Is it running?\n"
                "  • local model:  `ollama serve`  (http://localhost:11434)\n"
                "  • or set a hosted endpoint:  export OPENAI_API_KEY=…  /  "
                "export OPENROUTER_API_KEY=…\n"
                f"  detail: {reason}"
            )
        return f"Connection error: {reason}"

    # -------------------------------------------------------- list models

    def list_models(self) -> list[str]:
        request = urllib.request.Request(
            self.base_url + "/models", headers=self._headers(), method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=15) as response:
                data = json.loads(response.read().decode("utf-8", "replace"))
        except Exception as exc:
            raise LLMError(f"Could not list models from {self.base_url}: {exc}") from None
        ids = [m.get("id") for m in data.get("data", []) if m.get("id")]
        return sorted(ids)


def estimate_tokens(messages: list[dict]) -> int:
    """Rough token estimate (~4 chars/token) — used for context budgeting."""
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            total += sum(len(str(part)) for part in content)
        for tc in message.get("tool_calls") or []:
            total += len(json.dumps(tc))
    return max(1, total // 4)

