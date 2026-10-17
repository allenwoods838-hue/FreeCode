"""Configuration + session persistence (~/.freecode)."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

HOME = Path(os.environ.get("FREECODE_HOME", Path.home() / ".freecode"))
CONFIG_PATH = HOME / "config.json"
SESSIONS_DIR = HOME / "sessions"

DEFAULT_SYSTEM_PROMPT = """\
You are FreeCode, an expert terminal coding agent running inside the user's \
project. You help by reading, writing and editing files and running commands.

Rules:
- Use the provided tools to act; do not ask the user to paste code you can read.
- Prefer surgical edits (edit_file) over rewriting whole files.
- Keep answers short. Explain what you did, not what you are about to think.
- Never invent file contents. Read files before editing them.
- When a task is done, say so in one or two sentences.
"""

# Preference order for automatic fallback when the current model hits an
# OpenRouter limit. Current free tier, coding-first (checked 2026-10).
# Candidates are cross-checked against the endpoint's /models list, so
# stale IDs are skipped harmlessly.
DEFAULT_FALLBACK_MODELS = [
    "poolside/laguna-s-2.1:free",        # strongest free coding model
    "cohere/north-mini-code:free",       # fast agentic coding
    "thinkingmachines/inkling:free",     # reasoning, 1M context
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "qwen/qwen3.8-27b:free",
    "poolside/laguna-xs-2.1:free",
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-3-super-120b-a12b:free",
    "nvidia/nemotron-3.5-lightning:free",
    "thinkingmachines/inkling-small:free",
    "dots-studio/dots-3-note-preview:free",
    "openrouter/free",                   # router: any free model — last resort
]

# Free endpoints that are useless for an agent (moderation filters, tiny
# or domain-specific models) — never fall back to these.
FALLBACK_SKIP = {
    "nvidia/nemotron-3.5-content-safety:free",
    "liquid/lfm-2.5-2.6b:free",
    "inclusionai/ling-3.0-flash-sante:free",
}


@dataclass
class Config:
    # provider -------------------------------------------------------------
    provider: str = "auto"          # auto | ollama | openai | openrouter | custom
    base_url: str = ""
    api_key: str = ""
    model: str = ""
    temperature: float = 0.2
    max_tokens: int = 0             # 0 = provider default
    request_timeout: int = 600
    # agent ----------------------------------------------------------------
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    max_turns: int = 40             # tool-loop iterations per user message
    context_chars: int = 260_000    # soft budget; older tool output gets trimmed
    auto_compact: bool = True
    plan_mode: bool = False         # read-only research + plan; /plan toggles
    max_retries: int = 2            # per-request retries on 429/503/529
    fallback_models: list[str] = field(  # tried in order when rate limited
        default_factory=lambda: list(DEFAULT_FALLBACK_MODELS))
    # tools ----------------------------------------------------------------
    allow_bash: bool = False        # ask before running shell commands
    allow_write: bool = False       # ask before write_file / edit_file
    always_allow: list[str] = field(default_factory=list)  # tool names approved
    bash_timeout: int = 120
    # ui -------------------------------------------------------------------
    theme: str = "midnight"
    show_thoughts: bool = True      # reveal the model's <think> blocks dimmed

    # ------------------------------------------------------------------ io

    def load(self) -> "Config":
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text())
                for key, value in data.items():
                    if hasattr(self, key) and key != "always_allow":
                        setattr(self, key, value)
                    elif key == "always_allow" and isinstance(value, list):
                        self.always_allow = [str(v) for v in value]
            except (json.JSONDecodeError, OSError):
                pass
        self._apply_env()
        self._apply_provider_defaults()
        return self

    def save(self) -> None:
        HOME.mkdir(parents=True, exist_ok=True)
        data = asdict(self)
        CONFIG_PATH.write_text(json.dumps(data, indent=2) + "\n")
        # The file may contain an api_key — keep it owner-only.
        try:
            os.chmod(CONFIG_PATH, 0o600)
        except OSError:
            pass

    # ------------------------------------------------------------- helpers

    def _apply_env(self) -> None:
        env = os.environ
        self.base_url = env.get("FREECODE_BASE_URL", self.base_url)
        self.api_key = env.get("FREECODE_API_KEY", self.api_key)
        self.model = env.get("FREECODE_MODEL", self.model)
        if env.get("FREECODE_ALLOW_BASH"):
            self.allow_bash = env["FREECODE_ALLOW_BASH"] == "1"
        if env.get("FREECODE_ALLOW_WRITE"):
            self.allow_write = env["FREECODE_ALLOW_WRITE"] == "1"
        if env.get("FREECODE_PLAN_MODE"):
            self.plan_mode = env["FREECODE_PLAN_MODE"] == "1"
        if env.get("FREECODE_TEMPERATURE"):
            try:
                self.temperature = float(env["FREECODE_TEMPERATURE"])
            except ValueError:
                pass

    def _apply_provider_defaults(self) -> None:
        if not self.api_key:
            if os.environ.get("OPENROUTER_API_KEY"):
                self.api_key = os.environ["OPENROUTER_API_KEY"]
                if not self.provider or self.provider == "auto":
                    self.provider = "openrouter"
            elif os.environ.get("OPENAI_API_KEY"):
                self.api_key = os.environ["OPENAI_API_KEY"]
                if not self.provider or self.provider == "auto":
                    self.provider = "openai"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                # Anthropic exposes an OpenAI-compatible endpoint too.
                self.api_key = os.environ["ANTHROPIC_API_KEY"]
                if not self.provider or self.provider == "auto":
                    self.provider = "anthropic"

        provider = self.provider if self.provider != "auto" else "ollama"
        defaults = {
            "ollama": ("http://localhost:11434/v1", "", "qwen3-coder:8b"),
            "openai": ("https://api.openai.com/v1", "", "gpt-4o-mini"),
            "openrouter": ("https://openrouter.ai/api/v1", "", "cohere/north-mini-code:free"),
            "anthropic": ("https://api.anthropic.com/v1/", "", "claude-3-5-haiku-latest"),
        }
        if self.provider == "custom" or provider not in defaults:
            self.base_url = self.base_url or "http://localhost:11434/v1"
            self.model = self.model or "qwen3-coder:8b"
            return
        base, key, model = defaults[provider]
        self.base_url = self.base_url or base
        self.api_key = self.api_key or key
        self.model = self.model or model

    @property
    def endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/chat/completions"

    @property
    def models_endpoint(self) -> str:
        return self.base_url.rstrip("/") + "/models"

    def allow(self, tool: str) -> bool:
        return tool in self.always_allow

    def grant(self, tool: str) -> None:
        if tool not in self.always_allow:
            self.always_allow.append(tool)


# --------------------------------------------------------------- sessions ---


def new_session_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def save_session(session_id: str, messages: list[dict], meta: dict) -> None:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"id": session_id, "meta": meta, "messages": messages}
    session_path(session_id).write_text(json.dumps(payload, indent=1))


def load_session(session_id: str) -> dict | None:
    path = session_path(session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def list_sessions() -> list[dict]:
    if not SESSIONS_DIR.exists():
        return []
    out: list[dict] = []
    for path in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        first = next(
            (m["content"] for m in data.get("messages", [])
             if m.get("role") == "user" and isinstance(m.get("content"), str)),
            "(empty)",
        )
        out.append(
            {
                "id": data.get("id", path.stem),
                "model": data.get("meta", {}).get("model", "?"),
                "when": path.stat().st_mtime,
                "preview": first.replace("\n", " ")[:70],
                "turns": sum(1 for m in data.get("messages", []) if m.get("role") == "user"),
            }
        )
    return out
