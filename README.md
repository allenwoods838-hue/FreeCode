# FreeCode

**A free, local, open-source coding agent for your terminal** — a Claude Code
alternative with **zero dependencies** and your choice of free models.

```
myproject ❯ fix the failing test in tests/test_math.py

  📖 read_file  tests/test_math.py
    1→def test_add():
    2→    assert add(1, 2) == 4
  ✂️ edit_file  tests/test_math.py
    1→def test_add():
    2→    assert add(1, 2) == 3
  ⚙ bash  $ python3 -m pytest tests/ -q
    1 passed

Fixed the expected value in `tests/test_math.py` — `add(1, 2)` is 3.
```

- **Free** — runs against any OpenAI-compatible server: [Ollama](https://ollama.com)
  (fully local), LM Studio, vLLM, OpenRouter free models, or a hosted API.
- **Zero dependencies** — pure Python 3.10+ standard library. No `pip install`
  of anything at runtime, no telemetry, no account.
- **Coloured, live output** — streaming Markdown rendering with syntax
  highlighting, tool cards, a thinking spinner and context-usage stats.
- **Agentic** — reads/writes/edits files, runs shell commands, globs and greps,
  with permission prompts (`y` / `n` / `always`) and one-key `/undo`.
- **Plan mode** — `/plan` locks the agent into read-only research: it explores
  your code and proposes a numbered plan for approval before touching anything.
- **Resilient** — rate limits don't stop it: requests retry with backoff
  (honouring `Retry-After`), and a limited model automatically falls back to
  the next best free model on your endpoint.

## Install

**Option 1 — global command (works from any folder):**

```bash
./install.sh
freecode            # now available anywhere, not just inside the project
```

This drops a tiny launcher in `~/.local/bin/freecode` that runs your project
copy via `python -m freecode`, so it always uses your latest code with no
re-install. `~/.local/bin` is already on your `PATH`.

**Option 2 — from PyPI:**

```bash
pip install freecode-ai    # installs the `freecode` command
```

**Option 3 — no install at all:**

```bash
python3 -m freecode    # must run from inside the project directory
```

## Quick start

> **Everyone uses their own key.**
> Nothing in this project carries a credential. On a new machine you either run
> a local model (no key at all), or export your *own* `OPENROUTER_API_KEY`.
> freecode never writes your key into the project or into any shared file — if
> you ever set one with `/config api_key …`, it is stored owner-only (`600`)
> in `~/.freecode/config.json`, which is your machine's home directory, not
> this repo.

**1. Fully local (nothing leaves your machine):**

```bash
# install ollama.com, then:
ollama pull qwen3-coder:8b
ollama serve
freecode
```

**2. Free hosted models via OpenRouter:**

```bash
export OPENROUTER_API_KEY=sk-or-...   # free tier models available
freecode -m meta-llama/llama-3.3-70b-instruct:free
```

**3. Any OpenAI-compatible endpoint:**

```bash
freecode --base-url http://localhost:8000/v1 -m my-model
```

## Usage

```bash
freecode                     # interactive session
freecode "explain this repo" # one prompt, print, exit
freecode -y "do the refactor"  # auto-approve tool calls (no prompts)
freecode --plan "add caching"  # start in plan mode (read-only, proposes a plan)
freecode --resume <id>       # continue a saved session
freecode --cwd /path/to/proj # work in another directory
```

### Slash commands

| command | what it does |
| --- | --- |
| `/help` | list commands |
| `/model <name>` | switch model (saved to config) |
| `/models` | list models served by the current endpoint |
| `/config [key value]` | inspect or change settings |
| `/compact` | summarise history to free up context |
| `/undo` | revert the last file edit/write |
| `/sessions`, `/resume <id>` | list / resume saved sessions |
| `/paste` | multi-line input (finish with a lone `.`) |
| `/think` | toggle showing the model's reasoning |
| `/plan` | toggle plan mode (read-only research + proposed plan) |
| `/usage` | token usage and context size |
| `/clear`, `/cwd`, `/exit` | clear context / print cwd / quit |

Input tips: end a line with `\` to continue it; unbalanced `(`/`[`/`{`
automatically continues. `Ctrl-C` cancels a reply, `Ctrl-D` quits.

## Configuration

Settings live in `~/.freecode/config.json` (auto-created). Everything can also
be set by environment variables:

| variable | meaning |
| --- | --- |
| `FREECODE_BASE_URL` | OpenAI-compatible base URL |
| `FREECODE_API_KEY` | API key (also honours `OPENAI_API_KEY`, `OPENROUTER_API_KEY`) |
| `FREECODE_MODEL` | model name |
| `FREECODE_ALLOW_BASH=1` | don't ask before shell commands |
| `FREECODE_ALLOW_WRITE=1` | don't ask before file edits |
| `FREECODE_PLAN_MODE=1` | start in plan mode |
| `FREECODE_HOME` | override the `~/.freecode` directory |
| `NO_COLOR` | disable colours |

Useful config keys (via `/config key value`):
`temperature`, `max_turns`, `context_chars`, `bash_timeout`, `auto_compact`,
`show_thoughts`, `max_retries`, `fallback_models`
(e.g. `always_allow ["bash", "edit_file"]`,
`fallback_models '["poolside/laguna-s-2.1:free","openrouter/free"]'`).

Hit an OpenRouter limit and the agent backs off automatically (up to
`max_retries`, honouring the `Retry-After` header), then — if the limit
persists or the account is out of credits — switches to the next untried
model in `fallback_models`, followed by any other `:free` model your endpoint
serves. The switch is announced, applies to the current session only, and
never selects a paid model.

Sessions are persisted to `~/.freecode/sessions/`, so `/resume` restores full
conversation history — including tool results.

## How it works

```
freecode/
├── ansi.py       colour palette, wrapping, terminal width
├── markdown.py   streaming Markdown renderer + syntax highlighter
├── llm.py        stdlib SSE client for /chat/completions (tools + usage)
├── tools.py      read_file, write_file, edit_file, bash, glob, grep, list_dir, think
├── agent.py      the tool-calling loop, context budgeting, /compact
├── config.py     config + session persistence
├── repl.py       the interactive UI
└── __main__.py   CLI entry point
```

The agent talks to `/chat/completions` with function-calling tools, streams the
reply, executes approved tool calls, feeds the results back, and loops until the
model finishes (up to `max_turns`). Old tool output is trimmed automatically
when the transcript outgrows `context_chars`, and `/compact` folds the whole
session into a summary checkpoint.

## Safety

- Shell commands, file writes and edits require approval unless you opt in
  (`-y`, `always`, or `/config allow_bash true`).
- In plan mode (`/plan`, `--plan`) file writes and edits are blocked outright,
  and every shell command needs an individual yes — even under `-y`.
- Every file change is snapshotted, so `/undo` restores the previous content.
- Reads, searches and `think` never prompt.
- No network calls except to the model endpoint you configure.

## License

MIT
