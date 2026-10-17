"""CLI entry point: `freecode` / `python -m freecode`."""

from __future__ import annotations

import argparse
import sys

from . import __version__
from .config import Config
from .repl import REPL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="freecode",
        description="A free, local coding agent (open AI-compatible backends).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  freecode                          interactive session\n"
            "  freecode -p \"explain this repo\"   single prompt, then exit\n"
            "  freecode -m qwen3-coder:24b       pick a model\n"
            "  freecode --base-url http://localhost:11434/v1\n"
            "  FREECODE_BASE_URL=... FREECODE_API_KEY=... freecode\n"
        ),
    )
    parser.add_argument("prompt", nargs="*", help="prompt to run non-interactively")
    parser.add_argument("-m", "--model", help="model name (overrides config)")
    parser.add_argument("--base-url", help="OpenAI-compatible base URL")
    parser.add_argument("--provider", choices=["ollama", "openai", "openrouter", "anthropic", "custom"])
    parser.add_argument("-t", "--temperature", type=float, help="sampling temperature")
    parser.add_argument("-y", "--yolo", action="store_true",
                        help="auto-approve all tool calls (no prompts)")
    parser.add_argument("--plan", action="store_true",
                        help="start in plan mode (read-only research, no edits)")
    parser.add_argument("--resume", metavar="SESSION", help="resume a saved session id")
    parser.add_argument("--system", metavar="PROMPT", help="override the system prompt")
    parser.add_argument("--cwd", help="working directory (default: current)")
    parser.add_argument("--no-save", action="store_true", help="do not persist the session")
    parser.add_argument("--version", action="version", version=f"freecode {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = Config().load()

    if args.provider:
        config.provider = args.provider
    if args.base_url:
        config.base_url = args.base_url
        config.provider = "custom"
    if args.model:
        config.model = args.model
    if args.temperature is not None:
        config.temperature = args.temperature
    if args.plan:
        config.plan_mode = True
    if args.system:
        config.system_prompt = args.system

    repl = REPL(config, cwd=args.cwd, yolo=args.yolo)
    if args.resume:
        repl._cmd_resume(args.resume)

    if args.no_save:
        repl._save = lambda: None  # type: ignore[method-assign]

    prompt = " ".join(args.prompt).strip()
    if prompt:
        repl._ask(prompt)
        repl._save()
        return 0
    return repl.run()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print()
        sys.exit(130)
