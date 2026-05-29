"""Configuration — env vars and defaults.

Load order (later sources override earlier if not already set):
1. Default values in Config dataclass
2. ~/.corecoder/.env          (global user config)
3. .env walking up from cwd   (project-specific, up to home dir)
4. Environment variables       (always win — set in shell profile is safest)

This means you never need a .env in your project directory.
Set keys once in your shell or ~/.corecoder/.env and they apply everywhere.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv():
    """Load .env files in priority order.  No-op if python-dotenv missing."""
    try:
        from dotenv import load_dotenv

        # 1. Global user config — always loaded if it exists
        global_env = Path.home() / ".corecoder" / ".env"
        if global_env.exists():
            load_dotenv(global_env, override=False)

        # 2. Project .env — walk up from cwd to home
        cwd_env = Path(".env")
        if not cwd_env.exists():
            cur = Path.cwd()
            home = Path.home()
            while cur != home and cur != cur.parent:
                candidate = cur / ".env"
                if candidate.exists():
                    cwd_env = candidate
                    break
                cur = cur.parent
        load_dotenv(cwd_env, override=False)

    except ImportError:
        pass  # python-dotenv not installed, silently skip


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean env var using common truthy/falsey spellings."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Config:
    model: str = "gpt-4o"
    api_key: str = ""
    base_url: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.0
    max_context_tokens: int = 128_000
    provider: str = "openai"
    debug: bool = False

    @classmethod
    def from_env(cls) -> "Config":
        # load .env if present (won't override existing env vars)
        _load_dotenv()
        # pick up common env vars automatically
        api_key = (
            os.getenv("CORECODER_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or ""
        )
        return cls(
            model=os.getenv("CORECODER_MODEL", "gpt-4o"),
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or os.getenv("CORECODER_BASE_URL"),
            max_tokens=int(os.getenv("CORECODER_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("CORECODER_TEMPERATURE", "0")),
            max_context_tokens=int(os.getenv("CORECODER_MAX_CONTEXT", "128000")),
            provider=os.getenv("CORECODER_PROVIDER", "openai"),
            debug=_env_flag("CORECODER_DEBUG"),
        )
