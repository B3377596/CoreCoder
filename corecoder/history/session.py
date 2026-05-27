"""Session persistence - save and resume conversations.

Claude Code maintains session state via QueryEngine (1295 lines).
CoreCoder distills this to: JSON dump of SessionState + model config.

Schema version 2 (state-centric): saves SessionState fields.
Schema version 1 (legacy): saves raw messages list.
"""

import json
import re
import time
import uuid
from pathlib import Path
from typing import Any

SESSIONS_DIR = Path.home() / ".corecoder" / "sessions"
_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _normalize_session_id(session_id: str | None) -> str:
    if not session_id:
        return _new_session_id()

    name = session_id.strip().replace("\\", "/").split("/")[-1]
    name = _SAFE_SESSION_RE.sub("-", name).strip(".-_")
    return name or _new_session_id()


def _new_session_id() -> str:
    return f"session_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"


def _session_path(session_id: str) -> Path:
    path = (SESSIONS_DIR / f"{_normalize_session_id(session_id)}.json").resolve()
    root = SESSIONS_DIR.resolve()
    if root != path.parent:
        raise ValueError("Invalid session id")
    return path


def save_session(state: Any, model: str, session_id: str | None = None) -> str:
    """Save conversation state to disk. Returns the session ID.

    Accepts either a SessionState object (v2) or a raw messages list (v1 compat).
    """
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

    session_id = _normalize_session_id(session_id)

    # Detect SessionState vs legacy messages list
    if hasattr(state, 'to_dict'):
        # SessionState object (v2)
        data = {
            "id": session_id,
            "model": model,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "version": 2,
            **state.to_dict(),
        }
    else:
        # Legacy messages list (v1 compat)
        data = {
            "id": session_id,
            "model": model,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "messages": state,
        }

    path = _session_path(session_id)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return session_id


def load_session(session_id: str) -> tuple[Any, str] | None:
    """Load a saved session. Returns (SessionState_or_messages, model) or None.

    v2 sessions return a SessionState object.
    v1 sessions return a raw messages list for backward compatibility.
    """
    path = _session_path(session_id)
    if not path.exists():
        return None

    data = json.loads(path.read_text(encoding="utf-8"))

    if data.get("version") == 2:
        # v2: SessionState
        from corecoder.runtime.state import SessionState
        state = SessionState.from_dict(data)
        return state, data["model"]
    else:
        # v1: raw messages list
        return data["messages"], data["model"]


def list_sessions() -> list[dict]:
    """List available sessions, newest first."""
    if not SESSIONS_DIR.exists():
        return []

    sessions = []
    for f in sorted(SESSIONS_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            # grab first user message as preview
            preview = ""
            messages = data.get("persistent_history") or data.get("messages", [])
            for m in messages:
                if m.get("role") == "user" and m.get("content"):
                    preview = m["content"][:80]
                    break
            sessions.append({
                "id": data.get("id", f.stem),
                "model": data.get("model", "?"),
                "saved_at": data.get("saved_at", "?"),
                "preview": preview,
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return sessions[:20]  # cap at 20
