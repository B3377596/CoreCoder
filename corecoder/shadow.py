"""Shadow git repository for change tracking.

Like Claude Code's shadow repos, this maintains a separate git history
for the working directory so the agent can checkpoint, undo, and diff
without touching the user's ``.git``.

Git objects live under ``~/.corecoder/shadow/<project-hash>/`` while
``GIT_WORK_TREE`` points to the actual project root.  This means every
``git add`` / ``git commit`` / ``git reset`` operates on the real files
but stores history in the shadow directory.
"""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("corecoder.shadow")

SHADOW_ROOT = Path.home() / ".corecoder" / "shadow"

# files/dirs that should never be tracked by the shadow repo
_ALWAYS_IGNORE = [
    ".git/",
    ".corecoder/",
    "node_modules/",
    "__pycache__/",
    ".venv/",
    "venv/",
    ".tox/",
    "dist/",
    "build/",
    ".mypy_cache/",
    ".pytest_cache/",
    ".ruff_cache/",
    "*.pyc",
    ".DS_Store",
]


class ShadowGit:
    """Manages a shadow git repository for a specific working directory."""

    def __init__(self, work_tree: str | Path):
        self.work_tree = str(Path(work_tree).resolve())
        self.shadow_dir = self._shadow_path(self.work_tree)
        self._initialized = False

    @staticmethod
    def _shadow_path(work_tree: str) -> str:
        h = hashlib.md5(work_tree.encode()).hexdigest()[:12]
        return str(SHADOW_ROOT / h)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def init(self):
        """Initialize the shadow repository (idempotent)."""
        if self._initialized:
            return
        os.makedirs(self.shadow_dir, exist_ok=True)

        git_dir = Path(self.shadow_dir)
        if not (git_dir / "HEAD").exists():
            self._git("init")
            self._ensure_gitignore()
        self._initialized = True

    def _ensure_gitignore(self):
        """Write a .gitignore so the shadow repo doesn't track noise."""
        gitignore_path = Path(self.work_tree) / ".gitignore"
        existing: set[str] = set()
        if gitignore_path.exists():
            existing = set(gitignore_path.read_text().splitlines())

        # only add rules that aren't already present
        needed = [r for r in _ALWAYS_IGNORE if r not in existing]
        if needed:
            with open(gitignore_path, "a") as f:
                f.write("\n# CoreCoder shadow\n")
                for rule in needed:
                    f.write(rule + "\n")

    # ------------------------------------------------------------------
    # checkpoint / undo
    # ------------------------------------------------------------------

    def snapshot(self, message: str = "checkpoint"):
        """Commit the current working tree state. Fast no-op if clean."""
        self.init()
        self._git("add", "-A")
        # --allow-empty so we don't fail when nothing changed
        try:
            self._git("commit", "-m", message, "--allow-empty")
        except Exception:
            logger.debug("Shadow commit failed (may be empty or unchanged)")

    def undo(self):
        """Hard-reset to the previous commit, restoring all files."""
        self.init()
        try:
            self._git("reset", "--hard", "HEAD~1")
            logger.debug("Shadow undo: reset to HEAD~1")
        except Exception as e:
            logger.warning("Shadow undo failed: %s", e)
            raise

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def changed_files(self) -> list[str]:
        """Files modified in the working tree since the last snapshot (HEAD)."""
        try:
            out = self._git("diff", "--name-only", "HEAD")
            return [f for f in out.split("\n") if f]
        except Exception:
            return []

    def last_diff(self) -> str:
        """Unified diff of working tree vs last snapshot (HEAD)."""
        try:
            return self._git("diff", "HEAD")
        except Exception:
            return "(no diff available)"

    def session_diff(self) -> str:
        """Diff from first session commit to current state."""
        try:
            return self._git("diff", "SESSION_START", "HEAD")
        except Exception:
            # fallback: diff unstaged changes
            try:
                return self._git("diff", "HEAD")
            except Exception:
                return "(no diff available)"

    def tag_session_start(self):
        """Tag the current HEAD as the start of this session."""
        try:
            self._git("tag", "-f", "SESSION_START", "HEAD")
        except Exception:
            pass

    def checkout(self, ref: str = "HEAD"):
        """Checkout a specific ref (used for undo to specific checkpoint)."""
        self._git("checkout", ref)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _git(self, *args: str) -> str:
        """Run a git command with shadow GIT_DIR + GIT_WORK_TREE."""
        env = os.environ.copy()
        env["GIT_DIR"] = self.shadow_dir
        env["GIT_WORK_TREE"] = self.work_tree
        env["GIT_AUTHOR_NAME"] = env.get("GIT_AUTHOR_NAME") or "CoreCoder"
        env["GIT_AUTHOR_EMAIL"] = env.get("GIT_AUTHOR_EMAIL") or "corecoder@local"
        env["GIT_COMMITTER_NAME"] = env.get("GIT_COMMITTER_NAME") or "CoreCoder"
        env["GIT_COMMITTER_EMAIL"] = env.get("GIT_COMMITTER_EMAIL") or "corecoder@local"

        proc = subprocess.run(
            ["git"] + list(args),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=30,
            cwd=self.work_tree,
        )

        if proc.returncode != 0:
            stderr = proc.stderr.strip()
            raise RuntimeError(f"git {' '.join(args)}: {stderr}")

        return proc.stdout.strip()
