"""Persistence abstraction for task graph state.

The storage layer is deliberately thin — it saves and loads dictionaries.
This keeps the interface small and makes it easy to swap backends later
(JSON → SQLite → Postgres) without touching any orchestration logic.

Design principle: storage is a pure I/O concern.  It knows nothing about
task semantics, scheduling, or execution.  It just persists bytes.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseStorage(ABC):
    """Abstract persistence interface for orchestration state.

    Implementations: JSONStorage (filesystem), future SQLStorage, etc.
    """

    @abstractmethod
    def save_graph(self, graph_data: dict) -> None:
        """Persist the full task graph state."""

    @abstractmethod
    def load_graph(self) -> dict | None:
        """Load persisted graph state, or None if not found."""

    @abstractmethod
    def save_run_log(self, run_id: str, log_data: dict) -> None:
        """Save execution log for a specific run."""

    @abstractmethod
    def load_run_log(self, run_id: str) -> dict | None:
        """Load execution log for a specific run."""

    @abstractmethod
    def list_runs(self) -> list[str]:
        """List all saved run IDs."""

    @abstractmethod
    def delete_run(self, run_id: str) -> None:
        """Delete a saved run."""


class JSONStorage(BaseStorage):
    """Filesystem-backed JSON persistence.

    Stores data under `<base_dir>/` with the following layout:
        <base_dir>/
            graphs/
                <graph_name>.json      # current graph state
            runs/
                <run_id>.json           # per-run execution logs
            history/
                <timestamp>_<name>.json  # historical graph snapshots

    This is intentionally simple.  For production deployments with
    many concurrent runs, swap to SQLite or Postgres by implementing
    the BaseStorage interface.
    """

    def __init__(self, base_dir: str | Path = ".corecoder/orchestration"):
        self._base = Path(base_dir)
        self._graphs_dir = self._base / "graphs"
        self._runs_dir = self._base / "runs"
        self._history_dir = self._base / "history"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        for d in (self._graphs_dir, self._runs_dir, self._history_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # graph persistence
    # ------------------------------------------------------------------

    def save_graph(self, graph_data: dict) -> None:
        name = graph_data.get("name", "task_graph")
        path = self._graphs_dir / f"{name}.json"
        self._write_json(path, graph_data)

        # Also save a timestamped history snapshot
        ts = int(time.time() * 1000)
        hist_path = self._history_dir / f"{ts}_{name}.json"
        self._write_json(hist_path, graph_data)

    def load_graph(self) -> dict | None:
        """Load the most recently saved graph."""
        graphs = sorted(self._graphs_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        if not graphs:
            return None
        return self._read_json(graphs[0])

    def load_graph_by_name(self, name: str) -> dict | None:
        path = self._graphs_dir / f"{name}.json"
        if not path.exists():
            return None
        return self._read_json(path)

    # ------------------------------------------------------------------
    # run log persistence
    # ------------------------------------------------------------------

    def save_run_log(self, run_id: str, log_data: dict) -> None:
        path = self._runs_dir / f"{run_id}.json"
        self._write_json(path, log_data)

    def load_run_log(self, run_id: str) -> dict | None:
        path = self._runs_dir / f"{run_id}.json"
        if not path.exists():
            return None
        return self._read_json(path)

    def list_runs(self) -> list[str]:
        runs = sorted(self._runs_dir.glob("*.json"), key=os.path.getmtime, reverse=True)
        return [r.stem for r in runs]

    def delete_run(self, run_id: str) -> None:
        path = self._runs_dir / f"{run_id}.json"
        if path.exists():
            path.unlink()

    # ------------------------------------------------------------------
    # history
    # ------------------------------------------------------------------

    def list_history(self, graph_name: str | None = None) -> list[dict[str, Any]]:
        """List historical graph snapshots, newest first."""
        pattern = f"*_{graph_name}.json" if graph_name else "*.json"
        files = sorted(
            self._history_dir.glob(pattern), key=os.path.getmtime, reverse=True
        )
        result = []
        for f in files:
            parts = f.stem.split("_", 1)
            result.append({
                "timestamp": int(parts[0]) if parts else 0,
                "name": parts[1] if len(parts) > 1 else f.stem,
                "path": str(f),
            })
        return result

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _write_json(self, path: Path, data: dict) -> None:
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        tmp.replace(path)  # atomic on POSIX; best-effort on Windows

    def _read_json(self, path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
