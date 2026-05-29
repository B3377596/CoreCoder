"""Codebase services: indexing and shadow workspace management."""

from corecoder.codebase.indexing.index import RepoIndex, should_skip_path
from corecoder.codebase.shadow import ShadowGit

__all__ = ["RepoIndex", "ShadowGit", "should_skip_path"]
