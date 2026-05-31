"""Runtime-local verification helpers for staged Think-Execute execution.

This module keeps only the verification pieces that are still used by the
default staged runtime. It intentionally avoids the old DAG/workflow layer.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class VerificationInput:
    """Lightweight execution snapshot for verification."""

    success: bool
    output: str = ""
    error: str = ""
    tool_calls_made: int = 0
    artifacts: dict[str, Any] = field(default_factory=dict)


@dataclass
class VerificationResult:
    """Structured verification outcome for the staged runtime."""

    passed: bool
    checks_run: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    should_retry: bool = False
    should_replan: bool = False
    replan_hint: str = ""


@dataclass
class PatchAnalysis:
    """Facts extracted from runtime side effects on the filesystem."""

    modified_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    python_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    diff_text: str = ""

    @property
    def all_changed(self) -> list[str]:
        return sorted(set(self.modified_files + self.created_files + self.deleted_files))

    @property
    def has_changes(self) -> bool:
        return bool(self.all_changed)

    @classmethod
    def from_shadow(cls, shadow_git, working_dir: str = ".") -> PatchAnalysis:
        analysis = cls()
        cwd = Path(working_dir)

        try:
            changed = shadow_git.changed_files() if shadow_git else []
            diff_text = shadow_git.last_diff() if shadow_git else ""
            analysis.diff_text = diff_text

            for rel_path in changed:
                full = cwd / rel_path
                if not full.exists():
                    analysis.deleted_files.append(rel_path)
                else:
                    analysis.modified_files.append(rel_path)

            if diff_text:
                import re

                for match in re.finditer(r"^\+\+\+ b/(.+)$", diff_text, re.MULTILINE):
                    rel_path = match.group(1)
                    if rel_path not in changed and (cwd / rel_path).exists():
                        analysis.created_files.append(rel_path)
        except Exception:
            pass

        analysis._categorize()
        return analysis

    def _categorize(self) -> None:
        for rel_path in self.all_changed:
            if rel_path.endswith(".py"):
                self.python_files.append(rel_path)
                path = Path(rel_path)
                if "test" in path.stem or "test" in str(path.parent):
                    self.test_files.append(rel_path)
            elif any(rel_path.endswith(ext) for ext in (".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".env")):
                self.config_files.append(rel_path)


class BaseVerifier(ABC):
    @abstractmethod
    def verify(
        self,
        result: VerificationInput,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
        task_meta: dict[str, Any] | None = None,
    ) -> VerificationResult:
        """Run verification against runtime facts."""


class FileCreatedVerifier(BaseVerifier):
    """Verify that files mentioned in execution output actually exist."""

    def verify(
        self,
        result: VerificationInput,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
        task_meta: dict[str, Any] | None = None,
    ) -> VerificationResult:
        cwd = Path(working_dir) if working_dir else Path.cwd()

        import re

        mentioned_files: set[str] = set()
        for line in result.output.split("\n"):
            for match in re.finditer(
                r"(?:created|modified|wrote|updated|changed|file|path)[:\s]*[`\"']*([^\s`\"',;]+\.\w+)",
                line,
                re.IGNORECASE,
            ):
                mentioned_files.add(match.group(1))

        if not mentioned_files:
            if patch and patch.created_files:
                mentioned_files = set(patch.created_files)
            else:
                return VerificationResult(passed=True, checks_run=["file_exists:nothing_to_check"])

        failures: list[str] = []
        for rel_path in sorted(mentioned_files):
            full = cwd / rel_path
            if full.exists():
                continue
            candidates = list(cwd.rglob(Path(rel_path).name))
            if not candidates:
                failures.append(f"File mentioned in output not found: {rel_path}")

        return VerificationResult(
            passed=not failures,
            checks_run=["file_exists"],
            failures=failures,
            should_retry=bool(failures),
            should_replan=bool(failures),
            replan_hint="Agent claimed to create files that don't exist" if failures else "",
        )


class SyntaxVerifier(BaseVerifier):
    """Verify syntax for changed Python files."""

    def verify(
        self,
        result: VerificationInput,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
        task_meta: dict[str, Any] | None = None,
    ) -> VerificationResult:
        if not patch or not patch.python_files:
            return VerificationResult(passed=True, checks_run=["syntax:no_python_changes"])

        cwd = Path(working_dir) if working_dir else Path.cwd()
        failures: list[str] = []
        for rel_path in patch.python_files:
            full = cwd / rel_path
            if not full.exists():
                continue
            try:
                source = full.read_text(encoding="utf-8")
                compile(source, str(full), "exec")
            except SyntaxError as exc:
                failures.append(f"Syntax error in {rel_path}: {exc}")
            except Exception:
                pass

        return VerificationResult(
            passed=not failures,
            checks_run=[f"syntax:{len(patch.python_files)}_files"],
            failures=failures,
            should_retry=bool(failures),
        )


class TestVerifier(BaseVerifier):
    """Run a test command when test files were modified."""

    def verify(
        self,
        result: VerificationInput,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
        task_meta: dict[str, Any] | None = None,
    ) -> VerificationResult:
        cwd = working_dir or "."
        title = (task_meta or {}).get("title", "").lower()
        if any(keyword in title for keyword in ("init", "venv", "setup", "create project", "virtual environment")):
            return VerificationResult(passed=True, checks_run=["test:skipped_setup_task"])

        if patch and not patch.test_files:
            return VerificationResult(passed=True, checks_run=["test:no_test_changes"])

        for cmd in ["pytest -x --tb=short", "python -m pytest -x --tb=short"]:
            try:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                    cwd=cwd,
                )
                if proc.returncode == 0:
                    return VerificationResult(passed=True, checks_run=["test"])
                return VerificationResult(
                    passed=False,
                    checks_run=["test"],
                    failures=[f"Tests failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-500:]}"],
                    should_retry=True,
                )
            except FileNotFoundError:
                continue
            except subprocess.TimeoutExpired:
                return VerificationResult(
                    passed=False,
                    checks_run=["test"],
                    failures=["Tests timed out"],
                    should_retry=True,
                )
            except Exception:
                continue

        return VerificationResult(passed=True, checks_run=["test:no_runner_found"])


class CompositeVerifier(BaseVerifier):
    def __init__(self, verifiers: list[BaseVerifier] | None = None):
        self._verifiers = verifiers or []

    def verify(
        self,
        result: VerificationInput,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
        task_meta: dict[str, Any] | None = None,
    ) -> VerificationResult:
        if not self._verifiers:
            return VerificationResult(passed=True)

        aggregated = VerificationResult(passed=True)
        for verifier in self._verifiers:
            vr = verifier.verify(result, patch=patch, working_dir=working_dir, task_meta=task_meta)
            aggregated.checks_run.extend(vr.checks_run)
            aggregated.failures.extend(vr.failures)
            aggregated.warnings.extend(vr.warnings)
            aggregated.suggestions.extend(vr.suggestions)
            aggregated.passed = aggregated.passed and vr.passed
            aggregated.should_retry = aggregated.should_retry or vr.should_retry
            aggregated.should_replan = aggregated.should_replan or vr.should_replan
            if vr.replan_hint:
                sep = "; " if aggregated.replan_hint else ""
                aggregated.replan_hint += sep + vr.replan_hint
        return aggregated


class VerificationPolicyEngine:
    """Select lightweight verifiers from actual patch facts."""

    def select_verifiers(self, patch: PatchAnalysis | None) -> list[BaseVerifier]:
        verifiers: list[BaseVerifier] = [FileCreatedVerifier()]
        if patch is None or not patch.has_changes:
            return verifiers
        if patch.python_files:
            verifiers.append(SyntaxVerifier())
        if patch.test_files:
            verifiers.append(TestVerifier())
        return verifiers

    def build(self, patch: PatchAnalysis | None) -> CompositeVerifier:
        return CompositeVerifier(self.select_verifiers(patch))

    def verify(
        self,
        result: VerificationInput,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
        task_meta: dict[str, Any] | None = None,
    ) -> VerificationResult:
        return self.build(patch).verify(
            result,
            patch=patch,
            working_dir=working_dir,
            task_meta=task_meta,
        )
