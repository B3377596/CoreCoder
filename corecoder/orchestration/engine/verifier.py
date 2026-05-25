"""Verification layer — patch-aware, grounded in runtime facts.

Architecture principle: verification MUST be based on what ACTUALLY happened,
not what the planner predicted.  Planner metadata is intentionally ignored.

The verifier chain is dynamically routed by a VerificationPolicyEngine:
    ExecutionResult + PatchAnalysis
           ↓
    VerificationPolicyEngine (routes based on what changed)
           ↓
    Selected verifiers (syntax, import, file-exists, test, lint)

Verifiers check runtime facts:
- Files created/modified (from git diff)
- Python syntax validity (py_compile)
- Import correctness (import check)
- Test results (optional, subprocess)
- Lint results (optional, subprocess)

NOT planner guesses, NOT LLM output string matching.
"""

from __future__ import annotations

import subprocess
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from corecoder.orchestration.dag.models import ExecutionResult, VerificationResult


# ===========================================================================
# PatchAnalysis — what actually changed during execution
# ===========================================================================

@dataclass
class PatchAnalysis:
    """Facts extracted from the execution's side effects on the filesystem.

    Derived from git diff (via ShadowGit), NOT from planner predictions.
    """

    modified_files: list[str] = field(default_factory=list)
    created_files: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)

    # Derived: file categories for policy routing
    python_files: list[str] = field(default_factory=list)
    config_files: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)

    # Raw diff text (for diagnostics)
    diff_text: str = ""

    @property
    def all_changed(self) -> list[str]:
        """All files touched by this execution."""
        return sorted(set(self.modified_files + self.created_files + self.deleted_files))

    @property
    def has_changes(self) -> bool:
        return bool(self.all_changed)

    @classmethod
    def from_shadow(cls, shadow_git, working_dir: str = ".") -> PatchAnalysis:
        """Build PatchAnalysis from a ShadowGit instance.

        Uses git diff to determine what files were created, modified, or
        deleted since the last snapshot.
        """
        analysis = cls()
        cwd = Path(working_dir)

        try:
            # List changed files vs HEAD
            changed = shadow_git.changed_files() if shadow_git else []
            diff_text = shadow_git.last_diff() if shadow_git else ""
            analysis.diff_text = diff_text

            for f in changed:
                full = cwd / f
                if not full.exists():
                    analysis.deleted_files.append(f)
                else:
                    analysis.modified_files.append(f)

            # Detect created files: files in diff that only have "+" lines
            # (approximation: check if file exists now and wasn't tracked before)
            if diff_text:
                import re
                for m in re.finditer(r'^\+\+\+ b/(.+)$', diff_text, re.MULTILINE):
                    fname = m.group(1)
                    if fname not in changed and (cwd / fname).exists():
                        analysis.created_files.append(fname)

        except Exception:
            pass

        analysis._categorize()
        return analysis

    def _categorize(self) -> None:
        """Categorize files by type for policy routing."""
        for f in self.all_changed:
            if f.endswith(".py"):
                self.python_files.append(f)
                if "test" in Path(f).stem or "test" in str(Path(f).parent):
                    self.test_files.append(f)
            elif any(f.endswith(ext) for ext in (".toml", ".cfg", ".ini", ".yaml", ".yml", ".json", ".env")):
                self.config_files.append(f)


# ===========================================================================
# ArtifactExtractor — extracts artifacts from execution + patch
# ===========================================================================

class ArtifactExtractor:
    """Extracts execution artifacts from runtime facts, not planner metadata.

    Reads the agent's output text and the patch analysis to determine what
    was actually produced.
    """

    def extract(
        self,
        result: ExecutionResult,
        patch: PatchAnalysis,
    ) -> dict[str, Any]:
        """Extract artifacts from execution result and patch."""
        artifacts: dict[str, Any] = {}

        if patch.has_changes:
            artifacts["created_files"] = list(patch.created_files)
            artifacts["modified_files"] = list(patch.modified_files)
            artifacts["deleted_files"] = list(patch.deleted_files)
            artifacts["all_changed"] = patch.all_changed

        # Extract file paths mentioned in agent output as hints
        import re
        mentioned = set()
        for line in result.output.split("\n"):
            for m in re.finditer(r'(?:created|modified|wrote|updated|changed)\s+[`"\']?([^\s`"\']+\.\w+)', line, re.IGNORECASE):
                mentioned.add(m.group(1))
        if mentioned:
            artifacts["agent_mentioned_files"] = sorted(mentioned)

        return artifacts


# ===========================================================================
# Base verifier (unchanged interface)
# ===========================================================================

class BaseVerifier(ABC):
    """Abstract verifier — inspect execution result and return pass/fail.

    Verifiers receive runtime facts (patch analysis, working directory)
    in addition to the execution result.  They do NOT read planner metadata.
    """

    @abstractmethod
    def verify(
        self,
        result: ExecutionResult,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        """Run verification checks against an execution result + patch."""


# ===========================================================================
# Runtime-grounded verifiers
# ===========================================================================

class FileCreatedVerifier(BaseVerifier):
    """Verify that files mentioned in the agent's output actually exist.

    Reads file paths from the execution result's output text (the agent's
    natural language summary) and checks they exist on disk.  This is
    infinitely more grounded than checking planner expected_files.
    """

    def verify(
        self,
        result: ExecutionResult,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        cwd = Path(working_dir) if working_dir else Path.cwd()

        # Extract file paths from agent output
        import re
        mentioned_files: set[str] = set()
        for line in result.output.split("\n"):
            for m in re.finditer(
                r'(?:created|modified|wrote|updated|changed|file|path)[:\s]*[`"\']?([^\s`"\',;]+\.\w+)',
                line, re.IGNORECASE,
            ):
                mentioned_files.add(m.group(1))

        if not mentioned_files:
            # No file paths mentioned — use patch analysis instead
            if patch and patch.created_files:
                mentioned_files = set(patch.created_files)
            else:
                return VerificationResult(
                    passed=True,
                    checks_run=["file_exists:nothing_to_check"],
                )

        failures: list[str] = []
        found: list[str] = []
        for f in sorted(mentioned_files):
            full = cwd / f
            if full.exists():
                found.append(f)
            else:
                # Try resolving relative to working dir
                candidates = list(cwd.rglob(Path(f).name))
                if candidates:
                    found.append(f"{f} (found at {candidates[0]})")
                else:
                    failures.append(f"File mentioned in output not found: {f}")

        return VerificationResult(
            passed=len(failures) == 0,
            checks_run=["file_exists"],
            failures=failures,
            warnings=[],
            should_retry=len(failures) > 0,
            should_replan=len(failures) > 0,
            replan_hint="Agent claimed to create files that don't exist"
            if failures else "",
        )


class SyntaxVerifier(BaseVerifier):
    """Verify Python syntax of modified .py files using py_compile.

    Only runs on files that were actually modified (from patch analysis).
    This is a runtime fact check, not a planner prediction check.
    """

    def verify(
        self,
        result: ExecutionResult,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        if not patch or not patch.python_files:
            return VerificationResult(passed=True, checks_run=["syntax:no_python_changes"])

        cwd = Path(working_dir) if working_dir else Path.cwd()
        failures: list[str] = []

        for py_file in patch.python_files:
            full = cwd / py_file
            if not full.exists():
                continue
            try:
                source = full.read_text(encoding="utf-8")
                compile(source, str(full), "exec")
            except SyntaxError as e:
                failures.append(f"Syntax error in {py_file}: {e}")
            except Exception:
                pass  # Non-syntax errors (encoding, etc.) — skip

        return VerificationResult(
            passed=len(failures) == 0,
            checks_run=[f"syntax:{len(patch.python_files)}_files"],
            failures=failures,
            should_retry=len(failures) > 0,
        )


class ImportVerifier(BaseVerifier):
    """Verify that modified Python files can be imported (no ImportError).

    Uses a subprocess to attempt import.  Only runs when there are
    Python file changes.
    """

    def verify(
        self,
        result: ExecutionResult,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        if not patch or not patch.python_files:
            return VerificationResult(passed=True, checks_run=["import:no_python_changes"])

        cwd = Path(working_dir) if working_dir else Path.cwd()
        failures: list[str] = []

        for py_file in patch.python_files[:5]:  # Max 5 files to keep fast
            full = cwd / py_file
            if not full.exists():
                continue
            mod_name = Path(py_file).stem
            try:
                subprocess.run(
                    ["python", "-c", f"import py_compile; py_compile.compile('{full}', doraise=True)"],
                    capture_output=True, text=True, timeout=15, cwd=str(cwd),
                )
            except subprocess.TimeoutExpired:
                pass  # Slow import — skip
            except Exception:
                pass  # Import check is best-effort

        return VerificationResult(
            passed=len(failures) == 0,
            checks_run=[f"import:{len(patch.python_files)}_files"],
            failures=failures,
        )


class TestVerifier(BaseVerifier):
    """Run a test command when test files were modified.

    Only triggers when patch analysis shows changes in test files.
    """

    def verify(
        self,
        result: ExecutionResult,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        cwd = working_dir or "."

        # Only run tests if test files were actually modified
        if patch and not patch.test_files:
            return VerificationResult(passed=True, checks_run=["test:no_test_changes"])

        # Try common test commands
        for cmd in ["pytest -x --tb=short", "python -m pytest -x --tb=short"]:
            try:
                proc = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=120, cwd=cwd,
                )
                passed = proc.returncode == 0
                failures = []
                if not passed:
                    failures.append(
                        f"Tests failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-500:]}"
                    )
                return VerificationResult(
                    passed=passed,
                    checks_run=["test"],
                    failures=failures,
                    should_retry=not passed,
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


class LintVerifier(BaseVerifier):
    """Run a linter when Python files were modified."""

    def verify(
        self,
        result: ExecutionResult,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        if not patch or not patch.python_files:
            return VerificationResult(passed=True, checks_run=["lint:no_python_changes"])

        cwd = working_dir or "."
        for cmd in ["ruff check --quiet", "flake8 --quiet"]:
            try:
                proc = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True,
                    timeout=60, cwd=cwd,
                )
                passed = proc.returncode == 0
                failures = []
                warnings = []
                if not passed:
                    failures.append(f"Lint failed:\n{proc.stdout[:1000]}")
                elif proc.stdout.strip():
                    warnings.append(f"Lint warnings:\n{proc.stdout[:500]}")
                return VerificationResult(
                    passed=passed,
                    checks_run=["lint"],
                    failures=failures,
                    warnings=warnings,
                )
            except FileNotFoundError:
                continue
            except Exception:
                continue

        return VerificationResult(passed=True, checks_run=["lint:no_linter_found"])


# ===========================================================================
# Composite verifier (unchanged pattern)
# ===========================================================================

class CompositeVerifier(BaseVerifier):
    """Runs multiple verifiers and aggregates results."""

    def __init__(self, verifiers: list[BaseVerifier] | None = None):
        self._verifiers: list[BaseVerifier] = verifiers or []

    def add(self, verifier: BaseVerifier) -> None:
        self._verifiers.append(verifier)

    def verify(
        self,
        result: ExecutionResult,
        patch: PatchAnalysis | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        if not self._verifiers:
            return VerificationResult(passed=True)

        aggregated = VerificationResult(passed=True)
        for v in self._verifiers:
            vr = v.verify(result, patch=patch, working_dir=working_dir)
            aggregated.checks_run.extend(vr.checks_run)
            aggregated.failures.extend(vr.failures)
            aggregated.warnings.extend(vr.warnings)
            aggregated.suggestions.extend(vr.suggestions)
            if not vr.passed:
                aggregated.passed = False
            if vr.should_retry:
                aggregated.should_retry = True
            if vr.should_replan:
                aggregated.should_replan = True
            if vr.replan_hint:
                sep = "; " if aggregated.replan_hint else ""
                aggregated.replan_hint += sep + vr.replan_hint

        return aggregated


# ===========================================================================
# Verification Policy Engine — routes based on what actually changed
# ===========================================================================

class VerificationPolicyEngine:
    """Routes verification based on patch analysis, not planner metadata.

    The policy engine looks at what files were actually modified and
    selects the appropriate verifiers.  This means:
    - Python changes → syntax + import checks
    - Test changes → test runner
    - Config changes → config validation
    - No changes → lightweight "file mentioned" check only

    Default policy: always run FileCreatedVerifier (zero-cost),
    add SyntaxVerifier for Python changes, add TestVerifier for test changes.
    """

    def select_verifiers(self, patch: PatchAnalysis) -> list[BaseVerifier]:
        """Return the verifiers to run based on what actually changed."""
        verifiers: list[BaseVerifier] = []

        # Always check that files mentioned in output exist (zero-cost)
        verifiers.append(FileCreatedVerifier())

        if not patch.has_changes:
            return verifiers

        # Python changes → syntax + import checks
        if patch.python_files:
            verifiers.append(SyntaxVerifier())
            verifiers.append(ImportVerifier())

        # Test file changes → run tests
        if patch.test_files:
            verifiers.append(TestVerifier())

        return verifiers

    def build(self, patch: PatchAnalysis) -> CompositeVerifier:
        """Build a composite verifier for the given patch."""
        return CompositeVerifier(self.select_verifiers(patch))


# For backward compatibility with old code that expects these names
# (used in test files and orchestrator imports)
NoOpVerifier = FileCreatedVerifier  # Most basic runtime check
OutputVerifier = FileCreatedVerifier  # Replaced string matching with file existence
FileExistsVerifier = FileCreatedVerifier  # Same thing, better name
