"""Verification layer for task execution results.

After a task executes, the verifier inspects the output and decides
whether the task really succeeded.  This is the "trust but verify"
layer between execution and graph state updates.

Verification results influence:
- Whether the task is marked SUCCESS or FAILED
- Whether a retry should be attempted
- Whether the planner should replan
- What error context is fed back to the LLM on retry

The verifier is a composable pipeline: you can chain multiple checkers
together and run them all against a single execution result.
"""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from corecoder.orchestration.dag.models import ExecutionResult, VerificationResult


class BaseVerifier(ABC):
    """Abstract verifier — inspect an execution result and decide pass/fail.

    Verifiers should be stateless.  They receive the execution result,
    the task node's metadata, and return a VerificationResult.
    """

    @abstractmethod
    def verify(
        self,
        result: ExecutionResult,
        task_metadata: dict[str, Any] | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        """Run verification checks against an execution result."""


class NoOpVerifier(BaseVerifier):
    """Default verifier: trusts the executor's self-reported success/failure.

    Used when no specific verification hooks are configured.
    """

    def verify(
        self,
        result: ExecutionResult,
        task_metadata: dict[str, Any] | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        return VerificationResult(
            passed=result.success,
            checks_run=["self_report"],
            failures=[] if result.success else [result.error or "Task reported failure"],
        )


class CompositeVerifier(BaseVerifier):
    """Runs multiple verifiers and aggregates results.

    All verifiers run (no short-circuit) so the user sees every failure
    at once rather than fixing them one at a time.
    """

    def __init__(self, verifiers: list[BaseVerifier] | None = None):
        self._verifiers: list[BaseVerifier] = verifiers or []

    def add(self, verifier: BaseVerifier) -> None:
        self._verifiers.append(verifier)

    def verify(
        self,
        result: ExecutionResult,
        task_metadata: dict[str, Any] | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        if not self._verifiers:
            return VerificationResult(passed=result.success)

        aggregated = VerificationResult(passed=True)
        for v in self._verifiers:
            vr = v.verify(result, task_metadata, working_dir)
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
                if aggregated.replan_hint:
                    aggregated.replan_hint += "; " + vr.replan_hint
                else:
                    aggregated.replan_hint = vr.replan_hint

        return aggregated


class TestVerifier(BaseVerifier):
    """Run a test command and check its exit code.

    Configurable via task metadata:
        {"test_command": "pytest tests/ -x", "test_timeout": 60}
    """

    def verify(
        self,
        result: ExecutionResult,
        task_metadata: dict[str, Any] | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        meta = task_metadata or {}
        command = meta.get("test_command")
        if not command:
            return VerificationResult(passed=True, checks_run=["test_command:skipped"])

        timeout = meta.get("test_timeout", 120)
        cwd = working_dir or "."

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            passed = proc.returncode == 0
            failures = []
            if not passed:
                failures.append(
                    f"Test command failed (exit {proc.returncode}):\n"
                    f"STDOUT:\n{proc.stdout[-2000:]}\n"
                    f"STDERR:\n{proc.stderr[-1000:]}"
                )
            return VerificationResult(
                passed=passed,
                checks_run=["test_command"],
                failures=failures,
                should_retry=not passed,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                checks_run=["test_command"],
                failures=[f"Test command timed out after {timeout}s"],
                should_retry=True,
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                checks_run=["test_command"],
                failures=[f"Test command error: {e}"],
            )


class LintVerifier(BaseVerifier):
    """Run a linter/typechecker and check its exit code.

    Configurable via task metadata:
        {"lint_command": "ruff check .", "lint_timeout": 60}
    """

    def verify(
        self,
        result: ExecutionResult,
        task_metadata: dict[str, Any] | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        meta = task_metadata or {}
        command = meta.get("lint_command")
        if not command:
            return VerificationResult(passed=True, checks_run=["lint:skipped"])

        timeout = meta.get("lint_timeout", 60)
        cwd = working_dir or "."

        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd,
            )
            passed = proc.returncode == 0
            failures = []
            warnings = []
            if not passed:
                failures.append(f"Lint failed:\n{proc.stdout[-2000:]}\n{proc.stderr[-1000:]}")
            elif proc.stdout.strip():
                warnings.append(f"Lint warnings:\n{proc.stdout[:1000]}")
            return VerificationResult(
                passed=passed,
                checks_run=["lint"],
                failures=failures,
                warnings=warnings,
                should_retry=not passed,
            )
        except subprocess.TimeoutExpired:
            return VerificationResult(
                passed=False,
                checks_run=["lint"],
                failures=[f"Lint timed out after {timeout}s"],
                should_retry=True,
            )
        except Exception as e:
            return VerificationResult(
                passed=False,
                checks_run=["lint"],
                failures=[f"Lint error: {e}"],
            )


class OutputVerifier(BaseVerifier):
    """Check the execution output for required patterns or forbidden patterns.

    Configurable via task metadata:
        {
            "required_patterns": ["def test_", "SUCCESS"],
            "forbidden_patterns": ["TODO", "FIXME", "Error:"],
        }
    """

    def verify(
        self,
        result: ExecutionResult,
        task_metadata: dict[str, Any] | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        meta = task_metadata or {}
        required = meta.get("required_patterns", [])
        forbidden = meta.get("forbidden_patterns", [])

        if not required and not forbidden:
            return VerificationResult(passed=True, checks_run=["output_patterns:skipped"])

        output = result.output
        failures = []

        for pattern in required:
            if pattern not in output:
                failures.append(f"Required pattern not found in output: '{pattern}'")

        for pattern in forbidden:
            if pattern in output:
                failures.append(f"Forbidden pattern found in output: '{pattern}'")

        return VerificationResult(
            passed=len(failures) == 0,
            checks_run=["output_patterns"],
            failures=failures,
            should_retry=len(failures) > 0,
        )


class FileExistsVerifier(BaseVerifier):
    """Verify that specified files were created/modified.

    Configurable via task metadata:
        {"expected_files": ["src/app.py", "tests/test_app.py"]}
    """

    def verify(
        self,
        result: ExecutionResult,
        task_metadata: dict[str, Any] | None = None,
        working_dir: str | None = None,
    ) -> VerificationResult:
        meta = task_metadata or {}
        expected = meta.get("expected_files", [])

        if not expected:
            return VerificationResult(passed=True, checks_run=["file_exists:skipped"])

        cwd = Path(working_dir) if working_dir else Path.cwd()
        failures = []
        for filepath in expected:
            full_path = cwd / filepath
            if not full_path.exists():
                failures.append(f"Expected file not found: {filepath}")

        return VerificationResult(
            passed=len(failures) == 0,
            checks_run=["file_exists"],
            failures=failures,
            should_retry=len(failures) > 0,
            replan_hint="Missing expected output files — may need to create them"
            if failures
            else "",
        )
