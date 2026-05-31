"""Stage-driven Think-Execute runtime layered over the existing ReAct agent.

Outer loop:
    ThinkEngine -> StagePlan -> StageExecutor -> Evaluation -> State update

Inner loop:
    Existing Agent.chat() ReAct tool loop, constrained to one stage.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, TYPE_CHECKING

from corecoder.agent.dag.models import ExecutionResult as DagExecutionResult
from corecoder.agent.runtime.state import SessionState
from corecoder.agent.workflow.verifier import PatchAnalysis, VerificationPolicyEngine
from corecoder.context.models import ExecutionState
from corecoder.prompt import system_prompt
from corecoder.retrieval.task_understanding import TaskUnderstandingAnalyzer

if TYPE_CHECKING:
    from corecoder.agent.core import Agent
    from corecoder.context.orchestrator import ContextOrchestrator


ChatFn = Callable[..., Awaitable[str]]
RuntimeEventFn = Callable[[dict[str, Any]], None]


DEFAULT_STAGE_TOOLSETS: dict[str, list[str]] = {
    "understand": ["repo_info", "glob", "grep", "read_file"],
    "locate": ["repo_info", "glob", "grep", "read_file"],
    "analyze": ["read_file", "grep", "glob", "repo_info"],
    "modify": ["read_file", "edit_file", "write_file", "grep", "bash"],
    "verify": ["read_file", "grep", "bash"],
    "recover": ["repo_info", "glob", "grep", "read_file", "bash"],
    "finalize": [],
}

STAGE_CONTEXT_POLICIES: dict[str, str] = {
    "understand": "overview_first",
    "locate": "symbol_path_focus",
    "analyze": "targeted_analysis",
    "modify": "editable_exact_files",
    "verify": "verification_first",
    "recover": "failure_recovery",
    "finalize": "summary_only",
}

TOOL_ALIASES: dict[str, set[str]] = {
    "search_symbols": {"repo_info", "grep", "glob"},
    "read_file": {"read_file"},
    "grep": {"grep"},
    "glob": {"glob"},
    "edit": {"edit_file"},
    "edit_file": {"edit_file"},
    "write": {"write_file"},
    "write_file": {"write_file"},
    "run_command": {"bash"},
    "bash": {"bash"},
    "test": {"bash"},
    "repo_info": {"repo_info"},
    "agent": {"agent"},
}


def _stage_debug(title: str, payload: dict[str, Any]) -> None:
    payload = _sanitize_debug_payload(title, payload)
    print(f"\n{'=' * 68}")
    print(f"[STAGED DEBUG] {title}")
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))
    print(f"{'=' * 68}\n")
    sys.stdout.flush()


def _json_default(value: Any):
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)


def _sanitize_debug_payload(title: str, payload: dict[str, Any]) -> dict[str, Any]:
    if os.environ.get("DEBUG_FULL_TRACE", "").lower() == "true":
        return payload

    if title == "StageExecutor Output":
        trace = payload.get("trace", [])
        tool_names = []
        for step in trace:
            if isinstance(step, dict):
                tool_names.append(step.get("tool_name", ""))
            else:
                tool_names.append(getattr(step, "tool_name", ""))
        return {
            "stage": payload.get("stage"),
            "success": payload.get("success"),
            "tool_count": len(tool_names),
            "tool_names": tool_names,
            "trace_count": len(trace),
            "observation_count": len(payload.get("observations", [])),
            "evidence_count": len(payload.get("evidence", {})),
            "summary_preview": str(payload.get("summary_preview", ""))[:300],
            "needs_replan": payload.get("needs_replan"),
        }

    if title == "StageExecutor Input":
        return {
            "stage": payload.get("stage"),
            "objective": payload.get("objective"),
            "allowed_tools": payload.get("allowed_tools"),
            "retrieval_focus": payload.get("retrieval_focus"),
            "context_policy": payload.get("context_policy"),
            "target_file_count": len(payload.get("target_files", [])),
            "retrieval_mode": payload.get("retrieval_mode"),
            "retrieval_reason": payload.get("retrieval_reason"),
        }

    if title == "GlobalTaskState Update":
        evidence_store = payload.get("evidence_store", {})
        return {
            "current_stage": payload.get("current_stage"),
            "active_files": payload.get("active_files"),
            "changed_files": payload.get("changed_files"),
            "failures": payload.get("failures"),
            "stage_history": payload.get("stage_history"),
            "stage_summary_count": len(payload.get("stage_summaries", [])),
            "evidence_count": len(evidence_store) if isinstance(evidence_store, dict) else 0,
        }

    return payload


@dataclass
class StagePlan:
    stage: str
    objective: str
    rationale: str
    target_files: list[str] = field(default_factory=list)
    allowed_tools: list[str] = field(default_factory=list)
    retrieval_focus: str = ""
    context_policy: str = ""
    success_criteria: list[str] = field(default_factory=list)
    exit_conditions: list[str] = field(default_factory=list)
    max_tool_steps: int = 8


@dataclass
class ExecutionTraceStep:
    tool_name: str
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_result: str = ""


@dataclass
class StageSummary:
    stage: str
    text: str


@dataclass
class ExecutionResult:
    stage: str
    success: bool
    stage_summary: str
    changed_files: list[str] = field(default_factory=list)
    trace: list[ExecutionTraceStep] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)
    state_patch: dict[str, Any] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)
    needs_replan: bool = False


@dataclass
class GlobalTaskState:
    user_request: str
    task_type: str | None = None
    current_stage: str | None = None
    stage_history: list[StagePlan] = field(default_factory=list)
    execution_history: list[ExecutionResult] = field(default_factory=list)
    stage_summaries: list[StageSummary] = field(default_factory=list)
    evidence_store: dict[str, Any] = field(default_factory=dict)
    working_memory: dict[str, Any] = field(default_factory=dict)
    active_files: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    done: bool = False
    final_answer: str | None = None
    session_state: SessionState = field(default_factory=SessionState)


@dataclass
class ThinkDecision:
    type: str
    stage_plan: StagePlan | None = None
    answer: str | None = None
    reason: str = ""


@dataclass
class RuntimeEvaluation:
    done: bool = False
    final_answer: str | None = None
    needs_replan: bool = False
    reason: str = ""
    verification: Any | None = None


@dataclass
class LocalExecutionTrace:
    output: str
    tool_steps: int
    trace_steps: list[ExecutionTraceStep] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class CompletionAssessment:
    objective_satisfied: bool = False
    can_answer_now: bool = False
    missing_repository_understanding: bool = False
    missing_target_file: bool = False
    missing_implementation_detail: bool = False
    modification_required: bool = False
    validation_missing: bool = False
    requires_recovery: bool = False
    evidence: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class RetrievalDecision:
    should_retrieve: bool
    reason: str
    focus_signature: str = ""
    reuse_cache: bool = False


@dataclass
class OverviewBudget:
    max_files: int = 5


class AnswerComposer:
    """Compose a user-facing final answer from evidence, not from raw stage traces."""

    @staticmethod
    def compose(state: GlobalTaskState) -> str:
        task_type = state.task_type or "analysis"
        evidence = state.evidence_store
        summaries = [summary.text for summary in state.stage_summaries]

        if task_type in {"code_change", "bug_fix", "feature_change", "refactor"}:
            return AnswerComposer._compose_code_change(evidence, summaries)
        if evidence.get("implementation_notes"):
            return AnswerComposer._compose_explanation(evidence, summaries)
        return AnswerComposer._compose_overview(evidence, summaries)

    @staticmethod
    def _compose_overview(evidence: dict[str, Any], summaries: list[str]) -> str:
        overview = str(evidence.get("overview", "")).strip()
        entry_points = list(evidence.get("entry_points", []))
        core_modules = list(evidence.get("core_modules", []))
        target_files = list(evidence.get("target_files", []))
        risks = list(evidence.get("risks", []))

        lines: list[str] = []
        if overview:
            lines.append(overview)
        elif summaries:
            lines.append(summaries[-1])

        if entry_points:
            lines.append(f"Entry points: {', '.join(entry_points[:5])}")
        if core_modules:
            lines.append(f"Core modules: {', '.join(core_modules[:6])}")
        elif target_files:
            lines.append(f"Key files: {', '.join(target_files[:6])}")
        if risks:
            lines.append(f"Open risks: {'; '.join(str(risk) for risk in risks[:3])}")

        return "\n".join(line for line in lines if line).strip() or "No overview could be composed from the current evidence."

    @staticmethod
    def _compose_explanation(evidence: dict[str, Any], summaries: list[str]) -> str:
        overview = str(evidence.get("overview", "")).strip()
        implementation_notes = list(evidence.get("implementation_notes", []))
        target_files = list(evidence.get("target_files", []))
        observations = list(evidence.get("observations", []))
        risks = list(evidence.get("risks", []))

        lines: list[str] = []
        if overview:
            lines.append(overview)
        elif summaries:
            lines.append(summaries[-1])

        if implementation_notes:
            lines.append("Implementation details:")
            lines.extend(f"- {note}" for note in implementation_notes[:4])
        elif observations:
            lines.append("Key findings:")
            lines.extend(f"- {note}" for note in observations[:4])

        if target_files:
            lines.append(f"Relevant files: {', '.join(target_files[:6])}")
        if risks:
            lines.append(f"Open risks: {'; '.join(str(risk) for risk in risks[:3])}")

        return "\n".join(line for line in lines if line).strip() or "No explanation could be composed from the current evidence."

    @staticmethod
    def _compose_code_change(evidence: dict[str, Any], summaries: list[str]) -> str:
        changed_files = list(evidence.get("changed_files", []))
        verification = str(evidence.get("verification", "")).strip()
        modification_notes = list(evidence.get("modification_notes", []))
        risks = list(evidence.get("risks", []))

        lines: list[str] = []
        if modification_notes:
            lines.append("Completed the requested code change.")
            lines.extend(f"- {note}" for note in modification_notes[:3])
        elif summaries:
            lines.append(summaries[-1])

        if changed_files:
            lines.append(f"Changed files: {', '.join(changed_files[:8])}")
        if verification:
            lines.append(f"Verification: {verification}")
        if risks:
            lines.append(f"Open risks: {'; '.join(str(risk) for risk in risks[:3])}")

        return "\n".join(line for line in lines if line).strip() or "No change summary could be composed from the current evidence."


class MaxToolStepsExceededError(RuntimeError):
    """Raised when a stage-local tool loop exceeds its budget."""


class ToolNotAllowedError(RuntimeError):
    """Raised when the inner ReAct loop calls a tool outside the stage contract."""


class ThinkEngine:
    """Produces the next StagePlan from state, evidence gaps, and completion criteria."""

    def __init__(self):
        self._understanding = TaskUnderstandingAnalyzer()

    def think(self, state: GlobalTaskState) -> ThinkDecision:
        understanding = self._understanding.understand(goal=state.user_request)
        retrieval_family = self._understanding.infer_retrieval_family(understanding)
        task_type = state.task_type or self._infer_task_type(state.user_request)
        state.task_type = task_type

        assessment = self._assess_completion(
            state=state,
            task_type=task_type,
            retrieval_family=str(retrieval_family.value),
        )

        if assessment.objective_satisfied and assessment.can_answer_now:
            decision = ThinkDecision(
                type="final_answer",
                reason=assessment.reason or "current evidence is sufficient to answer",
            )
            self._debug_decision(state, decision, assessment)
            return decision

        next_stage = self._choose_next_stage(assessment, task_type)
        decision = ThinkDecision(
            type="stage_plan",
            stage_plan=self._build_stage_plan(next_stage, state, task_type),
            reason=assessment.reason or f"missing evidence requires `{next_stage}`",
        )
        self._debug_decision(state, decision, assessment)
        return decision

    def _debug_decision(
        self,
        state: GlobalTaskState,
        decision: ThinkDecision,
        assessment: CompletionAssessment | None = None,
    ) -> None:
        _stage_debug(
            "ThinkEngine Decision",
            {
                "user_request": state.user_request,
                "task_type": state.task_type,
                "completed_stages": [result.stage for result in state.execution_history if result.success],
                "current_stage": state.current_stage,
                "recent_failures": state.failures[-3:],
                "completion_assessment": assessment,
                "decision_type": decision.type,
                "decision_reason": decision.reason,
                "stage_plan": decision.stage_plan,
            },
        )

    def _build_stage_plan(self, stage: str, state: GlobalTaskState, task_type: str) -> StagePlan:
        target_files = list(dict.fromkeys(state.active_files + state.changed_files))[:8]
        replan_reason = str(state.working_memory.get("replan_reason", "")).strip()

        if stage == "understand":
            return StagePlan(
                stage=stage,
                objective="Build enough repository understanding to ground the task.",
                rationale="The current state does not yet contain enough high-level evidence to proceed confidently.",
                target_files=target_files,
                allowed_tools=DEFAULT_STAGE_TOOLSETS[stage],
                retrieval_focus="project overview, entry points, architecture summaries",
                context_policy=STAGE_CONTEXT_POLICIES[stage],
                success_criteria=[
                    "identify the main project goal or capability",
                    "find entry points or core modules",
                    "produce a concise project structure summary",
                ],
                exit_conditions=["overview established", "core modules identified"],
            )
        if stage == "locate":
            return StagePlan(
                stage=stage,
                objective="Ground the task in the most relevant files, symbols, and paths.",
                rationale="We need a narrower working set before deeper analysis or modification.",
                target_files=target_files,
                allowed_tools=DEFAULT_STAGE_TOOLSETS[stage],
                retrieval_focus="symbols, file paths, dependency neighborhoods related to the request",
                context_policy=STAGE_CONTEXT_POLICIES[stage],
                success_criteria=[
                    "identify candidate files or modules",
                    "locate key classes, functions, or config entry points",
                    "record the working set for follow-up analysis",
                ],
                exit_conditions=["identified target files", "located key symbols"],
            )
        if stage == "analyze":
            return StagePlan(
                stage=stage,
                objective="Understand implementation details, local call flow, and modification boundaries.",
                rationale="The task still lacks enough implementation detail for a confident answer or change.",
                target_files=target_files,
                allowed_tools=DEFAULT_STAGE_TOOLSETS[stage],
                retrieval_focus="targeted snippets, call graph, local dependencies, constraints",
                context_policy=STAGE_CONTEXT_POLICIES[stage],
                success_criteria=[
                    "understand the key implementation path",
                    "capture impact boundaries and constraints",
                    "produce enough evidence for the next decision",
                ],
                exit_conditions=["implementation understood", "constraints captured"],
            )
        if stage == "modify":
            return StagePlan(
                stage=stage,
                objective="Apply the required code changes within a bounded set of files.",
                rationale="The state indicates a concrete modification is still required.",
                target_files=target_files,
                allowed_tools=DEFAULT_STAGE_TOOLSETS[stage],
                retrieval_focus="exact files to edit, editable context, adjacent dependencies",
                context_policy=STAGE_CONTEXT_POLICIES[stage],
                success_criteria=[
                    "apply the requested change",
                    "record modified files and meaningful deltas",
                    "avoid unrelated edits",
                ],
                exit_conditions=["requested edits applied", "changed files recorded"],
                max_tool_steps=10,
            )
        if stage == "verify":
            return StagePlan(
                stage=stage,
                objective="Validate changes and gather concise verification evidence.",
                rationale="The requested changes exist, but we still need proof that they hold.",
                target_files=target_files,
                allowed_tools=DEFAULT_STAGE_TOOLSETS[stage],
                retrieval_focus="changed files, test logs, verification artifacts",
                context_policy=STAGE_CONTEXT_POLICIES[stage],
                success_criteria=[
                    "confirm the changed file set",
                    "run or collect the minimal necessary verification evidence",
                    "decide whether the task is ready to finish",
                ],
                exit_conditions=["verification completed", "success or actionable failure captured"],
                max_tool_steps=6,
            )
        if stage == "recover":
            return StagePlan(
                stage=stage,
                objective="Recover from the last failed attempt and find a smaller viable next move.",
                rationale=replan_reason or "The previous attempt failed or requested replanning.",
                target_files=target_files,
                allowed_tools=DEFAULT_STAGE_TOOLSETS[stage],
                retrieval_focus="failure-oriented retrieval, alternative files, wider dependency search",
                context_policy=STAGE_CONTEXT_POLICIES[stage],
                success_criteria=[
                    "identify the failure cause",
                    "propose an alternate approach or broader search",
                    "re-establish the next bounded working set",
                ],
                exit_conditions=["recovery hypothesis formed", "next-stage targets updated"],
            )
        return StagePlan(
            stage="finalize",
            objective="Produce a concise final answer grounded in the evidence collected so far.",
            rationale="The state indicates that the current evidence is sufficient to answer the user directly.",
            target_files=target_files,
            allowed_tools=[],
            retrieval_focus="final summaries only",
            context_policy=STAGE_CONTEXT_POLICIES["finalize"],
            success_criteria=[
                "summarize completed work or findings",
                "mention touched or relevant files when useful",
                "note residual risks or missing verification if any",
            ],
            exit_conditions=["final answer written"],
            max_tool_steps=1,
        )

    def _assess_completion(
        self,
        state: GlobalTaskState,
        task_type: str,
        retrieval_family: str,
    ) -> CompletionAssessment:
        successful = [result for result in state.execution_history if result.success]
        last = state.execution_history[-1] if state.execution_history else None
        last_summary = state.stage_summaries[-1].text.strip() if state.stage_summaries else ""
        stage_summaries = state.stage_summaries
        repo_summary = state.session_state.repo_summary.strip()
        evidence_store = state.evidence_store

        has_repo_understanding = bool(
            repo_summary
            or any(result.stage == "understand" for result in successful)
            or bool(evidence_store.get("overview"))
        )
        has_locate = any(result.stage == "locate" for result in successful)
        has_grounded_files = bool(state.active_files or has_locate or evidence_store.get("target_files"))
        has_analysis = any(result.stage == "analyze" for result in successful) or bool(evidence_store.get("implementation_notes"))
        has_modification = bool(state.changed_files) or any(result.stage == "modify" for result in successful)
        has_verification = any(result.stage == "verify" for result in successful) or bool(evidence_store.get("verification"))

        is_change_task = task_type in {"code_change", "bug_fix", "feature_change", "refactor"}
        is_navigation = retrieval_family == "navigation"
        is_explanation = retrieval_family == "explanation"
        is_understanding = retrieval_family == "understanding"

        requires_recovery = bool(
            last
            and (
                last.needs_replan
                or not last.success
                or (last.stage == "verify" and state.failures)
            )
            and state.current_stage != "recover"
        )

        assessment = CompletionAssessment(
            requires_recovery=requires_recovery,
            evidence=[
                item
                for item, present in [
                    ("repo_understanding", has_repo_understanding),
                    ("grounded_files", has_grounded_files),
                    ("implementation_detail", has_analysis),
                    ("code_changes", has_modification),
                    ("verification", has_verification),
                ]
                if present
            ],
        )

        if requires_recovery:
            assessment.reason = str(state.working_memory.get("replan_reason", "")).strip() or "previous stage requested recovery"
            return assessment

        if is_change_task:
            assessment.modification_required = not has_modification
            assessment.validation_missing = has_modification and not has_verification
            assessment.missing_repository_understanding = not has_repo_understanding
            assessment.missing_target_file = not has_grounded_files
            assessment.missing_implementation_detail = has_grounded_files and not has_analysis and not has_modification
            assessment.objective_satisfied = has_modification and has_verification
            assessment.can_answer_now = assessment.objective_satisfied
        elif is_understanding:
            assessment.missing_repository_understanding = not has_repo_understanding
            assessment.objective_satisfied = has_repo_understanding and bool(stage_summaries or last_summary or evidence_store.get("overview"))
            assessment.can_answer_now = assessment.objective_satisfied
        elif is_navigation:
            assessment.missing_repository_understanding = not has_repo_understanding
            assessment.missing_target_file = not has_grounded_files
            assessment.objective_satisfied = has_grounded_files
            assessment.can_answer_now = assessment.objective_satisfied
        elif is_explanation:
            assessment.missing_repository_understanding = not has_repo_understanding
            assessment.missing_target_file = not has_grounded_files
            assessment.missing_implementation_detail = has_grounded_files and not has_analysis
            assessment.objective_satisfied = has_analysis
            assessment.can_answer_now = assessment.objective_satisfied
        else:
            assessment.missing_repository_understanding = not has_repo_understanding
            assessment.missing_target_file = not has_grounded_files and bool(repo_summary)
            assessment.missing_implementation_detail = has_grounded_files and not has_analysis and not bool(last_summary)
            assessment.objective_satisfied = bool(last_summary or evidence_store.get("overview")) and (has_repo_understanding or has_grounded_files)
            assessment.can_answer_now = assessment.objective_satisfied

        assessment.missing = [
            name
            for name, present in [
                ("repository_understanding", assessment.missing_repository_understanding),
                ("target_file", assessment.missing_target_file),
                ("implementation_detail", assessment.missing_implementation_detail),
                ("modification", assessment.modification_required),
                ("validation", assessment.validation_missing),
            ]
            if present
        ]

        if assessment.objective_satisfied:
            assessment.reason = "the current state already satisfies the user request"
        elif assessment.missing:
            assessment.reason = f"missing evidence: {', '.join(assessment.missing)}"
        else:
            assessment.reason = "additional bounded work is still required"

        return assessment

    @staticmethod
    def _choose_next_stage(assessment: CompletionAssessment, task_type: str) -> str:
        if assessment.requires_recovery:
            return "recover"
        if assessment.missing_repository_understanding:
            return "understand"
        if assessment.missing_target_file:
            return "locate"
        if assessment.missing_implementation_detail:
            return "analyze"
        if assessment.modification_required:
            return "modify"
        if assessment.validation_missing:
            return "verify"
        if task_type in {"code_change", "bug_fix", "feature_change", "refactor"}:
            return "verify"
        return "finalize"

    @staticmethod
    def _infer_task_type(user_request: str) -> str:
        text = user_request.lower()
        if any(token in text for token in ("fix", "bug", "repair", "debug", "淇", "鎶ラ敊", "閿欒")):
            return "bug_fix"
        if any(token in text for token in ("modify", "change", "implement", "add", "update", "edit", "淇敼", "瀹炵幇", "鏂板")):
            return "code_change"
        if any(token in text for token in ("refactor", "restructure", "cleanup", "閲嶆瀯")):
            return "refactor"
        return "analysis"
class LocalReactExecutor:
    """Executes the existing ReAct loop inside a single stage boundary."""

    def __init__(self, chat_fn: ChatFn, agent_owner: Agent | None = None):
        self._chat_fn = chat_fn
        self._agent_owner = agent_owner

    async def execute(
        self,
        user_message: str,
        state_updates: dict[str, Any] | None = None,
        allowed_tools: list[str] | None = None,
        max_tool_steps: int = 8,
        on_token: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict], None] | None = None,
        on_event: RuntimeEventFn | None = None,
    ) -> LocalExecutionTrace:
        allowed = self._expand_allowed_tools(allowed_tools or [])
        trace_steps: list[ExecutionTraceStep] = []
        tool_steps = [0]
        before_history_len = len(self._agent_owner.state.persistent_history) if self._agent_owner else 0
        before_checkpoint_len = len(self._agent_owner._checkpoints) if self._agent_owner else 0
        if on_event is not None:
            on_event({
                "type": "react_loop_start",
                "allowed_tools": sorted(allowed),
                "max_tool_steps": max_tool_steps,
            })

        def _on_tool(name: str, kwargs: dict) -> None:
            if allowed and name not in allowed:
                raise ToolNotAllowedError(f"Tool '{name}' is not allowed in this stage")
            tool_steps[0] += 1
            if tool_steps[0] > max_tool_steps:
                raise MaxToolStepsExceededError(
                    f"Stage exceeded max_tool_steps ({max_tool_steps})"
                )
            try:
                args_preview = json.dumps(kwargs, ensure_ascii=False, sort_keys=True)
            except TypeError:
                args_preview = str(kwargs)
            del args_preview
            trace_steps.append(
                ExecutionTraceStep(
                    tool_name=name,
                    tool_args=dict(kwargs),
                )
            )
            if on_tool:
                on_tool(name, kwargs)

        try:
            with self._restrict_agent_tools(allowed):
                output = await self._chat_fn(
                    user_message,
                    state_updates=state_updates or {},
                    on_token=on_token,
                    on_tool=_on_tool,
                )
            transcript = self._collect_transcript(before_history_len)
            if not trace_steps:
                trace_steps = self._backfill_trace_steps_from_transcript(transcript)
            self._attach_tool_results(trace_steps, transcript)
            if on_event is not None:
                on_event({
                    "type": "react_loop_complete",
                    "tool_steps": tool_steps[0],
                    "trace_count": len(trace_steps),
                })
            return LocalExecutionTrace(
                output=output,
                tool_steps=tool_steps[0],
                trace_steps=trace_steps,
                transcript=transcript,
            )
        finally:
            self._restore_history(before_history_len)
            self._restore_checkpoints(before_checkpoint_len)

    def _collect_transcript(self, before_history_len: int) -> list[dict[str, Any]]:
        if not self._agent_owner:
            return []
        return [
            dict(msg)
            for msg in self._agent_owner.state.persistent_history[before_history_len:]
        ]

    def _restore_history(self, before_history_len: int) -> None:
        if not self._agent_owner:
            return
        self._agent_owner.state.persistent_history = self._agent_owner.state.persistent_history[:before_history_len]

    def _restore_checkpoints(self, before_checkpoint_len: int) -> None:
        if not self._agent_owner:
            return
        self._agent_owner._checkpoints = self._agent_owner._checkpoints[:before_checkpoint_len]

    @staticmethod
    def _attach_tool_results(trace_steps: list[ExecutionTraceStep], transcript: list[dict[str, Any]]) -> None:
        tool_results = [
            str(msg.get("content", ""))
            for msg in transcript
            if msg.get("role") == "tool"
        ]
        for step, result in zip(trace_steps, tool_results):
            step.tool_result = result

    @staticmethod
    def _backfill_trace_steps_from_transcript(transcript: list[dict[str, Any]]) -> list[ExecutionTraceStep]:
        trace_steps: list[ExecutionTraceStep] = []
        for message in transcript:
            if message.get("role") != "assistant":
                continue
            for tool_call in message.get("tool_calls", []) or []:
                function = tool_call.get("function", {})
                name = function.get("name", "")
                args = function.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"raw": args}
                trace_steps.append(
                    ExecutionTraceStep(
                        tool_name=name,
                        tool_args=dict(args) if isinstance(args, dict) else {"raw": str(args)},
                    )
                )
        return trace_steps

    @contextmanager
    def _restrict_agent_tools(self, allowed: set[str]):
        if not self._agent_owner or not allowed:
            yield
            return

        previous_tools = list(self._agent_owner.tools)
        previous_system = self._agent_owner._system
        filtered_tools = [tool for tool in previous_tools if tool.name in allowed]
        self._agent_owner.tools = filtered_tools
        self._agent_owner._system = system_prompt(filtered_tools)
        try:
            yield
        finally:
            self._agent_owner.tools = previous_tools
            self._agent_owner._system = previous_system

    @staticmethod
    def _expand_allowed_tools(allowed_tools: list[str]) -> set[str]:
        if not allowed_tools:
            return set()
        expanded: set[str] = set()
        for name in allowed_tools:
            normalized = name.strip().lower()
            expanded.update(TOOL_ALIASES.get(normalized, {normalized}))
        return expanded


class StageExecutor:
    """Runs a single StagePlan and returns a compact stage-scoped result."""

    def __init__(
        self,
        agent: Agent | None = None,
        chat_fn: ChatFn | None = None,
        context_orchestrator: ContextOrchestrator | None = None,
        working_dir: str = ".",
    ):
        if agent is None and chat_fn is None:
            raise ValueError("StageExecutor requires either an agent or a chat_fn")
        self._agent = agent
        self._chat_fn = chat_fn or agent.chat  # type: ignore[union-attr]
        self._context_orchestrator = context_orchestrator
        self._working_dir = working_dir
        self._local_executor = LocalReactExecutor(self._chat_fn, agent_owner=agent)

    async def execute(
        self,
        stage_plan: StagePlan,
        global_state: GlobalTaskState,
        on_token: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict], None] | None = None,
        on_event: RuntimeEventFn | None = None,
    ) -> ExecutionResult:
        user_message, state_updates = self._build_stage_context(stage_plan, global_state)
        session_state_updates = self._summarize_state_updates(stage_plan, state_updates)
        retrieval_meta = session_state_updates.get("retrieval_cache", {}).get("latest", {})
        self._emit_event(
            on_event,
            {
                "type": "execute_start",
                "stage": stage_plan.stage,
                "objective": stage_plan.objective,
                "allowed_tools": list(stage_plan.allowed_tools),
                "retrieval_mode": "fresh" if retrieval_meta.get("used_retrieval") else "cached",
                "retrieval_reason": retrieval_meta.get("reason", ""),
                "target_files": list(stage_plan.target_files[:8]),
            },
        )
        _stage_debug(
            "StageExecutor Input",
            {
                "stage": stage_plan.stage,
                "objective": stage_plan.objective,
                "allowed_tools": stage_plan.allowed_tools,
                "retrieval_focus": stage_plan.retrieval_focus,
                "context_policy": stage_plan.context_policy,
                "target_files": stage_plan.target_files,
                "retrieval_mode": "fresh" if retrieval_meta.get("used_retrieval") else "cached",
                "retrieval_reason": retrieval_meta.get("reason", ""),
            },
        )

        try:
            trace = await self._local_executor.execute(
                user_message=user_message,
                state_updates=state_updates,
                allowed_tools=stage_plan.allowed_tools,
                max_tool_steps=stage_plan.max_tool_steps,
                on_token=on_token,
                on_tool=on_tool,
                on_event=on_event,
            )
            changed_files = self._collect_changed_files()
            tool_failures = self._collect_failures(trace.trace_steps)
            needs_replan = bool(tool_failures)
            observations = self._derive_observations(stage_plan, trace, changed_files, session_state_updates)
            evidence = self._derive_evidence(stage_plan, observations, changed_files, session_state_updates, trace)
            summary = self._summarize_stage_output(stage_plan, observations, evidence)
            trace_preview = self._compress_trace(trace.trace_steps)
            _stage_debug(
                "StageExecutor Output",
                {
                    "stage": stage_plan.stage,
                    "success": not needs_replan,
                    "trace": trace_preview,
                    "observations": observations,
                    "evidence": evidence,
                    "summary_preview": summary[:1500],
                    "needs_replan": needs_replan,
                },
            )
            self._emit_event(
                on_event,
                {
                    "type": "execute_complete",
                    "stage": stage_plan.stage,
                    "success": not needs_replan,
                    "tool_count": len(trace_preview),
                    "observation_count": len(observations),
                    "evidence_count": len(evidence),
                    "needs_replan": needs_replan,
                    "summary_preview": summary[:300],
                },
            )
            return ExecutionResult(
                stage=stage_plan.stage,
                success=not needs_replan,
                stage_summary=summary,
                changed_files=changed_files,
                trace=trace_preview,
                observations=observations,
                evidence=evidence,
                state_patch=session_state_updates,
                failures=tool_failures,
                needs_replan=needs_replan,
            )
        except (MaxToolStepsExceededError, ToolNotAllowedError) as exc:
            self._emit_event(
                on_event,
                {
                    "type": "execute_complete",
                    "stage": stage_plan.stage,
                    "success": False,
                    "tool_count": 0,
                    "observation_count": 0,
                    "evidence_count": 1,
                    "needs_replan": True,
                    "summary_preview": str(exc)[:300],
                },
            )
            return ExecutionResult(
                stage=stage_plan.stage,
                success=False,
                stage_summary=f"{stage_plan.stage} stage failed: {exc}",
                changed_files=self._collect_changed_files(),
                observations=[],
                evidence={"failure_kind": type(exc).__name__},
                state_patch=session_state_updates,
                failures=[str(exc)],
                needs_replan=True,
            )
        except Exception as exc:
            self._emit_event(
                on_event,
                {
                    "type": "execute_complete",
                    "stage": stage_plan.stage,
                    "success": False,
                    "tool_count": 0,
                    "observation_count": 0,
                    "evidence_count": 1,
                    "needs_replan": True,
                    "summary_preview": f"{type(exc).__name__}: {exc}"[:300],
                },
            )
            return ExecutionResult(
                stage=stage_plan.stage,
                success=False,
                stage_summary=f"{stage_plan.stage} stage failed: {type(exc).__name__}: {exc}",
                changed_files=self._collect_changed_files(),
                observations=[],
                evidence={"failure_kind": type(exc).__name__},
                state_patch=session_state_updates,
                failures=[f"{type(exc).__name__}: {exc}"],
                needs_replan=True,
            )

    @staticmethod
    def _emit_event(on_event: RuntimeEventFn | None, payload: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(payload)

    def _build_stage_context(
        self,
        stage_plan: StagePlan,
        global_state: GlobalTaskState,
    ) -> tuple[str, dict[str, Any]]:
        retrieval_decision = self._decide_retrieval(stage_plan, global_state)
        if self._context_orchestrator is not None:
            if retrieval_decision.should_retrieve:
                assembly = self._context_orchestrator.build_for_stage(
                    stage=stage_plan.stage,
                    objective=stage_plan.objective,
                    target_files=stage_plan.target_files,
                    retrieval_focus=stage_plan.retrieval_focus,
                    global_state=global_state,
                    context_policy=stage_plan.context_policy,
                    allowed_tools=stage_plan.allowed_tools,
                    stage_plan=stage_plan,
                )
                user_message = assembly.user_message.strip() or global_state.user_request
                user_message += self._stage_instruction_suffix(stage_plan)
                state_updates = dict(assembly.state_updates)
                state_updates["retrieval_cache"] = self._build_retrieval_cache(
                    stage_plan,
                    retrieval_decision,
                    state_updates,
                )
                return user_message, state_updates
            return self._build_cached_stage_context(stage_plan, global_state, retrieval_decision)

        updates = {
            "current_goal": global_state.user_request,
            "current_task": stage_plan.objective,
            "active_files": list(stage_plan.target_files or global_state.active_files),
            "constraints": list(global_state.working_memory.get("constraints", [])),
            "failures": list(global_state.failures[-5:]),
            "allowed_actions": list(stage_plan.allowed_tools),
            "stop_conditions": "; ".join(stage_plan.exit_conditions),
            "retrieval_cache": self._build_retrieval_cache(
                stage_plan,
                retrieval_decision,
                {"active_files": list(stage_plan.target_files or global_state.active_files)},
            ),
        }
        user_message = (
            f"## Stage: {stage_plan.stage}\n"
            f"## Objective\n{stage_plan.objective}\n\n"
            f"## User Request\n{global_state.user_request}"
            f"{self._stage_instruction_suffix(stage_plan)}"
        )
        return user_message, updates

    def _build_cached_stage_context(
        self,
        stage_plan: StagePlan,
        global_state: GlobalTaskState,
        retrieval_decision: RetrievalDecision,
    ) -> tuple[str, dict[str, Any]]:
        cached = global_state.session_state.retrieval_cache.get("latest", {})
        active_files = list(
            dict.fromkeys(
                list(stage_plan.target_files)
                + list(global_state.active_files)
                + list(cached.get("active_files", []))
            )
        )[:12]
        state_updates = {
            "repo_summary": global_state.session_state.repo_summary or cached.get("repo_summary", ""),
            "active_files": active_files,
            "active_symbols": list(global_state.session_state.active_symbols),
            "current_goal": global_state.user_request,
            "current_task": stage_plan.objective,
            "constraints": list(global_state.working_memory.get("constraints", [])),
            "failures": list(global_state.failures[-5:]),
            "allowed_actions": list(stage_plan.allowed_tools),
            "stop_conditions": "; ".join(stage_plan.exit_conditions),
            "retrieval_cache": self._build_retrieval_cache(
                stage_plan,
                retrieval_decision,
                {
                    "repo_summary": global_state.session_state.repo_summary or cached.get("repo_summary", ""),
                    "active_files": active_files,
                    "active_symbols": list(global_state.session_state.active_symbols),
                },
            ),
        }
        user_message = (
            f"## Stage: {stage_plan.stage}\n"
            f"## Objective\n{stage_plan.objective}\n\n"
            f"## Goal\n{global_state.user_request}\n\n"
            f"## Retrieval Decision\n{retrieval_decision.reason}\n"
            f"{self._stage_instruction_suffix(stage_plan)}"
        )
        return user_message, state_updates

    @staticmethod
    def _stage_instruction_suffix(stage_plan: StagePlan) -> str:
        success = "\n".join(f"- {item}" for item in stage_plan.success_criteria)
        exit_conditions = "\n".join(f"- {item}" for item in stage_plan.exit_conditions)
        return (
            "\n\n## Stage Contract\n"
            f"Rationale: {stage_plan.rationale}\n"
            f"Allowed tools: {', '.join(stage_plan.allowed_tools) or 'none'}\n"
            f"Success criteria:\n{success or '- complete the stage objective'}\n"
            f"Exit conditions:\n{exit_conditions or '- stop when objective is satisfied'}\n"
            "Work only within this stage. Do not skip ahead to unrelated tasks."
        )

    def _collect_changed_files(self) -> list[str]:
        if self._agent is None:
            return []
        try:
            return list(self._agent.changed_files)
        except Exception:
            return []

    def _decide_retrieval(
        self,
        stage_plan: StagePlan,
        global_state: GlobalTaskState,
    ) -> RetrievalDecision:
        focus_signature = self._focus_signature(stage_plan, global_state)
        cache = global_state.session_state.retrieval_cache.get("latest", {})
        last_focus = str(cache.get("focus_signature", ""))
        repo_summary = global_state.session_state.repo_summary.strip()
        active_files = list(global_state.active_files)
        recent_failures = list(global_state.failures[-3:])
        has_stage_summary = bool(global_state.stage_summaries)

        if stage_plan.stage == "finalize":
            return RetrievalDecision(
                should_retrieve=False,
                reason="Finalization should reuse the evidence already gathered.",
                focus_signature=focus_signature,
                reuse_cache=True,
            )

        if stage_plan.stage == "recover":
            return RetrievalDecision(
                should_retrieve=True,
                reason="Recovery needs a fresh retrieval pass because the previous approach failed.",
                focus_signature=focus_signature,
            )

        if stage_plan.stage == "verify":
            return RetrievalDecision(
                should_retrieve=False,
                reason="Verification should prioritize changed files and existing evidence instead of re-retrieving the repo.",
                focus_signature=focus_signature,
                reuse_cache=True,
            )

        if stage_plan.stage == "modify" and (stage_plan.target_files or global_state.changed_files):
            return RetrievalDecision(
                should_retrieve=False,
                reason="Modification already has a bounded file set, so the runtime should reuse the existing working set.",
                focus_signature=focus_signature,
                reuse_cache=True,
            )

        if not repo_summary:
            return RetrievalDecision(
                should_retrieve=True,
                reason="No repository cognition is cached yet.",
                focus_signature=focus_signature,
            )

        if recent_failures and stage_plan.stage in {"understand", "locate", "analyze"}:
            return RetrievalDecision(
                should_retrieve=True,
                reason="Recent failures suggest the current evidence may be stale or insufficient.",
                focus_signature=focus_signature,
            )

        if stage_plan.stage == "understand" and has_stage_summary:
            return RetrievalDecision(
                should_retrieve=False,
                reason="A repository-level summary already exists, so we can reuse it for understanding.",
                focus_signature=focus_signature,
                reuse_cache=True,
            )

        if stage_plan.stage in {"locate", "analyze"} and active_files and focus_signature == last_focus:
            return RetrievalDecision(
                should_retrieve=False,
                reason="The retrieval focus and active working set are stable, so we can reuse cached repo cognition.",
                focus_signature=focus_signature,
                reuse_cache=True,
            )

        if stage_plan.stage in {"locate", "analyze"} and not active_files:
            return RetrievalDecision(
                should_retrieve=True,
                reason="The runtime still lacks grounded target files for this stage.",
                focus_signature=focus_signature,
            )

        return RetrievalDecision(
            should_retrieve=True,
            reason="This stage still needs fresh repository evidence.",
            focus_signature=focus_signature,
        )

    @staticmethod
    def _focus_signature(stage_plan: StagePlan, global_state: GlobalTaskState) -> str:
        focus_parts = [
            stage_plan.stage,
            stage_plan.retrieval_focus,
            ",".join(sorted(stage_plan.target_files)),
            ",".join(sorted(global_state.active_files)),
        ]
        return " | ".join(part for part in focus_parts if part)

    @staticmethod
    def _build_retrieval_cache(
        stage_plan: StagePlan,
        retrieval_decision: RetrievalDecision,
        state_updates: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "latest": {
                "stage": stage_plan.stage,
                "focus_signature": retrieval_decision.focus_signature,
                "reason": retrieval_decision.reason,
                "repo_summary": state_updates.get("repo_summary", ""),
                "active_files": list(state_updates.get("active_files", [])),
                "active_symbols": list(state_updates.get("active_symbols", [])),
                "used_retrieval": retrieval_decision.should_retrieve,
            }
        }

    @staticmethod
    def _collect_failures(trace_steps: list[ExecutionTraceStep]) -> list[str]:
        failures: list[str] = []
        for step in trace_steps:
            lowered = step.tool_result.lower()
            if lowered.startswith("error") or "traceback" in lowered or "tests failed" in lowered:
                preview = next((line.strip() for line in step.tool_result.splitlines() if line.strip()), "")[:240]
                failures.append(f"{step.tool_name}: {preview or 'tool failed'}")
        return failures[:5]

    def _derive_observations(
        self,
        stage_plan: StagePlan,
        trace: LocalExecutionTrace,
        changed_files: list[str],
        session_state_updates: dict[str, Any],
    ) -> list[str]:
        observations: list[str] = []
        output_lines = [
            line.strip("- ").strip()
            for line in trace.output.splitlines()
            if line.strip()
        ]
        for line in output_lines:
            if len(line) < 12:
                continue
            if line not in observations:
                observations.append(line[:220])

        active_files = list(session_state_updates.get("active_files", []))
        if stage_plan.stage == "understand" and active_files:
            observations.append(
                f"The overview focused on {', '.join(active_files[:5])}."
            )
        if stage_plan.stage == "verify" and changed_files:
            observations.append(
                f"Verification considered changed files: {', '.join(changed_files[:5])}."
            )
        if not observations:
            if changed_files:
                observations.append(f"The stage affected {', '.join(changed_files[:5])}.")
            elif trace.trace_steps:
                observations.append(f"The stage gathered evidence using {', '.join(step.tool_name for step in trace.trace_steps[:4])}.")
        return observations[:6]

    def _derive_evidence(
        self,
        stage_plan: StagePlan,
        observations: list[str],
        changed_files: list[str],
        session_state_updates: dict[str, Any],
        trace: LocalExecutionTrace,
    ) -> dict[str, Any]:
        active_files = list(session_state_updates.get("active_files", []))
        evidence: dict[str, Any] = {
            "observations": observations[:4],
            "target_files": active_files[:8] or changed_files[:8],
            "changed_files": changed_files[:8],
        }

        if stage_plan.stage == "understand":
            evidence["overview"] = observations[0] if observations else "Repository overview captured."
            evidence["entry_points"] = [path for path in active_files if path.endswith(("README.md", "pyproject.toml", "cli.py", "main.py", "app.py"))][:5]
            evidence["core_modules"] = [path for path in active_files if path.endswith(".py") and path not in evidence["entry_points"]][:5]
        elif stage_plan.stage == "locate":
            evidence["target_files"] = active_files[:8]
            evidence["located_symbols"] = list(session_state_updates.get("active_symbols", []))[:8]
        elif stage_plan.stage == "analyze":
            evidence["implementation_notes"] = observations[:4]
        elif stage_plan.stage == "modify":
            evidence["changed_files"] = changed_files[:8]
            evidence["modification_notes"] = observations[:4]
        elif stage_plan.stage == "verify":
            evidence["verification"] = observations[0] if observations else "Verification completed."
        elif stage_plan.stage == "recover":
            evidence["risks"] = observations[:3]

        if trace.trace_steps and os.environ.get("DEBUG_FULL_TRACE", "").lower() == "true":
            evidence["debug_trace"] = [
                {
                    "tool_name": step.tool_name,
                    "tool_args": step.tool_args,
                    "tool_result": step.tool_result,
                }
                for step in trace.trace_steps
            ]
        return evidence

    @staticmethod
    def _compress_trace(trace_steps: list[ExecutionTraceStep]) -> list[ExecutionTraceStep]:
        preview: list[ExecutionTraceStep] = []
        for step in trace_steps[:8]:
            first_line = next((line.strip() for line in step.tool_result.splitlines() if line.strip()), "")
            preview.append(
                ExecutionTraceStep(
                    tool_name=step.tool_name,
                    tool_args=dict(step.tool_args),
                    tool_result=first_line[:180],
                )
            )
        return preview

    @staticmethod
    def _summarize_state_updates(stage_plan: StagePlan, state_updates: dict[str, Any]) -> dict[str, Any]:
        keep_keys = {
            "repo_summary",
            "active_files",
            "active_symbols",
            "current_task",
            "current_goal",
            "constraints",
            "failures",
            "allowed_actions",
            "forbidden_actions",
            "stop_conditions",
            "retrieval_cache",
        }
        summarized: dict[str, Any] = {}
        for key, value in state_updates.items():
            if key not in keep_keys:
                continue
            if key == "repo_summary":
                summary_text = str(value)
                if stage_plan.stage == "understand":
                    summary_text = StageExecutor._apply_overview_budget(summary_text)
                summarized[key] = summary_text[:2500]
            elif isinstance(value, list):
                values = list(value)
                if key == "active_files" and stage_plan.stage == "understand":
                    values = StageExecutor._prioritize_overview_files(values)
                summarized[key] = values
            elif isinstance(value, dict):
                summarized[key] = dict(value)
            else:
                summarized[key] = value
        return summarized

    @staticmethod
    def _summarize_stage_output(
        stage_plan: StagePlan,
        observations: list[str],
        evidence: dict[str, Any],
    ) -> str:
        lead = observations[0] if observations else f"{stage_plan.stage} completed."
        evidence_bits: list[str] = []
        for key in ("target_files", "changed_files", "entry_points", "core_modules"):
            value = evidence.get(key)
            if value:
                evidence_bits.append(f"{key}: {', '.join(str(item) for item in value[:4])}")
        if evidence_bits:
            return lead + "\n" + "\n".join(evidence_bits[:3])
        return lead

    @staticmethod
    def _prioritize_overview_files(files: list[str]) -> list[str]:
        budget = OverviewBudget()

        def score(path: str) -> tuple[int, str]:
            normalized = path.replace("\\", "/").lower()
            if normalized.endswith("readme.md"):
                return (0, normalized)
            if normalized.endswith("pyproject.toml"):
                return (1, normalized)
            if normalized.endswith(("cli.py", "main.py", "app.py", "__main__.py")):
                return (2, normalized)
            if normalized.endswith(".py"):
                return (3, normalized)
            if normalized.endswith((".html", ".css", ".js")):
                return (5, normalized)
            return (4, normalized)

        return sorted(dict.fromkeys(files), key=score)[:budget.max_files]

    @staticmethod
    def _apply_overview_budget(repo_summary: str) -> str:
        lines = repo_summary.splitlines()
        kept: list[str] = []
        file_lines: list[str] = []
        in_files = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("## Key Files") or stripped.startswith("## Project Files"):
                in_files = True
                kept.append(line)
                continue
            if in_files and stripped.startswith("## "):
                in_files = False
            if in_files and stripped.startswith("- "):
                file_lines.append(line)
                continue
            if not in_files:
                kept.append(line)
        if file_lines:
            kept.extend(StageExecutor._prioritize_overview_files(file_lines))
        return "\n".join(kept)


class GlobalStateManager:
    """Applies stage results back into the persistent runtime state."""

    def apply_stage_result(
        self,
        state: GlobalTaskState,
        stage_plan: StagePlan,
        execution_result: ExecutionResult,
    ) -> GlobalTaskState:
        session_state_updates = execution_result.state_patch
        if isinstance(session_state_updates, dict) and session_state_updates:
            state.session_state.apply_state_updates(session_state_updates)

        state.current_stage = stage_plan.stage
        state.stage_history.append(stage_plan)
        state.execution_history.append(execution_result)

        state.active_files = list(dict.fromkeys(
            list(session_state_updates.get("active_files", []))
            + stage_plan.target_files
            + execution_result.changed_files
            + state.active_files
        ))[:12]
        state.changed_files = list(dict.fromkeys(state.changed_files + execution_result.changed_files))[:20]

        state.stage_summaries.append(StageSummary(stage=execution_result.stage, text=execution_result.stage_summary))
        state.working_memory["last_observations"] = execution_result.observations[-6:]
        state.working_memory["active_stage_objective"] = stage_plan.objective
        state.working_memory["last_retrieval_decision"] = session_state_updates.get("retrieval_cache", {}).get("latest", {})
        if execution_result.success:
            state.working_memory.pop("replan_reason", None)

        self._merge_evidence(state.evidence_store, execution_result.evidence)

        if execution_result.success:
            state.session_state.record_memory(execution_result.stage_summary)
        else:
            for failure in execution_result.failures:
                if failure not in state.failures:
                    state.failures.append(failure)
                state.session_state.record_failure(failure)

        state.session_state.active_files = list(state.active_files)
        state.session_state.current_task = stage_plan.objective
        state.session_state.current_goal = state.user_request
        _stage_debug(
            "GlobalTaskState Update",
            {
                "current_stage": state.current_stage,
                "active_files": state.active_files,
                "changed_files": state.changed_files,
                "failures": state.failures[-5:],
                "stage_summaries": [summary.text for summary in state.stage_summaries[-3:]],
                "evidence_store": state.evidence_store,
                "stage_history": [plan.stage for plan in state.stage_history],
            },
        )
        return state

    @staticmethod
    def _merge_evidence(store: dict[str, Any], evidence: dict[str, Any]) -> None:
        for key, value in evidence.items():
            if not value:
                continue
            if isinstance(value, list):
                existing = list(store.get(key, []))
                store[key] = list(dict.fromkeys(existing + value))[:12]
            elif isinstance(value, dict):
                existing = dict(store.get(key, {}))
                existing.update(value)
                store[key] = existing
            else:
                store[key] = value


class StageEvaluator:
    """Evaluates stage results and decides whether to finish or replan."""

    def __init__(self, agent: Agent | None = None, working_dir: str = "."):
        self._agent = agent
        self._working_dir = working_dir
        self._verifier = VerificationPolicyEngine()

    def evaluate(
        self,
        state: GlobalTaskState,
        execution_result: ExecutionResult,
    ) -> RuntimeEvaluation:
        if not execution_result.success:
            _stage_debug(
                "Stage Evaluation",
                {
                    "stage": execution_result.stage,
                    "success": False,
                    "needs_replan": execution_result.needs_replan or True,
                    "failures": execution_result.failures,
                },
            )
            return RuntimeEvaluation(
                needs_replan=execution_result.needs_replan or True,
                reason="; ".join(execution_result.failures) or execution_result.stage_summary,
            )

        if execution_result.stage == "verify":
            patch = self._patch_analysis()
            dag_result = DagExecutionResult(
                success=execution_result.success,
                output=execution_result.stage_summary,
                error="; ".join(execution_result.failures),
                tool_calls_made=len(execution_result.trace),
                artifacts=execution_result.evidence,
            )
            verification = self._verifier.verify(
                dag_result,
                patch=patch,
                working_dir=self._working_dir,
                task_meta={"title": execution_result.stage},
            )
            if not verification.passed:
                _stage_debug(
                    "Stage Evaluation",
                    {
                        "stage": execution_result.stage,
                        "success": False,
                        "needs_replan": True,
                        "verification_failures": verification.failures,
                        "replan_hint": verification.replan_hint,
                    },
                )
                return RuntimeEvaluation(
                    needs_replan=True,
                    reason="; ".join(verification.failures) or verification.replan_hint or "verification failed",
                    verification=verification,
                )

        _stage_debug(
            "Stage Evaluation",
            {
                "stage": execution_result.stage,
                "success": True,
                "done": False,
                "needs_replan": False,
            },
        )
        return RuntimeEvaluation()

    def _patch_analysis(self) -> PatchAnalysis | None:
        if self._agent is None:
            return None
        try:
            return PatchAnalysis.from_shadow(self._agent.shadow, working_dir=self._working_dir)
        except Exception:
            return None


class AgentRuntime:
    """Outer Think-Execute runtime that wraps the existing inner ReAct loop."""

    def __init__(
        self,
        think_engine: ThinkEngine,
        stage_executor: StageExecutor,
        state_manager: GlobalStateManager | None = None,
        evaluator: StageEvaluator | None = None,
        max_stages: int = 8,
    ):
        self.think_engine = think_engine
        self.stage_executor = stage_executor
        self.state_manager = state_manager or GlobalStateManager()
        self.evaluator = evaluator or StageEvaluator()
        self.max_stages = max_stages
        self.last_state: GlobalTaskState | None = None

    def initialize_state(self, user_request: str) -> GlobalTaskState:
        return GlobalTaskState(user_request=user_request)

    def exceed_limits(self, state: GlobalTaskState) -> bool:
        return len(state.stage_history) >= self.max_stages

    async def run_state(
        self,
        user_request: str,
        on_token: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict], None] | None = None,
        on_event: RuntimeEventFn | None = None,
    ) -> GlobalTaskState:
        state = self.initialize_state(user_request)

        while not state.done and not self.exceed_limits(state):
            self._emit_event(on_event, {
                "type": "think_start",
                "current_stage": state.current_stage,
                "completed_stages": [plan.stage for plan in state.stage_history],
                "failures": list(state.failures[-3:]),
            })
            decision = self.think_engine.think(state)
            self._emit_event(on_event, {
                "type": "think_complete",
                "decision_type": decision.type,
                "reason": decision.reason,
                "stage": decision.stage_plan.stage if decision.stage_plan else None,
                "objective": decision.stage_plan.objective if decision.stage_plan else None,
            })
            if decision.type in {"final", "final_answer"}:
                state.done = True
                state.final_answer = decision.answer or AnswerComposer.compose(state)
                self._emit_event(on_event, {
                    "type": "final_answer",
                    "reason": decision.reason,
                    "answer_preview": (state.final_answer or "")[:300],
                })
                break

            stage_plan = decision.stage_plan
            if stage_plan is None:
                state.done = True
                state.final_answer = self.build_incomplete_result(state)
                self._emit_event(on_event, {
                    "type": "runtime_incomplete",
                    "answer_preview": (state.final_answer or "")[:300],
                })
                break

            execution_result = await self.stage_executor.execute(
                stage_plan=stage_plan,
                global_state=state,
                on_token=on_token,
                on_tool=on_tool,
                on_event=on_event,
            )
            state = self.state_manager.apply_stage_result(state, stage_plan, execution_result)
            self._emit_event(on_event, {
                "type": "state_update",
                "current_stage": state.current_stage,
                "active_files": list(state.active_files[:8]),
                "changed_files": list(state.changed_files[:8]),
                "failures": list(state.failures[-3:]),
            })
            evaluation = self.evaluator.evaluate(state, execution_result)
            self._emit_event(on_event, {
                "type": "evaluation",
                "done": evaluation.done,
                "needs_replan": evaluation.needs_replan,
                "reason": evaluation.reason,
                "final_answer_preview": (evaluation.final_answer or "")[:300] if evaluation.final_answer else "",
            })

            if evaluation.done:
                state.done = True
                state.final_answer = evaluation.final_answer
                self._emit_event(on_event, {
                    "type": "final_answer",
                    "reason": evaluation.reason,
                    "answer_preview": (state.final_answer or "")[:300],
                })
                break

            if evaluation.needs_replan:
                state.working_memory["replan_reason"] = evaluation.reason

        if not state.final_answer:
            state.final_answer = self.build_incomplete_result(state)
            self._emit_event(on_event, {
                "type": "runtime_incomplete",
                "answer_preview": (state.final_answer or "")[:300],
            })
        self.last_state = state
        return state

    async def run(
        self,
        user_request: str,
        on_token: Callable[[str], None] | None = None,
        on_tool: Callable[[str, dict], None] | None = None,
        on_event: RuntimeEventFn | None = None,
    ) -> str:
        state = await self.run_state(user_request, on_token=on_token, on_tool=on_tool, on_event=on_event)
        return state.final_answer or self.build_incomplete_result(state)

    @staticmethod
    def _emit_event(on_event: RuntimeEventFn | None, payload: dict[str, Any]) -> None:
        if on_event is not None:
            on_event(payload)

    @staticmethod
    def build_incomplete_result(state: GlobalTaskState) -> str:
        completed = ", ".join(plan.stage for plan in state.stage_history) or "none"
        failures = "; ".join(state.failures[-3:]) or "none"
        summary = state.stage_summaries[-1].text if state.stage_summaries else ""
        return (
            "Task stopped before reaching a clean final state.\n"
            f"Completed stages: {completed}\n"
            f"Recent failures: {failures}\n"
            f"Latest summary: {summary}"
        )


def stage_to_execution_state(stage: str) -> ExecutionState:
    mapping = {
        "understand": ExecutionState.PLANNING,
        "locate": ExecutionState.EXPLORING,
        "analyze": ExecutionState.EXPLORING,
        "modify": ExecutionState.CODING,
        "verify": ExecutionState.VERIFYING,
        "recover": ExecutionState.DEBUGGING,
        "finalize": ExecutionState.VERIFYING,
    }
    return mapping.get(stage, ExecutionState.CODING)

