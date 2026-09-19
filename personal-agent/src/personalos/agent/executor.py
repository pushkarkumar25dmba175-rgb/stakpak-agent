"""Executing an approved plan, one step at a time.

The loop for each step is: resolve arguments → prepare (validate + classify
risk) → ask the policy engine → execute → verify → log. Approval happens
*here*, immediately before the step runs, not once for the whole plan up front.
That is what makes level-4 actions confirmable "immediately before execution",
and it means a plan whose later steps turn out riskier than expected still
stops and asks.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from personalos.agent.plan import Plan, PlanStep, resolve_templates
from personalos.agent.verification import VerificationResult, verify_step
from personalos.audit.audit_log import AuditLog
from personalos.database.models import ApprovalStatus
from personalos.errors import ToolNotFoundError, ToolValidationError
from personalos.security.permissions import ApprovalRequest
from personalos.security.policy_engine import PolicyEngine
from personalos.security.risk import RiskLevel
from personalos.security.secrets import redact_structure
from personalos.tools.base import PreparedCall, ToolContext, ToolResult
from personalos.tools.registry import ToolRegistry
from personalos.utils.textutil import truncate

#: Longest argument value kept in an episode. Enough for a path or a short
#: template, far short of a document.
TRACE_VALUE_LIMIT = 400


def _trace_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Redact and clip a step's arguments for long-term storage."""
    clipped = {
        key: truncate(value, TRACE_VALUE_LIMIT) if isinstance(value, str) else value
        for key, value in arguments.items()
    }
    return redact_structure(clipped)


@dataclass
class StepOutcome:
    """Everything that happened to one step."""

    step: PlanStep
    result: ToolResult
    risk_level: RiskLevel
    approval_status: ApprovalStatus
    approval_reason: str
    verification: VerificationResult | None = None
    attempts: int = 1
    skipped: bool = False

    @property
    def success(self) -> bool:
        return self.result.success and not (self.verification and self.verification.failed)


@dataclass
class ExecutionReport:
    """The result of running a whole plan."""

    plan: Plan
    outcomes: list[StepOutcome] = field(default_factory=list)
    halted_reason: str | None = None
    rollback_ids: list[int] = field(default_factory=list)

    @property
    def success(self) -> bool:
        if self.halted_reason:
            return False
        return all(outcome.success or outcome.step.optional for outcome in self.outcomes)

    @property
    def max_risk_level(self) -> RiskLevel:
        if not self.outcomes:
            return RiskLevel.READ_ONLY
        return max(outcome.risk_level for outcome in self.outcomes)

    def summary(self) -> str:
        """A short account of what happened, suitable for the CLI and memory."""
        if self.plan.is_empty():
            return self.plan.notes or "There was nothing to do."
        lines = []
        for outcome in self.outcomes:
            if outcome.skipped:
                mark = "skipped"
            elif outcome.success:
                mark = "done"
            else:
                mark = "failed"
            detail = outcome.result.error or truncate(outcome.result.output.replace("\n", " "), 120)
            lines.append(f"[{mark}] {outcome.step.description} — {detail}")
        if self.halted_reason:
            lines.append(f"Stopped: {self.halted_reason}")
        return "\n".join(lines)

    def action_trace(self) -> list[dict[str, Any]]:
        """The shape episodic memory and the pattern detector consume.

        Arguments are the step's *declared* ones, not the values they resolved
        to. That keeps ``{{ steps.scan.data.count }}`` intact, so a skill
        generated from this trace re-derives the value at run time instead of
        freezing whatever it happened to be. They are redacted and clipped
        because a write step's ``content`` can be arbitrarily large.
        """
        return [
            {
                "id": outcome.step.id,
                "tool": outcome.step.tool,
                "operation": outcome.step.operation,
                "description": outcome.step.description,
                "arguments": _trace_arguments(outcome.step.arguments),
                "depends_on": list(outcome.step.depends_on),
                "success": outcome.success,
                "risk_level": int(outcome.risk_level),
            }
            for outcome in self.outcomes
        ]


class Executor:
    """Runs plans. Owns no policy of its own — it asks the policy engine."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: PolicyEngine,
        audit: AuditLog,
        *,
        max_retries: int = 3,
    ) -> None:
        self.registry = registry
        self.policy = policy
        self.audit = audit
        self.max_retries = max_retries

    @staticmethod
    def _retryable(call: PreparedCall, result: ToolResult) -> bool:
        """Whether a failed step may be attempted again.

        Only read-only steps are retried. Re-running a move or a shell command
        that failed halfway is how an agent turns one problem into two.
        """
        if call.risk_level > RiskLevel.READ_ONLY:
            return False
        error = (result.error or "").lower()
        return any(
            marker in error
            for marker in ("timed out", "temporarily", "connection", "resource busy", "try again")
        )

    async def execute(
        self,
        plan: Plan,
        context: ToolContext,
        *,
        stop_on_failure: bool = True,
    ) -> ExecutionReport:
        """Run every step of ``plan`` in dependency order."""
        report = ExecutionReport(plan=plan)
        scope: dict[str, Any] = {"inputs": plan.inputs, "steps": {}}

        for step in plan.ordered():
            step_context = context.child(step.id)

            missing = [
                dependency
                for dependency in (set(step.depends_on) | step.references())
                if dependency not in scope["steps"]
            ]
            if missing:
                outcome = self._skip(
                    step,
                    f"Skipped: it depends on {', '.join(sorted(missing))}, which did not produce a result.",
                )
                report.outcomes.append(outcome)
                if stop_on_failure and not step.optional:
                    report.halted_reason = outcome.result.error
                    break
                continue

            arguments = resolve_templates(step.arguments, scope)

            try:
                tool = self.registry.get(step.tool)
                call = tool.prepare(step.operation, arguments, step_context)
            except (ToolNotFoundError, ToolValidationError) as exc:
                outcome = self._skip(step, str(exc))
                report.outcomes.append(outcome)
                self._log(step, outcome, context, arguments)
                if stop_on_failure and not step.optional:
                    report.halted_reason = str(exc)
                    break
                continue

            if call.blocked:
                outcome = StepOutcome(
                    step=step,
                    result=ToolResult.failed(call.blocked_reason or "Blocked."),
                    risk_level=call.risk_level,
                    approval_status=ApprovalStatus.USER_DENIED,
                    approval_reason=call.blocked_reason or "Blocked by a safety rule.",
                    skipped=True,
                )
                report.outcomes.append(outcome)
                self._log(step, outcome, context, arguments)
                if stop_on_failure and not step.optional:
                    report.halted_reason = call.blocked_reason
                    break
                continue

            request = ApprovalRequest(
                task_id=context.task_id,
                step_id=step.id,
                tool=step.tool,
                operation=step.operation,
                description=call.description,
                risk_level=call.risk_level,
                parameters=call.as_parameters(),
                affected_resources=call.affected_resources,
                concerns=call.concerns,
                reversible=call.reversible,
                rollback_hint=call.rollback_hint,
                project=context.project,
            )
            outcome_policy = self.policy.evaluate(request)

            if not outcome_policy.approved:
                outcome = StepOutcome(
                    step=step,
                    result=ToolResult.failed(outcome_policy.decision.reason),
                    risk_level=call.risk_level,
                    approval_status=outcome_policy.decision.status,
                    approval_reason=outcome_policy.decision.reason,
                    skipped=True,
                )
                report.outcomes.append(outcome)
                self._log(step, outcome, context, call.as_parameters())
                if stop_on_failure and not step.optional:
                    report.halted_reason = outcome_policy.decision.reason
                    break
                continue

            result, attempts = await self._run_with_retries(tool, call, step_context)
            verification = verify_step(step, result, resolver=context.paths)
            outcome = StepOutcome(
                step=step,
                result=result,
                risk_level=call.risk_level,
                approval_status=outcome_policy.decision.status,
                approval_reason=outcome_policy.decision.reason,
                verification=verification,
                attempts=attempts,
            )
            report.outcomes.append(outcome)
            report.rollback_ids.extend(result.rollback_ids)
            if outcome.success:
                # Only a step that actually worked publishes its result. A later
                # step that interpolates a failed step's output would otherwise
                # run with an unresolved placeholder, which is worse than not
                # running at all.
                scope["steps"][step.id] = {
                    "output": result.output,
                    "data": result.data,
                    "success": result.success,
                    "artifacts": result.artifacts,
                }
            self._log(step, outcome, context, call.as_parameters())

            if not outcome.success and stop_on_failure and not step.optional:
                report.halted_reason = (
                    result.error
                    or (verification.note if verification and verification.failed else None)
                    or "The step did not succeed."
                )
                break

        return report

    async def _run_with_retries(
        self, tool: Any, call: PreparedCall, context: ToolContext
    ) -> tuple[ToolResult, int]:
        """Execute a call, retrying only when it is safe to do so."""
        attempts = 0
        result = ToolResult.failed("Not executed.")
        while attempts < max(1, self.max_retries):
            attempts += 1
            result = await tool.run(call, context)
            if result.success or not self._retryable(call, result):
                break
            # Back off a little so a transient condition has time to clear.
            await asyncio.sleep(min(2.0 * attempts, 5.0))
        return result, attempts

    @staticmethod
    def _skip(step: PlanStep, reason: str) -> StepOutcome:
        return StepOutcome(
            step=step,
            result=ToolResult.failed(reason),
            risk_level=RiskLevel.READ_ONLY,
            approval_status=ApprovalStatus.NOT_REQUIRED,
            approval_reason=reason,
            skipped=True,
        )

    def _log(
        self,
        step: PlanStep,
        outcome: StepOutcome,
        context: ToolContext,
        parameters: dict[str, Any],
    ) -> None:
        self.audit.record(
            task_id=context.task_id,
            step_id=step.id,
            action=step.description,
            tool=step.tool,
            parameters={"operation": step.operation, **parameters},
            risk_level=outcome.risk_level,
            approval_status=outcome.approval_status,
            success=outcome.success,
            result={
                "output": truncate(outcome.result.output, 4000),
                "data_keys": sorted(outcome.result.data)[:20],
                "artifacts": outcome.result.artifacts,
            },
            error=outcome.result.error,
            execution_time_ms=outcome.result.duration_ms,
            verified=outcome.verification.verified if outcome.verification else None,
            verification_note=outcome.verification.note if outcome.verification else None,
            project=context.project,
        )
