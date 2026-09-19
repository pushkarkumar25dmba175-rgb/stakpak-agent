"""Shell tool: runs commands the user has approved, and only those.

The tool itself never decides that a command is acceptable. It parses the
command, classifies it with :func:`analyze_command`, and hands the verdict to
the policy engine along with the specific concerns a human should weigh. A
command matching a configured deny pattern is blocked outright — approval is
not offered, because "are you sure?" is the wrong question for ``rm -rf /``.
"""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, Field

from personalos.security.command_analyzer import analyze_command
from personalos.security.risk import RiskLevel
from personalos.tools.base import (
    OperationSpec,
    PreparedCall,
    Tool,
    ToolContext,
    ToolResult,
)


class RunInput(BaseModel):
    command: str = Field(description="The exact command line to run.")
    cwd: str | None = Field(default=None, description="Working directory; defaults to the agent workdir.")
    timeout_seconds: float | None = Field(default=None, gt=0)
    reason: str | None = Field(default=None, description="Why this command is needed.")


class ShellTool(Tool):
    """Executes a shell command inside the sandbox."""

    name: ClassVar[str] = "shell"
    description: ClassVar[str] = (
        "Run a shell command. Commands are screened for destructive, privileged "
        "and credential-accessing behaviour before you are asked to approve them."
    )

    operations: ClassVar[dict[str, OperationSpec]] = {
        "run": OperationSpec(
            name="run",
            description="Run a shell command and capture its output.",
            schema=RunInput,
            risk=RiskLevel.LOW,
            permissions=["shell.execute"],
            reversible=False,
        )
    }

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        args = call.arguments
        assert isinstance(args, RunInput)
        security = context.settings.security

        if not security.shell_enabled:
            call.blocked_reason = "Shell execution is disabled in configuration."
            return call

        analysis = analyze_command(
            args.command,
            allowlist=security.shell_allowlist,
            denied_patterns=security.shell_denied_patterns,
        )
        if analysis.blocked:
            call.blocked_reason = analysis.blocked_reason
            call.risk_level = RiskLevel.HIGH
            call.concerns = analysis.concerns
            return call

        call.risk_level = max(call.risk_level, analysis.risk_level)
        call.concerns = analysis.concerns
        call.reversible = False
        call.rollback_hint = "Commands are not automatically reversible."
        call.description = f"Run: {args.command}"
        if args.reason:
            call.description += f" ({args.reason})"

        if args.cwd:
            try:
                call.affected_resources = [str(context.paths.resolve(args.cwd))]
            except Exception as exc:  # noqa: BLE001
                call.blocked_reason = str(exc)
        else:
            call.affected_resources = [str(context.settings.workdir)]
        return call

    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        args = call.arguments
        assert isinstance(args, RunInput)
        cwd = context.paths.resolve(args.cwd) if args.cwd else context.settings.workdir
        timeout = args.timeout_seconds or context.settings.security.shell_timeout_seconds

        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would run in {cwd}: {args.command}")

        result = await context.sandbox.run(args.command, cwd=Path(cwd), timeout=timeout)
        context.journal.record_irreversible(
            context.task_id,
            f"ran shell command: {args.command}",
            step_id=context.step_id,
        )
        payload = {
            "command": result.command,
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
            "duration_seconds": round(result.duration_seconds, 3),
        }
        if result.success:
            return ToolResult.ok(result.summary(), data=payload)
        return ToolResult(
            success=False,
            output=result.summary(),
            data=payload,
            error=(
                f"Command timed out after {timeout}s."
                if result.timed_out
                else f"Command exited with code {result.return_code}."
            ),
        )
