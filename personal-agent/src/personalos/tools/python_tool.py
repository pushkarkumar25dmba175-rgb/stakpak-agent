"""Python tool: runs a script in an isolated working directory.

The script is written to a file under ``~/.personalos/work/`` and executed by a
fresh interpreter with a scrubbed environment. Writing the script out rather
than piping it to ``python -c`` means the code that ran is on disk afterwards,
next to the audit entry that references it — you can read exactly what the
agent executed.
"""

from __future__ import annotations

import sys
from typing import ClassVar

from pydantic import BaseModel, Field

from personalos.security.injection import scan_for_injection
from personalos.security.risk import RiskLevel
from personalos.tools.base import (
    OperationSpec,
    PreparedCall,
    Tool,
    ToolContext,
    ToolResult,
)
from personalos.utils.timeutil import utcnow

#: Imports that turn a data-processing script into something with system reach.
_SENSITIVE_IMPORTS = (
    "subprocess", "shutil.rmtree", "os.system", "os.remove", "os.rmdir",
    "socket", "requests", "urllib.request", "httpx", "ctypes", "multiprocessing",
)


class RunScriptInput(BaseModel):
    code: str = Field(description="Python source to execute.")
    filename: str = Field(default="script.py", description="Name for the saved script.")
    timeout_seconds: float | None = Field(default=None, gt=0)
    argv: list[str] = Field(default_factory=list)


class PythonTool(Tool):
    """Executes Python in a scratch directory."""

    name: ClassVar[str] = "python"
    description: ClassVar[str] = "Run a Python script in an isolated working directory."

    operations: ClassVar[dict[str, OperationSpec]] = {
        "run": OperationSpec(
            name="run",
            description="Execute Python source and capture its output.",
            schema=RunScriptInput,
            risk=RiskLevel.MODERATE,
            permissions=["python.execute"],
            reversible=False,
        )
    }

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        args = call.arguments
        assert isinstance(args, RunScriptInput)

        concerns: list[str] = []
        for marker in _SENSITIVE_IMPORTS:
            if marker in args.code:
                concerns.append(f"uses {marker}")
        if scan_for_injection(args.code):
            concerns.append("contains text that reads like instructions rather than code")

        if concerns:
            # A script that shells out or reaches the network is not a level-2
            # local change any more, whatever it claims to be doing.
            call.risk_level = RiskLevel.HIGH
        call.concerns = concerns
        call.reversible = False
        call.rollback_hint = "Script effects are not automatically reversible."
        call.description = f"Run a {len(args.code.splitlines())}-line Python script"
        call.affected_resources = [str(context.settings.workdir)]
        return call

    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        args = call.arguments
        assert isinstance(args, RunScriptInput)
        workdir = context.settings.workdir / f"py-{utcnow().strftime('%Y%m%d-%H%M%S%f')}"
        workdir.mkdir(parents=True, exist_ok=True)
        script = workdir / args.filename

        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would run {len(args.code)} characters of Python in {workdir}.")

        script.write_text(args.code, encoding="utf-8")
        timeout = args.timeout_seconds or context.settings.security.python_timeout_seconds
        result = await context.sandbox.run(
            [sys.executable, str(script), *args.argv],
            cwd=workdir,
            timeout=timeout,
        )
        context.journal.record_irreversible(
            context.task_id, f"ran python script {script}", step_id=context.step_id
        )
        payload = {
            "script_path": str(script),
            "return_code": result.return_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "timed_out": result.timed_out,
        }
        if result.success:
            return ToolResult.ok(result.summary(), data=payload, artifacts=[str(script)])
        return ToolResult(
            success=False,
            output=result.summary(),
            data=payload,
            error=(
                f"Script timed out after {timeout}s."
                if result.timed_out
                else f"Script exited with code {result.return_code}."
            ),
            artifacts=[str(script)],
        )
