"""Checking that a step did what it claimed.

A tool returning ``success=True`` means the call did not raise. It does not
mean the file is where it should be or that the report has anything in it.
Verification is the difference between "the command exited 0" and "the work is
done", and the agent reports the second, not the first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from personalos.agent.plan import PlanStep, VerificationSpec
from personalos.tools.base import ToolResult
from personalos.utils.paths import PathResolver


@dataclass
class VerificationResult:
    """The outcome of checking one step."""

    verified: bool | None
    """True, False, or None when the step is explicitly unverifiable."""
    note: str

    @property
    def failed(self) -> bool:
        return self.verified is False


def default_spec(step: PlanStep, result: ToolResult) -> VerificationSpec:
    """Pick a sensible check when the plan did not specify one.

    Steps that produced artifacts get a "does it exist and have content?"
    check; a process result gets its exit code checked; everything else is
    recorded as unverified rather than optimistically passed.
    """
    if result.artifacts:
        return VerificationSpec(kind="path_nonempty", target=result.artifacts[0])
    if "return_code" in result.data:
        return VerificationSpec(kind="return_code_zero")
    return VerificationSpec(kind="none")


def verify_step(
    step: PlanStep,
    result: ToolResult,
    *,
    resolver: PathResolver | None = None,
) -> VerificationResult:
    """Run the step's verification check against its result."""
    if not result.success:
        return VerificationResult(False, result.error or "The step reported failure.")

    spec = step.verification or default_spec(step, result)
    kind = (spec.kind or "none").lower()

    if kind == "none":
        return VerificationResult(None, "No automatic check applies to this step.")

    if kind in {"path_exists", "path_nonempty"}:
        target = spec.target or (result.artifacts[0] if result.artifacts else None)
        if not target:
            return VerificationResult(None, "Nothing to check: the step produced no path.")
        try:
            path = resolver.resolve(target) if resolver else Path(target)
        except Exception as exc:  # noqa: BLE001
            return VerificationResult(False, f"Could not check {target}: {exc}")
        if not path.exists():
            return VerificationResult(False, f"{path} does not exist after the step ran.")
        if kind == "path_nonempty":
            if path.is_dir():
                if not any(path.iterdir()):
                    return VerificationResult(False, f"{path} exists but is empty.")
                return VerificationResult(True, f"{path} exists and has contents.")
            if path.stat().st_size == 0:
                return VerificationResult(False, f"{path} exists but is empty.")
            return VerificationResult(True, f"{path} exists and is {path.stat().st_size} bytes.")
        return VerificationResult(True, f"{path} exists.")

    if kind == "output_contains":
        expected = spec.expected or ""
        haystack = f"{result.output}\n{result.data}"
        if expected and expected in haystack:
            return VerificationResult(True, f"Output contains {expected!r}.")
        return VerificationResult(False, f"Output does not contain {expected!r}.")

    if kind == "return_code_zero":
        code = result.data.get("return_code")
        if code is None:
            return VerificationResult(None, "The step did not report a return code.")
        if int(code) == 0:
            return VerificationResult(True, "The process exited 0.")
        return VerificationResult(False, f"The process exited {code}.")

    if kind == "count_at_least":
        field = spec.field or "count"
        value = result.data.get(field)
        if value is None:
            return VerificationResult(None, f"The step did not report {field!r}.")
        try:
            numeric = int(value)
        except (TypeError, ValueError):
            return VerificationResult(None, f"{field}={value!r} is not a number.")
        if numeric >= spec.minimum:
            return VerificationResult(True, f"{field} is {numeric} (wanted at least {spec.minimum}).")
        return VerificationResult(False, f"{field} is {numeric}, below the expected {spec.minimum}.")

    return VerificationResult(None, f"Unknown verification kind {spec.kind!r}.")
