"""The plan: what the agent intends to do, before it does any of it.

A plan is data, not code. It can be printed, diffed, approved step by step,
stored in the database, replayed as a skill and rolled back. Nothing executes
that is not first expressed as a :class:`PlanStep`.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from pydantic import BaseModel, Field

from personalos.security.risk import RiskLevel
from personalos.utils.timeutil import isoformat

#: ``{{ steps.find_pdfs.data.count }}`` or ``{{ inputs.folder }}``
TEMPLATE_PATTERN = re.compile(r"\{\{\s*([a-zA-Z0-9_.\[\]-]+)\s*\}\}")


class VerificationSpec(BaseModel):
    """How to check that a step really did what it said.

    ``kind`` values:

    ``path_exists``      the path in ``target`` (or the step's artifacts) exists
    ``path_nonempty``    …and has a non-zero size
    ``output_contains``  the step output contains ``expected``
    ``return_code_zero`` the step's process exited 0
    ``count_at_least``   the step's ``data[field]`` is >= ``minimum``
    ``none``             explicitly unverifiable; recorded as such
    """

    kind: str = "path_exists"
    target: str | None = None
    expected: str | None = None
    field: str | None = None
    minimum: int = 1


class PlanStep(BaseModel):
    """One tool invocation, with everything needed to approve and verify it."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    description: str
    tool: str
    operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    expected_outcome: str | None = None
    verification: VerificationSpec | None = None
    optional: bool = False
    """An optional step that fails does not fail the plan."""

    def references(self) -> set[str]:
        """Step ids this step's arguments interpolate from."""
        found: set[str] = set()
        for match in TEMPLATE_PATTERN.finditer(str(self.arguments)):
            path = match.group(1).split(".")
            if len(path) >= 2 and path[0] == "steps":
                found.add(path[1])
        return found


class Plan(BaseModel):
    """An ordered set of steps that together satisfy one request."""

    plan_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    goal: str
    steps: list[PlanStep] = Field(default_factory=list)
    notes: str | None = None
    source: str = "heuristic"
    """heuristic | llm | skill — how this plan was produced."""
    inputs: dict[str, Any] = Field(default_factory=dict)
    created_at: str = Field(default_factory=isoformat)

    def step(self, step_id: str) -> PlanStep | None:
        return next((step for step in self.steps if step.id == step_id), None)

    def is_empty(self) -> bool:
        return not self.steps

    def ordered(self) -> list[PlanStep]:
        """Steps in dependency order.

        A cycle, or a reference to a step that does not exist, is not an error
        here: the remaining steps are appended in declaration order and the
        executor fails them with a clear message when their input is missing.
        """
        remaining = list(self.steps)
        resolved: list[PlanStep] = []
        seen: set[str] = set()
        while remaining:
            progressed = False
            for step in list(remaining):
                dependencies = set(step.depends_on) | step.references()
                known = {s.id for s in self.steps}
                if dependencies & known <= seen:
                    resolved.append(step)
                    seen.add(step.id)
                    remaining.remove(step)
                    progressed = True
            if not progressed:
                resolved.extend(remaining)
                break
        return resolved

    def describe(self, risk_of: dict[str, RiskLevel] | None = None) -> str:
        """Render the plan for an approval prompt."""
        lines = [f"Goal: {self.goal}"]
        for index, step in enumerate(self.ordered(), start=1):
            marker = ""
            if risk_of and step.id in risk_of:
                level = risk_of[step.id]
                marker = f" [level {int(level)} — {level.label}]"
            lines.append(f"{index}. {step.description}{marker}")
            lines.append(f"     {step.tool}.{step.operation}")
        if self.notes:
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)


def resolve_templates(value: Any, scope: dict[str, Any]) -> Any:
    """Substitute ``{{ ... }}`` references using ``scope``.

    A reference that cannot be resolved is left as-is rather than replaced with
    an empty string, so a typo surfaces as an obviously wrong argument in the
    approval prompt instead of silently becoming ``""``.
    """
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            resolved = _lookup(match.group(1), scope)
            return match.group(0) if resolved is None else str(resolved)

        # A string that is exactly one reference keeps the referenced type.
        whole = TEMPLATE_PATTERN.fullmatch(value.strip())
        if whole:
            resolved = _lookup(whole.group(1), scope)
            return value if resolved is None else resolved
        return TEMPLATE_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {key: resolve_templates(item, scope) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_templates(item, scope) for item in value]
    return value


def _lookup(path: str, scope: dict[str, Any]) -> Any:
    cursor: Any = scope
    for part in path.split("."):
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        elif isinstance(cursor, list) and part.isdigit() and int(part) < len(cursor):
            cursor = cursor[int(part)]
        else:
            return None
    return cursor
