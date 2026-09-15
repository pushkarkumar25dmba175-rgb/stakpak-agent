"""Turning a request into a plan.

Two planners, in order of preference:

* :class:`LLMPlanner` asks the model for a structured plan, then *validates* it
  against the tool registry. A plan naming a tool that does not exist, or an
  operation with the wrong arguments, is rejected rather than half-executed.
* :class:`HeuristicPlanner` handles common shapes with no model at all. It is
  the fallback when there is no API key, when the model is unreachable, and
  when the model returns something unusable — so the agent stays useful
  offline instead of failing outright.

Neither planner executes anything or decides anything about permissions.
"""

from __future__ import annotations

import re
from pathlib import Path

from personalos.agent.plan import Plan, PlanStep, VerificationSpec
from personalos.errors import LLMError, PlanningError
from personalos.llm.base import LLMProvider, Message
from personalos.llm.context import ContextBudget
from personalos.memory.memory_manager import MemoryManager, RetrievedContext, classify_intent
from personalos.tools.registry import ToolRegistry
from personalos.utils.textutil import truncate

PLANNER_SYSTEM = """You are the planning component of PersonalOS, a local personal assistant.

Produce a plan as JSON. Rules you must follow:
- Use only the tools and operations listed. Never invent one.
- Prefer read-only steps first: look before you touch.
- Prefer reversible actions. Move files to a trash folder rather than deleting them.
- One tool call per step. Keep plans short; do not pad them with unnecessary steps.
- Arguments may reference an earlier step with {{ steps.<step_id>.data.<key> }}.
- If the request needs nothing done — it is a question you can answer from
  context — return an empty steps list and explain in "notes".

Respond with exactly this JSON shape:
{"goal": "...", "notes": "...", "steps": [
  {"id": "short_id", "description": "...", "tool": "...", "operation": "...",
   "arguments": {...}, "depends_on": [], "expected_outcome": "...",
   "verification": {"kind": "path_exists|path_nonempty|output_contains|return_code_zero|count_at_least|none",
                    "target": null, "expected": null, "field": null, "minimum": 1}}
]}"""


class HeuristicPlanner:
    """Produces plans for common requests without consulting a model."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    @staticmethod
    def _extract_path(request: str) -> str | None:
        """Pull a filesystem path out of a request, if there is an obvious one."""
        quoted = re.findall(r"['\"]([^'\"]+)['\"]", request)
        for candidate in quoted:
            if "/" in candidate or candidate.startswith("~"):
                return candidate
        for token in re.findall(r"(~?[\w./-]*/[\w./-]+)", request):
            return token
        for word in ("Downloads", "Documents", "Desktop", "Research"):
            if word.lower() in request.lower():
                return f"~/{word}"
        return None

    def plan(self, request: str, *, default_path: str | None = None) -> Plan:
        """Build a best-effort plan from keywords."""
        intent = classify_intent(request)
        path = self._extract_path(request) or default_path
        plan = Plan(goal=truncate(request, 200), source="heuristic")

        if intent in {"inspect", "organise", "general"} and path:
            plan.steps.append(
                PlanStep(
                    id="list_dir",
                    description=f"List what is in {path}",
                    tool="filesystem",
                    operation="list",
                    arguments={"path": path, "recursive": False},
                    expected_outcome="A listing of the directory.",
                    verification=VerificationSpec(kind="count_at_least", field="count", minimum=0),
                )
            )
        elif intent == "search" and path:
            terms = [word for word in re.findall(r"[\w.-]{3,}", request) if word.lower() not in {
                "find", "search", "look", "files", "file", "inside", "within", "containing"
            }]
            plan.steps.append(
                PlanStep(
                    id="search_files",
                    description=f"Search {path} for {terms[0] if terms else 'the requested text'}",
                    tool="filesystem",
                    operation="search",
                    arguments={"path": path, "query": terms[0] if terms else request[:40]},
                    verification=VerificationSpec(kind="none"),
                )
            )
        elif intent == "summarise" and path and Path(path).suffix:
            plan.steps.append(
                PlanStep(
                    id="extract",
                    description=f"Extract the text of {path}",
                    tool="documents",
                    operation="extract_text",
                    arguments={"path": path},
                    verification=VerificationSpec(kind="count_at_least", field="characters", minimum=1),
                )
            )

        if plan.is_empty():
            plan.notes = (
                "I could not work out concrete steps for this without a language model. "
                "Configure an LLM provider, or rephrase the request naming a specific folder or file."
            )
        return plan


class LLMPlanner:
    """Asks the model for a plan and validates it before anyone sees it."""

    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        memory: MemoryManager,
        *,
        max_output_tokens: int = 2048,
        context_budget: int = 12000,
    ) -> None:
        self.provider = provider
        self.registry = registry
        self.memory = memory
        self.max_output_tokens = max_output_tokens
        self.context_budget = context_budget
        self.fallback = HeuristicPlanner(registry)

    def build_prompt(
        self,
        request: str,
        retrieved: RetrievedContext,
        *,
        project: str | None = None,
        extra_context: str | None = None,
    ) -> tuple[str, ContextBudget]:
        """Assemble the planning prompt under the configured token budget."""
        budget = ContextBudget(
            max_tokens=self.context_budget, reserved_for_output=self.max_output_tokens
        )
        budget.add("Request", request, priority=1, truncatable=False)
        budget.add("Available tools", self.registry.catalogue(), priority=2, truncatable=False)
        if project:
            budget.add("Active project", project, priority=3)
        budget.add("What I remember", retrieved.render(), priority=4)
        if extra_context:
            budget.add("Additional context", extra_context, priority=5)
        return budget.build(), budget

    def _validate(self, plan: Plan) -> Plan:
        """Reject a plan referencing tools or operations that do not exist."""
        problems: list[str] = []
        seen_ids: set[str] = set()
        for step in plan.steps:
            if step.id in seen_ids:
                problems.append(f"duplicate step id {step.id!r}")
            seen_ids.add(step.id)
            if not self.registry.has(step.tool):
                problems.append(f"unknown tool {step.tool!r} in step {step.id}")
                continue
            tool = self.registry.get(step.tool)
            if step.operation not in tool.operations:
                problems.append(
                    f"{step.tool} has no operation {step.operation!r} (step {step.id})"
                )
        if problems:
            raise PlanningError(
                "The model produced a plan I cannot run: " + "; ".join(problems),
                remediation="Retrying with the offline planner.",
            )
        return plan

    async def plan(
        self,
        request: str,
        *,
        project: str | None = None,
        extra_context: str | None = None,
        default_path: str | None = None,
    ) -> Plan:
        """Produce a validated plan, falling back to heuristics on any failure."""
        if self.provider.name == "echo":
            # No real model is configured, so there is nothing to ask. Plan
            # from keywords rather than round-tripping through a stub.
            return self.fallback.plan(request, default_path=default_path)

        retrieved = self.memory.retrieve(request, project=project)
        prompt, _budget = self.build_prompt(
            request, retrieved, project=project, extra_context=extra_context
        )

        try:
            response = await self.provider.complete(
                [Message("user", prompt)],
                system=PLANNER_SYSTEM,
                max_tokens=self.max_output_tokens,
                temperature=0.1,
                json_mode=True,
            )
            payload = response.json()
        except (LLMError, PlanningError):
            return self.fallback.plan(request, default_path=default_path)

        if not isinstance(payload, dict):
            return self.fallback.plan(request, default_path=default_path)

        try:
            plan = Plan(
                goal=str(payload.get("goal") or truncate(request, 200)),
                notes=payload.get("notes"),
                steps=[PlanStep.model_validate(step) for step in payload.get("steps", [])],
                source="llm",
            )
            return self._validate(plan)
        except Exception:  # noqa: BLE001 - any malformed plan falls back
            fallback = self.fallback.plan(request, default_path=default_path)
            fallback.notes = (
                (fallback.notes or "")
                + " The model's plan was malformed, so I fell back to a simple one."
            ).strip()
            return fallback
