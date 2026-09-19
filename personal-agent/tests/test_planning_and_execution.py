"""Planning, template resolution, execution, verification and retries."""

from __future__ import annotations

from pathlib import Path

import pytest

from personalos.agent.executor import Executor
from personalos.agent.plan import Plan, PlanStep, VerificationSpec, resolve_templates
from personalos.agent.planner import HeuristicPlanner, LLMPlanner
from personalos.agent.verification import verify_step
from personalos.llm.providers.echo import EchoProvider
from personalos.tools.base import ToolResult


# ---- plan ordering and templating -----------------------------------------
def test_steps_are_ordered_by_dependency() -> None:
    plan = Plan(
        goal="g",
        steps=[
            PlanStep(id="write", description="w", tool="filesystem", operation="write",
                     arguments={"content": "{{ steps.scan.data.count }}"}, depends_on=["scan"]),
            PlanStep(id="scan", description="s", tool="filesystem", operation="list"),
        ],
    )
    assert [step.id for step in plan.ordered()] == ["scan", "write"]


def test_a_dependency_cycle_does_not_hang() -> None:
    plan = Plan(
        goal="g",
        steps=[
            PlanStep(id="a", description="a", tool="t", operation="o", depends_on=["b"]),
            PlanStep(id="b", description="b", tool="t", operation="o", depends_on=["a"]),
        ],
    )
    assert len(plan.ordered()) == 2


def test_templates_resolve_and_keep_types() -> None:
    scope = {"inputs": {"folder": "~/Downloads"}, "steps": {"scan": {"data": {"count": 7}}}}
    assert resolve_templates("{{ inputs.folder }}", scope) == "~/Downloads"
    assert resolve_templates("{{ steps.scan.data.count }}", scope) == 7
    assert resolve_templates("found {{ steps.scan.data.count }} files", scope) == "found 7 files"


def test_unresolvable_templates_are_left_visible() -> None:
    assert resolve_templates("{{ steps.missing.data.x }}", {"steps": {}}) == "{{ steps.missing.data.x }}"


def test_step_references_are_discovered() -> None:
    step = PlanStep(
        id="s", description="d", tool="t", operation="o",
        arguments={"path": "{{ steps.earlier.data.path }}"},
    )
    assert step.references() == {"earlier"}


# ---- heuristic planner -----------------------------------------------------
def test_heuristic_planner_handles_a_listing_request(registry, workspace: Path) -> None:
    plan = HeuristicPlanner(registry).plan(f"show me what is in {workspace}")
    assert plan.source == "heuristic"
    assert plan.steps[0].tool == "filesystem"


@pytest.mark.parametrize(
    "request_text,expected",
    [
        (r"list C:\Users\bob\PersonalOS", r"C:\Users\bob\PersonalOS"),
        (r"organise C:/Users/bob/Downloads", "C:/Users/bob/Downloads"),
        (r"what is in \\fileserver\share\reports", r"\\fileserver\share\reports"),
        ("list /home/me/docs", "/home/me/docs"),
        ("summarise ~/Research/notes.md", "~/Research/notes.md"),
        ('read "C:\\Program Files\\app\\log.txt"', "C:\\Program Files\\app\\log.txt"),
    ],
)
def test_paths_are_extracted_on_every_platform(request_text: str, expected: str) -> None:
    """A Windows user typing a native path must not get "I cannot help"."""
    assert HeuristicPlanner._extract_path(request_text) == expected


def test_prose_without_a_path_extracts_nothing() -> None:
    assert HeuristicPlanner._extract_path("write me a haiku about autumn") is None


def test_heuristic_planner_says_so_when_it_cannot_help(registry) -> None:
    plan = HeuristicPlanner(registry).plan("write me a haiku about autumn")
    assert plan.is_empty()
    assert "language model" in (plan.notes or "")


# ---- LLM planner -----------------------------------------------------------
async def test_llm_plan_is_validated_against_the_registry(registry, memory) -> None:
    provider = EchoProvider(
        "test",
        scripted=['{"goal": "g", "steps": [{"id": "x", "description": "d", '
                  '"tool": "teleporter", "operation": "go", "arguments": {}}]}'],
    )
    provider.name = "scripted"  # bypass the offline shortcut
    planner = LLMPlanner(provider, registry, memory)
    plan = await planner.plan("do something impossible", default_path="/tmp")
    assert plan.source == "heuristic", "an unrunnable plan must fall back, not be executed"


async def test_llm_plan_is_accepted_when_valid(registry, memory, workspace: Path) -> None:
    provider = EchoProvider(
        "test",
        scripted=['{"goal": "list files", "steps": [{"id": "x", "description": "list", '
                  f'"tool": "filesystem", "operation": "list", "arguments": {{"path": "{workspace}"}}}}]}}'],
    )
    provider.name = "scripted"
    plan = await LLMPlanner(provider, registry, memory).plan("list the files")
    assert plan.source == "llm"
    assert plan.steps[0].tool == "filesystem"


async def test_malformed_model_output_falls_back(registry, memory, workspace: Path) -> None:
    provider = EchoProvider("test", scripted=["not json at all"])
    provider.name = "scripted"
    plan = await LLMPlanner(provider, registry, memory).plan(
        f"list {workspace}", default_path=str(workspace)
    )
    assert plan.source == "heuristic"


def test_planner_prompt_respects_the_context_budget(registry, memory) -> None:
    for index in range(50):
        memory.semantic.remember(f"fact_{index}", "x" * 500, confidence=0.9)
    planner = LLMPlanner(EchoProvider("t"), registry, memory, context_budget=1500, max_output_tokens=500)
    prompt, budget = planner.build_prompt("do a thing", memory.retrieve("thing"))
    assert "## Request" in prompt
    assert "## Available tools" in prompt
    assert len(prompt) // 4 <= budget.available + 200


# ---- verification ----------------------------------------------------------
def test_verification_catches_a_missing_artifact(tmp_path: Path) -> None:
    step = PlanStep(
        id="s", description="write", tool="filesystem", operation="write",
        verification=VerificationSpec(kind="path_exists", target=str(tmp_path / "nope.txt")),
    )
    result = verify_step(step, ToolResult.ok("claimed success"))
    assert result.verified is False


def test_verification_catches_an_empty_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.md"
    empty.touch()
    step = PlanStep(
        id="s", description="write", tool="filesystem", operation="write",
        verification=VerificationSpec(kind="path_nonempty", target=str(empty)),
    )
    assert verify_step(step, ToolResult.ok("done")).verified is False


def test_verification_of_return_codes() -> None:
    step = PlanStep(id="s", description="run", tool="shell", operation="run",
                    verification=VerificationSpec(kind="return_code_zero"))
    assert verify_step(step, ToolResult.ok("", data={"return_code": 0})).verified is True
    assert verify_step(step, ToolResult.ok("", data={"return_code": 2})).verified is False


def test_unverifiable_steps_report_none_rather_than_true() -> None:
    step = PlanStep(id="s", description="x", tool="t", operation="o",
                    verification=VerificationSpec(kind="none"))
    assert verify_step(step, ToolResult.ok("done")).verified is None


def test_failed_results_are_never_verified_true() -> None:
    step = PlanStep(id="s", description="x", tool="t", operation="o")
    assert verify_step(step, ToolResult.failed("boom")).verified is False


# ---- executor --------------------------------------------------------------
async def test_executor_runs_a_plan_and_passes_data_between_steps(
    registry, policy, audit, tool_context, workspace: Path
) -> None:
    (workspace / "one.txt").write_text("a")
    (workspace / "two.txt").write_text("b")
    plan = Plan(
        goal="count and report",
        steps=[
            PlanStep(id="scan", description="list the folder", tool="filesystem",
                     operation="list", arguments={"path": str(workspace)}),
            PlanStep(
                id="report", description="write a report", tool="filesystem", operation="write",
                arguments={"path": str(workspace / "report.md"),
                           "content": "Found {{ steps.scan.data.count }} files."},
                depends_on=["scan"],
                verification=VerificationSpec(kind="path_nonempty"),
            ),
        ],
    )
    report = await Executor(registry, policy, audit).execute(plan, tool_context)
    assert report.success
    assert (workspace / "report.md").read_text() == "Found 2 files."


async def test_executor_stops_when_a_step_is_denied(
    registry, policy, audit, tool_context, workspace: Path, handler
) -> None:
    from personalos.security.permissions import ApprovalDecision

    class RefuseEverything:
        def request_approval(self, request):
            return ApprovalDecision.denied("no")

    policy.handler = RefuseEverything()
    plan = Plan(
        goal="delete something",
        steps=[
            PlanStep(id="rm", description="delete", tool="filesystem", operation="delete",
                     arguments={"path": str(workspace)}),
            PlanStep(id="after", description="should not run", tool="filesystem",
                     operation="list", arguments={"path": str(workspace)}),
        ],
    )
    report = await Executor(registry, policy, audit).execute(plan, tool_context)
    assert not report.success
    assert len(report.outcomes) == 1, "execution stops at the refusal"
    assert report.outcomes[0].skipped


async def test_executor_skips_steps_whose_dependency_failed(
    registry, policy, audit, tool_context, workspace: Path
) -> None:
    plan = Plan(
        goal="chain",
        steps=[
            PlanStep(id="bad", description="read a missing file", tool="filesystem",
                     operation="read", arguments={"path": str(workspace / "missing.txt")},
                     optional=True),
            PlanStep(id="next", description="use it", tool="filesystem", operation="write",
                     arguments={"path": str(workspace / "out.md"),
                                "content": "{{ steps.bad.data.content }}"},
                     depends_on=["bad"], optional=True),
        ],
    )
    report = await Executor(registry, policy, audit).execute(plan, tool_context, stop_on_failure=False)
    assert not report.outcomes[0].success
    assert report.outcomes[1].skipped


async def test_executor_does_not_retry_write_operations(
    registry, policy, audit, tool_context, workspace: Path
) -> None:
    plan = Plan(
        goal="write somewhere forbidden",
        steps=[
            PlanStep(id="w", description="write outside the workspace", tool="filesystem",
                     operation="write", arguments={"path": "/etc/passwd", "content": "x"}),
        ],
    )
    report = await Executor(registry, policy, audit, max_retries=3).execute(plan, tool_context)
    assert not report.success
    assert report.outcomes[0].attempts == 1


async def test_executor_logs_every_step(registry, policy, audit, tool_context, workspace: Path) -> None:
    plan = Plan(
        goal="list",
        steps=[PlanStep(id="s", description="list", tool="filesystem", operation="list",
                        arguments={"path": str(workspace)})],
    )
    await Executor(registry, policy, audit).execute(plan, tool_context)
    records = audit.query()
    assert records and records[0].tool == "filesystem"
    assert records[0].verified is not None or records[0].verification_note


async def test_unknown_tool_in_a_plan_fails_cleanly(
    registry, policy, audit, tool_context
) -> None:
    plan = Plan(goal="g", steps=[PlanStep(id="s", description="x", tool="nope", operation="go")])
    report = await Executor(registry, policy, audit).execute(plan, tool_context)
    assert not report.success
    assert "No tool named" in (report.halted_reason or "")


async def test_action_trace_shape_is_stable(registry, policy, audit, tool_context, workspace: Path) -> None:
    plan = Plan(
        goal="list",
        steps=[PlanStep(id="s", description="list", tool="filesystem", operation="list",
                        arguments={"path": str(workspace)})],
    )
    report = await Executor(registry, policy, audit).execute(plan, tool_context)
    trace = report.action_trace()
    assert trace[0]["tool"] == "filesystem"
    assert trace[0]["operation"] == "list"
    assert trace[0]["success"] is True
