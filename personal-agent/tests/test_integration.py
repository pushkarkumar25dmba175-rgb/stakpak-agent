"""End-to-end behaviour of the assembled agent, the scheduler and the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from personalos.agent.orchestrator import Agent
from personalos.agent.state import AgentState
from personalos.database.models import TaskStatus
from personalos.errors import ConfigurationError
from personalos.scheduler.jobs import parse_schedule
from personalos.scheduler.scheduler import Scheduler
from personalos.security.risk import RiskLevel


# ---- the agent loop --------------------------------------------------------
async def test_a_read_only_request_runs_without_asking(agent: Agent, workspace: Path, handler) -> None:
    (workspace / "note.md").write_text("hello")
    outcome = await agent.ask(f"list {workspace}")
    assert outcome.success
    assert handler.requests == [], "read-only work never interrupts the user"
    assert agent.state.state is AgentState.IDLE


async def test_every_task_is_recorded_and_explainable(agent: Agent, workspace: Path) -> None:
    outcome = await agent.ask(f"list {workspace}")
    task = agent.tasks.get(outcome.task_id)
    assert task is not None
    assert task.status == TaskStatus.COMPLETED
    records = agent.audit.query()
    assert any(record.task_id == outcome.task_id for record in records)


async def test_actions_land_in_the_jsonl_log(agent: Agent, workspace: Path) -> None:
    await agent.ask(f"list {workspace}")
    entries = list(agent.audit.iter_file())
    assert entries
    assert {"timestamp", "task_id", "tool", "risk_level", "approval_status"} <= set(entries[0])


async def test_a_task_creates_an_episode(agent: Agent, workspace: Path) -> None:
    await agent.ask(f"list {workspace}")
    episodes = agent.memory.episodic.recent(5)
    assert episodes
    assert episodes[0].action_signature.startswith("filesystem.list")


async def test_paused_agent_refuses_to_start_work(agent: Agent, workspace: Path) -> None:
    agent.state.pause()
    outcome = await agent.ask(f"list {workspace}")
    assert not outcome.success
    assert "paused" in outcome.answer.lower()


async def test_plan_only_executes_nothing(agent: Agent, workspace: Path) -> None:
    marker = workspace / "untouched.txt"
    outcome = await agent.ask(f"list {workspace}", plan_only=True)
    assert outcome.report is None
    assert not marker.exists()


async def test_running_a_skill_goes_through_approval(agent: Agent, workspace: Path, handler) -> None:
    from personalos.skills.schema import SkillDefinition, SkillInput, SkillStep

    agent.skills.create(
        SkillDefinition(
            name="make_note",
            description="Write a note.",
            inputs={"folder": SkillInput(type="string", required=True)},
            steps=[
                SkillStep(id="write", description="write a note", tool="filesystem",
                          operation="write",
                          arguments={"path": "{{ inputs.folder }}/note.md", "content": "hi"}),
            ],
        )
    )
    outcome = await agent.run_skill("make_note", {"folder": str(workspace)})
    assert outcome.success
    assert (workspace / "note.md").read_text() == "hi"
    summary = next(s for s in agent.skills.list_skills() if s.definition.name == "make_note")
    assert summary.run_count == 1


async def test_rollback_undoes_a_task(agent: Agent, workspace: Path) -> None:
    from personalos.agent.plan import Plan, PlanStep

    target = workspace / "created.md"
    plan = Plan(goal="write", steps=[
        PlanStep(id="w", description="write", tool="filesystem", operation="write",
                 arguments={"path": str(target), "content": "x"})
    ])
    task = agent.tasks.create("write a file")
    report = await agent.executor.execute(plan, agent.tool_context(task.task_id))
    assert report.success and target.exists()

    results = agent.rollback(task.task_id)
    assert any(result.applied for result in results)
    assert not target.exists()


async def test_rollback_dry_run_changes_nothing(agent: Agent, workspace: Path) -> None:
    from personalos.agent.plan import Plan, PlanStep

    target = workspace / "kept.md"
    task = agent.tasks.create("write a file")
    await agent.executor.execute(
        Plan(goal="write", steps=[
            PlanStep(id="w", description="write", tool="filesystem", operation="write",
                     arguments={"path": str(target), "content": "x"})
        ]),
        agent.tool_context(task.task_id),
    )
    results = agent.rollback(task.task_id, dry_run=True)
    assert all(not result.applied for result in results)
    assert target.exists()


async def test_rollback_of_an_unknown_task_is_reported(agent: Agent) -> None:
    from personalos.errors import RollbackError

    with pytest.raises(RollbackError, match="nothing to roll back"):
        agent.rollback("no-such-task")


async def test_feedback_becomes_a_stored_preference(agent: Agent) -> None:
    item = await agent.record_feedback("always save reports as markdown")
    assert item.kind == "preference"
    assert agent.memory.preferences.get("preferred_report_format") == "markdown"


async def test_repeating_a_task_produces_a_suggestion_not_an_automation(
    agent: Agent, workspace: Path
) -> None:
    for _ in range(3):
        await agent.ask(f"list {workspace}")
    pending = agent.suggestions.pending()
    assert pending, "a repeated shape should be noticed"
    generated = [s for s in agent.skills.list_skills() if s.definition.created_by == "agent"]
    assert generated == [], "and noticing is not automating"


async def test_accepting_a_suggestion_creates_a_readable_skill(
    agent: Agent, workspace: Path
) -> None:
    for _ in range(3):
        await agent.ask(f"list {workspace}")
    suggestion = agent.suggestions.pending()[0]
    definition = agent.accept_suggestion(suggestion.id, skill_name="listing_routine")

    assert definition.created_by == "agent"
    path = agent.skills.inspect("listing_routine").path
    assert path.exists() and path.suffix == ".yaml"
    assert agent.suggestions.get(suggestion.id).status == "accepted"


async def test_status_snapshot_is_complete(agent: Agent) -> None:
    snapshot = agent.status()
    assert snapshot["state"] == str(AgentState.IDLE)
    assert "filesystem" in snapshot["tools"]
    assert set(snapshot["memory"]) == {"episodes", "facts", "preferences", "workflows"}


# ---- projects --------------------------------------------------------------
def test_switching_projects_widens_the_allow_list(agent: Agent, tmp_path: Path) -> None:
    project_dir = tmp_path / "research"
    project_dir.mkdir()
    agent.projects.create("research", description="Reading", directories=[str(project_dir)])

    assert not agent.paths.is_allowed(project_dir / "a.md")
    agent.switch_project("research")
    assert agent.paths.is_allowed(project_dir / "a.md")
    assert agent.namespace == "research"


def test_unknown_project_is_reported(agent: Agent) -> None:
    with pytest.raises(ConfigurationError, match="no project called"):
        agent.switch_project("nonexistent")


def test_project_names_are_validated(agent: Agent) -> None:
    with pytest.raises(ConfigurationError, match="not a valid project name"):
        agent.projects.create("Not Valid")


# ---- scheduler -------------------------------------------------------------
@pytest.mark.parametrize(
    "text,kind",
    [
        ("every day at 18:00", "cron"),
        ("daily at 6pm", "cron"),
        ("every weekday at 17:30", "cron"),
        ("every monday at 9am", "cron"),
        ("every 30 minutes", "interval"),
        ("every 2 hours", "interval"),
        ("hourly", "cron"),
    ],
)
def test_schedule_parsing(text: str, kind: str) -> None:
    assert parse_schedule(text).kind == kind


def test_unparseable_schedule_is_refused() -> None:
    with pytest.raises(ConfigurationError, match="could not understand"):
        parse_schedule("whenever I feel like it")


def test_evening_time_conversion() -> None:
    spec = parse_schedule("every day at 6pm")
    assert spec.fields == {"hour": 18, "minute": 0}


def test_scheduled_jobs_are_capped_at_level_two(agent: Agent) -> None:
    scheduler = Scheduler(agent)
    job = scheduler.add("tidy", when="every day at 18:00", request="organize downloads",
                        max_risk_level=4)
    assert job.max_risk_level <= int(RiskLevel.MODERATE)


def test_a_job_needs_something_to_run(agent: Agent) -> None:
    with pytest.raises(ConfigurationError, match="either a request or a skill"):
        Scheduler(agent).add("empty", when="hourly")


async def test_running_a_job_restores_the_original_handler(agent: Agent, workspace: Path) -> None:
    scheduler = Scheduler(agent)
    job = scheduler.add("listing", when="hourly", request=f"list {workspace}")
    before = agent.policy.handler
    result = await scheduler.run_job(job.job_id)
    assert result.success
    assert agent.policy.handler is before, "the interactive handler must come back"
    assert scheduler.get(job.job_id).last_status == "ok"


def test_disabled_jobs_do_not_run(agent: Agent, workspace: Path) -> None:
    import asyncio

    scheduler = Scheduler(agent)
    job = scheduler.add("listing", when="hourly", request=f"list {workspace}")
    scheduler.set_enabled(job.job_id, False)
    result = asyncio.run(scheduler.run_job(job.job_id))
    assert not result.success
    assert "disabled" in result.detail


# ---- observation -----------------------------------------------------------
def test_folder_watching_is_off_by_default(agent: Agent) -> None:
    from personalos.observers.folder_watcher import FolderWatcher

    watcher = FolderWatcher(agent.settings, agent.database)
    assert watcher.enabled is False
    assert watcher.poll() == []


def test_watcher_records_paths_not_contents(agent: Agent, workspace: Path) -> None:
    from personalos.observers.folder_watcher import FolderWatcher

    agent.settings.observation.watch_folders_enabled = True
    agent.settings.observation.watched_folders = [str(workspace)]
    watcher = FolderWatcher(agent.settings, agent.database)

    (workspace / "new.pdf").write_text("confidential contents")
    events = watcher.poll()
    assert len(events) == 1
    assert events[0].path.endswith("new.pdf")
    assert "confidential" not in str(events[0].detail)


def test_pausing_observation_stops_recording(agent: Agent, workspace: Path) -> None:
    from personalos.observers.folder_watcher import FolderWatcher

    agent.settings.observation.watch_folders_enabled = True
    agent.settings.observation.watched_folders = [str(workspace)]
    agent.settings.observation.paused = True
    watcher = FolderWatcher(agent.settings, agent.database)
    (workspace / "new.pdf").write_text("x")
    assert watcher.poll() == []


# ---- daily briefs ----------------------------------------------------------
async def test_briefs_render_without_executing_anything(agent: Agent, workspace: Path) -> None:
    from personalos.interface.daily import evening_brief, morning_brief

    await agent.ask(f"list {workspace}")
    assert "Good morning" in morning_brief(agent, Scheduler(agent)).render()
    assert "how today went" in evening_brief(agent).render()


# ---- CLI -------------------------------------------------------------------
@pytest.fixture
def cli(settings, monkeypatch):
    """A CLI runner bound to the throwaway home."""
    from personalos.interface import cli as cli_module

    cli_module.state.home = settings.home
    cli_module.state._settings = settings
    cli_module.state._agent = None
    cli_module.state.assume_yes = True
    monkeypatch.setenv("PERSONALOS_LLM_PROVIDER", "echo")
    # Rich truncates table cells to the terminal width; give it room so the
    # assertions below test the output rather than the ellipsis.
    monkeypatch.setenv("COLUMNS", "220")
    return CliRunner()


def test_cli_version(cli) -> None:
    from personalos.interface.cli import app

    result = cli.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "PersonalOS Agent" in result.stdout


def test_cli_doctor_runs(cli) -> None:
    from personalos.interface.cli import app

    result = cli.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "workspace" in result.stdout


def test_cli_status(cli) -> None:
    from personalos.interface.cli import app

    result = cli.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "IDLE" in result.stdout


def test_cli_lists_tools_with_risk_levels(cli) -> None:
    from personalos.interface.cli import app

    result = cli.invoke(app, ["tools"])
    assert result.exit_code == 0
    assert "filesystem.delete" in result.stdout


def test_cli_skills_list(cli) -> None:
    from personalos.interface.cli import app

    result = cli.invoke(app, ["skills", "list"])
    assert result.exit_code == 0
    assert "organize_downloads" in result.stdout


def test_cli_ask_and_history(cli, workspace: Path) -> None:
    from personalos.interface.cli import app

    asked = cli.invoke(app, ["ask", f"list {workspace}"])
    assert asked.exit_code == 0

    history = cli.invoke(app, ["history"])
    assert history.exit_code == 0
    assert "filesystem" in history.stdout


def test_cli_permissions_show_explains_the_policy(cli) -> None:
    from personalos.interface.cli import app

    result = cli.invoke(app, ["permissions", "show"])
    assert result.exit_code == 0
    assert "always asks" in result.stdout


def test_cli_memory_set_and_search(cli) -> None:
    from personalos.interface.cli import app

    assert cli.invoke(app, ["memory", "set", "report_format", "markdown"]).exit_code == 0
    result = cli.invoke(app, ["memory", "search", "report"])
    assert result.exit_code == 0
    assert "report_format" in result.stdout


def test_cli_observe_status_states_that_it_is_off(cli) -> None:
    from personalos.interface.cli import app

    result = cli.invoke(app, ["observe", "status"])
    assert result.exit_code == 0
    assert "off" in result.stdout


def test_init_replaces_the_placeholder_workspace(tmp_path, monkeypatch) -> None:
    """`agent init --workspace X` should not leave the unused default behind."""
    from typer.testing import CliRunner

    from personalos.interface import cli as cli_module
    from personalos.settings import load_settings

    home = tmp_path / "home"
    workspace = tmp_path / "chosen"
    workspace.mkdir()

    cli_module.state.home = home
    cli_module.state._settings = None
    cli_module.state._agent = None
    monkeypatch.setenv("COLUMNS", "220")

    result = CliRunner().invoke(cli_module.app, ["--home", str(home), "init", "--workspace", str(workspace)])
    assert result.exit_code == 0

    settings = load_settings(home=home, use_env=False)
    assert settings.workspace.allowed_roots == [str(workspace)]
