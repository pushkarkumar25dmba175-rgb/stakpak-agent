"""The PersonalOS command line.

Every command goes through :class:`CLIState`, which builds the agent lazily so
that commands needing no model (``history``, ``memory``, ``skills``) work on a
machine with no API key configured.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.markup import escape
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from personalos.agent.orchestrator import Agent
from personalos.agent.state import AgentState
from personalos.audit.audit_log import AuditQuery
from personalos.errors import PersonalOSError
from personalos.interface import console as ui
from personalos.interface.approvals import InteractiveApprovalHandler
from personalos.interface.console import console
from personalos.interface.daily import evening_brief, morning_brief
from personalos.interface.doctor import run_checks
from personalos.scheduler.scheduler import Scheduler
from personalos.security.risk import RiskLevel
from personalos.settings import (
    Settings,
    WorkspaceSettings,
    initialise_home,
    load_settings,
    save_settings,
)
from personalos.skills.schema import load_skill_file
from personalos.utils.textutil import truncate
from personalos.version import __version__


@dataclass
class CLIState:
    """Shared state for a single CLI invocation."""

    home: Path | None = None
    project: str | None = None
    assume_yes: bool = False
    overrides: dict[str, Any] = field(default_factory=dict)
    _settings: Settings | None = None
    _agent: Agent | None = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = load_settings(home=self.home, overrides=self.overrides or None)
            self._settings.ensure_directories()
        return self._settings

    @property
    def agent(self) -> Agent:
        if self._agent is None:
            handler = InteractiveApprovalHandler(assume_yes=self.assume_yes)
            self._agent = Agent.build(self.settings, approval_handler=handler)
            project = self.project or self.settings.active_project
            if project:
                self._agent.switch_project(project)
        return self._agent

    def scheduler(self) -> Scheduler:
        return Scheduler(self.agent)


state = CLIState()

app = typer.Typer(
    name="agent",
    help="PersonalOS Agent — a local AI coworker that stays under your control.",
    no_args_is_help=True,
    add_completion=False,
)
skills_app = typer.Typer(help="List, inspect and manage reusable skills.", no_args_is_help=True)
memory_app = typer.Typer(help="Inspect and edit what the agent remembers.", no_args_is_help=True)
schedule_app = typer.Typer(help="Schedule recurring work.", no_args_is_help=True)
permissions_app = typer.Typer(help="See and change standing approval rules.", no_args_is_help=True)
project_app = typer.Typer(help="Work within a project context.", no_args_is_help=True)
suggest_app = typer.Typer(help="Review what the agent has noticed.", no_args_is_help=True)
observe_app = typer.Typer(help="Optional folder and clipboard observation.", no_args_is_help=True)

app.add_typer(skills_app, name="skills")
app.add_typer(memory_app, name="memory")
app.add_typer(schedule_app, name="schedule")
app.add_typer(permissions_app, name="permissions")
app.add_typer(project_app, name="project")
app.add_typer(suggest_app, name="suggestions")
app.add_typer(observe_app, name="observe")


@app.callback()
def main_callback(
    home: Annotated[Path | None, typer.Option(help="PersonalOS home directory.")] = None,
    project: Annotated[str | None, typer.Option(help="Run within this project.")] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Auto-approve up to level 2. Levels 3 and 4 still ask.")] = False,
    provider: Annotated[str | None, typer.Option(help="Override the LLM provider for this run.")] = None,
    model: Annotated[str | None, typer.Option(help="Override the model for this run.")] = None,
) -> None:
    """Set up shared state before any command runs."""
    state.home = home
    state.project = project
    state.assume_yes = yes
    overrides: dict[str, Any] = {}
    if provider:
        overrides.setdefault("llm", {})["provider"] = provider
    if model:
        overrides.setdefault("llm", {})["model"] = model
    state.overrides = overrides


# ---------------------------------------------------------------- core verbs
@app.command()
def version() -> None:
    """Show the version."""
    console.print(f"PersonalOS Agent {__version__}")


@app.command()
def init(
    workspace: Annotated[Path | None, typer.Option(help="A folder the agent may work in.")] = None,
) -> None:
    """Create the agent's home directory and a starter configuration."""
    settings = initialise_home(state.home)
    if workspace is not None:
        default_roots = WorkspaceSettings().allowed_roots
        current = settings.workspace.allowed_roots
        if current == default_roots:
            # The untouched placeholder is replaced, not added to: warning about
            # a ~/PersonalOS the user never asked for is noise on a fresh setup.
            settings.workspace.allowed_roots = [str(workspace)]
        else:
            settings.workspace.allowed_roots = sorted(set(current) | {str(workspace)})
        save_settings(settings)
    ui.success(f"PersonalOS is set up in {settings.home}")
    console.print(f"  config:    {settings.config_file}")
    console.print(f"  database:  {settings.database_path}")
    console.print(f"  logs:      {settings.logs_dir}")
    console.print(f"  workspace: {', '.join(settings.workspace.allowed_roots)}")
    console.print("\nNext: [bold]agent doctor[/bold], then [bold]agent ask \"…\"[/bold].")


@app.command()
def ask(
    request: Annotated[str, typer.Argument(help="What you want done, in plain language.")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would happen without doing it.")] = False,
    show_plan: Annotated[bool, typer.Option("--show-plan/--no-show-plan", help="Print the plan before running it.")] = True,
) -> None:
    """Ask the agent to do something."""
    try:
        agent = state.agent
        outcome = asyncio.run(agent.ask(request, dry_run=dry_run))
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc

    if show_plan and not outcome.plan.is_empty():
        ui.render_plan(outcome.plan)
    ui.render_answer(outcome.answer, success_state=outcome.success)

    if outcome.retrieved and outcome.retrieved.items:
        console.print(f"[dim]Context used: {outcome.retrieved.summary()}[/dim]")
    console.print(f"[dim]Task {outcome.task_id} — `agent explain {outcome.task_id}` for detail.[/dim]")

    for suggestion in outcome.suggestions:
        console.print(
            Panel(
                f"{suggestion.title}\n\n{suggestion.detail}\n\n"
                f"[dim]Accept with `agent suggestions accept {suggestion.id}`, "
                f"or dismiss with `agent suggestions dismiss {suggestion.id}`.[/dim]",
                title="I noticed something",
                border_style="cyan",
            )
        )
    if not outcome.success:
        raise typer.Exit(code=1)


@app.command(name="plan")
def plan_command(
    request: Annotated[str, typer.Argument(help="What you want done.")],
) -> None:
    """Show the plan for a request without executing any of it."""
    try:
        outcome = asyncio.run(state.agent.ask(request, plan_only=True))
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    ui.render_plan(outcome.plan)


@app.command()
def run(
    skill: Annotated[str, typer.Argument(help="Name of the skill to run.")],
    inputs: Annotated[list[str] | None, typer.Option("--input", "-i", help="key=value, repeatable.")] = None,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    """Run a saved skill."""
    parsed: dict[str, Any] = {}
    for item in inputs or []:
        key, _, value = item.partition("=")
        if not key or not _:
            ui.warn(f"Ignoring {item!r}: inputs must look like key=value.")
            continue
        parsed[key.strip()] = value.strip()

    try:
        outcome = asyncio.run(state.agent.run_skill(skill, parsed, dry_run=dry_run))
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc

    ui.render_plan(outcome.plan, title=f"Skill: {skill}")
    ui.render_answer(outcome.answer, success_state=outcome.success)
    if not outcome.success:
        raise typer.Exit(code=1)


@app.command()
def status() -> None:
    """Show what the agent is doing and how it is configured."""
    snapshot = state.agent.status()
    body = Text()
    body.append("State: ")
    body.append(ui.state_text(AgentState(snapshot["state"])))
    body.append(f"\n{snapshot['state_description']}\n\n")
    body.append(f"Project:       {snapshot['project'] or '(none)'}\n")
    body.append(f"Model:         {snapshot['provider']}\n")
    body.append(f"Approval mode: {snapshot['approval_mode']}\n")
    body.append(f"Home:          {snapshot['home']}\n")
    body.append(f"Tools:         {', '.join(snapshot['tools'])}\n")
    body.append(f"Skills:        {snapshot['skills']}\n")
    body.append(
        "Memory:        "
        + ", ".join(f"{k} {v}" for k, v in snapshot["memory"].items())
        + "\n"
    )
    if snapshot["pending_suggestions"]:
        body.append(f"\n{snapshot['pending_suggestions']} suggestion(s) waiting for you.\n", style="cyan")
    console.print(Panel(body, title="PersonalOS", border_style="cyan"))

    if snapshot["recent_tasks"]:
        ui.render_table(
            "Recent tasks",
            ["Task", "Request", "Status"],
            [
                [task["task_id"], task["request"], task["status"]]
                for task in snapshot["recent_tasks"]
            ],
        )


@app.command()
def history(
    today: Annotated[bool, typer.Option("--today", help="Only today's actions.")] = False,
    task: Annotated[str | None, typer.Option("--task", help="Only this task id.")] = None,
    failures: Annotated[bool, typer.Option("--failures", help="Only failures.")] = False,
    limit: Annotated[int, typer.Option(help="How many rows to show.")] = 25,
) -> None:
    """Show what the agent has done."""
    from personalos.utils.timeutil import utcnow

    since = None
    if today:
        since = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    records = state.agent.audit.query(
        AuditQuery(task_id=task, since=since, only_failures=failures, limit=limit)
    )
    if not records:
        ui.info("No matching actions recorded.")
        return
    ui.render_table(
        "Action history",
        ["When", "Task", "Tool", "Action", "Risk", "Approval", "Result"],
        [
            [
                record.created_at.strftime("%m-%d %H:%M"),
                record.task_id[:8],
                record.tool,
                truncate(record.action, 40),
                ui.risk_text(RiskLevel(record.risk_level)),
                record.approval_status.replace("_", " "),
                Text("ok", style="green") if record.success else Text("failed", style="red"),
            ]
            for record in records
        ],
    )


@app.command()
def explain(task_id: Annotated[str, typer.Argument(help="Task id from `agent history`.")]) -> None:
    """Explain everything that happened in one task."""
    agent = state.agent
    resolved = agent.tasks.resolve_id(task_id)
    task = agent.tasks.get(resolved) if resolved else None
    if task is None:
        ui.warn(
            f"No task matches {task_id!r}."
            + (" That prefix matches more than one task." if len(task_id) < 16 else "")
        )
        raise typer.Exit(code=1)
    task_id = task.task_id

    console.print(
        Panel(
            f"{task.request}\n\n"
            f"Status: {task.status}\n"
            f"Started: {task.created_at:%Y-%m-%d %H:%M}\n"
            f"Highest risk: {RiskLevel(task.max_risk_level).label}\n"
            + (f"\n{task.result_summary}" if task.result_summary else ""),
            title=f"Task {task.task_id}",
            border_style="cyan",
        )
    )

    records = agent.audit.query(AuditQuery(task_id=task_id, limit=200))
    for record in reversed(records):
        verification = (
            record.verification_note
            if record.verified is not None
            else "not automatically verified"
        )
        console.print(
            f"[bold]{record.action}[/bold]\n"
            f"  tool:        {record.tool}\n"
            f"  risk:        level {record.risk_level}\n"
            f"  approval:    {record.approval_status}\n"
            f"  took:        {record.execution_time_ms:.0f} ms\n"
            f"  verified:    {verification}\n"
            + (f"  [red]error: {record.error}[/red]\n" if record.error else "")
        )

    entries = agent.journal.entries_for(task_id, include_applied=True)
    if entries:
        ui.render_table(
            "Reversible changes",
            ["#", "Operation", "Undo", "Applied"],
            [
                [
                    entry.id,
                    entry.operation,
                    str(entry.rollback.get("operation", "-")),
                    "yes" if entry.applied else "no",
                ]
                for entry in entries
            ],
        )
        console.print(f"[dim]Undo with `agent rollback {task_id}`.[/dim]")


@app.command()
def rollback(
    task_id: Annotated[str, typer.Argument(help="Task to undo.")],
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be undone.")] = False,
) -> None:
    """Undo the changes a task made."""
    try:
        resolved = state.agent.tasks.resolve_id(task_id) or task_id
        results = state.agent.rollback(resolved, dry_run=dry_run)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc

    for result in results:
        marker = "[green]✓[/green]" if result.applied else "[yellow]•[/yellow]"
        console.print(f"{marker} {result.detail}")
    if dry_run:
        console.print("[dim]Nothing was changed. Re-run without --dry-run to apply.[/dim]")


@app.command()
def feedback(
    message: Annotated[str, typer.Argument(help='For example: "always save reports as Markdown".')],
    task: Annotated[str | None, typer.Option("--task", help="The task this is about.")] = None,
    path: Annotated[str | None, typer.Option("--path", help="Limit a rule to this folder.")] = None,
) -> None:
    """Tell the agent something it should remember."""
    try:
        item = asyncio.run(state.agent.record_feedback(message, task_id=task, path_scope=path))
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    ui.success(item.render())
    if item.explanation and item.explanation != item.render():
        console.print(f"[dim]{item.explanation}[/dim]")


@app.command()
def pause() -> None:
    """Stop the agent from starting any new task."""
    state.agent.state.pause()
    ui.success("Paused. No task will start until you run `agent resume`.")


@app.command()
def resume() -> None:
    """Resume after a pause."""
    state.agent.state.resume()
    ui.success("Resumed.")


@app.command()
def doctor() -> None:
    """Check the installation and report anything that needs attention."""
    settings = state.settings
    try:
        agent: Agent | None = state.agent
    except PersonalOSError:
        agent = None

    checks = run_checks(settings, agent)
    for check in checks:
        # Details contain things like `personalos-agent[docs]`, which Rich would
        # otherwise read as markup and swallow.
        console.print(
            f"[{check.style}]{check.symbol}[/{check.style}] {check.name:<20} "
            + escape(check.detail)
        )

    failures = [check for check in checks if check.status == "fail"]
    if failures:
        console.print(f"\n[red]{len(failures)} check(s) failed.[/red]")
        raise typer.Exit(code=1)
    console.print("\n[green]Ready.[/green]")


@app.command()
def tools() -> None:
    """List the tools the agent can use, and what each one risks."""
    rows = []
    for tool in state.agent.registry.all():
        for name, spec in tool.operations.items():
            rows.append(
                [
                    f"{tool.name}.{name}",
                    ui.risk_text(spec.risk),
                    "yes" if spec.reversible else "no",
                    truncate(spec.description, 60),
                ]
            )
    ui.render_table("Tools", ["Operation", "Risk", "Reversible", "Description"], rows)


@app.command()
def morning() -> None:
    """The morning brief: what is scheduled and what is open."""
    console.print(Panel(morning_brief(state.agent, state.scheduler()).render(), border_style="cyan"))


@app.command()
def evening() -> None:
    """The evening brief: what happened today and what keeps repeating."""
    console.print(Panel(evening_brief(state.agent).render(), border_style="cyan"))


@app.command()
def dashboard(
    port: Annotated[int | None, typer.Option(help="Port to listen on.")] = None,
) -> None:
    """Start the local dashboard (localhost only)."""
    from personalos.dashboard.app import serve

    settings = state.settings
    if port is not None:
        settings.dashboard.port = port
    try:
        serve(state.agent, settings)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc


# -------------------------------------------------------------------- skills
@skills_app.command("list")
def skills_list() -> None:
    """List every skill the agent knows."""
    summaries = state.agent.skills.list_skills()
    if not summaries:
        ui.info("No skills yet. Built-in examples live in the package; add your own to "
                f"{state.settings.skills_dir}.")
        return
    ui.render_table(
        "Skills",
        ["Name", "Version", "Origin", "Enabled", "Runs", "Success", "Description"],
        [
            [
                item.definition.name,
                item.definition.version,
                item.origin,
                "yes" if item.enabled else "no",
                item.run_count,
                f"{item.success_rate:.0%}" if item.success_rate is not None else "—",
                truncate(item.definition.description, 50),
            ]
            for item in summaries
        ],
    )
    for path, problem in state.agent.skills.registry.errors.items():
        ui.warn(f"{path.name} could not be loaded: {problem}")


@skills_app.command("inspect")
def skills_inspect(name: Annotated[str, typer.Argument()]) -> None:
    """Show a skill's full definition."""
    try:
        item = state.agent.skills.inspect(name)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    console.print(f"[dim]{item.path}[/dim]")
    console.print(Syntax(item.definition.to_yaml(), "yaml", theme="ansi_dark"))


@skills_app.command("import")
def skills_import(path: Annotated[Path, typer.Argument(help="A skill YAML file.")]) -> None:
    """Validate and install a skill from a file."""
    try:
        destination = state.agent.skills.import_file(path)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    ui.success(f"Installed to {destination}")


@skills_app.command("version")
def skills_version(
    name: Annotated[str, typer.Argument()],
    path: Annotated[Path, typer.Argument(help="The updated definition.")],
) -> None:
    """Replace a skill, keeping the previous version on disk."""
    try:
        definition = load_skill_file(path)
        destination = state.agent.skills.version(name, definition)
        updated = state.agent.skills.inspect(name).definition
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    ui.success(f"{name} is now version {updated.version} at {destination}")
    console.print("[dim]The previous version is kept in the `versions/` folder beside it.[/dim]")


@skills_app.command("enable")
def skills_enable(name: Annotated[str, typer.Argument()]) -> None:
    """Re-enable a disabled skill."""
    state.agent.skills.set_enabled(name, True)
    ui.success(f"{name} is enabled.")


@skills_app.command("disable")
def skills_disable(name: Annotated[str, typer.Argument()]) -> None:
    """Stop a skill from running, without deleting it."""
    state.agent.skills.set_enabled(name, False)
    ui.success(f"{name} is disabled.")


@skills_app.command("delete")
def skills_delete(
    name: Annotated[str, typer.Argument()],
    keep_backup: Annotated[bool, typer.Option("--keep-backup/--no-keep-backup")] = True,
) -> None:
    """Delete a skill."""
    if not typer.confirm(f"Delete the skill {name!r}?"):
        raise typer.Abort()
    try:
        path = state.agent.skills.delete(name, keep_backup=keep_backup)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    ui.success(f"Removed {path}" + (" (a copy is kept in `deleted/`)" if keep_backup else ""))


# -------------------------------------------------------------------- memory
@memory_app.command("show")
def memory_show() -> None:
    """Summarise everything the agent remembers."""
    agent = state.agent
    stats = agent.memory.statistics()
    console.print(
        Panel(
            "\n".join(f"{key}: {value}" for key, value in stats.items()),
            title="Memory",
            border_style="cyan",
        )
    )
    preferences = agent.memory.preferences.all()
    if preferences:
        ui.render_table(
            "Preferences",
            ["Key", "Value", "Source", "Confidence", "Used"],
            [
                [
                    item.key,
                    str(item.value),
                    "you told me" if item.explicit else item.source,
                    f"{item.effective_confidence:.0%}",
                    item.usage_count,
                ]
                for item in preferences
            ],
        )


@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument()],
    limit: Annotated[int, typer.Option()] = 10,
) -> None:
    """Search memory the way the agent does when planning."""
    retrieved = state.agent.memory.retrieve(query, project=state.agent.namespace, top_k=limit)
    if not retrieved.items:
        ui.info("Nothing relevant found.")
        return
    ui.render_table(
        f"Memory matching {query!r}",
        ["Kind", "Score", "Item"],
        [[item.kind, f"{item.score:.2f}", truncate(item.text, 90)] for item in retrieved.items],
    )
    console.print(f"[dim]{retrieved.summary()}[/dim]")


@memory_app.command("set")
def memory_set(
    key: Annotated[str, typer.Argument()],
    value: Annotated[str, typer.Argument()],
    namespace: Annotated[str, typer.Option()] = "global",
) -> None:
    """State a preference explicitly. Explicit always beats inferred."""
    resolved = state.agent.memory.learn_preference(key, value, explicit=True, namespace=namespace)
    ui.success(f"Stored {resolved.render()}")


@memory_app.command("forget")
def memory_forget(
    key: Annotated[str, typer.Argument(help="Preference key, fact key, or episode id.")],
    namespace: Annotated[str, typer.Option()] = "global",
) -> None:
    """Delete something from memory."""
    agent = state.agent
    if key.isdigit() and agent.memory.episodic.forget(int(key)):
        ui.success(f"Forgot episode {key}.")
        return
    if agent.memory.preferences.forget(key, namespace=namespace):
        ui.success(f"Forgot the preference {key!r}.")
        return
    if agent.memory.semantic.forget(key, namespace=namespace):
        ui.success(f"Forgot the fact {key!r}.")
        return
    ui.warn(f"Nothing stored under {key!r} in the {namespace!r} namespace.")


@memory_app.command("workflows")
def memory_workflows() -> None:
    """Show the repeated workflows the agent has noticed."""
    workflows = state.agent.memory.workflows.all()
    if not workflows:
        ui.info("No repeated workflows noticed yet.")
        return
    ui.render_table(
        "Workflows",
        ["Name", "Seen", "Promoted to skill", "Shape"],
        [
            [
                workflow.name,
                workflow.success_count,
                workflow.promoted_skill or "—",
                truncate(workflow.signature, 60),
            ]
            for workflow in workflows
        ],
    )


# ------------------------------------------------------------------ schedule
@schedule_app.command("add")
def schedule_add(
    name: Annotated[str, typer.Argument(help="A name for the job.")],
    when: Annotated[str, typer.Option("--when", help='e.g. "every day at 18:00".')],
    ask_text: Annotated[str | None, typer.Option("--ask", help="A request to run.")] = None,
    skill: Annotated[str | None, typer.Option("--skill", help="A skill to run.")] = None,
) -> None:
    """Schedule recurring work."""
    try:
        definition = state.scheduler().add(name, when=when, request=ask_text, skill=skill)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    ui.success(f"Scheduled: {definition.describe()}")
    console.print(
        f"[dim]It will run unattended, approving at most "
        f"{RiskLevel(definition.max_risk_level).label} actions. "
        "Anything riskier will stop and wait for you.[/dim]"
    )


@schedule_app.command("list")
def schedule_list() -> None:
    """List scheduled jobs."""
    scheduler = state.scheduler()
    jobs = scheduler.list_jobs()
    if not jobs:
        ui.info("Nothing scheduled.")
        return
    ui.render_table(
        "Scheduled jobs",
        ["Id", "Name", "When", "Kind", "Ceiling", "Enabled", "Last run"],
        [
            [
                job.job_id,
                job.name,
                job.spec,
                job.kind,
                f"level {job.max_risk_level}",
                "yes" if job.enabled else "no",
                job.last_run.strftime("%m-%d %H:%M") if job.last_run else "—",
            ]
            for job in jobs
        ],
    )


@schedule_app.command("run")
def schedule_run(job_id: Annotated[str, typer.Argument()]) -> None:
    """Run a scheduled job now."""
    result = asyncio.run(state.scheduler().run_job(job_id))
    ui.render_answer(result.detail, success_state=result.success)


@schedule_app.command("remove")
def schedule_remove(job_id: Annotated[str, typer.Argument()]) -> None:
    """Delete a scheduled job."""
    if state.scheduler().remove(job_id):
        ui.success(f"Removed job {job_id}.")
    else:
        ui.warn(f"No job with id {job_id}.")


@schedule_app.command("enable")
def schedule_enable(job_id: Annotated[str, typer.Argument()]) -> None:
    """Enable a scheduled job."""
    ui.success("Enabled.") if state.scheduler().set_enabled(job_id, True) else ui.warn("No such job.")


@schedule_app.command("disable")
def schedule_disable(job_id: Annotated[str, typer.Argument()]) -> None:
    """Disable a scheduled job."""
    ui.success("Disabled.") if state.scheduler().set_enabled(job_id, False) else ui.warn("No such job.")


@schedule_app.command("serve")
def schedule_serve() -> None:
    """Run the scheduler in the foreground until interrupted."""
    scheduler = state.scheduler()
    if not scheduler.start():
        ui.warn(
            "The scheduler could not start. Install APScheduler with "
            "`pip install 'personalos-agent[scheduler]'`, or enable scheduler.enabled in config."
        )
        raise typer.Exit(code=1)
    ui.success("Scheduler running. Press Ctrl-C to stop.")
    try:
        import time

        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        scheduler.shutdown()
        ui.info("Stopped.")


# --------------------------------------------------------------- permissions
@permissions_app.command("show")
def permissions_show() -> None:
    """Show what the agent may reach and which rules are in force."""
    agent = state.agent
    described = agent.paths.describe()
    console.print(
        Panel(
            "Allowed:\n"
            + "\n".join(f"  {path}" for path in described["allowed_roots"])
            + "\n\nNever accessible:\n"
            + "\n".join(f"  {path}" for path in described["denied_paths"]),
            title="Filesystem access",
            border_style="cyan",
        )
    )
    console.print(
        Panel(
            "\n".join(
                f"level {int(level)} ({level.label}): {behaviour}"
                for level, behaviour in _policy_summary(agent.settings).items()
            ),
            title="Approval policy",
            border_style="cyan",
        )
    )
    rules = agent.policy.list_rules()
    if rules:
        ui.render_table(
            "Standing rules",
            ["Name", "Effect", "Tool", "Operation", "Path", "Up to"],
            [
                [
                    rule.name,
                    rule.effect,
                    rule.tool or "any",
                    rule.operation or "any",
                    rule.path_prefix or "any",
                    f"level {rule.max_risk_level}",
                ]
                for rule in rules
            ],
        )


def _policy_summary(settings: Settings) -> dict[RiskLevel, str]:
    ceiling = RiskLevel.parse(settings.security.auto_approve_max_level)
    summary = {}
    for level in RiskLevel:
        if level.always_requires_confirmation:
            summary[level] = "always asks — no rule or habit can waive this"
        elif level <= ceiling:
            summary[level] = "runs automatically"
        else:
            summary[level] = "asks, unless a standing rule covers it"
    return summary


@permissions_app.command("allow")
def permissions_allow(
    name: Annotated[str, typer.Argument(help="A name for the rule.")],
    tool: Annotated[str | None, typer.Option(help="Limit to one tool.")] = None,
    operation: Annotated[str | None, typer.Option(help="Limit to one operation.")] = None,
    path: Annotated[str | None, typer.Option(help="Limit to paths under this prefix.")] = None,
    level: Annotated[int, typer.Option(help="Highest level this covers (max 2).")] = 2,
) -> None:
    """Grant a standing approval for routine actions."""
    rule = state.agent.policy.add_rule(
        name, effect="allow", tool=tool, operation=operation, path_prefix=path, max_risk_level=level
    )
    ui.success(f"Rule {rule.name!r} added, covering up to level {rule.max_risk_level}.")
    console.print(
        "[dim]Deletions, installs and anything leaving this machine still ask every time.[/dim]"
    )


@permissions_app.command("deny")
def permissions_deny(
    name: Annotated[str, typer.Argument()],
    tool: Annotated[str | None, typer.Option()] = None,
    operation: Annotated[str | None, typer.Option()] = None,
    path: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Forbid something outright."""
    rule = state.agent.policy.add_rule(
        name, effect="deny", tool=tool, operation=operation, path_prefix=path
    )
    ui.success(f"Rule {rule.name!r} added. This will be refused without asking.")


@permissions_app.command("revoke")
def permissions_revoke(name: Annotated[str, typer.Argument()]) -> None:
    """Remove a standing rule."""
    if state.agent.policy.revoke_rule(name):
        ui.success(f"Revoked {name!r}.")
    else:
        ui.warn(f"No rule called {name!r}.")


# ------------------------------------------------------------------ projects
@project_app.command("list")
def project_list() -> None:
    """List projects."""
    projects = state.agent.projects.list_projects()
    if not projects:
        ui.info("No projects yet. Create one with `agent project create <name>`.")
        return
    ui.render_table(
        "Projects",
        ["Name", "Description", "Directories"],
        [
            [
                project.definition.name,
                truncate(project.definition.description, 50),
                ", ".join(project.definition.directories) or "—",
            ]
            for project in projects
        ],
    )


@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Argument()],
    description: Annotated[str, typer.Option()] = "",
    directory: Annotated[list[str] | None, typer.Option("--directory", "-d", help="Repeatable.")] = None,
) -> None:
    """Create a project."""
    try:
        project = state.agent.projects.create(
            name, description=description, directories=list(directory or [])
        )
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    ui.success(f"Created {project.definition.name} at {project.path}")


@project_app.command("switch")
def project_switch(name: Annotated[str, typer.Argument()]) -> None:
    """Make a project the default for future commands."""
    try:
        state.agent.switch_project(name)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    settings = state.settings
    settings.active_project = name
    save_settings(settings)
    ui.success(f"Switched to {name}. Its directories are now reachable and its memory is active.")


@project_app.command("show")
def project_show(name: Annotated[str | None, typer.Argument()] = None) -> None:
    """Show a project's definition."""
    agent = state.agent
    target = name or agent.project_name
    if not target:
        ui.info("No project is active.")
        return
    try:
        project = agent.projects.load(target)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    console.print(Panel(project.render(), title=project.definition.name, border_style="cyan"))


# --------------------------------------------------------------- suggestions
@suggest_app.command("list")
def suggestions_list() -> None:
    """Show what the agent has noticed and is waiting to ask about."""
    pending = state.agent.suggestions.pending()
    if not pending:
        ui.info("Nothing to suggest right now.")
        return
    for suggestion in pending:
        console.print(
            Panel(
                f"{suggestion.title}\n\n{suggestion.detail}",
                title=f"#{suggestion.id} ({suggestion.kind}, {suggestion.confidence:.0%} confident)",
                border_style="cyan",
            )
        )


@suggest_app.command("accept")
def suggestions_accept(
    suggestion_id: Annotated[int, typer.Argument()],
    name: Annotated[str | None, typer.Option("--name", help="Name for the new skill.")] = None,
) -> None:
    """Accept a suggestion, creating a skill you can read before running."""
    try:
        definition = state.agent.accept_suggestion(suggestion_id, skill_name=name)
    except PersonalOSError as exc:
        ui.error(exc)
        raise typer.Exit(code=1) from exc
    ui.success(f"Created the skill {definition.name!r}.")
    console.print(Syntax(definition.to_yaml(), "yaml", theme="ansi_dark"))
    console.print(
        f"[dim]Read it, edit it if you like, then run it with `agent run {definition.name}`. "
        "It still asks for approval exactly as before.[/dim]"
    )


@suggest_app.command("dismiss")
def suggestions_dismiss(suggestion_id: Annotated[int, typer.Argument()]) -> None:
    """Dismiss a suggestion."""
    if state.agent.dismiss_suggestion(suggestion_id):
        ui.success("Dismissed. I will not raise it again for a while.")
    else:
        ui.warn(f"No suggestion #{suggestion_id}.")


@suggest_app.command("scan")
def suggestions_scan() -> None:
    """Look for repeated workflows now instead of waiting for the next task."""
    observations = state.agent.detector.scan()
    if not observations:
        ui.info("No repeated workflows found yet.")
        return
    ui.render_table(
        "Observed workflows",
        ["Workflow", "Times", "Suggested"],
        [
            [
                observation.workflow.name,
                observation.pattern.occurrences,
                "yes" if observation.suggestion else "no",
            ]
            for observation in observations
        ],
    )


# ------------------------------------------------------------------- observe
@observe_app.command("status")
def observe_status() -> None:
    """Show what is being observed, if anything."""
    observation = state.settings.observation
    console.print(
        Panel(
            f"Folder watching: {'on' if observation.watch_folders_enabled else 'off'}\n"
            f"Watched folders: {', '.join(observation.watched_folders) or '(none)'}\n"
            f"Clipboard:       {'on' if observation.watch_clipboard_enabled else 'off'}\n"
            f"App usage:       {'on' if observation.track_app_usage else 'off'}\n"
            f"Paused:          {'yes' if observation.paused else 'no'}",
            title="Observation",
            border_style="cyan",
        )
    )
    console.print(
        "[dim]All observation is off unless you turn it on in config.yaml, and it only "
        "ever records that a file changed — never its contents.[/dim]"
    )


@observe_app.command("events")
def observe_events(limit: Annotated[int, typer.Option()] = 25) -> None:
    """Show recent observation events."""
    from personalos.observers.folder_watcher import FolderWatcher

    watcher = FolderWatcher(state.settings, state.agent.database)
    events = watcher.events(limit=limit)
    if not events:
        ui.info("Nothing recorded.")
        return
    ui.render_table(
        "Observed events",
        ["When", "Type", "Path", "Handled"],
        [
            [
                event.created_at.strftime("%m-%d %H:%M"),
                event.event_type,
                truncate(event.path or "", 70),
                "yes" if event.handled else "no",
            ]
            for event in events
        ],
    )


@observe_app.command("poll")
def observe_poll() -> None:
    """Scan the watched folders once and record what is new."""
    from personalos.observers.folder_watcher import FolderWatcher

    watcher = FolderWatcher(state.settings, state.agent.database)
    if not watcher.enabled:
        ui.warn("Folder observation is off. Enable it in config.yaml first.")
        raise typer.Exit(code=1)
    events = watcher.poll()
    ui.success(f"Recorded {len(events)} event(s).")


@observe_app.command("pause")
def observe_pause(
    resume_it: Annotated[bool, typer.Option("--resume", help="Resume instead of pausing.")] = False,
) -> None:
    """Pause or resume all observation."""
    settings = state.settings
    settings.observation.paused = not resume_it
    save_settings(settings)
    ui.success("Observation resumed." if resume_it else "Observation paused.")


@app.command("config")
def show_config(
    as_json: Annotated[bool, typer.Option("--json", help="Print as JSON.")] = False,
) -> None:
    """Show the effective configuration."""
    from personalos.settings import dump_settings

    data = dump_settings(state.settings)
    data["home"] = str(state.settings.home)
    if as_json:
        console.print_json(json.dumps(data))
        return
    console.print(Syntax(json.dumps(data, indent=2), "json", theme="ansi_dark"))
    console.print(f"[dim]Edit {state.settings.config_file} to change any of this.[/dim]")


def main() -> None:
    """Console-script entry point."""
    try:
        app()
    except PersonalOSError as exc:  # pragma: no cover - top-level safety net
        ui.error(exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
