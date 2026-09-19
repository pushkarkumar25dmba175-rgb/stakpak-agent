"""Daily assistant mode: the morning and evening briefs.

Both briefs are strictly read-only. They tell you what is scheduled, what is
outstanding and what the agent noticed — and then stop. Nothing in daily mode
executes anything, which is what keeps "the agent starts with my computer" from
meaning "the agent starts doing things when my computer starts".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from personalos.agent.orchestrator import Agent
from personalos.database.models import Suggestion, Task
from personalos.scheduler.scheduler import Scheduler
from personalos.utils.timeutil import utcnow


@dataclass
class Brief:
    """A rendered brief, section by section."""

    title: str
    sections: list[tuple[str, list[str]]] = field(default_factory=list)

    def add(self, heading: str, lines: list[str]) -> None:
        if lines:
            self.sections.append((heading, lines))

    def render(self) -> str:
        if not self.sections:
            return f"{self.title}\n\nNothing to report."
        blocks = [self.title, ""]
        for heading, lines in self.sections:
            blocks.append(heading)
            blocks.extend(f"  • {line}" for line in lines)
            blocks.append("")
        return "\n".join(blocks).rstrip()


def _task_line(task: Task) -> str:
    mark = {True: "done", False: "failed", None: task.status}[task.success]
    return f"[{mark}] {task.request[:90]}"


def _suggestion_line(suggestion: Suggestion) -> str:
    return f"#{suggestion.id} {suggestion.title}"


def morning_brief(agent: Agent, scheduler: Scheduler | None = None) -> Brief:
    """What is coming up today, and what is still open."""
    brief = Brief(title="Good morning. Here is where things stand.")

    if scheduler is not None:
        jobs = scheduler.list_jobs(include_disabled=False)
        next_runs = scheduler.next_run_times()
        brief.add(
            "Scheduled today",
            [
                f"{job.name} — {job.spec}"
                + (f" (next: {next_runs[job.job_id]:%H:%M})" if next_runs.get(job.job_id) else "")
                for job in jobs
            ],
        )

    brief.add("Still open", [_task_line(task) for task in agent.tasks.unfinished()[:8]])
    brief.add(
        "Waiting on your answer",
        [_suggestion_line(item) for item in agent.suggestions.pending(limit=5)],
    )

    if agent.active_project:
        definition = agent.active_project.definition
        brief.add(
            f"Active project: {definition.name}",
            [definition.description or "(no description)"] + list(definition.rules),
        )
    return brief


def evening_brief(agent: Agent) -> Brief:
    """What happened today, what did not finish, and what keeps repeating."""
    brief = Brief(title="Here is how today went.")
    today = agent.tasks.recent(limit=50, today_only=True)

    done = [task for task in today if task.success]
    failed = [task for task in today if task.success is False]

    brief.add("Completed", [_task_line(task) for task in done[:10]])
    brief.add("Did not finish", [_task_line(task) for task in failed[:10]])
    brief.add("Still open", [_task_line(task) for task in agent.tasks.unfinished()[:5]])

    cutoff = utcnow() - timedelta(days=14)
    repeats = [
        f"{workflow.name}: seen {workflow.success_count} times"
        + (f" (already a skill: {workflow.promoted_skill})" if workflow.promoted_skill else "")
        for workflow in agent.memory.workflows.all()
        if workflow.success_count >= 2 and workflow.last_seen >= cutoff
    ]
    brief.add("Things you repeat", repeats[:5])
    brief.add(
        "Suggestions waiting",
        [_suggestion_line(item) for item in agent.suggestions.pending(limit=5)],
    )
    return brief
