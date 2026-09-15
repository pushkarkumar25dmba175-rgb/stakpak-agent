"""Local job scheduling.

Two properties matter more than the mechanics:

* **A scheduled job runs unattended, so nobody is there to approve anything.**
  Jobs therefore execute with a handler that auto-approves only up to the job's
  own ceiling — clamped by ``scheduler.skip_jobs_above_level``, which itself
  cannot exceed level 2. A job whose plan turns out to need level 3 stops and
  records why, and the user sees it in `agent history`.
* **Jobs survive restarts.** They live in the database, not in the scheduler's
  memory, and are reloaded on start.

APScheduler is optional; without it the store still works and
:meth:`Scheduler.run_due` lets an external cron driver fire jobs instead.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from sqlalchemy import select

from personalos.agent.orchestrator import Agent
from personalos.database.database import Database
from personalos.database.models import ScheduledJob
from personalos.errors import ConfigurationError
from personalos.scheduler.jobs import JobDefinition, ScheduleSpec, parse_schedule
from personalos.security.permissions import AutoApproveHandler
from personalos.security.policy_engine import PolicyEngine
from personalos.security.risk import RiskLevel
from personalos.utils.timeutil import utcnow


@dataclass
class JobRun:
    """The outcome of one firing."""

    job_id: str
    success: bool
    detail: str
    task_id: str | None = None


class Scheduler:
    """Stores jobs and, when APScheduler is installed, runs them."""

    def __init__(self, agent: Agent, database: Database | None = None) -> None:
        self.agent = agent
        self.database = database or agent.database
        self.settings = agent.settings
        self._scheduler: Any | None = None

    # ---- job storage -------------------------------------------------------
    def add(
        self,
        name: str,
        *,
        when: str,
        request: str | None = None,
        skill: str | None = None,
        inputs: dict[str, Any] | None = None,
        max_risk_level: int | None = None,
    ) -> JobDefinition:
        """Create a job from a plain-English schedule.

        Raises:
            ConfigurationError: for an unparseable schedule or a job that
                specifies neither a request nor a skill.
        """
        if not request and not skill:
            raise ConfigurationError(
                "A scheduled job needs either a request or a skill to run.",
                remediation='For example: agent schedule "tidy up" --when "every day at 18:00" --ask "organize my downloads".',
            )
        spec = parse_schedule(when)
        ceiling = self._ceiling(max_risk_level)

        definition = JobDefinition(
            job_id=uuid.uuid4().hex[:12],
            name=name,
            kind="skill" if skill else "ask",
            payload={"skill": skill, "inputs": inputs or {}} if skill else {"request": request},
            schedule=spec,
            max_risk_level=int(ceiling),
            project=self.agent.project_name,
        )
        with self.database.session() as session:
            session.add(
                ScheduledJob(
                    job_id=definition.job_id,
                    name=definition.name,
                    kind=definition.kind,
                    payload=definition.payload,
                    schedule_kind=spec.kind,
                    schedule=spec.fields,
                    spec=when,
                    max_risk_level=definition.max_risk_level,
                    project=definition.project,
                )
            )
        if self._scheduler is not None:
            self._register(definition)
        return definition

    def _ceiling(self, requested: int | None) -> RiskLevel:
        """Clamp a job's autonomy to what unattended execution may ever do."""
        configured = RiskLevel.parse(self.settings.scheduler.skip_jobs_above_level)
        hard_cap = RiskLevel.MODERATE
        ceiling = min(configured, hard_cap)
        if requested is not None:
            ceiling = min(ceiling, RiskLevel.parse(requested))
        return ceiling

    def list_jobs(self, *, include_disabled: bool = True) -> list[ScheduledJob]:
        with self.database.session() as session:
            statement = select(ScheduledJob).order_by(ScheduledJob.created_at)
            if not include_disabled:
                statement = statement.where(ScheduledJob.enabled.is_(True))
            jobs = list(session.scalars(statement))
            for job in jobs:
                session.expunge(job)
            return jobs

    def get(self, job_id: str) -> ScheduledJob | None:
        with self.database.session() as session:
            job = session.scalar(select(ScheduledJob).where(ScheduledJob.job_id == job_id))
            if job is not None:
                session.expunge(job)
            return job

    def set_enabled(self, job_id: str, enabled: bool) -> bool:
        with self.database.session() as session:
            job = session.scalar(select(ScheduledJob).where(ScheduledJob.job_id == job_id))
            if job is None:
                return False
            job.enabled = enabled
        if self._scheduler is not None:
            self._sync()
        return True

    def remove(self, job_id: str) -> bool:
        with self.database.session() as session:
            job = session.scalar(select(ScheduledJob).where(ScheduledJob.job_id == job_id))
            if job is None:
                return False
            session.delete(job)
        if self._scheduler is not None:
            try:
                self._scheduler.remove_job(job_id)
            except Exception:  # noqa: BLE001 - the job may never have been registered
                pass
        return True

    # ---- execution ---------------------------------------------------------
    async def run_job(self, job_id: str) -> JobRun:
        """Fire one job now, under an unattended approval ceiling."""
        job = self.get(job_id)
        if job is None:
            return JobRun(job_id, False, f"No job with id {job_id}.")
        if not job.enabled:
            return JobRun(job_id, False, f"The job {job.name!r} is disabled.")

        ceiling = self._ceiling(job.max_risk_level)
        original_handler = self.agent.policy.handler
        # Scheduled work runs with nobody watching, so the handler is swapped for
        # one that can only ever approve up to the job's ceiling — and never
        # above level 2, whatever the job asks for.
        self.agent.policy.handler = AutoApproveHandler(ceiling)
        try:
            if job.kind == "skill":
                outcome = await self.agent.run_skill(
                    str(job.payload.get("skill")),
                    dict(job.payload.get("inputs") or {}),
                    origin="scheduler",
                )
            else:
                outcome = await self.agent.ask(
                    str(job.payload.get("request", "")), origin="scheduler"
                )
        finally:
            self.agent.policy.handler = original_handler

        with self.database.session() as session:
            row = session.scalar(select(ScheduledJob).where(ScheduledJob.job_id == job_id))
            if row is not None:
                row.last_run = utcnow()
                row.last_status = "ok" if outcome.success else "failed"

        return JobRun(
            job_id=job_id,
            success=outcome.success,
            detail=outcome.answer,
            task_id=outcome.task_id,
        )

    def _run_job_sync(self, job_id: str) -> None:
        """Entry point APScheduler calls from its worker thread."""
        asyncio.run(self.run_job(job_id))

    # ---- APScheduler integration ------------------------------------------
    def start(self) -> bool:
        """Start the background scheduler. Returns False when unavailable."""
        if not self.settings.scheduler.enabled:
            return False
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            return False
        if self._scheduler is not None:
            return True
        self._scheduler = BackgroundScheduler(timezone=self.settings.scheduler.timezone)
        self._sync()
        self._scheduler.start()
        return True

    def shutdown(self) -> None:
        if self._scheduler is not None:
            self._scheduler.shutdown(wait=False)
            self._scheduler = None

    def _sync(self) -> None:
        """Make the running scheduler match the database."""
        if self._scheduler is None:
            return
        for job in self._scheduler.get_jobs():
            job.remove()
        for row in self.list_jobs(include_disabled=False):
            self._register(
                JobDefinition(
                    job_id=row.job_id,
                    name=row.name,
                    kind="skill" if row.kind == "skill" else "ask",
                    payload=dict(row.payload or {}),
                    schedule=ScheduleSpec(
                        "interval" if row.schedule_kind == "interval" else "cron",
                        dict(row.schedule or {}),
                        row.spec,
                    ),
                    max_risk_level=row.max_risk_level,
                    project=row.project,
                )
            )

    def _register(self, definition: JobDefinition) -> None:
        if self._scheduler is None:
            return
        self._scheduler.add_job(
            self._run_job_sync,
            trigger=definition.schedule.kind,
            args=[definition.job_id],
            id=definition.job_id,
            name=definition.name,
            replace_existing=True,
            max_instances=self.settings.scheduler.max_concurrent_jobs,
            **definition.schedule.fields,
        )

    def next_run_times(self) -> dict[str, datetime | None]:
        """When each registered job fires next, if the scheduler is running."""
        if self._scheduler is None:
            return {}
        return {job.id: getattr(job, "next_run_time", None) for job in self._scheduler.get_jobs()}


def unattended_policy(agent: Agent, ceiling: RiskLevel) -> PolicyEngine:
    """A policy engine for unattended contexts. Exposed for the folder watcher."""
    return PolicyEngine(agent.settings, agent.database, AutoApproveHandler(ceiling))
