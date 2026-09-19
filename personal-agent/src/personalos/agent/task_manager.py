"""Task records: creating them, moving them through states, reading them back."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Any

from sqlalchemy import select

from personalos.agent.state import AgentState
from personalos.database.database import Database
from personalos.database.models import Task, TaskStatus
from personalos.utils.timeutil import utcnow


class TaskManager:
    """Owns the lifecycle of :class:`Task` rows."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def create(
        self,
        request: str,
        *,
        project: str | None = None,
        origin: str = "cli",
    ) -> Task:
        """Open a new task and return the detached row."""
        with self.database.session() as session:
            task = Task(
                task_id=uuid.uuid4().hex[:16],
                request=request,
                project=project,
                origin=origin,
                status=TaskStatus.PENDING,
                agent_state=AgentState.PLANNING,
            )
            session.add(task)
            session.flush()
            session.refresh(task)
            session.expunge(task)
            return task

    def update(
        self,
        task_id: str,
        *,
        status: TaskStatus | str | None = None,
        agent_state: AgentState | str | None = None,
        plan: dict[str, Any] | None = None,
        result_summary: str | None = None,
        success: bool | None = None,
        error: str | None = None,
        max_risk_level: int | None = None,
        finished: bool = False,
    ) -> Task | None:
        """Patch a task row. Only the fields passed are touched."""
        with self.database.session() as session:
            task = session.scalar(select(Task).where(Task.task_id == task_id))
            if task is None:
                return None
            if status is not None:
                task.status = str(status)
            if agent_state is not None:
                task.agent_state = str(agent_state)
            if plan is not None:
                task.plan = plan
            if result_summary is not None:
                task.result_summary = result_summary
            if success is not None:
                task.success = success
            if error is not None:
                task.error = error
            if max_risk_level is not None:
                task.max_risk_level = max_risk_level
            if finished:
                task.finished_at = utcnow()
            session.flush()
            session.refresh(task)
            session.expunge(task)
            return task

    def get(self, task_id: str) -> Task | None:
        with self.database.session() as session:
            task = session.scalar(select(Task).where(Task.task_id == task_id))
            if task is not None:
                session.expunge(task)
            return task

    def resolve_id(self, partial: str) -> str | None:
        """Expand a task-id prefix to the full id.

        `agent history` abbreviates ids to eight characters, so the id a person
        can see is not the one the other commands want. Accepting the prefix
        removes that papercut. An ambiguous prefix resolves to nothing rather
        than to an arbitrary match.
        """
        if not partial:
            return None
        with self.database.session() as session:
            exact = session.scalar(select(Task.task_id).where(Task.task_id == partial))
            if exact:
                return exact
            matches = list(
                session.scalars(
                    select(Task.task_id).where(Task.task_id.startswith(partial)).limit(2)
                )
            )
        return matches[0] if len(matches) == 1 else None

    def recent(
        self,
        limit: int = 20,
        *,
        today_only: bool = False,
        project: str | None = None,
        status: TaskStatus | str | None = None,
    ) -> list[Task]:
        with self.database.session() as session:
            statement = select(Task).order_by(Task.created_at.desc()).limit(limit)
            if today_only:
                midnight = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
                statement = statement.where(Task.created_at >= midnight)
            if project is not None:
                statement = statement.where(Task.project == project)
            if status is not None:
                statement = statement.where(Task.status == str(status))
            tasks = list(session.scalars(statement))
            for task in tasks:
                session.expunge(task)
            return tasks

    def unfinished(self, *, within_days: int = 7) -> list[Task]:
        """Tasks that never reached a terminal state — what the evening brief shows."""
        cutoff = utcnow() - timedelta(days=within_days)
        terminal = {
            str(TaskStatus.COMPLETED),
            str(TaskStatus.FAILED),
            str(TaskStatus.CANCELLED),
            str(TaskStatus.REJECTED),
        }
        with self.database.session() as session:
            tasks = list(
                session.scalars(
                    select(Task)
                    .where(Task.created_at >= cutoff, Task.status.notin_(terminal))
                    .order_by(Task.created_at.desc())
                )
            )
            for task in tasks:
                session.expunge(task)
            return tasks
