"""Workflow memory: sequences the agent has seen work more than once.

A workflow is an *observation*, not an automation. It records that a shape of
task keeps recurring and how it was carried out. Turning one into something
that runs on its own is a separate, approved step — see
``learning.skill_generator``.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import select

from personalos.database.database import Database
from personalos.database.models import Workflow
from personalos.utils.timeutil import utcnow


class WorkflowMemory:
    """Stores recurring task shapes and their observed success rate."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def record(
        self,
        *,
        name: str,
        signature: str,
        steps: list[dict[str, Any]],
        description: str = "",
        trigger: str | None = None,
        example_request: str | None = None,
        project: str | None = None,
        success: bool = True,
    ) -> Workflow:
        """Insert or update a workflow, incrementing its tally."""
        with self.database.session() as session:
            row = session.scalar(select(Workflow).where(Workflow.name == name))
            if row is None:
                row = Workflow(name=name, signature=signature, steps=steps)
                session.add(row)
                # Flush so the column defaults land before the counters below
                # are incremented; a freshly constructed row has them as None.
                session.flush()
            row.description = description or row.description
            row.trigger = trigger or row.trigger
            row.signature = signature
            row.steps = steps
            row.project = project or row.project
            if example_request:
                examples = list(row.example_requests or [])
                if example_request not in examples:
                    examples.append(example_request)
                row.example_requests = examples[-10:]
            if success:
                row.success_count += 1
            else:
                row.failure_count += 1
            row.last_seen = utcnow()
            session.flush()
            session.refresh(row)
            session.expunge(row)
            return row

    def by_signature(self, signature: str) -> Workflow | None:
        with self.database.session() as session:
            row = session.scalar(select(Workflow).where(Workflow.signature == signature))
            if row is not None:
                session.expunge(row)
            return row

    def get(self, name: str) -> Workflow | None:
        with self.database.session() as session:
            row = session.scalar(select(Workflow).where(Workflow.name == name))
            if row is not None:
                session.expunge(row)
            return row

    def all(self, *, project: str | None = None) -> list[Workflow]:
        with self.database.session() as session:
            statement = select(Workflow).order_by(Workflow.success_count.desc())
            if project is not None:
                statement = statement.where(Workflow.project == project)
            rows = list(session.scalars(statement))
            for row in rows:
                session.expunge(row)
            return rows

    def mark_promoted(self, name: str, skill_name: str) -> bool:
        """Link a workflow to the skill the user approved creating from it."""
        with self.database.session() as session:
            row = session.scalar(select(Workflow).where(Workflow.name == name))
            if row is None:
                return False
            row.promoted_skill = skill_name
            return True

    def forget(self, name: str) -> bool:
        with self.database.session() as session:
            row = session.scalar(select(Workflow).where(Workflow.name == name))
            if row is None:
                return False
            session.delete(row)
            return True
