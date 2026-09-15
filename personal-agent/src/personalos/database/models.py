"""SQLAlchemy models — the agent's durable state.

Everything the agent learns or does ends up in one of these tables, and every
row is inspectable and deletable from the CLI. That is the whole point of the
"learning must be explicit, inspectable and reversible" rule: there is no
hidden state anywhere else.

JSON payloads are stored as ``JSON`` columns so SQLite keeps them queryable
while leaving room to move to Postgres later without a schema rewrite.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from personalos.utils.timeutil import ensure_aware, utcnow


class UTCDateTime(TypeDecorator):
    """A datetime column that is always timezone-aware UTC in Python.

    SQLite has no timezone type, so a value written as aware comes back naive.
    Comparing one of those against ``utcnow()`` raises, which would break decay,
    retention and every "how long ago" calculation. This decorator normalises
    on the way in and re-attaches UTC on the way out.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect: object) -> datetime | None:
        return None if value is None else ensure_aware(value)

    def process_result_value(self, value: datetime | None, dialect: object) -> datetime | None:
        return None if value is None else ensure_aware(value)


class Base(DeclarativeBase):
    """Declarative base with a JSON-friendly type map."""

    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


def _now() -> datetime:
    return utcnow()


class TaskStatus(StrEnum):
    PENDING = "pending"
    PLANNING = "planning"
    AWAITING_APPROVAL = "awaiting_approval"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class ApprovalStatus(StrEnum):
    NOT_REQUIRED = "not_required"
    AUTO_APPROVED = "auto_approved"
    RULE_APPROVED = "rule_approved"
    USER_APPROVED = "user_approved"
    USER_DENIED = "user_denied"
    PENDING = "pending"


class MemorySource(StrEnum):
    EXPLICIT = "explicit_user_instruction"
    OBSERVED = "observed"
    INFERRED = "inferred"
    SYSTEM = "system"


class Task(Base):
    """One user request, from natural language through to a verified result."""

    __tablename__ = "tasks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    request: Mapped[str] = mapped_column(Text)
    project: Mapped[str | None] = mapped_column(String(128), index=True, default=None)
    status: Mapped[str] = mapped_column(String(32), default=TaskStatus.PENDING, index=True)
    agent_state: Mapped[str] = mapped_column(String(32), default="IDLE")
    origin: Mapped[str] = mapped_column(String(32), default="cli")
    """cli | scheduler | watcher | dashboard | skill"""
    plan: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    result_summary: Mapped[str | None] = mapped_column(Text, default=None)
    success: Mapped[bool | None] = mapped_column(Boolean, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    max_risk_level: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(
        UTCDateTime, default=None
    )

    actions: Mapped[list[ActionRecord]] = relationship(
        back_populates="task", cascade="all, delete-orphan", order_by="ActionRecord.id"
    )


class ActionRecord(Base):
    """A single tool invocation. Mirrors the JSONL audit log for querying."""

    __tablename__ = "actions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_pk: Mapped[int | None] = mapped_column(
        ForeignKey("tasks.id", ondelete="CASCADE"), default=None, index=True
    )
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    step_id: Mapped[str | None] = mapped_column(String(64), default=None)
    action: Mapped[str] = mapped_column(String(128))
    tool: Mapped[str] = mapped_column(String(64), index=True)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    risk_level: Mapped[int] = mapped_column(Integer, default=0, index=True)
    approval_status: Mapped[str] = mapped_column(String(32), default=ApprovalStatus.NOT_REQUIRED)
    success: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, default=None)
    error: Mapped[str | None] = mapped_column(Text, default=None)
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0.0)
    verified: Mapped[bool | None] = mapped_column(Boolean, default=None)
    verification_note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now, index=True)

    task: Mapped[Task | None] = relationship(back_populates="actions")


class RollbackEntry(Base):
    """An undo record for one reversible operation.

    ``rollback`` describes the inverse operation in the same vocabulary as
    ``operation``, so replaying it needs no special-casing per tool.
    """

    __tablename__ = "rollback_journal"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    step_id: Mapped[str | None] = mapped_column(String(64), default=None)
    operation: Mapped[str] = mapped_column(String(32))
    source: Mapped[str | None] = mapped_column(Text, default=None)
    destination: Mapped[str | None] = mapped_column(Text, default=None)
    rollback: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    applied: Mapped[bool] = mapped_column(Boolean, default=False)
    """True once this entry has been used to undo the operation."""
    applied_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now, index=True)


class Episode(Base):
    """Episodic memory: what was asked, what was done, and how it went."""

    __tablename__ = "episodes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    task_id: Mapped[str] = mapped_column(String(36), index=True)
    project: Mapped[str | None] = mapped_column(String(128), index=True, default=None)
    task_text: Mapped[str] = mapped_column(Text)
    intent: Mapped[str | None] = mapped_column(String(64), index=True, default=None)
    action_signature: Mapped[str] = mapped_column(Text, default="")
    """Ordered ``tool:operation`` fingerprint used by the pattern detector."""
    actions: Mapped[list[Any]] = mapped_column(JSON, default=list)
    result: Mapped[str | None] = mapped_column(Text, default=None)
    success: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    user_feedback: Mapped[str | None] = mapped_column(Text, default=None)
    embedding_ref: Mapped[str | None] = mapped_column(String(128), default=None)
    """Opaque handle into an external vector store, when one is configured."""
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now, index=True)


class SemanticMemory(Base):
    """Durable facts: what a directory is for, what a project means, etc."""

    __tablename__ = "semantic_memory"
    __table_args__ = (UniqueConstraint("namespace", "key", name="uq_semantic_ns_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace: Mapped[str] = mapped_column(String(128), default="global", index=True)
    key: Mapped[str] = mapped_column(String(256), index=True)
    value: Mapped[str] = mapped_column(Text)
    category: Mapped[str] = mapped_column(String(64), default="fact", index=True)
    source: Mapped[str] = mapped_column(String(48), default=MemorySource.OBSERVED)
    confidence: Mapped[float] = mapped_column(Float, default=0.6)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding_ref: Mapped[str | None] = mapped_column(String(128), default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)
    last_used: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)


class Preference(Base):
    """A preference or standing rule, with provenance and confidence."""

    __tablename__ = "preferences"
    __table_args__ = (UniqueConstraint("namespace", "key", name="uq_pref_ns_key"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    namespace: Mapped[str] = mapped_column(String(128), default="global", index=True)
    key: Mapped[str] = mapped_column(String(128), index=True)
    value: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    """Always a JSON object: ``{"value": ...}``, so types survive round-tripping."""
    source: Mapped[str] = mapped_column(String(48), default=MemorySource.OBSERVED)
    confidence: Mapped[float] = mapped_column(Float, default=0.6)
    usage_count: Mapped[int] = mapped_column(Integer, default=0)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    note: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)
    last_used: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)


class PolicyRule(Base):
    """A standing approval rule the user granted explicitly.

    Rules can only ever *narrow* the number of questions for levels 0-2. The
    policy engine refuses to consult them for levels 3 and 4.
    """

    __tablename__ = "policy_rules"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True)
    effect: Mapped[str] = mapped_column(String(16), default="allow")
    """allow | deny. A deny rule always wins."""
    tool: Mapped[str | None] = mapped_column(String(64), default=None, index=True)
    operation: Mapped[str | None] = mapped_column(String(64), default=None)
    path_prefix: Mapped[str | None] = mapped_column(Text, default=None)
    max_risk_level: Mapped[int] = mapped_column(Integer, default=2)
    project: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    source: Mapped[str] = mapped_column(String(48), default=MemorySource.EXPLICIT)
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    expires_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)


class Workflow(Base):
    """A repeatable sequence the agent has seen succeed more than once."""

    __tablename__ = "workflows"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    description: Mapped[str] = mapped_column(Text, default="")
    trigger: Mapped[str | None] = mapped_column(Text, default=None)
    signature: Mapped[str] = mapped_column(Text, index=True, default="")
    steps: Mapped[list[Any]] = mapped_column(JSON, default=list)
    example_requests: Mapped[list[Any]] = mapped_column(JSON, default=list)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    promoted_skill: Mapped[str | None] = mapped_column(String(128), default=None)
    project: Mapped[str | None] = mapped_column(String(128), default=None, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)
    last_seen: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)


class SkillRecord(Base):
    """Bookkeeping for a skill whose definition lives on disk as YAML."""

    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    path: Mapped[str] = mapped_column(Text)
    created_by: Mapped[str] = mapped_column(String(32), default="user")
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    success_count: Mapped[int] = mapped_column(Integer, default=0)
    last_run: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)

    @property
    def success_rate(self) -> float | None:
        if not self.run_count:
            return None
        return self.success_count / self.run_count


class Suggestion(Base):
    """A proactive observation awaiting the user's yes or no.

    Suggestions are strictly separated from execution: nothing here runs until
    the user accepts it, and accepting creates a *skill definition*, not a job.
    """

    __tablename__ = "suggestions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(48), index=True)
    title: Mapped[str] = mapped_column(Text)
    detail: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    """pending | accepted | dismissed | expired"""
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now, index=True)
    resolved_at: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)


class ScheduledJob(Base):
    """A recurring or one-off job the local scheduler owns."""

    __tablename__ = "scheduled_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(128))
    kind: Mapped[str] = mapped_column(String(24), default="ask")
    """ask | skill"""
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    schedule_kind: Mapped[str] = mapped_column(String(16), default="cron")
    """cron | interval | date"""
    schedule: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    spec: Mapped[str] = mapped_column(Text, default="")
    """The human phrasing the user typed, kept for display."""
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    max_risk_level: Mapped[int] = mapped_column(Integer, default=1)
    project: Mapped[str | None] = mapped_column(String(128), default=None)
    last_run: Mapped[datetime | None] = mapped_column(UTCDateTime, default=None)
    last_status: Mapped[str | None] = mapped_column(String(32), default=None)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now)


class ObservationEvent(Base):
    """Something an opt-in observer noticed, e.g. a new file in a watched folder."""

    __tablename__ = "observation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    observer: Mapped[str] = mapped_column(String(48), index=True)
    event_type: Mapped[str] = mapped_column(String(32))
    path: Mapped[str | None] = mapped_column(Text, default=None)
    detail: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    handled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime, default=_now, index=True)


Index("ix_episodes_signature", Episode.action_signature)
Index("ix_actions_task_created", ActionRecord.task_id, ActionRecord.created_at)
