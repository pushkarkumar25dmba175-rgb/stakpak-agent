"""The audit log: an append-only record of everything the agent did.

Actions are written twice, on purpose:

* to ``logs/actions-YYYY-MM-DD.jsonl`` — append-only, greppable, survives a
  corrupted database, and is the artefact you would hand to someone reviewing
  what the agent has been doing on your machine;
* to the ``actions`` table — queryable, joined to tasks, what `agent history`
  and the dashboard read.

Every payload passes through redaction before it is written to either.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select

from personalos.database.database import Database
from personalos.database.models import ActionRecord, ApprovalStatus, Task
from personalos.security.risk import RiskLevel
from personalos.security.secrets import redact, redact_structure
from personalos.utils.timeutil import isoformat, utcnow


@dataclass
class AuditEntry:
    """One logged action. Field order matches the JSONL layout."""

    timestamp: str
    task_id: str
    action: str
    tool: str
    parameters: dict[str, Any]
    risk_level: int
    approval_status: str
    success: bool
    execution_time_ms: float
    step_id: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None
    verified: bool | None = None
    verification_note: str | None = None
    project: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, default=str)


@dataclass
class AuditQuery:
    """Filters for reading the log back."""

    task_id: str | None = None
    tool: str | None = None
    since: datetime | None = None
    only_failures: bool = False
    min_risk_level: int | None = None
    limit: int = 50


class AuditLog:
    """Writes and reads the action log.

    Args:
        logs_dir: Directory holding the daily JSONL files.
        database: Where the queryable mirror lives.
        redact_enabled: Turning this off is a deliberate debugging choice and
            is reported by ``agent doctor`` as a warning.
    """

    def __init__(self, logs_dir: Path, database: Database, *, redact_enabled: bool = True) -> None:
        self.logs_dir = logs_dir
        self.database = database
        self.redact_enabled = redact_enabled
        self._lock = threading.Lock()
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, day: date | None = None) -> Path:
        """The JSONL file for a given day (default: today, UTC)."""
        day = day or utcnow().date()
        return self.logs_dir / f"actions-{day.isoformat()}.jsonl"

    def _clean(self, value: Any) -> Any:
        return redact_structure(value) if self.redact_enabled else value

    def record(
        self,
        *,
        task_id: str,
        action: str,
        tool: str,
        parameters: dict[str, Any] | None = None,
        risk_level: RiskLevel | int = RiskLevel.READ_ONLY,
        approval_status: ApprovalStatus | str = ApprovalStatus.NOT_REQUIRED,
        success: bool = True,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        execution_time_ms: float = 0.0,
        step_id: str | None = None,
        verified: bool | None = None,
        verification_note: str | None = None,
        project: str | None = None,
    ) -> AuditEntry:
        """Append one action to both the JSONL file and the database."""
        entry = AuditEntry(
            timestamp=isoformat(),
            task_id=task_id,
            action=action,
            tool=tool,
            parameters=self._clean(parameters or {}),
            risk_level=int(risk_level),
            approval_status=str(approval_status),
            success=success,
            execution_time_ms=round(execution_time_ms, 3),
            step_id=step_id,
            result=self._clean(result) if result is not None else None,
            error=redact(error) if (error and self.redact_enabled) else error,
            verified=verified,
            verification_note=verification_note,
            project=project,
        )

        with self._lock:
            with self.path_for().open("a", encoding="utf-8") as handle:
                handle.write(entry.to_json() + "\n")

        with self.database.session() as session:
            task_pk = session.scalar(select(Task.id).where(Task.task_id == task_id))
            session.add(
                ActionRecord(
                    task_pk=task_pk,
                    task_id=entry.task_id,
                    step_id=entry.step_id,
                    action=entry.action,
                    tool=entry.tool,
                    parameters=entry.parameters,
                    risk_level=entry.risk_level,
                    approval_status=entry.approval_status,
                    success=entry.success,
                    result=entry.result,
                    error=entry.error,
                    execution_time_ms=entry.execution_time_ms,
                    verified=entry.verified,
                    verification_note=entry.verification_note,
                )
            )
        return entry

    # ---- reading -----------------------------------------------------------
    def query(self, spec: AuditQuery | None = None) -> list[ActionRecord]:
        """Read actions back from the database, newest first."""
        spec = spec or AuditQuery()
        statement = select(ActionRecord).order_by(ActionRecord.created_at.desc())
        if spec.task_id:
            statement = statement.where(ActionRecord.task_id == spec.task_id)
        if spec.tool:
            statement = statement.where(ActionRecord.tool == spec.tool)
        if spec.since:
            statement = statement.where(ActionRecord.created_at >= spec.since)
        if spec.only_failures:
            statement = statement.where(ActionRecord.success.is_(False))
        if spec.min_risk_level is not None:
            statement = statement.where(ActionRecord.risk_level >= spec.min_risk_level)
        statement = statement.limit(spec.limit)
        with self.database.session() as session:
            records = list(session.scalars(statement))
            for record in records:
                session.expunge(record)
            return records

    def iter_file(self, day: date | None = None) -> Iterator[dict[str, Any]]:
        """Stream raw entries from a day's JSONL file.

        Malformed lines are skipped rather than raising: a truncated final line
        after a crash should not make the whole day unreadable.
        """
        path = self.path_for(day)
        if not path.exists():
            return
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def files(self) -> list[Path]:
        """Every daily log file, oldest first."""
        return sorted(self.logs_dir.glob("actions-*.jsonl"))
