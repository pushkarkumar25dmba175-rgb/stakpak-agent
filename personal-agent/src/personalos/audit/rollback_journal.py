"""The rollback journal: how the agent undoes what it did.

Every reversible operation records the *inverse* operation at the moment it
succeeds. Undoing a task is then a matter of replaying those inverses in
reverse order — no tool-specific undo logic, no guessing after the fact.

Supported inverse operations:

``move``    move a file back from destination to source
``delete``  remove a file the agent created (only ever applied to paths the
            journal itself recorded as created)
``restore`` copy a saved backup back over the original
``noop``    the operation was not reversible; the journal says so explicitly
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from personalos.database.database import Database
from personalos.database.models import RollbackEntry
from personalos.errors import RollbackError
from personalos.utils.timeutil import utcnow


@dataclass
class RollbackResult:
    """What happened when a rollback was attempted."""

    entry_id: int
    operation: str
    applied: bool
    detail: str


class RollbackJournal:
    """Records inverse operations and replays them on request."""

    def __init__(self, database: Database, backups_dir: Path) -> None:
        self.database = database
        self.backups_dir = backups_dir
        self.backups_dir.mkdir(parents=True, exist_ok=True)

    # ---- recording ---------------------------------------------------------
    def backup_file(self, path: Path, task_id: str) -> Path:
        """Copy ``path`` into the backup store and return the copy's location."""
        target_dir = self.backups_dir / task_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stamp = utcnow().strftime("%H%M%S%f")
        target = target_dir / f"{stamp}-{path.name}"
        shutil.copy2(path, target)
        return target

    def record(
        self,
        *,
        task_id: str,
        operation: str,
        rollback: dict[str, Any],
        source: str | None = None,
        destination: str | None = None,
        step_id: str | None = None,
    ) -> int:
        """Store an inverse operation and return its journal id."""
        with self.database.session() as session:
            entry = RollbackEntry(
                task_id=task_id,
                step_id=step_id,
                operation=operation,
                source=source,
                destination=destination,
                rollback=rollback,
            )
            session.add(entry)
            session.flush()
            return entry.id

    def record_move(self, task_id: str, source: Path, destination: Path, **kwargs: Any) -> int:
        """Record that ``source`` became ``destination``."""
        return self.record(
            task_id=task_id,
            operation="move",
            source=str(source),
            destination=str(destination),
            rollback={"operation": "move", "source": str(destination), "destination": str(source)},
            **kwargs,
        )

    def record_create(self, task_id: str, path: Path, **kwargs: Any) -> int:
        """Record that ``path`` did not exist before the agent created it."""
        return self.record(
            task_id=task_id,
            operation="create",
            destination=str(path),
            rollback={"operation": "delete", "target": str(path)},
            **kwargs,
        )

    def record_modify(self, task_id: str, path: Path, backup: Path, **kwargs: Any) -> int:
        """Record that ``path`` was overwritten, with ``backup`` holding the original."""
        return self.record(
            task_id=task_id,
            operation="modify",
            source=str(path),
            rollback={"operation": "restore", "backup": str(backup), "target": str(path)},
            **kwargs,
        )

    def record_irreversible(self, task_id: str, description: str, **kwargs: Any) -> int:
        """Record that something happened which cannot be undone locally."""
        return self.record(
            task_id=task_id,
            operation="irreversible",
            rollback={"operation": "noop", "reason": description},
            **kwargs,
        )

    # ---- reading -----------------------------------------------------------
    def entries_for(self, task_id: str, *, include_applied: bool = False) -> list[RollbackEntry]:
        """Journal entries for a task, newest first — i.e. rollback order."""
        with self.database.session() as session:
            statement = (
                select(RollbackEntry)
                .where(RollbackEntry.task_id == task_id)
                .order_by(RollbackEntry.id.desc())
            )
            if not include_applied:
                statement = statement.where(RollbackEntry.applied.is_(False))
            entries = list(session.scalars(statement))
            for entry in entries:
                session.expunge(entry)
            return entries

    # ---- replaying ---------------------------------------------------------
    def rollback_task(self, task_id: str, *, dry_run: bool = False) -> list[RollbackResult]:
        """Undo every un-applied operation recorded for ``task_id``.

        Entries are replayed newest-first so that, for example, a file that was
        created and then moved ends up removed rather than orphaned.

        Raises:
            RollbackError: when the task has nothing recorded to undo.
        """
        entries = self.entries_for(task_id)
        if not entries:
            raise RollbackError(
                f"There is nothing to roll back for task {task_id}.",
                remediation="Run `agent history` to see which tasks made changes.",
            )

        results: list[RollbackResult] = []
        for entry in entries:
            result = self._apply(entry, dry_run=dry_run)
            results.append(result)
            if result.applied and not dry_run:
                with self.database.session() as session:
                    stored = session.get(RollbackEntry, entry.id)
                    if stored is not None:
                        stored.applied = True
                        stored.applied_at = utcnow()
        return results

    def _apply(self, entry: RollbackEntry, *, dry_run: bool) -> RollbackResult:
        spec = entry.rollback or {}
        operation = str(spec.get("operation", "noop"))

        if operation == "noop":
            return RollbackResult(
                entry.id, operation, False,
                f"Not reversible: {spec.get('reason', 'no inverse was recorded')}.",
            )

        if operation == "move":
            source, destination = Path(str(spec["source"])), Path(str(spec["destination"]))
            if not source.exists():
                return RollbackResult(entry.id, operation, False, f"{source} is no longer there.")
            if dry_run:
                return RollbackResult(entry.id, operation, False, f"Would move {source} back to {destination}.")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            return RollbackResult(entry.id, operation, True, f"Moved {source} back to {destination}.")

        if operation == "delete":
            target = Path(str(spec["target"]))
            if not target.exists():
                return RollbackResult(entry.id, operation, False, f"{target} is already gone.")
            if dry_run:
                return RollbackResult(entry.id, operation, False, f"Would remove the file the agent created at {target}.")
            target.unlink()
            return RollbackResult(entry.id, operation, True, f"Removed {target}, which the agent had created.")

        if operation == "restore":
            backup, target = Path(str(spec["backup"])), Path(str(spec["target"]))
            if not backup.exists():
                return RollbackResult(entry.id, operation, False, f"The backup {backup} is missing.")
            if dry_run:
                return RollbackResult(entry.id, operation, False, f"Would restore {target} from {backup}.")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, target)
            return RollbackResult(entry.id, operation, True, f"Restored {target} from its backup.")

        return RollbackResult(entry.id, operation, False, f"Unknown rollback operation {operation!r}.")
