"""Creating, versioning, running and retiring skills.

Two rules are enforced here rather than left to callers:

* **Creation is always explicit.** ``create`` is only ever reached from a CLI
  command or an accepted suggestion. Nothing in the learning pipeline can call
  it on its own.
* **A skill is not a permission.** Running a skill produces a plan, and the
  plan goes through the same policy engine as anything else. A skill that
  moves files still asks the first time, every time the rules say it should.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import select

from personalos.agent.plan import Plan
from personalos.database.database import Database
from personalos.database.models import SkillRecord
from personalos.errors import SkillError
from personalos.skills.schema import SkillDefinition, load_skill_file
from personalos.skills.skill_registry import SkillFile, SkillRegistry
from personalos.utils.timeutil import utcnow


@dataclass
class SkillSummary:
    """A skill plus its run statistics, for `agent skills`."""

    definition: SkillDefinition
    path: Path
    origin: str
    enabled: bool
    run_count: int
    success_count: int

    @property
    def success_rate(self) -> float | None:
        if not self.run_count:
            return None
        return self.success_count / self.run_count


class SkillManager:
    """The API behind every `agent skills …` command."""

    def __init__(self, registry: SkillRegistry, database: Database) -> None:
        self.registry = registry
        self.database = database

    # ---- reading -----------------------------------------------------------
    def _record(self, name: str) -> SkillRecord | None:
        with self.database.session() as session:
            record = session.scalar(select(SkillRecord).where(SkillRecord.name == name))
            if record is not None:
                session.expunge(record)
            return record

    def list_skills(self, *, include_disabled: bool = True) -> list[SkillSummary]:
        summaries: list[SkillSummary] = []
        for item in self.registry.all():
            record = self._record(item.definition.name)
            enabled = record.enabled if record else True
            if not enabled and not include_disabled:
                continue
            summaries.append(
                SkillSummary(
                    definition=item.definition,
                    path=item.path,
                    origin=item.origin,
                    enabled=enabled,
                    run_count=record.run_count if record else 0,
                    success_count=record.success_count if record else 0,
                )
            )
        return summaries

    def inspect(self, name: str) -> SkillFile:
        """Return a skill's definition and location for display."""
        return self.registry.get(name)

    # ---- writing -----------------------------------------------------------
    def create(self, definition: SkillDefinition, *, generated: bool = False) -> Path:
        """Write a new skill to disk.

        Raises:
            SkillError: when a skill of that name already exists. Updating one
                goes through :meth:`version`, which keeps the previous file.
        """
        if self.registry.exists(definition.name):
            raise SkillError(
                f"A skill called {definition.name!r} already exists.",
                remediation=f"Use `agent skills version {definition.name}` to supersede it.",
            )
        path = self.registry.path_for(definition.name, generated=generated)
        path.write_text(definition.to_yaml(), encoding="utf-8")
        with self.database.session() as session:
            session.add(
                SkillRecord(
                    name=definition.name,
                    version=definition.version,
                    path=str(path),
                    created_by=definition.created_by,
                )
            )
        self.registry.reload()
        return path

    def version(self, name: str, definition: SkillDefinition) -> Path:
        """Replace a skill, archiving the previous version beside it."""
        existing = self.registry.get(name)
        archive_dir = existing.path.parent / "versions"
        archive_dir.mkdir(parents=True, exist_ok=True)
        archive = archive_dir / f"{name}.v{existing.definition.version}.yaml"
        shutil.copy2(existing.path, archive)

        definition = definition.model_copy(update={"version": existing.definition.version + 1})
        existing.path.write_text(definition.to_yaml(), encoding="utf-8")
        with self.database.session() as session:
            record = session.scalar(select(SkillRecord).where(SkillRecord.name == name))
            if record is not None:
                record.version = definition.version
        self.registry.reload()
        return existing.path

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Disable a skill without deleting it."""
        self.registry.get(name)  # raises if unknown
        with self.database.session() as session:
            record = session.scalar(select(SkillRecord).where(SkillRecord.name == name))
            if record is None:
                item = self.registry.get(name)
                record = SkillRecord(
                    name=name,
                    version=item.definition.version,
                    path=str(item.path),
                    created_by=item.definition.created_by,
                )
                session.add(record)
            record.enabled = enabled

    def delete(self, name: str, *, keep_backup: bool = True) -> Path:
        """Remove a skill, keeping a copy unless told otherwise."""
        item = self.registry.get(name)
        if item.origin == "builtin":
            raise SkillError(
                f"{name} ships with PersonalOS and cannot be deleted.",
                remediation="Disable it instead: `agent skills disable " + name + "`.",
            )
        if keep_backup:
            backup_dir = item.path.parent / "deleted"
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item.path), str(backup_dir / item.path.name))
        else:
            item.path.unlink()
        with self.database.session() as session:
            record = session.scalar(select(SkillRecord).where(SkillRecord.name == name))
            if record is not None:
                session.delete(record)
        self.registry.reload()
        return item.path

    # ---- running -----------------------------------------------------------
    def build_plan(self, name: str, inputs: dict[str, Any] | None = None) -> Plan:
        """Instantiate a skill as a plan, refusing if it has been disabled."""
        item = self.registry.get(name)
        record = self._record(name)
        if record is not None and not record.enabled:
            raise SkillError(
                f"The skill {name!r} is disabled.",
                remediation=f"Re-enable it with `agent skills enable {name}`.",
            )
        return item.definition.to_plan(inputs)

    def record_run(self, name: str, *, success: bool) -> None:
        """Update a skill's run statistics after it executes."""
        with self.database.session() as session:
            record = session.scalar(select(SkillRecord).where(SkillRecord.name == name))
            if record is None:
                try:
                    item = self.registry.get(name)
                except SkillError:
                    return
                record = SkillRecord(
                    name=name,
                    version=item.definition.version,
                    path=str(item.path),
                    created_by=item.definition.created_by,
                )
                session.add(record)
                session.flush()  # let the column defaults populate the counters
            record.run_count += 1
            if success:
                record.success_count += 1
            record.last_run = utcnow()

    def import_file(self, path: Path, *, generated: bool = False) -> Path:
        """Validate an external YAML file and copy it into the skills directory."""
        definition = load_skill_file(path)
        return self.create(definition, generated=generated)
