"""Preference memory, with provenance, confidence and decay.

Two rules govern this module and are enforced in code, not just documented:

* An **explicit** instruction ("always save reports as Markdown") has
  confidence 1.0, never decays, and cannot be overwritten by an inference.
* An **inferred** preference decays with age and stops being injected into
  the model's context once it falls below the configured floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select

from personalos.database.database import Database
from personalos.database.models import MemorySource, Preference
from personalos.utils.timeutil import utcnow


@dataclass
class ResolvedPreference:
    """A preference with decay already applied."""

    key: str
    value: Any
    source: str
    confidence: float
    effective_confidence: float
    usage_count: int
    namespace: str = "global"
    note: str | None = None

    @property
    def explicit(self) -> bool:
        return self.source == MemorySource.EXPLICIT

    def render(self) -> str:
        marker = "stated" if self.explicit else f"inferred {self.effective_confidence:.0%}"
        return f"{self.key} = {self.value!r} ({marker})"


class PreferenceStore:
    """Reads and writes preferences, applying the precedence rules."""

    def __init__(
        self,
        database: Database,
        *,
        half_life_days: float = 45.0,
        confidence_floor: float = 0.25,
    ) -> None:
        self.database = database
        self.half_life_days = half_life_days
        self.confidence_floor = confidence_floor

    # ---- writing -----------------------------------------------------------
    def set(
        self,
        key: str,
        value: Any,
        *,
        namespace: str = "global",
        source: str = MemorySource.EXPLICIT,
        confidence: float | None = None,
        note: str | None = None,
    ) -> ResolvedPreference:
        """Store a preference.

        An inferred write against an existing explicit preference is ignored:
        the stored value is returned unchanged. That is what "explicit
        instructions always outrank inferred preferences" means mechanically.
        """
        explicit = source == MemorySource.EXPLICIT
        resolved_confidence = 1.0 if explicit else (confidence if confidence is not None else 0.6)

        with self.database.session() as session:
            row = session.scalar(
                select(Preference).where(
                    Preference.namespace == namespace, Preference.key == key
                )
            )
            if row is None:
                row = Preference(namespace=namespace, key=key)
                session.add(row)
            elif row.source == MemorySource.EXPLICIT and not explicit:
                session.expunge(row)
                return self._resolve(row)

            row.value = {"value": value}
            row.source = source
            row.confidence = min(1.0, max(0.0, resolved_confidence))
            row.note = note
            row.active = True
            row.updated_at = utcnow()
            session.flush()
            session.refresh(row)
            session.expunge(row)
            return self._resolve(row)

    def reinforce(self, key: str, *, namespace: str = "global", amount: float = 0.08) -> None:
        """Nudge an inferred preference upward after it proved useful.

        Explicit preferences are already at 1.0 and are left alone.
        """
        with self.database.session() as session:
            row = session.scalar(
                select(Preference).where(
                    Preference.namespace == namespace, Preference.key == key
                )
            )
            if row is None or row.source == MemorySource.EXPLICIT:
                return
            row.confidence = min(1.0, row.confidence + amount)
            row.usage_count += 1
            row.last_used = utcnow()
            row.updated_at = utcnow()

    def weaken(self, key: str, *, namespace: str = "global", amount: float = 0.25) -> None:
        """Reduce confidence after the user corrected the agent."""
        with self.database.session() as session:
            row = session.scalar(
                select(Preference).where(
                    Preference.namespace == namespace, Preference.key == key
                )
            )
            if row is None:
                return
            row.confidence = max(0.0, row.confidence - amount)
            row.updated_at = utcnow()
            if row.confidence <= 0.05:
                row.active = False

    def forget(self, key: str, *, namespace: str = "global") -> bool:
        with self.database.session() as session:
            row = session.scalar(
                select(Preference).where(
                    Preference.namespace == namespace, Preference.key == key
                )
            )
            if row is None:
                return False
            session.delete(row)
            return True

    # ---- reading -----------------------------------------------------------
    def _decay(self, row: Preference) -> float:
        """Exponential decay by age since last update, for inferred items only."""
        if row.source == MemorySource.EXPLICIT:
            return row.confidence
        reference = row.last_used or row.updated_at or row.created_at
        if reference is None:
            return row.confidence
        age_days = max(0.0, (utcnow() - reference).total_seconds() / 86400.0)
        factor = math.pow(0.5, age_days / self.half_life_days)
        return row.confidence * factor

    def _resolve(self, row: Preference) -> ResolvedPreference:
        payload = row.value or {}
        return ResolvedPreference(
            key=row.key,
            value=payload.get("value"),
            source=row.source,
            confidence=row.confidence,
            effective_confidence=self._decay(row),
            usage_count=row.usage_count,
            namespace=row.namespace,
            note=row.note,
        )

    def get(self, key: str, default: Any = None, *, namespace: str = "global") -> Any:
        """Return a preference value, or ``default`` when it is too weak to trust."""
        resolved = self.get_resolved(key, namespace=namespace)
        if resolved is None:
            return default
        return resolved.value

    def get_resolved(self, key: str, *, namespace: str = "global") -> ResolvedPreference | None:
        with self.database.session() as session:
            row = session.scalar(
                select(Preference).where(
                    Preference.namespace == namespace,
                    Preference.key == key,
                    Preference.active.is_(True),
                )
            )
            if row is None:
                return None
            session.expunge(row)
        resolved = self._resolve(row)
        if not resolved.explicit and resolved.effective_confidence < self.confidence_floor:
            return None
        return resolved

    def all(
        self, *, namespace: str | None = None, include_weak: bool = False
    ) -> list[ResolvedPreference]:
        """Every active preference, strongest first."""
        with self.database.session() as session:
            statement = select(Preference).where(Preference.active.is_(True))
            if namespace is not None:
                statement = statement.where(Preference.namespace == namespace)
            rows = list(session.scalars(statement))
            for row in rows:
                session.expunge(row)

        resolved = [self._resolve(row) for row in rows]
        if not include_weak:
            resolved = [
                item
                for item in resolved
                if item.explicit or item.effective_confidence >= self.confidence_floor
            ]
        resolved.sort(key=lambda item: (item.explicit, item.effective_confidence), reverse=True)
        return resolved
