"""Proactive suggestions — and the limits on them.

A suggestion is a sentence, never an action. Three rules keep the agent from
becoming the thing that interrupts you all day:

* **Cooldown.** The same kind of suggestion is not raised again for a
  configured period, even if the condition still holds.
* **Per-session cap.** At most N suggestions surface in one sitting.
* **Deduplication.** A suggestion whose key matches a pending or dismissed one
  is dropped silently.

Accepting a suggestion does not run anything either. It creates a skill
definition, which you then have to run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import select

from personalos.database.database import Database
from personalos.database.models import Suggestion
from personalos.utils.timeutil import utcnow


@dataclass
class SuggestionDraft:
    """A suggestion before it is stored."""

    kind: str
    key: str
    """Stable identity for deduplication, e.g. the workflow signature."""
    title: str
    detail: str = ""
    payload: dict[str, Any] | None = None
    confidence: float = 0.6


class SuggestionStore:
    """Stores suggestions and enforces the interruption limits."""

    def __init__(
        self,
        database: Database,
        *,
        cooldown_hours: float = 12.0,
        max_per_session: int = 3,
    ) -> None:
        self.database = database
        self.cooldown_hours = cooldown_hours
        self.max_per_session = max_per_session
        self._emitted_this_session = 0

    def offer(self, draft: SuggestionDraft) -> Suggestion | None:
        """Record a suggestion, or return ``None`` if it should not be raised."""
        if self._emitted_this_session >= self.max_per_session:
            return None

        payload = dict(draft.payload or {})
        payload["key"] = draft.key
        cutoff = utcnow() - timedelta(hours=self.cooldown_hours)

        with self.database.session() as session:
            existing = list(
                session.scalars(select(Suggestion).where(Suggestion.kind == draft.kind))
            )
            for row in existing:
                same_key = (row.payload or {}).get("key") == draft.key
                if same_key and row.status == "pending":
                    return None
                if same_key and row.resolved_at and row.resolved_at > cutoff:
                    return None
                if same_key and row.status == "accepted":
                    return None

            suggestion = Suggestion(
                kind=draft.kind,
                title=draft.title,
                detail=draft.detail,
                payload=payload,
                confidence=draft.confidence,
            )
            session.add(suggestion)
            session.flush()
            session.refresh(suggestion)
            session.expunge(suggestion)

        self._emitted_this_session += 1
        return suggestion

    def pending(self, limit: int = 20) -> list[Suggestion]:
        with self.database.session() as session:
            rows = list(
                session.scalars(
                    select(Suggestion)
                    .where(Suggestion.status == "pending")
                    .order_by(Suggestion.created_at.desc())
                    .limit(limit)
                )
            )
            for row in rows:
                session.expunge(row)
            return rows

    def get(self, suggestion_id: int) -> Suggestion | None:
        with self.database.session() as session:
            row = session.get(Suggestion, suggestion_id)
            if row is not None:
                session.expunge(row)
            return row

    def resolve(self, suggestion_id: int, status: str) -> bool:
        """Mark a suggestion accepted, dismissed or expired."""
        with self.database.session() as session:
            row = session.get(Suggestion, suggestion_id)
            if row is None:
                return False
            row.status = status
            row.resolved_at = utcnow()
            return True

    def reset_session(self) -> None:
        """Clear the per-session cap. Called when a new CLI session starts."""
        self._emitted_this_session = 0
