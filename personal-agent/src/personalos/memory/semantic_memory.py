"""Semantic memory: durable facts about the user's world.

Examples: what the ``threat-research`` project is, which directory holds
invoices, what "the deck" usually refers to. Facts are namespaced so a project
can carry its own vocabulary without polluting the global namespace.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import or_, select

from personalos.database.database import Database
from personalos.database.models import MemorySource
from personalos.database.models import SemanticMemory as SemanticRow
from personalos.utils.textutil import keyword_set, sequence_similarity
from personalos.utils.timeutil import utcnow


@dataclass
class FactMatch:
    fact: SemanticRow
    score: float


class SemanticMemory:
    """CRUD and search over durable facts."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def remember(
        self,
        key: str,
        value: str,
        *,
        namespace: str = "global",
        category: str = "fact",
        source: str = MemorySource.OBSERVED,
        confidence: float = 0.6,
    ) -> SemanticRow:
        """Insert or update a fact.

        An explicitly stated fact always replaces an observed one; an observed
        fact never downgrades an explicit one, because explicit instructions
        outrank inference everywhere in this system.
        """
        with self.database.session() as session:
            row = session.scalar(
                select(SemanticRow).where(
                    SemanticRow.namespace == namespace, SemanticRow.key == key
                )
            )
            if row is None:
                row = SemanticRow(namespace=namespace, key=key)
                session.add(row)
            elif row.source == MemorySource.EXPLICIT and source != MemorySource.EXPLICIT:
                session.expunge(row)
                return row
            row.value = value
            row.category = category
            row.source = source
            row.confidence = 1.0 if source == MemorySource.EXPLICIT else confidence
            session.flush()
            session.refresh(row)
            session.expunge(row)
            return row

    def get(self, key: str, *, namespace: str = "global") -> SemanticRow | None:
        with self.database.session() as session:
            row = session.scalar(
                select(SemanticRow).where(
                    SemanticRow.namespace == namespace, SemanticRow.key == key
                )
            )
            if row is None:
                return None
            row.usage_count += 1
            row.last_used = utcnow()
            session.flush()
            session.expunge(row)
            return row

    def all(self, *, namespace: str | None = None) -> list[SemanticRow]:
        with self.database.session() as session:
            statement = select(SemanticRow).order_by(SemanticRow.namespace, SemanticRow.key)
            if namespace is not None:
                statement = statement.where(SemanticRow.namespace == namespace)
            rows = list(session.scalars(statement))
            for row in rows:
                session.expunge(row)
            return rows

    def search(
        self,
        query: str,
        *,
        namespaces: list[str] | None = None,
        limit: int = 5,
        minimum_score: float = 0.2,
    ) -> list[FactMatch]:
        """Rank facts by keyword overlap against ``query``."""
        with self.database.session() as session:
            statement = select(SemanticRow)
            if namespaces:
                statement = statement.where(
                    or_(*[SemanticRow.namespace == ns for ns in namespaces])
                )
            rows = list(session.scalars(statement.limit(1000)))
            for row in rows:
                session.expunge(row)

        query_words = keyword_set(query)
        matches: list[FactMatch] = []
        for row in rows:
            text = f"{row.key} {row.value}"
            row_words = keyword_set(text)
            overlap = len(query_words & row_words) / len(query_words) if query_words else 0.0
            score = (0.7 * overlap + 0.3 * sequence_similarity(query, text)) * row.confidence
            if score >= minimum_score:
                matches.append(FactMatch(fact=row, score=score))
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:limit]

    def forget(self, identifier: int | str, *, namespace: str = "global") -> bool:
        """Delete a fact by row id or by key."""
        with self.database.session() as session:
            row = (
                session.get(SemanticRow, identifier)
                if isinstance(identifier, int)
                else session.scalar(
                    select(SemanticRow).where(
                        SemanticRow.namespace == namespace, SemanticRow.key == identifier
                    )
                )
            )
            if row is None:
                return False
            session.delete(row)
            return True
