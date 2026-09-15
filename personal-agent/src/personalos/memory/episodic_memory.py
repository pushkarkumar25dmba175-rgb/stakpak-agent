"""Episodic memory: a record of past tasks and how they went."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from sqlalchemy import delete, select

from personalos.database.database import Database
from personalos.database.models import Episode
from personalos.utils.textutil import keyword_set, sequence_similarity
from personalos.utils.timeutil import utcnow


@dataclass
class EpisodeMatch:
    """An episode plus why it was considered relevant."""

    episode: Episode
    score: float
    reason: str


class EpisodicMemory:
    """Stores and searches past tasks.

    Search is lexical by default (keyword overlap plus sequence similarity).
    The scoring is isolated in :meth:`_score` so a vector backend can replace
    it without touching callers.
    """

    def __init__(self, database: Database, *, retention_days: int = 365) -> None:
        self.database = database
        self.retention_days = retention_days

    def save(
        self,
        *,
        task_id: str,
        task_text: str,
        actions: Sequence[dict[str, Any]],
        result: str | None,
        success: bool,
        intent: str | None = None,
        project: str | None = None,
        user_feedback: str | None = None,
    ) -> Episode:
        """Persist one completed task."""
        signature = self.build_signature(actions)
        with self.database.session() as session:
            episode = Episode(
                task_id=task_id,
                project=project,
                task_text=task_text,
                intent=intent,
                action_signature=signature,
                actions=list(actions),
                result=result,
                success=success,
                user_feedback=user_feedback,
            )
            session.add(episode)
            session.flush()
            session.refresh(episode)
            session.expunge(episode)
            return episode

    @staticmethod
    def build_signature(actions: Sequence[dict[str, Any]]) -> str:
        """Fingerprint a task as an ordered ``tool.operation`` chain.

        Parameters are deliberately excluded: "rename then summarise then move"
        is the same shape whether it ran on three files or thirty.
        """
        parts = []
        for action in actions:
            tool = str(action.get("tool", "?"))
            operation = str(action.get("operation") or action.get("action") or "")
            parts.append(f"{tool}.{operation}" if operation else tool)
        return " > ".join(parts)

    def recent(self, limit: int = 20, *, project: str | None = None) -> list[Episode]:
        with self.database.session() as session:
            statement = select(Episode).order_by(Episode.created_at.desc()).limit(limit)
            if project is not None:
                statement = statement.where(Episode.project == project)
            episodes = list(session.scalars(statement))
            for episode in episodes:
                session.expunge(episode)
            return episodes

    def by_signature(self, signature: str, *, limit: int = 50) -> list[Episode]:
        with self.database.session() as session:
            episodes = list(
                session.scalars(
                    select(Episode)
                    .where(Episode.action_signature == signature)
                    .order_by(Episode.created_at.desc())
                    .limit(limit)
                )
            )
            for episode in episodes:
                session.expunge(episode)
            return episodes

    @staticmethod
    def _score(query: str, episode: Episode) -> float:
        """Blend keyword overlap with sequence similarity, favouring successes."""
        query_words = keyword_set(query)
        episode_words = keyword_set(f"{episode.task_text} {episode.result or ''}")
        overlap = (
            len(query_words & episode_words) / len(query_words) if query_words else 0.0
        )
        similarity = sequence_similarity(query, episode.task_text)
        score = 0.6 * overlap + 0.4 * similarity
        if not episode.success:
            # Failures are still worth retrieving — they tell the planner what
            # not to repeat — but a success is the better precedent.
            score *= 0.8
        return score

    def search(self, query: str, *, limit: int = 5, minimum_score: float = 0.15) -> list[EpisodeMatch]:
        """Return the most relevant past episodes for ``query``."""
        with self.database.session() as session:
            candidates = list(
                session.scalars(select(Episode).order_by(Episode.created_at.desc()).limit(500))
            )
            for episode in candidates:
                session.expunge(episode)

        scored = [
            EpisodeMatch(
                episode=episode,
                score=self._score(query, episode),
                reason="similar past task" if episode.success else "similar past task that failed",
            )
            for episode in candidates
        ]
        scored = [match for match in scored if match.score >= minimum_score]
        scored.sort(key=lambda match: match.score, reverse=True)
        return scored[:limit]

    def add_feedback(self, task_id: str, feedback: str) -> bool:
        """Attach the user's correction to the episode it refers to."""
        with self.database.session() as session:
            episode = session.scalar(
                select(Episode).where(Episode.task_id == task_id).order_by(Episode.id.desc())
            )
            if episode is None:
                return False
            episode.user_feedback = feedback
            return True

    def forget(self, episode_id: int) -> bool:
        with self.database.session() as session:
            episode = session.get(Episode, episode_id)
            if episode is None:
                return False
            session.delete(episode)
            return True

    def prune(self) -> int:
        """Delete episodes past the retention window. Returns how many went."""
        cutoff = utcnow() - timedelta(days=self.retention_days)
        with self.database.session() as session:
            result = session.execute(delete(Episode).where(Episode.created_at < cutoff))
            return int(result.rowcount or 0)
