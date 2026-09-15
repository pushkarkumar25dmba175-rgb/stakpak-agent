"""The memory façade: one object the rest of the agent talks to.

Its most important job is *not* storing things — it is deciding what small
subset of everything the agent knows is worth spending context on for the task
at hand. The whole database is never sent to the model. Each request gets:

1. a classification of the request,
2. a ranked search across preferences, facts, past episodes and workflows,
3. a hard token budget, filled highest-value first.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from personalos.database.database import Database
from personalos.database.models import MemorySource
from personalos.memory.episodic_memory import EpisodicMemory
from personalos.memory.preferences import PreferenceStore, ResolvedPreference
from personalos.memory.semantic_memory import SemanticMemory
from personalos.memory.vector_index import VectorIndex, build_vector_index
from personalos.memory.workflow_memory import WorkflowMemory
from personalos.memory.working_memory import WorkingMemory
from personalos.settings import Settings
from personalos.utils.textutil import token_estimate, truncate


@dataclass
class MemoryItem:
    """A single retrieved piece of context, ready to be rendered."""

    kind: str
    """preference | fact | episode | workflow"""
    text: str
    score: float
    identifier: str | None = None

    @property
    def tokens(self) -> int:
        return token_estimate(self.text)


@dataclass
class RetrievedContext:
    """What retrieval selected, and what it had to leave out."""

    items: list[MemoryItem] = field(default_factory=list)
    budget_tokens: int = 0
    used_tokens: int = 0
    dropped: int = 0

    def render(self) -> str:
        """Group items by kind into a compact block for the prompt."""
        if not self.items:
            return ""
        buckets: dict[str, list[str]] = {}
        for item in self.items:
            buckets.setdefault(item.kind, []).append(item.text)
        headings = {
            "preference": "Your stated preferences and rules",
            "fact": "Things I know about your setup",
            "episode": "Similar tasks I have done before",
            "workflow": "Workflows you repeat",
        }
        sections = []
        for kind, lines in buckets.items():
            sections.append(headings.get(kind, kind) + ":")
            sections.extend(f"  - {line}" for line in lines)
        return "\n".join(sections)

    def summary(self) -> str:
        return (
            f"{len(self.items)} memories, {self.used_tokens}/{self.budget_tokens} tokens"
            + (f", {self.dropped} dropped for budget" if self.dropped else "")
        )


#: Coarse intent labels. Used to bias retrieval and to pick a planning strategy.
INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "organise": ("organize", "organise", "tidy", "sort", "clean up", "file", "categorise", "categorize"),
    "summarise": ("summarize", "summarise", "summary", "tl;dr", "digest", "brief"),
    "search": ("find", "search", "locate", "look for", "where is", "grep"),
    "report": ("report", "write up", "index", "document", "generate"),
    "automate": ("automate", "every day", "schedule", "recurring", "skill", "workflow"),
    "edit": ("edit", "rename", "move", "rewrite", "update", "replace", "fix"),
    "run": ("run", "execute", "script", "command", "install", "build"),
    "inspect": ("show", "list", "inspect", "what is", "read", "open", "check"),
}


def classify_intent(request: str) -> str:
    """Label a request so retrieval and planning can specialise.

    Deliberately keyword-based: it runs with no model available, which matters
    because ``agent doctor`` and the offline provider both rely on it.
    """
    lowered = request.lower()
    scores: dict[str, int] = {}
    for intent, keywords in INTENT_KEYWORDS.items():
        hits = sum(1 for keyword in keywords if keyword in lowered)
        if hits:
            scores[intent] = hits
    if not scores:
        return "general"
    return max(scores.items(), key=lambda pair: pair[1])[0]


class MemoryManager:
    """Owns every memory layer and performs budgeted retrieval."""

    def __init__(self, settings: Settings, database: Database) -> None:
        self.settings = settings
        self.database = database
        self.episodic = EpisodicMemory(
            database, retention_days=settings.memory.episodic_retention_days
        )
        self.semantic = SemanticMemory(database)
        self.preferences = PreferenceStore(
            database,
            half_life_days=settings.memory.confidence_half_life_days,
            confidence_floor=settings.memory.inferred_confidence_floor,
        )
        self.workflows = WorkflowMemory(database)
        self.vector_index: VectorIndex = build_vector_index(settings.memory.vector_backend)
        self._working: dict[str, WorkingMemory] = {}

    # ---- working memory ----------------------------------------------------
    def open_working(self, task_id: str, request: str, *, project: str | None = None) -> WorkingMemory:
        working = WorkingMemory(
            task_id=task_id,
            request=request,
            project=project,
            intent=classify_intent(request),
        )
        self._working[task_id] = working
        return working

    def working(self, task_id: str) -> WorkingMemory | None:
        return self._working.get(task_id)

    def close_working(self, task_id: str) -> WorkingMemory | None:
        return self._working.pop(task_id, None)

    # ---- retrieval ---------------------------------------------------------
    def retrieve(
        self,
        request: str,
        *,
        project: str | None = None,
        budget_tokens: int | None = None,
        top_k: int | None = None,
    ) -> RetrievedContext:
        """Select the memories worth spending context on for ``request``."""
        budget = budget_tokens or self.settings.llm.memory_budget_tokens
        limit = top_k or self.settings.memory.retrieval_top_k
        namespaces = ["global"] + ([project] if project else [])

        candidates: list[MemoryItem] = []

        # Preferences come first: they are instructions, not background.
        for preference in self._relevant_preferences(request, namespaces):
            weight = 1.0 if preference.explicit else preference.effective_confidence
            candidates.append(
                MemoryItem(
                    kind="preference",
                    text=preference.render(),
                    # Explicit preferences are pinned above everything else.
                    score=2.0 if preference.explicit else 1.0 + weight,
                    identifier=f"{preference.namespace}/{preference.key}",
                )
            )

        for match in self.semantic.search(request, namespaces=namespaces, limit=limit):
            candidates.append(
                MemoryItem(
                    kind="fact",
                    text=f"{match.fact.key}: {truncate(match.fact.value, 200)}",
                    score=match.score,
                    identifier=str(match.fact.id),
                )
            )

        for match in self.episodic.search(request, limit=limit):
            episode = match.episode
            outcome = "succeeded" if episode.success else "failed"
            feedback = f" You said: {truncate(episode.user_feedback, 120)}" if episode.user_feedback else ""
            candidates.append(
                MemoryItem(
                    kind="episode",
                    text=(
                        f"{truncate(episode.task_text, 120)} → {outcome}"
                        f" via [{truncate(episode.action_signature, 120)}].{feedback}"
                    ),
                    score=match.score,
                    identifier=episode.task_id,
                )
            )

        for workflow in self.workflows.all(project=project)[:limit]:
            if workflow.success_count < 2:
                continue
            candidates.append(
                MemoryItem(
                    kind="workflow",
                    text=(
                        f"{workflow.name}: {truncate(workflow.description or workflow.signature, 160)} "
                        f"(ran {workflow.success_count}×)"
                    ),
                    score=0.4 + min(0.4, workflow.success_count / 25),
                    identifier=workflow.name,
                )
            )

        # Vector hits, when a backend is configured, are merged in as episodes.
        for hit in self.vector_index.search(request, limit=limit):
            candidates.append(
                MemoryItem(
                    kind="episode",
                    text=str(hit.payload.get("text", hit.identifier)),
                    score=hit.score,
                    identifier=hit.identifier,
                )
            )

        return self._fit_to_budget(candidates, budget)

    def _relevant_preferences(
        self, request: str, namespaces: list[str]
    ) -> list[ResolvedPreference]:
        """All trustworthy preferences for the active namespaces.

        Preferences are few and they are instructions, so they are not filtered
        by similarity to the request — a rule about filename formats matters
        even when the request never says "filename".
        """
        collected: dict[str, ResolvedPreference] = {}
        for namespace in namespaces:
            for preference in self.preferences.all(namespace=namespace):
                # A project-scoped preference overrides the global one.
                collected[preference.key] = preference
        return list(collected.values())

    @staticmethod
    def _fit_to_budget(candidates: list[MemoryItem], budget: int) -> RetrievedContext:
        """Greedily fill the budget, highest score first."""
        ordered = sorted(candidates, key=lambda item: item.score, reverse=True)
        context = RetrievedContext(budget_tokens=budget)
        seen: set[str] = set()
        for item in ordered:
            fingerprint = f"{item.kind}:{item.text}"
            if fingerprint in seen:
                continue
            if context.used_tokens + item.tokens > budget:
                context.dropped += 1
                continue
            seen.add(fingerprint)
            context.items.append(item)
            context.used_tokens += item.tokens
        return context

    # ---- convenience -------------------------------------------------------
    def learn_preference(
        self,
        key: str,
        value: Any,
        *,
        explicit: bool,
        namespace: str = "global",
        confidence: float = 0.6,
        note: str | None = None,
    ) -> ResolvedPreference:
        """Record a preference with the right provenance."""
        return self.preferences.set(
            key,
            value,
            namespace=namespace,
            source=MemorySource.EXPLICIT if explicit else MemorySource.OBSERVED,
            confidence=confidence,
            note=note,
        )

    def statistics(self) -> dict[str, int]:
        """Row counts, for `agent status` and `agent memory`."""
        from sqlalchemy import func, select

        from personalos.database.models import (
            Episode,
            Preference,
            Workflow,
        )
        from personalos.database.models import (
            SemanticMemory as SemanticRow,
        )

        with self.database.session() as session:
            return {
                "episodes": int(session.scalar(select(func.count()).select_from(Episode)) or 0),
                "facts": int(session.scalar(select(func.count()).select_from(SemanticRow)) or 0),
                "preferences": int(session.scalar(select(func.count()).select_from(Preference)) or 0),
                "workflows": int(session.scalar(select(func.count()).select_from(Workflow)) or 0),
            }
