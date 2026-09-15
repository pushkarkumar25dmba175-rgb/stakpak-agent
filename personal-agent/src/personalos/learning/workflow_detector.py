"""Noticing repeated workflows and offering — only offering — to automate them.

This is the component the specification's example is about: after the same
"download → rename → summarise → move" sequence happens a few times, the agent
says so and asks whether to make it reusable. It never makes that decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from personalos.database.models import Suggestion, Workflow
from personalos.learning.pattern_detector import DetectedPattern, find_repeated
from personalos.learning.suggestions import SuggestionDraft, SuggestionStore
from personalos.memory.memory_manager import MemoryManager
from personalos.settings import LearningSettings
from personalos.utils.textutil import slugify, truncate


@dataclass
class WorkflowObservation:
    """A pattern that has been recorded as a workflow, plus any suggestion raised."""

    pattern: DetectedPattern
    workflow: Workflow
    suggestion: Suggestion | None = None


class WorkflowDetector:
    """Scans episodic memory for recurring task shapes."""

    def __init__(
        self,
        memory: MemoryManager,
        suggestions: SuggestionStore,
        settings: LearningSettings,
    ) -> None:
        self.memory = memory
        self.suggestions = suggestions
        self.settings = settings

    @staticmethod
    def _workflow_name(pattern: DetectedPattern) -> str:
        """A stable, readable name derived from the pattern itself."""
        tools = [part.split(".")[0] for part in pattern.signature.split(" > ") if part]
        seen: list[str] = []
        for tool in tools:
            if tool not in seen:
                seen.append(tool)
        stem = "_".join(seen[:3]) or "workflow"
        return slugify(f"{stem}_{abs(hash(pattern.signature)) % 10000}")

    def scan(self, *, limit: int = 300) -> list[WorkflowObservation]:
        """Look for repeats and record them. Returns what was found.

        Suggestions are only attached once a pattern crosses the configured
        occurrence threshold *and* the suggestion store's cooldown allows it.
        """
        if not self.settings.enabled:
            return []

        episodes = self.memory.episodic.recent(limit=limit)
        patterns = find_repeated(
            episodes,
            threshold=self.settings.pattern_similarity_threshold,
            minimum_occurrences=2,
        )

        observations: list[WorkflowObservation] = []
        for pattern in patterns:
            existing = self.memory.workflows.by_signature(pattern.signature)
            name = existing.name if existing else self._workflow_name(pattern)
            workflow = self.memory.workflows.record(
                name=name,
                signature=pattern.signature,
                steps=self._steps_from(pattern),
                description=f"Observed {pattern.occurrences} times: {truncate(pattern.example_requests[0] if pattern.example_requests else pattern.signature, 120)}",
                example_request=pattern.example_requests[0] if pattern.example_requests else None,
                project=pattern.project,
                success=False,
            )
            # ``record`` counts observations; the tally that matters for
            # suggesting is the cluster size, which is what the user actually did.
            observation = WorkflowObservation(pattern=pattern, workflow=workflow)

            if (
                pattern.occurrences >= self.settings.min_occurrences_for_suggestion
                and not workflow.promoted_skill
            ):
                observation.suggestion = self.suggestions.offer(
                    SuggestionDraft(
                        kind="workflow",
                        key=pattern.signature,
                        title=(
                            f"You have run this sequence {pattern.occurrences} times. "
                            "Shall I turn it into a reusable skill?"
                        ),
                        detail=(
                            f"Steps: {pattern.describe()}\n"
                            f"For example: {truncate(pattern.example_requests[0], 160) if pattern.example_requests else '(no example recorded)'}\n"
                            "Accepting writes a skill definition you can read and edit. "
                            "It does not run anything, and it does not change what needs approval."
                        ),
                        payload={"workflow": workflow.name, "signature": pattern.signature},
                        confidence=min(0.95, 0.5 + pattern.occurrences * 0.1),
                    )
                )
            observations.append(observation)
        return observations

    @staticmethod
    def _steps_from(pattern: DetectedPattern) -> list[dict[str, object]]:
        """Reconstruct step definitions from the most recent episode in a cluster.

        The newest run is used because it reflects the current shape of the
        workflow, including any corrections the user made along the way.
        """
        newest = max(pattern.episodes, key=lambda episode: episode.created_at, default=None)
        if newest is None:
            return []
        steps: list[dict[str, object]] = []
        for index, action in enumerate(newest.actions or [], start=1):
            if not isinstance(action, dict):
                continue
            steps.append(
                {
                    "id": action.get("id") or f"step_{index}",
                    "tool": action.get("tool"),
                    "operation": action.get("operation"),
                    "description": action.get("description") or "",
                    "arguments": action.get("arguments") or {},
                    "risk_level": action.get("risk_level", 0),
                }
            )
        return steps
