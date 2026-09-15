"""Finding repeated shapes in what the user has asked for.

The unit of comparison is the *action signature* — the ordered chain of
``tool.operation`` calls a task produced. "Rename, summarise, move" is the same
shape whether it ran over three files or thirty, and that is exactly the
recurrence worth noticing.

Signatures are compared with a similarity ratio rather than exact equality, so
a run that had one extra listing step still clusters with the others.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from personalos.database.models import Episode
from personalos.utils.textutil import sequence_similarity


@dataclass
class DetectedPattern:
    """A cluster of similar, successful past tasks."""

    signature: str
    occurrences: int
    episodes: list[Episode] = field(default_factory=list)
    example_requests: list[str] = field(default_factory=list)
    cohesion: float = 1.0
    """Mean similarity within the cluster. Lower means a looser grouping."""

    @property
    def project(self) -> str | None:
        projects = {episode.project for episode in self.episodes if episode.project}
        return projects.pop() if len(projects) == 1 else None

    def describe(self) -> str:
        steps = self.signature.replace(" > ", " → ")
        return f"{steps} ({self.occurrences} times)"


def cluster_signatures(
    episodes: list[Episode],
    *,
    threshold: float = 0.78,
    successful_only: bool = True,
) -> list[DetectedPattern]:
    """Group episodes whose action signatures are similar enough.

    Args:
        threshold: Minimum similarity for two signatures to join a cluster.
        successful_only: Ignore failed tasks. A workflow worth suggesting is
            one that has actually worked, more than once.

    Returns:
        Clusters sorted by size, largest first.
    """
    candidates = [
        episode
        for episode in episodes
        if episode.action_signature and (episode.success or not successful_only)
    ]

    # Exact matches first — the common case, and it keeps the O(n²) pass small.
    exact: dict[str, list[Episode]] = defaultdict(list)
    for episode in candidates:
        exact[episode.action_signature].append(episode)

    clusters: list[DetectedPattern] = []
    for signature, members in exact.items():
        clusters.append(
            DetectedPattern(
                signature=signature,
                occurrences=len(members),
                episodes=list(members),
                example_requests=[m.task_text for m in members][:5],
            )
        )

    # Then merge clusters whose signatures are merely similar.
    merged: list[DetectedPattern] = []
    for cluster in sorted(clusters, key=lambda c: c.occurrences, reverse=True):
        for existing in merged:
            similarity = sequence_similarity(existing.signature, cluster.signature)
            if similarity >= threshold:
                existing.occurrences += cluster.occurrences
                existing.episodes.extend(cluster.episodes)
                existing.example_requests = (
                    existing.example_requests + cluster.example_requests
                )[:5]
                existing.cohesion = min(existing.cohesion, similarity)
                break
        else:
            merged.append(cluster)

    merged.sort(key=lambda c: c.occurrences, reverse=True)
    return merged


def find_repeated(
    episodes: list[Episode],
    *,
    threshold: float = 0.78,
    minimum_occurrences: int = 3,
) -> list[DetectedPattern]:
    """Clusters that have recurred often enough to be worth mentioning."""
    return [
        pattern
        for pattern in cluster_signatures(episodes, threshold=threshold)
        if pattern.occurrences >= minimum_occurrences
    ]
