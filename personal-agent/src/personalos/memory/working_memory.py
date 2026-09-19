"""Working memory: the scratchpad for the task currently in flight.

Nothing here is persisted. When the task ends, the interesting parts are
promoted into episodic, semantic or preference memory and the rest is dropped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from personalos.utils.textutil import truncate


@dataclass
class WorkingMemory:
    """Everything the agent is holding in mind for one task."""

    task_id: str
    request: str
    project: str | None = None
    intent: str | None = None
    focus_paths: list[Path] = field(default_factory=list)
    plan_id: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    step_results: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    retrieved_memory: list[str] = field(default_factory=list)

    def remember(self, key: str, value: Any) -> None:
        """Record a fact discovered mid-task, e.g. how many PDFs were found."""
        self.facts[key] = value

    def record_step(self, step_id: str, result: Any) -> None:
        self.step_results[step_id] = result

    def note(self, text: str) -> None:
        self.notes.append(text)

    def add_focus(self, path: Path) -> None:
        if path not in self.focus_paths:
            self.focus_paths.append(path)

    def render(self, limit: int = 1200) -> str:
        """A compact summary for prompt injection."""
        lines = [f"Current request: {self.request}"]
        if self.project:
            lines.append(f"Active project: {self.project}")
        if self.focus_paths:
            lines.append("Files in focus: " + ", ".join(str(p) for p in self.focus_paths[:10]))
        for key, value in list(self.facts.items())[:15]:
            lines.append(f"- {key}: {truncate(str(value), 160)}")
        for note in self.notes[-5:]:
            lines.append(f"- note: {truncate(note, 160)}")
        return truncate("\n".join(lines), limit)
