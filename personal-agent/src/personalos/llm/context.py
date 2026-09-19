"""Prompt assembly under a hard token budget.

The rule this module enforces is the one from the specification: *do not send
the entire memory database to the model*. A prompt is built from ranked parts,
each declaring how essential it is, and parts are dropped from the least
essential upward until the whole thing fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from personalos.llm.base import Message
from personalos.utils.textutil import token_estimate, truncate


@dataclass
class ContextPart:
    """One labelled section of the prompt."""

    label: str
    content: str
    priority: int = 5
    """1 is most essential. Parts are dropped highest-number-first."""
    truncatable: bool = True

    @property
    def tokens(self) -> int:
        return token_estimate(self.content)

    def render(self) -> str:
        return f"## {self.label}\n{self.content}"


@dataclass
class ContextBudget:
    """Accumulates parts and renders them within ``max_tokens``."""

    max_tokens: int
    reserved_for_output: int = 0
    parts: list[ContextPart] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)
    truncated: list[str] = field(default_factory=list)

    @property
    def available(self) -> int:
        return max(0, self.max_tokens - self.reserved_for_output)

    def add(self, label: str, content: str, *, priority: int = 5, truncatable: bool = True) -> None:
        """Queue a section. Empty content is ignored."""
        if not content or not content.strip():
            return
        self.parts.append(
            ContextPart(label=label, content=content.strip(), priority=priority, truncatable=truncatable)
        )

    def build(self) -> str:
        """Render the prompt, dropping and truncating to fit the budget.

        Essential parts (priority 1) are never dropped — if they alone exceed
        the budget, the caller has a configuration problem, and silently
        omitting the instructions would produce a confidently wrong answer
        instead of an error.
        """
        ordered = sorted(self.parts, key=lambda part: part.priority)
        kept: list[ContextPart] = []
        used = 0
        for part in ordered:
            remaining = self.available - used
            if part.tokens <= remaining:
                kept.append(part)
                used += part.tokens
                continue
            if part.priority == 1:
                kept.append(part)
                used += part.tokens
                continue
            if part.truncatable and remaining > 120:
                shortened = truncate(part.content, remaining * 4)
                kept.append(ContextPart(part.label, shortened, part.priority, True))
                self.truncated.append(part.label)
                used += token_estimate(shortened)
                continue
            self.dropped.append(part.label)

        kept.sort(key=lambda part: ordered.index(part) if part in ordered else 0)
        return "\n\n".join(part.render() for part in kept)

    def summary(self) -> str:
        parts = [f"{len(self.parts)} sections"]
        if self.truncated:
            parts.append("truncated: " + ", ".join(self.truncated))
        if self.dropped:
            parts.append("dropped: " + ", ".join(self.dropped))
        return "; ".join(parts)


def user_message(content: str) -> Message:
    return Message("user", content)


def assistant_message(content: str) -> Message:
    return Message("assistant", content)
