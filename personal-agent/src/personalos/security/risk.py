"""Risk levels and the default approval rules attached to them.

These five levels are the backbone of the security model. Every tool declares
a *base* level, every concrete invocation can be escalated above it (never
below), and the policy engine decides what to do based on the final level.
"""

from __future__ import annotations

from enum import IntEnum


class RiskLevel(IntEnum):
    """How much damage an action could do if it is wrong."""

    READ_ONLY = 0
    """Listing, reading, searching, summarising. No observable side effects."""

    LOW = 1
    """Additive local changes: create a new file, copy something."""

    MODERATE = 2
    """Changes existing local state: rename, move, overwrite, edit."""

    HIGH = 3
    """Destructive or system-level: delete, install, privileged commands."""

    EXTERNAL = 4
    """Effects outside this machine: send, publish, deploy, pay, call an API."""

    @property
    def label(self) -> str:
        return {
            RiskLevel.READ_ONLY: "read-only",
            RiskLevel.LOW: "low-risk local change",
            RiskLevel.MODERATE: "moderate local change",
            RiskLevel.HIGH: "high-risk action",
            RiskLevel.EXTERNAL: "external side effect",
        }[self]

    @property
    def description(self) -> str:
        return {
            RiskLevel.READ_ONLY: "Inspects information without changing anything.",
            RiskLevel.LOW: "Adds something new without touching what is already there.",
            RiskLevel.MODERATE: "Modifies, renames or moves things that already exist.",
            RiskLevel.HIGH: "Deletes data, installs software or changes system configuration.",
            RiskLevel.EXTERNAL: "Has effects outside this computer that cannot be undone locally.",
        }[self]

    @property
    def always_requires_confirmation(self) -> bool:
        """Levels 3 and 4 can never be waived by a standing rule or a habit."""
        return self >= RiskLevel.HIGH

    @property
    def requires_immediate_confirmation(self) -> bool:
        """Level 4 must be confirmed immediately before execution, not up front."""
        return self >= RiskLevel.EXTERNAL

    @classmethod
    def parse(cls, value: int | str | RiskLevel) -> RiskLevel:
        """Coerce ints, names and levels into a :class:`RiskLevel`."""
        if isinstance(value, RiskLevel):
            return value
        if isinstance(value, int):
            return cls(value)
        text = str(value).strip().upper().replace("-", "_").replace(" ", "_")
        if text.isdigit():
            return cls(int(text))
        aliases = {
            "READ": cls.READ_ONLY,
            "READONLY": cls.READ_ONLY,
            "READ_ONLY": cls.READ_ONLY,
            "LOW": cls.LOW,
            "MODERATE": cls.MODERATE,
            "MEDIUM": cls.MODERATE,
            "HIGH": cls.HIGH,
            "EXTERNAL": cls.EXTERNAL,
        }
        if text not in aliases:
            raise ValueError(f"Unknown risk level: {value!r}")
        return aliases[text]


#: The documented default policy. ``auto`` means the agent may proceed without
#: asking; ``confirm`` means a human has to say yes for this specific action.
DEFAULT_POLICY: dict[RiskLevel, str] = {
    RiskLevel.READ_ONLY: "auto",
    RiskLevel.LOW: "auto_if_requested",
    RiskLevel.MODERATE: "confirm_unless_rule",
    RiskLevel.HIGH: "confirm",
    RiskLevel.EXTERNAL: "confirm_immediately",
}


def escalate(base: RiskLevel, *reasons: RiskLevel | None) -> RiskLevel:
    """Return the highest of ``base`` and any escalations.

    Risk only ever goes up. A tool that declares itself read-only cannot be
    talked down to something safer by its inputs, but it *can* be pushed up —
    for example, ``shell`` is level 1 by default and level 3 when the command
    matches a destructive pattern.
    """
    highest = base
    for reason in reasons:
        if reason is not None and reason > highest:
            highest = reason
    return highest
