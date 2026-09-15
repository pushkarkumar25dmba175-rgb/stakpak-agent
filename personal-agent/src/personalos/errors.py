"""Exception hierarchy shared by every PersonalOS subsystem.

Every error carries a short, user-facing ``message`` and an optional
``remediation`` string. The CLI prints the remediation when present, which is
how the agent explains a failure instead of dumping a traceback.
"""

from __future__ import annotations


class PersonalOSError(Exception):
    """Base class for all errors raised deliberately by PersonalOS."""

    def __init__(self, message: str, *, remediation: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.remediation = remediation

    def __str__(self) -> str:  # pragma: no cover - trivial
        if self.remediation:
            return f"{self.message}\nSuggestion: {self.remediation}"
        return self.message


class ConfigurationError(PersonalOSError):
    """Configuration is missing or internally inconsistent."""


class PermissionDeniedError(PersonalOSError):
    """The policy engine or the user refused an action."""


class ApprovalRequiredError(PersonalOSError):
    """An action needs explicit approval and none was available."""


class ToolError(PersonalOSError):
    """A tool failed while executing."""


class ToolValidationError(ToolError):
    """Tool inputs failed validation before execution."""


class ToolNotFoundError(ToolError):
    """A plan referenced a tool that is not registered."""


class SandboxViolationError(PersonalOSError):
    """An operation tried to leave its allowed working area."""


class SkillError(PersonalOSError):
    """A skill is malformed, missing, or disabled."""


class MemoryError_(PersonalOSError):
    """Memory could not be read or written."""


class PlanningError(PersonalOSError):
    """The planner could not produce an executable plan."""


class VerificationError(PersonalOSError):
    """A step executed but its result could not be verified."""


class LLMError(PersonalOSError):
    """An LLM provider failed or is not configured."""


class RollbackError(PersonalOSError):
    """A rollback could not be completed."""
