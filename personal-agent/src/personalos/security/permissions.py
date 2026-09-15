"""Approval requests, decisions and the handler protocol.

The agent never decides for itself that it has permission. It builds an
:class:`ApprovalRequest` describing exactly what is about to happen, the policy
engine says whether that can be auto-approved, and anything else is handed to
an :class:`ApprovalHandler` — the CLI prompt, a dashboard, or in unattended
contexts a handler that refuses everything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from personalos.database.models import ApprovalStatus
from personalos.security.risk import RiskLevel


@dataclass
class ApprovalRequest:
    """Everything a human needs in order to answer "may I do this?".

    ``affected_resources`` and ``reversible`` exist so the prompt can state the
    blast radius rather than just the tool name.
    """

    task_id: str
    step_id: str | None
    tool: str
    operation: str
    description: str
    risk_level: RiskLevel
    parameters: dict[str, object] = field(default_factory=dict)
    affected_resources: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    reversible: bool = True
    rollback_hint: str | None = None
    project: str | None = None

    def headline(self) -> str:
        return f"{self.tool}.{self.operation} — {self.description}"


@dataclass
class ApprovalDecision:
    """The answer, plus why it came out that way."""

    approved: bool
    status: ApprovalStatus
    reason: str
    rule_name: str | None = None
    remember_as_rule: bool = False

    @classmethod
    def auto(cls, reason: str) -> ApprovalDecision:
        return cls(True, ApprovalStatus.AUTO_APPROVED, reason)

    @classmethod
    def by_rule(cls, rule_name: str, reason: str) -> ApprovalDecision:
        return cls(True, ApprovalStatus.RULE_APPROVED, reason, rule_name=rule_name)

    @classmethod
    def by_user(cls, reason: str = "Approved by the user.", *, remember: bool = False) -> ApprovalDecision:
        return cls(True, ApprovalStatus.USER_APPROVED, reason, remember_as_rule=remember)

    @classmethod
    def denied(cls, reason: str, *, rule_name: str | None = None) -> ApprovalDecision:
        return cls(False, ApprovalStatus.USER_DENIED, reason, rule_name=rule_name)


@runtime_checkable
class ApprovalHandler(Protocol):
    """Anything that can answer an :class:`ApprovalRequest`."""

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        """Ask a human (or a stand-in) and return their decision."""
        ...


class DenyAllHandler:
    """Refuses anything that needs a human. The safe default for unattended runs.

    Scheduled jobs and folder watchers use this: if a job turns out to need
    confirmation, it stops and records why instead of proceeding unsupervised.
    """

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.denied(
            f"{request.headline()} needs confirmation, but nobody is available to give it.",
        )


class AutoApproveHandler:
    """Approves up to a ceiling. Intended for tests and `--yes` style runs.

    The ceiling is clamped to level 2: this handler cannot be used to wave
    through deletions, installs or external side effects, no matter what it is
    constructed with.
    """

    MAX_ALLOWED = RiskLevel.MODERATE

    def __init__(self, max_level: RiskLevel | int = RiskLevel.MODERATE) -> None:
        requested = RiskLevel.parse(max_level)
        self.max_level = min(requested, self.MAX_ALLOWED)

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.risk_level <= self.max_level:
            return ApprovalDecision.by_user(
                f"Auto-approved at or below level {int(self.max_level)}."
            )
        return ApprovalDecision.denied(
            f"{request.headline()} is level {int(request.risk_level)}, "
            f"above the automatic ceiling of {int(self.max_level)}."
        )


class RecordingHandler:
    """Wraps another handler and remembers every request. Used in tests."""

    def __init__(self, inner: ApprovalHandler) -> None:
        self.inner = inner
        self.requests: list[ApprovalRequest] = []
        self.decisions: list[ApprovalDecision] = []

    def request_approval(self, request: ApprovalRequest) -> ApprovalDecision:
        self.requests.append(request)
        decision = self.inner.request_approval(request)
        self.decisions.append(decision)
        return decision
