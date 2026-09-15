"""The policy engine: the one place that decides whether an action may proceed.

Order of evaluation, highest authority first:

1. **Hard blocks.** A denied path, a denied command pattern, a disabled tool.
   Nothing overrides these — not a rule, not the user saying yes.
2. **Level 3 and 4.** Always escalated to a human, every single time. Standing
   rules are not even consulted, and repetition never earns an exemption.
3. **Deny rules.** A matching deny rule refuses the action.
4. **Allow rules.** A matching allow rule can auto-approve levels 0–2.
5. **Level defaults.** The documented behaviour per level.

The engine is pure with respect to the database: it reads rules, it never
writes them. Creating a rule is an explicit user action.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select

from personalos.database.database import Database
from personalos.database.models import ApprovalStatus, PolicyRule
from personalos.security.permissions import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
)
from personalos.security.risk import RiskLevel
from personalos.settings import ApprovalMode, Settings
from personalos.utils.paths import expand_path
from personalos.utils.timeutil import utcnow


@dataclass
class PolicyOutcome:
    """The engine's verdict, ready to be logged verbatim."""

    decision: ApprovalDecision
    risk_level: RiskLevel
    escalated_to_user: bool
    blocked: bool = False

    @property
    def approved(self) -> bool:
        return self.decision.approved


class PolicyEngine:
    """Evaluates :class:`ApprovalRequest` objects against configuration and rules."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        handler: ApprovalHandler,
    ) -> None:
        self.settings = settings
        self.database = database
        self.handler = handler

    # ---- rule matching -----------------------------------------------------
    def _active_rules(self, project: str | None) -> list[PolicyRule]:
        now = utcnow()
        with self.database.session() as session:
            rules = list(session.scalars(select(PolicyRule).where(PolicyRule.active.is_(True))))
        result = []
        for rule in rules:
            if rule.expires_at is not None and rule.expires_at <= now:
                continue
            if rule.project is not None and rule.project != project:
                continue
            result.append(rule)
        return result

    @staticmethod
    def _rule_matches(rule: PolicyRule, request: ApprovalRequest) -> bool:
        if rule.tool is not None and rule.tool != request.tool:
            return False
        if rule.operation is not None and rule.operation != request.operation:
            return False
        if rule.path_prefix:
            prefix = str(expand_path(rule.path_prefix))
            resources = request.affected_resources or []
            if not resources:
                return False
            if not all(str(expand_path(r)).startswith(prefix) for r in resources):
                return False
        return True

    # ---- evaluation --------------------------------------------------------
    def evaluate(self, request: ApprovalRequest) -> PolicyOutcome:
        """Decide what happens to ``request``, prompting the user when required."""
        level = request.risk_level

        # 1. Hard blocks are expressed by the caller via `concerns` + a level of
        #    HIGH plus `reversible=False`; an explicit block short-circuits here.
        if request.tool == "shell" and not self.settings.security.shell_enabled:
            return PolicyOutcome(
                decision=ApprovalDecision.denied(
                    "Shell execution is disabled in configuration.",
                ),
                risk_level=level,
                escalated_to_user=False,
                blocked=True,
            )

        mode = self.settings.security.approval_mode

        # 2. Levels 3 and 4 always go to a human, in every mode, every time.
        if level.always_requires_confirmation:
            decision = self.handler.request_approval(request)
            return PolicyOutcome(decision, level, escalated_to_user=True)

        # 3./4. Standing rules, deny first.
        rules = self._active_rules(request.project)
        if mode is not ApprovalMode.STRICT:
            for rule in rules:
                if rule.effect == "deny" and self._rule_matches(rule, request):
                    return PolicyOutcome(
                        decision=ApprovalDecision.denied(
                            f"Denied by your standing rule {rule.name!r}.",
                            rule_name=rule.name,
                        ),
                        risk_level=level,
                        escalated_to_user=False,
                        blocked=True,
                    )
            for rule in rules:
                if (
                    rule.effect == "allow"
                    and int(level) <= rule.max_risk_level
                    and self._rule_matches(rule, request)
                ):
                    return PolicyOutcome(
                        decision=ApprovalDecision.by_rule(
                            rule.name,
                            f"Covered by your standing rule {rule.name!r}.",
                        ),
                        risk_level=level,
                        escalated_to_user=False,
                    )

        # 5. Level defaults.
        if level == RiskLevel.READ_ONLY:
            return PolicyOutcome(
                ApprovalDecision.auto("Read-only actions run automatically."),
                level,
                escalated_to_user=False,
            )

        ceiling = RiskLevel.parse(self.settings.security.auto_approve_max_level)
        if mode is ApprovalMode.STRICT:
            ceiling = RiskLevel.READ_ONLY
        elif mode is ApprovalMode.RELAXED:
            ceiling = max(ceiling, RiskLevel.MODERATE)

        if level <= ceiling:
            return PolicyOutcome(
                ApprovalDecision.auto(
                    f"Level {int(level)} ({level.label}) is within your automatic ceiling "
                    f"of {int(ceiling)} in {mode.value} mode."
                ),
                level,
                escalated_to_user=False,
            )

        decision = self.handler.request_approval(request)
        return PolicyOutcome(decision, level, escalated_to_user=True)

    # ---- rule management ---------------------------------------------------
    def add_rule(
        self,
        name: str,
        *,
        effect: str = "allow",
        tool: str | None = None,
        operation: str | None = None,
        path_prefix: str | None = None,
        max_risk_level: int = 2,
        project: str | None = None,
        source: str = "explicit_user_instruction",
    ) -> PolicyRule:
        """Store a standing rule.

        Allow rules are clamped to level 2: a rule can shorten the approval
        path for routine local edits, never for deletions or external effects.
        """
        clamped = min(int(max_risk_level), int(RiskLevel.MODERATE)) if effect == "allow" else int(max_risk_level)
        with self.database.session() as session:
            existing = session.scalar(select(PolicyRule).where(PolicyRule.name == name))
            if existing is None:
                existing = PolicyRule(name=name)
                session.add(existing)
            existing.effect = effect
            existing.tool = tool
            existing.operation = operation
            existing.path_prefix = path_prefix
            existing.max_risk_level = clamped
            existing.project = project
            existing.source = source
            existing.active = True
            session.flush()
            session.refresh(existing)
            session.expunge(existing)
            return existing

    def list_rules(self, *, include_inactive: bool = False) -> list[PolicyRule]:
        with self.database.session() as session:
            statement = select(PolicyRule).order_by(PolicyRule.name)
            if not include_inactive:
                statement = statement.where(PolicyRule.active.is_(True))
            rules = list(session.scalars(statement))
            for rule in rules:
                session.expunge(rule)
            return rules

    def revoke_rule(self, name: str) -> bool:
        """Deactivate a rule. Returns False when there was nothing to revoke."""
        with self.database.session() as session:
            rule = session.scalar(select(PolicyRule).where(PolicyRule.name == name))
            if rule is None or not rule.active:
                return False
            rule.active = False
            return True


__all__ = ["ApprovalStatus", "PolicyEngine", "PolicyOutcome"]
