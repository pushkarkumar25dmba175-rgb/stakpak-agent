"""The policy engine: what runs automatically, what asks, and what never bends."""

from __future__ import annotations

from personalos.database.models import ApprovalStatus
from personalos.security.permissions import (
    ApprovalRequest,
    AutoApproveHandler,
    DenyAllHandler,
    RecordingHandler,
)
from personalos.security.policy_engine import PolicyEngine
from personalos.security.risk import RiskLevel
from personalos.settings import ApprovalMode


def make_request(level: RiskLevel, *, tool: str = "filesystem", operation: str = "write", paths=None):
    return ApprovalRequest(
        task_id="t1",
        step_id="s1",
        tool=tool,
        operation=operation,
        description="do a thing",
        risk_level=level,
        affected_resources=paths or [],
    )


def test_read_only_runs_without_asking(policy: PolicyEngine, handler: RecordingHandler) -> None:
    outcome = policy.evaluate(make_request(RiskLevel.READ_ONLY, operation="read"))
    assert outcome.approved
    assert outcome.decision.status is ApprovalStatus.AUTO_APPROVED
    assert handler.requests == []


def test_moderate_escalates_to_the_user(policy: PolicyEngine, handler: RecordingHandler) -> None:
    outcome = policy.evaluate(make_request(RiskLevel.MODERATE))
    assert outcome.escalated_to_user
    assert len(handler.requests) == 1


def test_high_risk_always_asks_even_with_a_matching_allow_rule(
    policy: PolicyEngine, handler: RecordingHandler
) -> None:
    policy.add_rule("trust-fs", effect="allow", tool="filesystem", max_risk_level=4)
    outcome = policy.evaluate(make_request(RiskLevel.HIGH, operation="delete"))
    assert outcome.escalated_to_user, "level 3 must never be covered by a standing rule"
    assert len(handler.requests) == 1


def test_external_actions_always_ask(policy: PolicyEngine, handler: RecordingHandler) -> None:
    policy.add_rule("trust-all", effect="allow", max_risk_level=4)
    outcome = policy.evaluate(make_request(RiskLevel.EXTERNAL, tool="http", operation="request"))
    assert outcome.escalated_to_user
    assert len(handler.requests) == 1


def test_allow_rules_are_clamped_to_level_two(policy: PolicyEngine) -> None:
    rule = policy.add_rule("wide", effect="allow", max_risk_level=4)
    assert rule.max_risk_level == int(RiskLevel.MODERATE)


def test_allow_rule_covers_moderate_without_asking(
    policy: PolicyEngine, handler: RecordingHandler
) -> None:
    policy.add_rule("fs-moves", effect="allow", tool="filesystem", operation="write")
    outcome = policy.evaluate(make_request(RiskLevel.MODERATE))
    assert outcome.approved
    assert outcome.decision.status is ApprovalStatus.RULE_APPROVED
    assert handler.requests == []


def test_deny_rule_beats_allow_rule(policy: PolicyEngine, handler: RecordingHandler) -> None:
    policy.add_rule("fs-allow", effect="allow", tool="filesystem")
    policy.add_rule("fs-deny", effect="deny", tool="filesystem", operation="write")
    outcome = policy.evaluate(make_request(RiskLevel.MODERATE))
    assert not outcome.approved
    assert outcome.blocked
    assert handler.requests == []


def test_path_scoped_rule_only_matches_that_path(policy: PolicyEngine, tmp_path) -> None:
    allowed = tmp_path / "project"
    policy.add_rule("scoped", effect="allow", tool="filesystem", path_prefix=str(allowed))

    inside = policy.evaluate(make_request(RiskLevel.MODERATE, paths=[str(allowed / "a.txt")]))
    assert inside.decision.status is ApprovalStatus.RULE_APPROVED

    outside = policy.evaluate(make_request(RiskLevel.MODERATE, paths=[str(tmp_path / "other.txt")]))
    assert outside.escalated_to_user


def test_strict_mode_ignores_standing_rules(settings, database, handler) -> None:
    settings.security.approval_mode = ApprovalMode.STRICT
    engine = PolicyEngine(settings, database, handler)
    engine.add_rule("fs-allow", effect="allow", tool="filesystem")
    outcome = engine.evaluate(make_request(RiskLevel.LOW))
    assert outcome.escalated_to_user


def test_relaxed_mode_still_stops_at_level_three(settings, database, handler) -> None:
    settings.security.approval_mode = ApprovalMode.RELAXED
    engine = PolicyEngine(settings, database, handler)
    assert engine.evaluate(make_request(RiskLevel.MODERATE)).decision.status is ApprovalStatus.AUTO_APPROVED
    assert engine.evaluate(make_request(RiskLevel.HIGH, operation="delete")).escalated_to_user


def test_disabled_shell_is_blocked_without_asking(settings, database, handler) -> None:
    settings.security.shell_enabled = False
    engine = PolicyEngine(settings, database, handler)
    outcome = engine.evaluate(make_request(RiskLevel.LOW, tool="shell", operation="run"))
    assert outcome.blocked
    assert not outcome.approved
    assert handler.requests == []


def test_deny_all_handler_refuses_everything(settings, database) -> None:
    engine = PolicyEngine(settings, database, DenyAllHandler())
    outcome = engine.evaluate(make_request(RiskLevel.HIGH, operation="delete"))
    assert not outcome.approved
    assert "nobody is available" in outcome.decision.reason


def test_auto_approve_handler_cannot_be_raised_past_level_two() -> None:
    handler = AutoApproveHandler(RiskLevel.EXTERNAL)
    assert handler.max_level is RiskLevel.MODERATE
    decision = handler.request_approval(make_request(RiskLevel.HIGH))
    assert not decision.approved


def test_revoking_a_rule_restores_the_prompt(policy: PolicyEngine, handler: RecordingHandler) -> None:
    policy.add_rule("temp", effect="allow", tool="filesystem", operation="write")
    assert policy.evaluate(make_request(RiskLevel.MODERATE)).approved
    assert policy.revoke_rule("temp")
    assert policy.evaluate(make_request(RiskLevel.MODERATE)).escalated_to_user
    assert policy.revoke_rule("temp") is False
