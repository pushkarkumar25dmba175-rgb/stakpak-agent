"""Security: risk classification, approvals, sandboxing, secrets, injection defence."""

from personalos.security.command_analyzer import CommandAnalysis, analyze_command
from personalos.security.injection import UntrustedContent, scan_for_injection, wrap_untrusted
from personalos.security.permissions import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    AutoApproveHandler,
    DenyAllHandler,
    RecordingHandler,
)
from personalos.security.policy_engine import PolicyEngine, PolicyOutcome
from personalos.security.risk import DEFAULT_POLICY, RiskLevel, escalate
from personalos.security.sandbox import ProcessResult, Sandbox, SubprocessSandbox
from personalos.security.secrets import (
    SecretRef,
    SecretResolver,
    redact,
    redact_structure,
    safe_environment,
)

__all__ = [
    "DEFAULT_POLICY",
    "ApprovalDecision",
    "ApprovalHandler",
    "ApprovalRequest",
    "AutoApproveHandler",
    "CommandAnalysis",
    "DenyAllHandler",
    "PolicyEngine",
    "PolicyOutcome",
    "ProcessResult",
    "RecordingHandler",
    "RiskLevel",
    "Sandbox",
    "SecretRef",
    "SecretResolver",
    "SubprocessSandbox",
    "UntrustedContent",
    "analyze_command",
    "escalate",
    "redact",
    "redact_structure",
    "safe_environment",
    "scan_for_injection",
    "wrap_untrusted",
]
