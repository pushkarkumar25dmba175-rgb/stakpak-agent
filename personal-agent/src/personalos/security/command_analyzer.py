"""Static analysis of shell commands before they are allowed to run.

The analyser never decides *whether* a command runs — that is the policy
engine's job. It answers three questions and hands them over:

* what programs does this command line actually invoke?
* what is the highest risk any of them carries?
* which specific concerns should be shown to the user before they approve?

It is deliberately conservative. Anything it cannot parse is treated as
higher risk, not lower.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field

from personalos.security.risk import RiskLevel

#: Shell metacharacters that chain or redirect commands. Their presence means
#: the command line is more than a single program invocation.
_CHAIN_TOKENS = ("&&", "||", "|", ";", "\n", "$(", "`", ">", ">>", "<")

_DESTRUCTIVE_PATTERNS: list[tuple[str, re.Pattern[str], RiskLevel]] = [
    ("recursive delete", re.compile(r"\brm\s+(-[a-zA-Z]*\s+)*-?[a-zA-Z]*[rf]"), RiskLevel.HIGH),
    ("filesystem format", re.compile(r"\b(mkfs(\.\w+)?|diskpart|fdisk|format\s+[a-zA-Z]:)\b"), RiskLevel.HIGH),
    ("raw disk write", re.compile(r"\bdd\s+(if|of)="), RiskLevel.HIGH),
    ("fork bomb", re.compile(r":\s*\(\s*\)\s*\{.*\|.*&\s*\}\s*;"), RiskLevel.HIGH),
    ("power state change", re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), RiskLevel.HIGH),
    ("recursive ownership change", re.compile(r"\b(chown|chmod)\s+(-[a-zA-Z]*\s+)*-R\b"), RiskLevel.HIGH),
    ("history rewrite", re.compile(r"\bgit\s+(push\s+.*--force|reset\s+--hard|clean\s+-[a-zA-Z]*f)"), RiskLevel.HIGH),
]

_PRIVILEGE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("privilege escalation", re.compile(r"\b(sudo|doas|runas|pkexec|su)\b")),
    ("setuid bit", re.compile(r"\bchmod\s+[0-7]*[4-7][0-7]{3}\b")),
]

_CREDENTIAL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("credential store access", re.compile(r"(?i)(\.ssh/|\.aws/credentials|\.netrc|id_rsa|id_ed25519|\.gnupg|keychain|secretsdump|lsass)")),
    ("environment dump", re.compile(r"(?i)\b(env|printenv|set)\b\s*(\||>|$)")),
    ("shadow file access", re.compile(r"/etc/(shadow|sudoers)")),
]

_PERSISTENCE_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("persistence mechanism", re.compile(r"(?i)\b(crontab|systemctl\s+(enable|start)|launchctl|schtasks|reg\s+add)\b")),
    ("shell profile modification", re.compile(r"(?i)(\.bashrc|\.zshrc|\.profile|autorun|Startup\\)")),
    ("service management", re.compile(r"(?i)\b(service|systemctl)\s+(stop|disable|mask)\b")),
]

_NETWORK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("pipes a download into a shell", re.compile(r"(?i)\b(curl|wget)\b[^|;]*\|\s*(sudo\s+)?(ba|z|k)?sh")),
    ("outbound network call", re.compile(r"(?i)\b(curl|wget|nc|ncat|netcat|ssh|scp|rsync|ftp|telnet)\b")),
    ("package installation", re.compile(r"(?i)\b(apt|apt-get|yum|dnf|brew|pacman|choco|winget)\s+(install|remove|upgrade)|\bpip3?\s+install|\bnpm\s+(i|install)\b")),
]


@dataclass
class CommandAnalysis:
    """The verdict on one shell command line."""

    command: str
    programs: list[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    concerns: list[str] = field(default_factory=list)
    blocked_reason: str | None = None
    chained: bool = False
    parse_error: str | None = None

    @property
    def blocked(self) -> bool:
        return self.blocked_reason is not None

    def summary(self) -> str:
        """A single line suitable for an approval prompt."""
        parts = [f"risk {int(self.risk_level)} ({self.risk_level.label})"]
        if self.programs:
            parts.append("runs: " + ", ".join(self.programs))
        if self.concerns:
            parts.append("concerns: " + "; ".join(self.concerns))
        return " | ".join(parts)


def _extract_programs(command: str) -> tuple[list[str], str | None]:
    """Best-effort extraction of every program name on a command line."""
    try:
        tokens = shlex.split(command, comments=True)
    except ValueError as exc:
        return [], str(exc)

    programs: list[str] = []
    expect_program = True
    for token in tokens:
        if token in {"&&", "||", "|", ";", "&"}:
            expect_program = True
            continue
        if expect_program:
            # `sudo apt install` should report both `sudo` and `apt`.
            name = token.split("/")[-1]
            if name and not name.startswith("-") and "=" not in name:
                programs.append(name)
                expect_program = name in {"sudo", "doas", "env", "nohup", "time", "xargs"}
            else:
                expect_program = True
    return programs, None


def analyze_command(
    command: str,
    *,
    allowlist: list[str] | None = None,
    denied_patterns: list[str] | None = None,
) -> CommandAnalysis:
    """Classify ``command`` and collect the concerns a human should see.

    Args:
        command: The raw command line.
        allowlist: Program names that may run without escalating risk. When
            provided, any program outside it escalates to at least MODERATE.
        denied_patterns: Regular expressions that block the command outright.

    Returns:
        A :class:`CommandAnalysis`. ``blocked_reason`` set means the command
        must not run at all, regardless of who approves it.
    """
    analysis = CommandAnalysis(command=command)
    stripped = command.strip()
    if not stripped:
        analysis.blocked_reason = "The command is empty."
        return analysis

    analysis.chained = any(token in stripped for token in _CHAIN_TOKENS)
    programs, parse_error = _extract_programs(stripped)
    analysis.programs = programs
    analysis.parse_error = parse_error

    for raw_pattern in denied_patterns or []:
        try:
            if re.search(raw_pattern, stripped):
                analysis.blocked_reason = (
                    f"The command matches the configured deny pattern {raw_pattern!r}."
                )
                analysis.risk_level = RiskLevel.HIGH
                return analysis
        except re.error:
            # A broken pattern in config must not silently disable screening.
            analysis.concerns.append(f"deny pattern {raw_pattern!r} is not valid regex")

    level = RiskLevel.LOW

    for label, pattern, pattern_level in _DESTRUCTIVE_PATTERNS:
        if pattern.search(stripped):
            analysis.concerns.append(label)
            level = max(level, pattern_level)

    for groups, escalate_to in (
        (_PRIVILEGE_PATTERNS, RiskLevel.HIGH),
        (_CREDENTIAL_PATTERNS, RiskLevel.HIGH),
        (_PERSISTENCE_PATTERNS, RiskLevel.HIGH),
    ):
        for label, pattern in groups:
            if pattern.search(stripped):
                analysis.concerns.append(label)
                level = max(level, escalate_to)

    for label, pattern in _NETWORK_PATTERNS:
        if pattern.search(stripped):
            analysis.concerns.append(label)
            # Reaching the network is an external side effect by definition.
            level = max(level, RiskLevel.EXTERNAL if "pipes" in label else RiskLevel.HIGH)

    if parse_error:
        analysis.concerns.append(f"could not be parsed ({parse_error}), treated as higher risk")
        level = max(level, RiskLevel.HIGH)

    if analysis.chained:
        analysis.concerns.append("chains or redirects multiple commands")
        level = max(level, RiskLevel.MODERATE)

    if allowlist is not None:
        unknown = [p for p in programs if p not in allowlist]
        if unknown:
            analysis.concerns.append("not on the shell allowlist: " + ", ".join(sorted(set(unknown))))
            level = max(level, RiskLevel.MODERATE)
        if not programs and not parse_error:
            analysis.concerns.append("no recognisable program")
            level = max(level, RiskLevel.MODERATE)

    analysis.risk_level = RiskLevel(level)
    return analysis
