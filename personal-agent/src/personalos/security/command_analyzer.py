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


#: Programs that run another program given to them as an argument. Seeing one
#: of these means the *next* token is also a program worth reporting.
_WRAPPERS = frozenset({"sudo", "doas", "env", "nohup", "time", "xargs", "nice", "ionice", "timeout"})

#: Shells that take a script to run via `-c`. Their payload is a command line
#: in its own right and has to be analysed as one.
_SHELLS = frozenset({"sh", "bash", "zsh", "fish", "dash", "ksh", "csh", "tcsh", "ash"})

#: Recursion limit for nested `-c` payloads and command substitutions.
_MAX_NESTING = 5

#: Numbers and durations, which appear as wrapper arguments rather than programs.
_VALUE_ARG = re.compile(r"\d+(\.\d+)?[smhd]?")

#: Unquoted operators that end one command and begin another.
_SEPARATORS = (";;", "&&", "||", ";", "|", "&", "\n")


def _split_segments(command: str) -> list[str]:
    """Split a command line into individual commands on unquoted separators.

    ``shlex`` alone is not enough: it tokenises ``echo hi; payload`` as
    ``["echo", "hi;", "payload"]``, gluing the separator to the previous word,
    so a scan looking for a bare ``;`` token never sees the boundary and every
    program after it becomes invisible. Splitting on the raw string first, with
    quote tracking, is what makes the boundary reliable.
    """
    segments: list[str] = []
    current: list[str] = []
    quote: str | None = None
    index = 0

    while index < len(command):
        char = command[index]

        if char == "\\" and quote != "'" and index + 1 < len(command):
            current.append(command[index : index + 2])
            index += 2
            continue

        if quote is not None:
            current.append(char)
            if char == quote:
                quote = None
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
            current.append(char)
            index += 1
            continue

        for separator in _SEPARATORS:
            if command.startswith(separator, index):
                segments.append("".join(current))
                current = []
                index += len(separator)
                break
        else:
            current.append(char)
            index += 1

    segments.append("".join(current))
    return [segment.strip() for segment in segments if segment.strip()]


def _substitutions(command: str) -> list[str]:
    """Extract the bodies of ``$(...)`` and backtick command substitutions.

    Their contents run as commands, so they are analysed like any other segment.
    """
    found: list[str] = []
    index = 0
    while index < len(command):
        if command.startswith("$(", index):
            depth, cursor = 1, index + 2
            while cursor < len(command) and depth:
                if command.startswith("$(", cursor):
                    depth += 1
                    cursor += 2
                    continue
                if command[cursor] == ")":
                    depth -= 1
                cursor += 1
            found.append(command[index + 2 : cursor - 1])
            index = cursor
            continue
        if command[index] == "`":
            end = command.find("`", index + 1)
            if end == -1:
                break
            found.append(command[index + 1 : end])
            index = end + 1
            continue
        index += 1
    return [item.strip() for item in found if item.strip()]


def _programs_in_segment(
    segment: str, programs: list[str], errors: list[str], depth: int
) -> None:
    """Collect program names from one command, recursing into what it runs."""
    if depth > _MAX_NESTING:
        errors.append("nesting limit exceeded")
        return

    try:
        tokens = shlex.split(segment, comments=True)
    except ValueError as exc:
        errors.append(str(exc))
        return

    index = 0
    while index < len(tokens):
        token = tokens[index]
        # `FOO=bar cmd` — assignments precede the program they apply to.
        if "=" in token and not token.startswith("-") and "/" not in token.split("=")[0]:
            index += 1
            continue
        if token.startswith("-"):
            index += 1
            continue
        # A wrapper's own value argument, e.g. the `5` in `timeout 5 ls` or the
        # `10` in `nice -n 10 cmd`. Never a program name in practice.
        if _VALUE_ARG.fullmatch(token):
            index += 1
            continue

        name = token.rsplit("/", 1)[-1]
        if not name:
            index += 1
            continue
        programs.append(name)

        if name in _SHELLS:
            # Analyse the script passed to `-c` as a command line of its own.
            # It goes through _extract_programs, not straight back into this
            # function, so that a payload which itself chains or pipes is split
            # into segments rather than read as one command.
            for offset in range(index + 1, len(tokens)):
                if tokens[offset] == "-c" and offset + 1 < len(tokens):
                    nested, error = _extract_programs(tokens[offset + 1], depth + 1)
                    programs.extend(nested)
                    if error:
                        errors.append(error)
                    break
            return
        if name in _WRAPPERS:
            # Skip this wrapper's own flags and report what it goes on to run.
            index += 1
            continue
        return


def _extract_programs(command: str, depth: int = 0) -> tuple[list[str], str | None]:
    """Extract every program a command line would invoke.

    Handles chained commands, nested shell payloads, command substitutions and
    wrapper programs. Anything it cannot parse is reported as an error, which
    the caller treats as *higher* risk rather than lower.
    """
    programs: list[str] = []
    errors: list[str] = []

    for segment in _split_segments(command):
        _programs_in_segment(segment, programs, errors, depth)
        for substitution in _substitutions(segment):
            nested, error = _extract_programs(substitution, depth + 1)
            programs.extend(nested)
            if error:
                errors.append(error)

    # Preserve order while dropping repeats, so the concern line stays readable.
    seen: set[str] = set()
    ordered = [p for p in programs if not (p in seen or seen.add(p))]
    return ordered, errors[0] if errors else None


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
