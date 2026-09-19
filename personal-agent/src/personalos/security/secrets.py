"""Secret handling: resolution by reference, and redaction everywhere else.

Two rules drive this module:

1. The agent never *stores* a credential. It stores a reference such as
   ``env:GITHUB_TOKEN`` or ``keyring:personalos/github``, and resolves it at
   the moment of use.
2. Anything on its way to a log, the database, the model or the screen goes
   through :func:`redact` first.

Redaction is pattern-based and therefore imperfect; it is a safety net, not a
licence to route secrets through the agent.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from personalos.errors import ConfigurationError

PLACEHOLDER = "«redacted:{kind}»"

#: Ordered patterns. Earlier, more specific patterns win.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.DOTALL)),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,255}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b")),
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b")),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\b")),
    ("bearer_token", re.compile(r"(?i)\b(?:authorization|bearer)\s*[:=]\s*[A-Za-z0-9._~+/=-]{12,}")),
    ("connection_string", re.compile(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s:@/]+:([^\s@/]+)@")),
    ("assignment", re.compile(
        r"(?i)\b([A-Z0-9_]*(?:PASSWORD|PASSWD|SECRET|TOKEN|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE[_-]?KEY|CREDENTIAL)[A-Z0-9_]*)"
        r"\s*[:=]\s*[\"']?([^\s\"';,]{4,})[\"']?"
    )),
]

#: Environment variable names whose value is treated as secret on sight.
_SENSITIVE_ENV_HINTS = re.compile(
    r"(?i)(password|passwd|secret|token|api[_-]?key|access[_-]?key|private[_-]?key|credential|session[_-]?id)"
)


def _mask(kind: str) -> str:
    return PLACEHOLDER.format(kind=kind)


def redact(text: str) -> str:
    """Replace anything that looks like a credential with a placeholder."""
    if not text:
        return text
    redacted = text
    for kind, pattern in _PATTERNS:
        if kind == "connection_string":
            redacted = pattern.sub(
                lambda m: m.group(0).replace(m.group(1), _mask("password")), redacted
            )
        elif kind == "assignment":
            redacted = pattern.sub(lambda m: f"{m.group(1)}={_mask('value')}", redacted)
        else:
            redacted = pattern.sub(_mask(kind), redacted)
    return redacted


def redact_structure(value: Any, *, _depth: int = 0) -> Any:
    """Recursively redact strings inside dicts, lists and tuples.

    Keys that *look* sensitive have their whole value replaced, because a
    pattern match on the value alone would miss e.g. a four-character PIN.
    """
    if _depth > 12:
        return value
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        result: dict[Any, Any] = {}
        for key, item in value.items():
            if isinstance(key, str) and _SENSITIVE_ENV_HINTS.search(key):
                result[key] = _mask("value")
            else:
                result[key] = redact_structure(item, _depth=_depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        rendered = [redact_structure(item, _depth=_depth + 1) for item in value]
        return type(value)(rendered) if isinstance(value, tuple) else rendered
    return value


def looks_sensitive(name: str) -> bool:
    """Whether an environment variable name should be withheld from the model."""
    return bool(_SENSITIVE_ENV_HINTS.search(name))


def safe_environment(base: dict[str, str] | None = None) -> dict[str, str]:
    """Return an environment with sensitive variables stripped out.

    This is what child processes launched by the shell and Python tools get, so
    a script the agent runs cannot read the user's API keys by accident.
    """
    source = dict(os.environ if base is None else base)
    return {k: v for k, v in source.items() if not looks_sensitive(k)}


@dataclass(frozen=True)
class SecretRef:
    """A pointer to a secret, safe to store and safe to log."""

    scheme: str
    identifier: str

    def __str__(self) -> str:
        return f"{self.scheme}:{self.identifier}"

    @classmethod
    def parse(cls, reference: str) -> SecretRef:
        scheme, _, identifier = reference.partition(":")
        if not identifier:
            raise ConfigurationError(
                f"{reference!r} is not a secret reference.",
                remediation="Use the form `env:NAME` or `keyring:service/user`.",
            )
        return cls(scheme.strip().lower(), identifier.strip())


class SecretResolver:
    """Resolves :class:`SecretRef` values from the environment or OS keyring.

    ``keyring`` is optional; when the package is not installed the resolver
    says so instead of failing obscurely, and the user can fall back to an
    environment variable.
    """

    def __init__(self, environ: dict[str, str] | None = None) -> None:
        self._environ = environ if environ is not None else os.environ

    def resolve(self, reference: str | SecretRef) -> str:
        ref = SecretRef.parse(reference) if isinstance(reference, str) else reference
        if ref.scheme == "env":
            value = self._environ.get(ref.identifier)
            if not value:
                raise ConfigurationError(
                    f"Environment variable {ref.identifier} is not set.",
                    remediation=f"Export {ref.identifier}, or add it to your .env file.",
                )
            return value
        if ref.scheme == "keyring":
            return self._from_keyring(ref.identifier)
        raise ConfigurationError(
            f"Unsupported secret scheme {ref.scheme!r}.",
            remediation="Supported schemes are `env:` and `keyring:`.",
        )

    def try_resolve(self, reference: str | SecretRef) -> str | None:
        """Like :meth:`resolve` but returns ``None`` instead of raising."""
        try:
            return self.resolve(reference)
        except ConfigurationError:
            return None

    @staticmethod
    def _from_keyring(identifier: str) -> str:
        try:
            import keyring  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on host
            raise ConfigurationError(
                "The `keyring` package is not installed, so OS credential storage is unavailable.",
                remediation="Run `pip install keyring`, or use an `env:` reference instead.",
            ) from exc
        service, _, username = identifier.partition("/")
        value = keyring.get_password(service, username or "default")
        if not value:
            raise ConfigurationError(
                f"No keyring entry found for {identifier}.",
                remediation=f"Store one with `keyring set {service} {username or 'default'}`.",
            )
        return value
