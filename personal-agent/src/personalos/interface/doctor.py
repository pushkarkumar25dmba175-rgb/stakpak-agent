"""`agent doctor` — check that the installation is sane before trusting it."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from personalos.agent.orchestrator import Agent
from personalos.security.risk import RiskLevel
from personalos.settings import Settings


@dataclass
class Check:
    """One diagnostic result."""

    name: str
    status: str
    """ok | warn | fail"""
    detail: str

    @property
    def symbol(self) -> str:
        return {"ok": "✓", "warn": "•", "fail": "✗"}[self.status]

    @property
    def style(self) -> str:
        return {"ok": "green", "warn": "yellow", "fail": "red"}[self.status]


def _optional(module: str, feature: str, install: str) -> Check:
    available = importlib.util.find_spec(module) is not None
    return Check(
        name=feature,
        status="ok" if available else "warn",
        detail=f"{module} is installed" if available else f"not installed — {install}",
    )


def run_checks(settings: Settings, agent: Agent | None = None) -> list[Check]:
    """Run every diagnostic and return the results."""
    checks: list[Check] = []

    checks.append(
        Check(
            "python",
            "ok" if sys.version_info >= (3, 12) else "fail",
            f"running {sys.version.split()[0]} (3.12+ required)",
        )
    )

    missing = [d for d in settings.directories() if not d.exists()]
    checks.append(
        Check(
            "directories",
            "ok" if not missing else "fail",
            f"{settings.home} is set up"
            if not missing
            else "missing: " + ", ".join(str(d) for d in missing),
        )
    )

    roots = [Path(root).expanduser() for root in settings.workspace.allowed_roots]
    missing_roots = [root for root in roots if not root.exists()]
    if not roots:
        checks.append(Check("workspace", "fail", "no allowed roots configured; nothing is reachable"))
    elif missing_roots:
        checks.append(
            Check(
                "workspace",
                "warn",
                "these roots do not exist yet: " + ", ".join(str(r) for r in missing_roots),
            )
        )
    else:
        checks.append(Check("workspace", "ok", f"{len(roots)} allowed root(s)"))

    # Security posture. These are the settings someone could loosen and forget.
    ceiling = RiskLevel.parse(settings.security.auto_approve_max_level)
    checks.append(
        Check(
            "approval policy",
            "ok" if ceiling <= RiskLevel.LOW else "warn",
            f"{settings.security.approval_mode} mode, auto-approving up to {risk_phrase(ceiling)}",
        )
    )
    checks.append(
        Check(
            "log redaction",
            "ok" if settings.security.redact_logs else "warn",
            "credentials are redacted from logs"
            if settings.security.redact_logs
            else "REDACTION IS OFF — secrets may be written to logs",
        )
    )
    checks.append(
        Check(
            "deletion policy",
            "ok" if settings.security.prefer_trash_over_delete else "warn",
            "deletions go to the agent trash"
            if settings.security.prefer_trash_over_delete
            else "deletions are permanent by default",
        )
    )
    checks.append(
        Check(
            "observation",
            "ok",
            "folder watching "
            + ("on" if settings.observation.watch_folders_enabled else "off")
            + ", clipboard "
            + ("on" if settings.observation.watch_clipboard_enabled else "off")
            + ", app tracking "
            + ("on" if settings.observation.track_app_usage else "off"),
        )
    )

    # LLM configuration.
    key_present = bool(os.environ.get(settings.llm.api_key_env))
    if settings.llm.provider in {"anthropic", "openai"} and not key_present:
        checks.append(
            Check(
                "llm",
                "warn",
                f"${settings.llm.api_key_env} is not set — running with the offline planner",
            )
        )
    else:
        checks.append(Check("llm", "ok", f"{settings.llm.provider}/{settings.llm.model}"))

    for module, feature, install in (
        ("apscheduler", "scheduler", "pip install 'personalos-agent[scheduler]'"),
        ("watchdog", "folder watching", "pip install 'personalos-agent[watch]'"),
        ("fastapi", "dashboard", "pip install 'personalos-agent[dashboard]'"),
        ("pypdf", "PDF reading", "pip install 'personalos-agent[docs]'"),
    ):
        checks.append(_optional(module, feature, install))

    if agent is not None:
        checks.append(
            Check("tools", "ok", ", ".join(agent.registry.available()))
        )
        try:
            stats = agent.memory.statistics()
            checks.append(
                Check(
                    "database",
                    "ok",
                    ", ".join(f"{key}: {value}" for key, value in stats.items()),
                )
            )
        except Exception as exc:  # noqa: BLE001
            checks.append(Check("database", "fail", str(exc)))

        broken = agent.skills.registry.errors
        checks.append(
            Check(
                "skills",
                "ok" if not broken else "warn",
                f"{len(agent.skills.list_skills())} loaded"
                + (f", {len(broken)} file(s) failed to parse" if broken else ""),
            )
        )

        ok, detail = asyncio.run(agent.provider.health_check())
        checks.append(Check("llm reachability", "ok" if ok else "warn", detail))

    return checks


def risk_phrase(level: RiskLevel) -> str:
    return f"level {int(level)} ({level.label})"
