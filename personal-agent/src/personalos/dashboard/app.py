"""The optional local dashboard.

Bound to localhost, read-only by default, and it exposes exactly the views the
CLI already provides: tasks, memory, skills, schedules, permissions and the
audit log. Two things it deliberately does not do:

* **It does not expose approvals.** A browser tab is a poor place to make a
  consequential decision, and an HTTP endpoint that grants permissions is an
  endpoint worth attacking. Approvals stay on the terminal.
* **It does not bind to anything but loopback.** The settings validator refuses
  any other host, so this cannot be misconfigured into being reachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from personalos.agent.orchestrator import Agent
from personalos.audit.audit_log import AuditQuery
from personalos.errors import ConfigurationError
from personalos.settings import Settings
from personalos.utils.textutil import truncate

STATIC_DIR = Path(__file__).parent / "static"


def build_app(agent: Agent, settings: Settings) -> Any:
    """Construct the FastAPI application.

    Raises:
        ConfigurationError: when FastAPI is not installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import HTMLResponse
    except ImportError as exc:
        raise ConfigurationError(
            "The dashboard needs FastAPI.",
            remediation="Install it with `pip install 'personalos-agent[dashboard]'`.",
        ) from exc

    api = FastAPI(
        title="PersonalOS Agent",
        version="0.1.0",
        docs_url="/docs",
        description="Local, read-only view of what the agent knows and has done.",
    )

    @api.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = STATIC_DIR / "index.html"
        return page.read_text(encoding="utf-8") if page.exists() else "<h1>PersonalOS</h1>"

    @api.get("/api/status")
    def status() -> dict[str, Any]:
        return agent.status()

    @api.get("/api/tasks")
    def tasks(limit: int = 25) -> list[dict[str, Any]]:
        return [
            {
                "task_id": task.task_id,
                "request": task.request,
                "status": task.status,
                "success": task.success,
                "project": task.project,
                "max_risk_level": task.max_risk_level,
                "created_at": task.created_at.isoformat(),
                "summary": truncate(task.result_summary or "", 400),
            }
            for task in agent.tasks.recent(limit=limit)
        ]

    @api.get("/api/tasks/{task_id}")
    def task_detail(task_id: str) -> dict[str, Any]:
        task = agent.tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="No such task.")
        records = agent.audit.query(AuditQuery(task_id=task_id, limit=200))
        return {
            "task_id": task.task_id,
            "request": task.request,
            "status": task.status,
            "plan": task.plan,
            "summary": task.result_summary,
            "actions": [
                {
                    "action": record.action,
                    "tool": record.tool,
                    "risk_level": record.risk_level,
                    "approval_status": record.approval_status,
                    "success": record.success,
                    "verified": record.verified,
                    "verification_note": record.verification_note,
                    "error": record.error,
                    "created_at": record.created_at.isoformat(),
                }
                for record in records
            ],
        }

    @api.get("/api/memory")
    def memory() -> dict[str, Any]:
        return {
            "statistics": agent.memory.statistics(),
            "preferences": [
                {
                    "key": item.key,
                    "value": item.value,
                    "source": item.source,
                    "confidence": round(item.effective_confidence, 3),
                    "explicit": item.explicit,
                    "usage_count": item.usage_count,
                }
                for item in agent.memory.preferences.all()
            ],
            "workflows": [
                {
                    "name": workflow.name,
                    "signature": workflow.signature,
                    "success_count": workflow.success_count,
                    "promoted_skill": workflow.promoted_skill,
                }
                for workflow in agent.memory.workflows.all()
            ],
        }

    @api.get("/api/skills")
    def skills() -> list[dict[str, Any]]:
        return [
            {
                "name": item.definition.name,
                "description": item.definition.description,
                "version": item.definition.version,
                "origin": item.origin,
                "enabled": item.enabled,
                "created_by": item.definition.created_by,
                "run_count": item.run_count,
                "success_rate": item.success_rate,
                "max_risk_level": item.definition.max_risk_level,
            }
            for item in agent.skills.list_skills()
        ]

    @api.get("/api/schedules")
    def schedules() -> list[dict[str, Any]]:
        from personalos.scheduler.scheduler import Scheduler

        return [
            {
                "job_id": job.job_id,
                "name": job.name,
                "spec": job.spec,
                "kind": job.kind,
                "enabled": job.enabled,
                "max_risk_level": job.max_risk_level,
                "last_run": job.last_run.isoformat() if job.last_run else None,
                "last_status": job.last_status,
            }
            for job in Scheduler(agent).list_jobs()
        ]

    @api.get("/api/permissions")
    def permissions() -> dict[str, Any]:
        return {
            "paths": agent.paths.describe(),
            "approval_mode": str(settings.security.approval_mode),
            "auto_approve_max_level": settings.security.auto_approve_max_level,
            "rules": [
                {
                    "name": rule.name,
                    "effect": rule.effect,
                    "tool": rule.tool,
                    "operation": rule.operation,
                    "path_prefix": rule.path_prefix,
                    "max_risk_level": rule.max_risk_level,
                }
                for rule in agent.policy.list_rules()
            ],
        }

    @api.get("/api/audit")
    def audit(limit: int = 50, failures: bool = False) -> list[dict[str, Any]]:
        return [
            {
                "created_at": record.created_at.isoformat(),
                "task_id": record.task_id,
                "tool": record.tool,
                "action": record.action,
                "risk_level": record.risk_level,
                "approval_status": record.approval_status,
                "success": record.success,
                "execution_time_ms": record.execution_time_ms,
            }
            for record in agent.audit.query(AuditQuery(limit=limit, only_failures=failures))
        ]

    @api.get("/api/suggestions")
    def suggestions() -> list[dict[str, Any]]:
        return [
            {
                "id": item.id,
                "kind": item.kind,
                "title": item.title,
                "detail": item.detail,
                "confidence": item.confidence,
            }
            for item in agent.suggestions.pending()
        ]

    @api.get("/api/settings")
    def settings_view() -> dict[str, Any]:
        from personalos.settings import dump_settings

        return dump_settings(settings)

    return api


def serve(agent: Agent, settings: Settings) -> None:
    """Run the dashboard until interrupted.

    Raises:
        ConfigurationError: when uvicorn is missing or the host is not loopback.
    """
    try:
        import uvicorn
    except ImportError as exc:
        raise ConfigurationError(
            "The dashboard needs uvicorn.",
            remediation="Install it with `pip install 'personalos-agent[dashboard]'`.",
        ) from exc

    if settings.dashboard.host not in {"127.0.0.1", "localhost", "::1"}:
        raise ConfigurationError(
            f"Refusing to bind the dashboard to {settings.dashboard.host}.",
            remediation="The dashboard is localhost-only. Use an SSH tunnel for remote access.",
        )

    api = build_app(agent, settings)
    uvicorn.run(api, host=settings.dashboard.host, port=settings.dashboard.port, log_level="warning")
