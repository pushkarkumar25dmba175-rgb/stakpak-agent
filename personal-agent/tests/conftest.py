"""Shared fixtures.

Every test runs against a throwaway PersonalOS home under ``tmp_path`` and the
offline echo provider, so the suite never touches the developer's real state
and never needs a network or an API key.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from personalos.agent.orchestrator import Agent
from personalos.audit.audit_log import AuditLog
from personalos.audit.rollback_journal import RollbackJournal
from personalos.database.database import build_database
from personalos.llm.providers.echo import EchoProvider
from personalos.memory.memory_manager import MemoryManager
from personalos.security.permissions import AutoApproveHandler, RecordingHandler
from personalos.security.policy_engine import PolicyEngine
from personalos.security.sandbox import SubprocessSandbox
from personalos.settings import Settings, load_settings
from personalos.tools import build_default_registry
from personalos.tools.base import ToolContext
from personalos.utils.paths import PathResolver


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """A directory the agent is allowed to work in."""
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def settings(tmp_path: Path, workspace: Path) -> Settings:
    """Settings pointing at a throwaway home, with the offline provider."""
    home = tmp_path / "home"
    resolved = load_settings(
        home=home,
        use_env=False,
        overrides={
            "workspace": {"allowed_roots": [str(workspace)], "denied_paths": []},
            "llm": {"provider": "echo", "model": "echo"},
            "learning": {"min_occurrences_for_suggestion": 2, "suggestion_cooldown_hours": 0},
        },
    )
    resolved.ensure_directories()
    return resolved


@pytest.fixture
def database(settings: Settings):
    db = build_database(settings.database_path)
    yield db
    db.dispose()


@pytest.fixture
def memory(settings: Settings, database) -> MemoryManager:
    return MemoryManager(settings, database)


@pytest.fixture
def audit(settings: Settings, database) -> AuditLog:
    return AuditLog(settings.logs_dir, database)


@pytest.fixture
def journal(settings: Settings, database) -> RollbackJournal:
    return RollbackJournal(database, settings.home / "backups")


@pytest.fixture
def handler() -> RecordingHandler:
    """An approval handler that records every request it is shown."""
    return RecordingHandler(AutoApproveHandler(2))


@pytest.fixture
def policy(settings: Settings, database, handler: RecordingHandler) -> PolicyEngine:
    return PolicyEngine(settings, database, handler)


@pytest.fixture
def registry():
    return build_default_registry(include_optional=True)


@pytest.fixture
def resolver(settings: Settings) -> PathResolver:
    return PathResolver(
        [*settings.workspace.allowed_roots, str(settings.home)],
        settings.workspace.denied_paths,
    )


@pytest.fixture
def tool_context(
    settings: Settings, resolver: PathResolver, journal: RollbackJournal, audit: AuditLog
) -> ToolContext:
    return ToolContext(
        settings=settings,
        paths=resolver,
        journal=journal,
        audit=audit,
        sandbox=SubprocessSandbox(),
        task_id="test-task",
    )


@pytest.fixture
def agent(settings: Settings, database, handler: RecordingHandler) -> Agent:
    """A fully wired agent using the offline provider and a recording handler."""
    built = Agent.build(
        settings,
        approval_handler=handler,
        provider=EchoProvider("echo"),
        database=database,
        include_optional_tools=True,
    )
    yield built
    built.state.transition(built.state.state)
