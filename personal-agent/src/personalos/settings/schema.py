"""Typed configuration schema for PersonalOS.

Configuration is layered, lowest priority first:

1. the defaults in this module,
2. ``$PERSONALOS_HOME/config.yaml`` (created on first run),
3. environment variables (``PERSONALOS_*``, and the provider API keys),
4. explicit command-line overrides.

Secrets are *never* part of this schema. The LLM sections hold the name of an
environment variable to read, not the key itself, so a config file can be
committed or shared without leaking anything.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from personalos.utils.paths import expand_path


class LLMProviderName(StrEnum):
    """Providers the agent knows how to talk to."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    ECHO = "echo"
    """A deterministic offline provider used by tests and `agent doctor`."""


class ApprovalMode(StrEnum):
    """How aggressively the agent asks before acting."""

    STRICT = "strict"
    """Confirm everything above read-only, ignoring standing rules."""
    NORMAL = "normal"
    """Follow the documented per-level defaults."""
    RELAXED = "relaxed"
    """Auto-approve level 2 inside the active project. Never affects 3 and 4."""


class LLMSettings(BaseModel):
    """Which model to use and how much of it to use."""

    provider: LLMProviderName = LLMProviderName.ANTHROPIC
    model: str = "claude-opus-5"
    api_key_env: str = "ANTHROPIC_API_KEY"
    """Name of the environment variable holding the key — not the key."""
    base_url: str | None = None
    max_output_tokens: int = Field(default=2048, ge=64, le=32768)
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    timeout_seconds: float = Field(default=120.0, gt=0)
    context_budget_tokens: int = Field(default=12000, ge=1000)
    """Hard ceiling on everything sent to the model, memory included."""
    memory_budget_tokens: int = Field(default=3000, ge=0)
    """Slice of the context budget that retrieved memory may occupy."""


class WorkspaceSettings(BaseModel):
    """The directories the agent is allowed to see and touch."""

    allowed_roots: list[str] = Field(default_factory=lambda: ["~/PersonalOS"])
    denied_paths: list[str] = Field(
        default_factory=lambda: [
            "~/.ssh",
            "~/.gnupg",
            "~/.aws",
            "~/.config/gcloud",
            "~/.kube",
            "~/.docker/config.json",
            "~/.netrc",
            "~/.password-store",
        ]
    )
    follow_symlinks: bool = False
    trash_dir: str = "agent_trash"
    """Relative to the PersonalOS home: where "deleted" files actually go."""
    max_read_bytes: int = Field(default=2_000_000, ge=1024)
    max_write_bytes: int = Field(default=10_000_000, ge=1024)


#: Read-mostly commands that do not escalate risk on their own. Everything
#: outside this list still runs — it just has to be approved.
_POSIX_ALLOWLIST = [
    "ls", "cat", "head", "tail", "wc", "grep", "find", "file", "stat",
    "du", "df", "echo", "pwd", "date", "which", "python3", "python",
    "git", "rg", "sort", "uniq", "diff", "tree",
]

_WINDOWS_ALLOWLIST = [
    "dir", "type", "findstr", "where", "echo", "cd", "tree", "fc",
    "more", "sort", "date", "time", "ver", "whoami", "python", "py",
    "git", "rg", "Get-ChildItem", "Get-Content", "Select-String",
]


def default_shell_allowlist() -> list[str]:
    """The shell allowlist appropriate to this machine.

    A POSIX list on Windows would mean every ordinary command — `dir`, `type`,
    `findstr` — came back flagged as unrecognised, which trains people to click
    through the warning that is supposed to mean something.
    """
    if sys.platform.startswith("win"):
        return list(_WINDOWS_ALLOWLIST)
    return list(_POSIX_ALLOWLIST)


class SecuritySettings(BaseModel):
    """Approval policy, command screening and destructive-action handling."""

    approval_mode: ApprovalMode = ApprovalMode.NORMAL
    auto_approve_max_level: int = Field(default=1, ge=0, le=2)
    """Never raise this above 2: levels 3 and 4 always ask, by design."""
    shell_enabled: bool = True
    shell_allowlist: list[str] = Field(default_factory=lambda: default_shell_allowlist())
    shell_denied_patterns: list[str] = Field(
        default_factory=lambda: [
            r"\brm\s+-[a-zA-Z]*[rf]", r"\bmkfs\b", r"\bdd\s+if=", r":\(\)\s*\{",
            r"\bshutdown\b", r"\breboot\b", r"\bdiskpart\b", r"\bformat\s+[A-Za-z]:",
            r"\bchown\s+-R\s+/", r"\bchmod\s+-R\s+777\s+/",
            r"curl[^|]*\|\s*(sudo\s+)?(ba)?sh", r"wget[^|]*\|\s*(sudo\s+)?(ba)?sh",
        ]
    )
    shell_timeout_seconds: float = Field(default=60.0, gt=0)
    python_timeout_seconds: float = Field(default=60.0, gt=0)
    prefer_trash_over_delete: bool = True
    max_retries: int = Field(default=3, ge=0, le=10)
    redact_logs: bool = True


class MemorySettings(BaseModel):
    """Retention, decay and retrieval behaviour."""

    episodic_retention_days: int = Field(default=365, ge=1)
    inferred_confidence_floor: float = Field(default=0.25, ge=0.0, le=1.0)
    """Inferred preferences below this confidence stop being injected."""
    confidence_half_life_days: float = Field(default=45.0, gt=0)
    """Half-life of the decay applied to *inferred* items. Explicit ones never decay."""
    retrieval_top_k: int = Field(default=8, ge=1, le=50)
    vector_backend: str = "none"
    """One of: none, chroma, qdrant, faiss. Only `none` ships wired up today."""


class LearningSettings(BaseModel):
    """Thresholds for workflow detection and proactive suggestions."""

    enabled: bool = True
    pattern_similarity_threshold: float = Field(default=0.78, ge=0.0, le=1.0)
    min_occurrences_for_suggestion: int = Field(default=3, ge=2)
    suggestion_cooldown_hours: float = Field(default=12.0, ge=0)
    max_suggestions_per_session: int = Field(default=3, ge=0)
    auto_create_skills: bool = False
    """Must stay False: a detected pattern is a suggestion, never an automation."""

    @field_validator("auto_create_skills")
    @classmethod
    def _never_auto_create(cls, value: bool) -> bool:
        if value:
            raise ValueError(
                "auto_create_skills cannot be enabled: skills require explicit approval."
            )
        return value


class ObservationSettings(BaseModel):
    """Optional, off-by-default monitoring of the user's machine."""

    watch_folders_enabled: bool = False
    watched_folders: list[str] = Field(default_factory=list)
    watch_clipboard_enabled: bool = False
    track_app_usage: bool = False
    paused: bool = False
    ignore_patterns: list[str] = Field(
        default_factory=lambda: ["*.tmp", "*.part", "*.crdownload", ".git/*", "~$*"]
    )


class SchedulerSettings(BaseModel):
    """Local job scheduling."""

    enabled: bool = True
    timezone: str = "UTC"
    max_concurrent_jobs: int = Field(default=2, ge=1, le=16)
    skip_jobs_above_level: int = Field(default=2, ge=0, le=4)
    """Scheduled jobs may not execute steps above this level unattended."""


class DashboardSettings(BaseModel):
    """The optional local web UI."""

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = Field(default=8765, ge=1024, le=65535)
    read_only: bool = True

    @field_validator("host")
    @classmethod
    def _localhost_only(cls, value: str) -> str:
        if value not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError(
                "The dashboard may only bind to localhost. "
                "Use an SSH tunnel if you need remote access."
            )
        return value


class NotificationSettings(BaseModel):
    """Desktop notifications."""

    enabled: bool = True
    backend: str = "auto"
    """auto | notify-send | osascript | powershell | console | none"""


class Settings(BaseModel):
    """The complete, validated agent configuration."""

    home: Path = Field(default_factory=lambda: expand_path("~/.personalos"))
    active_project: str | None = None
    llm: LLMSettings = Field(default_factory=LLMSettings)
    workspace: WorkspaceSettings = Field(default_factory=WorkspaceSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    memory: MemorySettings = Field(default_factory=MemorySettings)
    learning: LearningSettings = Field(default_factory=LearningSettings)
    observation: ObservationSettings = Field(default_factory=ObservationSettings)
    scheduler: SchedulerSettings = Field(default_factory=SchedulerSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)
    notifications: NotificationSettings = Field(default_factory=NotificationSettings)

    @field_validator("home", mode="before")
    @classmethod
    def _expand_home(cls, value: object) -> object:
        if isinstance(value, str):
            return expand_path(value)
        return value

    # ---- derived locations -------------------------------------------------
    @property
    def config_file(self) -> Path:
        return self.home / "config.yaml"

    @property
    def database_path(self) -> Path:
        return self.home / "personalos.db"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def skills_dir(self) -> Path:
        return self.home / "skills"

    @property
    def projects_dir(self) -> Path:
        return self.home / "projects"

    @property
    def workdir(self) -> Path:
        """Scratch directory handed to the Python tool and to script steps."""
        return self.home / "work"

    @property
    def trash_dir(self) -> Path:
        return self.home / self.workspace.trash_dir

    @property
    def state_file(self) -> Path:
        return self.home / "state.json"

    def directories(self) -> list[Path]:
        """Every directory that must exist before the agent can run."""
        return [
            self.home,
            self.logs_dir,
            self.skills_dir,
            self.skills_dir / "generated",
            self.projects_dir,
            self.workdir,
            self.trash_dir,
        ]

    def ensure_directories(self) -> None:
        """Create the agent's own directories. Idempotent."""
        for directory in self.directories():
            directory.mkdir(parents=True, exist_ok=True)
