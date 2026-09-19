"""Projects: scoping the agent to one body of work.

A project is a directory under ``~/.personalos/projects/<name>/`` holding a
``project.json``. Switching to a project changes four things:

* which directories the agent may touch (its own roots are added to the
  allow-list for the session),
* which memory namespace preferences and facts are read from and written to,
* which skills are offered first,
* which standing policy rules apply.

That last point is why project scoping is a security feature and not just
organisation: "don't ask before moving files" can be granted for one project
without applying anywhere else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from personalos.errors import ConfigurationError
from personalos.utils.paths import expand_path
from personalos.utils.timeutil import isoformat


class ProjectDefinition(BaseModel):
    """The contents of ``project.json``."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    description: str = ""
    directories: list[str] = Field(default_factory=list)
    preferred_tools: list[str] = Field(default_factory=list)
    memory_namespace: str | None = None
    skills: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    """Human-readable notes about how the agent should behave here."""
    created_at: str = Field(default_factory=isoformat)

    @property
    def namespace(self) -> str:
        return self.memory_namespace or self.name

    def resolved_directories(self) -> list[Path]:
        return [expand_path(directory) for directory in self.directories]


@dataclass
class ProjectContext:
    """A loaded project, ready to be applied to a session."""

    definition: ProjectDefinition
    path: Path
    extra_roots: list[Path] = field(default_factory=list)

    def render(self) -> str:
        """A short block describing the project, for the planner prompt."""
        lines = [f"Project: {self.definition.name}"]
        if self.definition.description:
            lines.append(self.definition.description)
        if self.definition.directories:
            lines.append("Directories: " + ", ".join(self.definition.directories))
        for rule in self.definition.rules:
            lines.append(f"Rule: {rule}")
        return "\n".join(lines)


class ProjectManager:
    """Creates, lists, loads and switches projects."""

    def __init__(self, projects_dir: Path) -> None:
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.projects_dir / name / "project.json"

    def create(
        self,
        name: str,
        *,
        description: str = "",
        directories: list[str] | None = None,
    ) -> ProjectContext:
        """Create a project, refusing to overwrite an existing one."""
        target = self.path_for(name)
        if target.exists():
            raise ConfigurationError(
                f"The project {name!r} already exists at {target}.",
                remediation="Edit it directly, or pick another name.",
            )
        try:
            definition = ProjectDefinition(
                name=name, description=description, directories=directories or []
            )
        except ValidationError as exc:
            raise ConfigurationError(
                f"{name!r} is not a valid project name: lowercase letters, digits, "
                "hyphens and underscores only.",
            ) from exc
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(definition.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
        return ProjectContext(definition=definition, path=target)

    def load(self, name: str) -> ProjectContext:
        """Load a project definition.

        Raises:
            ConfigurationError: when it does not exist or is malformed.
        """
        target = self.path_for(name)
        if not target.exists():
            known = ", ".join(self.list_names()) or "none yet"
            raise ConfigurationError(
                f"There is no project called {name!r}.",
                remediation=f"Existing projects: {known}. Create one with `agent project create {name}`.",
            )
        try:
            payload: dict[str, Any] = json.loads(target.read_text(encoding="utf-8"))
            definition = ProjectDefinition.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ConfigurationError(
                f"{target} is not a valid project definition: {exc}",
                remediation="Fix the JSON, or delete the directory and recreate the project.",
            ) from exc
        return ProjectContext(
            definition=definition,
            path=target,
            extra_roots=definition.resolved_directories(),
        )

    def list_names(self) -> list[str]:
        return sorted(
            entry.name
            for entry in self.projects_dir.iterdir()
            if entry.is_dir() and (entry / "project.json").exists()
        ) if self.projects_dir.is_dir() else []

    def list_projects(self) -> list[ProjectContext]:
        projects = []
        for name in self.list_names():
            try:
                projects.append(self.load(name))
            except ConfigurationError:
                continue
        return projects
