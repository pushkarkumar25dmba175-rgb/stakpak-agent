"""The tool registry: what the agent can do, and what it is allowed to load."""

from __future__ import annotations

from typing import Any

from personalos.errors import ToolNotFoundError
from personalos.tools.base import Tool


class ToolRegistry:
    """Holds the live tool instances and answers "can you do X?".

    Tools are registered explicitly rather than auto-discovered: a plugin
    mechanism that scans the filesystem for anything importable is a nice
    feature and a terrible idea for an agent with shell access.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._disabled: set[str] = set()

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"A tool named {tool.name!r} is already registered.")
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def disable(self, name: str) -> None:
        """Keep the tool registered but refuse to hand it out."""
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        self._disabled.discard(name)

    def get(self, name: str) -> Tool:
        if name in self._disabled:
            raise ToolNotFoundError(
                f"The tool {name!r} is disabled.",
                remediation="Enable it in config.yaml, then run the task again.",
            )
        try:
            return self._tools[name]
        except KeyError:
            known = ", ".join(sorted(self.available()))
            raise ToolNotFoundError(
                f"No tool named {name!r} is registered.",
                remediation=f"Available tools: {known}.",
            ) from None

    def has(self, name: str) -> bool:
        return name in self._tools and name not in self._disabled

    def available(self) -> list[str]:
        return sorted(name for name in self._tools if name not in self._disabled)

    def all(self) -> list[Tool]:
        return [self._tools[name] for name in self.available()]

    def describe(self) -> list[dict[str, Any]]:
        """Full description of every enabled tool, for prompts and diagnostics."""
        return [tool.describe() for tool in self.all()]

    def catalogue(self) -> str:
        """A compact, token-cheap listing for the planner prompt."""
        lines = []
        for tool in self.all():
            for operation, spec in tool.operations.items():
                lines.append(
                    f"{tool.name}.{operation} (risk {int(spec.risk)}): {spec.description}"
                )
        return "\n".join(lines)
