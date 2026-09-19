"""Clipboard access — opt-in, because the clipboard routinely holds secrets.

Both operations are disabled unless ``observation.watch_clipboard_enabled`` is
turned on, and reads are redacted before the content goes anywhere. A password
manager copy landing in the audit log would be exactly the kind of quiet leak
this project is meant to avoid.
"""

from __future__ import annotations

import shutil
import sys
from typing import ClassVar

from pydantic import BaseModel, Field

from personalos.security.risk import RiskLevel
from personalos.security.secrets import redact
from personalos.tools.base import OperationSpec, PreparedCall, Tool, ToolContext, ToolResult


class ReadClipboardInput(BaseModel):
    max_characters: int = Field(default=5000, ge=1, le=100000)


class WriteClipboardInput(BaseModel):
    content: str = Field(max_length=100000)


def _clipboard_commands() -> tuple[list[str] | None, list[str] | None]:
    """Return ``(read_command, write_command)`` for this platform, or ``(None, None)``."""
    if sys.platform == "darwin":
        return ["pbpaste"], ["pbcopy"]
    if sys.platform.startswith("win"):
        return (
            ["powershell", "-NoProfile", "-Command", "Get-Clipboard"],
            ["powershell", "-NoProfile", "-Command", "Set-Clipboard -Value ($input | Out-String)"],
        )
    if shutil.which("wl-paste") and shutil.which("wl-copy"):
        return ["wl-paste", "--no-newline"], ["wl-copy"]
    if shutil.which("xclip"):
        return ["xclip", "-selection", "clipboard", "-o"], ["xclip", "-selection", "clipboard"]
    if shutil.which("xsel"):
        return ["xsel", "--clipboard", "--output"], ["xsel", "--clipboard", "--input"]
    return None, None


class ClipboardTool(Tool):
    """Reads and writes the system clipboard when explicitly enabled."""

    name: ClassVar[str] = "clipboard"
    description: ClassVar[str] = "Read or replace the system clipboard (opt-in)."
    enabled_by_default: ClassVar[bool] = False

    operations: ClassVar[dict[str, OperationSpec]] = {
        "read": OperationSpec(
            name="read",
            description="Read the current clipboard contents.",
            schema=ReadClipboardInput,
            risk=RiskLevel.READ_ONLY,
            permissions=["clipboard.read"],
        ),
        "write": OperationSpec(
            name="write",
            description="Replace the clipboard contents.",
            schema=WriteClipboardInput,
            risk=RiskLevel.MODERATE,
            permissions=["clipboard.write"],
            reversible=False,
        ),
    }

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        observation = context.settings.observation
        if not observation.watch_clipboard_enabled:
            call.blocked_reason = (
                "Clipboard access is off. Enable observation.watch_clipboard_enabled "
                "in config.yaml if you want the agent to use it."
            )
        elif observation.paused:
            call.blocked_reason = "Observation is paused."
        if call.operation == "read":
            call.concerns.append("the clipboard often holds passwords; content is redacted")
        else:
            call.concerns.append("replaces whatever you currently have copied")
            call.rollback_hint = "The previous clipboard contents cannot be restored."
        return call

    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        read_command, write_command = _clipboard_commands()
        if read_command is None or write_command is None:
            return ToolResult.failed(
                "No clipboard utility is available. "
                "On Linux install `wl-clipboard` or `xclip`."
            )
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would {call.operation} the clipboard.")

        if call.operation == "read":
            args = call.arguments
            assert isinstance(args, ReadClipboardInput)
            result = await context.sandbox.run(read_command, cwd=context.settings.workdir, timeout=10)
            if not result.success:
                return ToolResult.failed(f"Could not read the clipboard: {result.stderr.strip()}")
            content = redact(result.stdout)[: args.max_characters]
            return ToolResult.ok(
                content,
                data={"characters": len(content), "content": content, "redacted": True},
            )

        args = call.arguments
        assert isinstance(args, WriteClipboardInput)
        result = await context.sandbox.run(
            write_command, cwd=context.settings.workdir, timeout=10, stdin=args.content
        )
        if not result.success:
            return ToolResult.failed(f"Could not write to the clipboard: {result.stderr.strip()}")
        return ToolResult.ok(f"Copied {len(args.content)} characters to the clipboard.")
