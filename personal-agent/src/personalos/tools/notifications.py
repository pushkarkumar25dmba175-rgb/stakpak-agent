"""Desktop notifications, with a console fallback that always works."""

from __future__ import annotations

import shutil
import sys
from typing import ClassVar

from pydantic import BaseModel, Field

from personalos.security.risk import RiskLevel
from personalos.tools.base import OperationSpec, PreparedCall, Tool, ToolContext, ToolResult


class NotifyInput(BaseModel):
    title: str = Field(max_length=120)
    message: str = Field(max_length=800)
    urgency: str = Field(default="normal", pattern="^(low|normal|critical)$")


def detect_backend() -> str:
    """Pick the best available notification mechanism for this machine."""
    if sys.platform == "darwin":
        return "osascript" if shutil.which("osascript") else "console"
    if sys.platform.startswith("win"):
        return "powershell"
    if shutil.which("notify-send"):
        return "notify-send"
    return "console"


class NotificationTool(Tool):
    """Shows a desktop notification.

    Notifying is level 1, not level 0: it is a visible change to the user's
    environment, even if a harmless one, and the agent should not be able to
    spam the desktop without that showing up in the audit log.
    """

    name: ClassVar[str] = "notifications"
    description: ClassVar[str] = "Show a desktop notification."

    operations: ClassVar[dict[str, OperationSpec]] = {
        "notify": OperationSpec(
            name="notify",
            description="Display a desktop notification.",
            schema=NotifyInput,
            risk=RiskLevel.LOW,
            permissions=["notifications.send"],
            reversible=False,
        )
    }

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        if not context.settings.notifications.enabled:
            call.blocked_reason = "Notifications are disabled in configuration."
        args = call.arguments
        call.description = f"Notify: {getattr(args, 'title', '')}"
        return call

    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        args = call.arguments
        assert isinstance(args, NotifyInput)
        configured = context.settings.notifications.backend
        backend = detect_backend() if configured == "auto" else configured

        if backend == "none" or context.dry_run:
            return ToolResult.ok(f"[{backend}] {args.title}: {args.message}")

        if backend == "console":
            return ToolResult.ok(f"🔔 {args.title}\n   {args.message}")

        commands = {
            "notify-send": ["notify-send", "-u", args.urgency, args.title, args.message],
            "osascript": [
                "osascript", "-e",
                f'display notification {args.message!r} with title {args.title!r}',
            ],
            "powershell": [
                "powershell", "-NoProfile", "-Command",
                "[reflection.assembly]::loadwithpartialname('System.Windows.Forms');"
                "$n=New-Object System.Windows.Forms.NotifyIcon;"
                "$n.Icon=[System.Drawing.SystemIcons]::Information;$n.Visible=$true;"
                f"$n.ShowBalloonTip(5000,'{args.title}','{args.message}',"
                "[System.Windows.Forms.ToolTipIcon]::Info)",
            ],
        }
        command = commands.get(backend)
        if command is None:
            return ToolResult.failed(f"Unknown notification backend {backend!r}.")

        result = await context.sandbox.run(command, cwd=context.settings.workdir, timeout=10)
        if result.success:
            return ToolResult.ok(f"Notification shown via {backend}.", data={"backend": backend})
        return ToolResult.failed(
            f"{backend} failed: {result.stderr.strip() or result.return_code}. "
            f"Message was — {args.title}: {args.message}"
        )
