"""Process inspection and control.

Listing is read-only. Stopping a process is level 3 and always asks — killing
the wrong PID is the kind of mistake that loses someone's unsaved work, and
there is no rollback for it.
"""

from __future__ import annotations

import os
import signal
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from personalos.security.risk import RiskLevel
from personalos.tools.base import OperationSpec, PreparedCall, Tool, ToolContext, ToolResult

#: Killing any of these takes the desktop session or the machine with it.
PROTECTED_NAMES = {
    "init", "systemd", "launchd", "kernel_task", "wininit.exe", "csrss.exe",
    "services.exe", "lsass.exe", "smss.exe", "explorer.exe", "Finder", "loginwindow",
}


class ListProcessesInput(BaseModel):
    filter: str | None = Field(default=None, description="Substring to match in the command line.")
    limit: int = Field(default=50, ge=1, le=500)


class StopProcessInput(BaseModel):
    pid: int = Field(ge=1)
    force: bool = Field(default=False, description="Send SIGKILL instead of SIGTERM.")


def _iter_processes() -> list[dict[str, Any]]:
    """Enumerate processes, preferring psutil and falling back to /proc."""
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return _iter_proc_fs()
    processes = []
    for proc in psutil.process_iter(["pid", "name", "username", "cmdline", "cpu_percent"]):
        try:
            info = proc.info
            processes.append(
                {
                    "pid": info["pid"],
                    "name": info.get("name") or "",
                    "user": info.get("username") or "",
                    "command": " ".join(info.get("cmdline") or [])[:200],
                }
            )
        except Exception:  # noqa: BLE001 - processes disappear mid-iteration
            continue
    return processes


def _iter_proc_fs() -> list[dict[str, Any]]:
    """Linux fallback so the tool works without psutil installed."""
    processes: list[dict[str, Any]] = []
    proc_root = "/proc"
    if not os.path.isdir(proc_root):
        return processes
    for entry in os.listdir(proc_root):
        if not entry.isdigit():
            continue
        try:
            with open(f"{proc_root}/{entry}/comm", encoding="utf-8") as handle:
                name = handle.read().strip()
            with open(f"{proc_root}/{entry}/cmdline", "rb") as handle:
                command = handle.read().replace(b"\0", b" ").decode("utf-8", "replace").strip()
        except OSError:
            continue
        processes.append({"pid": int(entry), "name": name, "user": "", "command": command[:200]})
    return processes


class ProcessManagerTool(Tool):
    """Lists processes and, with approval, stops one."""

    name: ClassVar[str] = "processes"
    description: ClassVar[str] = "List running processes, and stop one when explicitly approved."

    operations: ClassVar[dict[str, OperationSpec]] = {
        "list": OperationSpec(
            name="list",
            description="List running processes.",
            schema=ListProcessesInput,
            risk=RiskLevel.READ_ONLY,
            permissions=["process.read"],
        ),
        "stop": OperationSpec(
            name="stop",
            description="Stop a running process by PID.",
            schema=StopProcessInput,
            risk=RiskLevel.HIGH,
            permissions=["process.control"],
            reversible=False,
            destructive=True,
        ),
    }

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        if call.operation != "stop":
            return call
        args = call.arguments
        assert isinstance(args, StopProcessInput)

        target = next((p for p in _iter_processes() if p["pid"] == args.pid), None)
        if target is None:
            call.blocked_reason = f"No process with PID {args.pid} is running."
            return call
        if target["name"] in PROTECTED_NAMES:
            call.blocked_reason = (
                f"PID {args.pid} is {target['name']}, a critical system process. "
                "The agent will not stop it."
            )
            return call
        if args.pid == os.getpid():
            call.blocked_reason = "That PID is the agent itself."
            return call

        call.affected_resources = [f"pid:{args.pid} ({target['name']})"]
        call.description = f"Stop {target['name']} (PID {args.pid})"
        call.concerns.append(
            "unsaved work in that process will be lost"
            + (" — SIGKILL gives it no chance to save" if args.force else "")
        )
        call.reversible = False
        return call

    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        if call.operation == "list":
            args = call.arguments
            assert isinstance(args, ListProcessesInput)
            processes = _iter_processes()
            if args.filter:
                needle = args.filter.lower()
                processes = [
                    p for p in processes
                    if needle in p["name"].lower() or needle in p["command"].lower()
                ]
            processes = processes[: args.limit]
            lines = [f"{p['pid']:>7}  {p['name'][:30]:<30} {p['command'][:80]}" for p in processes]
            return ToolResult.ok(
                f"{len(processes)} processes:\n" + "\n".join(lines),
                data={"count": len(processes), "processes": processes},
            )

        args = call.arguments
        assert isinstance(args, StopProcessInput)
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would stop PID {args.pid}.")
        try:
            os.kill(args.pid, signal.SIGKILL if args.force else signal.SIGTERM)
        except ProcessLookupError:
            return ToolResult.failed(f"PID {args.pid} was already gone.")
        except PermissionError:
            return ToolResult.failed(
                f"Not permitted to stop PID {args.pid}. It belongs to another user."
            )
        context.journal.record_irreversible(
            context.task_id, f"stopped process {args.pid}", step_id=context.step_id
        )
        return ToolResult.ok(
            f"Sent {'SIGKILL' if args.force else 'SIGTERM'} to PID {args.pid}.",
            data={"pid": args.pid, "force": args.force},
        )
