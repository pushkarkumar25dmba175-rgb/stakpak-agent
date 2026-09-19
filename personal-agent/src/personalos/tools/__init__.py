"""Tools: everything the agent can actually do."""

from personalos.tools.base import (
    OperationSpec,
    PreparedCall,
    Tool,
    ToolContext,
    ToolResult,
)
from personalos.tools.browser import BrowserDriver, BrowserTool, PageSnapshot
from personalos.tools.clipboard import ClipboardTool
from personalos.tools.documents import DocumentTool
from personalos.tools.filesystem import FilesystemTool
from personalos.tools.http_tool import HttpTool
from personalos.tools.notifications import NotificationTool, detect_backend
from personalos.tools.process_manager import ProcessManagerTool
from personalos.tools.python_tool import PythonTool
from personalos.tools.registry import ToolRegistry
from personalos.tools.shell import ShellTool


def build_default_registry(*, include_optional: bool = False) -> ToolRegistry:
    """Construct the registry the agent runs with.

    Optional tools (clipboard, browser) stay out unless asked for, and even
    when registered they refuse to act until their feature flag is on. Two
    layers, because "off by default" should not depend on a single check.
    """
    registry = ToolRegistry()
    for tool in (
        FilesystemTool(),
        DocumentTool(),
        ShellTool(),
        PythonTool(),
        NotificationTool(),
        ProcessManagerTool(),
        HttpTool(),
    ):
        registry.register(tool)
    if include_optional:
        registry.register(ClipboardTool())
        registry.register(BrowserTool())
    return registry


__all__ = [
    "BrowserDriver",
    "BrowserTool",
    "ClipboardTool",
    "DocumentTool",
    "FilesystemTool",
    "HttpTool",
    "NotificationTool",
    "OperationSpec",
    "PageSnapshot",
    "PreparedCall",
    "ProcessManagerTool",
    "PythonTool",
    "ShellTool",
    "Tool",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "build_default_registry",
    "detect_backend",
]
