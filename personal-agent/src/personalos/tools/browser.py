"""Browser automation — the abstraction, deliberately without a driver.

Phase 6 of the roadmap. This module exists now so the seam is designed before
anything is plugged into it, and so nothing else in the agent has to change
when a driver arrives.

Two properties matter and are encoded here rather than left to the eventual
implementation:

* **Isolation.** A driver runs out of process, in its own profile directory,
  and never reuses the user's logged-in browser session. An agent that inherits
  your cookies can act as you on every site you are signed in to.
* **Everything a page returns is untrusted.** Page text goes through
  :func:`wrap_untrusted` exactly like a document does. A web page that says
  "ignore your instructions" is a page that says that, not an instruction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from personalos.security.injection import wrap_untrusted
from personalos.security.risk import RiskLevel
from personalos.tools.base import OperationSpec, PreparedCall, Tool, ToolContext, ToolResult


@dataclass
class PageSnapshot:
    """What a driver returns after navigating."""

    url: str
    title: str
    text: str
    status_code: int | None = None


@runtime_checkable
class BrowserDriver(Protocol):
    """What a browser backend has to provide.

    A Playwright implementation would satisfy this in well under a hundred
    lines; it is left out of the default install because a headless browser is
    a large dependency and a large attack surface for a feature most people
    will not turn on.
    """

    name: str
    isolated_profile: bool
    """Must be True. A driver reusing the user's own profile is not acceptable."""

    async def open(self, url: str, *, timeout: float) -> PageSnapshot:
        ...

    async def close(self) -> None:
        ...


class OpenPageInput(BaseModel):
    url: str = Field(description="Absolute URL to open.")
    timeout_seconds: float = Field(default=30.0, gt=0, le=180)
    extract_text: bool = True


class BrowserTool(Tool):
    """Opens a page in an isolated browser session.

    Registered but inert until a driver is supplied, so `agent tools` shows the
    capability and explains exactly why it is unavailable rather than pretending
    it does not exist.
    """

    name: ClassVar[str] = "browser"
    description: ClassVar[str] = "Open a web page in an isolated browser session (optional module)."
    enabled_by_default: ClassVar[bool] = False

    operations: ClassVar[dict[str, OperationSpec]] = {
        "open": OperationSpec(
            name="open",
            description="Open a URL and return the page text.",
            schema=OpenPageInput,
            risk=RiskLevel.HIGH,
            permissions=["browser.navigate"],
            reversible=False,
        )
    }

    def __init__(self, driver: BrowserDriver | None = None) -> None:
        self.driver = driver

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        if self.driver is None:
            call.blocked_reason = (
                "No browser driver is configured. Browser automation is an optional "
                "module; implement BrowserDriver and pass it to BrowserTool to enable it."
            )
            return call
        if not self.driver.isolated_profile:
            call.blocked_reason = (
                "The configured browser driver does not use an isolated profile. "
                "Refusing to drive a browser that carries your logged-in sessions."
            )
            return call
        args = call.arguments
        assert isinstance(args, OpenPageInput)
        call.affected_resources = [args.url]
        call.description = f"Open {args.url} in an isolated browser"
        call.concerns.append("loads remote content and executes page scripts")
        call.reversible = False
        return call

    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        if self.driver is None:
            return ToolResult.failed("No browser driver is configured.")
        args = call.arguments
        assert isinstance(args, OpenPageInput)
        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would open {args.url}.")

        snapshot = await self.driver.open(args.url, timeout=args.timeout_seconds)
        untrusted = wrap_untrusted(snapshot.text, source=f"web:{snapshot.url}")
        return ToolResult.ok(
            f"{snapshot.title} — {snapshot.url}",
            data={
                "url": snapshot.url,
                "title": snapshot.title,
                "text": snapshot.text if args.extract_text else "",
                "untrusted_block": untrusted.for_prompt(),
                "injection_flags": untrusted.flags,
            },
        )
