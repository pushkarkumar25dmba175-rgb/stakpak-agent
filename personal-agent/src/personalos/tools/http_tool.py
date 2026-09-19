"""HTTP tool: talking to APIs.

Anything that leaves this machine is a level 3 or 4 action:

* ``GET`` is level 3. It does not change the remote side, but it discloses that
  the request was made and can leak whatever is in the URL.
* every other method is level 4, because the whole point of a ``POST`` is to
  change something the agent cannot undo.

Credentials are passed by *reference* (``env:GITHUB_TOKEN``) and resolved
immediately before the request, so the token never reaches memory, the audit
log or the model.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from personalos.errors import ConfigurationError
from personalos.security.injection import wrap_untrusted
from personalos.security.risk import RiskLevel
from personalos.security.secrets import SecretResolver
from personalos.tools.base import OperationSpec, PreparedCall, Tool, ToolContext, ToolResult
from personalos.utils.textutil import truncate

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class RequestInput(BaseModel):
    url: str = Field(description="Absolute https:// URL.")
    method: str = Field(default="GET", pattern="^(GET|HEAD|OPTIONS|POST|PUT|PATCH|DELETE)$")
    headers: dict[str, str] = Field(default_factory=dict)
    auth_header: str | None = Field(
        default=None,
        description="Header name for a credential, e.g. 'Authorization'.",
    )
    auth_secret_ref: str | None = Field(
        default=None,
        description="Secret reference such as 'env:GITHUB_TOKEN'. Never the value itself.",
    )
    auth_prefix: str = Field(default="Bearer ")
    json_body: dict[str, Any] | None = None
    text_body: str | None = None
    timeout_seconds: float = Field(default=30.0, gt=0, le=300)
    max_response_characters: int = Field(default=20000, ge=100, le=200000)


class HttpTool(Tool):
    """Makes an HTTP request to an external service."""

    name: ClassVar[str] = "http"
    description: ClassVar[str] = "Call an HTTP API. Always requires confirmation."

    operations: ClassVar[dict[str, OperationSpec]] = {
        "request": OperationSpec(
            name="request",
            description="Send an HTTP request to an external service.",
            schema=RequestInput,
            risk=RiskLevel.EXTERNAL,
            permissions=["network.request"],
            reversible=False,
        )
    }

    def __init__(self, resolver: SecretResolver | None = None) -> None:
        self.resolver = resolver or SecretResolver()

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        args = call.arguments
        assert isinstance(args, RequestInput)

        if not args.url.lower().startswith(("https://", "http://")):
            call.blocked_reason = "The URL must be an absolute http(s) URL."
            return call
        if args.url.lower().startswith("http://"):
            call.concerns.append("plain HTTP: the request is not encrypted")

        method = args.method.upper()
        call.risk_level = RiskLevel.HIGH if method in SAFE_METHODS else RiskLevel.EXTERNAL
        call.reversible = False
        call.rollback_hint = "Network requests cannot be undone from here."
        call.affected_resources = [args.url]
        call.description = f"{method} {truncate(args.url, 120)}"
        if method not in SAFE_METHODS:
            call.concerns.append(f"{method} changes state on the remote service")
        if args.auth_secret_ref:
            call.concerns.append(f"authenticates using {args.auth_secret_ref}")
        for name in args.headers:
            if name.lower() in {"authorization", "cookie", "x-api-key"}:
                call.blocked_reason = (
                    f"Pass credentials via auth_secret_ref, not a raw {name} header, "
                    "so the value never enters the agent's memory or logs."
                )
        return call

    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        args = call.arguments
        assert isinstance(args, RequestInput)
        try:
            import httpx
        except ImportError:
            return ToolResult.failed("The `httpx` package is required for HTTP requests.")

        if context.dry_run:
            return ToolResult.ok(f"[dry run] Would send {args.method} {args.url}.")

        headers = dict(args.headers)
        if args.auth_secret_ref and args.auth_header:
            try:
                secret = self.resolver.resolve(args.auth_secret_ref)
            except ConfigurationError as exc:
                return ToolResult.failed(str(exc))
            headers[args.auth_header] = f"{args.auth_prefix}{secret}"

        content = args.text_body.encode() if args.text_body is not None else None
        try:
            async with httpx.AsyncClient(timeout=args.timeout_seconds, follow_redirects=True) as client:
                response = await client.request(
                    args.method.upper(),
                    args.url,
                    headers=headers,
                    json=args.json_body,
                    content=content,
                )
        except Exception as exc:  # noqa: BLE001 - network errors are expected
            return ToolResult.failed(f"{type(exc).__name__}: {exc}")

        body = response.text[: args.max_response_characters]
        untrusted = wrap_untrusted(body, source=f"http:{args.url}")
        context.journal.record_irreversible(
            context.task_id,
            f"{args.method.upper()} {args.url} → {response.status_code}",
            step_id=context.step_id,
        )
        payload: dict[str, Any] = {
            "url": str(response.url),
            "status_code": response.status_code,
            "body": body,
            "untrusted_block": untrusted.for_prompt(),
            "injection_flags": untrusted.flags,
        }
        try:
            payload["json"] = response.json()
        except (json.JSONDecodeError, ValueError):
            pass

        summary = f"{response.status_code} {response.reason_phrase} — {truncate(body, 800)}"
        if response.is_success:
            return ToolResult.ok(summary, data=payload)
        return ToolResult(
            success=False,
            output=summary,
            data=payload,
            error=f"The service returned {response.status_code}.",
        )
