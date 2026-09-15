"""The tool contract.

Every capability the agent has is a :class:`Tool` exposing one or more
*operations*. An operation declares its own input schema and its own base risk
level, because "read a file" and "delete a file" are the same tool but very
different propositions.

The execution path is always the same three phases, and the middle one is not
optional:

    prepare()   parse and validate inputs, resolve paths, classify the *actual*
                risk of this specific call, list what it would touch
    ← policy    the policy engine decides, possibly asking the user
    execute()   do the thing, recording an inverse operation as it goes

A tool that skips ``prepare`` cannot be approved, which is what keeps the
permission model from being something a tool author can forget.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from personalos.audit.audit_log import AuditLog
from personalos.audit.rollback_journal import RollbackJournal
from personalos.errors import ToolValidationError
from personalos.security.risk import RiskLevel
from personalos.security.sandbox import Sandbox
from personalos.settings import Settings
from personalos.utils.paths import PathResolver


@dataclass
class ToolContext:
    """Everything a tool is allowed to reach outside itself.

    Handing tools a context instead of global state is what makes them
    testable and what guarantees they cannot quietly widen their own reach:
    the ``paths`` resolver is the only way to turn a string into a filesystem
    location, and it enforces the workspace allow-list.
    """

    settings: Settings
    paths: PathResolver
    journal: RollbackJournal
    audit: AuditLog
    sandbox: Sandbox
    task_id: str
    step_id: str | None = None
    project: str | None = None
    dry_run: bool = False

    @property
    def workdir(self) -> Path:
        return self.settings.workdir

    def child(self, step_id: str) -> ToolContext:
        """A copy of this context bound to a specific plan step."""
        return ToolContext(
            settings=self.settings,
            paths=self.paths,
            journal=self.journal,
            audit=self.audit,
            sandbox=self.sandbox,
            task_id=self.task_id,
            step_id=step_id,
            project=self.project,
            dry_run=self.dry_run,
        )


@dataclass
class OperationSpec:
    """The declared contract of one operation."""

    name: str
    description: str
    schema: type[BaseModel]
    risk: RiskLevel
    permissions: list[str] = field(default_factory=list)
    """Capability names, e.g. ``filesystem.write`` — shown in `agent permissions`."""
    reversible: bool = True
    destructive: bool = False
    """True when the operation removes or overwrites user data."""

    def json_schema(self) -> dict[str, Any]:
        return self.schema.model_json_schema()


@dataclass
class PreparedCall:
    """A validated, risk-classified call that has not run yet."""

    tool: str
    operation: str
    arguments: BaseModel
    risk_level: RiskLevel
    description: str
    affected_resources: list[str] = field(default_factory=list)
    concerns: list[str] = field(default_factory=list)
    reversible: bool = True
    rollback_hint: str | None = None
    blocked_reason: str | None = None
    """Set when the call must not run even with approval."""

    @property
    def blocked(self) -> bool:
        return self.blocked_reason is not None

    def as_parameters(self) -> dict[str, Any]:
        return self.arguments.model_dump(mode="json")


@dataclass
class ToolResult:
    """What a tool did, in a form that can be logged, verified and shown."""

    success: bool
    output: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    rollback_ids: list[int] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    """Paths the operation created or modified — what verification checks."""
    duration_ms: float = 0.0

    @classmethod
    def ok(cls, output: str, **kwargs: Any) -> ToolResult:
        return cls(success=True, output=output, **kwargs)

    @classmethod
    def failed(cls, error: str, **kwargs: Any) -> ToolResult:
        return cls(success=False, error=error, **kwargs)


class Tool(ABC):
    """Base class for every capability.

    Subclasses declare ``name``, ``description`` and ``operations``, then
    implement :meth:`_run`. Overriding :meth:`prepare_operation` lets a tool
    escalate risk based on the actual arguments — the shell tool does this, and
    so does any filesystem operation that turns out to overwrite something.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    operations: ClassVar[dict[str, OperationSpec]]
    enabled_by_default: ClassVar[bool] = True

    def spec(self, operation: str) -> OperationSpec:
        try:
            return self.operations[operation]
        except KeyError:
            known = ", ".join(sorted(self.operations))
            raise ToolValidationError(
                f"{self.name} has no operation {operation!r}.",
                remediation=f"Available operations: {known}.",
            ) from None

    def prepare(
        self, operation: str, arguments: dict[str, Any], context: ToolContext
    ) -> PreparedCall:
        """Validate inputs and classify this specific call.

        Raises:
            ToolValidationError: when the arguments do not match the schema.
        """
        spec = self.spec(operation)
        try:
            parsed = spec.schema.model_validate(arguments)
        except ValidationError as exc:
            details = "; ".join(
                f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
            )
            raise ToolValidationError(
                f"Invalid arguments for {self.name}.{operation}: {details}",
                remediation=f"Expected fields: {', '.join(spec.schema.model_fields)}.",
            ) from exc

        call = PreparedCall(
            tool=self.name,
            operation=operation,
            arguments=parsed,
            risk_level=spec.risk,
            description=spec.description,
            reversible=spec.reversible,
        )
        return self.prepare_operation(spec, call, context)

    def prepare_operation(
        self, spec: OperationSpec, call: PreparedCall, context: ToolContext
    ) -> PreparedCall:
        """Hook for per-call risk escalation. Default: no change."""
        return call

    async def run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        """Execute an already-approved call, timing it."""
        if call.blocked:
            return ToolResult.failed(call.blocked_reason or "Blocked.")
        started = time.perf_counter()
        try:
            result = await self._run(call, context)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a failure
            result = ToolResult.failed(f"{type(exc).__name__}: {exc}")
        result.duration_ms = (time.perf_counter() - started) * 1000
        return result

    @abstractmethod
    async def _run(self, call: PreparedCall, context: ToolContext) -> ToolResult:
        """Perform the operation. Implemented by each tool."""

    def describe(self) -> dict[str, Any]:
        """Machine-readable description, used by the LLM and by `agent doctor`."""
        return {
            "name": self.name,
            "description": self.description,
            "operations": {
                name: {
                    "description": spec.description,
                    "risk_level": int(spec.risk),
                    "risk_label": spec.risk.label,
                    "permissions": spec.permissions,
                    "reversible": spec.reversible,
                    "destructive": spec.destructive,
                    "input_schema": spec.json_schema(),
                }
                for name, spec in self.operations.items()
            },
        }
