"""The skill format.

A skill is a YAML file describing a plan with named inputs. Declarative on
purpose: a skill you can read is a skill you can audit, diff and edit, and one
the agent cannot use to smuggle in arbitrary code. Generating a skill from an
observed workflow then amounts to writing out a document, which is a much
smaller thing to trust than generating Python.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from personalos.agent.plan import Plan, PlanStep, VerificationSpec
from personalos.errors import SkillError
from personalos.security.risk import RiskLevel
from personalos.utils.timeutil import isoformat


class SkillInput(BaseModel):
    """One parameter a skill accepts."""

    type: Literal["string", "integer", "number", "boolean"] = "string"
    description: str = ""
    required: bool = False
    default: Any = None

    def coerce(self, name: str, value: Any) -> Any:
        """Convert a CLI string into the declared type."""
        if value is None:
            return self.default
        try:
            if self.type == "integer":
                return int(value)
            if self.type == "number":
                return float(value)
            if self.type == "boolean":
                if isinstance(value, bool):
                    return value
                return str(value).strip().lower() in {"1", "true", "yes", "on"}
            return str(value)
        except (TypeError, ValueError) as exc:
            raise SkillError(
                f"Input {name!r} should be {self.type}, got {value!r}.",
            ) from exc


class SkillStep(BaseModel):
    """One step of a skill, mirroring :class:`PlanStep`."""

    id: str
    description: str
    tool: str
    operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    depends_on: list[str] = Field(default_factory=list)
    expected_outcome: str | None = None
    verification: VerificationSpec | None = None
    optional: bool = False


class SkillDefinition(BaseModel):
    """A complete, runnable skill."""

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{1,63}$")
    description: str = ""
    version: int = Field(default=1, ge=1)
    created_by: Literal["user", "agent"] = "user"
    created_at: str = Field(default_factory=isoformat)
    inputs: dict[str, SkillInput] = Field(default_factory=dict)
    permissions: list[str] = Field(default_factory=list)
    max_risk_level: int = Field(default=2, ge=0, le=4)
    """Declared ceiling. The policy engine still evaluates every step."""
    steps: list[SkillStep] = Field(default_factory=list)
    derived_from_workflow: str | None = None
    tags: list[str] = Field(default_factory=list)

    @field_validator("steps")
    @classmethod
    def _needs_steps(cls, value: list[SkillStep]) -> list[SkillStep]:
        if not value:
            raise ValueError("a skill needs at least one step")
        ids = [step.id for step in value]
        if len(ids) != len(set(ids)):
            raise ValueError("step ids must be unique")
        return value

    def resolve_inputs(self, provided: dict[str, Any] | None = None) -> dict[str, Any]:
        """Validate and fill in the skill's inputs.

        Raises:
            SkillError: when a required input is missing.
        """
        provided = provided or {}
        unknown = set(provided) - set(self.inputs)
        if unknown:
            raise SkillError(
                f"{self.name} does not take {', '.join(sorted(unknown))}.",
                remediation=f"Accepted inputs: {', '.join(self.inputs) or 'none'}.",
            )
        resolved: dict[str, Any] = {}
        for name, spec in self.inputs.items():
            value = spec.coerce(name, provided.get(name))
            if value is None and spec.required:
                raise SkillError(
                    f"{self.name} needs the input {name!r}: {spec.description or spec.type}.",
                    remediation=f"Run it as `agent run {self.name} --input {name}=...`.",
                )
            resolved[name] = value
        return resolved

    def to_plan(self, inputs: dict[str, Any] | None = None) -> Plan:
        """Instantiate the skill as a plan, with inputs bound."""
        resolved = self.resolve_inputs(inputs)
        return Plan(
            goal=self.description or f"Run the {self.name} skill",
            source="skill",
            inputs=resolved,
            notes=f"From skill {self.name} v{self.version}.",
            steps=[
                PlanStep(
                    id=step.id,
                    description=step.description,
                    tool=step.tool,
                    operation=step.operation,
                    arguments=step.arguments,
                    depends_on=step.depends_on,
                    expected_outcome=step.expected_outcome,
                    verification=step.verification,
                    optional=step.optional,
                )
                for step in self.steps
            ],
        )

    def declared_risk(self) -> RiskLevel:
        return RiskLevel.parse(self.max_risk_level)

    def to_yaml(self) -> str:
        payload = self.model_dump(mode="json", exclude_none=True)
        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)


def load_skill_file(path: Path) -> SkillDefinition:
    """Parse a skill YAML file.

    Raises:
        SkillError: when the file is unreadable or does not match the schema.
    """
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SkillError(f"Could not read the skill at {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SkillError(f"{path} does not contain a skill definition.")
    try:
        return SkillDefinition.model_validate(payload)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        raise SkillError(
            f"{path.name} is not a valid skill: {details}",
            remediation="Compare it against `agent skills inspect <name>` for a working example.",
        ) from exc
