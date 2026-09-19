"""Turning an observed workflow into a skill definition.

The generator produces a :class:`SkillDefinition` and hands it back. It does
not write it to disk, does not register it and does not run it — that is the
skill manager's job, reached only from an explicit user approval. Keeping the
generation and the persistence apart is what makes "the agent must not silently
turn observed behaviour into automation" a structural property rather than a
promise.

Paths that recur across observed runs are lifted into skill inputs, so the
generated skill is parameterised rather than hard-wired to one folder.
"""

from __future__ import annotations

import os
from typing import Any

from personalos.database.models import Workflow
from personalos.errors import SkillError
from personalos.security.risk import RiskLevel
from personalos.skills.schema import SkillDefinition, SkillInput, SkillStep
from personalos.utils.paths import expand_path
from personalos.utils.textutil import slugify, truncate

#: Argument names that hold a filesystem path and are worth parameterising.
_PATH_KEYS = ("path", "source", "destination", "folder", "directory", "output")


def _input_name(key: str, used: set[str]) -> str:
    base = {"path": "folder", "source": "source", "destination": "destination"}.get(key, key)
    name = base
    counter = 2
    while name in used:
        name = f"{base}_{counter}"
        counter += 1
    used.add(name)
    return name


def suggest_skill_name(workflow: Workflow) -> str:
    """Derive a readable, valid skill name from a workflow."""
    if workflow.example_requests:
        words = str(workflow.example_requests[0]).split()[:4]
        candidate = slugify(" ".join(words))
    else:
        candidate = slugify(workflow.name)
    candidate = candidate.strip("_")[:48] or "generated_skill"
    if not candidate[0].isalpha():
        candidate = f"skill_{candidate}"
    return candidate


def generate_skill(
    workflow: Workflow,
    *,
    name: str | None = None,
    description: str | None = None,
    parameterise_paths: bool = True,
) -> SkillDefinition:
    """Build a skill definition from a recorded workflow.

    Args:
        workflow: The observed workflow, with its recorded steps.
        name: Override the derived skill name.
        description: Override the derived description.
        parameterise_paths: Lift path arguments into named inputs.

    Raises:
        SkillError: when the workflow has no usable steps.
    """
    raw_steps: list[dict[str, Any]] = list(workflow.steps or [])
    if not raw_steps:
        raise SkillError(
            f"The workflow {workflow.name!r} has no recorded steps to build a skill from.",
            remediation="Run the task once more so the agent can record what it does.",
        )

    inputs: dict[str, SkillInput] = {}
    used_names: set[str] = set()
    path_bindings: dict[str, str] = {}
    steps: list[SkillStep] = []
    permissions: set[str] = set()
    highest = RiskLevel.READ_ONLY

    for index, raw in enumerate(raw_steps, start=1):
        tool = str(raw.get("tool", ""))
        operation = str(raw.get("operation") or raw.get("action") or "")
        if not tool or not operation:
            continue

        arguments = dict(raw.get("arguments") or {})
        if parameterise_paths:
            for key in list(arguments):
                if key not in _PATH_KEYS or not isinstance(arguments[key], str):
                    continue
                original = arguments[key]
                if original in path_bindings:
                    arguments[key] = path_bindings[original]
                    continue
                input_name = _input_name(key, used_names)
                default = _generalise_path(original)
                inputs[input_name] = SkillInput(
                    type="string",
                    description=f"{key.replace('_', ' ').title()} used by step {index}.",
                    required=False,
                    default=default,
                )
                reference = f"{{{{ inputs.{input_name} }}}}"
                path_bindings[original] = reference
                arguments[key] = reference

        step_id = str(raw.get("id") or f"step_{index}")
        steps.append(
            SkillStep(
                id=step_id,
                description=str(raw.get("description") or f"{tool}.{operation}"),
                tool=tool,
                operation=operation,
                arguments=arguments,
                depends_on=list(raw.get("depends_on") or []),
                expected_outcome=raw.get("expected_outcome"),
            )
        )
        permissions.add(f"{tool}.{operation}")
        level = RiskLevel.parse(int(raw.get("risk_level", 0)))
        highest = max(highest, level)

    if not steps:
        raise SkillError(f"None of {workflow.name!r}'s steps name a tool and an operation.")

    example = (workflow.example_requests or [""])[0]
    return SkillDefinition(
        name=name or suggest_skill_name(workflow),
        description=description
        or truncate(
            workflow.description or f"Repeats what you asked for {workflow.success_count} times: {example}",
            200,
        ),
        created_by="agent",
        inputs=inputs,
        permissions=sorted(permissions),
        # The declared ceiling is what was observed. It grants nothing: every
        # step is still evaluated by the policy engine at run time.
        max_risk_level=int(highest),
        steps=steps,
        derived_from_workflow=workflow.name,
        tags=["generated"],
    )


def _generalise_path(value: str) -> str:
    """Rewrite an absolute path under the user's home as a ``~`` path.

    A default of ``/home/alice/Downloads`` in a generated skill is brittle and
    leaks the account name; ``~/Downloads`` is the same folder and portable.
    """
    try:
        resolved = expand_path(value)
        home = expand_path("~")
        return f"~{os.sep}{resolved.relative_to(home)}"
    except (ValueError, OSError):
        return value
