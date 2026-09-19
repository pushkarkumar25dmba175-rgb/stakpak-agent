"""Learning from corrections.

When the user says "always save reports as Markdown" or "never rename files in
this folder", that is an instruction, not an observation: it is stored with
confidence 1.0, it does not decay, and an inference can never overwrite it.

Rules that *restrict* the agent are stored as deny policy rules. Rules that
would *widen* what the agent may do without asking are only ever accepted for
levels 0–2 — "don't ask before moving files in this project" is a reasonable
thing to want; "don't ask before deleting" is not, and the policy engine will
not honour it even if it is stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from personalos.agent.reasoning import FeedbackInterpretation
from personalos.database.models import MemorySource
from personalos.memory.memory_manager import MemoryManager
from personalos.security.policy_engine import PolicyEngine
from personalos.security.risk import RiskLevel
from personalos.utils.textutil import slugify, truncate


@dataclass
class LearnedItem:
    """What the agent stored in response to feedback, for it to report back."""

    kind: str
    key: str
    value: Any
    scope: str
    explanation: str
    stored: bool = True

    def render(self) -> str:
        if not self.stored:
            return self.explanation
        return f"Noted: {self.key.replace('_', ' ')} = {self.value!r} ({self.scope})."


#: Feedback keys that map to a policy rule rather than a preference.
_RULE_KEYS = {
    "forbid_rename": ("deny", "filesystem", "rename"),
    "forbid_delete": ("deny", "filesystem", "delete"),
    "forbid_move": ("deny", "filesystem", "move"),
    "reduce_prompts_scope": ("allow", "filesystem", None),
    "always_confirm": ("deny", None, None),
}


class FeedbackLearner:
    """Applies an interpreted correction to memory and policy."""

    def __init__(self, memory: MemoryManager, policy: PolicyEngine) -> None:
        self.memory = memory
        self.policy = policy

    def apply(
        self,
        interpretation: FeedbackInterpretation,
        *,
        project: str | None = None,
        path_scope: str | None = None,
    ) -> LearnedItem:
        """Store the correction. Returns what was stored, for display."""
        scope = project if (interpretation.scope == "project" and project) else "global"

        if not interpretation.actionable:
            # Even unparsed feedback is kept, as a dated note, so the user can
            # see that the agent heard them and nothing was silently dropped.
            key = f"note_{slugify(truncate(interpretation.raw, 40))}"
            self.memory.semantic.remember(
                key,
                interpretation.raw,
                namespace=scope,
                category="feedback",
                source=MemorySource.EXPLICIT,
            )
            return LearnedItem(
                kind="note",
                key=key,
                value=interpretation.raw,
                scope=scope,
                explanation=interpretation.explanation
                or "I kept this as a note but could not turn it into a rule.",
                stored=True,
            )

        if interpretation.kind == "fact":
            self.memory.semantic.remember(
                interpretation.key,
                str(interpretation.value),
                namespace=scope,
                source=MemorySource.EXPLICIT,
            )
            return LearnedItem(
                kind="fact",
                key=interpretation.key,
                value=interpretation.value,
                scope=scope,
                explanation=interpretation.explanation,
            )

        if interpretation.kind == "rule":
            return self._apply_rule(interpretation, scope=scope, path_scope=path_scope)

        self.memory.preferences.set(
            interpretation.key,
            interpretation.value,
            namespace=scope,
            source=MemorySource.EXPLICIT,
            note=truncate(interpretation.raw, 200),
        )
        return LearnedItem(
            kind="preference",
            key=interpretation.key,
            value=interpretation.value,
            scope=scope,
            explanation=interpretation.explanation,
        )

    def _apply_rule(
        self,
        interpretation: FeedbackInterpretation,
        *,
        scope: str,
        path_scope: str | None,
    ) -> LearnedItem:
        mapping = _RULE_KEYS.get(interpretation.key)
        if mapping is None:
            # An unrecognised rule is stored as a preference so it still shows
            # up in context, but it is not given policy force.
            self.memory.preferences.set(
                interpretation.key,
                interpretation.value,
                namespace=scope,
                source=MemorySource.EXPLICIT,
                note=truncate(interpretation.raw, 200),
            )
            return LearnedItem(
                kind="preference",
                key=interpretation.key,
                value=interpretation.value,
                scope=scope,
                explanation=(
                    "I recorded this as a preference. I did not turn it into an enforced "
                    "rule because I could not map it onto a specific tool and action."
                ),
            )

        effect, tool, operation = mapping
        rule_name = f"{interpretation.key}"
        if path_scope:
            rule_name = f"{interpretation.key}_{slugify(path_scope)}"[:120]

        self.policy.add_rule(
            rule_name,
            effect=effect,
            tool=tool,
            operation=operation,
            path_prefix=path_scope,
            # An allow rule is clamped to level 2 by the policy engine as well;
            # stating it here makes the intent explicit at the call site too.
            max_risk_level=int(RiskLevel.MODERATE),
            project=scope if scope != "global" else None,
            source=MemorySource.EXPLICIT,
        )
        explanation = interpretation.explanation or (
            f"I will {'not ' if effect == 'deny' else ''}"
            f"{operation or 'act'} {'on ' + path_scope if path_scope else ''} without asking."
        )
        if effect == "allow":
            explanation += (
                " This covers routine changes only — deletions, installs and anything "
                "that leaves this machine will still ask every time."
            )
        return LearnedItem(
            kind="rule",
            key=rule_name,
            value=effect,
            scope=scope,
            explanation=explanation,
        )

    def correct(self, preference_key: str, *, namespace: str = "global") -> None:
        """Weaken an inferred preference the user just contradicted."""
        self.memory.preferences.weaken(preference_key, namespace=namespace)

    def reinforce_from_success(self, keys: list[str], *, namespace: str = "global") -> None:
        """Strengthen inferred preferences that contributed to a successful task."""
        for key in keys:
            self.memory.preferences.reinforce(key, namespace=namespace)
