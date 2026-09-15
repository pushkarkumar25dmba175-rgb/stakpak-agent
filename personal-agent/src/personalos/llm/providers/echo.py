"""A deterministic offline provider.

It exists so the whole agent — planning, execution, verification, learning —
can be exercised in tests and by ``agent doctor`` with no network, no API key
and no cost. It is also what the agent falls back to when it is asked to plan
something it can handle with its own heuristics.
"""

from __future__ import annotations

import json
from typing import Any

from personalos.llm.base import LLMProvider, LLMResponse, Message


class EchoProvider(LLMProvider):
    """Returns canned, structurally valid responses.

    ``scripted`` lets a test queue exact replies; once it is exhausted the
    provider falls back to a generic answer rather than raising, so a test that
    makes one extra call fails on its assertion instead of on plumbing.
    """

    name = "echo"

    def __init__(self, model: str = "echo", *, scripted: list[str] | None = None, **options: Any) -> None:
        super().__init__(model, **options)
        self.scripted = list(scripted or [])
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append(
            {
                "system": system,
                "messages": [m.as_dict() for m in messages],
                "json_mode": json_mode,
            }
        )
        if self.scripted:
            text = self.scripted.pop(0)
        elif json_mode:
            text = json.dumps(
                {
                    "goal": "Answer the request without a model.",
                    "steps": [],
                    "notes": "The echo provider does not plan; the heuristic planner takes over.",
                }
            )
        else:
            last = next((m.content for m in reversed(messages) if m.role == "user"), "")
            text = f"[echo] {last.strip()[:500]}"

        return LLMResponse(
            text=text,
            model=self.model,
            provider=self.name,
            input_tokens=self.estimate_tokens(messages, system),
            output_tokens=len(text) // 4,
            stop_reason="end_turn",
        )

    async def health_check(self) -> tuple[bool, str]:
        return True, "echo provider is always available (offline, deterministic)"
