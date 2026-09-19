"""Provider-agnostic LLM interface.

The agent depends on this module, never on a vendor SDK. Swapping Claude for a
local Ollama model is a config change, and every provider implementation is
roughly a hundred lines of HTTP because that is all the agent actually needs:
send messages, get text back, optionally as JSON.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from personalos.errors import LLMError
from personalos.utils.textutil import token_estimate


@dataclass
class Message:
    """One turn in a conversation."""

    role: str
    """system | user | assistant"""
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """What a provider returned."""

    text: str
    model: str
    provider: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    stop_reason: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def json(self) -> Any:
        """Parse the response as JSON, tolerating a fenced code block.

        Models wrap JSON in ``` fences often enough that handling it here is
        worth more than insisting every caller strip it.

        Raises:
            LLMError: when the response is not JSON at all.
        """
        text = self.text.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        # Fall back to the outermost braces when the model added prose.
        if not text.startswith(("{", "[")):
            start = min(
                (index for index in (text.find("{"), text.find("[")) if index >= 0),
                default=-1,
            )
            end = max(text.rfind("}"), text.rfind("]"))
            if start >= 0 and end > start:
                text = text[start : end + 1]
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(
                f"The model did not return valid JSON: {exc}",
                remediation="Retry, or run with a larger model for planning.",
            ) from exc


class LLMProvider(ABC):
    """Base class for every provider adapter."""

    name: str = "base"

    def __init__(self, model: str, **options: Any) -> None:
        self.model = model
        self.options = options

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Send ``messages`` and return the model's reply."""

    async def health_check(self) -> tuple[bool, str]:
        """Return ``(ok, detail)``. Used by ``agent doctor``."""
        try:
            response = await self.complete(
                [Message("user", "Reply with the single word: ready")],
                max_tokens=16,
            )
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            return False, f"{type(exc).__name__}: {exc}"
        return True, f"{self.name}/{self.model} responded: {response.text.strip()[:60]}"

    @staticmethod
    def estimate_tokens(messages: list[Message], system: str | None = None) -> int:
        """Approximate the prompt size, for budget enforcement."""
        total = token_estimate(system) if system else 0
        return total + sum(token_estimate(message.content) for message in messages)
