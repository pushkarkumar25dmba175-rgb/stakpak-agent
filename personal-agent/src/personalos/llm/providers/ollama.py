"""Ollama adapter — a model running on this machine.

This is the configuration that makes PersonalOS fully local: no request leaves
the computer, which matters for an agent whose whole job is reading your files.
"""

from __future__ import annotations

from typing import Any

from personalos.errors import LLMError
from personalos.llm.base import LLMProvider, LLMResponse, Message

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout: float = 300.0,
        api_key: str | None = None,
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        # Ollama needs no key; the parameter exists so the factory can treat
        # every provider identically.
        self.timeout = timeout

    async def complete(
        self,
        messages: list[Message],
        *,
        system: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        json_mode: bool = False,
    ) -> LLMResponse:
        import httpx

        turns = ([{"role": "system", "content": system}] if system else []) + [
            m.as_dict() for m in messages
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if json_mode:
            payload["format"] = "json"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(f"{self.base_url}/api/chat", json=payload)
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Could not reach Ollama at {self.base_url}: {exc}",
                remediation="Start it with `ollama serve`, then `ollama pull " + self.model + "`.",
            ) from exc

        if response.status_code >= 400:
            raise LLMError(
                f"Ollama returned {response.status_code}: {response.text[:400]}",
                remediation=f"Check that the model is pulled: `ollama pull {self.model}`.",
            )

        body = response.json()
        return LLMResponse(
            text=(body.get("message") or {}).get("content", "") or "",
            model=body.get("model", self.model),
            provider=self.name,
            input_tokens=body.get("prompt_eval_count"),
            output_tokens=body.get("eval_count"),
            stop_reason=body.get("done_reason"),
            raw=body,
        )
