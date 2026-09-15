"""Anthropic Messages API adapter."""

from __future__ import annotations

from typing import Any

from personalos.errors import LLMError
from personalos.llm.base import LLMProvider, LLMResponse, Message

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"


class AnthropicProvider(LLMProvider):
    """Talks to Claude over the Messages API.

    The key is passed in at construction from a resolved secret reference; this
    class never reads the environment itself, so there is exactly one place in
    the codebase where a credential is looked up.
    """

    name = "anthropic"

    def __init__(
        self,
        model: str,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout: float = 120.0,
        **options: Any,
    ) -> None:
        super().__init__(model, **options)
        self.api_key = api_key
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
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

        # Anthropic takes the system prompt as a separate field, not a turn.
        conversation = [m for m in messages if m.role != "system"]
        system_text = system or next((m.content for m in messages if m.role == "system"), None)
        if json_mode:
            system_text = (system_text or "") + (
                "\n\nRespond with a single JSON object and nothing else. No prose, no code fences."
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [m.as_dict() for m in conversation],
        }
        if system_text:
            payload["system"] = system_text

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/v1/messages",
                    headers={
                        "x-api-key": self.api_key,
                        "anthropic-version": API_VERSION,
                        "content-type": "application/json",
                    },
                    json=payload,
                )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Could not reach the Anthropic API: {exc}",
                remediation="Check your network connection, or switch to `LLM_PROVIDER=ollama`.",
            ) from exc

        if response.status_code >= 400:
            raise LLMError(
                f"Anthropic returned {response.status_code}: {response.text[:400]}",
                remediation=(
                    "Check that your API key is valid and the model name is correct."
                    if response.status_code in (401, 403, 404)
                    else "Retry in a moment."
                ),
            )

        body = response.json()
        text = "".join(
            block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"
        )
        usage = body.get("usage", {})
        return LLMResponse(
            text=text,
            model=body.get("model", self.model),
            provider=self.name,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            stop_reason=body.get("stop_reason"),
            raw=body,
        )
