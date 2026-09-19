"""OpenAI-compatible Chat Completions adapter.

Works against OpenAI itself and against anything that speaks the same API —
vLLM, LM Studio, LiteLLM, OpenRouter — by pointing ``base_url`` at it.
"""

from __future__ import annotations

from typing import Any

from personalos.errors import LLMError
from personalos.llm.base import LLMProvider, LLMResponse, Message

DEFAULT_BASE_URL = "https://api.openai.com/v1"


class OpenAIProvider(LLMProvider):
    """Talks to any Chat Completions endpoint."""

    name = "openai"

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

        turns = ([{"role": "system", "content": system}] if system else []) + [
            m.as_dict() for m in messages
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": turns,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(
                f"Could not reach {self.base_url}: {exc}",
                remediation="Check the base URL and your network connection.",
            ) from exc

        if response.status_code >= 400:
            raise LLMError(
                f"The API returned {response.status_code}: {response.text[:400]}",
                remediation="Verify the model name and API key.",
            )

        body = response.json()
        choice = (body.get("choices") or [{}])[0]
        usage = body.get("usage", {})
        return LLMResponse(
            text=(choice.get("message") or {}).get("content", "") or "",
            model=body.get("model", self.model),
            provider=self.name,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            stop_reason=choice.get("finish_reason"),
            raw=body,
        )
