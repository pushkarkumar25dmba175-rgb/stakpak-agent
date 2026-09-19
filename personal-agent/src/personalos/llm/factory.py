"""Building the configured provider, and degrading gracefully without one."""

from __future__ import annotations

from personalos.errors import ConfigurationError
from personalos.llm.base import LLMProvider
from personalos.llm.providers import (
    AnthropicProvider,
    EchoProvider,
    OllamaProvider,
    OpenAIProvider,
)
from personalos.security.secrets import SecretResolver
from personalos.settings import LLMProviderName, Settings


def build_provider(
    settings: Settings,
    *,
    resolver: SecretResolver | None = None,
    fallback_to_echo: bool = False,
) -> LLMProvider:
    """Construct the provider named in configuration.

    Args:
        fallback_to_echo: When True, a missing API key produces the offline
            echo provider instead of an error. The CLI uses this so commands
            that do not need a model (``history``, ``memory``, ``skills``)
            still work on a machine with no key configured.

    Raises:
        ConfigurationError: when the provider needs a key and does not have one.
    """
    resolver = resolver or SecretResolver()
    llm = settings.llm
    provider = LLMProviderName(llm.provider)

    if provider is LLMProviderName.ECHO:
        return EchoProvider(llm.model)

    if provider is LLMProviderName.OLLAMA:
        return OllamaProvider(llm.model, base_url=llm.base_url, timeout=llm.timeout_seconds)

    api_key = resolver.try_resolve(f"env:{llm.api_key_env}")
    if not api_key:
        if fallback_to_echo:
            return EchoProvider(llm.model)
        raise ConfigurationError(
            f"{provider.value} needs an API key, but ${llm.api_key_env} is not set.",
            remediation=(
                f"Export {llm.api_key_env}, add it to your .env file, or set "
                "`llm.provider: ollama` in config.yaml to run fully locally."
            ),
        )

    if provider is LLMProviderName.ANTHROPIC:
        return AnthropicProvider(
            llm.model, api_key=api_key, base_url=llm.base_url, timeout=llm.timeout_seconds
        )
    if provider is LLMProviderName.OPENAI:
        return OpenAIProvider(
            llm.model, api_key=api_key, base_url=llm.base_url, timeout=llm.timeout_seconds
        )

    raise ConfigurationError(  # pragma: no cover - the enum covers every branch
        f"Unsupported provider {llm.provider!r}.",
        remediation="Valid values: anthropic, openai, ollama, echo.",
    )
