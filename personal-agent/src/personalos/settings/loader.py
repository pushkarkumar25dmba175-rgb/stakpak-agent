"""Loading, merging and saving configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from personalos.errors import ConfigurationError
from personalos.settings.schema import Settings
from personalos.utils.paths import expand_path

ENV_PREFIX = "PERSONALOS_"
HOME_ENV = "PERSONALOS_HOME"

_ENV_OVERRIDES: dict[str, tuple[str, ...]] = {
    # Flat, documented environment overrides. Anything not listed here has to
    # live in config.yaml, which keeps the env surface small and predictable.
    "LLM_PROVIDER": ("llm", "provider"),
    "LLM_MODEL": ("llm", "model"),
    "LLM_BASE_URL": ("llm", "base_url"),
    "LLM_API_KEY_ENV": ("llm", "api_key_env"),
    "APPROVAL_MODE": ("security", "approval_mode"),
    "SHELL_ENABLED": ("security", "shell_enabled"),
    "ACTIVE_PROJECT": ("active_project",),
    "DASHBOARD_PORT": ("dashboard", "port"),
    "LEARNING_ENABLED": ("learning", "enabled"),
}

_BOOL_TRUE = {"1", "true", "yes", "on"}
_BOOL_FALSE = {"0", "false", "no", "off"}


def _coerce(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in _BOOL_TRUE:
        return True
    if lowered in _BOOL_FALSE:
        return False
    if lowered.isdigit():
        return int(lowered)
    return value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` into ``base``, returning a new dict."""
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _set_path(target: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value


def default_home() -> Path:
    """Resolve the PersonalOS home directory, honouring ``$PERSONALOS_HOME``."""
    override = os.environ.get(HOME_ENV)
    return expand_path(override) if override else expand_path("~/.personalos")


def env_overlay(environ: dict[str, str] | None = None) -> dict[str, Any]:
    """Build a config overlay from the documented ``PERSONALOS_*`` variables."""
    environ = dict(os.environ if environ is None else environ)
    overlay: dict[str, Any] = {}
    for suffix, path in _ENV_OVERRIDES.items():
        raw = environ.get(ENV_PREFIX + suffix)
        if raw is not None and raw != "":
            _set_path(overlay, path, _coerce(raw))
    return overlay


def read_config_file(path: Path) -> dict[str, Any]:
    """Read a YAML config file, returning ``{}`` when it does not exist."""
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(
            f"{path} is not valid YAML: {exc}",
            remediation="Fix the syntax, or delete the file to regenerate defaults.",
        ) from exc
    if not isinstance(data, dict):
        raise ConfigurationError(
            f"{path} must contain a YAML mapping at the top level.",
            remediation="The file should start with keys like `llm:` or `security:`.",
        )
    return data


def load_settings(
    *,
    home: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    use_env: bool = True,
) -> Settings:
    """Assemble the effective settings from every configuration layer.

    Args:
        home: Override the PersonalOS home directory (tests pass a tmp path).
        overrides: Highest-priority overlay, normally from CLI flags.
        use_env: Whether to read ``PERSONALOS_*`` environment variables.

    Raises:
        ConfigurationError: when the merged configuration is invalid.
    """
    resolved_home = expand_path(home) if home is not None else default_home()

    layered: dict[str, Any] = {"home": str(resolved_home)}
    layered = _deep_merge(layered, read_config_file(resolved_home / "config.yaml"))
    if use_env:
        layered = _deep_merge(layered, env_overlay())
    if overrides:
        layered = _deep_merge(layered, overrides)
    # The home directory is decided by the environment/argument, never by the
    # file it was loaded from — otherwise config.yaml could redirect itself.
    layered["home"] = str(resolved_home)

    try:
        return Settings.model_validate(layered)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid configuration:\n{exc}",
            remediation=f"Edit {resolved_home / 'config.yaml'} or run `agent doctor`.",
        ) from exc


def dump_settings(settings: Settings) -> dict[str, Any]:
    """Serialise settings to plain YAML-ready data, excluding derived paths."""
    data = settings.model_dump(mode="json", exclude={"home"})
    return data


def save_settings(settings: Settings, path: Path | None = None) -> Path:
    """Write settings to ``config.yaml`` (or ``path``) and return the location."""
    target = path or settings.config_file
    target.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(dump_settings(settings), sort_keys=False, allow_unicode=True)
    header = (
        "# PersonalOS Agent configuration.\n"
        "# Secrets do not belong here: `llm.api_key_env` names an environment\n"
        "# variable, and the agent reads the key from there at call time.\n"
    )
    target.write_text(header + body, encoding="utf-8")
    return target


def initialise_home(home: Path | str | None = None) -> Settings:
    """Create the home directory tree and a default config file if absent."""
    settings = load_settings(home=home)
    settings.ensure_directories()
    if not settings.config_file.exists():
        save_settings(settings)
    return settings
