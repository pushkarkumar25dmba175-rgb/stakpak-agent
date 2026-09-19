"""Configuration loading, layering and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from personalos.errors import ConfigurationError
from personalos.settings import (
    ApprovalMode,
    Settings,
    dump_settings,
    env_overlay,
    initialise_home,
    load_settings,
    save_settings,
)


def test_defaults_are_conservative() -> None:
    settings = Settings()
    assert settings.security.approval_mode is ApprovalMode.NORMAL
    assert settings.security.auto_approve_max_level <= 1
    assert settings.security.prefer_trash_over_delete is True
    assert settings.security.redact_logs is True
    assert settings.observation.watch_folders_enabled is False
    assert settings.observation.watch_clipboard_enabled is False
    assert settings.observation.track_app_usage is False
    assert settings.dashboard.enabled is False


def test_auto_approve_ceiling_cannot_reach_destructive_levels() -> None:
    with pytest.raises(ValueError):
        Settings.model_validate({"security": {"auto_approve_max_level": 3}})


def test_skills_cannot_be_auto_created() -> None:
    with pytest.raises(ValueError, match="explicit approval"):
        Settings.model_validate({"learning": {"auto_create_skills": True}})


def test_dashboard_refuses_non_loopback_host() -> None:
    with pytest.raises(ValueError, match="localhost"):
        Settings.model_validate({"dashboard": {"host": "0.0.0.0"}})


def test_config_file_layers_over_defaults(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(
        yaml.safe_dump({"llm": {"model": "from-file"}, "memory": {"retrieval_top_k": 3}}),
        encoding="utf-8",
    )
    settings = load_settings(home=home, use_env=False)
    assert settings.llm.model == "from-file"
    assert settings.memory.retrieval_top_k == 3
    # Untouched sections keep their defaults.
    assert settings.security.approval_mode is ApprovalMode.NORMAL


def test_overrides_beat_the_config_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"llm": {"model": "file"}}), encoding="utf-8")
    settings = load_settings(home=home, use_env=False, overrides={"llm": {"model": "override"}})
    assert settings.llm.model == "override"


def test_config_file_cannot_redirect_the_home_directory(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text(yaml.safe_dump({"home": "/etc"}), encoding="utf-8")
    settings = load_settings(home=home, use_env=False)
    assert settings.home == home.absolute()


def test_env_overlay_coerces_types() -> None:
    overlay = env_overlay(
        {
            "PERSONALOS_LLM_PROVIDER": "ollama",
            "PERSONALOS_SHELL_ENABLED": "false",
            "PERSONALOS_DASHBOARD_PORT": "9000",
        }
    )
    assert overlay == {
        "llm": {"provider": "ollama"},
        "security": {"shell_enabled": False},
        "dashboard": {"port": 9000},
    }


def test_invalid_yaml_reports_the_file(tmp_path: Path) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.yaml").write_text("llm: [unclosed", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not valid YAML"):
        load_settings(home=home, use_env=False)


def test_saved_config_contains_no_secrets(tmp_path: Path) -> None:
    settings = initialise_home(tmp_path / "home")
    body = settings.config_file.read_text(encoding="utf-8")
    assert "api_key_env" in body
    assert "ANTHROPIC_API_KEY" in body  # the variable NAME is fine
    assert "sk-" not in body
    assert "home" not in dump_settings(settings)


def test_initialise_home_is_idempotent(tmp_path: Path) -> None:
    home = tmp_path / "home"
    first = initialise_home(home)
    first.llm.model = "changed"
    save_settings(first)
    second = initialise_home(home)
    assert second.llm.model == "changed"
    assert all(directory.exists() for directory in second.directories())
