from __future__ import annotations

import types

import pytest

from agent_evolve.cuga_wrapper import (
    RuntimeSettings,
    _construct_agent,
    _require_autonomous_mode,
    normalize_cuga_configuration_directory,
    resolve_skills_root,
)
from agent_evolve.cuga_wrapper import DEFAULT_SKILLS_ROOT


def test_normalize_cuga_configuration_directory_removes_blank_value(monkeypatch):
    monkeypatch.setenv("CUGA_CONFIGURATIONS_DIR", "   ")

    normalize_cuga_configuration_directory()

    assert "CUGA_CONFIGURATIONS_DIR" not in __import__("os").environ


def test_normalize_cuga_configuration_directory_preserves_non_blank_value(monkeypatch):
    monkeypatch.setenv("CUGA_CONFIGURATIONS_DIR", "/custom/cuga/config")

    normalize_cuga_configuration_directory()

    assert __import__("os").environ["CUGA_CONFIGURATIONS_DIR"] == "/custom/cuga/config"


def test_resolve_skills_root_maps_cuga_to_project_dir(monkeypatch):
    monkeypatch.setenv("SKILLS_ROOT", "cuga")

    resolved = resolve_skills_root()

    assert resolved == str(DEFAULT_SKILLS_ROOT)
    assert resolved.endswith(".cuga/skills")


def test_resolve_skills_root_preserves_absolute_path(monkeypatch, tmp_path):
    monkeypatch.setenv("SKILLS_ROOT", str(tmp_path))

    assert resolve_skills_root() == str(tmp_path)


def test_runtime_settings_from_env_prefers_cuga_vars_over_litellm(monkeypatch):
    monkeypatch.setenv("CUGA_MODEL", "cuga/model")
    monkeypatch.setenv("CUGA_BASE_URL", "https://cuga.example")
    monkeypatch.setenv("CUGA_API_KEY", "cuga-secret")
    monkeypatch.setenv("LITELLM_MODEL", "litellm/model")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://litellm.example")
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-secret")

    settings = RuntimeSettings.from_env()

    assert settings.model == "cuga/model"
    assert settings.base_url == "https://cuga.example"
    assert settings.api_key == "cuga-secret"


def test_require_autonomous_mode_raises_when_disabled(monkeypatch):
    config = types.ModuleType("cuga.config")

    class AdvancedFeatures:
        force_autonomous_mode = False

    class Settings:
        advanced_features = AdvancedFeatures()

    config.settings = Settings()
    monkeypatch.setitem(__import__("sys").modules, "cuga", types.ModuleType("cuga"))
    monkeypatch.setitem(__import__("sys").modules, "cuga.config", config)

    with pytest.raises(RuntimeError, match="autonomous"):
        _require_autonomous_mode()


def test_require_autonomous_mode_passes_when_enabled(monkeypatch):
    config = types.ModuleType("cuga.config")

    class AdvancedFeatures:
        force_autonomous_mode = True

    class Settings:
        advanced_features = AdvancedFeatures()

    config.settings = Settings()
    monkeypatch.setitem(__import__("sys").modules, "cuga", types.ModuleType("cuga"))
    monkeypatch.setitem(__import__("sys").modules, "cuga.config", config)

    _require_autonomous_mode()


def test_construct_agent_uses_verified_kwargs(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    cuga = types.ModuleType("cuga")
    cuga.CugaAgent = FakeAgent
    monkeypatch.setitem(__import__("sys").modules, "cuga", cuga)

    tools = ["tool-a", "tool-b"]
    agent = _construct_agent({}, tools, "default instructions")

    assert agent is not None
    assert captured["tools"] == tools
    assert captured["special_instructions"] == "default instructions"
    assert captured["enable_knowledge"] is True


def test_construct_agent_prefers_harness_tools_and_instructions(monkeypatch):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    cuga = types.ModuleType("cuga")
    cuga.CugaAgent = FakeAgent
    monkeypatch.setitem(__import__("sys").modules, "cuga", cuga)

    _construct_agent(
        {"tools": ["h-tool"], "instructions": "custom instructions"},
        ["default-tool"],
        "default instructions",
    )

    assert captured["tools"] == ["h-tool"]
    assert captured["special_instructions"] == "custom instructions"
    assert captured["enable_knowledge"] is True
