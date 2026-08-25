"""Token-budget audit (?08): no model-call path may cap below measured need.

``ox-alpha-free`` is a reasoning model: completion tokens are consumed by
reasoning before any content is emitted, and a small ``max_tokens`` budget
yields silently EMPTY answers (verified live this session: 10 tokens ->
empty content, 500 -> 'PONG'). Measured worst case on the live endpoint is a
rich fault analysis at 4281 completion tokens.

Our own adapter paths (analyzer, positivity judge, dedup adjudicator, editor
model calls) set no explicit ``max_tokens``, so the provider default applies.
CUGA-internal agents are different: ``cuga.backend.llm.models._create_llm_instance``
*asserts* a per-agent ``max_tokens`` from the TOML selected by
``AGENT_SETTING_CONFIG``. The shipped ``settings.openai.toml`` -- the wrapper's
historical default -- caps ``agent.action.model`` at 400 tokens while every
other agent gets 16000. A reasoning model driving tool calls under a
400-token budget starves before emitting its first action.

These tests pin the packaged replacement profile and the wrapper seam that
selects it.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

from agent_evolve.cuga_wrapper import (
    PACKAGED_MODEL_SETTINGS_PATH,
    RuntimeSettings,
)

#: Every CUGA agent section the shipped profiles define. If a CUGA upgrade
#: adds an agent, this test fails until the packaged profile covers it too --
#: a new agent must never silently fall back to the shipped defaults.
EXPECTED_AGENTS = frozenset(
    {
        "task_decomposition",
        "planner",
        "chat",
        "shortlister",
        "plan_controller",
        "final_answer",
        "code",
        "code_planner",
        "qa",
        "action",
    }
)

#: Floor is ~2x the measured worst-case single-call spend (4281 completion
#: tokens). The shipped sibling agents use 16000; the pin is a floor, not an
#: exact value, so deliberate tuning upward stays legal.
MIN_ACCEPTABLE_MAX_TOKENS = 8000


def test_packaged_profile_exists_and_is_loadable():
    assert PACKAGED_MODEL_SETTINGS_PATH is not None
    assert PACKAGED_MODEL_SETTINGS_PATH.is_file()


def test_packaged_profile_covers_every_agent_with_generous_caps():
    with open(PACKAGED_MODEL_SETTINGS_PATH, "rb") as fh:
        profile = tomllib.load(fh)
    agents = profile.get("agent", {})
    covered = {name for name, body in agents.items() if isinstance(body, dict) and "model" in body}
    assert EXPECTED_AGENTS <= covered, f"profile misses agents: {sorted(EXPECTED_AGENTS - covered)}"
    for name in sorted(EXPECTED_AGENTS):
        model = agents[name]["model"]
        assert model["platform"] == "openai", f"{name}: wrapper configures the OpenAI mode"
        assert model["max_tokens"] >= MIN_ACCEPTABLE_MAX_TOKENS, (
            f"{name}: max_tokens={model['max_tokens']} starves a reasoning model "
            f"(measured worst case 4281 completion tokens)"
        )


def test_configure_cuga_environment_defaults_to_packaged_profile(monkeypatch):
    monkeypatch.delenv("AGENT_SETTING_CONFIG", raising=False)
    RuntimeSettings(model="openai/probe").configure_cuga_environment()
    selected = os.environ["AGENT_SETTING_CONFIG"]
    assert selected == str(PACKAGED_MODEL_SETTINGS_PATH)


def test_explicit_agent_setting_config_is_respected(monkeypatch):
    monkeypatch.setenv("AGENT_SETTING_CONFIG", "settings.custom.toml")
    RuntimeSettings(model="openai/probe").configure_cuga_environment()
    assert os.environ["AGENT_SETTING_CONFIG"] == "settings.custom.toml"
