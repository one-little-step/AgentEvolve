"""Regressions for CUGA agent construction from a candidate harness.

CUGA resolves its skills directory from ``cuga_folder`` (see
``cuga.backend.skills.loader.get_skill_root``), not from ``skills_folder``.
Passing ``cuga_folder=None`` for a skills-only candidate makes CUGA silently
fall back to ``<cwd>/.cuga/skills``, loading stale project-level state instead
of the candidate's artifacts. That would make every candidate behave
identically while appearing to run correctly.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from agent_evolve.cuga_wrapper import _construct_agent


class _RecordingAgent:
    """Stands in for ``cuga.CugaAgent`` to capture constructor kwargs."""

    last_kwargs: dict[str, object] = {}

    def __init__(self, **kwargs):
        type(self).last_kwargs = dict(kwargs)


@pytest.fixture
def recorded(monkeypatch):
    module = types.ModuleType("cuga")
    module.CugaAgent = _RecordingAgent
    monkeypatch.setitem(sys.modules, "cuga", module)
    _RecordingAgent.last_kwargs = {}
    return _RecordingAgent


def test_skills_only_candidate_sets_cuga_folder_to_workspace(recorded, tmp_path):
    """Without this, CUGA scans <cwd>/.cuga/skills and ignores the candidate."""
    _construct_agent({"skills": {"s": "body"}}, ["tool"], "instr", str(tmp_path))

    assert recorded.last_kwargs["cuga_folder"] == str(tmp_path)
    assert recorded.last_kwargs["skills_folder"] == str(tmp_path)
    assert recorded.last_kwargs["enable_skills"] is True


def test_policies_only_candidate_sets_cuga_folder_to_workspace(recorded, tmp_path):
    _construct_agent({"policies": {"p": "body"}}, ["tool"], "instr", str(tmp_path))

    assert recorded.last_kwargs["cuga_folder"] == str(tmp_path)
    assert recorded.last_kwargs["auto_load_policies"] is True


def test_memory_only_candidate_still_isolates_cuga_folder(recorded, tmp_path):
    """Memory-only candidates must not inherit stale project skills/policies."""
    _construct_agent({"memory": {"m": "body"}}, ["tool"], "instr", str(tmp_path))

    assert recorded.last_kwargs["cuga_folder"] == str(tmp_path)


def test_no_workspace_leaves_cuga_folder_unset(recorded):
    """With no candidate workspace there is nothing to isolate."""
    _construct_agent({}, ["tool"], "instr", None)

    assert recorded.last_kwargs["cuga_folder"] is None
    assert recorded.last_kwargs["skills_folder"] is None
    assert recorded.last_kwargs["enable_skills"] is False


def test_cuga_resolves_candidate_skills_from_cuga_folder(tmp_path):
    """Pin the real SDK behavior this wiring depends on.

    If a future CUGA release keys discovery off ``skills_folder`` instead,
    this test fails loudly rather than letting candidates silently share
    stale global state.
    """
    loader = pytest.importorskip("cuga.backend.skills.loader")

    skill_dir = tmp_path / "skills" / "status-report"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: status-report\ndescription: d\n---\nbody\n", encoding="utf-8"
    )

    assert loader.get_skill_root(str(tmp_path)) == tmp_path / "skills"
    names = {e.name for e in loader.discover_skills(str(tmp_path))}
    assert "status-report" in names

    # cuga_folder=None escapes the candidate workspace entirely.
    assert loader.get_skill_root(None) != tmp_path / "skills"


# --------------------------------------------------------------------- #
# CUGA_FOLDER isolation
# --------------------------------------------------------------------- #


def test_construct_agent_exports_cuga_folder_env(recorded, tmp_path, monkeypatch):
    """The sandbox executor and policy loader only honor ``CUGA_FOLDER``.

    ``CugaAgent(cuga_folder=...)`` does NOT reach them: ``build_runtime_tools``
    calls ``create_sandbox_tools(thread_id=...)`` without ``cuga_folder``, and
    ``prepare_node`` reads ``os.getenv("CUGA_FOLDER", settings.policy.cuga_folder)``.
    Without exporting the env var, both fall back to ``<cwd>/.cuga`` and load
    stale project-level skills and playbooks into every candidate.
    """
    monkeypatch.delenv("CUGA_FOLDER", raising=False)

    _construct_agent({"skills": {"s": "body"}}, ["tool"], "instr", str(tmp_path))

    import os

    assert os.environ["CUGA_FOLDER"] == str(tmp_path)


def test_construct_agent_clears_cuga_folder_env_without_workspace(recorded, monkeypatch):
    """A candidate-free run must not inherit a previous candidate's folder."""
    monkeypatch.setenv("CUGA_FOLDER", "/stale/previous/candidate")

    _construct_agent({}, ["tool"], "instr", None)

    import os

    assert os.environ.get("CUGA_FOLDER") is None


def test_sibling_candidates_do_not_share_cuga_folder_env(recorded, tmp_path, monkeypatch):
    """Sequential candidates must each rebind the env var."""
    monkeypatch.delenv("CUGA_FOLDER", raising=False)
    first = tmp_path / "cand-A"
    second = tmp_path / "cand-B"
    first.mkdir()
    second.mkdir()

    import os

    _construct_agent({"skills": {"s": "a"}}, ["tool"], "instr", str(first))
    assert os.environ["CUGA_FOLDER"] == str(first)

    _construct_agent({"skills": {"s": "b"}}, ["tool"], "instr", str(second))
    assert os.environ["CUGA_FOLDER"] == str(second)


# --------------------------------------------------------------------- #
# Global policy-store isolation
# --------------------------------------------------------------------- #


def test_candidate_run_resets_shared_policy_storage(recorded, tmp_path):
    """CUGA persists policies in a process-global store, not per workspace.

    The store lives at ``<cuga package>/dbs/cuga.db`` (``config.DBS_DIR``) and
    survives every run regardless of ``cuga_folder``. Without a reset, a
    playbook written by any earlier run keeps matching for every later
    candidate, so all candidates inherit the same stale policy and score
    identically.
    """
    _construct_agent({"skills": {"s": "body"}}, ["tool"], "instr", str(tmp_path))

    assert recorded.last_kwargs["reset_policy_storage"] is True


def test_candidate_with_policies_still_autoloads_after_reset(recorded, tmp_path):
    """Reset must clear stale policies but keep the candidate's own."""
    _construct_agent({"policies": {"p": "body"}}, ["tool"], "instr", str(tmp_path))

    assert recorded.last_kwargs["reset_policy_storage"] is True
    assert recorded.last_kwargs["auto_load_policies"] is True


def test_no_workspace_does_not_reset_shared_storage(recorded):
    """Without a candidate workspace there is no candidate state to protect."""
    _construct_agent({}, ["tool"], "instr", None)

    assert recorded.last_kwargs["reset_policy_storage"] is False
