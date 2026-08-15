"""Regressions for the wrapper's single-invoke execution path.

``CugaAgent.invoke()`` does NOT initialize the policy system on this build --
that lazy init lives in ``CugaSupervisor.invoke()`` (sdk.py:3326), a different
class. So ``reset_policy_storage=True`` is stored on the agent but never acted
on unless ``initialize()`` is awaited explicitly.

Without that call, CUGA's process-global policy store at
``<cuga package>/dbs/cuga.db`` keeps matching playbooks written by earlier
runs, so every candidate inherits the same stale policy.
"""
from __future__ import annotations

import asyncio

import pytest

from agent_evolve.cuga_wrapper import _execute


class _Agent:
    """Records the order of lifecycle calls the wrapper makes."""

    def __init__(self, *, has_initialize: bool = True, initialize_raises: bool = False):
        self.calls: list[str] = []
        self._has_initialize = has_initialize
        self._initialize_raises = initialize_raises
        if has_initialize:
            self.initialize = self._initialize

    async def _initialize(self):
        if self._initialize_raises:
            raise RuntimeError("policy storage unavailable")
        self.calls.append("initialize")

    async def invoke(self, message, **kwargs):
        self.calls.append("invoke")
        return {"answer": "ok"}


def test_execute_initializes_agent_before_invoke():
    """Policy reset only happens inside initialize(); it must precede invoke."""
    agent = _Agent()

    asyncio.run(_execute(agent, "task", [], {}))

    assert agent.calls == ["initialize", "invoke"]


def test_execute_still_invokes_when_agent_has_no_initialize():
    """Older or stubbed agents without initialize() must still run."""
    agent = _Agent(has_initialize=False)

    asyncio.run(_execute(agent, "task", [], {}))

    assert agent.calls == ["invoke"]


def test_execute_fails_closed_when_initialize_errors():
    """A failed policy reset must not silently run with stale policies.

    Proceeding would produce a rollout contaminated by another candidate's
    playbook while reporting success -- fabricated evidence.
    """
    agent = _Agent(initialize_raises=True)

    with pytest.raises(RuntimeError, match="policy storage unavailable"):
        asyncio.run(_execute(agent, "task", [], {}))

    assert "invoke" not in agent.calls


def test_execute_ingests_memory_before_invoke():
    class _Knowledge:
        def __init__(self, agent):
            self.agent = agent

        async def ingest(self, doc):
            self.agent.calls.append(f"ingest:{doc}")

    agent = _Agent()
    agent.knowledge = _Knowledge(agent)

    asyncio.run(_execute(agent, "task", ["/tmp/a.md"], {}))

    assert agent.calls == ["initialize", "ingest:/tmp/a.md", "invoke"]
