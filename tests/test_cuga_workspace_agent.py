"""Tests for the shared Interface B workspace-agent runner.

Every test injects an agent_factory; no test constructs a real CugaAgent and no
test makes a network call.
"""
from __future__ import annotations

import inspect

from agent_evolve.adapters.cuga_workspace_agent import (
    WorkspaceAgentRun,
    run_workspace_agent,
)

APP_NAMES = {"read_thing": "evidence", "submit": "submit"}


def _callables() -> dict:
    def read_thing() -> str:
        """Read the thing."""
        return '{"thing": "value"}'

    def submit(verdict: str) -> str:
        """Submit the verdict."""
        return '{"status": "ok"}'

    return {"read_thing": read_thing, "submit": submit}


def test_returns_the_agent_answer() -> None:
    def factory(callables: dict, prompt: str) -> str:
        return "done"

    run = run_workspace_agent(
        _callables(), "do the thing", app_names=APP_NAMES, agent_factory=factory
    )

    assert isinstance(run, WorkspaceAgentRun)
    assert run.ok is True
    assert run.answer == "done"


def test_records_which_tools_actually_executed() -> None:
    def factory(callables: dict, prompt: str) -> str:
        callables["read_thing"]()
        callables["submit"](verdict="a")
        callables["read_thing"]()
        return "done"

    run = run_workspace_agent(
        _callables(), "p", app_names=APP_NAMES, agent_factory=factory
    )

    assert run.tools_called == ("read_thing", "submit", "read_thing")


def test_no_tool_calls_is_recorded_not_inferred() -> None:
    def factory(callables: dict, prompt: str) -> str:
        return "I would read the thing."  # narration only

    run = run_workspace_agent(
        _callables(), "p", app_names=APP_NAMES, agent_factory=factory
    )

    assert run.tools_called == ()
    assert run.ok is True
    assert run.no_tool_call is True


def test_agent_failure_is_returned_as_data_not_raised() -> None:
    def factory(callables: dict, prompt: str) -> str:
        raise RuntimeError("agent exploded")

    run = run_workspace_agent(
        _callables(), "p", app_names=APP_NAMES, agent_factory=factory
    )

    assert run.ok is False
    assert "agent exploded" in run.error
    assert "RuntimeError" in run.error
    assert run.answer == ""


def test_a_raising_tool_does_not_abort_the_run() -> None:
    def boom() -> str:
        """Explode."""
        raise ValueError("tool broke")

    callables = {"read_thing": boom, "submit": lambda verdict: "{}"}

    def factory(cs: dict, prompt: str) -> str:
        try:
            cs["read_thing"]()
        except ValueError:
            pass
        return "recovered"

    run = run_workspace_agent(
        callables, "p", app_names=APP_NAMES, agent_factory=factory
    )

    assert run.ok is True
    assert run.tools_called == ("read_thing",)


def test_recording_wrapper_preserves_docstring_and_signature() -> None:
    """LangChain's @tool refuses a body with no docstring and derives the args
    schema from the signature; a bare *args wrapper silently destroys both."""
    seen: dict = {}

    def factory(callables: dict, prompt: str) -> str:
        seen.update(callables)
        return "ok"

    run_workspace_agent(
        _callables(), "p", app_names=APP_NAMES, agent_factory=factory
    )

    assert seen["read_thing"].__doc__ == "Read the thing."
    assert seen["submit"].__doc__ == "Submit the verdict."
    assert list(inspect.signature(seen["submit"]).parameters) == ["verdict"]
    assert seen["read_thing"].__name__ == "read_thing"


def test_tools_called_survive_an_agent_failure() -> None:
    """Evidence of what ran must not be discarded because the run then failed."""

    def factory(callables: dict, prompt: str) -> str:
        callables["read_thing"]()
        raise RuntimeError("died after the tool ran")

    run = run_workspace_agent(
        _callables(), "p", app_names=APP_NAMES, agent_factory=factory
    )

    assert run.ok is False
    assert run.tools_called == ("read_thing",)
