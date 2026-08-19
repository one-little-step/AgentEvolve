"""Tests for the memory-exhaustion fixes (2026-08-19 90 GB crash).

A live 3-round RHO run grew past ~90 GB and the machine died. The investigation
(`feedback/rho-memory-leak-report.md`) named four leaks. These tests pin the
behaviour of each fix, because every one of them is invisible in normal operation:
a leak that is fixed and a leak that is not both produce correct answers, and the
difference only shows up hours later as RAM.

Two things are deliberately NOT tested here:

* Absolute memory figures. They are machine- and model-dependent, so asserting on
  RSS would produce a test that fails for reasons unrelated to the fix.
* That the SDK's internals actually free memory. We do not control CUGA's
  allocator. What we control, and therefore what we assert, is that our code
  *calls the teardown the SDK provides* and *bounds the payloads we hand it*.
"""
from __future__ import annotations

import json
import sys

import pytest

from agent_evolve.core.contracts import ExecutionTrace, TraceEvent


def _event(i: int, payload_bytes: int = 0) -> TraceEvent:
    return TraceEvent(
        event_id=f"e{i}",
        kind="tool",
        actor_id="agent",
        parent_event_id=None,
        payload={"blob": "x" * payload_bytes} if payload_bytes else {"step": i},
    )


def _trace(n_events: int, payload_bytes: int = 0) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id="t",
        candidate_id="v",
        task_id="task-1",
        events=tuple(_event(i, payload_bytes) for i in range(n_events)),
        final_output="done",
        status="ok",
    )


# ---------------------------------------------------------------------- #
# Leak 4: the preference judge rendered every payload unbounded.
# ---------------------------------------------------------------------- #
def test_render_trace_is_bounded_for_huge_payloads() -> None:
    """A 4 MB observation must not reach the judge's context verbatim.

    ``max_observation_bytes`` defaults to 4_194_304, and phase 9 renders *two*
    trajectories per comparison across N*k comparisons per round. Unbounded, one
    round can push hundreds of MB through the judge -- which is where the 90 GB
    run actually died.
    """
    from agent_evolve.adapters.cuga_preference_judge import _render_trace

    rendered = _render_trace(_trace(4, payload_bytes=1_000_000))
    assert len(rendered) < 200_000, (
        f"rendered trace is {len(rendered)} bytes; payload bodies are not bounded"
    )


def test_render_trace_keeps_event_structure_when_truncating() -> None:
    """Truncation must preserve the signal the rubric depends on.

    The judge compares *trajectories*: how many steps, which tools, in what
    order. Dropping payload bodies is safe for that; dropping the event list
    would make the two slots incomparable and silently void the verdict.
    """
    from agent_evolve.adapters.cuga_preference_judge import _render_trace

    payload = json.loads(_render_trace(_trace(7, payload_bytes=500_000)))
    assert payload["event_count"] == 7
    assert len(payload["events"]) == 7
    for event in payload["events"]:
        assert event["kind"] == "tool"
        assert event["actor_id"] == "agent"


def test_render_trace_marks_truncated_payloads_honestly() -> None:
    """A truncated payload must say so, not look like a small payload.

    Silent truncation would let the judge conclude a step did little work when in
    fact its output was megabytes.
    """
    from agent_evolve.adapters.cuga_preference_judge import _render_trace

    payload = json.loads(_render_trace(_trace(2, payload_bytes=500_000)))
    blob = json.dumps(payload["events"][0]["payload"])
    assert "truncated" in blob.lower(), (
        "a dropped payload body must be labelled, not silently shrunk"
    )


def test_render_trace_leaves_small_payloads_untouched() -> None:
    """The bound must not distort ordinary traces.

    Most events are small. If the cap rewrote them too, every existing judgement
    would change for no memory benefit.
    """
    from agent_evolve.adapters.cuga_preference_judge import _render_trace

    payload = json.loads(_render_trace(_trace(3)))
    assert payload["events"][0]["payload"] == {"step": 0}
    assert payload["events"][2]["payload"] == {"step": 2}


def test_render_trace_bounds_total_event_count() -> None:
    """A 100k-event trace must not be rendered event-by-event.

    Successful rollouts ran 25--127 events, so a trace with thousands is
    pathological. Rendering it in full would blow the context regardless of
    per-payload caps.
    """
    from agent_evolve.adapters.cuga_preference_judge import _render_trace

    payload = json.loads(_render_trace(_trace(5_000)))
    # The true count must survive even when the list is sampled.
    assert payload["event_count"] == 5_000
    assert len(payload["events"]) < 5_000


# ---------------------------------------------------------------------- #
# Leak 2: CugaAgent was constructed per call and never torn down.
# ---------------------------------------------------------------------- #
def test_workspace_agent_closes_the_agent_after_use(monkeypatch) -> None:
    """``aclose`` -- NOT ``close`` -- must be awaited after every invocation.

    Verified against the installed SDK: ``CugaAgent`` exposes ``aclose`` and has
    no ``close``. A teardown written as ``agent.close()`` would raise
    ``AttributeError``, and if that were swallowed the leak would persist while
    looking fixed.
    """
    import agent_evolve.adapters.cuga_workspace_agent as mod

    closed: list[str] = []

    class FakeAgent:
        def __init__(self, *a, **k) -> None:
            pass

        async def initialize(self) -> None:
            pass

        async def invoke(self, prompt: str, track_tool_calls: bool = False):
            class R:
                tool_calls = ()

                def __str__(self) -> str:
                    return "answer"

            return R()

        async def aclose(self) -> None:
            closed.append("aclose")

    fake_cuga = type(sys)("cuga")
    fake_cuga.CugaAgent = FakeAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cuga", fake_cuga)
    monkeypatch.setattr(mod, "prepare_workspace_environment", lambda *a, **k: None)
    monkeypatch.setattr(mod, "build_tracked_tools", lambda *a, **k: [])
    monkeypatch.setattr(mod, "workspace_agent_kwargs", lambda *a, **k: {})

    mod._run_real_agent({}, "prompt", {}, None, "")
    assert closed == ["aclose"], "workspace agent was not torn down after use"


def test_workspace_agent_closes_even_when_invoke_raises(monkeypatch) -> None:
    """Teardown must be in a ``finally``.

    A failing agent is the case most likely to repeat -- a retry loop that leaks
    on every failure leaks fastest exactly when things are going wrong.
    """
    import agent_evolve.adapters.cuga_workspace_agent as mod

    closed: list[str] = []

    class BoomAgent:
        def __init__(self, *a, **k) -> None:
            pass

        async def initialize(self) -> None:
            pass

        async def invoke(self, *a, **k):
            raise RuntimeError("model exploded")

        async def aclose(self) -> None:
            closed.append("aclose")

    fake_cuga = type(sys)("cuga")
    fake_cuga.CugaAgent = BoomAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cuga", fake_cuga)
    monkeypatch.setattr(mod, "prepare_workspace_environment", lambda *a, **k: None)
    monkeypatch.setattr(mod, "build_tracked_tools", lambda *a, **k: [])
    monkeypatch.setattr(mod, "workspace_agent_kwargs", lambda *a, **k: {})

    with pytest.raises(RuntimeError):
        mod._run_real_agent({}, "prompt", {}, None, "")
    assert closed == ["aclose"], "a failing agent leaked its resources"


def test_editor_agent_closes_the_agent_after_use(monkeypatch) -> None:
    """The genetic editor constructs one agent per ``propose_edit``."""
    import agent_evolve.adapters.cuga_editor as mod

    closed: list[str] = []

    class FakeAgent:
        def __init__(self, *a, **k) -> None:
            pass

        async def initialize(self) -> None:
            pass

        async def invoke(self, prompt: str, track_tool_calls: bool = False):
            class R:
                tool_calls = ()

                def __str__(self) -> str:
                    return "edited"

            return R()

        async def aclose(self) -> None:
            closed.append("aclose")

    fake_cuga = type(sys)("cuga")
    fake_cuga.CugaAgent = FakeAgent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "cuga", fake_cuga)

    editor = mod.CugaEditorAgent.__new__(mod.CugaEditorAgent)
    editor._active_ctx = None  # type: ignore[attr-defined]
    editor.last_sdk_tool_calls = ()  # type: ignore[attr-defined]

    monkeypatch.setattr(mod, "materialize_editor_skills", lambda p: p)
    monkeypatch.setattr(mod, "prepare_editor_environment", lambda d: None)
    monkeypatch.setattr(mod, "editor_agent_kwargs", lambda d: {})
    monkeypatch.setattr(
        "agent_evolve.adapters.cuga_editor_tools.build_editor_tools",
        lambda *a, **k: [],
    )

    editor._run_cuga_agent({}, "prompt")
    assert closed == ["aclose"], "editor agent was not torn down after use"


# ---------------------------------------------------------------------- #
# Leak 1 (dominant): workers reused one wrapper for the whole run.
# ---------------------------------------------------------------------- #
def test_pool_exposes_a_recycle_threshold() -> None:
    """The dominant growth term is unbounded rollouts per worker process.

    12 workers x hundreds of rollouts, each accumulating SDK state, is the 90 GB.
    Recycling caps it at per-worker steady state.
    """
    from agent_evolve.benchmarks.cuga_process_pool import CugaProcessPool

    pool = CugaProcessPool(root="/tmp/ae-test-root", trace_root="/tmp/ae-test-traces")
    assert pool.max_rollouts_per_worker >= 1


def test_recycle_threshold_is_configurable() -> None:
    from agent_evolve.benchmarks.cuga_process_pool import CugaProcessPool

    pool = CugaProcessPool(
        root="/tmp/ae-test-root",
        trace_root="/tmp/ae-test-traces",
        max_rollouts_per_worker=5,
    )
    assert pool.max_rollouts_per_worker == 5


def test_recycle_threshold_rejects_zero_and_negative() -> None:
    """Zero would mean "recycle before running", i.e. never make progress."""
    from agent_evolve.benchmarks.cuga_process_pool import CugaProcessPool

    for bad in (0, -1):
        with pytest.raises(ValueError):
            CugaProcessPool(
                root="/tmp/ae-test-root",
                trace_root="/tmp/ae-test-traces",
                max_rollouts_per_worker=bad,
            )


def test_worker_tracks_its_rollout_count() -> None:
    """Recycling needs a per-worker counter, not a global one.

    A global count would recycle every worker at once, stalling the whole pool.
    """
    import subprocess
    from pathlib import Path

    from agent_evolve.benchmarks.cuga_process_pool import _Worker

    worker = _Worker(
        worker_id="w0",
        harness_version="base-v0",
        process=subprocess.Popen(["true"], text=True),
        knowledge_dir=Path("/tmp"),
        dbs_dir=Path("/tmp"),
    )
    assert worker.rollouts_served == 0
    worker.rollouts_served += 1
    assert worker.rollouts_served == 1


# ---------------------------------------------------------------------- #
# Leak 3: browser orphans and workspace scratch (out-of-heap).
# ---------------------------------------------------------------------- #
def test_browser_scan_ignores_a_users_own_firefox() -> None:
    """A bare `firefox` must NOT be selected.

    The report's suggested `pkill -f firefox` would kill the developer's browser.
    Only Playwright-managed processes are eligible.
    """
    from agent_evolve.benchmarks.cleanup import find_orphaned_browsers

    ps = "\n".join([
        "  501 /Applications/Firefox.app/Contents/MacOS/firefox",
        "  502 /Users/x/Library/Caches/ms-playwright/firefox-1234/firefox --headless",
        "  503 /usr/bin/python3 -m playwright run-driver",
    ])
    assert find_orphaned_browsers(_ps_output=ps) == (502, 503)


def test_browser_scan_never_returns_own_pid() -> None:
    import os

    from agent_evolve.benchmarks.cleanup import find_orphaned_browsers

    ps = f"{os.getpid()} ms-playwright/firefox-1/firefox"
    assert find_orphaned_browsers(_ps_output=ps) == ()


def test_terminate_is_dry_run_by_default() -> None:
    """Killing processes must be opt-in, never a side effect of asking."""
    from agent_evolve.benchmarks.cleanup import terminate_orphaned_browsers

    killed, errors = terminate_orphaned_browsers((999999,))
    assert killed == () and errors == ()


def test_prune_keeps_recent_workspaces(tmp_path) -> None:
    """An in-flight rollout's workspace must survive.

    Age is the protection: a live rollout keeps its mtime fresh, so it cannot be
    deleted out from under itself.
    """
    import os
    import time

    from agent_evolve.benchmarks.cleanup import prune_workspace_scratch

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "f.txt").write_text("x" * 100)
    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "f.txt").write_text("y" * 100)
    old = time.time() - 10_000
    os.utime(stale, (old, old))

    removed, reclaimed, errors = prune_workspace_scratch(
        tmp_path, max_age_seconds=3600, dry_run=False
    )
    assert removed == 1 and errors == ()
    assert fresh.exists(), "a recent workspace was deleted"
    assert not stale.exists(), "a stale workspace survived"
    assert reclaimed >= 100


def test_prune_dry_run_reports_without_deleting(tmp_path) -> None:
    import os
    import time

    from agent_evolve.benchmarks.cleanup import prune_workspace_scratch

    stale = tmp_path / "stale"
    stale.mkdir()
    (stale / "f.txt").write_text("z" * 500)
    old = time.time() - 10_000
    os.utime(stale, (old, old))

    removed, reclaimed, _ = prune_workspace_scratch(
        tmp_path, max_age_seconds=3600, dry_run=True
    )
    assert removed == 1
    assert reclaimed >= 500
    assert stale.exists(), "dry run deleted a directory"


def test_prune_tolerates_a_missing_root() -> None:
    """Cleanup after a run that never created a workspace must not raise."""
    from agent_evolve.benchmarks.cleanup import prune_workspace_scratch

    assert prune_workspace_scratch("/nonexistent/ae-path") == (0, 0, ())


def test_run_cleanup_is_dry_by_default(tmp_path) -> None:
    from agent_evolve.benchmarks.cleanup import run_cleanup

    report = run_cleanup(workspace_root=tmp_path, kill_browsers=False)
    assert report.killed == ()
    assert report.errors == ()
