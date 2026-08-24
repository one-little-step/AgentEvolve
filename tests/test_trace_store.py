"""TS2: the cross-attempt trace store (D5 prerequisite, first build item).

Governing contracts:
* ``docs/design/issue-lifecycle.md`` D5.5 prerequisite 1 / D5.6 chain: Judge 2
  needs traces for the same task from *other* candidates, surviving the
  per-attempt resets of ``_last_validation_traces`` / ``_last_observation_traces``.

Capture policy (decided 2026-08-23, revising the original "winning traces"
sketch): complementarity is **relative per mechanism**, so capture applies **no
quality gate** — every *scorable* rollout is stored, pass or fail. A 0.4 may be
the best any candidate has done on a mechanism, and the editor tool degrades to
least-bad failures, which requires failure traces in the store. Unscorable
rollouts stay excluded everywhere (SV-9): no measurement is not evidence.

Persistence policy (decided 2026-08-23 with the user): **in-memory only**.
``_persist_attempt``'s invariant — raw traces never reach storage — stands, and
the storage sanitizer truncates strings at 2000 chars, so a persisted trace
would come back silently amputated. The tests pin that invariant against
future wiring.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionCandidate,
    EvolutionTask,
)
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from agent_evolve.core.storage import JSONFileStorage  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"


def _task(task_id: str = "task-a", expected: str = _TOKEN) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": expected},
    )


def _runner(
    adapter: FakeAdapter | None = None,
    storage: "JSONFileStorage | None" = None,
) -> SequentialGepaRunner:
    adapter = adapter or FakeAdapter()
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base",
            version="base-v0",
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in adapter.artifact_inventory("base-v0")
            },
        )
    )
    return SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        storage=storage,
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
    )


# ---------------------------------------------------------------------- #
# Capture across attempts, no quality gate
# ---------------------------------------------------------------------- #
def test_store_survives_attempt_resets_and_keeps_failures_and_passes() -> None:
    """Two attempts; the store accumulates across both, unfiltered by score."""
    runner = _runner()

    first = runner.run_attempt([_task("task-a")])
    assert first.accepted, f"attempt 1 should commit, got: {first.reason}"
    child_id = first.result_candidate_id
    assert child_id is not None

    base_entries = runner.traces_for("base", "task-a")
    child_entries = runner.traces_for(child_id, "task-a")

    # Base was observed failing (its artifacts carry no token); the child's
    # validated probe passed after the edit. BOTH must be present: the store
    # is not a winners-only list — relative comparison needs the spread.
    assert base_entries, "parent observation was not captured"
    assert all(entry.score is not None and entry.score.score < 0.5 for entry in base_entries)
    assert child_entries, "child validation rollout was not captured"
    assert all(entry.score is not None and entry.scorable for entry in child_entries)

    # A second attempt resets _last_validation_traces but must NOT reset this.
    second = runner.run_attempt([_task("task-a")])
    if second.accepted and second.result_candidate_id:
        grandchild_entries = runner.traces_for(second.result_candidate_id, "task-a")
        assert grandchild_entries
    assert len(runner.traces_for("base", "task-a")) >= len(base_entries)


def test_capture_applies_no_quality_gate_at_either_route() -> None:
    """validate() and build_issues() both feed the store for any scorable score."""
    runner = _runner()
    workspace = runner.adapter.materialize_candidate("base-v0", "att-store-1")
    report = runner.validate(workspace, _task("task-a"))

    # Default harness: the unedited child fails task-a -> stored despite failing.
    entries = runner.traces_for(workspace.version, "task-a")
    assert entries
    assert all(entry.score is not None and entry.score.score < 0.5 for entry in entries)

    issues = runner.build_issues([_task("task-a")])
    assert issues, "expected at least one issue from the failing base"
    base_entries = runner.traces_for("base", "task-a")
    assert base_entries


def test_unscorable_rollouts_never_reach_the_store() -> None:
    """SV-9 boundary holds here too: a crash is not evidence, anywhere."""
    class _CrashingAdapter(FakeAdapter):
        def capture_trace(self, rollout_result: object):  # type: ignore[override]
            trace = super().capture_trace(rollout_result)  # type: ignore[misc]
            return dataclasses.replace(trace, status="error", final_output="")

    adapter = _CrashingAdapter()
    runner = _runner(adapter=adapter)

    observed = runner.rollout_group("base-v0", (_task(),), prefix="probe")
    assert observed[0].scorable is False

    workspace = runner.adapter.materialize_candidate("base-v0", "att-crash")
    runner.validate(workspace, _task())
    runner.build_issues([_task()])

    assert runner.traces_for("base-v0", "task-a") == ()
    assert all(
        entry.scorable
        for entries in runner._trace_store.values()
        for entry in entries
    )


def test_traces_read_back_in_rollout_order_per_candidate_task() -> None:
    """Append semantics: same key accumulates in observation order."""
    runner = _runner()
    ws1 = runner.adapter.materialize_candidate("base-v0", "att-order-1")
    runner.validate(ws1, _task("task-a"))
    first_len = len(runner.traces_for(ws1.version, "task-a"))

    ws2 = runner.adapter.materialize_candidate("base-v0", "att-order-2")
    runner.validate(ws2, _task("task-a"))

    entries = runner.traces_for(ws1.version, "task-a")
    assert len(entries) == first_len  # keys are per candidate: ws2 did not mix in
    # The executor nests a per-probe workspace under the candidate version, so
    # traces identify their candidate through the STORE KEY, not through
    # trace.candidate_id (which is the probe workspace id). Pin that property:
    assert all(
        e.trace is not None
        and e.trace.task_id == "task-a"
        and e.trace.candidate_id.startswith(ws1.version)
        for e in entries
    )


# ---------------------------------------------------------------------- #
# Persistence boundary: memory only, pinned against future wiring
# ---------------------------------------------------------------------- #
def test_store_is_memory_only_even_when_storage_is_configured() -> None:
    """Raw traces never reach storage (_persist_attempt invariant), pinned."""
    import tempfile

    writes: list[str] = []

    class _SpyStorage(JSONFileStorage):
        def write_record(self, kind, record_id, payload):  # type: ignore[override]
            writes.append(kind)
            return super().write_record(kind, record_id, payload)

    with tempfile.TemporaryDirectory() as tmp:
        storage = _SpyStorage(Path(tmp))
        runner = _runner(storage=storage)

        outcome = runner.run_attempt([_task("task-a")])
        assert outcome.accepted, outcome.reason

        assert runner.traces_for("base", "task-a"), "store must still be live"

    assert "traces" not in writes and "trace" not in writes, (
        f"trace-bearing records reached storage: {sorted(set(writes))}"
    )
