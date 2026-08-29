"""W1: trace→pool provenance plumbing.

Gap 1 (``docs/plans/editor-tools-live-wiring.md``): the wrapper result carries
``causal_trace_path`` but the executor/adapter discard it, so after an iteration
the loop cannot answer *"where is the parent's tape?"*. ``ScoreProvenance``
records a ``trace_id`` (an id string), never a location.

This file pins, in three layers:

1. ``ExecutionTrace`` carries a ``trace_dir`` (empty = absent, never blanked).
2. ``cuga_adapter.capture_trace`` populates it from the wrapper's
   ``causal_trace_path`` instead of discarding it after loading rich events.
3. ``_record_rollout_score`` copies it into the cell's ``ScoreProvenance`` so
   the pool cell exposes the tape location.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace  # noqa: E402


class _RecordingWrapper:
    """Wrapper double exposing the ``causal_trace_path`` the adapter reads."""

    def __init__(self, causal_trace_path: Path | None = None) -> None:
        self._causal_trace_path = causal_trace_path

    def run_task(self, task_id: str, harness_config):
        result: dict[str, object] = {
            "task_id": task_id,
            "status": "success",
            "final_output": "four",
            "events": [{"event_id": f"{task_id}:started", "kind": "run_started"}],
        }
        if self._causal_trace_path is not None:
            result["causal_trace_path"] = str(self._causal_trace_path)
        return result

    def get_artifacts(self) -> dict[str, str]:
        return {}


def _rich_trace(directory: Path) -> None:
    (directory / "causal-trace.json").write_text(
        '{"run_id": "run-abc", "task_id": "task-1", "status": "success", '
        '"final_output": "four", "events": []}',
        encoding="utf-8",
    )


def test_execution_trace_defaults_trace_dir_to_absent() -> None:
    trace = ExecutionTrace(
        trace_id="t", candidate_id="c", task_id="task",
        events=(), final_output="x", status="success",
    )
    assert trace.trace_dir == ""


# ---------------------------------------------------------------------- #
# Adapter: capture_trace carries the path through
# ---------------------------------------------------------------------- #
def test_capture_trace_carries_causal_trace_path(tmp_path) -> None:
    from agent_evolve.adapters.cuga_adapter import CugaAdapter

    trace_dir = tmp_path / "trace"
    trace_dir.mkdir()
    _rich_trace(trace_dir)
    wrapper = _RecordingWrapper(causal_trace_path=trace_dir)
    adapter = CugaAdapter(wrapper)  # type: ignore[arg-type]
    adapter.register_candidate("base-1", {"skills/s": "body"})
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    result = adapter.run_full_rollout(
        workspace, EvolutionTask(task_id="task-1", input_text="go"), "rollout-1"
    )
    trace = adapter.capture_trace(result)

    assert trace.trace_dir == str(trace_dir)


def test_capture_trace_leaves_trace_dir_absent_without_path() -> None:
    from agent_evolve.adapters.cuga_adapter import CugaAdapter

    wrapper = _RecordingWrapper()  # no causal_trace_path
    adapter = CugaAdapter(wrapper)  # type: ignore[arg-type]
    adapter.register_candidate("base-1", {"skills/s": "body"})
    workspace = adapter.materialize_candidate("base-1", "attempt-1")

    result = adapter.run_full_rollout(
        workspace, EvolutionTask(task_id="task-1", input_text="go"), "rollout-1"
    )
    trace = adapter.capture_trace(result)

    assert trace.trace_dir == ""


# ---------------------------------------------------------------------- #
# Pool: the cell provenance exposes the tape location
# ---------------------------------------------------------------------- #
def test_record_rollout_score_writes_trace_dir_into_provenance() -> None:
    from agent_evolve.core.analyzer import FakeAnalyzerJudge
    from agent_evolve.core.clustering import LexicalEmbedder
    from agent_evolve.core.config import resolve_profile
    from agent_evolve.core.contracts import EvolutionCandidate
    from agent_evolve.core.evaluation import ObservedRollout, RolloutScore
    from agent_evolve.core.fake_editor import FakeEditor
    from agent_evolve.core.orchestrator import SequentialGepaRunner
    from agent_evolve.core.pool import PersistentPool
    from examples.fake_adapter import FakeAdapter

    task = EvolutionTask(
        task_id="task-a", input_text="produce", expected_contract={"expected_substring": "x"},
    )
    adapter = FakeAdapter()
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base", version="base-v0",
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in adapter.artifact_inventory("base-v0")
            },
        )
    )
    runner = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id="mechanism-default",
        seed=0,
    )

    trace = ExecutionTrace(
        trace_id="tr-1", candidate_id="base-v0", task_id="task-a",
        events=(), final_output="out", status="success",
        trace_dir=str(Path("data/traces/run-abc")),
    )
    rollout = ObservedRollout(
        task=task,
        trace=trace,
        score=RolloutScore(
            task_id="task-a", grader_name="g", score=1.0,
            scorable=True, passed=True,
        ),
    )
    runner._record_rollout_score("base", rollout)

    entry = runner.pool.get("base")
    cell = entry.cell("task-a", "mechanism-default")
    assert cell.provenance, "a score must have been recorded"
    assert cell.provenance[-1].trace_dir == str(Path("data/traces/run-abc"))


def test_record_rollout_score_leaves_trace_dir_absent_when_trace_has_none() -> None:
    from agent_evolve.core.analyzer import FakeAnalyzerJudge
    from agent_evolve.core.clustering import LexicalEmbedder
    from agent_evolve.core.config import resolve_profile
    from agent_evolve.core.contracts import EvolutionCandidate
    from agent_evolve.core.evaluation import ObservedRollout, RolloutScore
    from agent_evolve.core.fake_editor import FakeEditor
    from agent_evolve.core.orchestrator import SequentialGepaRunner
    from agent_evolve.core.pool import PersistentPool
    from examples.fake_adapter import FakeAdapter

    task = EvolutionTask(
        task_id="task-a", input_text="produce", expected_contract={"expected_substring": "x"},
    )
    adapter = FakeAdapter()
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base", version="base-v0",
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in adapter.artifact_inventory("base-v0")
            },
        )
    )
    runner = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id="mechanism-default",
        seed=0,
    )

    trace = ExecutionTrace(
        trace_id="tr-1", candidate_id="base-v0", task_id="task-a",
        events=(), final_output="out", status="success",
    )
    rollout = ObservedRollout(
        task=task,
        trace=trace,
        score=RolloutScore(
            task_id="task-a", grader_name="g", score=1.0,
            scorable=True, passed=True,
        ),
    )
    runner._record_rollout_score("base", rollout)

    entry = runner.pool.get("base")
    cell = entry.cell("task-a", "mechanism-default")
    assert cell.provenance[-1].trace_dir == ""
