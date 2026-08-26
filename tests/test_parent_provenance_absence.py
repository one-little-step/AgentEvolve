"""?07 — absent parent diagnosis must be recorded as ABSENCE, not blanks.

The pool's own contract (ScoreProvenance.blame_confidence docstring, pool.py)
says it plainly: ``None`` means *no diagnosis exists* and must never be
replaced by ``0.0``, which would read as a measured zero and make an
undiagnosed score look like a confidently diagnosed one.

``_record_rollout_score`` honored that for ``blame_confidence`` only in the
type widening (✓64) but kept writing the blanks at the call site:
``analyzer_model_id=""`` and ``blame_confidence=0.0`` whenever a scorable
rollout carries no analysis — which is precisely every *passing* rollout,
since the diagnose gate legitimately produces nothing for passes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

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
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"


def _task() -> EvolutionTask:
    return EvolutionTask(
        task_id="task-a",
        input_text=f"find {_TOKEN} and report it",
        expected_contract={"expected_substring": _TOKEN},
    )


def _runner() -> SequentialGepaRunner:
    """Runner whose single rollout PASSES (answer contains the token).

    A pass is never diagnosed (the diagnose gate only analyzes failures), so
    ``rollout.analysis`` is None — exactly the parent-provenance path ?07
    targets. The crashed-rollout file's adapter variant overrides status;
    here we override the answer instead.
    """

    class _PassingAdapter(FakeAdapter):
        def capture_trace(self, rollout_result: object):  # type: ignore[override]
            import dataclasses

            trace = super().capture_trace(rollout_result)  # type: ignore[misc]
            return dataclasses.replace(trace, final_output=f"answer: {_TOKEN}")

    adapter = _PassingAdapter()
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
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
    )


def test_passing_rollout_records_absence_not_blanks() -> None:
    """Scorable pass ⇒ no diagnosis ⇒ provenance fields must be None."""
    runner = _runner()
    observed = runner.rollout_group("base-v0", (_task(),), prefix="probe")

    assert len(observed) == 1
    assert observed[0].scorable is True
    assert observed[0].analysis is None  # diagnose gate: passes are undiagnosed

    runner._record_rollout_score("base", observed[0])

    cell = runner.pool.get("base").cell(_task().task_id, _CLUSTER)
    prov = cell.provenance[-1]
    assert prov.analyzer_model_id is None
    assert prov.blame_confidence is None


@pytest.mark.parametrize("blank", ["", 0.0])
def test_the_old_blank_shapes_are_gone(blank: object) -> None:
    """Pin the exact shapes this fix removes from the writer."""
    runner = _runner()
    observed = runner.rollout_group("base-v0", (_task(),), prefix="probe")
    runner._record_rollout_score("base", observed[0])
    prov = runner.pool.get("base").cell(_task().task_id, _CLUSTER).provenance[-1]
    assert prov.analyzer_model_id != ""
    assert prov.blame_confidence != 0.0
