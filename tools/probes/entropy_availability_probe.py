"""?06: measure EntropyAvailabilityReport on a multi-attempt offline loop.

Pre-SV-14 measurement (recorded in EntropyAvailabilityReport's own docstring):
a 4-attempt offline run reported ``no_analysis=3`` -- three cells existed but
offspring evidence never reached the tracker because ``validate`` discarded
child analyses. SV-14 wired commit-time filing; this probe re-measures.

Offline and free: FakeAdapter/FakeEditor/FakeAnalyzerJudge, no network.

What it reports:
* per-attempt acceptance trail (candidate ids),
* the aggregate EntropyAvailabilityReport (cells available/unavailable,
  reason tally, fallback_rate with its None-vs-zero distinction honoured),
* raw tracker cells for depth.

Exit code is nonzero only when the wiring looks DEAD (no cells AND no
reason categories at all -- nothing filed anywhere), or when the report's
arithmetic contradicts itself. A fully-unavailable result is a VALID
measurement and exits zero with the numbers printed.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import contract_score  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import EvolutionCandidate, EvolutionTask  # noqa: E402
from agent_evolve.core.entropy import EntropyTracker  # noqa: E402
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"
_ATTEMPTS = 4

# Stable fault/strength pair whose cosine clears the join band on this
# embedder, so the child JOINS the base fault's cluster instead of opening a
# parallel cell -- the crown case for cross-candidate comparability.
_FAULT_TEXT = (
    "the planner hit a retrieval timeout so the context held no documents "
    "and the model answered from memory"
)
_STRENGTH_TEXT = (
    "the planner avoided a retrieval timeout because the context held fresh "
    "documents so the model answered with grounded citations"
)


class _StableFaultAnalyzer:
    """FakeAnalyzerJudge with a FIXED failure mechanism (trace ids excluded)."""

    analyzer_model_id = "probe-analyzer"
    judge_model_id = "probe-judge"

    def analyze(self, task, trace):
        from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis

        if contract_score(task, trace) == 1.0:
            return CausalAnalysis(mechanism="none", severity=0.0, score=1.0,
                                  blame_graph=BlameGraph(nodes=()))
        return CausalAnalysis(
            mechanism=_FAULT_TEXT,
            severity=1.0,
            score=0.0,
            blame_graph=BlameGraph(nodes=(BlameNode(actor_id="planner", blame=1.0,
                                                    artifacts=("skills/retrieval",)),)),
            analyzer_model_id=self.analyzer_model_id,
            judge_model_id=self.judge_model_id,
        )


class _StrengthJudge:
    """Minimal positivity judge: one observed strength per passing rollout."""

    analyzer_model_id = "probe-positivity"

    def analyze_success(self, task, trace):
        from agent_evolve.core.blame import BlameGraph, BlameNode, CausalFinding

        finding = CausalFinding(
            verdict_id=f"strength-{trace.trace_id}",
            candidate_id=trace.candidate_id,
            task_id=task.task_id,
            trace_id=trace.trace_id,
            valence=-1,
            status="observed",
            mechanism_description=_STRENGTH_TEXT,
            mechanism_cluster_id="mechanism-cluster-unassigned",
            severity=0.9,
            confidence=0.9,
            blame_graph=BlameGraph(
                nodes=(BlameNode(actor_id="planner", blame=1.0,
                                 artifacts=("skills/retrieval",)),)
            ),
            evidence_refs=(trace.trace_id, "skills/retrieval"),
            rationale="probe",
        )
        return (finding,)


def _task(task_id: str) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": _TOKEN},
    )


def _run_loop(label: str, tracker: EntropyTracker):
    adapter = FakeAdapter()
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
    runner = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=_StableFaultAnalyzer(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        config=resolve_profile(
            "research_sequential",
            seed=0,
            entropy_min_comparable_candidates=tracker.min_comparable_candidates,
            entropy_min_rollouts_per_candidate=tracker.min_rollouts_per_candidate,
        ),
        mechanism_cluster_id=_CLUSTER,
        seed=0,
        entropy=tracker,
        positivity_judge=_StrengthJudge(),
    )

    tasks = [_task("task-a"), _task("task-b")]
    accepted_ids: list[str] = []
    print(f"===== configuration {label}")
    for i in range(1, _ATTEMPTS + 1):
        outcome = runner.run_attempt(tasks)
        print(f"attempt {i}: accepted={outcome.accepted} "
              f"candidate={outcome.result_candidate_id!r} reason={outcome.reason}")
        if outcome.accepted:
            accepted_ids.append(outcome.result_candidate_id)

    report = runner.entropy_availability()
    print("---")
    print("aggregate:", report.line())
    print(f"cells_available={report.cells_available} "
          f"cells_unavailable={report.cells_unavailable} "
          f"cells_total={report.cells_total}")
    print("reasons:", dict(sorted(report.reasons.items())))
    rate = report.fallback_rate
    print(f"fallback_rate={rate!r} (None means no cells ever observed)")
    print(f"entropy_never_available={report.entropy_never_available}")

    cells = sorted(runner.entropy.all_cells(), key=lambda k: (k.task_id, k.mechanism_cluster_id))
    print(f"raw tracker cells ({len(cells)}):")
    for key in cells:
        h = runner.entropy.entropy(key.task_id, key.mechanism_cluster_id)
        print(f"  {key.task_id} / {key.mechanism_cluster_id}: H={h!r}")

    # --- contract sanity (hard failures only) -------------------------- #
    recomputed = (
        None if report.cells_total == 0
        else report.cells_unavailable / report.cells_total
    )
    assert recomputed == rate, f"fallback_rate arithmetic broken: {rate!r} vs {recomputed!r}"
    assert report.entropy_never_available == (
        report.cells_total > 0 and report.cells_available == 0
    ), "entropy_never_available inconsistent with cell counts"

    if report.cells_total == 0 and not report.reasons:
        print("DEAD WIRING: no cells and no unavailability categories at all")
        raise SystemExit(1)

    print(f"{label} VERDICT: {report.cells_available} available / "
          f"{report.cells_unavailable} unavailable across {len(accepted_ids)} accepted attempts")
    return report


def main() -> int:
    # A: production defaults (3 comparable candidates x 2 rollouts each).
    _run_loop("default-floors", EntropyTracker())
    # B: floors matched to this loop's evidence shape (base + 1 child x 1
    # rollout). Demonstrates the SAME filing clearing floors when the floor
    # matches the evidence actually collectable at this scale -- the numbers
    # are then a measured availability, not a floor artefact.
    relaxed = EntropyTracker(min_comparable_candidates=2, min_rollouts_per_candidate=1)
    _run_loop("relaxed-floors", relaxed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
