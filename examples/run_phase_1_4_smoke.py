"""Phase-gate B0/B1 smoke harness over the deterministic FakeAdapter.

This standalone harness contrasts the two initial-pool strategies the research
core must distinguish before any H1 claim is made:

* **B0 (best-of-N):** evaluate every initial candidate on every fixed task,
  keep only the single highest-scoring candidate (``b0_retained == 1``).
* **B1 (persistent pool):** retain the base plus every initial candidate,
  record comparable score evidence, and derive a Pareto frontier that keeps
  more than one non-dominated candidate.

The candidates are deterministic and offline: ``c1`` satisfies task A's token,
``c2`` satisfies task B's token, and the base satisfies neither. No single
candidate dominates all others, so the Pareto frontier is non-trivial.

The expected-substring tokens are evaluator internals and are **never** written
to storage. Every persisted record (manifest, candidates, scores) is read back
and re-checked for token leakage before ``storage_records_are_redacted`` is set.
No LLM, CUGA, network, or merge/parallel service is used.

Usage
-----
    uv run python examples/run_phase_1_4_smoke.py
"""
from __future__ import annotations

import json
import statistics
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Make the project importable when run via `uv run python examples/...`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `examples.fake_adapter` is importable

from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    ArtifactEdit,
    CandidateWorkspace,
    EvolutionCandidate,
    EvolutionTask,
)
from agent_evolve.core.errors import PersistenceSafetyError  # noqa: E402
from agent_evolve.core.pool import PersistentPool, ScoreProvenance  # noqa: E402
from agent_evolve.core.storage import JSONFileStorage  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

# Evaluator-internal expected-substring tokens. These are NOT secrets in the
# credential sense, but they ARE evaluator internals: the core must never
# persist them. Keep them module-private and never write task.expected_contract.
_TOKEN_A = "token-a"
_TOKEN_B = "token-b"
_TOKEN_BY_CANDIDATE = {"c1": _TOKEN_A, "c2": _TOKEN_B}

# A single fixed mechanism cluster id makes every score cell comparable across
# candidates and tasks, which is what the persistent pool's Pareto frontier
# requires.
_MECHANISM_CLUSTER_ID = "mechanism-default"

# A fixed task coreset. The task_id strings deliberately avoid the token text.
_TASKS: tuple[EvolutionTask, ...] = (
    EvolutionTask(
        task_id="task-a",
        input_text="produce the A capability",
        expected_contract={"expected_substring": _TOKEN_A},
    ),
    EvolutionTask(
        task_id="task-b",
        input_text="produce the B capability",
        expected_contract={"expected_substring": _TOKEN_B},
    ),
)


@dataclass(frozen=True, slots=True)
class ComparisonOutcome:
    """Deterministic B0/B1 comparison result for one fixed seed."""

    b0_retained_candidate_count: int
    b1_retained_candidate_count: int
    b0_best_score: float
    b1_frontier_size: int
    comparison_coverage: float
    storage_records_are_redacted: bool
    seed: int
    accepted: int
    rejected: int
    no_op: int
    exhausted: int
    rollouts: int
    analyzer_judge_calls: int
    editor_calls: int
    validation_calls: int
    embedding_calls: int
    attempts: int


# ---------------------------------------------------------------------- #
# Harness construction
# ---------------------------------------------------------------------- #


def _build_harness(
    adapter: FakeAdapter,
) -> list[tuple[EvolutionCandidate, CandidateWorkspace]]:
    """Build the base plus two deterministic, token-injected candidates.

    Returns (candidate, workspace) pairs. The workspace is retained because
    ``run_full_rollout`` reads staging from the adapter's attempt id.
    """
    base_hashes = {
        d.artifact_id: d.version_hash for d in adapter.artifact_inventory("base-v0")
    }
    base = EvolutionCandidate(
        candidate_id="base",
        version="base-v0",
        artifact_hashes=base_hashes,
    )
    base_ws = adapter.materialize_candidate("base-v0", "attempt-base")

    harness: list[tuple[EvolutionCandidate, CandidateWorkspace]] = [(base, base_ws)]

    for candidate_id, token in _TOKEN_BY_CANDIDATE.items():
        ws = adapter.materialize_candidate("base-v0", f"attempt-{candidate_id}")
        adapter.apply_structured_edits(
            ws,
            (
                ArtifactEdit(
                    artifact_id="skills/retrieval",
                    operation="replace",
                    payload={"content": f"retrieve(query): use {token} for top_k docs"},
                ),
            ),
        )
        hashes = {
            d.artifact_id: d.version_hash
            for d in adapter.artifact_inventory(ws.version)
        }
        candidate = EvolutionCandidate(
            candidate_id=candidate_id,
            version=ws.version,
            artifact_hashes=hashes,
            attempt_ids=(f"attempt-{candidate_id}",),
        )
        harness.append((candidate, ws))

    return harness


def _score_rollout(
    adapter: FakeAdapter,
    workspace: CandidateWorkspace,
    task: EvolutionTask,
    rollout_id: str,
    token: str,
) -> float:
    result = adapter.run_full_rollout(workspace, task, rollout_id)
    trace = adapter.capture_trace(result)
    return 1.0 if token in trace.final_output else 0.0


# ---------------------------------------------------------------------- #
# Redaction verification
# ---------------------------------------------------------------------- #


def _verify_no_evaluator_tokens(storage: JSONFileStorage) -> bool:
    for record_type in ("manifest", "candidates", "scores"):
        for record in storage.list_records(record_type):
            blob = json.dumps(record, sort_keys=True)
            if any(token in blob for token in (_TOKEN_A, _TOKEN_B)):
                return False
    return True


# ---------------------------------------------------------------------- #
# The comparison harness
# ---------------------------------------------------------------------- #


def run_fixed_budget_comparison(seed: int, storage_root: Path) -> ComparisonOutcome:
    """Run one fixed-budget, deterministic B0-vs-B1 comparison for ``seed``.

    No LLM, CUGA, network, or merge/parallel service is used. The fake adapter
    is rebuilt per call so the run is reproducible regardless of storage state.
    """
    config = resolve_profile(
        "minimal",
        environ={"OLLAMA_EMBEDDING_MODEL": "embeddinggemma"},
        seed=seed,
    )
    adapter = FakeAdapter()
    tasks = _TASKS
    harness = _build_harness(adapter)

    # Evaluate every candidate on every task exactly once.
    scores: dict[str, dict[str, float]] = {
        candidate.candidate_id: {} for candidate, _ in harness
    }
    rollouts = 0
    for candidate, workspace in harness:
        for task in tasks:
            token = str(task.expected_contract["expected_substring"])
            score = _score_rollout(
                adapter,
                workspace,
                task,
                f"rollout-{candidate.candidate_id}-{task.task_id}",
                token,
            )
            scores[candidate.candidate_id][task.task_id] = score
            rollouts += 1

    # ---- B0: best-of-N keeps a single winner ----
    means = {
        candidate_id: (sum(per_task.values()) / len(per_task) if per_task else 0.0)
        for candidate_id, per_task in scores.items()
    }
    ranked = sorted(means.items(), key=lambda item: (-item[1], item[0]))
    best_candidate_id = ranked[0][0]
    b0_best_score = means[best_candidate_id]
    b0_retained_candidate_count = len(ranked[:1])

    # ---- B1: persistent pool keeps base + every candidate ----
    base_candidate = harness[0][0]
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(base_candidate)
    for candidate, _ in harness[1:]:
        pool.add_candidate(candidate, origin_attempt_ids=candidate.attempt_ids)

    for candidate, _ in harness:
        for task in tasks:
            score = scores[candidate.candidate_id][task.task_id]
            provenance = ScoreProvenance(
                task_id=task.task_id,
                mechanism_cluster_id=_MECHANISM_CLUSTER_ID,
                trace_id=f"trace-{candidate.candidate_id}-{task.task_id}",
                rollout_seq=0,
                analyzer_model_id="fake-analyzer",
                judge_model_id="fake-judge",
                blame_confidence=1.0,
                blame_stability=1.0,
                artifact_versions=dict(candidate.artifact_hashes),
            )
            pool.record_score(candidate.candidate_id, score, provenance)

    frontier = pool.pareto_frontier()
    b1_frontier_size = len(frontier)
    b1_retained_candidate_count = len(pool)

    # ---- Persist redacted evidence and verify no evaluator token leaked ----
    storage = JSONFileStorage(Path(storage_root))
    storage_records_are_redacted = False
    try:
        storage.write_record("manifest", "run", config.manifest_payload())
        for candidate, _ in harness:
            storage.write_record(
                "candidates",
                candidate.candidate_id,
                {
                    "candidate_id": candidate.candidate_id,
                    "version": candidate.version,
                    "artifact_hashes": dict(candidate.artifact_hashes),
                    "is_base": candidate.candidate_id == "base",
                    "attempt_ids": list(candidate.attempt_ids),
                },
            )
        score_rows = [
            {
                "candidate_id": candidate_id,
                "task_id": task_id,
                "mechanism_cluster_id": _MECHANISM_CLUSTER_ID,
                "score": score,
                "rollout_seq": 0,
            }
            for candidate_id, per_task in scores.items()
            for task_id, score in sorted(per_task.items())
        ]
        storage.write_record(
            "scores",
            "run",
            {"rows": score_rows, "rollout_count": rollouts},
        )
        storage_records_are_redacted = _verify_no_evaluator_tokens(storage)
    except PersistenceSafetyError:
        storage_records_are_redacted = False

    # Every candidate was evaluated on every task, so coverage is complete.
    comparison_coverage = 1.0

    return ComparisonOutcome(
        b0_retained_candidate_count=b0_retained_candidate_count,
        b1_retained_candidate_count=b1_retained_candidate_count,
        b0_best_score=b0_best_score,
        b1_frontier_size=b1_frontier_size,
        comparison_coverage=comparison_coverage,
        storage_records_are_redacted=storage_records_are_redacted,
        seed=seed,
        accepted=0,
        rejected=0,
        no_op=0,
        exhausted=0,
        rollouts=rollouts,
        analyzer_judge_calls=0,
        editor_calls=0,
        validation_calls=0,
        embedding_calls=0,
        attempts=0,
    )


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #


def _print_report(outcome: ComparisonOutcome) -> None:
    print(
        f"seed={outcome.seed} "
        f"b0_retained={outcome.b0_retained_candidate_count} "
        f"b0_best_score={outcome.b0_best_score:.4f} "
        f"b1_retained={outcome.b1_retained_candidate_count} "
        f"b1_frontier_size={outcome.b1_frontier_size} "
        f"coverage={outcome.comparison_coverage:.2f} "
        f"redacted={outcome.storage_records_are_redacted} "
        f"rollouts={outcome.rollouts} "
        f"attempts={outcome.attempts} "
        f"accepted={outcome.accepted} rejected={outcome.rejected} "
        f"no_op={outcome.no_op} exhausted={outcome.exhausted} "
        f"analyzer_judge={outcome.analyzer_judge_calls} "
        f"editor={outcome.editor_calls} "
        f"validation={outcome.validation_calls} "
        f"embedding={outcome.embedding_calls}"
    )


def main(storage_root: Path | None = None) -> int:
    base_root = (
        Path(storage_root)
        if storage_root is not None
        else Path(tempfile.gettempdir()) / "agent_evolve_phase_1_4_smoke"
    )
    outcomes = [run_fixed_budget_comparison(seed=s, storage_root=base_root / f"seed-{s}")
                for s in (0, 1, 2)]

    for outcome in outcomes:
        _print_report(outcome)

    b0_scores = [o.b0_best_score for o in outcomes]
    mean = statistics.fmean(b0_scores)
    dispersion = statistics.pstdev(b0_scores)
    print(
        f"cross-seed b0_best_score: mean={mean:.4f} "
        f"stdev={dispersion:.4f} (n={len(outcomes)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
