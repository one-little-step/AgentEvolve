"""Phase 6 B1 (persistent-pool) sequential GEPA experiment runner.

This standalone harness executes the Phase 6 deliverable from
``feedback/from_qwen/qf21.md`` Task B: an N-attempt sequential GEPA loop over a
persistent pool that reports a final champion and pool state, wired to the
``JSONFileStorage`` backend and the deterministic ``FakeAdapter``.

Contrast with the B0/B1 smoke harness
(``examples/run_phase_1_4_smoke.py``): that harness only *evaluates* the initial
candidates. This harness additionally *evolves* them through
:class:`agent_evolve.core.orchestrator.SequentialGepaRunner`:

    observe -> build_issues -> select_issues -> select_parent -> propose_edits
    -> validate -> commit_to_pool

The candidates are deterministic and offline: ``c1`` satisfies task A's token,
``c2`` satisfies task B's token, and the base satisfies neither, so no single
candidate dominates all others and the Pareto frontier is non-trivial.

Embeddings come from the configured provider (``embeddinggemma`` over Ollama by
default) with a deterministic lexical fallback whose use is recorded — never
silent. No LLM, CUGA, network beyond the local embedding service, or
merge/parallel service is used.

The expected-substring tokens are evaluator internals and are **never** written
to storage. Every persisted record is read back and re-checked for token
leakage before ``storage_records_are_redacted`` is set.

Usage
-----
    uv run python examples/run_phase_6_b1.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Make the project importable when run via `uv run python examples/...`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `examples.fake_adapter` is importable

from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    ArtifactEdit,
    CandidateWorkspace,
    EvolutionCandidate,
    EvolutionTask,
)
from agent_evolve.core.clustering import (  # noqa: E402
    LexicalEmbedder,
    MechanismEmbedder,
)
from agent_evolve.core.embeddings import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    FallbackEmbedder,
    OllamaEmbedder,
    build_embedder,
)
from agent_evolve.core.errors import PersistenceSafetyError  # noqa: E402
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
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
TASKS: tuple[EvolutionTask, ...] = (
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
class B1ExperimentResult:
    """Deterministic B1 persistent-pool experiment result for one fixed seed."""

    seed: int
    n_attempts: int
    attempts: int
    accepted: int
    rejected: int
    no_issue: int
    seeded_candidate_count: int
    pool_size: int
    pool_candidate_ids: tuple[str, ...]
    frontier_size: int
    champion_candidate_id: str
    champion_outcome: float
    champion_coverage: float
    champion_stability: float
    champion_regression_risk: float
    champion_aggregate: float
    champion_tie_breaker: str
    embedding_provider: str
    embedding_fallback_reason: str | None
    selection_fallback_reasons: tuple[str | None, ...]
    storage_records_are_redacted: bool


# ---------------------------------------------------------------------- #
# Harness construction (mirrors run_phase_1_4_smoke._build_harness)
# ---------------------------------------------------------------------- #


def _build_harness(
    adapter: FakeAdapter,
) -> list[tuple[EvolutionCandidate, CandidateWorkspace]]:
    """Seed the base plus two deterministic, token-injected candidates."""
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


def _record_initial_score(
    pool: PersistentPool,
    candidate: EvolutionCandidate,
    task: EvolutionTask,
    score: float,
) -> None:
    entry = pool.get(candidate.candidate_id)
    cell = entry.cell(task.task_id, _MECHANISM_CLUSTER_ID)
    pool.record_score(
        candidate.candidate_id,
        score,
        ScoreProvenance(
            task_id=task.task_id,
            mechanism_cluster_id=_MECHANISM_CLUSTER_ID,
            trace_id=f"trace-{candidate.candidate_id}-{task.task_id}",
            rollout_seq=cell.rollout_count,
            analyzer_model_id="fake-analyzer",
            judge_model_id="fake-judge",
            blame_confidence=1.0,
            blame_stability=1.0,
            artifact_versions=dict(candidate.artifact_hashes),
        ),
    )


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
    for record_type in ("manifest", "candidates", "scores", "attempts", "champion"):
        for record in storage.list_records(record_type):
            blob = json.dumps(record, sort_keys=True)
            if any(token in blob for token in (_TOKEN_A, _TOKEN_B)):
                return False
    return True


# ---------------------------------------------------------------------- #
# The experiment
# ---------------------------------------------------------------------- #


def _embedder_provider(embedder: MechanismEmbedder) -> str:
    """Report the concrete embedding provider backing ``embedder``."""
    if isinstance(embedder, LexicalEmbedder):
        return "lexical"
    if isinstance(embedder, OllamaEmbedder):
        return "ollama"
    if isinstance(embedder, FallbackEmbedder):
        return _embedder_provider(embedder.primary)
    return type(embedder).__name__


def _embedder_fallback_reason(embedder: MechanismEmbedder) -> str | None:
    return getattr(embedder, "fallback_reason", None) if isinstance(embedder, FallbackEmbedder) else None


def run_b1_experiment(
    seed: int,
    storage_root: Path,
    n_attempts: int,
    embedder: MechanismEmbedder | None = None,
) -> B1ExperimentResult:
    """Run one fixed-budget, deterministic B1 GEPA experiment for ``seed``.

    No LLM, CUGA, network beyond the local embedding service, or merge/parallel
    service is used. The fake adapter and pool are rebuilt per call so the run
    is reproducible regardless of storage state.

    ``embedder`` overrides the config-built embedding provider. Tests pass a
    :class:`LexicalEmbedder` to stay offline and fast; the smoke driver omits it
    to exercise the live ``embeddinggemma`` service.
    """
    if isinstance(n_attempts, bool) or not isinstance(n_attempts, int):
        raise ValueError("n_attempts must be a positive integer")
    if n_attempts < 1:
        raise ValueError("n_attempts must be a positive integer")

    config = resolve_profile(
        "research_sequential",
        environ={
            "OLLAMA_EMBEDDING_URL": os.environ.get(
                "OLLAMA_EMBEDDING_URL", "http://localhost:11434"
            ),
            "OLLAMA_EMBEDDING_MODEL": os.environ.get(
                "OLLAMA_EMBEDDING_MODEL", "embeddinggemma"
            ),
        },
        seed=seed,
    )

    adapter = FakeAdapter()
    tasks = TASKS
    harness = _build_harness(adapter)

    # Seed the persistent pool with base + every candidate and record one
    # comparable score cell per candidate per task (B1: retain all, evaluate
    # once each to preserve RHO-scale cost).
    base_candidate = harness[0][0]
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(base_candidate)
    for candidate, _ in harness[1:]:
        pool.add_candidate(candidate, origin_attempt_ids=candidate.attempt_ids)

    initial_scores: dict[str, dict[str, float]] = {}
    for candidate, workspace in harness:
        per_task: dict[str, float] = {}
        for task in tasks:
            token = str(task.expected_contract["expected_substring"])
            score = _score_rollout(
                adapter,
                workspace,
                task,
                f"rollout-{candidate.candidate_id}-{task.task_id}",
                token,
            )
            per_task[task.task_id] = score
            _record_initial_score(pool, candidate, task, score)
        initial_scores[candidate.candidate_id] = per_task

    # Build the embedding provider from the resolved config (records fallback),
    # unless the caller supplied an override.
    if embedder is None:
        embedder = build_embedder(config.embedding, dim=DEFAULT_EMBEDDING_DIM)

    # Run the sequential GEPA loop.
    runner = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=embedder,
        storage=None,  # storage wiring happens below so we control redaction
        config=config,
        mechanism_cluster_id=_MECHANISM_CLUSTER_ID,
        seed=seed,
    )
    run_result = runner.run(tasks, n_attempts=n_attempts)

    # Select the champion and extract the full manifest.
    champion = run_result.champion
    if champion is None:
        # No eligible candidate can only mean the pool carried no evidence;
        # with a seeded base + candidates this is an invariant violation.
        raise AssertionError("no eligible champion in a seeded B1 pool")

    # Persist redacted evidence and verify no evaluator token leaked.
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
            for candidate_id, per_task in initial_scores.items()
            for task_id, score in sorted(per_task.items())
        ]
        storage.write_record(
            "scores", "run", {"rows": score_rows, "rollout_count": len(score_rows)}
        )
        for attempt in run_result.attempts:
            storage.write_record(
                "attempts",
                attempt.attempt_id,
                {
                    "attempt_id": attempt.attempt_id,
                    "issue_id": attempt.issue_id,
                    "parent_candidate_id": attempt.parent_candidate_id,
                    "result_candidate_id": attempt.result_candidate_id,
                    "status": attempt.status.value,
                    "accepted": attempt.accepted,
                    "weighted_net_gain": attempt.weighted_net_gain,
                    "reason": attempt.reason,
                    "artifact_ids": list(attempt.artifact_ids),
                    "mechanism_cluster_id": _MECHANISM_CLUSTER_ID,
                },
            )
        storage.write_record(
            "champion",
            "run",
            {
                "candidate_id": champion.candidate_id,
                "outcome": champion.outcome,
                "coverage": champion.coverage,
                "stability": champion.stability,
                "regression_risk": champion.regression_risk,
                "aggregate": champion.aggregate,
                "tie_breaker": champion.tie_breaker,
                "disqualifications": list(champion.disqualifications),
            },
        )
        storage_records_are_redacted = _verify_no_evaluator_tokens(storage)
    except PersistenceSafetyError:
        storage_records_are_redacted = False

    return B1ExperimentResult(
        seed=seed,
        n_attempts=n_attempts,
        attempts=run_result.attempts_run,
        accepted=run_result.accepted_count,
        rejected=run_result.rejected_count,
        no_issue=run_result.no_issue_count,
        seeded_candidate_count=len(harness),
        pool_size=run_result.pool_size,
        pool_candidate_ids=pool.candidate_ids(),
        frontier_size=len(run_result.pareto_frontier),
        champion_candidate_id=champion.candidate_id,
        champion_outcome=champion.outcome,
        champion_coverage=champion.coverage,
        champion_stability=champion.stability,
        champion_regression_risk=champion.regression_risk,
        champion_aggregate=champion.aggregate,
        champion_tie_breaker=champion.tie_breaker,
        embedding_provider=_embedder_provider(embedder),
        embedding_fallback_reason=_embedder_fallback_reason(embedder),
        selection_fallback_reasons=tuple(a.fallback_reason for a in run_result.attempts),
        storage_records_are_redacted=storage_records_are_redacted,
    )


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #


def _print_report(result: B1ExperimentResult) -> None:
    print(
        f"seed={result.seed} attempts={result.attempts} "
        f"accepted={result.accepted} rejected={result.rejected} "
        f"no_issue={result.no_issue} "
        f"pool={result.pool_size} frontier={result.frontier_size} "
        f"champion={result.champion_candidate_id} "
        f"aggregate={result.champion_aggregate:.4f} "
        f"coverage={result.champion_coverage:.2f} "
        f"embedding={result.embedding_provider} "
        f"embedding_fallback={result.embedding_fallback_reason} "
        f"redacted={result.storage_records_are_redacted}"
    )


def main(storage_root: Path | None = None) -> int:
    base_root = (
        Path(storage_root)
        if storage_root is not None
        else Path(tempfile.gettempdir()) / "agent_evolve_phase_6_b1"
    )
    for seed in (0, 1, 2):
        result = run_b1_experiment(
            seed=seed, storage_root=base_root / f"seed-{seed}", n_attempts=4
        )
        _print_report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
