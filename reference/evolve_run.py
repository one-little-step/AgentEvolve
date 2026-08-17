"""Hardcoded progressive runner for offline Gaia wisdom evolution.

Edit the constants in the configuration block, then run:
    uv run python dataset/evolve_run.py
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from agent.gaia_lg_react.evolution.round import EvolutionRound
from agent.gaia_lg_react.evolution.gaia_adapter import GaiaEvolutionAdapter, GaiaEvolutionLLM
from agent.evolution_core.history import EditHistoryStore
from agent.evolution_core.population import PopulationEvolution
from agent.evolution_core.contracts import NormalizedTrajectory, RolloutLimits
from agent.gaia_lg_react.llm import LiteLLMClient


# ========================= EDIT THESE ======================================

# Historical Gaia runs used as the evolution corpus. Every configured source
# is loaded on every round, then their task records are combined before
# coreset selection. Paths are run names relative to DATASET_RUNS_ROOT.
SOURCE_RUNS = ["gaia_l3_validation_20260726_231430"]

# Wisdom version used as the parent in round 1. "base" is the vanilla bundle.
INITIAL_HARNESS = "base"

# Accepted candidates are materialized as <prefix>-1, <prefix>-2, etc.
# Multiple rounds form a progressive chain:
#   base -> rho-gaia-1 -> rho-gaia-2 -> rho-gaia-3
TARGET_HARNESS_NAME_PREFIX = "test2-gaia10b-L3_luna1-base-rho"

# Number of sequential evolution rounds. A rejected round stops the chain
# unless EXPERIMENTAL_PROMOTE_CANDIDATE is enabled.
ROUND_COUNT = 5

# Model used for agent rollouts, diagnosis, candidate optimization, and,
# unless JUDGE_MODEL is set, pairwise preference judging.
MODEL = "azure/gpt-5.6-luna"

# Optional separate model for pairwise judging. None intentionally means
# "use MODEL", so the current configuration uses gpt-oss-120b-a100 as judge.
JUDGE_MODEL = None

# Coreset selection strategy. Available strategies:
#   dpp        difficulty-weighted diversity (recommended/default)
#   random     seeded random baseline
#   difficulty highest-quality/difficulty records first
#   coverage   diversity/coverage-oriented selection
SELECTOR = "dpp"

# DPP difficulty/diversity tradeoff in [0, 1].
#   0.0 = pure diversity; 1.0 = strongest difficulty weighting.
# Values around 0.7 provide a balanced selection.
THETA = 0.7

# Minimum normalized quality used by DPP. Raise it to prevent very low-signal
# records from receiving near-zero quality; must be greater than zero.
SCORE_FLOOR = 0.1

# Seed for reproducible stochastic selectors such as SELECTOR="random".
# DPP, difficulty, and coverage are deterministic for the same input.
SEED = 0

# Number of independently generated candidate wisdom bundles per round.
# More candidates explore more possible edits but require more model calls.
CANDIDATE_COUNT = 1

# RHO-GEPA is opt-in. The default retains the established RHO-only chain.
GEPA_ENABLED = False
ELITE_COUNT = 3
OFFSPRING_COUNT = 6
MERGE_OFFSPRING_COUNT = 1
MODULE_JUDGE_MODEL = None
EDIT_HISTORY_RETRIEVAL_ENABLED = True
EDIT_HISTORY_SEMANTIC_ENABLED = True

# Explicit RHO terminology for the number of optimization samples. It must
# equal CANDIDATE_COUNT in this implementation.
OPTIMIZE_SAMPLES = CANDIDATE_COUNT

# Number of historical tasks selected for expensive diagnosis and rollouts.
# Set no higher than the number of valid tasks available in SOURCE_RUNS.
CORESET_SIZE = 6

# Number of parent and candidate rollouts per selected task. Higher values
# improve robustness but multiply rollout and judging cost.
GROUP_ROLLOUTS_PER_TASK = 3

# Maximum concurrently admitted task groups in a fresh evaluation phase.
MAX_RERUN_WORKERS = 10

# Maximum simultaneous repeated rollouts for one admitted task group.
MAX_ROLLOUT_WORKERS = 3

# Hard cap for every simultaneous fresh Gaia run in the active phase.
GLOBAL_MAX_WORKERS = 12

# Offline summary reconstruction and embedding preparation only.
MAX_TRAJECTORY_WORKERS = 10
TRAJECTORY_SUMMARY_CACHE_DIR = None
TRAJECTORY_EMBEDDING_CACHE_DIR = None

# Candidate mean pairwise preference must be strictly greater than this value
# to be accepted. 0.0 rejects neutral/no-op candidates by default.
ACCEPTANCE_THRESHOLD = 0.0

# Intended evaluation timeout metadata in seconds. The current LiteLLM client
# does not enforce a separate evolution-stage timeout; this is persisted for
# reproducibility and future client support.
EVALUATION_TIMEOUT_SECONDS = 300.0

# Response cache mode. Only "off" is currently implemented by this runner.
# The other RHO modes are rejected instead of being silently ignored.
CACHE_MODE = "off"

# Optional cache directory reserved for future response-cache support.
CACHE_DIR = None

# Safety gate for wisdom edits. Keep True to generate modified candidates;
# False allows a no-edit/copy path but cannot produce useful evolution edits.
ENABLE_WISDOM_EDITING = True

# Experimental promotion mode. Disabled by default: normal rounds only create
# a new harness when a candidate beats ACCEPTANCE_THRESHOLD. When enabled, the
# highest-scoring current-round candidate is promoted even for a non-positive
# score; if every pairwise judgment is unavailable, candidate_0 is promoted.
# All candidate bundles and rollouts are archived under the round artifacts.
EXPERIMENTAL_PROMOTE_CANDIDATE = True

# Repository and artifact locations. These normally do not need changing.
REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_RUNS_ROOT = REPO_ROOT / "dataset" / "runs_dataset_4rho"
WISDOM_ROOT = REPO_ROOT / "policies" / "evolved_context"
EVOLUTION_ARTIFACT_ROOT = DATASET_RUNS_ROOT / "evolution"
# ==========================================================================


def build_round_plan() -> list[tuple[int, str, str]]:
    """Return (round number, parent harness, target harness) entries."""
    parent = INITIAL_HARNESS
    plan: list[tuple[int, str, str]] = []
    for round_number in range(1, ROUND_COUNT + 1):
        target = f"{TARGET_HARNESS_NAME_PREFIX}-{round_number}"
        plan.append((round_number, parent, target))
        parent = target
    return plan


def _load_gepa_tasks() -> tuple[NormalizedTrajectory, ...]:
    """Load the configured offline corpus as the core's neutral task format."""
    from agent.gaia_lg_react.evolution.selection import select_coreset
    from agent.gaia_lg_react.evolution.trajectory_loader import TrajectoryRunLoader

    loader = TrajectoryRunLoader(DATASET_RUNS_ROOT)
    records = []
    for source in SOURCE_RUNS:
        loaded, _ = loader.load(source)
        records.extend(record for record in loaded if record.task_id)
    selected = select_coreset(records, CORESET_SIZE, seed=SEED, selector=SELECTOR, theta=THETA, score_floor=SCORE_FLOOR).selected_ids
    return tuple(
        NormalizedTrajectory(record.trajectory_id or record.task_id or "unknown", record.query, record.final_answer, record.status or ("success" if record.correct else "failure"), tuple(event for event in record.events if isinstance(event, dict)), record.source_paths)
        for record in records if (record.trajectory_id or record.task_id) in selected
    )


def main() -> int:
    if not SOURCE_RUNS:
        raise ValueError("SOURCE_RUNS must contain at least one run")
    if OPTIMIZE_SAMPLES != CANDIDATE_COUNT:
        raise ValueError("OPTIMIZE_SAMPLES must equal CANDIDATE_COUNT")
    if GEPA_ENABLED and ELITE_COUNT < 1:
        raise ValueError("ELITE_COUNT must be >= 1")
    if GEPA_ENABLED and OFFSPRING_COUNT < ELITE_COUNT:
        raise ValueError("OFFSPRING_COUNT must be >= ELITE_COUNT")
    if GEPA_ENABLED and not 0 <= MERGE_OFFSPRING_COUNT <= OFFSPRING_COUNT:
        raise ValueError("MERGE_OFFSPRING_COUNT must be between 0 and OFFSPRING_COUNT")
    if MAX_RERUN_WORKERS < 1:
        raise ValueError("MAX_RERUN_WORKERS must be >= 1")
    if MAX_ROLLOUT_WORKERS < 1:
        raise ValueError("MAX_ROLLOUT_WORKERS must be >= 1")
    if GLOBAL_MAX_WORKERS < 1:
        raise ValueError("GLOBAL_MAX_WORKERS must be >= 1")
    if MAX_TRAJECTORY_WORKERS < 1:
        raise ValueError("MAX_TRAJECTORY_WORKERS must be >= 1")
    if GLOBAL_MAX_WORKERS > MAX_RERUN_WORKERS * MAX_ROLLOUT_WORKERS:
        raise ValueError(
            "GLOBAL_MAX_WORKERS must be <= MAX_RERUN_WORKERS * MAX_ROLLOUT_WORKERS"
        )
    if CACHE_MODE != "off":
        raise ValueError("CACHE_MODE must be 'off'; response caching is not implemented")

    model = MODEL or os.environ.get("GAIA_MODEL")
    llm = LiteLLMClient(JUDGE_MODEL or model)
    runner = EvolutionRound(
        REPO_ROOT,
        llm=llm,
        model=model,
        dataset_runs_root=DATASET_RUNS_ROOT,
        wisdom_root=WISDOM_ROOT,
        target_root=WISDOM_ROOT,
    )

    if GEPA_ENABLED:
        from agent.gaia_lg_react.config import load_config
        from agent.gaia_lg_react.model_adapters.resolver import resolve_embedding_provider

        config = load_config()
        embedder = resolve_embedding_provider(config) if EDIT_HISTORY_SEMANTIC_ENABLED else None
        adapter = GaiaEvolutionAdapter(runner)
        population = PopulationEvolution(
            adapter,
            EVOLUTION_ARTIFACT_ROOT,
            version_root=WISDOM_ROOT,
            llm=GaiaEvolutionLLM(llm),
            history=EditHistoryStore(EVOLUTION_ARTIFACT_ROOT, adapter.agent_name, retrieval_enabled=EDIT_HISTORY_RETRIEVAL_ENABLED, semantic_enabled=EDIT_HISTORY_SEMANTIC_ENABLED, embedder=embedder),
            rollout_count=GROUP_ROLLOUTS_PER_TASK,
            limits=RolloutLimits(MAX_RERUN_WORKERS, MAX_ROLLOUT_WORKERS, GLOBAL_MAX_WORKERS),
        )
        tasks = _load_gepa_tasks()
        if not tasks:
            raise ValueError("coreset selection produced no tasks")
        for generation in range(1, ROUND_COUNT + 1):
            result = population.run_generation(initial_version=INITIAL_HARNESS, prefix=TARGET_HARNESS_NAME_PREFIX, generation=generation, elite_count=ELITE_COUNT, offspring_count=OFFSPRING_COUNT, crossover_count=MERGE_OFFSPRING_COUNT, tasks=tasks)
            print(f"GEPA generation {generation}: {result.round_dir}")
        return 0

    for round_number, parent, target in build_round_plan():
        print(f"Round {round_number}: {parent} -> {target}")
        result = runner.run(
            run_name=SOURCE_RUNS[0],
            source_runs=SOURCE_RUNS,
            parent_version=parent,
            target_version=target,
            candidate_count=CANDIDATE_COUNT,
            coreset_size=CORESET_SIZE,
            evolution_tools_enabled=ENABLE_WISDOM_EDITING,
            selector=SELECTOR,
            theta=THETA,
            score_floor=SCORE_FLOOR,
            seed=SEED,
            acceptance_threshold=ACCEPTANCE_THRESHOLD,
            judge_model=JUDGE_MODEL,
            group_rollouts_per_task=GROUP_ROLLOUTS_PER_TASK,
            max_rerun_workers=MAX_RERUN_WORKERS,
            max_rollout_workers=MAX_ROLLOUT_WORKERS,
            global_max_workers=GLOBAL_MAX_WORKERS,
            max_trajectory_workers=MAX_TRAJECTORY_WORKERS,
            trajectory_summary_cache_dir=Path(TRAJECTORY_SUMMARY_CACHE_DIR) if TRAJECTORY_SUMMARY_CACHE_DIR else None,
            trajectory_embedding_cache_dir=Path(TRAJECTORY_EMBEDDING_CACHE_DIR) if TRAJECTORY_EMBEDDING_CACHE_DIR else None,
            evaluation_timeout_seconds=EVALUATION_TIMEOUT_SECONDS,
            cache_mode=CACHE_MODE,
            cache_dir=Path(CACHE_DIR) if CACHE_DIR else None,
            experimental_promote_candidate=EXPERIMENTAL_PROMOTE_CANDIDATE,
            round_number=round_number,
        )
        print(f"  status: {result.status}")
        print(f"  artifacts: {result.round_dir}")
        if result.status != "completed":
            print("  stopping progressive chain")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
