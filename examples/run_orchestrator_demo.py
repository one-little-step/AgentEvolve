"""End-to-end demo: run the full RHO-Parallel-GEPA orchestrator against FakeAdapter.

This demonstrates the complete evolution loop implemented in
``src/agent_evolve/core/``:

    initialize base -> roll out base on tasks -> analyze failures ->
    editor proposes edits -> focused validation -> accept/reject ->
    add accepted candidates to persistent pool -> Pareto frontier.

Runs the ``minimal`` profile by default. Pass ``research_sequential`` or
``research_parallel`` as argv[1] to exercise those profiles.

Usage
-----
    uv run python examples/run_orchestrator_demo.py [profile_name]

Profiles: minimal (default), research_sequential, research_parallel
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.contracts import EvolutionCandidate, EvolutionTask  # noqa: E402
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.orchestrator import (  # noqa: E402
    MINIMAL,
    RESEARCH_PARALLEL,
    RESEARCH_SEQUENTIAL,
    Orchestrator,
)
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

PROFILES = {
    "minimal": MINIMAL,
    "research_sequential": RESEARCH_SEQUENTIAL,
    "research_parallel": RESEARCH_PARALLEL,
}


def main() -> int:
    profile_name = sys.argv[1] if len(sys.argv) > 1 else "minimal"
    if profile_name not in PROFILES:
        print(f"[FAIL] unknown profile: {profile_name!r}")
        print(f"       available: {sorted(PROFILES.keys())}")
        return 1
    profile = PROFILES[profile_name]

    adapter = FakeAdapter()
    pool = PersistentPool()
    orch = Orchestrator(
        adapter=adapter,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        pool=pool,
        profile=profile,
    )

    base = EvolutionCandidate(
        candidate_id="base",
        version="base-v0",
        artifact_hashes={
            "skills/retrieval": "h1",
            "policies/execution": "h2",
            "prompts/system": "h3",
        },
    )
    orch.initialize_base(base)
    print(f"[ok] base initialized: {base.candidate_id}")
    print(f"[ok] profile: {profile.name}")
    print(f"     features: causal_blame={profile.use_causal_blame}, "
          f"edit_memory={profile.use_edit_memory}, "
          f"focused_validation={profile.use_focused_validation}, "
          f"entropy_selection={profile.use_entropy_selection}, "
          f"parallel_batch={profile.use_parallel_batch}")

    tasks = [
        EvolutionTask(
            task_id="task-retrieval",
            input_text="Find and use the API.",
            expected_contract={"expected_substring": "graphrag-retrieval"},
        ),
        EvolutionTask(
            task_id="task-execution",
            input_text="Run the tool.",
            expected_contract={"expected_substring": "tool-executor-v2"},
        ),
    ]
    print(f"[ok] tasks: {[t.task_id for t in tasks]}")

    print("\n--- Iteration 1 ---")
    r1 = orch.run_iteration(tasks)
    _print_result(r1)

    print("\n--- Iteration 2 ---")
    r2 = orch.run_iteration(tasks)
    _print_result(r2)

    print("\n--- Final pool state ---")
    print(f"  pool size: {len(pool)}")
    print(f"  candidates: {pool.candidate_ids()}")
    print(f"  Pareto frontier: {pool.pareto_frontier()}")
    return 0


def _print_result(r) -> None:
    print(f"  iteration: {r.iteration}")
    print(f"  attempts: {len(r.attempts)}")
    print(f"  accepted: {len(r.accepted)} ({list(r.accepted)})")
    print(f"  rejected: {len(r.rejected)} ({list(r.rejected)})")
    print(f"  regressions: {len(r.regressions)} ({list(r.regressions)})")
    print(f"  pool size after: {r.pool_size}")


if __name__ == "__main__":
    raise SystemExit(main())
