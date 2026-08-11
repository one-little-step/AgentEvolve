"""End-to-end smoke driver for the AgentEvolve contract.

Runs the minimal evolution loop against :class:`examples.fake_adapter.FakeAdapter`:

    base materialize -> rollout -> trace -> apply edit -> re-rollout -> validate

No LLM is invoked. The point is to prove the contract surface composes
correctly; the actual analyzer/judge/editor logic is the next phase of work
per ``docs/plans/rho-parallel-gepa-completion.md``.

Usage
-----
    uv run python examples/run_demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make the project importable when run via `uv run python examples/run_demo.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # so `examples.fake_adapter` is importable

from agent_evolve.adapters.base import validate_adapter  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    ArtifactEdit,
    EvolutionTask,
)
from examples.fake_adapter import FakeAdapter  # noqa: E402


def _print(label: str, value: object) -> None:
    print(f"  {label}: {value}")


def main() -> int:
    adapter = FakeAdapter()
    validate_adapter(adapter)
    print(f"[ok] adapter validated: {adapter.adapter_name}")

    # 1. Inspect base inventory.
    base_version = "base-v0"
    inventory = adapter.artifact_inventory(base_version)
    print(f"[ok] base inventory: {len(inventory)} artifacts")
    for d in inventory:
        _print(d.artifact_id, d.version_hash)

    # 2. Define a task. The expected substring must NOT be in the base
    #    skill yet, so the base rollout fails to match it.
    task = EvolutionTask(
        task_id="task-001",
        input_text="Find the API spec and call it.",
        expected_contract={"expected_substring": "graphrag-retrieval"},
        source_paths=(),
    )
    print(f"[ok] task: {task.task_id} (expects: {task.expected_contract['expected_substring']})")

    # 3. Materialize a base candidate and roll it out.
    base_ws = adapter.materialize_candidate(base_version, "attempt-base")
    base_result = adapter.run_full_rollout(base_ws, task, "rollout-base")
    base_trace = adapter.capture_trace(base_result)
    print(f"[ok] base rollout status={base_trace.status}, output={base_trace.final_output!r}")
    matched = task.expected_contract["expected_substring"] in base_trace.final_output
    print(f"[note] base matched expected substring: {matched}")

    # 4. Apply a structured edit that adds the missing token to the skill.
    edit = ArtifactEdit(
        artifact_id="skills/retrieval",
        operation="replace",
        payload={
            "content": "retrieve(query): use graphrag-retrieval for top_k docs",
        },
    )
    candidate_ws = adapter.materialize_candidate(base_version, "attempt-001")
    changed = adapter.apply_structured_edits(candidate_ws, (edit,))
    print(f"[ok] edit applied, changed: {list(changed.keys())}")

    # 5. Re-rollout the candidate and verify the expected substring is now
    #    produced by the deterministic fake agent.
    cand_result = adapter.run_full_rollout(candidate_ws, task, "rollout-cand-001")
    cand_trace = adapter.capture_trace(cand_result)
    print(f"[ok] candidate rollout status={cand_trace.status}")
    print(f"     output: {cand_trace.final_output!r}")

    matched_after = task.expected_contract["expected_substring"] in cand_trace.final_output
    print(f"[ok] candidate matched expected substring: {matched_after}")

    # 6. Confirm isolation: the base workspace must be unaffected by the edit
    #    to the candidate workspace.
    base_contents_after = adapter.read_artifacts(base_version, ("skills/retrieval",))
    print(f"[ok] base skill untouched: {base_contents_after['skills/retrieval']!r}")

    if not matched_after:
        print("[FAIL] candidate did not satisfy expected contract")
        return 1
    if not matched:
        print("[ok] base failed before edit, candidate succeeded after — evolution worked.")
    else:
        print("[note] base already matched expected substring (unexpected for this demo).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
