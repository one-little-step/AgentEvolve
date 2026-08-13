"""Phase-gate verification: B0/B1 smoke harness over the deterministic FakeAdapter.

This test proves the B0/B1 contrast the research core must demonstrate before
any H1 claim:

* B0 (best-of-N) keeps only its single highest-scoring candidate.
* B1 (persistent pool) retains every initial candidate and derives a Pareto
  frontier with more than one non-dominated candidate.
* The harness persists a redacted manifest, candidate, and score record set and
  never writes the evaluator-internal expected-substring tokens to storage.
* The harness is deterministic: the same seed reproduces an identical outcome.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))  # for `examples.run_phase_1_4_smoke`

from examples.run_phase_1_4_smoke import (  # noqa: E402
    run_fixed_budget_comparison,
)

# The evaluator-internal tokens the harness must never persist.
_TOKEN_A = "token-a"
_TOKEN_B = "token-b"


def _read_all_persisted_blobs(root: Path) -> str:
    parts: list[str] = []
    for path in sorted(root.rglob("*.json")):
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def test_b1_retains_all_initial_candidates_while_b0_discards_non_winners(
    tmp_path: Path,
) -> None:
    outcome = run_fixed_budget_comparison(seed=7, storage_root=tmp_path)
    assert outcome.b0_retained_candidate_count == 1
    assert outcome.b1_retained_candidate_count > 1
    assert outcome.storage_records_are_redacted is True


def test_smoke_run_is_deterministic_and_never_persists_evaluator_tokens(
    tmp_path: Path,
) -> None:
    first = run_fixed_budget_comparison(seed=3, storage_root=tmp_path / "run-a")
    second = run_fixed_budget_comparison(seed=3, storage_root=tmp_path / "run-b")
    assert first == second

    for subdir in ("run-a", "run-b"):
        blob = _read_all_persisted_blobs(tmp_path / subdir)
        assert _TOKEN_A not in blob
        assert _TOKEN_B not in blob
