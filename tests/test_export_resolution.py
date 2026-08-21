"""SV-13d wiring — the exported champion is the *resolved* winner.

``resolve_final_candidate`` decides; this covers the decision actually reaching
``champion.json`` and the measurement report.

**Why this wiring is the point.** Before it, ``champion_version`` and
``export_pool`` each called ``select_champion`` independently, so the harness
carried into the next run via ``--harness`` was picked by an aggregate with two
open defects (SV-2, SV-3). Reproduced against the real pool::

    base     outcome=0.5000 coverage=1.0000 aggregate=0.6250
    cand-A   outcome=0.9000 coverage=0.5000 aggregate=0.7450   <- exported

``cand-A`` never ran the hard task. Both defects have since been fixed in the pool
itself -- ranking is now a pairwise comparison over shared cells -- so this wiring is
no longer the only guard against that export. It still matters because the judge
compares trajectories rather than scores, and because one memoised answer keeps
measurement and export in agreement.

**Memoisation is load-bearing, not an optimisation.** ``run_evolution.py`` asks for
the winner twice -- ``champion_version()`` at :1162 to measure it, ``export_pool()``
at :1170 to write it. Resolving twice would pay the survivors' rollout cost twice
*and*, since an LLM judge is not deterministic, could export a different candidate
than the one whose score was just printed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from agent_evolve.pipeline import build_offline_stack


class _Verdict:
    def __init__(self, score: float, available: bool = True) -> None:
        self.score = score
        self.available = available


class _Judge:
    """Prefers ``winner`` if named, else always prefers the candidate slot."""

    def __init__(self, winner: str | None = None, available: bool = True) -> None:
        self.winner = winner
        self._available = available
        self.calls = 0

    def compare_symmetric(self, task, baseline, candidate, **kw):  # noqa: ANN001
        self.calls += 1
        if not self._available:
            return _Verdict(0.0, False)
        if self.winner is None:
            return _Verdict(0.9)
        cid = getattr(candidate, "candidate_id", "")
        return _Verdict(0.9 if self.winner in str(cid) else -0.9)


def _stack(judge=None):
    stack = build_offline_stack()
    if judge is not None:
        stack.runner.compare_preference = judge.compare_symmetric
    return stack


# --------------------------------------------------------------------------- #
# 1. champion_version reflects the resolution
# --------------------------------------------------------------------------- #


def test_champion_version_uses_the_resolved_winner() -> None:
    judge = _Judge()
    stack = _stack(judge)
    stack.runner.run_attempt(stack.tasks)

    version = stack.champion_version()

    assert version == stack.pool.get(stack.winner().candidate_id).version


def test_champion_version_is_memoised() -> None:
    """Resolving twice would double the rollout cost and could disagree."""
    judge = _Judge()
    stack = _stack(judge)
    stack.runner.run_attempt(stack.tasks)

    first = stack.champion_version()
    calls_after_first = stack.adapter.rollout_calls  # type: ignore[attr-defined]
    second = stack.champion_version()
    calls_after_second = stack.adapter.rollout_calls  # type: ignore[attr-defined]

    assert first == second
    assert calls_after_second == calls_after_first


def test_champion_version_falls_back_to_base_on_an_unresolvable_pool() -> None:
    """A pool with no evidence must still name something runnable."""
    stack = _stack()

    assert stack.champion_version() == stack.base_version


# --------------------------------------------------------------------------- #
# 2. export_pool writes the resolved winner
# --------------------------------------------------------------------------- #


def _champion_payload(directory: Path) -> dict:
    champion = directory / "champion.json"
    assert champion.exists(), f"no champion.json in {sorted(directory.iterdir())}"
    return json.loads(champion.read_text())


def test_export_names_the_resolved_winner_as_champion(tmp_path: Path) -> None:
    judge = _Judge()
    stack = _stack(judge)
    stack.runner.run_attempt(stack.tasks)
    expected = stack.winner().candidate_id

    stack.export_pool(tmp_path / "harnesses")

    payload = _champion_payload(tmp_path / "harnesses")
    assert payload["provenance"]["candidate_id"] == expected


def test_the_exported_champion_matches_the_measured_champion(tmp_path: Path) -> None:
    """The defect memoisation prevents: measuring one candidate and exporting
    another, which would make the reported score describe a different harness."""
    judge = _Judge()
    stack = _stack(judge)
    stack.runner.run_attempt(stack.tasks)

    measured_version = stack.champion_version()
    stack.export_pool(tmp_path / "harnesses")

    payload = _champion_payload(tmp_path / "harnesses")
    # The exported ``version`` is namespaced (``evolved-<candidate version>``);
    # provenance carries the pool-side version verbatim.
    assert payload["provenance"]["candidate_version"] == measured_version


def test_export_records_how_the_champion_was_selected(tmp_path: Path) -> None:
    """A reader must be able to tell a judged verdict from a fallback, because a
    fallback champion was ranked by a known-defective aggregate."""
    judge = _Judge()
    stack = _stack(judge)
    stack.runner.run_attempt(stack.tasks)

    stack.export_pool(tmp_path / "harnesses")

    provenance = _champion_payload(tmp_path / "harnesses")["provenance"]
    assert provenance["selection_method"] in {
        "sole_survivor",
        "pairwise_ladder",
        "aggregate_fallback",
    }
    assert provenance["selection_reason"]
    assert provenance["is_champion"] is True


def test_a_fallback_champion_is_labelled_as_a_fallback(tmp_path: Path) -> None:
    """No judge configured: the aggregate is the only opinion available, and the
    export must say so rather than implying the winner was judged."""
    stack = _stack()
    stack.runner.run_attempt(stack.tasks)

    stack.export_pool(tmp_path / "harnesses")

    provenance = _champion_payload(tmp_path / "harnesses")["provenance"]
    assert provenance["selection_method"] in {
        "sole_survivor",
        "aggregate_fallback",
    }
    assert provenance["selection_judge_calls"] == 0


def test_every_candidate_is_still_exported(tmp_path: Path) -> None:
    """Retirement must not reduce what gets written: a retired parent cost real
    rollouts and is exactly what a seeded RHO run starts from."""
    judge = _Judge()
    stack = _stack(judge)
    stack.runner.run_attempt(stack.tasks)

    written = stack.export_pool(tmp_path / "harnesses")

    candidate_files = [p for p in written if p.name.startswith("candidate-")]
    assert len(candidate_files) == len(stack.pool.all_entries())


def test_a_retired_parent_is_exported_but_not_champion(tmp_path: Path) -> None:
    """Evidence is retained; only breeding and promotion eligibility changed.

    The offspring is promotable here because retirement *records its verdict* as
    the preference: the judge measured it against the version it was derived from,
    which is exactly what the SV-4 gate asks for. Without that recording a judged
    genetic candidate would supersede its parent and still be barred from export --
    an incoherent pair, and the reason this test caught a real gap.
    """
    judge = _Judge()
    stack = _stack(judge)
    outcome = stack.runner.run_attempt(stack.tasks)
    retired = outcome.retired_parent_id
    assert retired, "fixture did not retire a parent"
    child = outcome.result_candidate_id
    assert child is not None
    assert stack.pool.get(child).preference is not None, (
        "the retirement verdict was not recorded as preference evidence, so the "
        "offspring cannot be promoted despite superseding its parent"
    )

    stack.export_pool(tmp_path / "harnesses")

    payload = _champion_payload(tmp_path / "harnesses")
    assert payload["provenance"]["candidate_id"] == child
    exported = {
        json.loads(p.read_text())["provenance"]["candidate_id"]
        for p in (tmp_path / "harnesses").glob("candidate-*.json")
    }
    assert retired in exported


def test_an_unjudged_genetic_candidate_is_not_exported_as_champion(
    tmp_path: Path,
) -> None:
    """Documented SV-4 consequence, pinned so it is a choice and not a surprise.

    A run with **no** preference judge produces candidates with
    ``preference is None``, and the gate treats "never judged" as "no evidence of
    improvement". Such a run therefore always exports the base, no matter how well
    its offspring scored. That is the specified conservative reading
    (``tests/test_preference_gate.py::test_unjudged_candidate_is_gated_out``), and
    the escape hatch is ``--experimental-candidate-promotion``.
    """
    stack = _stack()  # no judge
    outcome = stack.runner.run_attempt(stack.tasks)
    assert outcome.accepted, outcome.reason
    child = outcome.result_candidate_id
    assert child is not None
    assert stack.pool.get(child).preference is None

    stack.export_pool(tmp_path / "harnesses")

    payload = _champion_payload(tmp_path / "harnesses")
    assert payload["provenance"]["candidate_id"] == stack.pool.base.candidate_id
    # The gate leaves only the base eligible, so resolution short-circuits as a
    # sole survivor rather than reaching the ladder. Either way no judge call is
    # spent and the exported harness is the base.
    assert payload["provenance"]["selection_method"] in {
        "sole_survivor",
        "aggregate_fallback",
    }
    assert payload["provenance"]["selection_judge_calls"] == 0


def test_single_file_export_writes_the_resolved_winner(tmp_path: Path) -> None:
    """The ``*.json`` shape is what gets handed straight to ``--harness``."""
    judge = _Judge()
    stack = _stack(judge)
    stack.runner.run_attempt(stack.tasks)
    expected = stack.winner().candidate_id

    written = stack.export_pool(tmp_path / "champion.json")

    assert len(written) == 1
    assert json.loads(written[0].read_text())["provenance"]["candidate_id"] == expected


# --------------------------------------------------------------------------- #
# 3. Failure modes never lose the export
# --------------------------------------------------------------------------- #


def test_an_unavailable_judge_still_exports_a_champion(tmp_path: Path) -> None:
    """A judge outage must not leave a finished run with no exported harness."""
    stack = _stack(_Judge(available=False))
    stack.runner.run_attempt(stack.tasks)

    written = stack.export_pool(tmp_path / "harnesses")

    assert written
    payload = _champion_payload(tmp_path / "harnesses")
    assert payload["provenance"]["candidate_id"]


def test_a_raising_judge_still_exports_a_champion(tmp_path: Path) -> None:
    stack = build_offline_stack()

    def boom(task, baseline, candidate, **kw):  # noqa: ANN001
        raise RuntimeError("judge exploded")

    stack.runner.compare_preference = boom
    stack.runner.run_attempt(stack.tasks)

    written = stack.export_pool(tmp_path / "harnesses")

    assert written
    assert _champion_payload(tmp_path / "harnesses")["provenance"]["candidate_id"]
