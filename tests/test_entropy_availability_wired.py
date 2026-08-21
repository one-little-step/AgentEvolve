"""SV-12 final remainder, part 2: the runner must actually produce the report.

``EntropyAvailabilityReport`` is only useful if something populates it. A
reporting dataclass that no code writes is the inert-field defect this project
has already been bitten by once (SV-10: ``weighted_score`` carried severity and
confidence terms that no production site ever set, so a "severity-weighted"
projection was arithmetically identical to the unweighted mean).

So these tests exercise the aggregation through :class:`SequentialGepaRunner`
against a real :class:`EntropyTracker`, not against a hand-built report.

The unit of counting is the **cell**, ``(task, mechanism)``, because that is what
the floors apply to. A task for which no mechanism could be established has no
cell at all, so it cannot be counted as one; it is recorded in ``reasons``
instead. Both facts are needed: the rate says how much of the measurable surface
was measurable, the tally says what stopped the rest.

Categories are recorded at the point of failure rather than parsed back out of a
prose reason string later. String-sniffing a human-readable message is exactly
the kind of coupling that breaks silently when the wording improves.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve.core.entropy import EntropyTracker  # noqa: E402
from agent_evolve.core.orchestrator import (  # noqa: E402
    ENTROPY_UNAVAILABLE_CATEGORIES,
    SequentialGepaRunner,
)


def _runner(tracker: EntropyTracker) -> SequentialGepaRunner:
    """A runner with only the entropy surface populated.

    ``object.__new__`` avoids constructing the full dependency graph: this suite
    is about aggregation arithmetic over a tracker, and building an adapter,
    pool, editor and judge would test those instead.
    """
    runner = object.__new__(SequentialGepaRunner)
    object.__setattr__(runner, "entropy", tracker)
    object.__setattr__(runner, "_last_entropy_unavailable_reasons", {})
    object.__setattr__(runner, "_entropy_unavailable_categories", {})
    return runner


def _tracker(**kwargs: object) -> EntropyTracker:
    defaults: dict[str, object] = {
        "min_comparable_candidates": 2,
        "min_rollouts_per_candidate": 1,
    }
    defaults.update(kwargs)
    return EntropyTracker(**defaults)  # type: ignore[arg-type]


def _fill(tracker: EntropyTracker, task: str, mech: str, scores: dict[str, float]) -> None:
    """Record one score per candidate and mark each comparable."""
    for candidate, score in scores.items():
        tracker.record_score(task, mech, candidate, score)
        tracker.mark_comparable(task, mech, candidate)


def test_report_counts_a_cell_that_cleared_the_floors_as_available() -> None:
    tracker = _tracker()
    _fill(tracker, "t1", "t1:c0", {"cand-A": 0.2, "cand-B": 0.9})

    report = _runner(tracker).entropy_availability()
    assert report.cells_available == 1
    assert report.cells_unavailable == 0
    assert report.fallback_rate == 0.0


def test_report_counts_a_floor_starved_cell_as_unavailable() -> None:
    """One candidate cannot yield cross-candidate variance.

    This is the honest common case on a short run, and the number the operator
    needs to see before reading anything into an entropy-guided result.
    """
    tracker = _tracker()
    _fill(tracker, "t1", "t1:c0", {"cand-A": 0.2})

    report = _runner(tracker).entropy_availability()
    assert report.cells_available == 0
    assert report.cells_unavailable == 1
    assert report.fallback_rate == 1.0
    assert report.entropy_never_available is True


def test_report_mixes_available_and_unavailable_cells() -> None:
    tracker = _tracker()
    _fill(tracker, "t1", "t1:c0", {"cand-A": 0.2, "cand-B": 0.9})
    _fill(tracker, "t2", "t2:c0", {"cand-A": 0.5})

    report = _runner(tracker).entropy_availability()
    assert report.cells_available == 1
    assert report.cells_unavailable == 1
    assert report.fallback_rate == 0.5
    assert report.entropy_never_available is False


def test_empty_tracker_reports_no_cells_rather_than_full_availability() -> None:
    report = _runner(_tracker()).entropy_availability()
    assert report.cells_total == 0
    assert report.fallback_rate is None


def test_floor_starved_cells_are_tallied_under_a_stable_category() -> None:
    """The category must be a fixed key, not the prose reason.

    A tally keyed by free text would fragment the moment a message is reworded,
    turning one recognisable cause into several unrecognisable ones -- the same
    fragmentation problem mechanism clustering exists to solve.
    """
    tracker = _tracker()
    _fill(tracker, "t1", "t1:c0", {"cand-A": 0.2})
    _fill(tracker, "t2", "t2:c0", {"cand-B": 0.4})

    report = _runner(tracker).entropy_availability()
    assert report.reasons == {ENTROPY_UNAVAILABLE_CATEGORIES.FLOOR_UNMET: 2}


def test_task_level_causes_are_tallied_even_with_no_cell() -> None:
    """A task whose mechanism could not be established has no cell to count.

    It must still appear in the tally, or a run where clustering failed outright
    would look like a run with nothing to report.
    """
    tracker = _tracker()
    runner = _runner(tracker)
    runner._note_entropy_unavailable(
        "t9", "no analysis", ENTROPY_UNAVAILABLE_CATEGORIES.NO_ANALYSIS
    )

    report = runner.entropy_availability()
    assert report.cells_total == 0
    assert report.reasons[ENTROPY_UNAVAILABLE_CATEGORIES.NO_ANALYSIS] == 1


def test_categories_are_distinct_so_the_fix_differs_per_cause() -> None:
    """Floors unmet needs more rollouts; a dedup outage needs an endpoint fix."""
    tracker = _tracker()
    runner = _runner(tracker)
    _fill(tracker, "t1", "t1:c0", {"cand-A": 0.2})
    runner._note_entropy_unavailable(
        "t2", "adjudicator down", ENTROPY_UNAVAILABLE_CATEGORIES.UNASSIGNED
    )

    report = runner.entropy_availability()
    assert report.reasons[ENTROPY_UNAVAILABLE_CATEGORIES.FLOOR_UNMET] == 1
    assert report.reasons[ENTROPY_UNAVAILABLE_CATEGORIES.UNASSIGNED] == 1


def test_an_existing_cell_is_never_relabelled_by_a_later_task_reason() -> None:
    """Regression: measured on a real 4-attempt offline run.

    The per-task category dict is keyed by task and last-write-wins. An earlier
    version of the aggregation consulted it for existing cells, so a later
    undiagnosed rollout on ``t1`` relabelled ``t1``'s already-filed cell as
    ``no_analysis`` -- reporting a missing analysis for a cell that could only
    exist *because* an analysis produced a mechanism. The offline run showed
    ``no_analysis=3`` for three existing cells.

    ``EntropyTracker.entropy`` returns ``None`` for a present cell only when the
    evidence floor is unmet, so that is the sole correct category here.
    """
    tracker = _tracker()
    runner = _runner(tracker)
    _fill(tracker, "t1", "t1:c0", {"cand-A": 0.2})
    # A later rollout on the SAME task produced no analysis.
    runner._note_entropy_unavailable(
        "t1", "no analysis", ENTROPY_UNAVAILABLE_CATEGORIES.NO_ANALYSIS
    )

    report = runner.entropy_availability()
    assert report.cells_unavailable == 1
    assert report.reasons == {ENTROPY_UNAVAILABLE_CATEGORIES.FLOOR_UNMET: 1}, (
        "an existing cell was relabelled by a task-level reason"
    )


def test_measured_zero_variance_is_available_not_a_fallback() -> None:
    """The distinction the whole report exists to preserve.

    Two candidates that scored identically give variance 0.0 -- a real
    measurement. It must count as available, or a genuinely uniform task would
    be indistinguishable from an unmeasurable one.
    """
    tracker = _tracker()
    _fill(tracker, "t1", "t1:c0", {"cand-A": 0.5, "cand-B": 0.5})

    runner = _runner(tracker)
    report = runner.entropy_availability()
    assert report.cells_available == 1
    assert report.cells_unavailable == 0
    assert runner._cell_entropy("t1") == 0.0
    assert runner.entropy_unavailable_reason("t1") is None


def test_line_is_renderable_for_the_cli() -> None:
    tracker = _tracker()
    _fill(tracker, "t1", "t1:c0", {"cand-A": 0.2})
    line = _runner(tracker).entropy_availability().line()
    assert "1/1" in line
    assert "100%" in line


# --------------------------------------------------------------------------- #
# Production surfaces: the report must reach a place an operator can read
# --------------------------------------------------------------------------- #
def test_run_result_is_populated_by_the_runner() -> None:
    """``GepaRunResult.entropy_availability`` must not stay inert.

    A reporting field nobody writes is the SV-10 defect repeated: there,
    ``weighted_score`` carried severity and confidence terms that no production
    site ever set.
    """
    import inspect

    from agent_evolve.core import orchestrator as orch

    src = inspect.getsource(orch.SequentialGepaRunner.run)
    assert "entropy_availability=" in src, (
        "run() builds GepaRunResult without the entropy report"
    )


def test_iteration_audit_records_entropy_availability() -> None:
    """The structured audit trail must carry the fallback rate.

    ``run_iterations`` is the production loop, and its ``_record`` entries are
    the durable evidence of a run. A rate printed to stdout and not recorded
    cannot be checked after the fact.
    """
    import inspect

    from agent_evolve import pipeline as pipe

    src = inspect.getsource(pipe.EvolutionStack.run_iterations)
    assert "entropy_availability" in src or "entropy_fallback" in src, (
        "run_iterations does not record entropy availability"
    )
