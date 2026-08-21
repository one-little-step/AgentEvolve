"""SV-12 final remainder: report the entropy fallback rate in the run summary.

``entropy_unavailable_reason(task_id)`` already reports, per task, why a cell's
entropy term could not be measured. What was missing is aggregation: a run could
select every issue on quality alone -- because no mechanism cell ever cleared the
evidence floors -- and the summary would look identical to a run where entropy
genuinely drove diversity. That is the register's own stated "until then"
mitigation for SV-12.

The distinction that must survive:

* **measured zero** -- floors met, variance genuinely 0.0, no reason string;
* **unavailable** -- floors unmet, ``EntropyTracker.entropy`` returns ``None``,
  and a reason string exists.

Both surface as ``H = 0.0``, so the count is the only way to tell a diversity-
driven run from a quality-only one after the fact. Following the shape already
established for preferences (``preferences_available`` /
``preferences_unavailable`` in ``rho/rounds.py:280``) rather than inventing a new
reporting idiom.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve.core.orchestrator import (  # noqa: E402
    EntropyAvailabilityReport,
    GepaRunResult,
)


def _report(**kwargs: object) -> EntropyAvailabilityReport:
    return EntropyAvailabilityReport(**kwargs)  # type: ignore[arg-type]


def test_report_counts_available_and_unavailable_cells() -> None:
    r = _report(
        cells_available=3,
        cells_unavailable=5,
        reasons={"floor_unmet": 5},
    )
    assert r.cells_total == 8
    assert r.fallback_rate == 5 / 8


def test_fallback_rate_of_one_means_entropy_never_drove_selection() -> None:
    """The case the operator must be able to see.

    Every cell unavailable means every issue was chosen on quality alone. A run
    summary that does not say so invites reading the result as evidence that
    entropy-guided selection worked.
    """
    r = _report(cells_available=0, cells_unavailable=4, reasons={"floor_unmet": 4})
    assert r.fallback_rate == 1.0
    assert r.entropy_never_available is True


def test_fallback_rate_is_zero_when_every_cell_cleared_the_floors() -> None:
    r = _report(cells_available=6, cells_unavailable=0, reasons={})
    assert r.fallback_rate == 0.0
    assert r.entropy_never_available is False


def test_no_cells_is_not_reported_as_a_zero_fallback_rate() -> None:
    """Zero observations must not read as "entropy was fully available".

    ``0/0`` is undefined; returning ``0.0`` would claim perfect availability for
    a run that measured nothing at all.
    """
    r = _report(cells_available=0, cells_unavailable=0, reasons={})
    assert r.cells_total == 0
    assert r.fallback_rate is None
    assert r.entropy_never_available is False


def test_reasons_are_tallied_so_the_cause_is_visible() -> None:
    """A rate says how often; the tally says why.

    "floors unmet" and "adjudicator outage" call for different operator actions,
    so collapsing them into one number would hide the actionable part.
    """
    r = _report(
        cells_available=1,
        cells_unavailable=3,
        reasons={"floor_unmet": 2, "adjudication_unavailable": 1},
    )
    assert r.reasons["floor_unmet"] == 2
    assert r.reasons["adjudication_unavailable"] == 1
    assert sum(r.reasons.values()) == r.cells_unavailable


def test_report_renders_a_summary_line() -> None:
    r = _report(cells_available=1, cells_unavailable=3, reasons={"floor_unmet": 3})
    line = r.line()
    assert "3" in line and "4" in line
    assert "75" in line or "0.75" in line


def test_run_result_carries_the_report() -> None:
    """The summary object the CLI prints must expose it."""
    result = GepaRunResult(
        attempts=(),
        champion=None,
        pool_size=1,
        pareto_frontier=(),
        entropy_availability=_report(
            cells_available=2, cells_unavailable=2, reasons={"floor_unmet": 2}
        ),
    )
    assert result.entropy_availability is not None
    assert result.entropy_availability.fallback_rate == 0.5


def test_run_result_entropy_report_defaults_to_none() -> None:
    """Absent aggregation must be absent, not a fabricated zero.

    Existing callers that construct a result without the report must not appear
    to claim entropy was fully available.
    """
    result = GepaRunResult(
        attempts=(), champion=None, pool_size=1, pareto_frontier=()
    )
    assert result.entropy_availability is None


def test_report_is_immutable() -> None:
    r = _report(cells_available=1, cells_unavailable=1, reasons={})
    try:
        r.cells_available = 99  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("report must be frozen")
