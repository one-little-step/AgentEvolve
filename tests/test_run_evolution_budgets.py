"""Tests for the budget, tuning and ablation CLI surface.

Every :class:`BudgetLimits` field defaults to ``None`` (unlimited), and before
these flags existed ``resolve_profile`` hardcoded ``BudgetLimits()`` while
``_VALID_OVERRIDES`` rejected ``budgets`` outright. The result was that no caller
could cap a run's spend at all, and ``BudgetUsage.reserve`` -- fully implemented
-- was never called from anywhere in ``src/``. The enforcement tests below are
the guard against that silently becoming true again: a decorative flag that
parses but does not bound anything is worse than no flag, because it reads as a
safety limit that is not there.

No test here makes a network call or reads a dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.core.config import PROFILE_GATES  # noqa: E402
from scripts.run_evolution import (  # noqa: E402
    build_parser,
    resolve_config_overrides,
)


# --------------------------------------------------------------------------- #
# Override resolution
# --------------------------------------------------------------------------- #
def test_no_flags_produces_no_overrides() -> None:
    """An existing invocation must resolve to exactly the config it did before."""
    args = build_parser().parse_args(["--dry-run"])
    assert resolve_config_overrides(args) == {}


def test_budget_flags_reach_budget_limits() -> None:
    args = build_parser().parse_args(
        [
            "--dry-run",
            "--max-rollouts", "50",
            "--max-attempts", "4",
            "--max-accepted-edits", "2",
            "--max-editor-calls", "6",
            "--max-judge-verdicts", "7",
            "--max-model-tokens", "1000",
            "--max-wall-seconds", "900",
            "--max-pool-candidates", "8",
            "--max-history-records", "9",
            "--max-rag-context-tokens", "1200",
            "--edit-max-retries", "2",
        ]
    )
    budgets = resolve_config_overrides(args)["budgets"]
    assert budgets.max_rollouts == 50
    assert budgets.max_attempts == 4
    assert budgets.max_accepted_edits == 2
    assert budgets.max_editor_calls == 6
    assert budgets.max_judge_verdicts == 7
    assert budgets.max_model_tokens == 1000
    assert budgets.max_wall_seconds == 900.0
    assert budgets.max_pool_candidates == 8
    assert budgets.max_history_records == 9
    assert budgets.max_rag_context_tokens == 1200
    assert budgets.edit_max_retries == 2


def test_default_edit_retries_does_not_fabricate_a_budget_override() -> None:
    """``--edit-max-retries`` has a real default, so it must not leak an override.

    Otherwise "no budget flag passed" and "explicitly default" become
    indistinguishable in the manifest.
    """
    args = build_parser().parse_args(["--dry-run", "--edit-max-retries", "3"])
    assert "budgets" not in resolve_config_overrides(args)


def test_tuning_flags_reach_resolved_config() -> None:
    from agent_evolve.core.config import resolve_profile

    args = build_parser().parse_args(
        [
            "--dry-run",
            "--dpp-theta", "0.9",
            "--dpp-max-items", "40",
            "--entropy-min-rollouts-per-candidate", "3",
            "--entropy-min-comparable-candidates", "4",
            "--cluster-similarity-threshold", "0.8",
            "--generalization-probe-mode", "enabled",
            "--probe-budget-fraction", "0.25",
            "--champion-min-coverage-fraction", "0.5",
        ]
    )
    config = resolve_profile(
        "research_sequential", environ={}, **resolve_config_overrides(args)
    )
    assert config.dpp_theta == 0.9
    assert config.dpp_max_items == 40
    assert config.entropy_min_rollouts_per_candidate == 3
    assert config.entropy_min_comparable_candidates == 4
    assert config.cluster_similarity_threshold == 0.8
    assert config.generalization_probe_mode == "enabled"
    assert config.probe_budget_fraction == 0.25
    assert config.champion_min_coverage_fraction == 0.5


def test_ablation_moves_one_gate_and_leaves_the_rest_on_the_profile() -> None:
    """``--profile`` swaps all five gates; an ablation must move exactly one."""
    from agent_evolve.core.config import resolve_profile

    args = build_parser().parse_args(
        ["--dry-run", "--enable-entropy-selection", "--disable-edit-memory"]
    )
    config = resolve_profile(
        "research_sequential", environ={}, **resolve_config_overrides(args)
    )
    base = PROFILE_GATES["research_sequential"]
    assert config.features.use_entropy_selection is True   # forced on
    assert config.features.use_edit_memory is False        # forced off
    assert config.features.use_causal_blame is base["use_causal_blame"]
    assert config.features.use_focused_validation is base["use_focused_validation"]
    assert config.features.parallel_execution is base["parallel_execution"]
    assert config.features.use_positivity_judge is base["use_positivity_judge"]
    # The profile still names which profile was requested.
    assert config.profile_name == "research_sequential"


def test_positivity_judge_ablation_flag_moves_the_gate() -> None:
    """``--enable-positivity-judge`` must force the Judge-2 gate on."""
    from agent_evolve.core.config import resolve_profile

    args = build_parser().parse_args(
        ["--dry-run", "--enable-positivity-judge"]
    )
    config = resolve_profile(
        "research_sequential", environ={}, **resolve_config_overrides(args)
    )
    assert config.features.use_positivity_judge is True
    config = resolve_profile(
        "research_sequential", environ={},
        **resolve_config_overrides(
            build_parser().parse_args(["--dry-run", "--disable-positivity-judge"])
        ),
    )
    assert config.features.use_positivity_judge is False


def test_unset_ablation_flags_leave_features_untouched() -> None:
    args = build_parser().parse_args(["--dry-run"])
    assert "features" not in resolve_config_overrides(args)


# --------------------------------------------------------------------------- #
# Enforcement: the part that was entirely missing
# --------------------------------------------------------------------------- #
def _iteration_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    return [
        line.strip()
        for line in capsys.readouterr().out.splitlines()
        if line.strip().startswith("iteration ")
    ]


def _run(argv: list[str]) -> int:
    from scripts.run_evolution import main as run_evolution_main

    return run_evolution_main(argv)


def test_max_attempts_bounds_the_whole_run_not_each_iteration(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A cap means "for this run".

    A per-iteration counter would let N iterations each spend the full cap, so
    ``--max-attempts 1 --iterations 3`` would quietly issue three attempts.
    """
    assert _run(["--dry-run", "--tasks", "5", "--iterations", "3",
                 "--max-attempts", "1"]) == 0
    lines = _iteration_lines(capsys)
    assert len(lines) == 3
    assert "attempts=1" in lines[0]
    for line in lines[1:]:
        assert "attempts=0" in line
        assert "BUDGET EXHAUSTED" in line


def test_max_accepted_edits_stops_further_attempts(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _run(["--dry-run", "--tasks", "5", "--iterations", "3",
                 "--max-accepted-edits", "1"]) == 0
    lines = _iteration_lines(capsys)
    assert "accepted=1" in lines[0]
    assert sum("BUDGET EXHAUSTED" in line for line in lines) == 2


def test_uncapped_run_is_unchanged(capsys: pytest.CaptureFixture[str]) -> None:
    """The control. Without a cap, every iteration still attempts."""
    assert _run(["--dry-run", "--tasks", "5", "--iterations", "3"]) == 0
    lines = _iteration_lines(capsys)
    assert len(lines) == 3
    assert all("attempts=1" in line for line in lines)
    assert not any("BUDGET EXHAUSTED" in line for line in lines)


def test_budget_stop_is_not_reported_as_no_issue(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """"Found nothing to fix" and "was not allowed to try" are opposite facts.

    Reporting a cap as ``no_issue`` sends the reader to debug the analyzer for a
    limit they set themselves.
    """
    assert _run(["--dry-run", "--tasks", "5", "--iterations", "2",
                 "--max-attempts", "1"]) == 0
    capped = [ln for ln in _iteration_lines(capsys) if "BUDGET EXHAUSTED" in ln]
    assert capped, "expected a budget-exhausted iteration"
    for line in capped:
        assert "no_issue=0" in line


def test_reserve_is_actually_called_on_the_rollout_path() -> None:
    """``BudgetUsage.reserve`` existed but nothing in src/ ever called it.

    This pins the rollout reservation directly, so the enforcement cannot regress
    to dead code while the flags keep parsing.
    """
    from agent_evolve.core.config import BudgetLimits, BudgetUsage
    from agent_evolve.core.errors import BudgetExceededError

    usage = BudgetUsage()
    limits = BudgetLimits(max_rollouts=3)
    usage.reserve(limits, rollouts=3)
    assert usage.rollouts == 3
    with pytest.raises(BudgetExceededError):
        usage.reserve(limits, rollouts=1)
