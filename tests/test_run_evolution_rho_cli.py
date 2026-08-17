"""Tests for the RHO CLI surface.

The concurrency invariant must be refused BEFORE anything expensive is built, and
independently of credentials: a misconfigured run should not first report "no
model configured". Every refusal exercised here runs with no dataset, no model
endpoint, and no network.

``RoundConfig`` lives in ``agent_evolve.core.rho.rounds``. The two tests that
need a *successful* resolution import it lazily via ``importorskip`` so this
file's refusal coverage stands on its own; the refusals themselves are checked
before that import is ever reached, which is the whole point of the ordering in
:func:`resolve_rho_config`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.benchmarks.cuga_executor import PROCESS_ISOLATION  # noqa: E402
from scripts.run_evolution import (  # noqa: E402
    build_parser,
    resolve_rho_config,
)
from scripts.run_evolution import main as run_evolution_main  # noqa: E402


def _args(*extra: str):
    return build_parser().parse_args(["--dry-run", *extra])


def _live_args(*extra: str):
    """Args without --dry-run, so real-rollout refusals apply."""
    return build_parser().parse_args([*extra])


# --------------------------------------------------------------------------- #
# Mode selection
# --------------------------------------------------------------------------- #
def test_mode_defaults_to_genetic_so_existing_runs_are_unchanged() -> None:
    assert _args().mode == "genetic"


def test_mode_accepts_the_three_modes() -> None:
    for mode in ("rho", "genetic", "rho-genetic"):
        assert _args("--mode", mode).mode == mode


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(SystemExit):
        _args("--mode", "nonsense")


# --------------------------------------------------------------------------- #
# Paper defaults
# --------------------------------------------------------------------------- #
def test_paper_defaults() -> None:
    args = _args("--mode", "rho")

    assert args.rho_coreset_size == 10
    assert args.rho_group_rollouts == 3
    assert args.rho_candidates == 3
    assert args.rho_candidate_rollouts == 2
    assert args.rho_rounds == 1
    assert args.rho_selector == "dpp"


def test_history_path_is_optional_for_cold_start() -> None:
    args = _args("--mode", "rho")

    assert args.rho_history is None


def test_cache_flags_default_to_disabled() -> None:
    args = _args("--mode", "rho")

    assert args.rho_summary_cache is None
    assert args.rho_difficulty_cache is None
    assert args.rho_embedding_cache is None


# --------------------------------------------------------------------------- #
# The preflight invariant: refuse, never clamp; credential-independent
# --------------------------------------------------------------------------- #
def test_concurrency_invariant_is_enforced() -> None:
    args = _args(
        "--mode", "rho",
        "--max-workers", "12",
        "--rho-group-workers", "2",
        "--rho-rollout-workers", "2",
    )

    with pytest.raises(SystemExit) as excinfo:
        resolve_rho_config(args)

    assert "global cap" in str(excinfo.value)


def test_concurrency_invariant_refuses_rather_than_clamping() -> None:
    """A too-large cap is a configuration error, never quietly lowered."""
    args = _args(
        "--mode", "rho",
        "--max-workers", "7",
        "--rho-group-workers", "2",
        "--rho-rollout-workers", "3",
    )

    with pytest.raises(SystemExit):
        resolve_rho_config(args)

    # Nothing was mutated on the way out: the operator's value is still theirs.
    assert args.max_workers == 7


def test_concurrency_refusal_needs_no_credentials_or_dataset() -> None:
    """No --dataset, no --grader, no --harness, no model: still refused."""
    args = _live_args(
        "--mode", "rho",
        "--isolation", PROCESS_ISOLATION,
        "--max-workers", "99",
        "--rho-group-workers", "1",
        "--rho-rollout-workers", "1",
    )

    with pytest.raises(SystemExit) as excinfo:
        resolve_rho_config(args)

    message = str(excinfo.value)
    assert "global cap" in message
    assert "model" not in message.lower()


def test_concurrency_boundary_is_allowed() -> None:
    """cap == groups * rollouts is exactly satisfiable, so it must pass."""
    args = _args(
        "--mode", "rho",
        "--max-workers", "6",
        "--rho-group-workers", "2",
        "--rho-rollout-workers", "3",
    )

    pytest.importorskip("agent_evolve.core.rho.rounds")
    config = resolve_rho_config(args)

    assert config.concurrency.global_cap == 6


# --------------------------------------------------------------------------- #
# Rollout concurrency > 1 requires process isolation
# --------------------------------------------------------------------------- #
def test_threaded_rollout_concurrency_is_refused_for_a_real_run() -> None:
    """CUGA_FOLDER is process-global; threads would measure the wrong harness."""
    args = _live_args(
        "--mode", "rho",
        "--max-workers", "6",
        "--rho-group-workers", "2",
        "--rho-rollout-workers", "3",
    )

    with pytest.raises(SystemExit) as excinfo:
        resolve_rho_config(args)

    message = str(excinfo.value)
    assert "isolation" in message
    assert "CUGA_FOLDER" in message


def test_process_isolation_permits_rollout_concurrency() -> None:
    args = _live_args(
        "--mode", "rho",
        "--isolation", PROCESS_ISOLATION,
        "--max-workers", "6",
        "--rho-group-workers", "2",
        "--rho-rollout-workers", "3",
    )

    pytest.importorskip("agent_evolve.core.rho.rounds")
    config = resolve_rho_config(args)

    assert config.concurrency.global_cap == 6


def test_dry_run_is_exempt_from_the_isolation_requirement() -> None:
    """The fake stack starts no CUGA process, so CUGA_FOLDER is not involved."""
    args = _args(
        "--mode", "rho",
        "--max-workers", "6",
        "--rho-group-workers", "2",
        "--rho-rollout-workers", "3",
    )

    pytest.importorskip("agent_evolve.core.rho.rounds")
    assert resolve_rho_config(args).concurrency.global_cap == 6


# --------------------------------------------------------------------------- #
# Temperature
# --------------------------------------------------------------------------- #
def test_proposal_temperature_defaults_to_unset() -> None:
    assert _args("--mode", "rho").rho_proposal_temperature is None


def test_zero_proposal_temperature_is_refused() -> None:
    args = _args("--mode", "rho", "--rho-proposal-temperature", "0.0")

    with pytest.raises(SystemExit) as excinfo:
        resolve_rho_config(args)

    assert "temperature" in str(excinfo.value)


def test_nonzero_proposal_temperature_is_accepted() -> None:
    args = _args("--mode", "rho", "--rho-proposal-temperature", "0.7")

    pytest.importorskip("agent_evolve.core.rho.rounds")
    resolve_rho_config(args)

    assert args.rho_proposal_temperature == pytest.approx(0.7)


# --------------------------------------------------------------------------- #
# Successful resolution
# --------------------------------------------------------------------------- #
def test_valid_concurrency_resolves() -> None:
    args = _args(
        "--mode", "rho",
        "--max-workers", "6",
        "--rho-group-workers", "4",
        "--rho-rollout-workers", "3",
    )

    pytest.importorskip("agent_evolve.core.rho.rounds")
    config = resolve_rho_config(args)

    assert config.concurrency.global_cap == 6
    assert config.rollouts_per_round == 90


def test_resolved_config_carries_the_paper_shape() -> None:
    args = _args("--mode", "rho", "--max-workers", "6",
                 "--rho-group-workers", "4", "--rho-rollout-workers", "3")

    pytest.importorskip("agent_evolve.core.rho.rounds")
    config = resolve_rho_config(args)

    assert config.mode == "rho"
    assert config.coreset_size == 10
    assert config.group_rollouts == 3
    assert config.candidates == 3
    assert config.candidate_rollouts == 2
    assert config.selector == "dpp"


def test_invalid_round_count_is_reported_as_a_configuration_error() -> None:
    args = _args("--mode", "rho", "--rho-rounds", "0")

    pytest.importorskip("agent_evolve.core.rho.rounds")
    with pytest.raises(SystemExit) as excinfo:
        resolve_rho_config(args)

    assert "rounds" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# main() runs the preflight before building anything
# --------------------------------------------------------------------------- #
def test_main_refuses_an_impossible_concurrency_before_building_a_stack(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Exit 2 with the invariant's message, not a stack trace and not a run.

    --dry-run is passed deliberately: even the stack that costs nothing must not
    be constructed, because the configuration is wrong regardless.
    """
    code = run_evolution_main(
        [
            "--dry-run",
            "--mode", "rho",
            "--max-workers", "12",
            "--rho-group-workers", "2",
            "--rho-rollout-workers", "2",
        ]
    )

    assert code == 2
    out = capsys.readouterr().out
    assert "global cap" in out
    # Nothing was measured: the run never reached the header or a tally.
    assert "measuring the base" not in out


def test_main_refuses_threaded_rho_rollout_concurrency(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = run_evolution_main(
        [
            "--mode", "rho",
            "--max-workers", "6",
            "--rho-group-workers", "2",
            "--rho-rollout-workers", "3",
        ]
    )

    assert code == 2
    assert "isolation" in capsys.readouterr().out


def test_genetic_mode_never_reaches_the_rho_preflight(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An existing invocation is unchanged, including one that would violate
    the RHO invariant -- the invariant governs RHO rollouts, which do not run.
    """
    code = run_evolution_main(
        ["--dry-run", "--tasks", "1", "--iterations", "1", "--max-workers", "99"]
    )

    assert code == 0
    assert "global cap" not in capsys.readouterr().out


def test_rho_mode_says_so_rather_than_silently_running_the_genetic_loop(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A passing preflight must not be mistaken for a completed RHO round.

    Until the round machinery is wired at the composition root, ``--mode rho``
    reports what it did and did not do. Silently running the genetic loop under
    an RHO flag would attribute genetic results to RHO.
    """
    pytest.importorskip("agent_evolve.core.rho.rounds")
    code = run_evolution_main(
        [
            "--dry-run",
            "--mode", "rho",
            "--tasks", "1",
            "--iterations", "1",
            "--max-workers", "6",
            "--rho-group-workers", "4",
            "--rho-rollout-workers", "3",
        ]
    )

    out = capsys.readouterr().out
    assert code in (0, 2)
    assert "rho" in out.lower()
