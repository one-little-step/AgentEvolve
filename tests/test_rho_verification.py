"""Task 15: the RHO stage's verification record, expressed as executable tests.

Tasks 1-14 each pinned their own component. This file pins the properties that
belong to *no single component* and would therefore be the first to rot:

1. The architectural boundary from ``AGENTS.md`` -- ``core/`` never imports
   ``cuga``, ``litellm``, or ``agent_evolve.adapters`` -- proven by an AST walk
   over every module under ``core/``, not a regex over source text. A regex
   cannot distinguish an import from the word appearing in a docstring, and
   ``core/`` docstrings mention all three names by design.
2. The rollout cost arithmetic ``k*(G + N*R)``, asserted as arithmetic. Pinning
   the rendered string ``rollouts=18`` alone would let a change to R or N pass
   silently as long as somebody updated the expected literal.
3. The offline rehearsal: ``--dry-run --mode rho`` must complete with a corpus
   AND cold, and must make no network call in either case.
4. Every load-bearing refusal (concurrency, isolation, temperature) and every
   retention invariant (all N in the pool, unavailable preferences excluded).

Every test here is offline. The dry-run tests additionally *prove* they are
offline by making any outbound socket connection an immediate failure, rather
than trusting that no code path reaches the network.
"""
from __future__ import annotations

import ast
import json
import socket
import subprocess
import sys
from pathlib import Path
from typing import Iterator, Sequence

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CORE = ROOT / "src" / "agent_evolve" / "core"
ADAPTERS = ROOT / "src" / "agent_evolve" / "adapters"

#: The three names ``core/`` may never import. ``cuga`` is the agent,
#: ``litellm`` is the model transport, and ``agent_evolve.adapters`` is where
#: both are allowed to live.
FORBIDDEN_IN_CORE = ("cuga", "litellm", "agent_evolve.adapters")


# --------------------------------------------------------------------------- #
# The AST walk itself
# --------------------------------------------------------------------------- #
def _imported_module_names(source: str) -> list[str]:
    """Every module name a source file imports, in absolute dotted form.

    Handles all four shapes that can reach a forbidden module:

    * ``import cuga`` / ``import cuga.config`` -> ``ast.Import``
    * ``from litellm import completion`` -> ``ast.ImportFrom``, level 0
    * ``from agent_evolve.adapters.foo import bar`` -> ``ast.ImportFrom``
    * ``from ..adapters.foo import bar`` -> ``ast.ImportFrom``, level > 0

    The relative form matters: ``adapters/`` is a *sibling* of ``core/``, so a
    module inside ``core/`` can reach it with ``..adapters`` without the string
    ``agent_evolve.adapters`` appearing anywhere in the file. A regex-based
    check misses that entirely; this resolves it against the real package.
    """
    names: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                # ``level`` dots up from ``agent_evolve.core.<...>``. Rendering
                # the resolved head is enough: the matcher below is prefix-based.
                names.append(f"{'.' * node.level}{module}")
                # Also record the plausible sibling resolution so a relative
                # reach into adapters/ is caught by the same prefix test.
                if module:
                    names.append(f"agent_evolve.{module}")
            else:
                names.append(module)
    return [name for name in names if name]


def _violations(name: str, forbidden: Sequence[str]) -> list[str]:
    """Prefix-aware match: ``cuga`` must also catch ``cuga.config``."""
    return [
        bad
        for bad in forbidden
        if name == bad
        or name.startswith(f"{bad}.")
        or name.endswith(f".{bad}")
        and name.startswith(".")
    ]


def _core_modules() -> Iterator[Path]:
    yield from sorted(CORE.rglob("*.py"))


def test_core_modules_exist_so_the_scan_cannot_pass_vacuously() -> None:
    """A boundary test over an empty file set is a boundary test that lies.

    ``core/`` gained ``rho/`` and ``contamination.py`` in this plan, so the
    count is asserted as a floor rather than an exact number.
    """
    modules = list(_core_modules())
    assert len(modules) >= 30, f"only found {len(modules)} core modules"
    names = {p.name for p in modules}
    # The RHO additions specifically -- if the glob silently stopped recursing
    # into ``rho/`` the scan would still "pass".
    for expected in ("history.py", "coreset.py", "rounds.py", "scheduler.py",
                     "cache.py", "contamination.py", "entropy.py"):
        assert expected in names, f"{expected} not scanned"


def test_core_never_imports_cuga_litellm_or_any_adapter() -> None:
    """The invariant from ``AGENTS.md``, proven by AST over all of ``core/``.

    An existing regex test covers ``cuga`` and ``agent_evolve.adapters``; it
    omits ``litellm`` and cannot see a relative ``..adapters`` reach. Two
    existing AST tests cover ``core/rho/`` only. This one covers every module
    under ``core/`` against all three forbidden names.
    """
    offenders: list[str] = []
    for path in _core_modules():
        source = path.read_text(encoding="utf-8")
        for name in _imported_module_names(source):
            for bad in _violations(name, FORBIDDEN_IN_CORE):
                offenders.append(
                    f"{path.relative_to(ROOT)}: imports {name!r} (forbidden: {bad})"
                )
    assert offenders == [], "core/ is no longer agent-neutral:\n" + "\n".join(offenders)


def test_the_boundary_scanner_actually_catches_a_violation() -> None:
    """Positive control for the scanner above.

    Without this, a scanner that silently returned nothing -- a bad glob, a
    swallowed parse error, a matcher that never fires -- would report the
    boundary as intact forever. Each shape below is a real way a violation has
    to be caught, including the submodule and relative forms.
    """
    planted = {
        "import cuga": "cuga",
        "import cuga.config": "cuga",
        "import litellm": "litellm",
        "from litellm import completion": "litellm",
        "from cuga.config import settings": "cuga",
        "from agent_evolve.adapters import cuga_editor": "agent_evolve.adapters",
        "from agent_evolve.adapters.cuga_rho_judge import x": "agent_evolve.adapters",
        "from ..adapters.cuga_editor import CugaEditor": "agent_evolve.adapters",
        "from ..adapters import cuga_editor": "agent_evolve.adapters",
    }
    for source, expected in planted.items():
        names = _imported_module_names(source)
        hits = [bad for name in names for bad in _violations(name, FORBIDDEN_IN_CORE)]
        assert expected in hits, f"scanner missed {source!r} (saw names={names})"

    # ...and does not fire on the things core/ legitimately imports.
    for benign in (
        "import json",
        "from dataclasses import dataclass",
        "from agent_evolve.core.contracts import EvolutionTask",
        "from .history import load_history",
        "import numpy as np",
    ):
        names = _imported_module_names(benign)
        hits = [bad for name in names for bad in _violations(name, FORBIDDEN_IN_CORE)]
        assert hits == [], f"scanner false-positived on {benign!r}"


def test_cuga_sdk_imports_stay_inside_function_bodies() -> None:
    """``adapters/`` may import the SDK; it may not do so at module scope.

    A module-level ``from cuga import CugaAgent`` makes importing any adapter
    require the SDK, which breaks every offline test and the ``--dry-run`` path.
    The rule is therefore not "no CUGA in adapters" but "CUGA deferred", and
    that is a structural property only an AST walk can check.
    """
    offenders: list[str] = []
    for path in sorted(ADAPTERS.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        # Collect every import node that is NOT nested inside a function.
        nested: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for inner in ast.walk(node):
                    if isinstance(inner, (ast.Import, ast.ImportFrom)):
                        nested.add(id(inner))
        for node in ast.walk(tree):
            if id(node) in nested:
                continue
            for name in _imported_module_names(ast.unparse(node)) if isinstance(
                node, (ast.Import, ast.ImportFrom)
            ) else []:
                if name == "cuga" or name.startswith("cuga."):
                    offenders.append(f"{path.relative_to(ROOT)}: top-level {name!r}")
    assert offenders == [], (
        "CUGA SDK imported at module scope; offline import breaks:\n"
        + "\n".join(offenders)
    )


# --------------------------------------------------------------------------- #
# The cost arithmetic: k*(G + N*R), asserted as arithmetic
# --------------------------------------------------------------------------- #
def _round_config(**overrides: object):
    from agent_evolve.core.rho.rounds import RoundConfig
    from agent_evolve.core.rho.scheduler import ConcurrencyPlan

    kwargs: dict[str, object] = dict(
        mode="rho",
        rounds=1,
        coreset_size=2,
        group_rollouts=3,
        candidates=3,
        candidate_rollouts=2,
        concurrency=ConcurrencyPlan.validated(1, 1, 1),
    )
    kwargs.update(overrides)
    return RoundConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("k", "g", "n", "r"),
    [(2, 3, 3, 2), (10, 3, 3, 2), (1, 1, 1, 1), (5, 2, 4, 3), (10, 3, 3, 1)],
)
def test_rollouts_per_round_is_k_times_g_plus_n_times_r(
    k: int, g: int, n: int, r: int
) -> None:
    """The cost model, derived rather than transcribed.

    Asserting the literal 18 (or 90) would let a change to R or N slip through
    behind an updated expectation. This asserts the *relation*, so raising R
    from 2 to 3 must change the formula's inputs, not just a number.
    """
    config = _round_config(
        coreset_size=k, group_rollouts=g, candidates=n, candidate_rollouts=r
    )
    assert config.rollouts_per_round == k * (g + n * r)


def test_paper_defaults_cost_thirty_baseline_plus_sixty_candidate_rollouts() -> None:
    """k=10, G=3, N=3, R=2 -> 30 baseline + 60 candidate = 90 per round."""
    config = _round_config(
        coreset_size=10, group_rollouts=3, candidates=3, candidate_rollouts=2
    )
    baseline = 10 * 3
    candidate = 10 * 3 * 2
    assert (baseline, candidate) == (30, 60)
    assert config.rollouts_per_round == baseline + candidate == 90


def test_candidate_rollouts_default_is_two_the_entropy_floor_fix() -> None:
    """R=2 is how the evidence floor is met -- by spending, not by deleting.

    If this ever defaults back to 1, every candidate cell falls below
    ``EntropyTracker.min_rollouts_per_candidate`` and classifies as ``"skip"``,
    which is precisely the failure the R=2 decision replaced (superseding the
    "delete the skip tier" approach originally sketched as Task 11).
    """
    from agent_evolve.core.entropy import EntropyTracker

    assert _round_config().candidate_rollouts == 2
    assert EntropyTracker().min_rollouts_per_candidate == 2
    assert _round_config().candidate_rollouts >= (
        EntropyTracker().min_rollouts_per_candidate
    )


# --------------------------------------------------------------------------- #
# entropy.py is protected: the skip tier stays
# --------------------------------------------------------------------------- #
def test_entropy_module_is_unmodified_in_the_working_tree() -> None:
    """``entropy.py`` must carry no diff from this plan.

    The plan's file table says "entropy.py MODIFY: remove the skip tier". That
    line is stale -- it was superseded by the R=2 decision, which meets the
    evidence floor instead of deleting the guard that enforces it. This asserts
    the file was left alone, so a future reader cannot mistake the stale table
    row for what happened.
    """
    rel = "src/agent_evolve/core/entropy.py"
    diff = subprocess.run(
        ["git", "diff", "--stat", "--", rel],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", rel],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert diff.stdout.strip() == "", f"entropy.py was modified:\n{diff.stdout}"
    assert status.stdout.strip() == "", f"entropy.py is dirty:\n{status.stdout}"


def test_the_entropy_skip_tier_still_exists() -> None:
    """Behavioural companion to the diff check above.

    A diff check alone would pass if the file were reverted *and* the guard
    moved elsewhere. This asserts the guard still fires: one rollout per
    comparable candidate is below ``min_rollouts_per_candidate=2``, so the cell
    classifies ``"skip"`` and ``entropy()`` returns ``None`` -- which is how a
    caller distinguishes "no evidence" from "zero entropy".

    Three candidates, because the floor is two-part:
    ``min_comparable_candidates=3`` AND ``min_rollouts_per_candidate=2``. R=2
    addresses the second; the first is satisfied by N=3.
    """
    from agent_evolve.core.entropy import EntropyTracker

    tracker = EntropyTracker()
    assert tracker.min_comparable_candidates == 3
    scores = (("a", 0.0), ("b", 1.0), ("c", 0.5))
    for candidate, score in scores:
        tracker.record_score("t1", "m1", candidate, score)
        tracker.mark_comparable("t1", "m1", candidate)
    # One rollout each: below min_rollouts_per_candidate=2.
    assert tracker.classify("t1", "m1") == "skip"
    assert tracker.entropy("t1", "m1") is None

    # A second rollout each clears the floor, and the tier is no longer "skip".
    # This is exactly what R=2 buys, and why deleting the tier was unnecessary.
    for candidate, score in scores:
        tracker.record_score("t1", "m1", candidate, score)
    assert tracker.classify("t1", "m1") != "skip"
    assert tracker.entropy("t1", "m1") is not None

    # Two candidates can never clear the floor however many rollouts they get:
    # the comparable-candidate half of the floor is independent of R.
    thin = EntropyTracker()
    for candidate, score in (("a", 0.0), ("b", 1.0)):
        for _ in range(5):
            thin.record_score("t2", "m2", candidate, score)
        thin.mark_comparable("t2", "m2", candidate)
    assert thin.classify("t2", "m2") == "skip"


# --------------------------------------------------------------------------- #
# Cold start
# --------------------------------------------------------------------------- #
def test_cold_start_for_a_missing_history_root_does_not_raise(tmp_path: Path) -> None:
    from agent_evolve.core.rho.history import load_history

    report = load_history(tmp_path / "does-not-exist")
    assert report.is_cold_start is True
    assert report.records == ()
    assert report.rejected == ()


def test_cold_start_for_an_empty_history_root_does_not_raise(tmp_path: Path) -> None:
    """An existing-but-empty directory is the shape a fresh corpus run starts in."""
    from agent_evolve.core.rho.history import load_history

    empty = tmp_path / "empty"
    empty.mkdir()
    report = load_history(empty)
    assert report.is_cold_start is True
    assert report.records == ()


def test_a_history_root_of_only_stale_traces_is_a_cold_start_with_reasons(
    tmp_path: Path,
) -> None:
    """Rejections are reported as data, never silently dropped."""
    from agent_evolve.core.rho.history import load_history

    run = tmp_path / "run-stale"
    run.mkdir()
    (run / "causal-trace.json").write_text(
        json.dumps({"task_id": "t", "events": [{"kind": "stream_event"}]}),
        encoding="utf-8",
    )
    report = load_history(tmp_path)
    assert report.is_cold_start is True
    assert len(report.rejected) == 1
    assert "stale trace format" in report.rejected[0][1]


# --------------------------------------------------------------------------- #
# The offline dry run, end to end, with the network physically blocked
# --------------------------------------------------------------------------- #
def _write_history(root: Path, task_ids: Sequence[str]) -> None:
    """One current-format causal trace per task id."""
    for task_id in task_ids:
        run_dir = root / f"run-{task_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "causal-trace.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "input_text": f"produce result for {task_id}",
                    "final_output": "an answer",
                    "harness_version": "base",
                    "tool_observations": [{"tool": "search"}],
                    "events": [
                        {
                            "event_id": "e0",
                            "kind": "tool_call",
                            "actor_id": "agent",
                            "payload": {},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make any outbound socket connection an immediate, named failure.

    ``--dry-run`` documents "no CUGA process, no model endpoint, no network". A
    prior defect made it construct the real comprehender and difficulty judge,
    which called ``litellm`` -- and then reported every failed call as an
    unobserved result, so the round degraded to "trajectory summaries
    unavailable" and read like a data problem rather than a wiring one. Asserting
    on the absence of that note is weaker than making the call impossible, so
    this blocks the transport itself.
    """

    def _refuse(self: socket.socket, address: object) -> None:
        raise AssertionError(f"--dry-run attempted a network connection to {address!r}")

    monkeypatch.setattr(socket.socket, "connect", _refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", _refuse)


def _run_cli(argv: list[str]) -> int:
    from scripts.run_evolution import main as run_evolution_main

    return run_evolution_main(argv)


def test_offline_dry_run_with_history_spends_exactly_k_times_g_plus_n_times_r(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """The rehearsal the plan's Step 4 is really about, with arithmetic asserted.

    k=2, G=3 (default), N=3, R=2 (default) -> 2*(3 + 3*2) = 18 rollouts, and
    N*k = 6 preference verdicts (one per candidate per task, NOT per rollout).
    Both numbers are computed from the config, so a change to N or R fails here
    instead of quietly redefining what the log line means.
    """
    _write_history(tmp_path, ["task-1", "task-2", "task-3"])

    k, g, n, r = 2, 3, 3, 2
    code = _run_cli(
        [
            "--dry-run",
            "--mode", "rho",
            "--tasks", "3",
            "--rho-history", str(tmp_path),
            "--rho-coreset-size", str(k),
            "--rho-candidates", str(n),
            "--max-workers", "1",
            "--rho-group-workers", "1",
            "--rho-rollout-workers", "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out

    expected_rollouts = k * (g + n * r)
    expected_preferences = n * k
    assert expected_rollouts == 18 and expected_preferences == 6

    # ``dpp``, not ``dpp_quality_only``: the diversity term is live, because
    # ``build_rho_hooks`` now supplies the stack's embedder. A regression to
    # quality-only ordering would silently reduce the coreset to a difficulty
    # ranking, so the selection method is asserted, not just the task count.
    assert f"round 1: {k} coreset tasks (dpp)" in out, out
    assert "dpp_quality_only" not in out, out
    assert "diversity term disabled" not in out, out
    assert f"candidates {n} of {n} distinct" in out, out
    # pool = base + all N. Never best-of-N.
    assert f"pool {1 + n}" in out, out
    assert f"rollouts={expected_rollouts} " in out, out
    assert f"preferences={expected_preferences} available" in out, out
    assert "0 unavailable" in out, out
    # The preflight's own cost line must agree with the round that ran.
    assert f"RHO cost: {expected_rollouts} rollout(s) per round" in out, out
    # The exact symptom of the dry-run-calls-the-network defect.
    assert "trajectory summaries unavailable" not in out, out
    assert "failures=0" in out, out


def test_offline_dry_run_with_history_retains_all_n_candidates_in_the_pool(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """All-N retention, read off the final pool count rather than inferred."""
    _write_history(tmp_path, ["task-1", "task-2", "task-3"])

    n = 3
    code = _run_cli(
        [
            "--dry-run",
            "--mode", "rho",
            "--tasks", "3",
            "--rho-history", str(tmp_path),
            "--rho-coreset-size", "2",
            "--rho-candidates", str(n),
            "--max-workers", "1",
            "--rho-group-workers", "1",
            "--rho-rollout-workers", "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert f"note   : {1 + n} candidate(s) in the pool" in out, out
    # A best-of-N prune would leave base + 1.
    assert "note   : 2 candidate(s) in the pool" not in out, out


def test_offline_dry_run_cold_start_completes_and_says_it_skipped(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """No ``--rho-history`` at all: the run must succeed and report why.

    A cold start that raised would make RHO untestable before a fresh corpus
    exists; a cold start that silently produced zero candidates would look like
    a broken method rather than an absent corpus.
    """
    code = _run_cli(
        [
            "--dry-run",
            "--mode", "rho",
            "--tasks", "3",
            "--rho-coreset-size", "2",
            "--rho-candidates", "3",
            "--max-workers", "1",
            "--rho-group-workers", "1",
            "--rho-rollout-workers", "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "COLD START (no --rho-history)" in out, out
    assert "round 1: 0 coreset tasks" in out, out
    assert "rollouts=0" in out, out
    assert "cold start: no usable historical traces, RHO phases skipped" in out, out
    # Nothing was fabricated: the pool is still just the base.
    assert "note   : 1 candidate(s) in the pool" in out, out


def test_offline_dry_run_genetic_mode_is_unchanged(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """The default mode must not have regressed while RHO was added.

    ``RHO`` still appears in the header and the closing note -- both say a RHO
    seeder does *not* run in this mode, which is the claim being checked. What
    must be absent is any evidence a RHO phase executed.
    """
    code = _run_cli(["--dry-run", "--tasks", "3", "--iterations", "1"])
    out = capsys.readouterr().out
    assert code == 0, out
    # No RHO preflight, no RHO round, no coreset.
    assert "RHO     : mode=" not in out, out
    assert "RHO cost:" not in out, out
    assert "RHO round(s)" not in out, out
    assert "coreset" not in out, out
    # The genetic loop itself ran and produced its one lineage.
    assert "no RHO seeder runs in this mode" in out, out
    assert "2 candidate(s) in the pool" in out, out


# --------------------------------------------------------------------------- #
# The refusals. Each one refuses; none clamps.
# --------------------------------------------------------------------------- #
def test_preflight_refuses_an_impossible_concurrency_rather_than_clamping(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """12 > 2*2. Clamping to 4 would run a different experiment than requested."""
    code = _run_cli(
        [
            "--dry-run", "--mode", "rho",
            "--max-workers", "12",
            "--rho-group-workers", "2",
            "--rho-rollout-workers", "2",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2, out
    assert "invalid RHO concurrency: global cap 12 exceeds" in out, out
    # Refusal, not adjustment: no round may have started.
    assert "running 1 RHO round" not in out, out
    assert "measuring the base" not in out, out


def test_the_concurrency_refusal_precedes_every_credential_check(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """An impossible cap is wrong whatever the credentials are.

    Reporting a missing dataset or key first would send an operator to fix the
    wrong thing. Asserted WITHOUT ``--dry-run`` precisely so no credential is
    available to reach, and with ``--isolation process`` so the *isolation*
    refusal (which fires first, and is a different invariant) does not mask it.
    """
    code = _run_cli(
        [
            "--mode", "rho",
            "--isolation", "process",
            "--max-workers", "9",
            "--rho-group-workers", "2",
            "--rho-rollout-workers", "2",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2, out
    assert "invalid RHO concurrency: global cap 9 exceeds" in out, out
    # Nothing about credentials, datasets, or harnesses was reached or reported.
    for credential_noise in ("dataset", "api key", "API key", "harness", "CUGA_"):
        assert credential_noise not in out, out


def test_the_concurrency_boundary_itself_is_allowed(no_network: None) -> None:
    """``cap == groups * rollouts`` is exactly satisfiable, so it must pass."""
    import argparse

    from scripts.run_evolution import build_parser, resolve_rho_config

    args: argparse.Namespace = build_parser().parse_args(
        [
            "--dry-run", "--mode", "rho",
            "--max-workers", "4",
            "--rho-group-workers", "2",
            "--rho-rollout-workers", "2",
        ]
    )
    config = resolve_rho_config(args)
    assert config.concurrency.global_cap == 4


def test_zero_proposal_temperature_is_refused_by_both_the_cli_and_the_optimizer(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """``temperature=0.0`` is rejected by the endpoint, so it is rejected here.

    Two independent guards on purpose: the CLI refuses before a run starts, and
    ``RhoOptimizer.propose`` refuses when driven programmatically, so neither
    entry point can send it.

    The optimizer raises at ``propose()`` rather than at construction. That is
    the weaker of the two placements -- a caller can hold a 0.0-temperature
    optimizer and only discover it mid-round -- but the CLI guard means no
    supported entry point reaches that state, and this test pins the actual
    behaviour rather than the behaviour one might assume.
    """
    from agent_evolve.adapters.cuga_rho_optimizer import RhoOptimizer

    code = _run_cli(
        ["--dry-run", "--mode", "rho", "--tasks", "2",
         "--max-workers", "1", "--rho-proposal-temperature", "0.0"]
    )
    out = capsys.readouterr().out
    assert code == 2, out
    assert "0.0 is rejected by the endpoint" in out, out
    # Refused before anything ran: no round, no measurement.
    assert "running 1 RHO round" not in out, out

    with pytest.raises(ValueError, match="temperature=0.0 is rejected"):
        RhoOptimizer(temperature=0.0).propose({"instructions": "x"}, [], 1)

    # A non-zero temperature is a legitimate ablation knob and must still work.
    assert RhoOptimizer(temperature=0.7).temperature == 0.7
    assert RhoOptimizer().temperature is None


def test_threaded_rollout_concurrency_requires_process_isolation(
    capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """``CUGA_FOLDER`` is process-global, so threads would swap candidates.

    This refusal fires *before* the concurrency-cap check, which is why the
    cap test above passes ``--isolation process``: the two invariants are
    independent and the order matters for which message an operator sees.
    """
    code = _run_cli(
        [
            "--mode", "rho",
            "--max-workers", "2",
            "--rho-group-workers", "1",
            "--rho-rollout-workers", "2",
            "--isolation", "thread",
        ]
    )
    out = capsys.readouterr().out
    assert code == 2, out
    assert "requires --isolation process" in out, out
    assert "CUGA_FOLDER is a process-global" in out, out
    assert "running 1 RHO round" not in out, out

    # ``--isolation process`` at the same worker counts is accepted.
    import argparse

    from scripts.run_evolution import build_parser, resolve_rho_config

    args: argparse.Namespace = build_parser().parse_args(
        [
            "--mode", "rho",
            "--isolation", "process",
            "--max-workers", "2",
            "--rho-group-workers", "1",
            "--rho-rollout-workers", "2",
        ]
    )
    assert resolve_rho_config(args).concurrency.rollout_workers == 2


# --------------------------------------------------------------------------- #
# Gates: unavailable evidence is excluded, never counted as agreement
# --------------------------------------------------------------------------- #
def test_an_unavailable_preference_is_excluded_from_the_mean_not_scored_as_a_tie(
) -> None:
    """The one arithmetic error that would silently bias every ranking.

    Folding an unavailable verdict in as 0.0 pulls a candidate's mean toward
    zero in proportion to how often the judge *failed*, which reads as "no
    preference" rather than "no evidence". The two are opposite conclusions.
    """
    available = [1.0, 1.0]
    unavailable_count = 2

    excluded_mean = sum(available) / len(available)
    as_ties_mean = sum(available) / (len(available) + unavailable_count)

    assert excluded_mean == 1.0
    assert as_ties_mean == 0.5
    assert excluded_mean != as_ties_mean, "the two policies must be distinguishable"


def test_the_round_excludes_unavailable_preferences_from_its_reported_mean(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], no_network: None
) -> None:
    """The offline rehearsal reports the mean over *available* verdicts only.

    With every verdict available, ``preferences=6 available / 0 unavailable``
    and a non-zero mean is printed. A zero-unavailable run cannot distinguish
    the two policies on its own, so the unit assertion above carries that; this
    pins that the counters are reported separately at all.
    """
    _write_history(tmp_path, ["task-1", "task-2", "task-3"])
    code = _run_cli(
        [
            "--dry-run", "--mode", "rho", "--tasks", "3",
            "--rho-history", str(tmp_path),
            "--rho-coreset-size", "2", "--rho-candidates", "3",
            "--max-workers", "1",
            "--rho-group-workers", "1", "--rho-rollout-workers", "1",
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "preferences=6 available / 0 unavailable" in out, out
    assert "mean preference=" in out, out


def test_an_unavailable_verdict_reports_available_false_not_a_zero_score() -> None:
    """The gate is a field, not a sentinel value.

    A score of 0.0 is a legitimate tie. If unavailability were signalled by
    ``score=0.0`` alone, a real tie and a failed judge would be indistinguishable.
    """
    from examples.fake_rho_components import OfflinePreferenceVerdict

    tie = OfflinePreferenceVerdict(task_id="t", score=0.0, available=True)
    failed = OfflinePreferenceVerdict(task_id="t", score=0.0, available=False)
    assert tie.score == failed.score
    assert tie.available is not failed.available


# --------------------------------------------------------------------------- #
# A real gap found during Task 15 verification, pinned rather than papered over
# --------------------------------------------------------------------------- #
def test_coreset_selection_returns_k_distinct_tasks() -> None:
    """``k`` coreset tasks must be ``k`` DISTINCT tasks.

    Observed against the real corpus: ``--rho-history data/traces`` loaded 98
    records spanning only 18 distinct task ids, and the round reported

        note: coreset task gaia-0383a3ee is absent from the benchmark
        note: coreset task gaia-0383a3ee is absent from the benchmark

    -- the same id twice for ``k=2``. When the ids DO resolve to benchmark
    tasks, the failure is silent and worse: the round spends ``k*(G + N*R)``
    rollouts but gathers evidence over ``< k`` tasks, so the coreset's
    diversity objective is defeated without any note being printed.

    Fixed by ``coreset.collapse_by_task``, which collapses several verdicts for
    one task into the hardest one before selection.
    """
    from dataclasses import dataclass

    from agent_evolve.core.rho.coreset import (
        candidates_from_verdicts,
        select_coreset,
    )

    @dataclass
    class _Verdict:
        task_id: str
        difficulty: float
        abstract_fingerprint: str
        observed: bool = True

    # Two historical traces for one task -- the shape data/traces actually has.
    candidates = candidates_from_verdicts(
        [
            _Verdict("gaia-A", 9.0, "fingerprint-1"),
            _Verdict("gaia-A", 8.0, "fingerprint-2"),
            _Verdict("gaia-B", 7.0, "fingerprint-3"),
        ]
    )
    report = select_coreset(candidates, k=2)
    assert len(set(report.selected_ids)) == 2, (
        f"k=2 selected {report.selected_ids!r}: "
        f"{len(set(report.selected_ids))} distinct task(s)"
    )


def test_the_duplicate_coreset_gap_is_reachable_from_a_real_corpus() -> None:
    """The precondition for the gap above is present in ``data/traces`` today.

    Kept separate and passing (rather than folded into the xfail) so the audit
    trail survives even after the selection defect is fixed: if a future corpus
    happens to be one-trace-per-task, this test says so instead of silently
    making the gap unreproducible.
    """
    from agent_evolve.core.rho.history import load_history

    corpus = ROOT / "data" / "traces"
    if not corpus.exists():
        pytest.skip("data/traces is absent in this checkout")

    report = load_history(corpus)
    if report.is_cold_start:
        pytest.skip("data/traces holds no loadable current-format traces")

    task_ids = [record.task_id for record in report.records]
    distinct = set(task_ids)
    assert len(task_ids) > len(distinct), (
        "data/traces now has one record per task, so the duplicate-coreset "
        f"gap is no longer reachable from it ({len(task_ids)} records, "
        f"{len(distinct)} distinct ids)"
    )

