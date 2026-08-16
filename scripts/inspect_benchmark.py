"""Summarize completed benchmark runs: noise floor, grader spread, coverage.

Prints, per run: task count, each grader's pass count/rate WITH its explicit
denominator, missing-key coverage, timeout/error counts, and the number of
FAILED evaluation batches. Then a cross-run comparison that refuses to compare
mismatched denominators.

No grading material (patterns, gold answers, judge reasons, answer spans) is
ever printed -- only aggregate counts.

Usage::

    uv run python scripts/inspect_benchmark.py
    uv run python scripts/inspect_benchmark.py --root datasets/gaia
    uv run python scripts/inspect_benchmark.py --run <run_dir> --run <run_dir>
    uv run python scripts/inspect_benchmark.py --coverage-limit 0   # all keys
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

from agent_evolve.benchmarks import (
    GaiaBenchmark,
    RunStatistics,
    compare_runs,
    compute_run_statistics,
    discover_gaia_runs,
    outcomes_disagree,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = REPO_ROOT / "datasets" / "gaia"


def _fmt_rate(value: float | None) -> str:
    return "n/a" if value is None else f"{value:6.2%}"


def _print_run(bench: GaiaBenchmark, stats: RunStatistics, coverage_limit: int) -> None:
    print(f"\n=== run: {stats.run_name}")
    print(f"    benchmark      : {stats.benchmark_name}")
    print(f"    model          : {stats.config.get('model', '<unknown>')}")
    print(f"    tasks          : {stats.task_count}")
    print(f"    timed out      : {stats.timed_out}")
    print(f"    errored        : {stats.errored}")
    print(f"    failed eval batches (excluded from results): {stats.failed_eval_batches}")

    print("    graders:")
    for grader in bench.graders():
        gs = stats.grader_stats[grader]
        flag = "  <-- PARTIAL DENOMINATOR" if gs.is_partial else ""
        print(
            f"      {grader:<24} pass {gs.passed:>3} / {gs.evaluated:>3} evaluated"
            f"  (of {gs.total_tasks} tasks)  rate {_fmt_rate(gs.pass_rate)}"
            f"  ungraded {gs.unavailable:>3}{flag}"
        )

    coverage = stats.key_coverage
    raw_missing = coverage.get("missing", {})
    raw_empty = coverage.get("empty", {})
    missing: dict[str, int] = dict(raw_missing) if isinstance(raw_missing, dict) else {}
    empty: dict[str, int] = dict(raw_empty) if isinstance(raw_empty, dict) else {}
    print(
        f"    coverage       : {coverage.get('record_count', 0)} result.json read; "
        f"{coverage.get('task_dirs_without_record', 0)} task dir(s) without result.json; "
        f"{coverage.get('unreadable_records', 0)} unreadable; "
        f"{coverage.get('recorded_verdict_conflicts', 0)} verdict conflict(s)"
    )
    gaps = [
        (key, missing.get(key, 0), empty.get(key, 0))
        for key in sorted(set(missing) | set(empty))
        if missing.get(key, 0) or empty.get(key, 0)
    ]
    if not gaps:
        print("    key gaps       : none (all keys present and non-empty)")
    else:
        shown = gaps if coverage_limit <= 0 else gaps[:coverage_limit]
        print("    key gaps       : key (absent / no-usable-value) out of "
              f"{coverage.get('record_count', 0)}")
        for key, absent, blank in shown:
            print(f"        {key:<20} absent {absent:>3}   no-usable-value {blank:>3}")
        if len(shown) < len(gaps):
            print(f"        ... {len(gaps) - len(shown)} more (use --coverage-limit 0)")


def _print_grader_agreement(bench: GaiaBenchmark, stats: RunStatistics) -> None:
    """Quantify whether the two ground-truth measures agree on this run."""
    both = 0
    agree = 0
    only_regex_passed = 0
    only_other_passed = 0
    graders = bench.graders()
    if len(graders) < 2:
        return
    primary, secondary = graders[0], graders[1]
    for task_id in stats.task_ids:
        outcomes = bench.score_all(task_id, bench.recorded_answer(task_id) or "")
        if primary not in outcomes or secondary not in outcomes:
            continue
        both += 1
        if not outcomes_disagree(outcomes):
            agree += 1
        elif outcomes[primary].passed:
            only_regex_passed += 1
        else:
            only_other_passed += 1
    rate = "n/a" if both == 0 else f"{agree / both:.2%}"
    print(
        f"    grader agreement: {agree}/{both} comparable tasks agree ({rate}); "
        f"{primary}-only pass {only_regex_passed}, {secondary}-only pass {only_other_passed}"
    )


def _print_comparison(a: RunStatistics, b: RunStatistics) -> None:
    comparison = compare_runs(a, b)
    print(f"\n--- {comparison.run_a}  ->  {comparison.run_b}")
    print(
        f"    same task set: {comparison.same_task_set}"
        f"  shared tasks: {comparison.shared_task_count}"
    )
    for note in comparison.notes:
        print(f"    note: {note}")
    for grader, delta in comparison.deltas.items():
        rate = delta.pass_rate_delta
        rate_text = "n/a (not comparable)" if rate is None else f"{rate:+.2%}"
        print(
            f"    {grader:<24} {delta.passed_a}/{delta.evaluated_a} -> "
            f"{delta.passed_b}/{delta.evaluated_b}"
            f"   passed {delta.passed_delta:+d}   rate {rate_text}"
        )
        for note in delta.notes:
            print(f"        ! {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize completed benchmark runs: noise floor, grader spread, coverage."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"benchmark root containing run directories (default: {DEFAULT_ROOT})",
    )
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        default=None,
        help="explicit run directory (repeatable); overrides --root discovery",
    )
    parser.add_argument(
        "--coverage-limit",
        type=int,
        default=8,
        help="max key-gap rows to print per run (0 = all)",
    )
    parser.add_argument(
        "--no-agreement",
        action="store_true",
        help="skip the per-run grader agreement replay",
    )
    args = parser.parse_args(argv)

    run_dirs = tuple(args.run) if args.run else discover_gaia_runs(args.root)
    if not run_dirs:
        print(f"no benchmark runs found under {args.root}")
        return 1

    print(f"benchmark root: {args.root}")
    print(f"runs found    : {len(run_dirs)}")

    loaded: list[tuple[GaiaBenchmark, RunStatistics]] = []
    for run_dir in run_dirs:
        bench = GaiaBenchmark.from_run_dir(run_dir)
        stats = compute_run_statistics(bench, bench.observations())
        _print_run(bench, stats, args.coverage_limit)
        if not args.no_agreement:
            _print_grader_agreement(bench, stats)
        loaded.append((bench, stats))

    print("\n=== cross-run comparison")
    if len(loaded) < 2:
        print("    only one run available; no comparison possible")
        return 0
    for (_, a), (_, b) in itertools.combinations((item for item in loaded), 2):
        _print_comparison(a, b)

    print("\n=== spread over runs sharing an identical task set")
    groups: dict[tuple[str, ...], list[RunStatistics]] = {}
    for _, stats in loaded:
        groups.setdefault(stats.task_ids, []).append(stats)
    for task_ids, group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), len(kv[0]))):
        if len(group) < 2:
            continue
        print(f"\n    task set of {len(task_ids)} tasks, {len(group)} runs:")
        for stats in group:
            for grader, gs in stats.grader_stats.items():
                print(
                    f"      {stats.run_name:<48} {grader:<24} "
                    f"{gs.passed}/{gs.evaluated}  {_fmt_rate(gs.pass_rate)}"
                    f"{'  PARTIAL' if gs.is_partial else ''}"
                )
        for grader in group[0].grader_stats:
            full = [s.grader_stats[grader] for s in group if not s.grader_stats[grader].is_partial]
            partial = [s.grader_stats[grader] for s in group if s.grader_stats[grader].is_partial]
            rates = [gs.pass_rate for gs in full if gs.pass_rate is not None]
            if len(rates) >= 2:
                print(
                    f"      spread[{grader}] over {len(rates)} FULL-denominator runs: "
                    f"min {min(rates):.2%} max {max(rates):.2%} "
                    f"range {max(rates) - min(rates):.2%}"
                )
            else:
                print(
                    f"      spread[{grader}]: only {len(rates)} full-denominator run(s); "
                    f"{len(partial)} partial run(s) excluded -- no valid spread"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
