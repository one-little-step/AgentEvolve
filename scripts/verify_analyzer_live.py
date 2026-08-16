"""Live verification: does the LLM analyzer emit genuinely distinct mechanisms?

Makes REAL model calls. This is the only check that can falsify the module's
reason to exist. The placeholder analyzer emitted ``failed-to-match-<task_id>``
for every failure on a task, so two different root causes collapsed into one
mechanism and downstream clustering/entropy/DPP became degenerate. The question
this script answers is not "does the analyzer run" but:

    Given three trajectories that fail for three genuinely different reasons,
    does the analyzer produce three genuinely different causal sentences?

Four synthetic trajectories are analyzed, each in its own report (one model call
each):

  A. A required tool was never called at all.
  B. A tool was called with arguments transposed into the wrong fields.
  C. A plan was produced and then never executed.
  D. Almost no evidence at all -- the abstention probe. A model that invents a
     mechanism here is fabricating, and that is a failure even though the run
     "worked".

Everything is printed verbatim so a human can judge the mechanisms directly.
Distinctness is reported with token-overlap numbers AND the full strings; the
overlap statistic is a signal, not a verdict.

Usage:
    uv run python scripts/verify_analyzer_live.py \
        2>&1 | tee terminal_output/analyzer/live.log
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from agent_evolve.adapters.cuga_analyzer import CugaTrajectoryAnalyzer  # noqa: E402
from agent_evolve.core.blame import CausalFinding  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.evidence import rollout_group_report  # noqa: E402

# Loading .env before RuntimeSettings.from_env() reads the environment is the
# whole reason the model configuration is visible at all in a fresh shell.
try:
    from agent_evolve.cuga_wrapper import RuntimeSettings, prepare_cuga_environment

    prepare_cuga_environment()
except Exception as exc:  # noqa: BLE001 - reported below, not swallowed
    print(f"could not load the CUGA wrapper environment: {exc}")
    RuntimeSettings = None  # type: ignore[assignment]


# ---------------------------------------------------------------------- #
# Synthetic trajectories with genuinely different root causes
# ---------------------------------------------------------------------- #
def _event(event_id, kind, actor_id, parent, payload):
    return TraceEvent(
        event_id=event_id,
        kind=kind,
        actor_id=actor_id,
        parent_event_id=parent,
        payload=payload,
    )


def case_a_tool_never_called() -> tuple[EvolutionTask, ExecutionTrace]:
    """Root cause: the currency conversion tool exists but is never invoked."""
    task = EvolutionTask(
        task_id="task-currency",
        input_text=(
            "The invoice total is 4820 JPY. Convert it to USD using the "
            "convert_currency tool and report the USD amount."
        ),
        # The answer key never reaches the analyzer (the bridge strips it).
        expected_contract={"expected_substring": "31.44"},
    )
    trace = ExecutionTrace(
        trace_id="trace-a-no-tool-call",
        candidate_id="cand-base",
        task_id="task-currency",
        events=(
            _event("a1", "llm_call_start", "planner", None, {"prompt_ref": "blob:1"}),
            _event(
                "a2",
                "tool_call",
                "planner",
                "a1",
                {
                    "tool": "list_available_tools",
                    "result": ["convert_currency", "fx_rate_lookup", "calculator"],
                },
            ),
            _event(
                "a3",
                "plan_emitted",
                "planner",
                "a2",
                {"plan": "estimate the conversion from memory"},
            ),
            _event(
                "a4",
                "tool_call",
                "answer_agent",
                "a3",
                {
                    "tool": "submit_answer",
                    "arguments": {"text": "roughly 48 USD"},
                    "note": "no convert_currency call appears anywhere in this trace",
                },
            ),
        ),
        final_output="roughly 48 USD",
        status="failure",
    )
    return task, trace


def case_b_wrong_arguments() -> tuple[EvolutionTask, ExecutionTrace]:
    """Root cause: the right tool, called with origin and destination swapped."""
    task = EvolutionTask(
        task_id="task-flight",
        input_text=(
            "Find the cheapest one-way flight from Boston (BOS) to Denver (DEN) "
            "on 2026-03-04 and report the carrier and price."
        ),
        expected_contract={"expected_substring": "UA 2317"},
    )
    trace = ExecutionTrace(
        trace_id="trace-b-swapped-args",
        candidate_id="cand-base",
        task_id="task-flight",
        events=(
            _event(
                "b1",
                "plan_emitted",
                "planner",
                None,
                {"plan": "call search_flights with origin BOS and destination DEN"},
            ),
            _event(
                "b2",
                "tool_call",
                "api_agent",
                "b1",
                {
                    "tool": "search_flights",
                    "arguments": {
                        "origin": "DEN",
                        "destination": "BOS",
                        "date": "2026-03-04",
                    },
                    "result_summary": "3 itineraries, all departing DEN arriving BOS",
                },
            ),
            _event(
                "b3",
                "tool_call",
                "answer_agent",
                "b2",
                {
                    "tool": "submit_answer",
                    "arguments": {"text": "cheapest is F9 118 departing Denver"},
                },
            ),
        ),
        final_output="cheapest is F9 118 departing Denver",
        status="failure",
    )
    return task, trace


def case_c_plan_never_executed() -> tuple[EvolutionTask, ExecutionTrace]:
    """Root cause: a correct plan is emitted, then no step of it is ever run."""
    task = EvolutionTask(
        task_id="task-report",
        input_text=(
            "Read the three quarterly CSV files in data/finance/, sum the "
            "revenue column across them, and write the total to summary.txt."
        ),
        expected_contract={"expected_substring": "1284900"},
    )
    trace = ExecutionTrace(
        trace_id="trace-c-plan-not-executed",
        candidate_id="cand-base",
        task_id="task-report",
        events=(
            _event(
                "c1",
                "plan_emitted",
                "planner",
                None,
                {
                    "plan": (
                        "1. read_csv q1.csv 2. read_csv q2.csv 3. read_csv q3.csv "
                        "4. sum revenue 5. write_file summary.txt"
                    ),
                    "step_count": 5,
                },
            ),
            _event(
                "c2",
                "control_transfer",
                "planner",
                "c1",
                {"to": "answer_agent", "steps_dispatched": 0},
            ),
            _event(
                "c3",
                "tool_call",
                "answer_agent",
                "c2",
                {
                    "tool": "submit_answer",
                    "arguments": {
                        "text": "I will read the three CSVs and sum the revenue column."
                    },
                    "note": "no read_csv, no write_file, and no executor event follow c1",
                },
            ),
        ),
        final_output="I will read the three CSVs and sum the revenue column.",
        status="failure",
    )
    return task, trace


def case_d_thin_evidence() -> tuple[EvolutionTask, ExecutionTrace]:
    """Abstention probe: one opaque event. Any mechanism here is fabricated."""
    task = EvolutionTask(
        task_id="task-opaque",
        input_text="Summarize the attached contract's termination clause.",
        expected_contract={"expected_substring": "30 days"},
    )
    trace = ExecutionTrace(
        trace_id="trace-d-thin-evidence",
        candidate_id="cand-base",
        task_id="task-opaque",
        events=(
            _event("d1", "runtime_update", None, None, {}),
        ),
        final_output="",
        status="error",
    )
    return task, trace


CASES = (
    ("A: required tool never called", case_a_tool_never_called),
    ("B: tool called with swapped arguments", case_b_wrong_arguments),
    ("C: plan emitted but never executed", case_c_plan_never_executed),
    ("D: almost no evidence (abstention probe)", case_d_thin_evidence),
)


# ---------------------------------------------------------------------- #
# Distinctness measurement
# ---------------------------------------------------------------------- #
_STOPWORDS = frozenset(
    """a an the and or but so that this it its of to in on at for with from by as
    was were is are be been being had has have did does do not never no then than
    which who whom whose when where while because thus therefore into onto over
    under after before during agent model""".split()
)


def _tokens(text: str) -> set[str]:
    words = re.findall(r"[a-z_][a-z0-9_]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


# ---------------------------------------------------------------------- #
# Reporting
# ---------------------------------------------------------------------- #
def print_finding(label: str, finding: CausalFinding) -> None:
    print(f"\n{'=' * 78}")
    print(f"CASE {label}")
    print(f"{'=' * 78}")
    print(f"trace_id  : {finding.trace_id}")
    print(f"status    : {finding.status}")
    print(f"severity  : {finding.severity}")
    print(f"confidence: {finding.confidence}")
    print(f"cluster_id: {finding.mechanism_cluster_id}")
    print("\nMECHANISM (verbatim):")
    print(f"  {finding.mechanism_description!r}")
    print("\nBLAMED ACTORS:")
    if finding.blame_graph.nodes:
        for node in finding.blame_graph.nodes:
            artifacts = ", ".join(node.artifacts) or "(none)"
            print(f"  {node.actor_id:<16} blame={node.blame:<6} artifacts={artifacts}")
    else:
        print("  (none attributed)")
    print("\nCAUSAL LINKS:")
    if finding.blame_graph.edges:
        for edge in finding.blame_graph.edges:
            print(f"  {edge.from_actor} -> {edge.to_actor}: {edge.mechanism}")
    else:
        print("  (none)")
    print(f"\nEVIDENCE REFS: {list(finding.evidence_refs)}")
    print(f"\nRATIONALE:\n  {finding.rationale}")
    if finding.counterfactual_notes:
        print("\nCOUNTERFACTUAL NOTES:")
        for note in finding.counterfactual_notes:
            print(f"  - {note}")


def main() -> int:
    if RuntimeSettings is None:
        print("ABORT: the CUGA wrapper could not be imported; cannot resolve a model.")
        return 2
    try:
        settings = RuntimeSettings.from_env()
    except RuntimeError as exc:
        print("ABORT: no model configured, so no live call can be made.")
        print(f"  {exc}")
        print(
            "  Set CUGA_MODEL (or LITELLM_MODEL) plus CUGA_BASE_URL / CUGA_API_KEY "
            "in .env or the environment, then re-run."
        )
        return 2

    print("LIVE ANALYZER VERIFICATION")
    print(f"model config (credentials never printed): {settings.public_config()}")
    print(f"cases: {len(CASES)}; one model call per case")

    analyzer = CugaTrajectoryAnalyzer()

    results: list[tuple[str, CausalFinding | None, str]] = []
    for label, build in CASES:
        task, trace = build()
        report = rollout_group_report(task, trace)
        try:
            findings = analyzer.analyze(report)
        except Exception as exc:  # noqa: BLE001 - a transport failure must be visible
            print(f"\nCASE {label}: MODEL CALL FAILED: {type(exc).__name__}: {exc}")
            results.append((label, None, f"{type(exc).__name__}: {exc}"))
            continue
        if not findings:
            print(f"\nCASE {label}: analyzer returned no findings")
            results.append((label, None, "no findings returned"))
            continue
        print_finding(label, findings[0])
        results.append((label, findings[0], ""))

    # ------------------------------------------------------------------ #
    # The key assertion
    # ------------------------------------------------------------------ #
    print(f"\n\n{'#' * 78}")
    print("# KEY CHECK: are the three distinct root causes described distinctly?")
    print(f"{'#' * 78}")

    causal = [
        (label, f.mechanism_description)
        for label, f, _ in results[:3]
        if f is not None and f.mechanism_description
    ]
    if len(causal) < 3:
        print(
            f"INCONCLUSIVE: only {len(causal)}/3 causal cases produced a mechanism. "
            "Distinctness cannot be judged."
        )
        for label, f, err in results[:3]:
            state = err or (f.status if f else "unknown")
            print(f"  {label}: {state}")
        return 1

    print("\nThe three mechanisms, verbatim:\n")
    for label, mechanism in causal:
        print(f"  [{label}]")
        print(f"    {mechanism}\n")

    exact_unique = len({m for _, m in causal}) == 3
    print(f"all three strings differ exactly: {exact_unique}")

    print("\npairwise token overlap (Jaccard; lower means more distinct):")
    max_overlap = 0.0
    for i in range(len(causal)):
        for j in range(i + 1, len(causal)):
            (la, ma), (lb, mb) = causal[i], causal[j]
            overlap = _jaccard(ma, mb)
            max_overlap = max(max_overlap, overlap)
            print(f"  {la[0]} vs {lb[0]}: {overlap:.2f}")

    # 0.5 is a reporting threshold, not a proven one. Some shared vocabulary is
    # expected (all three describe agent steps); near-identical wording is not.
    print(f"\nmax pairwise overlap: {max_overlap:.2f}")
    distinct = exact_unique and max_overlap < 0.5
    print(
        "VERDICT: mechanisms are "
        + ("DISTINCT" if distinct else "NOT CLEARLY DISTINCT")
        + " by the automated check. Read the strings above and judge for yourself: "
        "the automated check cannot tell a real causal difference from a reworded "
        "template."
    )

    # ------------------------------------------------------------------ #
    # Abstention probe
    # ------------------------------------------------------------------ #
    print(f"\n{'#' * 78}")
    print("# ABSTENTION PROBE (case D): did the analyzer refuse to fabricate?")
    print(f"{'#' * 78}")
    _, d_finding, d_error = results[3]
    if d_finding is None:
        print(f"case D did not produce a finding: {d_error}")
        honest = False
    else:
        honest = d_finding.status in {
            "insufficient_evidence",
            "uncertain",
            "malformed",
        }
        print(f"status: {d_finding.status}")
        print(f"mechanism: {d_finding.mechanism_description!r}")
        print(f"blamed actors: {[n.actor_id for n in d_finding.blame_graph.nodes]}")
        print(
            "VERDICT: "
            + (
                "abstained or hedged, as required."
                if honest
                else "FABRICATED an observed mechanism from one opaque event. "
                "This is a real failure."
            )
        )

    print(f"\n{'#' * 78}")
    print(
        "NOTE: this is ONE sample per case. A single successful run does not "
        "establish that the analyzer reliably produces distinct, grounded "
        "mechanisms; it only shows it can."
    )
    print(f"{'#' * 78}")

    return 0 if (distinct and honest) else 1


if __name__ == "__main__":
    raise SystemExit(main())
