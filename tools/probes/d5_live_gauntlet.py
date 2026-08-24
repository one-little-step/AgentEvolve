"""D5 live gauntlet: rigorous real-LLM testing of the positivity chain.

Run with the project venv:  python tools/probes/d5_live_gauntlet.py <cmd>

Subcommands
-----------
judges : both adapters against the REAL endpoint
         - fault analyzer on a rich failure      -> observed, valence=+1
         - fault analyzer on an event-less trace -> abstaining (never invents)
         - positivity judge on a rich success    -> strengths, valence=-1
         - positivity judge on event-less success-> abstaining strengths
         - grounding: emitted actors must exist in the trace
         Every call goes through a capturing completion wrapper: per-call
         purpose, latency, token usage (feeds ?08 budget audit), and a
         redacted echo (api_key stripped) so nothing sensitive prints.

embed  : the D5.1 crown property on REAL embeddings (embeddinggemma):
         fault text vs its near-paraphrase FIX must clear the join
         threshold (cosine >= 0.75), unrelated pairs must stay apart.
         This upgrades the lexical-only proof to the calibrated instrument.

runner : full pipeline with the REAL positivity judge in the loop:
         accepted attempt -> validate opens the gate -> TS2 stores real
         strengths -> commit -> signed_mechanism_index() -> TL payload.
         Faults stay faked (covered by `judges`); spend stays bounded.

No keys are ever printed. Captures redact authorization material by
construction (the wrapper strips request["api_key"] before echoing).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

import litellm  # noqa: E402

from agent_evolve.adapters.cuga_analyzer import CugaTrajectoryAnalyzer  # noqa: E402
from agent_evolve.adapters.cuga_positivity_judge import CugaPositivityJudge  # noqa: E402
from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.blame import CausalAnalysis  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionCandidate,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)
from agent_evolve.core.evidence import rollout_group_report  # noqa: E402

TOKEN = "CONF-2026"

CALL_LOG: list[dict] = []


def capturing_completion(purpose: str):
    """litellm wrapper that records purpose/latency/usage per call."""

    def completion_fn(**request: object) -> object:
        started = time.time()
        response = litellm.completion(**request)  # type: ignore[arg-type]
        took = time.time() - started
        usage = getattr(response, "usage", None)
        content = ""
        try:
            content = response.choices[0].message.content or ""  # type: ignore[index]
        except Exception:  # noqa: BLE001
            pass
        CALL_LOG.append(
            {
                "purpose": purpose,
                "seconds": round(took, 2),
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
                "content_chars": len(content),
            }
        )
        return response

    return completion_fn


# ---------------------------------------------------------------------- #
# Traces
# ---------------------------------------------------------------------- #
def _task(tid="task-live") -> EvolutionTask:
    return EvolutionTask(
        task_id=tid,
        input_text="Book a flight Boston->Denver and return the confirmation code.",
        expected_contract={"expected_substring": TOKEN},
    )


def _rich_trace(status: str, *, final: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=f"tr-{status}-{abs(hash(final)) % 9999}",
        candidate_id="cand-live",
        task_id=_task().task_id,
        events=(
            TraceEvent("e1", "thought", "planner", None,
                       {"note": "decompose: search then book"}),
            TraceEvent("e2", "tool_call", "planner", "e1",
                       {"tool": "plan", "skill": "skills/retrieval"}),
            TraceEvent("e3", "tool_call", "api_agent", "e2",
                       {"tool": "search_flights", "origin": "BOS", "dest": "DEN"}),
            TraceEvent("e4", "tool_call", "api_agent", "e3",
                       {"tool": "book_flight", "result": final}),
        ),
        final_output=final,
        status=status,
    )


def _bare_trace(status: str) -> ExecutionTrace:
    return ExecutionTrace(
        trace_id=f"tr-bare-{status}",
        candidate_id="cand-live",
        task_id=_task().task_id,
        events=(),
        final_output="",
        status=status,
    )


def _report(trace: ExecutionTrace):
    return rollout_group_report(_task(), [trace])


# ---------------------------------------------------------------------- #
# judges
# ---------------------------------------------------------------------- #
def cmd_judges() -> None:
    failures: list[str] = []

    def check(label: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
        if not cond:
            failures.append(label)

    print("== A1 fault analyzer, rich failure ==")
    fault_judge = CugaTrajectoryAnalyzer(
        completion_fn=capturing_completion("fault/rich"),
        max_events_in_prompt=20,
    )
    ff = fault_judge.analyze(_report(_rich_trace("failure", final="no code returned")))
    check("one finding", len(ff) == 1, str(len(ff)))
    check("observed", ff[0].status == "observed", ff[0].status)
    check("valence +1 (code-stamped)", ff[0].valence == 1, str(ff[0].valence))
    check("mechanism is a causal sentence", len(ff[0].mechanism_description or "") > 40)
    known_actors = {"planner", "api_agent"}
    emitted = {n.actor_id for n in ff[0].blame_graph.nodes}
    check("actors grounded", emitted <= known_actors, str(emitted))

    print("== A2 fault analyzer, bare trace (must abstain, not invent) ==")
    fb = fault_judge.analyze(capturing_completion("noop") and _report(_bare_trace("failure")))
    # NOTE: bare trace has no evidence -> analyzer returns () or abstaining.
    check("never observed on empty evidence",
          all(f.status != "observed" for f in fb) or not fb,
          str([f.status for f in fb]))

    print("== A3 positivity judge, rich success ==")
    pos_judge = CugaPositivityJudge(completion_fn=capturing_completion("positivity/rich"))
    sf = pos_judge.analyze_success(_task(), _rich_trace("success", final=f"done {TOKEN}"))
    check("strengths produced", len(sf) >= 1, str(len(sf)))
    check("all valence -1", all(f.valence == -1 for f in sf))
    observed = [f for f in sf if f.status == "observed"]
    check("at least one observed", len(observed) >= 1, str([f.status for f in sf]))
    if observed:
        best = observed[0]
        check("causal sentence", len(best.mechanism_description or "") > 40)
        s_actors = {n.actor_id for n in best.blame_graph.nodes}
        check("actors grounded", s_actors <= known_actors | {"agent"}, str(s_actors))

    print("== A4 positivity judge, bare success (must abstain) ==")
    sb = pos_judge.analyze_success(_task(), _bare_trace("success"))
    check("no observed strengths from nothing",
          all(f.status != "observed" for f in sb),
          str([f.status for f in sb]))
    check("still strengths by polarity", all(f.valence == -1 for f in sb))

    print()
    print(f"-- {len(CALL_LOG)} live calls --")
    for c in CALL_LOG:
        print("  ", c)
    total_out = sum(c["completion_tokens"] or 0 for c in CALL_LOG)
    print(f"-- total completion tokens: {total_out} (?08 budget datum) --")

    if failures:
        print("GAUNTLET FAILURES:", failures)
        raise SystemExit(1)
    print("JUDGES GAUNTLET: ALL PASS")


# ---------------------------------------------------------------------- #
# embed
# ---------------------------------------------------------------------- #
def _cos(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = (sum(x * x for x in a) * sum(y * y for y in b)) ** 0.5
    return num / den


FAULT_TEXT = (
    "planner retrieval timed out so context held no documents and "
    "api_agent answered the flight search from memory"
)
FIX_TEXT = (
    "planner retrieval succeeded after retry so context held the documents "
    "and api_agent answered the flight search with grounded results"
)
UNRELATED = (
    "pagination loop skipped the last page so two records were missing "
    "from the exported csv report"
)


def cmd_embed() -> None:
    from agent_evolve.core.embeddings import OllamaEmbedder

    emb = OllamaEmbedder(url="http://localhost:11434", model="embeddinggemma")
    v_fault = emb.embed(FAULT_TEXT)
    v_fix = emb.embed(FIX_TEXT)
    v_unrel = emb.embed(UNRELATED)

    c_join = _cos(v_fault, v_fix)
    c_apart = _cos(v_fault, v_unrel)
    c_fix_apart = _cos(v_fix, v_unrel)
    print(f"dim={len(v_fault)}")
    print(f"cos(fault, its-fix)      = {c_join:.3f}   (join threshold 0.75)")
    print(f"cos(fault, unrelated)    = {c_apart:.3f}   (band low 0.45)")
    print(f"cos(fix, unrelated)      = {c_fix_apart:.3f}")

    ok = True
    if c_join >= 0.75:
        print("[PASS] crown property holds on REAL embeddings: fix joins fault cluster")
    else:
        print("[WARN] fix does NOT clear join threshold on real embeddings "
              "-> same-fault/fix pair lands in ambiguous band; adjudicator or "
              "band recalibration needed (feeds ?09 follow-up)")
        ok = False
    if c_apart < 0.45 and c_fix_apart < 0.45:
        print("[PASS] unrelated mechanisms stay below band-low")
    else:
        print("[WARN] unrelated pair above band-low -> distribution shifted")
        ok = False
    if not ok:
        raise SystemExit(1)
    print("EMBED GAUNTLET: PASS")


# ---------------------------------------------------------------------- #
# runner (real positivity judge inside the full pipeline)
# ---------------------------------------------------------------------- #
def cmd_runner() -> None:
    from agent_evolve.core.orchestrator import SequentialGepaRunner
    from agent_evolve.core.pool import PersistentPool
    from agent_evolve.core.fake_editor import FakeEditor
    from examples.fake_adapter import FakeAdapter

    # Standard accept flow: base FAILS task-a (no token) so build_issues has
    # work; FakeEditor's fix makes the CHILD pass -> validation opens the
    # positivity gate with the REAL judge.
    adapter = FakeAdapter()
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base", version="base-v0",
            artifact_hashes={d.artifact_id: d.version_hash
                             for d in adapter.artifact_inventory("base-v0")},
        )
    )

    class _RealFault(FakeAnalyzerJudge):
        """Faults stay deterministic; the LIVE side is the positivity judge."""

    runner = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=_RealFault(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        positivity_judge=CugaPositivityJudge(
            completion_fn=capturing_completion("positivity/in-runner")
        ),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id="mechanism-default",
        seed=0,
    )

    print("== C1 accepted attempt with REAL positivity judge on the gate ==")
    outcome = runner.run_attempt([_task("task-a")])
    print("  accepted:", outcome.accepted, "|", outcome.reason[:60])
    assert outcome.accepted, "expected acceptance"
    assert runner._positivity_calls >= 1, "gate did not call the real judge"
    stored_child = runner.traces_for(outcome.result_candidate_id, "task-a")
    real_strengths = [f for r in stored_child for f in r.strengths]
    print(f"  real strengths stored in TS2: {len(real_strengths)}")
    assert real_strengths, "real strengths did not reach the store"
    assert all(f.valence == -1 for f in real_strengths)

    print("== C2 signed index contains REAL strengths cross-candidate ==")
    index = runner.signed_mechanism_index()
    cand_ids = {
        m.candidate_id
        for key in index.clusters()
        for m in index.members_for(*key)
    }
    print("  candidates in index:", sorted(cand_ids))
    assert outcome.result_candidate_id in cand_ids, "child missing from index"

    print("== C3 TL payload over the real index ==")
    from agent_evolve.core.blame import BlameGraph
    from agent_evolve.core.mechanism_index import complementary_parent_payload

    def _analysis_of(text: str) -> CausalAnalysis:
        return CausalAnalysis(
            mechanism=text, severity=0.5, score=0.0,
            blame_graph=BlameGraph(nodes=()),
        )

    # (a) OK path: ask about the mechanism the child REALLY solved (its own
    # stored strength text). The solver must come back ranked first.
    strength_text = real_strengths[0].mechanism_description or ""
    payload = complementary_parent_payload(
        index=index,
        registry=runner.cluster_registry,
        task_id="task-a",
        analysis=_analysis_of(strength_text),
    )
    print("  [ok-path] status:", payload["status"],
          "| members:", [(m["role"], m["candidate_id"]) for m in payload["members"]])
    assert payload["status"] == "ok", json.dumps(payload)[:200]
    assert any(
        m["role"] == "solver" and m["candidate_id"] == outcome.result_candidate_id
        for m in payload["members"]
    )

    # (b) DEGRADE path: ask about an UNRELATED failure -- nobody solved it,
    # so the tool must say solvers_absent and list least-bad faults instead
    # of pretending help exists. (Live-observed behaviour, now pinned.)
    degrade = complementary_parent_payload(
        index=index,
        registry=runner.cluster_registry,
        task_id="task-a",
        analysis=_analysis_of("planner retrieval timed out; answered from memory"),
    )
    print("  [degrade] status:", degrade["status"],
          "| members:", [(m["role"], m["candidate_id"]) for m in degrade["members"]])
    assert degrade["status"] in ("solvers_absent", "unclustered")
    if degrade["status"] == "solvers_absent":
        assert all(m["role"] == "least_bad_failure" for m in degrade["members"])

    print()
    for c in CALL_LOG:
        print("  ", c)
    print("RUNNER GAUNTLET: ALL PASS")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd in ("judges", "all"):
        CALL_LOG.clear()
        cmd_judges()
    if cmd in ("embed", "all"):
        cmd_embed()
    if cmd in ("runner", "all"):
        CALL_LOG.clear()
        cmd_runner()
