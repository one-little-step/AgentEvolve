"""Analyzer+judge protocol, a deterministic fake, and the dual-protocol shim.

The analyzer+judge is the LLM-driven component that produces a
:class:`CausalAnalysis` for one (candidate, task, rollout) triple. The
generic core defines the Protocol; an adapter or host application supplies
a concrete implementation backed by a real LLM.

For testing and offline demos, :class:`FakeAnalyzerJudge` produces a
deterministic analysis derived from the trace's ``final_output`` and the
task's ``expected_contract``. It does NOT call any LLM.

Two protocols, one call site
----------------------------
Two ``AnalyzerJudge`` protocols exist in the core and they are not compatible:

* **Legacy / trace-based** -- :class:`AnalyzerJudge` in this module.
  ``analyze(task, trace) -> CausalAnalysis``. Every orchestrator call site
  invokes this one. It conflates diagnosis with measurement (``CausalAnalysis``
  carries a ``score``) and cannot express "I could not tell".
* **Report-based** -- :class:`agent_evolve.core.analysis.AnalyzerJudge`.
  ``analyze(report) -> tuple[CausalFinding, ...]``. Rollout-group aware, carries
  ``status``/``confidence`` so abstention is expressible, and carries no score,
  so judging stays separate from diagnosing.

The report-based contract is the target. Rather than rewrite every call site at
once, :func:`as_legacy_analyzer` adapts a report-based analyzer to the legacy
call signature: it builds a sanitized report via
:func:`~agent_evolve.core.evidence.rollout_group_report`, calls the analyzer, and
projects the returned finding back with
:func:`~agent_evolve.core.blame.analysis_from_finding` -- supplying the score
from the caller's own contract evaluation, never from the analyzer.

Protocol detection is **structural** (:func:`is_report_analyzer` inspects the
``analyze`` signature). It is deliberately not a ``try``/``except`` around a real
call: a ``TypeError`` raised from deep inside a live analyzer would be
indistinguishable from a signature mismatch, and the retry would double-charge a
model call.

This module stays agent-neutral: it imports only core protocols and types.
"""
from __future__ import annotations

import inspect
import re
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol, Sequence

from agent_evolve.core.analysis import RolloutGroupReport
from agent_evolve.core.blame import (
    BlameEdge,
    BlameGraph,
    BlameNode,
    CausalAnalysis,
    CausalFinding,
    abstained_analysis,
    analysis_from_finding,
)
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace
from agent_evolve.core.evidence import rollout_group_report
from agent_evolve.core.parallel_analysis import (
    AnalysisOutcome,
    ParallelAnalysisRunner,
)


class AnalyzerJudge(Protocol):
    """Maps a (task, trace) to a causal analysis verdict."""

    analyzer_model_id: str
    judge_model_id: str

    def analyze(
        self,
        task: EvolutionTask,
        trace: ExecutionTrace,
    ) -> CausalAnalysis: ...


class ReportAnalyzerJudge(Protocol):
    """Maps a rollout-group report to trace-backed findings.

    Structurally identical to :class:`agent_evolve.core.analysis.AnalyzerJudge`;
    restated here so this module can name both sides of the shim.
    """

    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]: ...


class PositivityJudge(Protocol):
    """D5/J2B: maps a SUCCESSFUL (task, trace) to strength findings.

    The mirror of :class:`AnalyzerJudge`: Judge 1 diagnoses failures and may
    only emit ``valence=+1`` findings; a positivity judge diagnoses successes
    and may only emit ``valence=-1``. Polarity is stamped by code at the
    runner's boundary -- an implementation returning any other sign has its
    whole batch refused and recorded, never flipped.

    ``core/`` never imports an adapter; the CUGA implementation lives in
    ``agent_evolve.adapters`` behind exactly this protocol.
    """

    analyzer_model_id: str

    def analyze_success(
        self,
        task: EvolutionTask,
        trace: ExecutionTrace,
    ) -> tuple[CausalFinding, ...]: ...


@dataclass(frozen=True, slots=True)
class FakePositivityJudge:
    """Deterministic offline positivity judge.

    Mirrors :class:`FakeAnalyzerJudge`'s role for successes: one observed
    strength finding per passing rollout, attributing the trace's actors.
    Deliberately simple -- the interesting behaviour (gate wiring, polarity
    refusal, storage) is pinned in the runner tests.
    """

    analyzer_model_id: str = "fake-positivity"

    def analyze_success(
        self,
        task: EvolutionTask,
        trace: ExecutionTrace,
    ) -> tuple[CausalFinding, ...]:
        actor_list = sorted({e.actor_id for e in trace.events if e.actor_id})
        if not actor_list:
            actor_list = ["agent"]
        # Observed findings must cite trace-backed evidence: prefer the
        # events' own ids; an event-less trace cites its trace id, which is
        # still a genuine pointer into stored evidence.
        evidence = tuple(
            e.event_id for e in trace.events if getattr(e, "event_id", "")
        ) or (trace.trace_id,)
        nodes = tuple(
            BlameNode(actor_id=a, blame=1.0 / len(actor_list), artifacts=())
            for a in actor_list
        )
        finding = CausalFinding(
            verdict_id=f"strength-{trace.trace_id}",
            candidate_id=trace.candidate_id,
            task_id=task.task_id,
            trace_id=trace.trace_id,
            valence=-1,
            status="observed",
            mechanism_description=(
                f"contract satisfied on task {task.task_id} by "
                + ", ".join(actor_list)
            ),
            # Provisional identity: real per-task clusters are assigned when
            # the signed index (IDX2) builds; the observed-status contract
            # requires a non-empty id already.
            mechanism_cluster_id=f"strength:{task.task_id}",
            severity=0.8,
            confidence=0.9,
            blame_graph=BlameGraph(nodes=nodes),
            evidence_refs=evidence,
            rationale="fake positivity judge: success attributed to actors",
        )
        return (finding,)



# ---------------------------------------------------------------------- #
# Contract scoring
# ---------------------------------------------------------------------- #
def contract_score(task: EvolutionTask, trace: ExecutionTrace) -> float:
    """Measure a rollout against its task contract.

    This is the *measurement* half of what the legacy ``CausalAnalysis``
    conflates with diagnosis. It is factored out of :class:`FakeAnalyzerJudge` so
    the shim can score a rollout without asking a report-based analyzer for a
    score it deliberately does not produce.

    * ``expected_substring`` present in ``final_output`` -> 1.0
    * else ``expected_regex`` matches ``final_output`` -> 1.0
    * else 0.0 (including a task with no contract: an unlabeled task cannot be
      scored, and claiming success would be worse than claiming failure)
    """
    substring = task.expected_contract.get("expected_substring")
    if substring is not None and str(substring) in trace.final_output:
        return 1.0
    regex = task.expected_contract.get("expected_regex")
    if regex is not None and re.search(str(regex), trace.final_output):
        return 1.0
    return 0.0


@dataclass(slots=True)
class FakeAnalyzerJudge:
    """Deterministic offline analyzer+judge.

    Scoring rule
    ------------
    * If ``task.expected_contract["expected_substring"]`` is set and appears
      in ``trace.final_output``, score = 1.0 (success).
    * Else if ``task.expected_contract["expected_regex"]`` is set and matches
      ``trace.final_output``, score = 1.0.
    * Else score = 0.0 (failure).

    Blame assignment
    ----------------
    * On success: empty blame graph, severity 0.0.
    * On failure: blame is split across all actors that appear in the trace
      events. The first actor (by event order) gets 0.6 blame, the rest get
      equal shares of 0.4. Severity is 1.0.

    This is deliberately simple and deterministic; production analyzer+judge
    implementations will be backed by LLMs.
    """

    analyzer_model_id: str = "fake-analyzer"
    judge_model_id: str = "fake-judge"

    def analyze(self, task: EvolutionTask, trace: ExecutionTrace) -> CausalAnalysis:
        if contract_score(task, trace) == 1.0:
            return CausalAnalysis(
                mechanism="none",
                severity=0.0,
                score=1.0,
                blame_graph=BlameGraph(nodes=()),
                analyzer_model_id=self.analyzer_model_id,
                judge_model_id=self.judge_model_id,
            )

        # Failure: blame actors from the trace.
        actors: list[str] = []
        for e in trace.events:
            if e.actor_id and e.actor_id not in actors:
                actors.append(e.actor_id)
        if not actors:
            actors = ["unknown"]

        nodes: list[BlameNode] = []
        if len(actors) == 1:
            nodes.append(BlameNode(actor_id=actors[0], blame=1.0, artifacts=()))
        else:
            nodes.append(BlameNode(actor_id=actors[0], blame=0.6, artifacts=()))
            rest_share = 0.4 / (len(actors) - 1)
            for a in actors[1:]:
                nodes.append(BlameNode(actor_id=a, blame=rest_share, artifacts=()))

        edges: list[BlameEdge] = []
        for i in range(len(actors) - 1):
            edges.append(
                BlameEdge(
                    from_actor=actors[i],
                    to_actor=actors[i + 1],
                    mechanism=f"chain-{i}",
                )
            )

        return CausalAnalysis(
            mechanism=f"trace-{trace.trace_id}-failed-to-match",
            severity=1.0,
            score=0.0,
            blame_graph=BlameGraph(nodes=tuple(nodes), edges=tuple(edges)),
            analyzer_model_id=self.analyzer_model_id,
            judge_model_id=self.judge_model_id,
        )


# ---------------------------------------------------------------------- #
# Protocol detection
# ---------------------------------------------------------------------- #
def is_report_analyzer(analyzer: object) -> bool:
    """True for the report-based protocol, False for the legacy (task, trace) one.

    Detection is by ``analyze`` arity, inspected without calling it. A
    ``try``/``except TypeError`` around a live call would be wrong twice over: a
    ``TypeError`` raised *inside* a working analyzer would be misread as a
    signature mismatch, and the fallback call would charge a second model
    invocation for the same rollout.

    Raises ``TypeError`` when ``analyzer`` has no ``analyze`` method or its arity
    matches neither protocol. An unrecognised analyzer is a wiring bug and must
    fail at wiring time, not silently take a branch.
    """
    analyze = getattr(analyzer, "analyze", None)
    if analyze is None or not callable(analyze):
        raise TypeError(
            f"{type(analyzer).__name__} has no callable analyze method; it "
            "implements neither AnalyzerJudge protocol"
        )
    try:
        signature = inspect.signature(analyze)
    except (TypeError, ValueError) as exc:  # builtins / C callables
        raise TypeError(
            f"cannot inspect {type(analyzer).__name__}.analyze to detect its "
            f"analyzer protocol: {exc}"
        ) from exc

    positional = [
        parameter
        for parameter in signature.parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and parameter.default is inspect.Parameter.empty
    ]
    if len(positional) == 1:
        return True
    if len(positional) == 2:
        return False
    raise TypeError(
        f"{type(analyzer).__name__}.analyze takes {len(positional)} required "
        "positional arguments; expected 1 (report-based) or 2 (task, trace)"
    )


# ---------------------------------------------------------------------- #
# The shim
# ---------------------------------------------------------------------- #
#: Judge identity for a report-based analyzer. The report contract separates
#: diagnosis from judging and names no judge model, so the shim does not invent
#: one; the score comes from ``score_fn``, which is what actually judged.
_SHIM_JUDGE_MODEL_ID = "contract-scorer"


@dataclass(slots=True)
class ReportAnalyzerShim:
    """Adapts a report-based analyzer to the legacy ``(task, trace)`` call site.

    Per call it:

    1. Builds a sanitized single-rollout :class:`RolloutGroupReport` via
       :func:`~agent_evolve.core.evidence.rollout_group_report`, so the analyzer
       never sees ``trace.final_output`` or ``task.expected_contract``.
    2. Calls ``analyzer.analyze(report)``.
    3. Scores the rollout with ``score_fn`` -- the shim's own measurement, not
       the analyzer's -- and projects the returned finding onto a
       :class:`CausalAnalysis` with :func:`analysis_from_finding`.

    Errors from the analyzer propagate. A model outage is not an abstention: the
    report protocol expresses abstention as a ``status``, and swallowing an
    exception into ``insufficient_evidence`` would make an infrastructure failure
    look like an honest "I could not tell".

    Zero findings *is* treated as an abstention, because an analyzer that
    returned an empty tuple made no claim about the rollout.
    """

    analyzer: ReportAnalyzerJudge
    score_fn: Callable[[EvolutionTask, ExecutionTrace], float] = contract_score
    max_events_per_trace: int = 50

    @property
    def analyzer_model_id(self) -> str:
        return str(getattr(self.analyzer, "analyzer_model_id", "") or "")

    @property
    def judge_model_id(self) -> str:
        return _SHIM_JUDGE_MODEL_ID

    def analyze(self, task: EvolutionTask, trace: ExecutionTrace) -> CausalAnalysis:
        report = rollout_group_report(
            task, trace, max_events_per_trace=self.max_events_per_trace
        )
        findings = tuple(self.analyzer.analyze(report))
        score = float(self.score_fn(task, trace))

        if not findings:
            return abstained_analysis(
                "insufficient_evidence",
                score=score,
                evidence=("the analyzer returned no finding for this rollout",),
                analyzer_model_id=self.analyzer_model_id,
                judge_model_id=self.judge_model_id,
            )
        if len(findings) > 1:
            raise ValueError(
                f"a single-rollout report must yield one finding, got "
                f"{len(findings)} for trace {trace.trace_id}; the legacy call "
                "site cannot represent more than one verdict without discarding "
                "evidence -- use analyze_groups for multi-rollout analysis"
            )
        return analysis_from_finding(
            findings[0],
            score=score,
            analyzer_model_id=self.analyzer_model_id,
            judge_model_id=self.judge_model_id,
        )


def as_legacy_analyzer(
    analyzer: object,
    *,
    score_fn: Callable[[EvolutionTask, ExecutionTrace], float] = contract_score,
    max_events_per_trace: int = 50,
) -> AnalyzerJudge:
    """Return ``analyzer`` if it is already legacy, else wrap it in a shim.

    A legacy analyzer is returned *by identity*, not re-wrapped, so existing
    behaviour is unchanged: the same object, the same ``analyze``, the same
    verdicts.
    """
    if not is_report_analyzer(analyzer):
        return analyzer  # type: ignore[return-value]
    return ReportAnalyzerShim(
        analyzer=analyzer,  # type: ignore[arg-type]
        score_fn=score_fn,
        max_events_per_trace=max_events_per_trace,
    )


# ---------------------------------------------------------------------- #
# Group analysis
# ---------------------------------------------------------------------- #
def analyze_groups(
    analyzer_factory: Callable[[], ReportAnalyzerJudge],
    groups: Sequence[tuple[EvolutionTask, Sequence[ExecutionTrace]]],
    *,
    max_workers: int = 1,
    max_events_per_trace: int = 50,
) -> tuple[AnalysisOutcome, ...]:
    """Analyze many (task, rollout-group) pairs with bounded concurrency.

    This is the entry point the report-based protocol exists for: it forwards a
    whole rollout group to one ``analyze`` call, so an analyzer can reason about
    cross-rollout variance instead of seeing one trace at a time.

    ``analyzer_factory`` is a zero-arg builder, not an analyzer:
    :class:`~agent_evolve.core.parallel_analysis.ParallelAnalysisRunner` builds
    one instance per worker thread because a stateful agent shared across threads
    would interleave two trajectories into one conversation.

    Outcomes come back in input order with per-group failure isolation; see
    :class:`~agent_evolve.core.parallel_analysis.AnalysisOutcome`.
    """
    if not groups:
        return ()
    probe = analyzer_factory()
    if not is_report_analyzer(probe):
        raise TypeError(
            f"analyze_groups needs a report-based analyzer; "
            f"{type(probe).__name__} implements the legacy (task, trace) "
            "protocol and cannot see a rollout group"
        )
    reports = tuple(
        rollout_group_report(
            task, tuple(traces), max_events_per_trace=max_events_per_trace
        )
        for task, traces in groups
    )
    runner = ParallelAnalysisRunner(
        analyzer_factory=analyzer_factory, max_workers=max_workers
    )
    return runner.run(reports)
