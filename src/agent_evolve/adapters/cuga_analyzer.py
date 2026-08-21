"""LLM-backed trajectory analyzer emitting semantic mechanisms and blame graphs.

This is an adapter, not core: it makes model calls. It implements the core
:class:`~agent_evolve.core.analysis.AnalyzerJudge` protocol
(``analyze(report) -> tuple[CausalFinding, ...]``) and imports nothing from
CUGA's agent surface -- only ``RuntimeSettings`` for connection settings.

Why this exists
---------------
The placeholder analyzer emits one mechanism string per task
(``failed-to-match-<task_id>``). Every failure on a task therefore lands in the
same mechanism, so mechanism clustering, entropy cells, and DPP
mechanism-diversity cannot distinguish two different root causes: they are
degenerate by construction. This module's only reason to exist is to produce a
*specific causal sentence* per trace, grounded in trace evidence.

Design
------
* **One model call per report.** A report may carry several traces; the model
  returns one finding per trace in a single JSON response. The call site is a
  single injected callable (``completion_fn``) so it can be swapped for a
  multi-turn CUGA agent later without touching the parse/validation layer.
* **Thread-safe per-instance reuse.** ``analyze`` mutates nothing except a
  lazily cached, idempotent request base, so
  :class:`~agent_evolve.core.parallel_analysis.ParallelAnalysisRunner` can build
  one analyzer per worker thread via :meth:`CugaTrajectoryAnalyzer.factory` and
  call it repeatedly.
* **Grounding over fluency.** Actor IDs must appear in the report's evidence,
  event IDs must exist, and an artifact is only attached to a blame node when
  its ID literally appears in the sanitized evidence. Anything the model invents
  is dropped and the drop is recorded in the finding's rationale. A blame node
  is never manufactured for an actor with no trace evidence.
* **Abstention is a first-class outcome.** Thin evidence maps to
  ``insufficient_evidence``; a model response we cannot trust maps to
  ``malformed``. Neither invents a mechanism.

Failure-handling split (deliberate)
-----------------------------------
Problems *in the model's response* become findings with a non-``observed``
status, because they are information about that trajectory's analysis. Problems
*reaching the model* (transport errors, missing credentials) propagate as
exceptions, because they say nothing about the trajectory; ``ParallelAnalysisRunner``
records them as ``ok=False`` outcomes so they stay visible instead of being
laundered into per-trace "malformed" verdicts.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable

from agent_evolve.core.analysis import RolloutGroupReport
from agent_evolve.core.blame import (
    BlameEdge,
    BlameGraph,
    BlameNode,
    CausalFinding,
)
from agent_evolve.core.run_logging import RunLogSink

# ---------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------- #

#: Placeholder cluster ID. ``status="observed"`` requires a non-null
#: ``mechanism_cluster_id``, but real cluster assignment is the downstream
#: clusterer's job (see :mod:`agent_evolve.core.clustering`, which produces
#: ``"<task_id>:c<N>"``). This analyzer therefore stamps every observed finding
#: with the same explicit "not yet clustered" sentinel: a value that is
#: obviously not a cluster is safer than a plausible-looking fake one, and a
#: caller that forgets to re-stamp it produces a visibly degenerate grouping
#: rather than a silently wrong one.
UNASSIGNED_MECHANISM_CLUSTER_ID = "mechanism-cluster-unassigned"

_VALID_STATUSES = frozenset(
    {"observed", "uncertain", "insufficient_evidence", "malformed"}
)

#: Lenient synonyms for the abstention statuses a model tends to invent.
_STATUS_ALIASES = {
    "observed": "observed",
    "confirmed": "observed",
    "uncertain": "uncertain",
    "unsure": "uncertain",
    "ambiguous": "uncertain",
    "insufficient_evidence": "insufficient_evidence",
    "insufficient-evidence": "insufficient_evidence",
    "insufficient": "insufficient_evidence",
    "abstain": "insufficient_evidence",
    "no_evidence": "insufficient_evidence",
    "unknown": "insufficient_evidence",
    "malformed": "malformed",
}

#: A mechanism must be a causal sentence, not a label. Anything shorter than
#: this many words cannot describe a step and its consequence.
_MIN_MECHANISM_WORDS = 6

#: Whole-phrase degenerate mechanisms. Matched with ``fullmatch`` against the
#: normalized mechanism so a genuinely causal sentence that merely *starts* with
#: one of these phrasings ("the agent failed to call load_skill, so ...") is not
#: rejected.
_DEGENERATE_MECHANISMS = tuple(
    re.compile(pattern)
    for pattern in (
        r"failed[- ]to[- ]match([- ]\S+)*",
        r"(the )?(agent|model|planner|executor)?\s*(task )?fail(ed|ure)( the task)?",
        r"(tool|planning|execution|reasoning|retrieval|memory) (error|failure|issue|problem)s?",
        r"(incorrect|wrong|bad) (answer|output|result|response)",
        r"unknown( error| cause| failure)?",
        r"(the )?output (did not|didn't) match( the)?( expected)?( output| answer| result)?",
        r"none",
        r"n/?a",
    )
)

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

_SYSTEM_PROMPT = """\
You are a trajectory failure analyst for an agent-evolution research system.

You are given sanitized evidence from one or more rollouts of a single agent
harness on a single task. You must explain, per rollout, WHY the trajectory
went wrong, and attribute blame to the specific actors visible in the evidence.

You never see the expected answer and you never see the agent's final output.
Do not speculate about what the correct answer was. Reason only about the
observable causal structure of the trajectory: which actor did what, what it
omitted, what it passed downstream, and what consequence followed.

THE HARNESS YOU ARE LOOKING AT
------------------------------
The rollout agent is a SINGLE-AGENT ReAct/CodeAct loop, not a hierarchy of
planners. The graph is:

  CugaLiteSubgraph -> prepare -> call_model <-> sandbox -> SDKCallback
                                                       -> FinalAnswerAgent

* prepare assembles the turn's context (always-on instructions, any loaded
  skill or policy text, retrieved memory) before the model is called.
* call_model produces one assistant message. If that message contains a fenced
  code block, the harness extracts it and runs it in sandbox, and the loop
  returns to call_model with the execution output. If it contains NO extractable
  code block, there is nothing to run: the loop exits toward the final answer
  without any tool ever being invoked.
* sandbox is the ONLY place a tool can be called. A tool that is never named
  inside executed code is never called, whatever the assistant message claims.
* FinalAnswerAgent produces the terminal answer from whatever context exists at
  that point, including none.

Actor ids in the evidence are graph node names from this shape. There is no
separate planner actor and no plan-controller actor on this path; do not
attribute blame to a stage that does not appear in the evidence.

READING THE EVIDENCE HONESTLY
-----------------------------
* The model's own words are a CLAIM, not an observation. Statements like "I am
  unable to call that tool", "the tool returned nothing", "I don't have access
  to the web" are self-reports and are frequently FALSE. A self-report must be
  corroborated against the event stream before you build a mechanism on it: a
  tool failure requires an actual tool_call event whose payload shows the
  failure. If no tool_call event exists, the tool was not attempted, and the
  cause lies upstream of the tool.
* Absence of an actor is different from failure of an actor. An actor with no
  events did not run.
* payload_redacted marks a payload that was withheld, not a payload that was
  empty. Do not read a redaction as evidence of nothing happening.

FAILURE PATTERNS WORTH RECOGNISING
----------------------------------
These are recurring shapes, not a taxonomy to label with. When one fits, still
write a specific causal sentence about THIS trace.

1. NO EXECUTABLE CODE EMITTED. The assistant message narrates an intention
   ("first I will fetch the page, then I will count them"), contains no fenced
   code block, and often ends with an inability claim or an apology. Signature
   in the evidence: zero tool_call events for the whole trace, the loop never
   re-enters call_model from sandbox, and the run reaches the final answer in
   one model turn. The mechanism here is that the turn produced narration
   instead of runnable code, so no capability was ever exercised. This is a
   failure of the turn-level output contract the model was operating under -- it
   is NOT a tool failure, NOT a missing tool, and NOT a missing capability, even
   when the model says it is.
2. CODE EXECUTED BUT WRONG. tool_call events exist and show wrong arguments,
   the wrong tool for the goal, or results that were ignored. Blame belongs to
   the step that produced or consumed the bad values.
3. CAPABILITY PRESENT BUT NEVER SELECTED. An optional capability exists and the
   model had to opt into it (for instance by calling a loader tool) and did not,
   so its content was absent from context at answer time. Distinguish this from
   pattern 1: here code ran, it simply never selected the thing.
4. LOOP EXHAUSTION OR PREMATURE TERMINATION. Repeated equivalent calls with no
   progress, or a jump to the final answer while the evidence shows the needed
   information had not been obtained yet.

MAKE THE MECHANISM ACTIONABLE
-----------------------------
Downstream, an editor can only change these four surfaces of the harness:

* instructions -- always-present, unconditional per-turn behavioral
  configuration; governs how every turn is conducted.
* skills -- optional procedures the model must choose to load; govern how a
  multi-step job is carried out once selected.
* policies -- conditional guidance that applies only when the request matches
  its trigger.
* memory -- retrievable facts and context, not behavior.

A mechanism that no surface can act on produces no edit, so the analysis is
wasted. Prefer the most specific mechanism the evidence supports, and locate it
where an editor could plausibly intervene: which stage of the loop misbehaved,
what the harness's own configuration led it to do there, and what followed. Do
NOT name a surface or prescribe a remedy in the mechanism -- choosing the fix is
the editor's job. Your obligation is that the mechanism be concrete enough for
that choice to be possible.

Answer with a single JSON object and nothing else. Schema:

{
  "findings": [
    {
      "trace_id": "<the trace_id this finding is about; must match the evidence>",
      "status": "observed" | "uncertain" | "insufficient_evidence",
      "mechanism": "<ONE specific causal sentence; see MECHANISM RULES>",
      "severity": <number in [0,1]: how much this broke the task>,
      "confidence": <number in [0,1]: how sure you are of the attribution>,
      "rationale": "<why the evidence supports this, citing event ids>",
      "blamed_actors": [
        {
          "actor_id": "<must be one of the actors listed for this trace>",
          "blame": <number in [0,1]>,
          "artifacts": ["<only artifact names that literally appear in the evidence>"],
          "why": "<optional short note>"
        }
      ],
      "causal_links": [
        {"from_actor": "<blamed actor>", "to_actor": "<blamed actor>",
         "mechanism": "<what propagated from one to the other>"}
      ],
      "evidence_event_ids": ["<event_id values copied verbatim from the evidence>"],
      "counterfactual_notes": ["<optional: what would plausibly have changed the outcome>"]
    }
  ]
}

MECHANISM RULES (this is the point of the whole task):
- The mechanism must be a specific causal sentence naming the concrete step
  that went wrong AND the consequence that followed. Shape:
  "<actor> <did or omitted something concrete>, so <what consequence followed>".
- GOOD: "the planner emitted code that never called load_skill, so the required
  skill body was never in the model's context when it answered".
- GOOD: "the api_agent called search_flights with the destination string in the
  origin field, so every returned itinerary started from the wrong city".
- GOOD: "call_model answered with a prose description of the steps it intended
  to take and no runnable block, so sandbox never executed and no tool was
  reached before FinalAnswerAgent produced the answer".
- FORBIDDEN: category labels ("tool error", "planning failure", "bad output").
- FORBIDDEN: repeating a self-report as if it were an observation ("the tool was
  unavailable" when no tool_call event exists).
- FORBIDDEN: restatements of failure ("the agent failed the task", "the output
  did not match", "incorrect answer", "failed-to-match").
- FORBIDDEN: templated text that would read identically for a different root
  cause on the same task. Two different root causes must yield two clearly
  different mechanism sentences.

GROUNDING RULES:
- Use only actor_id values listed in that trace's "actors".
- Use only event_id values that appear in that trace's "events".
- Name an artifact only if its identifier literally appears in the evidence.
- If the evidence does not let you attribute a cause, set
  status "insufficient_evidence", leave blamed_actors empty, and say in the
  rationale exactly what evidence was missing. Do NOT invent a mechanism.
- Emit exactly one finding per trace_id present in the evidence.
"""


class AnalyzerConfigurationError(RuntimeError):
    """No model is configured and no completion callable was injected."""


# ---------------------------------------------------------------------- #
# The analyzer
# ---------------------------------------------------------------------- #
@dataclass(slots=True)
class CugaTrajectoryAnalyzer:
    """Report-based analyzer+judge backed by a single structured model call.

    Every field is optional, so ``CugaTrajectoryAnalyzer()`` is a valid
    ``analyzer_factory`` for :class:`ParallelAnalysisRunner`. Model, base URL and
    key default to ``RuntimeSettings.from_env()`` resolved lazily on the first
    call that needs them, so constructing an analyzer never requires credentials
    (the runner builds one per worker thread and would otherwise fail a whole
    batch at construction time).

    Parameters
    ----------
    completion_fn:
        Called as ``completion_fn(**request)`` and expected to return an
        OpenAI/litellm-shaped response. Defaults to a live ``litellm.completion``
        call. This is the single swap point for a future multi-turn CUGA agent.
    """

    completion_fn: Callable[..., object] | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    #: Forwarded only when set. Deliberately ``None``: a live run against
    #: ``azure/gpt-5.6-luna`` rejected the request outright with "Unsupported
    #: value: 'temperature' does not support 0.0 with this model. Only the
    #: default (1) value is supported." Pinning a value we prefer would make the
    #: analyzer unusable on such models, so the provider default stands unless a
    #: caller opts in. Determinism is therefore NOT guaranteed by default.
    temperature: float | None = None
    max_events_in_prompt: int = 40
    request_json_object: bool = False
    analyzer_model_id: str = "cuga-trajectory-analyzer"
    #: When set and active, one record per model call: the request messages, the
    #: raw response text, and the statuses the response produced. Off by default
    #: because a measurement run must be able to spend nothing on capture.
    log_sink: RunLogSink | None = None
    _request_base: dict[str, object] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def factory(cls, **kwargs: object) -> Callable[[], "CugaTrajectoryAnalyzer"]:
        """A zero-arg builder for ``ParallelAnalysisRunner.analyzer_factory``.

        Configuration is captured now; each worker thread gets its own instance,
        which matters once the call site becomes a stateful agent.
        """
        return lambda: cls(**kwargs)  # type: ignore[arg-type]

    # ------------------------------------------------------------------ #
    # Protocol surface
    # ------------------------------------------------------------------ #
    def analyze(self, report: RolloutGroupReport) -> tuple[CausalFinding, ...]:
        """One finding per trace in ``report``, in evidence order.

        Raises only if the model call itself cannot be made or fails; response
        content problems become non-``observed`` findings.
        """
        evidences = tuple(
            e for e in report.sanitized_evidence if isinstance(e, Mapping)
        )
        if not evidences:
            return ()

        response, messages = self._call_model(report, evidences)

        text = _response_text(response)
        if not text.strip():
            return self._captured(
                report,
                messages,
                text,
                self._all_malformed(
                    report, evidences, "model returned an empty response body"
                ),
            )

        parsed, parse_error = _parse_findings_payload(text)
        if parse_error is not None:
            return self._captured(
                report,
                messages,
                text,
                self._all_malformed(report, evidences, parse_error),
            )

        by_trace, unmatched = _index_by_trace(
            parsed, tuple(str(e.get("trace_id", "")) for e in evidences)
        )

        findings: list[CausalFinding] = []
        for evidence in evidences:
            trace_id = str(evidence.get("trace_id") or "")
            raw = by_trace.get(trace_id)
            if raw is None:
                findings.append(
                    self._malformed(
                        report,
                        trace_id or "unknown-trace",
                        "model response carried no finding for this trace"
                        + (
                            f"; unmatched trace ids in response: {sorted(unmatched)}"
                            if unmatched
                            else ""
                        ),
                    )
                )
                continue
            findings.append(self._build_finding(report, evidence, raw))
        return self._captured(report, messages, text, tuple(findings))

    # ------------------------------------------------------------------ #
    # Transcript capture
    # ------------------------------------------------------------------ #
    def _captured(
        self,
        report: RolloutGroupReport,
        messages: Sequence[Mapping[str, object]],
        response_text: str,
        findings: tuple[CausalFinding, ...],
    ) -> tuple[CausalFinding, ...]:
        """Record one judge transcript, then return ``findings`` unchanged.

        Placed on the return path of every branch so an abstention and a
        malformed response are as recoverable as a well-formed one. Returns the
        findings so a call site cannot record a transcript and forget the value.
        """
        self._write_transcript(
            report,
            {
                "event": "analyzer_call",
                "request_messages": [dict(m) for m in messages],
                "response_text": response_text,
                "finding_statuses": [f.status for f in findings],
            },
        )
        return findings

    def _write_transcript(
        self, report: RolloutGroupReport, record: Mapping[str, object]
    ) -> None:
        """Best-effort write. Never raises: an observer must not fail the run.

        A logging failure that killed an analysis would throw away a model call
        that has already been paid for, so every error here is swallowed --
        including a sink that does not behave like one.
        """
        sink = self.log_sink
        if sink is None:
            return
        try:
            sink.write_record(
                f"{report.candidate_id}__{report.task_id}",
                {
                    "candidate_id": report.candidate_id,
                    "task_id": report.task_id,
                    "analyzer_model_id": self.analyzer_model_id,
                    **dict(record),
                },
            )
        except Exception:  # noqa: BLE001 - capture is an observer, never a gate
            pass

    # ------------------------------------------------------------------ #
    # Model call
    # ------------------------------------------------------------------ #
    def _call_model(
        self,
        report: RolloutGroupReport,
        evidences: Sequence[Mapping[str, object]],
    ) -> tuple[object, tuple[Mapping[str, object], ...]]:
        """The response and the messages that produced it.

        The messages are returned rather than rebuilt for the transcript: a
        second ``_user_prompt`` call would trim the evidence again and could
        record a prompt the model never received.
        """
        request = dict(self._resolve_request_base())
        messages: tuple[Mapping[str, object], ...] = (
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(report, evidences)},
        )
        request["messages"] = [dict(m) for m in messages]
        invoke = self.completion_fn or _litellm_completion
        try:
            return invoke(**request), messages
        except Exception as exc:
            # The request is the only artifact a failed call can leave, and
            # without it an over-long prompt is indistinguishable from an
            # endpoint outage. The exception still propagates: a transport
            # failure says nothing about the trajectory.
            self._write_transcript(
                report,
                {
                    "event": "analyzer_call_failed",
                    "request_messages": [dict(m) for m in messages],
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
            raise

    def _resolve_request_base(self) -> dict[str, object]:
        """Model/connection settings, resolved once and reused.

        Idempotent, so concurrent first calls from two threads cannot produce an
        inconsistent request; the loser simply recomputes the same dict.
        """
        if self._request_base is not None:
            return self._request_base

        model, base_url, api_key = self.model, self.base_url, self.api_key
        if model is None or base_url is None or api_key is None:
            env_model, env_base, env_key = _env_settings()
            model = model or env_model
            base_url = base_url or env_base
            api_key = api_key or env_key
        if not model:
            raise AnalyzerConfigurationError(
                "no analyzer model configured: set CUGA_MODEL or LITELLM_MODEL, "
                "or pass model= explicitly"
            )

        request: dict[str, object] = {"model": model}
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if base_url:
            request["api_base"] = base_url
        if api_key:
            request["api_key"] = api_key
        if self.request_json_object:
            request["response_format"] = {"type": "json_object"}
        self._request_base = request
        return request

    def _user_prompt(
        self,
        report: RolloutGroupReport,
        evidences: Sequence[Mapping[str, object]],
    ) -> str:
        trimmed = [
            _trim_evidence(e, self.max_events_in_prompt) for e in evidences
        ]
        trace_ids = [str(e.get("trace_id") or "") for e in evidences]
        return (
            f"candidate_id: {report.candidate_id}\n"
            f"task_id: {report.task_id}\n"
            f"traces to analyze (one finding each): {trace_ids}\n"
            f"tool invocations actually observed:\n{_tool_call_census(evidences)}\n\n"
            "SANITIZED EVIDENCE (JSON):\n"
            f"{json.dumps(trimmed, indent=2, default=_jsonable)}\n\n"
            "Return the JSON object described in the schema. No prose outside it."
        )

    # ------------------------------------------------------------------ #
    # Finding construction
    # ------------------------------------------------------------------ #
    def _build_finding(
        self,
        report: RolloutGroupReport,
        evidence: Mapping[str, object],
        raw: Mapping[str, object],
    ) -> CausalFinding:
        trace_id = str(evidence.get("trace_id") or "unknown-trace")
        try:
            return self._build_finding_unsafe(report, evidence, raw, trace_id)
        except Exception as exc:  # noqa: BLE001 - a bad verdict is data, not a crash
            return self._malformed(
                report,
                trace_id,
                f"could not construct a valid finding from the model response: "
                f"{type(exc).__name__}: {exc}",
            )

    def _build_finding_unsafe(
        self,
        report: RolloutGroupReport,
        evidence: Mapping[str, object],
        raw: Mapping[str, object],
        trace_id: str,
    ) -> CausalFinding:
        notes: list[str] = []

        status = _normalize_status(raw.get("status"))
        if status is None:
            return self._malformed(
                report,
                trace_id,
                f"model returned an unrecognized status {raw.get('status')!r}",
            )

        rationale = _first_text(raw, ("rationale", "reason", "explanation"))
        mechanism = _first_text(raw, ("mechanism", "mechanism_description"))

        if status == "malformed":
            return self._malformed(
                report,
                trace_id,
                rationale or "model self-reported a malformed analysis",
            )

        # Scalars. Out-of-range or non-numeric values mean the response cannot
        # be trusted at all: a severity of 7 is not a rounding problem.
        severity, sev_err = _unit_number(raw.get("severity"), "severity")
        confidence, conf_err = _unit_number(raw.get("confidence"), "confidence")
        scalar_errors = [e for e in (sev_err, conf_err) if e]
        if scalar_errors:
            return self._malformed(
                report, trace_id, "; ".join(scalar_errors)
            )

        if status == "insufficient_evidence":
            return CausalFinding(
                verdict_id=self._verdict_id(report, trace_id),
                candidate_id=report.candidate_id,
                task_id=report.task_id,
                trace_id=trace_id,
                status="insufficient_evidence",
                rationale=rationale
                or "model abstained: evidence insufficient to attribute a cause",
            )

        if status == "observed" and not mechanism:
            return self._malformed(
                report,
                trace_id,
                "status=observed but the model supplied no mechanism sentence",
            )

        # Anti-degeneracy gate: a category label or a restatement of failure is
        # exactly the pathology this analyzer replaces, so it is not promoted to
        # an observed mechanism.
        if status == "observed" and _is_degenerate_mechanism(mechanism):
            status = "uncertain"
            notes.append(
                "downgraded from observed: mechanism was generic or a "
                "restatement of failure, not a causal sentence"
            )

        known_actors = _known_actors(evidence)
        known_events = _known_event_ids(evidence)
        evidence_blob = json.dumps(evidence, default=_jsonable)

        nodes, edges, node_notes, artifact_refs = _grounded_blame_graph(
            raw, known_actors, evidence_blob
        )
        notes.extend(node_notes)

        event_refs, ref_notes = _grounded_event_refs(
            raw, known_events, trace_id
        )
        notes.extend(ref_notes)

        evidence_refs = tuple(dict.fromkeys((*event_refs, *artifact_refs)))

        # An observed mechanism must be anchored to at least one real event.
        # Artifact refs alone are weaker grounding (a literal substring match
        # against the evidence blob), so they do not qualify a finding as
        # trace-backed on their own.
        if status == "observed" and not event_refs:
            status = "uncertain"
            notes.append(
                "downgraded from observed: the model cited no event id that "
                "exists in this trace, so the attribution is not trace-backed"
            )

        if status == "observed" and (severity is None or confidence is None):
            missing = [
                name
                for name, value in (("severity", severity), ("confidence", confidence))
                if value is None
            ]
            status = "uncertain"
            notes.append(
                "downgraded from observed: missing " + ", ".join(missing)
            )

        # Edges cannot survive without their endpoints; ``_grounded_blame_graph``
        # already drops any edge touching an ungrounded actor, so an empty node
        # set implies an empty edge set.
        graph = BlameGraph(nodes=nodes, edges=edges)

        full_rationale = _join_rationale(rationale, notes) or (
            "analyzer produced a finding with no model rationale"
        )

        return CausalFinding(
            verdict_id=self._verdict_id(report, trace_id),
            candidate_id=report.candidate_id,
            task_id=report.task_id,
            trace_id=trace_id,
            status=status,  # type: ignore[arg-type]
            mechanism_description=mechanism or None,
            mechanism_cluster_id=(
                UNASSIGNED_MECHANISM_CLUSTER_ID if status == "observed" else None
            ),
            severity=severity,
            confidence=confidence,
            blame_graph=graph,
            evidence_refs=evidence_refs,
            rationale=full_rationale,
            counterfactual_notes=_string_tuple(raw.get("counterfactual_notes")),
        )

    # ------------------------------------------------------------------ #
    # Small helpers
    # ------------------------------------------------------------------ #
    def _verdict_id(self, report: RolloutGroupReport, trace_id: str) -> str:
        return f"verdict-{report.candidate_id}-{report.task_id}-{trace_id}"

    def _malformed(
        self, report: RolloutGroupReport, trace_id: str, reason: str
    ) -> CausalFinding:
        return CausalFinding(
            verdict_id=self._verdict_id(report, trace_id),
            candidate_id=report.candidate_id,
            task_id=report.task_id,
            trace_id=trace_id,
            status="malformed",
            rationale=reason or "malformed analyzer response",
        )

    def _all_malformed(
        self,
        report: RolloutGroupReport,
        evidences: Sequence[Mapping[str, object]],
        reason: str,
    ) -> tuple[CausalFinding, ...]:
        return tuple(
            self._malformed(
                report, str(e.get("trace_id") or "unknown-trace"), reason
            )
            for e in evidences
        )


# ---------------------------------------------------------------------- #
# Response parsing
# ---------------------------------------------------------------------- #
def _litellm_completion(**request: object) -> object:
    """Live model call. Imported lazily so unit tests stay offline.

    Attaches the ambient ``X-AE-*`` run correlation so the observability proxy can
    tie each captured call to its ``(candidate, task, rollout, phase)``. mitmproxy
    sees only socket bytes, and the addon strips these before the request goes
    upstream, so no vendor endpoint receives internal identifiers. Outside a
    correlation scope nothing is added and the call is unchanged.

    Caller-supplied ``extra_headers`` are merged into, never replaced, and the
    caller's own dict is not mutated.
    """
    import litellm

    from agent_evolve.core.correlation import correlation_headers

    if correlation := correlation_headers():
        supplied = request.get("extra_headers") or {}
        request = {
            **request,
            "extra_headers": {**supplied, **correlation},
        }
    return litellm.completion(**request)


def _response_text(response: object) -> str:
    """First assistant text from an OpenAI/litellm-shaped response.

    Returns ``""`` rather than raising for any unexpected shape; the caller maps
    an empty body to a malformed finding.
    """
    choices = (
        response.get("choices")
        if isinstance(response, Mapping)
        else getattr(response, "choices", None)
    )
    if isinstance(response, str):
        return response
    if not choices:
        return ""
    choice = choices[0]
    message = (
        choice.get("message")
        if isinstance(choice, Mapping)
        else getattr(choice, "message", None)
    )
    content = (
        message.get("content")
        if isinstance(message, Mapping)
        else getattr(message, "content", None)
    )
    if content is None:
        content = (
            choice.get("text")
            if isinstance(choice, Mapping)
            else getattr(choice, "text", None)
        )
    return "" if content is None else str(content)


def _parse_findings_payload(
    text: str,
) -> tuple[tuple[Mapping[str, object], ...], str | None]:
    """Extract the findings list, tolerating fences and shape variation.

    Returns ``(findings, None)`` or ``((), error_message)``.
    """
    candidate = text.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()
    else:
        # Some models prepend prose; fall back to the outermost JSON braces.
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start > 0 and end > start:
            candidate = candidate[start : end + 1]

    try:
        payload = json.loads(candidate)
    except (ValueError, TypeError) as exc:
        return (), f"model response was not valid JSON: {exc}"

    if isinstance(payload, Mapping):
        raw = payload.get("findings", payload.get("results"))
        if raw is None:
            # A single bare finding object is acceptable.
            raw = [payload]
    elif isinstance(payload, list):
        raw = payload
    else:
        return (), (
            f"model response JSON was a {type(payload).__name__}, expected an "
            "object with a 'findings' list"
        )

    if not isinstance(raw, list):
        return (), "model response 'findings' was not a list"
    findings = tuple(item for item in raw if isinstance(item, Mapping))
    if not findings:
        return (), "model response contained no finding objects"
    return findings, None


def _index_by_trace(
    findings: Sequence[Mapping[str, object]],
    trace_ids: Sequence[str],
) -> tuple[dict[str, Mapping[str, object]], set[str]]:
    """Map each trace_id to its finding; report ids that matched nothing.

    Single-trace reports accept a finding with a missing or wrong ``trace_id``:
    there is no ambiguity about what it describes, and rejecting it would throw
    away a valid analysis over a label. Multi-trace reports require an exact
    match, because guessing which trace a finding refers to would silently
    attribute a mechanism to the wrong rollout.
    """
    known = set(trace_ids)
    by_trace: dict[str, Mapping[str, object]] = {}
    unmatched: set[str] = set()

    for item in findings:
        tid = str(item.get("trace_id") or "")
        if tid in known and tid not in by_trace:
            by_trace[tid] = item
        else:
            unmatched.add(tid or "<missing>")

    if len(trace_ids) == 1 and not by_trace and findings:
        by_trace[trace_ids[0]] = findings[0]
        unmatched.discard(str(findings[0].get("trace_id") or "") or "<missing>")
    return by_trace, unmatched


# ---------------------------------------------------------------------- #
# Grounding
# ---------------------------------------------------------------------- #
def _known_actors(evidence: Mapping[str, object]) -> frozenset[str]:
    actors: set[str] = set()
    listed = evidence.get("actors")
    if isinstance(listed, (list, tuple)):
        actors.update(str(a) for a in listed if a)
    for event in _events(evidence):
        actor = event.get("actor_id")
        if actor:
            actors.add(str(actor))
    return frozenset(actors)


def _known_event_ids(evidence: Mapping[str, object]) -> frozenset[str]:
    return frozenset(
        str(event.get("event_id"))
        for event in _events(evidence)
        if event.get("event_id")
    )


def _events(evidence: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    events = evidence.get("events")
    if not isinstance(events, (list, tuple)):
        return ()
    return tuple(e for e in events if isinstance(e, Mapping))


def _grounded_blame_graph(
    raw: Mapping[str, object],
    known_actors: frozenset[str],
    evidence_blob: str,
) -> tuple[tuple[BlameNode, ...], tuple[BlameEdge, ...], list[str], tuple[str, ...]]:
    """Build a blame graph containing only trace-backed actors and artifacts.

    Returns ``(nodes, edges, notes, artifact_refs)``. ``artifact_refs`` are the
    artifact IDs that survived grounding and must therefore be present in
    ``evidence_refs`` for an ``observed`` finding to validate.
    """
    notes: list[str] = []
    listed = raw.get("blamed_actors")
    if not isinstance(listed, (list, tuple)):
        listed = ()

    nodes: list[BlameNode] = []
    seen: set[str] = set()
    artifacts_kept: list[str] = []
    dropped_actors: list[str] = []
    dropped_artifacts: list[str] = []
    dropped_weights: list[str] = []

    for item in listed:
        if not isinstance(item, Mapping):
            continue
        actor_id = str(item.get("actor_id") or item.get("actor") or "").strip()
        if not actor_id:
            continue
        if actor_id not in known_actors:
            dropped_actors.append(actor_id)
            continue
        if actor_id in seen:
            continue
        blame, blame_err = _unit_number(item.get("blame"), "blame")
        if blame_err or blame is None:
            dropped_weights.append(actor_id)
            continue

        kept: list[str] = []
        for artifact in _string_tuple(item.get("artifacts")):
            if artifact and artifact in evidence_blob:
                kept.append(artifact)
            elif artifact:
                dropped_artifacts.append(artifact)

        seen.add(actor_id)
        artifacts_kept.extend(kept)
        nodes.append(
            BlameNode(
                actor_id=actor_id,
                blame=blame,
                artifacts=tuple(dict.fromkeys(kept)),
            )
        )

    if dropped_actors:
        notes.append(
            "dropped blame for actors absent from the trace evidence: "
            + ", ".join(sorted(set(dropped_actors)))
        )
    if dropped_weights:
        notes.append(
            "dropped blame nodes with an invalid weight: "
            + ", ".join(sorted(set(dropped_weights)))
        )
    if dropped_artifacts:
        notes.append(
            "dropped artifacts not named anywhere in the trace evidence: "
            + ", ".join(sorted(set(dropped_artifacts)))
        )

    node_ids = {n.actor_id for n in nodes}
    edges: list[BlameEdge] = []
    dropped_edges = 0
    links = raw.get("causal_links")
    for item in links if isinstance(links, (list, tuple)) else ():
        if not isinstance(item, Mapping):
            continue
        src = str(item.get("from_actor") or item.get("from") or "").strip()
        dst = str(item.get("to_actor") or item.get("to") or "").strip()
        mech = str(item.get("mechanism") or "").strip()
        if src in node_ids and dst in node_ids and mech:
            edges.append(BlameEdge(from_actor=src, to_actor=dst, mechanism=mech))
        else:
            dropped_edges += 1
    if dropped_edges:
        notes.append(
            f"dropped {dropped_edges} causal link(s) referencing an ungrounded "
            "actor or carrying no mechanism"
        )

    return (
        tuple(nodes),
        tuple(edges),
        notes,
        tuple(dict.fromkeys(artifacts_kept)),
    )


def _grounded_event_refs(
    raw: Mapping[str, object],
    known_events: frozenset[str],
    trace_id: str,
) -> tuple[tuple[str, ...], list[str]]:
    """Turn cited event ids into ``<trace_id>#<event_id>`` references.

    Event refs are namespaced by trace so a multi-trace group cannot conflate
    two rollouts' events; artifact refs stay bare because
    :class:`CausalFinding` matches blame-node artifacts against ``evidence_refs``
    by exact string.
    """
    notes: list[str] = []
    cited = _string_tuple(
        raw.get("evidence_event_ids")
        or raw.get("evidence_refs")
        or raw.get("event_ids")
    )
    kept: list[str] = []
    dropped: list[str] = []
    for event_id in cited:
        bare = event_id.split("#")[-1]
        if bare in known_events:
            kept.append(f"{trace_id}#{bare}")
        else:
            dropped.append(event_id)
    if dropped:
        notes.append(
            "dropped cited event ids that do not exist in this trace: "
            + ", ".join(sorted(set(dropped)))
        )
    return tuple(dict.fromkeys(kept)), notes


# ---------------------------------------------------------------------- #
# Value helpers
# ---------------------------------------------------------------------- #
def _normalize_status(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    key = value.strip().lower().replace(" ", "_")
    if key in _VALID_STATUSES:
        return key
    return _STATUS_ALIASES.get(key)


def _unit_number(value: object, name: str) -> tuple[float | None, str | None]:
    """Parse a [0, 1] number. Returns ``(value, error)``; both may be None."""
    if value is None:
        return None, None
    if isinstance(value, bool):
        return None, f"{name} was a boolean, expected a number in [0, 1]"
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None, f"{name} was not a number: {value!r}"
    if not isinstance(value, (int, float)):
        return None, f"{name} was a {type(value).__name__}, expected a number"
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        return None, f"{name} was not finite: {number}"
    if not (0.0 <= number <= 1.0):
        return None, f"{name} was out of range [0, 1]: {number}"
    return number, None


def _is_degenerate_mechanism(mechanism: str) -> bool:
    """True when the mechanism is a label or a restatement, not a cause."""
    text = mechanism.strip()
    if not text:
        return True
    if len(text.split()) < _MIN_MECHANISM_WORDS:
        return True
    normalized = re.sub(r"[^a-z0-9/\s-]+", "", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return any(p.fullmatch(normalized) for p in _DEGENERATE_MECHANISMS)


def _first_text(raw: Mapping[str, object], keys: Sequence[str]) -> str:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _string_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value.strip(),) if value.strip() else ()
    if isinstance(value, (list, tuple)):
        return tuple(
            str(item).strip()
            for item in value
            if item is not None and str(item).strip()
        )
    return ()


def _join_rationale(rationale: str, notes: Sequence[str]) -> str:
    parts = [p for p in (rationale.strip(), *notes) if p]
    return " | ".join(parts)


def _tool_call_census(evidences: Sequence[Mapping[str, object]]) -> str:
    """One line per trace stating how many tool_call events it really has.

    Derived only from event ``kind`` values already present in the sanitized
    evidence, so it exposes nothing the model cannot see -- but it removes the
    step the judge was measured to get wrong. A zero here is the discriminating
    observation between "the model could not use a tool" (a claim) and "no tool
    was ever invoked" (a fact), and asking a model to count occurrences inside a
    multi-kilobyte JSON blob is exactly the operation it does unreliably.
    """
    lines: list[str] = []
    for evidence in evidences:
        trace_id = str(evidence.get("trace_id") or "unknown-trace")
        count = sum(
            1 for e in _events(evidence) if str(e.get("kind")) == "tool_call"
        )
        note = (
            "  (zero: no tool was invoked in this trace, whatever the model said)"
            if count == 0
            else ""
        )
        lines.append(f"  {trace_id}: {count} tool_call event(s){note}")
    return "\n".join(lines)


def _trim_evidence(
    evidence: Mapping[str, object], max_events: int
) -> dict[str, object]:
    """Bound prompt size without hiding that events were cut."""
    trimmed = dict(evidence)
    events = _events(evidence)
    if max_events >= 0 and len(events) > max_events:
        trimmed["events"] = list(events[:max_events])
        trimmed["events_truncated"] = True
    return trimmed


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (set, frozenset, tuple)):
        return list(value)
    return repr(value)


def _env_settings() -> tuple[str | None, str | None, str | None]:
    """``(model, base_url, api_key)`` from the environment, blanks when absent.

    Imported lazily: :mod:`agent_evolve.cuga_wrapper` pulls in dotenv and the
    trace model, which an offline unit test that injects ``completion_fn`` and a
    model name has no reason to load.
    """
    try:
        from agent_evolve.cuga_wrapper import RuntimeSettings
    except Exception:  # noqa: BLE001 - absence of the wrapper is not fatal here
        return None, None, None
    try:
        settings = RuntimeSettings.from_env()
    except RuntimeError:
        return None, None, None
    return settings.model, settings.base_url, settings.api_key
