"""LLM-backed positivity judge emitting strengths for SUCCESSFUL rollouts.

D5/J2B adapter: implements the core
:class:`~agent_evolve.core.analyzer.PositivityJudge` protocol
(``analyze_success(task, trace) -> tuple[CausalFinding, ...]``). This is the
mirror of :mod:`agent_evolve.adapters.cuga_analyzer` -- it reuses that
module's grounding machinery wholesale (invented actors/events/artifacts are
dropped; abstention is first-class; transport errors propagate) and changes
three things:

* **The prompt asks only what went well**, and demands a specific causal
  sentence, not a category label.
* **Polarity is stamped by code.** The model is never asked for a sign and
  any client-supplied ``valence`` field in its payload is ignored: every
  finding this adapter emits carries ``valence=-1``, so a mis-prompted model
  cannot smuggle faults into the success side. The runner's receive wall
  refuses them anyway; this adapter simply never produces one.
* **Multiple strengths per rollout are allowed** -- one success may involve
  several actors doing the right thing. Each becomes its own finding.

Everything else follows ``cuga_analyzer.py``'s documented design: one model
call via an injectable ``completion_fn`` (live default: litellm), lazy
credential resolution from ``RuntimeSettings``, thread-safe per-instance
reuse, sanitized evidence only.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable

from agent_evolve.adapters.cuga_analyzer import (
    UNASSIGNED_MECHANISM_CLUSTER_ID,
    AnalyzerConfigurationError,
    CugaTrajectoryAnalyzer,
    _env_settings,
    _first_text,
    _grounded_blame_graph,
    _grounded_event_refs,
    _is_degenerate_mechanism,
    _jsonable,
    _known_actors,
    _known_event_ids,
    _litellm_completion,
    _normalize_status,
    _parse_findings_payload,
    _response_text,
    _tool_call_census,
    _trim_evidence,
    _unit_number,
)
from agent_evolve.core.analysis import RolloutGroupReport
from agent_evolve.core.blame import BlameGraph, CausalFinding
from agent_evolve.core.contracts import EvolutionTask, ExecutionTrace
from agent_evolve.core.evidence import rollout_group_report
from agent_evolve.core.run_logging import RunLogSink

_STRENGTH_SYSTEM_PROMPT = """\
You are a trajectory STRENGTH analyst. You examine a SUCCESSFUL agent \
rollout -- the task was completed -- and extract what specifically caused \
the success.

Rules:
- Describe CAUSES, not categories: "the planner decomposed the itinerary \
before searching, so the flight query was fully specified" -- not "good \
planning".
- Attribute each cause to an actor_id that appears in the evidence.
- Cite evidence_event_ids that exist in the trace.
- Name artifacts only when their id literally appears in the evidence.
- severity/confidence are numbers in [0, 1]: severity means how important \
this cause was to the success.
- If the evidence is too thin to attribute anything, return one finding \
with status "insufficient_evidence" and no invented details.

Return ONLY this JSON object:
{"findings": [
  {"status": "observed|uncertain|insufficient_evidence",
   "mechanism": "<causal sentence>",
   "severity": 0.0-1.0, "confidence": 0.0-1.0,
   "blamed_actors": [{"actor_id": "...", "blame": 0.0-1.0,
                      "artifacts": ["..."]}],
   "evidence_event_ids": ["..."],
   "rationale": "<why this is grounded>"}
]}

Every finding you emit will be recorded as a strength (valence=-1); you do \
not choose polarity and must not include it."""


@dataclass(slots=True)
class CugaPositivityJudge:
    """Success-side judge behind the core ``PositivityJudge`` protocol.

    Field semantics mirror ``CugaTrajectoryAnalyzer`` exactly (lazy settings,
    injectable completion_fn, optional transcript sink). Constructing one
    never requires credentials.
    """

    completion_fn: Callable[..., object] | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float | None = None
    max_events_in_prompt: int = 40
    request_json_object: bool = False
    analyzer_model_id: str = "cuga-positivity-judge"
    log_sink: RunLogSink | None = None
    _request_base: dict[str, object] | None = field(
        default=None, init=False, repr=False, compare=False
    )

    # ------------------------------------------------------------------ #
    # Protocol surface
    # ------------------------------------------------------------------ #
    def analyze_success(
        self,
        task: EvolutionTask,
        trace: ExecutionTrace,
    ) -> tuple[CausalFinding, ...]:
        """Strengths found in ONE successful rollout.

        Raises only if the model call itself fails; response-content problems
        become abstaining findings (still ``valence=-1``).
        """
        report = rollout_group_report(task, [trace])
        evidences = tuple(
            e for e in report.sanitized_evidence if isinstance(e, Mapping)
        )
        if not evidences:
            return self._abstain_all(
                report,
                "insufficient_evidence",
                "no sanitized evidence could be built from this trace",
            )

        response, messages = self._strength_call(report, evidences)
        text = _response_text(response)

        def done(findings: tuple[CausalFinding, ...]) -> tuple[CausalFinding, ...]:
            self._write_transcript(report, messages, text, findings)
            return findings

        if not text.strip():
            return done(
                self._abstain_all(report, "malformed", "model returned an empty body")
            )

        entries, parse_error = _parse_findings_payload(text)
        if parse_error is not None:
            return done(self._abstain_all(report, "malformed", parse_error))

        return done(
            tuple(
                self._build_strength(report, evidences[0], entry, index)
                for index, entry in enumerate(entries)
            )
        )

    # ------------------------------------------------------------------ #
    # Finding construction -- polarity stamped HERE, in code
    # ------------------------------------------------------------------ #
    def _build_strength(
        self,
        report: RolloutGroupReport,
        evidence: Mapping[str, object],
        raw: Mapping[str, object],
        index: int,
    ) -> CausalFinding:
        trace_id = str(evidence.get("trace_id") or "unknown-trace")
        try:
            return self._build_strength_unsafe(report, evidence, raw, trace_id, index)
        except Exception as exc:  # noqa: BLE001 - a bad verdict is data, not a crash
            return self._abstaining(
                report,
                trace_id,
                index,
                "malformed",
                f"could not construct a valid strength: "
                f"{type(exc).__name__}: {exc}",
            )

    def _build_strength_unsafe(
        self,
        report: RolloutGroupReport,
        evidence: Mapping[str, object],
        raw: Mapping[str, object],
        trace_id: str,
        index: int,
    ) -> CausalFinding:
        status = _normalize_status(raw.get("status"))
        if status is None:
            status = "uncertain"

        rationale = _first_text(raw, ("rationale", "reason", "explanation"))
        mechanism = _first_text(raw, ("mechanism", "mechanism_description"))

        severity, sev_err = _unit_number(raw.get("severity"), "severity")
        confidence, conf_err = _unit_number(raw.get("confidence"), "confidence")

        known_actors = _known_actors(evidence)
        known_events = _known_event_ids(evidence)
        evidence_blob = json.dumps(evidence, default=_jsonable)

        nodes, edges, node_notes, artifact_refs = _grounded_blame_graph(
            raw, known_actors, evidence_blob
        )
        event_refs, ref_notes = _grounded_event_refs(raw, known_events, trace_id)

        notes = [*node_notes, *ref_notes]
        if sev_err:
            notes.append(sev_err)
        if conf_err:
            notes.append(conf_err)
            confidence = None

        if status == "observed":
            if not mechanism:
                status = "uncertain"
                notes.append("downgraded: observed without a mechanism sentence")
            elif _is_degenerate_mechanism(mechanism):
                status = "uncertain"
                notes.append(
                    "downgraded: mechanism was generic, not a causal sentence"
                )
            elif not event_refs:
                status = "uncertain"
                notes.append(
                    "downgraded: cited no event id that exists in this trace"
                )
            elif severity is None or confidence is None:
                status = "uncertain"
                notes.append("downgraded: missing severity/confidence")

        if rationale and notes:
            rationale = f"{rationale} [{'; '.join(notes)}]"
        elif notes:
            rationale = "; ".join(notes)
        elif not rationale:
            rationale = "model-reported strength"

        # THE POLARITY STAMP. raw["valence"] is deliberately never read.
        return CausalFinding(
            verdict_id=f"strength:{report.candidate_id}:{trace_id}:{index}",
            candidate_id=report.candidate_id,
            task_id=report.task_id,
            trace_id=trace_id,
            valence=-1,
            status=status,  # type: ignore[arg-type]
            mechanism_description=mechanism or None,
            # Same sentinel convention as the fault side: obviously-not-a-
            # cluster beats a plausible fake one; IDX2 assigns real clusters.
            mechanism_cluster_id=UNASSIGNED_MECHANISM_CLUSTER_ID,
            severity=severity,
            confidence=confidence,
            blame_graph=BlameGraph(nodes=nodes, edges=edges),
            evidence_refs=tuple(dict.fromkeys((*event_refs, *artifact_refs))),
            rationale=rationale,
        )

    def _abstaining(
        self,
        report: RolloutGroupReport,
        trace_id: str,
        index: int,
        status: str,
        reason: str,
    ) -> CausalFinding:
        """An abstaining finding -- still a strength by polarity."""
        return CausalFinding(
            verdict_id=f"strength:{report.candidate_id}:{trace_id}:{index}",
            candidate_id=report.candidate_id,
            task_id=report.task_id,
            trace_id=trace_id,
            valence=-1,
            status=status,  # type: ignore[arg-type]
            rationale=reason,
        )

    def _abstain_all(
        self, report: RolloutGroupReport, status: str, reason: str
    ) -> tuple[CausalFinding, ...]:
        trace_id = report.trace_refs[0] if report.trace_refs else "unknown-trace"
        return (self._abstaining(report, trace_id, 0, status, reason),)

    # ------------------------------------------------------------------ #
    # Model call + transcript (mirrors cuga_analyzer's split)
    # ------------------------------------------------------------------ #
    def _resolve_request_base(self) -> dict[str, object]:
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
                "no positivity model configured: set CUGA_MODEL or LITELLM_MODEL, "
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
        trimmed = [_trim_evidence(e, self.max_events_in_prompt) for e in evidences]
        return (
            f"candidate_id: {report.candidate_id}\n"
            f"task_id: {report.task_id}\n"
            f"rollout OUTCOME: success\n"
            f"tool invocations actually observed:\n{_tool_call_census(evidences)}\n\n"
            "SANITIZED EVIDENCE (JSON):\n"
            f"{json.dumps(trimmed, indent=2, default=_jsonable)}\n\n"
            "Return the JSON object described in the schema. No prose outside it."
        )

    def _strength_call(
        self,
        report: RolloutGroupReport,
        evidences: Sequence[Mapping[str, object]],
    ) -> tuple[object, tuple[Mapping[str, object], ...]]:
        messages: tuple[Mapping[str, object], ...] = (
            {"role": "system", "content": _STRENGTH_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(report, evidences)},
        )
        request = dict(self._resolve_request_base())
        request["messages"] = [dict(m) for m in messages]
        invoke = self.completion_fn or _litellm_completion
        return invoke(**request), messages

    def _write_transcript(
        self,
        report: RolloutGroupReport,
        messages: Sequence[Mapping[str, object]],
        response_text: str,
        findings: tuple[CausalFinding, ...],
    ) -> None:
        sink = self.log_sink
        if sink is None:
            return
        try:
            sink.write_record(
                f"positivity__{report.candidate_id}__{report.task_id}",
                {
                    "candidate_id": report.candidate_id,
                    "task_id": report.task_id,
                    "analyzer_model_id": self.analyzer_model_id,
                    "messages": [dict(m) for m in messages],
                    "response_text": response_text,
                    "statuses": [f.status for f in findings],
                },
            )
        except Exception:  # noqa: BLE001 - capture is an observer, never a gate
            pass


__all__ = ["CugaPositivityJudge"]
