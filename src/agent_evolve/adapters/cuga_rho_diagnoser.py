"""Interface B: diagnose one task's G rollouts in a single workspace-agent call.

Per ``RHO_agents_context.md:59-94``, self-validation and self-consistency are two
signals extracted *within one diagnosis invocation*, not separate calls. With
``k=10`` that is 10 invocations, not 20.

The agent gets tools to inspect each rollout selectively rather than one enormous
serialized prompt. That is the reason the paper uses an agent here, and it keeps
the prompt bounded when ``G=3`` trajectories of 19-56 events each would otherwise
be concatenated.

Three properties are load-bearing and are asserted by the tests:

1. **The result is a captured side effect.** ``submit_diagnosis`` writes into a
   per-call context. A model that emits a perfectly shaped JSON object in its
   prose without executing the tool is reported as ``NO_TOOL_CALL``, not parsed
   as a success. Prose is not evidence, and neither is
   ``InvokeResult.tool_calls``.
2. **Validation happens inside the loop.** A malformed submission returns a
   rejection *with the reason* so the same invocation can correct itself. Only
   an invocation that never produces an acceptable submission is unobserved.
   Post-hoc-only validation would throw away a whole diagnosis over a
   fixable field.
3. **The mechanism must be discriminative.** The mechanism string is what the
   optimizer (phase 6) tries to fix and what cross-candidate variance is computed
   over. "The agent did not follow instructions" is not a mechanism; it is a
   restatement of failure that no edit can act on. Such answers are rejected in
   loop with an explanation, and the prompt names them as banned up front.

The prompt states the real SDK graph shape and must distinguish "narrated without
emitting an executable code block, so ``sandbox`` was never reached" from a
genuine tool failure. Ground truth for tool execution is recorded tool
observations, never model prose.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from agent_evolve.adapters.cuga_workspace_agent import run_workspace_agent
from agent_evolve.core.contracts import ExecutionTrace

DIAGNOSER_MODEL_ID = "cuga-rho-group-diagnoser"

#: The four surfaces ``CugaAdapter`` can actually deliver. A diagnosis pointing
#: anywhere else names a surface the optimizer cannot write, so it is rejected
#: rather than silently dropped later.
VALID_SURFACES = frozenset({"instructions", "skills", "policies", "memory"})

#: Restatements of failure that carry no mechanism. Each one is true of almost
#: every failing rollout, so it cannot discriminate between candidates and gives
#: the optimizer nothing to change. Listed in the prompt as banned, and rejected
#: in loop when they appear anyway.
BANNED_MECHANISM_PHRASES: tuple[str, ...] = (
    "did not follow instructions",
    "failed to follow instructions",
    "failed to complete the task",
    "made a mistake",
    "was not careful",
    "lacks sufficient guidance",
    "insufficient detail",
    "poor reasoning",
    "needs to be more accurate",
    "did not perform well",
)

#: Minimum words in a mechanism. "bad" or "wrong tool" cannot locate a surface.
_MIN_MECHANISM_WORDS = 4

#: Bounds that keep one pathological rollout from blowing the agent's context.
_MAX_EVENTS_PER_READ = 40
_MAX_PAYLOAD_CHARS = 400

APP_NAMES: dict[str, str] = {
    "get_task": "evidence",
    "list_rollouts": "evidence",
    "read_rollout_events": "evidence",
    "note_rollout": "evidence",
    "submit_diagnosis": "submit",
}


# --------------------------------------------------------------------------- #
# Behavioural configuration (always present) vs. the task prompt (per call).
#
# CUGA injects ``special_instructions`` on every turn, so the invariants that
# must hold whatever the agent decides live here; the per-task analysis order
# lives in the prompt. The shared WORKSPACE_AGENT_TOOL_CONTRACT owns the fenced
# execution protocol and is appended by run_workspace_agent -- nothing here may
# contradict it.
# --------------------------------------------------------------------------- #
DIAGNOSER_INSTRUCTIONS = """\
You are an offline trajectory analyst. You diagnose why an agent harness
underperformed. You never edit the harness, and you never solve the task.

Invariants, which hold whatever you decide:

1. Ground truth for whether a tool ran is a recorded tool observation. The
   agent's own prose is not evidence: a rollout that says it searched, and shows
   zero tool observations, did not search.
2. No trajectory is ground truth for the answer. You are diagnosing the harness,
   not grading a submission, and you may not have any way to know the correct
   answer. Say what the rollouts did and where they diverged.
3. Agreement is not correctness. Three rollouts converging on the same wrong
   answer is a strong harness weakness, not a success; converging by luck is not
   evidence of a good procedure either.
4. Diagnose the harness, not the task. Your output is read by an optimizer that
   may only change always-on instructions, loadable skills, triggered policies,
   and retrievable memory. A finding that no such change could address is not
   useful to it.
5. Never put a task-specific expected answer, or any value you inferred as the
   answer, into text meant to be reused. Reusable text describes procedure.
6. Finish by executing the terminal submit tool. An analysis you only narrate is
   discarded and counts as no engagement.
"""

#: Per-call prompt. ``{count}`` and ``{task_id}`` are the only placeholders, so
#: literal braces are avoided throughout (``str.format`` would choke on them).
DIAGNOSIS_PROMPT = """\
Write and execute Python code that calls the diagnosis tools listed below, then
report the exact values they returned.

Diagnose why one agent harness underperformed on task {task_id}.

WHAT YOU ARE LOOKING AT

You have {count} independent rollouts of the SAME task with the SAME harness.
Same task, same harness, fresh execution each time: every difference between
them was produced by the harness leaving something underdetermined.

The agent under study runs this graph:

    CugaLiteSubgraph -> prepare -> call_model <-> sandbox -> SDKCallback
        -> FinalAnswerAgent

A tool runs ONLY when call_model emits an executable fenced Python block that
reaches sandbox, and only the FIRST fenced block in a turn executes. So there
are two completely different failures that look alike in the final text:

  * NEVER REACHED THE SANDBOX - no tool observation exists. The model described
    what it would do and then answered. The harness failed to make execution
    happen at all.
  * REACHED THE SANDBOX AND THE CALL WENT WRONG - a tool observation exists and
    shows an error, an empty result, or a result the model then misread.

These need opposite fixes, so never merge them. Never claim a tool failed unless
a tool observation shows it failing. list_rollouts reports the tool-observation
count per rollout; that count, not the prose, decides which case you are in.

TOOLS - you act only by executing these

    get_task()                        the task input
    list_rollouts()                   rollout ids, final outputs, statuses,
                                      tool-observation counts
    read_rollout_events(rollout_id)   that rollout's event stream
    note_rollout(rollout_id, likely_successful, verified_own_answer, issue)
                                      record your per-rollout assessment
    submit_diagnosis(...)             terminal; call last

Read the events of every rollout before you conclude anything. You must call
note_rollout once for each rollout id; submit_diagnosis is refused until you
have.

ANALYSIS ORDER

1. get_task, then list_rollouts. Note which rollouts executed tools at all, and
   which never got started -- a rollout whose events are mostly errors and which
   produced no answer tells you about the infrastructure, not the harness. Say so
   and exclude it from your conclusions rather than reading it as fast or lean.
2. Per rollout, read its events and decide:
   - did it likely satisfy the task's actual requirements?
   - did it VERIFY its own answer before committing - re-derived it, cross
     checked a second source, or re-ran a computation? Reasserting the answer
     more confidently is not verification. Answer yes only if some concrete
     checking step happened.
   - what evidence, tools, or reasoning steps did it actually rely on?
   - which of these four went wrong, naming them explicitly when present:
       UNNECESSARY WORK   steps it chose to take that were not needed - redundant
                          calls, re-deriving something already established,
                          circling a step already done. Retries forced by a tool
                          that errored do not count; those are not its choice.
       MISSED INFORMATION something available to it that it never looked at
       MISLEADING EVIDENCE something it trusted that did not support the
                          conclusion it drew
       INCORRECT DECISION a choice that was wrong given what it already knew
   Record each one with note_rollout. Name which of the four applies, so the
   optimizer can attend to the same category across tasks; a single word like
   "wasteful" does not survive that comparison.
3. Compare the rollouts. Where do they DISAGREE in interpretation of the task,
   plan, actions taken, or final answer? Separate harmless variation (different
   wording, different but equivalent route) from consequential divergence
   (different interpretation, different answer, one verified and others did
   not). Report only the consequential ones.
4. Name the failure mode that RECURS across rollouts. If nothing recurs, say so
   and set a low severity: an accident in one rollout is not a harness weakness.

MECHANISM, NOT SYMPTOM

recurring_failure_mode is the single most important thing you produce. An
optimizer will try to fix exactly what you name, and your wording will be
compared against other tasks' diagnoses to find shared weaknesses. So it must
name the MECHANISM - the specific step at which behaviour went wrong - and not
the SYMPTOM, which is just that the answer was wrong.

Test your sentence two ways before you submit it:
  * Could it be pasted onto almost any failing run? Then it is a symptom.
  * Could someone locate the point in the run it refers to, and say what text
    would have prevented it? If not, it is a symptom.

Banned, because each is true of nearly every failure and names no step:
    "did not follow instructions", "failed to complete the task",
    "made a mistake", "was not careful", "lacks sufficient guidance",
    "insufficient detail", "poor reasoning", "did not perform well"

Symptom (rejected):  the agent produced the wrong number
Mechanism (useful):  committed the first candidate value it saw without
                     re-deriving it, in every rollout that answered at all

Symptom (rejected):  the agent did not use its tools properly
Mechanism (useful):  narrated the tool call in prose instead of emitting a
                     fenced block, so sandbox never ran and no observation
                     exists in any rollout

SEVERITY, ANCHORED

severity orders every task's diagnosis for the optimizer, so an uncalibrated
number destroys that ordering. Use these anchors, not a default:

    0.0  nothing recurs; all rollouts handled the task accurately and without
         wasted work
    0.2  a minor inefficiency or a weak concern; the task was still handled, and
         this alone is not worth a harness edit
    0.4  one rollout was derailed by it and the others absorbed it, OR the
         rollouts disagreed consequentially even though none outright failed
    0.6  it recurs and degrades results, but some rollout still recovered
    0.8  it recurs in most rollouts and none recovered
    1.0  it recurs in every rollout and blocks the task outright

CONSISTENCY IS RELIABILITY

Recurrence is not the only thing that earns severity. Identical inputs that
produce consequentially DIFFERENT behaviour are themselves a harness weakness,
because it means the harness underdetermined what the agent should do and left
the outcome to chance. A task where the rollouts disagreed on the answer, or
where one verified and the others did not, belongs at 0.4 or above even if you
cannot point to a single rollout that failed -- you have learned that success
here is not reproducible.

Two things that are NOT inconsistency of this kind, and must not be scored as it:
  * harmless variation -- different wording, or a different but equivalent route
    to the same answer;
  * divergence caused by a tool erroring, timing out, or being unavailable in one
    rollout. That is infrastructure noise, not a harness gap. A rollout that
    crashed before doing any work is not evidence about the harness at all;
    exclude it and say you excluded it.

IMPROVEMENT DIRECTION

Describe a change to the HARNESS that would prevent this mechanism on tasks you
have not seen. It must be general, not task-specific: name the procedural rule
or capability that is missing, never this task's answer, entities, or values. Do
not write the edit; name the direction and let the optimizer write it.

candidate_surfaces names where that change belongs, any of:
    instructions   always-on text, reaches the model every turn unconditionally
    skills         a procedure, only enters context if the model loads it
    policies       conditional guidance, applies only when its trigger matches
    memory         retrievable facts; changes what can be looked up, not
                   behaviour
Pick by delivery route. If the mechanism is that some step never happens, do not
choose a surface that only arrives once that step happens.

SUBMIT

Execute submit_diagnosis exactly once at the end, with:
    recurring_failure_mode     one or two sentences, mechanism not symptom
    disagreements              list of consequential disagreements, may be empty
    self_validation_observed   true only if some rollout really checked itself
    severity                   number in 0.0 to 1.0, per the anchors
    improvement_direction      general, not task-specific
    candidate_surfaces         list from instructions, skills, policies, memory

If it returns a rejection, read the reason, fix that field, and call it again in
your next block. A rejected submission is not recorded.
"""


# --------------------------------------------------------------------------- #
# Result contracts
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class RolloutAssessment:
    """The agent's per-rollout judgement, captured while it worked.

    Held separately from the group-level fields because self-validation is a
    per-rollout observation that the single group-level flag necessarily
    flattens.
    """

    rollout_id: str
    likely_successful: bool = False
    verified_own_answer: bool = False
    issue: str = ""


@dataclass(frozen=True, slots=True)
class GroupDiagnosis:
    """One task's diagnosis over its whole rollout group.

    ``observed`` is the only success test. ``status`` says *why* an unobserved
    diagnosis is unobserved, because those cases demand different responses:
    ``NO_TOOL_CALL`` is a prompt problem, ``REJECTED`` is a compliance problem,
    ``UNAVAILABLE`` is an infrastructure problem.
    """

    task_id: str
    recurring_failure_mode: str = ""
    disagreements: tuple[str, ...] = ()
    self_validation_observed: bool = False
    severity: float = 0.0
    improvement_direction: str = ""
    candidate_surfaces: tuple[str, ...] = ()
    rollouts_seen: int = 0
    observed: bool = False
    error: str = ""
    #: One of NO_ROLLOUTS, UNAVAILABLE, NO_TOOL_CALL, NO_OP, REJECTED, OK.
    status: str = "UNOBSERVED"
    per_rollout: tuple[RolloutAssessment, ...] = ()
    tools_called: tuple[str, ...] = ()


# --------------------------------------------------------------------------- #
# Validation (pure, reusable, and run in loop)
# --------------------------------------------------------------------------- #
def validate_diagnosis_payload(payload: Mapping[str, object]) -> str:
    """Return an actionable error string, or ``""`` when the payload is usable.

    Every message names the offending field and what would fix it, because the
    same string is handed back to the agent mid-run as its retry instruction.
    """
    raw = payload.get("severity")
    try:
        severity = float(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return f"severity must be a number in 0.0-1.0; got {raw!r}"
    if not 0.0 <= severity <= 1.0:
        return f"severity {severity} is outside 0.0-1.0; rescale it to the anchors"

    surfaces = payload.get("candidate_surfaces") or []
    if isinstance(surfaces, str) or not isinstance(surfaces, (list, tuple)):
        return "candidate_surfaces must be a list of surface names, not a string"
    for surface in surfaces:
        if str(surface) not in VALID_SURFACES:
            return (
                f"unknown surface {str(surface)!r}; the harness only delivers "
                f"{sorted(VALID_SURFACES)}"
            )

    disagreements = payload.get("disagreements") or []
    if isinstance(disagreements, str) or not isinstance(
        disagreements, (list, tuple)
    ):
        return "disagreements must be a list of strings, not a string"

    mechanism = str(payload.get("recurring_failure_mode") or "").strip()
    if not mechanism:
        return "recurring_failure_mode is empty; name the mechanism"
    if len(mechanism.split()) < _MIN_MECHANISM_WORDS:
        return (
            "recurring_failure_mode is too short to name a mechanism: "
            f"{mechanism!r}. State the step at which behaviour went wrong."
        )
    lowered = mechanism.lower()
    for phrase in BANNED_MECHANISM_PHRASES:
        if phrase in lowered:
            return (
                f"recurring_failure_mode is vague: {phrase!r} is true of nearly "
                "every failing run and names no step. Name the mechanism -- the "
                "specific point at which behaviour went wrong."
            )
    return ""


# --------------------------------------------------------------------------- #
# Per-call tool context
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class DiagnoserContext:
    """State one diagnosis call's tools close over. Nothing here is global."""

    task_id: str
    task_input: str
    traces: tuple[ExecutionTrace, ...]
    notes: dict[str, RolloutAssessment] = field(default_factory=dict)
    payload: dict | None = None
    last_rejection: str = ""


def _ok(**payload: object) -> str:
    return json.dumps(payload, default=str)


def _err(message: str) -> str:
    return json.dumps({"status": "error", "message": message})


def _tool_observations(trace: ExecutionTrace) -> int:
    """Count recorded tool observations. This, not prose, is execution evidence."""
    return sum(1 for event in trace.events if "tool" in str(event.kind).lower())


def _clip(value: object) -> object:
    text = value if isinstance(value, str) else None
    if text is None or len(text) <= _MAX_PAYLOAD_CHARS:
        return value
    return f"{text[:_MAX_PAYLOAD_CHARS]}... [truncated]"


#: Closing directive for THIS prompt. ``run_workspace_agent`` also appends a
#: shared execute directive after the tool roster; this one names the specific
#: first calls, which the shared text cannot know.
_ACT_NOW = """\

WHERE TO START

Call get_task() and list_rollouts() first, in one executed block, and print both
results before you assess anything.
"""


def build_diagnosis_prompt(
    task_id: str, task_input: str, traces: Sequence[ExecutionTrace]
) -> str:
    """The per-call prompt.

    ``task_input`` is deliberately NOT interpolated: the agent reads it through
    ``get_task``, which keeps the prompt bounded and keeps the one evidence route
    tool-mediated so reading it is observable.
    """
    return DIAGNOSIS_PROMPT.format(count=len(traces), task_id=task_id) + _ACT_NOW


def build_diagnoser_callables(
    ctx: DiagnoserContext,
) -> dict[str, Callable[..., str]]:
    """Plain callables with no CUGA dependency, so the rules stay testable.

    Each one has a real typed signature and a docstring: LangChain's ``@tool``
    refuses an undocumented body, and it derives the args schema from the
    signature, so a signature-less tool silently tells the model it takes no
    arguments.
    """

    def get_task() -> str:
        """Return the task input text the rollouts were asked to solve."""
        return _ok(task_id=ctx.task_id, input=ctx.task_input)

    def list_rollouts() -> str:
        """List every rollout with its final output, status and tool-observation count."""
        return _ok(
            rollouts=[
                {
                    "rollout_id": trace.trace_id,
                    "final_output": _clip(trace.final_output),
                    "status": trace.status,
                    "event_count": len(trace.events),
                    "tool_observations": _tool_observations(trace),
                    "executed_any_tool": _tool_observations(trace) > 0,
                }
                for trace in ctx.traces
            ],
            note=(
                "tool_observations is the only evidence a tool ran; "
                "zero means the sandbox was never reached"
            ),
        )

    def read_rollout_events(rollout_id: str) -> str:
        """Return one rollout's event stream, in order, bounded in size."""
        for trace in ctx.traces:
            if trace.trace_id != rollout_id:
                continue
            events = trace.events[:_MAX_EVENTS_PER_READ]
            return _ok(
                rollout_id=rollout_id,
                events=[
                    {
                        "event_id": event.event_id,
                        "kind": event.kind,
                        "actor_id": event.actor_id,
                        "payload": {
                            key: _clip(value)
                            for key, value in dict(event.payload).items()
                        },
                    }
                    for event in events
                ],
                truncated=len(trace.events) > len(events),
                total_events=len(trace.events),
            )
        known = [trace.trace_id for trace in ctx.traces]
        return _err(f"unknown rollout_id {rollout_id!r}; known ids are {known}")

    def note_rollout(
        rollout_id: str,
        likely_successful: bool = False,
        verified_own_answer: bool = False,
        issue: str = "",
    ) -> str:
        """Record your assessment of one rollout: did it likely succeed, did it verify itself, what went wrong."""
        if rollout_id not in {trace.trace_id for trace in ctx.traces}:
            known = [trace.trace_id for trace in ctx.traces]
            return _err(f"unknown rollout_id {rollout_id!r}; known ids are {known}")
        ctx.notes[rollout_id] = RolloutAssessment(
            rollout_id=rollout_id,
            likely_successful=bool(likely_successful),
            verified_own_answer=bool(verified_own_answer),
            issue=str(issue),
        )
        outstanding = [
            trace.trace_id for trace in ctx.traces if trace.trace_id not in ctx.notes
        ]
        return _ok(status="ok", recorded=rollout_id, still_unassessed=outstanding)

    def submit_diagnosis(
        recurring_failure_mode: str,
        disagreements: list | None = None,
        self_validation_observed: bool = False,
        severity: float = 0.0,
        improvement_direction: str = "",
        candidate_surfaces: list | None = None,
    ) -> str:
        """Finalize the diagnosis for this task. Returns ok, or a rejection with the reason to fix."""
        outstanding = [
            trace.trace_id for trace in ctx.traces if trace.trace_id not in ctx.notes
        ]
        if outstanding:
            reason = (
                "call note_rollout for every rollout before submitting; still "
                f"unassessed: {outstanding}"
            )
            ctx.last_rejection = reason
            return _ok(status="rejected", reason=reason)

        payload = {
            "recurring_failure_mode": recurring_failure_mode,
            "disagreements": list(disagreements or []),
            "self_validation_observed": bool(self_validation_observed),
            "severity": severity,
            "improvement_direction": improvement_direction,
            "candidate_surfaces": list(candidate_surfaces or []),
        }
        # Validate before capture, so a rejected submission is recoverable in the
        # same invocation instead of costing the whole task's diagnosis.
        reason = validate_diagnosis_payload(payload)
        if reason:
            ctx.last_rejection = reason
            return _ok(status="rejected", reason=reason)

        ctx.payload = payload
        ctx.last_rejection = ""
        return _ok(status="ok", accepted=True)

    return {
        "get_task": get_task,
        "list_rollouts": list_rollouts,
        "read_rollout_events": read_rollout_events,
        "note_rollout": note_rollout,
        "submit_diagnosis": submit_diagnosis,
    }


# --------------------------------------------------------------------------- #
# The diagnoser
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RhoGroupDiagnoser:
    """Interface B group diagnoser: exactly one invocation per task."""

    agent_factory: Callable[[dict, str], str] | None = None
    app_names: Mapping[str, str] = field(default_factory=lambda: dict(APP_NAMES))

    def diagnose(
        self,
        task_id: str,
        task_input: str,
        traces: Sequence[ExecutionTrace],
    ) -> GroupDiagnosis:
        """Diagnose one task's rollout group, returning failure as data.

        Never raises: a raised exception here would discard a whole round's
        evidence for one bad invocation.
        """
        group = tuple(traces)
        if not group:
            return GroupDiagnosis(
                task_id=task_id,
                error="no rollouts to diagnose",
                status="NO_ROLLOUTS",
            )

        ctx = DiagnoserContext(task_id=task_id, task_input=task_input, traces=group)
        run = run_workspace_agent(
            build_diagnoser_callables(ctx),
            build_diagnosis_prompt(task_id, task_input, group),
            app_names=self.app_names,
            special_instructions=DIAGNOSER_INSTRUCTIONS,
            agent_factory=self.agent_factory,
        )

        assessments = tuple(
            ctx.notes[trace.trace_id]
            for trace in group
            if trace.trace_id in ctx.notes
        )
        base = {
            "task_id": task_id,
            "rollouts_seen": len(group),
            "per_rollout": assessments,
            "tools_called": run.tools_called,
        }

        # The captured side effect decides success. It outranks a later crash,
        # because an accepted submission already happened, and it outranks the
        # answer text, which is never parsed.
        if ctx.payload is not None:
            payload = ctx.payload
            return GroupDiagnosis(
                **base,
                recurring_failure_mode=str(payload["recurring_failure_mode"]).strip(),
                disagreements=tuple(str(d) for d in payload["disagreements"]),
                self_validation_observed=bool(payload["self_validation_observed"]),
                severity=float(payload["severity"]),
                improvement_direction=str(payload["improvement_direction"] or ""),
                candidate_surfaces=tuple(str(s) for s in payload["candidate_surfaces"]),
                observed=True,
                status="OK",
                error=run.error,
            )

        if not run.ok:
            return GroupDiagnosis(**base, error=run.error, status="UNAVAILABLE")
        if run.no_tool_call:
            return GroupDiagnosis(
                **base,
                error="agent executed no tool; nothing was diagnosed",
                status="NO_TOOL_CALL",
            )
        if ctx.last_rejection:
            return GroupDiagnosis(
                **base,
                error=f"every submission was rejected: {ctx.last_rejection}",
                status="REJECTED",
            )
        return GroupDiagnosis(
            **base,
            error="agent used tools but never submitted a diagnosis",
            status="NO_OP",
        )
