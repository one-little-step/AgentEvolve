"""Interface B x N: propose N independent candidate harnesses.

Per the paper (``RHO_agents_context.md:125-163``) each candidate is its own agent
invocation with its own workspace::

    candidate_0 = optimizer.run(fresh_workspace_0)
    candidate_1 = optimizer.run(fresh_workspace_1)
    candidate_2 = optimizer.run(fresh_workspace_2)

Diversity therefore comes from independent multi-step tool trajectories, not from
token sampling. Two agents that read different diagnoses in a different order and
stage different edits differ for a substantive reason; two samples from one
request differ only by which token the decoder happened to draw. Sampling
temperature is an ablation knob, not the mechanism, and is omitted by default
(``0.0`` is additionally rejected by the endpoint).

A candidate is captured from **staged artifacts**, never parsed from the agent's
final text -- the paper's "captured from the filesystem, rather than parsed from
the optimizer's final textual answer". Unfinalized staging is discarded.

A no-op is not a candidate. Byte-identical-to-base and duplicate artifact sets
are discarded before evaluation so the pairwise judge never compares a harness
with itself. Every discard is reported with a status from ``DISCARD_STATUSES``,
so a collapse from N to 1 is visible rather than silent.

**All surviving candidates are retained.** This module never selects a best-of-N;
ranking happens downstream and every distinct candidate enters the pool as a
parent.

Diagnoses are consumed **structurally** -- either mappings or objects with the
same attribute names -- so this module does not import the diagnoser and the two
stay independently testable.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Callable, Mapping, Sequence

from agent_evolve.adapters.cuga_editor_state import (
    DEFAULT_CREATABLE_PREFIXES,
    EditStagingArea,
)
from agent_evolve.adapters.cuga_workspace_agent import run_workspace_agent

OPTIMIZER_MODEL_ID = "cuga-rho-optimizer"

#: Created ids must carry the CUGA group first: ``_harness_slot`` in
#: ``cuga_adapter.py`` accepts only ``instructions`` or a
#: ``skills|policies|memory/<name>`` prefix, so a flat ``generated/<name>`` would
#: raise at registration and the creation path would be dead code.
#:
#: SV-8: one prefix per surface. A scalar ``skills/generated-`` meant the only
#: artifact the optimizer could *create* was a skill, so ``memory/`` and
#: ``policies/`` were reachable by replacement alone -- and until multi-surface
#: seeding landed there was nothing on those surfaces to replace.
CREATABLE_PREFIXES: tuple[str, ...] = DEFAULT_CREATABLE_PREFIXES
#: Retained for callers reading a single prefix; see ``CREATABLE_PREFIXES``.
CREATABLE_PREFIX = CREATABLE_PREFIXES[0]

#: Tool name -> CUGA app group. The keys are the exact tool surface.
APP_NAMES: dict[str, str] = {
    "list_diagnoses": "evidence",
    "read_diagnosis": "evidence",
    "list_artifacts": "harness",
    "read_artifact": "harness",
    "stage_replace": "harness",
    "stage_create": "harness",
    "list_staged": "harness",
    "unstage": "harness",
    "submit_candidate": "submit",
}

#: Every reason a proposal can fail to become a candidate. A discard reason is
#: always ``"<STATUS>: <detail>"``, so a round log can aggregate by status.
DISCARD_STATUSES: tuple[str, ...] = (
    # the invocation raised or timed out
    "UNAVAILABLE",
    # the agent finished having executed no tool at all -- it narrated
    "NO_TOOL_CALL",
    # edits were staged but submit_candidate was never executed
    "NOT_FINALIZED",
    # finalized with nothing staged
    "NO_OP",
    # staged, but the resulting artifact set equals the base byte for byte
    "IDENTICAL",
    # equals a candidate an earlier invocation in this batch already produced
    "DUPLICATE",
)


# ====================================================================== #
# Doctrine: always-present instructions
# ====================================================================== #
#
# This text is the entire delta of the RHO stage. Everything else here is
# plumbing: if this text is weak the optimizer produces cosmetic rewording, the
# candidate survives every discard rule, and the measured result is noise.
#
# Three failure modes it is written against, all observed on the genetic path:
#
# 1. Placing a fix on a surface whose delivery route depends on the broken
#    behaviour (a skill that repairs "the model never calls tools" can only be
#    read by calling a tool).
# 2. Writing exhortation ("be careful", "think step by step") instead of a
#    checkable procedure. Exhortation validates, materializes, and changes
#    nothing.
# 3. Fixing the single most vivid observation instead of the mode that recurs
#    across tasks -- which is overfitting to one rollout under another name.
OPTIMIZER_INSTRUCTIONS = f"""\
You are improving the persistent HARNESS of an AI agent from accumulated failure
evidence gathered over many tasks. The model weights are fixed and you cannot
change them. The only thing you can change is the text the harness delivers into
that model's context, and the only thing that matters is whether your text
reaches the model at the moment the failure happens.

WHAT "BETTER" MEANS

Better performance means the agent's final answer more directly and correctly
answers what each task asks, WITH FEWER WASTED STEPS. Both halves count. A harness
edit that improves accuracy while sending the agent down a longer route is a
partial win; one that makes the agent reliably reach the same answer with less
redundant work is a real one.

Reliability is part of this. If the diagnoses show the rollouts for one task
disagreeing with each other, the harness left something underdetermined, and
pinning that down is as valuable as fixing an outright failure.

Do not pursue brevity for its own sake. Cutting the agent's steps by making it
stop before it has verified anything trades a wrong answer for a fast one, which
is a regression however short the trajectory looks.

WHAT THE AGENT ACTUALLY IS

The agent under study runs this graph:

    CugaLiteSubgraph -> prepare -> call_model <-> sandbox -> SDKCallback
                                                 -> FinalAnswerAgent

`prepare` assembles the context. `call_model` emits a response. A tool executes
ONLY when that response contains an executable fenced Python block, which is
what routes control to `sandbox`. If no fenced block is emitted, `sandbox` is
never reached, no tool runs, and the model typically reports that the tools were
unavailable. That specific failure -- narrating instead of emitting an
executable block -- is a turn-level defect, not a tool defect, and no amount of
procedural advice reachable only BY calling a tool can repair it.

THE FOUR SURFACES, AND WHERE EACH ONE ACTUALLY WORKS

The surfaces are not interchangeable. They differ in whether their text reaches
the model unconditionally, only if the model opts in, only if a trigger matches,
or only if a retrieval hits. A well-written artifact on the wrong surface passes
validation, lands on disk, and changes NOTHING -- the worst outcome available to
you, because it is indistinguishable from progress.

* instructions -- a single scalar artifact, assembled into the model's context on
  EVERY turn, unconditionally, with no opt-in and no matching step. The most
  direct path to the model that exists here.
  Fit: contracts about how a turn itself is conducted -- what form a response
  must take, what must always accompany it, how to proceed when something is
  unclear. If a behaviour must happen every turn, this is the ONLY surface that
  guarantees it.

* skills/<name> -- an optional procedure. Advertised to the model as a loadable
  item; its BODY enters context only if the model chooses to call `load_skill`.
  Two consequences you must reason about:
    - Selection is driven by the description, derived from the FIRST LINE of the
      body. A trigger-oriented opening ("Use when a question requires combining
      two retrieved values ...") gets selected; a passive title does not, and an
      unselected skill is inert however good its body is.
    - A skill cannot repair a failure that consists of the model not calling
      tools. That is circular and the artifact is dead on arrival.
  Fit: reusable multi-step procedures for a job the model already engages with
  and can recognise it needs.

* policies/<name> -- conditional guidance, loaded up front but applied only when
  its trigger matches the request's intent. A policy that declares it always
  applies loads successfully and then never matches, so it is silently inert.
  Fit: guidance that should apply to some requests and not others.

* memory/<name> -- retrievable facts and context. It changes what the model can
  look up, not how it behaves. Writing a behavioural rule here is a common and
  invisible mistake: it is stored, it may even be retrieved, and it governs
  nothing. Fit: durable domain facts, not behaviour.

HOW TO CHOOSE A SURFACE

1. Ask at which point in the run the failure mode bit: before any tool was used,
   during a procedure, or only for a certain kind of request. That locates the
   surface. Whichever artifact you happen to find most familiar does not.
2. Prefer the surface with the most direct, least conditional path to the model
   for that mode. Every conditional step between your text and the model -- a
   selection the model must make, a trigger that must match, a retrieval that
   must hit -- is a place your edit silently fails to apply.
3. Reject any surface whose delivery route depends on the very behaviour that is
   broken.
4. State in your rationale WHY the surface you chose will reach the model for
   this mode. If you cannot say that, you guessed.

WHAT AN EDIT MUST LOOK LIKE

An edit that a reader cannot check compliance against is worthless. Write
operational text with a TRIGGER (when this applies), an ACTION (the concrete
thing to do), and a CHECK (how the model knows it complied).

  Worthless, because nothing is checkable:
    "Think step by step and be careful to verify your answer."
    "Try to use the available tools effectively."
    "Pay close attention to the details of the question."

  Operational, because each part is checkable:
    "TRIGGER: the question names a quantity you have not yet retrieved.
     ACTION: emit one fenced Python block that calls the retrieval tool and
     prints its raw result before writing any prose about the value.
     CHECK: your next message must quote the printed value verbatim; if you
     cannot, the call did not execute -- re-issue it."

Rules:
- Prioritize failure modes that RECUR across tasks with high severity. A mode
  seen once at low severity is an anecdote; editing for it is overfitting to one
  rollout.
- Fix the MECHANISM, not the symptom. "Got the wrong number" is a symptom;
  "asserted a value it never retrieved" is a mechanism.
- Make surgical, generalizable edits. NEVER hardcode anything task-specific:
  no task ids, no expected answers, no literal strings copied from an observed
  question, no evaluator or grader internals, no regexes matched against a
  known answer. Such an edit inflates the measured score and teaches the agent
  nothing; it is the one failure this stage cannot tolerate.
- Write for the NEXT unseen task, not for the tasks in the evidence.
- Delete or replace harness text that the evidence shows is actively misleading.
  Removing a wrong rule is a legitimate improvement.
- `stage_replace` OVERWRITES the artifact wholesale: it does not append. If you
  intend to keep existing content, `read_artifact` first and include what you
  are keeping in the content you stage.
- A newly created artifact must be named `<surface>/generated-<name>`, where
  `<surface>` is one of `skills`, `memory` or `policies`. Choose the surface that
  fits what you are writing: a reusable procedure is a skill, a durable fact the
  agent should recall is memory, a hard constraint is a policy.
- Prefer ONE well-placed edit that addresses the top recurring mode over several
  shallow edits spread across surfaces.
- If nothing in the evidence justifies a change, stage nothing. That outcome is
  recorded honestly and is better than a cosmetic rewrite.

HOW TO WORK

1. `list_diagnoses` to see every mode, most severe first.
2. `read_diagnosis` on the ones that look like they share a mechanism.
3. `list_artifacts` and `read_artifact` on the surfaces those modes implicate --
   you cannot judge whether the harness already says something without reading
   it. An edit that restates existing text is a no-op.
4. Stage your edits, `list_staged` to confirm.
5. `submit_candidate(rationale=...)` exactly once. Work that is never finalized
   is DISCARDED ENTIRELY, including a decision to change nothing.
"""


# ====================================================================== #
# Result types
# ====================================================================== #


@dataclass(frozen=True, slots=True)
class ProposedCandidate:
    """One surviving candidate harness.

    ``artifacts`` is the COMPLETE artifact set (base carried forward with the
    staged edits applied), so a consumer can register it without also holding
    the base. Every id maps onto a CUGA harness slot: ``instructions`` or
    ``skills|policies|memory/<name>``.
    """

    candidate_index: int
    artifacts: Mapping[str, str]
    rationale: str = ""
    tools_called: tuple[str, ...] = ()
    observed: bool = False
    error: str = ""
    edited_ids: tuple[str, ...] = ()
    created_ids: tuple[str, ...] = ()
    fingerprint: str = ""


@dataclass(frozen=True, slots=True)
class ProposalReport:
    """Every proposal outcome, so a collapse to one candidate is visible.

    ``candidates`` holds ALL survivors. This stage never prunes to a best-of-N;
    downstream ranking does not remove parents from the pool.
    """

    candidates: tuple[ProposedCandidate, ...] = ()
    requested: int = 0
    discarded: tuple[tuple[int, str], ...] = ()

    @property
    def distinct(self) -> int:
        """Number of distinct surviving candidates."""
        return len(self.candidates)

    @property
    def collapsed(self) -> bool:
        """True when fewer candidates survived than were requested."""
        return self.distinct < self.requested

    def status_counts(self) -> dict[str, int]:
        """Discard counts keyed by ``DISCARD_STATUSES`` status."""
        return dict(Counter(reason.split(":", 1)[0] for _, reason in self.discarded))


# ====================================================================== #
# Diagnosis access: structural, so the diagnoser is not imported
# ====================================================================== #


def _get(diagnosis: object, name: str, default: Any) -> Any:
    """Read ``name`` from a mapping-shaped or attribute-shaped diagnosis."""
    if isinstance(diagnosis, Mapping):
        value = diagnosis.get(name, default)
    else:
        value = getattr(diagnosis, name, default)
    return default if value is None else value


def _task_id(diagnosis: object) -> str:
    return str(_get(diagnosis, "task_id", ""))


def _severity(diagnosis: object) -> float:
    try:
        return float(_get(diagnosis, "severity", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _surfaces(diagnosis: object) -> tuple[str, ...]:
    raw = _get(diagnosis, "candidate_surfaces", ())
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)


def _strings(diagnosis: object, name: str) -> tuple[str, ...]:
    raw = _get(diagnosis, name, ())
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(item) for item in raw)


def select_diagnoses(diagnoses: Sequence[object]) -> tuple[object, ...]:
    """Keep observed diagnoses only, ordered most severe first.

    An unobserved diagnosis is one whose own invocation failed. Feeding its
    empty fields to the optimizer would present "no failure mode" as evidence,
    which is worse than presenting nothing: the optimizer cannot tell a clean
    run from a lost one.
    """
    observed = [d for d in diagnoses if bool(_get(d, "observed", False))]
    return tuple(sorted(observed, key=lambda d: (-_severity(d), _task_id(d))))


def _render_diagnoses(diagnoses: Sequence[object]) -> str:
    lines: list[str] = []
    for diagnosis in diagnoses:
        surfaces = ", ".join(_surfaces(diagnosis)) or "unspecified"
        lines.append(
            f"- task {_task_id(diagnosis)} "
            f"(severity {_severity(diagnosis):.2f}, "
            f"{_get(diagnosis, 'rollouts_seen', 0)} rollouts observed)\n"
            f"    recurring failure mode: "
            f"{_get(diagnosis, 'recurring_failure_mode', '(unstated)')}\n"
            f"    improvement direction: "
            f"{_get(diagnosis, 'improvement_direction', '(unstated)')}\n"
            f"    surfaces the diagnosis implicates: {surfaces}"
        )
    if not lines:
        return (
            "- (no observed diagnoses; every diagnosis invocation failed, so there\n"
            "  is no evidence to act on -- stage nothing and say so)"
        )
    return "\n".join(lines)


# ====================================================================== #
# Per-invocation prompt
# ====================================================================== #


def build_optimizer_prompt(
    base_artifacts: Mapping[str, str],
    diagnoses: Sequence[object],
) -> str:
    """Render the task prompt for one proposal.

    Artifact bodies and diagnosis text are interpolated, never used as a format
    template: an artifact containing ``{json}`` would otherwise raise or be
    silently rewritten.
    """
    ordered = select_diagnoses(diagnoses)
    inventory = "\n".join(
        f"  - {artifact_id} ({len(content)} chars)"
        for artifact_id, content in sorted(base_artifacts.items())
    ) or "  - (none)"
    return (
        "Improve this agent's harness so that the recurring failure modes below\n"
        "stop happening on future, unseen tasks.\n"
        "\n"
        "FAILURE EVIDENCE, most severe first:\n"
        f"{_render_diagnoses(ordered)}\n"
        "\n"
        "Read the evidence for a mode that repeats across MORE THAN ONE task.\n"
        "That is the mode worth an edit; a single-task observation is an anecdote.\n"
        "\n"
        "HARNESS ARTIFACTS YOU MAY READ AND EDIT:\n"
        f"{inventory}\n"
        "\n"
        "YOUR TOOLS:\n"
        "  list_diagnoses()                     every diagnosis, most severe first\n"
        "  read_diagnosis(task_id)              one diagnosis in full\n"
        "  list_artifacts()                     the harness artifacts above\n"
        "  read_artifact(artifact_id)           current content of one artifact\n"
        "  stage_replace(artifact_id, content)  stage a rewrite (OVERWRITES it all)\n"
        "  stage_create(artifact_id, content)   stage a new "
        f"{'|'.join(p.split('/', 1)[0] for p in CREATABLE_PREFIXES)}"
        "/generated-<name>\n"
        "  list_staged()                        what you have staged so far\n"
        "  unstage(artifact_id)                 drop one staged edit\n"
        "  submit_candidate(rationale)          finalize ONCE when you are done\n"
        "\n"
        "Read the artifacts you intend to change before changing them, then stage\n"
        "your edits, then call submit_candidate exactly once. Nothing you staged\n"
        "counts until submit_candidate has executed.\n"
        "\n"
        # Present in the live round that yielded 3 of 3 distinct candidates.
        # Removing it, together with shortening the shared contract, dropped the
        # optimizer to 0 of 3 NO_TOOL_CALL on the same dataset.
        "Write and execute Python code that calls these tools, then report the\n"
        "exact values they returned. Narration stages nothing.\n"
    )


# ====================================================================== #
# Tool callables: plain Python, no CUGA dependency
# ====================================================================== #


def build_optimizer_callables(
    base_artifacts: Mapping[str, str],
    diagnoses: Sequence[object],
    staging: EditStagingArea,
    plan: dict,
) -> dict[str, Callable[..., str]]:
    """Build one invocation's tool callables.

    Every callable has a docstring and a real typed signature: LangChain's
    ``@tool`` refuses a body with no docstring, and it derives the args schema
    from the signature -- a ``*args`` body silently tells the model that every
    tool takes no arguments.

    Rejections are RETURNED, never raised: an exception inside a CUGA tool body
    can abort the whole run, turning a recoverable authorization mistake into a
    lost invocation.
    """
    ordered = select_diagnoses(diagnoses)
    by_id = {_task_id(d): d for d in ordered}

    def list_diagnoses() -> str:
        """List every observed task diagnosis, most severe first."""
        return json.dumps(
            {
                "diagnoses": [
                    {
                        "task_id": _task_id(d),
                        "severity": _severity(d),
                        "recurring_failure_mode": str(
                            _get(d, "recurring_failure_mode", "")
                        ),
                        "candidate_surfaces": list(_surfaces(d)),
                    }
                    for d in ordered
                ]
            }
        )

    def read_diagnosis(task_id: str) -> str:
        """Return one task's diagnosis in full, including any disagreements."""
        diagnosis = by_id.get(task_id)
        if diagnosis is None:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"unknown task_id {task_id!r}",
                    "known": sorted(by_id),
                }
            )
        return json.dumps(
            {
                "status": "ok",
                "task_id": task_id,
                "recurring_failure_mode": str(
                    _get(diagnosis, "recurring_failure_mode", "")
                ),
                "improvement_direction": str(
                    _get(diagnosis, "improvement_direction", "")
                ),
                "severity": _severity(diagnosis),
                "rollouts_seen": _get(diagnosis, "rollouts_seen", 0),
                "disagreements": list(_strings(diagnosis, "disagreements")),
                "self_validation_observed": bool(
                    _get(diagnosis, "self_validation_observed", False)
                ),
                "candidate_surfaces": list(_surfaces(diagnosis)),
            }
        )

    def list_artifacts() -> str:
        """List the harness artifacts you may read and edit."""
        return json.dumps(
            {
                "artifacts": sorted(base_artifacts),
                "creatable_prefixes": list(CREATABLE_PREFIXES),
            }
        )

    def read_artifact(artifact_id: str) -> str:
        """Return one artifact's current content verbatim."""
        if artifact_id not in base_artifacts:
            return json.dumps(
                {
                    "status": "error",
                    "message": f"unknown artifact {artifact_id!r}",
                    "known": sorted(base_artifacts),
                }
            )
        return json.dumps(
            {
                "status": "ok",
                "artifact_id": artifact_id,
                "content": base_artifacts[artifact_id],
            }
        )

    def stage_replace(artifact_id: str, content: str) -> str:
        """Stage a full rewrite of an existing artifact. This overwrites it."""
        outcome = staging.stage_replace(artifact_id, content)
        return json.dumps({"accepted": outcome.accepted, "reason": outcome.reason})

    def stage_create(artifact_id: str, content: str) -> str:
        """Stage a brand-new artifact. The id must carry the creatable prefix."""
        outcome = staging.stage_create(artifact_id, content)
        return json.dumps({"accepted": outcome.accepted, "reason": outcome.reason})

    def list_staged() -> str:
        """List the artifact ids you have staged so far."""
        return json.dumps({"staged": list(staging.staged_ids())})

    def unstage(artifact_id: str) -> str:
        """Drop one staged edit, restoring the base content for that artifact."""
        outcome = staging.unstage(artifact_id)
        return json.dumps({"accepted": outcome.accepted, "reason": outcome.reason})

    def submit_candidate(rationale: str) -> str:
        """Finalize this candidate. Call exactly once, or your work is discarded."""
        if "rationale" in plan:
            return json.dumps(
                {
                    "status": "error",
                    "message": "already finalized; the first rationale stands",
                }
            )
        plan["rationale"] = str(rationale)
        return json.dumps(
            {"status": "ok", "staged": list(staging.staged_ids())}
        )

    return {
        "list_diagnoses": list_diagnoses,
        "read_diagnosis": read_diagnosis,
        "list_artifacts": list_artifacts,
        "read_artifact": read_artifact,
        "stage_replace": stage_replace,
        "stage_create": stage_create,
        "list_staged": list_staged,
        "unstage": unstage,
        "submit_candidate": submit_candidate,
    }


def _fingerprint(artifacts: Mapping[str, str]) -> str:
    payload = json.dumps(dict(sorted(artifacts.items())), sort_keys=True)
    return sha256(payload.encode()).hexdigest()


# ====================================================================== #
# The optimizer
# ====================================================================== #


#: Per-candidate framings, appended so each of the N invocations reads a
#: DIFFERENT final instruction.
#:
#: Two problems are solved by the same change, both observed on live rounds:
#:
#: 1. Correlated failure. Tool invocation is a deterministic, all-or-nothing
#:    function of prompt wording, and reasoning models skip temperature
#:    ("Skipping temperature for reasoning model" in CUGA's log), so decoding is
#:    effectively greedy. N invocations of one byte-identical prompt are ONE
#:    sample repeated N times: one live round discarded 3 of 3 as NO_TOOL_CALL,
#:    another discarded 3 of 3 as NO_OP. Either all N narrate or all N act.
#: 2. Collapsed diversity. The paper's design draws its diversity from divergent
#:    tool trajectories across N independent invocations. Identical prompts under
#:    greedy decoding cannot diverge, so ``distinct`` is capped at 1 even when
#:    every invocation succeeds.
#:
#: Each framing names a different surface to consider first, so the N candidates
#: explore genuinely different repairs instead of re-deriving one.
# _CANDIDATE_FRAMINGS: tuple[str, ...] = (
#     "Write and execute Python code that calls list_artifacts(), then "
#     "read_artifact() on the artifact the diagnosis most directly implicates. "
#     "Make the SMALLEST edit that closes the gap, then submit_candidate.",
#     "Write and execute Python code that calls list_artifacts() and read every "
#     "artifact before deciding. Prefer changing the artifact that DELIVERS the "
#     "missing behaviour at the moment it is needed, even if the diagnosis names "
#     "a different one. Then stage your edit and submit_candidate.",
#     "Write and execute Python code that calls list_artifacts(), then "
#     "read_artifact() on each candidate surface. Consider whether a NEW "
#     "generated skill closes the gap more directly than editing existing text; "
#     "stage either, then submit_candidate.",
# )
#: EMPTY by design. The framings existed to work around a misdiagnosis: identical
#: prompts were collapsing because the upstream gateway served a CACHED response
#: (verified: four identical requests shared one response ``id``), not because
#: reasoning models decode greedily. With the cache disabled at the transport
#: layer the N invocations diverge on their own, and a per-candidate nudge is a
#: confound -- it permanently biases which surface each candidate considers
#: first, so "candidate 3 created a skill" would be an artifact of the prompt
#: rather than a finding. Restore only as a deliberate, documented ablation.
_CANDIDATE_FRAMINGS: tuple[str, ...] = ()

def _per_candidate_prompt(prompt: str, index: int, n: int) -> str:
    """Append a per-candidate framing so the N prompts are not byte-identical.

    Deterministic in ``index``, so rerunning a round reproduces the same N
    prompts. Only the framing rotates; the evidence is identical for all N.

    With ``_CANDIDATE_FRAMINGS`` empty the N prompts become byte-identical by
    design -- the intended state now that the upstream response cache is
    disabled, since diversity should come from independent trajectories rather
    than from a per-candidate nudge that also biases WHICH repair each candidate
    explores. ``APPROACH i of n`` is still appended so a transcript remains
    attributable to its invocation.
    """
    framing = (
        _CANDIDATE_FRAMINGS[index % len(_CANDIDATE_FRAMINGS)]
        if _CANDIDATE_FRAMINGS
        else ""
    )
    return f"{prompt}\nAPPROACH {index + 1} of {n}\n\n{framing}\n"


@dataclass(slots=True)
class RhoOptimizer:
    """Interface B optimizer: N independent candidate proposals.

    ``agent_factory`` is the offline test seam. When ``None`` a real
    ``CugaAgent`` is constructed inside ``run_workspace_agent``.
    """

    agent_factory: Callable[[dict, str], str] | None = None
    app_names: Mapping[str, str] = field(default_factory=lambda: dict(APP_NAMES))
    #: Ablation only, and NEVER 0.0 -- the endpoint rejects that value. Diversity
    #: comes from N independent trajectories, not from sampling.
    temperature: float | None = None

    def propose(
        self,
        base_artifacts: Mapping[str, str],
        diagnoses: Sequence[object],
        n: int,
    ) -> ProposalReport:
        """Run ``n`` independent proposals and return every outcome.

        ``diagnoses`` accepts mappings or objects carrying ``task_id``,
        ``severity``, ``recurring_failure_mode``, ``improvement_direction``,
        ``candidate_surfaces``, ``rollouts_seen``, ``disagreements``,
        ``self_validation_observed``, and ``observed``.

        Never raises for an agent failure: a failed invocation is one discarded
        proposal, not a lost round.
        """
        if self.temperature is not None and float(self.temperature) == 0.0:
            raise ValueError(
                "temperature=0.0 is rejected by the endpoint; omit it instead"
            )
        if n < 1:
            raise ValueError("n must be >= 1")
        if not base_artifacts:
            raise ValueError("base_artifacts must not be empty")

        base = dict(base_artifacts)
        ordered = select_diagnoses(diagnoses)
        prompt = build_optimizer_prompt(base, ordered)

        base_fingerprint = _fingerprint(base)
        seen: set[str] = set()
        survivors: list[ProposedCandidate] = []
        discarded: list[tuple[int, str]] = []

        for index in range(n):
            # A fresh staging area per invocation is what makes these workspaces
            # independent; sharing one would let candidate 1 inherit candidate
            # 0's edits and collapse the diversity this stage exists to produce.
            staging = EditStagingArea(
                write_set=tuple(sorted(base)),
                creatable_prefixes=CREATABLE_PREFIXES,
            )
            plan: dict = {}
            callables = build_optimizer_callables(base, ordered, staging, plan)
            run = run_workspace_agent(
                callables,
                _per_candidate_prompt(prompt, index, n),
                app_names=self.app_names,
                special_instructions=OPTIMIZER_INSTRUCTIONS,
                agent_factory=self.agent_factory,
            )

            if not run.ok:
                discarded.append((index, f"UNAVAILABLE: {run.error}"))
                continue
            if run.no_tool_call:
                discarded.append(
                    (
                        index,
                        "NO_TOOL_CALL: the agent executed no tool; it narrated "
                        "instead of emitting an executable block",
                    )
                )
                continue
            if "rationale" not in plan:
                discarded.append(
                    (
                        index,
                        "NOT_FINALIZED: never executed submit_candidate, so the "
                        "staged work is discarded",
                    )
                )
                continue

            edits = staging.edits()
            if not edits:
                discarded.append(
                    (
                        index,
                        "NO_OP: finalized with nothing staged; a no-op is not a "
                        "candidate",
                    )
                )
                continue

            artifacts = dict(base)
            for edit in edits:
                artifacts[edit.artifact_id] = str(edit.payload.get("content", ""))

            fingerprint = _fingerprint(artifacts)
            if fingerprint == base_fingerprint:
                discarded.append(
                    (
                        index,
                        "IDENTICAL: artifact set is byte-identical to the base; "
                        "a no-op is not a candidate",
                    )
                )
                continue
            if fingerprint in seen:
                discarded.append(
                    (
                        index,
                        "DUPLICATE: identical to a candidate an earlier "
                        "invocation already produced",
                    )
                )
                continue

            seen.add(fingerprint)
            survivors.append(
                ProposedCandidate(
                    candidate_index=index,
                    artifacts=artifacts,
                    rationale=str(plan["rationale"]),
                    tools_called=run.tools_called,
                    observed=True,
                    edited_ids=tuple(
                        e.artifact_id for e in edits if e.operation == "replace"
                    ),
                    created_ids=tuple(
                        e.artifact_id for e in edits if e.operation == "create"
                    ),
                    fingerprint=fingerprint,
                )
            )

        # Every survivor is retained. This stage never prunes to a best-of-N;
        # each distinct candidate enters the pool as a parent.
        return ProposalReport(
            candidates=tuple(survivors),
            requested=n,
            discarded=tuple(discarded),
        )
