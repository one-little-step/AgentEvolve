"""Interface A: historical-task difficulty and abstract failure fingerprint.

Why this is a plain structured call, not an agentic one
------------------------------------------------------
This matches the paper's own split (``RHO_agents_context.md:194-236``): the
stage needs a bounded classification plus a bounded abstraction, over evidence
that is already in hand. There is nothing to explore, no tool to call, and no
artifact to stage, so an agentic invocation would only add cost and variance.

What this feeds
---------------
The two outputs are the *only* two inputs to the coreset DPP
(``core/rho/coreset.py``): ``difficulty`` becomes the quality vector and
``abstract_fingerprint`` becomes the text that gets embedded for the diversity
kernel. Both failure modes of this module are therefore silent and expensive:

* If ``difficulty`` collapses onto "the agent failed, so this was hard", quality
  degenerates into a restatement of the base harness's error set and RHO stops
  preferring *intrinsically* hard tasks.
* If ``abstract_fingerprint`` carries task vocabulary (filenames, people,
  places, domain nouns), cosine similarity measures shared subject matter rather
  than shared failure structure, every pair looks moderately similar, the
  diversity term stops discriminating, and the coreset degenerates to a
  difficulty ranking. This is the same dilution failure that §4.2 of the spec
  documents for raw-trace embeddings, one level up.

The prompt below is built against those two degeneracies specifically: an
anchored difficulty rubric that separates task hardness from observed outcome,
and a three-axis fingerprint over a *controlled vocabulary* so that two tasks
with the same structure emit literally overlapping phrases and two tasks with
different structure do not.

Input policy
------------
The judge consumes the comprehended *summary text* (Task 3's semantic summary)
as a plain ``str``, not the raw trace: the summary is bounded prose about
behaviour, whereas ``record.final_output`` may contain the expected answer.

Ground truth is an explicit opt-in parameter (``expected_answer``). The user has
overridden the repository-wide no-labels rule for this judge, because a
calibrated difficulty score needs to know whether the committed answer was
actually wrong or merely unverified. Containment is therefore enforced two ways:
the prompt forbids echoing the answer, and any fingerprint that contains it is
rejected as unobserved rather than accepted. Verdicts judged with ground truth
are cached under a distinct key so a GT-calibrated score can never be served to
a caller that asked for a GT-free one.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable

from agent_evolve.core.rho.cache import JsonDiskCache
from agent_evolve.core.rho.history import HistoricalRecord

#: Identifies this judge in cache keys, so changing the prompt family or the
#: output contract does not silently reuse verdicts from the old one.
JUDGE_MODEL_ID = "cuga-rho-difficulty-judge/v1"

MIN_DIFFICULTY = 0.0
MAX_DIFFICULTY = 10.0

#: Fingerprints longer than this are truncation-prone noise for the embedder and
#: usually mean the model narrated the task instead of abstracting it.
MAX_FINGERPRINT_CHARS = 1200


class JudgeConfigurationError(RuntimeError):
    """No usable model could be resolved for a live judge call."""


JUDGE_SYSTEM_PROMPT = """\
You are a task-difficulty analyst for an agent-improvement pipeline. You are
given a behavioural summary of ONE past attempt by an autonomous agent at ONE
task. You return two things: a difficulty score for the TASK, and an abstract
structural fingerprint of the task and of where the attempt broke down.

Return ONLY a JSON object, no prose outside it, with exactly these two keys:

  "difficulty"            a number in [0, 10], one decimal place
  "abstract_fingerprint"  three lines, in the fixed format given below

=========================== 1. difficulty ===========================

Score the INTRINSIC difficulty of the task for a competent tool-using agent.
Score the task, not the transcript.

The single most common error here is treating failure as evidence of
difficulty. It is not. An agent that never called a tool at all tells you
almost nothing about how hard the task was; an agent that executed six correct
retrievals and lost the thread on the seventh tells you a great deal. A failed
attempt is not evidence of a hard task, and a successful attempt is not
evidence of an easy one. Judge the work the task actually demands.

Anchored bands - pick the band first, then place within it to one decimal:

  0-1  One fact, one obvious source, no composition. A competent agent gets
       this right essentially always.
  2-3  Two or three steps with an unambiguous order, each step's target named
       plainly in the task. Little judgement required.
  4-5  Real multi-step work: the agent must decide what to look for before it
       can look for it, or must reconcile two sources, or must apply a stated
       constraint while gathering. One plausible wrong turn exists.
  6-7  The task under-specifies something the agent must infer, OR requires
       composing results across several steps where an early mistake silently
       poisons the rest, OR demands an exact form (unit, ordering, rounding,
       enumeration) that is easy to get almost right.
  8-9  Several interacting constraints, or a search space with no obvious
       entry point, or evidence that must be cross-verified because sources
       plausibly disagree. A competent agent fails this a meaningful fraction
       of the time.
   10  Requires capability or access the agent plausibly does not have, or is
       under-determined enough that no reliable procedure exists.

Calibration guards:
- Do not cluster on 7-8. If you cannot justify a band above 5, use a lower one.
- If the summary says the agent never executed any tool, that is a harness
  failure. Score the task on its own demands and do NOT inflate the score.
- If ground truth is supplied, use it only to tell a genuinely wrong answer
  apart from an unverified one, and to see how exacting the required output
  form is. It is not a difficulty signal by itself.

====================== 2. abstract_fingerprint ======================

This string is embedded and compared against other tasks' fingerprints. Its
only job is to make two structurally similar failures land near each other and
two structurally different failures land far apart. Shared subject matter must
NOT pull two fingerprints together.

Emit EXACTLY three lines, each beginning with the given label:

  task shape: <one phrase from the list below>
  binding constraint: <one clause: the thing that actually made this hard>
  failure locus: <one phrase from the list below>, <one clause of detail>

Choose "task shape" from this closed list; pick the closest, do not invent:
  single-hop lookup
  multi-hop composition
  aggregation over a set
  constraint satisfaction
  comparison between candidates
  procedural execution
  extraction from a structured artifact
  ambiguous or under-specified request

Choose "failure locus" from this closed list; pick the closest, do not invent:
  never acted
  wrong tool
  right tool wrong query
  retrieved but misread
  looped without progress
  lost intermediate state
  committed without verifying
  correct process wrong output form
  no failure

Hard rules for all three lines:
- Never name a specific file, repository, framework, function, product, person,
  place, organisation, date, or number drawn from the task. Describe the ROLE a
  thing played, not its identity: "a reference page for one entity" rather than
  the entity's name.
- Never include the answer, any candidate answer, or any value from the task.
- Write "binding constraint" so that a task about a completely different
  subject with the same structure would produce nearly the same clause.
- If two attempts broke in the same way, their fingerprints must read almost
  identically. Do not add distinguishing colour for its own sake.

Example of a correct fingerprint (structure only, no identifiers):

  task shape: multi-hop composition
  binding constraint: the second step's target is only knowable after the first
    step resolves, so the agent must sequence retrieval rather than batch it
  failure locus: never acted, the agent narrated a complete plan in prose and
    then emitted an answer without executing any step of it
"""

_GT_PREAMBLE = """\
GROUND TRUTH (calibration only): the recorded correct answer is given below.
Use it ONLY to distinguish a wrong answer from an unverified one and to gauge
how exacting the required output form is. It must not appear, in whole or in
part, anywhere in your abstract_fingerprint, and it must not be treated as a
difficulty signal in itself.
"""

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


@dataclass(frozen=True, slots=True)
class DifficultyVerdict:
    """One judged historical record.

    ``observed`` is the only field a caller may trust without checking anything
    else. When it is ``False`` the record must be excluded from
    difficulty-weighted selection and counted in the round report; ``difficulty``
    stays at its neutral default precisely so that a caller who ignores
    ``observed`` produces a visibly wrong coreset rather than a plausible one
    built on a fabricated score.
    """

    task_id: str
    difficulty: float = 0.0
    abstract_fingerprint: str = ""
    observed: bool = False
    error: str = ""
    #: True when ground truth was supplied to the call that produced this
    #: verdict. Carried so a manifest can show which scores are GT-calibrated.
    ground_truth_used: bool = False


@dataclass(slots=True)
class RhoDifficultyJudge:
    """Interface A difficulty and abstract-fingerprint judge.

    Every field is optional so the judge can be constructed without credentials
    and resolved lazily on first use, matching
    :class:`~agent_evolve.adapters.cuga_analyzer.CugaTrajectoryAnalyzer`.
    """

    completion_fn: Callable[..., object] | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    #: Forwarded only when set. NEVER pass 0.0: the endpoint rejects the request
    #: outright ("'temperature' does not support 0.0 with this model"), so a
    #: pinned zero would make the judge unusable rather than deterministic.
    temperature: float | None = None
    #: Defaults to a disabled cache: a measurement run must be able to spend
    #: nothing on capture.
    cache: JsonDiskCache = field(default_factory=lambda: JsonDiskCache(None))
    max_fingerprint_chars: int = MAX_FINGERPRINT_CHARS

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #
    def judge(
        self,
        record: HistoricalRecord,
        summary_text: str,
        *,
        expected_answer: str | None = None,
    ) -> DifficultyVerdict:
        """Judge one record from its comprehended summary text.

        ``summary_text`` is the phase-2 semantic summary rendered as prose (a
        plain ``str``, so this module does not depend on the comprehender's
        dataclass). ``expected_answer`` is opt-in ground truth used for
        calibration only.

        Never raises for a model or response problem: those become unobserved
        verdicts, because a failure to judge a record is information about that
        record. Raises only for a caller error (no model resolvable, or an
        explicitly configured ``temperature=0.0``).
        """
        gt = (expected_answer or "").strip()
        summary = (summary_text or "").strip()
        if not summary:
            # No summary means the comprehension stage already failed. Spending
            # a judge call on an empty prompt would buy a fabricated score.
            return DifficultyVerdict(
                task_id=record.task_id,
                error="empty trajectory summary: nothing to judge",
                ground_truth_used=bool(gt),
            )

        key = self.cache_key(record, expected_answer=expected_answer)
        cached = self.cache.get(key)
        if cached is not None:
            return self._verdict_from_payload(
                record, cached, gt, source="cached verdict"
            )

        request = self._request(record, summary, gt)
        invoke = self.completion_fn or _litellm_completion
        try:
            response = invoke(**request)
        except Exception as exc:  # noqa: BLE001 - a failed call is data
            return DifficultyVerdict(
                task_id=record.task_id,
                error=f"judge call failed: {type(exc).__name__}: {exc}",
                ground_truth_used=bool(gt),
            )

        payload, parse_error = _parse_payload(_response_text(response))
        if payload is None:
            return DifficultyVerdict(
                task_id=record.task_id,
                error=parse_error,
                ground_truth_used=bool(gt),
            )

        verdict = self._verdict_from_payload(
            record, payload, gt, source="judge response"
        )
        if verdict.observed:
            self.cache.put(
                key,
                {
                    "difficulty": verdict.difficulty,
                    "abstract_fingerprint": verdict.abstract_fingerprint,
                },
            )
        return verdict

    def cache_key(
        self,
        record: HistoricalRecord,
        *,
        expected_answer: str | None = None,
    ) -> str:
        """Cache key for one record under this judge's configuration.

        Includes the trace content hash, so an edited trace can never produce a
        false hit, and a ground-truth marker, so a GT-calibrated score is never
        served to a GT-free caller (or the reverse).
        """
        gt_marker = "gt" if (expected_answer or "").strip() else "nogt"
        return (
            f"{JUDGE_MODEL_ID}|{self.model}|{gt_marker}"
            f"|{record.task_id}|{record.content_hash}"
        )

    # ------------------------------------------------------------------ #
    # Request construction
    # ------------------------------------------------------------------ #
    def _request(
        self, record: HistoricalRecord, summary: str, gt: str
    ) -> dict[str, object]:
        model, base_url, api_key = self._resolve_settings()
        request: dict[str, object] = {"model": model}
        if self.temperature is not None:
            if self.temperature == 0.0:
                raise ValueError(
                    "temperature=0.0 is rejected by the endpoint; omit it "
                    "instead of pinning a zero"
                )
            request["temperature"] = self.temperature
        if base_url:
            request["api_base"] = base_url
        if api_key:
            request["api_key"] = api_key
        request["messages"] = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(record, summary, gt)},
        ]
        return request

    def _user_prompt(
        self, record: HistoricalRecord, summary: str, gt: str
    ) -> str:
        parts = [f"TASK REQUEST:\n{record.input_text}"]
        if gt:
            parts.append(f"{_GT_PREAMBLE}\nRECORDED CORRECT ANSWER:\n{gt}")
        parts.append(f"BEHAVIOURAL SUMMARY OF THE ATTEMPT:\n{summary}")
        parts.append(
            "Return the JSON object now: difficulty, then the three-line "
            "abstract_fingerprint."
        )
        return "\n\n".join(parts)

    def _resolve_model(self) -> str:
        return self._resolve_settings()[0]

    def _resolve_settings(self) -> tuple[str, str | None, str | None]:
        """``(model, base_url, api_key)``, falling back to the environment.

        All three must fall back together. Resolving only the model leaves a
        live run pointed at the right model name with no credentials, which the
        endpoint rejects as ``Missing credentials`` -- and upstream that surfaces
        only as an unobserved verdict, i.e. as a task with no difficulty signal
        rather than as a configuration error.
        """
        model, base_url, api_key = self.model, self.base_url, self.api_key
        if model is None or base_url is None or api_key is None:
            env_model, env_base, env_key = _env_settings()
            model = model or env_model
            base_url = base_url or env_base
            api_key = api_key or env_key
        if not model:
            raise JudgeConfigurationError(
                "no judge model configured: set CUGA_MODEL or LITELLM_MODEL, "
                "or pass model= explicitly"
            )
        return model, base_url, api_key

    # ------------------------------------------------------------------ #
    # Validation
    # ------------------------------------------------------------------ #
    def _verdict_from_payload(
        self,
        record: HistoricalRecord,
        payload: Mapping[str, object],
        gt: str,
        *,
        source: str,
    ) -> DifficultyVerdict:
        """Validate a payload from either the model or the cache.

        Cached entries are revalidated rather than trusted: the cache is a plain
        directory of JSON files that a parallel batch or a human may have
        touched, and a poisoned entry must fail visibly instead of silently
        steering the coreset.
        """
        used_gt = bool(gt)

        def rejected(reason: str) -> DifficultyVerdict:
            return DifficultyVerdict(
                task_id=record.task_id,
                error=f"{source} rejected: {reason}",
                ground_truth_used=used_gt,
            )

        raw = payload.get("difficulty")
        if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
            return rejected(f"difficulty is not a number: {raw!r}")
        try:
            difficulty = float(raw)
        except (TypeError, ValueError):
            return rejected(f"difficulty is not a number: {raw!r}")
        if difficulty != difficulty:  # NaN
            return rejected("difficulty is not a number: NaN")
        if not MIN_DIFFICULTY <= difficulty <= MAX_DIFFICULTY:
            return rejected(
                f"difficulty {difficulty} outside "
                f"[{MIN_DIFFICULTY}, {MAX_DIFFICULTY}]"
            )

        fingerprint_raw = payload.get("abstract_fingerprint")
        fingerprint = (
            fingerprint_raw.strip() if isinstance(fingerprint_raw, str) else ""
        )
        if not fingerprint:
            return rejected("abstract_fingerprint is missing or empty")
        if len(fingerprint) > self.max_fingerprint_chars:
            return rejected(
                f"abstract_fingerprint is {len(fingerprint)} chars, over the "
                f"{self.max_fingerprint_chars}-char limit"
            )
        if gt and gt.lower() in fingerprint.lower():
            # Containment is enforced on the output, not only asked for in the
            # prompt: a leaked answer in a fingerprint would be embedded and
            # then exported with the coreset.
            return rejected(
                "abstract_fingerprint contains the recorded answer"
            )

        return DifficultyVerdict(
            task_id=record.task_id,
            difficulty=difficulty,
            abstract_fingerprint=fingerprint,
            observed=True,
            ground_truth_used=used_gt,
        )


# ---------------------------------------------------------------------- #
# Response handling
# ---------------------------------------------------------------------- #
def _parse_payload(text: str) -> tuple[dict | None, str]:
    """``(payload, error)`` from a response body, tolerating a code fence."""
    body = text.strip()
    if not body:
        return None, "judge returned an empty response body"
    fenced = _FENCE.search(body)
    if fenced:
        body = fenced.group(1).strip()
    else:
        start, end = body.find("{"), body.rfind("}")
        if start != -1 and end > start:
            body = body[start : end + 1]
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, f"unparseable judge response: {exc}"
    if not isinstance(parsed, dict):
        return None, (
            f"unparseable judge response: expected a JSON object, got "
            f"{type(parsed).__name__}"
        )
    return parsed, ""


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
    from agent_evolve.cuga_wrapper.retry_policy import resolve_max_retries as _ae_rr

    _ae_retries = _ae_rr()
    if _ae_retries and "num_retries" not in request:
        request["num_retries"] = _ae_retries
    return litellm.completion(**request)


def _response_text(response: object) -> str:
    """First assistant text from an OpenAI/litellm-shaped response.

    Returns ``""`` for any unexpected shape; the caller maps that to an
    unobserved verdict rather than raising.
    """
    if isinstance(response, str):
        return response
    choices = (
        response.get("choices")
        if isinstance(response, Mapping)
        else getattr(response, "choices", None)
    )
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


def _env_settings() -> tuple[str | None, str | None, str | None]:
    """``(model, base_url, api_key)`` from the environment, blanks when absent.

    Imported lazily: an offline unit test that injects ``completion_fn`` and a
    model name has no reason to load the CUGA wrapper.
    """
    try:
        from agent_evolve.cuga_wrapper import RuntimeSettings
    except Exception:  # noqa: BLE001 - absence of the wrapper is not fatal
        return None, None, None
    try:
        settings = RuntimeSettings.from_env()
    except RuntimeError:
        return None, None, None
    return settings.model, settings.base_url, settings.api_key
