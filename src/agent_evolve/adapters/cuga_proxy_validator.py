"""Counterfactual proxy validator: a cheap A/B over ONE recorded LLM call.

Purpose. Deciding whether an edit to an artifact would have fixed a failure
normally costs a full agent rollout. This module estimates the same thing for
roughly two LLM requests: it takes one *recorded* LLM call out of a persisted
trace, replays it twice - once verbatim (arm A) and once with the artifact text
substituted inside the recorded prompt (arm B) - and compares how often a
predicate holds across the sampled completions.

Scope, stated bluntly. This is NOT counterfactual agent replay. Nothing here
restores agent state or resumes a trajectory, and
``supports_counterfactual_replay`` stays ``False``. A verdict produced here is
*proxy* evidence about one prompt boundary, never a confirmed outcome about a
task. ``ProxyVerdict.evidence_kind`` is pinned to ``"proxy"`` and the dataclass
refuses any other value so a proxy result cannot be laundered into a confirmed
one downstream.

Sampling design, forced by measured endpoint behaviour
(``openai/azure/gpt-5.6-luna``, measured live):

1. ``temperature`` is rejected for any non-default value: ``temperature=0.0``
   returns ``BadRequestError: Unsupported value: 'temperature' does not support
   0.0``. This module therefore never sends ``temperature``.
2. Identical *sequential* requests are served from a cache. The same prompt
   issued three times returned the identical string three times (1 distinct of
   3). ``k`` sequential requests do not yield ``k`` samples.
3. ``n=3`` inside a *single* request does return three genuinely distinct
   completions (measured: ``['Boustrophedon', 'Brindleweed', 'Brumation']``).

So each arm is exactly ONE request carrying ``n=k``, and the two arms run in
parallel threads because they are independent network calls.

Cache caveat, stated honestly. Because the cache keys on the request, an
identical ``n=k`` request repeated later returns the SAME ``k`` completions.
Re-running an identical comparison is therefore NOT a statistically independent
second trial: it re-reads the first trial's answer. Two verdicts over the same
(call, substitution, k) triple are one observation, not two. Any confidence
statement built by repeating an identical A/B is invalid.
"""
from __future__ import annotations

import re
import threading
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from agent_evolve.cuga_wrapper import RecordedCall, replay_single_llm_call

__all__ = [
    "ProxyArmResult",
    "ProxySubstitutionError",
    "ProxyVerdict",
    "artifact_text_substitution",
    "calls_tool",
    "contains_all",
    "matches_regex",
    "run_proxy_ab",
]

Messages = list[dict[str, str]]
Substitution = Callable[[Messages], Sequence[Mapping[str, object]]]
Predicate = Callable[[str], bool]

_EVIDENCE_KIND = "proxy"


class ProxySubstitutionError(RuntimeError):
    """Raised when a substitution cannot produce a meaningful A/B comparison.

    A substitution that silently fails to change anything is the single worst
    failure mode of this module: both arms would then be byte-identical, the
    delta would be an artefact of sampling noise, and the verdict would look
    like real evidence about an edit that was never applied. Every such case
    fails loudly instead.
    """


@dataclass(frozen=True, slots=True)
class ProxyArmResult:
    """One arm of the A/B: the sampled completions and their predicate scores.

    ``predicate_errors`` counts completions the predicate raised on. Those are
    counted as non-passes (a predicate that cannot score a completion has not
    observed a pass) but are reported separately so a broken predicate is not
    mistaken for a genuinely failing arm.

    ``request_count`` records how many provider requests this arm actually
    cost. The design intends exactly 1. A value above 1 means the provider
    returned fewer than ``k`` choices and the replay topped up sequentially -
    and because sequential identical requests are cached, those top-ups are
    likely duplicates rather than samples.
    """

    completions: tuple[str, ...]
    pass_count: int
    k: int
    predicate_errors: int = 0
    request_count: int = 1

    @property
    def pass_rate(self) -> float:
        if self.k <= 0:
            return 0.0
        return self.pass_count / self.k

    @property
    def distinct_count(self) -> int:
        """How many of the ``k`` completions are distinct strings.

        This is the control measurement for the sampling design: if this is 1
        for ``k > 1``, ``n=k`` did not actually give independent samples on this
        prompt and any delta is not interpretable as a rate difference.
        """
        return len(set(self.completions))

    @property
    def scored_count(self) -> int:
        return len(self.completions) - self.predicate_errors


@dataclass(frozen=True, slots=True)
class ProxyVerdict:
    """The comparison of the two arms. Proxy evidence only, never confirmed."""

    baseline: ProxyArmResult
    edited: ProxyArmResult
    predicate_name: str
    k: int
    evidence_kind: str = _EVIDENCE_KIND
    substitution_summary: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_kind != _EVIDENCE_KIND:
            raise ValueError(
                "ProxyVerdict.evidence_kind must be 'proxy': a single recorded-call "
                f"A/B can never produce {self.evidence_kind!r} evidence"
            )

    @property
    def delta(self) -> float:
        """``edited.pass_rate - baseline.pass_rate``. Zero when inconclusive."""
        if self.label == "inconclusive":
            return 0.0
        return self.edited.pass_rate - self.baseline.pass_rate

    @property
    def label(self) -> str:
        """``improved`` | ``no_change`` | ``regressed`` | ``inconclusive``."""
        if self.baseline.scored_count == 0 and self.edited.scored_count == 0:
            return "inconclusive"
        raw = self.edited.pass_rate - self.baseline.pass_rate
        # Pass rates are i/k, so exact equality is meaningful; the epsilon only
        # absorbs float division error.
        if raw > 1e-9:
            return "improved"
        if raw < -1e-9:
            return "regressed"
        return "no_change"


# ---------------------------------------------------------------------------
# substitution helpers
# ---------------------------------------------------------------------------


def artifact_text_substitution(
    old_text: str,
    new_text: str,
    *,
    roles: Sequence[str] = ("system",),
) -> Substitution:
    """Build a substitution that swaps ``old_text`` for ``new_text`` in a prompt.

    Harness/skill/policy artifact content reaches the model inside the recorded
    system message, so ``roles`` defaults to ``("system",)``. Every occurrence in
    every targeted message is replaced.

    Raises ``ProxySubstitutionError`` when ``old_text`` appears in no targeted
    message. A silent no-op substitution would produce an A/B whose two arms are
    identical prompts, which is a meaningless comparison dressed as evidence.
    """
    if not old_text:
        raise ValueError("old_text must not be empty: an empty match is not a substitution")
    if old_text == new_text:
        raise ValueError("old_text and new_text are identical: this substitution is a no-op")
    targeted = tuple(roles)
    if not targeted:
        raise ValueError("roles must name at least one message role")

    def substitution(messages: Messages) -> Messages:
        edited: Messages = []
        hits = 0
        for message in messages:
            content = str(message.get("content", ""))
            if message.get("role") in targeted and old_text in content:
                hits += content.count(old_text)
                edited.append({**dict(message), "content": content.replace(old_text, new_text)})
            else:
                edited.append(dict(message))
        if hits == 0:
            raise ProxySubstitutionError(
                f"old_text was not found in any {list(targeted)} message "
                f"({len(old_text)} chars, starts {old_text[:60]!r}); "
                "the A/B would compare two identical prompts"
            )
        return edited

    return substitution


# ---------------------------------------------------------------------------
# predicate factories
# ---------------------------------------------------------------------------


class _NamedPredicate:
    """A predicate carrying a stable ``name`` so verdicts are self-describing."""

    __slots__ = ("_fn", "name")

    def __init__(self, name: str, fn: Predicate) -> None:
        self.name = name
        self._fn = fn

    def __call__(self, completion: str) -> bool:
        return self._fn(completion)

    def __repr__(self) -> str:
        return f"<predicate {self.name}>"


def contains_all(terms: Sequence[str], *, case_sensitive: bool = False) -> _NamedPredicate:
    """Pass when every term occurs in the completion (case-insensitive default)."""
    selected = tuple(terms)
    if not selected:
        raise ValueError("contains_all requires at least one term")
    if any(not term for term in selected):
        raise ValueError("contains_all terms must be non-empty")

    def predicate(completion: str) -> bool:
        haystack = completion if case_sensitive else completion.lower()
        return all(
            (term if case_sensitive else term.lower()) in haystack for term in selected
        )

    return _NamedPredicate(f"contains_all({','.join(selected)})", predicate)


def matches_regex(pattern: str, *, flags: int = 0) -> _NamedPredicate:
    """Pass when the pattern is found anywhere in the completion."""
    compiled = re.compile(pattern, flags)

    def predicate(completion: str) -> bool:
        return compiled.search(completion) is not None

    return _NamedPredicate(f"matches_regex({pattern})", predicate)


def calls_tool(tool_name: str) -> _NamedPredicate:
    """Pass when the completion actually *invokes* ``tool_name``.

    Two invocation shapes are recognised, matching how the recorded agent emits
    tool use: a code-style call (``tool_name(...)``) and a JSON tool-call object
    (``{"name": "tool_name", ...}``, also ``tool_name``/``tool``/``function``
    keys). A bare prose mention is deliberately NOT a pass - "I could use
    search_docs but I will not" is not a tool call, and counting it would make
    the predicate reward talking about the tool instead of using it.
    """
    if not tool_name:
        raise ValueError("tool_name must not be empty")
    escaped = re.escape(tool_name)
    code_call = re.compile(rf"(?<![\w.]){escaped}\s*\(")
    json_call = re.compile(
        rf'"(?:name|tool_name|tool|function)"\s*:\s*"{escaped}"'
    )

    def predicate(completion: str) -> bool:
        return bool(code_call.search(completion) or json_call.search(completion))

    return _NamedPredicate(f"calls_tool({tool_name})", predicate)


# ---------------------------------------------------------------------------
# the A/B itself
# ---------------------------------------------------------------------------


class _CountingCompletion:
    """Wrap a completion callable to count how many requests an arm cost.

    The count matters: the replay helper tops up sequentially when a provider
    returns fewer than ``n`` choices, and identical sequential requests are
    cached, so a top-up silently substitutes duplicates for samples. Counting
    lets a caller see that happen instead of trusting a degraded sample.
    """

    __slots__ = ("_fn", "_lock", "count")

    def __init__(self, fn: Callable[..., object]) -> None:
        self._fn = fn
        self._lock = threading.Lock()
        self.count = 0

    def __call__(self, **request: object) -> object:
        with self._lock:
            self.count += 1
        return self._fn(**request)


def _resolve_completion_fn(completion_fn: Callable[..., object] | None) -> Callable[..., object]:
    if completion_fn is not None:
        return completion_fn
    # Reach for the wrapper's live completion so it can be wrapped in a counter.
    # Passing ``completion_fn=None`` through would let the replay pick its own
    # callable and the per-arm request count would be unobservable.
    from agent_evolve import cuga_wrapper

    return cuga_wrapper._litellm_completion


def _score(
    completions: Sequence[str], predicate: Predicate, k: int, request_count: int
) -> ProxyArmResult:
    passes = 0
    errors = 0
    for completion in completions:
        try:
            outcome = bool(predicate(completion))
        except Exception:  # noqa: BLE001 - a broken predicate must not kill the run
            errors += 1
            continue
        passes += 1 if outcome else 0
    return ProxyArmResult(
        completions=tuple(completions),
        pass_count=passes,
        k=k,
        predicate_errors=errors,
        request_count=request_count,
    )


def _substitution_summary(baseline: Messages, edited: Sequence[Mapping[str, object]]) -> dict[str, object]:
    before = "".join(str(message.get("content", "")) for message in baseline)
    after = "".join(str(message.get("content", "")) for message in edited)
    return {
        "baseline_chars": len(before),
        "edited_chars": len(after),
        "chars_added": max(0, len(after) - len(before)),
        "chars_removed": max(0, len(before) - len(after)),
        "baseline_message_count": len(baseline),
        "edited_message_count": len(edited),
    }


def run_proxy_ab(
    call: RecordedCall,
    *,
    substitution: Substitution,
    predicate: Predicate,
    predicate_name: str | None = None,
    k: int = 3,
    model: str | None = None,
    completion_fn: Callable[..., object] | None = None,
) -> ProxyVerdict:
    """Replay one recorded LLM call twice - verbatim and edited - and compare.

    Arm A replays ``call.messages`` unchanged; arm B replays
    ``substitution(call.messages)``. Each arm is exactly ONE request carrying
    ``n=k`` (see the module docstring for why ``k`` sequential requests would be
    cached rather than sampled), and the two arms run concurrently in threads.

    ``temperature`` is never sent: the reference endpoint rejects non-default
    values.

    Every completion is scored with ``predicate``. A predicate that raises is
    counted as a non-pass and tallied in ``ProxyArmResult.predicate_errors``.

    Raises ``ProxySubstitutionError`` before issuing any request when the
    substitution produced messages identical to the baseline (or produced none).
    Failing loudly is chosen over returning ``"inconclusive"``: identical arms
    are a caller bug, not an uncertain measurement, and an ``inconclusive``
    return would be silently discarded by a caller that only branches on
    improved/regressed.

    A returned verdict is proxy evidence about a single prompt boundary. It is
    not a confirmed task outcome.
    """
    if k < 1:
        raise ValueError("k must be a positive integer")

    baseline_messages = [dict(message) for message in call.messages]
    edited_messages = [dict(message) for message in substitution(baseline_messages)]
    if not edited_messages:
        raise ProxySubstitutionError("substitution returned no messages")
    if edited_messages == baseline_messages:
        raise ProxySubstitutionError(
            "substitution produced messages identical to the baseline: both arms "
            "would run the same prompt and the delta would be sampling noise"
        )

    resolved = _resolve_completion_fn(completion_fn)
    baseline_counter = _CountingCompletion(resolved)
    edited_counter = _CountingCompletion(resolved)

    def arm(
        messages: Sequence[Mapping[str, object]], counter: _CountingCompletion
    ) -> tuple[str, ...]:
        return replay_single_llm_call(
            call,
            messages=messages,
            model=model,
            # temperature deliberately omitted: any explicit value is rejected.
            n=k,
            completion_fn=counter,
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        baseline_future = pool.submit(arm, baseline_messages, baseline_counter)
        edited_future = pool.submit(arm, edited_messages, edited_counter)
        baseline_completions = baseline_future.result()
        edited_completions = edited_future.result()

    resolved_name = predicate_name or getattr(predicate, "name", None) or repr(predicate)
    return ProxyVerdict(
        baseline=_score(baseline_completions, predicate, k, baseline_counter.count),
        edited=_score(edited_completions, predicate, k, edited_counter.count),
        predicate_name=str(resolved_name),
        k=k,
        substitution_summary=_substitution_summary(baseline_messages, edited_messages),
    )
