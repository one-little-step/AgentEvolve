"""Model-backed adjudicator for ambiguous mechanism-cluster merges.

Embedding cosine decides the clear cases for free and is measurably unreliable in
the middle: on this codebase, genuine paraphrases of one fault scored ``0.769``
against a ``0.75`` join threshold, and four descriptions of one identical fault
using different vocabulary produced four separate clusters. This adjudicator is
consulted **only** inside that ambiguous band and on a forced merge at the cluster
cap, which is what keeps a model in this path affordable.

It implements :class:`~agent_evolve.core.clustering.MechanismAdjudicator`
structurally and is injected, never imported by ``core/``.

**Its own model role, deliberately.** Rollout, analyzer, judge and editor usually
want a strong reasoning model; deciding whether two one-line fault descriptions
name the same fault does not. ``AE_MECHANISM_DEDUP_MODEL`` /
``AE_MECHANISM_DEDUP_BASE_URL`` / ``AE_MECHANISM_DEDUP_API_KEY`` address this role
separately so a small cheap model can serve it.

**Conservative on every failure.** A provider outage, an unparseable answer, or a
model that will not commit all return ``None`` (abstain), and the caller keeps the
cosine decision and records that it was not adjudicated. A dedup outage must never
silently split or merge a mechanism cell.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

#: Nothing about the task, the expected answer, or the grader may appear here.
#: The adjudicator sees two mechanism descriptions and nothing else.
DEDUP_SYSTEM_PROMPT = """\
You compare two descriptions of why an autonomous agent failed a task, and decide
whether they describe THE SAME underlying failure mechanism.

Answer with exactly one word, lowercase, nothing else:

  same       the two describe one mechanism, differently worded
  different  the two describe genuinely different mechanisms
  unsure     you cannot tell from what you were given

What counts as the same mechanism: the same thing went wrong for the same reason.
Different wording, different level of detail, and different surface symptoms are
all still the same mechanism if the underlying fault is one fault.

What counts as different: two faults that would need two different fixes. If
repairing one would leave the other intact, they are different.

Judge the mechanism, not the phrasing. "Forgot to filter by date" and "the query
lacked a temporal constraint" are the SAME mechanism. "Forgot to filter by date"
and "never called the pricing API" are DIFFERENT.

Prefer "unsure" over a guess. A wrong "same" merges two unrelated faults into one
measurement bucket and produces a confident but meaningless variance reading; a
wrong "different" fragments the evidence for one fault. "unsure" costs only a
fallback to the cheaper heuristic, which is the safe outcome.
"""


class DedupConfigurationError(RuntimeError):
    """No usable model could be resolved for a live adjudication call."""


def _env_settings() -> tuple[str, str, str]:
    """``(model, base_url, api_key)`` from the environment; blanks when absent."""
    return (
        os.environ.get("AE_MECHANISM_DEDUP_MODEL", ""),
        os.environ.get("AE_MECHANISM_DEDUP_BASE_URL", ""),
        os.environ.get("AE_MECHANISM_DEDUP_API_KEY", ""),
    )


def _litellm_completion(**request: object) -> object:
    """Indirected so tests never import or reach a provider.

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


@dataclass
class CugaMechanismAdjudicator:
    """Adjudicates same-mechanism questions with a small configured model.

    Every field is optional so it can be constructed without credentials and
    resolved lazily on first use, matching the other adapters.
    """

    completion_fn: Callable[..., object] | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    #: Left unset by default: some endpoints reject ``temperature=0.0`` outright.
    temperature: float | None = None
    #: Cache verdicts so repeated pairs within a run cost one call. Mechanism text
    #: is deterministic input, so this is safe and materially cheaper.
    _cache: dict[tuple[str, str], bool | None] | None = None

    def __post_init__(self) -> None:
        if self._cache is None:
            self._cache = {}

    # ------------------------------------------------------------------ #
    # MechanismAdjudicator
    # ------------------------------------------------------------------ #
    def same_mechanism(self, left: str, right: str) -> bool | None:
        """``True`` same, ``False`` different, ``None`` abstain.

        Never raises for a provider or parsing failure: the caller treats ``None``
        as "not adjudicated" and keeps its cosine decision. Raising here would let
        a dedup outage take down a run whose clustering was merely going to be
        slightly coarser.
        """
        if not left or not right:
            return None
        if left == right:
            return True
        # Order-independent: the same pair must not depend on arrival order.
        key = (left, right) if left <= right else (right, left)
        assert self._cache is not None
        if key in self._cache:
            return self._cache[key]

        try:
            verdict = self._ask(key[0], key[1])
        except Exception:  # noqa: BLE001 - any failure is an abstention
            return None
        self._cache[key] = verdict
        return verdict

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _ask(self, left: str, right: str) -> bool | None:
        model, base_url, api_key = self._resolve_settings()
        request: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": DEDUP_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"A: {left}\nB: {right}\n\nsame, different, or unsure?",
                },
            ],
        }
        if self.temperature is not None:
            if self.temperature == 0.0:
                raise ValueError(
                    "temperature=0.0 is rejected by some endpoints; omit it "
                    "rather than passing zero"
                )
            request["temperature"] = self.temperature
        if base_url:
            request["api_base"] = base_url
        if api_key:
            request["api_key"] = api_key

        fn = self.completion_fn or _litellm_completion
        return _parse_verdict(_extract_text(fn(**request)))

    def _resolve_settings(self) -> tuple[str, str, str]:
        model, base_url, api_key = self.model, self.base_url, self.api_key
        if model is None or base_url is None or api_key is None:
            env_model, env_base, env_key = _env_settings()
            model = model or env_model
            base_url = base_url or env_base
            api_key = api_key or env_key
        if not model:
            raise DedupConfigurationError(
                "AE_MECHANISM_DEDUP_MODEL is required for a live mechanism "
                "adjudication call, or pass model= explicitly"
            )
        return model, base_url or "", api_key or ""


def _extract_text(response: object) -> str:
    """Pull the assistant text out of an OpenAI-shaped response."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, dict):
        message = first.get("message")
    content = getattr(message, "content", None)
    if content is None and isinstance(message, dict):
        content = message.get("content")
    return content if isinstance(content, str) else ""


def _parse_verdict(text: str) -> bool | None:
    """Map the model's word to a verdict; anything unrecognised abstains.

    Deliberately strict. A model that answers with a paragraph, hedges, or invents
    a fourth category has not given a usable verdict, and guessing at its
    intention is how a wrong merge gets made silently.
    """
    token = text.strip().lower().strip(".!\"'` \n\t")
    if not token:
        return None
    # Take the first word so a trailing explanation does not defeat the parse,
    # but refuse anything whose first word is not one of the three.
    first = token.split()[0].strip(".,:;")
    if first == "same":
        return True
    if first == "different":
        return False
    return None
