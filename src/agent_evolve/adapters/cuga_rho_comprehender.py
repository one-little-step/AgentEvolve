"""Interface A: turn a raw causal trace into a bounded semantic summary.

Why this stage exists at all
----------------------------
Measured on the canonical current-format trace
(``data/traces/0cb88c5a-.../causal-trace.json``, 9,610 bytes, 19 events):
40 UUIDs (1,440 bytes), 20 long hex hashes (1,280 bytes) and 260 JSON keys
(3,121 bytes) account for **60.8%** of the payload before braces and quotes.
Every trace shares the same schema vocabulary, so embedding raw traces makes
cosine similarity saturate near-uniformly high and the DPP diversity term stops
discriminating -- leaving coreset selection driven by difficulty alone.

Mechanical truncation does not fix this: it removes prose and keeps the schema
noise. So a model call converts the trace into prose about *behaviour*, and that
prose is what gets embedded. The previous Gaia RHO reached the same conclusion
(``trajectory_summary.md`` as the "preferred input to embedding").

This is an ordinary structured LLM call (Interface A), not a workspace agent:
the stage needs a bounded abstraction, not filesystem inspection.

Why the prompt is *rendered*, not dumped
----------------------------------------
The obvious implementation -- ``json.dumps(raw_trace["events"][:40])`` -- would
reproduce exactly the disease this module exists to cure. Verified against the
267-trace corpus in ``data/traces/``, a CUGA event payload is:

===================  =============================================================
``graph_node_start`` ``node``, ``run_id``, ``parent_run_id``, ``state_before_ref``, ``step``
``graph_node_end``   ``node``, ``routed_to``, ``run_id``, ``state_after_ref``
``llm_call_start``   ``messages_ref``, ``run_id``, ``parent_run_id``
``llm_call_end``     ``response_ref``, ``run_id``, ``parent_run_id``
``graph_tool_start`` ``tool_name``, ``run_id``, ``parent_run_id``
``graph_tool_end``   ``run_id``, ``parent_run_id``
``tool_call``        ``tool_call``: ``name``, ``arguments``, ``result``, ``error``, ``duration_ms``
===================  =============================================================

Prompt-relevant content is therefore *exactly*: the ``node``/``routed_to``
control-flow sequence, and the ``tool_call``/``tool_observations`` rows. The
``*_ref`` fields are content-addressed pointers to payloads that are not inlined
in the trace -- they carry zero semantics for a reader and are pure cost. So the
user prompt renders three derived views and nothing else:

1. **CONTROL FLOW** -- the node sequence with repeated cycles collapsed
   (``(sandbox -> call_model) x8``). A collapsed cycle is the single most
   diagnostic signal available: it distinguishes "looped without progress" from
   "answered in one turn", and it is invisible in a truncated raw dump.
2. **TOOL EXECUTIONS** -- name, arguments and a bounded result head per executed
   call, taken from ``tool_observations`` (authoritative: these are recorded at
   the wrapper, so they cannot be inflated by the model's narration).
3. **CAPTURE LIMITS** -- which capabilities were unavailable, so the model does
   not read a capture gap as agent behaviour.

Two further corpus facts shape the prompt: ``input_text`` is empty in 0 of 267
real traces (so a blank is labelled ``not recorded`` rather than sent as an
empty line), and ``tool_observations`` is empty in the majority (so "no tool ever
executed" is stated as a positive fact rather than left as an absent section).

Failure is data
---------------
A transport error, an unparseable body, or a rejected temperature all return a
``TrajectorySummary`` with ``observed=False`` and a populated ``error``. Nothing
fabricates a summary, and nothing unobserved is ever cached.
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Callable

from agent_evolve.core.rho.cache import JsonDiskCache
from agent_evolve.core.rho.history import HistoricalRecord

COMPREHENDER_MODEL_ID = "cuga-rho-comprehender"

#: The closed outcome vocabulary. Published so the difficulty judge and the
#: coreset stage agree with this module instead of re-deriving the strings.
OUTCOME_VALUES: tuple[str, ...] = (
    "correct_answer",
    "wrong_answer",
    "no_committed_answer",
    "error",
)

#: How much of one tool result to show. Enough to see *what came back* (an HTTP
#: error page, an empty result set, a wall of prose) without pasting a document.
_RESULT_HEAD_CHARS = 400
_ARGUMENT_HEAD_CHARS = 240

#: Longest node cycle worth naming. Beyond this a "cycle" is really the whole
#: run, and reporting it as a repeat says nothing.
_MAX_CYCLE_WIDTH = 4

#: Payload keys that are content-addressed pointers or run correlation ids.
#: Every one of them is identifier noise; none is ever rendered.
_OPAQUE_PAYLOAD_KEYS = frozenset(
    {
        "run_id",
        "parent_run_id",
        "state_before_ref",
        "state_after_ref",
        "messages_ref",
        "response_ref",
        "content_digest",
    }
)

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
_LONG_HEX_RE = re.compile(r"\b[0-9a-f]{16,}\b", re.I)
#: A JSON key fragment (``"some_key":``) leaking into prose.
_JSON_KEY_RE = re.compile(r'"\s*[A-Za-z0-9_]+\s*"\s*:')

_SYSTEM_PROMPT = """\
You are reading one agent trajectory and writing a short behavioural summary of
it. Many such summaries are compared against each other afterwards, so yours
must describe what makes THIS run different from a run that went differently.

Return ONLY a JSON object with exactly these keys:

  what_was_attempted   One sentence naming the goal the agent pursued, in the
                       task's own domain terms (what quantity, about what).
  approach_taken       One or two sentences describing the strategy actually
                       observed: what it tried first, whether it acted or only
                       described acting, and whether it changed strategy.
  where_it_went_wrong  One or two sentences naming the FIRST decisive failure
                       and the mechanism behind it. Use "" only if the run
                       genuinely succeeded.
  tools_used           Array of tool names that ACTUALLY EXECUTED. Empty array
                       if none did.
  outcome              Exactly one of: correct_answer, wrong_answer,
                       no_committed_answer, error

HOW TO JUDGE THE EVIDENCE

- TOOL EXECUTIONS is authoritative. It is recorded at the tool boundary, so it
  reflects what ran. The agent's own narration is not evidence that anything ran.
  If TOOL EXECUTIONS says no tool ever executed, then no tool ever executed --
  no matter how confidently the final output describes searching or computing.
- A collapsed cycle such as "(sandbox -> call_model) x8" means the agent looped.
  Say whether the loop made progress (different arguments, narrowing) or spun
  (equivalent calls repeated).
- A run that reaches its final answer with no tool execution at all did not fail
  at a tool and did not lack a capability: it failed to emit an action. Name that
  mechanism, because it is the most common and the most fixable.
- Distinguish "the tool returned nothing useful" from "the tool result was
  returned but ignored" from "the right tool was never selected". These demand
  different fixes, so they must read as different sentences.
- CAPTURE LIMITS lists what the tracer could not record. Never interpret a
  capture gap as agent behaviour, and never claim a failure you cannot see.
- If the evidence does not show a failure mechanism, say plainly in
  where_it_went_wrong what is missing. Do not invent a cause.

HOW TO WRITE IT

- Write prose about BEHAVIOUR, in the vocabulary of the task and the actions.
- Never write a UUID, a hex hash, an event id, a run id, a state or response
  reference, a JSON key name, or a raw JSON fragment. They carry no meaning here
  and they actively poison downstream comparison between summaries.
- Do not quote the expected or correct answer to the task.
- Be specific over hedged: "issued the same web search three times with only
  quoting changed" beats "had difficulty searching effectively".
- Two runs that failed for different reasons must produce two clearly different
  where_it_went_wrong sentences. Two runs that failed the same way should read
  alike.
"""


# --------------------------------------------------------------------------- #
# Result type
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class TrajectorySummary:
    """A bounded, identifier-free semantic summary of one trajectory."""

    task_id: str
    what_was_attempted: str = ""
    approach_taken: str = ""
    where_it_went_wrong: str = ""
    tools_used: tuple[str, ...] = ()
    outcome: str = ""
    observed: bool = False
    error: str = ""

    @property
    def embedding_text(self) -> str:
        """The text that gets embedded for DPP diversity.

        Deliberately excludes ``task_id``: an id is exactly the kind of
        surface token that would make two structurally different failures look
        different for the wrong reason.
        """
        parts = [
            self.what_was_attempted,
            self.approach_taken,
            self.where_it_went_wrong,
            " ".join(self.tools_used),
            self.outcome,
        ]
        return "\n".join(p for p in parts if p)


# --------------------------------------------------------------------------- #
# Scrubbing
# --------------------------------------------------------------------------- #
def _scrub(text: str) -> str:
    """Remove identifier tokens a model echoed back despite being told not to.

    The embedding invariant is the whole point of this stage, so it is enforced
    on the output rather than merely requested in the prompt.
    """
    cleaned = _UUID_RE.sub("", text)
    cleaned = _LONG_HEX_RE.sub("", cleaned)
    cleaned = _JSON_KEY_RE.sub("", cleaned)
    return " ".join(cleaned.split())


# --------------------------------------------------------------------------- #
# Trace rendering: raw events -> behaviour
# --------------------------------------------------------------------------- #
def _node_sequence(events: Sequence[object]) -> list[str]:
    """Node names in execution order, from ``graph_node_start`` events."""
    nodes: list[str] = []
    for event in events:
        if not isinstance(event, Mapping) or event.get("kind") != "graph_node_start":
            continue
        payload = event.get("payload")
        node = payload.get("node") if isinstance(payload, Mapping) else None
        if node:
            nodes.append(str(node))
    return nodes


def _collapse(nodes: Sequence[str]) -> str:
    """Render a node sequence with adjacent repeats and cycles collapsed.

    ``a a b c b c b c`` becomes ``a -> b -> c`` with the repeat marked as
    ``(b -> c) x3``. Real traces alternate ``call_model``/``sandbox`` up to eight
    times; that repetition count is the diagnostic signal, and spelling every
    repetition out would crowd out the tool evidence.
    """
    # CUGA emits a node twice (subgraph + node); drop adjacent duplicates first.
    deduped: list[str] = []
    for node in nodes:
        if not deduped or deduped[-1] != node:
            deduped.append(node)

    parts: list[str] = []
    index = 0
    total = len(deduped)
    while index < total:
        matched = False
        # Shortest cycle first, deliberately. The real corpus alternates
        # call_model/sandbox up to eight times; a longest-first search would
        # report that as one eight-node cycle repeated twice and destroy the
        # repetition count, which is the single most diagnostic number here.
        for width in range(1, min(_MAX_CYCLE_WIDTH, (total - index) // 2) + 1):
            window = deduped[index : index + width]
            repeats = 1
            probe = index + width
            while (
                probe + width <= total and deduped[probe : probe + width] == window
            ):
                repeats += 1
                probe += width
            if repeats > 1:
                if width == 1:
                    parts.append(f"{window[0]} x{repeats}")
                else:
                    parts.append("(" + " -> ".join(window) + f") x{repeats}")
                index = probe
                matched = True
                break
        if not matched:
            parts.append(deduped[index])
            index += 1
    return " -> ".join(parts)


def _tool_rows(trace: Mapping[str, object], limit: int) -> tuple[list[str], int]:
    """Bounded prose rows for executed tool calls, plus a not-shown count.

    ``tool_observations`` is preferred because it is recorded at the tool
    boundary. ``tool_call`` events are the fallback for traces captured before
    observations existed.
    """
    rows: list[str] = []
    observations = trace.get("tool_observations")
    source: list[Mapping[str, object]] = []
    if isinstance(observations, list):
        source = [o for o in observations if isinstance(o, Mapping)]

    if not source:
        events = trace.get("events")
        if isinstance(events, list):
            for event in events:
                if not isinstance(event, Mapping) or event.get("kind") != "tool_call":
                    continue
                payload = event.get("payload")
                call = payload.get("tool_call") if isinstance(payload, Mapping) else None
                if isinstance(call, Mapping):
                    source.append(
                        {
                            "tool_name": call.get("name") or call.get("app_name"),
                            "canonical_arguments": json.dumps(
                                call.get("arguments"), default=str, sort_keys=True
                            ),
                            "result": call.get("result"),
                            "error": call.get("error"),
                        }
                    )

    withheld = max(0, len(source) - limit)
    for index, obs in enumerate(source[:limit], start=1):
        name = str(obs.get("tool_name") or "unnamed-tool")
        args = str(obs.get("canonical_arguments") or "")[:_ARGUMENT_HEAD_CHARS]
        error = obs.get("error")
        if error:
            body = f"raised {str(error)[:_RESULT_HEAD_CHARS]}"
        else:
            result = str(obs.get("result") or "")
            if not result:
                body = "returned an empty result"
            else:
                head = " ".join(result[:_RESULT_HEAD_CHARS].split())
                suffix = " [...]" if len(result) > _RESULT_HEAD_CHARS else ""
                body = f"returned {head}{suffix}"
        if obs.get("truncated"):
            body += " (result truncated at capture)"
        rows.append(f"{index}. {name} called with {args}\n   -> {body}")
    return rows, withheld


def _capture_limits(trace: Mapping[str, object]) -> list[str]:
    """Capabilities the tracer could not capture, so absence is not overread."""
    limits: list[str] = []
    capabilities = trace.get("capabilities")
    if isinstance(capabilities, Mapping):
        for name, value in capabilities.items():
            status = value.get("status") if isinstance(value, Mapping) else value
            if status and str(status) != "captured":
                reason = (
                    value.get("reason") if isinstance(value, Mapping) else None
                ) or "no reason recorded"
                limits.append(f"- {name}: {status} ({reason})")
    if trace.get("events_truncated"):
        dropped = trace.get("dropped_event_count") or "unknown"
        limits.append(f"- event stream truncated; {dropped} events dropped")
    return limits


# --------------------------------------------------------------------------- #
# Transport
# --------------------------------------------------------------------------- #
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


def _litellm_completion(**request: object) -> object:
    """Live model call. Imported lazily so unit tests stay offline."""
    import litellm

    return litellm.completion(**request)


def _response_text(response: object) -> str:
    """First assistant text from an OpenAI/litellm-shaped response."""
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, Mapping):
        choices = response.get("choices")
    if not choices:
        return ""
    first = choices[0]
    message = getattr(first, "message", None)
    if message is None and isinstance(first, Mapping):
        message = first.get("message")
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if content is None and isinstance(message, Mapping):
        content = message.get("content")
    return str(content or "")


def _strip_fence(text: str) -> str:
    """Drop a surrounding ```json fence; models add one despite instructions."""
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    body = stripped[3:]
    if body.lower().startswith("json"):
        body = body[4:]
    return body.rsplit("```", 1)[0].strip() if "```" in body else body.strip()


# --------------------------------------------------------------------------- #
# Comprehender
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class RhoComprehender:
    """Interface A trajectory comprehender.

    Parameters
    ----------
    completion_fn:
        Called as ``completion_fn(**request)`` and expected to return an
        OpenAI/litellm-shaped response. Defaults to a live ``litellm.completion``
        call. This is the single swap point that keeps every test offline.
    cache:
        Keyed by trace content hash, so a re-run over an unchanged corpus costs
        nothing and a changed trace can never produce a false hit. Defaults to a
        disabled cache.
    max_events:
        Upper bound on rendered tool-execution rows.
    """

    completion_fn: Callable[..., object] | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    #: Omitted unless set. NEVER pass 0.0: the endpoint rejects it outright with
    #: "Unsupported value: 'temperature' does not support 0.0".
    temperature: float | None = None
    cache: JsonDiskCache = field(default_factory=lambda: JsonDiskCache(None))
    max_events: int = 40

    # -- prompt ----------------------------------------------------------- #
    def _user_prompt(self, record: HistoricalRecord) -> str:
        trace = record.raw_trace
        events = trace.get("events")
        event_list = events if isinstance(events, list) else []

        task_text = record.input_text.strip() or (
            "not recorded in this trace; infer the goal from the trajectory"
        )
        final_output = record.final_output.strip() or "(empty final output)"

        flow = _collapse(_node_sequence(event_list)) or "no node transitions recorded"
        rows, withheld = _tool_rows(trace, self.max_events)
        if rows:
            tools_block = "\n".join(rows)
            if withheld:
                tools_block += f"\n({withheld} further tool executions not shown)"
        else:
            tools_block = (
                "no tool ever executed in this run: zero tool executions were "
                "recorded from start to final answer"
            )

        limits = _capture_limits(trace)
        limits_block = "\n".join(limits) if limits else "- none; capture was complete"

        status = str(trace.get("status") or "unknown")
        error = trace.get("error")
        run_status = f"{status}" + (f" (harness error: {error})" if error else "")

        return (
            f"TASK GIVEN TO THE AGENT:\n{task_text}\n\n"
            f"CONTROL FLOW (node sequence, repeated cycles collapsed):\n{flow}\n\n"
            f"TOOL EXECUTIONS (authoritative; recorded at the tool boundary, "
            f"{record.tool_observation_count} total):\n{tools_block}\n\n"
            f"FINAL OUTPUT THE AGENT COMMITTED TO:\n{final_output}\n\n"
            f"HARNESS RUN STATUS: {run_status}\n"
            f"HARNESS VERSION: {record.harness_version}\n\n"
            f"CAPTURE LIMITS (what the tracer could not record; never read a "
            f"gap here as agent behaviour):\n{limits_block}\n\n"
            "Now return the JSON object described in the instructions."
        )

    def _resolved_settings(self) -> tuple[str, str | None, str | None]:
        """``(model, base_url, api_key)``, falling back to the environment.

        Resolved lazily and per call, matching ``CugaTrajectoryAnalyzer``: a
        caller that injects ``completion_fn`` must never be forced to hold
        credentials, but a live run must not silently send ``model="unset"``
        either -- that reaches the endpoint as a ``BadRequestError`` whose only
        visible symptom upstream is ``N trajectory summaries unavailable``,
        which reads as a data problem rather than a configuration one.
        """
        model, base_url, api_key = self.model, self.base_url, self.api_key
        if model is None or base_url is None or api_key is None:
            env_model, env_base, env_key = _env_settings()
            model = model or env_model
            base_url = base_url or env_base
            api_key = api_key or env_key
        if not model:
            raise ValueError(
                "no comprehender model configured: set CUGA_MODEL or "
                "LITELLM_MODEL, or pass model= explicitly"
            )
        return model, base_url, api_key

    def _request(self, record: HistoricalRecord) -> dict[str, object]:
        model, base_url, api_key = self._resolved_settings()
        request: dict[str, object] = {"model": model}
        if self.temperature is not None:
            if self.temperature == 0.0:
                raise ValueError(
                    "temperature=0.0 is rejected by the endpoint; omit it instead"
                )
            request["temperature"] = self.temperature
        if base_url:
            request["api_base"] = base_url
        if api_key:
            request["api_key"] = api_key
        request["messages"] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": self._user_prompt(record)},
        ]
        return request

    def _cache_key(self, record: HistoricalRecord) -> str:
        return (
            f"{COMPREHENDER_MODEL_ID}|{self.model}|{record.task_id}|"
            f"{record.content_hash}"
        )

    # -- main entry point -------------------------------------------------- #
    def comprehend(self, record: HistoricalRecord) -> TrajectorySummary:
        """Summarize one record, returning an unobserved summary on failure."""
        key = self._cache_key(record)
        cached = self.cache.get(key)
        if cached is not None:
            return self._summary_from(record.task_id, cached)

        try:
            request = self._request(record)
        except ValueError as exc:
            # A misconfigured temperature must surface, not reach the endpoint.
            return TrajectorySummary(task_id=record.task_id, error=str(exc))

        invoke = self.completion_fn or _litellm_completion
        try:
            response = invoke(**request)
        except Exception as exc:  # noqa: BLE001 - a failure is data
            return TrajectorySummary(
                task_id=record.task_id,
                error=f"{type(exc).__name__}: {exc}",
            )

        text = _strip_fence(_response_text(response))
        if not text:
            return TrajectorySummary(
                task_id=record.task_id,
                error="empty comprehension response body",
            )
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise ValueError("response JSON is not an object")
        except (json.JSONDecodeError, ValueError) as exc:
            return TrajectorySummary(
                task_id=record.task_id,
                error=f"unparseable comprehension response: {exc}",
            )

        summary = self._summary_from(record.task_id, payload)
        # Only observed summaries are cached; a failure must be retried.
        self.cache.put(
            key,
            {
                "what_was_attempted": summary.what_was_attempted,
                "approach_taken": summary.approach_taken,
                "where_it_went_wrong": summary.where_it_went_wrong,
                "tools_used": list(summary.tools_used),
                "outcome": summary.outcome,
            },
        )
        return summary

    @staticmethod
    def _summary_from(task_id: str, payload: Mapping[str, object]) -> TrajectorySummary:
        tools = payload.get("tools_used")
        return TrajectorySummary(
            task_id=task_id,
            what_was_attempted=_scrub(str(payload.get("what_was_attempted", ""))),
            approach_taken=_scrub(str(payload.get("approach_taken", ""))),
            where_it_went_wrong=_scrub(str(payload.get("where_it_went_wrong", ""))),
            tools_used=(
                tuple(_scrub(str(t)) for t in tools) if isinstance(tools, list) else ()
            ),
            outcome=_scrub(str(payload.get("outcome", ""))),
            observed=True,
        )
