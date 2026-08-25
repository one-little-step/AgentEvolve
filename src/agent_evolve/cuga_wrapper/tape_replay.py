"""TapeChatModel — serves recorded LLM responses back as a chat model (?15).

Seam (R9 candidate, refined by evidence): CUGA's ``LLMManager`` singleton
exposes the documented ``set_llm(model: BaseChatModel)`` override and
``get_model`` returns that pre-instantiated model for EVERY agent before any
platform-specific construction (``cuga/backend/llm/models.py:1383``). A
tape-backed model injected there therefore covers planner, code agents,
final-answer and every other CUGA-internal caller uniformly — no HTTP server,
no client patching, no wire-format translation.

Two production shapes drive the mechanics (reference trace ``3306905e``):

* ``messages_ref`` blobs are OUR capture projection: ``_json_safe`` over the
  message batches, so live incoming messages projected through the same
  function must compare byte-equal — this is the divergence gate. It fires
  loudly at the first misaligned call instead of feeding wrong context
  forward.
* ``response_ref`` blobs are native ``LLMResult`` pydantic dumps:
  ``generations`` is a list of per-prompt lists. Reconstruction rebuilds the
  inner messages (the objects that enter graph state) and adapts to the flat
  ``ChatResult`` that ``BaseChatModel._generate`` must return.

Fidelity contract: the reconstructed message re-projected through
``_json_safe`` equals the recorded message dict (modulo our ``__type__``
marker, which native dumps never carry). Envelope extras such as ``run`` are
capture-context keys, not state content; they are not reconstructed.

This module is adapter-side by design: it imports LangChain output types and
our own capture serializer, both forbidden in ``core/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from agent_evolve.core.tape import TapeIndex

__all__ = [
    "TapeReplayError",
    "TapeCallSequenceExhausted",
    "TapeDivergence",
    "TapeEntry",
    "TapeState",
    "serve",
    "load_tape_model",
    "TapeModel",
]


class TapeReplayError(RuntimeError):
    """Base class for tape replay failures."""


class TapeCallSequenceExhausted(TapeReplayError):
    """A live call arrived after every recorded response was served."""


class TapeDivergence(TapeReplayError):
    """The incoming prompt does not match the recording at this position."""


@dataclass(frozen=True)
class TapeEntry:
    sequence: int
    run_id: str
    messages_ref: str
    response_ref: str | None


@dataclass
class TapeState:
    entries: list[TapeEntry]
    messages_canonical: list[str] = field(default_factory=list)
    responses: list[dict] = field(default_factory=list)
    pointer: int = 0
    expected_calls: int | None = None
    scrub_patterns: tuple[Any, ...] = ()
    gate_enabled: bool = True

    def compare(self, incoming_canonical: str, index: int) -> bool:
        """Gate comparison; scrubbed when a volatility registry is supplied.

        ``gate_enabled=False`` is MUTATION mode (RQ4): an edited artifact
        legitimately changes prompts from the first call, so prefix gates are
        opened deliberately and the arm measures behavioural effect rather
        than prefix fidelity.
        """
        if not self.gate_enabled:
            return True
        recorded = self.messages_canonical[index]
        if not self.scrub_patterns:
            return incoming_canonical == recorded
        for pattern in self.scrub_patterns:
            incoming_canonical = pattern.sub("<volatile>", incoming_canonical)
            recorded = pattern.sub("<volatile>", recorded)
        return incoming_canonical == recorded


def _canonical(value: object) -> str:
    from agent_evolve import cuga_wrapper as pkg

    return json.dumps(pkg._json_safe(value), sort_keys=True, ensure_ascii=False)


def _reconstruct_message(recorded: dict) -> Any:
    """Rebuild an AIMessage from a native-dump dict (no ``__type__`` marker)."""
    from langchain_core.messages import AIMessage

    fields = {
        key: value
        for key, value in recorded.items()
        if key in ("content", "additional_kwargs", "response_metadata",
                   "tool_calls", "id", "name", "usage_metadata")
        and value is not None
    }
    return AIMessage(**fields)


def reconstruct_result(state: TapeState, index: int):
    """Rebuild the recorded ``LLMResult`` as a flat ``ChatResult``."""
    from langchain_core.outputs import ChatGeneration, ChatResult

    recorded = state.responses[index]
    generations_outer = recorded.get("generations")
    if not isinstance(generations_outer, list) or len(generations_outer) != 1:
        count = len(generations_outer) if isinstance(generations_outer, list) else -1
        raise TapeReplayError(
            f"sequence {index}: recording holds {count} prompt batch(es); "
            f"only single-batch calls are replayable")
    generations: list[ChatGeneration] = []
    for gen in generations_outer[0]:
        if not isinstance(gen, dict):
            raise TapeReplayError(
                f"sequence {index}: malformed generation entry")
        message_dict = gen.get("message")
        if not isinstance(message_dict, dict):
            raise TapeReplayError(
                f"sequence {index}: generation carries no message dict")
        text = gen.get("text")
        gen_kwargs: dict[str, Any] = {
            "message": _reconstruct_message(message_dict),
            "generation_info": gen.get("generation_info") or {},
        }
        if isinstance(text, str):
            gen_kwargs["text"] = text
        generations.append(ChatGeneration(**gen_kwargs))
    llm_output = recorded.get("llm_output")
    return ChatResult(
        generations=generations,
        llm_output=llm_output if isinstance(llm_output, dict) else {},
    )


def serve(state: TapeState, messages: Sequence[Any]) -> Any:
    """Serve the next recorded response, enforcing pointer + symmetry."""
    import os as _os

    if _os.environ.get("AE_TAPE_DEBUG"):
        print(f"[tape-debug] serve id={id(state):#x} pointer={state.pointer} "
              f"entries={len(state.entries)} gate={state.gate_enabled}",
              flush=True)
    if state.pointer >= len(state.entries):
        consumed = state.pointer
        suffix = "" if state.expected_calls is None else \
            f" (expected {state.expected_calls})"
        raise TapeCallSequenceExhausted(
            f"tape exhausted: consumed {consumed} recorded call(s){suffix}; "
            f"a live call arrived that has no recording")
    index = state.pointer
    entry = state.entries[index]

    incoming = _canonical([list(messages)])
    if not state.compare(incoming, index):
        raise TapeDivergence(
            f"sequence {index} ({entry.run_id}): incoming prompt does not "
            f"match the recorded messages_ref {entry.messages_ref[:12]}...")

    result = reconstruct_result(state, index)
    state.pointer += 1
    return result


def load_tape_model(trace_dir: Path | str,
                    expect_calls: int | None = None,
                    scrub_patterns: tuple[Any, ...] = ()) -> "TapeModel":
    """Load a trace directory into a ready-to-inject :class:`TapeModel`.

    ``scrub_patterns`` is the volatility registry (R5 amendment): compiled
    regexes applied identically to both gate sides so wall-clock/task-id
    drift between the original run and the replay cannot masquerade as
    behavioural divergence. Raw comparison is used when empty.
    """
    directory = Path(trace_dir)
    index = TapeIndex.load(directory)
    entries: list[TapeEntry] = []
    messages_canonical: list[str] = []
    responses: list[dict] = []
    for boundary in index.llm_boundaries:
        raw_messages = index.resolve(boundary.messages_ref)
        messages_canonical.append(raw_messages.decode("utf-8"))
        response_blob: dict = {}
        if boundary.response_ref:
            raw_response = json.loads(
                index.resolve(boundary.response_ref).decode("utf-8"))
            if isinstance(raw_response, dict):
                response_blob = raw_response
        responses.append(response_blob)
        entries.append(TapeEntry(
            sequence=boundary.sequence,
            run_id=f"seq-{boundary.sequence}",
            messages_ref=boundary.messages_ref,
            response_ref=boundary.response_ref,
        ))
    return TapeModel(tape_state=TapeState(
        entries=entries,
        messages_canonical=messages_canonical,
        responses=responses,
        expected_calls=expect_calls,
        scrub_patterns=tuple(scrub_patterns),
    ))


try:  # LangChain availability guard keeps helpers importable everywhere.
    from langchain_core.callbacks import CallbackManagerForLLMRun
    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from pydantic import PrivateAttr

    class TapeModel(BaseChatModel):
        """A ``BaseChatModel`` that replays a recorded call sequence.

        Injection point: ``LLMManager.set_llm(tape_model)`` before graph
        construction makes every CUGA agent draw from the tape (models.py
        checks the pre-instantiated model before any platform path).

        Parameters ``temperature`` / ``max_tokens`` / ``max_completion_tokens``
        / ``model_kwargs`` / ``model_name`` exist so CUGA's
        ``_update_model_parameters`` can mutate them harmlessly.
        """

        temperature: float = 0.0
        max_tokens: int | None = None
        max_completion_tokens: int | None = None
        model_kwargs: dict[str, Any] = {}
        model_name: str = "tape-replay"

        _tape_state: TapeState = PrivateAttr()

        def __init__(self, tape_state: TapeState | None = None, **kwargs: Any) -> None:
            super().__init__(**kwargs)
            if tape_state is not None:
                self._tape_state = tape_state

        @property
        def entries(self) -> list[TapeEntry]:
            return list(self._tape_state.entries)

        @property
        def pointer(self) -> int:
            return self._tape_state.pointer

        @property
        def recorded_responses(self) -> list[dict]:
            return list(self._tape_state.responses)

        def reconstruct(self, index: int) -> ChatResult:
            """Rebuild the recorded result at ``index`` without consuming tape."""
            return reconstruct_result(self._tape_state, index)

        @property
        def _llm_type(self) -> str:
            return "tape-replay"

        def bind_tools(self, *args: Any, **kwargs: Any) -> "TapeModel":
            # Tool schemas shape provider requests; under replay the response
            # is already fixed, so binding is a no-op returning self.
            return self

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Sequence[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            return serve(self._tape_state, messages)

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: Sequence[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            # Offline replay has no IO; direct sync execution is exact.
            return serve(self._tape_state, messages)

except ImportError:  # pragma: no cover - langchain always present with cuga
    TapeModel = None  # type: ignore[assignment,misc]

    class HybridTapeModel:  # pragma: no cover - placeholder, never constructed
        pass

else:

    class HybridTapeModel(BaseChatModel):
        """LIVE-TAIL mode (R4): taped prefix, live tail.

        Serves recorded responses up to ``cutoff`` boundaries; the first call
        past the cutoff lazily constructs the live model via the
        caller-supplied ``live_factory`` (the driver wires it to
        ``LLMManager.get_model(settings.agent.code.model)`` after clearing the
        pre-instantiated override) and forwards everything later to it.
        Bound tool schemas are stored and re-bound onto the live branch —
        dropping them would silently break tool-calling in the tail.
        """

        cutoff: int = 0
        temperature: float = 0.0
        max_tokens: int | None = None
        max_completion_tokens: int | None = None
        model_kwargs: dict[str, Any] = {}
        model_name: str = "hybrid-tape-live"

        _tape_state: TapeState = PrivateAttr()
        _live_factory: Any = PrivateAttr()
        _live_model: Any = PrivateAttr(default=None)
        _bound_tools: Any = PrivateAttr(default=None)

        def __init__(
            self,
            tape_state: TapeState | None = None,
            live_factory: Any = None,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            if tape_state is not None:
                self._tape_state = tape_state
            if live_factory is not None:
                self._live_factory = live_factory

        @classmethod
        def from_trace(
            cls,
            trace_dir: Path | str,
            cutoff: int,
            scrub_patterns: tuple[Any, ...] = (),
            live_factory: Any = None,
            gate_enabled: bool = True,
        ) -> "HybridTapeModel":
            base = load_tape_model(
                trace_dir, expect_calls=cutoff, scrub_patterns=scrub_patterns)
            base._tape_state.gate_enabled = gate_enabled
            return cls(tape_state=base._tape_state, live_factory=live_factory,
                       cutoff=cutoff)

        @property
        def entries(self) -> list[TapeEntry]:
            return list(self._tape_state.entries)

        @property
        def pointer(self) -> int:
            return self._tape_state.pointer

        @property
        def live_calls(self) -> int:
            return max(0, self._tape_state.pointer - self.cutoff)

        @property
        def _llm_type(self) -> str:
            return "hybrid-tape-live"

        def bind_tools(self, *args: Any, **kwargs: Any) -> "HybridTapeModel":
            self._bound_tools = args[0] if args else None
            if kwargs:
                self._bound_tools = kwargs
            return self

        def _live_branch(self, messages: list[BaseMessage]) -> ChatResult:
            import os as _os

            if _os.environ.get("AE_TAPE_DEBUG"):
                print(f"[tape-debug] LIVE id={id(self._tape_state):#x} "
                      f"pointer_before={self._tape_state.pointer}", flush=True)
            if getattr(self, "_live_model", None) is None:
                self._live_model = self._live_factory()
            target = self._live_model
            bound = getattr(self, "_bound_tools", None)
            runnable = target.bind_tools(bound) if bound is not None else target
            message = runnable.invoke(messages)
            self._tape_state.pointer += 1  # live calls count too (?16 fix)
            return ChatResult(generations=[ChatGeneration(message=message)])

        def _generate(
            self,
            messages: list[BaseMessage],
            stop: Sequence[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            state = self._tape_state
            if state.pointer < self.cutoff:
                return serve(state, messages)
            return self._live_branch(messages)

        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: Sequence[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> ChatResult:
            state = self._tape_state
            if state.pointer < self.cutoff:
                return serve(state, messages)
            return self._live_branch(messages)
