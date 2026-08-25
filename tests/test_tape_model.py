"""Phase 4 — TapeModel seam prototype (?15 / R9).

The candidate seam: CUGA's ``LLMManager.set_llm`` accepts any
``BaseChatModel`` and ``get_model`` returns it for EVERY agent before any
platform-specific construction, so a tape-backed chat model injected there
serves recorded responses uniformly without patching internals or speaking
HTTP. Decision rule (plan Phase 4): this seam settles only once the R5
hash-chain oracle passes on the real reference trace (Phase 5); these tests
prove the mechanics offline against production-shaped serializations.

Fixtures build their blobs THROUGH the real capture-path serializer
(``_json_safe`` + canonical dumps) over real LangChain objects, mirroring the
two shapes observed in reference trace 3306905e:

* ``messages_ref``  -> our projection: ``[[{...msg, "__type__": ...}]]``
* ``response_ref``  -> native ``LLMResult`` model_dump tree: ``[[gen]]``

Fidelity is asserted where it matters for R5: the reconstructed *message*
must project identically to the recorded generation's message, because that
is the object that enters graph state.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from agent_evolve.cuga_wrapper.tape_replay import (
    TapeCallSequenceExhausted,
    TapeModel,
    TapeDivergence,
    load_tape_model,
)


def _canonical(value: object) -> str:
    from agent_evolve import cuga_wrapper as pkg

    return json.dumps(pkg._json_safe(value), sort_keys=True, ensure_ascii=False)


def _strip_type_marker(projected: object) -> dict:
    assert isinstance(projected, dict)
    return {key: value for key, value in projected.items() if key != "__type__"}


def _fixture_answer(answer: str) -> AIMessage:
    return AIMessage(
        content=answer,
        id=f"run-{answer}",
        additional_kwargs={"model_provider": "fixture"},
        response_metadata={"model_name": "fixture-model"},
    )


@pytest.fixture()
def two_call_trace(tmp_path: Path) -> Path:
    """Synthetic trace dir whose blobs pass through the REAL serializer."""
    payloads = tmp_path / "payloads"
    payloads.mkdir()

    def store(value: object) -> str:
        raw = _canonical(value).encode("utf-8")
        ref = hashlib.sha256(raw).hexdigest()
        (payloads / f"{ref}.json").write_bytes(raw)
        return ref

    calls = [
        (
            [HumanMessage(content="prompt-one")],
            LLMResult(
                generations=[[ChatGeneration(
                    message=_fixture_answer("answer-one"), text="answer-one")]],
                llm_output={"token_usage": {"total_tokens": 10},
                            "model_name": "fixture-model"},
            ),
        ),
        (
            [HumanMessage(content="prompt-two")],
            LLMResult(
                generations=[[ChatGeneration(
                    message=_fixture_answer("answer-two"), text="answer-two")]],
                llm_output={"token_usage": {"total_tokens": 12},
                            "model_name": "fixture-model"},
            ),
        ),
    ]

    lines = []
    for seq, (messages, result) in enumerate(calls):
        start_seq = seq * 2
        end_seq = seq * 2 + 1
        lines.append(json.dumps({
            "actor_id": None, "event_id": f"graph:{start_seq}",
            "kind": "llm_call_start", "parent_event_id": None,
            "payload": {"run_id": f"llm-{seq}", "parent_run_id": None,
                        "messages_ref": store([[m for m in messages]])},
            "sequence": start_seq, "timestamp": "2026-08-25T14:10:45Z",
        }))
        lines.append(json.dumps({
            "actor_id": None, "event_id": f"graph:{end_seq}",
            "kind": "llm_call_end", "parent_event_id": None,
            "payload": {"run_id": f"llm-{seq}", "parent_run_id": None,
                        "response_ref": store(result)},
            "sequence": end_seq, "timestamp": "2026-08-25T14:10:46Z",
        }))
    (tmp_path / "events.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return tmp_path


def _invoke(model: TapeModel, content: str):
    return model.generate([[HumanMessage(content=content)]])


class TestRoundtripFidelity:
    def test_reconstructed_fields_preserve_recording_verbatim(
            self, two_call_trace: Path) -> None:
        """Every recorded field survives reconstruction (field-level roundtrip).

        Whole-envelope equality through ``generate()`` is NOT the contract:
        LangChain's packaging deterministically enriches outbound messages
        (tool_calls/usage_metadata/token_usage merge) on BOTH the original run
        and the replay, so it cancels; what must hold is that reconstruction
        loses or alters nothing the recording carried.
        """
        model = load_tape_model(two_call_trace)
        message = model.reconstruct(0).generations[0].message
        projected = _strip_type_marker(_json_safe_of(message))
        recorded_raw = json.loads(
            (two_call_trace / "payloads" / f"{model.entries[0].response_ref}.json")
            .read_bytes().decode("utf-8"))
        recorded_message = recorded_raw["generations"][0][0]["message"]
        for key, value in recorded_message.items():
            if key == "type":
                continue  # discriminator; class identity asserted by isinstance below
            assert key in projected, f"reconstruction dropped field {key!r}"
            assert projected[key] == value, f"field {key!r} altered"
        assert isinstance(message, AIMessage)

    def test_reconstruct_carries_recorded_envelope(self, two_call_trace: Path) -> None:
        model = load_tape_model(two_call_trace)
        result = model.reconstruct(0)
        assert result.llm_output["token_usage"]["total_tokens"] == 10

    def test_second_call_serves_its_own_recording(self, two_call_trace: Path) -> None:
        model = load_tape_model(two_call_trace)
        _invoke(model, "prompt-one")
        result = _invoke(model, "prompt-two")
        assert result.generations[0][0].message.content == "answer-two"


class TestPointerDiscipline:
    def test_exhaustion_raises_loudly_naming_position(self, two_call_trace: Path) -> None:
        model = load_tape_model(two_call_trace)
        _invoke(model, "prompt-one")
        _invoke(model, "prompt-two")
        with pytest.raises(TapeCallSequenceExhausted) as excinfo:
            _invoke(model, "anything")
        assert "consumed 2" in str(excinfo.value)

    def test_unexpected_extra_call_reports_expected_total(self, two_call_trace: Path) -> None:
        model = load_tape_model(two_call_trace, expect_calls=2)
        _invoke(model, "prompt-one")
        _invoke(model, "prompt-two")
        with pytest.raises(TapeCallSequenceExhausted) as excinfo:
            _invoke(model, "surprise")
        assert "expected 2" in str(excinfo.value)


class TestSymmetryGate:
    def test_mismatching_prompt_raises_naming_sequence(self, two_call_trace: Path) -> None:
        model = load_tape_model(two_call_trace)
        with pytest.raises(TapeDivergence) as excinfo:
            _invoke(model, "a completely different prompt")
        assert "sequence 0" in str(excinfo.value)

    def test_gate_fires_on_later_call_with_earlier_matching(
            self, two_call_trace: Path) -> None:
        model = load_tape_model(two_call_trace)
        _invoke(model, "prompt-one")
        with pytest.raises(TapeDivergence) as excinfo:
            _invoke(model, "prompt-one")  # repeat instead of advancing
        assert "sequence 1" in str(excinfo.value)


class TestVolatilityScrubbing:
    """R5 amendment: wall-clock/task-id drift must not masquerade as divergence."""

    def test_scrubbed_gate_passes_where_raw_fails(self) -> None:
        import re

        from agent_evolve.cuga_wrapper.tape_replay import TapeEntry, TapeState

        state = TapeState(
            entries=[TapeEntry(0, "s0", "ref-a", "ref-b")],
            messages_canonical=[json.dumps(
                [{"content": "ran at 2026-08-25 19:45:00"}])],
            scrub_patterns=(re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?"),),
        )
        replayed = json.dumps([{"content": "ran at 2026-08-26 09:01:02"}])
        assert replayed != state.messages_canonical[0]  # raw comparison would fail
        assert state.compare(replayed, 0) is True

    def test_scrubbing_does_not_absorb_real_differences(self) -> None:
        import re

        from agent_evolve.cuga_wrapper.tape_replay import TapeEntry, TapeState

        state = TapeState(
            entries=[TapeEntry(0, "s0", "ref-a", "ref-b")],
            messages_canonical=[json.dumps(
                [{"content": "total is 105000 USD"}])],
            scrub_patterns=(re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}(:\d{2})?"),),
        )
        replayed = json.dumps([{"content": "total is 120000 USD"}])
        assert state.compare(replayed, 0) is False


class TestTraceLoading:
    def test_load_pairs_boundaries_in_order(self, two_call_trace: Path) -> None:
        model = load_tape_model(two_call_trace)
        assert [entry.sequence for entry in model.entries] == [0, 2]
        assert len(model.entries) == 2  # paired start/end collapse to one entry

    def test_entries_carry_both_refs(self, two_call_trace: Path) -> None:
        model = load_tape_model(two_call_trace)
        entry = model.entries[0]
        assert entry.messages_ref and entry.response_ref

    def test_reference_trace_integration(self) -> None:
        reference = Path(
            "terminal_output/live-run-prep/traces/"
            "3306905e-668f-41a3-adb0-e7a0ba33e332")
        if not reference.exists():  # gitignored; skip on machines without it
            pytest.skip("reference trace not present")
        model = load_tape_model(reference)
        assert len(model.entries) == 4  # four LLM boundaries in the real trace


def _json_safe_of(value: object) -> object:
    from agent_evolve import cuga_wrapper as pkg

    return pkg._json_safe(value)
