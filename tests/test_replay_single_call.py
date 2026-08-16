"""Tests for single-LLM-call replay from a recorded CUGA causal trace.

These tests never touch the network: the live completion function is injected.
They also never depend on ``data/traces/`` (which is gitignored); every fixture
trace directory is synthesized in ``tmp_path`` using the payload layout observed
in the real reference trace (``payloads/<sha256>.json``).
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import pytest

from agent_evolve.cuga_wrapper import (
    RecordedCall,
    list_recorded_llm_calls,
    load_recorded_call,
    replay_single_llm_call,
)


def _write_blob(trace_dir: Path, value: object) -> str:
    """Persist ``value`` the way the wrapper's PayloadStore does and return its digest."""
    payloads = trace_dir / "payloads"
    payloads.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    (payloads / f"{digest}.json").write_text(serialized, encoding="utf-8")
    return digest


def _langchain_message(kind: str, content: str) -> dict[str, object]:
    """Mirror the recorded shape: ``__type__`` plus a lowercase ``type`` discriminator."""
    types = {"system": "SystemMessage", "human": "HumanMessage", "ai": "AIMessage"}
    return {
        "__type__": types[kind],
        "additional_kwargs": {},
        "content": content,
        "id": None,
        "name": None,
        "response_metadata": {},
        "type": kind,
    }


def _llm_result(text: str) -> dict[str, object]:
    return {
        "__type__": "LLMResult",
        "generations": [
            [
                {
                    "generation_info": {"finish_reason": "stop"},
                    "message": _langchain_message("ai", text),
                    "text": text,
                }
            ]
        ],
    }


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    """A minimal two-LLM-call trace directory in the recorded on-disk format."""
    directory = tmp_path / "run-1"
    directory.mkdir()

    # Recorded messages are a list of message *batches* (LangChain callback shape).
    first_messages = _write_blob(
        directory,
        [[_langchain_message("system", "SYS ONE"), _langchain_message("human", "HUMAN ONE")]],
    )
    second_messages = _write_blob(
        directory,
        [
            [
                _langchain_message("system", "SYS TWO"),
                _langchain_message("human", "HUMAN TWO"),
                _langchain_message("ai", "AI TWO"),
            ]
        ],
    )
    first_response = _write_blob(directory, _llm_result("BASELINE ONE"))

    trace = {
        "run_id": "run-1",
        "model": "openai/azure/gpt-5.6-luna",
        "events": [
            {
                "event_id": "graph:4",
                "kind": "llm_call_start",
                "sequence": 4,
                "payload": {"messages_ref": first_messages, "run_id": "r1"},
            },
            {
                "event_id": "graph:5",
                "kind": "llm_call_end",
                "sequence": 5,
                "payload": {"response_ref": first_response, "run_id": "r1"},
            },
            {
                "event_id": "graph:9",
                "kind": "graph_node_start",
                "sequence": 9,
                "payload": {"state_before_ref": "irrelevant"},
            },
            {
                # No paired llm_call_end: baseline response must be absent.
                "event_id": "graph:13",
                "kind": "llm_call_start",
                "sequence": 13,
                "payload": {"messages_ref": second_messages, "run_id": "r2"},
            },
        ],
    }
    (directory / "causal-trace.json").write_text(json.dumps(trace), encoding="utf-8")
    return directory


class TestListRecordedLlmCalls:
    def test_enumerates_llm_call_start_event_ids_in_sequence_order(self, trace_dir: Path) -> None:
        assert list_recorded_llm_calls(trace_dir) == ("graph:4", "graph:13")

    def test_ignores_non_llm_events(self, trace_dir: Path) -> None:
        assert "graph:9" not in list_recorded_llm_calls(trace_dir)

    def test_missing_trace_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            list_recorded_llm_calls(tmp_path / "nope")


class TestLoadRecordedCall:
    def test_resolves_messages_to_role_content_dicts(self, trace_dir: Path) -> None:
        call = load_recorded_call(trace_dir, "graph:4")

        assert call.messages == [
            {"role": "system", "content": "SYS ONE"},
            {"role": "user", "content": "HUMAN ONE"},
        ]

    def test_maps_ai_messages_to_assistant_role(self, trace_dir: Path) -> None:
        call = load_recorded_call(trace_dir, "graph:13")

        assert [message["role"] for message in call.messages] == ["system", "user", "assistant"]

    def test_carries_event_id_and_trace_model(self, trace_dir: Path) -> None:
        call = load_recorded_call(trace_dir, "graph:4")

        assert isinstance(call, RecordedCall)
        assert call.event_id == "graph:4"
        assert call.model == "openai/azure/gpt-5.6-luna"

    def test_resolves_baseline_response_text_from_paired_end_event(self, trace_dir: Path) -> None:
        assert load_recorded_call(trace_dir, "graph:4").baseline_response == "BASELINE ONE"

    def test_baseline_response_is_none_without_paired_end_event(self, trace_dir: Path) -> None:
        assert load_recorded_call(trace_dir, "graph:13").baseline_response is None

    def test_has_system_message_reports_system_prompt_presence(self, trace_dir: Path) -> None:
        assert load_recorded_call(trace_dir, "graph:4").has_system_message is True

    def test_unknown_event_id_raises_key_error(self, trace_dir: Path) -> None:
        with pytest.raises(KeyError, match="graph:999"):
            load_recorded_call(trace_dir, "graph:999")

    def test_non_llm_event_id_raises_key_error(self, trace_dir: Path) -> None:
        with pytest.raises(KeyError):
            load_recorded_call(trace_dir, "graph:9")

    def test_missing_payload_blob_raises_file_not_found(self, trace_dir: Path) -> None:
        for blob in (trace_dir / "payloads").glob("*.json"):
            blob.unlink()

        with pytest.raises(FileNotFoundError):
            load_recorded_call(trace_dir, "graph:4")

    def test_model_falls_back_to_manifest_when_trace_omits_it(self, trace_dir: Path) -> None:
        trace = json.loads((trace_dir / "causal-trace.json").read_text(encoding="utf-8"))
        del trace["model"]
        (trace_dir / "causal-trace.json").write_text(json.dumps(trace), encoding="utf-8")
        (trace_dir / "manifest.json").write_text(
            json.dumps({"model": "openai/from-manifest"}), encoding="utf-8"
        )

        assert load_recorded_call(trace_dir, "graph:4").model == "openai/from-manifest"

    def test_empty_recorded_messages_raise_value_error(self, tmp_path: Path) -> None:
        directory = tmp_path / "empty"
        directory.mkdir()
        ref = _write_blob(directory, [[]])
        (directory / "causal-trace.json").write_text(
            json.dumps(
                {
                    "model": "m",
                    "events": [
                        {
                            "event_id": "graph:1",
                            "kind": "llm_call_start",
                            "sequence": 1,
                            "payload": {"messages_ref": ref, "run_id": "r"},
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="no messages"):
            load_recorded_call(directory, "graph:1")


class _FakeCompletion:
    """Records call kwargs and returns a litellm-shaped mapping response."""

    def __init__(self, texts: tuple[str, ...] = ("REPLAYED",)) -> None:
        self.texts = texts
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> dict[str, object]:
        self.calls.append(kwargs)
        requested = int(kwargs.get("n", 1) or 1)
        available = self.texts[:requested] or ("",)
        return {"choices": [{"message": {"content": text}} for text in available]}


class TestReplaySingleLlmCall:
    @pytest.fixture
    def call(self, trace_dir: Path) -> RecordedCall:
        return load_recorded_call(trace_dir, "graph:4")

    def test_sends_recorded_messages_and_model_by_default(self, call: RecordedCall) -> None:
        fake = _FakeCompletion()

        replay_single_llm_call(call, completion_fn=fake)

        assert fake.calls[0]["messages"] == call.messages
        assert fake.calls[0]["model"] == "openai/azure/gpt-5.6-luna"

    def test_returns_completion_text_tuple(self, call: RecordedCall) -> None:
        fake = _FakeCompletion(("REPLAYED",))

        assert replay_single_llm_call(call, completion_fn=fake) == ("REPLAYED",)

    def test_substituted_messages_override_recorded_messages(self, call: RecordedCall) -> None:
        fake = _FakeCompletion()
        substituted = [{"role": "system", "content": "PATCHED"}, {"role": "user", "content": "Q"}]

        replay_single_llm_call(call, messages=substituted, completion_fn=fake)

        assert fake.calls[0]["messages"] == substituted

    def test_model_override_is_used(self, call: RecordedCall) -> None:
        fake = _FakeCompletion()

        replay_single_llm_call(call, model="openai/other-model", completion_fn=fake)

        assert fake.calls[0]["model"] == "openai/other-model"

    def test_temperature_is_forwarded_when_supplied(self, call: RecordedCall) -> None:
        fake = _FakeCompletion()

        replay_single_llm_call(call, temperature=1.2, completion_fn=fake)

        assert fake.calls[0]["temperature"] == 1.2

    def test_temperature_is_omitted_when_not_supplied(self, call: RecordedCall) -> None:
        fake = _FakeCompletion()

        replay_single_llm_call(call, completion_fn=fake)

        assert "temperature" not in fake.calls[0]

    def test_requests_n_samples_from_provider(self, call: RecordedCall) -> None:
        fake = _FakeCompletion(("A", "B", "C"))

        assert replay_single_llm_call(call, n=3, completion_fn=fake) == ("A", "B", "C")
        assert fake.calls[0]["n"] == 3

    def test_tops_up_with_extra_calls_when_provider_returns_fewer_choices(
        self, call: RecordedCall
    ) -> None:
        fake = _FakeCompletion(("ONLY",))  # ignores n>1, always returns one choice

        results = replay_single_llm_call(call, n=3, completion_fn=fake)

        assert results == ("ONLY", "ONLY", "ONLY")
        assert len(fake.calls) == 3

    def test_non_positive_n_raises_value_error(self, call: RecordedCall) -> None:
        with pytest.raises(ValueError, match="n must be"):
            replay_single_llm_call(call, n=0, completion_fn=_FakeCompletion())

    def test_forwards_credentials_from_runtime_settings_env(
        self, call: RecordedCall, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("CUGA_MODEL", "openai/env-model")
        monkeypatch.setenv("CUGA_BASE_URL", "https://example.invalid/v1")
        monkeypatch.setenv("CUGA_API_KEY", "secret-key")
        fake = _FakeCompletion()

        replay_single_llm_call(call, completion_fn=fake)

        assert fake.calls[0]["api_base"] == "https://example.invalid/v1"
        assert fake.calls[0]["api_key"] == "secret-key"
        # The recorded model still wins: replay must reproduce the recorded call.
        assert fake.calls[0]["model"] == "openai/azure/gpt-5.6-luna"

    def test_works_without_any_model_env_configured(
        self, call: RecordedCall, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for name in (
            "CUGA_MODEL",
            "LITELLM_MODEL",
            "CUGA_BASE_URL",
            "LITELLM_BASE_URL",
            "CUGA_API_KEY",
            "LITELLM_API_KEY",
        ):
            monkeypatch.delenv(name, raising=False)
        fake = _FakeCompletion()

        assert replay_single_llm_call(call, completion_fn=fake) == ("REPLAYED",)

    def test_extracts_text_from_object_style_response(self, call: RecordedCall) -> None:
        class _Message:
            content = "OBJECT STYLE"

        class _Choice:
            message = _Message()

        class _Response:
            choices = [_Choice()]

        assert replay_single_llm_call(call, completion_fn=lambda **_: _Response()) == (
            "OBJECT STYLE",
        )

    def test_response_without_choices_raises_replay_error(self, call: RecordedCall) -> None:
        from agent_evolve.cuga_wrapper import SingleCallReplayError

        with pytest.raises(SingleCallReplayError):
            replay_single_llm_call(call, completion_fn=lambda **_: {"choices": []})


class TestReplayIsNotAgentStateReplay:
    def test_docstring_disclaims_checkpoint_and_agent_state_replay(self) -> None:
        docstring = (replay_single_llm_call.__doc__ or "").lower()

        assert "single" in docstring
        assert "agent state" in docstring
        assert "checkpoint" in docstring

    def test_wrapper_exposes_no_counterfactual_replay_capability_claim(self) -> None:
        import agent_evolve.cuga_wrapper as wrapper

        assert not hasattr(wrapper, "supports_counterfactual_replay")
