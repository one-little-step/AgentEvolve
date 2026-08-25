"""Phase 6 — HybridTapeModel: taped prefix, live tail (LIVE-TAIL mode).

The editor-experiment seam: consume recorded responses up to a cutoff
boundary, then hand every later call to a LIVE model constructed lazily via a
caller-supplied factory (the driver wires it to ``LLMManager.get_model``).
Bound tool schemas must survive the handoff — under tape they were droppable,
under live they shape provider requests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from agent_evolve.cuga_wrapper.tape_replay import (
    HybridTapeModel,
    TapeDivergence,
)


def _trace_dir(tmp_path: Path, prompts: list[str]) -> Path:
    """One-recorded-call-per-prompt trace built through _json_safe."""
    from agent_evolve import cuga_wrapper as pkg

    payloads = tmp_path / "payloads"
    payloads.mkdir()

    def store(value: object) -> str:
        import hashlib

        raw = json.dumps(pkg._json_safe(value), sort_keys=True,
                         ensure_ascii=False).encode("utf-8")
        ref = hashlib.sha256(raw).hexdigest()
        (payloads / f"{ref}.json").write_bytes(raw)
        return ref

    lines = []
    for i, prompt in enumerate(prompts):
        from langchain_core.outputs import ChatGeneration, LLMResult

        result = LLMResult(
            generations=[[ChatGeneration(
                message=AIMessage(content=f"taped-answer-{i}", id=f"run-{i}"),
                text=f"taped-answer-{i}")]],
            llm_output={},
        )
        lines.append(json.dumps({
            "kind": "llm_call_start", "sequence": i * 2,
            "event_id": f"g:{i*2}",
            "payload": {"run_id": f"r{i}",
                        "messages_ref": store([[HumanMessage(content=prompt)]])},
        }))
        lines.append(json.dumps({
            "kind": "llm_call_end", "sequence": i * 2 + 1,
            "event_id": f"g:{i*2+1}",
            "payload": {"run_id": f"r{i}", "response_ref": store(result)},
        }))
    (tmp_path / "events.jsonl").write_text("\n".join(lines) + "\n",
                                           encoding="utf-8")
    return tmp_path


class StubLive:
    """Records calls; returns a marker message so live traffic is identifiable."""

    def __init__(self) -> None:
        self.calls: list[list] = []
        self.bound_tools: list = []

    def _record(self, messages) -> AIMessage:
        self.calls.append([getattr(m, "content", m) for m in messages])
        return AIMessage(content="LIVE-ANSWER", id="live-1")

    def invoke(self, messages, **kwargs):
        return self._record(messages)

    def bind_tools(self, tools):
        self.bound_tools.append(tools)

        class _Bound:
            def __init__(self, outer: "StubLive") -> None:
                self._outer = outer

            def invoke(self, messages, **kwargs):
                return self._outer._record(messages)

        return _Bound(self)


@pytest.fixture()
def two_call_trace(tmp_path: Path) -> Path:
    return _trace_dir(tmp_path, ["p-zero", "p-one"])


def _make(trace_dir: Path, cutoff: int, stub: StubLive) -> HybridTapeModel:
    return HybridTapeModel.from_trace(
        trace_dir, cutoff=cutoff, scrub_patterns=(),
        live_factory=lambda: stub)


class TestHandoff:
    def test_tape_serves_before_cutoff(self, two_call_trace: Path) -> None:
        stub = StubLive()
        model = _make(two_call_trace, cutoff=1, stub=stub)
        result = model.invoke([HumanMessage(content="p-zero")])
        assert result.content == "taped-answer-0"
        assert stub.calls == []  # factory untouched before cutoff

    def test_live_factory_used_after_cutoff(self, two_call_trace: Path) -> None:
        stub = StubLive()
        model = _make(two_call_trace, cutoff=1, stub=stub)
        model.invoke([HumanMessage(content="p-zero")])
        result = model.invoke([HumanMessage(content="a different prompt")])
        assert result.content == "LIVE-ANSWER"
        assert len(stub.calls) == 1

    def test_bound_tools_forwarded_to_live_branch(
            self, two_call_trace: Path) -> None:
        stub = StubLive()
        model = _make(two_call_trace, cutoff=1, stub=stub)
        model.invoke([HumanMessage(content="p-zero")])
        bound = model.bind_tools([{"type": "function", "function": {"name": "f"}}])
        bound.invoke([HumanMessage(content="go live")])
        assert stub.bound_tools, "tools were dropped at the handoff"

    def test_gate_still_enforced_on_taped_portion(
            self, two_call_trace: Path) -> None:
        stub = StubLive()
        model = _make(two_call_trace, cutoff=2, stub=stub)
        with pytest.raises(TapeDivergence):
            model.invoke([HumanMessage(content="wrong prompt")])
        assert stub.calls == []

    def test_multiple_live_calls_reuse_one_model(
            self, tmp_path: Path) -> None:
        trace = _trace_dir(tmp_path, ["only"])
        factory_calls: list[int] = []
        stub = StubLive()

        def factory() -> StubLive:
            factory_calls.append(1)
            return stub

        model = HybridTapeModel.from_trace(
            trace, cutoff=0, scrub_patterns=(), live_factory=factory)
        model.invoke([HumanMessage(content="x")])
        model.invoke([HumanMessage(content="y")])
        assert len(factory_calls) == 1  # constructed lazily exactly once


class TestLiveAccounting:
    """Regression: live calls MUST advance the shared pointer (?16 metric bug).

    The first LIVE-TAIL experiment reported ``live_calls: 0`` while two
    provider calls actually fired, because only ``serve()`` incremented the
    pointer — every live branch entry left it frozen at the cutoff and the
    report misread that as models bypassing the injection.
    """

    def test_live_calls_counted_and_pointer_advances(
            self, two_call_trace: Path) -> None:
        stub = StubLive()
        model = _make(two_call_trace, cutoff=1, stub=stub)
        model.invoke([HumanMessage(content="p-zero")])   # taped
        model.invoke([HumanMessage(content="live-one")])
        model.invoke([HumanMessage(content="live-two")])
        assert model.live_calls == 2
        assert model.pointer == 3          # 1 taped + 2 live
        assert len(stub.calls) == 2

    def test_mixed_counts_after_full_run(self, tmp_path: Path) -> None:
        trace = _trace_dir(tmp_path, ["a", "b"])
        stub = StubLive()
        model = HybridTapeModel.from_trace(
            trace, cutoff=2, scrub_patterns=(), live_factory=lambda: stub,
            gate_enabled=True)
        model.invoke([HumanMessage(content="a")])
        model.invoke([HumanMessage(content="b")])
        model.invoke([HumanMessage(content="c")])
        assert model.pointer == 3 and model.live_calls == 1
