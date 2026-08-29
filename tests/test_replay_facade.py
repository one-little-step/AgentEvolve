"""W3: the ReplayExperimentFacade's orchestration + report contract.

Everything here runs offline: the two CUGA seams (``build_model`` and the
wrapper run) are injected, so no SDK or endpoint is touched. What is pinned:

1. Explicit ``resume`` wins; the model is built with that cutoff and the given
   gate setting.
2. ``resume=None`` + an ``analysis`` resolves via ``boundary_for_fault``.
3. Neither -> ``status="error"`` (fall through to full validation), no crash.
4. A raising run is captured as an error report, never raised into the caller.
5. The report carries raw fields only -- no verdict enum.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.cuga_wrapper.replay_facade import (  # noqa: E402
    ReplayExperimentFacade,
)


class _FakeModel:
    def __init__(self, cutoff: int, gate_enabled: bool) -> None:
        self.cutoff = cutoff
        self.gate_enabled = gate_enabled
        self.pointer = 3
        self.live_calls = 1
        self.divergence = None


def _facade(**overrides) -> ReplayExperimentFacade:
    built: list[tuple[int, bool]] = []

    def build(parent_trace_dir: Path, resume: int, gate_enabled: bool) -> _FakeModel:
        built.append((resume, gate_enabled))
        return _FakeModel(resume, gate_enabled)

    def run(model: _FakeModel, task_id: str, harness_config) -> dict:
        return {
            "status": "success",
            "final_output": f"answer for {task_id}",
            "causal_trace_path": str(Path("out/traces/xyz")),
        }

    facade = ReplayExperimentFacade(
        _build_model=build, _run_task=run, _inject_model=lambda m: None
    )
    facade._built = built  # type: ignore[attr-defined]
    for key, value in overrides.items():
        setattr(facade, key, value)
    return facade


def test_explicit_resume_wins_and_gate_is_passed_through() -> None:
    facade = _facade()

    report = facade.run(
        parent_trace_dir=Path("ignored"),
        task_id="task-a",
        harness_config={"input": "x"},
        resume=2,
        gate_enabled=False,
    )

    assert report.status == "ok"
    assert report.resume_boundary == 2
    assert report.gate_enabled is False
    assert facade._built == [(2, False)]  # type: ignore[attr-defined]


def test_analysis_resolves_resume_via_boundary_for_fault(tmp_path) -> None:
    from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
    from agent_evolve.core.tape import TapeIndex

    # A trace whose only node "sandbox" sits at seq 3, after one boundary.
    events = [
        {"event_id": "g0", "kind": "graph_node_start", "parent_event_id": None,
         "sequence": 0, "payload": {"node": "prepare", "step": 0, "state_before_ref": "r"}},
        {"event_id": "g1", "kind": "llm_call_start", "parent_event_id": None,
         "sequence": 1, "payload": {"run_id": "a", "messages_ref": "m-a"}},
        {"event_id": "g2", "kind": "llm_call_end", "parent_event_id": None,
         "sequence": 2, "payload": {"run_id": "a", "response_ref": "r-a"}},
        {"event_id": "g3", "kind": "graph_node_start", "parent_event_id": None,
         "sequence": 3, "payload": {"node": "sandbox", "step": 1, "state_before_ref": "r"}},
        {"event_id": "g4", "kind": "llm_call_start", "parent_event_id": None,
         "sequence": 4, "payload": {"run_id": "b", "messages_ref": "m-b"}},
        {"event_id": "g5", "kind": "llm_call_end", "parent_event_id": None,
         "sequence": 5, "payload": {"run_id": "b", "response_ref": "r-b"}},
    ]
    (tmp_path / "payloads").mkdir()
    TapeIndex(events=events, payloads_dir=tmp_path / "payloads")
    (tmp_path / "events.jsonl").write_text(
        "\n".join(__import__("json").dumps(e) for e in events), encoding="utf-8"
    )

    analysis = CausalAnalysis(
        mechanism="m", severity=0.7, score=0.0,
        blame_graph=BlameGraph(nodes=(BlameNode(actor_id="sandbox", blame=0.9),)),
    )
    facade = _facade()

    report = facade.run(
        parent_trace_dir=tmp_path,
        task_id="task-a",
        harness_config={"input": "x"},
        resume=None,
        analysis=analysis,
    )
    assert report.status == "ok"
    assert report.resume_boundary == 1  # boundaries < seq 3 = {seq1}
    assert facade._built == [(1, True)]  # type: ignore[attr-defined]


def test_no_resume_and_no_analysis_reports_error_and_does_not_run() -> None:
    facade = _facade()

    report = facade.run(
        parent_trace_dir=Path("ignored"),
        task_id="task-a",
        harness_config={"input": "x"},
        resume=None,
        analysis=None,
    )

    assert report.status == "error"
    assert "fall through" in (report.error or "")
    assert facade._built == []  # type: ignore[attr-defined] - nothing built


def test_raising_run_is_captured_as_error_report() -> None:
    def boom(model, task_id, harness_config):
        raise RuntimeError("provider down")

    facade = _facade(_run_task=boom)

    report = facade.run(
        parent_trace_dir=Path("ignored"),
        task_id="task-a",
        harness_config={"input": "x"},
        resume=2,
    )

    assert report.status == "error"
    assert "provider down" in (report.error or "")


def test_report_carries_raw_fields_not_a_verdict() -> None:
    facade = _facade()

    report = facade.run(
        parent_trace_dir=Path("ignored"),
        task_id="task-a",
        harness_config={"input": "x"},
        resume=2,
    ).as_dict()

    assert report["status"] == "ok"
    assert report["taped_calls"] == 2      # pointer 3 - live_calls 1
    assert report["live_calls"] == 1
    assert report["final_output_chars"] > 0
    assert "verdict" not in report
    assert "cleared" not in report
