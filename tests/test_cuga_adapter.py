from __future__ import annotations

from pathlib import Path

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.core.contracts import CandidateWorkspace, EvolutionTask


class WrapperStub:
    def run_task(self, task_id, harness_config):
        assert harness_config == {"input": "compute"}
        return {
            "task_id": task_id,
            "status": "success",
            "final_output": "four",
            "events": [{"event_id": "tool-1", "kind": "tool_call"}],
        }

    def get_artifacts(self):
        return {"skills/default": "be helpful"}

    def update_artifact(self, artifact_id, content):
        assert artifact_id == "skills/default"
        assert content == "be precise"


def test_cuga_adapter_maps_wrapper_trajectory_to_execution_trace(tmp_path):
    adapter = CugaAdapter(WrapperStub())
    workspace = CandidateWorkspace("attempt-1", "candidate-1", Path(tmp_path), "base-1")
    result = adapter.run_full_rollout(
        workspace,
        EvolutionTask(task_id="task-1", input_text="compute"),
        "rollout-1",
    )

    trace = adapter.capture_trace(result)

    assert trace.trace_id == "rollout-1"
    assert trace.candidate_id == "candidate-1"
    assert trace.task_id == "task-1"
    assert trace.final_output == "four"
    assert trace.events[0].event_id == "tool-1"
