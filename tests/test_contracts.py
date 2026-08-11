from pathlib import Path

from agent_evolve.adapters.base import validate_adapter
from agent_evolve.core.contracts import (
    ArtifactDescriptor,
    ArtifactEdit,
    CandidateWorkspace,
    CheckpointDescriptor,
    EvolutionTask,
    ExecutionTrace,
    TraceEvent,
)


def test_contracts_do_not_assume_gaia_wisdom_files():
    artifact = ArtifactDescriptor(
        artifact_id="skills/research",
        kind="skill",
        format="text",
        version_hash="sha256:test",
        readable=True,
        writable=True,
        merge_strategy="adapter-defined",
        bindings=("retriever",),
    )
    assert artifact.artifact_id == "skills/research"


def test_trace_contract_carries_agent_state_provenance():
    trace = ExecutionTrace(
        trace_id="trace-1",
        candidate_id="candidate-1",
        task_id="task-1",
        events=(TraceEvent("event-1", "state", "retriever", None, {"state_hash": "x"}),),
        final_output="answer",
        status="success",
        checkpoint_ids=("checkpoint-1",),
    )
    assert trace.events[0].actor_id == "retriever"


def test_adapter_validation_accepts_optional_replay_implementation():
    class FakeAdapter:
        adapter_name = "fake"

        def artifact_inventory(self, version): return ()
        def read_artifacts(self, version, artifact_ids): return {}
        def materialize_candidate(self, parent_version, attempt_id):
            return CandidateWorkspace(attempt_id, attempt_id, Path("."), parent_version)
        def apply_structured_edits(self, workspace, edits): return {}
        def run_full_rollout(self, workspace, task, rollout_id): return object()
        def capture_trace(self, rollout_result):
            return ExecutionTrace("t", "c", "task", (), "", "success")
        def supports_counterfactual_replay(self): return False
        def discover_checkpoints(self, trace): return ()
        def replay_from_checkpoint(self, checkpoint, workspace, task, rollout_id):
            raise RuntimeError("replay is unsupported")

    validate_adapter(FakeAdapter())
