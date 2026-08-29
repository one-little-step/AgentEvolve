"""TL: the voluntary ``list_complementary_parents`` editor tool (D5 finale).

Layers under test
-----------------
1. **Core payload** (`complementary_parent_payload`): resolves the CURRENT
   failure's cluster through the same clusterer that built the index, and
   returns structured statuses -- never exceptions for absent data:
   ``ok`` (solvers exist) / ``solvers_absent`` (degrade to least-bad) /
   ``unclustered`` (no analysis or clusterer refusal) / members always present.
2. **Tool layer** (`build_tool_callables`): the tool is registered, returns a
   JSON string (CUGA requirement), NEVER raises into the agent, and reports
   itself unavailable when no provider was attached -- zero cost by default.
3. **Attach point**: `attach_complement_provider` wires a per-request factory
   so each tool call reads live TS2 state.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.adapters.cuga_editor import attach_complement_provider  # noqa: E402
from agent_evolve.adapters.cuga_editor_tools import (  # noqa: E402
    TOOL_APP_NAMES,
    EditorToolContext,
    build_tool_callables,
)
from agent_evolve.core.blame import CausalAnalysis  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.editor import EditorRequest  # noqa: E402
from agent_evolve.core.mechanism_index import (  # noqa: E402
    IndexEntry,
    SignedMechanismIndex,
    complementary_parent_payload,
)

_FAULT_MECH = (
    "the planner hit a retrieval timeout so the context held no documents "
    "and the model answered from memory"
)


def _analysis() -> CausalAnalysis:
    return CausalAnalysis(
        mechanism=_FAULT_MECH,
        severity=0.7,
        score=0.0,
        blame_graph=BlameGraphShim().graph,
    )


class BlameGraphShim:
    """Minimal blame graph without importing more of core into this test."""

    def __init__(self) -> None:
        from agent_evolve.core.blame import BlameGraph, BlameNode

        self.graph = BlameGraph(
            nodes=(
                BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)),
            )
        )


class _RegistryShim:
    """assign() returning a fixed cluster id -- the indexer already ran."""

    def __init__(self, cluster: str = "c0") -> None:
        self.cluster = cluster
        self.clusterer_for_calls: list[str] = []

    def clusterer_for(self, task_id: str):
        self.clusterer_for_calls.append(task_id)

        class _C:
            def __init__(self, c: str) -> None:
                self._c = c

            def assign(self, analysis):  # type: ignore[no-untyped-def]
                class _A:
                    def __init__(self, c: str) -> None:
                        self.cluster_id = c

                return _A(self._c)

        return _C(self.cluster)


def _index_with(solver_severity: float = 0.9, fault_severity: float = 0.4):
    index = SignedMechanismIndex()
    index.add(
        IndexEntry(
            valence=-1, severity=solver_severity, candidate_id="solver-cand",
            task_id="task-a", cluster_id="task-a:c0",
            artifact_ids=("skills/retrieval",), trace_id="tr-s",
        )
    )
    index.add(
        IndexEntry(
            valence=1, severity=fault_severity, candidate_id="least-bad",
            task_id="task-a", cluster_id="task-a:c0",
            artifact_ids=("policies/x",), trace_id="tr-f",
        )
    )
    return index


# ---------------------------------------------------------------------- #
# Core payload
# ---------------------------------------------------------------------- #
def test_payload_ranks_solvers_first_with_roles() -> None:
    payload = complementary_parent_payload(
        index=_index_with(), registry=_RegistryShim(),
        task_id="task-a", analysis=_analysis(),
    )

    assert payload["status"] == "ok"
    roles = [m["role"] for m in payload["members"]]
    assert roles == ["solver", "least_bad_failure"]
    assert payload["members"][0]["candidate_id"] == "solver-cand"
    assert payload["cluster_id"] == "task-a:c0"


def test_degrades_to_least_bad_when_no_solver_exists() -> None:
    index = SignedMechanismIndex()
    index.add(
        IndexEntry(
            valence=1, severity=0.4, candidate_id="only-failure",
            task_id="task-a", cluster_id="task-a:c0",
            artifact_ids=(), trace_id="tr",
        )
    )

    payload = complementary_parent_payload(
        index=index, registry=_RegistryShim(),
        task_id="task-a", analysis=_analysis(),
    )

    assert payload["status"] == "solvers_absent"
    assert payload["members"][0]["role"] == "least_bad_failure"


def test_no_analysis_reports_unclustered_never_raises() -> None:
    payload = complementary_parent_payload(
        index=_index_with(), registry=_RegistryShim(),
        task_id="task-a", analysis=None,
    )

    assert payload["status"] == "unclustered"
    assert payload["members"] == []


def test_clusterer_refusal_is_reported_not_swallowed() -> None:
    registry = _RegistryShim(cluster="")  # refused assignment

    payload = complementary_parent_payload(
        index=_index_with(), registry=registry,
        task_id="task-a", analysis=_analysis(),
    )

    assert payload["status"] == "unclustered"


def test_payload_top_k_limits_returned_members() -> None:
    index = SignedMechanismIndex()
    for i, sev in enumerate((0.9, 0.8, 0.7)):
        index.add(
            IndexEntry(
                valence=-1, severity=sev, candidate_id=f"solver-{i}",
                task_id="task-a", cluster_id="task-a:c0",
                artifact_ids=(), trace_id=f"tr-{i}",
            )
        )

    payload = complementary_parent_payload(
        index=index, registry=_RegistryShim(),
        task_id="task-a", analysis=_analysis(), limit=2,
    )

    assert payload["status"] == "ok"
    assert [m["candidate_id"] for m in payload["members"]] == [
        "solver-0", "solver-1",
    ]


# ---------------------------------------------------------------------- #
# Tool layer
# ---------------------------------------------------------------------- #
def _tool_ctx(provider) -> EditorToolContext:  # type: ignore[no-untyped-def]
    from agent_evolve.adapters.cuga_editor_evidence import EvidenceView
    from agent_evolve.adapters.cuga_editor_state import EditStagingArea
    from agent_evolve.core.contracts import (
        CandidateWorkspace,
        EvolutionTask,
        ExecutionTrace,
    )

    task = EvolutionTask(
        task_id="task-a", input_text="x", expected_contract={}
    )
    analysis = CausalAnalysis(
        mechanism="m", severity=0.1, score=0.0, blame_graph=BlameGraphShim().graph
    )
    trace = ExecutionTrace(
        trace_id="tr", candidate_id="v1", task_id="task-a",
        events=(), final_output="", status="success",
    )
    request = EditorRequest(
        base_workspace=CandidateWorkspace("att", "v1", Path("."), "v0"),
        task=task,
        analysis=analysis,
        issue_id="i-1",
        write_set=("skills/retrieval",),
        current_artifacts={},
        parents=(),
    )
    return EditorToolContext(
        staging=EditStagingArea(
            write_set=("skills/retrieval",),
            creatable_prefixes=(),
            pool_created_count=0,
        ),
        evidence=EvidenceView(
            analysis=analysis, trace=trace, task=task, contamination_terms=()
        ),
        request=request,
        adapter=object(),
        memory=None,  # type: ignore[arg-type]
        complement_provider=provider,
    )


def test_tool_registered_and_returns_provider_payload_as_json() -> None:
    assert TOOL_APP_NAMES.get("list_complementary_parents") == "parents"

    provider = lambda top_k=5: {"status": "ok", "cluster_id": "task-a:c0", "members": [{"role": "solver"}]}  # noqa: E731
    tools = build_tool_callables(_tool_ctx(provider))

    raw = tools["list_complementary_parents"]()
    decoded = json.loads(raw)
    assert decoded["status"] == "ok"
    assert decoded["members"][0]["role"] == "solver"


def test_tool_threads_top_k_to_the_provider() -> None:
    seen: list[int] = []

    def provider(top_k: int) -> dict:
        seen.append(top_k)
        return {"status": "ok", "members": [{"role": "solver"}]}

    tools = build_tool_callables(_tool_ctx(provider))

    tools["list_complementary_parents"](top_k=2)

    assert seen == [2]


def test_tool_defaults_top_k_to_5() -> None:
    seen: list[int] = []

    def provider(top_k: int) -> dict:
        seen.append(top_k)
        return {"status": "ok", "members": []}

    tools = build_tool_callables(_tool_ctx(provider))

    tools["list_complementary_parents"]()

    assert seen == [5]


def test_tool_without_provider_reports_unavailable_at_zero_cost() -> None:
    tools = build_tool_callables(_tool_ctx(None))

    decoded = json.loads(tools["list_complementary_parents"]())
    assert decoded["status"] == "unavailable"
    assert decoded["members"] == []


def test_tool_converts_a_raising_provider_into_an_error_string() -> None:
    def boom(top_k=5):
        raise RuntimeError("index exploded")

    tools = build_tool_callables(_tool_ctx(boom))

    decoded = json.loads(tools["list_complementary_parents"]())
    assert decoded["status"] == "error"
    assert "index exploded" in decoded["message"]


# ---------------------------------------------------------------------- #
# Attach point
# ---------------------------------------------------------------------- #
def test_attach_sets_the_factory_on_the_agent() -> None:
    from agent_evolve.adapters.cuga_editor import CugaEditorAgent

    agent = CugaEditorAgent.__new__(CugaEditorAgent)  # skip SDK init
    factory = lambda request: (lambda top_k=5: {"status": "ok", "members": []})  # noqa: E731

    attach_complement_provider(agent, factory)

    assert agent.complement_provider_factory is factory