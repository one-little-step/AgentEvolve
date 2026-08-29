"""S1.1/S1.2: the live-path wiring of Judge 2 (positivity judge).

Judge 2 was built and offline-proven (D5/J2B) but never constructed on the live
path: ``build_live_stack`` never passed ``positivity_judge``, so strengths never
entered TS2/index/complementary evidence in production. This file pins:

1. **Behavioural**: ``_build_positivity_judge`` returns a judge exposing the
   ``PositivityJudge`` protocol surface (``analyze_success`` + ``analyzer_model_id``).
2. **Structural (gate-on)**: the composition root passes a positivity judge to
   ``SequentialGepaRunner`` gated on ``config.features.use_positivity_judge``.
3. **Structural (gate-off)**: with the gate off (the default), the runner's
   ``positivity_judge`` is ``None`` -- a run stays byte-identical to today.
"""
from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.pipeline import _build_positivity_judge  # noqa: E402


def test_build_positivity_judge_exposes_the_protocol_surface() -> None:
    judge = _build_positivity_judge(log_sink=None)

    assert hasattr(judge, "analyze_success")
    assert hasattr(judge, "analyzer_model_id")
    assert callable(judge.analyze_success)


def _runner_calls(tree: ast.AST) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == "SequentialGepaRunner"
    ]


def _build_live_tree() -> ast.AST:
    from agent_evolve import pipeline as pipeline_module

    return ast.parse(inspect.getsource(pipeline_module.build_live_stack))


def test_composition_root_passes_positivity_judge_gated_on_feature() -> None:
    """The runner call must carry a positivity_judge keyword whose value is
    conditioned on the feature gate -- not dropped, not hardwired on."""
    tree = _build_live_tree()
    calls = _runner_calls(tree)
    assert calls, "build_live_stack must construct SequentialGepaRunner"

    gate_kwargs = [
        kw
        for call in calls
        for kw in call.keywords
        if kw.arg == "positivity_judge"
    ]
    assert gate_kwargs, (
        "build_live_stack must pass positivity_judge to the runner -- Judge 2 "
        "would otherwise be permanently dead on the live path (S1 regression)"
    )

    # The value must be a conditional on the feature gate so the default stays
    # OFF (None); otherwise an operator who never asks for Judge 2 still pays
    # for it on every passing rollout.
    for kw in gate_kwargs:
        value = kw.value
        assert isinstance(value, ast.IfExp), (
            "positivity_judge must be an IfExp over use_positivity_judge so the "
            "default (gate off) stays None and byte-identical"
        )


def test_gate_off_yields_none_judge() -> None:
    """The conditional's else branch must be None."""
    tree = _build_live_tree()
    calls = _runner_calls(tree)
    gate_kwargs = [
        kw
        for call in calls
        for kw in call.keywords
        if kw.arg == "positivity_judge"
    ]
    assert gate_kwargs
    for kw in gate_kwargs:
        value = kw.value
        assert isinstance(value, ast.IfExp)
        assert isinstance(value.orelse, ast.Constant) and value.orelse.value is None, (
            "gate off must pass None, not a judge"
        )


# ---------------------------------------------------------------------- #
# S1.5: end-to-end -- a passing rollout's strengths reach the complement payload
# ---------------------------------------------------------------------- #
def test_strengths_flow_to_complement_payload_end_to_end() -> None:
    """A strength stored in TS2 must land in the signed index and then in the
    editor's complement payload -- the whole D5 chain, through the production
    route (the same provider factory the composition root attaches)."""
    from agent_evolve.adapters.cuga_editor import attach_complement_provider
    from agent_evolve.core.analyzer import FakeAnalyzerJudge
    from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis, CausalFinding
    from agent_evolve.core.clustering import LexicalEmbedder
    from agent_evolve.core.config import resolve_profile
    from agent_evolve.core.contracts import EvolutionCandidate, EvolutionTask, ExecutionTrace
    from agent_evolve.core.evaluation import ObservedRollout, RolloutScore
    from agent_evolve.core.fake_editor import FakeEditor
    from agent_evolve.core.mechanism_index import complementary_parent_payload
    from agent_evolve.core.orchestrator import SequentialGepaRunner
    from agent_evolve.core.pool import PersistentPool
    from examples.fake_adapter import FakeAdapter

    task = EvolutionTask(
        task_id="task-a", input_text="produce task-a",
        expected_contract={"expected_substring": "graphrag-retrieval"},
    )
    adapter = FakeAdapter()
    pool = PersistentPool(min_comparable_rollouts=1)
    pool.add_base(
        EvolutionCandidate(
            candidate_id="base",
            version="base-v0",
            artifact_hashes={
                d.artifact_id: d.version_hash
                for d in adapter.artifact_inventory("base-v0")
            },
        )
    )
    runner = SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=FakeEditor(),
        embedder=LexicalEmbedder(dim=32),
        config=resolve_profile("research_sequential", seed=0),
        mechanism_cluster_id="mechanism-default",
        seed=0,
    )

    # The proven join pair (fault-vs-fix cosine 0.963): seed one parent fault
    # and one near-identical child strength into the TS2 store.
    fault_text = (
        "the planner hit a retrieval timeout so the context held no documents "
        "and the model answered from memory"
    )
    strength_text = (
        "the planner avoided a retrieval timeout because the context held fresh "
        "documents so the model answered with grounded citations"
    )

    def _trace(candidate_id: str, trace_id: str) -> ExecutionTrace:
        return ExecutionTrace(
            trace_id=trace_id, candidate_id=candidate_id, task_id="task-a",
            events=(), final_output="x", status="success",
        )

    runner._record_stored_trace(
        "parent",
        ObservedRollout(
            task=task, trace=_trace("parent-v", "tr-fault"),
            score=RolloutScore(
                task_id="task-a", grader_name="g", score=0.0,
                scorable=True, passed=False,
            ),
            analysis=CausalAnalysis(
                mechanism=fault_text, severity=0.7, score=0.0,
                blame_graph=BlameGraph(
                    nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)),)
                ),
            ),
        ),
    )
    runner._record_stored_trace(
        "child",
        ObservedRollout(
            task=task, trace=_trace("child-v", "tr-strength"),
            score=RolloutScore(
                task_id="task-a", grader_name="g", score=1.0,
                scorable=True, passed=True,
            ),
            strengths=(
                CausalFinding(
                    verdict_id="strength-tr-strength", candidate_id="child",
                    task_id="task-a", trace_id="tr-strength", valence=-1,
                    status="observed", mechanism_description=strength_text,
                    mechanism_cluster_id="mechanism-cluster-unassigned",
                    severity=0.9, confidence=0.9,
                    blame_graph=BlameGraph(
                        nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)),)
                    ),
                    evidence_refs=("skills/retrieval",), rationale="test",
                ),
            ),
        ),
    )

    # The same provider factory the composition root attaches.
    class _EditorProbe:
        def __init__(self):
            self.complement_provider_factory = None

    probe = _EditorProbe()
    attach_complement_provider(
        probe,
        lambda request: lambda top_k=5: complementary_parent_payload(
            index=runner.signed_mechanism_index(),
            registry=runner.cluster_registry,
            task_id=request.task.task_id,
            analysis=request.analysis,
            limit=top_k,
        ),
    )
    request = SimpleNamespace(
        task=task,
        analysis=CausalAnalysis(
            mechanism=fault_text, severity=0.7, score=0.0,
            blame_graph=BlameGraph(
                nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=("skills/retrieval",)),)
            ),
        ),
    )
    payload = probe.complement_provider_factory(request)()
    assert payload["status"] == "ok"
    assert payload["members"], "the complement tool must return real members"
    assert payload["members"][0]["role"] == "solver"
    assert payload["members"][0]["candidate_id"] == "child"
