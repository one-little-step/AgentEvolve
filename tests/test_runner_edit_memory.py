"""SV-6: the production runner must keep an edit memory the editor can read.

Governing contracts:
* ``docs/architecture/orchestration-lifecycle.md:40-66`` -- the attempt
  lifecycle includes *retrieve history* before the editor call.
* ``docs/architecture/editing-and-validation.md`` -- retry exhaustion is one of
  the named edit-validation mechanisms.

Why this file exists
--------------------
:class:`Orchestrator` records every attempt into :class:`EditMemory`
(``core/orchestrator.py:474``) and consults ``retry_budget`` before proposing
(``:657``). But ``Orchestrator`` is constructed **only in tests**; production
runs :class:`SequentialGepaRunner` (``pipeline.py:819`` and ``:988``), which had
no edit memory at all. The consequence was not merely that
``search_edit_history`` returned ``[]``: nothing ever called
:meth:`EditMemory.record`, so

* ``search_edit_history`` returned ``[]`` on every call,
* ``get_attempt_outcome`` raised ``KeyError`` for every id, and
* ``RetryBudget`` never counted an attempt, so retry exhaustion never fired.

Every assertion here is behavioral: each one drives ``run_attempt`` and then
inspects what a *tool* would return, never a substring of a prompt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.adapters.cuga_editor import CugaEditorAgent  # noqa: E402
from agent_evolve.adapters.cuga_editor_evidence import EvidenceView  # noqa: E402
from agent_evolve.adapters.cuga_editor_state import EditStagingArea  # noqa: E402
from agent_evolve.adapters.cuga_editor_tools import (  # noqa: E402
    EditorToolContext,
    build_tool_callables,
)
from agent_evolve.core.analyzer import FakeAnalyzerJudge  # noqa: E402
from agent_evolve.core.blame import BlameGraph, CausalAnalysis  # noqa: E402
from agent_evolve.core.clustering import LexicalEmbedder  # noqa: E402
from agent_evolve.core.config import resolve_profile  # noqa: E402
from agent_evolve.core.contracts import (  # noqa: E402
    EvolutionCandidate,
    EvolutionTask,
    ExecutionTrace,
)
from agent_evolve.core.editor import (  # noqa: E402
    ArtifactEdit,
    EditorRequest,
    EditorResponse,
)
from agent_evolve.core.fake_editor import FakeEditor  # noqa: E402
from agent_evolve.core.memory import (  # noqa: E402
    AttemptStatus,
    EditMemory,
    artifact_group_of,
)
from agent_evolve.core.orchestrator import SequentialGepaRunner  # noqa: E402
from agent_evolve.core.pool import PersistentPool  # noqa: E402
from agent_evolve.core.storage import JSONFileStorage  # noqa: E402
from agent_evolve.pipeline import build_offline_stack  # noqa: E402
from examples.fake_adapter import FakeAdapter  # noqa: E402

_TOKEN = "graphrag-retrieval"
_CLUSTER = "mechanism-default"


def _task(task_id: str = "task-a", expected: str = _TOKEN) -> EvolutionTask:
    return EvolutionTask(
        task_id=task_id,
        input_text=f"produce {task_id}",
        expected_contract={"expected_substring": expected},
    )


class _NoOpEditor:
    """Editor whose edit never satisfies the task, so every attempt is rejected.

    Needed to exercise retry exhaustion: an accepted attempt moves the lineage
    forward, which is a different budget scope by design.
    """

    editor_model_id: str = "noop-editor"

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        target = request.write_set[0]
        content = request.current_artifacts.get(target, "")
        return EditorResponse(
            rationale="deliberately ineffective edit",
            edits=(
                ArtifactEdit(
                    artifact_id=target,
                    operation="replace",
                    payload={"content": content},
                ),
            ),
            reads=dict(request.current_artifacts),
            writes={target: content},
            risks={},
            expected_effects={},
            editor_model_id=self.editor_model_id,
        )


def _runner(
    *,
    seed: int = 0,
    storage: JSONFileStorage | None = None,
    edit_memory: EditMemory | None = None,
    editor: object | None = None,
) -> SequentialGepaRunner:
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
    kwargs: dict[str, object] = {}
    if edit_memory is not None:
        kwargs["edit_memory"] = edit_memory
    return SequentialGepaRunner(
        adapter=adapter,
        pool=pool,
        analyzer_judge=FakeAnalyzerJudge(),
        editor=editor if editor is not None else FakeEditor(),  # type: ignore[arg-type]
        embedder=LexicalEmbedder(dim=32),
        storage=storage,
        config=resolve_profile("research_sequential", seed=seed),
        mechanism_cluster_id=_CLUSTER,
        seed=seed,
        **kwargs,  # type: ignore[arg-type]
    )


# ---------------------------------------------------------------------- #
# The memory exists and is written
# ---------------------------------------------------------------------- #
def test_runner_exposes_an_edit_memory() -> None:
    """Without this field there is nowhere for attempt history to accumulate."""
    runner = _runner()

    assert isinstance(runner.edit_memory, EditMemory)


def test_run_attempt_records_the_attempt_in_edit_memory() -> None:
    """The lifecycle's history step needs a write, not just a read path."""
    runner = _runner()

    outcome = runner.run_attempt([_task()])

    assert outcome.issue_id, "precondition: an attempt actually ran"
    assert len(runner.edit_memory) == 1


def test_recorded_attempt_is_retrievable_by_its_own_id() -> None:
    """``get_attempt_outcome`` raised KeyError for every id in production."""
    runner = _runner()

    outcome = runner.run_attempt([_task()])

    attempt = runner.edit_memory.get(outcome.attempt_id)
    assert attempt.attempt_id == outcome.attempt_id
    assert attempt.issue_id == outcome.issue_id
    assert attempt.status is outcome.status


def test_recorded_attempt_is_discoverable_by_issue_without_knowing_its_id() -> None:
    """The tool that lists ids was the broken one, so the working tool was unreachable."""
    runner = _runner()

    outcome = runner.run_attempt([_task()])

    found = runner.edit_memory.for_issue(outcome.issue_id)
    assert [a.attempt_id for a in found] == [outcome.attempt_id]


# ---------------------------------------------------------------------- #
# search_edit_history's actual read path
# ---------------------------------------------------------------------- #
def test_retrieve_returns_the_attempt_that_search_edit_history_reads() -> None:
    """``search_edit_history`` calls ``retrieve``; RAM-only writes left it empty.

    This is the assertion that fails loudest before the fix: ``record`` indexed
    ``_records_by_issue`` only when a storage backend was configured, and
    ``retrieve`` reads nothing else.
    """
    runner = _runner(storage=None)

    outcome = runner.run_attempt([_task()])

    records = runner.edit_memory.retrieve(outcome.issue_id, max_records=5)
    assert len(records) == 1
    assert records[0].attempt_id == outcome.attempt_id


def test_retrieve_works_identically_with_and_without_storage(tmp_path: Path) -> None:
    """History must not silently depend on a persistence flag."""
    without = _runner(storage=None)
    outcome_a = without.run_attempt([_task()])

    with_storage = _runner(storage=JSONFileStorage(tmp_path))
    outcome_b = with_storage.run_attempt([_task()])

    assert len(without.edit_memory.retrieve(outcome_a.issue_id, max_records=5)) == 1
    assert len(with_storage.edit_memory.retrieve(outcome_b.issue_id, max_records=5)) == 1


def test_second_attempt_on_the_same_issue_sees_the_first_in_history() -> None:
    """The point of history: do not repeat a strategy that already failed."""
    runner = _runner()

    first = runner.run_attempt([_task()])
    runner.run_attempt([_task()])

    records = runner.edit_memory.retrieve(first.issue_id, max_records=5)
    assert len(records) >= 1
    assert first.attempt_id in {r.attempt_id for r in records}


# ---------------------------------------------------------------------- #
# Retry budget enforcement
# ---------------------------------------------------------------------- #
def test_recording_an_attempt_consumes_retry_budget() -> None:
    """``RetryBudget.record_attempt`` is only reachable through ``EditMemory.record``.

    With no memory write in the runner, the budget counted zero attempts
    forever, so retry exhaustion -- a documented edit-validation mechanism --
    never fired in production.
    """
    runner = _runner()

    outcome = runner.run_attempt([_task()])

    attempt = runner.edit_memory.get(outcome.attempt_id)
    group = artifact_group_of(attempt.artifact_ids)
    budget = runner.edit_memory.retry_budget
    # Lineage for a base-parented attempt is the parent version it forked from.
    remaining = budget.remaining(outcome.issue_id, group, "base-v0")

    assert remaining == budget.max_attempts - 1


def test_retry_budget_exhausts_after_max_attempts_on_one_lineage() -> None:
    """Repeated *rejected* attempts on one scope must exhaust the budget.

    Rejected rather than accepted on purpose. An accepted attempt commits a new
    candidate, and the next attempt forks from that one, so its ``lineage`` key
    legitimately differs and the budget correctly does not exhaust. Retry
    exhaustion is about hammering the *same* lineage, which is what a failing
    editor produces.
    """
    runner = _runner(editor=_NoOpEditor())
    budget = runner.edit_memory.retry_budget

    first = runner.run_attempt([_task()])
    assert first.accepted is False, "precondition: the edit must be rejected"
    attempt = runner.edit_memory.get(first.attempt_id)
    group = artifact_group_of(attempt.artifact_ids)
    lineage = "base-v0"
    for _ in range(budget.max_attempts - 1):
        runner.run_attempt([_task()])

    assert budget.is_exhausted(first.issue_id, group, lineage)


# ---------------------------------------------------------------------- #
# The editor tool surface, end to end
# ---------------------------------------------------------------------- #
def test_search_edit_history_tool_reports_the_prior_attempt_to_the_model() -> None:
    """Drives the real tool closure: what the model literally receives.

    Asserting on the store alone would not prove the tool stopped saying "no
    prior attempts", because ``search_edit_history`` reads through
    :meth:`EditMemory.retrieve` rather than the attempt index.
    """
    runner = _runner()
    outcome = runner.run_attempt([_task()])

    analysis = CausalAnalysis(
        mechanism="m",
        severity=0.5,
        score=0.0,
        blame_graph=BlameGraph(nodes=()),
    )
    trace = ExecutionTrace(
        trace_id="probe-trace",
        candidate_id="base",
        task_id="task-a",
        events=(),
        final_output="",
        status="success",
    )
    tools = build_tool_callables(
        EditorToolContext(
            staging=EditStagingArea(write_set=("skills/retrieval",), creatable_prefixes=()),
            evidence=EvidenceView(
                analysis=analysis,
                trace=trace,
                task=_task(),
                contamination_terms=(),
            ),
            request=EditorRequest(
                base_workspace=runner.adapter.materialize_candidate("base-v0", "probe"),
                task=_task(),
                analysis=analysis,
                issue_id=outcome.issue_id,
                write_set=("skills/retrieval",),
                current_artifacts={},
            ),
            adapter=runner.adapter,
            memory=runner.edit_memory,
        )
    )

    payload = json.loads(tools["search_edit_history"]())

    assert payload, "the model must not be told 'no prior attempts'"
    assert outcome.attempt_id in {entry["attempt_id"] for entry in payload}


# ---------------------------------------------------------------------- #
# Persistence safety is preserved
# ---------------------------------------------------------------------- #
def test_edit_memory_writes_never_leak_the_expected_substring(tmp_path: Path) -> None:
    """Adding a write path must not add a contamination path."""
    runner = _runner(storage=JSONFileStorage(tmp_path))

    runner.run([_task()], n_attempts=2)

    blob = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(tmp_path.rglob("*.json"))
    )
    assert blob
    assert _TOKEN not in blob


# ---------------------------------------------------------------------- #
# The production wiring invariant
# ---------------------------------------------------------------------- #
def test_offline_stack_shares_one_edit_memory_with_a_memory_owning_editor() -> None:
    """Two EditMemory objects would leave the editor reading an empty store.

    This is the regression that reintroduces SV-6 silently: the runner records
    into its own memory, the editor reads its own, both are internally
    consistent, and every history tool still reports nothing.

    Driven with a memory-owning editor because that is the shape production
    uses (:class:`CugaEditorAgent` owns a ``memory``); ``FakeEditor`` has no
    memory attribute, so it could not exercise the invariant.
    """
    editor = CugaEditorAgent(
        adapter=FakeAdapter(),
        memory=EditMemory(),
        agent_factory=lambda tools, prompt: "{}",
    )

    stack = build_offline_stack(task_count=1, editor=editor)

    assert stack.runner.edit_memory is editor.memory


def test_live_stack_factory_shares_one_edit_memory() -> None:
    """The CUGA factory must hand the same object to runner and editor.

    Read statically: constructing a live stack needs credentials and a real
    agent, so the assertion is on the wiring rather than on a run.
    """
    import ast
    import inspect

    from agent_evolve import pipeline as pipeline_module

    tree = ast.parse(inspect.getsource(pipeline_module.build_live_stack))
    shared: list[bool] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == "SequentialGepaRunner"
        ):
            kwargs = {k.arg: k.value for k in node.keywords}
            runner_memory = kwargs.get("edit_memory")
            editor_call = kwargs.get("editor")
            assert isinstance(runner_memory, ast.Name), (
                "build_live_stack must pass edit_memory to the runner"
            )
            assert isinstance(editor_call, ast.Call)
            editor_kwargs = {k.arg: k.value for k in editor_call.keywords}
            editor_memory = editor_kwargs.get("memory")
            assert isinstance(editor_memory, ast.Name), (
                "the editor's memory must be a shared variable, not a fresh "
                "EditMemory() literal"
            )
            shared.append(editor_memory.id == runner_memory.id)

    assert shared == [True]
