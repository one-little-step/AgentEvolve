"""propose_edits must deliver parents and creation authority to the editor.

Without this wiring the multi-parent editor is unreachable and the loop silently
falls back to single-parent editing.
"""
from __future__ import annotations

from agent_evolve.core.contracts import ArtifactEdit
from agent_evolve.core.editor import EditorRequest, EditorResponse

from test_phase_6_orchestrator import _task, _runner  # type: ignore


class RecordingEditor:
    """Captures the request it was given and returns a minimal valid edit."""

    editor_model_id = "recording-editor"

    def __init__(self) -> None:
        self.seen: EditorRequest | None = None
        self.last_parents_read: tuple[str, ...] = ()

    def propose_edit(self, request: EditorRequest) -> EditorResponse:
        self.seen = request
        target = request.write_set[0]
        content = request.current_artifacts.get(target, "") + " edited"
        return EditorResponse(
            rationale="recorded",
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


_TASKS = (_task("task-a"), _task("task-b"))


def _wired() -> tuple[object, RecordingEditor]:
    runner = _runner()
    editor = RecordingEditor()
    runner.editor = editor
    return runner, editor


def _first_issue_and_task(runner):
    tasks = _TASKS
    issues = runner.build_issues(tasks)
    assert issues, "expected at least one failing task to produce an issue"
    issue = issues[0]
    task = runner._task_for(issue, tasks)
    return issue, task


def test_propose_edits_passes_parents_to_the_editor() -> None:
    runner, editor = _wired()
    issue, task = _first_issue_and_task(runner)
    _, analysis = runner.observe(runner.pool.base, task)
    runner.propose_edits(
        runner.pool.base, issue, task, analysis, "att-wiring-1"
    )
    assert editor.seen is not None
    assert editor.seen.parents, "editor received no parent context"
    primaries = [p for p in editor.seen.parents if p.is_primary]
    assert len(primaries) == 1


def test_propose_edits_passes_the_creatable_prefix() -> None:
    runner, editor = _wired()
    issue, task = _first_issue_and_task(runner)
    _, analysis = runner.observe(runner.pool.base, task)
    runner.propose_edits(
        runner.pool.base, issue, task, analysis, "att-wiring-2"
    )
    assert editor.seen.creatable_prefix != ""


def test_propose_edits_returns_observed_parent_ids() -> None:
    runner, editor = _wired()
    issue, task = _first_issue_and_task(runner)
    _, analysis = runner.observe(runner.pool.base, task)
    result = runner.propose_edits(
        runner.pool.base, issue, task, analysis, "att-wiring-3"
    )
    assert len(result) == 4
    assert result[3] == ()


def test_observed_parent_ids_come_from_the_editor_not_the_offer() -> None:
    """An editor that reads a donor reports it; merely offering does not."""
    runner, editor = _wired()
    editor.last_parents_read = ("donor-x",)
    issue, task = _first_issue_and_task(runner)
    _, analysis = runner.observe(runner.pool.base, task)
    result = runner.propose_edits(
        runner.pool.base, issue, task, analysis, "att-wiring-4"
    )
    assert result[3] == ("donor-x",)


def test_run_attempt_still_completes_with_the_new_arity() -> None:
    runner, _ = _wired()
    tasks = _TASKS
    outcome = runner.run_attempt(tasks)
    assert outcome.attempt_id
