"""Tests for the real-CUGA benchmark executor.

No network, no model calls, no CUGA SDK. A fake wrapper implementing the narrow
:class:`agent_evolve.benchmarks.cuga_executor.TracingWrapper` protocol stands in
for the real one, so the properties that actually matter are verifiable:

* the harness version reaches ``run_task`` unchanged;
* a missing / None / blank answer is an execution failure, never an empty answer;
* every rollout's trace path is recorded and verified to exist;
* one executor is built per worker thread when driven through ``run_benchmark``;
* absent model configuration fails with a message that names the problem;
* the recorded Gaia result shape (``answer`` / ``trace``) is understood;
* a wrapper that raises fails one task, not the run.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping, cast

import pytest

from agent_evolve.benchmarks import (
    BenchmarkTask,
    TaskOutcome,
    UnknownTaskError,
    run_benchmark,
)
from agent_evolve.benchmarks.cuga_executor import (
    ANSWER_KEYS,
    BUILTIN_HARNESS_NAMES,
    PROCESS_ISOLATION,
    THREAD_ISOLATION,
    TRACE_PATH_KEYS,
    VANILLA_HARNESS,
    ConcurrencyUnsupportedError,
    ConcurrentWorkspaceError,
    CugaEnvironmentError,
    CugaExecutorError,
    HarnessSpecError,
    HarnessVersion,
    MissingTraceError,
    NoAnswerError,
    NoQuestionError,
    TraceRecorder,
    TracingDisabledError,
    make_cuga_executor_factory,
    missing_trace_task_ids,
    preflight,
    require_model_env,
)
from agent_evolve.benchmarks.cuga_process_pool import (
    CugaProcessPool,
    WorkerProtocolError,
)

GRADER = "exact"


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #


def write_trace_dir(root: Path, run_id: str, *, with_document: bool = True) -> Path:
    """Create a trace directory shaped like the one ``TraceWriter`` writes."""
    directory = root / run_id
    directory.mkdir(parents=True, exist_ok=True)
    if with_document:
        (directory / "causal-trace.json").write_text(
            json.dumps({"run_id": run_id, "events": []}), encoding="utf-8"
        )
    return directory


class FakeWrapper:
    """Records what it was asked to run and returns a configurable result."""

    def __init__(
        self,
        trace_root: Path,
        *,
        answer: str | None = "an answer",
        status: str = "success",
        tracing: bool = True,
        write_trace: bool = True,
        answer_key: str = "final_output",
        trace_key: str = "causal_trace_path",
        raises: BaseException | None = None,
        delay: float = 0.0,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        self.trace_root = trace_root
        self.answer = answer
        self.status = status
        self.tracing = tracing
        self.write_trace = write_trace
        self.answer_key = answer_key
        self.trace_key = trace_key
        self.raises = raises
        self.delay = delay
        self.extra = dict(extra or {})
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.thread_names: list[str] = []

    def supports_recorded_environment_replay(self) -> bool:
        return self.tracing

    def run_task(
        self, task_id: str, harness_config: Mapping[str, object]
    ) -> dict[str, object]:
        self.calls.append((task_id, dict(harness_config)))
        self.thread_names.append(threading.current_thread().name)
        if self.delay:
            time.sleep(self.delay)
        if self.raises is not None:
            raise self.raises
        result: dict[str, object] = {
            "task_id": task_id,
            "status": self.status,
            "harness_version": harness_config.get("version"),
        }
        if self.answer is not None or self.answer_key in ("final_output", "answer"):
            result[self.answer_key] = self.answer
        if self.write_trace:
            trace = write_trace_dir(self.trace_root, f"trace-{task_id}")
            result[self.trace_key] = str(trace)
        result.update(self.extra)
        return result


class FakeBenchmark:
    """Minimal Benchmark: passes when the answer equals the expected string."""

    name = "fake"

    def __init__(
        self,
        expected: Mapping[str, str],
        *,
        questions: Mapping[str, str] | None = None,
    ) -> None:
        self._expected = dict(expected)
        self._questions = dict(questions or {})

    def load_tasks(self) -> tuple[BenchmarkTask, ...]:
        return tuple(
            BenchmarkTask(
                task_id=tid,
                question=self._questions.get(tid, f"question for {tid}"),
            )
            for tid in self._expected
        )

    def grading_for(self, task_id: str):  # pragma: no cover - unused
        return None

    def graders(self) -> tuple[str, ...]:
        return (GRADER,)

    def score(self, task_id: str, answer: str, *, grader: str) -> TaskOutcome:
        if task_id not in self._expected:
            raise UnknownTaskError(task_id)
        passed = answer == self._expected[task_id]
        return TaskOutcome(
            task_id=task_id,
            score=1.0 if passed else 0.0,
            passed=passed,
            grader_name=grader,
        )

    def score_all(self, task_id: str, answer: str) -> Mapping[str, TaskOutcome]:
        return {GRADER: self.score(task_id, answer, grader=GRADER)}


def build(
    wrapper: FakeWrapper,
    harness: HarnessVersion = VANILLA_HARNESS,
    recorder: TraceRecorder | None = None,
):
    """A factory over a single shared fake wrapper (fine: the fake is safe)."""
    return make_cuga_executor_factory(
        harness, recorder=recorder, wrapper_factory=lambda: wrapper
    )


TASK = BenchmarkTask(task_id="t-1", question="what is 2+2?")


# --------------------------------------------------------------------------- #
# harness version resolution
# --------------------------------------------------------------------------- #


def test_builtin_vanilla_harness_is_resolvable_and_named():
    harness = HarnessVersion.resolve("vanilla")
    assert harness.version == "vanilla"
    assert harness is VANILLA_HARNESS
    assert "vanilla" in BUILTIN_HARNESS_NAMES


def test_harness_version_is_required_and_non_empty():
    with pytest.raises(HarnessSpecError):
        HarnessVersion(version="")
    with pytest.raises(HarnessSpecError):
        HarnessVersion(version="   ")


def test_harness_json_file_must_declare_a_version(tmp_path):
    """A version is never inferred from a filename.

    The label is stamped onto every trace the run writes; guessing it would make
    two different harnesses indistinguishable after the fact.
    """
    path = tmp_path / "b1.json"
    path.write_text(json.dumps({"instructions": "be brief"}), encoding="utf-8")
    with pytest.raises(HarnessSpecError, match="version"):
        HarnessVersion.resolve(str(path))


def test_harness_json_file_loads_all_artifact_groups(tmp_path):
    path = tmp_path / "b1.json"
    path.write_text(
        json.dumps(
            {
                "version": "b1-v3",
                "instructions": "be exact",
                "skills": {"arith": "use the calculator"},
                "memory": {"note": "remember this"},
                "policies": {"fmt": "end with ANSWER:"},
            }
        ),
        encoding="utf-8",
    )
    harness = HarnessVersion.resolve(str(path))
    assert harness.version == "b1-v3"
    assert harness.instructions == "be exact"
    assert harness.skills == {"arith": "use the calculator"}
    assert harness.memory == {"note": "remember this"}
    assert harness.policies == {"fmt": "end with ANSWER:"}
    assert harness.source == str(path)
    assert harness.requires_workspace is True


def test_unknown_harness_name_is_rejected_with_the_available_options():
    with pytest.raises(HarnessSpecError, match="unknown harness"):
        HarnessVersion.resolve("b7-does-not-exist")


def test_empty_harness_spec_is_rejected():
    with pytest.raises(HarnessSpecError):
        HarnessVersion.resolve("   ")


def test_malformed_harness_json_is_rejected(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HarnessSpecError, match="not valid JSON"):
        HarnessVersion.resolve(str(path))


def test_non_object_harness_json_is_rejected(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(HarnessSpecError, match="JSON object"):
        HarnessVersion.resolve(str(path))


def test_vanilla_harness_needs_no_workspace():
    assert VANILLA_HARNESS.requires_workspace is False


# --------------------------------------------------------------------------- #
# harness version flows through to run_task unchanged
# --------------------------------------------------------------------------- #


def test_harness_version_reaches_run_task_unchanged(tmp_path):
    """The whole point: a run executes against exactly the named harness."""
    harness = HarnessVersion(version="b3-candidate-7")
    wrapper = FakeWrapper(tmp_path)
    executor = build(wrapper, harness)()
    executor(TASK)

    task_id, config = wrapper.calls[0]
    assert task_id == "t-1"
    # ``version`` is the key ``_artifact_metadata`` reads to stamp
    # ``harness_version`` onto the persisted trace.
    assert config["version"] == "b3-candidate-7"


def test_all_harness_artifacts_reach_run_task(tmp_path):
    harness = HarnessVersion(
        version="b4",
        instructions="answer exactly",
        skills={"s1": "skill body"},
        memory={"m1": "memory body"},
        policies={"p1": "policy body"},
    )
    wrapper = FakeWrapper(tmp_path)
    build(wrapper, harness)()(TASK)

    _, config = wrapper.calls[0]
    assert config == {
        "version": "b4",
        "input": "what is 2+2?",
        "instructions": "answer exactly",
        "skills": {"s1": "skill body"},
        "memory": {"m1": "memory body"},
        "policies": {"p1": "policy body"},
    }


def test_question_is_sent_as_input_following_the_adapter_convention(tmp_path):
    wrapper = FakeWrapper(tmp_path)
    build(wrapper)()(BenchmarkTask(task_id="t-9", question="the real question"))
    _, config = wrapper.calls[0]
    assert config["input"] == "the real question"


def test_absent_artifact_groups_are_omitted_not_sent_empty(tmp_path):
    """An empty ``skills`` dict would make CUGA build a needless workspace."""
    wrapper = FakeWrapper(tmp_path)
    build(wrapper, HarnessVersion(version="bare"))()(TASK)
    _, config = wrapper.calls[0]
    assert set(config) == {"version", "input"}


def test_a_task_with_no_question_is_refused(tmp_path):
    wrapper = FakeWrapper(tmp_path)
    executor = build(wrapper)()
    with pytest.raises(NoQuestionError):
        executor(BenchmarkTask(task_id="t-blank", question="   "))
    assert wrapper.calls == []


# --------------------------------------------------------------------------- #
# answer extraction: no answer is never a wrong answer
# --------------------------------------------------------------------------- #


def test_answer_is_extracted_from_final_output(tmp_path):
    wrapper = FakeWrapper(tmp_path, answer="4")
    assert build(wrapper)()(TASK) == "4"


def test_answer_is_extracted_from_the_recorded_gaia_result_shape(tmp_path):
    """``datasets/gaia/*/tasks/*/result.json`` and ``stdout.log`` use ``answer``.

    A run_task-shaped dict carrying ``answer`` + ``trace`` (rather than
    ``final_output`` + ``causal_trace_path``) must still be understood, because
    that is the shape the real recorded baseline produced.
    """
    trace_file = tmp_path / "841373bc.json"
    trace_file.write_text("{}", encoding="utf-8")
    result = {
        "run_id": "841373bc-9eaf-420f-90ff-e5285f5572ae",
        "answer": "The fish bag's calculated volume was approximately 0.1777 m^3.",
        "tool_calls": [],
        "trace": str(trace_file),
    }

    class RecordedShapeWrapper:
        def supports_recorded_environment_replay(self) -> bool:
            return True

        def run_task(self, task_id, harness_config):
            return result

    recorder = TraceRecorder()
    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, recorder=recorder, wrapper_factory=RecordedShapeWrapper
    )
    answer = factory()(TASK)
    assert "0.1777" in answer
    assert recorder.trace_path("t-1") == trace_file
    assert "answer" in ANSWER_KEYS and "trace" in TRACE_PATH_KEYS


def test_none_answer_raises_rather_than_returning_an_empty_string(tmp_path):
    """A crashed harness must never look like a wrong answer.

    Returning "" would place the task in the scoring denominator and count it as
    failed, deflating the pass rate of a harness that never actually answered.
    """
    wrapper = FakeWrapper(tmp_path, answer=None)
    with pytest.raises(NoAnswerError):
        build(wrapper)()(TASK)


def test_missing_answer_field_raises(tmp_path):
    class NoAnswerFieldWrapper:
        def supports_recorded_environment_replay(self) -> bool:
            return True

        def run_task(self, task_id, harness_config):
            return {"task_id": task_id, "status": "success"}

    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=NoAnswerFieldWrapper
    )
    with pytest.raises(NoAnswerError, match="no answer field"):
        factory()(TASK)


@pytest.mark.parametrize("blank", ["", "   ", "\n\t"])
def test_blank_answer_raises_rather_than_being_scored_as_wrong(tmp_path, blank):
    wrapper = FakeWrapper(tmp_path, answer=blank)
    with pytest.raises(NoAnswerError, match="blank"):
        build(wrapper)()(TASK)


def test_status_error_raises_and_surfaces_the_reason(tmp_path):
    wrapper = FakeWrapper(
        tmp_path,
        answer="partial text",
        status="error",
        extra={"error": "ConnectionError: endpoint refused"},
    )
    with pytest.raises(NoAnswerError, match="endpoint refused"):
        build(wrapper)()(TASK)


def test_non_string_answer_is_not_stringified(tmp_path):
    class BadTypeWrapper:
        def supports_recorded_environment_replay(self) -> bool:
            return True

        def run_task(self, task_id, harness_config):
            return {"status": "success", "final_output": {"nested": "object"}}

    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=BadTypeWrapper
    )
    with pytest.raises(NoAnswerError, match="not str"):
        factory()(TASK)


def test_non_mapping_result_is_refused(tmp_path):
    class NotAMappingWrapper:
        def supports_recorded_environment_replay(self) -> bool:
            return True

        def run_task(self, task_id, harness_config):
            return "just a string"

    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=cast(Any, NotAMappingWrapper)
    )
    with pytest.raises(NoAnswerError, match="not a mapping"):
        factory()(TASK)


# --------------------------------------------------------------------------- #
# trace capture is mandatory
# --------------------------------------------------------------------------- #


def test_trace_path_is_recorded_per_task(tmp_path):
    wrapper = FakeWrapper(tmp_path)
    recorder = TraceRecorder()
    executor = build(wrapper, VANILLA_HARNESS, recorder)()
    executor(BenchmarkTask(task_id="a", question="qa"))
    executor(BenchmarkTask(task_id="b", question="qb"))

    assert len(recorder) == 2
    assert recorder.trace_path("a") == tmp_path / "trace-a"
    assert recorder.trace_path("b") == tmp_path / "trace-b"
    assert all(p.is_dir() for p in recorder.trace_paths)


def test_rollout_record_carries_provenance_but_no_answer_text(tmp_path):
    wrapper = FakeWrapper(tmp_path, answer="the secret answer")
    recorder = TraceRecorder()
    build(wrapper, HarnessVersion(version="b2"), recorder)()(TASK)

    record = recorder.records[0]
    assert record.task_id == "t-1"
    assert record.harness_version == "b2"
    assert record.status == "success"
    assert record.answer_chars == len("the secret answer")
    assert "the secret answer" not in repr(record)


def test_answer_without_a_trace_is_an_execution_failure(tmp_path):
    """A run that produces answers but no traces is useless to the analyzer."""
    wrapper = FakeWrapper(tmp_path, write_trace=False)
    with pytest.raises(MissingTraceError, match="no causal trace"):
        build(wrapper)()(TASK)


def test_a_trace_path_that_does_not_exist_is_refused(tmp_path):
    class GhostTraceWrapper:
        def supports_recorded_environment_replay(self) -> bool:
            return True

        def run_task(self, task_id, harness_config):
            return {
                "status": "success",
                "final_output": "answer",
                "causal_trace_path": str(tmp_path / "never-written"),
            }

    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=GhostTraceWrapper
    )
    with pytest.raises(MissingTraceError, match="nothing"):
        factory()(TASK)


def test_a_trace_directory_without_causal_trace_json_is_refused(tmp_path):
    """``load_recorded_call`` and ``CugaAdapter`` both open causal-trace.json."""
    empty = write_trace_dir(tmp_path, "empty-trace", with_document=False)

    class HollowTraceWrapper:
        def supports_recorded_environment_replay(self) -> bool:
            return True

        def run_task(self, task_id, harness_config):
            return {
                "status": "success",
                "final_output": "answer",
                "causal_trace_path": str(empty),
            }

    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=HollowTraceWrapper
    )
    with pytest.raises(MissingTraceError, match="causal-trace.json"):
        factory()(TASK)


def test_a_wrapper_with_tracing_disabled_is_refused_before_any_task(tmp_path):
    """Fail loudly at construction, not after 42 untraced billed rollouts."""
    wrapper = FakeWrapper(tmp_path, tracing=False)
    with pytest.raises(TracingDisabledError, match="tracing disabled"):
        build(wrapper)()
    assert wrapper.calls == []


def test_missing_trace_task_ids_reports_in_input_order(tmp_path):
    wrapper = FakeWrapper(tmp_path)
    recorder = TraceRecorder()
    executor = build(wrapper, VANILLA_HARNESS, recorder)()
    tasks = [
        BenchmarkTask(task_id="z", question="qz"),
        BenchmarkTask(task_id="y", question="qy"),
        BenchmarkTask(task_id="x", question="qx"),
    ]
    executor(tasks[1])
    assert missing_trace_task_ids(tasks, recorder) == ("z", "x")


# --------------------------------------------------------------------------- #
# one executor per worker thread
# --------------------------------------------------------------------------- #


def test_one_executor_is_built_per_worker_thread_via_run_benchmark(tmp_path):
    """The runner's factory contract: one stateful agent per thread, reused.

    Sharing one CUGA agent across threads would interleave two task
    trajectories into a single conversation.
    """
    built: list[str] = []
    lock = threading.Lock()
    wrappers: list[FakeWrapper] = []

    def wrapper_factory():
        with lock:
            built.append(threading.current_thread().name)
        wrapper = FakeWrapper(tmp_path, delay=0.02)
        wrappers.append(wrapper)
        return wrapper

    bench = FakeBenchmark({f"t{i}": "an answer" for i in range(24)})
    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=wrapper_factory
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=4)

    assert result.ok_count == 24
    # One per worker thread, not one per task.
    assert 1 < len(built) <= 4
    assert len(set(built)) == len(built), "a thread built more than one executor"
    assert factory.executors_built == len(built)
    # Every task's trace was recorded, across all threads.
    assert len(factory.recorder) == 24


def test_max_workers_one_builds_exactly_one_executor(tmp_path):
    calls = []

    def wrapper_factory():
        calls.append(1)
        return FakeWrapper(tmp_path)

    bench = FakeBenchmark({f"t{i}": "an answer" for i in range(5)})
    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=wrapper_factory
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=1)

    assert result.ok_count == 5
    assert len(calls) == 1


def test_workspace_bound_harness_refuses_a_second_concurrent_executor(tmp_path):
    """CUGA_FOLDER is process-global; two workspaces would race on it.

    Refusing is the honest option: a silent race would make the run measure a
    harness that never existed.
    """
    harness = HarnessVersion(version="b5", skills={"s": "body"})
    factory = make_cuga_executor_factory(
        harness, wrapper_factory=lambda: FakeWrapper(tmp_path)
    )
    factory()
    with pytest.raises(ConcurrentWorkspaceError, match="CUGA_FOLDER"):
        factory()


def test_preflight_rejects_a_workspace_harness_with_multiple_workers(monkeypatch):
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    harness = HarnessVersion(version="b6", policies={"p": "body"})
    with pytest.raises(ConcurrentWorkspaceError, match="max-workers"):
        preflight(harness, max_workers=10, tasks=5)


def test_preflight_accepts_a_workspace_harness_with_one_worker(monkeypatch):
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    preflight(HarnessVersion(version="b6", memory={"m": "b"}), max_workers=1, tasks=5)


def test_preflight_rejects_an_empty_task_selection(monkeypatch):
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    with pytest.raises(CugaExecutorError, match="no tasks"):
        preflight(VANILLA_HARNESS, max_workers=1, tasks=0)


# --------------------------------------------------------------------------- #
# env guard
# --------------------------------------------------------------------------- #


def test_missing_model_env_fails_with_a_message_that_names_the_problem(monkeypatch):
    monkeypatch.delenv("CUGA_MODEL", raising=False)
    monkeypatch.delenv("LITELLM_MODEL", raising=False)
    # ``prepare_environment`` reloads .env, which would repopulate the vars on a
    # configured machine; stub it so the guard itself is what is under test.
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    with pytest.raises(CugaEnvironmentError, match="CUGA_MODEL"):
        require_model_env()


def test_present_model_env_is_returned(monkeypatch):
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.delenv("CUGA_MODEL", raising=False)
    monkeypatch.setenv("LITELLM_MODEL", "openai/some-model")
    assert require_model_env() == "openai/some-model"


def test_preflight_fails_when_no_model_is_configured(monkeypatch):
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.delenv("CUGA_MODEL", raising=False)
    monkeypatch.delenv("LITELLM_MODEL", raising=False)
    with pytest.raises(CugaEnvironmentError):
        preflight(VANILLA_HARNESS, max_workers=1, tasks=3)


def test_preflight_warms_cuga_imports_only_for_parallel_runs(monkeypatch):
    """The warmup exists to prevent a real observed import deadlock.

    A live two-worker run lost a task to
    ``_DeadlockError: ... _ModuleLock('cuga.backend.cuga_graph.policy.configurable')``
    because CUGA imports 172 modules lazily inside ``invoke()``. Serial runs
    cannot race, so they must not pay the ~10s cost.
    """
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    calls: list[int] = []
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.warm_up_cuga_imports",
        lambda: calls.append(1) or 0,
    )

    preflight(VANILLA_HARNESS, max_workers=1, tasks=3)
    assert calls == [], "a serial run must not pay the warmup cost"

    preflight(
        VANILLA_HARNESS, max_workers=4, tasks=3, allow_unsafe_concurrency=True
    )
    assert calls == [1]


def test_a_failing_warmup_does_not_block_a_run(monkeypatch, capsys):
    """Warming is best-effort: no warmup is better than no run."""
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")

    def boom():
        raise ImportError("no cuga installed")

    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.warm_up_cuga_imports", boom
    )
    preflight(
        VANILLA_HARNESS, max_workers=4, tasks=3, allow_unsafe_concurrency=True
    )
    assert "warmup failed" in capsys.readouterr().out


def test_preflight_refuses_parallel_real_execution(monkeypatch):
    """CUGA's knowledge engine is a process-wide singleton; parallel loses tasks.

    Verified against cuga 0.3.1: the engine constructor takes an exclusive flock
    on ``<cwd>/.cuga/knowledge/.lock``, the path ignores ``CUGA_FOLDER``, and
    flock conflicts between two file objects in the same process. Observed live
    on tiny5: ``--max-workers 2`` answered 1 of 2 tasks with
    ``RuntimeError('Knowledge engine already running in another process')``,
    while ``--max-workers 1`` answered 2 of 2.
    """
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    with pytest.raises(ConcurrencyUnsupportedError, match="knowledge engine"):
        preflight(VANILLA_HARNESS, max_workers=2, tasks=5)


def test_serial_real_execution_is_allowed(monkeypatch):
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    preflight(VANILLA_HARNESS, max_workers=1, tasks=5)


def test_parallel_execution_can_be_forced_explicitly(monkeypatch):
    """An escape hatch must be opt-in and never the default."""
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.warm_up_cuga_imports", lambda: 0
    )
    preflight(
        VANILLA_HARNESS, max_workers=4, tasks=5, allow_unsafe_concurrency=True
    )


def test_status_error_recovers_the_reason_from_the_trace(tmp_path):
    """``CugaWrapper.run_task`` drops the runtime's ``error`` field.

    Without recovering it from the trace, a failed rollout reports
    ``status=error`` with no diagnosis at all -- which is exactly how the
    knowledge-engine lock conflict initially presented.
    """
    trace = write_trace_dir(tmp_path, "err-trace", with_document=False)
    (trace / "causal-trace.json").write_text(
        json.dumps(
            {
                "run_id": "err-trace",
                "error": "RuntimeError('Knowledge engine already running in "
                "another process. Start with --workers 1')",
                "events": [],
            }
        ),
        encoding="utf-8",
    )

    class ErrorNoDetailWrapper:
        def supports_recorded_environment_replay(self) -> bool:
            return True

        def run_task(self, task_id, harness_config):
            # Exactly the shape CugaWrapper returns: status but no error field.
            return {
                "task_id": task_id,
                "status": "error",
                "final_output": "",
                "causal_trace_path": str(trace),
            }

    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=ErrorNoDetailWrapper
    )
    with pytest.raises(NoAnswerError, match="Knowledge engine already running"):
        factory()(TASK)


def test_status_error_without_a_readable_trace_says_so(tmp_path):
    class OpaqueWrapper:
        def supports_recorded_environment_replay(self) -> bool:
            return True

        def run_task(self, task_id, harness_config):
            return {"task_id": task_id, "status": "error", "final_output": ""}

    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=OpaqueWrapper
    )
    with pytest.raises(NoAnswerError, match="does not forward"):
        factory()(TASK)


# --------------------------------------------------------------------------- #
# failures are isolated, not fatal
# --------------------------------------------------------------------------- #


def test_a_wrapper_that_raises_becomes_one_failed_task_not_a_crashed_run(tmp_path):
    good = FakeWrapper(tmp_path)
    bad = FakeWrapper(tmp_path, raises=RuntimeError("CUGA exploded"))
    state = {"n": 0}

    def per_task_wrapper():
        # One wrapper per thread; with max_workers=1 the same wrapper is reused,
        # so route the failure through a switching proxy instead.
        class Switching:
            def supports_recorded_environment_replay(self) -> bool:
                return True

            def run_task(self, task_id, harness_config):
                state["n"] += 1
                target = bad if task_id == "t2" else good
                return target.run_task(task_id, harness_config)

        return Switching()

    bench = FakeBenchmark({f"t{i}": "an answer" for i in range(5)})
    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=per_task_wrapper
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=1)

    assert result.executed_count == 5
    assert result.failed_count == 1
    assert result.ok_count == 4
    # The failed task is excluded from the denominator, not scored as wrong.
    assert result.scored_count == 4
    assert result.pass_count == 4
    failed = result.failed_executions[0]
    assert failed.task.task_id == "t2"
    assert failed.answer is None
    assert "CUGA exploded" in failed.error


def test_a_no_answer_task_is_excluded_from_the_denominator(tmp_path):
    """The measurement property: no answer must not deflate the pass rate."""

    def wrapper_factory():
        class Partial:
            def supports_recorded_environment_replay(self) -> bool:
                return True

            def run_task(self, task_id, harness_config):
                trace = write_trace_dir(tmp_path, f"tr-{task_id}")
                answer = None if task_id in ("t1", "t3") else "an answer"
                return {
                    "status": "success",
                    "final_output": answer,
                    "causal_trace_path": str(trace),
                }

        return Partial()

    bench = FakeBenchmark({f"t{i}": "an answer" for i in range(5)})
    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=wrapper_factory
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=1)

    assert result.failed_count == 2
    assert result.scored_count == 3
    assert result.pass_count == 3
    assert result.pass_rate == pytest.approx(1.0)
    # Partial coverage is flagged rather than presented as a clean 100%.
    assert result.grader_stats.is_partial is True


def test_executor_construction_failure_does_not_abort_the_batch(tmp_path):
    """A wrapper that cannot even be built must fail tasks, not the run."""

    def exploding_factory():
        raise RuntimeError("no CUGA SDK installed")

    bench = FakeBenchmark({f"t{i}": "an answer" for i in range(3)})
    factory = make_cuga_executor_factory(
        VANILLA_HARNESS, wrapper_factory=exploding_factory
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=1)

    assert result.failed_count == 3
    assert result.scored_count == 0
    assert result.pass_rate is None
    assert all("no CUGA SDK" in e.error for e in result.failed_executions)


# --------------------------------------------------------------------------- #
# integration: through run_benchmark, at scale
# --------------------------------------------------------------------------- #


def test_integration_input_ordering_and_isolation_over_24_tasks(tmp_path):
    """Input-ordered results and isolated failures, driven concurrently.

    Ordering must not vary with thread scheduling: selection, entropy accounting
    and run-to-run diffing all depend on it.
    """
    import random

    task_ids = [f"task-{i:02d}" for i in range(24)]
    failing = {"task-03", "task-11", "task-20"}

    def wrapper_factory():
        class Jittery:
            def supports_recorded_environment_replay(self) -> bool:
                return True

            def run_task(self, task_id, harness_config):
                time.sleep(random.uniform(0.001, 0.02))
                if task_id in failing:
                    raise RuntimeError(f"harness crashed on {task_id}")
                trace = write_trace_dir(tmp_path, f"tr-{task_id}")
                assert harness_config["version"] == "b-int"
                return {
                    "status": "success",
                    "final_output": f"answer-{task_id}",
                    "harness_version": harness_config["version"],
                    "causal_trace_path": str(trace),
                }

        return Jittery()

    bench = FakeBenchmark({tid: f"answer-{tid}" for tid in task_ids})
    recorder = TraceRecorder()
    factory = make_cuga_executor_factory(
        HarnessVersion(version="b-int"),
        recorder=recorder,
        wrapper_factory=wrapper_factory,
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=6)

    # Input order, regardless of completion order.
    assert [e.task.task_id for e in result.executions] == task_ids
    assert [o.task_id for o in result.outcomes] == [
        tid for tid in task_ids if tid not in failing
    ]

    # Failures isolated as data; the other 21 survive.
    assert result.failed_count == 3
    assert {e.task.task_id for e in result.failed_executions} == failing
    assert result.ok_count == 21
    assert result.scored_count == 21
    assert result.pass_count == 21

    # Evidence exists for every task that answered, and only those.
    assert len(recorder) == 21
    assert set(missing_trace_task_ids(bench.load_tasks(), recorder)) == failing
    assert {r.harness_version for r in recorder.records} == {"b-int"}


def test_integration_results_match_at_one_and_six_workers(tmp_path):
    """A deterministic executor gives an identical verdict at any concurrency."""

    def wrapper_factory():
        class Det:
            def supports_recorded_environment_replay(self) -> bool:
                return True

            def run_task(self, task_id, harness_config):
                trace = write_trace_dir(tmp_path, f"tr-{task_id}")
                answer = "wrong" if task_id.endswith(("2", "7")) else f"answer-{task_id}"
                return {
                    "status": "success",
                    "final_output": answer,
                    "causal_trace_path": str(trace),
                }

        return Det()

    task_ids = [f"task-{i:02d}" for i in range(22)]
    bench = FakeBenchmark({tid: f"answer-{tid}" for tid in task_ids})

    def run(workers: int):
        return run_benchmark(
            bench,
            make_cuga_executor_factory(
                HarnessVersion(version="b-det"), wrapper_factory=wrapper_factory
            ),
            grader=GRADER,
            max_workers=workers,
        )

    a, b = run(1), run(6)
    fingerprint = lambda r: (  # noqa: E731 - local comparison helper
        [(e.task.task_id, e.answer, e.ok) for e in r.executions],
        [(o.task_id, o.passed) for o in r.outcomes],
    )
    assert fingerprint(a) == fingerprint(b)
    assert a.pass_count == 18  # 22 tasks, 4 ending in 2 or 7


# --------------------------------------------------------------------------- #
# process-isolated parallel execution
# --------------------------------------------------------------------------- #
#
# Threading was measured and rejected. Two probes settle it (logs under
# ``terminal_output/benchmarks/``):
#
# * ``probe-knowledge-persist-dir.log`` -- two knowledge engines with DISTINCT
#   ``persist_dir`` coexist in one process, while two with the SAME one lose a
#   worker to the flock. So the knowledge lock alone is fixable in-process.
# * ``probe-cuga-folder-threadsafety.log`` -- but it is not the only global. Two
#   threads calling the real ``cuga_wrapper._construct_agent`` with two
#   different workspaces left BOTH threads observing ``h1``'s workspace, because
#   ``CUGA_FOLDER`` is one process-wide environment variable and CUGA's sandbox
#   executors and ``prepare_node`` read it during ``invoke()``, long after
#   construction. A serialized build lock cannot help: the read happens later.
#
# That is the disqualifying result. Threading would let a task run against
# another candidate's harness while the trace still stamped its own
# ``harness_version`` -- silently wrong evidence, which is worse than slow
# evidence. So parallelism is per-process.


class FakeProcessPool:
    """Stands in for real subprocess workers.

    Records the isolation each worker was given, so the properties that matter
    are checked without paying for a CUGA process: distinct knowledge stores,
    distinct policy stores, and a harness that cannot leak between workers.
    """

    def __init__(self, root: Path, *, fail: set[str] | None = None) -> None:
        self.root = root
        self.fail = set(fail or ())
        self.lock = threading.Lock()
        self.leases: list[dict[str, object]] = []
        self.ran: list[tuple[str, str, str]] = []  # (worker, task, harness)
        self.locks_held: set[str] = set()
        self.collisions: list[str] = []

    def lease(self, worker_id: str, harness_version: str) -> dict[str, object]:
        with self.lock:
            env: dict[str, object] = {
                "worker_id": worker_id,
                "harness_version": harness_version,
                "knowledge_dir": str(self.root / worker_id / "knowledge"),
                "dbs_dir": str(self.root / worker_id / "dbs"),
            }
            self.leases.append(env)
            if env["knowledge_dir"] in self.locks_held:
                self.collisions.append(str(env["knowledge_dir"]))
            self.locks_held.add(str(env["knowledge_dir"]))
            return env

    def run(
        self, lease: Mapping[str, object], task_id: str, harness_config: Mapping[str, object]
    ) -> dict[str, object]:
        worker = str(lease["worker_id"])
        version = str(harness_config["version"])
        with self.lock:
            self.ran.append((worker, task_id, version))
        if task_id in self.fail:
            raise RuntimeError(f"worker {worker} crashed on {task_id}")
        trace = write_trace_dir(self.root / "traces", f"tr-{task_id}")
        return {
            "task_id": task_id,
            "status": "success",
            "final_output": f"answer-{task_id}",
            # Stamped by the worker that actually ran it, exactly as
            # ``_artifact_metadata`` stamps it from the harness config.
            "harness_version": version,
            "causal_trace_path": str(trace),
        }


def test_parallel_real_execution_is_permitted_when_isolation_is_per_process(
    monkeypatch,
):
    """The gate must open only for a mode that was proven safe.

    ``--max-workers 2`` with process isolation is the whole point of this work:
    a 42-task run at ~40-200s/task is otherwise hours.
    """
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.warm_up_cuga_imports", lambda: 0
    )
    preflight(
        VANILLA_HARNESS, max_workers=2, tasks=5, isolation=PROCESS_ISOLATION
    )


def test_parallel_in_process_execution_is_still_refused(monkeypatch):
    """Threading stays refused, and the refusal must name the measured reason.

    Not the knowledge flock -- that one is fixable with a per-worker
    ``persist_dir``. The unfixable one is ``CUGA_FOLDER``.
    """
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    with pytest.raises(ConcurrencyUnsupportedError, match="CUGA_FOLDER"):
        preflight(
            VANILLA_HARNESS, max_workers=2, tasks=5, isolation=THREAD_ISOLATION
        )


def test_process_isolation_is_not_the_default_for_a_serial_run(monkeypatch):
    """One worker needs no subprocess, and must not silently pay for one."""
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    preflight(VANILLA_HARNESS, max_workers=1, tasks=5)


def test_a_workspace_harness_runs_in_parallel_under_process_isolation(monkeypatch):
    """The real evolution case: candidates with skills/memory/policies.

    Refused for threads (``CUGA_FOLDER``), allowed for processes, because a
    subprocess has its own environment.
    """
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.prepare_environment", lambda: None
    )
    monkeypatch.setenv("CUGA_MODEL", "test-model")
    monkeypatch.setattr(
        "agent_evolve.benchmarks.cuga_executor.warm_up_cuga_imports", lambda: 0
    )
    harness = HarnessVersion(version="cand-7", skills={"s": "body"})
    preflight(harness, max_workers=3, tasks=9, isolation=PROCESS_ISOLATION)
    with pytest.raises(ConcurrentWorkspaceError):
        preflight(harness, max_workers=3, tasks=9, isolation=THREAD_ISOLATION)


def test_process_isolation_gives_every_worker_its_own_knowledge_and_policy_store(
    tmp_path,
):
    """The two stores that collide must be distinct per worker, or tasks die.

    Measured: two engines on one ``persist_dir`` lose one to the flock; two on
    distinct dirs both start.
    """
    pool = FakeProcessPool(tmp_path)
    bench = FakeBenchmark({f"task-{i:02d}": f"answer-task-{i:02d}" for i in range(12)})
    factory = make_cuga_executor_factory(
        HarnessVersion(version="b-iso"), worker_pool=pool
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=4)

    assert result.ok_count == 12
    assert pool.collisions == [], "two workers shared a knowledge store"
    knowledge = [lease["knowledge_dir"] for lease in pool.leases]
    dbs = [lease["dbs_dir"] for lease in pool.leases]
    assert len(set(knowledge)) == len(knowledge), "knowledge_dir was reused"
    assert len(set(dbs)) == len(dbs), "policy store was reused"
    assert 1 < len(pool.leases) <= 4, "one lease per worker, not per task"


def test_every_task_is_stamped_with_the_harness_its_own_worker_ran(tmp_path):
    """Cross-contamination check, on the field a trace actually carries.

    A threaded run cannot promise this: ``CUGA_FOLDER`` is process-global, so a
    task can execute against another candidate's artifacts while its trace still
    claims its own version. Per-worker processes can, and this asserts it on the
    recorded ``harness_version`` rather than assuming it.
    """
    pool = FakeProcessPool(tmp_path)
    recorder = TraceRecorder()
    bench = FakeBenchmark({f"task-{i:02d}": f"answer-task-{i:02d}" for i in range(10)})
    factory = make_cuga_executor_factory(
        HarnessVersion(version="cand-A", skills={"s": "A body"}),
        recorder=recorder,
        worker_pool=pool,
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=3)

    assert result.ok_count == 10
    assert {version for _, _, version in pool.ran} == {"cand-A"}
    assert {record.harness_version for record in recorder.records} == {"cand-A"}
    assert {str(lease["harness_version"]) for lease in pool.leases} == {"cand-A"}


def test_two_different_harnesses_never_share_a_worker(tmp_path):
    """Two candidates evaluated concurrently is the evolution use case.

    Each factory leases its own workers, so no worker can serve two harnesses --
    the property whose absence makes threading unusable.
    """
    pool = FakeProcessPool(tmp_path)
    bench_a = FakeBenchmark({f"a-{i}": f"answer-a-{i}" for i in range(6)})
    bench_b = FakeBenchmark({f"b-{i}": f"answer-b-{i}" for i in range(6)})

    def run(bench, version):
        return run_benchmark(
            bench,
            make_cuga_executor_factory(
                HarnessVersion(version=version, skills={"s": version}),
                worker_pool=pool,
            ),
            grader=GRADER,
            max_workers=2,
        )

    assert run(bench_a, "cand-A").ok_count == 6
    assert run(bench_b, "cand-B").ok_count == 6

    by_worker: dict[str, set[str]] = {}
    for worker, _task, version in pool.ran:
        by_worker.setdefault(worker, set()).add(version)
    mixed = {w: v for w, v in by_worker.items() if len(v) > 1}
    assert mixed == {}, f"a worker served two harnesses: {mixed}"


def test_a_crashed_worker_fails_one_task_not_the_run(tmp_path):
    """Failures stay data. A dead worker must not become a scored empty answer."""
    pool = FakeProcessPool(tmp_path, fail={"task-02", "task-05"})
    bench = FakeBenchmark({f"task-{i:02d}": f"answer-task-{i:02d}" for i in range(8)})
    factory = make_cuga_executor_factory(
        HarnessVersion(version="b-crash"), worker_pool=pool
    )
    result = run_benchmark(bench, factory, grader=GRADER, max_workers=3)

    assert result.failed_count == 2
    assert {e.task.task_id for e in result.failed_executions} == {"task-02", "task-05"}
    assert all(e.answer is None for e in result.failed_executions)
    # Excluded from the denominator, never graded as wrong.
    assert result.scored_count == 6
    assert result.pass_count == 6


def test_process_isolated_results_match_at_one_and_four_workers(tmp_path):
    """Determinism w.r.t. worker count, in input order.

    A pass rate that moves with ``--max-workers`` is not a measurement.
    """
    task_ids = [f"task-{i:02d}" for i in range(16)]
    bench = FakeBenchmark({tid: f"answer-{tid}" for tid in task_ids})

    def run(workers: int):
        return run_benchmark(
            bench,
            make_cuga_executor_factory(
                HarnessVersion(version="b-det2"),
                worker_pool=FakeProcessPool(tmp_path / f"w{workers}"),
            ),
            grader=GRADER,
            max_workers=workers,
        )

    a, b = run(1), run(4)
    fingerprint = lambda r: (  # noqa: E731 - local comparison helper
        [(e.task.task_id, e.answer, e.ok) for e in r.executions],
        [(o.task_id, o.passed) for o in r.outcomes],
    )
    assert fingerprint(a) == fingerprint(b)
    assert [e.task.task_id for e in a.executions] == task_ids
    assert a.ok_count == 16


def test_a_worker_pool_and_a_wrapper_factory_are_mutually_exclusive():
    """Two execution backends at once would silently pick one; refuse instead."""
    with pytest.raises(CugaExecutorError, match="worker_pool"):
        make_cuga_executor_factory(
            VANILLA_HARNESS,
            wrapper_factory=lambda: cast(Any, None),
            worker_pool=cast(Any, object()),
        )


# --------------------------------------------------------------------------- #
# the real subprocess worker pool
# --------------------------------------------------------------------------- #
#
# These exercise the pool's *isolation contract* without starting CUGA: the env
# a worker is given, and the fact that two workers are never given the same
# stores. What the child then does with that env is covered by the live runs
# recorded under ``terminal_output/benchmarks/``.


def test_each_worker_gets_its_own_knowledge_and_policy_store(tmp_path):
    """The two globals that collide, made per-worker.

    Measured (``probe-knowledge-persist-dir.log``): two knowledge engines on one
    ``persist_dir`` lose one to the flock; on distinct dirs both start.
    """
    pool = CugaProcessPool(root=tmp_path, trace_root=tmp_path / "traces")
    a = pool.worker_environment("w0001", "cand-A")
    b = pool.worker_environment("w0002", "cand-A")

    assert a["DYNACONF_KNOWLEDGE__PERSIST_DIR"] != b["DYNACONF_KNOWLEDGE__PERSIST_DIR"]
    assert a["CUGA_DBS_DIR"] != b["CUGA_DBS_DIR"]
    for env in (a, b):
        assert str(tmp_path) in env["DYNACONF_KNOWLEDGE__PERSIST_DIR"]
        assert str(tmp_path) in env["CUGA_DBS_DIR"]


def test_a_worker_never_inherits_a_stale_cuga_folder(tmp_path, monkeypatch):
    """A leaked CUGA_FOLDER is how a task reads the wrong candidate's skills.

    The parent may hold one from an earlier serial run; a child must start clean
    and let its own ``_construct_agent`` bind it.
    """
    monkeypatch.setenv("CUGA_FOLDER", "/stale/previous/candidate")
    pool = CugaProcessPool(root=tmp_path, trace_root=tmp_path / "traces")
    env = pool.worker_environment("w0001", "cand-A")
    assert "CUGA_FOLDER" not in env


def test_worker_environment_preserves_model_credentials(tmp_path, monkeypatch):
    """A worker that cannot reach the model produces no rollout at all."""
    monkeypatch.setenv("CUGA_MODEL", "some/model")
    monkeypatch.setenv("CUGA_BASE_URL", "https://endpoint.invalid/v1")
    monkeypatch.setenv("CUGA_API_KEY", "k")
    pool = CugaProcessPool(root=tmp_path, trace_root=tmp_path / "traces")
    env = pool.worker_environment("w0001", "cand-A")
    assert env["CUGA_MODEL"] == "some/model"
    assert env["CUGA_BASE_URL"] == "https://endpoint.invalid/v1"
    assert env["CUGA_API_KEY"] == "k"


def test_workers_write_traces_to_the_shared_trace_root(tmp_path):
    """Evidence must land in one place regardless of which worker produced it.

    Otherwise the analyzer has to know the worker topology to find a trace.
    """
    pool = CugaProcessPool(root=tmp_path, trace_root=tmp_path / "traces")
    a = pool.worker_environment("w0001", "cand-A")
    b = pool.worker_environment("w0002", "cand-A")
    assert a["AGENT_EVOLVE_TRACE_ROOT"] == b["AGENT_EVOLVE_TRACE_ROOT"]
    assert a["AGENT_EVOLVE_TRACE_ROOT"] == str((tmp_path / "traces").resolve())


def test_the_worker_runs_from_the_repo_so_cuga_config_resolves(tmp_path):
    """cwd is deliberately NOT moved.

    ``knowledge.persist_dir`` is overridden by env instead (verified cold-start
    in ``probe-env-only-isolation.log``), which keeps settings.toml discovery,
    ``.cuga/skills`` and every other cwd-relative CUGA path byte-identical to a
    serial run. Relocating cwd would silently change the configuration under
    measurement.
    """
    pool = CugaProcessPool(root=tmp_path, trace_root=tmp_path / "traces")
    assert pool.worker_cwd == Path.cwd()


def test_a_dead_worker_is_reported_as_a_failed_task_not_an_empty_answer(tmp_path):
    """A worker that dies must never look like a wrong answer."""
    pool = CugaProcessPool(
        root=tmp_path,
        trace_root=tmp_path / "traces",
        python_executable="/nonexistent/python",
    )
    bench = FakeBenchmark({f"t{i}": "an answer" for i in range(3)})
    factory = make_cuga_executor_factory(VANILLA_HARNESS, worker_pool=pool)
    try:
        result = run_benchmark(bench, factory, grader=GRADER, max_workers=1)
    finally:
        pool.close()

    assert result.failed_count == 3
    assert result.ok_count == 0
    assert all(e.answer is None for e in result.failed_executions)
    # No answers means nothing to score -- not a 0% pass rate.
    assert result.scored_count == 0
    assert result.pass_rate is None


def test_the_pool_reports_a_worker_error_verbatim(tmp_path):
    """A worker's own diagnosis is the only clue a caller gets; keep it."""
    pool = CugaProcessPool(root=tmp_path, trace_root=tmp_path / "traces")
    lease = pool.lease("w0001", "cand-A")
    with pytest.raises(WorkerProtocolError, match="not valid JSON"):
        pool._decode("this is not json", "t-1")
    pool.close()
    assert lease is not None


# --------------------------------------------------------------------------- #
# knowledge seeding: making a parallel run comparable to a serial one
# --------------------------------------------------------------------------- #
#
# Found by running it, and it is a *scoring* bug, not a throughput one. A serial
# run reuses the repository's populated ``.cuga/knowledge``; a fresh worker
# started empty. Same tasks, same harness, identical tool surface (verified),
# only the knowledge store differing:
#
#   serial, in-process, 4 tasks      -> 3/4 passed, twice
#   process-isolated, empty store    -> 0/3 passed, at 1 worker AND at 4
#   process-isolated, seeded store   -> passes restored
#
# A pass rate that moves with ``--isolation`` is not a measurement, so a worker
# is given a copy of the reference store instead of an empty one.


def test_each_worker_is_seeded_from_the_reference_knowledge_store(tmp_path):
    """A worker must start from the same knowledge state a serial run uses."""
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "metadata.db").write_text("metadata", encoding="utf-8")
    (reference / "knowledge_vectors.db").write_text("vectors", encoding="utf-8")
    (reference / "files").mkdir()
    (reference / "files" / "note.md").write_text("a note", encoding="utf-8")

    pool = CugaProcessPool(
        root=tmp_path / "workers",
        trace_root=tmp_path / "traces",
        knowledge_seed=reference,
    )
    seeded = pool.seed_knowledge_store("w0001")

    assert (seeded / "metadata.db").read_text(encoding="utf-8") == "metadata"
    assert (seeded / "knowledge_vectors.db").read_text(encoding="utf-8") == "vectors"
    assert (seeded / "files" / "note.md").read_text(encoding="utf-8") == "a note"


def test_a_worker_never_copies_the_reference_lock_file(tmp_path):
    """Copying ``.lock`` would hand a worker a foreign lock's state."""
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / ".lock").write_bytes(b"")
    (reference / "metadata.db").write_text("m", encoding="utf-8")

    pool = CugaProcessPool(
        root=tmp_path / "workers",
        trace_root=tmp_path / "traces",
        knowledge_seed=reference,
    )
    seeded = pool.seed_knowledge_store("w0001")
    assert not (seeded / ".lock").exists()
    assert (seeded / "metadata.db").exists()


def test_seeding_copies_so_a_worker_cannot_corrupt_the_reference(tmp_path):
    """Isolation must be two-way: a worker's writes must not reach the source.

    Otherwise two workers would mutate one store through their copies' backing
    file and the reference would drift mid-run.
    """
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "metadata.db").write_text("original", encoding="utf-8")

    pool = CugaProcessPool(
        root=tmp_path / "workers",
        trace_root=tmp_path / "traces",
        knowledge_seed=reference,
    )
    a = pool.seed_knowledge_store("w0001")
    b = pool.seed_knowledge_store("w0002")
    (a / "metadata.db").write_text("worker one wrote this", encoding="utf-8")

    assert (reference / "metadata.db").read_text(encoding="utf-8") == "original"
    assert (b / "metadata.db").read_text(encoding="utf-8") == "original"


def test_an_absent_reference_store_seeds_nothing_and_does_not_fail(tmp_path):
    """A fresh checkout has no knowledge store; that must not block a run."""
    pool = CugaProcessPool(
        root=tmp_path / "workers",
        trace_root=tmp_path / "traces",
        knowledge_seed=tmp_path / "does-not-exist",
    )
    seeded = pool.seed_knowledge_store("w0001")
    assert seeded.is_dir()
    assert list(seeded.iterdir()) == []


def test_seeding_can_be_switched_off_explicitly(tmp_path):
    """An empty-store run is a legitimate experiment -- just not the default."""
    reference = tmp_path / "reference"
    reference.mkdir()
    (reference / "metadata.db").write_text("m", encoding="utf-8")

    pool = CugaProcessPool(
        root=tmp_path / "workers",
        trace_root=tmp_path / "traces",
        knowledge_seed=None,
    )
    seeded = pool.seed_knowledge_store("w0001")
    assert list(seeded.iterdir()) == []


def test_the_default_reference_store_is_the_one_a_serial_run_uses(tmp_path):
    """Comparability is the point, so the default must not be an empty dir.

    ``<cwd>/.cuga/knowledge`` is exactly what ``KnowledgeConfig`` resolves to
    when ``persist_dir`` is unset, which is what a serial run gets.
    """
    pool = CugaProcessPool(root=tmp_path / "workers", trace_root=tmp_path / "traces")
    assert pool.knowledge_seed == Path.cwd() / ".cuga" / "knowledge"
