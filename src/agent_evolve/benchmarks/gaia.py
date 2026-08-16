"""Gaia benchmark adapter.

Reads a completed Gaia run directory produced by the CUGA wrapper runner and
exposes it through the agent-neutral :mod:`agent_evolve.benchmarks.base`
contract.

On-disk shape (verified against ``datasets/gaia/<run_name>/``)::

    <run_dir>/config.json
    <run_dir>/result.json                 # optional run-level rollup
    <run_dir>/tasks/<task_dir>/result.json
    <run_dir>/tasks/<task_dir>/cuga_trace.json
    <run_dir>/tasks/<task_dir>/{stdout,stderr}.log
    <run_dir>/evaluations/batch_NNNN.json        # JSON list of judgments
    <run_dir>/evaluations/batch_NNNN_error.json  # FAILED batch -- not a result

Gaia carries **two independent ground-truth measures**:

``expected_regex``
    A live, deterministic grader. Given a pattern it can score any answer,
    including answers produced by a future rollout.

``recorded_llm_verdict``
    A *replay* of judgments an LLM judge issued during the original run
    (``result.json:llm_verdict`` and ``evaluations/batch_NNNN.json``). It is not
    a live judge: it can only be replayed for the exact answer it judged. This
    module deliberately implements no live LLM judge.

Neither measure is ever placed on a :class:`BenchmarkTask`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

from .base import (
    BenchmarkGrading,
    BenchmarkTask,
    GradingUnavailableError,
    RunObservations,
    TaskOutcome,
    UnknownGraderError,
    UnknownTaskError,
)

__all__ = [
    "GAIA_RESULT_KEYS",
    "GRADER_EXPECTED_REGEX",
    "GRADER_RECORDED_LLM_VERDICT",
    "GaiaBenchmark",
    "GaiaGrading",
    "discover_gaia_runs",
]

GRADER_EXPECTED_REGEX = "expected_regex"
GRADER_RECORDED_LLM_VERDICT = "recorded_llm_verdict"

#: Union of ``result.json`` keys observed across real runs. Coverage is reported
#: for every one of these even when a run omits the key entirely.
GAIA_RESULT_KEYS: tuple[str, ...] = (
    "answer",
    "category",
    "difficulty",
    "direct_regex",
    "elapsed",
    "elapsed_seconds",
    "ended_at",
    "error",
    "expected_regex",
    "gaia_task_id",
    "llm_verdict",
    "question",
    "return_code",
    "run_id",
    "started_at",
    "status",
    "task_id",
    "timed_out",
    "tool_calls",
    "trace",
)

_PASS_VERDICTS = frozenset({"correct"})
_FAIL_VERDICTS = frozenset({"wrong", "incorrect"})


class GaiaGrading(BenchmarkGrading):
    """Gaia grading material. SCORER-ONLY.

    Inherits the redacted ``repr()`` from :class:`BenchmarkGrading`; each
    accessor below is an explicit, auditable read of one secret field.
    """

    def __repr__(self) -> str:
        return (
            f"GaiaGrading(task_id={self.task_id!r}, "
            f"grader_names={self.grader_names!r}, "
            f"payload=<redacted {len(self.payload)} field(s)>)"
        )

    __str__ = __repr__

    @property
    def expected_regex(self) -> str | None:
        value = self.payload.get("expected_regex")
        return value if isinstance(value, str) and value else None

    @property
    def recorded_verdict(self) -> str | None:
        value = self.payload.get("recorded_verdict")
        return value if isinstance(value, str) and value else None

    @property
    def recorded_regex_passed(self) -> bool | None:
        value = self.payload.get("recorded_regex_passed")
        return value if isinstance(value, bool) else None

    @property
    def judged_answer(self) -> str | None:
        """The exact answer the recorded judge saw, used to gate replay."""
        value = self.payload.get("judged_answer")
        return value if isinstance(value, str) else None


def discover_gaia_runs(root: Path | str) -> tuple[Path, ...]:
    """Return sorted run directories under ``root`` (empty when absent)."""
    root = Path(root)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            (child for child in root.iterdir() if child.is_dir() and (child / "tasks").is_dir()),
            key=lambda p: p.name,
        )
    )


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _is_blank(value: object) -> bool:
    return value is None or value == "" or value == [] or value == {}


class GaiaBenchmark:
    """Gaia adapter over one completed run directory."""

    name = "gaia"

    def __init__(
        self,
        *,
        run_dir: Path,
        records: Mapping[str, Mapping[str, object]],
        recorded_verdicts: Mapping[str, Mapping[str, object]],
        failed_eval_batches: int,
        coverage: Mapping[str, object],
        config: Mapping[str, object],
    ) -> None:
        self._run_dir = run_dir
        self._records = dict(records)
        self._recorded_verdicts = dict(recorded_verdicts)
        self._failed_eval_batches = failed_eval_batches
        self._coverage = dict(coverage)
        self._config = dict(config)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_run_dir(cls, run_dir: Path | str) -> "GaiaBenchmark":
        run_dir = Path(run_dir)
        tasks_dir = run_dir / "tasks"

        records: dict[str, dict[str, object]] = {}
        task_dirs_without_record = 0
        unreadable_records = 0
        records_without_task_id = 0

        task_dirs = sorted(tasks_dir.iterdir()) if tasks_dir.is_dir() else []
        for task_dir in task_dirs:
            if not task_dir.is_dir():
                continue
            result_path = task_dir / "result.json"
            if not result_path.is_file():
                task_dirs_without_record += 1
                continue
            payload = _read_json(result_path)
            if not isinstance(payload, dict):
                unreadable_records += 1
                continue
            task_id = payload.get("task_id")
            if not isinstance(task_id, str) or not task_id:
                # Fall back to the directory name so the record is not lost.
                task_id = task_dir.name.replace("_", "-")
                records_without_task_id += 1
            records[task_id] = payload

        coverage = cls._compute_coverage(
            records.values(),
            task_dirs_without_record=task_dirs_without_record,
            unreadable_records=unreadable_records,
            records_without_task_id=records_without_task_id,
        )

        verdicts, failed_batches, batch_conflicts = cls._load_recorded_verdicts(run_dir, records)
        coverage["recorded_verdict_conflicts"] = batch_conflicts

        config = _read_json(run_dir / "config.json")
        return cls(
            run_dir=run_dir,
            records=records,
            recorded_verdicts=verdicts,
            failed_eval_batches=failed_batches,
            coverage=coverage,
            config=config if isinstance(config, dict) else {},
        )

    @staticmethod
    def _compute_coverage(
        records: Iterable[Mapping[str, object]],
        *,
        task_dirs_without_record: int,
        unreadable_records: int,
        records_without_task_id: int,
    ) -> dict[str, object]:
        records = list(records)
        observed_keys = set(GAIA_RESULT_KEYS)
        for record in records:
            observed_keys.update(str(k) for k in record)

        missing: dict[str, int] = {}
        empty: dict[str, int] = {}
        for key in sorted(observed_keys):
            absent = sum(1 for r in records if key not in r)
            unusable = sum(1 for r in records if key not in r or _is_blank(r.get(key)))
            missing[key] = absent
            # "empty" = no usable value: absent, null, or blank.
            empty[key] = unusable
        return {
            "record_count": len(records),
            "task_dirs_without_record": task_dirs_without_record,
            "unreadable_records": unreadable_records,
            "records_without_task_id": records_without_task_id,
            "missing": missing,
            "empty": empty,
        }

    @staticmethod
    def _load_recorded_verdicts(
        run_dir: Path,
        records: Mapping[str, Mapping[str, object]],
    ) -> tuple[dict[str, dict[str, object]], int, int]:
        """Collect replayable judgments and count FAILED evaluation batches.

        ``batch_NNNN_error.json`` files record a batch that never produced
        judgments; they are counted so a partial denominator is visible, and
        never parsed as results.
        """
        verdicts: dict[str, dict[str, object]] = {}

        # 1) per-task llm_verdict embedded in result.json (fallback source)
        for task_id, record in records.items():
            embedded = record.get("llm_verdict")
            if isinstance(embedded, dict) and isinstance(embedded.get("verdict"), str):
                verdicts[task_id] = dict(embedded)

        # 2) evaluations/ batches (authoritative when present)
        failed_batches = 0
        conflicts = 0
        eval_dir = run_dir / "evaluations"
        if eval_dir.is_dir():
            for path in sorted(eval_dir.iterdir()):
                if not path.is_file() or path.suffix != ".json":
                    continue
                if path.stem.endswith("_error"):
                    failed_batches += 1
                    continue
                payload = _read_json(path)
                if not isinstance(payload, list):
                    failed_batches += 1
                    continue
                for item in payload:
                    if not isinstance(item, dict):
                        continue
                    task_id = item.get("task_id")
                    verdict = item.get("verdict")
                    if not isinstance(task_id, str) or not isinstance(verdict, str):
                        continue
                    previous = verdicts.get(task_id)
                    if previous is not None and previous.get("verdict") != verdict:
                        conflicts += 1
                    verdicts[task_id] = dict(item)

        return verdicts, failed_batches, conflicts

    # -- identity ---------------------------------------------------------

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    @property
    def run_name(self) -> str:
        return self._run_dir.name

    @property
    def config(self) -> Mapping[str, object]:
        return dict(self._config)

    # -- task-facing surface ---------------------------------------------

    def load_tasks(self) -> tuple[BenchmarkTask, ...]:
        tasks: list[BenchmarkTask] = []
        for task_id in sorted(self._records):
            record = self._records[task_id]
            question = record.get("question")
            metadata: dict[str, object] = {
                "benchmark": self.name,
                "run_name": self.run_name,
                "question_present": isinstance(question, str) and question != "",
            }
            # Only operational / descriptive fields. `status` is deliberately
            # excluded: its values ("passed_direct", "failed_llm", ...) encode
            # the verdict and are therefore grading material.
            for key in ("category", "difficulty", "gaia_task_id", "timed_out", "elapsed_seconds"):
                if key in record:
                    metadata[key] = record[key]
            tasks.append(
                BenchmarkTask(
                    task_id=task_id,
                    question=question if isinstance(question, str) else "",
                    metadata=metadata,
                )
            )
        return tuple(tasks)

    def recorded_answer(self, task_id: str) -> str | None:
        """The answer the agent produced during this run (not grading data)."""
        record = self._records.get(task_id)
        if record is None:
            return None
        answer = record.get("answer")
        return answer if isinstance(answer, str) else None

    def key_coverage(self) -> Mapping[str, object]:
        return json.loads(json.dumps(self._coverage))

    def recorded_verdict_count(self) -> int:
        return sum(
            1
            for item in self._recorded_verdicts.values()
            if str(item.get("verdict", "")).strip().lower() in (_PASS_VERDICTS | _FAIL_VERDICTS)
        )

    # -- scorer-only surface ---------------------------------------------

    def grading_for(self, task_id: str) -> GaiaGrading | None:
        record = self._records.get(task_id)
        if record is None:
            return None
        judgment = self._recorded_verdicts.get(task_id) or {}
        payload: dict[str, object] = {
            "expected_regex": record.get("expected_regex"),
            "recorded_verdict": judgment.get("verdict"),
            "recorded_reason": judgment.get("reason"),
            "recorded_answer_span": judgment.get("answer_span"),
            "judged_answer": record.get("answer"),
        }
        direct = record.get("direct_regex")
        if isinstance(direct, dict):
            payload["recorded_regex_passed"] = direct.get("passed")
            payload["recorded_regex_error"] = direct.get("error")
        return GaiaGrading(
            task_id=task_id,
            grader_names=self.graders(),
            payload=payload,
        )

    def graders(self) -> tuple[str, ...]:
        return (GRADER_EXPECTED_REGEX, GRADER_RECORDED_LLM_VERDICT)

    def score(self, task_id: str, answer: str, *, grader: str) -> TaskOutcome:
        if grader not in self.graders():
            raise UnknownGraderError(
                f"unknown grader {grader!r} for benchmark {self.name!r}; "
                f"available: {self.graders()}"
            )
        if task_id not in self._records:
            raise UnknownTaskError(f"unknown task_id {task_id!r} in run {self.run_name!r}")
        grading = self.grading_for(task_id)
        assert grading is not None  # task presence already checked
        if grader == GRADER_EXPECTED_REGEX:
            return self._score_expected_regex(grading, answer)
        return self._score_recorded_verdict(grading, answer)

    def try_score(self, task_id: str, answer: str, *, grader: str) -> TaskOutcome | None:
        """Like :meth:`score` but returns ``None`` when grading is unavailable."""
        try:
            return self.score(task_id, answer, grader=grader)
        except GradingUnavailableError:
            return None

    def score_all(self, task_id: str, answer: str) -> Mapping[str, TaskOutcome]:
        """Every *available* grader's verdict, so disagreement is visible.

        Graders with no material for this task/answer are omitted rather than
        recorded as failures.
        """
        outcomes: dict[str, TaskOutcome] = {}
        for grader in self.graders():
            outcome = self.try_score(task_id, answer, grader=grader)
            if outcome is not None:
                outcomes[grader] = outcome
        return outcomes

    # -- graders ----------------------------------------------------------

    @staticmethod
    def _score_expected_regex(grading: GaiaGrading, answer: str) -> TaskOutcome:
        pattern = grading.expected_regex
        if pattern is None:
            raise GradingUnavailableError(
                f"task {grading.task_id!r} has no expected_regex; the regex "
                f"grader cannot measure it (this is not a failing score)"
            )
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise GradingUnavailableError(
                f"task {grading.task_id!r} has an uncompilable expected_regex: {exc}"
            ) from exc
        matched = compiled.search(answer or "") is not None
        return TaskOutcome(
            task_id=grading.task_id,
            score=1.0 if matched else 0.0,
            passed=matched,
            grader_name=GRADER_EXPECTED_REGEX,
            detail={
                "live": True,
                "match_found": matched,
                "answer_chars": len(answer or ""),
            },
        )

    @staticmethod
    def _score_recorded_verdict(grading: GaiaGrading, answer: str) -> TaskOutcome:
        verdict = (grading.recorded_verdict or "").strip().lower()
        if not verdict:
            raise GradingUnavailableError(
                f"task {grading.task_id!r} has no recorded judgment (its "
                f"evaluation batch may have failed); no live judge is "
                f"implemented, so this task cannot be graded"
            )
        if verdict not in _PASS_VERDICTS and verdict not in _FAIL_VERDICTS:
            raise GradingUnavailableError(
                f"task {grading.task_id!r} has an unrecognised recorded "
                f"verdict token; refusing to guess its polarity"
            )
        judged = grading.judged_answer
        if (judged or "").strip() != (answer or "").strip():
            raise GradingUnavailableError(
                f"the recorded judgment for task {grading.task_id!r} applies "
                f"only to the answer it judged; it is a replayed historical "
                f"judgment, not a live judge, and cannot score a new answer"
            )
        passed = verdict in _PASS_VERDICTS
        return TaskOutcome(
            task_id=grading.task_id,
            score=1.0 if passed else 0.0,
            passed=passed,
            grader_name=GRADER_RECORDED_LLM_VERDICT,
            detail={
                "live": False,
                "replayed": True,
                "answer_chars": len(answer or ""),
            },
        )

    # -- run-level --------------------------------------------------------

    def observations(self) -> RunObservations:
        task_ids = tuple(sorted(self._records))
        answers: dict[str, str] = {}
        timed_out: list[str] = []
        errored: list[str] = []
        for task_id in task_ids:
            record = self._records[task_id]
            answer = record.get("answer")
            answers[task_id] = answer if isinstance(answer, str) else ""
            if record.get("timed_out") is True or str(record.get("status", "")).lower() in {
                "timeout",
                "timed_out",
            }:
                timed_out.append(task_id)
            if not _is_blank(record.get("error")) or str(record.get("status", "")).lower() in {
                "errored",
                "error",
            }:
                errored.append(task_id)
        return RunObservations(
            run_name=self.run_name,
            task_ids=task_ids,
            answers=answers,
            timed_out_task_ids=tuple(timed_out),
            errored_task_ids=tuple(errored),
            failed_eval_batches=self._failed_eval_batches,
            key_coverage=self._coverage,
            config=self._config,
        )
