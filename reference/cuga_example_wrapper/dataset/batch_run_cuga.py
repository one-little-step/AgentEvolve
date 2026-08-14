"""
Batch-run the CUGA GAIA wrapper (run2.py) with task-level parallelism.

Features:
- configurable dataset / run2.py / Python executable
- parallel task execution
- per-task timeout
- direct regex validation
- optional LLM-driven validation every EVALUATION_BATCH_SIZE completed tasks
- cumulative accuracy + validation statistics
- per-task stdout/stderr
- CUGA trace preservation
- batch summary JSON
- ablation-friendly run labels/config metadata
- no trace cleanup unless CLEANUP_CHUNKS=True

Typical usage:
    python batch_run_cuga.py

Or:
    GAIA_EXPERIMENT=gaia_l1_validation_tiny5.json python batch_run_cuga.py

The script assumes run2.py accepts:
    python run2.py "<task prompt>"

and prints a JSON object containing at least:
    {
      "run_id": "...",
      "answer": "...",
      "tool_calls": [...],
      "trace": "..."
    }

The dataset format is the format supplied in gaia_l1_validation_tiny5.json:
    queries[].id
    queries[].input.query
    queries[].regex

LLM evaluation uses an OpenAI-compatible /v1/chat/completions endpoint.
Configure:
    EVAL_MODEL
    EVAL_BASE_URL
    EVAL_API_KEY

For LiteLLM:
    EVAL_BASE_URL=http://localhost:4000/v1
    EVAL_API_KEY=...
    EVAL_MODEL=your-evaluator-model

For OpenAI:
    EVAL_BASE_URL=https://api.openai.com/v1
    EVAL_API_KEY=...
    EVAL_MODEL=...

The LLM evaluator is deliberately given:
- the original question
- the regex validator
- the agent answer
- the direct regex result

It must return one verdict per task: correct / wrong.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


# Load evaluator settings from the repository .env before reading them below.
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


# ============================================================================
# CONFIG — EDIT THESE
# ============================================================================

REPO_ROOT = Path(__file__).resolve().parent.parent

RUNNER = REPO_ROOT / "run2.py"

# Dataset path. Can also be overridden by GAIA_EXPERIMENT.
DEFAULT_EXPERIMENT_FILES = [
    # "dataset/exp/gaia_l1_validation_tiny1.json",
    # "dataset/exp/gaia_l1_validation_tiny5.json",
    "dataset/exp/gaia_l1_validation.json",
    "dataset/exp/gaia_l2_validation.json",
    "dataset/exp/gaia_l3_validation.json",

]

EXPERIMENT_FILES = [
    os.environ["GAIA_EXPERIMENT"]
] if os.environ.get("GAIA_EXPERIMENT") else DEFAULT_EXPERIMENT_FILES

# Results root.
TARGET_DIR = REPO_ROOT / "dataset" / "cuga_runs"

# Parallelism.
MAX_WORKERS = int(os.environ.get("GAIA_MAX_WORKERS", "10"))

# Per-task hard timeout.
TASK_TIMEOUT_SECONDS = int(os.environ.get("GAIA_TASK_TIMEOUT", "1200"))

# Maximum stdout/stderr stored in summary metadata. Full logs are preserved
# separately unless LOG_MAX_CHARS is explicitly lowered.
LOG_MAX_CHARS = int(os.environ.get("GAIA_LOG_MAX_CHARS", "200000"))

# Keep per-task directories.
CLEANUP_CHUNKS = False

# ---------------------------------------------------------------------------
# Direct validation
# ---------------------------------------------------------------------------

DIRECT_REGEX_VALIDATION = True

# If True, regex matching is done against the complete answer.
# If False, only the final extracted answer field is used.
REGEX_SEARCH_COMPLETE_OUTPUT = True

# ---------------------------------------------------------------------------
# LLM validation
# ---------------------------------------------------------------------------

LLM_VALIDATION_ENABLED = os.environ.get(
    "GAIA_LLM_EVAL", "true"
).lower() in {"1", "true", "yes", "on"}

# After this many completed tasks, evaluate the newly completed batch.
EVALUATION_BATCH_SIZE = int(os.environ.get("GAIA_EVAL_BATCH_SIZE", "10"))

# If the final batch has fewer than EVALUATION_BATCH_SIZE tasks, evaluate it
# anyway at the end.
EVALUATE_FINAL_PARTIAL_BATCH = True

EVAL_MODEL = (
    os.environ.get("EVAL_MODEL")
    or os.environ.get("GAIA_EVAL_MODEL")
    or ""
)

EVAL_BASE_URL = (
    os.environ.get("EVAL_BASE_URL")
    or os.environ.get("GAIA_EVAL_BASE_URL")
    or ""
).rstrip("/")

EVAL_API_KEY = (
    os.environ.get("EVAL_API_KEY")
    or os.environ.get("GAIA_EVAL_API_KEY")
    or ""
)

# Keep the evaluator compatible with the same endpoint configuration used by
# run2.py when only CUGA credentials are provided in .env.
if not EVAL_BASE_URL:
    EVAL_BASE_URL = (
        os.environ.get("CUGA_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or ""
    ).rstrip("/")
if not EVAL_API_KEY:
    EVAL_API_KEY = (
        os.environ.get("CUGA_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    )
if not EVAL_MODEL:
    EVAL_MODEL = os.environ.get("CUGA_MODEL") or os.environ.get("MODEL_NAME") or ""

EVAL_TIMEOUT_SECONDS = int(os.environ.get("GAIA_EVAL_TIMEOUT", "160"))

# LLM evaluator temperature. Keep 0 for reproducibility.
EVAL_TEMPERATURE = 0

# ---------------------------------------------------------------------------
# Experiment / ablation metadata
# ---------------------------------------------------------------------------

EXPERIMENT_NAME = os.environ.get("GAIA_EXPERIMENT_NAME", "cuga-baseline")

# Put whatever changed in this field:
#
# Examples:
#   baseline
#   no_wikipedia
#   no_web_fetch
#   skill_v2
#   memory_v3
#   model_X
#
ABLATION_NAME = os.environ.get("GAIA_ABLATION", "baseline")

# Free-form JSON string, e.g.
# GAIA_ABLATION_CONFIG='{"skills":"v1","memory":"v0","tools":["web_search"]}'
ABLATION_CONFIG_JSON = os.environ.get("GAIA_ABLATION_CONFIG", "{}")

# Optional label for the CUGA configuration.
CUGA_MODEL = (
    os.environ.get("CUGA_MODEL")
    or os.environ.get("MODEL_NAME")
    or "unknown"
)


# ============================================================================
# UTILITIES
# ============================================================================

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_task_id(task_id: str) -> str:
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("Invalid task id")
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_id).replace("-", "_")
    if not safe or safe.startswith("."):
        raise ValueError(f"Unsafe task id: {task_id!r}")
    return safe


def format_elapsed(seconds: float) -> str:
    hundredths = int(round(seconds * 100))
    hours, rem = divmod(hundredths, 360000)
    minutes, rem = divmod(rem, 6000)
    secs, hs = divmod(rem, 100)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{hs:02d}"


def load_experiment(path_str: str) -> dict[str, Any]:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate_dataset(queries: list[dict[str, Any]]) -> None:
    ids = [q.get("id") for q in queries]
    if any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("Every query needs a non-empty string id")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task ids in dataset")
    for task_id in ids:
        safe_task_id(task_id)


def task_question(task: dict[str, Any]) -> str:
    value = task.get("input", {})
    if isinstance(value, dict):
        q = value.get("query")
        if isinstance(q, str):
            return q
    if isinstance(value, str):
        return value
    raise ValueError(f"Task {task.get('id')} has no input.query")


def regex_validate(answer: str, pattern: str | None) -> dict[str, Any]:
    if not pattern:
        return {
            "available": False,
            "passed": None,
            "pattern": None,
            "error": None,
        }

    try:
        matched = re.search(pattern, answer or "", flags=0) is not None
        return {
            "available": True,
            "passed": matched,
            "pattern": pattern,
            "error": None,
        }
    except re.error as exc:
        return {
            "available": True,
            "passed": None,
            "pattern": pattern,
            "error": f"invalid regex: {exc}",
        }


def extract_runner_json(stdout: str) -> dict[str, Any] | None:
    """
    run2.py prints a JSON object at the end. CUGA/logging can also emit text.
    First try whole stdout, then scan backwards for a JSON object.
    """
    stdout = stdout.strip()

    try:
        obj = json.loads(stdout)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Find JSON object candidates from the end.
    for match in reversed(list(re.finditer(r"\{", stdout))):
        candidate = stdout[match.start():].strip()
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text[:LOG_MAX_CHARS], encoding="utf-8")


def result_path_for(run_dir: Path, task_id: str) -> Path:
    return run_dir / "tasks" / safe_task_id(task_id) / "result.json"


def task_dir_for(run_dir: Path, task_id: str) -> Path:
    return run_dir / "tasks" / safe_task_id(task_id)


# ============================================================================
# TASK EXECUTION
# ============================================================================

def run_one_task(
    runner: Path,
    task: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    task_id = task["id"]
    question = task_question(task)
    expected_regex = task.get("regex")

    task_dir = task_dir_for(run_dir, task_id)
    task_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    started_at = utc_now()

    cmd = [sys.executable, str(runner), question]

    print(f"[START] {task_id}", flush=True)

    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            text=True,
            capture_output=True,
            timeout=TASK_TIMEOUT_SECONDS,
        )
        return_code = proc.returncode
        stdout = proc.stdout or ""
        stderr = proc.stderr or ""
        timed_out = False
        error = None if return_code == 0 else (
            f"run2.py exited with code {return_code}"
        )
    except subprocess.TimeoutExpired as exc:
        return_code = None
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        timed_out = True
        error = f"task timeout after {TASK_TIMEOUT_SECONDS}s"
    except Exception as exc:
        return_code = None
        stdout = ""
        stderr = ""
        timed_out = False
        error = f"subprocess error: {type(exc).__name__}: {exc}"

    elapsed = time.monotonic() - started

    runner_json = extract_runner_json(stdout)
    answer = ""
    run_id = None
    trace = None
    tool_calls: list[Any] = []

    if runner_json:
        answer = str(runner_json.get("answer") or "")
        run_id = runner_json.get("run_id")
        trace = runner_json.get("trace")
        tool_calls = runner_json.get("tool_calls") or []

    direct = regex_validate(answer if REGEX_SEARCH_COMPLETE_OUTPUT else answer, expected_regex)

    task_result = {
        "task_id": task_id,
        "gaia_task_id": task.get("gaia_task_id"),
        "category": task.get("category"),
        "difficulty": task.get("difficulty"),
        "question": question,
        "expected_regex": expected_regex,
        "started_at": started_at,
        "ended_at": utc_now(),
        "elapsed_seconds": elapsed,
        "elapsed": format_elapsed(elapsed),
        "return_code": return_code,
        "timed_out": timed_out,
        "error": error,
        "run_id": run_id,
        "answer": answer,
        "tool_calls": tool_calls,
        "trace": trace,
        "direct_regex": direct,
        "llm_verdict": None,
        "status": (
            "errored" if error or timed_out
            else ("passed_direct" if direct.get("passed") is True else "failed_direct")
        ),
    }

    write_text(task_dir / "stdout.log", stdout)
    write_text(task_dir / "stderr.log", stderr)
    (task_dir / "result.json").write_text(
        json.dumps(task_result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # Copy the CUGA trace into the batch run if it is a real local path.
    if trace:
        try:
            trace_path = Path(trace)
            if not trace_path.is_absolute():
                trace_path = REPO_ROOT / trace_path
            if trace_path.exists() and trace_path.is_file():
                shutil.copy2(trace_path, task_dir / "cuga_trace.json")
        except Exception:
            pass

    print(
        f"[DONE] {task_id} | {format_elapsed(elapsed)} | "
        f"regex={'PASS' if direct.get('passed') else 'FAIL/NA'}",
        flush=True,
    )

    return task_result


# ============================================================================
# LLM EVALUATION
# ============================================================================

EVALUATOR_SYSTEM = """You are the correctness judge for a GAIA benchmark run.

Your job is to judge whether an agent's answer correctly answers the original
question.

You receive:
- the original GAIA question
- the dataset's direct regex validator
- the agent's final answer
- the direct regex result

The regex is evidence, not absolute truth. It can be insufficient or overly
permissive. Use the original question and answer to decide correctness.

Return ONLY valid JSON with this exact schema:

{
  "verdict": "correct" | "wrong",
  "reason": "short explanation",
  "answer_span": "the part of the agent answer that supports the verdict"
}

Do not reward unsupported claims. If the agent gives the right-looking number
but the question asks for a specific fact/calculation and the answer is
substantively wrong, mark wrong.
"""


def call_openai_compatible_evaluator(
    items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not EVAL_MODEL:
        raise RuntimeError("EVAL_MODEL is not configured")
    if not EVAL_BASE_URL:
        raise RuntimeError("EVAL_BASE_URL is not configured")
    if not EVAL_API_KEY:
        raise RuntimeError("EVAL_API_KEY is not configured")

    user_content = json.dumps(
        [
            {
                "task_id": item["task_id"],
                "question": item["question"],
                "expected_regex": item["expected_regex"],
                "agent_answer": item["answer"],
                "direct_regex_passed": item["direct_regex"]["passed"],
            }
            for item in items
        ],
        ensure_ascii=False,
        indent=2,
    )

    payload = {
        "model": EVAL_MODEL,
        "messages": [
            {"role": "system", "content": EVALUATOR_SYSTEM},
            {
                "role": "user",
                "content": (
                    "Judge each task independently. Return a JSON array with "
                    "one object per task, preserving task_id.\n\n"
                    + user_content
                ),
            },
        ],
    }
    # Azure's GPT-5 model rejects an explicit temperature=0 and requires its
    # default value. Omitting the field preserves compatibility with it.
    if EVAL_TEMPERATURE != 0:
        payload["temperature"] = EVAL_TEMPERATURE

    request = urllib.request.Request(
        f"{EVAL_BASE_URL}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {EVAL_API_KEY}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=EVAL_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Evaluator HTTP {exc.code} from {EVAL_BASE_URL}: {body[:2000]}"
        ) from exc

    content = data["choices"][0]["message"]["content"]

    # Handle markdown fences defensively.
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)

    judged = json.loads(content)
    if not isinstance(judged, list):
        raise ValueError("Evaluator did not return a JSON array")

    by_id = {}
    for row in judged:
        if isinstance(row, dict) and row.get("task_id"):
            by_id[row["task_id"]] = row

    output = []
    for item in items:
        row = by_id.get(item["task_id"])
        if not row:
            output.append({
                "task_id": item["task_id"],
                "verdict": "wrong",
                "reason": "Evaluator omitted this task.",
                "answer_span": "",
            })
        else:
            verdict = str(row.get("verdict", "")).lower()
            if verdict not in {"correct", "wrong"}:
                verdict = "wrong"
            output.append({
                "task_id": item["task_id"],
                "verdict": verdict,
                "reason": str(row.get("reason", "")),
                "answer_span": str(row.get("answer_span", "")),
            })

    return output


def evaluate_batch(batch: list[dict[str, Any]], run_dir: Path, batch_number: int) -> None:
    if not batch:
        return

    if not LLM_VALIDATION_ENABLED:
        return

    if not EVAL_MODEL or not EVAL_BASE_URL or not EVAL_API_KEY:
        print(
            "[LLM-EVAL] skipped: configure EVAL_MODEL, EVAL_BASE_URL and "
            "EVAL_API_KEY",
            flush=True,
        )
        return

    print(
        f"[LLM-EVAL] evaluating {len(batch)} tasks "
        f"(batch #{batch_number})...",
        flush=True,
    )

    try:
        verdicts = call_openai_compatible_evaluator(batch)
    except Exception as exc:
        error_path = run_dir / "evaluations" / f"batch_{batch_number:04d}_error.json"
        error_path.parent.mkdir(parents=True, exist_ok=True)
        error_path.write_text(
            json.dumps(
                {
                    "batch_number": batch_number,
                    "error": f"{type(exc).__name__}: {exc}",
                    "created_at": utc_now(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"[LLM-EVAL] ERROR: {exc}", file=sys.stderr, flush=True)
        return

    verdict_by_id = {v["task_id"]: v for v in verdicts}

    eval_dir = run_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"batch_{batch_number:04d}.json").write_text(
        json.dumps(verdicts, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    for item in batch:
        verdict = verdict_by_id.get(item["task_id"])
        item["llm_verdict"] = verdict
        if verdict:
            item["status"] = (
                "passed_llm"
                if verdict["verdict"] == "correct"
                else "failed_llm"
            )

        path = result_path_for(run_dir, item["task_id"])
        path.write_text(
            json.dumps(item, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    correct = sum(
        1 for v in verdicts if v.get("verdict") == "correct"
    )
    print(
        f"[LLM-EVAL] batch #{batch_number}: "
        f"{correct}/{len(verdicts)} correct",
        flush=True,
    )


# ============================================================================
# SUMMARY
# ============================================================================

def calculate_summary(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(tasks)

    regex_available = [
        t for t in tasks if t["direct_regex"].get("available")
    ]
    regex_correct = [
        t for t in regex_available if t["direct_regex"].get("passed") is True
    ]

    llm_available = [
        t for t in tasks
        if isinstance(t.get("llm_verdict"), dict)
        and t["llm_verdict"].get("verdict") in {"correct", "wrong"}
    ]
    llm_correct = [
        t for t in llm_available
        if t["llm_verdict"]["verdict"] == "correct"
    ]

    return {
        "total_tasks": total,
        "completed_tasks": sum(
            1 for t in tasks
            if not t.get("timed_out") and t.get("return_code") == 0
        ),
        "errored_tasks": sum(
            1 for t in tasks if t.get("error") or t.get("timed_out")
        ),
        "direct_regex": {
            "evaluated": len(regex_available),
            "correct": len(regex_correct),
            "accuracy": (
                len(regex_correct) / len(regex_available)
                if regex_available else None
            ),
        },
        "llm": {
            "evaluated": len(llm_available),
            "correct": len(llm_correct),
            "accuracy": (
                len(llm_correct) / len(llm_available)
                if llm_available else None
            ),
        },
        "mean_task_seconds": (
            sum(float(t["elapsed_seconds"]) for t in tasks) / total
            if total else None
        ),
        "total_task_seconds": sum(
            float(t["elapsed_seconds"]) for t in tasks
        ),
    }


def write_summary(run_dir: Path, experiment: dict[str, Any], tasks: list[dict[str, Any]],
                  started_at: str, ended_at: str, wall_seconds: float) -> Path:
    summary = calculate_summary(tasks)

    data = {
        "run_name": run_dir.name,
        "experiment_name": EXPERIMENT_NAME,
        "ablation": {
            "name": ABLATION_NAME,
            "config": json.loads(ABLATION_CONFIG_JSON),
        },
        "cuga": {
            "runner": str(RUNNER),
            "model": CUGA_MODEL,
            "max_workers": MAX_WORKERS,
            "task_timeout_seconds": TASK_TIMEOUT_SECONDS,
        },
        "validation": {
            "direct_regex": DIRECT_REGEX_VALIDATION,
            "llm_enabled": LLM_VALIDATION_ENABLED,
            "evaluation_batch_size": EVALUATION_BATCH_SIZE,
            "evaluator_model": EVAL_MODEL or None,
        },
        "dataset": {
            "name": experiment.get("name"),
            "version": experiment.get("version"),
            "description": experiment.get("description"),
            "task_count": len(experiment.get("queries", [])),
        },
        "started_at": started_at,
        "ended_at": ended_at,
        "wall_seconds": wall_seconds,
        "wall_elapsed": format_elapsed(wall_seconds),
        "summary": summary,
        "tasks": tasks,
    }

    path = run_dir / "result.json"
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# ============================================================================
# EXPERIMENT
# ============================================================================

def make_run_name(experiment_path: Path) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{experiment_path.stem}__{ABLATION_NAME}__{stamp}"


def run_experiment(experiment_file: str) -> Path:
    experiment_path = Path(experiment_file)
    if not experiment_path.is_absolute():
        experiment_path = REPO_ROOT / experiment_path

    experiment = load_experiment(str(experiment_path))
    queries = experiment.get("queries", [])
    if not isinstance(queries, list) or not queries:
        raise ValueError(f"No queries in {experiment_path}")

    validate_dataset(queries)

    run_dir = TARGET_DIR / make_run_name(experiment_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "experiment_file": str(experiment_path),
                "experiment_name": EXPERIMENT_NAME,
                "ablation": ABLATION_NAME,
                "ablation_config": json.loads(ABLATION_CONFIG_JSON),
                "runner": str(RUNNER),
                "model": CUGA_MODEL,
                "max_workers": MAX_WORKERS,
                "task_timeout_seconds": TASK_TIMEOUT_SECONDS,
                "evaluation_batch_size": EVALUATION_BATCH_SIZE,
                "llm_validation_enabled": LLM_VALIDATION_ENABLED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=" * 72)
    print(f"[EXPERIMENT] {experiment_path}")
    print(f"[RUN]        {run_dir}")
    print(f"[ABLATION]   {ABLATION_NAME}")
    print(f"[MODEL]      {CUGA_MODEL}")
    print(f"[WORKERS]    {MAX_WORKERS}")
    print(f"[TIMEOUT]    {TASK_TIMEOUT_SECONDS}s/task")
    print(f"[LLM EVAL]   {LLM_VALIDATION_ENABLED}")
    print(f"[EVAL BATCH] {EVALUATION_BATCH_SIZE}")
    print("=" * 72)

    started_monotonic = time.monotonic()
    started_at = utc_now()

    tasks_by_id: dict[str, dict[str, Any]] = {}
    completed_order: list[dict[str, Any]] = []
    pending_for_llm: list[dict[str, Any]] = []

    workers = max(1, min(MAX_WORKERS, len(queries)))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(run_one_task, RUNNER, task, run_dir): task["id"]
            for task in queries
        }

        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {
                    "task_id": task_id,
                    "question": next(
                        task_question(q) for q in queries if q["id"] == task_id
                    ),
                    "answer": "",
                    "error": f"worker exception: {type(exc).__name__}: {exc}",
                    "timed_out": False,
                    "return_code": None,
                    "elapsed_seconds": 0,
                    "direct_regex": {
                        "available": False,
                        "passed": None,
                        "pattern": None,
                        "error": None,
                    },
                    "llm_verdict": None,
                    "status": "errored",
                }

            tasks_by_id[task_id] = result
            completed_order.append(result)
            pending_for_llm.append(result)

            # Evaluate every N completed tasks.
            if (
                LLM_VALIDATION_ENABLED
                and len(pending_for_llm) >= EVALUATION_BATCH_SIZE
            ):
                batch = pending_for_llm[:EVALUATION_BATCH_SIZE]
                pending_for_llm = pending_for_llm[EVALUATION_BATCH_SIZE:]
                evaluate_batch(
                    batch,
                    run_dir,
                    batch_number=(len(completed_order) // EVALUATION_BATCH_SIZE),
                )

            # Write a live summary after every completed task.
            ordered = [
                tasks_by_id[q["id"]]
                for q in queries
                if q["id"] in tasks_by_id
            ]
            write_summary(
                run_dir,
                experiment,
                ordered,
                started_at,
                utc_now(),
                time.monotonic() - started_monotonic,
            )

    # Evaluate final partial batch.
    if (
        LLM_VALIDATION_ENABLED
        and EVALUATE_FINAL_PARTIAL_BATCH
        and pending_for_llm
    ):
        evaluate_batch(
            pending_for_llm,
            run_dir,
            batch_number=(
                (len(completed_order) // EVALUATION_BATCH_SIZE) + 1
            ),
        )

    ordered = [
        tasks_by_id[q["id"]]
        for q in queries
        if q["id"] in tasks_by_id
    ]

    ended_at = utc_now()
    wall_seconds = time.monotonic() - started_monotonic

    result_path = write_summary(
        run_dir,
        experiment,
        ordered,
        started_at,
        ended_at,
        wall_seconds,
    )

    summary = calculate_summary(ordered)

    print("\n" + "=" * 72)
    print("BATCH COMPLETE")
    print("=" * 72)
    print(f"Run:              {run_dir}")
    print(f"Tasks:            {summary['total_tasks']}")
    print(
        f"Regex accuracy:   "
        f"{summary['direct_regex']['correct']}/"
        f"{summary['direct_regex']['evaluated']} "
        f"({summary['direct_regex']['accuracy']})"
    )
    print(
        f"LLM accuracy:     "
        f"{summary['llm']['correct']}/"
        f"{summary['llm']['evaluated']} "
        f"({summary['llm']['accuracy']})"
    )
    print(f"Wall time:        {format_elapsed(wall_seconds)}")
    print(f"Result:           {result_path}")
    print("=" * 72)

    return result_path


def main() -> None:
    if not RUNNER.exists():
        raise SystemExit(f"run2.py not found: {RUNNER}")

    all_results = []

    for experiment_file in EXPERIMENT_FILES:
        all_results.append(run_experiment(experiment_file))

    print("\nAll experiments complete:")
    for path in all_results:
        print(f"  {path}")


if __name__ == "__main__":
    main()
