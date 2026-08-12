from __future__ import annotations

import importlib.util
import json
from pathlib import Path


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_vanilla_baseline.py"
SPEC = importlib.util.spec_from_file_location("run_vanilla_baseline", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def test_mock_baseline_writes_one_trajectory_per_configured_task(tmp_path):
    setattr(RUNNER, "OUTPUT_DIRECTORY", tmp_path)
    setattr(RUNNER, "USE_MOCK_RUNTIME", True)
    setattr(
        RUNNER,
        "TASKS",
        (
            {"task_id": "baseline-1", "input": "first task"},
            {"task_id": "baseline-2", "input": "second task"},
        ),
    )

    assert RUNNER.main() == 0

    assert json.loads((tmp_path / "baseline-1.json").read_text())["task_id"] == "baseline-1"
    assert json.loads((tmp_path / "baseline-2.json").read_text())["final_output"] == "second task"
