from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import cast


RUNNER_PATH = Path(__file__).resolve().parents[1] / "scripts" / "inference_run.py"
SPEC = importlib.util.spec_from_file_location("inference_run", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
CONFIG = cast(object, RUNNER)


def test_mock_inference_writes_json_trace(tmp_path, capsys):
    output_path = tmp_path / "trace.json"
    setattr(CONFIG, "OUTPUT_PATH", output_path)
    setattr(CONFIG, "TASK_ID", "demo-1")
    setattr(CONFIG, "TASK_INPUT", "hello")
    setattr(CONFIG, "USE_MOCK_RUNTIME", True)

    exit_code = RUNNER.main()

    assert exit_code == 0
    assert json.loads(output_path.read_text()) == {
        "task_id": "demo-1",
        "status": "success",
        "model": "mock-model",
        "final_output": "hello",
        "events": [
            {"event_id": "demo-1:started", "kind": "run_started"},
            {"event_id": "demo-1:completed", "kind": "run_completed"},
        ],
    }
    assert str(output_path) in capsys.readouterr().out


def test_live_inference_loads_dotenv_before_reading_runtime_settings(monkeypatch):
    calls = []

    class StubWrapper:
        def run_task(self, task_id, harness_config):
            return {
                "task_id": task_id,
                "status": "success",
                "model": "test-model",
                "final_output": "answer",
                "events": [],
            }

    monkeypatch.setattr(RUNNER, "load_dotenv", lambda: calls.append("dotenv"))
    monkeypatch.setattr(RUNNER.RuntimeSettings, "from_env", lambda: object())
    monkeypatch.setattr(RUNNER.CugaWrapper, "from_cuga", lambda settings: StubWrapper())
    setattr(CONFIG, "USE_MOCK_RUNTIME", False)

    assert RUNNER.main() == 0
    assert calls == ["dotenv"]
