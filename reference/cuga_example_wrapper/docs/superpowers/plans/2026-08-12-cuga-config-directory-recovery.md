# CUGA Configuration Directory Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent blank CUGA configuration-directory values from stopping batch workers before trajectories are created.

**Architecture:** Normalize the optional configuration-directory environment variable before importing CUGA. Preserve explicit non-blank custom configuration directories while removing blank values so CUGA falls back to its installed configurations.

**Tech Stack:** Python 3.14, pytest, python-dotenv, CUGA.

## Global Constraints

- Preserve non-blank `CUGA_CONFIGURATIONS_DIR` values.
- A blank or whitespace-only directory must behave as unset.
- Keep the fix limited to CUGA startup configuration.

---

### Task 1: Normalize the CUGA Configuration Directory

**Files:**
- Modify: `run2.py:19-32`
- Modify: `.env:14-16`
- Test: `tests/test_run2_configuration.py`

**Interfaces:**
- Produces: `normalize_cuga_configuration_directory() -> None`
- Consumes: `os.environ["CUGA_CONFIGURATIONS_DIR"]`

- [ ] **Step 1: Write the failing test**

```python
import importlib.util
from pathlib import Path


def load_run2_module():
    path = Path(__file__).resolve().parents[1] / "run2.py"
    spec = importlib.util.spec_from_file_location("run2_configuration", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_blank_configuration_directory_is_removed(monkeypatch):
    monkeypatch.setenv("CUGA_CONFIGURATIONS_DIR", "  ")
    run2 = load_run2_module()

    run2.normalize_cuga_configuration_directory()

    assert "CUGA_CONFIGURATIONS_DIR" not in run2.os.environ
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_run2_configuration.py::test_blank_configuration_directory_is_removed -v`

Expected: FAIL because `normalize_cuga_configuration_directory` does not exist.

- [ ] **Step 3: Write minimal implementation**

```python
def normalize_cuga_configuration_directory() -> None:
    value = os.getenv("CUGA_CONFIGURATIONS_DIR")
    if value is not None and not value.strip():
        os.environ.pop("CUGA_CONFIGURATIONS_DIR", None)
```

Call it immediately after `load_dotenv(ROOT / ".env")` and before importing
`cuga`. Remove `CUGA_CONFIGURATIONS_DIR=` from `.env`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_run2_configuration.py::test_blank_configuration_directory_is_removed -v`

Expected: PASS.

- [ ] **Step 5: Verify CUGA startup and batch trajectories**

Run: `uv run dataset/batch_run_cuga.py`

Expected: task results have non-null `run_id` values and each successful task
directory contains `cuga_trace.json`.
