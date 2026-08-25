"""?12: production wiring of the TL complement provider at the composition root.

``attach_complement_provider`` existed since the TL step, but no production
code called it -- during live runs the editor's ``list_complementary_parents``
tool would honestly report itself unavailable forever. These tests pin:

1. **Behavioural**: ``wire_editor_complements`` attaches a factory whose
   provider reads the runner's LIVE signed-mechanism state at call time
   (never a frozen snapshot) and returns the core payload contract
   (status + members, never an exception).
2. **Structural**: the pipeline composition root actually CALLS the wiring in
   the same function that constructs the real ``CugaEditorAgent`` -- an AST
   pin so a future refactor cannot silently drop the attach again.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from agent_evolve.adapters.cuga_editor import CugaEditorAgent  # noqa: E402
from agent_evolve.pipeline import wire_editor_complements  # noqa: E402


class _RecordingRunner:
    """Duck-typed stand-in exposing exactly the surface the wiring closes over."""

    def __init__(self):
        self.index_calls = 0

    def signed_mechanism_index(self):
        self.index_calls += 1

        class _Empty:
            entries = ()

        return _Empty()

    cluster_registry = object()


def _editor_agent() -> CugaEditorAgent:
    agent = CugaEditorAgent.__new__(CugaEditorAgent)  # skip SDK init
    agent.complement_provider_factory = None
    return agent


def _request():
    return SimpleNamespace(
        task=SimpleNamespace(task_id="task-live-1"),
        analysis=None,
    )


def test_wire_editor_complements_attaches_live_factory():
    runner = _RecordingRunner()
    editor = _editor_agent()
    assert editor.complement_provider_factory is None

    wire_editor_complements(runner, editor)

    assert editor.complement_provider_factory is not None
    # Wiring must NOT read the index eagerly -- zero calls until a provider is built.
    assert runner.index_calls == 0

    request = _request()
    provider = editor.complement_provider_factory(request)
    assert runner.index_calls == 0, "building the factory must stay lazy"

    result = provider()
    assert runner.index_calls == 1, "each tool call must read live runner state"
    assert result["status"] == "unclustered"
    assert result["members"] == []


def _pipeline_functions(tree: ast.AST) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def test_pipeline_composition_root_calls_the_wiring():
    source = (ROOT / "src" / "agent_evolve" / "pipeline.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = _pipeline_functions(tree)

    builders = [
        fn
        for fn in functions.values()
        for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", "") == "CugaEditorAgent"
    ]
    assert builders, "no function in pipeline.py constructs CugaEditorAgent"

    wired = [
        fn
        for fn in functions.values()
        for call in ast.walk(fn)
        if isinstance(call, ast.Call)
        and getattr(call.func, "id", "") == "wire_editor_complements"
    ]
    assert wired, (
        "wire_editor_complements is never called in pipeline.py -- the editor "
        "tool would be permanently unavailable in production (?12 regression)"
    )
    # The wiring must live where the editor is born, not in some unrelated helper.
    builder_names = {id(fn) for fn in builders}
    assert any(id(fn) in builder_names for fn in wired), (
        "wire_editor_complements exists but is not called by the composition "
        "root that constructs CugaEditorAgent"
    )
