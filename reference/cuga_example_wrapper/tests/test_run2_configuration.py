import ast
import os
from pathlib import Path
from typing import Any, Callable


def load_configuration_helpers() -> dict[str, Any]:
    source = (Path(__file__).resolve().parents[1] / "run2.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    imports = [
        node
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            isinstance(node, ast.Import)
            and any(alias.name == "os" for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and node.module == "typing"
        )
    ]
    helpers = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "normalize_cuga_configuration_directory"
    ]
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=[*imports, *helpers], type_ignores=[]), "run2.py", "exec"),
        namespace,
    )
    return namespace


def load_run_helpers(*names: str) -> dict[str, Any]:
    source = (Path(__file__).resolve().parents[1] / "run2.py").read_text(
        encoding="utf-8"
    )
    module = ast.parse(source)
    imports = [
        node
        for node in module.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            isinstance(node, ast.Import)
            and any(alias.name in {"json", "re"} for alias in node.names)
            or isinstance(node, ast.ImportFrom)
            and node.module in {"typing", "urllib.parse"}
        )
    ]
    helpers = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {*names, "json_safe"}
    ]
    for helper in helpers:
        helper.decorator_list = []
    namespace: dict[str, Any] = {}
    exec(
        compile(ast.Module(body=[*imports, *helpers], type_ignores=[]), "run2.py", "exec"),
        namespace,
    )
    return namespace


def test_blank_configuration_directory_is_removed(monkeypatch):
    monkeypatch.setenv("CUGA_CONFIGURATIONS_DIR", "  ")
    helpers = load_configuration_helpers()

    normalizer = helpers["normalize_cuga_configuration_directory"]
    assert callable(normalizer)
    normalizer()

    assert "CUGA_CONFIGURATIONS_DIR" not in os.environ


def test_wikipedia_search_can_encode_a_query():
    helpers = load_run_helpers("wikipedia_search")

    assert "quote_plus" in helpers


def test_stream_result_uses_final_answer_and_tool_calls():
    helpers = load_run_helpers("extract_stream_result")
    streamed = [
        [
            [],
            {
                "FinalAnswerAgent": {
                    "final_answer": "The final answer",
                    "tool_calls": [{"name": "calculator"}],
                    "thread_id": "thread-1",
                }
            },
        ]
    ]

    result = helpers["extract_stream_result"](streamed)

    assert result == {
        "answer": "The final answer",
        "tool_calls": [{"name": "calculator"}],
        "error": None,
        "thread_id": "thread-1",
    }
