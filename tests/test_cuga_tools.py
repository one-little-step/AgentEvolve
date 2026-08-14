from __future__ import annotations

import types

from agent_evolve.cuga_wrapper import tools


def test_calculator_returns_product():
    assert tools.calculator("17 * 24") == "408"


def test_calculator_returns_structured_error_on_unsupported_expression():
    result = tools.calculator("__import__('os').system('x')")

    assert '"status": "error"' in result
    assert '"message"' in result


def test_web_fetch_returns_structured_error_on_non_http_scheme():
    result = tools.web_fetch("ftp://example.com")

    assert '"status": "error"' in result
    assert '"message"' in result


def test_save_note_writes_file_and_reports_path(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "NOTES_PATH", tmp_path / "notes.txt")

    result = tools.save_note("All 5 tools verified successfully")

    assert (tmp_path / "notes.txt").read_text() == "All 5 tools verified successfully\n"
    assert "notes.txt" in result


def test_build_tools_returns_five_tools_with_expected_names(monkeypatch):
    def fake_tracked_tool(_func=None, *, app_name=None):
        def decorator(func):
            func.__cuga_app_name__ = app_name
            return func

        if _func is not None:
            return decorator(_func)
        return decorator

    cuga = types.ModuleType("cuga")
    cuga.tracked_tool = fake_tracked_tool
    monkeypatch.setitem(__import__("sys").modules, "cuga", cuga)

    built = tools.build_tools()

    names = [tool.name for tool in built]
    assert names == ["calculator", "web_search", "web_fetch", "wikipedia_search", "save_note"]
