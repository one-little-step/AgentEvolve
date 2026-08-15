"""The editing surface must import and run without the CUGA SDK (qf31).

A blocker that only overrides the removed ``find_module`` hook is silently
ignored on Python 3.12+, so this uses ``find_spec`` and asserts the blocker
itself works before trusting any result from it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_PROBE = '''
import sys
from importlib.abc import MetaPathFinder

BLOCKED = ("cuga", "langchain_core", "langchain")


class Blocker(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked by probe: " + name)
        return None


for name in list(sys.modules):
    if name.split(".")[0] in BLOCKED:
        del sys.modules[name]
sys.meta_path.insert(0, Blocker())

try:
    import cuga  # noqa: F401
except ImportError:
    pass
else:
    print("BLOCKER_INERT")
    raise SystemExit(1)

for module in (
    "cuga_editor_state",
    "cuga_editor_evidence",
    "cuga_editor_skills",
    "cuga_editor_tools",
):
    __import__("agent_evolve.adapters." + module)

import agent_evolve.adapters.cuga_editor_tools as tools

assert hasattr(tools, "build_tool_callables")
try:
    tools.build_editor_tools(None)
except ImportError:
    print("OK")
else:
    print("SDK_NOT_REQUIRED")
    raise SystemExit(1)
'''


def test_editing_surface_imports_without_the_cuga_sdk() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        cwd=root,
    )
    assert "BLOCKER_INERT" not in result.stdout, "import blocker did not work"
    assert "SDK_NOT_REQUIRED" not in result.stdout
    assert "OK" in result.stdout, result.stdout + result.stderr
