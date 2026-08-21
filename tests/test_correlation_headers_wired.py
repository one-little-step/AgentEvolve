"""SV-7 step 2: the four adapter wrappers must send the correlation headers.

``core/correlation.py`` builds the ``X-AE-*`` mapping; these tests pin that every
live-call wrapper actually attaches it. Verified against the running proxy: the
capture side works and strips the headers before upstream, so the only missing
half was emission -- a grep for ``X-AE-`` across ``src/`` and ``scripts/``
returned nothing, meaning every capture to date has been uncorrelated and could
not answer SV-7.

There are exactly four wrappers, all structurally identical:

* ``cuga_analyzer._litellm_completion``
* ``cuga_mechanism_adjudicator._litellm_completion``
* ``cuga_rho_comprehender._litellm_completion``
* ``cuga_rho_judge._litellm_completion``

Tested through the wrapper rather than by asserting on source text: a
substring assertion would pass on a commented-out call. ``litellm`` is patched at
``sys.modules`` so no provider is reached and no credential is reqired.

The four are enumerated dynamically, so a fifth wrapper added later without
correlation is a test failure rather than a silent gap.
"""
from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve.core.correlation import correlation_scope  # noqa: E402

WRAPPER_MODULES = (
    "agent_evolve.adapters.cuga_analyzer",
    "agent_evolve.adapters.cuga_mechanism_adjudicator",
    "agent_evolve.adapters.cuga_rho_comprehender",
    "agent_evolve.adapters.cuga_rho_judge",
)


class _FakeLitellm(types.ModuleType):
    """Stands in for ``litellm`` so the wrapper can be exercised offline."""

    def __init__(self) -> None:
        super().__init__("litellm")
        self.seen: list[dict[str, object]] = []

    def completion(self, **request: object) -> object:
        self.seen.append(request)
        return {"ok": True}


@pytest.fixture()
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> _FakeLitellm:
    fake = _FakeLitellm()
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


def _call(module_name: str, **kwargs: object) -> None:
    mod = importlib.import_module(module_name)
    mod._litellm_completion(model="m", messages=[], **kwargs)


@pytest.mark.parametrize("module_name", WRAPPER_MODULES)
def test_wrapper_attaches_correlation_headers(
    module_name: str, fake_litellm: _FakeLitellm
) -> None:
    """Inside a scope, the wrapper must pass the headers to the provider call."""
    with correlation_scope(
        run="run-1", candidate="cand-A", task="t1", rollout=2, phase="rollout"
    ):
        _call(module_name)

    assert len(fake_litellm.seen) == 1
    headers = fake_litellm.seen[0].get("extra_headers")
    assert headers == {
        "X-AE-Run": "run-1",
        "X-AE-Candidate": "cand-A",
        "X-AE-Task": "t1",
        "X-AE-Rollout": "2",
        "X-AE-Phase": "rollout",
    }, f"{module_name} did not attach correlation headers: {headers!r}"


@pytest.mark.parametrize("module_name", WRAPPER_MODULES)
def test_wrapper_sends_no_header_key_outside_a_scope(
    module_name: str, fake_litellm: _FakeLitellm
) -> None:
    """Uncorrelated calls must stay clean.

    Sending ``extra_headers={}`` would be harmless but noise; more importantly,
    the wrapper must not invent placeholder identifiers, which would put a wrong
    correlation into the audit trail.
    """
    _call(module_name)
    request = fake_litellm.seen[0]
    assert request.get("extra_headers") in (None, {})
    assert not any(str(k).lower().startswith("x-ae-") for k in request)


@pytest.mark.parametrize("module_name", WRAPPER_MODULES)
def test_wrapper_preserves_caller_supplied_extra_headers(
    module_name: str, fake_litellm: _FakeLitellm
) -> None:
    """Correlation must merge into, never replace, existing headers.

    A wrapper that assigns ``extra_headers`` outright would silently drop a
    caller's own header -- the kind of regression that shows up only in
    production.
    """
    with correlation_scope(run="run-1", candidate="cand-A"):
        _call(module_name, extra_headers={"X-Custom": "keep-me"})

    headers = fake_litellm.seen[0]["extra_headers"]
    assert isinstance(headers, dict)
    assert headers["X-Custom"] == "keep-me"
    assert headers["X-AE-Candidate"] == "cand-A"
    assert headers["X-AE-Run"] == "run-1"


@pytest.mark.parametrize("module_name", WRAPPER_MODULES)
def test_wrapper_does_not_mutate_the_callers_header_dict(
    module_name: str, fake_litellm: _FakeLitellm
) -> None:
    """Merging must not write into the dict the caller still holds."""
    caller_headers: dict[str, str] = {"X-Custom": "keep-me"}
    with correlation_scope(run="run-1", candidate="cand-A"):
        _call(module_name, extra_headers=caller_headers)

    assert caller_headers == {"X-Custom": "keep-me"}


def test_every_litellm_wrapper_in_adapters_is_covered() -> None:
    """A fifth wrapper added without correlation must fail, not slip through.

    Guards the enumeration itself: the risk is not that these four regress, but
    that a new live-call site is added and nobody wires correlation into it.
    """
    import ast

    adapters = _ROOT / "src/agent_evolve/adapters"
    found: set[str] = set()
    for path in sorted(adapters.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_litellm_completion":
                found.add(f"agent_evolve.adapters.{path.stem}")
    assert found == set(WRAPPER_MODULES), (
        "the set of live-call wrappers changed; wire correlation into any new "
        f"one and add it here. found={sorted(found)}"
    )
