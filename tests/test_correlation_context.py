"""SV-7 step 1: ambient run-correlation for LLM call capture.

The observability proxy (``docker/observability/``) records every LLM call, but
mitmproxy sees only bytes on a socket -- it cannot know which candidate, task or
rollout produced a call. The addon lifts ``X-AE-*`` request headers into the
capture record and strips them before the request goes upstream, so no vendor
ever receives internal experiment identifiers
(``docker/observability/addons/correlate.py:139-152``).

Verified end-to-end against the running proxy before these tests were written: a
call carrying the five headers produced the capture record
``{"run": ..., "candidate": ..., "task": ..., "rollout": ..., "phase": ...}``
with ``X-AE-*`` absent from the forwarded request and ``Authorization``
redacted. What was missing is the other half: **nothing in ``src/`` ever sent
those headers**, so every capture to date has been uncorrelated. A grep for
``X-AE-`` across ``src/`` and ``scripts/`` returned no matches.

Why ambient rather than threaded through call signatures: the four
``_litellm_completion`` wrappers sit at the bottom of long call chains
(orchestrator -> issue selection -> analyzer -> adapter), and the correlation
facts are known only at the top. Threading five parameters through every
intermediate signature would touch far more code than the problem warrants and
would still miss any path that forgets to forward them. A ``contextvars``
context is set once at the top and read at the bottom.

``contextvars``, not a module global, because :data:`ResolvedConfig` allows
parallel execution: a global would let one worker's candidate id label another
worker's calls, which is precisely the mislabelling this exists to prevent.

This module is agent-neutral: it must never import ``litellm`` or any adapter.
It only *builds* the header mapping; the adapters attach it.
"""
from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve.core.correlation import (  # noqa: E402
    CorrelationContext,
    correlation_headers,
    current_correlation,
    correlation_scope,
)


def test_no_context_yields_no_headers() -> None:
    """Absent a scope, calls must be unlabelled rather than mislabelled.

    Inventing a placeholder candidate id would put a *wrong* correlation into
    the audit trail, which is worse than an absent one.
    """
    assert current_correlation() is None
    assert correlation_headers() == {}


def test_scope_produces_the_exact_header_names_the_addon_reads() -> None:
    """The addon reads ``X-AE-{key.capitalize()}`` for five specific keys.

    Pinned literally: a header the addon does not recognise is silently dropped,
    so a typo here produces uncorrelated captures with no error anywhere.
    """
    with correlation_scope(
        run="run-1", candidate="cand-A", task="t1", rollout=3, phase="rollout"
    ):
        assert correlation_headers() == {
            "X-AE-Run": "run-1",
            "X-AE-Candidate": "cand-A",
            "X-AE-Task": "t1",
            "X-AE-Rollout": "3",
            "X-AE-Phase": "rollout",
        }


def test_headers_are_all_strings() -> None:
    """HTTP header values must be strings; ``rollout`` is naturally an int."""
    with correlation_scope(run="r", candidate="c", task="t", rollout=0, phase="p"):
        for key, value in correlation_headers().items():
            assert isinstance(value, str), f"{key} is {type(value).__name__}"


def test_partial_context_omits_absent_keys_rather_than_sending_blanks() -> None:
    """An empty header value is indistinguishable from a real empty id.

    Omitting the key leaves the capture record honestly silent on that field.
    """
    with correlation_scope(run="run-1", phase="analysis"):
        headers = correlation_headers()
    assert headers == {"X-AE-Run": "run-1", "X-AE-Phase": "analysis"}
    assert "X-AE-Candidate" not in headers


def test_scope_restores_the_previous_context_on_exit() -> None:
    """Nested phases must not leak outward."""
    with correlation_scope(run="r", candidate="outer", phase="rollout"):
        assert current_correlation().candidate == "outer"
        with correlation_scope(run="r", candidate="inner", phase="analysis"):
            assert current_correlation().candidate == "inner"
            assert current_correlation().phase == "analysis"
        assert current_correlation().candidate == "outer"
        assert current_correlation().phase == "rollout"
    assert current_correlation() is None


def test_scope_restores_context_even_when_the_body_raises() -> None:
    """A failed rollout must not leave its label attached to later calls."""
    with pytest.raises(RuntimeError):
        with correlation_scope(run="r", candidate="cand-A"):
            raise RuntimeError("rollout failed")
    assert current_correlation() is None


def test_parallel_workers_do_not_see_each_others_correlation() -> None:
    """The reason this is a contextvar and not a module global.

    ``parallel_execution`` is a supported feature gate. With a global, one
    worker's candidate id would label another worker's calls -- silently
    attributing evidence to the wrong candidate, which is unrecoverable after
    the fact.
    """

    def work(candidate: str) -> str:
        with correlation_scope(run="r", candidate=candidate, phase="rollout"):
            return correlation_headers()["X-AE-Candidate"]

    with ThreadPoolExecutor(max_workers=4) as pool:
        seen = list(pool.map(work, ["c0", "c1", "c2", "c3"]))
    assert seen == ["c0", "c1", "c2", "c3"]


def test_context_is_immutable() -> None:
    """A correlation label must not be editable after the scope is entered."""
    with correlation_scope(run="r", candidate="cand-A"):
        ctx = current_correlation()
        with pytest.raises(Exception):
            ctx.candidate = "cand-B"  # type: ignore[misc]


def test_core_correlation_imports_no_adapter_or_provider() -> None:
    """``core/`` is agent-neutral; this module builds headers, nothing more."""
    import ast

    src = Path(_ROOT / "src/agent_evolve/core/correlation.py").read_text()
    forbidden = {"litellm", "openai", "httpx", "requests", "cuga"}
    for node in ast.walk(ast.parse(src)):
        mods: list[str] = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            root = m.split(".")[0]
            assert root not in forbidden, f"core imports {m}"
            assert not m.startswith("agent_evolve.adapters"), f"core imports {m}"


def test_context_can_be_built_directly_and_rendered() -> None:
    """The dataclass is usable without the scope, for callers that store one."""
    ctx = CorrelationContext(
        run="r", candidate="c", task="t", rollout=7, phase="validate"
    )
    assert ctx.headers()["X-AE-Rollout"] == "7"
    assert ctx.headers()["X-AE-Phase"] == "validate"
