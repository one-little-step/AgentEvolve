"""Behavioral tests for the model-backed mechanism adjudicator.

No network: ``completion_fn`` is injected in every test. What matters here is that
the adjudicator is *conservative* -- an outage, a hedge, or an unparseable answer
must abstain rather than guess, because a wrong "same" merges two unrelated faults
into one measurement bucket and produces a confident but meaningless variance
reading.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve.adapters.cuga_mechanism_adjudicator import (  # noqa: E402
    CugaMechanismAdjudicator,
    DedupConfigurationError,
)
from agent_evolve.core.clustering import (  # noqa: E402
    LexicalEmbedder,
    MechanismAdjudicator,
    MechanismClusterer,
)


def _reply(text: str):
    """A minimal OpenAI-shaped response."""
    return {"choices": [{"message": {"content": text}}]}


def _fn(text: str, calls: list | None = None):
    def _call(**request):
        if calls is not None:
            calls.append(request)
        return _reply(text)

    return _call


# ---------------------------------------------------------------------- #
# Protocol conformance
# ---------------------------------------------------------------------- #
def test_satisfies_the_core_protocol_without_core_importing_it():
    """Structural conformance: injected, never imported by core."""
    adj = CugaMechanismAdjudicator(completion_fn=_fn("same"), model="m")
    assert isinstance(adj, MechanismAdjudicator)


def test_core_clustering_does_not_import_any_adapter_or_provider():
    """``core/clustering.py`` must stay agent-neutral."""
    import ast

    src = (_ROOT / "src/agent_evolve/core/clustering.py").read_text()
    mods: list[str] = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods.extend(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.append(node.module or "")
    forbidden = [
        m
        for m in mods
        if any(k in m for k in ("cuga", "litellm", "adapters", "openai", "httpx"))
    ]
    assert forbidden == [], f"core/clustering.py imports {forbidden}"


# ---------------------------------------------------------------------- #
# Verdict parsing
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "answer,expected",
    [
        ("same", True),
        ("SAME", True),
        ("same.", True),
        ("same — one fault, two phrasings", True),
        ("different", False),
        ("Different.", False),
        ("unsure", None),
        ("", None),
        ("   ", None),
        ("maybe?", None),
        ("I think they are probably the same", None),
        ("yes", None),
        ("true", None),
    ],
)
def test_verdict_parsing_is_strict(answer, expected):
    """Anything that is not one of the three words abstains.

    ``"yes"`` and ``"true"`` deliberately abstain: a model that will not use the
    requested vocabulary has not demonstrably understood the question, and
    guessing at its intent is how a wrong merge gets made silently.
    """
    adj = CugaMechanismAdjudicator(completion_fn=_fn(answer), model="m")
    assert adj.same_mechanism("mechanism a", "mechanism b") is expected


# ---------------------------------------------------------------------- #
# Conservative failure
# ---------------------------------------------------------------------- #
def test_provider_outage_abstains_rather_than_raising():
    """A dedup outage must not take down a run."""

    def _boom(**request):
        raise OSError("connection refused")

    adj = CugaMechanismAdjudicator(completion_fn=_boom, model="m")
    assert adj.same_mechanism("a mechanism", "another mechanism") is None


def test_malformed_response_shape_abstains():
    adj = CugaMechanismAdjudicator(completion_fn=lambda **_: {"nope": 1}, model="m")
    assert adj.same_mechanism("a mechanism", "another mechanism") is None


def test_missing_model_raises_only_when_actually_called():
    """Constructible without credentials; resolved lazily.

    The configuration error surfaces as an abstention to the caller, because
    ``same_mechanism`` never raises -- but the underlying error type exists and is
    reachable directly for a loud preflight.
    """
    adj = CugaMechanismAdjudicator(completion_fn=_fn("same"), model="", base_url="", api_key="")
    assert adj.same_mechanism("a", "b") is None
    with pytest.raises(DedupConfigurationError, match="AE_MECHANISM_DEDUP_MODEL"):
        adj._ask("a", "b")


def test_empty_text_never_reaches_the_model():
    calls: list = []
    adj = CugaMechanismAdjudicator(completion_fn=_fn("same", calls), model="m")
    assert adj.same_mechanism("", "something") is None
    assert adj.same_mechanism("something", "") is None
    assert calls == [], "an empty mechanism description is not worth a model call"


def test_identical_text_never_reaches_the_model():
    calls: list = []
    adj = CugaMechanismAdjudicator(completion_fn=_fn("different", calls), model="m")
    assert adj.same_mechanism("one fault", "one fault") is True
    assert calls == [], "identical text needs no model call"


# ---------------------------------------------------------------------- #
# Cost control
# ---------------------------------------------------------------------- #
def test_repeated_pairs_cost_one_call():
    calls: list = []
    adj = CugaMechanismAdjudicator(completion_fn=_fn("same", calls), model="m")
    for _ in range(5):
        adj.same_mechanism("fault a", "fault b")
    assert len(calls) == 1, f"expected 1 call, made {len(calls)}"


def test_verdicts_are_order_independent():
    """(a, b) and (b, a) must agree, and cost one call between them.

    Order dependence is one of the three measured defects of the cosine-only
    clusterer; the adjudicator must not reintroduce it.
    """
    calls: list = []
    adj = CugaMechanismAdjudicator(completion_fn=_fn("same", calls), model="m")
    first = adj.same_mechanism("fault a", "fault b")
    second = adj.same_mechanism("fault b", "fault a")
    assert first == second
    assert len(calls) == 1


def test_zero_temperature_is_refused():
    """Some endpoints reject ``temperature=0.0`` outright; omit rather than pass."""
    adj = CugaMechanismAdjudicator(
        completion_fn=_fn("same"), model="m", temperature=0.0
    )
    # Surfaces as an abstention to the caller; loud when asked directly.
    assert adj.same_mechanism("a fault", "b fault") is None
    with pytest.raises(ValueError, match="temperature=0.0"):
        adj._ask("a fault", "b fault")


# ---------------------------------------------------------------------- #
# The prompt must leak nothing
# ---------------------------------------------------------------------- #
def test_prompt_carries_only_the_two_mechanism_descriptions():
    """No task id, no expected answer, no grader, no evaluator internals.

    Scoped to the **user** message, which is the only part built from run data.
    The system prompt is a fixed literal in this repo and legitimately contains
    words like "answer" in its instructions ("Answer with exactly one word"), so
    substring-scanning it produces false positives rather than findings.
    """
    calls: list = []
    adj = CugaMechanismAdjudicator(completion_fn=_fn("same", calls), model="m")
    adj.same_mechanism("units unchecked", "no unit verification")
    assert len(calls) == 1
    messages = calls[0]["messages"]
    user = [m for m in messages if m["role"] == "user"]
    assert len(user) == 1, "expected exactly one user message"
    payload = user[0]["content"]
    # Only the two descriptions and the question form.
    assert "units unchecked" in payload
    assert "no unit verification" in payload
    for leak in (
        "expected_regex",
        "expected answer",
        "grader",
        "regex",
        "task-",
        "gaia",
        "verdict_id",
        "trace_id",
    ):
        assert leak not in payload.lower(), (
            f"{leak!r} leaked into the dedup prompt: {payload!r}"
        )


def test_system_prompt_carries_no_evaluator_internals():
    """The fixed instruction text must not name graders, datasets or answers."""
    from agent_evolve.adapters.cuga_mechanism_adjudicator import (
        DEDUP_SYSTEM_PROMPT,
    )

    lowered = DEDUP_SYSTEM_PROMPT.lower()
    for leak in ("expected_regex", "grader", "gaia", "expected answer", "regex"):
        assert leak not in lowered, f"{leak!r} appears in the dedup system prompt"


def test_api_key_is_passed_to_the_provider_but_not_logged_by_us():
    calls: list = []
    adj = CugaMechanismAdjudicator(
        completion_fn=_fn("same", calls),
        model="m",
        base_url="http://x",
        api_key="unit-test-key",
    )
    adj.same_mechanism("a fault", "b fault")
    assert calls[0]["api_key"] == "unit-test-key"
    assert calls[0]["api_base"] == "http://x"


# ---------------------------------------------------------------------- #
# End to end through the clusterer
# ---------------------------------------------------------------------- #
def test_clusterer_uses_the_adapter_to_merge_a_drifted_fault():
    """The measured drift case, fixed through the real adapter."""
    from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis

    def _analysis(mechanism: str) -> CausalAnalysis:
        return CausalAnalysis(
            mechanism=mechanism,
            severity=0.6,
            score=0.2,
            blame_graph=BlameGraph(
                nodes=(
                    BlameNode(actor_id="agent", artifacts=("skills/a.md",), blame=0.9),
                )
            ),
        )

    adj = CugaMechanismAdjudicator(completion_fn=_fn("same"), model="m")
    cl = MechanismClusterer(
        task_id="t",
        embedder=LexicalEmbedder(dim=768),
        adjudicator=adj,
        band_low=0.0,
        band_high=1.0,
    )
    cl.begin_iteration(1)
    ids = [
        cl.assign(_analysis(m)).cluster_id
        for m in (
            "date filter missing",
            "date filter absent",
            "filter for dates omitted",
            "dates unfiltered entirely",
        )
    ]
    assert len(set(ids)) == 1, f"one fault still fragmented: {ids}"
