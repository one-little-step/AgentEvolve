"""Counterfactual proxy validator: cheap A/B over ONE recorded LLM call.

Every test here is offline. The network is replaced by an injected
``completion_fn``, so these tests also pin the *request shape* the live
endpoint requires: one request per arm carrying ``n=k``, and never a
``temperature`` (the reference endpoint rejects any non-default value).
"""
from __future__ import annotations

import re
import threading

import pytest

from agent_evolve.adapters.cuga_proxy_validator import (
    ProxyArmResult,
    ProxySubstitutionError,
    ProxyVerdict,
    artifact_text_substitution,
    calls_tool,
    contains_all,
    matches_regex,
    run_proxy_ab,
)
from agent_evolve.cuga_wrapper import RecordedCall


def _call(system: str = "harness body: always be careful", user: str = "do the task") -> RecordedCall:
    return RecordedCall(
        event_id="ev-1",
        model="openai/azure/test-model",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        baseline_response="recorded baseline",
    )


class _Recorder:
    """Injected completion_fn that returns scripted texts and records requests."""

    def __init__(self, baseline: list[str], edited: list[str], marker: str = "EDITED") -> None:
        self._baseline = baseline
        self._edited = edited
        self._marker = marker
        self.requests: list[dict] = []
        self._lock = threading.Lock()

    def __call__(self, **request: object) -> dict:
        with self._lock:
            self.requests.append(dict(request))
        messages = request["messages"]
        blob = "".join(str(message.get("content", "")) for message in messages)  # type: ignore[union-attr]
        texts = self._edited if self._marker in blob else self._baseline
        return {"choices": [{"message": {"content": text}} for text in texts]}

    @property
    def arm_requests(self) -> tuple[dict, dict]:
        baseline = [r for r in self.requests if self._marker not in _blob(r)]
        edited = [r for r in self.requests if self._marker in _blob(r)]
        return tuple(baseline), tuple(edited)  # type: ignore[return-value]


def _blob(request: dict) -> str:
    return "".join(str(message.get("content", "")) for message in request["messages"])


def _append_marker(text: str = "EDITED: call the search tool first"):
    def substitution(messages: list[dict]) -> list[dict]:
        edited = [dict(message) for message in messages]
        edited[0] = {**edited[0], "content": edited[0]["content"] + "\n" + text}
        return edited

    return substitution


# --------------------------------------------------------------------------
# scoring, delta, labels
# --------------------------------------------------------------------------


def test_both_arms_are_scored_independently() -> None:
    recorder = _Recorder(baseline=["no", "no", "yes"], edited=["yes", "yes", "yes"])

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: "yes" in text,
        predicate_name="says_yes",
        k=3,
        completion_fn=recorder,
    )

    assert verdict.baseline.pass_count == 1
    assert verdict.baseline.k == 3
    assert verdict.baseline.pass_rate == pytest.approx(1 / 3)
    assert verdict.edited.pass_count == 3
    assert verdict.edited.pass_rate == pytest.approx(1.0)
    assert verdict.predicate_name == "says_yes"
    assert verdict.k == 3


def test_label_improved_when_edited_arm_passes_more() -> None:
    recorder = _Recorder(baseline=["no", "no"], edited=["yes", "yes"])

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: "yes" in text,
        predicate_name="says_yes",
        k=2,
        completion_fn=recorder,
    )

    assert verdict.delta == pytest.approx(1.0)
    assert verdict.label == "improved"


def test_label_no_change_when_pass_rates_match() -> None:
    recorder = _Recorder(baseline=["yes", "no"], edited=["no", "yes"])

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: "yes" in text,
        predicate_name="says_yes",
        k=2,
        completion_fn=recorder,
    )

    assert verdict.delta == pytest.approx(0.0)
    assert verdict.label == "no_change"


def test_label_regressed_when_edited_arm_passes_less() -> None:
    recorder = _Recorder(baseline=["yes", "yes"], edited=["nope", "nope"])

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: "yes" in text,
        predicate_name="says_yes",
        k=2,
        completion_fn=recorder,
    )

    assert verdict.delta == pytest.approx(-1.0)
    assert verdict.label == "regressed"


def test_verdict_evidence_kind_is_proxy_and_never_confirmed() -> None:
    recorder = _Recorder(baseline=["no"], edited=["yes"])

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: "yes" in text,
        predicate_name="says_yes",
        k=1,
        completion_fn=recorder,
    )

    assert verdict.evidence_kind == "proxy"
    with pytest.raises(ValueError):
        ProxyVerdict(
            baseline=verdict.baseline,
            edited=verdict.edited,
            predicate_name="says_yes",
            k=1,
            evidence_kind="confirmed",
        )


def test_arm_reports_distinct_completion_count() -> None:
    arm = ProxyArmResult(completions=("a", "a", "b"), pass_count=1, k=3)
    assert arm.distinct_count == 2
    assert arm.pass_rate == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# request shape: ONE request per arm, n=k, never temperature
# --------------------------------------------------------------------------


def test_exactly_one_request_per_arm_carrying_n_equals_k() -> None:
    recorder = _Recorder(baseline=["a", "b", "c"], edited=["d", "e", "f"])

    run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: True,
        predicate_name="always",
        k=3,
        completion_fn=recorder,
    )

    assert len(recorder.requests) == 2, "k sequential requests would be cached, not sampled"
    baseline_requests, edited_requests = recorder.arm_requests
    assert len(baseline_requests) == 1
    assert len(edited_requests) == 1
    for request in recorder.requests:
        assert request["n"] == 3


def test_temperature_is_never_sent() -> None:
    recorder = _Recorder(baseline=["a"], edited=["b"])

    run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: True,
        predicate_name="always",
        k=1,
        completion_fn=recorder,
    )

    for request in recorder.requests:
        assert "temperature" not in request


def test_k_of_one_issues_one_request_per_arm() -> None:
    recorder = _Recorder(baseline=["no"], edited=["yes"])

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: "yes" in text,
        predicate_name="says_yes",
        k=1,
        completion_fn=recorder,
    )

    assert len(recorder.requests) == 2
    # n is omitted for a single sample; the provider default is 1.
    for request in recorder.requests:
        assert request.get("n", 1) == 1
    assert verdict.baseline.pass_rate == pytest.approx(0.0)
    assert verdict.edited.pass_rate == pytest.approx(1.0)


@pytest.mark.parametrize("k", [0, -1])
def test_k_must_be_at_least_one(k: int) -> None:
    with pytest.raises(ValueError):
        run_proxy_ab(
            _call(),
            substitution=_append_marker(),
            predicate=lambda text: True,
            predicate_name="always",
            k=k,
            completion_fn=_Recorder(baseline=["a"], edited=["b"]),
        )


def test_arms_run_in_parallel() -> None:
    """Both arms must be in flight simultaneously: a 2-party barrier proves it."""
    barrier = threading.Barrier(2, timeout=10)

    def completion_fn(**request: object) -> dict:
        barrier.wait()
        return {"choices": [{"message": {"content": "ok"}}]}

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=lambda text: True,
        predicate_name="always",
        k=1,
        completion_fn=completion_fn,
    )

    assert verdict.baseline.pass_count == 1
    assert verdict.edited.pass_count == 1


def test_arm_failure_propagates() -> None:
    def completion_fn(**request: object) -> dict:
        raise RuntimeError("endpoint exploded")

    with pytest.raises(RuntimeError, match="endpoint exploded"):
        run_proxy_ab(
            _call(),
            substitution=_append_marker(),
            predicate=lambda text: True,
            predicate_name="always",
            k=1,
            completion_fn=completion_fn,
        )


# --------------------------------------------------------------------------
# identical-substitution guard
# --------------------------------------------------------------------------


def test_identical_substitution_raises_because_both_arms_would_match() -> None:
    recorder = _Recorder(baseline=["a"], edited=["b"])

    with pytest.raises(ProxySubstitutionError, match="identical"):
        run_proxy_ab(
            _call(),
            substitution=lambda messages: [dict(message) for message in messages],
            predicate=lambda text: True,
            predicate_name="always",
            k=1,
            completion_fn=recorder,
        )
    assert recorder.requests == [], "no request may be issued for a degenerate A/B"


def test_substitution_returning_no_messages_raises() -> None:
    with pytest.raises(ProxySubstitutionError):
        run_proxy_ab(
            _call(),
            substitution=lambda messages: [],
            predicate=lambda text: True,
            predicate_name="always",
            k=1,
            completion_fn=_Recorder(baseline=["a"], edited=["b"]),
        )


# --------------------------------------------------------------------------
# substitution helper: the single most important correctness property
# --------------------------------------------------------------------------


def test_substitution_helper_edits_the_system_message() -> None:
    substitution = artifact_text_substitution("always be careful", "always call search first")
    call = _call()

    edited = substitution(call.messages)

    assert edited[0]["content"] == "harness body: always call search first"
    assert edited[1] == {"role": "user", "content": "do the task"}
    assert call.messages[0]["content"] == "harness body: always be careful", "input must not mutate"


def test_substitution_helper_raises_when_old_text_absent() -> None:
    substitution = artifact_text_substitution("text that is not there", "replacement")

    with pytest.raises(ProxySubstitutionError, match="not found"):
        substitution(_call().messages)


def test_substitution_helper_raises_when_old_text_only_in_untargeted_role() -> None:
    call = _call(system="nothing relevant", user="always be careful")
    substitution = artifact_text_substitution("always be careful", "always call search")

    with pytest.raises(ProxySubstitutionError, match="not found"):
        substitution(call.messages)


def test_substitution_helper_can_target_other_roles() -> None:
    call = _call(system="nothing relevant", user="always be careful")
    substitution = artifact_text_substitution(
        "always be careful", "always call search", roles=("user",)
    )

    edited = substitution(call.messages)

    assert edited[1]["content"] == "always call search"
    assert edited[0]["content"] == "nothing relevant"


def test_substitution_helper_replaces_every_occurrence() -> None:
    call = _call(system="rule X. more text. rule X.")
    substitution = artifact_text_substitution("rule X", "rule Y")

    edited = substitution(call.messages)

    assert edited[0]["content"] == "rule Y. more text. rule Y."


def test_substitution_helper_rejects_a_noop_replacement() -> None:
    with pytest.raises(ValueError, match="identical"):
        artifact_text_substitution("same", "same")


def test_substitution_helper_rejects_empty_old_text() -> None:
    with pytest.raises(ValueError, match="empty"):
        artifact_text_substitution("", "replacement")


def test_substitution_helper_composes_with_run_proxy_ab() -> None:
    recorder = _Recorder(baseline=["no"], edited=["yes"], marker="call search first")
    substitution = artifact_text_substitution("always be careful", "call search first")

    verdict = run_proxy_ab(
        _call(),
        substitution=substitution,
        predicate=lambda text: "yes" in text,
        predicate_name="says_yes",
        k=1,
        completion_fn=recorder,
    )

    assert verdict.label == "improved"


# --------------------------------------------------------------------------
# predicate errors
# --------------------------------------------------------------------------


def test_predicate_error_counts_as_non_pass_and_is_tallied() -> None:
    recorder = _Recorder(baseline=["boom", "yes"], edited=["yes", "yes"])

    def predicate(text: str) -> bool:
        if text == "boom":
            raise ValueError("predicate blew up")
        return "yes" in text

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=predicate,
        predicate_name="says_yes",
        k=2,
        completion_fn=recorder,
    )

    assert verdict.baseline.predicate_errors == 1
    assert verdict.baseline.pass_count == 1
    assert verdict.edited.predicate_errors == 0
    assert verdict.label == "improved"


def test_all_predicate_errors_yields_inconclusive() -> None:
    recorder = _Recorder(baseline=["a", "b"], edited=["c", "d"])

    def predicate(text: str) -> bool:
        raise ValueError("never scorable")

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=predicate,
        predicate_name="broken",
        k=2,
        completion_fn=recorder,
    )

    assert verdict.baseline.predicate_errors == 2
    assert verdict.edited.predicate_errors == 2
    assert verdict.label == "inconclusive"
    assert verdict.delta == pytest.approx(0.0)


# --------------------------------------------------------------------------
# predicate factories
# --------------------------------------------------------------------------


def test_contains_all_requires_every_term() -> None:
    predicate = contains_all(["search", "filter"])

    assert predicate("first SEARCH then Filter") is True
    assert predicate("only search") is False
    assert predicate.name == "contains_all(search,filter)"


def test_contains_all_can_be_case_sensitive() -> None:
    predicate = contains_all(["Search"], case_sensitive=True)

    assert predicate("Search") is True
    assert predicate("search") is False


def test_contains_all_rejects_empty_term_list() -> None:
    with pytest.raises(ValueError):
        contains_all([])


def test_matches_regex() -> None:
    predicate = matches_regex(r"step\s+2")

    assert predicate("go to step  2 now") is True
    assert predicate("go to step three") is False
    assert "matches_regex" in predicate.name


def test_matches_regex_accepts_flags() -> None:
    predicate = matches_regex(r"^done$", flags=re.MULTILINE)
    assert predicate("first\ndone\n") is True


def test_calls_tool_detects_a_python_style_call() -> None:
    predicate = calls_tool("search_docs")

    assert predicate("result = search_docs(query='x')") is True
    assert predicate("result = other_tool(query='x')") is False


def test_calls_tool_detects_a_json_tool_call() -> None:
    predicate = calls_tool("search_docs")

    assert predicate('{"name": "search_docs", "arguments": {}}') is True
    assert predicate('{"tool_name": "search_docs"}') is True
    assert predicate('{"name": "search_docs_v2"}') is False
    assert predicate.name == "calls_tool(search_docs)"


def test_calls_tool_does_not_fire_on_a_bare_mention() -> None:
    predicate = calls_tool("search_docs")

    assert predicate("I could use search_docs but I will not") is False


def test_predicate_name_defaults_to_the_factory_name() -> None:
    recorder = _Recorder(baseline=["no"], edited=["search filter"])

    verdict = run_proxy_ab(
        _call(),
        substitution=_append_marker(),
        predicate=contains_all(["search", "filter"]),
        k=1,
        completion_fn=recorder,
    )

    assert verdict.predicate_name == "contains_all(search,filter)"
