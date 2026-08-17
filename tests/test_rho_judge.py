"""Tests for the RHO difficulty + abstract-fingerprint judge (Interface A).

Every test injects a fake ``completion_fn``; no test makes a network call.

The judge consumes the *comprehended summary text* as a plain ``str`` so it does
not import the comprehender module. Ground truth is an explicit opt-in
parameter: the user has overridden the repo-wide no-labels rule for this judge,
so containment is enforced by prompt instruction *and* by an output check, not
by withholding the answer.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from agent_evolve.adapters.cuga_rho_judge import (
    JUDGE_SYSTEM_PROMPT,
    DifficultyVerdict,
    JudgeConfigurationError,
    RhoDifficultyJudge,
)
from agent_evolve.core.rho.cache import JsonDiskCache
from agent_evolve.core.rho.history import HistoricalRecord

ANSWER_KEY = "Zq7Denver-Intl-4417"

SUMMARY_TEXT = (
    "what_was_attempted: count albums\n"
    "approach_taken: narrated a plan\n"
    "where_it_went_wrong: never executed a tool\n"
    "outcome: no_committed_answer"
)


def _record(task_id: str = "gaia-1", content_hash: str = "sha256:abc") -> HistoricalRecord:
    return HistoricalRecord(
        task_id=task_id,
        input_text="how many albums",
        trace_path="/traces/x/causal-trace.json",
        raw_trace={"events": []},
        final_output="unknown",
        tool_observation_count=0,
        harness_version="vanilla",
        content_hash=content_hash,
    )


def _response(payload: dict) -> dict:
    return {"choices": [{"message": {"content": json.dumps(payload)}}]}


def _ok_fake(calls: list[dict] | None = None, **payload: object):
    body = {"difficulty": 5.0, "abstract_fingerprint": "x", **payload}

    def fake(**request: object) -> dict:
        if calls is not None:
            calls.append(request)
        return _response(body)

    return fake


# --------------------------------------------------------------------- #
# Happy path and schema validation
# --------------------------------------------------------------------- #
def test_parses_a_valid_verdict() -> None:
    def fake(**request: object) -> dict:
        return _response(
            {
                "difficulty": 7.8,
                "abstract_fingerprint": (
                    "A multi-step retrieval task where the agent plans in prose "
                    "but never executes a retrieval step."
                ),
            }
        )

    verdict = RhoDifficultyJudge(completion_fn=fake, model="m").judge(
        _record(), SUMMARY_TEXT
    )

    assert isinstance(verdict, DifficultyVerdict)
    assert verdict.observed is True
    assert verdict.task_id == "gaia-1"
    assert verdict.difficulty == 7.8
    assert "retrieval" in verdict.abstract_fingerprint
    assert verdict.error == ""


def test_integer_difficulty_is_coerced_to_float() -> None:
    verdict = RhoDifficultyJudge(
        completion_fn=_ok_fake(difficulty=8), model="m"
    ).judge(_record(), SUMMARY_TEXT)

    assert verdict.observed is True
    assert isinstance(verdict.difficulty, float)
    assert verdict.difficulty == 8.0


def test_out_of_range_difficulty_is_rejected() -> None:
    verdict = RhoDifficultyJudge(
        completion_fn=_ok_fake(difficulty=11.0), model="m"
    ).judge(_record(), SUMMARY_TEXT)

    assert verdict.observed is False
    assert "difficulty" in verdict.error


def test_negative_difficulty_is_rejected() -> None:
    verdict = RhoDifficultyJudge(
        completion_fn=_ok_fake(difficulty=-1.0), model="m"
    ).judge(_record(), SUMMARY_TEXT)

    assert verdict.observed is False


def test_non_numeric_difficulty_is_rejected() -> None:
    verdict = RhoDifficultyJudge(
        completion_fn=_ok_fake(difficulty="quite hard"), model="m"
    ).judge(_record(), SUMMARY_TEXT)

    assert verdict.observed is False
    assert "difficulty" in verdict.error


def test_missing_fingerprint_is_rejected() -> None:
    def fake(**request: object) -> dict:
        return _response({"difficulty": 5.0})

    verdict = RhoDifficultyJudge(completion_fn=fake, model="m").judge(
        _record(), SUMMARY_TEXT
    )

    assert verdict.observed is False
    assert "fingerprint" in verdict.error


def test_blank_fingerprint_is_rejected() -> None:
    verdict = RhoDifficultyJudge(
        completion_fn=_ok_fake(abstract_fingerprint="   "), model="m"
    ).judge(_record(), SUMMARY_TEXT)

    assert verdict.observed is False
    assert "fingerprint" in verdict.error


def test_fenced_json_is_parsed() -> None:
    def fake(**request: object) -> dict:
        body = json.dumps({"difficulty": 6.0, "abstract_fingerprint": "shape"})
        return {
            "choices": [
                {"message": {"content": f"Sure.\n```json\n{body}\n```\n"}}
            ]
        }

    verdict = RhoDifficultyJudge(completion_fn=fake, model="m").judge(
        _record(), SUMMARY_TEXT
    )

    assert verdict.observed is True
    assert verdict.difficulty == 6.0


def test_non_object_json_is_rejected() -> None:
    def fake(**request: object) -> dict:
        return {"choices": [{"message": {"content": "[1, 2, 3]"}}]}

    verdict = RhoDifficultyJudge(completion_fn=fake, model="m").judge(
        _record(), SUMMARY_TEXT
    )

    assert verdict.observed is False
    assert verdict.error


def test_empty_response_body_is_rejected() -> None:
    def fake(**request: object) -> dict:
        return {"choices": [{"message": {"content": ""}}]}

    verdict = RhoDifficultyJudge(completion_fn=fake, model="m").judge(
        _record(), SUMMARY_TEXT
    )

    assert verdict.observed is False
    assert "empty" in verdict.error


# --------------------------------------------------------------------- #
# Failures are observable outcomes, never silent success
# --------------------------------------------------------------------- #
def test_transport_failure_becomes_an_unobserved_verdict() -> None:
    def fake(**request: object) -> dict:
        raise RuntimeError("endpoint down")

    verdict = RhoDifficultyJudge(completion_fn=fake, model="m").judge(
        _record(), SUMMARY_TEXT
    )

    assert verdict.observed is False
    assert verdict.difficulty == 0.0
    assert "RuntimeError" in verdict.error
    assert "endpoint down" in verdict.error


def test_empty_summary_is_refused_without_a_model_call() -> None:
    calls: list[dict] = []

    verdict = RhoDifficultyJudge(completion_fn=_ok_fake(calls), model="m").judge(
        _record(), "   "
    )

    assert calls == []
    assert verdict.observed is False
    assert "summary" in verdict.error


def test_unresolvable_model_for_a_live_call_raises(monkeypatch) -> None:
    import agent_evolve.adapters.cuga_rho_judge as mod

    monkeypatch.setattr(mod, "_env_settings", lambda: (None, None, None))

    with pytest.raises(JudgeConfigurationError):
        RhoDifficultyJudge().judge(_record(), SUMMARY_TEXT)


# --------------------------------------------------------------------- #
# Ground-truth containment
# --------------------------------------------------------------------- #
def test_prompt_never_contains_the_expected_answer_by_default() -> None:
    calls: list[dict] = []

    record = dataclasses.replace(
        _record(), final_output=f"the answer is {ANSWER_KEY}"
    )
    # The judge sees the summary, not the raw final output.
    RhoDifficultyJudge(completion_fn=_ok_fake(calls), model="m").judge(
        record, SUMMARY_TEXT
    )

    rendered = json.dumps(calls[0], default=str)
    assert ANSWER_KEY not in rendered


def test_expected_answer_is_forwarded_only_when_explicitly_supplied() -> None:
    calls: list[dict] = []

    RhoDifficultyJudge(completion_fn=_ok_fake(calls), model="m").judge(
        _record(), SUMMARY_TEXT, expected_answer=ANSWER_KEY
    )

    rendered = json.dumps(calls[0], default=str)
    assert ANSWER_KEY in rendered
    assert "must not appear" in rendered


def test_fingerprint_that_leaks_the_expected_answer_is_rejected() -> None:
    verdict = RhoDifficultyJudge(
        completion_fn=_ok_fake(
            abstract_fingerprint=f"single-hop lookup for {ANSWER_KEY}"
        ),
        model="m",
    ).judge(_record(), SUMMARY_TEXT, expected_answer=ANSWER_KEY)

    assert verdict.observed is False
    assert "fingerprint" in verdict.error


def test_leak_check_is_case_insensitive() -> None:
    verdict = RhoDifficultyJudge(
        completion_fn=_ok_fake(abstract_fingerprint=f"shape {ANSWER_KEY.lower()}"),
        model="m",
    ).judge(_record(), SUMMARY_TEXT, expected_answer=ANSWER_KEY)

    assert verdict.observed is False


# --------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------- #
def test_cache_hit_skips_the_model_call(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    calls: list[dict] = []

    judge = RhoDifficultyJudge(
        completion_fn=_ok_fake(calls, difficulty=6.5, abstract_fingerprint="y"),
        model="m",
        cache=cache,
    )
    judge.judge(_record(), SUMMARY_TEXT)
    second = judge.judge(_record(), SUMMARY_TEXT)

    assert len(calls) == 1
    assert second.difficulty == 6.5
    assert second.observed is True
    assert cache.hits == 1


def test_a_changed_trace_hash_misses_the_cache(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    calls: list[dict] = []
    judge = RhoDifficultyJudge(
        completion_fn=_ok_fake(calls), model="m", cache=cache
    )

    judge.judge(_record(content_hash="sha256:aaa"), SUMMARY_TEXT)
    judge.judge(_record(content_hash="sha256:bbb"), SUMMARY_TEXT)

    assert len(calls) == 2


def test_ground_truth_calibrated_verdicts_use_a_separate_cache_entry(
    tmp_path: Path,
) -> None:
    cache = JsonDiskCache(tmp_path)
    calls: list[dict] = []
    judge = RhoDifficultyJudge(
        completion_fn=_ok_fake(calls), model="m", cache=cache
    )

    judge.judge(_record(), SUMMARY_TEXT)
    judge.judge(_record(), SUMMARY_TEXT, expected_answer=ANSWER_KEY)

    assert len(calls) == 2


def test_cached_payloads_are_revalidated(tmp_path: Path) -> None:
    cache = JsonDiskCache(tmp_path)
    calls: list[dict] = []
    judge = RhoDifficultyJudge(
        completion_fn=_ok_fake(calls), model="m", cache=cache
    )
    record = _record()
    cache.put(
        judge.cache_key(record),
        {"difficulty": 99.0, "abstract_fingerprint": "poisoned"},
    )

    verdict = judge.judge(record, SUMMARY_TEXT)

    assert calls == []
    assert verdict.observed is False
    assert "difficulty" in verdict.error


def test_cache_is_disabled_by_default() -> None:
    calls: list[dict] = []
    judge = RhoDifficultyJudge(completion_fn=_ok_fake(calls), model="m")

    judge.judge(_record(), SUMMARY_TEXT)
    judge.judge(_record(), SUMMARY_TEXT)

    assert len(calls) == 2


# --------------------------------------------------------------------- #
# Request shape
# --------------------------------------------------------------------- #
def test_temperature_is_omitted_by_default() -> None:
    calls: list[dict] = []

    RhoDifficultyJudge(completion_fn=_ok_fake(calls), model="m").judge(
        _record(), SUMMARY_TEXT
    )

    assert "temperature" not in calls[0]


def test_zero_temperature_is_refused() -> None:
    with pytest.raises(ValueError, match="0.0"):
        RhoDifficultyJudge(
            completion_fn=_ok_fake(), model="m", temperature=0.0
        ).judge(_record(), SUMMARY_TEXT)


def test_non_zero_temperature_is_forwarded() -> None:
    calls: list[dict] = []

    RhoDifficultyJudge(
        completion_fn=_ok_fake(calls), model="m", temperature=0.4
    ).judge(_record(), SUMMARY_TEXT)

    assert calls[0]["temperature"] == 0.4


def test_connection_settings_are_forwarded() -> None:
    calls: list[dict] = []

    RhoDifficultyJudge(
        completion_fn=_ok_fake(calls),
        model="m",
        base_url="https://example.invalid",
        api_key="k",
    ).judge(_record(), SUMMARY_TEXT)

    assert calls[0]["model"] == "m"
    assert calls[0]["api_base"] == "https://example.invalid"
    assert calls[0]["api_key"] == "k"


def test_prompt_carries_the_summary_and_the_task_question() -> None:
    calls: list[dict] = []

    RhoDifficultyJudge(completion_fn=_ok_fake(calls), model="m").judge(
        _record(), SUMMARY_TEXT
    )

    user = calls[0]["messages"][1]["content"]  # type: ignore[index]
    assert "never executed a tool" in user
    assert "how many albums" in user


# --------------------------------------------------------------------- #
# Prompt quality is the product here: pin the discriminative scaffold
# --------------------------------------------------------------------- #
def test_system_prompt_pins_an_anchored_difficulty_scale() -> None:
    prompt = JUDGE_SYSTEM_PROMPT
    # Anchored bands, not a bare "0 easy 10 hard".
    for anchor in ("0-1", "2-3", "4-5", "6-7", "8-9"):
        assert anchor in prompt
    # The failure-implies-maximum shortcut is the main degeneracy risk.
    assert "not evidence of" in prompt
    assert "one decimal" in prompt


def test_system_prompt_pins_the_structural_fingerprint_vocabulary() -> None:
    prompt = JUDGE_SYSTEM_PROMPT.lower()
    for axis in ("task shape", "binding constraint", "failure locus"):
        assert axis in prompt
    for shape in (
        "single-hop lookup",
        "multi-hop composition",
        "aggregation over a set",
        "constraint satisfaction",
    ):
        assert shape in prompt
    for locus in (
        "never acted",
        "wrong tool",
        "retrieved but misread",
        "looped without progress",
        "committed without verifying",
    ):
        assert locus in prompt


def test_system_prompt_forbids_task_specific_identifiers() -> None:
    prompt = JUDGE_SYSTEM_PROMPT.lower()
    assert "never name" in prompt
    for banned in ("file", "person", "place", "date"):
        assert banned in prompt
