"""Editor-call budget cap (spec §12).

Each editor call becomes 10-40 internal LLM calls, so the invocation count
must be capable of being capped. Default stays None (uncapped) so existing
profiles and tests are unaffected.
"""
from __future__ import annotations

import pytest

from agent_evolve.core.config import BudgetLimits, BudgetUsage, resolve_profile
from agent_evolve.core.errors import BudgetExceededError


def test_editor_calls_budget_refuses_operation_above_limit() -> None:
    limits = BudgetLimits(max_editor_calls=2)
    usage = BudgetUsage(editor_calls=2)
    with pytest.raises(BudgetExceededError):
        usage.reserve(limits, editor_calls=1)


def test_editor_calls_budget_allows_operation_within_limit() -> None:
    limits = BudgetLimits(max_editor_calls=2)
    usage = BudgetUsage(editor_calls=1)
    usage.reserve(limits, editor_calls=1)
    assert usage.editor_calls == 2


def test_editor_calls_uncapped_by_default() -> None:
    limits = BudgetLimits()
    usage = BudgetUsage()
    assert limits.max_editor_calls is None
    usage.reserve(limits, editor_calls=1000)
    assert usage.editor_calls == 1000


def test_manifest_payload_exposes_editor_call_cap() -> None:
    config = resolve_profile("research_sequential", environ={})
    assert "max_editor_calls" in config.manifest_payload()["budgets"]
