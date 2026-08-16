"""LIVE proof of the counterfactual proxy validator against a real recorded call.

Issues REAL model requests. Two requests total (one per arm), each carrying
``n=k``, against the recorded model from
``data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781``.

What is being measured
----------------------
1. That a real recorded prompt can be A/B'd: baseline arm vs an arm whose
   recorded SYSTEM message has one plausible directive injected.
2. **The control that matters:** whether ``n=k`` in a single request actually
   yields DISTINCT completions on a large real prompt (~36.5k system chars),
   not just on a toy probe. Distinct count is reported per arm. If an arm
   reports 1 distinct of k, the pass rate for that arm is one observation
   repeated k times and the delta is not a rate difference.

The A/B
-------
Recorded task: chain ``fetch_alpha_token`` -> ``exchange_alpha_for_beta`` ->
``checksum_beta``. The recorded baseline response commits to only the FIRST step
("I'll retrieve the ALPHA token first ... for the next tool call"). The injected
directive tells the agent to emit one block covering the whole fixed chain. The
primary predicate detects whether the completion commits to the WHOLE chain
rather than one step.

MEASURED LIMITATION of this boundary (found while building this script, reported
rather than hidden). On this endpoint the assistant ``message.content`` for this
prompt carries ONLY the short pre-code note; ``finish_reason`` is ``stop``,
``tool_calls`` is ``None``, and no fenced code block is present in any sampled
completion. The recorded baseline response has the same shape. So a
``calls_tool(...)`` predicate is structurally UNMEASURABLE at this boundary: it
scores 0 for every completion in both arms regardless of the edit, which would
render a "no_change" verdict that says nothing about the edit. This script runs
that predicate too - re-scoring the SAME completions offline, at no extra request
cost - specifically to show the failure mode rather than pretend the tool-call
predicate was validated live.

Usage:
    uv run python scripts/verify_proxy_validator_live.py [TRACE_DIR] [EVENT_ID]
"""
from __future__ import annotations

import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_evolve.adapters.cuga_proxy_validator import (  # noqa: E402
    ProxySubstitutionError,
    artifact_text_substitution,
    calls_tool,
    matches_regex,
    run_proxy_ab,
)
from agent_evolve.cuga_wrapper import (  # noqa: E402
    list_recorded_llm_calls,
    load_recorded_call,
    prepare_cuga_environment,
)

DEFAULT_TRACE_DIR = REPO_ROOT / "data" / "traces" / "5d434903-bc26-4dc4-9229-8d886d2c6781"
DEFAULT_EVENT_ID = "graph:13"
K = 3

ANCHOR = "# FINAL REMINDER"
DIRECTIVE = (
    "# FINAL REMINDER\n\n"
    "* **CHAIN THE WHOLE TASK IN ONE BLOCK:** when the task names a fixed "
    "sequence of tool calls where each call consumes the previous return value, "
    "emit ONE code block that awaits every call in that sequence - including the "
    "final one - instead of stopping after the first call. Do not split a fully "
    "specified chain across turns."
)

# Primary predicate: does the completion commit to the WHOLE chain (all three
# calls) rather than only the first step? This is measurable in the text this
# endpoint actually returns at this boundary.
WHOLE_CHAIN_PATTERN = r"(all three|the three tools|three calls|each tool|every call)"
# Secondary predicate, expected to be unmeasurable here - see module docstring.
TARGET_TOOL = "checksum_beta"


def _truncate(text: str, limit: int = 200) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 3] + "..."


def _report_arm(name: str, arm) -> None:
    print(f"--- arm {name} ---")
    print(f"  requests issued : {arm.request_count} (design intends 1)")
    print(f"  completions     : {len(arm.completions)} (k={arm.k})")
    print(f"  DISTINCT        : {arm.distinct_count} of {arm.k}")
    print(f"  passes          : {arm.pass_count}  pass_rate={arm.pass_rate:.3f}")
    print(f"  predicate errors: {arm.predicate_errors}")
    for index, completion in enumerate(arm.completions):
        print(f"    [{index}] {_truncate(completion)}")


def _rescore(completions, predicate) -> int:
    passes = 0
    for completion in completions:
        try:
            passes += 1 if predicate(completion) else 0
        except Exception:  # noqa: BLE001
            pass
    return passes


def main(argv: list[str]) -> int:
    prepare_cuga_environment()

    trace_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_TRACE_DIR
    event_id = argv[2] if len(argv) > 2 else DEFAULT_EVENT_ID

    if not (trace_dir / "causal-trace.json").exists():
        print(f"SKIP: no causal-trace.json under {trace_dir}")
        return 1

    model_env = os.environ.get("CUGA_MODEL") or os.environ.get("LITELLM_MODEL")
    if not model_env:
        print("SKIP: neither CUGA_MODEL nor LITELLM_MODEL is set; a live A/B needs a model.")
        return 1
    if not (os.environ.get("CUGA_API_KEY") or os.environ.get("LITELLM_API_KEY")):
        print("SKIP: neither CUGA_API_KEY nor LITELLM_API_KEY is set.")
        return 1

    available = list_recorded_llm_calls(trace_dir)
    if event_id not in available:
        print(f"SKIP: {event_id} is not an llm_call_start event. Available: {list(available)}")
        return 1

    call = load_recorded_call(trace_dir, event_id)
    system = next((m["content"] for m in call.messages if m["role"] == "system"), "")

    print("=" * 78)
    print("LIVE counterfactual proxy A/B over ONE recorded LLM call")
    print("=" * 78)
    print(f"trace dir          : {trace_dir}")
    print(f"event id           : {call.event_id}")
    print(f"recorded model     : {call.model}")
    print(f"env model          : {model_env}")
    print(f"messages           : {len(call.messages)} {[m['role'] for m in call.messages]}")
    print(f"system chars       : {len(system)}")
    print(f"total prompt chars : {call.total_content_chars}")
    print(f"k (n per request)  : {K}")
    print(f"primary predicate  : matches_regex({WHOLE_CHAIN_PATTERN})")
    print(f"recorded baseline  : {_truncate(call.baseline_response or '<none>')}")
    print()

    substitution = artifact_text_substitution(ANCHOR, DIRECTIVE)
    try:
        edited = substitution(call.messages)
    except ProxySubstitutionError as exc:
        print(f"FAIL: substitution anchor missing: {exc}")
        return 1

    before = sum(len(m.get("content", "")) for m in call.messages)
    after = sum(len(str(m.get("content", ""))) for m in edited)
    print("--- substitution diff summary ---")
    print(f"  anchor            : {ANCHOR!r}")
    print(f"  baseline chars    : {before}")
    print(f"  edited chars      : {after}")
    print(f"  chars added       : {max(0, after - before)}")
    print(f"  chars removed     : {max(0, before - after)}")
    print(f"  arms identical?   : {'YES (INVALID)' if edited == call.messages else 'no'}")
    print()
    print("Issuing 2 REAL requests (one per arm, n=%d each), arms in parallel..." % K)
    print()

    try:
        verdict = run_proxy_ab(
            call,
            substitution=substitution,
            predicate=matches_regex(WHOLE_CHAIN_PATTERN),
            k=K,
        )
    except Exception as exc:  # noqa: BLE001 - report the live failure verbatim
        print(f"FAIL: live A/B raised {type(exc).__name__}: {exc}")
        return 1

    _report_arm("A (baseline, recorded prompt verbatim)", verdict.baseline)
    print()
    _report_arm("B (edited, directive injected into SYSTEM)", verdict.edited)
    print()
    print("--- verdict (primary predicate) ---")
    print(f"  predicate       : {verdict.predicate_name}")
    print(f"  baseline rate   : {verdict.baseline.pass_rate:.3f}")
    print(f"  edited rate     : {verdict.edited.pass_rate:.3f}")
    print(f"  delta           : {verdict.delta:+.3f}")
    print(f"  label           : {verdict.label}")
    print(f"  evidence_kind   : {verdict.evidence_kind}")
    print(f"  substitution    : {dict(verdict.substitution_summary)}")
    print()

    print("--- CONTROL: did n=k actually sample on a real %d-char prompt? ---" % len(system))
    sampled = True
    for name, arm in (("A", verdict.baseline), ("B", verdict.edited)):
        word = "SAMPLED" if arm.distinct_count > 1 else "NOT SAMPLED (all identical)"
        if arm.distinct_count <= 1 and arm.k > 1:
            sampled = False
        print(f"  arm {name}: {arm.distinct_count}/{arm.k} distinct -> {word}")
    if not sampled:
        print("  WARNING: an arm returned k identical completions. Its pass rate is ONE")
        print("           observation repeated k times, so the delta is not a rate")
        print("           difference and this verdict carries far less evidence than k.")
    print()

    # No extra requests: re-score the SAME completions with the tool-call
    # predicate to expose the unmeasurable-predicate failure mode.
    tool_predicate = calls_tool(TARGET_TOOL)
    baseline_tool = _rescore(verdict.baseline.completions, tool_predicate)
    edited_tool = _rescore(verdict.edited.completions, tool_predicate)
    fenced = sum(
        1
        for completion in verdict.baseline.completions + verdict.edited.completions
        if "```" in completion
    )
    print(f"--- NEGATIVE FINDING: {tool_predicate.name} at this boundary ---")
    print("  (re-scored offline over the same completions; no extra request)")
    print(f"  baseline passes : {baseline_tool}/{K}")
    print(f"  edited passes   : {edited_tool}/{K}")
    print(f"  completions containing a fenced code block: {fenced}/{2 * K}")
    if baseline_tool == 0 and edited_tool == 0:
        print("  UNMEASURABLE: no completion at this boundary emits a tool call in")
        print("  message.content, so this predicate returns 'no_change' for ANY edit.")
        print("  A caller reading that as 'the edit did not help' would be wrong: the")
        print("  predicate never had a chance to observe the behaviour it tests.")
    print()

    one_request_per_arm = verdict.baseline.request_count == 1 and verdict.edited.request_count == 1
    print("--- checks ---")
    print(f"  one request per arm       : {'PASS' if one_request_per_arm else 'FAIL'}")
    print("  arms were distinct prompts: PASS (guard would have raised otherwise)")
    print(f"  evidence_kind is proxy    : {'PASS' if verdict.evidence_kind == 'proxy' else 'FAIL'}")
    print(f"  n=k yielded samples       : {'PASS' if sampled else 'FAIL'}")
    print()
    print("SUMMARY: " + ("PASS" if one_request_per_arm and sampled else "FAIL"))
    print(
        "Scope: this is PROXY evidence about one prompt boundary of one recorded call. "
        "It is NOT a confirmed task outcome, NOT agent-state replay, and one run of "
        "one event proves nothing about reliability. Repeating this identical A/B "
        "re-reads the provider cache and is not an independent second trial. A "
        "predicate that cannot observe the target behaviour at the chosen boundary "
        "yields a confident-looking 'no_change' that means nothing - see the "
        "NEGATIVE FINDING section above."
    )
    return 0 if (one_request_per_arm and sampled) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
