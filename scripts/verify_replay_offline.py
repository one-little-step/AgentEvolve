"""Offline proof that recorded LLM calls in a real trace are single-call replayable.

Makes NO network call and issues NO model request. It only resolves the recorded
``messages_ref``/``response_ref`` payload blobs for every ``llm_call_start`` event
and reports whether each one yields a complete, replayable message array.

A call PASSES when it resolves to at least one message, includes a system
message, and carries non-empty content. Baseline response availability is
reported separately: it is useful evidence, not a replayability requirement.

Usage:
    uv run python scripts/verify_replay_offline.py [TRACE_DIR]
"""
from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from agent_evolve.cuga_wrapper import (  # noqa: E402
    RecordedCall,
    list_recorded_llm_calls,
    load_recorded_call,
)

DEFAULT_TRACE_DIR = REPO_ROOT / "data" / "traces" / "5d434903-bc26-4dc4-9229-8d886d2c6781"


def _evaluate(call: RecordedCall) -> tuple[bool, str]:
    if not call.messages:
        return False, "no messages resolved"
    if not call.has_system_message:
        return False, "no SystemMessage present"
    if call.total_content_chars == 0:
        return False, "all message content empty"
    if not call.model:
        return False, "no model recorded"
    return True, "complete replayable message array"


def main(argv: list[str]) -> int:
    trace_dir = Path(argv[1]) if len(argv) > 1 else DEFAULT_TRACE_DIR
    print(f"Trace directory : {trace_dir}")
    if not (trace_dir / "causal-trace.json").exists():
        print(f"FAIL: no causal-trace.json under {trace_dir}")
        return 1

    event_ids = list_recorded_llm_calls(trace_dir)
    print(f"llm_call_start  : {len(event_ids)} event(s)")
    print("NOTE: this script makes no network call and issues no model request.")
    print()

    header = (
        f"{'event_id':<12} {'msgs':>4} {'system':>7} {'chars':>7} "
        f"{'baseline':>9} {'model':<26} {'result':<8} detail"
    )
    print(header)
    print("-" * len(header))

    passed = 0
    baselines = 0
    roles: dict[str, int] = {}
    failures: list[tuple[str, str]] = []
    for event_id in event_ids:
        try:
            call = load_recorded_call(trace_dir, event_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            failures.append((event_id, f"{type(exc).__name__}: {exc}"))
            print(
                f"{event_id:<12} {'-':>4} {'-':>7} {'-':>7} {'-':>9} "
                f"{'-':<26} {'FAIL':<8} {type(exc).__name__}: {exc}"
            )
            continue

        ok, detail = _evaluate(call)
        passed += 1 if ok else 0
        baselines += 1 if call.baseline_response else 0
        for message in call.messages:
            role = message.get("role", "?")
            roles[role] = roles.get(role, 0) + 1
        if not ok:
            failures.append((event_id, detail))
        print(
            f"{event_id:<12} {len(call.messages):>4} "
            f"{('yes' if call.has_system_message else 'NO'):>7} "
            f"{call.total_content_chars:>7} "
            f"{('yes' if call.baseline_response else 'no'):>9} "
            f"{(call.model or '-'):<26} {('PASS' if ok else 'FAIL'):<8} {detail}"
        )

    print()
    print(f"Replayable message arrays : {passed}/{len(event_ids)}")
    print(f"Baseline responses resolved: {baselines}/{len(event_ids)}")
    print(f"Role histogram            : {dict(sorted(roles.items()))}")
    if failures:
        print()
        print("Failures:")
        for event_id, detail in failures:
            print(f"  {event_id}: {detail}")

    overall = bool(event_ids) and passed == len(event_ids)
    print()
    print(f"SUMMARY: {'PASS' if overall else 'FAIL'}")
    print(
        "Scope: single-LLM-call replay inputs only. Agent-state/checkpoint replay "
        "is NOT proven and remains unsupported."
    )
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
