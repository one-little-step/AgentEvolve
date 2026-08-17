"""Does an ``instructions`` artifact actually reach the rollout model?

We are about to make ``instructions`` the editor's primary lever, on the strength
of a code path: ``_harness_config`` puts it under the ``instructions`` key,
``run_task`` forwards it, and ``_construct_agent`` passes it as
``special_instructions``. That is structural evidence, and structural evidence has
already misled this project once -- ``enable_skills=True`` looked correctly wired
for a long time while CUGA silently discarded the entire skills block because
``enable_shell_tool`` was false.

So this asserts the only thing that matters: a marker that exists **nowhere**
except the artifact appears in the model's output. If it does, the artifact
reached the model. If it does not, no amount of editing that slot can ever move a
score, and the whole plan is void.

The marker is a random token per run, so it cannot be produced from prior
knowledge, recalled from a cached completion, or guessed from the task. Anything
weaker (asserting on a correct answer, or on a phrase the model might invent)
would prove nothing.

Runs the *real* rollout path -- ``CugaWrapper.run_task`` with a harness_config,
exactly as ``CugaAdapter.run_full_rollout`` builds it -- not a hand-made
``CugaAgent``, because the question is whether OUR plumbing delivers it.

Usage (needs credentials in .env; costs 2 rollouts):
    uv run python scripts/verify_instructions_reach_model.py 2>&1 \
      | tee terminal_output/instructions_reach/run.log
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


def probe(marker: str, *, with_instructions: bool) -> dict:
    """One rollout, with or without the instructions artifact."""
    from agent_evolve.cuga_wrapper import (
        CugaWrapper,
        RuntimeSettings,
        prepare_cuga_environment,
    )

    prepare_cuga_environment()
    wrapper = CugaWrapper.from_cuga(RuntimeSettings.from_env())

    # Shaped exactly like CugaAdapter._harness_config output.
    harness: dict[str, object] = {
        "input": "What is 2 + 2? Answer briefly.",
        "version": "instructions-probe",
    }
    if with_instructions:
        harness["instructions"] = (
            "You are answering a short arithmetic question.\n"
            f"Mandatory: end your reply with the exact line {marker}\n"
            "This line is required for the reply to be considered complete."
        )

    try:
        result = wrapper.run_task(f"probe-{uuid.uuid4().hex[:8]}", harness)
        final = str(result.get("final_output") or "")
        status = str(result.get("status") or "")
    except Exception as exc:  # noqa: BLE001 - the reason IS the finding
        final, status = f"<exception> {exc!r}", "error"

    return {
        "with_instructions": with_instructions,
        "status": status,
        "marker_present": marker in final,
        "final_output": final.strip()[-400:],
    }


def main() -> int:
    marker = f"MARKER-{uuid.uuid4().hex[:12].upper()}"
    print(f"marker (exists only in the artifact): {marker}\n", flush=True)

    print("=" * 70)
    print("ARM 1: instructions artifact PRESENT")
    print("=" * 70, flush=True)
    treated = probe(marker, with_instructions=True)
    print(f"  status={treated['status']} marker_present={treated['marker_present']}")
    print(f"  output: {treated['final_output'][:300]}\n", flush=True)

    print("=" * 70)
    print("ARM 2: instructions artifact ABSENT (control)")
    print("=" * 70, flush=True)
    control = probe(marker, with_instructions=False)
    print(f"  status={control['status']} marker_present={control['marker_present']}")
    print(f"  output: {control['final_output'][:300]}\n", flush=True)

    dest = REPO_ROOT / "terminal_output" / "instructions_reach"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "result.json").write_text(
        json.dumps({"marker": marker, "treated": treated, "control": control}, indent=2)
    )

    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    if treated["marker_present"] and not control["marker_present"]:
        print("REACHES THE MODEL. The marker appeared only when the artifact was")
        print("supplied, so `instructions` is a live, editable lever.")
        return 0
    if treated["marker_present"] and control["marker_present"]:
        print("INCONCLUSIVE: the control also emitted the marker, so it leaked")
        print("from somewhere other than the artifact. Do not trust this result.")
        return 1
    print("DOES NOT REACH THE MODEL. Editing `instructions` cannot change")
    print("behavior; making it the editor's primary lever would be a no-op.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
