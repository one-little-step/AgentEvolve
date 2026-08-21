"""SV-8 probe: what artifact surface was the editor OFFERED vs which did it CHOOSE?

Runs the *real* ``CugaEditorAgent`` -> real ``CugaAgent`` so that CUGA-internal
LLM traffic crosses the mitmproxy interceptor. Regular proxy mode is the whole
point here: the editor deliberately goes through ``CugaAgent`` (a design choice,
not a defect), so our four LiteLLM wrappers never see these calls and
``X-AE-*`` headers cannot be emitted from our side. The proxy is therefore the
only instrument that can observe the editor's LLM layer at all.

What this probe establishes, in order:

1. **Offered surface.** ``list_artifacts`` is the only tool that tells the agent
   what it may write. We capture its literal return value, so "offered" is the
   bytes the model saw, not our belief about them.
2. **Chosen surface.** Every ``stage_replace``/``stage_create`` call is recorded
   with its artifact id, so the chosen surfaces are counted from tool calls
   rather than inferred from the final response.
3. **Interception coverage.** The probe reports whether any CUGA-internal call
   reached the proxy. If zero calls are captured while the agent demonstrably
   ran, ``HTTPS_PROXY`` is not honoured by CUGA's internal client -- the one
   thing ``docker/observability/README.md`` lists as unverified.

Run it through the proxy, never directly:

    ./docker/observability/proxy.sh run -- \
        python3 tools/probes/sv8_editor_surface_probe.py

This lives under ``tools/probes/`` rather than ``terminal_output/`` because
``terminal_output/`` is gitignored (``.gitignore:13``) and a clean would destroy
it. ``REPO`` resolves via ``parents[2]``, which is the repo root from either
location, so the two copies are interchangeable.

Cost control: set ``AE_SV8_MOCK=1`` and enable a mock rule so no upstream call
is billed. A mocked run answers (1) and (3) but *not* (2), because the chosen
surface is then whatever the mock dictates -- the script labels its own output
accordingly rather than letting a mocked verdict read as a live one.

Nothing here writes an expected answer, a grader internal, or a credential into
the capture: the task input is synthetic and the analysis mechanism is generic.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

from dotenv import load_dotenv

load_dotenv(REPO / ".env")

from agent_evolve.adapters.cuga_adapter import CugaAdapter
from agent_evolve.adapters.cuga_editor import CugaEditorAgent
from agent_evolve.benchmarks.cuga_executor import VANILLA_HARNESS
from agent_evolve.core.blame import BlameGraph, BlameNode, CausalAnalysis
from agent_evolve.core.contracts import CandidateWorkspace, EvolutionTask
from agent_evolve.core.correlation import correlation_scope
from agent_evolve.core.memory import EditMemory
from agent_evolve.core.editor import EditorRequest
from agent_evolve.pipeline import _harness_artifacts

CAPTURES = REPO / "docker" / "observability" / "captures" / "calls.jsonl"
RUN_ID = f"sv8-probe-{int(time.time())}"


class _Wrapper:
    """Minimal artifact host. The probe exercises the editor, not a rollout."""

    def __init__(self, artifacts: dict[str, str]) -> None:
        self._a = dict(artifacts)

    def get_artifacts(self) -> dict[str, str]:
        return dict(self._a)

    def update_artifact(self, artifact_id: str, content: str) -> None:
        self._a[artifact_id] = content


def _capture_offset() -> int:
    """Line count before the run, so only this probe's calls are analyzed."""
    if not CAPTURES.exists():
        return 0
    with CAPTURES.open(encoding="utf-8") as fh:
        return sum(1 for _ in fh)


def _new_records(offset: int) -> list[dict]:
    if not CAPTURES.exists():
        return []
    out: list[dict] = []
    with CAPTURES.open(encoding="utf-8") as fh:
        for index, line in enumerate(fh):
            if index < offset:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def main() -> int:
    artifacts = _harness_artifacts(VANILLA_HARNESS)
    # CugaAdapter only calls get_artifacts/update_artifact on its wrapper, so a
    # duck-typed host is sufficient and avoids booting a real CUGA runtime for
    # a probe that never performs a rollout.
    adapter = CugaAdapter(cast("Any", _Wrapper(artifacts)))
    adapter.register_candidate("base", artifacts)

    inventory = adapter.artifact_inventory("base")
    write_set = tuple(sorted(d.artifact_id for d in inventory if d.writable))

    print("=" * 72)
    print("SV-8 PROBE — offered vs chosen artifact surface")
    print("=" * 72)
    print(f"run_id            : {RUN_ID}")
    print(f"offered write_set : {list(write_set)}")
    print(f"offered surfaces  : {sorted({w.split('/')[0] for w in write_set})}")
    print(f"creatable_prefixes: {list(adapter.creatable_prefixes)}")

    task = EvolutionTask(
        task_id="sv8-probe-task",
        # Synthetic and self-contained: no benchmark item, no expected answer.
        input_text="Summarize the two most recent entries in the log.",
    )
    analysis = CausalAnalysis(
        mechanism=(
            "the agent restated the request instead of consulting the log, so "
            "no entry was ever read"
        ),
        severity=0.7,
        score=0.0,
        blame_graph=BlameGraph(
            nodes=(BlameNode(actor_id="planner", blame=1.0, artifacts=write_set),)
        ),
    )

    request = EditorRequest(
        base_workspace=CandidateWorkspace(
            attempt_id="sv8-probe-attempt",
            version="base",
            path=REPO / "terminal_output" / "sv8" / "workspace",
            parent_version="base",
        ),
        task=task,
        analysis=analysis,
        issue_id="sv8-probe-issue",
        write_set=write_set,
        current_artifacts={
            a: c for a, c in artifacts.items() if a in set(write_set)
        },
        creatable_prefixes=adapter.creatable_prefixes,
    )

    editor = CugaEditorAgent(adapter=adapter, memory=EditMemory())

    offset = _capture_offset()
    t0 = time.time()
    outcome = "?"
    error = ""
    # Correlation labels our own wrappers; CUGA-internal calls will arrive
    # unlabelled, which is itself the measurement for coverage.
    try:
        with correlation_scope(
            run=RUN_ID, candidate="base", task=task.task_id, phase="sv8-editor"
        ):
            response = editor.propose_edit(request)
        outcome = "VALID"
        chosen = sorted(response.writes)
    except Exception as exc:  # noqa: BLE001 - a decline is a real result here
        outcome = getattr(editor.last_outcome, "name", str(editor.last_outcome))
        error = f"{type(exc).__name__}: {exc}"
        chosen = []
    elapsed = time.time() - t0

    tools = list(editor.last_tools_called)
    print("\n--- editor result ---")
    print(f"outcome        : {outcome}")
    if error:
        print(f"detail         : {error}")
    print(f"elapsed        : {elapsed:.1f}s")
    print(f"tools called   : {tools or '(none)'}")
    print(f"artifacts writ.: {chosen or '(none)'}")
    if chosen:
        print(f"chosen surfaces: {sorted({c.split('/')[0] for c in chosen})}")

    records = _new_records(offset)
    print("\n--- proxy interception ---")
    print(f"new captured LLM calls: {len(records)}")
    if not records:
        print(
            "  NO CALLS CAPTURED. Either the agent never reached the LLM layer, "
            "or CUGA's internal client ignores HTTPS_PROXY."
        )
    for rec in records:
        corr = rec.get("correlation") or {}
        req = rec.get("request") or {}
        resp = rec.get("response") or {}
        body = req.get("body") or ""
        print(
            f"  {req.get('host')}{req.get('path')} "
            f"status={resp.get('status')} mocked={rec.get('mocked')} "
            f"labelled={bool(corr)} bytes={len(body)}"
        )
        # The decisive bytes: did the offered roster reach the model's context?
        #
        # ``instructions`` is deliberately EXCLUDED from this check. It is both a
        # concrete artifact id and an English word that occurs throughout
        # EDITOR_INSTRUCTIONS prose, so a substring hit on it proves nothing --
        # it matched turn 1 on the first version of this probe, where no roster
        # had yet been fetched. Only the three group ids carry an unguessable
        # slot name, so only they can evidence roster delivery.
        roster_ids = [a for a in write_set if "/" in a]
        present = [a for a in roster_ids if a in body]
        print(f"    roster ids present: {present or '(none)'}")
        # The literal tool return is the authoritative "offered" record.
        if '"writable"' in body:
            print("    list_artifacts return present in context: YES")

    print("\n--- verdict inputs ---")
    print(f"offered surfaces (code)  : {sorted({w.split('/')[0] for w in write_set})}")
    print(f"chosen surfaces (tools)  : {sorted({c.split('/')[0] for c in chosen})}")
    print(f"interception coverage    : {len(records)} call(s)")
    if os.getenv("AE_SV8_MOCK"):
        print(
            "NOTE: AE_SV8_MOCK set — 'chosen' reflects the mock rule, not a "
            "live model preference."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
