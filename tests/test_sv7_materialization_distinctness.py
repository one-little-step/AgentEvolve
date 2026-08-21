"""SV-7: is the upstream materialization path handing the judge identical traces?

SV-7 recorded a live observation -- the judge reporting ``Are raw strings
identical? True`` -- with two possible explanations. Prior work
(``tests/test_judge_slot_distinctness.py``) exonerated everything from the
rollout grid onward: the slot closures, the version stamping and
``compare_symmetric`` wiring. The register narrowed the remainder to exactly two
candidates:

1. the edit genuinely produced no behavioural change, which given SV-8 (every
    candidate edits only ``instructions``) is plausible and would be *correct*
    judge behaviour; or
2. the two versions were materialized to byte-identical harnesses **upstream** of
    the grid, i.e. in ``materialize_candidate``/``register_candidate``.

These tests settle (2) offline. It needs no proxy and no live model: the CUGA
adapter's artifact store is an in-memory mapping and the harness payload is a
pure function of it, so the question "do two versions produce byte-identical
harnesses" is directly decidable.

The path under test mirrors production exactly. ``orchestrator._execute_rollouts``
materializes a *child* workspace per probe (``:1249``) and then runs it
(``:1250``), so a test that only compared the registered parents would miss a
defect introduced by that extra materialization step -- which is precisely where
an aliasing bug would live.

Result: **no defect on this path.** Distinct artifacts produce distinct harness
payloads, and a child's edits never write back into its parent. Explanation (2)
is eliminated, leaving (1).
"""
from __future__ import annotations

import sys
from hashlib import sha256
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT / "src"), str(_ROOT), str(_ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from agent_evolve.adapters.cuga_adapter import CugaAdapter  # noqa: E402
from agent_evolve.core.contracts import ArtifactEdit, EvolutionTask  # noqa: E402


class _FakeWrapper:
    """Records the harness payload CUGA would receive, without running CUGA."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def get_artifacts(self) -> dict[str, str]:
        return {"instructions": "BASE instructions text"}

    def run_task(self, task_id: str, harness: dict[str, object]) -> dict[str, object]:
        self.calls.append((task_id, harness))
        return {"harness_seen": harness}


def _adapter() -> tuple[CugaAdapter, _FakeWrapper]:
    wrapper = _FakeWrapper()
    return CugaAdapter(wrapper=wrapper), wrapper


def _digest(harness: dict[str, object]) -> str:
    return sha256(repr(sorted(harness.items())).encode()).hexdigest()


def test_two_versions_produce_distinct_harness_payloads() -> None:
    """The direct test of SV-7 explanation (2).

    Runs the same two-step path production uses: materialize a child workspace
    per probe, then roll it out. If these payloads were equal, every preference
    score ever collected would be void.
    """
    adapter, wrapper = _adapter()
    adapter.register_candidate("cand-A", {"instructions": "A: always verify units"})
    adapter.register_candidate("cand-B", {"instructions": "B: cheapest tool first"})
    task = EvolutionTask(task_id="t1", input_text="question?")

    payloads: dict[str, dict[str, object]] = {}
    for version in ("cand-A", "cand-B"):
        workspace = adapter.materialize_candidate(version, f"probe-{version}")
        result = adapter.run_full_rollout(workspace, task, f"probe-{version}")
        payloads[version] = result["trace"]["harness_seen"]  # type: ignore[index]

    assert _digest(payloads["cand-A"]) != _digest(payloads["cand-B"]), (
        "two different candidates produced byte-identical harnesses -- SV-7 "
        "explanation (2) reproduced"
    )
    assert payloads["cand-A"]["instructions"] == "A: always verify units"
    assert payloads["cand-B"]["instructions"] == "B: cheapest tool first"
    assert len(wrapper.calls) == 2


def test_rollout_stamps_the_child_version_not_the_parent() -> None:
    """Each rollout must be attributable to the workspace that produced it."""
    adapter, _ = _adapter()
    adapter.register_candidate("cand-A", {"instructions": "A"})
    adapter.register_candidate("cand-B", {"instructions": "B"})
    task = EvolutionTask(task_id="t1", input_text="q")

    ids = []
    for version in ("cand-A", "cand-B"):
        workspace = adapter.materialize_candidate(version, "probe")
        result = adapter.run_full_rollout(workspace, task, "probe")
        ids.append(result["candidate_id"])  # type: ignore[index]

    assert ids[0] != ids[1]
    assert ids[0].startswith("cand-A")
    assert ids[1].startswith("cand-B")


def test_materialized_child_starts_identical_to_its_parent() -> None:
    """A pure copy is the intended behaviour, stated so a change is visible.

    This is *not* the defect: an unedited child SHOULD match its parent. Pinning
    it means a future change to copy semantics cannot pass unnoticed.
    """
    adapter, _ = _adapter()
    adapter.register_candidate("cand-A", {"instructions": "A: verify units"})
    workspace = adapter.materialize_candidate("cand-A", "att-1")

    parent = adapter.read_artifacts("cand-A", ("instructions",))
    child = adapter.read_artifacts(workspace.version, ("instructions",))
    assert parent == child


def test_child_edits_do_not_write_back_into_the_parent() -> None:
    """The aliasing defect that WOULD produce identical judge trajectories.

    If the child shared its parent's mapping, editing the child would mutate the
    parent, and a base-versus-candidate comparison would see two identical
    harnesses -- exactly SV-7's live observation.
    """
    adapter, _ = _adapter()
    adapter.register_candidate("cand-A", {"instructions": "A: verify units"})
    workspace = adapter.materialize_candidate("cand-A", "att-1")

    adapter.apply_structured_edits(
        workspace,
        [
            ArtifactEdit(
                artifact_id="instructions",
                operation="replace",
                payload={"content": "CHILD: verify units AND cite sources"},
            )
        ],
    )

    parent = adapter.read_artifacts("cand-A", ("instructions",))["instructions"]
    child = adapter.read_artifacts(workspace.version, ("instructions",))["instructions"]
    assert parent == "A: verify units", "child edit leaked into the parent"
    assert child == "CHILD: verify units AND cite sources"


def test_sibling_candidates_do_not_alias_each_other() -> None:
    """Two children of one parent must be independently editable."""
    adapter, _ = _adapter()
    adapter.register_candidate("cand-A", {"instructions": "A: base"})
    left = adapter.materialize_candidate("cand-A", "att-1")
    right = adapter.materialize_candidate("cand-A", "att-2")

    adapter.apply_structured_edits(
        left,
        [
            ArtifactEdit(
                artifact_id="instructions",
                operation="replace",
                payload={"content": "LEFT"},
            )
        ],
    )

    assert adapter.read_artifacts(left.version, ("instructions",))[
        "instructions"
    ] == "LEFT"
    assert adapter.read_artifacts(right.version, ("instructions",))[
        "instructions"
    ] == "A: base"


def test_identical_artifacts_do_produce_identical_harnesses() -> None:
    """The honest converse, preserved deliberately.

    When two versions genuinely carry the same artifacts, their harnesses SHOULD
    be identical. Masking that would replace one blind spot with another -- a
    no-op edit must remain visible as a no-op, which is the same reasoning
    ``test_identical_traces_are_reported_identically_not_hidden`` applies at the
    judge boundary. It also shows the test above can distinguish the two cases,
    rather than passing because the digests always differ.
    """
    adapter, _ = _adapter()
    adapter.register_candidate("cand-A", {"instructions": "same text"})
    adapter.register_candidate("cand-B", {"instructions": "same text"})
    task = EvolutionTask(task_id="t1", input_text="q")

    digests = []
    for version in ("cand-A", "cand-B"):
        workspace = adapter.materialize_candidate(version, "probe")
        result = adapter.run_full_rollout(workspace, task, "probe")
        digests.append(_digest(result["trace"]["harness_seen"]))  # type: ignore[index]

    assert digests[0] == digests[1]
