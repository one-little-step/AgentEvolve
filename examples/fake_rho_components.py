"""Offline, deterministic RHO components for the ``--dry-run`` rehearsal.

``--dry-run`` documents "no CUGA process, no model endpoint, no network". The
five real RHO components all reach a model: the comprehender and difficulty judge
call ``litellm`` directly, and the diagnoser, optimizer and preference judge each
construct a CUGA agent. Handing a dry run the real ones therefore breaks the
promise the flag makes, and it breaks it *quietly* -- every component reports a
failed call as an unobserved result, so the round degrades to "summaries
unavailable / no coreset task resolved", which reads like a data problem rather
than a wiring one.

These fakes live outside ``src/agent_evolve/core`` for the same reason
:mod:`examples.fake_adapter` does: they are a concrete runtime, and the core must
stay agent-neutral. They satisfy exactly the duck-typed shapes ``RhoHooks``
documents and nothing more.

What the rehearsal does and does not prove
------------------------------------------
It proves the wiring: phase ordering, the ``observed``/``available`` gates,
all-N retention, pool commit with provenance, entropy cell population, coreset
resolution, and the report. It proves **nothing** about the method -- every
difficulty score, diagnosis and preference here is a deterministic function of a
task id, not a judgement. A run using these is a plumbing test, and the CLI says
so.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from typing import Mapping, Sequence

__all__ = [
    "OfflineDifficultyJudge",
    "OfflineDiagnosis",
    "OfflineDiagnoser",
    "OfflineOptimizer",
    "OfflinePreferenceJudge",
    "OfflineProposal",
    "OfflineProposalReport",
    "OfflineSummary",
    "OfflineTrajectoryComprehender",
    "OfflineVerdict",
    "OfflinePreferenceVerdict",
    "offline_rho_components",
]

#: Where a generated artifact id must start for ``CugaAdapter`` to accept it.
CREATABLE_PREFIX = "skills/generated-"


def _unit(text: str) -> float:
    """A deterministic value in ``[0, 1)`` derived from ``text``.

    Hashed rather than counted so two adjacent task ids do not produce two
    adjacent scores: a rehearsal whose difficulty ranking is monotonic in task
    order would make the DPP selector look like it was working when it was only
    reading the task list back.
    """
    digest = sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") / float(1 << 32)


# --------------------------------------------------------------------------- #
# Phase 2: trajectory comprehension
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OfflineSummary:
    """The shape ``RhoHooks.comprehend`` must return."""

    task_id: str
    observed: bool = True
    error: str = ""
    embedding_text: str = ""


@dataclass(slots=True)
class OfflineTrajectoryComprehender:
    """Summarizes a historical record without a model call."""

    calls: int = 0

    def comprehend(self, record: object) -> OfflineSummary:
        self.calls += 1
        task_id = str(getattr(record, "task_id", ""))
        # The embedded text is the fingerprint plus the summary in the real
        # pipeline; here it carries the tool-observation count so two records
        # with different trajectory shapes embed differently.
        observations = int(getattr(record, "tool_observation_count", 0) or 0)
        return OfflineSummary(
            task_id=task_id,
            embedding_text=(
                f"offline summary for {task_id}: attempted the task using "
                f"{observations} tool observation(s) and produced an answer"
            ),
        )


# --------------------------------------------------------------------------- #
# Phase 3: difficulty and abstract fingerprint
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OfflineVerdict:
    """The shape ``RhoHooks.judge`` must return."""

    task_id: str
    difficulty: float = 0.0
    abstract_fingerprint: str = ""
    observed: bool = True
    error: str = ""
    ground_truth_used: bool = False


@dataclass(slots=True)
class OfflineDifficultyJudge:
    """Scores difficulty deterministically from the task id.

    An empty summary is refused for the same reason the real judge refuses it:
    with no summary there is nothing abstract to reason over, and returning a
    number anyway would put a fabricated score into coreset selection.
    """

    calls: int = 0

    def judge(
        self,
        record: object,
        summary_text: str,
        *,
        expected_answer: str | None = None,
    ) -> OfflineVerdict:
        self.calls += 1
        task_id = str(getattr(record, "task_id", ""))
        if not (summary_text or "").strip():
            return OfflineVerdict(
                task_id=task_id,
                observed=False,
                error="empty trajectory summary: nothing to judge",
            )
        return OfflineVerdict(
            task_id=task_id,
            difficulty=round(_unit(f"difficulty:{task_id}") * 10.0, 3),
            abstract_fingerprint=(
                f"offline fingerprint for {task_id}: the trajectory committed "
                f"to an answer without verifying it"
            ),
            # Deliberately never true: no expected answer is read, so no ground
            # truth can leak into a fingerprint a manifest would then carry.
            ground_truth_used=False,
        )


# --------------------------------------------------------------------------- #
# Phase 6: group diagnosis
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OfflineDiagnosis:
    """The shape ``RhoHooks.diagnose`` must return.

    Carries the fields ``select_diagnoses`` and the optimizer prompt read, so the
    rehearsal exercises the same selection and rendering path a live run does.
    """

    task_id: str
    observed: bool = True
    status: str = "OK"
    error: str = ""
    severity: float = 0.5
    rollouts_seen: int = 0
    recurring_failure_mode: str = ""
    improvement_direction: str = ""
    candidate_surfaces: tuple[str, ...] = ()
    disagreements: tuple[str, ...] = ()
    self_validation_observed: bool = False


@dataclass(slots=True)
class OfflineDiagnoser:
    """Diagnoses a rollout group without invoking an agent."""

    calls: int = 0

    def diagnose(
        self, task_id: str, task_input: str, traces: Sequence[object]
    ) -> OfflineDiagnosis:
        self.calls += 1
        group = tuple(traces)
        if not group:
            # Mirrors the real diagnoser: no rollouts is a status, not a
            # diagnosis with empty fields.
            return OfflineDiagnosis(
                task_id=task_id,
                observed=False,
                status="NO_ROLLOUTS",
                error="no rollouts to diagnose",
            )
        return OfflineDiagnosis(
            task_id=task_id,
            severity=round(_unit(f"severity:{task_id}"), 3),
            rollouts_seen=len(group),
            recurring_failure_mode=(
                f"offline: every rollout for {task_id} answered without "
                f"consulting the retrieval skill"
            ),
            improvement_direction=(
                "offline: instruct the agent to retrieve before answering"
            ),
            candidate_surfaces=("skills/retrieval",),
        )


# --------------------------------------------------------------------------- #
# Phase 7: candidate proposal
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OfflineProposal:
    """The shape each ``propose`` candidate must have.

    ``artifacts`` is the COMPLETE set -- base carried forward with the edit
    applied -- because ``register_candidate`` registers a whole workspace, not a
    diff.
    """

    candidate_index: int
    artifacts: Mapping[str, str]
    rationale: str = ""
    observed: bool = True
    error: str = ""
    edited_ids: tuple[str, ...] = ()
    created_ids: tuple[str, ...] = ()
    fingerprint: str = ""
    tools_called: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OfflineProposalReport:
    """The shape ``RhoHooks.propose`` must return."""

    candidates: tuple[OfflineProposal, ...] = ()
    requested: int = 0
    discarded: tuple[tuple[int, str], ...] = ()

    @property
    def distinct(self) -> int:
        return len(self.candidates)

    @property
    def collapsed(self) -> bool:
        return self.distinct < self.requested

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _index, reason in self.discarded:
            key = reason.split(":", 1)[0]
            counts[key] = counts.get(key, 0) + 1
        return counts


@dataclass(slots=True)
class OfflineOptimizer:
    """Produces N *distinct* candidates without invoking an agent.

    Distinctness is the property that matters: N identical proposals would make
    the preference judge compare a harness against itself, and cross-candidate
    entropy would see one candidate wearing N names. Each proposal therefore
    creates a differently-named artifact with differently-hashed content.
    """

    calls: int = 0

    def propose(
        self,
        base_artifacts: Mapping[str, str],
        diagnoses: Sequence[object],
        n: int,
    ) -> OfflineProposalReport:
        self.calls += 1
        if n < 1:
            raise ValueError("n must be >= 1")
        if not base_artifacts:
            raise ValueError("base_artifacts must not be empty")

        base = dict(base_artifacts)
        directions = [
            str(getattr(d, "improvement_direction", "") or "")
            for d in diagnoses
            if bool(getattr(d, "observed", False))
        ]
        summary = "; ".join(directions) or "no observed diagnosis"

        proposals: list[OfflineProposal] = []
        for index in range(n):
            created_id = f"{CREATABLE_PREFIX}offline-{index}"
            body = (
                f"# offline candidate {index}\n\n"
                f"Derived from: {summary}\n"
                f"Variant token: {_unit(f'variant:{index}:{summary}'):.9f}\n"
            )
            artifacts = dict(base)
            artifacts[created_id] = body
            proposals.append(
                OfflineProposal(
                    candidate_index=index,
                    artifacts=artifacts,
                    rationale=f"offline variant {index}: {summary}",
                    created_ids=(created_id,),
                    fingerprint=sha256(body.encode("utf-8")).hexdigest(),
                )
            )
        return OfflineProposalReport(
            candidates=tuple(proposals), requested=n
        )


# --------------------------------------------------------------------------- #
# Phase 9: preference judging
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class OfflinePreferenceVerdict:
    """The shape ``RhoHooks.compare`` must return.

    ``score`` is signed and oriented ``baseline -> candidate``; ``available`` is
    the gate a consumer needs, and an unavailable verdict must be excluded from
    averages rather than folded in as a tie.
    """

    task_id: str
    score: float = 0.0
    winner: str = "tie"
    rationale: str = ""
    gt_available: bool = False
    available: bool = True
    error: str = ""
    status: str = "ok"
    orientation: str = "symmetric"
    position_bias: float = 0.0
    comparisons: int = 2
    inspected_both: bool = True
    tools_called: tuple[str, ...] = ()


@dataclass(slots=True)
class OfflinePreferenceJudge:
    """Compares two trajectories deterministically, without an agent.

    ``position_bias`` is reported as exactly 0.0 rather than omitted: this judge
    is symmetric by construction, so its bias is genuinely zero, and a live run's
    non-zero value is then a real measurement rather than a format difference.
    """

    calls: int = 0

    def compare_symmetric(
        self,
        task: object,
        baseline: object,
        candidate: object,
        *,
        baseline_summary: str = "",
        candidate_summary: str = "",
    ) -> OfflinePreferenceVerdict:
        self.calls += 1
        task_id = str(getattr(task, "task_id", ""))
        candidate_id = str(getattr(candidate, "candidate_id", ""))
        # Centred on zero so the rehearsal produces both signs, and keyed by the
        # candidate so two candidates on one task disagree.
        score = round(_unit(f"pref:{task_id}:{candidate_id}") * 2.0 - 1.0, 6)
        return OfflinePreferenceVerdict(
            task_id=task_id,
            score=score,
            winner=(
                "candidate" if score > 0 else "baseline" if score < 0 else "tie"
            ),
            rationale=f"offline deterministic verdict for {task_id}",
            # No expected answer is ever read, so ground truth is never
            # available and cannot leak into a rationale.
            gt_available=False,
        )


# --------------------------------------------------------------------------- #
# The set, as ``build_rho_hooks`` keyword arguments
# --------------------------------------------------------------------------- #
def offline_rho_components() -> dict[str, object]:
    """The five offline components, keyed for ``build_rho_hooks(**...)``."""
    return {
        "comprehender": OfflineTrajectoryComprehender(),
        "difficulty_judge": OfflineDifficultyJudge(),
        "diagnoser": OfflineDiagnoser(),
        "optimizer": OfflineOptimizer(),
        "preference_judge": OfflinePreferenceJudge(),
    }
