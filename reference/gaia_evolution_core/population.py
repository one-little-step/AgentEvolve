"""Agent-neutral immutable RHO-GEPA population lifecycle."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .contracts import AgentEvolutionAdapter, DiagnosisRecord, EvolutionBundle, EvolutionLLM, NormalizedTrajectory, RolloutLimits
from .history import EditHistoryRecord, EditHistoryStore, HistoryRetrieval
from .operators import OperatorResult, run_crossover, run_mutation


def elite_version_name(prefix: str, generation: int, rank: int) -> str:
    if generation < 1 or rank < 1:
        raise ValueError("generation and rank must be >= 1")
    return f"{prefix}-g{generation}-elite-{rank}"


def champion_version_name(prefix: str, generation: int) -> str:
    if generation < 1:
        raise ValueError("generation must be >= 1")
    return f"{prefix}-g{generation}-champion"


def preflight_targets(root: Path, prefix: str, generation: int, elite_count: int) -> tuple[tuple[Path, ...], Path]:
    elite_paths = tuple(Path(root) / elite_version_name(prefix, generation, rank) for rank in range(1, elite_count + 1))
    champion_path = Path(root) / champion_version_name(prefix, generation)
    for path in (*elite_paths, champion_path):
        if path.exists():
            raise FileExistsError(17, "immutable evolution target already exists", str(path))
    return elite_paths, champion_path


@dataclass(frozen=True, slots=True)
class PopulationResult:
    generation: int
    round_dir: Path
    child_count: int
    elite_versions: tuple[Path, ...]
    champion_version: Path
    champion_id: str
    errors: tuple[str, ...] = ()


@dataclass(slots=True)
class _Candidate:
    candidate_id: str
    bundle: EvolutionBundle
    parent_ids: tuple[str, ...]
    ancestor_id: str | None
    operator: str
    changed_modules: tuple[str, ...]
    task_scores: dict[str, float | None]
    average_score: float | None
    artifact_dir: Path
    history: HistoryRetrieval


class PopulationEvolution:
    """Run immutable, adapter-owned RHO-GEPA generations.

    The core only creates candidate directories and invokes the adapter editor;
    it never imports or writes any agent-specific policy representation.
    """

    def __init__(
        self,
        adapter: AgentEvolutionAdapter,
        artifact_root: Path,
        *,
        llm: EvolutionLLM,
        version_root: Path | None = None,
        history: EditHistoryStore | None = None,
        rollout_count: int = 1,
        limits: RolloutLimits = RolloutLimits(),
    ) -> None:
        self.adapter = adapter
        self.artifact_root = Path(artifact_root)
        self.version_root = Path(version_root) if version_root is not None else self.artifact_root
        self.llm = llm
        self.history = history or EditHistoryStore(self.artifact_root, adapter.agent_name, retrieval_enabled=True, semantic_enabled=False)
        self.rollout_count = rollout_count
        self.limits = limits
        self._rollout_cache: dict[str, dict[str, Sequence[NormalizedTrajectory]]] = {}

    def run_generation(
        self,
        *,
        initial_version: str,
        prefix: str,
        generation: int,
        elite_count: int,
        offspring_count: int,
        crossover_count: int,
        tasks: Sequence[NormalizedTrajectory],
    ) -> PopulationResult:
        if generation < 1 or elite_count < 1 or offspring_count < elite_count:
            raise ValueError("generation >= 1 and offspring_count >= elite_count >= 1 are required")
        if not 0 <= crossover_count <= offspring_count:
            raise ValueError("crossover_count must be between zero and offspring_count")
        elite_paths, champion_path = preflight_targets(self.version_root, prefix, generation, elite_count)
        round_dir = self.artifact_root / "evolution" / f"g{generation}"
        if round_dir.exists():
            raise FileExistsError(17, "generation artifact already exists", str(round_dir))
        round_dir.mkdir(parents=True)
        parent_versions = ((initial_version,) if generation == 1 else tuple(
            elite_version_name(prefix, generation - 1, rank) for rank in range(1, elite_count + 1)
        ))
        parents = [self.adapter.load_bundle(version) for version in parent_versions]
        parent_candidates = [self._evaluate_parent(bundle, tasks, round_dir) for bundle in parents]
        children: list[_Candidate] = []
        errors: list[str] = []
        eligible = [
            (left, right, ancestor)
            for index, left in enumerate(parent_candidates)
            for right in parent_candidates[index + 1:]
            if (ancestor := self._common_ancestor(left, right)) is not None
        ]
        for index in range(crossover_count):
            if eligible:
                left, right, ancestor = eligible[index % len(eligible)]
                child = self._crossover(generation, len(children), left, right, ancestor, tasks, round_dir, errors)
                if child is not None:
                    children.append(child)
                    continue
            children.append(self._mutation(generation, len(children), parent_candidates[index % len(parent_candidates)], tasks, round_dir, errors))
        remaining = offspring_count - len(children)
        for index in range(remaining):
            parent = parent_candidates[index % len(parent_candidates)]
            children.append(self._mutation(generation, len(children), parent, tasks, round_dir, errors))
        selection = self._select([*parent_candidates, *children], elite_count)
        for rank, candidate in enumerate(selection, 1):
            self.adapter.materialize_bundle(candidate.bundle, elite_paths[rank - 1])
            self._write_lineage(elite_paths[rank - 1], candidate)
        self.adapter.materialize_bundle(selection[0].bundle, champion_path)
        self._write_lineage(champion_path, selection[0])
        self._persist_history(children, generation)
        history_mode = next((candidate.history.mode for candidate in children), "off")
        manifest = {
            "adapter": self.adapter.agent_name,
            "generation": generation,
            "configuration": {"elite_count": elite_count, "offspring_count": offspring_count, "crossover_count": crossover_count},
            "parents": list(parent_versions), "child_count": len(children),
            "history": {"mode": history_mode, "path": str(self.history.path), "fallback_reasons": [c.history.fallback_reason for c in children if c.history.fallback_reason]},
            "candidates": [self._candidate_manifest(candidate) for candidate in [*parent_candidates, *children]],
            "elite_ids": [candidate.candidate_id for candidate in selection],
            "elite_paths": [str(path) for path in elite_paths], "champion_id": selection[0].candidate_id,
            "champion_path": str(champion_path), "errors": errors,
        }
        (round_dir / "population.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return PopulationResult(generation, round_dir, len(children), elite_paths, champion_path, selection[0].candidate_id, tuple(errors))

    def _evaluate_parent(self, bundle: EvolutionBundle, tasks: Sequence[NormalizedTrajectory], round_dir: Path) -> _Candidate:
        scores = self._scores(bundle, tasks, round_dir / "parents" / bundle.version)
        lineage = self._read_lineage(bundle.version)
        return _Candidate(bundle.version, bundle, tuple(lineage.get("parents", ())), lineage.get("ancestor"), "parent", (), scores, _average(scores), round_dir / "parents" / bundle.version, HistoryRetrieval("off", ()))

    def _mutation(self, generation: int, index: int, parent: _Candidate, tasks: Sequence[NormalizedTrajectory], round_dir: Path, errors: list[str]) -> _Candidate:
        candidate_id = f"g{generation}-mutation-{index}"
        candidate_dir = round_dir / "candidates" / candidate_id
        candidate_dir.parent.mkdir(parents=True, exist_ok=True)
        self.adapter.materialize_bundle(parent.bundle, candidate_dir)
        target = self.adapter.module_names[index % len(self.adapter.module_names)]
        diagnoses = self.adapter.diagnose(tasks, parent.bundle)
        history = self.history.retrieve(self._history_query(target, diagnoses), lineage_id=parent.bundle.version, module=target, minimum_records=1)
        result: OperatorResult
        try:
            result = run_mutation(self.llm, self.adapter.open_editor(candidate_dir, candidate_id), parent.bundle, target_module=target, diagnoses=diagnoses, trajectories=tasks, history=history)
        except Exception as exc:  # LLM unavailability remains an evaluated no-op child.
            errors.append(f"mutation {candidate_id}: {exc}")
            result = OperatorResult((), "", ({"reason": str(exc)},), history.mode)
        bundle = self._load_candidate(candidate_id, candidate_dir, parent.bundle)
        scores = self._scores(bundle, tasks, candidate_dir / "rollouts", reference=parent)
        return _Candidate(candidate_id, bundle, (parent.candidate_id,), parent.candidate_id, "mutation", result.changed_modules, scores, _average(scores), candidate_dir, history)

    def _crossover(self, generation: int, index: int, left: _Candidate, right: _Candidate, ancestor_version: str, tasks: Sequence[NormalizedTrajectory], round_dir: Path, errors: list[str]) -> _Candidate | None:
        candidate_id = f"g{generation}-crossover-{index}"
        candidate_dir = round_dir / "candidates" / candidate_id
        candidate_dir.parent.mkdir(parents=True, exist_ok=True)
        ancestor = self.adapter.load_bundle(ancestor_version)
        self.adapter.materialize_bundle(ancestor, candidate_dir)
        target = self.adapter.module_names[index % len(self.adapter.module_names)]
        diagnoses = self.adapter.diagnose(tasks, ancestor)
        history = self.history.retrieve(self._history_query(target, diagnoses), lineage_id=ancestor.version, module=target, minimum_records=1)
        try:
            result = run_crossover(self.llm, self.adapter.open_editor(candidate_dir, candidate_id), ancestor, left.bundle, right.bundle, diagnoses=diagnoses, trajectories=tasks, history=history, left_scores=left.task_scores, right_scores=right.task_scores)
        except Exception as exc:
            errors.append(f"crossover {candidate_id}: {exc}")
            return None
        bundle = self._load_candidate(candidate_id, candidate_dir, ancestor)
        scores = self._scores(bundle, tasks, candidate_dir / "rollouts", reference=left)
        return _Candidate(candidate_id, bundle, (left.candidate_id, right.candidate_id), ancestor.version, "crossover", result.changed_modules, scores, _average(scores), candidate_dir, history)

    def _load_candidate(self, candidate_id: str, candidate_dir: Path, fallback: EvolutionBundle) -> EvolutionBundle:
        modules = {module: (candidate_dir / module).read_text(encoding="utf-8") for module in self.adapter.module_names}
        return EvolutionBundle(candidate_id, modules) if modules else fallback

    def _scores(self, bundle: EvolutionBundle, tasks: Sequence[NormalizedTrajectory], artifact_dir: Path, *, reference: _Candidate | None = None) -> dict[str, float | None]:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        rollouts = self._rollout_cache.get(bundle.version)
        if rollouts is None:
            rollouts = dict(self.adapter.run_rollouts(bundle, tasks, rollout_count=self.rollout_count, limits=self.limits, artifact_dir=artifact_dir))
            self._rollout_cache[bundle.version] = rollouts
        reference_rollouts = None
        if reference is not None:
            reference_rollouts = self._rollout_cache.get(reference.bundle.version)
            if reference_rollouts is None:
                reference_rollouts = dict(self.adapter.run_rollouts(
                    reference.bundle, tasks, rollout_count=self.rollout_count, limits=self.limits,
                    artifact_dir=reference.artifact_dir / "rollouts",
                ))
                self._rollout_cache[reference.bundle.version] = reference_rollouts
        scores = self.adapter.score_rollouts(
            bundle, tasks, rollouts, reference_bundle=reference.bundle if reference else None,
            reference_rollouts=reference_rollouts, artifact_dir=artifact_dir,
        )
        return {task.task_id: scores.get(task.task_id) for task in tasks}

    @staticmethod
    def _common_ancestor(left: _Candidate, right: _Candidate) -> str | None:
        if left.ancestor_id and left.ancestor_id == right.ancestor_id:
            return left.ancestor_id
        return None

    def _read_lineage(self, version: str) -> dict[str, object]:
        path = self.version_root / version / ".rho-gepa-lineage.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            ancestor = data.get("ancestor")
            parents = data.get("parents", [])
            if isinstance(ancestor, str) and isinstance(parents, list) and all(isinstance(item, str) for item in parents):
                return {"ancestor": ancestor, "parents": parents}
        except (OSError, ValueError, TypeError):
            pass
        return {"ancestor": version, "parents": []}

    @staticmethod
    def _write_lineage(path: Path, candidate: _Candidate) -> None:
        (path / ".rho-gepa-lineage.json").write_text(json.dumps({
            "schema_version": "1", "candidate_id": candidate.candidate_id,
            "parents": list(candidate.parent_ids),
            "ancestor": candidate.ancestor_id or candidate.bundle.version,
        }, indent=2, sort_keys=True), encoding="utf-8")

    def _select(self, candidates: Sequence[_Candidate], elite_count: int) -> list[_Candidate]:
        def dominates(left: _Candidate, right: _Candidate) -> bool:
            pairs = [(left.task_scores[key], right.task_scores[key]) for key in left.task_scores.keys() & right.task_scores.keys() if left.task_scores[key] is not None and right.task_scores[key] is not None]
            pairs = [(float(left_score), float(right_score)) for left_score, right_score in pairs]
            return bool(pairs) and all(left_score >= right_score for left_score, right_score in pairs) and any(left_score > right_score for left_score, right_score in pairs)
        frontier = [candidate for candidate in candidates if not any(dominates(other, candidate) for other in candidates if other is not candidate)]
        rank = lambda candidate: (-(candidate.average_score if candidate.average_score is not None else float("-inf")), candidate.candidate_id)
        return sorted(frontier, key=rank)[:elite_count] + sorted([c for c in candidates if c not in frontier], key=rank)[:max(0, elite_count - len(frontier))]

    def _persist_history(self, children: Sequence[_Candidate], generation: int) -> None:
        for candidate in children:
            for module in candidate.changed_modules or self.adapter.module_names[:1]:
                self.history.append(EditHistoryRecord(f"{generation}-{candidate.candidate_id}-{module}", candidate.ancestor_id or candidate.candidate_id, module, f"{candidate.operator} {module} score={candidate.average_score}", "helpful" if (candidate.average_score or 0) >= 0 else "harmful"))

    @staticmethod
    def _history_query(target_module: str, diagnoses: Sequence[DiagnosisRecord]) -> str:
        context = [target_module]
        for diagnosis in diagnoses:
            for field in ("failure_mode", "root_cause", "fix", "evidence", "phase"):
                value = getattr(diagnosis, field, "")
                if value:
                    context.append(str(value))
        return " ".join(context)

    @staticmethod
    def _candidate_manifest(candidate: _Candidate) -> dict[str, object]:
        return {"candidate_id": candidate.candidate_id, "parents": list(candidate.parent_ids), "ancestor": candidate.ancestor_id, "operator": candidate.operator, "changed_modules": list(candidate.changed_modules), "task_scores": candidate.task_scores, "average_score": candidate.average_score, "artifact_dir": str(candidate.artifact_dir)}


def _average(scores: dict[str, float | None]) -> float | None:
    available = [score for score in scores.values() if score is not None]
    return sum(available) / len(available) if available else None
