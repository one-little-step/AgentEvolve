"""LLM-driven, editor-gated mutation and crossover operators."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

def _safe(value: object) -> object:
    """Remove prohibited field names as well as their sensitive values."""
    if isinstance(value, dict):
        return {
            key: _safe(item)
            for key, item in value.items()
            if not any(term in str(key).lower() for term in ("api_key", "token", "secret", "expected", "evaluator", "regex", "label"))
        }
    if isinstance(value, list):
        return [_safe(item) for item in value]
    if isinstance(value, str):
        return value[:4000]
    return value

from .contracts import CandidateEditor, DiagnosisRecord, EvolutionBundle, EvolutionLLM, NormalizedTrajectory
from .history import HistoryRetrieval, redact_history_value


def history_packet(retrieval: HistoryRetrieval) -> str:
    """Render prior experiments as explicit positive and negative guidance."""
    groups = {
        "Previously Helpful Changes": [],
        "Previously Harmful Or Rejected Changes": [],
        "Uncertain Or Inconclusive Experiments": [],
    }
    for record in retrieval.records:
        heading = (
            "Previously Helpful Changes" if record.outcome == "helpful"
            else "Previously Harmful Or Rejected Changes" if record.outcome in {"harmful", "rejected"}
            else "Uncertain Or Inconclusive Experiments"
        )
        groups[heading].append(f"- [{record.module}] {redact_history_value(record.text)}")
    return "\n\n".join(
        f"{heading}\n" + ("\n".join(lines) if lines else "- None retrieved")
        for heading, lines in groups.items()
    )


@dataclass(frozen=True, slots=True)
class OperatorResult:
    changed_modules: tuple[str, ...]
    raw_output: str
    skipped_edits: tuple[dict[str, object], ...]
    history_mode: str


def _phase_packet(trajectories: Sequence[NormalizedTrajectory], module: str) -> list[dict[str, object]]:
    packet: list[dict[str, object]] = []
    for trajectory in trajectories:
        for event in trajectory.events:
            if event.get("phase") in {module, module.removesuffix(".md")}:
                safe = _safe(event)
                if isinstance(safe, dict):
                    packet.append(safe)
    return packet[:20]


def _apply_edits(editor: CandidateEditor, raw: str, allowed_modules: tuple[str, ...]) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    try:
        data = json.loads(raw[raw.find("{"):raw.rfind("}") + 1])
        edits = data["edits"]
        if not isinstance(edits, list):
            raise ValueError("edits must be a list")
    except Exception as exc:  # noqa: BLE001
        return (), ({"reason": f"invalid model output: {exc}"},)
    changed: list[str] = []
    skipped: list[dict[str, object]] = []
    operation_aliases = {
        "add": "append_section",
        "append": "append_section",
        "replace": "replace_section",
        "delete": "delete_section",
    }
    for edit in edits:
        try:
            operation = operation_aliases.get(edit["operation"], edit["operation"])
            filename = edit["filename"]
            heading = re.sub(r"^#{1,6}\s+", "", str(edit.get("heading", "")).strip())
            if filename not in allowed_modules:
                raise ValueError("module is not allowed")
            if operation == "append_section":
                editor.append_section(filename, heading, str(edit.get("content", "")))
            elif operation == "replace_section":
                editor.replace_section(filename, heading, str(edit.get("content", "")))
            elif operation == "delete_section":
                editor.delete_section(filename, heading)
            else:
                raise ValueError("unsupported operation")
            changed.append(filename)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"edit": edit, "reason": str(exc)})
    return tuple(sorted(set(changed))), tuple(skipped)


def run_mutation(
    llm: EvolutionLLM,
    editor: CandidateEditor,
    parent: EvolutionBundle,
    *,
    target_module: str,
    diagnoses: Sequence[DiagnosisRecord],
    trajectories: Sequence[NormalizedTrajectory],
    history: HistoryRetrieval,
) -> OperatorResult:
    """Ask an LLM to improve one bundle from current evidence and history."""
    packet = {
        "parent": parent.modules,
        "target_module": target_module,
        "diagnoses": [_safe(asdict(d)) for d in diagnoses],
        "phase_evidence": _phase_packet(trajectories, target_module),
        "history": history_packet(history),
    }
    prompt = (
        "Mutate this advisory agent bundle using only evidence-supported changes. "
        "Do not repeat harmful history unless current evidence materially differs. "
        "Return JSON {edits:[{operation,filename,heading,content}]} using only allowed section edits.\n"
        + json.dumps(packet, default=str)
    )
    raw = llm.complete(
        "You are an offline agent-policy improver. Return only the requested JSON edits.",
        prompt,
    )
    changed, skipped = _apply_edits(editor, str(raw), tuple(parent.modules))
    editor.close()
    return OperatorResult(changed, str(raw), skipped, history.mode)


def run_crossover(
    llm: EvolutionLLM,
    editor: CandidateEditor,
    ancestor: EvolutionBundle,
    left: EvolutionBundle,
    right: EvolutionBundle,
    *,
    diagnoses: Sequence[DiagnosisRecord],
    trajectories: Sequence[NormalizedTrajectory],
    history: HistoryRetrieval,
    left_scores: dict[str, float | None],
    right_scores: dict[str, float | None],
) -> OperatorResult:
    """Ask an LLM to synthesize a child from ancestor and two evidence-scored parents."""
    packet = {
        "ancestor": ancestor.modules,
        "left_parent": left.modules,
        "right_parent": right.modules,
        "global_task_scores": {"left_parent": left_scores, "right_parent": right_scores},
        "diagnoses": [_safe(asdict(d)) for d in diagnoses],
        "failure_evidence": [_safe(event) for trajectory in trajectories for event in trajectory.events][:30],
        "history": history_packet(history),
    }
    prompt = (
        "Create an evidence-aware crossover child from the shared common ancestor and both parents. "
        "Preserve supported improvements, resolve conflicts, and avoid rejected strategies. "
        "Return JSON {edits:[{operation,filename,heading,content}]} only.\n"
        + json.dumps(packet, default=str)
    )
    raw = llm.complete(
        "You are an offline agent-policy crossover improver. Return only the requested JSON edits.",
        prompt,
    )
    changed, skipped = _apply_edits(editor, str(raw), tuple(ancestor.modules))
    editor.close()
    return OperatorResult(changed, str(raw), skipped, history.mode)
