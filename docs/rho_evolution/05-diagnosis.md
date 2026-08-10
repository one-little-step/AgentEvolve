# 05 - Diagnosis

Diagnosis explains why each selected trajectory failed or underperformed, using
the parent wisdom bundle as context. The diagnosis drives candidate generation.

## Location

agent/gaia_lg_react/evolution/prompts.py, function diagnose_trajectory.

## Input

The prompt contains:

1. A trajectory digest from _build_digest.
2. A truncated summary of the parent wisdom bundle.

Both are credential-free. The digest captures status, correctness, event names,
answer metadata, and normalized text fingerprints.

## Prompt instruction

```text
Diagnose the following Gaia trajectory and propose a targeted wisdom improvement.

Return a single JSON object with exactly these keys:
- severity
- failure_mode
- phase
- evidence
- wisdom_direction
- root_cause
- fix
```

Allowed phases: planner, react, critic, consolidator, scratchpad, synthesis, cross_phase.

## Output

A Diagnosis dataclass. If the model output is not valid JSON, a parse_failure
diagnosis is returned and the raw output is preserved.

## Example diagnosis

```json
{
  "severity": "high",
  "failure_mode": "trivial_output_synthesis",
  "phase": "synthesis",
  "evidence": "answer_metadata shows length=1, has_number=true, normalized=\"0\" while correct=false; source_count=4 but final answer is just a single token.",
  "wisdom_direction": "In synthesis.md add a guard that validates the synthesized answer is non-trivial.",
  "root_cause": "The planner mis-parsed the garbled query fingerprint; synthesis collapsed data into a placeholder token.",
  "fix": "Add answer-richness checks in synthesis and strengthen intent parsing."
}
```

## How diagnoses are used

All diagnoses for the selected coreset are passed to optimize_candidate. The
optimizer sees the parent bundle and the JSON array of diagnoses, then proposes
wisdom edits that address the aggregate failure patterns.

## Persistence

Diagnoses are written to:

```text
dataset/runs/evolution/<round_id>/diagnoses.json
```

Each entry maps a task_id to its diagnosis dictionary.
