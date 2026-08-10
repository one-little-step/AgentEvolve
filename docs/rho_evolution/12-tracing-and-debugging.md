# 12 - Tracing and Debugging

This page explains how to read the artifacts and diagnose what happened in a
round.

## Did the round complete or reject?

Check the terminal output or manifest.json:

```text
dataset/runs/evolution/<round_id>/manifest.json
```

Fields to inspect:

- status: "completed" or "rejected"
- promotion_mode: "acceptance_gate", "experimental_scored", "experimental_fallback", "experimental_unavailable"
- winner_candidate_id
- winner_version
- errors

## Why was it rejected?

Look at candidate_scores.json:

```json
[
  {
    "candidate_id": "candidate_0",
    "average_score": 0.0,
    "scores": [
      {
        "available": true,
        "normalized_score": 0.0,
        "rationale": "Both trajectories resulted in failure..."
      }
    ]
  }
]
```

If average_score is not strictly greater than ACCEPTANCE_THRESHOLD, normal mode
rejects.

## What did the optimizer change?

Read the candidate edit log:

```text
dataset/runs/evolution/<round_id>/candidates/candidate_0/edit_log.jsonl
```

Each line is a JSON object with operation, target file, heading, timestamp, and
unified diff.

## Inspect all candidates

Even after promotion, losers are archived at:

```text
dataset/runs/evolution/<round_id>/all_candidates/<round_number>/
```

## Verify parent was not mutated

Compare parent snapshots in manifest.json, or manually diff:

```bash
diff -r policies/evolved_context/base policies/evolved_context/base.bak
```

## Trace model calls

Each candidate rollout writes its trajectory to:

```text
dataset/runs/evolution/<round_id>/candidates/<candidate_id>/rollouts/<task_id>.json
```

Parent rollouts are in:

```text
dataset/runs/evolution/<round_id>/baseline/<task_id>.json
```

These contain query, final_answer, correct, status, and events.

## Re-run a candidate manually

Point a manual config at the candidate directory:

```yaml
wisdom_version: candidate_0
evolution:
  wisdom_root: dataset/runs/evolution/<round_id>/candidates
```

## Common issues

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Round rejected with score 0.0 | Candidate did not improve | Enable EXPERIMENTAL_PROMOTE_CANDIDATE or improve source runs |
| All scores unavailable | Judge parse failure | Check judge model output format |
| DPP fallback used | Kernel singular or too few records | Increase source run size or check theta/score_floor |
| `similarity_mode=handcrafted_fallback` | Summary or embedding unavailable/invalid | Inspect source trace, summary cache metadata, Ollama availability, and `fallback_reason` in selection.json |
| Sparse reconstructed summary | Older artifact lacks readable state or extractor cannot parse nested OTel fields | Inspect `agent_spans.log`, `agent_result.json`, and `agent_trajectory.txt`; check provenance in the summary |
| Summary cache miss every round | Source fingerprint or summary schema changes, or cache root is not persistent | Inspect cache metadata and configure a persistent trajectory-summary cache root |
| Embedding cache miss every round | Summary hash, embedding model, schema, or dimension differs | Check embedding metadata and keep the Ollama model configuration stable |
| Missing phase file error | Wisdom bundle is incomplete | Ensure all six phase files exist: planner, ReAct, critic, consolidator, scratchpad, and synthesis |
| Parent snapshot mismatch | Bug or external edit | Do not modify parent bundle during evolution |

## Tests to run after changes

Focused evolution tests:

```bash
uv run pytest tests/unit/test_evolution_round.py tests/integration/test_evolution_round.py tests/unit/test_evolution_selection.py tests/unit/test_evolution_trajectory_cache.py tests/unit/test_evolution_trajectory_loader.py tests/unit/test_trajectory_summary.py
```

Batch tests:

```bash
uv run pytest tests/unit/test_batch_run.py
```

All together:

```bash
uv run pytest tests/unit/test_evolution_round.py tests/integration/test_evolution_round.py tests/unit/test_evolution_selection.py tests/unit/test_evolution_trajectory_cache.py tests/unit/test_evolution_trajectory_loader.py tests/unit/test_trajectory_summary.py tests/unit/test_batch_run.py
```
