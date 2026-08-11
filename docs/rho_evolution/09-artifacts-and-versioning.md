# 09 - Artifacts and Versioning

Every round produces a complete audit trail. This page describes the directory
layout and what each file contains.

## Round directory

```text
dataset/runs/evolution/<YYYYMMDD_HHMMSS_xxxxxx>/
```

Created at the start of the round and never deleted.

## Files inside a round directory

```text
manifest.json              # round-level operational manifest
source_trajectory_run.json # parse report for the loaded source runs
selection.json             # selected task IDs and selector parameters
diagnoses.json             # one diagnosis per selected task
candidate_scores.json      # scores for every candidate
winner.json                # winner metadata (only if accepted)
baseline/                  # parent rollouts on selected tasks
  <task_id>.json
candidates/                # per-candidate bundles and rollouts
  candidate_0/
    intent_planner.md
    reAct.md
    critic.md
    consolidator.md
    scratchpad.md
    synthesis.md
    edit_log.jsonl
    rollouts/
      <task_id>.json
  candidate_1/
    ...
all_candidates/            # archive copy of all candidates before promotion
  <round_number>/
    candidate_0/
    candidate_1/
    ...
```

## Source summary and embedding caches

Outside each round directory, RHO maintains reusable source-artifact caches:

```text
<dataset-runs-root>/
  cache_trajectory_summaries/<source-run>/<task-id>/
    trajectory_summary.md
    metadata.json
  cache_trajectory_embeddings/<source-run>/<task-id>/
    embedding.npy
    metadata.json
```

Summary metadata records summary schema version, redacted source fingerprint,
summary SHA-256, and provenance. Embedding metadata records embedding schema
version, summary SHA-256, embedding model, and vector dimension. The cached
content itself is rehashed on read. Invalid or stale cache entries are rebuilt
using unique temporary files followed by atomic replacement.

## Round manifest.json

Contains:

```text
round_id
run_name
parent_version
model
candidate_count
coreset_size
selector, theta, score_floor, seed
acceptance_threshold
judge_model
group_rollouts_per_task
evaluation_timeout_seconds
cache_mode, cache_dir
evolution_tools_enabled
experimental_promote_candidate
promotion_mode
selected_ids
selection_method
similarity_mode
summary_cache_hits, embedding_cache_hits
max_trajectory_workers
winner_candidate_id
winner_version
errors
created_at
```

## winner.json

Only present on acceptance:

```json
{
  "winner_candidate_id": "candidate_0",
  "version": "rho-gaia-1",
  "average_score": 0.5,
  "scores": [...]
}
```

## Version directory

When accepted, the winner is copied to:

```text
policies/evolved_context/<version>/
  intent_planner.md
  reAct.md
  critic.md
  consolidator.md
  scratchpad.md
  synthesis.md
  manifest.json   # version-level EvolutionManifest
```

The version manifest captures:

```text
version
parent_version
source_digest
model_identifier
candidate_count
coreset_size
artifact_paths back to the round directory
```

## Parent immutability

Before the round, the parent directory is snapshotted:

```python
parent_snapshot = _dir_snapshot(parent_dir)
```

After the round:

```python
parent_dir_snapshot_after = _dir_snapshot(parent_dir)
```

The EvolutionResult contains both snapshots. They should be identical.

## Candidate archive

Before selecting a winner, every candidate directory is copied to:

```text
all_candidates/<round_number>/<candidate_id>/
```

This preserves losers for later analysis even after a winner is materialized.
