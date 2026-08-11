# 03 - Control Flow

This page walks through a complete evolution round step by step. All code paths
live in agent/gaia_lg_react/evolution/round.py unless otherwise noted.

## Entry point

The runner is dataset/evolve_run.py. It builds a progressive plan and calls
EvolutionRound.run() once per round.

```python
plan = [
    (1, "base", "rho-gaia-1"),
    (2, "rho-gaia-1", "rho-gaia-2"),
    ...
]
```

If any round is rejected, the chain stops unless experimental promotion is
enabled.

## Round phases

### 1. Create round directory

```text
dataset/runs/evolution/<YYYYMMDD_HHMMSS_xxxxxx>/
```

A unique round ID is generated from UTC timestamp plus a random suffix. This
directory will hold every artifact produced by the round.

### 2. Load and validate parent bundle

```python
parent_bundle = WisdomBundle.load(wisdom_root, parent_version)
parent_bundle.validate()
```

The parent is loaded read-only and snapshotted with MD5 hashes. The snapshot is
compared after the round to detect accidental mutation.

### 3. Load source trajectory runs

```python
loader = TrajectoryRunLoader(dataset_runs_root)
all_records = []
for source_run in source_runs:
    records, report = loader.load(source_run)
    all_records.extend(records)
```

TrajectoryRunLoader recursively discovers:

```text
agent_spans.log
trajectory_summary.md
*.json
```

It normalizes task IDs, merges records that share an ID, redacts credentials and
evaluator-only content, and returns a parse report. A captured
`trajectory_summary.md` is preferred. Otherwise it reconstructs a bounded,
verbose semantic summary from nested OTel state/messages plus result artifacts.
The report is written to source_trajectory_run.json.

### 4. Prepare semantic summaries and embeddings

Before selection, RHO prepares each source-run/task pair in a bounded pool of
`max_trajectory_workers` workers, default 10. This limit is independent from
fresh baseline/candidate rollout worker limits.

```text
TrajectoryRecord
  -> captured or reconstructed trajectory_summary.md
  -> summary cache validation
  -> local embeddinggemma embedding cache validation
  -> prepared vector keyed by selection identity
```

Default cache roots are sibling directories under the dataset-runs root:

```text
cache_trajectory_summaries/<source-run>/<task-id>/
cache_trajectory_embeddings/<source-run>/<task-id>/
```

Cache entries validate source/summary hashes, schema versions, embedding model,
and vector dimensions. Missing or stale entries are regenerated with atomic
per-task writes. One preparation failure does not stop the round; that record
can use handcrafted selection similarity.

### 5. Coreset selection

```python
selection = select_coreset(
    valid_records,
    coreset_size,
    seed=seed,
    selector=selector,
    theta=theta,
    score_floor=score_floor,
    prepared_embeddings=prepared_vectors,
)
```

When compatible summary embeddings exist, DPP uses semantic summary similarity
blended with structural features. Otherwise it uses the deterministic
handcrafted path. See 04-coreset-selection.md for the operational overview and
selection_algo_explaination.md for detailed math. The result is written to
selection.json.

### 6. Baseline rollout

```python
base_config = _make_base_config(parent_version)
parent_trajectories = self._run_baseline(selected_records, base_config, ...)
```

The parent harness is run on every selected task. Each result is converted to a
TrajectoryRecord and saved under:

```text
dataset/runs/evolution/<round_id>/baseline/<task_id>.json
```

This is a fresh rollout, not a replay of the historical trace.

### 7. Diagnosis

```python
diagnoses = self._diagnose_selected(selected_records, parent_bundle, errors)
```

Each selected historical trajectory is diagnosed against the parent bundle. The
LLM returns a structured diagnosis. Diagnoses are saved to diagnoses.json.

### 8. Candidate generation

For each candidate index from 0 to candidate_count - 1:

```text
1. Create candidate directory
2. Materialize parent bundle into it
3. Call optimize_candidate(...) with diagnoses
4. The optimizer proposes edits to intent_planner.md, reAct.md, critic.md, consolidator.md, scratchpad.md, or synthesis.md
5. Edit registry applies allowed edits and writes edit_log.jsonl
6. Validate the resulting bundle
```

If evolution_tools_enabled is False, the candidate is just a copy of the parent.

### 9. Candidate rollout

```python
candidate_trajectories = self._run_candidate(
    candidate_id, candidate_dir, selected_records, candidate_config, ...
)
```

The candidate harness is run on the same selected tasks. Results are saved to:

```text
dataset/runs/evolution/<round_id>/candidates/<candidate_id>/rollouts/<task_id>.json
```

### 10. Pairwise scoring

```python
scores = self._score_candidate(
    candidate_id, candidate_trajectories, parent_trajectories, errors
)
```

For each selected task, the parent rollout and candidate rollout are compared
by the judge model. See 07-pairwise-judging.md.

### 11. Archive all candidates

```python
all_candidates_dir = round_dir / "all_candidates" / str(round_number)
```

Every candidate directory, including edit logs and rollouts, is copied into the
archive before promotion. This lets you inspect losers even after a winner is
chosen.

### 12. Winner selection

Normal mode:

```python
winner_index = _pick_winner(candidate_results, acceptance_threshold)
```

Requires average_score > acceptance_threshold.

Experimental mode:

```python
winner_index, promotion_mode = _pick_experimental_winner(candidate_results)
```

Promotes the highest-scoring candidate even with non-positive scores, or falls
back to candidate_0 when no scores are available.

### 13. Materialization or rejection

If a winner is found:

```text
1. Load winner bundle
2. Create version directory: policies/evolved_context/<target_version>/
3. Materialize files
4. Write winner.json and round manifest.json
5. Write version manifest.json inside the version directory
6. Verify parent snapshot unchanged
```

If no winner passes:

```text
1. Write manifest.json with winner_candidate_id=null
2. Return status="rejected"
```

## Call graph summary

```text
dataset/evolve_run.py
  -> EvolutionRound.run
       -> WisdomBundle.load (parent)
       -> TrajectoryRunLoader.load (source runs)
       -> TrajectoryCache.prepare_many (summary/embedding cache)
       -> select_coreset
       -> _run_baseline
       -> _diagnose_selected
            -> diagnose_trajectory (prompts.py)
       -> for each candidate:
            -> WisdomBundle.materialize
            -> optimize_candidate (prompts.py)
                 -> WisdomEditRegistry.append/replace/delete
            -> _run_candidate
            -> _score_candidate
                 -> pairwise_preference (prompts.py)
       -> _pick_winner / _pick_experimental_winner
       -> WisdomBundle.materialize (winner)
       -> write manifest.json, winner.json, version manifest
```
