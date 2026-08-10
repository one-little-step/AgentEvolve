# 04 - Coreset Selection

Coreset selection chooses the small set of historical trajectories that the
round will diagnose, rerun, and judge. The default strategy is DPP, which
balances failure-biased difficulty against semantic and structural diversity.

## Location

agent/gaia_lg_react/evolution/selection.py

The detailed current algorithm and mathematics are in
[selection_algo_explaination.md](selection_algo_explaination.md).

## Available selectors

| Selector | Method | Use case |
|----------|--------|----------|
| dpp | Greedy DPP MAP approximation on quality-diversity kernel | Default; balances hard failures with diversity |
| difficulty | Sort by quality score descending | Focus on hardest failures |
| coverage | Greedy distance maximization | Maximize behavioral diversity |
| random | Seeded random sampling | Baseline or ablation |

## Semantic-summary preparation

Before selection, `EvolutionRound` loads each task's captured
`trajectory_summary.md` or reconstructs one from historical OTel/result
artifacts. `TrajectoryCache` embeds the bounded summary with local Ollama
`embeddinggemma`, caches it by source run and task ID, and supplies prepared
vectors to the selector. The selector never calls Ollama itself.

When multiple source runs are combined, selection uses
`<source-run>::<task-id>` to avoid collisions; inference reruns still use the
original task ID.

When all valid candidates have compatible prepared vectors, similarity is:

```text
0.85 * semantic cosine similarity
+ 0.15 * structured feature cosine similarity
```

If summaries or vectors are unavailable, the selector records
`handcrafted_fallback` and uses its prior deterministic feature-only
similarity.

## Structured feature construction

For each trajectory, _build_digest produces:

```python
{
    "task_id": "...",
    "status": "success" or "failure",
    "correct": True or False,
    "event_names": ["web_search", "calculator", ...],
    "event_count": N,
    "query_fingerprint": "sorted unique words",
    "answer_metadata": {"length": L, "has_number": True, "normalized": "..."},
    "metadata_keys": [...],
    "source_count": N,
}
```

Credentials and evaluator-only content are redacted before summaries,
fingerprints, embeddings, and feature digests are used.

_records_to_features turns these digests into a numeric matrix with:

- one-hot status vector
- one-hot event vector
- one-hot metadata-key vector
- binary query-word vector
- answer length and has-number
- numeric fields: correct, event_count, source_count

Each column is normalized to unit variance.

## Quality scores

_quality_scores rewards:

- incorrect trajectories (+1.0)
- failure status (+0.5)
- rich event structure (+0.05 per event, capped at 20)

Then the score is floored and normalized:

```python
floored = np.maximum(raw_quality / 2.5, score_floor)
normalized = floored / max(np.max(floored), score_floor)
```

## DPP kernel

The kernel is built as:

```python
similarity = 0.85 * semantic_similarity + 0.15 * structured_similarity
quality_diag = np.diag(quality)
kernel = quality_diag @ similarity @ quality_diag + jitter
```

The diagonal carries quality, while the off-diagonals carry blended semantic and
structural similarity. A small jitter is added for positive definiteness. The
handcrafted fallback maps structured cosine to `exp(-(1 - cosine))` before
building the same kernel.

## DPP sampling

_dpp_sample is a greedy Cholesky-based MAP approximation:

1. Start with the item that has the highest diagonal kernel value.
2. Maintain the Cholesky factor L of the selected sub-kernel.
3. Repeatedly pick the item that maximizes the Schur complement gain.
4. Update L incrementally.

This is deterministic. The seed parameter exists for future stochastic
variants but is unused by the DPP path.

## Difficulty weighting

Theta controls how much the DPP weights quality versus diversity:

```python
alpha = theta / (2.0 * max(1.0 - theta, 1e-6))
quality = normalized**alpha if theta < 1.0 else normalized
```

- theta near 0: nearly uniform quality, diversity dominates
- theta near 1: quality dominates
- theta = 0.7: balanced

## Score floor

score_floor ensures very low-quality records do not receive near-zero weights,
which would make the kernel degenerate. It must be > 0.

## Fallbacks

The DPP path falls back to deterministic quality-sorted selection when:

- fewer than two valid records
- missing, malformed, or incompatible prepared embeddings
- all records have identical handcrafted features when semantic similarity is unavailable
- the kernel is numerically singular (condition number > 1e12)
- any exception during kernel construction or sampling

The fallback reason is recorded in selection.json.

## Output format

selection.json:

```json
{
  "selected_ids": ["gaia-ec09fa32", "gaia-8e867cd7"],
  "method": "dpp",
  "requested_size": 2,
  "valid_count": 5,
  "fallback_reason": null,
  "similarity_mode": "semantic_summary",
  "summary_cache_hits": 4,
  "embedding_cache_hits": 4,
  "max_trajectory_workers": 10,
  "embedding_model": "embeddinggemma",
  "selector": "dpp",
  "theta": 0.7,
  "score_floor": 0.1,
  "seed": 0
}
```
