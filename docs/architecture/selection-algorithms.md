# Selection Algorithms

## Purpose

This document removes algorithmic ambiguity from `core/issues.py`,
`core/entropy.py`, and `core/pool.py`. Every formula, algorithm choice, bound,
and fallback below is mandatory. An implementation must not substitute an
alternative heuristic, and must not present a quality-ranked selector as a
diversity selector.

## Entropy Statistics

For each `(task_id, mechanism_cluster_id)` cell, maintain incremental statistics
over comparable candidate scores only:

```text
count, sum, sum_of_squares
candidate_id -> current score
maximum score and its owner
rollout counts per candidate
cluster freshness marker
```

Population variance is used:

```text
mean = sum / count
variance = max(0.0, (sum_of_squares / count) - mean * mean)
```

Entropy with a score floor:

```text
H(t, m) = variance * max(max_score, GEPA_ENTROPY_SCORE_FLOOR)
```

Evidence floors must be satisfied before a cell contributes to entropy-driven
selection:

```text
comparable candidates >= GEPA_ENTROPY_MIN_COMPARABLE_CANDIDATES (default 3)
rollouts per candidate >= GEPA_ENTROPY_MIN_ROLLOUTS_PER_CANDIDATE (default 2)
```

A cell failing the floors is marked `entropy_unavailable` with a reason and falls
back to severity/coverage quality. It must never contribute a high-variance
signal derived from a single sample.

Update complexity is `O(1)` per score, except when the previous maximum owner is
updated downward, which requires a bounded rescan of that cell's candidate map.
Priority uses a max-heap keyed by entropy with lazy invalidation:

```text
push/update: O(log N)
top-K: O(K log N)
```

In parallel mode, statistics and heap updates occur only at a successful
coordinator barrier, never inside a worker.

## Issue Quality

Issue quality is a bounded product-free weighted sum in `[0, 1]`:

```text
quality(i) =
    w_severity   * severity(i)
  + w_confidence * confidence(i)
  + w_entropy    * normalized_entropy(i)
  + w_coverage   * coverage_need(i)
  + w_pareto     * pareto_relevance(i)
```

Requirements:

- Weights are configuration, sum to `1.0`, and are recorded in the manifest.
- `normalized_entropy(i)` is min-max normalized within the current candidate
  issue set; if entropy is unavailable for a cell, the term contributes `0.0` and
  the fallback reason is recorded.
- Frontier-exploration cells, where variance is meaningful but maximum score is
  below `GEPA_ENTROPY_RECOMBINATION_SCORE_THRESHOLD`, are multiplied by
  `GEPA_ENTROPY_FRONTIER_WEIGHT` rather than discarded.
- Any issue lacking trace-backed artifact attribution is rejected before ranking.

## Hard Constraints Before Selection

Applied as filters, not as scoring penalties:

```text
reject issues without an attributable, inventory-declared writable artifact
reject duplicate (parent_candidate, write_set) pairs within one batch
reject overlapping write sets within one batch
reject exhausted (issue_fingerprint, artifact_group, lineage) contexts
cap issues per mechanism cluster per batch
reserve capacity for under-covered task/mechanism regions
```

## DPP Selection: Mandated Algorithm

Similarity uses cosine similarity over issue embeddings, clamped to `[0, 1]`:

```text
sim(i, j) = max(0.0, cosine_similarity(e_i, e_j))
```

The kernel is quality-weighted:

```text
L[i][j] = quality(i) * sim(i, j) * quality(j)
L[i][i] = quality(i)^2 + JITTER      # JITTER default 1e-9
```

Selection uses **greedy MAP inference with Cholesky-style incremental
log-determinant updates**. This is mandatory:

```text
1. Prefilter to at most GEPA_DPP_MAX_ITEMS (default 100) items using the entropy
   heap and quality ranking. Record the prefilter threshold and item counts.
2. Initialize d[i]^2 = L[i][i] for all candidates.
3. Repeat until k items are selected or no positive gain remains:
     select j = argmax_i log(d[i]^2) over remaining items
     if d[j]^2 <= GEPA_DPP_MIN_GAIN (default 1e-12): stop
     append j to the selected set
     for each remaining i:
         e = (L[i][j] - dot(c[i], c[j])) / d[j]
         c[i].append(e)
         d[i]^2 = d[i]^2 - e * e
4. Break ties by ascending stable issue/attempt ID for determinism.
```

Forbidden implementations:

- Exact eigendecomposition or dense kernel factorization when `N > 100`.
- Any selector that adds similarity to quality, which rewards redundancy.
- Any selector that ignores `sim` and returns top-K by quality while being named
  or documented as DPP.
- Any selector whose output depends on unseeded randomness.

Marginal-gain semantics: selecting an item must reduce the remaining marginal
gain of similar items. A required test asserts that, given two near-duplicate
high-quality issues and one dissimilar high-quality issue, the selector returns
one duplicate plus the dissimilar issue.

Hierarchical selection is mandatory to bound cost:

```text
Stage 1: select tasks using aggregate task entropy and task embeddings.
Stage 2: within each selected task, select mechanism clusters.
```

Alternate modes remain available for ablations and must be recorded:

```text
GEPA_ISSUE_SELECTION_MODE = dpp | severity_rank | random
severity_rank: order by (severity, confidence, entropy availability, stable ID)
random: seeded deterministic RNG, seed recorded in manifest
```

## Embedding Fallback

If the configured embedding provider is unavailable or returns an unexpected
dimension, the selector uses a deterministic lexical similarity fallback only
when the active profile permits it. The manifest records `embedding_fallback`,
the reason, and the affected selection. Silent substitution is forbidden.

## Pareto Dominance

For candidates `a` and `b` on task `t`, using comparable cells only:

```text
Let M = mechanism clusters with comparable evaluated cells for both a and b.
If M is empty: neither dominates.

a dominates b iff:
  for all m in M: weighted(a, m) >= weighted(b, m) - EPSILON
  and exists m in M: weighted(a, m) > weighted(b, m) + EPSILON
  and no protected floor for (t, m) is violated by a

weighted(c, m) = score(c, t, m) * severity(t, m) * confidence(c, t, m)
```

Rules:

- Mechanisms present for only one candidate are excluded from `M`, never treated
  as zero.
- `EPSILON` is configuration, defaults to `1e-9`, and is recorded.
- Comparison coverage, that is `|M|` and excluded-cell counts, is retained with
  the result.

## Parent Sampling

```text
frequency(c) = sum over (t, m) of
    severity(t, m) * confidence(c, t, m) * indicator[c wins (t, m)]
```

A candidate wins `(t, m)` when it holds the strict maximum comparable weighted
score for that cell. Ties award all tied winners. Sampling is proportional to
`frequency(c)` using a seeded RNG, with the seed recorded.

## Aggregation Key Rule

Every aggregation over tasks, candidates, mechanisms, or artifacts uses the
complete identifier string as the dictionary key. Using a character, prefix,
slice, truncation, or derived hash of an ID as a grouping key is a defect. A
required test asserts that `task-a` and `test-b` never aggregate together.

## Champion Selection

```text
aggregate(c) =
    alpha * Outcome(c)
  + beta  * ProcessCoverage(c)
  + gamma * Stability(c)
  - delta * RegressionRisk(c)
```

Defaults: `alpha=0.55`, `beta=0.20`, `gamma=0.15`, `delta=0.10`, all recorded in
the manifest. Protected floors are disqualifying and evaluated before the
aggregate. Candidates lacking the configured minimum comparison coverage are
reported separately and cannot win by default. The manifest records every
component, coverage figure, tie-breaker, and the disqualification list.
