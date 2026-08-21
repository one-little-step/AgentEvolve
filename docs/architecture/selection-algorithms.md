# Selection Algorithms

## Purpose

This document removes algorithmic ambiguity from `core/issues.py`,
`core/entropy.py`, and `core/pool.py`. Every formula, algorithm choice, bound,
and fallback below is mandatory. An implementation must not substitute an
alternative heuristic.

DPP selection optimizes **quality and diversity together**. A selector that
maximizes only one of the two is a different, separately named mode and must not
be presented as DPP. The kernel construction, quality normalization, and
`theta` balance follow the RHO selection specification in
[selection_algo_explaination.md](../rho_evolution/selection_algo_explaination.md),
which remains the authoritative mathematical reference.

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

DPP optimizes **quality and diversity jointly**. Quality enters the kernel
diagonal, diversity enters the off-diagonals, and `theta` controls the balance.
Neither objective may be dropped: a quality-only selector is `severity_rank`, and
a diversity-only selector is `coverage`. Both exist separately as ablations.

Raw issue quality combines the available evidence signals:

```text
raw_quality(i) =
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

Quality is then floored and normalized, following the RHO construction in
[selection_algo_explaination.md](../rho_evolution/selection_algo_explaination.md):

```text
floored(i)    = max(raw_quality(i), GEPA_DPP_SCORE_FLOOR)     # floor default 0.1
normalized(i) = floored(i) / max_j floored(j)
```

The floor is strictly positive. It prevents low-quality items from collapsing to
near-zero weight, which would make the kernel numerically degenerate.

## Theta: Quality Versus Diversity Balance

`theta` scales how strongly quality influences selection, without altering
embeddings or similarity:

```text
alpha    = theta / (2 * max(1 - theta, 1e-6))
quality(i) = normalized(i) ** alpha        if theta < 1.0
quality(i) = normalized(i)                 if theta == 1.0
```

`theta == 1.0` is special-cased to avoid an unbounded exponent.

| `theta` | Selection tendency |
| ---: | --- |
| 0.0 | Quality effectively uniform; diversity dominates |
| 0.3 | Diversity first, modest preference for high-value issues |
| 0.7 | Default balance of value and coverage |
| 0.9 | Stronger focus on high-value issues |
| 1.0 | Quality-oriented limit |

Default: `GEPA_DPP_THETA=0.7`, giving `alpha ≈ 1.167`. The resolved `theta`,
`alpha`, and score floor are mandatory manifest fields.

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

DPP jointly maximizes quality and diversity. The selection objective is:

```text
argmax over |Y| = k of det(L_Y)
```

Similarity uses cosine similarity over issue embeddings, clamped to `[0, 1]`:

```text
sim(i, j) = clip(cosine_similarity(e_i, e_j), 0.0, 1.0)
```

Embeddings are L2-normalized before use. A structural similarity signal may be
blended when the profile declares it, following the RHO construction:

```text
sim(i, j) = clip(w_sem * semantic(i, j) + w_struct * structural(i, j), 0.0, 1.0)
```

with defaults `w_sem = 0.85` and `w_struct = 0.15`, both recorded.

The kernel carries quality on the diagonal and quality-weighted similarity off
the diagonal:

```text
L = Q S Q + JITTER * I          # Q = diag(quality), JITTER default 1e-6
L[i][j] = quality(i) * sim(i, j) * quality(j)
L[i][i] = quality(i)^2 + JITTER
```

Why this is a joint objective, not a similarity penalty: for a pair,

```text
det([[q_i^2, q_i q_j s], [q_i q_j s, q_j^2]]) = q_i^2 * q_j^2 * (1 - s^2)
```

High quality raises the determinant; high similarity lowers it. Both terms are
required. An implementation that maximizes only `q` or only `(1 - s)` is not a
DPP and must not be labelled one.

Selection uses **greedy MAP inference with Cholesky-style incremental
Schur-complement updates**. This is mandatory:

```text
1. Prefilter to at most GEPA_DPP_MAX_ITEMS (default 100) items using the entropy
   heap and quality ranking. Record the prefilter threshold and item counts.
2. Initialize gain[i] = L[i][i] for all candidates; c[i] = [] for all candidates.
3. Repeat until k items are selected or no candidate has sufficient gain:
     select j = the remaining item with maximum gain[i],
              breaking ties by ascending stable issue/attempt ID
     if gain[j] <= GEPA_DPP_MIN_GAIN (default 1e-12): stop
     append j to the selected set and remove it from the remaining set
     d_j = sqrt(gain[j])
     for each remaining i:
         e = (L[i][j] - dot(c[i], c[j])) / d_j
         c[i].append(e)
         gain[i] = gain[i] - e * e        # selecting j reduces redundant gain
4. Return the selected set in selection order.
```

Notes:

- The first pick maximizes `L[i][i]`, which is the highest quality-weighted item.
- Compare gains directly. Do not take a logarithm of the gain, since gains reach
  zero or small negative values from floating-point error and `log` is undefined
  there.
- Clamp `gain[i]` at `0.0` after each update to absorb floating-point drift.

Forbidden implementations:

- Exact eigendecomposition or dense kernel factorization when `N > 100`.
- Any selector that adds similarity to quality, which rewards redundancy.
- Any selector that ignores `sim` and returns top-K by quality while being named
  or documented as DPP.
- Any selector that ignores `quality` and returns a pure farthest-first set while
  being named or documented as DPP.
- Any selector whose output depends on unseeded randomness.

Required behavioral tests:

```text
test_dpp_penalizes_similarity_and_promotes_diversity
    Two near-duplicate high-quality issues plus one dissimilar high-quality
    issue, k=2: the dissimilar issue is selected and both duplicates are not.

test_dpp_prefers_quality_among_equally_diverse_items
    Three mutually dissimilar issues with distinct quality, k=1: the highest
    quality issue is selected.

test_dpp_theta_shifts_quality_diversity_balance
    Raising theta increases selection of high-quality near-duplicates; lowering
    theta increases distinct-family coverage.

test_dpp_is_deterministic
    Identical input and configuration produce identical selections.
```

Hierarchical selection is mandatory to bound cost:

```text
Stage 1: select tasks using aggregate task entropy and task embeddings.
Stage 2: within each selected task, select mechanism clusters.
```

Alternate modes remain available for ablations and must be recorded:

```text
GEPA_ISSUE_SELECTION_MODE = dpp | severity_rank | coverage | random
dpp:           joint quality and diversity (default)
severity_rank: quality only; order by (severity, confidence, entropy availability, stable ID)
coverage:      diversity only; greedy farthest-first over embeddings
random:        seeded deterministic RNG, seed recorded in manifest
```

## Degenerate-Kernel Fallback

The DPP path falls back to deterministic quality-ordered selection, recording the
reason, when any of the following holds:

```text
fewer than two valid candidate issues
missing, malformed, or dimension-incompatible embeddings
non-positive eigenvalues or condition number above 1e12
an exception during kernel construction or greedy selection
```

The fallback orders by descending quality and breaks ties by stable ID. Silent
fallback is forbidden; the manifest records `fallback_reason`.

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

Ranking is **pairwise on shared evidence**, not by a weighted aggregate. Eligible
candidates are compared king-of-the-hill in insertion order, and an incumbent is
displaced only by a challenger that scores better over the `(task, mechanism)`
cells *both* candidates measured, at or above `min_comparable_rollouts`. A tie, a
loss, or an empty overlap all leave the incumbent standing. Genuine ties between
comparable candidates break by ascending `candidate_id`.

The manifest still reports this aggregate:

```text
aggregate(c) =
    alpha * Outcome(c)
  + beta  * ProcessCoverage(c)
  + gamma * Stability(c)
  - delta * RegressionRisk(c)
```

Defaults: `alpha=0.55`, `beta=0.20`, `gamma=0.15`, `delta=0.10`, all recorded in
the manifest. **It is a diagnostic, not a ranking key** — no weight can change
which candidate wins. Three findings forced that:

- **SV-2** — `Outcome` averaged over whatever cells each candidate happened to
  measure, so *not attempting* a hard task raised a candidate's score. Comparison
  is now restricted to the shared-cell intersection.
- **SV-3** — `ProcessCoverage` measures how much was measured, not how good the
  candidate is, and held 27% of the live weight, so a strictly worse candidate
  could win on breadth alone. Coverage is now an **eligibility floor**
  (`champion_min_coverage_fraction`), which *is* enforced.
- **SV-5** — `Stability` is hardcoded `1.0` and `RegressionRisk` hardcoded `0.0`.
  Neither was ever implemented, so they cancel in every comparison and `gamma`
  and `delta` weight nothing.

Protected floors are disqualifying and evaluated before any comparison, as are
retired candidates and candidates failing the `S_j > 0` acceptance gate. The
manifest records every component, the coverage figure, the shared-cell count
behind the decision (`comparable_cells`), the tie-breaker, and the
disqualification list.
