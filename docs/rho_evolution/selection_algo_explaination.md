# RHO Coreset Selection Algorithm

This document explains the coreset-selection algorithm currently used by the
Gaia RHO evolution pipeline. It describes what is implemented today, not a
future design.

RHO does not diagnose or rerun every historical trajectory. Instead, it picks a
small coreset that should contain both:

1. trajectories that are valuable to improve, especially failures; and
2. trajectories that represent different task, strategy, and failure patterns.

The default selector is a deterministic greedy approximation of a
Determinantal Point Process (DPP) maximum-a-posteriori selection.

## End-To-End Flow

```text
historical Gaia run artifacts
  -> load one TrajectoryRecord per task
  -> load or reconstruct trajectory_summary.md
  -> cache summary and embedding by source run and task ID
  -> embed summary with local embeddinggemma
  -> build quality-weighted similarity kernel
  -> greedy DPP MAP selection
  -> diagnose and rerun only selected trajectories
```

The main implementation files are:

| File | Responsibility |
| --- | --- |
| `agent/gaia_lg_react/evolution/trajectory_loader.py` | Loads historical result, span, and summary artifacts. |
| `agent/gaia_lg_react/trajectory_summary.py` | Builds/reconstructs bounded semantic trajectory summaries. |
| `agent/gaia_lg_react/evolution/trajectory_cache.py` | Caches summaries and `embeddinggemma` vectors. |
| `agent/gaia_lg_react/evolution/selection.py` | Builds the DPP kernel and performs greedy selection. |
| `agent/gaia_lg_react/evolution/round.py` | Orchestrates loading, cache preparation, selection, diagnosis, reruns, and judging. |

## What Is A Candidate?

One candidate is one historical Gaia task trajectory. A candidate record contains
the final outcome plus the preserved behavior observed while solving the task.

The evolution loader uses these sources when available:

| Artifact | Main purpose |
| --- | --- |
| `trajectory_summary.md` | Compact semantic representation; preferred input to embedding. |
| `result.json` / `agent_result.json` | Outcome, correctness/pass status, final answer, runtime data, explicit failure events. |
| `agent_spans.log` | Planner/intent state, ReAct/tool behavior, scratchpad snapshots, critic data, and phase traces. |
| `agent_trajectory.txt` | Readable fallback chronology for older runs. |

For historical runs that do not have a native `trajectory_summary.md`, RHO
deterministically reconstructs one from the available result and trace artifacts.
No summary LLM call is required for historical reconstruction.

## Selection IDs And Multiple Source Runs

An evolution round can combine multiple source runs. Different runs can contain
the same original GAIA task ID, so task ID alone cannot identify a candidate in a
combined coreset.

For a single source run, RHO keeps the original task ID:

```text
gaia-73c1b9fe
```

For multiple source runs, RHO uses this selection identity:

```text
<source-run>::<task-id>
```

For example:

```text
gaia10_luna_batch1::gaia-73c1b9fe
gaia10_luna_batch2::gaia-73c1b9fe
```

The original task ID remains attached to the trajectory. Gaia uses it when
rerunning the task; the source-qualified ID is only used to avoid coreset and
embedding collisions.

## Semantic Summary Input

The semantic embedding input is the complete bounded `trajectory_summary.md`.
It may contain:

- terminal outcome and runtime profile;
- grouped, normalized tool and execution failures;
- intent and planner plan;
- high-value ReAct and tool timeline entries;
- scratchpad snapshots, including pre-compaction/final context;
- critic review and recovery outcome;
- an optional causal narrative for failed or unresolved new runs;
- artifact provenance and omissions.

The summary excludes evaluator expected answers, evaluator regexes, credentials,
raw web page bodies, full raw tool outputs, signed URLs, and raw benchmark
labels. Secret-like values are redacted before summary hashing, caching,
embedding, or optional LLM narrative construction.

The inference engine targets a 10,000-token soft summary budget and enforces a
12,000-token hard limit. A fresh negative or unresolved run may use one optional
summary LLM call. Its carefully prioritized input has a 25,000-token target and
a 30,000-token hard cap.

## Cache Preparation

Before DPP selection, RHO prepares summaries and embeddings. This work is
separate from inference rollout workers and is controlled by:

```python
max_trajectory_workers
```

The hardcoded progressive runner in `dataset/evolve_run.py` defaults it to 10.
The valid range is 1 through 10.

Default cache layout:

```text
<dataset-runs-root>/
  cache_trajectory_summaries/
    <source-run>/<task-id>/
      trajectory_summary.md
      metadata.json
  cache_trajectory_embeddings/
    <source-run>/<task-id>/
      embedding.npy
      metadata.json
```

Summary cache metadata validates:

- summary schema version;
- fingerprint of redacted source trajectory content;
- SHA-256 of the actual summary content;
- provenance: captured or reconstructed.

Embedding cache metadata validates:

- embedding schema version;
- SHA-256 of the summary being embedded;
- embedding model identifier;
- vector dimension.

Stale, unreadable, incomplete, tampered, or incompatible cache entries are cache
misses. Workers use unique temporary sibling files followed by atomic rename, so
one task cache entry is not partially written.

## Embedding Vectors

For each candidate summary, RHO uses the existing local Ollama
`embeddinggemma` provider. Let the raw embedding of candidate \(i\) be:

\[
e_i \in \mathbb{R}^{d}
\]

The implementation L2-normalizes every nonzero vector:

\[
\bar{e_i} = \frac{e_i}{\|e_i\|_2}
\]

The semantic cosine similarity of trajectories \(i\) and \(j\) is then:

\[
S^{semantic}_{ij} = \bar{e_i}^{T}\bar{e_j}
\]

Cosine values are clamped to \([-1, 1]\) for numerical safety.

Two trajectories that have different wording but share the same failure pattern,
such as web-source rate limiting followed by unsupported synthesis, should be
close in this semantic space. Two unrelated patterns, such as a Python
calculation error and a critic evidence-conflict rejection, should be farther
apart.

## Structured Features

Embeddings are the primary similarity signal, but RHO retains a smaller,
deterministic structural signal. For every trajectory, it builds a feature vector
from:

- one-hot encoded terminal status;
- one-hot encoded event/span names;
- one-hot encoded metadata keys;
- normalized query-word presence;
- final-answer length;
- whether the final answer contains a number;
- correctness as 0 or 1;
- event count;
- source artifact count.

Each feature column is normalized by its population standard deviation:

\[
x'_{ij} = \frac{x_{ij}}{\sigma_j}
\]

If a column has zero variance, its divisor is treated as 1. This prevents a
large-scale value such as event count or answer length from dominating similarity
only because of its numeric range.

The structured cosine similarity is:

\[
S^{structured}_{ij} =
\frac{x_i^T x_j}{\|x_i\|_2\|x_j\|_2}
\]

## Blended Similarity

When every valid candidate has a compatible prepared embedding, the DPP uses:

\[
S_{ij} =
\operatorname{clip}
\left(
0.85 S^{semantic}_{ij} + 0.15 S^{structured}_{ij},
0, 1
\right)
\]

This means:

- 85% of similarity is semantic trajectory/failure similarity;
- 15% is explicit observable structure such as status, events, and query shape.

The final clamp keeps the similarity in a safe nonnegative range for the kernel.

If a summary or embedding is unavailable, malformed, incompatible, or cannot be
prepared, RHO does not stop the evolution round. It uses the existing handcrafted
similarity instead and records:

```text
similarity_mode = handcrafted_fallback
```

The handcrafted default begins with feature cosine similarity and maps it to:

\[
S_{ij} = \exp\left(-(1 - \cos(x_i, x_j))\right)
\]

## Difficulty / Quality Weight

DPP needs more than diversity. RHO also needs to prioritize trajectories that
can teach the evolution pipeline something useful. The current quality score is
failure-biased:

\[
q_i^{raw} =
1.0
+ 1.0 \cdot \mathbb{1}[\text{trajectory is incorrect}]
+ 0.5 \cdot \mathbb{1}[\text{status is not success}]
+ 0.05 \cdot \min(\text{event count}, 20)
\]

Interpretation:

| Trajectory | Raw quality |
| --- | ---: |
| Correct success with 2 events | \(1.0 + 0.10 = 1.10\) |
| Incorrect failure with 5 events | \(1.0 + 1.0 + 0.5 + 0.25 = 2.75\) |
| Incorrect failure with at least 20 events | \(1.0 + 1.0 + 0.5 + 1.0 = 3.50\) |

The event contribution caps at 20 so an excessively verbose trace does not win
solely due to length.

The raw score is floored and normalized:

\[
f_i = \max\left(\frac{q_i^{raw}}{2.5}, \text{score floor}\right)
\]

\[
\hat{q_i} = \frac{f_i}{\max_j f_j}
\]

The default score floor is 0.1. It prevents low-score records from collapsing to
near-zero quality and making the kernel numerically weak.

## Theta: Hardness Versus Diversity

`theta` controls how strongly the quality score influences selection.

\[
\alpha = \frac{\theta}{2\max(1-\theta, 10^{-6})}
\]

For \(\theta < 1\), the final quality weight is:

\[
q_i = \hat{q_i}^{\alpha}
\]

At \(\theta = 1\), the code uses \(q_i = \hat{q_i}\) to avoid an infinite
exponent.

The default is:

```text
theta = 0.7
```

At this default:

\[
\alpha = \frac{0.7}{2(1-0.7)} = \frac{0.7}{0.6} \approx 1.167
\]

Practical interpretation:

| Theta | Selection tendency |
| ---: | --- |
| 0.0 | Pure diversity; quality is effectively uniform. |
| 0.3 | Diversity first, with modest preference for difficult failures. |
| 0.7 | Default balance between difficult failures and diverse coverage. |
| 0.9 | Stronger focus on high-value failures. |
| 1.0 | Difficulty-oriented limit. |

`theta` does not change the semantic embedding itself. It changes how much high
quality/failure trajectories are favored when DPP constructs the selected set.

## The DPP Kernel

Let:

- \(S\) be the blended similarity matrix;
- \(q_i\) be the theta-adjusted quality weight for trajectory \(i\);
- \(Q = \operatorname{diag}(q_1, q_2, \ldots, q_n)\);
- \(I\) be the identity matrix.

RHO builds the DPP kernel:

\[
L = Q S Q + 10^{-6}I
\]

Equivalently, each entry is approximately:

\[
L_{ij} = q_i S_{ij} q_j
\]

The diagonal is approximately:

\[
L_{ii} = q_i^2
\]

because \(S_{ii}=1\). The small diagonal jitter makes the kernel more stable
for Cholesky factorization.

This construction gives the intended behavior:

- high-quality failures have large diagonal values and are attractive;
- near-duplicate trajectories have large off-diagonal similarity and repel each
  other in a selected set;
- distinct trajectories add new diversity volume.

## Why The Determinant Prefers Diverse Trajectories

For a selected set \(Y\), DPP prefers a large determinant:

\[
\underset{|Y|=k}{\operatorname{argmax}}\; \det(L_Y)
\]

For two trajectories \(i\) and \(j\), their two-by-two determinant is:

\[
\det
\begin{bmatrix}
q_i^2 & q_iq_jS_{ij} \\
q_iq_jS_{ij} & q_j^2
\end{bmatrix}
= q_i^2 q_j^2 (1 - S_{ij}^2)
\]

This makes the tradeoff visible:

- If \(S_{ij}\) is close to 1, the trajectories are near-duplicates. Then
  \(1-S_{ij}^2\) is close to zero, so selecting both contributes little value.
- If \(S_{ij}\) is close to 0, the trajectories are distinct. Then
  \(1-S_{ij}^2\) is close to one, so selecting both adds much more value.
- Larger \(q_i\) and \(q_j\) make a pair more attractive when it is diverse.

Therefore RHO should choose a high-value web evidence failure and a distinct
Python calculation failure, rather than many almost identical web rate-limit
failures.

## Greedy MAP Approximation

Exact fixed-size DPP optimization is expensive. Gaia uses a deterministic greedy
maximum-a-posteriori approximation.

### First Pick

The first trajectory is the largest kernel diagonal:

\[
i_1 = \operatorname{argmax}_i L_{ii}
\]

Since \(L_{ii}\) is approximately \(q_i^2\), the first selected trajectory is
normally the highest difficulty-weighted candidate.

### Later Picks

Suppose \(Y\) is the set already selected. For each remaining candidate \(i\),
RHO computes its Schur-complement gain:

\[
g_i = L_{ii} - L_{iY}L_{YY}^{-1}L_{Yi}
\]

This measures how much new determinant volume the candidate adds after its
redundancy with the selected set has been removed.

The next pick is:

\[
i_{next} = \operatorname{argmax}_{i \notin Y} g_i
\]

The implementation maintains a Cholesky factor \(C\) for the selected kernel:

\[
CC^T = L_{YY}
\]

This avoids explicitly recomputing an inverse or determinant for every possible
candidate at every step. The process is deterministic: there is no random
tie-breaking in the DPP path.

## Example

Imagine three failed trajectories:

| ID | Summary pattern | Relative difficulty |
| --- | --- | ---: |
| A | `web_fetch` repeatedly returns HTTP 429; primary evidence remains missing | High |
| B | Browser/search rate limit; critic rejects unsupported source verification | High |
| C | Python `NameError` stops a numerical calculation; planner omitted a verification step | High |

The semantic similarities may look approximately like:

\[
S^{semantic} =
\begin{bmatrix}
1.00 & 0.91 & 0.15 \\
0.91 & 1.00 & 0.18 \\
0.15 & 0.18 & 1.00
\end{bmatrix}
\]

A and B are both source-access/evidence-verification failures. C is a
calculation/planning failure. With default `theta = 0.7`, DPP will generally
select one of A/B plus C, rather than A and B together.

If `theta` is raised toward 1, RHO increasingly favors the highest-quality
failures, even when they are somewhat redundant. If `theta` is lowered toward
0, it increasingly favors distinct trajectory families.

## Selector Modes

The available selectors are:

| Selector | Behavior |
| --- | --- |
| `dpp` | Default. Semantic/structured similarity plus theta-weighted difficulty. |
| `difficulty` | Sort by quality score only. No diversity objective. |
| `coverage` | Greedy farthest-first selection on handcrafted structural features. |
| `random` | Seeded random baseline. |

The `seed` is relevant to `random`. The greedy DPP, difficulty, and coverage
paths are deterministic for the same input data and configuration.

## Safety And Fallbacks

The selector checks for invalid or degenerate conditions, including:

- fewer than two valid candidate trajectories;
- absent, incomplete, invalid, or incompatible embedding vectors;
- all-identical handcrafted feature rows when semantic similarity is unavailable;
- non-positive eigenvalues or a kernel condition number above \(10^{12}\);
- exceptions during kernel construction or greedy selection.

On failure, it uses a deterministic fallback:

1. sort by descending quality score;
2. break ties by stable source-qualified selection ID.

The round writes the chosen method, similarity mode, cache hit counts, worker
limit, embedding model, selector controls, and fallback reason to:

```text
<evolution-round>/selection.json
```

## Current Default Configuration

```text
selector                  = dpp
theta                     = 0.7
score_floor               = 0.1
semantic similarity weight= 0.85
structured similarity     = 0.15
max_trajectory_workers    = 10
embedding model           = embeddinggemma
```

In plain terms, the current system tries to select a small set of trajectories
that are both difficult enough to improve Gaia and different enough to expose
different weaknesses in its planner, ReAct loop, tools, scratchpad handling,
critic, and synthesis behavior.
