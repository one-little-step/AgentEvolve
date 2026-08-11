# DPP Selector Completion Design

## Scope

Complete the existing bounded hierarchical DPP selector so its behavior matches
the binding architecture in `docs/architecture/selection-algorithms.md`.
This increment changes no adapter, orchestrator, storage, or CUGA behavior.
The selector remains in `core/entropy.py` temporarily; moving it to `issues.py`
is a later package-structure refactor after the foundation modules exist.

## Current Deficit

The current selector has the correct greedy-MAP Schur-complement core, but does
not yet construct quality or report selection in the form required by the
architecture. In particular it lacks the configurable raw-quality formula,
theta transformation, deterministic prefilter record, degeneracy fallback, and
the `coverage` ablation.

## Boundaries

`core/entropy.py` owns pure, agent-neutral selection math. It accepts issue
evidence and caller-supplied deterministic similarity functions. It must not
create embeddings, read manifests, persist provenance, or import an adapter.

The future caller owns configured embeddings, profile resolution, and manifest
persistence. Until `core/storage.py` exists, the selector returns an immutable
selection report that contains every value the future manifest needs.

## Inputs

`Issue` gains these optional, validated evidence inputs:

- `confidence`, `coverage_need`, and `pareto_relevance`, each normalized to
  `[0, 1]`.
- `trace_backed_writable_artifact`, which must be true for DPP eligibility.

Existing `severity`, entropy, and freshness supply the remaining raw-quality
signals. Missing entropy contributes zero and is represented in the report,
not treated as successful evidence.

`DPPSelectionConfig` is immutable and validates:

- quality weights for severity, confidence, entropy, coverage, and Pareto
  relevance sum to one;
- score floor is positive;
- `theta` is in `[0, 1]`;
- jitter and minimum gain are positive;
- maximum dense items is between one and 100;
- semantic and structural similarity weights sum to one;
- a finite positive condition-number limit defaults to `1e12`.

Defaults follow the current binding architecture: score floor `0.1`, theta
`0.7`, jitter `1e-6`, minimum gain `1e-12`, maximum dense items `100`, and
semantic/structural weights `0.85` and `0.15`.

## Quality And Kernel

For each eligible item, compute:

```text
raw_quality =
    w_severity * severity
  + w_confidence * confidence
  + w_entropy * normalized_entropy
  + w_coverage * coverage_need
  + w_pareto * pareto_relevance

floored = max(raw_quality, score_floor)
normalized = floored / max(floored for all eligible items)
alpha = theta / (2 * max(1 - theta, 1e-6))  when theta < 1
quality = normalized ** alpha                when theta < 1
quality = normalized                         when theta == 1
```

The selector builds a symmetric kernel with an explicit `quality` diagonal and
quality-weighted similarity off diagonal:

```text
L[i, i] = quality(i)^2 + jitter
L[i, j] = quality(i) * clipped_similarity(i, j) * quality(j)
```

The existing `greedy_map_dpp()` remains the only DPP inference operation. It
uses incremental Cholesky/Schur-complement marginal gains; it does not use
eigendecomposition or select by top-K quality.

## Hierarchy And Prefilter

Stage 1 produces one aggregate item per task. Its quality combines that task's
eligible issue evidence and its similarity comes from the configured task
similarity function.

Stage 2 applies the same selector independently to mechanism clusters within
each selected task using mechanism similarities.

Before every dense kernel build, candidates are deterministically ranked by
descending entropy, descending raw quality, then stable ID and truncated to
`max_dense_items`. The report includes the original count, eligible count,
prefiltered count, IDs, and the final threshold tuple. A caller can persist
this report verbatim later; no persistence is added in this increment.

## Modes And Fallback

`dpp` is the only quality-diversity mode.

- `severity_rank` is quality-only and does not inspect similarity.
- `coverage` is deterministic farthest-first diversity-only selection. It
  selects the stable smallest ID first and thereafter maximizes minimum
  distance (`1 - similarity`) to selected items, breaking ties by stable ID.
- `random` remains seeded and is an explicit ablation.

The DPP mode falls back to deterministic descending-quality ordering when
there are fewer than two valid candidates, similarities are invalid/non-finite,
the kernel is non-finite, the condition number exceeds the configured limit, or
greedy selection raises. The report carries an explicit fallback reason. A
missing similarity function is valid and means zero similarity; it is not an
embedding fallback because this prototype does not own embeddings.

## Public Result

`SelectionReport` contains immutable selected IDs plus selection metadata:

- mode, resolved theta, alpha, score floor, and jitter;
- candidate counts and prefilter information for each stage;
- whether fallback occurred and its reason;
- whether supplied similarities were used.

The existing `select()` method remains and returns only `Issue` values for
compatibility. A new `select_with_report()` returns both selected issues and
the report.

## Tests

Tests are written before implementation and prove:

- qf4's near-duplicate marginal-gain result;
- quality wins among equally diverse candidates;
- theta changes the quality/diversity tradeoff;
- deterministic tie breaking;
- prefilter cap and recorded threshold;
- invalid or ill-conditioned kernels use recorded quality fallback;
- coverage is diversity-only and deterministic;
- missing trace-backed writable attribution excludes an issue before ranking;
- configuration rejects invalid weights, limits, and theta values.

## Documentation Changes

Update `docs/architecture/selection-algorithms.md` to make the temporary
`core/entropy.py` location and report interface explicit, distinguish supplied
similarity from future embedding ownership, and define the exact deterministic
prefilter order.

Update `docs/architecture/component-contracts.md` to list
`DPPSelectionConfig` and `SelectionReport` as the temporary entropy-module
boundary and to state that manifest persistence becomes a storage-layer duty.

Update `docs/architecture/implementation-mapping.md` so the migration sequence
is accurate: complete and test bounded DPP behavior in `entropy.py` now; move
it into `issues.py` only after foundational storage/config contracts are in
place.
