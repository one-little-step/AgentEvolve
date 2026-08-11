# Merge Resolution Algorithm

## Purpose

This document removes ambiguity from `core/merge.py`. Crossover is a
provenance-preserving three-way merge over adapter-declared artifact units. It is
never token, line, paragraph, or prompt splicing.

## Eligibility

All checks must pass and each result is recorded in `MergeProvenance`:

```text
1. left and right are distinct admitted candidates
2. neither is an ancestor or descendant of the other
3. a common ancestor exists via full lineage traversal
4. no prior merge attempt exists for the same (ancestor, left, right) triple
5. at least one side improves over the ancestor on comparable objectives
6. neither side violates a catastrophic protected floor
7. complementarity >= GEPA_MERGE_MIN_COMPLEMENTARITY
8. at least one artifact differs from the ancestor
9. every conflicting artifact is a single declared artifact unit
```

Complementarity uses comparable cells only:

```text
complementarity(left, right) =
    sum over comparable (t, m) of
        severity(t, m) * abs(score(left, t, m) - score(right, t, m))
  + GEPA_MERGE_DISJOINT_BONUS * disjoint_changed_artifact_count
```

## Per-Artifact Resolution

For each artifact unit in the union of ancestor, left, and right inventories:

```text
left_hash == ancestor_hash and right_hash != ancestor_hash  -> inherit right
left_hash != ancestor_hash and right_hash == ancestor_hash  -> inherit left
left_hash == right_hash                                     -> inherit shared
left_hash == ancestor_hash and right_hash == ancestor_hash  -> inherit ancestor
both differ from ancestor and from each other                -> conflict
```

Deletion and addition are explicit cases:

```text
present in ancestor, absent in exactly one side -> conflict, never silent deletion
absent in ancestor, present in exactly one side -> inherit that side
absent in ancestor, present in both with different hashes -> conflict
```

An artifact whose `resulting_hash` equals `ancestor_hash` must not emit an edit
operation. A no-op merge across all artifacts is recorded as `no_op`, not as an
accepted merge.

## Conflict Resolution By Evidence

For a conflicting artifact `A`, evidence is restricted to blame-graph findings
that actually cite `A`:

```text
relevant(side, A) = comparable evaluated cells (t, m) such that
    some observed CausalFinding for that side and (t, m)
    lists A in its blame_graph artifact_candidates

EvidenceScore(side, A) =
    sum over relevant(side, A) of
        severity(t, m) * confidence(side, t, m) * score(side, t, m)
```

Resolution rules, applied in order:

```text
1. Both sides have zero relevant evidence            -> retain ancestor
2. Exactly one side has non-zero relevant evidence   -> inherit that side
3. Scores differ by more than GEPA_MERGE_EVIDENCE_EPSILON -> inherit higher side
4. Scores are within epsilon (a tie):
     if GEPA_LLM_CONFLICT_REFINEMENT_ENABLED and coverage is sufficient
         -> request scoped conflict refinement
     else
         -> retain ancestor
```

Coverage guard for rule 3: a side may win only if its relevant comparison
coverage meets the configured minimum. A single low-coverage cell must not
override a broader evidence base. Coverage figures for both sides are recorded.

Deliberate deviation from a global-average tie-breaker: aggregate cross-task
averages are forbidden here because they discard causal attribution and would
let unrelated task performance decide an artifact-level conflict.

## Scoped Conflict Refinement

When refinement is permitted, the request contains only:

```text
the single conflicting artifact_id
ancestor, left, and right content for that artifact only
relevant sanitized findings and evidence references
worked-set and regression obligations for that artifact
the authorized write set containing exactly that artifact
```

The refiner cannot read or modify any other artifact. A refinement result that
targets a different artifact raises `WriteAuthorizationError`. A malformed
refinement receives one bounded correction request, then falls back to retaining
the ancestor and records `malformed`.

## Child Validation And Admission

A merged child follows the same path as a mutation attempt:

```text
focused validation: origin, worked, regression obligations for every inherited
                    changed artifact
protected floors:   disqualifying
acceptance:         positive primary gain and positive weighted net gain
admission:          durable transactional commit with full merge provenance
```

## Recorded Metrics

```text
merge attempts and eligibility failures by check name
complementarity distribution
per-artifact inheritance counts by kind
evidence-resolved versus ancestor-retained conflicts
refinement requests, successes, and malformed outcomes
crossover_yield = accepted merges / attempted merges
```

Very low yield indicates over-strict eligibility. Very high yield indicates
insufficient selectivity. Both are research signals, not silent conditions.
