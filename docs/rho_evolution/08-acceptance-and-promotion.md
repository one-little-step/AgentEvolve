# 08 - Acceptance and Promotion

After scoring, the round either promotes a candidate to a new wisdom version or
rejects it.

## Normal acceptance gate

In normal mode, a candidate must satisfy:

```python
average_score > acceptance_threshold
```

The default threshold is 0.0, so a candidate with an average score of 0.0 is
rejected. Among passing candidates, the highest average score wins.

This prevents neutral or harmful edits from becoming new versions.

## Rejection

If no candidate passes:

```text
1. manifest.json records winner_candidate_id=null and winner_version=null
2. promotion_mode is "acceptance_gate"
3. result.status is "rejected"
4. The progressive chain stops
```

No version directory is created.

## Experimental promotion

For experimentation, set in dataset/evolve_run.py:

```python
EXPERIMENTAL_PROMOTE_CANDIDATE = True
```

When enabled:

1. If any candidate has a numeric average_score, promote the highest-scoring
candidate regardless of sign. promotion_mode is "experimental_scored".
2. If no candidate has any numeric score because all judgments failed, promote
candidate_0 as a deterministic fallback. promotion_mode is
"experimental_fallback".
3. If there are no candidates at all, the round still rejects.
promotion_mode is "experimental_unavailable".

The experimental mode does not compare candidates across rounds. The candidate
pool is only the current round's candidates.

## Progressive chain

The runner builds a chain before starting:

```python
parent = INITIAL_HARNESS
for round_number in range(1, ROUND_COUNT + 1):
    target = f"{TARGET_HARNESS_NAME_PREFIX}-{round_number}"
    yield (round_number, parent, target)
    parent = target
```

For example, with ROUND_COUNT=3:

```text
Round 1: base -> rho-gaia-1
Round 2: rho-gaia-1 -> rho-gaia-2
Round 3: rho-gaia-2 -> rho-gaia-3
```

If Round 2 is rejected, Round 3 never runs unless experimental promotion is
enabled.

## Version naming

Target versions are deterministic from the configuration:

```text
<TARGET_HARNESS_NAME_PREFIX>-<round_number>
```

Examples: rho-gaia-1, rho-gaia-2.

If target_version is not provided, a sanitized fallback name is generated from
the source run, parent version, and round ID.
