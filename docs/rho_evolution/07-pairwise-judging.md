# 07 - Pairwise Judging

Pairwise judging compares a candidate rollout against the parent rollout for
the same task. The judge model produces a numeric preference score.

## Location

agent/gaia_lg_react/evolution/prompts.py, function pairwise_preference.

## What is compared

For each selected task:

```text
parent rollout on task T   -> before trajectory
candidate rollout on task T -> after trajectory
```

The judge receives a digest of both trajectories, not just final answers. The
digest includes status, correctness, event names, answer metadata, and query
fingerprints.

## Judge prompt

```text
Compare the following "before" and "after" trajectories for the same task
and judge whether the after trajectory is better.

Return a single JSON object with:
- before_id
- after_id
- confidence (0 to 1)
- rationale
- score (-10 to 10)
```

## Score semantics

| Range | Meaning |
|-------|---------|
| -10 to -1 | Parent rollout was better |
| 0 | No meaningful difference |
| +1 to +10 | Candidate rollout was better |

## Aggregation

For each candidate, available scores are averaged across the selected tasks:

```python
available_scores = [
    s.normalized_score
    for s in scores
    if s.available and s.normalized_score is not None
]
average_score = sum(available_scores) / len(available_scores) if available_scores else None
```

Unavailable scores are ignored in the average.

## Judge model

By default the same model is used for rollouts, diagnosis, optimization, and
judging. JUDGE_MODEL in evolve_run.py sets the LLM client, but the current
implementation uses that client for all evolution-stage LLM calls.

## Failure handling

If the judge returns invalid JSON, an out-of-range score, or raises an
exception, the comparison is marked unavailable:

```python
CandidateScore(available=False, normalized_score=None, rationale=str(exc))
```

These do not contribute to the candidate average.

## Example score from a real run

```json
{
  "available": true,
  "normalized_score": 0.0,
  "confidence": 0.85,
  "rationale": "Both trajectories resulted in failure and incorrect answers; the only change is a different normalized value.",
  "before_id": "gaia-8e867cd7",
  "after_id": "gaia-8e867cd7"
}
```
