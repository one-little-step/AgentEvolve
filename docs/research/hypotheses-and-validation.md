# RHO-Parallel-GEPA Research Hypotheses And Validation

## Hypotheses

| ID | Claim | Compare | Failure threshold |
| --- | --- | --- | --- |
| H1 | Persistent pool beats RHO best-of-N under matched budget | B0 vs B1 | No repeatable held-out gain or worse efficiency |
| H2 | Causal blame improves editing | B1 vs B2 | Accepted edit rate drops >20% and held-out gain is <=1pp |
| H3 | Worked/failed edit memory reduces loops | B2 vs B3 | No regression reduction or unacceptable validation cost |
| H4 | Entropy prioritizes harness-sensitive tasks/issues | B3 vs B4 | No gain over severity-rank at matched budget |
| H5 | Deterministic merge combines complementary branches | B4 vs B5 | Low yield or no held-out/process benefit |
| H6 | Parallel batches reduce time without quality loss | B5 vs B6 | Any state error, speedup <1.5x, or quality drop >1pp |

## Baseline Ladder

```text
B0: Legacy RHO best-of-N
B1: RHO + persistent pool + outcome Pareto + severity-directed editing
B2: B1 + causal blame graph
B3: B2 + structured edit memory and feedback validation
B4: B3 + entropy/DPP quality and issue selection
B5: B4 + deterministic merge
B6: B5 + parallel snapshot/lease batches
```

## Required Measurements

```text
held-out outcome and dispersion
agent rollouts, analyzer/judge calls, editor calls, embeddings, model tokens
accepted/rejected/no-op/exhausted attempts
generalization rate and regression-probe violations
pool size, mechanism cluster coverage, entropy availability
blame stability and calibration agreement when enabled
merge complementarity and crossover yield
parallel speedup and shared-state correctness
```

## Causal Reliability

Single analyzer+judge verdict is the default cost profile. Consensus and
intervention calibration are ablations. Calibration must use controlled artifact
or agent-output interventions and report predicted-versus-observed agreement.

## Entropy Reliability

Only calculate entropy from cells with at least three comparable candidates and
two rollouts per candidate. Mechanism alignment uses task-local semantic clusters
anchored by base harness mechanisms. Report cluster fragmentation, freshness,
creation/merge/split events, and embedding configuration.

## Generalization Probes

Probes run after a mechanism edit cluster completes, not after every edit. Their
budget is capped by `GEPA_PROBE_BUDGET_FRACTION`; skipped probes are explicitly
`generalization_unverified`. Replay is adapter-gated; otherwise use full rollout.
