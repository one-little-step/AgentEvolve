# RHO-Parallel-GEPA Research Hypotheses And Validation Protocol

## Purpose

This document separates research claims from the target architecture in
[18-rho-parallel-gepa-target-architecture.md](18-rho-parallel-gepa-target-architecture.md).
The architecture specifies what the system will do. This document states which
claims require empirical validation, their baselines, evidence, ablations, and
failure criteria.

## 1. Research Claims

| ID | Claim | Required evidence | Failure criterion |
| --- | --- | --- | --- |
| H1 | Preserving base plus all RHO proposals in a persistent pool outperforms RHO best-of-N under matched budgets | Held-out outcome comparison against sequential RHO | No repeatable held-out gain or materially worse budget efficiency |
| H2 | Causal-blame-directed artifact editing improves attribution and edit efficiency over non-causal severity-directed editing | Accepted-edit rate, targeted artifact agreement, held-out outcome | Accepted-edit rate drops by more than 20% relative to B1 and held-out outcome gain is not greater than 1 percentage point |
| H3 | Feedback validation with worked/regression sets reduces pass-fail-pass loops | Regression-probe violation rate, repeated issue retries, generalization rate | No reduction in regressions or excessive validation cost |
| H4 | Candidate-relative entropy prioritizes harness-sensitive tasks/issues better than severity alone | Crossover yield, accepted edits, held-out score per rollout | No gain against severity-rank under matched task/rollout budget |
| H5 | DPP issue selection creates useful diversity beyond severity ranking/random selection | Mutation cluster diversity, duplicate rate, accepted edit yield, wall time | Mutation diversity is below 0.50, or DPP exceeds 10% of iteration wall time without at least 15% accepted-edit-yield improvement over severity rank |
| H6 | Provenance-preserving merge combines complementary branches productively | Merge acceptance, crossover yield, held-out gain attributable to merged artifacts | Yield remains low or merged children regress more than mutation-only baseline |
| H7 | Lock-safe parallel batches reduce wall time without reducing result quality | Wall-clock speedup, budget parity, selection/acceptance quality | Any shared-state correctness error, speedup below 1.5x, or held-out quality drop above 1 percentage point relative to sequential |

## 1.1 Reference Adapter Roadmap

The current Gaia implementation is a baseline adapter for historical RHO and
existing evolution tests. The next target-phase reference adapter is CUGA because
the research design needs exact state/event tracing and optional valid checkpoint
replay for causal attribution and deferred generalization validation.

At the time of this document, CUGA source and documentation are external to this
workspace. Consequently:

```text
- no CUGA behavior is claimed or simulated here;
- generic capability contracts are planned before CUGA integration;
- Gaia remains available for baseline compatibility tests;
- CUGA integration, repository placement, and artifact mapping are separate
  planning decisions after source/docs are imported and reviewed.
```

The eventual CUGA adapter validation must demonstrate trace provenance for agent
state transitions, skill/policy/memory reads, tool/subagent interactions,
artifact versions, checkpoint boundaries, and final outputs before replay-based
claims are enabled.

## 2. Minimum Baselines

Every experiment must compare against enough baselines to identify which added
mechanism caused any observed improvement.

```text
B0: Legacy RHO best-of-N
B1: RHO + persistent pool + outcome Pareto only
B2: B1 + causal blame graph, sequential
B3: B2 + feedback validation and structured edit memory
B4: B3 + entropy task/issue quality
B5: B4 + deterministic merge
B6: B5 + parallel batch execution
```

`B1` tests the core pool hypothesis. Later systems must not claim a pool benefit
when only the full composite system was evaluated.

## 3. Default Evaluation Protocol

### 3.1 Data partitions

```text
Historical source corpus:
  preserved trajectories used for RHO/DPP input.

Evolution coreset:
  fixed by default, or explicitly anchor-refresh/full-entropy mode.

Generalization probes:
  same/near mechanism-cluster tasks not used as origin or worked-set cases.

Held-out evaluation:
  tasks never used for candidate mutation, focused validation, or pool selection.
```

### 3.2 Budget parity

Every result reports separate costs:

```text
agent rollouts
analyzer/judge calls
editor calls
input/output model tokens by role
embedding calls
wall-clock time
cached versus fresh work
```

Comparisons are valid only when they use either matched rollout budgets or a
clearly reported efficiency curve across budget levels.

### 3.3 Statistical reporting

For each configuration and seed, report:

```text
mean and dispersion of held-out task outcome
mean and dispersion of process/mechanism score
accepted/rejected/no-op/exhausted attempt counts
generalization rate
regression-probe violation rate
pool size trajectory
blame stability and calibration agreement when enabled
mutation diversity and DPP selection wall time
crossover yield and merge complementarity distribution
```

No causal claim should rest on one smoke run or one model response.

## 4. Blame Graph Validation

The default operational mode uses one analyzer+judge verdict per selected
evidence unit:

```bash
GEPA_BLAME_CONSENSUS_RUNS=1
GEPA_BLAME_CALIBRATION_ENABLED=false
```

This is a cost-controlled operating mode. It does not validate causal accuracy.

Two optional research checks assess reliability:

### 4.1 Consensus ablation

```text
single verdict: 1 call
consensus verdict: 2-3 independent calls
```

Measure agreement in blame distribution, primary artifact selection, and
mechanism-cluster assignment. Preserve disagreement rather than averaging it
away without trace.

### 4.2 Controlled intervention calibration

For a bounded stratified sample of high-impact or low-stability findings:

```text
1. Identify the highest-blame artifact/agent node.
2. Substitute a controlled alternative output/artifact behavior.
3. Re-run the affected task trajectory.
4. Observe whether predicted downstream behavior changes.
5. Record agreement between predicted and observed effect.
```

This creates `blame_calibration_agreement`, not universal causal proof. It is a
research metric and must be charged to rollout budget.

## 5. Entropy and Mechanism Alignment Validation

### 5.1 Entropy evidence floor

Do not calculate strong cross-candidate entropy from sparse single-rollout data.
For a task/mechanism cluster to influence entropy ranking:

```text
minimum comparable admitted candidates: 3
minimum rollouts per candidate: 2
compatible score provenance: required
```

Otherwise the item is marked `entropy_unavailable` and uses severity/coverage
fallback quality.

### 5.2 Dynamic mechanism clustering

"Same mechanism" means same `mechanism_cluster_id`, assigned by task-local
incremental semantic clustering. The clustering evaluation reports:

```text
anchor coverage
cluster creation/merge/split counts
mean member similarity
cluster fragmentation rate
cluster collision audit sample
cluster lineage across outer iterations
```

Within an outer iteration, existing cluster identities and centroids remain
stable for comparability, while new mechanism observations may join existing
clusters. Cluster creation, merge, and split are deferred to an entropy-refresh
barrier. The engine reports `cluster_freshness`, the fraction of member
observations produced after the last accepted edit touching the cluster's primary
artifact targets. Potentially stale clusters receive reduced entropy quality
weight rather than being silently treated as current evidence.

Embedding model, threshold, maximum clusters per task, and clustering mode are
all manifest fields. Base-harness mechanisms seed anchor clusters; assignments
freeze within one outer iteration.

### 5.3 DPP issue-selection baselines

```bash
GEPA_ISSUE_SELECTION_MODE=dpp
GEPA_ISSUE_SELECTION_MODE=severity_rank
GEPA_ISSUE_SELECTION_MODE=random
```

Measure the marginal value of DPP through:

```text
distinct mechanism clusters / selected work items
duplicate or conflicting write-set rejection rate
accepted edit yield
generalization rate
selection wall-clock time
held-out gain per rollout
```

## 6. Deferred Generalization Probe Protocol

Generalization probes are cluster-level by default, not per-edit. The default
cluster key is `mechanism_cluster_id`; batch and artifact write-set grouping are
ablations.

```bash
GEPA_GENERALIZATION_PROBE_MODE=deferred
GEPA_PROBE_CLUSTER_BY=mechanism_cluster
GEPA_PROBE_BUDGET_FRACTION=0.15
```

Probe selection uses one or two related task/mechanism cases outside origin,
worked, and regression sets. Their rollouts are charged to `GEPA_MAX_ROLLOUTS`
and cannot exceed the configured fraction. When probe capacity is exhausted, the
cluster is marked `generalization_unverified`; it is excluded from the
generalization-rate denominator.

Counterfactual replay is allowed only through an adapter that declares exact
checkpoint/state reconstruction support. Otherwise the protocol performs a full
rollout. Probe failures are retained as next-cluster regression evidence.

## 7. Merge Validation

Merge eligibility uses at least one ancestor-improving descendant, no
catastrophic protected-floor regression in the other descendant, and sufficient
complementarity. It must report:

```text
attempted merges
eligible merges
accepted merges
rejected merges by reason
complementarity score distribution
crossover yield = accepted merges / attempted merges
artifact provenance of every accepted merge
```

The merge hypothesis fails when deterministic merge adds cost without improving
held-out outcomes, generalization, or process coverage relative to mutation-only
pool evolution.

## 8. Standard Profiles

| Profile | Goal | Required comparison |
| --- | --- | --- |
| `minimal` | Test H1 | B0 versus B1 |
| `research_sequential` | Test H2-H5 without parallel scheduling | B1 through B5 feature ablations |
| `research_parallel` | Test H7 | sequential B5 versus parallel B6 at matched budget |
| `full_ablation` | Complete research matrix | all baselines, profiles, and configured modes |

The full target is feature-gated, but the experiment order should remain
interpretable: establish pool value first, then causal/edit-memory value, then
entropy/merge value, then parallel speedup.
