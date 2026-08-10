# RHO Offline Wisdom Evolution Pipeline

> **Historical archive status:** These documents preserve the detailed Gaia-era
> RHO and RHO-GEPA analysis that informed AgentEvolve. They are authoritative for
> established rationale, schemas, debugging evidence, and target architecture;
> Gaia-specific paths and runtime assumptions are historical examples, not active
> AgentEvolve dependencies. Read `../vision-and-decision-record.md` and
> `../migration/cuga-adaptation-guide.md` before implementing against CUGA.

This directory contains a complete, self-contained explanation of the
implemented offline wisdom-evolution system for the Gaia agent.

The system is a minimal, hardcoded, versioned implementation of an iterative
**R**esponse-based **H**arness **O**ptimization loop (RHO). It takes historical
trajectory runs, selects a representative subset, diagnoses failures, proposes
edited wisdom bundles, evaluates them against the current parent harness, and
optionally materializes the best candidate as a new immutable version.

## Documents

| File | Purpose |
|------|---------|
| [01-overview.md](01-overview.md) | High-level goals, terminology, and RHO concepts |
| [02-data-model.md](02-data-model.md) | Trajectory records, digests, diagnoses, scores, and wisdom bundles |
| [03-control-flow.md](03-control-flow.md) | Full round execution flow, from source runs to materialized version |
| [04-coreset-selection.md](04-coreset-selection.md) | DPP and other selector strategies, feature construction, and fallbacks |
| [selection_algo_explaination.md](selection_algo_explaination.md) | Detailed current semantic-summary DPP input, cache, math, and greedy MAP explanation |
| [05-diagnosis.md](05-diagnosis.md) | How trajectories are diagnosed against the parent wisdom bundle |
| [06-candidate-generation.md](06-candidate-generation.md) | How candidate wisdom bundles are proposed and applied |
| [07-pairwise-judging.md](07-pairwise-judging.md) | Pairwise preference scoring and the role of the judge model |
| [08-acceptance-and-promotion.md](08-acceptance-and-promotion.md) | Acceptance gate, experimental promotion, and progressive chains |
| [09-artifacts-and-versioning.md](09-artifacts-and-versioning.md) | Directory layout, manifests, and candidate archive |
| [10-runner-and-batch-integration.md](10-runner-and-batch-integration.md) | `evolve_run.py`, `batch_run.py`, and how to evaluate a version |
| [11-configuration-reference.md](11-configuration-reference.md) | Every tunable constant and its effect |
| [12-tracing-and-debugging.md](12-tracing-and-debugging.md) | How to trace failures and audit a round |
| [13-rho-gepa-population-evolution.md](13-rho-gepa-population-evolution.md) | Compact lifecycle and configuration reference for opt-in population evolution |
| [14-agent-integration-and-history-rag.md](14-agent-integration-and-history-rag.md) | Adapter contract and edit-history RAG reference |
| [15-rho-gepa-architecture-and-debugging.md](15-rho-gepa-architecture-and-debugging.md) | Primary RHO-GEPA architecture dossier, control flow, artifacts, failure analysis, verified findings, and improvement backlog |
| [16-rho-gepa-execution-atlas.md](16-rho-gepa-execution-atlas.md) | Function-by-function active execution atlas from configuration to artifacts and planned-state comparison |
| [17-rho-gepa-prompts-and-data-contracts.md](17-rho-gepa-prompts-and-data-contracts.md) | Exact active evolution prompt templates, JSON contracts, parser rules, and artifact schemas |
| [18-rho-parallel-gepa-target-architecture.md](18-rho-parallel-gepa-target-architecture.md) | Approved target architecture: analyzer, GEPA judge, hierarchical Pareto, feedback-validated edits, RAG, merge, locks, and parallel batches |
| [19-rho-parallel-gepa-research-hypotheses.md](19-rho-parallel-gepa-research-hypotheses.md) | Research claims, baselines, calibration, entropy/clustering validation, merge metrics, and ablation protocol |

## Quick Start

Run one evolution round:

```bash
uv run python dataset/evolve_run.py
```

Evaluate a specific evolved version in batch:

```python
# dataset/batch_run.py
WISDOM_VERSION = "rho-gaia-1"
```

```bash
uv run python dataset/batch_run.py
```

The rest of this folder explains exactly what happens under the hood.

For RHO-GEPA investigation, start with
[15-rho-gepa-architecture-and-debugging.md](15-rho-gepa-architecture-and-debugging.md).
Then use [16-rho-gepa-execution-atlas.md](16-rho-gepa-execution-atlas.md) to
trace a run and [17-rho-gepa-prompts-and-data-contracts.md](17-rho-gepa-prompts-and-data-contracts.md)
to inspect model-facing data flow.

For the approved future design rather than current behavior, use
[18-rho-parallel-gepa-target-architecture.md](18-rho-parallel-gepa-target-architecture.md).
Use [19-rho-parallel-gepa-research-hypotheses.md](19-rho-parallel-gepa-research-hypotheses.md)
to distinguish target mechanisms from empirically validated claims.
