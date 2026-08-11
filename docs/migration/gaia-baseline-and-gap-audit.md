# Gaia Baseline And Gap Audit

## Purpose

This is the migration record for relevant legacy RHO-GEPA work. It preserves
what was learned without importing Gaia runtime code into AgentEvolve.

## Useful Legacy Foundation

The legacy repository implemented these reusable concepts:

| Legacy area | Value retained in AgentEvolve |
| --- | --- |
| `agent/evolution_core/contracts.py` | Motivation for a generic adapter protocol |
| `agent/evolution_core/history.py` | Redaction-first history persistence and lexical/semantic fallback concepts |
| `agent/evolution_core/operators.py` | LLM output must be parsed and passed through an editor capability boundary |
| `agent/evolution_core/population.py` | Immutable candidate versioning, lineage sidecars, and candidate artifact isolation |
| Gaia `WisdomEditRegistry` | Candidate-local, allowlisted write operations and diff logs |
| Gaia rollout scheduler | Bounded concurrent task/rollout execution |
| One-task smoke diagnostics | Validate live model edit vocabulary before assuming a JSON schema works |

## Verified Legacy Defects That Must Not Be Copied

| Defect | Why it matters | Target correction |
| --- | --- | --- |
| Parents had synthetic `0.0` score while children had deltas against different parents | Candidate scores were not globally comparable | Common provenance-bearing score tensor |
| Only prior elites were parents in later generations | Task specialists were discarded | Persistent candidate pool |
| Parent selection was round-robin | No GEPA Pareto exploration pressure | Pareto task/mechanism winner selection |
| Nominal target module could edit every wisdom file | Provenance and merge semantics were unreliable | Explicit adapter-declared artifact write sets and leases |
| Every child received full-cohort evaluation | No inexpensive admission policy | Focused feedback validation and bounded budgets |
| History stored only module plus average score | RAG could not learn what edit worked or failed | Structured attempt records with rationale, diff, evidence, outcomes |
| `add` operation from live model was initially rejected | LLM output vocabularies drift | Schema validation plus deliberately supported aliases |
| Active crossover was unconstrained LLM synthesis | No deterministic inheritance provenance | Artifact-level deterministic merge first, limited conflict refinement second |

## Historical Context Files

The original source repository contained extensive current-state and target-state
documents. AgentEvolve carries the distilled, self-contained versions in:

```text
docs/architecture/target-rho-parallel-gepa.md
docs/research/hypotheses-and-validation.md
docs/plans/rho-parallel-gepa-completion.md
```

Do not add a runtime dependency on the original Gaia repository. If a historical
implementation detail is needed, treat it as reference-only and re-evaluate it
against the CUGA adapter contract.
