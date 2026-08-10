# AgentEvolve Vision And Decision Record

## Vision

AgentEvolve evolves externally configurable agent harnesses, not model weights.
It improves reusable agent behavior by analyzing execution evidence, preserving
promising candidate variants, editing declared artifacts such as skills, memory,
policies, prompts, and workflows, and validating improvements against regressions.

## Approved Target

The approved target is RHO-Parallel-GEPA, not a request to reinvent an evolution
algorithm. Its pipeline is historical trajectories and DPP coreset ->
base plus every RHO candidate in a persistent GEPA pool ->
provenance-bearing evaluation -> causal blame and artifact-targeted edits ->
structured edit memory and focused regression validation -> entropy/DPP selection,
deterministic merge, and optional safe parallelism.

## CUGA Boundary

CUGA is the intended reference adapter through its SDK. We do not invent CUGA APIs,
artifact types, trace fields, checkpoint behavior, replay semantics, or package
names. Inspect official SDK documentation and source first; then map only proven
public capabilities to the active adapter contract.

## First Implementation Path

Implement and evaluate the `minimal` profile first: persistent pool, common
outcome-score provenance, base plus every RHO proposal, fixed historical coreset,
and sequential editing. Compare B0 and B1 under matched budget before enabling
causal blame, edit memory, entropy, merge, or parallelism.

## Settled Decisions

| # | Decision | Rationale | Source |
| --- | --- | --- | --- |
| 1 | Use the CUGA SDK, not a CUGA fork | Preserves agent-neutrality, version pinning, and upstream separation | `docs/migration/cuga-sdk-integration-notes.md` |
| 2 | Keep `src/agent_evolve/core/` free of CUGA and agent imports | Generic orchestration must work with any adapter | `src/agent_evolve/core/contracts.py` |
| 3 | CUGA is the target reference adapter | Need exact trace provenance, artifacts, and optional checkpoint replay | `docs/architecture/target-rho-parallel-gepa.md` |
| 4 | Gaia is historical context only | Avoid coupling to legacy runtime paths and assumptions | `docs/migration/gaia-baseline-and-gap-audit.md` |
| 5 | Persistent pool: base plus every RHO proposal | Avoid elite-only loss of task-specialist candidates | `docs/rho_evolution/13-rho-gepa-population-evolution.md` |
| 6 | Base receives `G` rollout group evidence; post-RHO candidates start with one rollout per task | Preserve RHO-scale cost while seeding the pool | `docs/plans/rho-parallel-gepa-completion.md` |
| 7 | Default roles: rollout, analyzer+judge, editor | Minimum model separation for evidence, diagnosis, and editing | `docs/rho_evolution/17-rho-gepa-prompts-and-data-contracts.md` |
| 8 | Dynamic causal blame graphs replace fixed taxonomy | Mechanisms emerge from evidence rather than a pre-defined list | `docs/research/hypotheses-and-validation.md` |
| 9 | Cross-candidate entropy uses comparable evidence floors | Prevents noisy entropy from driving selection too early | `docs/rho_evolution/19-rho-parallel-gepa-research-hypotheses.md` |
| 10 | Edit validation uses origin, worked, regression, and deferred cluster probes | Balance focused validation against over-generalization | `docs/architecture/target-rho-parallel-gepa.md` |
| 11 | Parallel batches use immutable snapshots, exclusive write leases, and coordinator commits | Workers cannot corrupt shared state | `docs/architecture/target-rho-parallel-gepa.md` |
| 12 | Crossover is deterministic provenance-preserving merge by default | Auditable inheritance before any LLM refinement | `docs/migration/gaia-baseline-and-gap-audit.md` |

## Historical Versus Active

| Historical material | Status | How to use it |
| --- | --- | --- |
| `docs/rho_evolution/` (21 files) | Preserved archive | Authoritative rationale, schemas, target architecture, debugging evidence |
| `reference/gaia_evolution_core/` | Read-only reference | Reusable implementation ideas; never a runtime import |
| Gaia runtime adapters, datasets, evaluator internals | Excluded | Do not copy paths, fixtures, labels, or regexes |
| CUGA SDK/package | Pending inspection | Verify package, license, APIs, and artifact model before writing `CUGAAdapter` |
| `src/agent_evolve/core/` and `src/agent_evolve/adapters/` | Active implementation | All new code belongs here and must pass active tests |

## Explicit Deferrals

The following are **not** approved for immediate implementation:

- Parallel proposal batches until `research_sequential` is stable and profile-gated.
- Counterfactual replay until an adapter proves valid checkpoint/state reconstruction.
- Mechanism entropy as a primary selection driver until comparable evidence floors exist.
- Crossover beyond deterministic merge until artifact conflict documentation exists.
- Specialized model-role ablations until the default three-role pipeline is validated.

## Do Not Redo

Do not revisit the following questions without new evidence:

- Whether to fork CUGA instead of using the SDK.
- Whether to import Gaia runtime code into AgentEvolve.
- Whether to allow `src/agent_evolve/core/` to depend on an agent implementation.
- Whether the initial pool should discard RHO proposals.
- Whether to use a fixed failure taxonomy instead of causal blame graphs.

## Links

- `docs/START_HERE.md` — handoff summary and required reading order.
- `docs/architecture/target-rho-parallel-gepa.md` — target system and interfaces.
- `docs/research/hypotheses-and-validation.md` — validation criteria before claims.
- `docs/migration/gaia-baseline-and-gap-audit.md` — retained and rejected legacy ideas.
- `docs/migration/cuga-sdk-integration-notes.md` — SDK-only integration constraints.
- `docs/migration/cuga-adaptation-guide.md` — mapping historical concepts to CUGA-neutral capabilities.
- `docs/migration/self-contained-migration-inventory.md` — what is included, excluded, and active.
- `docs/rho_evolution/README.md` — entry point to the historical archive.
- `reference/gaia_evolution_core/README.md` — reference baseline notes.
