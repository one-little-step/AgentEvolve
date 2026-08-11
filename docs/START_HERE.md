# AgentEvolve Handoff

## Why This Repository Exists

This repository was created to continue a RHO-GEPA evolution effort without
coupling the next phase to the legacy Gaia agent repository. The legacy code
proved several useful primitives but also exposed important flaws: local-relative
scores were compared as a pool-wide Pareto matrix, only elites survived, edit
history was too coarse, and active crossover was LLM synthesis rather than an
auditable system-aware merge.

AgentEvolve starts from the target architecture, not from those constraints.

## What Is Here Now

- An independent Git repository and standalone Python package.
- Agent-neutral adapter contracts in `src/agent_evolve/core/contracts.py`.
- Adapter capability validation in `src/agent_evolve/adapters/base.py`.
- Fresh-agent instructions in `AGENTS.md`.
- Self-contained architecture, research, migration, and implementation context.
- Contract tests proving the foundation does not assume Gaia-specific wisdom
  filenames or replay support.

## What Is Not Implemented

The following are planned, not implemented:

- CUGA SDK dependency or adapter.
- Persistent pool, common score tensor, or Pareto parent selection.
- Analyzer/judge, causal blame graph, semantic clustering, entropy tracker, or
  DPP issue selection.
- Edit-memory RAG, worked/regression sets, merge, locks, batch workers, or replay.
- A production task dataset, rollout runner, or model-provider implementation.

Do not describe these as existing features until implementation and tests prove
them.

## Decision Record

| Decision | Status |
| --- | --- |
| Use CUGA SDK, not a CUGA fork | Approved |
| Keep generic core free of CUGA imports | Approved |
| Use CUGA as target reference adapter | Approved, pending SDK/docs inspection |
| Use Gaia only as historical baseline/reference | Approved |
| Initial pool includes base plus all RHO proposals | Approved |
| Base uses group rollouts; post-RHO candidates initially use one rollout/task | Approved |
| Default roles: rollout, analyzer+judge, editor | Approved |
| Default blame consensus calls | One; consensus/calibration are ablations |
| Dynamic causal blame graphs replace fixed taxonomy | Approved |
| Entropy uses dynamically clustered mechanism alignment | Approved |
| Parallelism is feature-gated and snapshot/lease based | Approved |

## Required Reading

1. `architecture/target-rho-parallel-gepa.md`: target system and interfaces.
2. `research/hypotheses-and-validation.md`: what must be tested before claims.
3. `migration/gaia-baseline-and-gap-audit.md`: what was retained and rejected
   from the legacy implementation.
4. `migration/cuga-sdk-integration-notes.md`: SDK-only integration constraints.
5. `plans/rho-parallel-gepa-completion.md`: phased execution sequence.
6. `vision-and-decision-record.md`: high-level project decisions, CUGA boundary,
   and handoff context.
7. `rho_evolution/README.md`: entry point to the complete historical RHO-GEPA archive.
8. `rho_evolution/18-rho-parallel-gepa-target-architecture.md`: historical target
   architecture that informed AgentEvolve.
9. `rho_evolution/19-rho-parallel-gepa-research-hypotheses.md`: hypotheses and
   validation criteria carried forward from Gaia-era research.
10. `migration/cuga-adaptation-guide.md`: how to map historical concepts to active
    CUGA-neutral capabilities without inventing SDK APIs.
11. `migration/self-contained-migration-inventory.md`: what is included, excluded,
    and active in this repository.
12. `reference/gaia_evolution_core/README.md`: reference implementation notes for
    the Gaia evolution core.

A fresh agent must read these local materials before changing architecture or
attempting CUGA integration. Do not guess CUGA APIs, artifact types, trace
fields, checkpoint behavior, or package names; use only official SDK documentation
and source after inspection.

The historical RHO-GEPA archive in `rho_evolution/` eliminates any need to
rediscover the RHO-GEPA design from scratch; use it as authoritative rationale,
schemas, debugging evidence, and target architecture, while treating Gaia-specific
paths and runtime assumptions as historical examples.

## Fresh-Agent First Actions

1. Read `vision-and-decision-record.md` and `migration/self-contained-migration-inventory.md`
   to understand what is decided, what is excluded, and what must not be redone.
2. Inspect and install the official CUGA SDK only after confirming its package
   name, supported Python versions, licensing, and trace/checkpoint API.
3. Clone CUGA read-only into ignored `vendor/` only if source inspection is
   needed; never modify it inside this repository.
4. Use `migration/cuga-adaptation-guide.md` to map CUGA's public artifact, state
   trace, tool/subagent, and checkpoint APIs to `EvolutionAdapter` capability
   methods.
5. Add adapter contract tests using fakes before writing `CUGAAdapter`.
6. Implement the `minimal` research profile before enabling causal blame,
   entropy selection, merge, or parallelism.
