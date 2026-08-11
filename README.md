# AgentEvolve

AgentEvolve is an agent-neutral research framework for **RHO-Parallel-GEPA**:
retrospective harness optimization with persistent Pareto candidate pools,
causal trace analysis, feedback-validated artifact editing, structured edit
memory, provenance-preserving crossover, and optional lock-safe parallel
proposal batches.

The next reference adapter is IBM CUGA, integrated through its SDK after its
official source and documentation are inspected. This repository does not fork
or vendor CUGA and must not assume undocumented CUGA APIs.

## Start Here

1. Read `AGENTS.md`.
2. Read `docs/START_HERE.md`.
3. Read `docs/architecture/target-rho-parallel-gepa.md`.
4. Read `docs/research/hypotheses-and-validation.md`.
5. Read `docs/migration/cuga-sdk-integration-notes.md` before installing or
   implementing a CUGA adapter.
6. Read `docs/vision-and-decision-record.md`.
7. Read `docs/rho_evolution/README.md`.
8. Read `docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md`.
9. Read `docs/rho_evolution/19-rho-parallel-gepa-research-hypotheses.md`.
10. Read `docs/migration/cuga-adaptation-guide.md`.
11. Read `docs/migration/self-contained-migration-inventory.md`.
12. Read `reference/gaia_evolution_core/README.md`.

A fresh agent must use these local materials before changing architecture or
attempting CUGA integration. Do not invent CUGA APIs, artifact types, trace
fields, checkpoint behavior, replay semantics, or package names.

The historical RHO-GEPA archive in `docs/rho_evolution/` eliminates any need to
rediscover the RHO-GEPA design from scratch; use it as authoritative rationale,
schemas, debugging evidence, and target architecture, while treating Gaia-specific
paths and runtime assumptions as historical examples.

## Setup

```bash
uv venv
uv pip install -e ".[dev]"
uv run pytest 2>&1 | tee terminal_output/setup/tests.log
```

## Repository Boundary

```text
src/agent_evolve/core/
  Generic evolution contracts and orchestration. Never imports CUGA.

src/agent_evolve/adapters/
  Adapter protocol and concrete CUGA/Pi/Gaia integrations.

docs/
  Target architecture, research hypotheses, migration context, and execution plans.
```

`vendor/` is ignored and reserved only for a read-only CUGA source clone used
for API inspection. The intended production integration is a version-pinned SDK
dependency, not a fork.
