# Gaia Evolution-Core Reference

This directory is a **read-only historical baseline** copied from the Gaia RHO-
GEPA effort. It preserves reusable implementation ideas and known behavioral
limitations for a future CUGA-neutral implementation. It is not production code,
is not part of the `src` package, and **must never be imported** by active
AgentEvolve code.

## What It Preserves

- Initial agent-neutral bundle, trajectory, adapter, editor, and LLM contracts.
- Append-only redacted edit history with lexical/semantic retrieval fallback.
- Editor-gated mutation and LLM-synthesis crossover protocols.
- Immutable generation artifacts, lineage manifests, rollout caching, and simple
  task-score Pareto selection.

## Why It Is Not The Target Implementation

The baseline has known gaps: parent-relative and synthetic score comparability,
elite-only retention instead of a persistent pool, round-robin target selection,
coarse edit-history outcomes, LLM-first rather than deterministic merge, and
Gaia-shaped module/Markdown assumptions. The approved target is documented in
`../../docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md`.

## Porting Rule

Port behavior selectively into `src/agent_evolve/` only after writing tests
against the active artifact and adapter contracts. Replace Gaia-shaped types with
declared artifact capabilities. Do not patch this snapshot and do not let it set
the CUGA API boundary.
