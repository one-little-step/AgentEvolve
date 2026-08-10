# AgentEvolve Instructions

## Mission

AgentEvolve is an independent, agent-neutral RHO-Parallel-GEPA project. It will
use IBM CUGA through its SDK, not a CUGA fork. CUGA source code and documentation
are not yet vendored or installed; do not invent CUGA APIs, trace fields,
checkpoint behavior, artifact types, or package names.

The immediate objective is to build a generic evolution core and an adapter
boundary. The intended reference adapter is CUGA because the next research phase
requires exact agent-state tracing, artifact provenance, and optionally valid
checkpoint replay. Gaia is historical context only, not a runtime dependency of
this repository.

## Non-Negotiable Boundaries

- `src/agent_evolve/core/` is agent-neutral and must never import `cuga` or any
  agent implementation.
- `src/agent_evolve/adapters/` contains the abstract adapter contract and future
  CUGA/Pi/Gaia adapters.
- Use the CUGA SDK as a pinned dependency after inspecting its official package
  and documentation. Keep any CUGA clone read-only under ignored `vendor/`.
- Never assume a generic trace can be replayed. Replay is available only when an
  adapter explicitly reports a valid checkpoint/state reconstruction capability.
- Artifacts can be wisdom, skills, memory, policies, workflows, or other adapter
  declared editable units. Do not hardcode Gaia wisdom filenames or Markdown
  section editing in the generic core.
- Capture every test, smoke run, and migration verification command with
  `2>&1 | tee terminal_output/<topic>/<name>.log`.
- Never persist credentials, expected answers, evaluator internals, labels, or
  regexes to edit memory, embeddings, prompts, manifests, or terminal logs.
- Add tests before implementation changes. Preserve a clean distinction between
  current implementation, research hypothesis, and target architecture.

## Architecture Decisions Already Made

- Use a persistent pool: base plus every initial RHO proposal are retained.
- Base receives `G` rollout group evidence; post-RHO candidates initially receive
  one rollout per selected task to preserve RHO-scale cost.
- Default model roles: rollout, analyzer+judge, editor. Specialized model roles
  are optional ablation overrides.
- Causal blame graphs replace a fixed failure taxonomy.
- Cross-candidate entropy requires comparable evidence floors before it drives
  selection. Mechanisms align through task-local semantic clusters anchored by
  base-harness observations.
- Edit validation uses origin cases, worked sets, regression probes, deferred
  cluster-level generalization probes, retry exhaustion, and protected floors.
- Parallel batches use immutable snapshots, exclusive artifact write leases, and
  coordinator-only shared-state commits.
- Crossover is provenance-preserving deterministic merge by default; an editor
  may resolve only documented same-artifact conflicts.

## Required Reading Order

1. `docs/START_HERE.md`
2. `docs/architecture/target-rho-parallel-gepa.md`
3. `docs/research/hypotheses-and-validation.md`
4. `docs/migration/gaia-baseline-and-gap-audit.md`
5. `docs/migration/cuga-sdk-integration-notes.md`
6. `docs/plans/rho-parallel-gepa-completion.md`
7. `docs/vision-and-decision-record.md`
8. `docs/rho_evolution/README.md`
9. `docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md`
10. `docs/rho_evolution/19-rho-parallel-gepa-research-hypotheses.md`
11. `docs/migration/cuga-adaptation-guide.md`
12. `reference/gaia_evolution_core/README.md`

The historical RHO-GEPA archive in `docs/rho_evolution/` eliminates any need to
rediscover the RHO-GEPA design from scratch; use it as authoritative rationale,
schemas, debugging evidence, and target architecture, while treating Gaia-specific
paths and runtime assumptions as historical examples.

## Current State

Only the capability contracts and migration context are initialized. Do not claim
that persistent pool selection, causal blame, semantic clustering, CUGA tracing,
counterfactual replay, or parallel GEPA are implemented until tests and adapters
prove them.
