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

- Use a persistent pool: base plus every initial RHO proposal are retained. Every
  score cell ever recorded stays in the tensor; nothing is deleted mid-run.
- **Generational retirement (SV-13).** Retention is about *evidence*, not about
  breeding rights. When an accepted offspring is preferred over its parent by the
  RHO symmetric pairwise judge, the parent is **soft-retired**: excluded from
  parent sampling, the Pareto frontier and champion selection, while its score
  cells, lineage and preference record are all kept. An offspring is generated to
  fix its parent's diagnosed faults, so continuing to breed from a version its own
  descendant improved on spends rollouts re-deriving a fix that already exists.
  - Soft, never pruned: hard-deleting an entry would destroy the comparable cells
    cross-candidate entropy needs and the negative evidence a later analysis
    wants. `pool.prune()` remains ablation-only.
  - The judge decides, not the arithmetic. Numeric dominance cannot see whether a
    child solved the parent's failure *mechanism*; the pairwise judge reads
    trajectories. One instrument governs retirement, promotion (SV-4) and final
    resolution, so a candidate can never be retired by one standard and promoted
    by another.
  - Conservative on missing evidence: no judge, an unavailable verdict, a tie, an
    incomplete trace pair, or a raising judge all leave the parent alive. A judge
    outage must never silently shrink the breeding population.
  - The live population is never emptied, and the base is retired only if some
    descendant supersedes it while another live entry remains.
  - Terminal condition: if the live pool shrinks to one, that candidate is the
    winner outright. Otherwise survivors are resolved by symmetric pairwise
    preference over the coreset.
  - Costs judge calls only. Both trace sets already exist at commit time — the
    parent's from `build_issues`, the child's from `validate` — so retirement adds
    `2k` model calls and **zero** rollouts.

- Base receives `G` rollout group evidence; post-RHO candidates initially receive
  one rollout per selected task to preserve RHO-scale cost.
- Default model roles: rollout, analyzer+judge, editor. Specialized model roles
  are optional ablation overrides.
- Causal blame graphs replace a fixed failure taxonomy.
- Cross-candidate entropy requires comparable evidence floors before it drives
  selection. Mechanisms align through task-local semantic clusters formed
  dynamically as mechanisms arrive: an embedding cosine pre-filter decides the
  clear cases for free, and a dedicated small dedup model adjudicates only the
  ambiguous band, because measurement shows cosine alone cannot separate analyzer
  paraphrase from a genuinely different fault (the same-fault and
  different-fault similarity distributions overlap). See
  `docs/design/issue-lifecycle.md`. Mechanism identity is deliberately
  **task-local**: variance is computed within one task across candidates, so an id
  never needs meaning outside its task. Cross-task pooling is deferred.
  - Superseded 2026-08-21: this line previously read *"anchored by base-harness
    observations"*. `MechanismClusterer.add_anchor(force_new=True)` exists but has
    never had a caller in `src/`, and as built it does not work — anchors embed
    bare mechanism text while observations embed mechanism plus actor plus
    artifacts, so an identical mechanism scored only 0.756 against its own anchor
    and two anchors plus their two matching observations produced four clusters
    rather than two.
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
12. `docs/migration/self-contained-migration-inventory.md`
13. `reference/gaia_evolution_core/README.md`

A fresh agent must use these local materials before changing architecture or
attempting CUGA integration. Do not invent CUGA APIs, artifact types, trace
fields, checkpoint behavior, replay semantics, or package names.

The historical RHO-GEPA archive in `docs/rho_evolution/` eliminates any need to
rediscover the RHO-GEPA design from scratch; use it as authoritative rationale,
schemas, debugging evidence, and target architecture, while treating Gaia-specific
paths and runtime assumptions as historical examples.

## Current State

Only the capability contracts and migration context are initialized. Do not claim
that persistent pool selection, causal blame, semantic clustering, CUGA tracing,
counterfactual replay, or parallel GEPA are implemented until tests and adapters
prove them.
