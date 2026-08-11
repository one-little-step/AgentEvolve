# Self-Contained RHO-Parallel-GEPA Migration Design

## Status

Verified migration foundation. This document defines the migration that makes
AgentEvolve independently understandable and implementable without access to
the Gaia repository, its local files, or this conversation.

Verification on 2026-08-11: `uv run pytest -q` passed (8 tests), and the
source-boundary and local-link validation passed. Command output is retained in
the ignored `terminal_output/migration/09_full_test_suite.log` and
`terminal_output/migration/10_structural_validation.log` logs.

## Vision

AgentEvolve evolves externally configurable agent harnesses, not model weights.
It improves reusable agent behavior by analyzing execution evidence, preserving
promising candidate variants, editing declared artifacts such as skills, memory,
policies, prompts, and workflows, and validating improvements against regressions.

The approved long-term target is RHO-Parallel-GEPA:

```text
historical trajectories and a DPP coreset
  -> base plus every RHO candidate in a persistent GEPA pool
  -> common provenance-bearing evaluation
  -> causal blame and artifact-targeted edits
  -> structured edit memory and focused regression validation
  -> entropy/DPP selection, deterministic merge, and optional safe parallelism
  -> transparent, budgeted, feature-gated research ablations
```

CUGA is the intended reference integration because the target needs exact state
and event provenance plus optionally valid checkpoint replay. The generic engine
remains agent-neutral. It adapts to CUGA only through interfaces proven after
inspection of official CUGA SDK documentation and source.

## Problem

The initial AgentEvolve migration retained only distilled architecture and
research summaries. A fresh agent working only in AgentEvolve would therefore
need to rediscover settled decisions, detailed schemas, historical behavior,
debugging findings, feature-gate semantics, and baseline implementation logic.
That duplicates work and risks introducing incompatible redesigns.

## Goals

- Make all RHO/GEPA context required for continuation available inside
  AgentEvolve.
- Preserve the complete evolution documentation set and its original filenames
  and cross-links.
- Preserve portable baseline implementation code without coupling active code to
  Gaia runtime modules.
- Clearly distinguish settled target decisions, historical current behavior,
  research hypotheses, and CUGA-dependent unknowns.
- Make the first implementation task and phased order unambiguous for an agent
  that has no access to the source repository.

## Non-Goals

- Do not copy the Gaia runtime, its agent harness, datasets, secrets, generated
  runs, feedback artifacts, or model credentials.
- Do not claim or simulate CUGA APIs, state fields, artifact types, checkpoints,
  or replay behavior before official inspection.
- Do not make legacy reference code importable by the active AgentEvolve package.
- Do not treat historical baseline behavior as target-complete behavior.

## Repository Layout

```text
docs/
  START_HERE.md
  vision-and-decision-record.md
  rho_evolution/                         # complete preserved documentation set
    README.md
    01-overview.md ... 19-rho-parallel-gepa-research-hypotheses.md
  migration/
    cuga-adaptation-guide.md
    gaia-baseline-and-gap-audit.md
    cuga-sdk-integration-notes.md
  architecture/
    target-rho-parallel-gepa.md
  research/
    hypotheses-and-validation.md
  plans/
    rho-parallel-gepa-completion.md
reference/
  gaia_evolution_core/                   # preserved generic baseline only
    README.md
    contracts.py
    history.py
    operators.py
    population.py
src/agent_evolve/                        # active, independent implementation
  core/
  adapters/
tests/
```

The existing distilled documents remain useful entry points. The complete
`docs/rho_evolution/` archive becomes the authoritative detailed record and
must remain internally navigable.

## Documentation Responsibilities

`docs/vision-and-decision-record.md` is the prominent continuation brief. It
must contain:

- The vision above and the intended RHO-Parallel-GEPA pipeline.
- The distinction between the target architecture and current/legacy Gaia
  implementation.
- Settled decisions: persistent pool, base plus all RHO proposals, common score
  provenance, causal blame, feature gates, entropy evidence floors,
  deterministic merge, safe parallel batches, and replay only through declared
  adapter support.
- The exact CUGA boundary: SDK integration rather than a fork unless inspection
  proves an unsupported required extension; no invented APIs.
- A concise “do not redo” list directing agents to existing detailed documents
  before proposing architecture changes.
- A first-session checklist for inspecting CUGA and implementing the `minimal`
  profile before advanced mechanisms.

`docs/migration/cuga-adaptation-guide.md` maps historical Gaia-specific terms
and code responsibilities to target agent-neutral capabilities. It must show
that `wisdom` is a historical artifact kind, not a generic restriction, and that
future adapters can expose skills, memory, policies, prompts, workflows, and
other declared artifacts.

The guide must explicitly identify each reference module’s reusable behavior and
known limitations. It must direct future work to port selectively into
`src/agent_evolve/`, with tests, rather than import or patch the reference copy.

## Reference Code Boundary

The complete current generic legacy core is copied verbatim into
`reference/gaia_evolution_core/`, excluding only compiled/cache files. A local
README must state:

- It is a read-only historical baseline and behavioral reference.
- It captures useful contracts, edit-history/RAG machinery, mutator and
  crossover protocols, and population orchestration.
- It is not active production code and must never be imported by
  `src/agent_evolve`.
- Its known semantic gaps include synthetic/relative scoring, elite-only
  retention, coarse history, fixed/insufficient pool behavior, and Gaia-shaped
  assumptions described in the migration audit.
- Any port must preserve the target’s agent-neutral adapter boundary and add or
  update tests in AgentEvolve.

## Data and Safety Requirements

- Documentation and reference code must not introduce secrets, evaluator
  internals, expected answers, labels, or regexes into committed traces, prompts,
  history, embeddings, manifests, or logs.
- New runs and verification commands retain tee logging under ignored
  `terminal_output/`.
- Existing paths and links must resolve within AgentEvolve after migration. Gaia
  paths in preserved historical docs must be clearly labelled historical and
  accompanied by adaptation guidance rather than silently rewritten into false
  CUGA claims.

## Verification

The migration is complete only when:

1. Every `docs/rho_evolution/*.md` source document exists in AgentEvolve.
2. Every `agent/evolution_core/*.py` source module exists under
   `reference/gaia_evolution_core/` with a reference-boundary README.
3. The active `src/agent_evolve/` package has no imports from `reference`, Gaia,
   or CUGA.
4. The vision/decision record names the target, settled decisions, deferred CUGA
   work, and first implementation path.
5. Internal relative links in the copied documentation resolve locally.
6. The existing AgentEvolve contract tests still pass.
7. A migration inventory report records source-to-destination document and code
   coverage, plus any intentionally excluded non-portable material.

## Implementation Sequence

1. Copy the complete RHO evolution documentation archive.
2. Copy the generic legacy evolution-core modules into the read-only reference
   boundary and add its README.
3. Add the prominent vision/decision record and CUGA adaptation guide.
4. Update `START_HERE.md`, `README.md`, and `AGENTS.md` to make the detailed
   archive and reference baseline mandatory first reading.
5. Add a migration inventory and automated structural checks.
6. Run structural validation, import-boundary validation, and AgentEvolve tests.
7. Commit only the intended AgentEvolve files.
