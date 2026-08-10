# 01 - Overview and Goals

## What this system does

The RHO wisdom-evolution pipeline is an offline, self-contained trainer for the
Gaia agent's prompt-level wisdom. It looks at historical task trajectories,
finds a small but diverse subset of failures, figures out why the current
prompt bundle caused those failures, proposes edited prompt bundles, reruns the
same tasks with the old and new bundles, and promotes the best edit to a new
versioned bundle.

## Key design decisions

1. **Offline.** The pipeline only reads and writes files under the repository.
   It makes LLM calls, but those are the only external dependency.
2. **Versioned and immutable.** Every accepted candidate becomes a new
   directory under `policies/evolved_context/`. Old versions are never mutated.
3. **Progressive.** Multiple rounds can form a chain:
   ```text
   base -> rho-gaia-1 -> rho-gaia-2 -> ...
   ```
4. **Hardcoded configuration.** The runner stores all knobs in the same file as
   constants (`dataset/evolve_run.py`). This is intentional for reproducibility.
5. **Gated edits.** Wisdom editing is only enabled inside evolution. Normal
   agent runs cannot edit wisdom files.
6. **Non-destructive.** The parent bundle is snapshotted before the round and
   its content is verified after the round.

## Terminology

| Term | Meaning |
|------|---------|
| **Wisdom bundle** | A versioned set of six phase prompt files: planner, ReAct, critic, consolidator, scratchpad, and synthesis guidance. |
| **Parent** | The wisdom bundle used as the starting point for the current round. |
| **Candidate** | A proposed new wisdom bundle derived from the parent. |
| **Trajectory** | One attempt at one task, including query, final answer, correctness, status, and events. |
| **Coreset** | The small set of trajectories selected for diagnosis and evaluation. |
| **DPP** | Determinantal Point Process, the default diversity-aware selector. |
| **Judge model** | The LLM that compares a parent rollout against a candidate rollout. |
| **Round** | One complete pass: load, select, diagnose, generate, evaluate, promote/reject. |
| **Progressive chain** | Sequential rounds where each accepted version becomes the next parent. |

## Files that matter

| File | Role |
|------|------|
| `dataset/evolve_run.py` | Hardcoded runner and configuration. |
| `agent/gaia_lg_react/evolution/round.py` | `EvolutionRound`: main orchestrator. |
| `agent/gaia_lg_react/evolution/selection.py` | Coreset selection strategies. |
| `agent/gaia_lg_react/evolution/trajectory_loader.py` | Historical trajectory discovery and normalization. |
| `agent/gaia_lg_react/evolution/prompts.py` | Diagnosis, optimization, and pairwise-judge prompts. |
| `agent/gaia_lg_react/evolution/wisdom.py` | Loading, validation, and materialization of bundles. |
| `agent/gaia_lg_react/evolution/edit_tools.py` | Gated, candidate-scoped section editing. |
| `agent/gaia_lg_react/evolution/models.py` | Frozen dataclasses for trajectories, scores, diagnoses, manifests. |
| `agent/gaia_lg_react/runner.py` | The actual agent execution function that runs a query with a config. |

## What the system does NOT do

- It does not mutate source runs.
- It does not use `agent_spans.log` as a replacement for live rollouts.
- It does not automatically push or merge versions.
- It does not implement response caching yet (`CACHE_MODE` only accepts `"off"`).
- It does not compare candidates across rounds.
