# CUGA Adaptation Guide

This guide maps historical RHO/GEPA concepts into AgentEvolve's active,
CUGA-neutral capabilities. It does not invent CUGA APIs; it identifies what must
be investigated in the official CUGA SDK before writing `CUGAAdapter`.

## Historical-To-Active Mapping

| Historical concept | Target AgentEvolve capability | CUGA investigation required |
| --- | --- | --- |
| `EvolutionBundle.modules` | `EvolutionCandidate` plus adapter `ArtifactDescriptor` inventory | Exact CUGA artifact grouping, versioning, and write policy |
| Gaia wisdom module | Any declared artifact kind: skill, memory, policy, prompt, workflow, or adapter-defined unit | CUGA artifact metadata and edit surface |
| `NormalizedTrajectory.events` | `ExecutionTrace.events` with immutable provenance | CUGA event, tool, subagent, artifact-read, and final-output data |
| `run_rollouts` | `run_full_rollout` then `capture_trace` | Public task execution and trace retrieval APIs |
| legacy `open_editor` section operations | `apply_structured_edits` in a candidate workspace | Artifact mutation/override and lifecycle APIs |
| legacy replay absence | optional `discover_checkpoints` and `replay_from_checkpoint` | Valid checkpoint/state reconstruction and artifact dependency boundary |
| legacy score map | common provenance-bearing score tensor | CUGA evaluator/task-contract integration |

## Reference-Module Reuse

`reference/gaia_evolution_core/` preserves five read-only modules that may inform
the CUGA adapter design:

- `contracts.py` — initial agent-neutral bundle, trajectory, adapter, editor, and LLM contracts.
- `history.py` — append-only redacted edit history with lexical/semantic retrieval fallback.
- `operators.py` — editor-gated mutation and LLM-synthesis crossover protocols.
- `population.py` — immutable candidate versioning, lineage sidecars, rollout caching, and simple task-score Pareto selection.

These modules are **inputs to design only**. They must never be imported by
`src/agent_evolve/`.

## Reference-Module Limitations

The baseline has known gaps that AgentEvolve corrects:

- Parent-relative and synthetic score comparability.
- Elite-only retention instead of a persistent pool.
- Round-robin target selection.
- Coarse edit-history outcomes.
- LLM-first rather than deterministic merge.
- Gaia-shaped module and Markdown assumptions.

Do not let these limitations set the CUGA adapter boundary.

## CUGA Inspection Checklist

Before implementing `CUGAAdapter`, verify each item from official CUGA documentation or source:

1. **Package / version / license**
   - Official package name and install target.
   - Supported Python versions.
   - License compatible with AgentEvolve usage.
   - Version pinning policy.

2. **Artifacts**
   - Public artifact kinds: skills, policies, memory, workflows, prompts, tools, or others.
   - Artifact grouping and version identification.
   - Read/write API and content format.
   - Materialization/override semantics for candidate workspaces.

3. **Traces**
   - Execution trace export format.
   - Stable event, tool, subagent, and artifact-read identifiers.
   - State transition and final-output fields.
   - Provenance required by causal blame graphs.

4. **Tool / subagent provenance**
   - How tool calls and subagent invocations are recorded.
   - Parent/child relationships between events.
   - How artifact reads are linked to events.

5. **Candidate workspaces**
   - Mechanism to isolate a candidate version for editing.
   - How edits are applied atomically or transactionally.
   - Artifact write policy and lifecycle.

6. **State checkpoints**
   - Whether checkpoints are exposed publicly.
   - Format and identifiers.
   - State hash or reconstruction contract.

7. **Replay validity**
   - Public support status for resume/replay.
   - Artifact override behavior during replay.
   - Determinism guarantees and limits.

8. **Concurrency behavior**
   - Thread-safety and process-safety of SDK clients.
   - Concurrent agent execution limitations.
   - Shared-state commit semantics if any.

9. **Error semantics**
   - Exception types and retry guidance.
   - How partial failures are reported.
   - Timeout and cancellation behavior.

## Mapping Output

Use the checklist above to fill in the following adapter mapping before writing code:

| AgentEvolve method | CUGA SDK call(s) | Notes |
| --- | --- | --- |
| `artifact_inventory(version)` | TBD | |
| `read_artifacts(version, artifact_ids)` | TBD | |
| `materialize_candidate(parent_version, attempt_id)` | TBD | |
| `apply_structured_edits(workspace, edits)` | TBD | |
| `run_full_rollout(workspace, task, rollout_id)` | TBD | |
| `capture_trace(rollout_result)` | TBD | |
| `supports_counterfactual_replay()` | TBD | Return `False` until proven valid. |
| `discover_checkpoints(trace)` | TBD | |
| `replay_from_checkpoint(checkpoint, workspace, task, rollout_id)` | TBD | |

Keep this table updated as SDK inspection proceeds.
