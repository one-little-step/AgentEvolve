# 02 - Data Model

The pipeline works with a small set of frozen dataclasses and filesystem artifacts. This page describes each one and how they relate.

## TrajectoryRecord

Defined in agent/gaia_lg_react/evolution/models.py.

Fields:
- query: the question or task text
- final_answer: the agent output for the task
- correct: boolean correctness signal
- metadata: extra fields from the source artifact, sanitized
- task_id: normalized identifier such as gaia-8e867cd7
- source_paths: relative paths of source files that contributed
- events: sanitized event records from agent_spans.log or similar
- status: success or failure derived from correctness or explicit status
- summary_text: bounded semantic `trajectory_summary.md` used for embedding when available
- summary_provenance: `captured`, `reconstructed`, or `unavailable`
- trajectory_id: stable selection identity. For combined source runs this is
  `<source-run>::<task-id>`; task_id remains the original value used for reruns.

Historical trajectories come from TrajectoryRunLoader. Fresh rollouts come from _agent_result_to_trajectory, which converts an AgentResult back into a TrajectoryRecord for judging.

### Semantic trajectory summaries

New inference runs write task-local `trajectory_summary.md` artifacts. They
contain bounded outcome, failure, plan, selected ReAct/tool, scratchpad, critic,
and provenance information. A failed or unresolved fresh run may add one bounded
causal narrative; routine successes do not require that extra model call.

For older runs, the loader first prefers a captured summary. When it is absent,
it reconstructs a verbose, redacted summary from nested Traceloop/LangGraph OTel
attributes, result files, and preserved trajectory data. The reconstruction does
not call an LLM and does not pad a genuinely short trace. It prioritizes outcome,
failure signatures, intent/plan, critic, and scratchpad before admitting lower
priority chronology under the 10k soft and 12k hard summary budgets.

### TrajectoryPreparation

`TrajectoryCache` prepares summaries and optional local embedding vectors before
DPP selection. Its result contains the selection identity, prepared summary,
embedding, provenance, and summary/embedding cache hit or miss status.

## CandidateScore

Fields:
- correctness, efficiency, cost: reserved numeric dimensions, currently unused for ranking
- available: False if the judge response could not be parsed
- normalized_score: pairwise preference in [-10, 10]
- confidence: judge confidence in [0, 1]
- rationale: human-readable explanation
- before_id, after_id: task identifiers
- raw_output: the raw model response

Score semantics:
- negative: parent rollout was better
- 0: no meaningful difference
- positive: candidate rollout was better

## Diagnosis

Fields:
- failure_mode: short label for the failure pattern
- root_cause: why it happened
- fix: concise fix description
- severity: low, medium, high, critical
- phase: planner, react, synthesis, or cross_phase
- evidence: concrete evidence from the trajectory digest
- wisdom_direction: actionable guidance for the optimizer

A diagnosis is produced once per selected coreset task by diagnose_trajectory in prompts.py, using the parent bundle as context.

## SelectionResult

Fields:
- selected_ids: task IDs chosen for the round
- method: dpp, random, difficulty, coverage, or deterministic_fallback
- requested_size: the configured coreset size
- valid_count: number of valid records seen
- fallback_reason: why fallback was used, if any
- similarity_mode: `semantic_summary`, `handcrafted`, or
  `handcrafted_fallback`

## WisdomBundle

Fields:
- version: the bundle name, such as base or rho-gaia-1
- files: mapping from filename to content

Allowed files:
- intent_planner.md
- reAct.md
- critic.md
- consolidator.md
- scratchpad.md
- synthesis.md

Bundles are immutable in memory and materialized through materialize().

## EvolutionResult

Return value of EvolutionRound.run():
- status: completed or rejected
- round_dir: path to the round artifact directory
- version_dir: path to the materialized version, empty if rejected
- parent_dir: path to the parent bundle
- parent_snapshot: MD5 hashes of parent files before the round
- parent_dir_snapshot_after: MD5 hashes after the round
- winner_candidate_id: winning candidate or None
- errors: non-fatal errors collected during the round

## EvolutionManifest

A version-level manifest written into the materialized bundle directory. It captures version, parent version, aggregate source digest, aggregate diagnosis, model identifier, and artifact paths back to the round directory.

This is distinct from the round-level manifest.json, which captures operational parameters.
