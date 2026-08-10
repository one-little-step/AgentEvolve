# RHO-GEPA Prompts And Data Contracts

## Purpose

This is an inspection catalog for exact active LLM-facing data flow in the
offline Gaia evolution pipeline. It documents the current prompt templates,
system messages, JSON response contracts, parsing rules, and artifact record
formats. It does not expose secrets because the implementation attempts to
redact prohibited fields before history persistence and prompt construction.

The active prompt sources are:

| Function | File | Used by active GEPA path? |
| --- | --- | --- |
| `diagnose_trajectory()` | `agent/gaia_lg_react/evolution/prompts.py` | Yes, via Gaia adapter |
| `run_mutation()` | `agent/evolution_core/operators.py` | Yes |
| `run_crossover()` | `agent/evolution_core/operators.py` | Yes for configured crossover slots |
| `pairwise_preference()` | `agent/gaia_lg_react/evolution/prompts.py` | Yes, via Gaia adapter |
| `optimize_candidate()` | `agent/gaia_lg_react/evolution/prompts.py` | No, legacy `EvolutionRound.run()` path only |
| `evaluate_module()` | `agent/gaia_lg_react/evolution/gepa.py` | No, currently dormant helper |

## 1. Shared LLM Boundary

The runtime client protocol is:

```python
class LLMClient(Protocol):
    def complete(
        self,
        messages: list[LLMMessage],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse: ...
```

For generic population operators, `GaiaEvolutionLLM.complete()` adapts:

```python
complete(system_prompt: str, user_prompt: str) -> str
```

to:

```python
[
  LLMMessage(role="system", content=system_prompt),
  LLMMessage(role="user", content=user_prompt),
]
```

with `temperature=0.0`, returning `response.content or ""`.

No function call/tool schema is given to the evolution LLM. The model produces
text containing JSON; the application parses and gates that JSON itself.

## 2. Active GEPA Mutation Prompt

### Call site

```text
PopulationEvolution._mutation()
  -> run_mutation(
       GaiaEvolutionLLM(LiteLLMClient),
       enabled candidate editor,
       parent bundle,
       nominal target module,
       diagnoses,
       normalized source trajectories,
       history retrieval
     )
```

### System message

```text
You are an offline agent-policy improver. Return only the requested JSON edits.
```

### User-message prefix

```text
Mutate this advisory agent bundle using only evidence-supported changes. Do not
repeat harmful history unless current evidence materially differs. Return JSON
{edits:[{operation,filename,heading,content}]} using only allowed section edits.
```

The prefix is followed by `json.dumps(packet, default=str)`.

### Packet schema

```json
{
  "parent": {
    "intent_planner.md": "full parent text",
    "reAct.md": "full parent text",
    "critic.md": "full parent text",
    "consolidator.md": "full parent text",
    "scratchpad.md": "full parent text",
    "synthesis.md": "full parent text"
  },
  "target_module": "reAct.md",
  "diagnoses": [
    {
      "failure_mode": "...",
      "root_cause": "...",
      "fix": "...",
      "severity": "...",
      "phase": "...",
      "evidence": "..."
    }
  ],
  "phase_evidence": [
    {"phase": "reAct", "...": "sanitized event fields"}
  ],
  "history": "Previously Helpful Changes\n..."
}
```

### Active data preparation

| Field | Producer | Current behavior |
| --- | --- | --- |
| `parent` | `EvolutionBundle.modules` | Full text for every parent module is sent |
| `target_module` | `_mutation()` | Selected round-robin, not weakness-guided |
| `diagnoses` | `GaiaEvolutionAdapter.diagnose()` | One diagnosis per active source task, not filtered by target module |
| `phase_evidence` | `_phase_packet()` | Up to 20 normalized events whose `phase` equals target filename or filename minus `.md` |
| `history` | `history_packet()` | Rendered coarse records separated into helpful, harmful/rejected, uncertain sections |

`_safe()` removes dictionary keys whose name contains `api_key`, `token`,
`secret`, `expected`, `evaluator`, `regex`, or `label`; it recursively handles
lists and truncates each string to 4,000 characters. It does not inspect generic
string content for secret-like substrings. The history store applies additional
inline assignment redaction.

### Required response shape

```json
{
  "edits": [
    {
      "operation": "append_section",
      "filename": "reAct.md",
      "heading": "Recovery Strategy",
      "content": "After an identical failed tool call, change the query or source."
    }
  ]
}
```

Canonical operations are:

```text
append_section
replace_section
```

The current parser additionally accepts:

```text
add, append -> append_section
replace     -> replace_section
```

Leading Markdown heading syntax is normalized:

```text
"## Recovery Strategy" -> "Recovery Strategy"
```

### Parser behavior

`_apply_edits()`:

1. Finds text between first `{` and final `}`.
2. Parses JSON and requires `edits` to be a list.
3. Checks each filename against the supplied allowed module tuple.
4. Dispatches to `CandidateEditor` section methods.
5. Captures individual operation failure as:

```json
{
  "edit": {"original model edit object": "..."},
  "reason": "..."
}
```

6. Returns:

```python
OperatorResult(
    changed_modules=("reAct.md",),
    raw_output="original model text",
    skipped_edits=(...),
    history_mode="lexical",
)
```

7. Calls `editor.close()` regardless of whether any edit succeeded after a
successful model call.

### Current scope caveat

For mutation, the allowed module tuple is currently `tuple(parent.modules)`.
The model can therefore edit every parent module even though the prompt labels
one module as `target_module`. The planned implementation will instead use:

```python
allowed_modules=(target_module,)
```

and will preserve unrelated output as skipped-edit evidence.

## 3. Active GEPA Crossover Prompt

### Call site

```text
PopulationEvolution._crossover()
  -> run_crossover(
       LLM adapter,
       enabled editor over an ancestor-materialized workspace,
       ancestor bundle,
       left parent bundle,
       right parent bundle,
       diagnoses, trajectories, history,
       left task scores, right task scores
     )
```

### System message

```text
You are an offline agent-policy crossover improver. Return only the requested JSON edits.
```

### User-message prefix

```text
Create an evidence-aware crossover child from the shared common ancestor and
both parents. Preserve supported improvements, resolve conflicts, and avoid
rejected strategies. Return JSON {edits:[{operation,filename,heading,content}]}
only.
```

### Packet schema

```json
{
  "ancestor": {"<module>": "full ancestor text"},
  "left_parent": {"<module>": "full left text"},
  "right_parent": {"<module>": "full right text"},
  "global_task_scores": {
    "left_parent": {"task-id": 2.0},
    "right_parent": {"task-id": -1.0}
  },
  "diagnoses": ["sanitized diagnosis records"],
  "failure_evidence": ["at most 30 sanitized events"],
  "history": "rendered history packet"
}
```

The active packet lacks:

- complete ancestor graph information;
- explicit changed-module sets inferred from bundle text;
- proof that either parent beat the ancestor;
- deterministic per-module source choices;
- prior merge-attempt state.

Consequently this is an LLM synthesis request, not GEPA's deterministic
system-aware merge. The planned merge stage removes LLM authority for disjoint
module inheritance and reserves LLM use for explicit same-module conflict
resolution only.

## 4. Active Trajectory Diagnosis Prompt

### Call site

```text
GaiaEvolutionAdapter.diagnose()
  -> EvolutionRound._diagnose_selected()
  -> diagnose_trajectory(llm, record, parent_bundle)
```

Diagnosis runs once for each supplied `NormalizedTrajectory` converted back into
a `TrajectoryRecord`. It currently diagnoses historical/cohort task records, not
fresh parent rollout traces from the current candidate pool.

### System message

```text
You are an expert trajectory diagnostician for an autonomous question-answering
agent. Respond only with the requested JSON.
```

### User-message template

```text
Diagnose the following Gaia trajectory and propose a targeted wisdom improvement.

Trajectory digest (no raw credentials included):
<JSON trajectory digest>

Parent wisdom bundle contents:
<phase-labelled, maximum-800-character preview of every module>

Return a single JSON object with exactly these keys:
- severity: one of "low", "medium", "high", "critical"
- failure_mode: a short label for the failure pattern
- phase: one of "planner", "react", "synthesis", "cross_phase"
- evidence: concrete evidence from the trajectory digest
- wisdom_direction: actionable guidance to add to the appropriate wisdom file
- root_cause: why the failure occurred
- fix: a concise fix description
```

### Parsed response

```json
{
  "severity": "high",
  "failure_mode": "repeated failed retrieval",
  "phase": "react",
  "evidence": "The same search query was repeated after an empty response.",
  "wisdom_direction": "Require source escalation after repeated empty results.",
  "root_cause": "The ReAct policy lacks a recovery branch.",
  "fix": "Add a tool-failure recovery rule."
}
```

Invalid JSON creates:

```python
Diagnosis(
    failure_mode="parse_failure",
    root_cause="model output was not valid JSON",
    fix="retry with stricter prompt",
    raw_output=raw,
)
```

Unknown phases become `cross_phase`. Note the active diagnosis phase vocabulary
uses `planner` and `react`, while Gaia wisdom filenames use `intent_planner.md`
and `reAct.md`. Planned strict module targeting must normalize these names
explicitly instead of relying on informal string matching.

## 5. Active Pairwise Preference Prompt

### Call site

```text
GaiaEvolutionAdapter.score_rollouts()
  -> pairwise_preference(llm, before, after)
```

For each task, the adapter zips reference and candidate rollout sequences. It
averages available `normalized_score` values. When no reference exists, it
returns a synthetic `0.0` parent baseline without calling the judge.

### System message

```text
You are an expert evaluator comparing agent trajectories. Respond only with the requested JSON.
```

### User-message template

```text
Compare the following "before" and "after" trajectories for the same task and
judge whether the after trajectory is better.

Before trajectory digest:
<JSON before digest>

After trajectory digest:
<JSON after digest>

Return a single JSON object with exactly these keys:
- before_id: the task/trajectory identifier for the before case
- after_id: the task/trajectory identifier for the after case
- confidence: a number between 0 and 1
- rationale: a short explanation of the judgment
- score: an integer or float in [-10, 10] where negative means before is better,
  positive means after is better, and 0 means no meaningful difference
```

### Parsed response and failure behavior

```json
{
  "before_id": "task-42",
  "after_id": "task-42",
  "confidence": 0.85,
  "rationale": "The after trajectory changed source after the initial failure.",
  "score": 4
}
```

Invalid JSON, missing/non-numeric score, or score outside `[-10, 10]` returns:

```python
CandidateScore(available=False, normalized_score=None, raw_output=raw)
```

The current active engine discards confidence during population selection. The
planned common-score and minibatch work should preserve comparison availability
and confidence/provenance for later quality weighting rather than reducing all
results to `float | None`.

## 6. Legacy-Only Candidate Optimization Prompt

`optimize_candidate()` is active only in legacy `EvolutionRound.run()` when
`evolution_tools_enabled=True`. It is included here because it explains why the
repository contains two edit prompt implementations.

Its system message is:

```text
You are an expert prompt engineer optimizing agent wisdom files. Respond only with the requested JSON.
```

It sends previews of the parent bundle, all diagnoses, and a prose description
of allowed operations. Its user message inconsistently lists only three filename
examples in the response requirements even though the editor and earlier prose
allow all six files. The active GEPA path does not call this function; do not
mistake legacy prompt behavior for population behavior.

## 7. Active History Contracts

### Current persisted record

`EditHistoryStore` persists this five-field record:

```json
{
  "record_id": "1-g1-mutation-0-reAct.md",
  "lineage_id": "base",
  "module": "reAct.md",
  "text": "mutation reAct.md score=-1.0",
  "outcome": "harmful"
}
```

For no-change children, `_persist_history()` uses the first adapter module even
if no edit happened. `outcome` is `helpful` whenever `(average_score or 0) >= 0`;
an unavailable score (`None`) consequently becomes helpful. This is current
behavior, not a reliable experimental label.

### Current retrieval result

```python
HistoryRetrieval(
    mode="semantic" | "lexical" | "off",
    records=(EditHistoryRecord(...), ...),
    fallback_reason="..." | None,
)
```

Selection cascade:

```text
same lineage + same module
  -> same module in other lineages
  -> any remaining agent-scoped history
```

The population asks for `minimum_records=1`; this is a fallback threshold, not a
top-K limit. With a large history, all selected records can reach the prompt.

### Planned replacement

The completion plan introduces `EditAttemptRecord` with:

```text
attempt ID, parent IDs, ancestor ID, target module, issue context,
sanitized applied edit summary/diff, skipped edits, retrieved attempt IDs,
minibatch deltas, Pareto scores, explicit status, timestamps, selection result
```

This is necessary for RAG to say what was tried and what happened, rather than
only returning a module name and an average score.

## 8. Artifact Schemas For Inspection

### Active lineage sidecar

```json
{
  "schema_version": "1",
  "candidate_id": "g2-mutation-1",
  "parents": ["rho-g1-elite-1"],
  "ancestor": "rho-g1-elite-1"
}
```

### Active population candidate summary

```json
{
  "candidate_id": "g1-mutation-0",
  "parents": ["base"],
  "ancestor": "base",
  "operator": "mutation",
  "changed_modules": ["reAct.md"],
  "task_scores": {"gaia-task": -1.0},
  "average_score": -1.0,
  "artifact_dir": ".../evolution/g1/candidates/g1-mutation-0"
}
```

`changed_modules` records only accepted editor calls. It does not prove that the
resulting prompt was useful, and it does not include rejected model edits.

### Planned attempt summary

The target artifact should add fields such as:

```json
{
  "attempt_id": "g4-b2-a1",
  "pool_snapshot": 12,
  "parent_id": "candidate-17",
  "target_module": "reAct.md",
  "issue_context": "repeated retrieval after empty result",
  "retrieved_attempt_ids": ["g2-a4", "g3-a1"],
  "status": "accepted",
  "minibatch": {"task_ids": ["..."], "parent_mean": 0.1, "child_mean": 0.3},
  "pareto_score_source": "fixed_baseline_pairwise",
  "edit_log_path": ".../edit_log.jsonl",
  "selection_reason": "task_winner"
}
```

## 9. Prompt/Response Inspection Checklist

For a real generation, audit in this order:

1. Confirm `SOURCE_RUNS`, coreset ID selection, and source parse quality.
2. Inspect diagnoses for phase vocabulary and evidence specificity.
3. Inspect candidate `edit_log.jsonl` to see successful applied operations.
4. If the edit log is absent, inspect captured model output or add structured
   attempt observability; current `population.json` cannot explain all skips.
5. Inspect rollout JSON files for parent and child task behavior.
6. Inspect pairwise judge raw output where persisted by rollout/legacy paths.
7. Compare task-score provenance before trusting Pareto/champion results.
8. Inspect history mode and records, remembering current history does not contain
   actual edit diffs or reliable acceptance labels.
