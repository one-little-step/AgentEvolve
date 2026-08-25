# Unified CUGA Editor Agent — Design

Date: 2026-08-15
Status: design approved, not implemented
Supersedes: `FakeEditor` as the only `Editor` implementation

## 1. Problem

`src/agent_evolve/core/fake_editor.py` is the only implementation of the
`Editor` protocol. It reads the expected answer out of the task contract and
writes it into the artifact:

```python
expected = request.task.expected_contract.get("expected_substring", "")
```

That is deterministic answer injection, not optimization. It is acceptable as a
fixture for pool and validation mechanics; it must never produce a reported
research result. Until a real editor exists, the evolution loop cannot
demonstrate self-improvement, and the seed generator / RHO proposal stage has
nothing meaningful to seed.

Phase 8 proved the adapter boundary carries edits to the agent and the causal
DAG back to the analyzer. The editor is the remaining inert component.

## 2. Goals

1. Replace `FakeEditor` with an editor that proposes artifact changes from
   causal evidence, never from the expected contract.
2. Give one editor component both capabilities — refining a single parent and
   combining several parents — in a single invocation, so it reasons about the
   full editing context at once.
3. Keep `src/agent_evolve/core/` agent-neutral. The core must not learn that
   the editor is a CUGA agent.
4. Make every failure mode of the editor an explicitly recorded outcome, so a
   non-functioning editor is detectable rather than silently indistinguishable
   from a legitimate decision not to edit.

### Non-goals

- Implementing the RHO outer stage or a seed generator.
- Fixing `core/merge.py` (see §14).
- Counterfactual simulation or tentative-state construction (see §5, Deliberate omissions).
- Making the editor agent's own skills evolvable.

## 3. Approved decisions

| Decision             | Choice                                          | Rationale                                                                                                                            |
| -------------------- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Editor substrate     | CUGA agent, multi-turn                          | A single LLM call handling evidence reading, history, mutation and combination degrades under attention load.                        |
| Mutation / crossover | One unified call, no mode flag                  | Maximum flexibility; the agent picks its own strategy, nudged toward balance by prompting.                                           |
| Crossover authority  | Free-form                                       | Deliberate deviation from `merge-resolution.md`; the agent may propose any artifact content rather than resolving a scoped conflict. |
| Isolation            | Same process, explicit trace detachment         | Simpler than subprocess isolation; residual risk accepted and guarded by test.                                                       |
| Evidence scope       | Blame + artifacts + history + task `input_text` | Excludes `expected_contract` and `final_output`.                                                                                     |
| Trace payloads       | Event metadata + `tool_call` payloads           | Tool observations are environment evidence; guarded by the contamination check in §8.                                                |
| Edit extraction      | Terminal submit tool                            | Bets on tool-body execution (proven live) rather than model narration (proven unreliable).                                           |
| Artifact creation    | Allowed, namespaced and capped                  | Adding a missing skill is a high-value mutation; bounded to stay auditable.                                                          |
| Parent set           | Primary sample + K−1 Pareto donors, K=3         | Bounds prompt growth to a constant as the pool grows.                                                                                |

## 4. Architecture

```
core/                          (agent-neutral, protocol unchanged)
  editor.py                    Editor protocol, EditorRequest/Response,
                               validate_editor_plan, repair_once_then_classify
  orchestrator.py              propose_edits, select_parents, commit_to_pool

adapters/
  cuga_editor.py       NEW     CugaEditorAgent implements Editor
  cuga_editor_tools.py NEW     adapter-interfaced tool clusters
  cuga_adapter.py      EXTEND  create operation, namespaced write authorization
```

Three invariants:

1. **`CugaEditorAgent.propose_edit(request) -> EditorResponse`** satisfies the
   existing protocol. The entire multi-turn agent loop is inside that one call.
   `repair_once_then_classify` continues to work unmodified.
2. **Tools close over the request, not over global state.** Each
   `propose_edit` builds a fresh toolset bound to that request's write set.
   Authorization is enforced inside the tool body (the actual write boundary)
   and re-checked independently by `validate_editor_plan` afterward.
3. **No tool touches CUGA internals or the filesystem.** Every tool reads or
   writes through the adapter, which keeps the editor decoupled from the
   runtime and testable with an in-memory adapter.

## 5. Tool clusters

Clustered with `@tool` over `@tracked_tool(app_name=...)`, following the
existing pattern in `src/agent_evolve/cuga_wrapper/tools.py`.

### `app_name="evidence"` — read the failure, no writes

| Tool                                          | Returns                                                                                                         |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| `get_mechanism()`                             | mechanism description, severity                                                                                 |
| `list_blamed_actors()`                        | actor ids, blame weights, attributed artifacts                                                                  |
| `get_task_input()`                            | the task's `input_text` only                                                                                    |
| `list_trace_actors()`                         | distinct actor ids in the trace                                                                                 |
| `read_trace_events(kind=, actor_id=, limit=)` | filtered event window: `event_id`, `kind`, `actor_id`, `parent_event_id`, `sequence`, plus `tool_call` payloads |

`*_ref` payload values are stripped and never dereferenced. Content-addressed
blobs hold raw prompts and AgentState and remain closed, per the standing
decision recorded in the Phase 8 handoff.

### `app_name="harness"` — read and write artifacts

| Tool                                  | Behavior                                                                         |
| ------------------------------------- | -------------------------------------------------------------------------------- |
| `list_artifacts()`                    | write-set ids and kinds; separately, readable-but-not-writable ids               |
| `read_artifact(artifact_id)`          | current content                                                                  |
| `stage_replace(artifact_id, content)` | stages a replacement; authorization checked immediately                          |
| `stage_create(artifact_id, content)`  | stages a new artifact; namespace prefix and per-attempt cap enforced immediately |
| `list_staged()`                       | the edits staged so far                                                          |
| `unstage(artifact_id)`                | discards one staged edit                                                         |

Writes are **staged incrementally, then finalized once**. Each staging call
validates authorization on the spot, so the agent gets per-artifact feedback
while it works rather than a single all-or-nothing rejection at the end. Staged
state is wrapper-side and discarded if the agent never finalizes.

#### Creation namespace and caps

| Rule                 | Value                                |
| -------------------- | ------------------------------------ |
| Permitted id pattern | `skills/generated-<name>`            |
| Per-attempt cap      | 2 new artifacts                      |
| Pool-wide cap        | 10 generated artifacts, configurable |

The prefix is **CUGA-group-first by necessity, not style**. `_harness_slot`
(`cuga_adapter.py:122-138`) accepts only `instructions` or a
`skills|policies|memory/<name>` prefix, so a flat `generated/<name>` would raise
`ValueError` at registration and the creation path would be dead on arrival. The
`generated-` infix keeps created artifacts distinguishable from seeded ones for
provenance and cap accounting.

Creation is restricted to `skills` for now. Policy and memory creation are
plausible — a created policy could supply the trigger that CUGA's unreliable
`load_skill` needs — but neither is proven, and each widens the gaming surface.
Revisit once skill creation is observed working.

Both caps are enforced in the `stage_create` tool body. Exceeding either returns
a rejection with the current count, not an exception.

### `app_name="history"` — learn from past attempts

| Tool                              | Behavior                                                           |
| --------------------------------- | ------------------------------------------------------------------ |
| `search_edit_history(limit<=5)`   | sanitized `MemoryRecord`s for this issue via `EditMemory.retrieve` |
| `get_attempt_outcome(attempt_id)` | worked / failed / regression, with reason                          |

Retrieval is bounded, consistent with `memory.py`'s existing contract. Records
are reference-based and already sanitized; no raw payloads are exposed.

### `app_name="parents"` — the combination surface

| Tool                                           | Behavior                                                      |
| ---------------------------------------------- | ------------------------------------------------------------- |
| `list_parents()`                               | primary parent id plus donor ids, with per-task score summary |
| `read_parent_artifact(parent_id, artifact_id)` | that parent's artifact content                                |

`read_parent_artifact` records which parents were actually read. That record —
not the agent's prose — is the provenance source for `parent_ids` (§9).

### `app_name="submit"` — terminal

`submit_edit_plan(rationale, risks, expected_effect)`

Finalizes whatever is currently staged. It takes no `edits` argument: the edits
are the staged set, which was already validated per-artifact at staging time.
This keeps one mechanism for writes rather than two.

It re-validates the complete staged set, captures the plan into wrapper-side
state, and returns either a confirmation or a precise rejection reason. It
**returns** a rejection rather than raising, so the agent can correct itself
within the same run before the core's one-correction repair protocol is needed.

Calling it with nothing staged is the explicit decline path (`no_op` in §10),
provided a rationale is given.

### Deliberate omissions

- **No simulation tools.** `build_tentative_state` and
  `simulate_counterfactual` (qf28 P1/P2) require the tag-injection scheme,
  which does not exist, and `supports_counterfactual_replay()` is `False` by
  standing decision. Real validation already runs after submission.
- **No `validate_edit_syntax` / `preview_edit_effect`.** For whole-artifact
  markdown replacement these are near-tautological;
  `submit_edit_plan`'s own validation covers the real constraint.

## 6. Editor agent skills and prompting

The agent needs to know *how* to evolve a harness effectively, not merely which
tools exist. Guidance is split by CUGA's own injection semantics, per
`feedback/gpt_context/cuga_skills_polices_etc.md`: instructions are
always-relevant behavioral configuration; skills are on-demand procedures loaded
via `load_skill`.

**`special_instructions` (always present)** — the invariants that must never be
violated regardless of what the agent decides to do:

- write only into the authorized set; a rejection means rethink, not retry-harder
- prefer minimal targeted change over wholesale rewriting
- ground every edit in the blame evidence, not in general improvement instincts
- declining to edit is a legitimate outcome; say so with a rationale
- always finalize with `submit_edit_plan`, including when declining

**Skills (loaded on demand)** — procedural playbooks for distinct strategies:

| Skill                | Teaches                                                                                                      |
| -------------------- | ------------------------------------------------------------------------------------------------------------ |
| `refine-artifact`    | reading blame, locating the responsible artifact, making a minimal targeted change                           |
| `combine-parents`    | comparing donor artifacts against the primary, transplanting a capability without discarding working content |
| `create-artifact`    | recognizing that no artifact covers the failure mode, and authoring a new one within namespace rules         |
| `learn-from-history` | interpreting worked/failed/regression records; not repeating a strategy that regressed                       |

**Balance nudge.** Since mutation and combination share one call with no mode
flag, the instructions state both are available and that the choice should
follow the evidence: refine when blame points at a specific artifact the primary
already owns; combine when a donor scores better on the failing task; create
when no artifact addresses the mechanism at all. This is a prompt-level nudge,
not an enforced policy — the agent's chosen strategy is recorded so the actual
mutation/combination mix is measurable rather than assumed.

**Honest limitation.** These skills are hand-authored by us. Edit quality is
therefore bounded by our prompt writing, and CUGA's known unreliable
`load_skill` invocation means a skill may never load at all. Which skills were
actually loaded is recorded per attempt, so this is observable rather than
silent.

## 7. Request and response flow

```
orchestrator.propose_edits
  -> select_parents()            primary sample + K-1 Pareto donors
  -> materialize_candidate(primary.version, attempt_id)
  -> read_artifacts(primary.version, write_set)
  -> EditorRequest(analysis, write_set, current_artifacts,
                   history_refs, parents=(...))
  -> repair_once_then_classify(CugaEditorAgent, request)
       -> propose_edit:
            build toolset bound to this request
            construct CugaAgent: tracing detached, no workspace
            await initialize(); await invoke(prompt)
            read captured plan from wrapper state
            ignore the prose answer entirely
  -> EditorResponse | None
```

`EditorRequest` gains optional fields with defaults. It is a frozen dataclass,
so this is backward-compatible: existing tests and `FakeEditor` continue to
work untouched.

```python
parents: tuple[ParentContext, ...] = ()
```

`ParentContext` carries `candidate_id`, `version`, and a per-task score
summary. Donor artifact content is fetched on demand through
`read_parent_artifact`, not inlined into the request.

The write set remains the primary parent's `issue.writable_artifact_ids`,
widened only by the creation namespace. Donors are read-only: the agent may
draw content from a donor but always writes into the primary's workspace.

## 8. Evidence boundary and contamination guard

The editor never receives:

- `task.expected_contract` — the expected answer
- `trace.final_output` — frequently contains or hints at the expected answer
- payload blobs — raw prompts and AgentState

Because `tool_call` payloads are exposed and a tool result could contain
answer-shaped free text, a **fail-closed contamination guard** runs before the
agent is invoked: every assembled tool payload is scanned for any string value
from the task's `expected_contract`; a match causes that payload to be dropped
and the redaction recorded.

The guard *consumes* `expected_contract` but never *shows* it. Without the
guard, `tool_call` payload exposure is an unmonitored channel; with it, it is a
monitored one. This mirrors the read-back verification already used for storage
in `examples/run_phase_6_b1.py:216`.

Key-name denylisting alone (`memory.sanitize_payload`) is insufficient here: it
matches keys such as `expected_answer`, not an expected answer appearing as
free text inside a tool result string.

## 9. Provenance

`commit_to_pool` currently hardcodes a single parent
(`orchestrator.py:1221`). Under unified editing that would be wrong: lineage
would claim one parent when the agent drew content from several.

`parent_ids` is therefore derived from `read_parent_artifact` tool-execution
evidence — the same tool-body-over-narration principle that makes
`ingest_sdk_tool_calls` correct and `ToolObservationRecorder.wrap` dead on live
CUGA. The primary parent is always included; donors appear only if actually
read.

`lineage_of` (`editor.py:484-492`) already accepts multiple parent versions and
joins them sorted, so retry-budget scoping needs no change.

## 10. Outcome taxonomy

| Outcome                         | Meaning                                                         | Classification                                                                     |
| ------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| finalized, valid                | staged set captured, authorization passed                       | normal path                                                                        |
| staging rejected                | a `stage_*` call refused the write                              | agent may correct in-run; if it finalizes nothing, falls through to the rows below |
| `submit_edit_plan` never called | agent produced prose, or staged without finalizing              | **`no_tool_call`**                                                                 |
| explicit decline                | `submit_edit_plan` called with nothing staged, plus a rationale | `no_op`                                                                            |
| agent raised                    | CUGA execution error                                            | `unavailable`                                                                      |

`no_tool_call` must remain distinct from `no_op`. Collapsing them would let
"the agent did not engage" masquerade as "the agent judged no edit warranted" —
the same category of error that produced the retracted Phase 8 E2E PASS.

## 11. Testing strategy

Tests precede implementation, per AGENTS.md.

**Offline unit tests (no CUGA, no network)** — the tool bodies are plain
functions over an in-memory adapter, so the whole authorization and evidence
surface is testable offline:

- `stage_create` rejects ids outside the authorized namespace
- `stage_create` rejects a flat `generated/` id, proving the CUGA-group-first
  prefix requirement is enforced rather than assumed
- `stage_create` enforces the per-attempt cap of 2
- `stage_create` enforces the pool-wide cap
- `stage_create` rejects `policies/generated-*` and `memory/generated-*` while
  skills-only creation is in force
- `stage_replace` rejects ids outside the write set
- `stage_*` returns a rejection (does not raise) on unauthorized writes
- `unstage` removes a staged edit; `list_staged` reflects current state
- staged edits are discarded when `submit_edit_plan` is never called
- `read_trace_events` strips `*_ref` values
- the contamination guard drops a payload containing an `expected_contract` value
- `get_task_input` exposes `input_text` and nothing else
- no tool exposes `expected_contract` or `final_output` — asserted by scanning
  every tool's output for task-contract values
- `parent_ids` reflects exactly the parents read via `read_parent_artifact`
- `lineage_of` produces a stable sorted key for a multi-parent attempt
  (confirms the §15 verification with a regression test)
- `reserve()` raises `BudgetExceededError` once `max_editor_calls` is reached,
  and does not cap when it is `None`
- each outcome in §10 maps to its recorded classification, with `no_tool_call`
  distinct from `no_op`
- which editor skills were loaded is recorded per attempt

**Agent-level tests with a stubbed CUGA agent** — a fake agent that invokes a
scripted tool sequence, verifying `propose_edit` returns a valid
`EditorResponse` and that the prose answer is ignored.

**Isolation regression test** — asserts the editor agent's own LLM calls never
appear in a rollout trace. This guards the residual risk accepted in §13: if a
future CUGA version breaks the same-process assumption, a test fails rather
than an experiment silently passing.

**Live verification script** — one real editor invocation against the existing
56-event reference trace at
`data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781`, reporting tools called,
outcome classification, and whether any contamination was detected.

All runs captured with `2>&1 | tee terminal_output/<topic>/<name>.log`.

## 12. Budget

`BudgetUsage.editor_calls` exists (`config.py:49`) but is the **only** occurrence
of that name in the module: `reserve()`'s `limit_fields` map (`config.py:57-63`)
has no entry for it, so editor calls are counted and never capped. That was
tolerable when an editor call was one LLM call. It is not tolerable now.

Required change:

- add `max_editor_calls: int | None = None` to `BudgetLimits`
- add `"editor_calls": "max_editor_calls"` to `reserve()`'s `limit_fields`
- default `None` (uncapped), so no existing test or profile changes behavior;
  experiment runners set it explicitly

Amplification is larger than a naive per-attempt count suggests. One attempt is
one editor *call* but 10-40 internal LLM calls, and
`repair_once_then_classify` can invoke the editor twice. So a 20-attempt run
carries up to ~1600 internal editor-side LLM calls, on top of the origin and
regression **rollouts** each attempt already performs. Capping
`max_editor_calls` bounds the invocation count; `max_model_tokens` remains the
backstop for total spend.

## 13. Known limitations

- **The editor agent's own skills are hand-authored.** Edit quality is bounded
  by our prompt writing. Making those skills evolvable would be recursive
  self-improvement and is out of scope.
- **CUGA process-global state remains shared.** The singleton
  `ActivityTracker` (`tracker.py:92-94`) and the global policy DB are shared
  between the editor agent and rollout agents in one process. Accepted;
  guarded by the isolation regression test.
- **Tool invocation reliability is unproven for this model.** In isolated live
  runs the model stopped at ~92 completion tokens with `finish_reason:"stop"`
  and never called `load_skill`. It may not call these tools either. This is
  why `no_tool_call` is a first-class recorded outcome rather than an anomaly.
  If tool invocation fails consistently, the fallbacks are: a model with better
  tool-calling reliability, structured-output mode instead of tools, or a
  simpler single-call editor. The live verification script reveals this on the
  first run.
- **The `instructions` artifact is a single flat scalar.**
  `cuga_adapter.py:24` maps it to one `special_instructions` argument, while
  `feedback/gpt_context/cuga_skills_polices_etc.md` establishes that CUGA
  instructions are per-component (`api_planner.md`, `code_agent.md`,
  `answer.md`). The editor therefore cannot target the planner separately from
  the answer node, even though blame graphs name those actors distinctly. This
  bounds achievable edit precision. Not addressed here.

## 14. Deviations from the architecture

- **Free-form crossover bypasses `merge-resolution.md:96-104`**, which mandates
  deterministic three-way inheritance with the LLM receiving only a single tied
  conflicting artifact's ancestor/left/right content. Chosen deliberately for
  editor flexibility. Consequence: `MergeProvenance` and
  `ArtifactMergeDecision` (`contracts.py:326-406`) are unused on this path, and
  per-artifact three-way provenance is replaced by the observed-parent lineage
  in §9.
- **`core/merge.py` is not fixed.** It has three defects against
  `merge-resolution.md`: identical parent edits are misclassified as conflicts
  (`merge.py:335` never compares left to right); evidence resolution rules 1–3
  are absent, so every conflict would reach an LLM; and `_blame_for_artifact`
  (`merge.py:250`) sums raw blame instead of computing
  `severity * confidence * score` over citing cells. It also has a lossy hole
  rather than a permissive one: `plan_merge:305` drops artifacts absent from
  base with `continue`, so "present in ancestor, absent on one side" never
  becomes a conflict and an artifact one parent deleted can vanish with no
  decision recorded. `plan_merge` has zero production callers, so none of this
  is currently reachable.

## 15. Review record (qf30)

`feedback/from_qwen/qf30.md` reviewed this spec: **approved**, with 3 required
actions and 6 claims marked unverified. Every unverified claim was checked
against the code and confirmed:

| Claim                                       | Evidence                                                                                              |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| `lineage_of` accepts multiple parents       | `editor.py:490-491` joins sorted; `record_attempt:499` already threads `parent_versions`              |
| `commit_to_pool` hardcodes one parent       | `orchestrator.py:1221` `parent_ids=(parent_entry.candidate_id,)`                                      |
| `merge.py:335` never compares left to right | bare `else:` after three `changed` tests; `left_contents[aid] == right_contents[aid]` never evaluated |
| `merge.py:250` sums raw blame               | `total += n.blame`, no severity/confidence factor                                                     |
| `plan_merge:305` drops absent artifacts     | `continue` on `aid not in base_contents`                                                              |
| `plan_merge` has zero production callers    | 15 references, all in `tests/test_merge.py`                                                           |
| `editor_calls` counted but uncapped         | `config.py:49` is the only occurrence of the name in the module                                       |

Resolutions:

- **Action 1 (budget cap)** — resolved in §12.
- **Action 2 (namespace + caps)** — resolved in §5, with a correction: qf30
  proposed a flat `generated/` prefix, which `_harness_slot`
  (`cuga_adapter.py:122-138`) would reject with `ValueError`. The prefix must be
  CUGA-group-first, hence `skills/generated-<name>`.
- **Action 3 (verify `lineage_of`)** — already satisfied by the cited lines; no
  spec change needed. A regression test still covers it (§11).

One inaccuracy in the review, recorded for future readers: qf30's §V table
confirms `EditMemory.retrieve` is bounded "via `max_records` parameter". The
conclusion is right but the mechanism is conflated — `retrieve()`'s parameter
`max_records` (`memory.py:373`) is unrelated to the class field `max_records`
(`memory.py:251`), which is a deprecated alias that `__post_init__` forces to
`None` (`memory.py:271`). The live bound is `max_history_records`.

## 16. Open questions

None blocking. Deferred items are tracked in §13 (limitations) and §14
(deviations); Phase 5 merge activation requires fixing the `merge.py` defects
listed there first.
