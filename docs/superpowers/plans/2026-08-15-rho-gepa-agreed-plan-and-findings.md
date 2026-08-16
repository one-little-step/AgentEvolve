# RHO-Parallel-GEPA: Agreed Plan And Session Findings

Written 2026-08-15 before a context compaction. Supersedes the plan sections of
`2026-08-15-cuga-editor-verification-handoff.md`; that document's *findings* and
*bug list* remain valid, but its "Immediate Next Steps" are replaced by §3 here.

Authoritative sources, in precedence order: `AGENTS.md`, `docs/architecture/`,
`feedback/from_qwen/qf33_rho_gepa.md` (the full research deck), then this file.

---

## 1. What We Are Building (the loop)

The user's stated target, mapped to code. **We are replacing exactly one box**
(the editor/mutator). Everything else already exists in some form.

```
rollouts (base G/task, post-RHO candidates 1/task)
   │
   ▼
(1) TRAJECTORY ANALYZER AGENT
    proposes root cause of failure + faulty-node blame
    "can be a single LLM call now, a full CUGA agent in future" (user)
    core/analyzer.py -- currently FakeAnalyzerJudge, a SUBSTRING MATCHER
   │
   ▼
(2) FILTERING: entropy + hierarchical DPP
    selects harness-DEPENDENT issues = high variance ACROSS candidate harnesses
    core/entropy.py   H(t,m) = Var({Q(h_i,t,m)}) * max(max_i Q, score_floor)
    core/issues.py    HierarchicalDPPSelector (task level, then mechanism level)
    evidence floors: >=3 comparable candidates, >=2 rollouts each -- IMPLEMENTED
   │
   ▼
(3) EDITOR-CUGA AGENT  <-- THE BOX WE ARE CHANGING
    mutate + crossover, with:
      - edit history / RAG            core/memory.py (exists; no real records yet)
      - single-LLM-call simulator k=3 TO BUILD (the "GT proxy")
    adapters/cuga_editor.py -- built and live-verified this session
   │
   ▼
(4) VALIDATOR AGENT
    full rollout validation AFTER A SET OF EDITS (not per edit)
    ValidationPlanner + emit_generalization_probes -- EXISTS
    cluster-completion trigger that sets it True -- MISSING
   │
   ▼
(5) accept/reject -> pool -> budget check -> loop until budget exhausted
    core/pool.py PersistentPool / Pareto / champion -- IMPLEMENTED
    core/config.py BudgetLimits + BudgetUsage       -- IMPLEMENTED
```

### The dependency that orders the work

Entropy (2) needs **comparable cells** keyed by `mechanism_cluster_id`. That key
is produced by embedding-clustering the analyzer's mechanism string
(`clustering.py`), and `verdict_id = f"{task_id}:{mechanism_cluster_id}"`
(`orchestrator.py:985`) is *also* the edit-history key.

Today the mechanism is `f"failed-to-match-{task_id}"` (`orchestrator.py:246`) --
a constant per task. Consequences, all currently true:

* one cluster per task, so `Var` inside a cell mixes unrelated failure causes;
* DPP "mechanism diversity" degenerates to diversity over task IDs;
* every failure on a task shares one `issue_id`, so edit history cannot
  distinguish causes.

**Therefore (1) is the minimum unblock for (2) and (3) to function at all.** It
does not need to be a CUGA agent yet; it needs to emit a real mechanism string.

---

## 2. Scope Boundary (user-confirmed)

**In scope:** the editor box plus the analyzer's mechanism output, plus the
trigger that activates the already-built batched validator.

**Out of scope / do not rewrite:** verdict sampler, score tensor, pool + Pareto,
issue selection/DPP internals, deterministic merge/crossover rules, focused
validation internals, champion selection. `qf33` slide 5 is the topology; we
change the `M["GEPA mutator / crossover model"]` node.

**Retracted earlier proposals** (recorded so they are not revived):

* Building a full `CugaAnalyzerAgent` as step 1 -- out of scope; a single LLM
  call emitting a real mechanism is what is needed.
* Wiring an embedder into `EditMemory.retrieve()` -- **redundant**. Semantic
  generalization already happens upstream in `clustering.py`; exact-match on a
  semantically-derived key is a legitimate design. The real gap is that
  `issue_id` embeds `task_id`, so history never crosses tasks.
* Exposing `validation_summary` to the editor *now* -- no code path populates it
  yet, so the tool would return `{}` and a skill telling the editor to consult
  it would be measuring nothing (bug-3 class error). Deferred behind (4)/(6).
* Treating the delta loop as per-edit validation -- the architecture batches
  full rollouts after a *set* of edits (`target-rho-parallel-gepa.md:153-155`).

---

## 3. Agreed Next Steps (in order)

0. **DONE (2026-08-16): parallel analyzer fan-out.** Built ahead of the analyzer
   itself so the CUGA swap is a drop-in. See §10.

1. **Trajectory analyzer emitting real mechanisms.** Single LLM call *for now*;
   **the user intends to replace it with a CUGA agent** (see §10). Produces
   free-form mechanism + faulty-node blame per the `CausalFinding` contract
   (`component-contracts.md:60-75`): status
   `observed|uncertain|insufficient_evidence|malformed`, severity, confidence,
   artifact candidates **only when trace-backed**, evidence refs. Keeps the
   existing `AnalyzerJudge` protocol so nothing downstream changes shape.
   Fix-signature authorship is now **editor-derived** (§6.1), so the analyzer's
   output shape is unchanged from the existing contract.

2. **`replay_single_llm_call`** on the wrapper. Re-issue one recorded LLM call
   from `(messages, model)`. Prove standalone on
   `data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781` before any wiring.
   MUST NOT flip `supports_counterfactual_replay()` (stays `False` per
   `cuga-adapter/sdk-verification-matrix.md:51`); name it separately so we never
   imply agent-state reconstruction we do not have.

3. **k-parallel A/B counterfactual simulator, `k=3` configurable.** Replay the
   baseline prompt k times AND the edited prompt k times, in parallel, score each
   with a machine-checkable predicate, return both rates plus agreement.
   **A/B is mandatory, not optional:** one-arm self-consistency measures the
   model's stability, not the edit's effect. The effect is the difference between
   two distributions.

4. **Skill restructuring.** Move the history practice into `EDITOR_INSTRUCTIONS`
   (always present, no selection required); keep skills one-per-decision; add
   `skills_loaded` to the verification report so selection is measured rather
   than inferred.

5. **Editor validator tool + history/RAG recording.** Expose the simulator to the
   editor; record `baseline_rate`, `edited_rate`, `k`, predicate into
   `EditAttempt`, **labeled as proxy evidence**, so edit-history RAG carries
   measured outcomes rather than editor prose.

6. **Cluster-completion trigger.** Set `emit_generalization_probes=True` when a
   mechanism edit cluster completes, activating the existing batched full-rollout
   validator. This is also what calibrates the step-3 proxy's false-positive
   rate: compare "proxy said fixed" against "full rollout scored better".

---

## 4. User Decision This Session: Cell Comparability By Semantic Similarity

The user resolved the "entropy needs comparable cells" problem:

> "we can consider two of them same based on semantic sim (ie., if sim > 0.95
> --> consider as same)"

So two mechanism clusters whose embeddings are >= **0.95** similar are treated as
the **same cell** for entropy comparability, even when their `cluster_id`s
differ. This lets variance be computed across candidate harnesses that failed the
same way under slightly different mechanism wording.

Implementation notes and existing thresholds (do not conflate them -- three
distinct numbers):

| threshold | value | meaning |
|---|---|---|
| `clustering.py:149` `join_threshold` | 0.75 | join an existing cluster vs spawn new |
| `config.py:118` `cluster_similarity_threshold` | 0.80 | configured clustering threshold |
| **new: comparability threshold** | **0.95** | treat two clusters as one cell |

0.95 is deliberately stricter than the join threshold: joining shapes the
cluster space, whereas comparability decides what may be *statistically
compared*, and a false merge there silently corrupts entropy. Relevant code:
`EntropyTracker.mark_comparable` / `_comparable_candidates` /
`_meets_evidence_floor` (`entropy.py:158-201`).

---

## 5. Hard Findings From This Session (do not re-litigate)

### 5.1 `learn-from-history` never fires -- willingness, not access

Measured across all 12 live logs, **37 total skill invocations**:

| skill | invocations |
|---|---|
| `refine-artifact` | 22 |
| `combine-parents` | 9 |
| `create-artifact` | 6 |
| **`learn-from-history`** | **0** |

Access is provably fine. The rendered catalog (via CUGA's own
`SkillRegistry` + `format_available_skills_block`) lists all four with correct
descriptions. Cause: CUGA's catalog instruction is *"When a task matches a
skill"* -- singular, match-based routing. Three skills answer "what should I
do?" (task-type triggers); `learn-from-history` answers "what should I do
first?" (a temporal/phase trigger). The router matches task type, so a
phase-scoped skill is structurally unselectable. Model-agnostic; not a
gpt-5.6-luna quirk.

**Corollary -- a metric was lying.** `consulted_history` measured the *tool*, and
the tool fires from `combine-parents` step 1, whose procedure mentions history.
So a skill the model never read looked satisfied. This also fully explains the
apparent run-to-run "variance" in history consultation: it is an accident of
which strategy skill was selected, not a flaky rule.

Fix: cross-cutting practices belong in `special_instructions` (always present);
skills stay one-per-decision.

### 5.2 State reconstruction: prompts YES, agent state NO

From `data/traces/5d434903-.../` (56 events, 44 payload blobs, all resolvable):

* **7/7 LLM calls have fully reconstructable inputs** -- complete message arrays
  including `SystemMessage`, 1.7KB-43KB each, plus `response_ref` holding the
  actual output, plus `model: openai/azure/gpt-5.6-luna` in the trace.
  => **single-call replay needs only `(messages, model)`. No checkpointer.**
* `checkpoints/000000.json` has **`state_keys` only, no `channel_values`**;
  `graph_history: unavailable_no_checkpointer`; `AgentState` blobs have 64 keys
  but only 8-13 non-empty. => **no graph replay, no true counterfactual.**
* `CugaLiteState` carries `prepared_prompt` (36.5KB) and a discrete
  **`skills_prompt_section`** field -- the natural substitution point for edited
  artifact text (no regex tagging needed; CUGA already isolates it).
  CAVEAT: in this trace `reflection_skills_enabled=False` and
  `skills_prompt_section_len=0` (a skills-disabled rollout). The field is the
  right hook but has **not** been observed populated. Confirm on a
  skills-enabled trace before depending on it.

I earlier said "no counterfactual replay" when the accurate statement was
"no *graph* replay". Single-call replay is feasible and is what the plan uses.

Cost: worst-case prompt ~10.8K input tokens; `k=3` A/B ~65K input tokens per
validation. Calls are independent and parallelize.

### 5.3 Bug 11 (found by the creation scenario): sandbox-authored indentation

The editor authors artifact bodies inside Python string literals; the first live
creation produced a skill whose every line after the first began with four
spaces. Markdown reads uniformly indented lines as a code block, so the skill
would materialize as a literal listing instead of instructions.

Fixed via `normalize_authored_content()` at both staging entry points in
`cuga_editor_state.py`. **`inspect.cleandoc`, not `textwrap.dedent`**: dedent
takes the common prefix over *all* lines and the literal's first line is flush,
so the prefix is `""` and dedent is a **no-op on exactly the observed shape**.
cleandoc ignores the first line. Relative indentation is preserved (nested lists,
fenced blocks) -- regression-tested.

### 5.4 Creation path verified live (twice)

`stage_create` works: `skills/generated-pagination`, correct namespace, caps
respected, unrelated auth skill untouched. Downstream survival proven end-to-end:
adapter accepted -> in `artifact_inventory` -> counted by
`created_artifact_count` -> lands in the `skills` group of `_harness_config` ->
`materialize_harness` writes a loadable `SKILL.md` with a populated
(non-`None`) description.

Tool reachability now **12/16**. Never reached: `get_attempt_outcome`,
`get_task_input`, `list_trace_actors`, `unstage`.

---

## 6. Decisions Resolved By The User (2026-08-16)

1. **Fix signature authorship: EDITOR-derived.** The editor writes the
   machine-checkable predicate, not the analyzer. Rationale: the editor knows what
   its own edit was *trying* to change, so the predicate tests the intervention
   rather than the diagnosis. Consequence: the analyzer's contract stays as it is
   today (mechanism + blame + status/confidence), and the predicate becomes an
   output of the edit proposal.

   The known risk, to be watched in live runs: an editor that authors both the
   edit and its passing criterion can write a predicate that trivially matches
   its own output. Mitigation is to keep the predicate anchored to the *mechanism*
   (which the editor did not author) and to calibrate proxy verdicts against
   batched full rollouts (step 7). If observed proxy precision is poor, revisit.

2. **Temperature: NOT AN ISSUE. My earlier concern was misplaced.** I had
   conflated two distinct call paths:

   | path | who sets sampling | temperature relevance |
   |---|---|---|
   | rollout | CUGA config (`agent.temperature`) | affects rollouts; record for benchmark hygiene |
   | **proxy counterfactual** | **our own single LLM call** | we own the params directly |

   The proxy counterfactual simulator replays the exact recorded state
   (`messages`) with the edited artifact substituted, as a **single LLM call that
   does not go through the CUGA wrapper at all**. So no wrapper temperature
   setting and no "replay temperature policy" is needed; the `k=3` sampling
   parameters are ours to pass at call time.

   Still worth recording from `feedback/gpt_context/cuga_temperature.md` §4, for
   the *rollout* side only: CUGA has multiple model layers (outer agent model plus
   internal planner/coder/final-answer nodes), so "CUGA temperature" is not one
   knob. For a vanilla-vs-evolved comparison, log `model`, `temperature`,
   `max_tokens`, execution mode, and planner/coder configuration -- otherwise an
   apparent evolutionary gain could actually be a model-parameter change.

3. **Cross-task history: YES.** `search_edit_history` must cross task boundaries
   so the editor can learn from the same mechanism seen on a different task.
   Blocker to resolve in implementation: `issue_id` is `f"{task_id}:{cluster_id}"`
   and `EditMemory.retrieve()` is an exact-fingerprint dict lookup
   (`memory.py:372`), so cross-task retrieval is structurally impossible today.
   Retrieval must key on the mechanism component, with `task_id` retained on the
   record so the editor can see that a precedent came from another task.

4. Residual repo hygiene (still open): tracked stale `.cuga/` artifacts; whether
   `.vscode/` should be gitignored.

---

## 7. Working Agreements (hold across compaction)

* TDD: a failing test before implementation, every step.
* **Never `git commit` without explicit user approval.**
* `uv run pytest` / `uv run python`; system Python lacks dependencies.
* Capture runs with `2>&1 | tee terminal_output/<topic>/<name>.log`.
* `src/agent_evolve/core/` must never import `cuga` or any adapter.
* Proxy verdicts are recorded as **proxy**, never as confirmed outcomes.
* Per `qf33` slide 18: *"No causal claim rests on one smoke run or one model
  response."* Report n; never upgrade "worked once" into "works".
* Do not claim a capability from a single live run; report negative results.
* Models: `openai/azure/gpt-5.6-luna` for development;
  `openai/azure/gpt-5.6-terra` reserved for expensive ablations.
* `timeout` is unavailable in this macOS shell. `rg` is gitignore-aware and skips
  `.venv` -- use `grep -r` or `rg --no-ignore` for SDK source inspection.

## 8. Current Repository State

* Branch `dev4`, HEAD `f3c21aa "wiring cuga editor 1"`.
* Suite: **803 passed, 1 skipped, 0 failures** (`uv run pytest -p no:randomly`).
* Uncommitted (staged + unstaged), awaiting user approval:
  `scripts/verify_editor_rigorous.py` (creation scenario + per-scenario checks),
  `adapters/cuga_editor_state.py` (`normalize_authored_content`),
  `tests/test_cuga_editor_state.py` (+5 dedent tests),
  `adapters/cuga_editor.py`, `adapters/cuga_editor_skills.py`,
  `cuga_wrapper/__init__.py`, `tests/test_cuga_editor_skills.py`,
  `tests/test_harness_materialization.py`,
  `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md`
  (+3 transferable entries), and this file plus the earlier handoff.

## 9. Key Files

| path | why it matters |
|---|---|
| `core/analyzer.py` | `FakeAnalyzerJudge`; step 1 replaces its mechanism output |
| `core/orchestrator.py:246` | the `f"failed-to-match-{task_id}"` template mechanism |
| `core/orchestrator.py:985` | `verdict_id = f"{task_id}:{mechanism_cluster_id}"` |
| `core/clustering.py` | embedding clustering -> `mechanism_cluster_id` |
| `core/entropy.py:158-201` | comparability + evidence floors (0.95 change lands here) |
| `core/issues.py` | `HierarchicalDPPSelector` |
| `core/editor.py:462-491` | `ValidationPlanner`, `emit_generalization_probes` |
| `core/memory.py:372` | `EditMemory.retrieve()` -- exact key, semantics live upstream |
| `adapters/cuga_editor*.py` | the editor agent, tools, skills, staging, evidence |
| `cuga_wrapper/__init__.py` | env/model config, `materialize_harness`; step 2 lands here |
| `scripts/verify_editor_rigorous.py` | live scenarios: history, crossover, creation |
| `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md` | cross-repo CUGA findings |

---

## 10. Parallel Analysis (user direction 2026-08-16, IMPLEMENTED)

### 10.1 User direction

* The analyzer+judge will become a **CUGA agent**, not a single LLM call.
* Analysis of different trajectories is **independent**, so run it in parallel.
* Configurable `max_analyzer_workers`, user-specified default intent **10**.
* Interim: a single LLM call is fine, but **wire the parallel path now** so the
  CUGA swap changes only the factory.

### 10.2 What was built

`src/agent_evolve/core/parallel_analysis.py` + `tests/test_parallel_analysis.py`
(15 tests). `ResolvedConfig.max_analyzer_workers` (default **1**, override
verified to 10; rejects `0`, `-1`, `True`, `2.5`; appears in `manifest_payload`).

Default is 1, not 10, deliberately: parallelism is opt-in per profile so a
sequential debug run stays reproducible and stack traces stay readable.

### 10.3 Why a separate module from `parallel.py` (user was right)

The user noted `parallel.py` predates the decision to use CUGA as the editor and
was aimed at parallel *edits*. Confirmed by reading it: `SnapshotLeaseManager`
and `BatchCoordinator` exist to make concurrent artifact **writes** safe
(snapshots, exclusive write leases, commit barrier). **Analysis performs no
writes**, so reusing that machinery would impose write-serialization on
read-only work. Hence a separate, smaller module.

### 10.4 Design decisions encoded as tests

| decision | why | test |
|---|---|---|
| `analyzer_factory`, not an analyzer instance | a CUGA agent carries conversation state; sharing one across threads would interleave two trajectories into one conversation | `test_one_analyzer_instance_per_worker_thread_not_shared` |
| one analyzer per **thread**, reused across items | agent construction is expensive; per-item construction would waste it | `test_workers_never_exceed_item_count` (10 workers, 2 items -> <=2 built) |
| results in **input order** | clustering/entropy must not vary with thread scheduling | `test_results_are_ordered_by_input_position...` plus a 40-item randomized-delay stress test |
| failure is **data**, not an exception | one bad trajectory must not discard the batch's findings | `test_failing_analyzer_isolates_to_its_own_work_item` |
| factory failure also isolated | a missing model env must not abort the batch | `test_analyzer_construction_failure_is_reported_not_raised` |
| `max_workers=1` runs **inline** | sequential debugging needs a plain stack, not a pool | `test_max_workers_one_runs_inline_without_spawning_threads` |
| **no budget accounting in workers** | concurrent ledger mutation is a race; the caller charges budget on the coordinator thread when consuming outcomes | (constraint documented in module docstring) |

Ordering is structural (`executor.map` preserves input order), not timing luck;
the 40-item randomized-delay test exists specifically to catch a completion-order
regression that a 3-item test could pass by chance.

### 10.5 Editor parallelism (user's second point) -- ANALYZED, NOT YET BUILT

User observation: the editor takes `1 + (k-1)` harnesses from the Pareto pool for
mutation/crossover, so that can parallelize too when the candidates are disjoint.

Correct, with one refinement: crossover **reads** donor artifacts but **writes**
only the primary's. So the safety condition is **disjoint write sets**, not
disjoint candidate sets. Two attempts may safely share a donor for reading.

**Defect found in the existing machinery while checking this.** In
`parallel.py:109` the lease key is a bare `artifact_id`, and the
`commit_barrier` clash check (`parallel.py:~245`) also keys on
`e.artifact_id` alone. But artifact IDs are **candidate-relative paths**
(`cuga_adapter.py:77-89` builds them from harness slots, e.g. `skills/default`),
so two different candidates editing their own `skills/default` collide on a
single lease. Consequences: false `LeaseConflict`, and a `commit_barrier` that
rejects a legitimate batch. Fix (not yet made): key leases by
`(candidate_id, artifact_id)`, or namespace the ID by candidate before leasing.
There is no test covering two candidates with same-named artifacts.

Not fixed yet because editor parallelism is not on the critical path; the
analyzer chain is. Recorded so it is not rediscovered.

### 10.6 Temperature, for the record

The user is right that the proxy counterfactual does **not** call the CUGA
wrapper: it replays the exact recorded state with the edited artifact substituted
as a single LLM call we issue ourselves. So we own its sampling parameters, and
no wrapper/CUGA temperature policy is needed for the proxy. CUGA's
`agent.temperature` remains relevant only to **rollouts** -- and per
`feedback/gpt_context/cuga_temperature.md` §4 it is not one knob (outer model
plus internal planner/coder/final-answer nodes), so vanilla-vs-evolved
comparisons must log model, temperature, max_tokens, mode, and planner/coder
config to avoid crediting a model-parameter change to evolution.

---

## 11. Session 2026-08-16 (speed phase): components built + live sampling facts

Suite went **803 -> 1034 passed**, 1 skipped, 0 failures. Nothing committed.

### 11.1 MEASURED LIVE — sampling on `openai/azure/gpt-5.6-luna`

These are measurements, not assumptions. They decided the proxy design.

| probe | result |
|---|---|
| `temperature=0.0` | **REJECTED** - BadRequestError "does not support 0.0" |
| `temperature=1.0` explicit, x3 | 1 distinct of 3 |
| identical prompt sequential x3 | **1 distinct of 3** (cached) |
| cache-busted prompts x3 | 2 distinct of 3 |
| **`n=3` in ONE request** | **3 distinct of 3** |
| `n=3` repeated later | identical triple returned (cached) |

**Consequences, binding on all replay/proxy work:**
1. **Never pass `temperature`** to this endpoint. Omit it.
2. **k sequential calls do NOT produce k samples** - identical requests are served from cache.
3. **`n=k` in a single request is the ONLY working variance source.** USER DECISION: option A.
4. Repeating an identical A/B re-reads the first trial. Two verdicts over the same
   `(call, substitution, k)` triple are **one observation**; a CI built by repeating
   an identical A/B is invalid.
5. The analyzer is **non-deterministic by default** (cannot pin temperature=0).

Live-confirmed on a **36.5k-char real prompt**: `n=3` gave **3/3 distinct per arm**.
So variance is real on production-scale prompts, not just toy probes.

### 11.2 Built this session

| component | file | evidence |
|---|---|---|
| parallel analyzer fan-out | `core/parallel_analysis.py` | 15 tests; one analyzer per worker thread (CUGA agents are stateful) |
| `max_analyzer_workers` | `core/config.py` | override->10 verified; rejects 0/-1/True/2.5 |
| evidence bridge + shared guard | `core/evidence.py` | 24 tests; guard de-duplicated with the editor's copy |
| **LLM trajectory analyzer** | `adapters/cuga_analyzer.py` | 55 tests; **live: 3 distinct causal mechanisms, max pairwise Jaccard 0.16** |
| single-call replay | `cuga_wrapper/__init__.py` | **7/7 events** resolved; offline proof passes with sockets blocked |
| **k=3 A/B proxy validator** | `adapters/cuga_proxy_validator.py` | 34 tests; live 0.333 vs 1.000, delta +0.667 |
| finding<->analysis bridge + dual-protocol shim | `core/blame.py`, `core/analyzer.py` | 74 tests; **zero existing tests modified** |

The degenerate-mechanism blocker is **broken**: `failed-to-match-{task_id}` is gone.
Live mechanisms are specific and causal, e.g. *"the planner emitted a plan without
scheduling convert_currency after discovering it, so the answer_agent submitted a
rough USD estimate without a tool-derived conversion."*

### 11.3 Trace format corrections (my brief was wrong; agent caught it)

* Blobs are under **`payloads/<sha256>.json`**, NOT `blobs/`.
* `messages_ref` blobs are **doubly nested** `[[msg, ...]]` (LangChain message batches).
* **No event carries a model**; model is at trace top level only. A run with
  per-node models could not be distinguished in this format.
* start/end pairing is by `payload["run_id"]`, not event_id adjacency.

### 11.4 DANGEROUS FAILURE MODE (found live, no automated guard)

A proxy predicate can be **structurally unmeasurable** at a given boundary and then
returns `no_change` for **any** edit. Measured: at `graph:13` the endpoint returned
only the short pre-code note in `message.content` - `tool_calls=None`, and **0/6
completions contained a fenced code block**. So `calls_tool(...)` scored 0/3 vs 0/3
-> `no_change`. Reading that as "the edit did not help" would be **flatly wrong**.

Choosing a measurable predicate for a boundary is currently the caller's
**unverified** responsibility. Mitigation candidate (not built): assert the
predicate can fire on at least one baseline completion before trusting a verdict.

Also: a `+0.667` delta measured **stated intention** in prose, not achieved
behaviour. One arm-B completion contained "I'm sorry, but I couldn't..." and still
passed a regex predicate. **A passing completion is not a good completion.**

### 11.5 Placeholder discipline (replaces the degenerate template)

`__placeholder__:unanalyzed` (minimal profile) and `__placeholder__:abstained:<status>`
(analyzer declined), with `is_placeholder_mechanism()`. Abstention is deliberately
lossy: blame graph dropped, mechanism hint dropped, severity forced 0.0 - so a hunch
can never enter clustering as a real mechanism. `score` always passes through:
failing to *diagnose* a rollout does not un-*measure* it. `"none"` is NOT a
placeholder - it is the real verdict for a successful rollout.

Second degenerate template found and fixed at `orchestrator.py:~504`
(`base-failed-{t}-{m}`): it fabricated `BlameNode(actor_id="agent")` while the real
analysis was computed 30 lines earlier and **discarded**. Now retained per cell.

### 11.6 NOT wired end-to-end (next decisions)

1. **Nothing constructs a real analyzer in production** - every default is still
   `FakeAnalyzerJudge`. The shim makes the real one *acceptable*, not *active*.
2. `max_analyzer_workers` is **not threaded through**; `Orchestrator` has no
   `config` field. `analyze_groups(...)` exists but no caller uses it.
   Best first conversion: `run_iteration` step 1 already loops
   `base_rollout_group_size` (3) traces per task.
3. Nothing filters on `is_placeholder_mechanism` - placeholders are *detectable*
   but still ingested by clustering/entropy/DPP as ordinary strings.
4. `SequentialGepaRunner.finding_from_analysis` is now the reverse of the new
   converter; a report analyzer's finding makes a lossy round-trip
   (finding->analysis->finding), losing `confidence` and `evidence_refs`.
5. `replay_single_llm_call` **tops up sequentially** if the provider returns <n
   choices - on a caching endpoint that substitutes duplicates for samples.
   Surfaced via `ProxyArmResult.request_count`; suggest it should raise instead.
6. `empty_analysis()` hardcodes `score=1.0` (same diagnosis/measurement conflation).
7. Proxy verdicts are unvalidated against full rollouts - no delta threshold has
   been established, and stability across the other 6 trace events is untested.

---

## 12. Benchmark layer + THE NOISE FLOOR (2026-08-16)

Suite: **1104 passed, 1 skipped, 0 failures.** Nothing committed.

### 12.1 RHO SEEDER: NOT IMPLEMENTED (user asked; confirmed by search)

Zero matches for `seed_pool`/`RhoSeeder`/`rho_seed`/`initial_propos`/`seeder` in
`src/`, `tests/`, `scripts/`. The pool has ONE base entry point (`pool.py:264
add_base`, which raises `"base already exists"` on a second call).

**We are therefore NOT running RHO today.** The proposal-generation step that
creates the N initial variants is absent. Nor is it "N dummy candidates" -- no code
creates N of anything.

**BUT the pipeline does support N candidates** -- proved empirically:
`add_base` + 5x `add_candidate` -> pool size 6, `base count: 1`, all retained.
`add_candidate` is unlimited, dup-ID guarded, and carries
`parent_ids`/`ancestor_ids` provenance. A future seeder just calls it N times; no
structural change needed and no selection code cares whether N is 1 or 20.

Consequence while N=1: entropy needs >=3 comparable candidates per cell
(`min_comparable_candidates=3`) and DPP needs alternatives, so **both are inert in
practice at N=1**. A live run today exercises the plumbing, NOT the selection.

### 12.2 THE NOISE FLOOR -- the most important number in the project

Same 42 Gaia L1 tasks, same model (`gpt-5.6-luna`), baseline vs baseline:

| grader | run 193706 | run 035541 | spread |
|---|---|---|---|
| `expected_regex` | 10/42 = 23.81% | 17/42 = 40.48% | **16.67 pp** |
| `recorded_llm_verdict` | 6/22 (PARTIAL) | 18/42 = 42.86% | **NOT COMPARABLE** |

**An evolution result must beat ~17 pp on this harness before it means anything.**
Two baseline runs of identical tasks differ by 7 tasks out of 42.

My earlier "27.3% -> 42.9%" comparison was **invalid**: 6/22 vs 18/42 mixes a
partial denominator (one eval batch died of a URLError, losing 20/42 tasks) with a
full one. The only valid comparison is regex's 10/42 -> 17/42. The tooling now
refuses to compute that delta and says `denominator mismatch`.

Run `20260813_035233` is half-broken: 27 task dirs but 17 `result.json`, 10 errored.
A naive loader reports 3/17 = 17.65% and compares it against 42.

### 12.3 Grader agreement: 96.88% (n=64 paired)

62/64 agree; 2 disagreements, one in each direction, so they cancel. Grader choice
does **not** silently set the headline number on this data -- the 10-vs-6 and
17-vs-18 gaps are **coverage**, not disagreement.

Caveats: n=64 with 2 disagreements gives a 95% CI of roughly 89-99%, so a 3-10%
disagreement rate is consistent with this data -- and at a 17 pp noise floor, 5%
disagreement is a third of the signal. Regex patterns are strict anchored matches
(`(?i)\bRockhopper\ penguin\b`), so expect disagreement to RISE as agents get more
verbose. All 42 regex patterns recompute identically to the recorded
`direct_regex.passed` -- zero drift.

**DECISION: `expected_regex` is the primary selection signal** (deterministic, live,
42/42 available, replayable on new answers). `recorded_llm_verdict` is a secondary
check on regex strictness, and can NEVER grade a live rollout -- by design it
refuses any answer other than the one it judged.

### 12.4 Built: benchmark layer, decoupled + parallel (user requirement)

`benchmarks/base.py` (contract), `gaia.py` (adapter), `runner.py` (parallel).
29 + 41 = 70 tests.

* **Leakage prevented structurally, not by convention:** a `GRADING_KEY_DENYLIST`
  raises `LeakageError` at construction; `BenchmarkTask` has no grading attribute to
  read; `BenchmarkGrading.__repr__` is redacted so a traceback cannot dump the
  answer key. Note `status` (`passed_direct`/`failed_llm`) was excluded from task
  metadata -- it looks operational but IS the verdict.
* **`GradingUnavailableError` instead of a failing score.** "We could not measure"
  must never collapse into "the agent was wrong" -- that is exactly how a 22
  becomes a fake 42.
* **Parallel runner, `max_workers=10` default** (matching Gaia's own baseline
  config), per-task timeout, input-ordered results, per-task failure isolation,
  one executor per worker thread (stateful agents), scoring on the coordinator.
* **Verified:** replay of run 035541 reproduces **17/42 = 40.48%** exactly, and
  `--max-workers 1` vs `10` are **byte-identical**. Synthetic speedup 42x0.4s:
  1w=16.96s -> 10w=2.02s = **8.39x**.

### 12.5 Timeout honesty (empirically probed)

Guaranteed: no single task stalls the run; the run terminates even when every worker
is lost to a hung task, and starved tasks are recorded as failures rather than
dropped.

**NOT guaranteed: a timed-out task does not stop.** Python threads cannot be killed.
Confirmed directly -- after the run returned, the abandoned thread was still alive,
later completed its side effect, and produced an answer that was discarded. So it
may still hold a session/subprocess and still be billing tokens, its worker slot
stays occupied (effective concurrency drops), and the interpreter will not exit while
it lives. Hard termination needs the executor's own cancellation.

A bug the tests caught: the first implementation judged deadlines against the
coordinator's wall clock, so 4 tasks that finished in ~1ms were all reported timed
out -- the verdict was a function of thread scheduling. Now judged on each task's own
elapsed time.

### 12.6 Benchmark abstraction is ~70% general (honest assessment)

General: multi-grader-by-name, `GradingUnavailableError`, explicit denominators,
noise-floor math, leakage denylist.

**Over-fit to Gaia, will break on the first non-QA benchmark:**
1. `score(task_id, answer: str)` assumes a text answer. **AppWorld grades
   environment state**; **tau-2 grades a multi-turn dialogue + final DB state.**
   Needs `score(task_id, rollout: RolloutResult, *, grader)` with an
   adapter-declared opaque rollout. Deliberately NOT pre-generalized -- inventing
   APIs without a real schema in hand is what AGENTS.md forbids.
2. `score` is sync and pure; AppWorld's grader boots a sandbox. Needs batch/async.
3. No partial credit: `TaskOutcome.score` is a float but both Gaia graders are
   binary and `GraderStats` only counts `passed`.

### 12.7 Still not wired

* Nothing in `core/` consumes `Benchmark` -- `core/evaluation.py` must accept a
  benchmark + grader name, and the grader name must reach recorded results or the
  headline number is ambiguous at point of use.
* `run_benchmark` is unused by the orchestrator; rollout execution should route
  through it, and `failed_count`/`timeout_count` MUST reach evidence -- a candidate
  whose harness crashed must not be scored as answering wrongly.
* `max_workers`/`task_timeout_seconds` still hardcoded (10, 1200) in runner+CLI
  rather than read from `ResolvedConfig`.
* No live agent execution through the runner yet: "one stateful CUGA agent per
  thread" is verified only for object construction/reuse, not against real CUGA.

---

## 13. Real CUGA execution through the benchmark runner (2026-08-16)

Suite **1156 passed, 1 skipped, 0 failures.** Nothing committed.

### 13.1 Confirmed: the wrapper chain the user described already exists

`run_full_rollout` -> `_harness_config(workspace.version, task)` ->
`wrapper.run_task(task_id, harness_config)` -> `materialize_harness` writes
`skills`/`policies`/`memory` and binds `CUGA_FOLDER`. So **the evolved harness
version is what CUGA actually loads, settable per rollout.**
`_rich_events` then reads the persisted `causal-trace.json` preserving the DAG --
the same format `load_recorded_call` proved 7/7 on, so rollout traces feed the
analyzer AND the proxy replay with no conversion.

Built `benchmarks/cuga_executor.py` binding this to the runner's
`executor_factory` seam. 52 tests. **Live tiny run executed: 2/2 answered,
2 traces written, both loadable by `load_recorded_call` (6 and 2 recorded LLM
calls), `harness_version=vanilla` stamped on trace + manifest.** 1/2 passed.

Tracing is enforced at three points (capability probe before any token is spent;
per-task missing/absent/`causal-trace.json`-less trace refused; count printed),
because a run with answers but no traces is useless to us. `RAW_OPT_IN` +
`capture_node_payloads` is required -- without the `payloads/` blobs a trace
records THAT a call happened but not WHAT was sent.

### 13.2 THROUGHPUT BLOCKER: real `--execute` is currently SERIAL

Two bugs found by running, not by reading:

1. **Import deadlock at max_workers>1.** `CugaAgent` eagerly imports 6 `cuga.*`
   modules but a rollout pulls **172 more lazily inside `invoke()`**; two threads
   race CPython import locks -> `_DeadlockError`. Fixed with a one-time
   single-threaded `warm_up_cuga_imports()` (~10s, leaves 0 of 172 unimported).

2. **Knowledge-engine singleton lock.** `engine.py:1758` opens
   `config.persist_dir/".lock"` and takes `flock(LOCK_EX|LOCK_NB)`, raising
   `"Knowledge engine already running in another process. Start with --workers 1"`.
   Reproducible: `--max-workers 2` -> 1 of 2 answered, every attempt.

`--execute` now defaults to `--max-workers 1` and refuses >1 up front.
**Refusing beats silently halving the denominator** (a lost task is not a wrong
answer).

### 13.3 I CORRECTED the subagent's conclusion (probed the lock directly)

The subagent concluded process isolation was the ONLY fix. Its *diagnosis* was
right, its *conclusion* too pessimistic. Measured with `interprocess_lock`:

| case | result |
|---|---|
| same `.lock` path, two file objects, ONE process | **CONFLICT** (subagent correct) |
| **different `persist_dir` paths, one process** | **NO CONFLICT** |

And `knowledge/config.py:455` shows `persist_dir` defaults to
`Path.cwd()/".cuga"/"knowledge"` but is overridable via a `persist_dir` key
(`config.py:1013`). **So a per-worker `persist_dir` should permit in-process
threaded concurrency** -- far cheaper than one CUGA process per worker.

NOT yet proven end-to-end: that a real threaded rollout works with per-worker
`persist_dir` (only the lock primitive was probed). Remaining shared-global risk:
`CUGA_FOLDER` is process-global, so per-thread harness binding still needs care.
**Next step for parallel execution: set per-worker `persist_dir`, then re-test
`--max-workers 2` live.** If `CUGA_FOLDER` proves irreducibly process-global,
process isolation becomes necessary after all.

Consequence if unfixed: a 42-task run is serial at ~40-200s/task = potentially
hours per candidate, and an evolution loop multiplies that by candidates x
iterations. **This is the single biggest throughput risk to getting research
numbers.**

### 13.4 Smaller finding (out of scope, not fixed)

`CugaWrapper.run_task` copies a fixed key set and **drops the runtime's `error`
field**, so a failed rollout reports `status=error` with no diagnosis. Worked
around by recovering it from the trace; the real fix belongs in the wrapper.

Also open: whether `enable_knowledge=True` is needed for every rollout -- it is
what pins the singleton lock in the first place.

---

## 14. PARALLEL EXECUTION SOLVED (process isolation) + two silent-corruption traps

Suite **1179 passed, 1 skipped, 0 failures.** Nothing committed.

### 14.1 THE MOST IMPORTANT FINDING: `CUGA_FOLDER` is NOT thread-safe

Threaded parallel rollouts would have **silently measured the wrong harness.**
Measured on the real `cuga_wrapper._construct_agent`, two threads, two workspaces,
reading the var where a rollout actually reads it (during `invoke()`):

```
h0:expected   = .../h0
h0:at_rollout = .../h1     <-- WRONG WORKSPACE
VERDICT = CUGA_FOLDER_NOT_THREAD_SAFE
```

A build lock cannot help: the read happens AFTER construction. **And the trace's
`harness_version` CANNOT detect the swap** -- it is copied from the harness config,
so a contaminated run looks clean while measuring a harness that never existed.

This is the worst class of bug for this project: it would have produced
self-improvement deltas attributable to nothing. Threaded `--execute` is now
REFUSED (verified: exits with a measurement-grounded explanation).

Verification used unfakeable evidence instead of the stamp -- captured prompt
payloads containing each candidate's uniquely-named skill:
```
cand-alpha  own(alpha_marker)=42  foreign(beta_marker)=0   CLEAN
cand-beta   own(beta_marker)=14   foreign(alpha_marker)=0  CLEAN
"You are ALPHA": 6/0 in alpha's trace, 0/6 in beta's
VERDICT = NO_CROSS_CONTAMINATION   (process isolation)
```

### 14.2 My per-worker `persist_dir` hypothesis: CORRECT but INSUFFICIENT

It does fix the knowledge flock (mechanism confirmed supported:
`knowledge_settings.toml` `persist_dir = ""`, overridable via
`DYNACONF_KNOWLEDGE__PERSIST_DIR`, honoured on cold start). But `CUGA_FOLDER` is a
*second* global that no persist_dir change can fix. So the fix is **process
isolation**: `benchmarks/cuga_process_pool.py`, one CUGA subprocess per worker,
NDJSON protocol. cwd deliberately NOT moved (CUGA resolves `settings.toml`/`.cuga/*`
from cwd; moving it would silently change the configuration under measurement).
Only colliding state is redirected; `CUGA_FOLDER` is stripped so no stale workspace
is inherited.

**Live: `--max-workers 2` now answers 2 of 2** (was 1 of 2). Speedup **1.63-1.92x at
4 workers** on 8 tasks -- sub-linear because task times are skewed (8-120s) and one
long task sets the floor; should improve on 42 tasks.

### 14.3 SECOND silent-corruption trap: empty worker knowledge store

Caught only because per-task verdicts were compared, not just wall-clock:

| run | passed |
|---|---|
| serial, in-process | 3/4 |
| process-isolated, empty store, **1 worker** | **0/3** |
| process-isolated, empty store, 4 workers | 0/3 |
| process-isolated, **seeded** store, 4 workers | **3/4 (matches serial)** |

The 1-worker row rules out concurrency as the cause. Real cause: a serial run
inherits the repo's populated `.cuga/knowledge` (2 indexed docs) so knowledge
searches hit; an empty store returns dry and the model answers differently.
Workers are now seeded with a **copy** (copy, not share -- sharing recreates the
flock collision). `--empty-worker-knowledge` opts out; the run header prints which
store was used.

**Lesson to carry forward: wall-clock-only validation would have shipped a 1.9x
faster runner reporting 0% instead of 75%.** Any performance change must be
validated on per-task verdicts, not just speed.

### 14.4 Unproven / next

* Tested only to **4 workers**; 42x4 is extrapolation. Memory at 8-10 workers
  (each holds a CUGA process + ~1.9MB knowledge copy + model client) untested.
* **Model nondeterminism is real**: one arm returned a blank answer once and
  answered on rerun. Correctly recorded `ok=False` (excluded from denominator, not
  scored wrong) -- the contract held -- but single-sample runs are not reproducible
  at ANY worker count. Reinforces the 16.67 pp noise floor.
* Determinism verified on 8 tasks/one seed: identical per-task verdicts at 1 vs 4.
* Recommended first real run: 42 tasks, `--isolation process`, 4 workers,
  `--verbose`, **diffed against a serial run before trusting any evolution number.**

---

## 15. FULL PIPELINE WIRED + a bug that made evolution mathematically inert

Suite **1220 passed, 1 skipped, 0 failures** (fixed order AND random order).
Nothing committed.

### 15.1 THE SHOWSTOPPER: passing regression probes blocked every edit

`core/editor.py weighted_net_gain` weighted REGRESSION at **`-1.0 * score`**.
Since real producers set `passed = score >= 0.5` with `score` = the task score
(`orchestrator.py:486,1735`), a regression probe that **PASSED** with score 1.0
subtracted **1.0**. Proved by direct probe:

```
origin pass + 0 PASSING regression probes -> gain=+1.00 accepted=True
origin pass + 1 PASSING regression probes -> gain=+0.00 accepted=False
origin pass + 2 PASSING regression probes -> gain=-1.00 accepted=False
origin pass + 3 PASSING regression probes -> gain=-2.00 accepted=False
```

**A perfect edit -- origin fixed, nothing regressed -- was REJECTED as soon as one
regression probe existed.** At >=2 tasks no edit could EVER be accepted, so every
self-improvement delta would have been exactly 0.0 for arithmetic reasons,
regardless of agent quality. We would have concluded "RHO-GEPA does not work".

**Fix:** charge a regression probe only when it FAILED, in proportion to its
shortfall (`1 - score`). A high-scoring probe is a task that still works and is
evidence *for* the edit. Genuine regressions stay gated by `regression_violated`
and protected floors, which already existed -- the weight must not double as that
gate. Post-fix: gain=+1.00 and accepted=True at 0,1,2,3 passing probes.

### 15.2 Three tests encoded the inverted reading; I changed them WITH justification

* `test_weighted_net_gain_default_weights` asserted 0.325, arithmetic that only
  works if passing probes are charged. Now 1.225.
* `test_decide_acceptance_allows_small_regression_when_net_gain_positive` used
  `score=0.1, passed=False` and called it "small". Under the old formula a WORSE
  probe cost LESS (0.1 charged 0.1), so "small" was backwards. Changed to
  `score=0.45` -- an actual small dip below the 0.5 bar.
* `test_decide_acceptance_rejects_when_net_gain_below_threshold` relied on a
  passing probe exactly cancelling the origin. Now uses a threshold above the
  origin's gain.
* `tests/test_pipeline.py` characterization test (written by a subagent to pin the
  defect, and which explicitly said "when fixed, this should fail and be replaced
  by its inverse") -- replaced by its inverse.

Resolved by checking what REAL producers emit rather than by preference. Two
subagents had independently written tests around this and neither questioned the
semantics; the direct probe settled it.

### 15.3 CORRECTION to section 14.3: the knowledge "seeding" story was wrong

I reported that seeding worker knowledge stores restored scores 0/3 -> 3/4. The
user asked what it was seeded WITH. Inspected: `.cuga/knowledge` contains exactly
two leftover smoke-test fixtures --

```
favorite-color.md         -> "blue"
project-clearance-code.md -> "...clearance code is MEM-5331780485..."
```

**Neither can possibly help answer a Gaia question** about BBC Earth videos or
flight prices, so the causal story "serial hits knowledge, empty store comes back
dry" does NOT hold. What the 0/3 vs 3/4 actually was: **n=3-4 tasks on an endpoint
with measured nondeterminism** -- i.e. noise, or an unexamined empty-store error
path. Retracted as a finding.

Worse, seeding from that store is **contamination**: it plants a fake
"clearance code" fact into the knowledge base during measured runs. The real
requirement is only that **every arm uses an IDENTICAL store**, so
identical-and-empty beats identical-and-junk. Default is now a clean empty store,
printed in the run header.

### 15.4 Pipeline is wired end to end

`pipeline.py` composition root (`build_offline_stack` / `build_live_stack` ->
one `EvolutionStack`), `scripts/run_evolution.py` CLI. Dry run proves the whole
lifecycle: base 0/3 -> analyze -> DPP select -> edit -> validate -> **accept** ->
champion 3/3, pool 1->2, delta printed WITH the 16.67 pp noise-floor caveat and an
explicit note that cross-candidate entropy/DPP contributed nothing at N=1.

A failed rollout cannot enter a denominator -- four independent layers:
`RolloutOutcome` forbids a traceless outcome without an error; a **whitelist** of
answered statuses (unknown status = "no answer", never "wrong answer");
`RolloutScore.scorable=False` structurally cannot carry `passed=True`;
`_record_rollout_score` RAISES on an unscorable rollout rather than skipping
quietly. `tally_scores` is the single definition of the denominator.

Live invocation (not yet run):
```
uv run python scripts/run_evolution.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --harness vanilla \
  --tasks 42 --iterations 3 --max-workers 6 --isolation process --analyzer-workers 6
```

### 15.5 Still not wired / next

* `cuga_proxy_validator` is NOT wired into validation (real-rollout probes only).
* One rollout per task -- no G-group for the base, so no within-candidate variance.
* `Orchestrator` (older class) untouched; `SequentialGepaRunner` is the live path.
* **Biggest remaining risk to a trustworthy number: N=1 plus a 16.67 pp noise
  floor.** A single run cannot produce a credible delta; repeated runs per arm are
  required. Recommend a baseline replication (n>=3, parallel diffed against
  serial) BEFORE any evolution claim.

---

## 16. LIVE BASELINE REPLICATION ON tiny5 -- THE MEASUREMENT IS NOT VIABLE YET

Three IDENTICAL live runs: same 5 tasks, same `vanilla` harness, same model
(`gpt-5.6-luna`), same grader (`expected_regex`), process isolation, 5 workers,
empty knowledge store. Real CUGA rollouts, 15 rollouts total, 15 traces.

| run | pass rate | wall |
|---|---|---|
| A | **3/5 = 60%** | 106s |
| B | **1/5 = 20%** | 49s |
| C | **2/5 = 40%** | 203s |

**Range = 40 percentage points on identical inputs. Mean 40%, and the spread is
as large as the mean.**

### 16.1 Per-task: 2 of 5 tasks flip verdict across identical runs

```
task                 A      B      C     regex
gaia-5d0080cb        True   False  False  (?i)\b0\.1777\b   <-- FLIPS
gaia-8e867cd7        True   True   True   (?i)\b3\b
gaia-a1e91b78        True   False  True   (?i)\b3\b         <-- FLIPS
gaia-e1fc63a2        False  False  False  (?i)\b17\b
gaia-ec09fa32        False  False  False  (?i)\b3\b
```

Only 3 of 5 tasks are stable (1 always-pass, 2 always-fail). **40% of the task set
is pure coin-flip.**

### 16.2 THE ROOT CAUSE IS NOT MODEL SAMPLING -- IT IS TRUNCATED AGENT RUNS

The flipped answers are not wrong answers. They are **mid-progress narration**:

```
[a] True  'The fish bag's calculated volume was 0.1777 m3.'
[b] False 'The knowledge-base result is unrelated, so I'm retrying the public-web
           search with the exact title and distinc...'
[c] False 'The PDF-tool installation produced no usable output, so I'm verifying
           imports and using a small Python script ...'

[a] True  'The highest number of bird species on camera simultaneously is 3.'
[b] False 'The indexed result identifies the video as a "Spy in the Snow" clip but
           not the answer itself. I'm checking th...'
[c] True  'The highest number is 3 bird species on camera simultaneously.'
```

B and C did not answer incorrectly -- **they stopped mid-investigation** and the
harness reported `status=success` with a progress note as `final_output`. The same
pattern appears in the historical dataset (`gaia_0383a3ee`: *"I'll search for the
specific BBC Earth video and verify..."* scored as a wrong answer).

**This is the same class of bug as the ones already caught: a run that did not
finish is being scored as a wrong answer.** Our pipeline correctly protects
against *failed* rollouts entering the denominator, but CUGA reports these as
`success`, so no guard fires. The agent simply ran out of steps/turns.

Consequence: **the "wrong" bucket is mostly incomplete runs, not reasoning
errors.** An evolution loop pointed at this signal would mostly be optimizing
"finish before the step budget", and its measured delta would be dominated by
whether runs happened to complete.

### 16.3 What this means for the research plan

* The earlier **16.67 pp** noise floor (42 tasks, historical) understated it.
  On tiny5 the live spread is **40 pp**, because n=5 makes each task worth 20 pp.
* **tiny5 cannot measure a self-improvement delta at all.** One flipped task = 20
  pp, so any plausible improvement is inside the noise. tiny5 is a smoke test for
  the plumbing, never an evidence source.
* A credible measurement needs, in order:
  1. **Fix or bound the truncation** -- either raise the step/turn budget so runs
     complete, or detect "did not reach a final answer" and record it as
     UNSCORABLE rather than a wrong answer. This is the highest-value fix
     available: it likely converts most of the variance into signal.
  2. **Larger task set** (42+) so one task is ~2 pp, not 20 pp.
  3. **Repeated runs per arm** (n>=3) with the delta reported against the
     measured spread, never as a single number.

### 16.4 What DID verify (the point of this run)

The `weighted_net_gain` fix is confirmed against real LLM rollouts: the pipeline
executed 15 real CUGA rollouts across 3 runs with **zero failures, zero timeouts,
zero scoring errors, 15/15 traces written**, all `harness_version=vanilla`,
process isolation at 5 workers. Before the fix, no edit could have been accepted
at 5 tasks; the machinery now runs end to end on real data.

### 16.5 ROOT CAUSE ISOLATED: PERFECT separation between truncation and failure

Cross-tabulating all 15 live rollouts:

```
            PASS  FAIL
completed      6     0
truncated      0     9
```

**100% of failures are truncated runs. 100% of completed runs pass.**
Zero exceptions in 15 rollouts. 9/15 = 60% of rollouts end mid-investigation.

So on tiny5 the harness has **no reasoning-quality problem at all** -- every run
that actually finished got the right answer. The entire measured "failure rate" is
runs that stopped early and returned narration such as *"I'm retrying the
public-web search..."* as `final_output`.

### 16.6 It is NOT the step budget

* `settings.toml`: `cuga_lite_max_steps = 70`, `max_steps = 55`.
* Truncated runs used **2-4 `call_model` cycles** and 19-34 events -- nowhere near
  either limit.
* Every trace reports `status=success`, `error=None`, and **no** `max_steps` /
  `recursion` / `limit reached` signal anywhere in the trace.

**The agent is voluntarily stopping and handing back a progress note.** Raising
the step budget would therefore change nothing. This is a harness/prompt-behaviour
problem: the agent treats an intermediate narration turn as a final answer.

`gaia-ec09fa32` produces **byte-identical output (19 events, 362 chars) in all
three runs** -- a deterministic early stop, not sampling noise.

### 16.7 Why this is GOOD news for the research plan

This is the ideal target for harness evolution. The failure mechanism is
* real,
* dominant (100% of failures),
* mechanically detectable (did the run produce a final answer or a progress note),
* and plausibly fixable by editing the harness (skill/instruction telling the
  agent to keep going until it has an answer, and never to return narration).

That is exactly the kind of artifact edit our editor produces, and it means a
genuine self-improvement delta is available here rather than noise-chasing.

**Two changes required before any evolution claim:**
1. **Detect non-answers.** A run returning a progress note must be recorded
   `unscorable` (or given its own status), NOT counted as a wrong answer. Without
   this, the analyzer will "diagnose" reasoning failures that never happened, and
   the delta will be dominated by whether runs happened to finish.
2. **Report the delta against the measured spread**, never as a single number:
   the live tiny5 baseline is 60% / 20% / 40% (mean 40%, range **40 pp**).

tiny5 remains a plumbing smoke test only -- one flipped task is worth 20 pp.

---

## 17. CORRECTION to section 16: autonomous mode IS on; truncation has TWO causes

The user pointed at `reference/.../cuga-integration-learnings.md` and `run2.py`,
where a truncation issue was previously fixed with
`DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true` +
`..._CUGA_LITE_NL_AUTO_CONTINUE=true`. Checked rather than assumed:

**Both flags ARE set in our `.env` and ARE effective**, verified in-process AND in
a simulated worker environment (with `CUGA_FOLDER` stripped, as the pool does):
```
settings.force_autonomous_mode      : True
settings.cuga_lite_nl_auto_continue : True
CHILD force_autonomous_mode         : True   (worker env)
```
Only `CUGA_FOLDER` is stripped by `_STRIPPED_ENV`, and the worker calls
`prepare_environment()` BEFORE importing cuga. So this is NOT a repeat of the
earlier bug.

### 17.1 My "100% of failures are truncations" claim was too strong

I derived it from a narration regex over 15 rollouts. Deeper inspection of the
event graphs shows the autonomous loop **is** running:

```
run          task            events  call_model  sandbox  gave_up
autoprobe    gaia-5d0080cb     91       10          8      True
baseline-c   gaia-5d0080cb    102       12         10      False
baseline-a   gaia-ec09fa32     19        1          0      True
```
mean `call_model` cycles = 3.9; 7/16 ran >=2 sandbox cycles; one ran 12 model
cycles and 10 sandbox executions. **That is a working multi-step loop, not a
premature single-turn finalization.** My earlier reading of "2-4 cycles = agent
voluntarily stops immediately" was wrong -- I sampled run B, the shortest run.

### 17.2 There are actually TWO distinct mechanisms

**(a) Genuine give-up after a tool failure** -- 6/16 rollouts contain explicit
give-up language. The clearest case (`autoprobe`, 91 events, 10 model cycles,
8 sandbox executions):
```
"The failure was only a missing import. I'm re-running the extraction with `re`
 explicitly imported, then I'll fetch the first Leicester-hosted PDF URL found.
 I'm sorry, but I can't reliably determine ..."
```
The agent diagnosed its own tool error, said it would retry, and then quit. This
is a **real, evolvable harness weakness**: recovery-after-tool-error.

**(b) A deterministic early stop on one task.** `gaia-ec09fa32` produced
**19 events, 1 call_model cycle, 0 sandbox executions, byte-identical output in
all 4 runs.** Zero sandbox means no code was ever emitted -- exactly the
"deterministic function of prompt wording" failure documented in the reference
learnings (`extract_code_from_model_response` returns `""` -> no-code branch).
This one is NOT noise and NOT a give-up; it never started.

### 17.3 What still stands from section 16

* The pass-rate spread is real: **60% / 20% / 40%** on identical inputs (range
  40 pp on n=5).
* The PASS/FAIL vs narration separation was 6/0 and 0/9 -- narration in the final
  answer remains a strong failure signal, and **scoring a narration or a give-up
  as a "wrong answer" still corrupts the analyzer's input**. Recording them
  distinctly is still the right fix.
* `FinalAnswerAgent` appeared in all 15 rollouts and `PlanControllerAgent` in
  none. NOT yet explained: with `force_autonomous_mode=True`,
  `cuga_lite_node.py:529-571` routes success to `PlanControllerAgent`. Either
  `_has_error` fired on every run, or CugaLite's internal loop terminates before
  that branch. Worth resolving, because it decides whether the agent gets a
  planning pass at all. **Open.**

### 17.4 Instrumentation gap found

Worker subprocess CUGA logs are **not captured** into the run log
(`is_autonomous_subtask`, `Routing to:` never appear). They go to the child's
stderr and are lost. That is why this took a direct single-task rerun to see.
Capturing worker stderr per task is needed before the first real evolution run,
or every rollout diagnosis will require a manual re-run.
