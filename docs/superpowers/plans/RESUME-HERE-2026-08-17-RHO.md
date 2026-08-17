# RESUME HERE — 2026-08-17 — RHO Stage, ready for subagent-driven execution

Read this first after compaction. Everything needed to continue is here or linked.

---

## 0. Immediate next action

Execute the RHO implementation plan **task-by-task using subagent-driven
development**. The user approved this path; the plan is written, reviewed, and
verified.

**Plan:** `docs/superpowers/plans/2026-08-17-rho-stage.md` (4,824 lines, 15 tasks)
**Spec:** `docs/superpowers/specs/2026-08-17-rho-stage-design.md` (812 lines)

Before Task 1, establish the baseline:

```bash
mkdir -p terminal_output/rho
uv run pytest -p no:warnings 2>&1 | tee terminal_output/rho/baseline-suite.log | tail -3
```

Expected: **1359 passed, 1 skipped** (verified 2026-08-17).

---

## 1. Verified repo state

| Fact | Value |
| --- | --- |
| Branch | **`dev5`** (NOT dev4 — earlier notes were wrong) |
| Suite | **1359 passed, 1 skipped, 0 failed** |
| Working tree | clean except untracked new files |
| Commits made this session | **none** |

Untracked (all new, none committed):

```
docs/from_rho_paper_referance/
docs/superpowers/plans/2026-08-17-rho-stage.md
docs/superpowers/specs/2026-08-17-rho-stage-design.md
feedback/from_qwen/qf34.md
feedback/from_qwen/qf35.md
reference/evolve_run.py
```

**Commit policy: do NOT commit without explicit user approval.** Each plan task
ends with a staged commit command; ask before running it.

---

## 2. What RHO is, in this repo

RHO = Retrospective Harness Optimization. Select a difficult-and-diverse coreset
from historical traces, diagnose group rollouts, propose N independent candidate
harnesses, rank by pairwise self-preference, and **retain all N** in the
persistent pool as parents for the existing genetic loop.

**The one deliberate deviation from the paper:** the paper takes best-of-N and
discards the rest. We keep all N. That is the entire point — N distinct harness
hypotheses become the parents whose *disagreement* the genetic stage exploits.

### Two execution interfaces (this was a mid-session correction)

The paper drives **Codex CLI** for trajectory-rich stages and an ordinary LLM
client only for difficulty/fingerprint. We mirror that split, substituting the
**CUGA SDK** for Codex CLI.

| Stage | Paper | Ours |
| --- | --- | --- |
| Rollouts (solve + re-solve) | Codex CLI | **CUGA SDK rollout** (exists) |
| Trajectory comprehension | (digest) | **Interface A** structured LLM |
| Difficulty + fingerprint | Ordinary LLM | **Interface A** |
| Embeddings, DPP, aggregation | Local compute | **Deterministic Python** |
| Group diagnosis | Codex CLI x k | **Interface B** workspace agent x10 |
| Harness optimization | Codex CLI x N | **Interface B** x3 |
| Pairwise ranking | Codex CLI | **Interface B** x30 |

- **Interface A** = structured stateless LLM call, injectable `completion_fn`
- **Interface B** = CUGA workspace agent with tools, result captured from
  **staged artifacts**, injectable `agent_factory`

**Key discovery: Interface B is not new work.** `cuga_editor.propose_edit`
already IS a workspace agent — multi-turn
`agent.invoke(prompt, track_tool_calls=True)` over 19 tools, results from staged
artifacts, unfinalized staging discarded. Task 6 extracts that mechanism.

### Cost per round (k=10, G=3, N=3, R=2)

- **90 rollouts** = 30 baseline (k x G) + 60 candidate (k x N x R)
- **43 workspace-agent invocations** = 10 diagnose + 3 optimize + 30 judge
- up to 84 cached Interface A calls

At `--rho-rounds 3` that is 270 rollouts before genetic work — the dominant cost.
Paper's own accounting is `30+10+3+30+30 = 103` agent invocations; ours matches.

---

## 3. Decisions the user made (do not relitigate)

| Decision | Rationale |
| --- | --- |
| `--rho-history` trace dir with **cold-start fallback** | lets RHO be built/tested before a fresh corpus exists |
| **Preference judge gets GT** when the split has it | judge is "grader-llm preference judge", not a regex grader |
| **AGENTS.md no-labels rule OVERRIDDEN** for judge+editor | user's explicit call; containment by prompting |
| Single outer loop, **phases selected by mode** | user corrected an earlier mis-selection of the phase-list option |
| **N independent agent invocations**, not `n=N` sampling | paper-faithful; diversity from tool trajectories |
| **Paper defaults** k=10, G=3, N=3 | |
| **Pairwise judge = Interface B** workspace agent | paper-faithful, cost accepted |
| **Shared preference judge** for RHO and genetic | it is task-metadata/GT aware in both |
| RHO analyzer/editor are **separate** from genetic ones | only the CUGA-SDK call plumbing is shared |
| **Two-level concurrency** group x rollout + global cap | from `reference/evolve_run.py` |
| **Trajectory comprehension** phase added | raw traces are 60.8% identifiers |
| **`R=2` instead of removing the entropy skip tier** | user's fix, better than mine — see §4 |
| Task 11 **SKIPPED** | superseded by R=2 |
| Contamination detector included | cheap, restricts nothing |

---

## 4. The entropy decision (user's correction — important)

I originally planned to delete the `"skip"` tier from `EntropyTracker.classify`.
The user proposed instead setting candidate rollouts to 2 so the floor is met
naturally. **That is the better fix and is what the plan now does.**

Verified floors in `src/agent_evolve/core/entropy.py:110-111`:

```python
min_comparable_candidates: int = 3
min_rollouts_per_candidate: int = 2
```

With R=2: base (G=3) + 3 candidates (R=2 each) = **4 comparable candidates**, each
with >= 2 rollouts. Both floors met, so `classify` never returns `"skip"` for a
RHO-populated cell, and the guard still protects genuinely thin cells.

Why it is better: the floor exists because a mean from one rollout is
untrustworthy. Deleting it hides that. R=2 is informative because CUGA rollouts
are **stochastic** — tiny5 gave 3/5 then 1/5 on the same harness (40pp spread).

**`src/agent_evolve/core/entropy.py` must NOT be modified.**

### Wiring gotcha (pinned by a test in Task 13)

`EntropyTracker._comparable_candidates` counts only candidates promoted via
**`mark_comparable()`** (`entropy.py:158`). Rollout count alone is NOT enough —
without that call `comp` is empty and entropy is `None` regardless of R.
`Orchestrator._cell_entropy` does not share this requirement (it reads
`pool.all_entries()`), so **both paths must be satisfied independently.**

---

## 5. Genetic phase task scope (user asked; answer is load-bearing)

In `rho-genetic`, the genetic phase runs on the **coreset tasks only** (k=10),
not all 42.

Forced by the data, not chosen for cost: cross-candidate variance needs a
`(task, mechanism)` cell populated for multiple candidates, and cells are created
by rollouts. After a RHO round, cells exist only for the k coreset tasks — on the
rest, variance is **undefined**, not low. All 42 x 4 candidates x R=2 would be
**336** rollouts/round instead of 90.

Full dataset is used exactly twice: coreset-selection input, and final champion
measurement.

### Two different score signals — do not conflate

- **grader (`expected_regex`)** populates the score tensor → this is what
  cross-candidate variance is computed from; free (regex, no LLM)
- **preference judge (Interface B)** decides RHO candidate *ranking* → stays at
  N x k = 30 invocations, one verdict per (candidate, task), **not per rollout**

So R doubles rollouts but **not** judge calls.

### Two quality functions, one DPP

| | RHO coreset | Genetic issues |
| --- | --- | --- |
| Quality | judge **difficulty** | **cross-candidate variance** |
| Diversity | fingerprint cosine | mechanism cosine |

Same `build_kernel`/`greedy_map`; different quality vector. Do not unify.

---

## 6. Measured facts that justify design choices

### Raw traces are mostly identifiers (justifies trajectory comprehension)

Measured on `data/traces/0cb88c5a-1a6e-4aea-8ce0-f84c3f926e68/causal-trace.json`
(9,610 bytes, 19 events):

| Component | Count | Bytes | Share |
| --- | --- | --- | --- |
| UUIDs | 40 | 1,440 | 15.0% |
| Long hex hashes | 20 | 1,280 | 13.3% |
| JSON keys | 260 | 3,121 | 32.5% |
| **Total** | | **5,841** | **60.8%** |

Event kinds all structural (`graph_node_start` x7, `graph_node_end` x8,
`llm_call_start` x2, `llm_call_end` x2); `final_output` 202 bytes;
`tool_observations` 0. Embedding raw traces saturates cosine similarity and
neutralizes DPP diversity. Truncation does not help (cuts prose, keeps schema).

### Variance decomposition (recorded as FUTURE work, spec §6.1)

`total = between + within` (law of total variance), verified numerically:

| cell | total | between | within |
| --- | --- | --- | --- |
| A: harnesses disagree, each stable | 0.2222 | **0.2222** | 0.0000 |
| B: harnesses identical, all flaky | **0.2500** | 0.0000 | 0.2500 |
| C: mixed | 0.2500 | 0.1667 | 0.0833 |
| D: all agree, stable | 0.0000 | 0.0000 | 0.0000 |

B (pure flicker, nothing to recombine) outranks A (real disagreement). Future
fix: rank on `between`, route high-`within`/low-`between` to an instability work
type. **Not implemented now.**

### Temperature — my earlier claim was WRONG

Correct: **`temperature` IS supported; only `0.0` is rejected.**

- forwarded when supplied: `cuga_wrapper/__init__.py:909`,
  `adapters/cuga_analyzer.py:511`
- `0.0` → `BadRequestError: Unsupported value: 'temperature' does not support 0.0`
- `0.2` / `1.2` forwarded and tested (`tests/test_cuga_analyzer.py:576`)
- default omits it → greedy decode → identical prompts byte-identical
- `n=k` sampling already works: `cuga_wrapper/__init__.py:938`
- CUGA's `agent.temperature` affects **rollouts only**; editor path bypasses CUGA

Sampling is an **ablation knob**, not the diversity mechanism.

### GT availability differs by split (justifies the judge's GT branch)

| exp file | tasks | distinct regex | GT usable |
| --- | --- | --- | --- |
| `gaia_l1_validation.json` | 42 | 39 | **yes** |
| `gaia_l1_validation_tiny5.json` | 5 | 3 | yes |
| `gaia_l1_test.json` | 68 | **1** | **no — placeholder** |
| `gaia_l1_test_tiny10.json` | 10 | **1** | **no — placeholder** |

Test splits carry `(?i)\?` for every task — matches any question mark, passes
vacuously. **Never treat it as ground truth.** Judge must branch on
`gt_available`.

Real GT literals for the contamination detector: `17`, `0.1777`,
`Mapping Human Oriented Information to Software Agents for Online Systems Usage`.

---

## 7. Plan structure (15 tasks, 77 steps)

| # | Task | Interface | Depends on |
| --- | --- | --- | --- |
| 1 | History load + stale rejection + cold start | deterministic | — |
| 2 | Content-hash disk cache | deterministic | — |
| 3 | Trajectory comprehension | **A** | 1, 2 |
| 4 | Difficulty + fingerprint judge | **A** | 1, 2, 3 |
| 5 | Coreset DPP (quality = difficulty) | deterministic | 1 |
| 6 | Shared workspace-agent runner | **B** mechanism | — |
| 7 | Group diagnoser (1 call/task) | **B** | 6 |
| 8 | Optimizer x N independent | **B** | 6, 7 |
| 9 | Preference judge (signed, shared) | **B** | 6 |
| 10 | Two-level scheduler + global cap | deterministic | — |
| 11 | **SKIPPED** (R=2 supersedes) | — | — |
| 12 | Contamination detector | deterministic | — |
| 13 | Round config, phases, summaries | deterministic | 10 |
| 14 | CLI flags + preflight invariant | — | 13 |
| 15 | Full-suite verification + dry run | — | all |

Parallelizable: 1, 2, 5, 6, 10, 12 are independent. 7/8/9 need 6. 13 needs 10.
14 needs 13.

### Files created

```
src/agent_evolve/core/rho/{__init__,history,coreset,scheduler,rounds,cache}.py
src/agent_evolve/core/contamination.py
src/agent_evolve/adapters/cuga_workspace_agent.py
src/agent_evolve/adapters/cuga_rho_{comprehender,judge,diagnoser,optimizer}.py
src/agent_evolve/adapters/cuga_preference_judge.py
```

Modified: `scripts/run_evolution.py` (CLI + `resolve_rho_config`),
`src/agent_evolve/pipeline.py` (mode dispatch — deferred, see §9).

---

## 8. Non-negotiable constraints for every task

- `src/agent_evolve/core/**` MUST NOT import `cuga`, `litellm`, or
  `agent_evolve.adapters`. Task 15 verifies via AST walk.
- Interface A → injectable `completion_fn`; Interface B → injectable
  `agent_factory`. **No test makes a network call.**
- **NEVER send `temperature=0.0`.**
- Rollout concurrency > 1 REQUIRES `--isolation process` (`CUGA_FOLDER` is
  process-global; two threads were observed both reading the second workspace
  while each trace stamped its own `harness_version`).
- Preflight invariant:
  `--max-workers <= --rho-group-workers * --rho-rollout-workers`.
  Refuse, never clamp. Credential-independent.
- Artifact ids: `instructions` scalar, or `skills|policies|memory/<name>`.
  Created ids start `skills/generated-`.
- **All N candidates retained. NEVER prune to best-of-N.**
- CUGA SDK imports stay inside function bodies.
- Capture citable commands: `2>&1 | tee terminal_output/<topic>/<name>.log`.
- Tests before implementation, always.

---

## 9. Explicitly NOT in this plan

- **Live pipeline wiring** (`pipeline.py` RHO stack, executing phases against
  real adapters). Tasks 1-14 build + unit-test components; `--mode rho` will NOT
  be runnable at the end of this plan. Next plan.
- **Fresh 42-task baseline collection** — required before any RHO run is
  meaningful; existing traces are stale-format.
- Entropy between/within decomposition (§6.1 future work).
- `cuga_proxy_validator` wiring, predicate-measurability guard, Option C
  planning knobs, semantic result search.
- `cuga_editor` refactor onto `run_workspace_agent` — **accepted tech debt**, two
  similar mechanisms will coexist. Deliberate: the genetic path produced the
  measured baseline and must not be destabilized under deadline.

---

## 10. External review status (qf35)

`feedback/from_qwen/qf35.md` reviews the plan: **APPROVED FOR EXECUTION**.
Verified `dev5` alignment for `build_kernel`/`greedy_map`, `max_editor_calls`,
`_harness_slot`, `register_candidate`, `creatable_prefix`.

**Two of its notes are STALE — do not act on them:**

1. **§III.1 "Task 11 entropy skip removal"** — superseded. Task 11 is SKIPPED;
   R=2 replaces it. Do not remove the skip tier.
2. **§III.2 "`propose_edits` arity change 3→4"** — **hallucinated.** Verified:
   the plan never mentions `propose_edits`, and it already returns 4 values
   (`orchestrator.py:1622`). That note describes a different plan. Ignore it.

Its valid points: track the Task 6 duplication as tech debt (agreed, §9), and
run the baseline suite before Task 1 (agreed, §0).

---

## 11. Key file references

| Path | Why it matters |
| --- | --- |
| `docs/superpowers/plans/2026-08-17-rho-stage.md` | THE PLAN — execute this |
| `docs/superpowers/specs/2026-08-17-rho-stage-design.md` | design rationale |
| `docs/from_rho_paper_referance/RHO_agents_context.md` | Codex-CLI vs ordinary-LLM split (:6-16, :288-299) |
| `docs/from_rho_paper_referance/RHO_summary.md` | pipeline, DPP math, hyperparameters |
| `reference/evolve_run.py` | two-level concurrency (:99-105), invariant (:194) |
| `docs/rho_evolution/` | historical Gaia RHO; rationale only, different edit surface |
| `src/agent_evolve/core/issues.py` | `build_kernel:233`, `greedy_map:251` — reuse |
| `src/agent_evolve/core/entropy.py` | floors :110-111, `mark_comparable:158` — DO NOT MODIFY |
| `src/agent_evolve/adapters/cuga_editor.py` | existing workspace agent to extract from |
| `src/agent_evolve/adapters/cuga_editor_tools.py` | 19 tools, `build_tool_callables` |
| `src/agent_evolve/adapters/cuga_editor_state.py` | `EditStagingArea`, `StageOutcome` |
| `src/agent_evolve/adapters/cuga_adapter.py` | `register_candidate`, `_harness_slot` |
| `src/agent_evolve/benchmarks/cuga_executor.py` | `HarnessVersion:309`, `VANILLA_HARNESS` |
| `data/traces/0cb88c5a-.../causal-trace.json` | canonical fixture; the 60.8% measurement |
| `docs/USER-MANUAL.md` | operational CLI/config/export manual |

---

## 12. Carry-over facts from earlier phases

- SDK runs a simplified graph:
  `CugaLiteSubgraph → prepare → call_model ⇄ sandbox → SDKCallback → FinalAnswerAgent`.
  **No `PlanControllerAgent`** — that exists only in the prebuilt local
  server/full `DynamicAgentGraph`. Staying on SDK Option A.
- Describe the system as single-agent ReAct/CodeAct, **not** hierarchical
  planner/executor.
- CUGA executes only the **first** fenced Python block per turn.
- Tools available to rollouts: `calculator`, `web_search`, `web_fetch`,
  `wikipedia_search`, `save_note`.
- Ground truth for tool execution is tool-body execution /`tool_observations`,
  **never** model prose, never `InvokeResult.tool_calls`.
- Baseline model is usually *unwilling*, not unable, to emit executable code. A
  temporary fence directive made tools fire and solved `3f57289b` with `519`.
  **Do NOT hand-insert that fix into `vanilla`** — evolution must discover it.
- Non-answer policy: `core/non_answer.py`; historical baseline `17/42 = 40.48%`
  full denominator, `17/32 = 53.13%` committed-answer denominator, 10
  non-answers. Old 16.67pp noise floor is stale.
- Attempt records still not persisted (`storage=None`) → always use
  `--capture-logs` and `--export-harness`.
- `data/harnesses/` intentionally not gitignored.
- `weighted_net_gain` is correct; passing regression probes cost nothing.

---

## 13. Execution protocol

Use **superpowers:subagent-driven-development**. Per task:

1. Dispatch a fresh subagent with the task's full text (it sees only its own task,
   so the plan's `**Interfaces:**` blocks are how it learns neighbouring names).
2. Subagent writes the failing test, verifies it fails, implements minimally,
   verifies it passes.
3. Review the diff before moving on.
4. **Ask the user before committing.**
5. After Tasks 9, 14, and 15, run the full suite.

Tell the user honestly if a task reveals the plan was wrong — the plan is a
hypothesis, not scripture.
