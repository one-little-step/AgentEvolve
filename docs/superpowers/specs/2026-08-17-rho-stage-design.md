# RHO Stage Design (CUGA / AgentEvolve)

Date: 2026-08-17
Status: approved, pending implementation plan
Scope: add a Retrospective Harness Optimization (RHO) stage to AgentEvolve and
compose it with the existing genetic (mutation/crossover) loop.

Authoritative references:

- `docs/from_rho_paper_referance/RHO_summary.md` (pipeline and hyperparameters)
- `docs/from_rho_paper_referance/RHO_agents_context.md` (which stage uses an
  agentic call vs. an ordinary LLM call)
- `docs/from_rho_paper_referance/RHO_prompts_summary.md` (prompt families P1-P5)
- `docs/rho_evolution/` (historical Gaia RHO-GEPA implementation; rationale and
  schemas only, **not** a template for the edit surface)
- `reference/evolve_run.py` (previous Gaia RHO runner; source of the two-level
  concurrency model and cache/manifest fields adopted in §4.5.1-4.5.2)

The Gaia implementation in `docs/rho_evolution/` targeted a different edit
surface (Markdown wisdom bundles). This design targets the CUGA SDK's four
editable surfaces and reuses none of Gaia's artifact assumptions.

---

## 1. Objective

RHO optimizes the **harness** around a fixed model by examining accumulated
trajectories, diagnosing recurring weaknesses, and proposing modified
harnesses. It requires no ground-truth validation labels to *select* a harness;
the paper's mechanism is the agent's own pairwise preference.

AgentEvolve deviates from the paper in one deliberate, load-bearing way:

> **The paper takes best-of-N and discards the rest. We retain all N candidates
> in the persistent pool and use them as parents for the post-RHO genetic
> stage.**

This is the point of the whole exercise. RHO produces N materially different
harness hypotheses; the genetic stage then finds issues where those hypotheses
*disagree* and combines what each got right.

---

## 2. Architecture: two cognition stacks, one invocation layer

RHO and the genetic loop each get their own analyzer and their own editor, with
their own prompts, instructions, skills, and output contracts. What they share
is the decoupled CUGA-SDK invocation layer beneath them.

### 2.1 Two execution interfaces

The published RHO implementation drives **Codex CLI** for its trajectory-rich
and filesystem-rich stages, and an **ordinary LLM client** only for difficulty
and fingerprint generation. Embeddings, DPP, and aggregation are deterministic
computation (`RHO_agents_context.md:6-16, 426-434`).

We mirror that split exactly, substituting the **CUGA SDK** wherever the paper
uses Codex CLI. Two interfaces, chosen per stage:

**Interface A — structured stateless LLM call.** Bounded context in, schema-shaped
JSON out. No filesystem, no tools.

**Interface B — CUGA SDK workspace agent.** Instructions plus a workspace plus
tools, producing a multi-step trajectory and a result captured **from staged
artifacts, not parsed from final text**.

| Stage | Paper mechanism | Our mechanism |
| --- | --- | --- |
| Task solve / group rollouts | Codex CLI, 1 per rollout | **CUGA SDK rollout** (exists) |
| Trajectory comprehension | (bounded digest) | **Interface A** — semantic summary |
| Difficulty + abstract fingerprint | Ordinary LLM client | **Interface A** |
| Fingerprint embeddings | Local embedding model | **Deterministic** (`OllamaEmbedder`) |
| DPP selection, score aggregation | Deterministic Python | **Deterministic Python** |
| Group diagnosis (self-validation + self-consistency, one call) | Codex CLI, 1 per task | **Interface B** |
| Harness optimization | Codex CLI × N | **Interface B × N** |
| Candidate re-solving | Codex CLI | **CUGA SDK rollout** (exists) |
| Pairwise self-preference ranking | Codex CLI | **Interface B** |

Interface B is **not new work**: `adapters/cuga_editor.py` is already a
workspace agent. Its `propose_edit` runs a multi-turn
`agent.invoke(prompt, track_tool_calls=True)` loop over 19 tools
(`cuga_editor_tools.py`), including `read_trace_events`, `list_artifacts`,
`read_artifact`, `stage_replace`, `stage_create`, `list_staged`, `unstage`,
`read_parent_artifact`, and `submit_edit_plan`. Results come from staged
artifacts, and unfinalized staging is discarded rather than silently applied.
That is precisely the paper's "captured from the filesystem, rather than parsed
from the optimizer's final textual answer."

Consequence: **RHO's N candidates are N independent workspace-agent
invocations**, as in the paper — not one sampled request. Diversity therefore
comes from independent tool trajectories rather than from token sampling, which
is both more faithful and a stronger diversity guarantee.

```
              ┌────────── shared CUGA-SDK invocation layer ──────────┐
              │ CugaRolloutRunner · cuga_process_pool · trace capture│
              │ CugaAdapter.register_candidate · artifact isolation  │
              │ CandidatePool (append-only) · export · run_logging   │
              └─────────────────────────────────────────────────────┘
                        ▲                              ▲
        ┌───────────────┴──────────┐      ┌────────────┴──────────────┐
        │      RHO cognition       │      │    genetic cognition      │
        │ A: RhoComprehender       │      │ cuga_analyzer (causal)    │
        │ A: RhoDifficultyJudge    │      │ cuga_editor (mut/cross)   │
        │ B: RhoGroupDiagnoser     │      │ focused validation        │
        │ B: RhoOptimizer × N      │      │                           │
        └──────────────────────────┘      └───────────────────────────┘
                        └──────────┬───────────────────┘
                    B: PreferenceJudge (shared workspace agent)
```

| Layer | RHO | Genetic | Shared |
| --- | --- | --- | --- |
| CUGA rollout invocation, process pool, trace capture | yes | yes | **shared** |
| Artifact injection / isolation / `register_candidate` | yes | yes | **shared** |
| Pool, export, run logging, budgets | yes | yes | **shared** |
| Workspace-agent harness (`cuga_editor` machinery) | yes | yes | **shared mechanism, separate prompts** |
| Analyzer prompt + output contract | `RhoGroupDiagnoser` | `cuga_analyzer` | separate |
| Editor prompt + skills + output contract | `RhoOptimizer` | `cuga_editor` | separate |
| Candidate ranking | `PreferenceJudge` | `PreferenceJudge` | **shared** |

`src/agent_evolve/core/` remains agent-neutral and imports no CUGA. Interface A
calls take an injectable `completion_fn`; Interface B calls take an injectable
agent factory, so the entire test suite runs offline.

---

## 3. Edit surfaces

RHO edits exactly the four surfaces the CUGA adapter can deliver, as validated
by `CugaAdapter._harness_slot`:

- `instructions` — scalar (`_SCALAR_ARTIFACTS`), unconditional turn-level behavior
- `skills/<name>` — opt-in, reached only via `load_skill`
- `policies/<name>` — requires a real trigger
- `memory/<name>` — retrieval and facts, not behavior

These map onto `HarnessVersion.{instructions,skills,policies,memory}`
(`benchmarks/cuga_executor.py:309`). Artifact ids that do not map to a slot are
rejected at registration, not silently dropped.

---

## 4. Round phases

Defaults follow the paper: `k=10` coreset tasks, `G=3` group rollouts,
`N=3` candidates.

| # | Phase | Interface | Cost at k=10, G=3, N=3 |
| --- | --- | --- | --- |
| 1 | History load | deterministic | 0 |
| 2 | Trajectory comprehension | **A** structured LLM, cached | ≤ 42 calls, cached |
| 3 | Difficulty + fingerprint | **A** structured LLM, cached | ≤ 42 calls, cached |
| 4 | Coreset selection | deterministic (embed + DPP) | 0 (local) |
| 5 | Group rollouts | **CUGA SDK rollout** | **30 rollouts** |
| 6 | Group diagnosis | **B** workspace agent, 1 per task | 10 agent runs |
| 7 | Candidate proposal | **B** workspace agent × N, independent | **3 agent runs** |
| 8 | Candidate rollouts | **CUGA SDK rollout** | **60 rollouts** (N x k x R, R=2) |
| 9 | Preference judging | **B** workspace agent | 30 agent runs |
| 10 | Pool commit | deterministic | 0 |

Total per round: **90 rollouts** (30 baseline + 60 candidate) plus **43
workspace-agent invocations** (10 diagnose + 3 optimize + 30 judge) plus up to 84
cached structured calls.

Candidate rollouts are `k x N x R` with **R = 2** rollouts per candidate per
coreset task. R exists to satisfy the cross-candidate entropy evidence floor
(section 6.1) honestly rather than by deleting the guard. Preference judging stays
at `N x k = 30` invocations: one verdict per (candidate, task), not per rollout.

The 103-invocation accounting in `RHO_agents_context.md:288-299` is
`30 solve + 10 diagnose + 3 optimize + 30 candidate solve + 30 evaluate`; ours
matches, with the 60 solves counted as rollouts.

### 4.1 History load (`core/rho/history.py`)

Reads `--rho-history <trace-root>` in the current causal-trace layout
(`<trace-root>/<run_id>/causal-trace.json`, with `manifest.json`,
`events.jsonl`, `tool_observations`, `final_output`).

A `HistoricalRecord` carries: `task_id`, `input_text`, a bounded trajectory
digest, `tool_observations`, `final_output`, and `harness_version`.

Stale-format traces are rejected with a named error, not silently accepted.
Old 42-task traces have 8 generic `stream_event`s, `actor_id=None`, and no
`tool_call` events; they cannot support diagnosis and must not be treated as
history.

**Cold start.** When `--rho-history` is absent or yields zero valid records,
RHO skips phase 2, selects the coreset by seeded-random or dataset order
(recorded as `selection_method`), and lets phase 4 generate the evidence. This
lets RHO be built and tested before a fresh baseline corpus exists.

### 4.2 Trajectory comprehension (`adapters/cuga_rho_comprehender.py`)

**Interface A.** One structured LLM call per historical record, producing a
bounded semantic summary of what the agent actually did and where it went wrong.

This phase exists because **a raw causal trace is mostly identifiers, and
embedding it destroys DPP diversity.** Measured on the canonical current-format
trace `data/traces/0cb88c5a-.../causal-trace.json` (9,610 bytes, 19 events):

| Component | Count | Bytes | Share |
| --- | --- | --- | --- |
| UUIDs | 40 | 1,440 | 15.0% |
| Long hex hashes | 20 | 1,280 | 13.3% |
| JSON keys | 260 | 3,121 | 32.5% |
| **Identifiers + schema total** | | **5,841** | **60.8%** |

Over 60% of the payload is identifiers and schema vocabulary, before counting
braces and quotes. Event kinds are entirely structural
(`graph_node_start` ×7, `graph_node_end` ×8, `llm_call_start` ×2,
`llm_call_end` ×2); `final_output` is 202 bytes.

Embedding that directly means cosine similarity is dominated by *shared schema*
rather than shared failure structure. Every trace contains the same keys and the
same event kinds, so all traces appear mutually similar, similarity saturates
near-uniformly high, and the DPP diversity term stops discriminating — leaving
selection driven by difficulty alone. Mechanical head-and-tail truncation does
not fix this; it truncates prose while preserving the schema noise.

The previous Gaia RHO reached the same conclusion and shipped the same remedy:
`trajectory_summary.md` is documented as the "preferred input to embedding,"
with reconstruction for historical runs lacking one
(`docs/rho_evolution/selection_algo_explaination.md:48-53`,
`docs/rho_evolution/04-coreset-selection.md:26`).

Output contract:

```json
{
  "what_was_attempted": "...",
  "approach_taken": "...",
  "where_it_went_wrong": "...",
  "tools_used": ["web_search"],
  "outcome": "no_committed_answer"
}
```

The summary is prose about behavior, carrying no UUIDs, event ids, hashes, or
JSON keys. It is the input to phase 3 (difficulty and fingerprint) and the text
that gets embedded in phase 4.

Summaries are cached by trace content hash (§4.5.2). A record whose
comprehension call fails is excluded from difficulty-weighted selection and
counted in the report, rather than falling back to a raw-trace embedding.

### 4.3 Difficulty and abstract fingerprint (`adapters/cuga_rho_judge.py`)

**Interface A** — an ordinary structured LLM call, not an agentic one, matching
the paper (`RHO_agents_context.md:194-236`). It consumes the phase-2 semantic
summary rather than the raw trace.

Per record it returns:

```json
{"difficulty": 7.8, "abstract_fingerprint": "A multi-step retrieval task where ..."}
```

`difficulty` is in `[0, 10]`. The fingerprint describes structural form and
failure shape without task-specific identifiers (no filenames, proper nouns, or
answer values), so that similarity reflects failure structure rather than
surface vocabulary.

Results are cached by `(task_id, trace content hash, judge model)`, so
re-running a round costs no judge calls.

### 4.4 Coreset selection (`core/rho/coreset.py`)

Reuses the existing primitives in `core/issues.py` — `build_kernel:233` and
`greedy_map:251` — with a different quality vector.

- quality = normalized difficulty (phase 2), floored by `dpp_score_floor`
- diversity = cosine similarity over fingerprint embeddings
  (`core/embeddings.py`, `OllamaEmbedder`, lexical fallback with recorded
  `fallback_reason`)
- kernel = `diag(q) · S · diag(q) + jitter`, greedy MAP selection

Modes: `dpp` (default), `difficulty_rank`, `random` — the last two are ablation
baselines.

There is **no second DPP implementation**. RHO and the genetic stage call the
same kernel and greedy selection code with different quality inputs.

### 4.5 Group rollouts

Reuses `Orchestrator.rollout_group` (`core/orchestrator.py:1098`) and the
existing process pool. `G` independent rollouts of the base harness per coreset
task, under `--isolation process`.

Process isolation is mandatory: `CUGA_FOLDER`, the policy DB, and the knowledge
store are global and not thread-safe, and a trace's `harness_version` cannot
reveal a workspace swap.

#### 4.5.1 Two-level rollout concurrency

RHO's rollout phase is inherently a **group** structure (`k` tasks × `G`
rollouts), and a group's diagnosis cannot start until all `G` of its rollouts
finish. A single flat worker count cannot express this: with a flat cap of 6 the
scheduler cannot distinguish "6 tasks × 1 rollout" from "2 tasks × 3 rollouts".

This design adopts the two-level model from the previous Gaia RHO runner
(`reference/evolve_run.py:99-105`):

| Knob | Meaning | Default |
| --- | --- | --- |
| `--rho-group-workers` | concurrently admitted task groups | 4 |
| `--rho-rollout-workers` | concurrent rollouts within one group | 3 (= G) |
| `--max-workers` | **global** hard cap on simultaneous CUGA runs | 6 |
| `--analyzer-workers` | diagnosis / judging fan-out (LLM-bound) | 6 |

**Invariant, enforced at preflight** (from `reference/evolve_run.py:194`):

```
--max-workers <= --rho-group-workers * --rho-rollout-workers
```

A global cap larger than the two-level structure can produce is a configuration
error, not something to silently clamp.

**Group-major admission.** Groups are admitted up to `--rho-group-workers`, and
each admitted group runs its `G` rollouts up to `--rho-rollout-workers`, with
every simultaneous CUGA run counted against the global cap. Completing whole
groups early is preferred over spreading thinly across all `k` tasks, so
diagnosis for a finished group starts while later groups are still executing.

**Isolation.** Process isolation remains mandatory for any concurrency above 1,
enforced by the existing `require_safe_rollout_concurrency`
(`pipeline.py:196`). `CUGA_FOLDER` is a process-global environment variable read
during `invoke()`; two threads binding different workspaces were observed both
reading the second one's while each trace still stamped its own
`harness_version`. Thread-level rollout concurrency is therefore refused, before
anything expensive is constructed.

**What parallelizes where:**

- rollouts (phases 4, 7) — process-isolated, two-level, global cap
- diagnosis (phase 5), difficulty judging (phase 2), preference judging
  (phase 8) — LLM-latency-bound and workspace-free, so they fan out under
  `--analyzer-workers` via the existing `parallel_analysis` thread pool
- coreset DPP, embeddings, kernel construction (phase 3) — local and cheap

**Artifact writes are never parallel.** Only the coordinator commits to the
pool; workers produce results and commit nothing, per the existing snapshot /
exclusive-lease / coordinator-commit model in `core/parallel.py`.

#### 4.5.2 Caching

Following the reference's `TRAJECTORY_SUMMARY_CACHE_DIR` /
`TRAJECTORY_EMBEDDING_CACHE_DIR` and its `summary_cache_hits` /
`embedding_cache_hits` manifest fields, three caches are keyed by content hash
and reported in the round summary:

- difficulty + fingerprint verdicts (§4.3)
- trajectory digests
- fingerprint embeddings

Round 2 and later re-select a coreset over the same history, so these caches
remove nearly all repeat phase-2 and phase-3 cost. Cache hits and misses are
recorded per round; a cache is never allowed to mask a stale trace, because the
key includes the trace content hash.

### 4.6 Group diagnosis (`adapters/cuga_rho_diagnoser.py`)

**Interface B — one workspace-agent invocation per coreset task**, receiving all
`G` trajectories together. Per `RHO_agents_context.md:59-94`, self-validation and
self-consistency are two signals extracted **within one diagnosis invocation**,
not separate calls: `k=10` means 10 diagnosis runs, not 20.

The workspace gives the agent the task input, the harness, and the `G` trajectory
event streams as inspectable files, so it can read selectively instead of
receiving one enormous serialized prompt. This is the reason the paper uses an
agent here rather than a plain call, and it is also what keeps the prompt bounded
when `G=3` trajectories of ~19-56 events each would otherwise be concatenated.

The per-rollout causal analyzer (`cuga_analyzer`) continues to run; its findings
are supplied as supporting evidence.

Output contract:

```json
{
  "recurring_failure_mode": "...",
  "disagreements": ["..."],
  "self_validation_observed": false,
  "severity": 0.0,
  "improvement_direction": "...",
  "candidate_surfaces": ["instructions"]
}
```

- **self-consistency**: meaningful disagreements between the G trajectories,
  distinguishing harmless variation from consequential divergence
- **self-validation**: whether the agent checked its own answer before
  committing
- `severity` in `[0, 1]` orders the diagnosis bundle in phase 6

The prompt states the actual SDK graph shape
(`CugaLiteSubgraph → prepare → call_model ⇄ sandbox → SDKCallback →
FinalAnswerAgent`) and must distinguish "narrated without emitting an
executable code block, so `sandbox` was never reached" from a genuine tool
failure. Ground truth for tool execution is tool-body execution and
`tool_observations`, never model prose and never `InvokeResult.tool_calls`.

### 4.7 Candidate proposal (`adapters/cuga_rho_optimizer.py`)

**Interface B — N independent workspace-agent invocations**, matching the paper
(`RHO_agents_context.md:125-163`). Not one sampled request.

Each of the N optimizers gets its own workspace containing the current harness
(writable) and the full severity-ordered diagnosis bundle, and reaches its
result through the same staged-artifact mechanism `cuga_editor` already
implements: `list_artifacts`, `read_artifact`, `stage_replace`, `stage_create`,
`list_staged`, `unstage`, `submit_edit_plan`. The candidate is captured from
staged artifacts; unfinalized staging is discarded rather than silently applied.

```
candidate_0 = rho_optimizer.run(fresh_workspace_0)   # independent
candidate_1 = rho_optimizer.run(fresh_workspace_1)   # independent
candidate_2 = rho_optimizer.run(fresh_workspace_2)   # independent
```

Diversity therefore comes from **independent multi-step tool trajectories**, not
token sampling. Two agents that read different artifacts in a different order
and stage different edits are distinct for a substantive reason. Sampling
temperature is consequently not the diversity mechanism; the default omits
temperature and the provider default stands.

Recorded for completeness, since an earlier draft of this spec relied on it: the
claim "temperature is unsupported" was wrong. `temperature` is forwarded when
supplied (`cuga_wrapper/__init__.py:909`, `adapters/cuga_analyzer.py:511`), only
`0.0` is rejected by the endpoint, and `n=k` sampling works
(`cuga_wrapper/__init__.py:938`). Those remain available as an ablation knob but
are not the design.

The optimizer prompt is RHO's own — separate from `cuga_editor`'s mutation and
crossover prompts. It presents all four surfaces with their delivery mechanics
and instructs the agent to prioritize recurring, high-severity, cross-task
failures and to avoid task-specific hardcoding.

**Discard rules.** A candidate is dropped before evaluation when the invocation
fails or times out, when nothing was staged, or when its artifact set is
byte-identical to the base. A no-op is not a candidate. Surviving candidates are
deduplicated by artifact hash. The round reports `candidates_requested` and
`candidates_distinct`, so a collapse to 1 is visible rather than silently
comparing a harness against itself.

### 4.8 Candidate rollouts and preference judging

Each surviving candidate is registered via `CugaAdapter.register_candidate`
(which validates every artifact id against `_harness_slot`) and run on each
coreset task.

`adapters/cuga_preference_judge.py` is **shared by both stages** and is
**Interface B — a workspace agent**, as in the published implementation
(`RHO_agents_context.md:165-192`). The judge receives an evaluation workspace
holding the original task, the baseline harness and trajectory, and the candidate
harness and trajectory, and can inspect artifacts and diffs with tools before
committing to a verdict.

It returns a **signed** preference score for the transition
`baseline harness → candidate harness`: positive favors the candidate, negative
favors the baseline, zero is a tie or indeterminate.

This is the most expensive stage in the round (30 agent invocations at
`k=10, N=3`). It was chosen over a cheaper structured call deliberately, for
fidelity to the paper, with the cost accepted.

**Task metadata, including ground truth when available, is supplied to the
judge.** The judge is explicitly told whether GT is present, because the
available splits differ:

| exp file | tasks | distinct regex | GT usable |
| --- | --- | --- | --- |
| `gaia_l1_validation.json` | 42 | 39 | yes |
| `gaia_l1_validation_tiny5.json` | 5 | 3 | yes |
| `gaia_l1_test.json` | 68 | 1 | **no — placeholder** |
| `gaia_l1_test_tiny10.json` | 10 | 1 | **no — placeholder** |

The test splits carry a single shared `regex` of `(?i)\?`, which matches any
question mark and so passes vacuously. It is a placeholder, not ground truth.
When GT is absent the judge must fall back to process-quality comparison and
record `gt_available: false`; it must never treat the placeholder as an answer.

An unparseable or out-of-range verdict is recorded as unavailable and
contributes nothing to the candidate average, rather than defaulting to a tie.

**Relationship to `--grader`.** The `expected_regex` grader continues to run and
remains the reported pass-rate metric, so a run stays comparable to the existing
baseline. It does **not** rank RHO candidates: ranking is the preference judge's
job. The grader is therefore a recorded observable, and the preference judge is
the selection signal. Both are written to the round summary so their agreement
(or disagreement) is measurable.

### 4.9 Pool commit

All surviving candidates plus the base are committed to the existing
append-only `CandidatePool`. Nothing is discarded on the basis of rank; rank
determines reported ordering and champion selection only.

Every candidate is exportable and re-runnable through the existing
`--export-harness` directory mode and `--harness PATH`.

---

## 5. Modes

A single outer loop; the mode selects which phases run per outer iteration.

- **`rho`** — phases 1-9, repeated `--rho-rounds` times. Each round's base is
  the current champion.
- **`genetic`** — the existing mutation/crossover loop until budget exhaustion.
  Unchanged behavior.
- **`rho-genetic`** — `[RHO round → genetic iterations]` repeated
  `--rho-rounds` times, with per-phase budgets. The RHO round seeds and
  refreshes the pool; the genetic phase then exploits cross-candidate variance.

**Task scope of the genetic phase.** In `rho-genetic` the genetic phase operates
on the **coreset tasks only**, not the full dataset. This is forced by the data
rather than chosen for cost: cross-candidate variance can only be computed where a
`(task, mechanism)` cell exists for multiple candidates, and cells are created by
rollouts. After a RHO round, cells exist only for the `k` coreset tasks; on the
remaining tasks the variance is *undefined*, not low. Rolling out all 42 tasks
against 4 candidates at R=2 would cost 336 rollouts per round instead of 90.

The full dataset is used exactly twice: as coreset-selection input, and for the
final champion measurement.

Pool, export, run logging, budgets, and the adapter are shared across all three
modes.

---

## 6. Two quality functions, one DPP

This is the load-bearing distinction between the stages.

| | RHO coreset selection | Genetic issue selection |
| --- | --- | --- |
| Unit selected | historical task | (task, mechanism) issue cell |
| **Quality** | judge-assigned difficulty | **cross-candidate score variance** |
| Diversity | fingerprint cosine | mechanism-embedding cosine |
| Goal | hard and varied tasks that expose weakness | issues where harnesses *disagree*, so the good parts of each RHO candidate can be combined |

RHO fills the pool with N distinct harnesses; that is precisely what makes
cross-candidate variance measurable for the genetic stage.

### 6.1 Entropy: current decision and a recorded future improvement

`EntropyTracker._entropy` pools every rollout of every candidate into one flat
list and takes the variance, so it blends between-candidate disagreement with
within-candidate rollout noise. `Orchestrator._cell_entropy` instead uses one
mean per pool entry, which is true between-candidate variance. Two
implementations of the same documented quantity currently disagree.

**Decision for this deliverable (revised 2026-08-17): keep the blended
definition AND keep the skip guard.** The floor is satisfied by evidence instead
of by deletion: each candidate gets `R = 2` rollouts per coreset task, so every
RHO-populated cell has base (G=3) plus N candidates (R=2 each) = 4 comparable
candidates, each with at least 2 rollouts. `classify` therefore never returns
`"skip"` for a cell RHO populated, and the guard still protects cells that
genuinely lack evidence.

This is the better repair: the floor exists because a mean built from one rollout
is untrustworthy, and deleting it would have hidden that rather than fixed it.
R=2 is informative rather than wasteful because CUGA rollouts are stochastic --
the tiny5 run produced 3/5 then 1/5 on the same harness and tasks, a 40pp spread.

`EntropyTracker` is therefore left unchanged by this deliverable.

**Wiring requirement.** `EntropyTracker._comparable_candidates` counts only
candidates promoted through `mark_comparable()`; rollout count alone is not
sufficient. `Orchestrator._cell_entropy` does not share this requirement because
it reads `pool.all_entries()` directly, so both paths must be satisfied
independently.

**Recorded for future work**, with measured numbers. Because
`total = between + within` (law of total variance), blended is richer as a
number but inverts the ranking:

| cell | total (blended) | between | within |
| --- | --- | --- | --- |
| A: harnesses disagree, each stable `[0,0][1,1][0,0]` | 0.2222 | **0.2222** | 0.0000 |
| B: harnesses identical, all flaky `[0,1][1,0][0,1]` | **0.2500** | 0.0000 | 0.2500 |
| C: mixed `[0,0][1,1][0,1]` | 0.2500 | 0.1667 | 0.0833 |
| D: all agree, stable `[1,1][1,1][1,1]` | 0.0000 | 0.0000 | 0.0000 |

Case B — three harnesses behaving identically, each flickering randomly — ranks
**above** case A, a genuine harness disagreement. There is no
harness-dependent issue in B and nothing to recombine.

The future fix is to decompose rather than collapse: record all three, rank on
`between`, and route high-`within`/low-`between` cells to an instability work
type (needs more evidence, or a self-validation/determinism fix) instead of
discarding them. `Orchestrator._cell_entropy` should then delegate to
`EntropyTracker` so exactly one implementation exists.

The `min_comparable_candidates=3` / `min_rollouts_per_candidate=2` floors are
both met by construction: base + N = 4 candidate means clears the candidate
floor, and R = 2 clears the rollout floor. Setting `--rho-candidate-rollouts 1`
halves candidate rollout cost but drops every candidate mean onto a single
stochastic rollout, which fails the floor and returns those cells to `"skip"`.

---

## 7. Ground truth handling and the AGENTS.md override

`AGENTS.md` states:

> Never persist credentials, expected answers, evaluator internals, labels, or
> regexes to edit memory, embeddings, prompts, manifests, or terminal logs.

**This rule is explicitly overridden for the preference judge and the editor,
by decision on 2026-08-17.** Ground truth and full task metadata are supplied
to the judge, and the judge's output — including free-text rationale — may
reach the editor. Containment is by prompting rather than by a hard firewall.

Recorded consequence: a candidate can in principle improve its score by
carrying an answer in an artifact rather than by improving procedure, and
nothing in the data path prevents it.

**Mitigation (included, non-restricting): post-hoc contamination detector.**
After a run, scan every exported artifact for literal GT strings drawn from the
dataset regexes (for example `17`, `0.1777`,
`Mapping Human Oriented Information to Software Agents for Online Systems Usage`)
and report any hit with artifact id and surface. It constrains no prompt and
blocks no run; it exists so that a contaminated harness is discovered by the
detector rather than by a reviewer. Long, distinctive literals are reported at
high confidence; short numerics are reported as low-confidence hints, since
`17` legitimately occurs in prose.

---

## 8. New and reused code

**New — agent-neutral (`core/`, imports no CUGA):**

- `core/rho/history.py` — `HistoricalRecord`, trace loading, stale-format
  rejection, cold-start
- `core/rho/coreset.py` — difficulty × diversity DPP over historical records
- `core/rho/rounds.py` — round state machine, phase sequencing, mode dispatch

**New — CUGA adapters:**

- `adapters/cuga_rho_comprehender.py` — **A** trajectory semantic summary
- `adapters/cuga_rho_judge.py` — **A** difficulty and abstract fingerprint
- `adapters/cuga_rho_diagnoser.py` — **B** group diagnosis over G trajectories
- `adapters/cuga_rho_optimizer.py` — **B** × N independent candidate proposals
- `adapters/cuga_preference_judge.py` — **B** shared signed pairwise ranking

Interface B adapters reuse the existing workspace-agent machinery in
`cuga_editor.py` / `cuga_editor_tools.py` / `cuga_editor_state.py` (agent
invocation loop, tool registration, staging, transcript capture) with their own
prompts and tool subsets. The shared mechanism is extracted where needed; the
prompts are not shared.

**Reused unchanged:** `CugaRolloutRunner`, `cuga_process_pool`, trace capture,
`CugaAdapter.register_candidate`, `CandidatePool`, `export_pool` /
`--export-harness`, `run_logging`, `build_kernel` / `greedy_map`,
`OllamaEmbedder`, `non_answer`, and the `cuga_editor` workspace-agent harness.

**Modified:** `EntropyTracker` (remove skip), `pipeline.py` (mode dispatch and
RHO stack construction), `scripts/run_evolution.py` (CLI).

---

## 9. CLI

```
--mode {rho,genetic,rho-genetic}     default: genetic (existing behavior)
--rho-rounds N                       default: 1
--rho-history PATH                   trace root; omitted ⇒ cold start
--rho-coreset-size 10
--rho-group-rollouts 3
--rho-candidates 3
--rho-candidate-rollouts 2           R: rollouts per candidate per coreset task
--rho-proposal-temperature FLOAT     ablation only; unset by default (diversity
                                     comes from independent agent invocations,
                                     not sampling). 0.0 is rejected by the endpoint.
--rho-selector {dpp,difficulty_rank,random}
--rho-difficulty-cache PATH
--rho-summary-cache PATH             trajectory comprehension cache
--rho-group-workers 4                concurrent task groups
--rho-rollout-workers 3              concurrent rollouts within one group
--rho-embedding-cache PATH
--genetic-iterations-per-round N     rho-genetic only
```

`--max-workers` is the global hard cap and must satisfy
`--max-workers <= --rho-group-workers * --rho-rollout-workers`.

Existing flags keep their meaning. `--capture-logs` and `--export-harness`
remain mandatory in practice for any run whose result must outlive the process:
attempt records are still not persisted by default (`storage=None`).

Example:

```bash
uv run python scripts/run_evolution.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --harness vanilla \
  --mode rho-genetic --rho-rounds 3 \
  --rho-coreset-size 10 --rho-group-rollouts 3 --rho-candidates 3 \
  --tasks 42 --max-workers 6 --isolation process \
  --analyzer-workers 6 --capture-logs \
  --trace-root data/traces/rho-$(date +%Y%m%d-%H%M) \
  --export-harness data/harnesses/rho-$(date +%Y%m%d-%H%M)/
```

---

## 10. Error handling

| Failure | Behavior |
| --- | --- |
| Stale-format history trace | rejected with named error and count; run continues on valid records |
| Zero valid history records | cold start, `selection_method` recorded |
| Trajectory comprehension call fails | record excluded from difficulty-weighted selection and counted; never falls back to embedding a raw trace |
| Embedding provider down | profile-permitted lexical fallback, `fallback_reason` recorded; never a silent zero vector |
| Difficulty judge returns invalid JSON | record excluded from difficulty-weighted selection, counted in the report |
| Optimizer invocation stages nothing | no candidate produced; counted, not treated as a no-op edit |
| Proposal returns a no-op or duplicate | candidate discarded; `candidates_requested` vs `candidates_distinct` reported |
| Unmappable artifact id from optimizer | rejected at `register_candidate` (fails loudly, never a silent no-op rollout) |
| Preference judge unparseable | verdict unavailable, excluded from the average, not a tie |
| All candidates discarded | round rejects and reports why; pool unchanged |
| Rollout non-answer | existing `core/non_answer.py` classification; excluded from the scored denominator and surfaced in the summary |
| `--max-workers > groups * rollouts` | preflight rejects as a configuration error; never silently clamped |
| Rollout concurrency > 1 without process isolation | refused by `require_safe_rollout_concurrency` before any expensive construction |
| One rollout in a group fails | recorded as a traceless outcome; the group is diagnosed on its surviving rollouts, with the reduced `G` recorded |
| Every rollout in a group fails | group excluded from the diagnosis bundle and counted in the report |
| Cache key hit on a changed trace | impossible by construction: keys include the trace content hash |

---

## 11. Testing

Tests precede implementation. Interface A calls take an injectable
`completion_fn`; Interface B calls take an injectable agent factory. The full
suite runs offline with `FakeAdapter`, a fake workspace agent, and recorded
fixtures.

Pinned behaviors:

1. History loader rejects stale-format traces and accepts current-format ones
2. Cold start selects a coreset with no history and records `selection_method`
3. Trajectory comprehension produces a summary containing no UUIDs, event ids,
   hashes, or JSON keys (asserted against the real 9,610-byte fixture trace)
4. A failed comprehension call excludes the record and is counted, and never
   falls back to embedding a raw trace
5. Difficulty judge parses valid JSON, rejects out-of-range scores, and caches
   by content hash
6. Fingerprints carry no task-specific identifiers (asserted against a fixture)
7. Coreset DPP prefers high-difficulty and penalizes near-duplicate
   fingerprints; `difficulty_rank` and `random` modes behave as specified
8. Coreset DPP over comprehended summaries discriminates more than over raw
   traces on the same fixtures (pins the §4.2 dilution rationale)
9. Group diagnosis is **one** workspace-agent invocation per task covering both
   self-validation and self-consistency, not two
10. Candidate proposal issues **N independent** workspace-agent invocations, each
    with its own workspace
11. Candidates are captured from staged artifacts; an invocation that stages
    nothing yields no candidate
12. No-op and duplicate candidates are discarded; `candidates_distinct` is
    reported
13. Every proposed artifact id maps to a `_harness_slot`; an unmappable id raises
    at registration
14. All N candidates are retained in the pool (no best-of-N pruning)
15. Export and reload round-trip for every pool candidate; exported harness
    re-runs via `--harness PATH`
16. Preference judge returns a **signed** score and handles GT-present and
    GT-absent splits, never treating `(?i)\?` as ground truth
17. Unavailable verdicts are excluded rather than counted as ties
18. Mode dispatch runs the correct phase sequence for `rho`, `genetic`, and
    `rho-genetic`; `genetic` is byte-identical in behavior to today
19. Entropy no longer skips cells below the evidence floor
20. Contamination detector flags a planted GT literal in an artifact
21. Preflight rejects `--max-workers > --rho-group-workers * --rho-rollout-workers`
22. Group-major admission: a group's `G` rollouts complete together, and its
    diagnosis is dispatched before later groups finish
23. Global cap is never exceeded across concurrent groups (asserted with a
    counting fake executor)
24. A partially failed group is diagnosed on surviving rollouts with the reduced
    `G` recorded; a fully failed group is excluded and counted
25. Summary, difficulty, and embedding caches hit on identical content and miss
    when the trace content hash changes; hit counts appear in the summary

Every citable command is captured with
`2>&1 | tee terminal_output/<topic>/<name>.log`.

---

## 12. Sequencing note

RHO code can be built and tested offline now. A *meaningful* RHO run needs a
fresh current-format trace corpus, because the existing 42-task traces are
stale-format. Order of operations: implement and test RHO offline, collect the
fresh 42-task baseline, then run `rho-genetic` against real history.

---

## 13. Out of scope

- CUGA checkpoint replay and counterfactual replay (adapter reports no valid
  checkpoint capability)
- The full `DynamicAgentGraph` / `PlanControllerAgent` server path; this design
  stays on SDK Option A
- Wiring `cuga_proxy_validator` into validation
- Predicate-measurability guard
- Persisting attempt records by default
- Semantic result search or a storage tool (0 truncated tool observations
  observed in 240 traces; no evidence of need)
- The entropy between/within decomposition (recorded in §6.1 as future work)
