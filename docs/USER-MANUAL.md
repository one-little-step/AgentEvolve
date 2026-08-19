# AgentEvolve User Manual

How to run inference (benchmark) and evolution, every configuration flag, and
exactly where each artifact lands on disk.

Verified against branch `dev5`, `cuga==0.2.20`, models `azure/gpt-5.6-luna`
(CUGA / Interface B) and `gcp/gemini-3.6-flash` (Interface A) via a LiteLLM
OpenAI-compatible endpoint, on 2026-08-17. Flag lists are transcribed from
`--help` output, not from memory: all 76 flags are documented and verified
present. Re-check with `--help` after any CLI change.

> **Read `docs/OPEN-ISSUES.md` before trusting any measurement.** Seven of the
> eleven budget caps currently do nothing, `--max-rollouts` crashes instead of
> stopping cleanly, and candidate rollouts are stamped `harness_version: base`.
> Each entry there states how it was observed.

> **Read `docs/superpowers/plans/RESUME-HERE-2026-08-16.md` first** if you are
> picking this project up cold. This manual is operational; that file carries the
> research state and the open decisions.

---

## 0. Two things to know before your first run

**Every number currently on record is stale.** The `vanilla` harness gained an
`instructions` artifact (§6.1), which changes the rollout prompt, and the recorded
42-task dataset is in a pre-causal-tracing trace format the analyzer cannot read
(§7.4). Re-collect a baseline before comparing anything.

**Evolution does not persist attempt records by default.** `build_live_stack` and
`build_offline_stack` take `storage=None`, and `SequentialGepaRunner._record`
returns early when storage is `None` (`core/orchestrator.py:1992`). Neither CLI
passes a storage backend, so **no `attempts` records are written by
`run_evolution.py` today**. Traces and captured logs are still written. See §7.6.

**Always pass `--export-harness`.** Without it the evolved harness is destroyed at
process exit and the run's only output is a number on stdout — unreproducible and
unshippable. See §5.5.

---

## 1. Prerequisites

```bash
uv sync                        # install, including the pinned cuga==0.3.1
uv run pytest -q               # expect: 1343 passed, 1 skipped
```

`.env` at the repo root is loaded by the wrapper before CUGA is imported. Do not
rely on `printenv` to check it — it is read inside the Python process, not
exported to your shell.

| Variable | Purpose | Required |
| --- | --- | --- |
| `LITELLM_MODEL` / `CUGA_MODEL` | rollout + analyzer + editor model | **yes** for live runs |
| `LITELLM_BASE_URL` / `CUGA_BASE_URL` | endpoint | **yes** |
| `LITELLM_API_KEY` / `CUGA_API_KEY` | credential | **yes** |
| `OLLAMA_EMBEDDING_URL` | mechanism clustering embeddings | falls back to lexical |
| `OLLAMA_EMBEDDING_MODEL` | e.g. `embeddinggemma` | falls back to lexical |
| `DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true` | keeps completed substeps in the loop instead of finalizing early | yes |
| `DYNACONF_SKILLS__ENABLED=true` | enables the skills surface | yes, if using skills |
| `DYNACONF_ADVANCED_FEATURES__ENABLE_SHELL_TOOL=true` | **without this CUGA silently discards the entire skills prompt block** | yes, if using skills |
| `DYNACONF_KNOWLEDGE__ENABLED=true` | enables `agent.knowledge.ingest/search` | yes, if using memory |

Absent model config fails fast with `RuntimeError: CUGA_MODEL or LITELLM_MODEL is
required for a live inference run`. Absent credentials surface as
`openai.OpenAIError: Missing credentials`.

---

## 2. Quick start

```bash
# 1. Offline lifecycle proof: no CUGA, no network, no dataset, no cost.
uv run python scripts/run_evolution.py --dry-run --tasks 3 --iterations 1

# 2. Replay a recorded benchmark: no model calls, reproduces a known pass rate.
uv run python scripts/run_benchmark.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --replay --max-workers 10

# 3. Real rollouts on 5 tasks (~50-200s, costs tokens).
uv run python scripts/run_benchmark.py \
  --dataset datasets/gaia/gaia_l1_validation_tiny5__baseline__20260812_180239 \
  --grader expected_regex --execute --harness vanilla \
  --isolation process --max-workers 5 \
  --trace-root data/traces/tiny5-$(date +%Y%m%d-%H%M) --capture-logs
```

Always tee a run you intend to cite:

```bash
... 2>&1 | tee terminal_output/<topic>/<name>.log
```

---

## 3. Inference / benchmark — `scripts/run_benchmark.py`

Three mutually-exclusive execution modes:

| Mode | Flag | Cost | Use |
| --- | --- | --- | --- |
| Replay | `--replay` | free | reproduce a recorded run's pass rate |
| Real | `--execute --harness X` | tokens + time | measure a harness |
| Custom | `--executor mod:factory` | depends | your own callable |

### Required

- `--dataset PATH` — a benchmark run directory (§7.3)
- `--grader NAME` — `expected_regex` or `recorded_llm_verdict`. Never defaulted:
  two graders on the same benchmark disagree.

### Execution and isolation

- `--execute` — real CUGA rollouts, causal tracing mandatory
- `--harness NAME|PATH` — **required with `--execute`**; `vanilla` or a JSON file (§6)
- `--isolation {thread,process}` — default `thread`, safe only at `--max-workers 1`
- `--max-workers N` — default 10 for replay/executor, **1** for `--execute`
- `--allow-unsafe-concurrency` — threaded parallel real rollouts. **Tasks WILL be
  lost** to CUGA's knowledge lock and candidates can silently swap workspaces via
  the process-global `CUGA_FOLDER`. Only for experiments that accept corrupt evidence.
- `--task-timeout SECONDS` — omit for none; recorded baseline used `1200`
- `--limit N` — first N tasks only

### Worker stores

- `--worker-root PATH` — default `data/workers`
- `--empty-worker-knowledge` — start workers with an empty knowledge store.
  Measured to change the pass rate (3/4 serial vs 0/3 empty on tiny5, at one
  worker as well as four), so results are **not comparable** to a serial run.

### Output

- `--trace-root PATH` — default `data/traces`
- `--capture-logs` — capture each worker's CUGA stderr. **Only channel where CUGA
  reports `is_autonomous_subtask` and `Routing to:`**; discarded by default.
- `--log-root PATH` — default `<trace-root>/logs`
- `--progress`, `--verbose`

> `--capture-logs` on `run_benchmark.py` only takes effect with
> `--isolation process`. Threaded runs construct no pool, so there is no child
> stderr to capture; the flag parses and writes nothing.

---

## 4. Evolution — `scripts/run_evolution.py`

Runs rollout → analyze → select → edit → validate → record.

### Mode

- `--dry-run` — fake stack, offline. No CUGA, no endpoint, no network, no dataset.
- Live requires `--dataset`, `--grader`, **and `--harness`**.
- `--mode {genetic,rho,rho-genetic}` — which phases run. Default `genetic` is the
  existing loop, unchanged. See §4a for `rho` and `rho-genetic`.

### Scale

- `--tasks N` — default 3
- `--iterations N` — outer iterations, one GEPA attempt each; default 1
- `--max-workers N` — rollout concurrency, default 1. Above 1 **requires
  `--isolation process`**.
- `--analyzer-workers N` — analyzer fan-out, default 1. Threads are safe here:
  pure LLM calls, no CUGA process.
- `--isolation {thread,process}` — default `thread`
- `--task-timeout SECONDS` — default `1200.0`

### What ends the loop

**Only the iteration count you pass.** `run_iterations` is a bounded `for` with a
single exit; there is no convergence, plateau, or patience check anywhere in the
codebase. `--iterations 10` on a run that stopped improving at iteration 2 still
executes all 10 and still bills for them.

The complete list of things that end a genetic run:

| Ends the run | How |
| --- | --- |
| `--iterations N` (or `--genetic-iterations-per-round N` in `rho-genetic`) | loop bound |
| `--max-attempts` / `--max-accepted-edits` | iterations still run but issue no attempt; line reads `BUDGET EXHAUSTED (no attempt issued)` |
| `--max-rollouts` | raises `BudgetExceededError` — crashes, see OPEN-ISSUES S3-1 |
| an unhandled exception | crash |

Not on that list, deliberately or otherwise: zero accepted edits, an empty issue
set, or a plateaued score. `no_issue=1` iterations keep looping.

`RetryBudget` (3 attempts per `issue × artifact_group × lineage`) does **not** end
the loop — it skips that scope and moves on. Its `reset()` is never called, so an
exhausted scope stays exhausted for the whole run; see OPEN-ISSUES S1-2.

### Research config

- `--profile NAME` — default `research_sequential`. Also `minimal`,
  `research_parallel`, `full_ablation`.
- `--seed N` — RNG seed for parent sampling and DPP

### Budgets (spend caps)

**Every cap is UNLIMITED by default.** Before these existed a run had no
reachable ceiling: the loop issued rollouts and editor calls until the dataset
ran out. Set at least one on any run you are not watching.

A cap is a **run total, not per iteration** — `--max-attempts 1 --iterations 3`
issues one attempt, not three. Caps are checked *before* the work is issued, so a
cap refuses rather than abandoning a half-finished attempt, and the iteration
line says `BUDGET EXHAUSTED (no attempt issued)` rather than pretending nothing
was wrong.

| Flag | Caps | Status |
| --- | --- | --- |
| `--max-attempts N` | total edit attempts | **works** (clean stop) |
| `--max-accepted-edits N` | stop accepting after N edits | **works** (clean stop) |
| `--max-rollouts N` | total rollouts for the run | enforced but **crashes** — see OPEN-ISSUES S3-1 |
| `--max-editor-calls N` | editor-agent invocations | **no effect yet** (S2-1) |
| `--max-judge-verdicts N` | analyzer/judge calls | **no effect yet** (S2-1) |
| `--max-model-tokens N` | model tokens | **no effect yet** (S2-1) |
| `--max-wall-seconds S` | wall-clock seconds | **no effect yet** (S2-2) |
| `--max-pool-candidates N` | persistent pool size | **no effect yet** (S2-1) |
| `--max-history-records N` | edit-memory records shown to the editor | **no effect yet** (S2-1) |
| `--max-rag-context-tokens N` | retrieved-context tokens for the editor | **no effect yet** (S2-1) |
| `--edit-max-retries N` | retries per attempt | **works**; default **3**, the one non-unlimited default |

> **Only `--max-attempts` and `--max-accepted-edits` currently bound a run.** The
> others parse, validate, and appear in the manifest while bounding nothing —
> measured, see `docs/OPEN-ISSUES.md` S2-1. Do not leave a long run unattended
> relying on `--max-wall-seconds` or `--max-model-tokens`.

> `--max-pool-candidates` conflicts with RHO's all-N retention. RHO keeps every
> surviving candidate by design; capping the pool can refuse a retention the
> design requires. Leave it unset for RHO runs.

```bash
# a bounded exploratory run (using the two caps that actually work)
uv run python scripts/run_evolution.py --dry-run --tasks 5 --iterations 3 \
  --max-attempts 4 --max-accepted-edits 2
```

### Algorithm tuning

Each flag overrides one `ResolvedConfig` field. **Unset means the `--profile`
default stands**, so an existing command line resolves to exactly the config it
did before.

Coreset / issue selection (DPP):
`--dpp-max-items` (100), `--dpp-theta` (0.7, quality-vs-diversity in [0,1]),
`--dpp-score-floor` (0.1), `--dpp-min-gain` (1e-12).

Entropy-guided selection:
`--entropy-refresh-mode {outer_iteration,accepted_edits,pool_growth}`,
`--entropy-score-floor`, `--entropy-recombination-score-threshold`,
`--entropy-frontier-weight`, `--entropy-min-comparable-candidates` (3),
`--entropy-min-rollouts-per-candidate` (2).

> These two entropy floors are why `--rho-candidate-rollouts` defaults to 2 and
> `--rho-candidates` to 3. Set `--rho-candidates` below
> `--entropy-min-comparable-candidates`, or `--rho-candidate-rollouts` below
> `--entropy-min-rollouts-per-candidate`, and cross-candidate entropy stays
> **inert** — cells never reach the comparability floor, and selection silently
> falls back to score alone.

Mechanism clustering: `--cluster-similarity-threshold`, `--max-clusters-per-task`.

Validation probes: `--generalization-probe-mode {deferred,enabled}` (default
`deferred` records probes without spending rollouts), `--probe-budget-fraction`
(0.15).

Champion selection weights: `--champion-alpha` (0.55, mean score),
`--champion-beta` (0.20, coverage), `--champion-gamma` (0.15, worst case),
`--champion-delta` (0.10, novelty), `--champion-min-coverage-fraction` (0.0).

> Raise `--champion-min-coverage-fraction` to stop a candidate becoming champion
> on one lucky task. At `0.0` a candidate measured on a single task can outrank
> one measured on forty.

### Ablations (one gate at a time)

`--profile` sets five feature gates as a bundle. An ablation study needs to move
exactly one, so each gate has a tri-state pair; unset leaves the profile's value.

| Gate | Flags |
| --- | --- |
| causal blame graphs | `--enable-causal-blame` / `--disable-causal-blame` |
| edit memory | `--enable-edit-memory` / `--disable-edit-memory` |
| focused validation | `--enable-focused-validation` / `--disable-focused-validation` |
| entropy selection | `--enable-entropy-selection` / `--disable-entropy-selection` |
| parallel execution | `--enable-parallel-execution` / `--disable-parallel-execution` |

```bash
# research_sequential, but with causal blame OFF -- everything else unchanged
uv run python scripts/run_evolution.py --dry-run --tasks 5 \
  --profile research_sequential --disable-causal-blame
```

### Storage

- `--trace-root PATH` — default `data/traces`
- `--worker-root PATH` — default **`data/cuga-workers`** (note: differs from
  `run_benchmark.py`'s `data/workers`)
- `--seed-worker-knowledge PATH` — default is an **empty** store. This repo's
  `.cuga/knowledge` holds leftover fixtures unrelated to the benchmark, which
  would be contamination. **Both arms of any comparison must use the same choice.**

### Logging

- `--capture-logs` — off by default
- `--log-root PATH` — default `<trace-root>/logs`
- `--log-channels a,b,c` — default all four: `pipeline,workers,analyzer,editor`.
  Narrow to skip the expensive one: `workers` is per-rollout CUGA stderr, while
  debugging the editor needs only `editor,pipeline`.

### Full live example

```bash
uv run python scripts/run_evolution.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex \
  --harness vanilla \
  --tasks 42 --iterations 3 \
  --max-workers 6 --isolation process \
  --analyzer-workers 6 \
  --trace-root data/traces/evolve-$(date +%Y%m%d-%H%M) \
  --capture-logs \
  --export-harness data/harnesses/evolve-$(date +%Y%m%d-%H%M)/ \
  --profile research_sequential --seed 0 \
  2>&1 | tee terminal_output/evolution/run.log
```

---

## 4a. RHO — Retrospective Harness Optimization (`--mode rho`)

RHO reads a corpus of **past traces**, picks a difficult-and-diverse coreset,
diagnoses why the harness failed on those tasks, proposes N independent candidate
harnesses through CUGA workspace agents, and ranks them by pairwise preference.
All N survivors are retained in the pool as parents for the genetic loop.

### Modes

- `--mode rho` — RHO rounds only.
- `--mode rho-genetic` — each RHO round is followed by genetic iterations. The
  genetic phase runs on the **coreset tasks only**, not the full dataset (see
  "Why coreset-only" below).

### The knobs

| Flag | Meaning | Default |
| --- | --- | --- |
| `--rho-rounds N` | RHO rounds | 1 |
| `--rho-history PATH` | trace corpus to learn from | none → **cold start** |
| `--rho-coreset-size k` | tasks to diagnose and re-solve | 10 |
| `--rho-group-rollouts G` | baseline rollouts per coreset task | 3 |
| `--rho-candidates N` | independent candidate proposals | 3 |
| `--rho-candidate-rollouts R` | rollouts per candidate per task | 2 |
| `--rho-selector {dpp,difficulty_rank,random}` | coreset selection | `dpp` |
| `--rho-group-workers` | concurrent task groups | 4 |
| `--rho-rollout-workers` | concurrent rollouts within a group | 3 |
| `--genetic-iterations-per-round N` | genetic iterations per round (`rho-genetic`) | 1 |
| `--rho-summary-cache PATH` | cache: trajectory comprehension | off |
| `--rho-difficulty-cache PATH` | cache: difficulty/fingerprint verdicts | off |
| `--rho-embedding-cache PATH` | cache: fingerprint embeddings | off |
| `--rho-proposal-temperature T` | **ablation only**; `0.0` is refused | unset |

### Cost model

**Rollouts per round = `k × (G + N×R)`**

- paper defaults (k=10, G=3, N=3, R=2) → `10 × (3 + 6)` = **90** per round
- a cheap smoke config (k=3, G=2, N=3, R=2) → `3 × (2 + 6)` = **24**

Preference-judge comparisons per round = `N × k`. Multiply everything by
`--rho-rounds`. At `--rho-rounds 3` with paper defaults that is 270 rollouts
before any genetic work, which dominates the run.

If a candidate is discarded (see below), the round spends less than the formula
predicts: with 2 of 3 candidates surviving, `3 × (2 + 2×2)` = 18, not 24.

### Preflight invariant

```
--max-workers  <=  --rho-group-workers × --rho-rollout-workers
```

This is **refused, never clamped**, and is checked before any credential is
needed. Rollout concurrency above 1 also requires `--isolation process`.

### Cold start

Omit `--rho-history` and the run **cold-starts**: no usable corpus, so difficulty
judging is skipped, no coreset is selected, and no candidates are produced. The
run reports `cold start: no usable historical traces, RHO phases skipped` and
exits 0. That proves the plumbing, not the method. To get a corpus, run
`--mode genetic` first and point `--rho-history` at its `--trace-root`.

### Why coreset-only for the genetic phase

Cross-candidate variance needs a `(task, mechanism)` cell populated for several
candidates, and cells are created by rollouts. After a RHO round, cells exist only
for the k coreset tasks — on every other task variance is **undefined, not low**.
Running the genetic phase on all 42 tasks would cost `42 × 4 × 2` = 336 rollouts
per round instead of 90, for cells that cannot inform selection. The full dataset
is used exactly twice: as coreset-selection input, and for final champion
measurement.

### Reading the round line

```
round 1: 3 coreset tasks (dpp), candidates 3 of 3 distinct, pool 4
  rollouts=24 failures=0 diagnoses_observed=2 preferences=9 available / 0 unavailable
```

- `(dpp)` means the diversity term was live. **`(dpp_quality_only)` means it was
  not** — no embedder was available, so selection degraded to a plain difficulty
  ranking. Half the selection design is inert when you see that.
- `candidates 3 of 3 distinct` — all N retained. `pool 4` = base + 3.
- `diagnoses_observed=2` out of 3 coreset tasks means one diagnosis failed; an
  unobserved diagnosis never reaches the optimizer.
- `preferences ... unavailable` are **excluded** from the mean, never counted as
  ties.

Discarded candidates are named individually with a status:

| Status | Meaning |
| --- | --- |
| `NO_TOOL_CALL` | the agent narrated instead of executing a tool |
| `NO_OP` | it finalized with nothing staged |
| `NOT_FINALIZED` | it staged work but never called the submit tool |
| `IDENTICAL` / `DUPLICATE` | same artifacts as the base or another candidate |
| `UNAVAILABLE` | the invocation itself failed |

`candidates COLLAPSED to 0 of 3` means every proposal was discarded — the round
produced no candidate, and the pool is unchanged.

### Working live example

```bash
# 1. build a corpus
uv run python scripts/run_evolution.py \
  --dataset datasets/gaia/gaia_l1_validation_tiny5__baseline__20260812_180239 \
  --grader expected_regex --harness vanilla \
  --mode genetic --tasks 5 --iterations 1 \
  --max-workers 10 --isolation process \
  --trace-root data/live_traces/genetic

# 2. run RHO against it
uv run python scripts/run_evolution.py \
  --dataset datasets/gaia/gaia_l1_validation_tiny5__baseline__20260812_180239 \
  --grader expected_regex --harness vanilla \
  --mode rho --tasks 5 --rho-rounds 1 \
  --rho-history data/live_traces/genetic \
  --rho-coreset-size 3 --rho-group-rollouts 2 \
  --rho-candidates 3 --rho-candidate-rollouts 2 \
  --max-workers 10 --isolation process \
  --rho-group-workers 3 --rho-rollout-workers 4 \
  --trace-root data/live_traces/rho \
  --rho-embedding-cache data/live_cache/rho/embedding \
  --export-harness data/live_harnesses/rho \
  2>&1 | tee terminal_output/rho_live/rho/run-rho.log
```

Measured wall-clock for that config on 12 cores, 10 process workers: **~32 min**
for `rho`, **~34 min** for `rho-genetic`, **~9 min** for `genetic`. Most of it is
model latency (20-47 s per reasoning-model call), not compute.

### Known operational hazards

- **Stale knowledge lock.** A crashed run can leave `.cuga/knowledge/.lock`
  behind, after which every CUGA tool call fails with
  `AttributeError: 'NoneType' object has no attribute '_config'` and rollouts fail
  en masse. Fix: `rm -f .cuga/knowledge/.lock`.
- **`.env` is not auto-loaded early enough.** `RuntimeSettings.from_env()` runs
  before the dotenv load, so a live run needs the variables already exported:
  `set -a && . ./.env && set +a` before the command.
- **Two different models.** `LITELLM_MODEL` drives CUGA (Interface B); the
  Interface A calls resolve their own model from `RuntimeSettings`. They can be
  different models — check both before attributing behaviour to "the model".
- **Interface B tool use is prompt-sensitive.** Whether a workspace agent calls
  any tool is close to a deterministic function of prompt wording. If you edit
  `WORKSPACE_AGENT_TOOL_CONTRACT` or an agent's instructions, re-measure on a live
  round; a toy single-tool probe does **not** predict real-agent behaviour.

---

## 4b. Exporting and re-running an evolved harness

### 4b.1 Why this is mandatory

`CugaAdapter._workspaces` is an **in-memory dict** (`adapters/cuga_adapter.py:52`).
Candidates the editor produces exist only there. Without `--export-harness`,
`run_evolution.py` prints the champion's name and score, calls `stack.close()`, and
**the improved harness is gone**. `data/workspaces/` is transient CUGA scratch, not
persistence.

### 4b.2 Export

`--export-harness PATH` — off by default. The suffix selects the shape:

| PATH | Result |
| --- | --- |
| ends in `.json` | one file, champion only |
| anything else | a directory: `candidate-<id>.json` per pool member **plus** `champion.json` |

Directory mode is what you want with RHO: it captures the whole frontier, not just
the winner. The champion is duplicated into `champion.json` so you do not have to
re-derive selection.

Each file is a valid harness JSON (§6.2) plus a `provenance` block —
`candidate_id`, lineage (`parent_ids`, `ancestor_ids`, `attempt_ids`),
`scored_cells`, `mean_score`, `source_base_version`, `grader_name`, `task_ids`.
`HarnessVersion.from_path` ignores unknown keys, so provenance travels *inside* the
harness file rather than in a sibling that can get separated from it. Grader **name**
only — no payloads, regexes, or expected answers.

`version` is `evolved-<candidate_id>`, never derived from the filename.

An artifact with no CUGA harness slot is recorded under
`provenance.unexported_artifacts` rather than silently dropped or mislabelled as a
skill.

### 4b.2b Reading an exported harness

```bash
# one file: surfaces + provenance, explained
uv run python scripts/read_harness.py data/live_harnesses/rho/champion.json

# what actually changed vs the base
uv run python scripts/read_harness.py data/live_harnesses/rho/champion.json \
  --base data/live_harnesses/rho/candidate-base.json

# whole export dir: one row per candidate, base first
uv run python scripts/read_harness.py data/live_harnesses/rho --lineage

# every artifact body in full
uv run python scripts/read_harness.py <file> --full
```

**The schema.** Surfaces are implicit in the key names — there is no `artifacts`
wrapper:

| JSON key | Artifact id(s) | Shape |
| --- | --- | --- |
| `instructions` | `instructions` | scalar string |
| `skills` | `skills/<name>` | `{name: body}` |
| `policies` | `policies/<name>` | `{name: body}` |
| `memory` | `memory/<name>` | `{name: body}` |
| `version` | — | what the harness *claims to be*; stamped on every trace |
| `export_format` | — | `agent-evolve-harness-v1` |
| `provenance` | — | lineage + score record (below) |

An **absent key means that surface is empty**, not defaulted. `version` is
authoritative, not the filename: `HarnessVersion.from_path` refuses to infer a
version from a filename, so renaming a file cannot change what it claims to be.

Artifacts created by evolution are named `skills/generated-<name>`, so
`grep generated- ` on a harness tells you whether the run *created* anything or
only edited what already existed.

**`provenance` — what each field means.** Written by `harness_payload`
(`pipeline.py:1533`); ignored by the loader, so it travels with the file without
affecting `--harness`.

| Field | Meaning |
| --- | --- |
| `candidate_id` | identity in the persistent pool |
| `candidate_version` | the version rollouts actually ran under |
| `source_base_version` | the base this was derived from |
| `parent_ids` | immediate parent(s) edited from |
| `ancestor_ids` | full lineage back to the base |
| `origin_attempt_ids` | the attempt(s) that produced it |
| `attempt_ids` | attempts recorded *against* it |
| `is_base` / `is_champion` | whether it is the unmodified base / won champion selection |
| `mean_score` | mean over scored cells — **read with `scored_cells`** |
| `scored_cells` | how many `(task, mechanism)` cells were measured |
| `grader_name` | which grader produced those scores |
| `task_ids` | tasks in the run it was measured in |
| `unexported_artifacts` | artifact ids with **no CUGA harness slot** — kept verbatim so they are recoverable, but the agent will **not** load them |

> **`mean_score` without `scored_cells` is meaningless.** A candidate showing
> `mean_score: 1.00` on `scored_cells: 1` beat one lucky task; the base showing
> `0.75` on `4` cells is far better evidenced. The reader script prints a caveat
> when `scored_cells <= 3`. See also `--champion-min-coverage-fraction`, which at
> its default `0.0` permits exactly this.

**Only pool survivors are exported.** Discarded candidates (`NO_TOOL_CALL`,
`NO_OP`, `NOT_FINALIZED`, `IDENTICAL`, `DUPLICATE`) are named in the round output
but their artifacts are not persisted, so a failed proposal cannot be inspected
after the run.

### 4b.3 Re-run it as inference

`--harness PATH` already accepts an exported file, on either CLI:

```bash
# benchmark the evolved harness
uv run python scripts/run_benchmark.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex \
  --execute --harness data/harnesses/evolve-20260816-2130/champion.json \
  --isolation process --max-workers 6 \
  --trace-root data/traces/champion-rerun --capture-logs

# or evolve further from it
uv run python scripts/run_evolution.py \
  --dataset ... --grader expected_regex \
  --harness data/harnesses/evolve-20260816-2130/champion.json \
  --tasks 42 --iterations 3 \
  --export-harness data/harnesses/gen2/
```

Round-trip verified: an accepted edit survives export and reload. Diffing an
exported champion against the exported base showed exactly the edited artifact
differing.

> `data/harnesses/` is deliberately **not** gitignored — unlike traces and logs, an
> evolved harness is a *result*, and it is small. Do not add a blanket `data/`
> ignore rule; that would silently restore the defect this export exists to fix.

### 4b.4 What "nothing was accepted" means

A run that accepts nothing prints a diagnostic listing the real causes: no issue
attributed, editor declined, validation rejected on a genuine regression or a
protected-floor violation, or the retry budget was exhausted.

**Acceptance is reachable at any task count.** An older version of this warning
claimed evolution was "arithmetically inert above one task" — that was true of a
since-fixed `weighted_net_gain` bug and is now false. Measured on the current code:

| Scenario | `weighted_net_gain` |
| --- | --- |
| origin 1.0 + 0/1/2/3/5 **passing** regression probes | `+1.00` in every case |
| origin 1.0 + 1 **failing** probe, score 0.1 (collapsed) | `+0.10` |
| origin 1.0 + 1 **failing** probe, score 0.9 (mild dip) | `+0.90` |

Passing probes are free regardless of count; a failing probe is charged `1 - score`.
Pinned by tests so the false claim cannot return.

---

## 5. The four editable surfaces

What the editor may change, and how each reaches the model. Choosing the wrong
surface produces a valid-looking edit with **zero effect**.

| Surface | Artifact id | Delivery | Conditionality |
| --- | --- | --- | --- |
| Instructions | `instructions` | `special_instructions=` kwarg | **Unconditional, every turn.** Highest leverage |
| Skills | `skills/<name>` | `<ws>/skills/<name>/SKILL.md` | Model must call `load_skill` |
| Policies | `policies/<name>` | `<ws>/playbooks/<name>.md` | A trigger must match |
| Memory | `memory/<name>` | knowledge ingest | Must be retrieved |

Verified delivery mechanics and their traps:

- `instructions` **reaches the model** — confirmed with an unguessable marker
  (`scripts/verify_instructions_reach_model.py`).
- A skill's **description is the selection criterion**. Trigger-oriented ("Use
  when blame points at X") gets invoked; a passive title does not.
- A policy with only `always: true` **loads and then never matches** — no
  evaluator in cuga 0.3.1 selects an `AlwaysTrigger`. Use `natural_language` or
  `keywords`.
- Skills require `ENABLE_SHELL_TOOL=true` or the whole block is silently dropped.
- A skill **cannot** repair a failure that consists of the model not invoking
  tools: loading the skill itself requires a tool call. Circular.

Editor creation is capped and confined to `creatable_prefix = "skills/generated-"`.
`instructions` is replaced, not created.

### 5.5 The evolved harness must be exported or it is lost

Candidate artifacts live in an in-memory dict. See §4b — always pass
`--export-harness`.

---

## 6. Custom harness versions

### 6.1 Built-in: `vanilla`

`--harness vanilla`. Carries neutral `instructions` (`benchmarks/cuga_executor.py:477`,
`VANILLA_INSTRUCTIONS`), no skills/policies/memory, and the wrapper's default
5-tool set: `calculator`, `web_search`, `web_fetch`, `wikipedia_search`,
`save_note`.

The instructions are deliberately neutral — no mention of fenced blocks, code
execution, or tool-calling — because discovering that contract is what evolution
is being measured on. `test_vanilla_instructions_carry_no_code_execution_directive`
pins this; do not "helpfully" add a fence directive.

### 6.2 Custom harness JSON

`--harness path/to/harness.json`. `version` is **required** and never inferred
from the filename: it is stamped onto every trace and is how a result is
attributed later.

```json
{
  "version": "my-harness-v1",
  "instructions": "You are a question-answering agent. Answer concisely.",
  "skills": {
    "verify-with-source": "# Verify with a source\n\nUse when an answer needs a citation.\n\n1. Search for the primary source.\n2. Fetch it.\n3. Quote the figure."
  },
  "policies": {
    "cite-sources": "---\nname: cite-sources\nid: playbook_cite-sources\ntriggers:\n  natural_language:\n    - \"user asks a factual question needing a source\"\n  target: intent\n  threshold: 0.5\n---\nAlways name the source you used."
  },
  "memory": {
    "domain-notes": "Baseball-Reference lists regular-season batting separately from postseason."
  }
}
```

All of `instructions`, `skills`, `memory`, `policies` are optional. Frontmatter
for skills is generated by `materialize_harness`; for policies you supply it, and
**a colon in an unquoted YAML trigger phrase drops the whole policy silently** —
parse your generated frontmatter with `yaml.safe_load` in a test.

A harness with any of skills/policies/memory sets `requires_workspace = True`,
which forces per-task workspace materialization and constrains threading.

---

## 7. Where everything is stored

### 7.1 Traces — `<trace-root>/<run_id>/` (default `data/traces/`)

```
data/traces/<run_id>/
├── causal-trace.json     # full trace: events, tool_observations, final_output,
│                         #   status, harness_version, model, capabilities
├── manifest.json         # run metadata + file index + payload_level
├── events.jsonl          # one event per line, append order
├── graph-topology.json   # parent/child node structure
├── payloads/             # content-addressed blobs, <sha256>.json
└── checkpoints/          # checkpoint records, when captured
```

Gotchas, each of which has cost someone an hour:

- `final_output` is in **`causal-trace.json`**, not `manifest.json`.
- Payload blobs live in **`payloads/`**, not `blobs/`.
- `messages_ref` blobs are **doubly nested** `[[msg, ...]]` (LangChain batches).
- **No event carries a model name**; model is at trace top level only.
- start/end pairing is by `payload["run_id"]`, not event adjacency.
- Tool calls are in the top-level **`tool_observations`** array, and as
  `kind="tool_call"` events. `InvokeResult.tool_calls` from the SDK is
  unreliable — it has been observed empty while tools demonstrably executed.

Event kinds in the current format: `graph_node_start/end`, `llm_call_start/end`,
`graph_tool_start/end`, `graph_tool_error`, `tool_call`.

### 7.2 Captured logs — `<log-root>/<channel>/` (default `<trace-root>/logs/`)

Only written with `--capture-logs`. Off means **nothing written and no directory
created**.

```
<log-root>/
├── workers/<worker_id>.log        # raw CUGA stderr per worker (routing decisions)
├── analyzer/<candidate>__<task>.jsonl   # prompt + raw response + finding statuses
├── editor/<version>__<task>.jsonl       # prompt, answer, tool ledger, outcome
└── pipeline/iteration-<N>.jsonl         # iteration start/end, tallies, accept/reject
```

`data/logs/` is gitignored: these hold raw prompts and request bodies.

### 7.3 Datasets — `datasets/gaia/<run_name>/`

```
datasets/gaia/<run_name>/
├── config.json           # model, max_workers, task_timeout, ablation
├── result.json           # aggregate
├── tasks/<task_id>/
│   ├── result.json       # question, expected_regex, answer, tool_calls,
│   │                     #   direct_regex, llm_verdict, status
│   ├── cuga_trace.json   # the rollout's trace
│   ├── stdout.log
│   └── stderr.log
└── evaluations/
```

Available: three `gaia_l1_validation__baseline__*` (42 tasks) and one
`gaia_l1_validation_tiny5__baseline__*` (5 tasks). **tiny5 is a plumbing smoke
test only** — its observed spread is 40 pp (3/5 then 1/5), far wider than any
effect you would claim.

### 7.4 Which traces the analyzer can actually read

The recorded 42-task datasets predate the causal-tracing rewrite: their events are
8 undifferentiated `stream_event` entries with `actor_id=None` and **no
`tool_call` kind**. An analyzer reading one sees no actors and no tool vocabulary,
so it cannot produce a grounded mechanism. Verified: probing that format returned
a WEAK verdict; the same probe on a current-format trace passed.

**Evolution must run against freshly collected traces.**

### 7.5 Worker and workspace state (all gitignored, all regenerated)

| Path | Contents |
| --- | --- |
| `data/workers/<worker_id>/{knowledge,dbs}/` | `run_benchmark.py` per-worker stores |
| `data/cuga-workers/<worker_id>/...` | `run_evolution.py` per-worker stores |
| `data/workspaces/<version>/` | materialized `skills/`, `playbooks/` per candidate — **transient scratch, NOT the evolved harness** |
| `data/logs/` | captured run logs |
| `cuga_workspace/` | CUGA per-thread sandbox scratch |
| `.cuga/knowledge/`, `.cuga/skills/` | CUGA global state — a policy or skill from an earlier run **persists into later ones** |
| `terminal_output/` | tee'd command logs |

### 7.5b Exported harnesses — `data/harnesses/` (NOT gitignored)

The only durable record of what evolution produced. Written only with
`--export-harness` (§4b). Directory mode:

```
data/harnesses/<run>/
├── champion.json            # the selected champion, re-runnable via --harness
└── candidate-<id>.json      # one per pool member (the frontier)
```

Committed deliberately: an evolved harness is a result, and it is small.

### 7.6 Evolution attempt records — currently not written

`JSONFileStorage` (`core/storage.py:177`) writes
`<root>/<collection>/<record_id>.json` and would persist `attempts` records. But
both stack builders default `storage=None` and neither CLI supplies one, so
`_record` returns early. **Evidence of what the editor tried survives only in the
`editor`/`pipeline` log channels — so pass `--capture-logs` on any evolution run
you intend to analyze.**

---

## 8. Verification and diagnostic scripts

| Script | Purpose | Cost |
| --- | --- | --- |
| `verify_instructions_reach_model.py` | proves `instructions` reaches the model (unguessable marker) | 2 rollouts |
| `diagnose_tool_invocation.py` | unwilling vs unable: fence emission, sandbox routing, tool-body execution | ~6 rollouts |
| `probe_analyzer_editor_chain.py` | does the judge name an actionable mechanism on a real trace | 1 LLM call |
| `verify_cuga_tools.py` | five-tool live check | 2 rollouts |
| `run_vanilla_baseline.py` | baseline collection | 42 rollouts |
| `inspect_benchmark.py` | dataset structure, no model calls | free |
| `verify_replay_offline.py` | replay path, no model calls | free |

There are ~20 more `diagnose_*` / `verify_*` / `probe_*` scripts; each carries a
module docstring stating exactly what it measured and when.

---

## 9. Concurrency rules (measured, not theoretical)

1. **Real parallel rollouts require `--isolation process`.** `CUGA_FOLDER` is
   process-global; two threads binding two workspaces were observed reading each
   other's. A build lock cannot help — the read happens after construction.
2. **The trace cannot detect the swap.** `harness_version` is copied from config,
   so a contaminated run looks clean while measuring a harness that never existed.
3. **Analyzer fan-out may thread** (`--analyzer-workers`): pure LLM calls, no CUGA.
4. **A timed-out task does not stop.** Python threads cannot be killed; an
   abandoned thread was observed finishing later and producing a discarded answer.
5. Per-worker `persist_dir` fixes the knowledge `flock` but is **not sufficient
   alone**.

---

## 10. Reading a result honestly

`pass_rate` is computed over **scored** answers only. Non-answers (the agent
committed no answer) are excluded from the denominator and reported separately —
recording them as wrong inflates the denominator and understates the rate.

```
non-answer (gave up): 10
scored (DENOMINATOR): 32
passed              : 17
pass rate           : 17/32 = 53.12%
```

**Watch the non-answer count as a guardrail.** Excluding non-answers creates a way
to game the metric: an agent that learns to say "I'm unable to determine this" on
hard tasks removes them from the denominator and the pass rate climbs while the
agent gets worse. A rising non-answer count means the delta is an artifact.

Report both framings when citing a delta:

```
pass rate (full denominator)   17/42 = 40.48%
pass rate (committed answers)  17/32 = 53.13%
non-answers                    10        <- must not rise
```

The historical noise floor (16.67 pp) was computed on **inflated** denominators
and must be recomputed on whichever basis you report.

---

## 11. Failure modes worth recognising

| Symptom | Real cause | Where |
| --- | --- | --- |
| "I'm unable to call the tool" | **Often false.** Model emitted no fenced block; `extract_code_from_model_response` returned `""`, so the graph never reached the sandbox | `shared_nodes.py:233` |
| Skill loads but has no effect | `ENABLE_SHELL_TOOL=false` silently discards the skills block | `prompt_utils.py:682-689` |
| Policy loads but never fires | `always: true` is never selected by any evaluator | `policy/agent.py:929` |
| "Loaded 1 policies" but `list()` is empty | policy file missing frontmatter `id`; filesystem sync deletes it | `filesystem_sync.py` |
| Loads someone else's skills | `cuga_folder=None` falls back to `<cwd>/.cuga/skills` | `_construct_agent` |
| `tool_calls == []` but tools ran | SDK aggregation unreliable; trust `tool_observations` or a side effect | — |
| Candidates behave identically | `reset_policy_storage` not set; policies persist in the package DB | `_construct_agent` |
| `PlanControllerAgent` never appears | It does not exist in the SDK graph. Only the server graph registers it | `sdk.py:2144-2154` |

**Never diagnose tool execution with a guessable task.** A model asked to
"use the calculator to compute 1234*5678" will just do the arithmetic and answer
correctly without calling anything — `tool_calls: []` is then truthful reporting,
not a defect. Use an unguessable per-run token and treat the tool function body
executing as the only ground truth.

**Never repeat an identical prompt and call it N trials.** Reasoning models skip
temperature and decode greedily, so the same prompt gives byte-identical output.
Vary the prompt to get real samples.

---

## 12. Architecture in one paragraph

The SDK graph we run is
`CugaLiteSubgraph → prepare → call_model ⇄ sandbox → SDKCallback → FinalAnswerAgent`
— a single-agent ReAct/CodeAct loop, identical across all 104 traces inspected.
There is **no `PlanControllerAgent`** on this path; it exists only in CUGA's server
graph (`DynamicAgentGraph`, `graph.py:164`), which we do not construct. Do not
describe results as hierarchical planner/executor decomposition: the numbers are
valid but the prose would contradict the trace. `force_autonomous_mode` is
consumed for **prompt content**, not routing.
