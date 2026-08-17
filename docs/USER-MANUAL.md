# AgentEvolve User Manual

How to run inference (benchmark) and evolution, every configuration flag, and
exactly where each artifact lands on disk.

Verified against branch `dev4`, `cuga==0.3.1`, model `azure/gpt-5.6-luna` via a
LiteLLM OpenAI-compatible endpoint, on 2026-08-16. Flag lists are transcribed
from `--help` output, not from memory. Re-check with `--help` after any CLI
change.

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

### Scale

- `--tasks N` — default 3
- `--iterations N` — outer iterations, one GEPA attempt each; default 1
- `--max-workers N` — rollout concurrency, default 1. Above 1 **requires
  `--isolation process`**.
- `--analyzer-workers N` — analyzer fan-out, default 1. Threads are safe here:
  pure LLM calls, no CUGA process.
- `--isolation {thread,process}` — default `thread`
- `--task-timeout SECONDS` — default `1200.0`

### Research config

- `--profile NAME` — default `research_sequential`. Also `minimal`,
  `research_parallel`, `full_ablation`.
- `--seed N` — RNG seed for parent sampling and DPP

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
