# RESUME HERE — AgentEvolve Session Handoff (2026-08-16, evening)

Supersedes `RESUME-HERE-2026-08-16.md`. Written to survive context compaction.
Operational instructions live in `docs/USER-MANUAL.md`; this file is state,
evidence, and open decisions.

---

## 0. STATE

- Branch `dev4`, worktree **dirty, nothing committed** (user has not approved a commit)
- Suite: **1359 passed, 1 skipped, 0 failed** (was 1313 at session start)
- Next task: **RHO integration** — user said proceed. Previously blocked on user
  context files; user has released that block.

New files this session:
`src/agent_evolve/core/run_logging.py`, `src/agent_evolve/core/non_answer.py`,
`docs/USER-MANUAL.md`, `docs/superpowers/plans/2026-08-16-plan-controller-diagnosis.md`,
`scripts/{diagnose_tool_invocation,verify_instructions_reach_model,probe_analyzer_editor_chain}.py`,
`tests/test_{run_logging,non_answer}.py`

---

## 1. WORKING AGREEMENTS (hold across compaction)

- **No commits without explicit approval.**
- **No secret/credential redaction.** User cut it deliberately: research repo, no
  time. Do not re-add scrubbing/masking anywhere.
- Speed over cosmetic rigor; number-affecting correctness is never negotiable.
- Tee anything citable: `2>&1 | tee terminal_output/<topic>/<name>.log`
- Parallelize independent work (subagents), keep CPU/stateful work serial.
- `core/` must never import `cuga` or any adapter.
- Tests before implementation. Mutation-test the guarantees that matter.

---

## 2. WHAT WAS BUILT THIS SESSION (all tested)

1. **Configurable log capture** — `core/run_logging.py`: `LogCaptureConfig`,
   `RunLogSink`, `build_sinks`, channels `pipeline|workers|analyzer|editor`. Off by
   default; off means no file **and no directory**. Wired into `ResolvedConfig`
   (+`_VALID_OVERRIDES`, +`manifest_payload`), `CugaProcessPool` (fixed
   `stderr=DEVNULL`, the lost channel), analyzer, editor, pipeline, both CLIs
   (`--capture-logs/--log-root/--log-channels`).
2. **Non-answer detection** — `core/non_answer.py`, wired at
   `evaluation.py:_answer_or_reason` and `runner.py:_score_all`. Changed the
   reported baseline (§4).
3. **`vanilla` gained neutral `instructions`** (`cuga_executor.py:477`,
   `VANILLA_INSTRUCTIONS`) so the editor has a real lever.
4. **`list_rollout_tools`** editor tool, derived by introspection over
   `tools._RAW_TOOLS` (never a hardcoded list).
5. **Judge prompt upgraded** — real graph shape, the no-code pattern, an explicit
   rule that a model's self-report is a *claim not an observation*, actionable-
   mechanism requirement. Also now states the **literal `tool_call` count** per
   trace (LLMs count unreliably inside multi-KB JSON).
6. **Editor prompt upgraded** — all four surfaces with delivery mechanics and
   which is effective where, plus the silent-failure traps.
7. `.gitignore` += `data/logs/` (raw prompts). `data/harnesses/` deliberately NOT
   ignored — an evolved harness is a result.
8. **`--export-harness PATH`** on `run_evolution.py`. `.json` = champion only;
   anything else = directory with `candidate-<id>.json` per pool member +
   `champion.json`. Output is directly re-runnable via the existing `--harness PATH`
   on either CLI. Round-trip verified: an accepted edit survives export+reload.
   Provenance (lineage, scores, grader NAME only) rides inside the file —
   `from_path` ignores unknown keys. Artifacts with no CUGA slot go to
   `provenance.unexported_artifacts` rather than being dropped or mislabelled.
9. Fixed the **stale multi-task inertness warning** in `run_evolution.py` and
   widened its trigger to any task count (a single-task run accepting nothing was
   previously silent).

---

## 3. HARD-WON FACTS — DO NOT RE-LITIGATE

### 3.1 CUGA ships TWO graphs; the SDK gets the simplified one
`CugaAgent` → `sdk.py:2014 _create_hitl_wrapper_graph` → 5 real nodes + 3 dummy
stubs. **No `PlanControllerAgent`.** The server graph (`DynamicAgentGraph`,
`graph.py:63`, used by a local FastAPI app) registers it at `graph.py:164`.
`cuga_lite_node.py:529-571` routing applies **only** to the server graph — that
file is imported by `graph.py:46` only.

Observed shape, all 104 traces, identical:
`CugaLiteSubgraph → prepare → call_model ⇄ sandbox → SDKCallback → FinalAnswerAgent`

Both earlier hypotheses were **refuted**: `_has_error` never fires (0/16), nothing
terminates early (all reach `FinalAnswerAgent`). `DynamicAgentGraph` is prebuilt
and constructible in-process (`utils/controller.py:193`) — the migration cost is a
**different constructor surface**, not building anything. Decision: **Option A**
(stay on SDK). Option C (`enable_todos`/`reflection_enabled`, both false,
`cuga_lite_graph.py:173`) saved for later.

**Writeup constraint:** single-agent ReAct under autonomous prompting. NOT
hierarchical planner/executor.

### 3.2 The agent is UNWILLING, not unable (measured)
`scripts/diagnose_tool_invocation.py`, clean run:

| task | arm A baseline | arm B + fence directive |
| --- | --- | --- |
| `3f57289b` | 0 fences, **0 tools** | 2 fences, `web_search`+`web_fetch`, **correct (519)** |
| `5188369a` | 0 fences, **0 tools** | 2 fences, correct |
| `e142056d` | 1 fence, calculator | 1 fence, calculator |

Baseline arm A output was one turn: *"I'll verify the 1977 Yankees table..."* — a
plan, no fence. **0 tool calls across all 42 baseline tasks; 0 truncated tool
observations across 240 traces.** The model's *"I'm unable to retrieve the source
page"* is **FALSE**. Tools work.

Consequence: the semantic-search tool idea is **premature** — adding a 6th tool to
an agent calling zero tools cannot help. Park until truncation is evidenced.

### 3.3 `instructions` reaches the model (VERIFIED)
Unguessable marker present with the artifact, absent in control
(`verify_instructions_reach_model.py`). Path: `_harness_config` → `run_task` →
`_construct_agent(special_instructions=...)` (`cuga_wrapper/__init__.py:144`).

### 3.4 The judge works on real evidence (VERIFIED)
`probe_analyzer_editor_chain.py` on trace `0cb88c5a` (0 tool calls, false
inability claim) → PASS:

> *"call_model completed its turn without producing an executable code block that
> invoked a tool, so sandbox was never reached and FinalAnswerAgent had no
> tool-derived evidence before producing the terminal answer."*

Anchored to real event ids; did **not** repeat the false claim. **Editor half not
yet live-probed.**

### 3.5 The recorded 42-task datasets are in a STALE trace format
8 undifferentiated `stream_event` entries, `actor_id=None`, **no `tool_call`
kind**. The analyzer cannot reason about them (first probe scored WEAK purely
because of this). Current format: `graph_node_start/end`, `llm_call_start/end`,
`graph_tool_start/end`, `graph_tool_error`, `tool_call`, with real actor ids.
**Evolution requires freshly collected traces.**

### 3.6 `CUGA_FOLDER` is not thread-safe, and the trace cannot detect the swap
Real parallel rollouts **must** use `--isolation process`. `harness_version` is
copied from config, so a contaminated run looks clean. Analyzer fan-out may thread.

### 3.7 Endpoint sampling
`temperature=0.0` unsupported on `azure/gpt-5.6-luna`; identical prompts decode
greedily → byte-identical output. **Vary the prompt, not the trial index.** `n=3`
in one request is the usable variance source.

### 3.8 Trace format gotchas
`final_output` in `causal-trace.json` (not manifest); blobs in `payloads/` (not
`blobs/`); `messages_ref` doubly nested `[[msg,...]]`; no event carries a model;
start/end pair by `payload["run_id"]`; SDK `InvokeResult.tool_calls` can be empty
while tools ran — trust `tool_observations` or a side effect.

---

## 4. THE MEASUREMENT SITUATION

```
GAIA 42-task replay:  17/42 = 40.48%  →  17/32 = 53.13%
```
Numerator unchanged. 10 rollouts committing no answer were being counted wrong.
All 10 audited: none would have matched its `expected_regex`.

**Two live risks:**
1. **Goodhart.** Excluding non-answers means an agent that learns to say "I'm
   unable" removes hard tasks from the denominator and the rate climbs while it
   gets worse. **Report both framings and treat the non-answer count as a
   guardrail that must not rise.**
2. **The 16.67 pp noise floor was computed on inflated denominators** (10/42 →
   17/42) and gates whether any delta is signal. **Recompute on the reported basis.**

**Baseline is invalid for two independent reasons** — `vanilla` now sends
`instructions` (changed control arm) and the old traces are unreadable (§3.5).
User has accepted this: *"we will collect live traces later."*

---

## 5. KNOWN GAPS (each verified)

1. **Evolution persists no attempt records.** Both stack builders default
   `storage=None`; `orchestrator.py:1992` returns early. Neither CLI passes a
   backend. Evidence survives only in the `editor`/`pipeline` log channels →
   **always pass `--capture-logs` on evolution runs.** (Still open.)
1b. **Evolved harnesses were destroyed at exit — NOW FIXED** via
   `--export-harness` (§2.8). `_workspaces` is still in-memory; export is the
   persistence boundary. **A run without `--export-harness` still loses its
   harness**, so pass it on every evolution run.
2. `cuga_proxy_validator` built, **not wired** into validation.
3. No predicate-measurability guard: a proxy predicate can be structurally
   unmeasurable and return `no_change` for any edit (measured: 0/6 completions had
   a code fence, so `calls_tool` could never fire).
4. One rollout per task — no G-group, no within-candidate variance.
5. `benchmarks` abstraction ~70% general; `score(task_id, answer: str)` will break
   on AppWorld (grades env state) and tau-2 (multi-turn + DB state).
6. A timed-out task does not stop (Python threads can't be killed).
7. Editor tool coverage incomplete: `get_attempt_outcome`, `list_trace_actors`,
   `read_trace_events`, `unstage` never reached live. Do not claim all 16 verified.
8. At N=1 candidate, entropy (needs ≥3 comparable) and DPP diversity are **inert**.
   Pool already supports N (`add_base` + 5×`add_candidate` = 6).

---

## 6. THE FOUR EDIT SURFACES (what the editor can change)

| Surface | Delivery | Conditionality |
| --- | --- | --- |
| `instructions` | `special_instructions=` kwarg | **unconditional, every turn — highest leverage** |
| `skills/<n>` | `<ws>/skills/<n>/SKILL.md` | model must call `load_skill` |
| `policies/<n>` | `<ws>/playbooks/<n>.md` | a trigger must match |
| `memory/<n>` | knowledge ingest | must be retrieved |

Traps: `always: true`-only policy **never matches**; skill description is the
selection criterion; skills need `ENABLE_SHELL_TOOL=true` or the block is silently
dropped; **a skill cannot fix an inability to invoke tools (circular)**;
`cuga_folder=None` loads other people's skills.
`creatable_prefix="skills/generated-"`; `instructions` is replaced, not created.

---

## 7. ANTI-CONTAMINATION (pinned by tests — do not "helpfully" break)

- `test_vanilla_instructions_carry_no_code_execution_directive` — vanilla
  instructions must contain no fence/code directive.
- `test_surface_guidance_prescribes_no_specific_remedy` — editor prompts must not
  contain `add a fence directive`, `519`, `web_fetch`, `gaia`, etc.

Both mutation-verified. **The fence fix must be discovered by evolution, not
written by us** — that is the research result. The editor/judge are taught
*mechanisms and surfaces*, never the remedy.

---

## 8. NEXT STEPS (user's order)

0. `weighted_net_gain` **verified correct** — passing regression probes are free at
   N=0..5 (gain `+1.00`); a failing probe is charged `1 - score` (0.1→`+0.10`,
   0.9→`+0.90`). Multi-task runs are NOT inert. The old warning claiming otherwise
   was stale and has been rewritten. Do not re-introduce it.
1. **RHO integration** — user said proceed (this is the active task). Note export
   (§2.8) is a prerequisite: RHO seeds N candidates and directory-mode export is
   what captures the frontier.
2. Re-collect the 42-task baseline on the current harness + current trace format.
3. Live-probe the editor on trace `0cb88c5a` (~1 LLM call): does it pick
   `instructions` over a circular skill edit?
4. Recompute the noise floor on the reported denominator basis.
5. Wire `cuga_proxy_validator`; add the predicate-measurability guard.
6. Later: Option C (`enable_todos`/`reflection_enabled`); semantic-search tool
   **only if** truncation is ever evidenced.

---

## 9. LIVE COMMANDS

```bash
# offline lifecycle proof (free)
uv run python scripts/run_evolution.py --dry-run --tasks 3 --iterations 1

# replay (free) — was 17/42, now reports 17/32 = 53.13%
uv run python scripts/run_benchmark.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --replay --max-workers 10

# fresh baseline collection (42 rollouts)
uv run python scripts/run_benchmark.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --execute --harness vanilla \
  --isolation process --max-workers 6 --task-timeout 1200 \
  --trace-root data/traces/baseline-$(date +%Y%m%d-%H%M) --capture-logs

# full evolution -- ALWAYS with --capture-logs AND --export-harness
uv run python scripts/run_evolution.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --harness vanilla \
  --tasks 42 --iterations 3 --max-workers 6 --isolation process \
  --analyzer-workers 6 --capture-logs \
  --trace-root data/traces/evolve-$(date +%Y%m%d-%H%M) \
  --export-harness data/harnesses/evolve-$(date +%Y%m%d-%H%M)/

# re-run an evolved champion as inference
uv run python scripts/run_benchmark.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --execute \
  --harness data/harnesses/<run>/champion.json \
  --isolation process --max-workers 6 --capture-logs
```

---

## 10. KEY REFERENCE DOCS

- `docs/USER-MANUAL.md` — how to run everything, every flag, every storage path
- `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md` — CUGA SDK
  facts (~1180 lines; includes the two-graph section added this session)
- `docs/superpowers/plans/2026-08-16-plan-controller-diagnosis.md`
- `AGENTS.md` — non-negotiable boundaries
- `terminal_output/{tool_diagnosis,instructions_reach,chain_probe}/` — this
  session's live evidence
