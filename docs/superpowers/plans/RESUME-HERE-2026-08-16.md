# RESUME HERE — AgentEvolve Session Handoff (2026-08-16)

**Read this file first. It is the compaction-survival document.**
Long-form detail lives in `docs/superpowers/plans/2026-08-15-rho-gepa-agreed-plan-and-findings.md`
(1161 lines — do NOT reload it wholesale; grep or `ctx_search` it by topic).

---

## 0. STATE

* Branch `dev4`, HEAD **`39a5295 "conplete evolve wiring fix1"`**, **worktree CLEAN**.
* Suite: **1220 passed, 1 skipped, 0 failed** (`uv run pytest -p no:randomly`; also green random-order).
* Everything below is committed. Never `git commit` without explicit user approval.

## 1. THE IMMEDIATE TASK (user's last instruction)

> "we should capture and save all the logs and traces (even for the workers, judge,
> editor and whole evolve pipeline too), and keep these logging as configurable
> (enable/disable)"

**Not started.** Requirements:
* Capture logs from: **worker subprocesses** (currently LOST — see §5.1), analyzer/judge,
  editor, and the whole evolve pipeline.
* Persist alongside traces.
* **Configurable enable/disable.**
* Follow the existing convention: `2>&1 | tee terminal_output/<topic>/<name>.log`.
* Config precedent: `ResolvedConfig` in `core/config.py` (see `max_analyzer_workers`),
  validated in `_POSITIVE_INT_FIELDS` / `_FLOAT_UNIT_FIELDS`, serialized via
  `manifest_payload`, and whitelisted in `_VALID_OVERRIDES`.

## 2. WORKING AGREEMENTS (hold across compaction)

* **Speed matters** — user is time-pressed. Do independent work in PARALLEL subagents.
* **Do NOT compromise testing that affects the research number** (the self-improvement
  delta). MAY relax rigor on things that don't affect outcomes (exhaustive sanitization,
  cosmetic validation).
* TDD: failing test first. Report negative results. Never upgrade "worked once" into "works".
* `uv run pytest` / `uv run python`. System Python lacks deps.
* `src/agent_evolve/core/` must NEVER import `cuga` or `adapters`. Composition happens in
  `src/agent_evolve/pipeline.py`.
* Models: `openai/azure/gpt-5.6-luna` (dev), `gpt-5.6-terra` (expensive ablations only).
* macOS: no `timeout` command. `rg` is gitignore-aware and skips `.venv` — use `grep -r`
  or `rg --no-ignore` for SDK inspection. Quote globs in zsh (`--include="*.py"`).
* **STOP before implementing the RHO seeder** — user is supplying context files.

## 3. HARD-WON EMPIRICAL FACTS — DO NOT RE-LITIGATE

### 3.1 Endpoint sampling (`gpt-5.6-luna`)
| probe | result |
|---|---|
| `temperature=0.0` (any non-default) | **REJECTED**, BadRequestError |
| identical prompt sequential ×3 | **1 distinct of 3** (cached) |
| **`n=3` in ONE request** | **3 distinct of 3** |
| `n=3` repeated later | same triple (cached) |

**Never pass `temperature`. `n=k` in a single request is the ONLY variance source**
(user chose this: "option A"). Repeating an identical A/B re-reads trial 1 — two verdicts
over the same `(call, substitution, k)` are ONE observation. Confirmed on a 36.5k-char
real prompt: 3/3 distinct per arm.

### 3.2 `CUGA_FOLDER` is NOT thread-safe (PROVEN, and undetectable)
Two threads binding two workspaces: the one that bound `h0` was observed reading `h1`
during `invoke()`. A build lock cannot help (read happens after construction).
**The trace's `harness_version` CANNOT detect the swap** — it is copied from the config,
so a contaminated run looks clean while measuring a harness that never existed.
=> **Real parallel rollouts MUST use process isolation** (`benchmarks/cuga_process_pool.py`).
Threaded `--execute` is refused. Analyzer/proxy fan-out (pure LLM, no CUGA) may thread.
Per-worker `persist_dir` fixes the knowledge flock but is NOT sufficient alone.

### 3.3 Trace format (my earlier brief was WRONG; a subagent caught it)
* Blobs live in **`payloads/<sha256>.json`**, NOT `blobs/`.
* `messages_ref` blobs are **doubly nested** `[[msg, ...]]` (LangChain batches).
* **No event carries a model**; model is at trace top level only.
* start/end pairing is by `payload["run_id"]`, not event_id adjacency.
* `final_output` is in `causal-trace.json`, NOT `manifest.json`.

### 3.4 Autonomous mode IS already correct (user asked; verified)
`DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true` and
`..._CUGA_LITE_NL_AUTO_CONTINUE=true` are in `.env` and **effective in-process AND in
worker children** (only `CUGA_FOLDER` is stripped; `prepare_environment()` runs before
the cuga import). The earlier `run2.py` truncation bug is NOT recurring.

## 4. WHAT IS BUILT (all committed, all tested)

| component | file | key evidence |
|---|---|---|
| parallel analyzer fan-out | `core/parallel_analysis.py` | one analyzer per worker THREAD (CUGA agents are stateful); takes a **factory**, not an instance |
| evidence bridge + shared guard | `core/evidence.py` | `rollout_group_report(task, traces)`; guard shared with editor |
| **LLM trajectory analyzer** | `adapters/cuga_analyzer.py` | `CugaTrajectoryAnalyzer.factory(...)`; **live: 3 distinct causal mechanisms, max pairwise Jaccard 0.16**; honest abstention |
| single-call replay | `cuga_wrapper/__init__.py` | `list_recorded_llm_calls`, `load_recorded_call`, `replay_single_llm_call`; **7/7 events**; offline proof passes with sockets blocked |
| **k=3 A/B proxy validator** | `adapters/cuga_proxy_validator.py` | `run_proxy_ab(...)`; live 0.333 vs 1.000, delta +0.667; `evidence_kind` pinned to `"proxy"` |
| finding↔analysis bridge | `core/blame.py` | `analysis_from_finding(finding, *, score)` — score MUST come from caller |
| dual-protocol shim | `core/analyzer.py` | `is_report_analyzer`, `as_legacy_analyzer`, `analyze_groups(...)` |
| benchmark layer | `benchmarks/{base,gaia}.py` | `expected_regex` (PRIMARY) + `recorded_llm_verdict` (replay-only, refuses new answers) |
| parallel benchmark runner | `benchmarks/runner.py` | `run_benchmark(...)`, input-ordered, failure-as-data, per-task timeout |
| CUGA executor | `benchmarks/cuga_executor.py` | `HarnessVersion.resolve`, `make_cuga_executor_factory`; trace capture ENFORCED |
| process pool | `benchmarks/cuga_process_pool.py` | NDJSON protocol; cwd deliberately NOT moved |
| **composition root** | `pipeline.py` | `build_offline_stack` / `build_live_stack` → `EvolutionStack` |
| CLI | `scripts/run_evolution.py` | `--dry-run` proves full lifecycle offline |

Verification scripts: `scripts/verify_replay_offline.py`, `verify_analyzer_live.py`,
`verify_proxy_validator_live.py`, `inspect_benchmark.py`, `run_benchmark.py`,
`verify_editor_rigorous.py`.

## 5. TWO BUGS FIXED THIS SESSION (both would have destroyed the research number)

### 5.1 `weighted_net_gain` made evolution MATHEMATICALLY INERT
`core/editor.py` weighted REGRESSION at `-1.0 * score`. Real producers set
`passed = score >= 0.5` with `score` = task score, so a **PASSING** probe subtracted 1.0:
```
origin pass + 1 PASSING regression probe -> gain=0.00 accepted=False
origin pass + 2 PASSING regression probes -> gain=-1.00 accepted=False
```
A perfect edit was REJECTED once one regression probe existed. At ≥2 tasks **no edit could
ever be accepted** — every delta would be exactly 0.0 for arithmetic reasons.
**Fixed:** charge only FAILED probes, proportional to shortfall (`1 - score`).
Three tests encoded the inverted reading and were changed with justification in-docstring.

### 5.2 Knowledge "seeding" claim was FALSE (user caught this)
I reported seeding restored 0/3 → 3/4. User asked what it was seeded with. It was two
leftover smoke fixtures: `favorite-color.md` ("blue") and `project-clearance-code.md`.
**Neither can help answer a Gaia question**, so the causal story was wrong — that was
noise at n=3–4. Worse, seeding from it is **contamination**. Default is now a clean EMPTY
store; requirement is only that **both arms use an IDENTICAL store**.

## 6. THE MEASUREMENT PROBLEM (blocks any credible delta)

### 6.1 Noise floor
* Historical, 42 tasks, two identical baseline runs: `expected_regex` **10/42 vs 17/42 =
  16.67 pp spread**.
* **Live tiny5, 3 identical runs: 60% / 20% / 40% — range 40 pp** (n=5, so 1 task = 20 pp).
* `recorded_llm_verdict` 6/22 vs 18/42 is **NOT comparable** (partial denominator; one eval
  batch died of a URLError). Tooling now refuses that delta.
* Grader agreement: **96.88%** (62/64 paired) — differences are coverage, not disagreement.
* **tiny5 is a plumbing smoke test ONLY, never evidence.**

### 6.2 Why runs fail (2 distinct mechanisms — my first single-cause claim was WRONG)
The autonomous loop IS working (mean 3.9 `call_model` cycles; one run 12 cycles/10 sandbox).
1. **Give-up after tool failure** — 6/16 rollouts. Example after 10 model cycles + 8 sandbox
   executions: *"The failure was only a missing import. I'm re-running... I'm sorry, but I
   can't reliably determine"*. **This is the evolvable target.**
2. **Deterministic never-starts** — `gaia-ec09fa32`: 19 events, 1 model cycle, **0 sandbox**,
   byte-identical across 4 runs. No code ever emitted (prompt-wording failure, documented in
   `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md`).

Narration-vs-verdict separation was 6/0 PASS and 0/9 FAIL — **scoring a give-up or narration
as a "wrong answer" corrupts the analyzer's input.** Recording them distinctly is still needed.

## 7. OPEN QUESTIONS / NEXT STEPS

1. **[USER'S TASK] Configurable log+trace capture** for workers, judge, editor, pipeline.
2. **Worker CUGA logs are LOST** — child stderr is not captured, so `is_autonomous_subtask`
   and `Routing to:` never appear. This is why §6.2 needed a manual re-run. Overlaps (1).
3. **UNEXPLAINED: `PlanControllerAgent` appears in 0 of 15 rollouts**, but
   `cuga_lite_node.py:529-571` routes success there when `force_autonomous_mode=True`
   (which IS true). Either `_has_error` fires every time, or CugaLite terminates earlier.
   Decides whether the agent gets a planning pass at all.
4. **Non-answer detection**: record give-up/narration as `unscorable`, NOT wrong answer.
5. **RHO seeder — STOP, await user's context files.** Pool ALREADY supports N candidates
   (proved: `add_base` + 5 × `add_candidate` = 6 entries). At N=1 entropy
   (needs ≥3 comparable) and DPP diversity are **inert** — user has acknowledged this.
6. `cuga_proxy_validator` is built but **NOT wired** into validation.
7. Proxy predicate can be **structurally unmeasurable** and return `no_change` for ANY edit
   (measured: 0/6 completions had a code fence, so `calls_tool` could never fire). No guard yet.
8. One rollout per task — no G-group, so no within-candidate variance.
9. `benchmarks` abstraction is ~70% general: `score(task_id, answer: str)` **will break** on
   AppWorld (grades env state) and tau-2 (multi-turn + DB state). Deliberately not
   pre-generalized without a real schema.
10. A timed-out task **does not stop** (Python threads can't be killed) — verified an
    abandoned thread later finished and produced a discarded answer.

## 8. LIVE COMMANDS

```bash
# full evolution, live
uv run python scripts/run_evolution.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --harness vanilla \
  --tasks 42 --iterations 3 --max-workers 6 --isolation process --analyzer-workers 6

# offline lifecycle proof (no CUGA, no network)
uv run python scripts/run_evolution.py --dry-run --tasks 3 --iterations 1

# benchmark replay (MUST reproduce 17/42 = 40.48%)
uv run python scripts/run_benchmark.py \
  --dataset datasets/gaia/gaia_l1_validation__baseline__20260813_035541 \
  --grader expected_regex --replay --max-workers 10

# real rollouts on tiny5 (5 tasks, ~50-200s)
uv run python scripts/run_benchmark.py \
  --dataset datasets/gaia/gaia_l1_validation_tiny5__baseline__20260812_180239 \
  --grader expected_regex --execute --harness vanilla \
  --isolation process --max-workers 5 --trace-root data/traces/<name>
```

Live trace evidence already on disk: `data/traces/tiny5-baseline-{a,b,c}/`,
`data/traces/tiny5-autoprobe/`, reference trace
`data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781/` (7 llm_call_start events).

## 9. KEY REFERENCE DOCS

* `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md` — 1020 lines of
  verified CUGA behavior. **Consult before any CUGA debugging.** Covers: autonomous mode,
  skills needing `enable_shell_tool`, `always: true` playbooks never matching, one-fenced-
  block-per-turn, `cuga_folder=None` loading stale skills, prompt-wording determinism.
* `reference/cuga_example_wrapper/run2.py` — the working reference wrapper.
* `docs/superpowers/plans/2026-08-15-rho-gepa-agreed-plan-and-findings.md` — full detail,
  §10 parallel analysis, §12 benchmarks/noise floor, §14 parallel execution, §15 pipeline,
  §16–17 live tiny5 results + corrections.
* `AGENTS.md` — boundaries (core is agent-neutral; no invented CUGA APIs).
