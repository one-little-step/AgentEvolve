# RESUME HERE — AgentEvolve Session Handoff (2026-08-18, cache-fix session)

Supersedes `RESUME-HERE-2026-08-17-EVENING.md` for everything about sampling,
temperature, and rollout diversity. That file remains valid for RHO plumbing,
flags, and the harness reader.

Branch `dev5`. Suite **1757 passed, 1 skipped**. Nothing committed (user approval
required for every commit).

---

## 1. What this session actually established

### 1.1 The headline correction: it was a CACHE, not greedy decoding

The prior session recorded "reasoning models skip temperature, so decoding is
greedy, so N identical prompts are one sample repeated N times." **That was
wrong.** The real mechanism is an upstream response cache on the IBM
`ete-litellm` gateway.

Proof — four identical requests share ONE response `id`:

```text
                        luna                gemini-3.6-flash
default          distinct_ids=1  text=1     distinct_ids=1  text=1
extra_body off   distinct_ids=4  text=4     distinct_ids=4  text=4
```

A repeated response `id` means the request never reached the model. The gateway
also returns `x-litellm-cache-key`.

**Diagnostic rule: use the response `id`, NEVER text equality.** `"what is 2+2"`
gives `distinct_ids=4, distinct_text=1` — four real samples that legitimately
agree. Text equality cannot separate cache from agreement. This is the single
most reusable lesson here.

### 1.2 Temperature is per-model and NOT the lever

```text
model                  temp=0.0   temp=0.7   temp=1.0
azure/gpt-5.6-luna     HTTP 400   HTTP 400   accepted
gcp/gemini-3.6-flash   accepted   accepted   accepted
```

- On `luna`, temperature is unavailable at any non-default value. CUGA's
  `_is_reasoning_model` suppression (`cuga/backend/llm/models.py:973-985`, applied
  `:1302-1317`) is **load-bearing** — defeating it guarantees a 400.
- On `gemini`, the prefix test does NOT match, so CUGA already sends
  `temperature = 0.1` (from `configurations/models/settings.litellm.toml`, every
  role).
- Cache-off gemini sweep: 4/4 distinct at 0.0, 0.1, 1.0, 2.0. So temperature does
  not drive diversity; cache control does. `2.0` visibly degrades output into
  leaked scratchpad.

**Do not build a LiteLLM proxy to inject temperature.** The upstream rejects the
value regardless of who sends it. (A proxy for *request/response logging* is still
a good idea — see §5.)

### 1.3 The fix, and the only injection point that works

```text
extra_params={"caching": False}                      dropped before the wire
model_kwargs={"caching": False}                      reaches litellm, still DISTINCT=1
model_kwargs={"extra_body": {"caching": False}}      DISTINCT=4  <-- works
```

Why: CUGA merges `extra_params` into `litellm_params` (`models.py:1311`) but the
langchain client is a pydantic model with no `caching` field and no extras
(`has 'caching' field? False`, `__pydantic_extra__ None`). A bare `caching` kwarg
*does* reach litellm (wire-spy confirmed) but litellm consumes it as its own
client-side setting and never forwards it. Only `extra_body` travels in the HTTP
body where the gateway reads it.

Implemented in `src/agent_evolve/cuga_wrapper/__init__.py`:
- `CACHE_BYPASS_EXTRA_BODY` — sends both `caching: False` and
  `cache: {"no-cache": True}`.
- `apply_response_cache_policy(model, *, disable_cache)` — merges into any
  existing `extra_body`; never raises.
- `install_response_cache_policy(manager, *, disable_cache)` — patches
  `LLMManager._update_model_parameters`, **the only choke point covering all three
  `get_model` paths** (pre-instantiated, cache hit, fresh). Patching
  `_create_llm_instance` would miss every cache hit.
- `response_cache_disabled(default=True)` + `ALLOW_RESPONSE_CACHE_ENV`.

Env var, not a plain argument, because `--isolation process` rollout workers are
separate processes building their own wrapper (`cuga_process_pool.py:632`).

CLI: `--allow-response-cache` (OFF by default). Header prints
`response cache : disabled (each rollout is an independent sample)` or the ALLOWED
warning. 13 tests in `tests/test_cuga_response_cache.py`.

**Live-verified on the REAL path.** Live runs use `platform="openai"` →
`ReasoningChatOpenAI`, *not* litellm. Confirmed there: `DISTINCT=1` → `DISTINCT=4`
(`terminal_output/cache_probe/openai_path.log`).

### 1.4 `_CANDIDATE_FRAMINGS` is now empty — and why

User directed this. The framings' justifying comment cited greedy decoding (wrong
mechanism) and they were also a **confound**: they permanently biased which
surface each candidate considered first, so "candidate 3 created a skill" would be
a prompt artifact, not a finding.

Now `_CANDIDATE_FRAMINGS: tuple[str, ...] = ()` in `cuga_rho_optimizer.py`.

**Trap caught:** the user's first edit was `("")`, which is an empty *string*, so
`index % len(...)` raised `ZeroDivisionError` on the first proposal. Fixed by
making it a real empty tuple AND guarding `_per_candidate_prompt`. `APPROACH i of
n` is still appended for attributability.

**Unverified:** whether N byte-identical prompts now yield distinct candidates.
The run that would have shown this was interrupted.

---

## 2. The aborted live run (2026-08-18)

Log: `terminal_output/cache_fix/rho_live/run-rho.log` (25,709 lines, 4.3 MB).
Traces: `data/cachefix_traces/` (29 dirs). Harnesses: **none — lost**.

Config: tiny5, `--mode rho`, `k=3 G=2 N=3 R=2`, `--max-workers 24`.

**Where it stopped** — phase 4 of 4:

| phase | log lines | status |
|---|---|---|
| base group rollouts | — | done (29 trace dirs) |
| diagnosis | 97–7,687 | complete, 3 accepted |
| optimizer (N=3) | 7,791–13,018 | complete, 3/3 submitted |
| preference judge | 13,050–25,593 | **aborted mid-phase** |

Never ran: champion selection, final `after` measurement, delta, export.

**Duration: ~66 min of logging / ~73 min rollout wall-clock**, killed at 21:04
(~7 min past last log line, workers blocked in inference). Remaining work was
~5 rollouts (the `after` measurement) plus the judge tail.

**`MallocStackLogging` is NOT an error.** 11 lines, 11 distinct PIDs, all in the
final 11 lines. macOS `libmalloc` emits it during process *exit* when the feature
was never enabled — an exit-path no-op notice. There is **no traceback anywhere**:
this was a signal-kill (user interrupted the Bash call), not a crash. The line
before them is a healthy `call_model: 6 messages → model`.

The 6 `APIConnectionError`/`APITimeoutError` rollouts did NOT stop the run — the
pipeline treats a failed rollout as `status=error` and continues. Likely caused by
`--max-workers 24`; the known-good value is **10**. 34 knowledge-engine lock
warnings also appeared.

### 2.1 Rollout diversity DID improve — measured correctly

The right unit is **same `task_id` + same `started_at`** = one concurrent batch.

```text
batches with >=2 rollouts: 11    DIVERSE=7    COLLAPSED=4
```

All 4 "collapsed" are benign: 2 are `gaia-a1e91b78` answering `'3'` twice (the
correct answer — real agreement), 2 are `[error]` rollouts with no model output.
**Zero cache-driven collapses.**

Do **not** compare aggregate ratios (`0.72` new vs `0.94` old). The old run's
distinctness came largely from per-candidate framing narration — the very
confound that was removed. Aggregate distinctness across different batches
conflates unrelated things; within-batch is the only meaningful unit.

**Measurement trap I fell into twice — do not repeat:** each trace directory is
ONE rollout (`run_id` == dir name); `payloads/` are content-addressed files
*inside* it. Grouping by directory and hashing `causal-trace.json`'s
`final_answer` (which does not exist at top level) made every answer hash to the
string `'None'` and printed fake perfect determinism. Extract answers from
`payloads/*.json`.

---

## 3. New issues registered

`docs/OPEN-ISSUES.md` is now 465 lines, 20 entries. Added this session:

- **S4-8** An interrupted run destroys every candidate it produced. All 3
  candidates submitted OK at 20:12/20:14/20:18; `data/cachefix_harnesses/` does
  not exist. `--export-harness` writes only *after* champion selection, so passing
  it is **not sufficient**. Fix: persist at `submit_candidate` time.
- **S1-6** A rollout's failure reason is dropped from its manifest
  (`status=error`, `err=None`; real cause only in `events.jsonl` as
  `trace.error`). S1 because it conflates transport failure with harness failure,
  which need opposite responses and change the honest denominator.
- **S1-7 (UNCONFIRMED)** Preference judge may receive identical trajectories.
- **U-1 rewritten as RESOLVED** with the cache evidence replacing the greedy claim.

---

## 4. S1-7 — the next task, and it is offline

The judge verified in its own executed code:

```text
Are events identical? True
Are raw strings identical? True
score=0.0  "Both trajectories executed the exact same sequence of 55 graph steps"
```

**The judge is behaving correctly** — `JUDGE_INSTRUCTIONS`'s `CALIBRATION` section
mandates exactly 0.0 for indistinguishable pairs, and it also refused to reward
the more verbose side. The prompt (`cuga_preference_judge.py`:
`JUDGE_INSTRUCTIONS` line 129, `_PROMPT_TEMPLATE` line 167, `_GT_PRESENT` 218,
`_GT_ABSENT` 227) is not suspect. Position bias is handled structurally by
`compare_symmetric` (line 559): runs both slot orders, reports
`(fwd - rev)/2` as score and `(fwd + rev)/2` as observable `position_bias`.

Two unseparated explanations:
1. **Legitimate** — all 3 candidates edited only `instructions`; if the edit did
   not change behaviour, identical trajectories are honest.
2. **Wiring defect** — same trace object in both slots, which would make every
   score 0.0 and destroy ranking.

**Next step (agreed, no live spend):** offline unit test constructing a candidate
whose artifacts differ from base, asserting `read_baseline()` (line 392) and
`read_candidate()` (line 400) return different payloads.

---

## 5. The LiteLLM proxy question — settled position

- **For temperature injection: NO.** Upstream rejects the value; a proxy only
  moves where the rejected field is set. Also note the upstream *already is* a
  LiteLLM proxy (`LITELLM_BASE_URL=https://ete-litellm...`), so this would be
  LiteLLM-in-front-of-LiteLLM.
- **For request/response logging: YES, worth building.** The user raised this
  twice and was right. It would have answered S1-7 without a re-run, and it also
  addresses S1-1 (`harness_version: base`) and S1-3 (model mismatch invisible in
  the header). Official prebuilt image `ghcr.io/berriai/litellm:main-stable`
  (GHCR has `linux/arm64`; host is arm64, Docker 29.7.2 present). Mount
  `config.yaml` + a `CustomLogger` hook; pin a version tag for reproducibility.
  Notes in `feedback/gpt_context/gpt_litellm_context.md`.

---

## 6. Verified facts to carry forward

- Suite **1757 passed, 1 skipped**. `entropy.py` untouched (protected; R=2 stands).
- **76+1 = 77** CLI flags; `--allow-response-cache` is the new one.
- Known-good `--max-workers` is **10**. 24 produced connection errors/timeouts.
- Every candidate in every run so far has edited **only `instructions`**.
  `skills`, `policies`, `memory`, and the `skills/generated-` creation path remain
  completely unexercised. Possible reachability problem in the editor surface.
- Before any live run:
  ```bash
  set -a && . ./.env && set +a
  rm -f .cuga/knowledge/.lock
  ```
- Dataset: `datasets/gaia/gaia_l1_validation_tiny5__baseline__20260812_180239`.
- The exact prior working command is in
  `RESUME-HERE-2026-08-17-EVENING.md` §"the exact working run command".
- `--rho-history` omitted = COLD START (RHO phases skip, pool stays 1).
- Live model split is real and unlogged (S1-3): `LITELLM_MODEL=openai/azure/gpt-5.6-luna`
  (Interface B) vs `CUGA_MODEL=openai/gcp/gemini-3.6-flash` (Interface A).
- CUGA docs updated: `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md`
  now has a verified "Sampling, Temperature, And The Upstream Response Cache"
  section, and the two stale greedy-decoding bullets were corrected.

## 7. Uncommitted state

```text
 M docs/USER-MANUAL.md
 M reference/cuga_example_wrapper/docs/cuga-integration-learnings.md
 M scripts/run_evolution.py
 M src/agent_evolve/adapters/cuga_rho_optimizer.py     <- framings emptied
 M src/agent_evolve/core/config.py
 M src/agent_evolve/core/orchestrator.py
 M src/agent_evolve/cuga_wrapper/__init__.py           <- cache policy
 M src/agent_evolve/pipeline.py
 M tests/test_config.py
?? docs/OPEN-ISSUES.md
?? docs/superpowers/plans/RESUME-HERE-2026-08-17-EVENING.md
?? scripts/read_harness.py
?? tests/test_cuga_response_cache.py
?? tests/test_run_evolution_budgets.py
?? data/cachefix_traces/            <- 29 rollout traces from the aborted run
?? feedback/gpt_context/gpt_litellm_context.md
```

Probe logs worth keeping: `terminal_output/cache_probe/`
(`two_models.log`, `gemini_temp.log`, `openai_path.log`, `install_live.log`,
`mechanisms.log`, `wire_test.log`, `cuga_extra_params.log`).

## 8. Priority order from here

**READ FIRST: `docs/research/rho-paper-prompt-fidelity.md`.** The RHO paper
(`reference/rho_ref/RHO_2606.05922.pdf`, Appendix B Listings 1-5, pp11-16) rewards
the **process/trajectory**, not just the final outcome. Our prompts do not, and
the user's direction is to **fix the prompts before any further measurement** —
these are the measurement instrument, so a delta collected with the current
rubrics is not the delta the method claims.

Seven gaps identified, verbatim paper quotes and our line numbers in that file.
The two that matter most:

- **GAP 1** — our preference judge has **zero** mentions of efficiency
  (`grep -cin "efficien|wasted step|..." cuga_preference_judge.py` → `0`), while
  paper Listing 5 conjoins them: `+10` = trajectory *EFFICIENT* **and** answer
  correct; `-10` = *INEFFICIENT* **and** wrong. Right answer in 30 thrashing steps
  currently ties with 5 clean steps.
- **GAP 2** — paper severity band `0.4-0.7` = "mixed success, **INCONSISTENCY**,
  or a plausible harness gap": divergence alone earns mid-severity even with no
  failure. Our anchors (`cuga_rho_diagnoser.py:212-223`) key every band above 0.2
  to *recurrence*. This signal only became available now that the response cache
  is off — before the fix, trajectories were often byte-identical.

Also recorded there: GAP 7, the largest fidelity gap and **not** a prompt fix —
paper Table 5's "full harness" axis means editable executable **tools and skills**
(Listings 8/11/13 are real executables), whereas every candidate we have ever
produced edited **only `instructions`**. On the paper's own axis we are currently
"prompt text alone".

That file also lists what **not** to change (our surface-selection doctrine,
MECHANISM-not-SYMPTOM, anti-sycophancy block, regex-GT warning are better than or
deliberately different from the paper).

Sequenced work:

1. **Paper-fidelity prompt fixes** (GAP 1 → 2 → 4 → 3), then verify the
   `S_j > 0` acceptance gate (GAP 6) and decide GAP 5 with the user (2× judge
   cost).
2. **S1-7 offline test** (cheap, no spend).
3. **S4-8** persist candidates at `submit_candidate` time.
4. **S1-6** propagate `trace.error` into the manifest.
5. **S3-1** `--max-rollouts` crashes *and* reports exit 0.
6. **S1-1** `harness_version: base` on candidate traces.
7. **GAP 7** editor-surface reachability (why only `instructions` is ever edited).
8. Then a complete tiny5 RHO run at `--max-workers 10`.
9. Only then the 42-task experiment.

**Caveat that applies to all of it:** every prompt change alters measured results.
No baseline collected before these edits is comparable to one after. Record which
rubric version produced any number.
