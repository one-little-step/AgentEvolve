# Open Issues

Central register of known defects, unenforced surfaces, and unverified claims.

**Rules for this file**
- Every entry states how it was *observed*, not how it was inferred. If a claim is
  unverified, it says so.
- Nothing is marked fixed without a command whose output proves it.
- A defect that produces a *silently wrong number* ranks above a crash. A crash
  stops you; a wrong number gets published.

**Not tracked here**
- **SEVERE, measurement-instrument defects** live in `docs/SEVERE-OPEN-ISSUES.md`
  (SV-1..SV-12). Those are **deferred until the LiteLLM logging proxy exists**,
  because each one runs without error and produces a plausible number or a silent
  empty result that cannot currently be distinguished from success. That file
  supersedes the former **S5-1** (champion acceptance gate → **SV-4**) and
  **S5-2** (crashed rollouts → **SV-9**), which were moved there.
  **SV-11 is the most structural**: the analyzer observes only `pool.base`, so no
  candidate is ever mechanism-analyzed — which makes `--mode genetic` repeated
  mutation of a fixed base rather than population evolution, starves
  cross-candidate entropy (**SV-12**), and blocks parent-targeted editing
  (**SV-10**).
- RHO **paper-fidelity prompt gaps** live in
  `docs/research/rho-paper-prompt-fidelity.md`, not in this register. They are not
  defects in the implementation; they are divergences between our reward rubrics
  and the paper's (`reference/rho_ref/RHO_2606.05922.pdf`, Appendix B). The paper
  rewards the **process/trajectory**, ours leans on outcome. Because the rubric IS
  the measurement instrument, those are scheduled ahead of the items below.
  GAPs 1-4 implemented 2026-08-19; GAP 6 is now **SV-4**; GAPs 5 and 7 open
  (GAP 7 is **SV-8**).
- The wiring map, with every formula and what is dead code, is
  `docs/architecture/IMPLEMENTED-PIPELINE-MAP.md`.

Last verified against the tree at: suite `1757 passed, 1 skipped`, branch `dev5`.

**Fixed since the last revision**
- Upstream response cache is now disabled for every rollout (was U-1, misdiagnosed
  as greedy decoding). `apply_response_cache_policy` /
  `install_response_cache_policy` in `cuga_wrapper/__init__.py`, installed on the
  `LLMManager` singleton so all agent roles and all cache-hit paths are covered.
  Opt back in with `--allow-response-cache`. Verified live on the real
  `platform="openai"` / `ReasoningChatOpenAI` path: `DISTINCT=1` → `DISTINCT=4`.
  13 unit tests in `tests/test_cuga_response_cache.py`.

---

## Severity legend

| Tag | Meaning |
| --- | --- |
| **S1** | Silently produces a wrong or misleading measurement. |
| **S2** | Feature advertised (CLI flag / docstring) but does not work. |
| **S3** | Crash or hard stop; loud, so it cannot corrupt a result. |
| **S4** | Design gap or missing capability; nothing is wrong, something is absent. |

---

## S1 — silently wrong measurements

### S1-1 Candidate rollouts are stamped `harness_version: base`

Every trace written by the live RHO runs carries `harness_version: "base"`, even
for rollouts executed against a candidate harness.

Observed:
```
rho:         34 traces total | by harness {'base': 34} | distinct tasks 5
rho_genetic: 31 traces total | by harness {'base': 31} | distinct tasks 5
```
(`data/live_traces/rho`, `data/live_traces/rho_genetic`, 2026-08-17)

Why it matters: provenance is how a candidate's evidence is attributed. If every
trace claims to be the base, then per-candidate evidence cannot be separated from
baseline evidence by reading the corpus, and any later analysis keyed on
`harness_version` silently mixes arms. `pipeline.py:_restamped` exists to correct
`trace.candidate_id`, so the in-memory path may be correct while the persisted
trace is not — that would make this a *persistence* bug rather than an attribution
bug. **Not yet determined which.** Next step: compare `candidate_id` against
`harness_version` inside a candidate's own trace file.

### S1-2 `RetryBudget.reset()` is never called, contradicting its own contract

`RetryBudget` (`core/memory.py:197`) documents: *"Once a budget is exhausted,
further attempts for the same scope are rejected and the orchestrator must move on
**or wait for the next outer iteration**"*, and `reset()` is documented as *"used
on outer iteration refresh"*.

Observed: `grep -rn 'retry_budget.reset' src/` returns nothing. No caller exists.

Consequence: an `(issue, artifact_group, lineage)` scope that exhausts its 3
attempts stays exhausted for the **entire run**, not just the current iteration.
A long run therefore has a monotonically shrinking set of addressable issues, and
later iterations do less work than the operator believes they are paying for. The
iteration line reports this as `no_issue`, which reads as "nothing was wrong".

Either the docstring is wrong or the reset call is missing. Both are load-bearing;
pick one deliberately.

### S1-3 Interface A and Interface B can silently run different models

`LITELLM_MODEL=openai/azure/gpt-5.6-luna` drives CUGA (Interface B).
`RuntimeSettings.model` resolved to `openai/gcp/gemini-3.6-flash` and drives
Interface A (comprehender, difficulty judge).

Observed in the live RHO run: CUGA logged `Using MODEL_NAME from environment:
azure/gpt-5.6-luna` while `RuntimeSettings.from_env().model` returned
`gcp/gemini-3.6-flash`.

Consequence: an ablation that "changes the model" may change only one of the two
stacks. Nothing in the run header prints both, so the discrepancy is invisible.
Fix direction: print both resolved models in the run header.

### S1-4 `provenance.mean_score` is trivially mistaken for the benchmark result

Two different numbers describe the same harness and disagree:

| | rho base | rho champion |
| --- | --- | --- |
| `provenance.mean_score` | `0.50` on 2 cells | `0.50` on 2 cells |
| final benchmark line | `2/2 = 100%` | `1/3 = 33%` |

`mean_score` (`pipeline.py:1661`) averages `(task, mechanism)` cell means from the
**evolution loop's internal rollouts** — the evidence selection saw. The
`before`/`after`/`delta` lines come from a **separate final measurement** over all
tasks. Different rollouts, different denominators, different purpose.

Nothing in the exported file says which is which, and the field name reads like a
result. I misread it myself in this session and reported "all three candidates
scored worse than base" from a comparison across different `scored_cells`, which is
not a sound comparison.

Fix direction: rename to `selection_mean_score`, or emit both alongside a label.
`scripts/read_harness.py` now prints a caveat when `scored_cells <= 3`, which is a
mitigation, not a fix.

### S1-5 Champion selection promoted a candidate on thinner evidence than the base

Observed in `rho_genetic`:

```
candidate-base.json             mean_score=0.75  scored_cells=4   (incumbent)
candidate-rho-cand-002-c1.json  mean_score=1.00  scored_cells=2   -> CHAMPION
```

The promoted candidate was measured on **half as many cells** as the base it
displaced. The outcome then contradicted the promotion: that champion benchmarked
`33%` against the base's `50%` (`delta: -16.67 pp`). Selection chose a harness that
was worse on the benchmark.

`--champion-min-coverage-fraction` exists for exactly this and defaults to `0.0`,
so no coverage floor is applied unless asked. Whether the default is wrong, or the
champion formula should weight coverage harder (`--champion-beta`, default `0.20`),
is an open decision. Do not treat any champion selected at `scored_cells <= 3` as
evidence of improvement.

### S1-6 A rollout's failure reason is dropped from its manifest

**Observed.** Six rollouts in the aborted 2026-08-18 run recorded
`status="error"` while the manifest's error field read `None`. The real cause was
only in `events.jsonl`:

```text
manifest.json : status=error   err=None
events.jsonl  : llm_call_error   APIConnectionError('Connection error.')
                llm_call_error   APITimeoutError('Request timed out.')
                graph_node_error node=call_model
```

Ranks S1 rather than S4 because the two causes it conflates need opposite
responses: a transport failure means "retry, the harness is untested on this
task", whereas a harness failure means "this harness loses on this task". A reader
seeing `status=error, err=None` cannot tell them apart, and the honest denominator
for a pass rate depends on which it was. Cost one diagnostic round this session.

Fix direction: propagate the first `*_error` payload into the manifest. The value
already exists in `causal-trace.json` as `trace.error`.

**Status update 2026-08-19.** This is no longer a blocker for *excluding* crashed
rollouts from evidence — SV-9 closed that on both the GEPA and RHO paths using
`trace.status` against the `ANSWERED_TRACE_STATUSES` whitelist, which does not
need the manifest. S1-6 remains open for *attribution*: a reader still cannot tell
a transport failure from a harness failure, and the two need opposite responses.

### S1-7 UNCONFIRMED — preference judge may receive identical trajectories

**Not yet diagnosed; recorded so it is not lost.** In the aborted run the
preference judge repeatedly verified, in its own executed code, that the two
trajectories it was comparing were byte-identical:

```text
Are events identical? True
Are raw strings identical? True
rationale="Both trajectories executed the exact same sequence of 55 graph
           steps/events and arrived at the same answer (ball 100)"
score=0.0
```

The judge's behaviour is **correct** — `JUDGE_INSTRUCTIONS`'s `CALIBRATION`
section mandates exactly 0.0 for indistinguishable pairs, and it also declined to
reward the more verbose side. The prompt is not suspect.

Two candidate explanations, not yet separated:

1. **Legitimate.** All three candidates edited only `instructions`, and if that
   edit did not change behaviour on the task, identical trajectories are the
   honest result.
2. **Wiring defect.** The same trace object reaching both slots, which would make
   every candidate score 0.0 and render ranking meaningless.

The log captured only the judge's prose summary, not the raw `read_baseline()` /
`read_candidate()` payloads, so it cannot be settled from this run. Cheapest
next step is offline: construct a candidate whose artifacts differ from base and
assert the two tool payloads differ. No live spend.

---

## S2 — advertised but non-functional

### S2-1 Seven of eleven budget caps do nothing

Added 2026-08-17. Enforcement was wired for attempts/accepted-edits/rollouts only.
The rest parse, validate, reach `BudgetLimits`, appear in `--help` and in the
manifest — and bound nothing.

Measured (`--dry-run --tasks 3 --iterations 4`, control = 4/4 iterations attempted):

| Flag | Attempted | Verdict |
| --- | --- | --- |
| `--max-attempts 2` | 2/4 | **enforced**, clean stop |
| `--max-accepted-edits 2` | 2/4 | **enforced**, clean stop |
| `--max-rollouts 4` | — | **enforced but CRASHES** — see S3-1 |
| `--max-editor-calls 1` | 4/4 | **no effect** |
| `--max-judge-verdicts 1` | 4/4 | **no effect** |
| `--max-model-tokens 10` | 4/4 | **no effect** |
| `--max-wall-seconds 0.001` | 4/4 | **no effect** |
| `--max-pool-candidates 2` | 4/4 | **no effect** |
| `--max-history-records 1` | 4/4 | **no effect** |
| `--max-rag-context-tokens 10` | 4/4 | **no effect** |

Root causes differ per flag:
- `max_editor_calls`, `max_judge_verdicts`, `max_model_tokens` are in
  `reserve()`'s `limit_fields` map, but **no call site ever reserves them**, so
  their counters stay at 0 forever.
- `max_wall_seconds`, `max_pool_candidates`, `max_history_records`,
  `max_rag_context_tokens` are **not in the map at all** — `reserve()` cannot
  check them even if a call site existed.

A cap that reads as a safety limit but is not one is worse than no cap: it invites
an unattended long run. This is the highest-priority S2.

### S2-2 `--max-wall-seconds` has no timer anywhere

Special case of S2-1, called out because it is the cap most likely to be trusted
for an overnight run. There is no wall-clock check in the loop at all; the value
is stored and ignored.

---

## S3 — loud failures

### S3-1 `--max-rollouts` raises an uncaught `BudgetExceededError`

Observed:
```
$ uv run python scripts/run_evolution.py --dry-run --tasks 3 --iterations 4 --max-rollouts 4
...
  File ".../core/orchestrator.py", line 1134, in _execute_rollouts
    self._budget.reserve(self.config.budgets, rollouts=len(tasks))
  File ".../core/config.py", line 74, in reserve
    raise BudgetExceededError(f"{field} budget exceeded")
agent_evolve.core.errors.BudgetExceededError: rollouts budget exceeded
```

The cap *is* enforced, but by crashing mid-run: the traceback reaches the user, no
champion is measured, and no summary is printed. Reaching a planned cap must be a
clean stop like `--max-attempts` produces (`BUDGET EXHAUSTED (no attempt issued)`),
not an exception. Also note the exit code was reported as `0`, which is wrong for
a crash — a wrapper script would treat this run as successful.

---

## S4 — design gaps

### S4-1 The genetic loop has no convergence / plateau stop

`run_iterations` (`pipeline.py:539`) is a bounded `for` with a single exit
(`return tuple(summaries)` at line 604). Verified: no `break`, `continue`, or
early `return` in the body; `grep` for
`converge|plateau|stagnat|early_stop|no_improvement|patience` across
`orchestrator.py` and `pipeline.py` returns nothing.

Observed: `--iterations 10` on a plateaued run executes all 10 iterations.

Termination is therefore **exactly** whatever count is passed, plus the budget
caps that work, plus a crash. Consequences:
- you pay for iterations after the harness stops improving;
- no output line says "this plateaued at iteration 4".

Arguably correct for a research harness (predictable cost, no stopping heuristic
biasing results) — but it should be a stated decision, not an accident.

### S4-2 `--max-pool-candidates` contradicts RHO's all-N retention

RHO retains every surviving candidate by design ("All N candidates are retained.
NEVER prune to best-of-N"). A pool cap can refuse a retention the design requires.
Currently harmless only because the flag does nothing (S2-1). When S2-1 is fixed,
this must either refuse the combination or be documented as an ablation-only knob.
Documented as a warning in `USER-MANUAL.md §4 Budgets`.

### S4-9 RESOLVED 2026-08-27 - Absence of artifact use is never actionable blame, so the "create a new artifact" path is unreachable live

**How observed:** three paid live runs 2026-08-27 (`terminal_output/live-test-judge2/run1-3.log`,
logs under `logs/`, `logs2/`, `logs3/`), real endpoint + CUGA + GAIA, Judge 2 gate ON.
Run 3: base scored 1/2; the failed task (`gaia-0383a3ee`) got a correct causal
diagnosis — `call_model` emitted code that sandbox executed four times but which
never named any tool (0 `tool_call` events) — with `blamed_actors[].artifacts`
**empty**. `no_issue=1` in every iteration of all three runs; the editor never ran.

**The mechanism (traced, not inferred):** `core/evidence.py:48`
`_PAYLOAD_BEARING_KINDS = frozenset({"tool_call"})` — `_safe_payload` strips every
non-`tool_call` payload to `{}`, so `cuga_analyzer.py:934` can only keep an artifact
whose id is a literal substring of a `tool_call` payload. A failure whose cause is
"the guidance that would have made the agent look something up was never there"
produces no surface evidence at all. (`finding_from_analysis`,
`orchestrator.py:1741`, stamps the adapter's full writable set onto the top-blame
actor, so issue-building itself is not blocked by empty `artifacts` — run 3's
`no_issue=1` also had a flaky-task contribution: the BBC task passed on the
iteration's re-observation. The structural gap is that **surface absence carries no
information** anywhere in the chain.)

**Why this is a defect, not just a finding (user's ruling 2026-08-27):** if you pay
someone to guide you and he is lazy, the *absence* of guidance is itself
misguidance. Structured absence is evidence: it may justify **creating** a brand-new
skill/policy (or an instruction) rather than only improving an existing one. The
analyzer currently cannot express "this failed because nothing was loaded, and the
fix is a new artifact", so the create-new-artifact capability (`stage_create` tool
exists in `cuga_editor_tools.py:155`; `apply_edits` allows unknown ids only as
creates) is unreachable on the live path.

**Why this is a defect, not just a finding (user's ruling 2026-08-27):** if you pay
someone to guide you and he is lazy, the *absence* of guidance is itself
misguidance. Empty artifact use is evidence: it may justify **creating** a brand-new
skill/policy (or an instruction) rather than only improving an existing one. The
analyzer/schema currently has no way to express "this failed because nothing was
loaded, and the fix is a new artifact" — so the create-new-artifact capability
(`ArtifactEdit` operation `create`, `cuga_adapter.apply_edits` allows unknown ids
only as creates) is unreachable on the live path.

**Fix shape (decided 2026-08-27, before the synthetic-dataset run):**
1. Surface *structured absence* as analyzer evidence: the sanitized trace gains a
   `surface_activity` summary (which artifacts each surface-bearing rollout actually
   loaded — derived from tool_call `load_skill`-shaped payloads and the prepare
   inventory, never prompt bodies), so "no skill/policy/memory was ever loaded"
   is visible as data.
2. Analyzer contract: when the mechanism is absence-shaped and the surface summary
   is empty, the finding may say so (`unloaded_surface` in `blamed_actors[].artifacts`
   is NOT allowed — instead a dedicated `absent_surfaces` field on the finding) and
   `build_issue` treats a finding with `absent_surfaces` as actionable even when
   attributed artifacts are empty, mapping it to the parent's declared-but-unused
   writable artifacts (or to a create recommendation when the surface is empty of
   members).
3. Editor: the issue carries the absence signal so `stage_create` on a new artifact
   id is a first-class answer to it.
4. Tests-first; contamination guard must still hold (the surface summary carries
   artifact *ids and load counts* only, never contents).

**RESOLVED 2026-08-27 (same day).** Implemented:
- `core/evidence.py`: `surface_activity_from()` derives a per-surface
  (`skills`/`policies`/`memory`) summary of artifact ids actually loaded, from
  `load_skill`-shaped `tool_call` payloads only, over the FULL event list
  (beyond the 50-event trim window); ids matching answer-key terms are withheld
  and counted into `redaction_count`; every trace carries the summary, empty
  members included (explicit absence).
- `core/blame.py`: `CausalFinding.absent_surfaces` / `CausalAnalysis.absent_surfaces`
  with a closed vocabulary (`instructions/skills/policies/memory`) validated on
  both; `analysis_from_finding` and `abstained_analysis` forward it.
- `adapters/cuga_analyzer.py`: prompt documents the absence semantics
  ("SURFACE ABSENCE IS EVIDENCE"); schema gains `absent_surfaces`;
  `_grounded_absent_surfaces()` keeps only claims the trace's
  `surface_activity` corroborates and notes dropped claims.
- `core/issues.py`: `build_issue` treats absence + empty attribution as
  actionable, attributing the declared-but-unused writable artifacts of the
  absent surfaces; `Issue.absent_surfaces` carries the signal to the editor.
- `core/orchestrator.py`: `finding_from_analysis` forwards absence (and keeps
  the judged severity on an absence verdict); `build_issues` no longer drops
  absence-bearing findings as evidence vacuums.
- `adapters/cuga_editor_evidence.py` + `cuga_editor.py`: the editor prompt shows
  `MEASURED ABSENT SURFACES` with the stage_create-is-first-class guidance.
Tests: `tests/test_absence_evidence.py` (17), analyzer absence tests (5),
orchestrator absence tests (2), evidence-view tests (2). Non-vacuity: reverting
the `build_issue` absence fallback killed exactly the 5 absence-path tests.
Suite after: 2270 passed / 9 known Windows platform failures / 2 skipped.
Core neutrality re-verified: 37 files, 0 violations.

**Then:** run on a synthetic complex dataset testing BOTH capabilities — improve
existing artifact vs create new artifact.

### S4-10 NEW 2026-08-30, RESOLVED same day - Unscorable reasons are discarded: "no measurement" is indistinguishable from an outage

**How observed:** synthetic honing6 live run 2 (2026-08-30,
`terminal_output/synthetic-honing6/run2.log`). 4 of 6 tasks unscorable in the
before-measurement. The tally (`pipeline/before__base.jsonl`) records only
counts and task ids. Diagnosing WHY required forensic reconstruction: replaying
judge calls against recorded trace answers, and matching trace mtimes to the
measurement window. Two causes found (2+2): two tasks had no
`llm_grading_notes` so the judge grader was deterministically unwired for them
(dataset materialization gap, since fixed), and the other two failed at the
judge/rollout layer with a reason string the scorer built and then threw away.
`BenchmarkScorer.score_rollout` puts the exact cause into
`RolloutScore.reason` ("grader has no measurement for this task/answer: ..."),
but `tally_scores` drops it.

**Fix (implemented same day, tests-first):** `ScoreTally.unscorable_reasons`
maps each unscorable task to its reason; `measure` logs it in the pipeline
record; `run_evolution._print_tally` prints one line per unscorable task.
Test: `test_tally_carries_the_unscorable_reason_per_task`,
`test_tally_has_no_unscorable_reasons_when_everything_scored`. Every future
run answers "why unscorable" from the log, not from archaeology.

### S4-3 `.env` is not loaded before `RuntimeSettings.from_env()`

`cuga_wrapper.prepare_cuga_environment()` calls `load_dotenv(DOTENV_PATH)`, but
`RuntimeSettings.from_env()` is reached first on the live path, so a live run fails
with `CUGA_MODEL or LITELLM_MODEL is required for a live inference run` unless the
variables are already exported.

Current workaround (used for every live run so far, and documented in
`USER-MANUAL.md §4a`):
```bash
set -a && . ./.env && set +a
```
Fix direction: load dotenv at CLI entry, before any settings resolution.

### S4-11 NEW 2026-08-30 - Rollout-quality coin-flip: narration-vs-work variance makes single-pass measurements meaningless

**Where it happened (exact):** synthetic honing6 run 3, 2026-08-30,
`terminal_output/synthetic-honing6/run3.log`, before-measurement pass
10:54-11:02 local vs champion/after passes 11:22-11:27 local. Tally in
`terminal_output/synthetic-honing6/logs3/pipeline/before__base.jsonl` vs
`after__base.jsonl`: before 4/5 scored (80%), after 1/6 (16.7%) — same harness
(the champion was the base), delta -63.33 pp printed as if it were signal.

**The evidence chain (all matched programmatically,
`terminal_output/synthetic-honing6/match_judged_answers_to_traces.py`):**
- Every judged-correct answer traced back to a real-work rollout (>=32 events,
  tools executed in 5 of 6; task-04 ev=107 sandbox-only).
- Every judged-wrong answer traced back to a 17-event, 1-LLM-call, 0-tool
  rollout (7 of 7). The judge mislabeled nothing — its notes cite the real
  advisor (Devoret's is Anatole Abragam) when refuting fabricated chains.
- Narration-rollout response blob (`data/traces/4fbb61ec-.../payloads/89e860c3...json`):
  `finish_reason=stop`, 496 completion tokens, content = *"Your multi-part
  building puzzle is queued — I'll pinpoint the tallest 2025 tower…"* — the
  model ended its turn with narration, no fenced code block. CUGA then routed
  to FinalAnswerAgent with no work done. This is exactly the failure mode
  documented in `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md`
  ("Why Multi-Step Runs Can End Early": `extract_code_from_model_response`
  returns empty -> NL auto-continue classifier -> finalize).
- Judge variance ruled out: 22/22 verdicts consistent with rollout depth.

**Were the two rollout classes under the same env? YES — verified, not assumed:**
- Model: run2.log shows 269 `Set model profile ... for
  meta/muse-spark-1.2-contributor` lines; run3.log shows 162, and **no other
  model name appears in either log** (count_models_run3.py). Trace-blob
  metadata redacts `model_name`/`system_fingerprint`, so log-line counts are
  the surviving model identity evidence.
- Both runs launched with the identical in-repo `.env`
  (sha256 prefix `c49d3ddd5bdd570f`), carrying: `CUGA_MODEL` =
  `LITELLM_MODEL` = `openai/meta/muse-spark-1.2-contributor`,
  `DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true`,
  `DYNACONF_ADVANCED_FEATURES__ENABLE_SHELL_TOOL=true`,
  `DYNACONF_SKILLS__ENABLED=true`, `DYNACONF_KNOWLEDGE__ENABLED=true`,
  same base URL, same keys (redacted). Same `--harness vanilla`, same
  dataset `datasets/synthetic/honing6_v1`, same seed and worker settings.
- Same-process alternation: within a single pass, real-work and narration
  rollouts interleave minutes apart under the same worker — this is not a
  config change between passes.

**Why this is a defect, not just a finding:** with G=1 rollout per
(task, pass), the before/after delta measures sampling luck of a bimodal
rollout distribution (work vs narration), not the harness. The 2026-08-30
"evolution" numbers (-63.33 pp) are noise and must not be quoted as signal.
The doc's own methodology ("Repeat each configuration before attributing a
failure"; "three trials of one prompt may be one observation") applies.

**Fix direction:** (1) measurement passes must use rollout groups (G>1) and
report per-pass narration rate next to pass rate; (2) the narration mode is
itself the dominant failure mode — it is what the evolution loop's
instructions edit should target (force one fenced code block per turn, per
the doc's verified prompt contract); (3) the editor prompt contract already
exists (EDITOR_INSTRUCTIONS requires code-on-first-turn) — the rollout
harness does not; a vanilla-harness instruction requiring code-first turns
is the obvious first candidate edit for the loop to accept.

### S4-4 Interface B tool invocation is prompt-wording dependent

Whether a CUGA workspace agent calls **any** tool is close to a deterministic
function of prompt wording, all-or-nothing per phrasing.

Measured, one variable changed, same dataset:
- long shared contract + optimizer "write and execute" tail → 2/2 diagnoses
  observed, **3/3** distinct candidates
- two-line contract, optimizer tail removed → **0/3** candidates, all discarded
  `NO_TOOL_CALL`

A shorter contract won on a *one-tool toy probe* and that result did **not**
transfer to the real agents. Current wording is pinned by
`tests/test_rho_optimizer.py::test_shared_contract_keeps_the_live_round_verified_wording`.

Operational rule: never "tidy" `WORKSPACE_AGENT_TOOL_CONTRACT` or an agent's
instructions without re-measuring on a live round. A probe A/B is not evidence.

### S4-5 Stale CUGA knowledge lock poisons every subsequent run

A crashed run leaves `.cuga/knowledge/.lock` behind. Afterwards CUGA logs
`Knowledge engine already running in another process` and every tool call fails
with `AttributeError: 'NoneType' object has no attribute '_config'`.

Observed: 11 such errors and 12 of 16 failed rollouts in one live RHO round.

Workaround: `rm -f .cuga/knowledge/.lock`. Disabling the knowledge engine is
**not** a valid fix — tested, and it stopped the agent calling any tool at all
(the knowledge tools are part of the surface CodeAct is primed on).

Fix direction: clear a stale lock at startup when no live process holds it.

### S4-6 RHO phases are not covered by `--capture-logs`

`--capture-logs` writes `pipeline`, `workers`, `analyzer`, `editor`. The RHO
comprehender, difficulty judge, diagnoser, optimizer and preference judge write to
none of them, so diagnosing an RHO phase means grepping ~20k lines of interleaved
stdout. Observed while diagnosing S2-1 and the `NO_TOOL_CALL` failures.

### S4-7 Evolution attempt records are not persisted — FIXED 2026-08-19

Already noted in `USER-MANUAL.md §7.6`. Carried here so it is tracked in one place.

**Resolved as part of SV-6.** The root cause was not a persistence bug: nothing
ever called `EditMemory.record()` in production, because `SequentialGepaRunner`
(the class the pipeline actually builds) had no edit memory at all. Attempts are
now recorded on every terminal path, and the pipeline hands the runner an
`EditMemory(storage=storage)` shared with the editor. See
`docs/SEVERE-OPEN-ISSUES.md` § SV-6 and `tests/test_runner_edit_memory.py`.

### S4-8 An interrupted run destroys every candidate it already produced

**Observed, and it has already cost a real run.** `--export-harness` writes only
after the whole run finishes: champion selection happens first, then the export.
Until that moment every candidate lives solely in the adapter's in-memory
registry, so any interruption — `Ctrl-C`, a killed parent, an OOM, a crashed
phase — discards work that already succeeded.

Evidence from the aborted 2026-08-18 RHO run
(`terminal_output/cache_fix/rho_live/run-rho.log`):

```text
submit_candidate result: {"status": "ok", "staged": ["instructions"]}   20:12:50
submit_candidate result: {"status": "ok", "staged": ["instructions"]}   20:14:21
submit_candidate result: {"status": "ok", "staged": ["instructions"]}   20:18:32
```

All three RHO proposals survived (3 of 3, no `NO_TOOL_CALL`, no `NO_OP`). The run
was then interrupted during the *preference-judge* phase, and:

```bash
$ ls data/cachefix_harnesses/
ls: data/cachefix_harnesses/: No such file or directory
```

~66 minutes of live optimizer work, unrecoverable. The rollout traces survived
(`data/cachefix_traces/`, 29 dirs) because traces are written incrementally; the
candidates did not, because harnesses are not.

This is the "nothing survives the run" property stated in `run_evolution.py`'s own
module docstring. That docstring frames it as a reason to pass
`--export-harness`; the real behaviour is that passing it is **not sufficient**,
because the write is deferred to the end.

Why it matters beyond lost time: it makes long runs unsafe to interrupt, so an
operator who sees a run misbehaving must choose between wasting more budget and
destroying the evidence already earned. It also blocks resuming a run.

Fix direction: persist each candidate at the moment `submit_candidate` finalizes
it — the artifacts and the fingerprint are both available there — rather than only
after ranking. Ranking metadata (`provenance`) can be back-filled or written
separately. A candidate on disk with no verdict is strictly more useful than no
candidate at all.

---

## Unverified claims to re-check before relying on them

### U-1 RESOLVED — the collapse was an upstream response CACHE, not greedy decoding

**Superseded 2026-08-18 by direct probes. The old claim was wrong and is corrected
here rather than deleted, because it was load-bearing in a design decision.**

What was previously claimed: "reasoning models skip temperature, so decoding is
effectively greedy, so N identical prompts are one sample repeated N times."

What is now verified. Four identical requests returned **one shared response
`id`** with identical text, and the gateway returned an `x-litellm-cache-key`
header. A repeated response `id` proves the request never reached the model.
Reproduced on **both** models:

```text
                              luna            gemini-3.6-flash
default          n=4   distinct_ids=1  1     distinct_ids=1  1
extra_body off   n=4   distinct_ids=4  4     distinct_ids=4  4
```

Diagnostic rule this establishes: **use the response `id`, never text equality.**
A low-entropy prompt legitimately yields identical text from four genuine samples
(`"what is 2+2"` → `distinct_ids=4, distinct_text=1`), so text equality cannot
separate a cache hit from real agreement. Verified in
`terminal_output/cache_probe/two_models.log`.

Temperature, separately verified (and it is per-model, not universal):

```text
model                  temperature=0.0   =0.7   =1.0
azure/gpt-5.6-luna     HTTP 400          400    accepted
gcp/gemini-3.6-flash   accepted          ok     accepted
```

So on `luna` temperature is genuinely unavailable at any non-default value, and
CUGA's `_is_reasoning_model` suppression (`cuga/backend/llm/models.py:973-985`,
applied at `:1302-1317`) is **load-bearing, not a defect** — defeating it turns a
working call into a guaranteed 400. On `gemini` the prefix test does not match, so
CUGA already sends `temperature = 0.1` from
`configurations/models/settings.litellm.toml`. A cache-off temperature sweep on
gemini gave 4/4 distinct at 0.0, 0.1, 1.0 and 2.0, so **temperature is not the
diversity lever; cache control is**.

Consequence for `_CANDIDATE_FRAMINGS`: its justifying comment cited greedy
decoding and is now known to be the wrong mechanism. The framings were a
workaround for a transport defect and also a confound — they permanently biased
which surface each candidate considered first. Now emptied to `()` in
`cuga_rho_optimizer.py`, with `_per_candidate_prompt` guarded against the empty
case (a bare `("")` is a *string*, so `index % len(...)` raised
`ZeroDivisionError`; caught before the run).

Still unverified: whether N byte-identical optimizer prompts produce genuinely
distinct candidates now that the cache is off. The run that would have shown this
was interrupted (S4-8).

### U-2 `data/traces` is not a usable RHO corpus

`load_history(data/traces)` yields **98 records / 176 rejected**, spanning only
**18 distinct task ids**, with ids (`gaia-*`, `task-1`, `e2e-*`) that do not match
benchmark task ids. A `--rho-history data/traces` run produces
`0 coreset tasks ... pool 1`. The "267 traces" figure in earlier handoffs is
wrong. Build a fresh corpus with `--mode genetic` instead.

---

## Measurement caveats for every result so far

Not defects, but any number produced to date is subject to these.

- **No arm has demonstrated improvement.** Live deltas were `-66.67`, `-16.67`,
  `-25.00` pp against a measured **16.67 pp noise floor**, on 5 tasks with 1–3
  scored rollouts per side. Those measure variance, not evolution.
- **The champion was the base in every live arm** — nothing was accepted, so
  before/after are two measurements of the same harness.
- **Budgets used in the live runs were the minimum**: `--rho-rounds 1`,
  `--genetic-iterations-per-round 1` (`rho-genetic`), `--iterations 1`
  (`genetic`). One shot at editing, on 3 coreset tasks. Enough to prove plumbing;
  not enough to show self-improvement.
- A real delta needs the 42-task split, `--rho-rounds` > 1,
  `--genetic-iterations-per-round` > 1, and repeated runs to clear the noise floor.
