# Session Handoff — SV-8, SV-11, SV-13, judge wiring, champion math (SV-2/3/5)

**Written for a post-compaction agent.** Every fact here was verified by executing
code, not recalled. Anchors are `file:line` at time of writing plus a durable
`rg` pattern, because line numbers drift.

---

## 1. State

| | |
|---|---|
| Branch | `dev7` |
| HEAD | `8d48a8f` "some issues fixed" |
| Suite | **1952 collected, exit 0, 0 failed** |
| Committed this session | **nothing** |

`8d48a8f` (already committed, *not* this session's work) contains the prior
session: mitmproxy stack under `docker/observability/`, memory-leak fixes
(`benchmarks/cleanup.py`, worker recycling, bounded judge context), SV-4/SV-6/SV-9,
and the `IMPLEMENTED-PIPELINE-MAP.md` rewrite.

**This session's work is entirely uncommitted** — 26 modified files, 15 untracked
(13 code/test + this handoff + `.jspace/`). Do not `git checkout`/`stash` anything
without asking.

Suite arithmetic, so a drift is detectable: `1935` baseline collected
(`1934 passed + 1 skipped`) `+ 9` config-threading tests `+ 8` SV-2 tests `= 1952`.
Note `test_champion_intersection.py` holds **8** tests, not 9 — an earlier claim of
9 in conversation was a miscount, corrected here.

New files (all untracked):
```
src/agent_evolve/core/retirement.py
src/agent_evolve/core/resolution.py
tests/test_multi_surface_seeding.py     tests/test_parent_observation.py
tests/test_generational_retirement.py   tests/test_retirement_decision.py
tests/test_retirement_wiring.py         tests/test_final_resolution.py
tests/test_export_resolution.py         tests/test_judge_in_every_mode.py
tests/test_resolution_config.py         tests/test_resolution_config_wiring.py
tests/test_champion_intersection.py
```

`.jspace/` is the j-space ledger for this task. It is scratch state, not a
deliverable; leave it untracked.

---

## 2. Standing constraints

- **Do not commit** without explicit user approval.
- `src/agent_evolve/core/**` must not import `cuga`, `litellm`, or `adapters`.
  Verified by AST across **all 34 core files** — 0 violations. Re-check after any
  core edit. Note a plain `rg` for `adapters` gives false positives: the word
  appears in comments. Use an AST walk over `ast.Import` / `ast.ImportFrom`.
- `core/entropy.py` is untouched and stays that way (`git diff --stat` = 0 lines).
- **Tests before implementation.** Prove a new test fails against unfixed source.
  Where a fix is a one-line wiring change, prove it by *temporarily reverting* the
  line, running the tests, then restoring the file and confirming it is
  byte-identical. Both fixes this session were verified that way, each across every
  test in its own file.
- No prompt-substring or tautological assertions; assert behaviour. Specifically:
  do not assert "an argument was forwarded" — assert the *winner id it changes*.
  A test that patches a function and checks `config is not None` passes against a
  callee that ignores the argument.
- Every behavioural test that pins a fix needs a **control** proving the old
  behaviour really did differ, or the test can pass for the wrong reason.
- Log verification to `terminal_output/<topic>/<name>.log`.
- Never persist credentials, expected answers, evaluator internals, or labels.
- Do not rerun an expensive full live RHO experiment.
- `timeout` does **not** exist on this macOS box — do not use it in commands.
- The LSP index is stale and emits false positives (notably
  `pipeline.py` `python_executable` — the reported line is `**(`, unrelated to any
  edit; `cuga_editor.py:381` CugaAgent kwargs; `orchestrator.py`
  `"base" is not defined`; several `tests/test_cuga_*` type complaints).
  **Verify against source or the suite before acting on any LSP error.**

Live run shape:
```bash
set -a && . ./.env && set +a
rm -f .cuga/knowledge/.lock
python scripts/run_evolution.py --mode rho --max-workers 6 --isolation process \
  --max-rollouts-per-worker 20 --cleanup-on-exit
```
`--max-workers 10` or fewer, never 24. `--cleanup-on-exit` is destructive (kills
Playwright-managed browsers, deletes stale `cuga_workspace/`); ask before running.

---

## 3. Closed this session

### SV-8 — the optimizer only ever saw `instructions`
Two independent blockers, both fixed (user chose "empty slot per surface + widen
creatable prefixes"):

1. **Empty roster.** `VANILLA_HARNESS` — the only builtin — has
   `skills=0, memory=0, policies=0` (verified by constructing the harness object
   and counting all three surfaces individually), so
   `_harness_artifacts` returned `{"instructions"}` alone.
   Fix: `pipeline.py:1209` seeds one **empty** slot per surface,
   `_SEEDED_SLOT = "generated-evolved"` (`pipeline.py:1206`). Never overwrites
   real content. Empty on purpose — starter text would be an authored prior that
   confounds attribution.
2. **Creation confined to `skills/`.** `creatable_prefix` (scalar) →
   `creatable_prefixes` (tuple) at `cuga_editor_state.py:58`. `generated-` marker
   retained on every surface so provenance and the creation cap still work.
   Touched 7 files including `examples/fake_adapter.py` — that one matters,
   because its comment says it deliberately mirrors `CugaAdapter`, and leaving it
   scalar would make every offline test rehearse a path the real adapter no longer
   takes.

`rg -n "_SEEDED_SLOT|DEFAULT_CREATABLE_PREFIXES" src/`

### SV-11 — the analyzer observed only `pool.base`
**The register was partly wrong; measurement corrected it.**
- `orchestrator.py:541` (`Orchestrator.run_iteration`) has **zero callers**
  anywhere — dead code, like `SequentialGepaRunner.run()`. Production uses
  `run_attempt`. Editing it would have changed nothing.
- `run_attempt` **already** rolled out and analyzed `select_parent()`.
- The real defect was the single site in `build_issues`, which hardcoded
  `self.pool.base` for rollout, write set, inventory *and* score attribution.
  Measured over 6 attempts: base **12** rollouts, every candidate **2**, and
  **every cell stuck at 1 comparable candidate** against an entropy floor of 3.

Fix: `orchestrator.py:1490` `parent = self.select_parent()` plus all four
attribution sites (score, finding, pareto, lineage).
Worse than documented: base scored `0.0` forever, so the loop kept re-diagnosing
failures a candidate had **already fixed**.

`rg -n "parent = self.select_parent\(\)" src/agent_evolve/core/orchestrator.py`

### SV-13 — generational retirement (user's proposal)
Rule: when an accepted offspring is **preferred by the RHO pairwise judge** over
its parent, the parent is **soft-retired**. Recorded in `AGENTS.md:38`.

| Piece | Anchor | Cost |
|---|---|---|
| `PoolEntry.retired` / `superseded_by` | `pool.py:227` | free |
| `live_candidate_ids()` | `pool.py:725` | free |
| `retire()` | `pool.py:739` | free |
| `has_sole_survivor()` / `sole_survivor()` | `pool.py:777` / `:785` | free |
| `decide_retirement()` | `retirement.py:70` | `2k` judge calls, **0 rollouts** |
| `_maybe_retire_parent()` | `orchestrator.py:2146`, called `:2122` | — |
| `resolve_final_candidate()` | `resolution.py:117` | `2(N-1)` calls |
| `stack.resolve_winner()` / `winner()` | `pipeline.py:639` / `:628` | rollouts |

Key properties, each a decision:
- **Soft, never pruned.** Score cells stay, so entropy comparability and negative
  evidence survive. `pool.prune()` remains ablation-only and its docstring now
  says so explicitly.
- **The judge decides, not `dominates()`.** Numeric dominance cannot see whether a
  child fixed the parent's *mechanism*. One instrument governs retirement,
  promotion (SV-4) and final resolution.
- **Conservative.** No judge / unavailable verdict / tie / incomplete traces /
  raising judge → parent lives, candidate still committed.
- **Retirement costs zero rollouts** — corrected downward from my own estimate.
  Both trace sets already exist: parent's from `build_issues`, child's from
  `validate`. `ValidationResult` kept only `trace_id`, so they were being
  discarded; now retained in `_last_observation_traces` / `_last_validation_traces`.
- Retired entries excluded from `parent_frequencies`, `pareto_frontier`,
  `select_champion`.
- Terminal condition: live pool of 1 → that candidate wins, **zero judge calls**.

### SV-4 preserved through resolution — two defects *I* introduced
Caught by the pre-existing
`test_rho_wiring.py::test_gated_candidate_does_not_become_the_exported_champion`.
1. **The ladder overturned the gate.** A candidate with `preference = -0.5` was
   exported, because a fresh comparison beat the recorded RHO verdict.
   Fix: `_is_promotable` filter (`resolution.py:101`) runs **before** the
   sole-survivor short-circuit — otherwise retiring a parent hands the win to an
   ineligible candidate by default.
2. **Fallback grabbed `live[0]`**, which can be the candidate the gate just
   disqualified. Fix: fall back to **base** (`resolution.py:76`).

### Judge in every mode
**Gap:** `compare_preference` was bound only in `build_rho_hooks`, which
`--mode genetic` never reaches (`run_evolution.py:1099` builds `rho_config` only
when mode != genetic; `:1146` vs `:1149` dispatch). So genetic offspring had
`preference is None` forever and the SV-4 gate rejected all of them — base was the
only exportable harness.

Fix: judge bound at **stack construction**, which every mode goes through.
- `_bind_preference_judge` (`pipeline.py:848`) — accepts `compare_symmetric` or a
  bare callable; requires the symmetric two-call form.
- `_default_preference_judge` (`pipeline.py:837`) — lazy import, so the module
  never forces the CUGA SDK.
- `build_live_stack` **defaults to the real judge** (`pipeline.py:1179`).
- `build_offline_stack` defaults to `None` (`pipeline.py:979`) so the
  deterministic suite gains no model dependency.
- `--dry-run` passes `OfflinePreferenceJudge` (`run_evolution.py:967`, `:1139`).
- `build_rho_hooks` still overrides, so RHO keeps its injectable judge.
- Retirement verdict is now **recorded as preference** (`orchestrator.py:2201`),
  which is what makes a genetic offspring promotable at all.

Proven end-to-end via the real CLI, running both arms through the full dry-run
path (this covers mode dispatch and judge binding; it does not cover a live model):
```
--mode genetic, judge removed  →  measuring the champion (base-v0)
--mode genetic, judge wired    →  measuring the champion (base-v0+att-i001-s0000)
```
`--experimental-candidate-promotion` disables the **gate only**; the judge remains
(`test_an_experimental_promotion_run_still_has_a_judge`). This was the user's
explicit requirement: an ablation must not lose the instrument it is compared to.

### Champion aggregate config threading — closed

`resolve_final_candidate` reached `select_champion` through `_aggregate_fallback`
with **no `config`**, so every `champion_*` value reverted to its dataclass default
on the one path that still ranks by score. The weights only corrupt a reported
number; `champion_min_coverage_fraction` silently became `0.0`, which *readmits a
candidate the operator disqualified* — and that path fires precisely when the judge
is unavailable, so the floor was dropped at the one moment it was the only guard.

- `resolution.py:141` `resolve_final_candidate(..., config=None)`;
  `resolution.py:82` `_aggregate_fallback(pool, reason, config=None)`.
  All **6** internal fallback sites forward it (AST-verified).
- `pipeline.py:673` and `pipeline.py:695` pass `config=self.runner.config`.
  No new field was needed: `runner.config` is `ResolvedConfig | None`
  (`orchestrator.py:951`).
- Tests: `tests/test_resolution_config.py` (5), `tests/test_resolution_config_wiring.py` (4).
- Non-vacuity: removing both `config=` forwards fails 3 tests with
  `- broad / + narrow` — the under-measured candidate gets exported.
- `rg -n 'def _aggregate_fallback|config=self.runner.config'`

### SV-2 — CLOSED. Ranking is pairwise over shared cells

`_champion_outcome` averaged over whatever cells each entry measured, so means were
compared across candidates measured on *different* tasks:
```
base   easy(0.9) + hard(0.1)  ->  outcome 0.500   agg 0.6250
candA  easy(0.9) only         ->  outcome 0.900   agg 0.7450   <- used to win
```
`candA` is identical to base on the only task both attempted and won by skipping.

**Fix (the register's option 3, already resolved — do not redesign):** gate on
`S_j > 0` first (SV-4, closed), then rank survivors on the pairwise intersection.

- `pool.py:613` `_intersection_outcome`, `pool.py:633`
  `_pairwise_outcome_preference` (reuses `comparable_cells`, so
  `min_comparable_rollouts` applies exactly as in `dominates`).
- `pool.py:795` replaces the scalar sort with a king-of-the-hill pass in insertion
  order. Tie, loss, or empty overlap all leave the incumbent standing.
- `pool.py:309` `ChampionReport.comparable_cells` — the "report the intersection
  size" requirement.
- `pool.py:596` `_champion_outcome` retained **for the manifest only**, documented
  as non-ranking.
- Tests: `tests/test_champion_intersection.py` (8). 5 failed against unfixed
  source; the 3 that passed are counterparts (better/worse on shared cell,
  determinism) proving the suite is not satisfiable by freezing the base.
- `rg -n 'def _pairwise_outcome_preference|SV-2: rank by'`

**Why the ranking key could not stay a scalar** — two candidates may share no cell:
```
base  vs candA: shared={easy}  0.9 vs 0.9  -> tie
base  vs candC: shared={hard}  0.1 vs 0.4  -> candC
candA vs candC: shared={}                  -> no verdict expressible
```

### SV-3 — CLOSED by construction, and SV-5 as documentation

SV-2 subsumed SV-3 exactly as SV-3's own fix direction predicted. No weight can
flip a winner, so coverage-as-quality is not expressible; coverage survives only as
the enforced `champion_min_coverage_fraction` floor.

That made **all four** champion weights non-selecting, which the user accepted as
"aggregate becomes report-only". Consequences, all documentation:

- `run_evolution.py:485-503` — every `--champion-*` flag now says "does not affect
  selection". `gamma`/`delta` are labelled **reserved, currently constant** instead
  of "worst-case"/"novelty", which described a specification, not the code.
- `select_champion` docstring and `docs/architecture/selection-algorithms.md`
  rewritten: the aggregate is a reported diagnostic.
- 3 tests that asserted weights flip the winner were rewritten to assert pairwise
  behaviour plus reported-aggregate reproducibility:
  `test_pool.py::test_select_champion_ranks_pairwise_not_by_config_weights`,
  `test_resolution_config.py::test_the_configured_weights_reach_the_reported_aggregate`,
  `test_final_resolution.py::test_the_ladder_can_reject_the_candidate_the_aggregate_would_export`.

**Live option deliberately left on the table:** implementing `gamma` as real
worst-case (min-over-tasks) score. It is the signal that would independently catch
the SV-2 exploit. Recorded in `docs/SEVERE-OPEN-ISSUES.md` under SV-5 so the choice
is not lost.

---

## 4. Open, in the documented order

### Others
- **SV-10** — parent vulnerabilities never reach the editor;
  `ParentContext.score_summary` is a lossy projection. This is what makes the
  SV-13 premise ("the child fixes the parent's faults") reliably true, so it is
  higher-value than its severity suggests.
- **SV-12** — entropy starvation. SV-11 fixed the *distribution* (rollouts follow
  the parent), but the floor of 3 comparable candidates per cell was **not** shown
  to be met: offline, an accepted candidate scores a perfect 1.0 and evolution
  correctly stops at one candidate. Needs live-shaped tasks the parent keeps
  partially failing. **Do not claim SV-12 closed.**
- **SV-1** — reclassified, not a defect. `ScoreProvenance.severity`/`.confidence`
  are never supplied in production, so `weighted_score() == mean`. Two unrelated
  fields are named `severity`; `CausalAnalysis.severity` *does* matter.
- **SV-7** — narrowed to MEDIUM. Judge and rollout grid exonerated; only upstream
  candidate materialization remains. Needs a live proxy-captured run.
- **`X-AE-*` correlation headers** — `docker/observability/addons/correlate.py`
  reads them; no production caller emits them yet.
- **`core/merge.py` (crossover)** — 393 lines, **zero** production callers; only
  `tests/test_merge.py` imports it. Genetic stage is mutation-only.
- **Docs** — `docs/SEVERE-OPEN-ISSUES.md`, `IMPLEMENTED-PIPELINE-MAP.md` and
  `selection-algorithms.md` are all current as of 2026-08-20 (SV-13, judge wiring,
  and the champion-math closures are described). `AGENTS.md` covers soft
  generational retirement. No known stale doc claims remain — a sweep for
  "aggregate is wrong" / "SV-2 and SV-3 stay open" phrasing returns nothing.

---

## 5. Decisions the user made (do not re-litigate)

| Question | Answer |
|---|---|
| SV-8 seeding | Empty slot per surface **+** widen creatable prefixes |
| SV-11 observation budget | Selected parent instead of base (cost-neutral) |
| Retirement mechanism | **Soft** retire; keep evidence |
| Retirement condition | Judge-preferred, **not** numeric `dominates()` |
| Retirement judging cost | Always judge; `2k` acceptable since k is 5-10 |
| Final resolution | Sole survivor, else symmetric ladder `2(N-1)` |
| `resolve_winner` in export | Yes — wired |
| Judge in ablation arms | **Required**; a judge-less experiment measures nothing |
| `AGENTS.md` edits | Authorized |
| SV-5 direction | **Document as reserved, change nothing** — terms stay inert; do *not* delete the terms or the flags, and do not implement worst-case without asking |
| SV-2/SV-3/SV-5 order | **Follow the register**: SV-2, then SV-3, then SV-5 — the sequence was load-bearing and is now complete |
| Champion weights | **Aggregate is report-only.** Pairwise intersection is the sole ranking key; all four `--champion-*` weights are reported diagnostics |

---

## 6. Verification logs

```
terminal_output/sv8/           full-suite.log, sv8-verify.log
terminal_output/sv11/          baseline-evidence.log, verify.log
terminal_output/sv13/          retirement-verify.log, decision-verify.log,
                               resolution-verify.log, export-wiring-verify.log
terminal_output/judge_all_modes/  dryrun-genetic.log, verify.log
terminal_output/resolution_config/  01-tests-fail-before-fix.log, 02-control-passes.log,
                               03-tests-pass-after-fix.log, 04-related-suites.log,
                               05-full-suite.log
terminal_output/sv2/           01-tests-fail-before-fix.log,
                               02-full-suite-after-impl.log,
                               03-full-suite-after-test-updates.log
terminal_output/sv3_sv5/       01-full-suite-after-docs.log, 02-full-suite-final.log,
                               03-final-with-register.log, 04-final-all-docs.log
```

Counting suite results — **`-q` suppresses the summary line on this box**, so the
`N passed` regex finds nothing and `tee | tail` truncates it. Use the exit code plus
an explicit `FAILED` scan, which is what every log above was verified with:
```bash
python -m pytest -p no:warnings --no-header -q --tb=short 2>&1 | tee /tmp/suite.log
echo "exit=$?"; grep -c '^FAILED' /tmp/suite.log || echo "0 failures"
```
For a collected count (expect **1952**):
```bash
python -m pytest -p no:warnings --co -q | awk -F': ' '/: [0-9]+$/{s+=$NF} END{print s}'
```

---

## 7. First actions after compaction

1. Re-read this file in full.
2. Confirm state: `git status --porcelain` — expect **26 modified** files (lines
   beginning with `M`) and **15 untracked** (lines beginning with a double
   question mark). Then `git rev-parse --short HEAD` (`8d48a8f`), and the suite at
   **1952 collected, exit 0**.
3. Re-read `AGENTS.md` (boundaries) and `docs/SEVERE-OPEN-ISSUES.md` (register).
4. State the pass being taken. **Next action: SV-10** — parent vulnerabilities never
   reach the editor; `ParentContext.score_summary` is a lossy projection. Write
   behavioural tests first, asserting the editor *receives* the parent's diagnosed
   faults, not that a prompt contains a substring.

**Why SV-10 and not SV-12 or SV-7.** SV-10 affects every generated edit, and it is
the premise generational retirement rests on: SV-13 retires a parent because its
child "fixed the parent's diagnosed faults", which is only reliably true if those
faults actually reached the editor. SV-12 needs live-shaped tasks that do not exist
offline; SV-7 needs a bounded live proxy capture. SV-10 is fixable offline today.

**Before committing** (only on explicit approval): the diff spans 26 modified and
13 code/test files across two logical changes — the SV-8/SV-11/SV-13/judge work and
the champion-math chain. They are separable if the user wants two commits.

**Two corrections made this session, recorded so they are not re-derived:**
- I claimed SV-2 was "fallback-only, robustness not correctness". **False.**
  `select_champion` is reached by the SV-4 eligibility gate and by every
  `aggregate_fallback` — i.e. any judge outage. It was load-bearing.
- I recommended deleting the SV-3 coverage term and the SV-5 inert terms. That
  would have silently dead-ended three **public CLI flags** and destroyed a
  specified-but-unbuilt worst-case objective. Reading the code before deleting it
  is what caught this. Prefer reading the flags/docs a change would invalidate
  *before* proposing deletion.
