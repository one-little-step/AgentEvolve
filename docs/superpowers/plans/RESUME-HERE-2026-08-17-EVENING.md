# RESUME HERE — 2026-08-17 evening — RHO stage complete and running live

Written immediately before a context compaction. Everything needed to continue is
either here or in a file named here. **Nothing in this session was committed.**

---

## 0. State in one paragraph

The RHO stage (all 15 plan tasks) is implemented, wired, and **runs live
end-to-end** in all three modes. The suite is **1744 passed, 1 skipped** (baseline
was 1359). Six real defects were found and fixed during live bring-up. Three live
arms were measured on `gaia_l1_validation_tiny5`; **none showed improvement**, and
the dataset is too small to show one. A full budget/tuning/ablation CLI surface was
added (76 flags, all documented) but **only 4 of 11 budget caps actually enforce**.
The next substantive work is fixing `--max-rollouts` (crashes) and the
`harness_version: base` provenance bug, then a 42-task run.

---

## 1. Read these first, in this order

| File | Why |
| --- | --- |
| `docs/OPEN-ISSUES.md` | **17 tracked issues**, severity-ranked, each with how it was observed. Written this session. Read before trusting any number. |
| `docs/USER-MANUAL.md` | Operational reference. §4a is the new RHO section, §4b.2b explains harness files, "What ends the loop" answers loop termination. |
| `scripts/read_harness.py` | New. Reads an exported harness in human terms (`--base` to diff, `--lineage` for a directory). |
| `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md` | Verified CUGA behaviour. The prompt-sensitivity findings there are load-bearing. |
| `docs/superpowers/plans/RESUME-HERE-2026-08-17-RHO.md` | The plan this session executed. Design rationale and user decisions. |

---

## 2. Uncommitted changes (branch `dev5`)

```
 M docs/USER-MANUAL.md                   +358
 M scripts/run_evolution.py              +270   (41 new flags + resolve_config_overrides)
 M src/agent_evolve/core/config.py        +22    (PROFILE_GATES; budgets/features overridable)
 M src/agent_evolve/core/orchestrator.py  +92    (budget enforcement)
 M src/agent_evolve/pipeline.py           +39    (config_overrides, budget_exhausted reporting)
 M tests/test_config.py                   +66
?? docs/OPEN-ISSUES.md
?? scripts/read_harness.py
?? tests/test_run_evolution_budgets.py            (11 tests)
```

Plus, from earlier in the session and already in the tree: the whole RHO stage
(`core/rho/*`, `core/contamination.py`, `adapters/cuga_rho_*`,
`adapters/cuga_workspace_agent.py`, `adapters/cuga_preference_judge.py`), its
tests, and the `pipeline.py` `build_rho_hooks` wiring.

`src/agent_evolve/core/entropy.py` is **unchanged and protected** — verified with
`git diff --stat`. The plan's file table saying "MODIFY: remove the skip tier" is
stale, superseded by the user's `R=2` decision. Task 11 is SKIPPED.

---

## 3. How to run it (this exact form works)

```bash
# .env is NOT auto-loaded early enough (OPEN-ISSUES S4-3) -- export first
set -a && . ./.env && set +a
rm -f .cuga/knowledge/.lock          # stale lock poisons runs (S4-5)

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
  --export-harness data/live_harnesses/rho
```

- Omitting `--rho-history` = **cold start**: RHO phases skip entirely, pool stays 1.
  Build a corpus with `--mode genetic` first.
- `--max-workers > 1` requires `--isolation process`. 10 workers on 12 cores is fine.
- Preflight: `--max-workers <= --rho-group-workers * --rho-rollout-workers`,
  refused not clamped.

**Measured wall clock** (10 process workers, 12 cores): `rho` **32 min**,
`rho-genetic` **34 min**, `genetic` **9 min**. Dominated by 20-47 s per
reasoning-model call, not compute. An earlier serial run took 2 hr+ and was aborted.

---

## 4. Live results — the honest version

| Arm | Coreset | Candidates | Rollouts | delta |
| --- | --- | --- | --- | --- |
| `rho` | 3 (dpp) | **3 of 3 distinct**, pool 4 | 24, 0 failures | `-66.67 pp` |
| `rho-genetic` | 3 (dpp) | 2 of 3, pool 3, + 1 genetic iter | 18, 0 failures | `-16.67 pp` |
| `genetic` | — | pool 1 (no seeder, by design) | — | `-25.00 pp` |

**No arm improved anything.** With 5 tasks, 1-3 scored rollouts per side, and a
measured **16.67 pp noise floor**, these measure variance. The champion was the
base in the `rho` arm; in `rho-genetic` a candidate won and then benchmarked
*worse* than the base (see OPEN-ISSUES S1-5).

Budgets used were the **minimum**: `--rho-rounds 1`,
`--genetic-iterations-per-round 1`, `--iterations 1`. One shot at editing on 3
tasks. Enough to prove plumbing; not enough to show self-improvement.

Cost model: **rollouts = `k × (G + N×R)`**. Paper defaults (10,3,3,2) = **90/round**.

---

## 5. Six defects found and fixed during live bring-up

1. **Comprehender sent `model="unset"`** — never read env settings; every trajectory
   summary failed. Fixed with `_resolved_settings()` mirroring the analyzer.
2. **Judge resolved `model` but not `api_key`/`base_url`** → `Missing credentials`.
   Fixed: all three fall back together.
3. **Coreset did not dedupe by `task_id`** — several traces of one task could occupy
   several coreset slots, burning the full budget on fewer than k tasks, silently.
   Fixed with `coreset.collapse_by_task` (keep-hardest).
4. **DPP diversity term was dead** — `run_round` never passed an embedder, so every
   run silently degraded to `dpp_quality_only`. Fixed via `RhoHooks.embedder`; this
   also made `--rho-embedding-cache` non-inert (verified: 3 cache hits on re-run).
5. **Stale `.cuga/knowledge/.lock`** → `AttributeError: 'NoneType' has no attribute
   '_config'`, 11 occurrences, 12/16 failed rollouts. Workaround: delete the lock.
6. **All N optimizer invocations got a byte-identical prompt** — one sample repeated
   N times, so all 3 failed together and `distinct` was capped at 1. Fixed with
   `_CANDIDATE_FRAMINGS` (`cuga_rho_optimizer.py:588`), rotated by index.

---

## 6. The CUGA prompt-sensitivity trap (most important operational fact)

Whether an Interface B workspace agent calls **any** tool is close to a
deterministic, all-or-nothing function of prompt wording.

Measured, one variable changed, same dataset:

| Config | Result |
| --- | --- |
| long shared contract + optimizer "write and execute" tail | 2/2 diagnoses, **3/3** candidates |
| two-line contract, optimizer tail removed | **0/3**, all `NO_TOOL_CALL` |

**Position matters as much as content.** A prompt ending on a tool roster or a
submission schema produced full narration with an empty tool ledger — including a
fabricated "Diagnosis submitted successfully". Moving the execute directive to the
**very end** flipped the same task to 7 real tool calls.

Current assembled order (`run_workspace_agent`):
```
evidence -> artifacts -> tools+doctrine -> APPROACH n of N (per-candidate)
         -> TOOLS AVAILABLE roster -> BEGIN NOW (execute directive, LAST)
```

I "improved" the contract based on a **one-tool toy probe** and it regressed the
optimizer from 3/3 to 0/3 on a live round. Pinned by
`tests/test_rho_optimizer.py::test_shared_contract_keeps_the_live_round_verified_wording`.

**Rule: never edit `WORKSPACE_AGENT_TOOL_CONTRACT` or an agent's instructions
without re-measuring on a live round. A probe A/B is not evidence.**

---

## 7. What to do next (priority order)

1. **S3-1 — `--max-rollouts` crashes** (~10 min). Raises an uncaught
   `BudgetExceededError` mid-run *and* reports exit `0`, so a wrapper script records
   a crashed run as successful. It is the most natural cap to reach for on a long
   run. Should stop cleanly like `--max-attempts` does.
2. **S1-1 — `harness_version: base` on every candidate trace.** All 34/31 traces
   claim to be the base. Determine first whether it is an attribution bug or only a
   persistence bug (`pipeline.py:_restamped` fixes the in-memory path). This can
   invalidate candidate evidence in the 42-task run.
3. **S2-1 — 7 of 11 budget caps do nothing.** Two root causes: three are in
   `reserve()`'s map but never reserved at a call site; four are not in the map at
   all. A cap that reads as a safety limit but is not one invites an unattended run.
4. **Then the 42-task run** with `--rho-rounds 3`,
   `--genetic-iterations-per-round 3`, repeated for the noise floor. Only this can
   show a real delta.

Also queued, smaller: persist discarded candidates (currently only pool survivors
export, so failures cannot be inspected); RHO phases are not covered by
`--capture-logs` (S4-6); investigate why **only `instructions` is ever edited** —
no run has ever touched `skills`, `policies`, `memory`, or used the
`skills/generated-` creation path, even when a framing explicitly invited it.

---

## 8. Facts that are easy to get wrong

- **Two different models.** `LITELLM_MODEL=openai/azure/gpt-5.6-luna` drives CUGA
  (Interface B); `RuntimeSettings.model` resolved to `openai/gcp/gemini-3.6-flash`
  and drives Interface A. Neither is printed in the run header (S1-3).
- **Temperature:** the skip is a **CUGA SDK client-side name check**
  (`cuga/backend/llm/models.py:973`, matches `o1|o3|o4|gpt-5`), *not* an endpoint
  behaviour. "Therefore greedy decoding" is **not established** — logged as U-1.
- **`provenance.mean_score` is not the benchmark result.** It averages the
  evolution loop's internal `(task, mechanism)` cells. Comparing it across
  different `scored_cells` is unsound. Trust `before`/`after`/`delta` (S1-4).
- **`data/traces` is not a usable RHO corpus**: 98 records / 176 rejected, 18
  distinct task ids that do not match benchmark ids. The "267 traces" figure in
  older handoffs is wrong (U-2).
- **The genetic loop has no convergence stop.** `run_iterations` is a bounded
  `for`; `--iterations 10` on a plateaued run executes all 10 (S4-1).
- **`RetryBudget.reset()` is never called**, so an exhausted
  `(issue, artifact_group, lineage)` scope stays dead for the whole run, reported
  as `no_issue` (S1-2).
- **`--genetic-iterations-per-round` DOES work** — verified behaviourally: `0` skips
  the phase, `1` → pool 4, `3` → pool 6. An earlier "no effect" reading was my own
  test error (I forgot `--rho-history`, so cold start produced no candidates).

---

## 9. Verification commands

```bash
uv run pytest -p no:warnings                       # expect 1744 passed, 1 skipped
git diff --stat src/agent_evolve/core/entropy.py   # MUST be empty (protected)
uv run python scripts/run_evolution.py --help      # 76 flags, all in USER-MANUAL
uv run python scripts/read_harness.py data/live_harnesses/rho --lineage
```

Artifacts on disk: `terminal_output/rho_live/{smoke,genetic,rho,rho_genetic}/`
(87 logs), `data/live_traces/*` (traces), `data/live_harnesses/*` (exported
harnesses, **not** gitignored).

**Commit policy: do not commit without explicit user approval.**
