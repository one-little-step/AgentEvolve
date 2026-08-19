# RHO Run Memory Exhaustion — Investigation Report

**Run:** `scripts/run_evolution.py --mode rho` (3 rounds, k=10, G=3, N=3, R=1,
12 process workers) over `data/traces` → `terminal_output/rho/rho_all.log`.
**Symptom:** process grew past ~90 GB RAM over a few hours and was killed; no
Python `MemoryError`/traceback in the log — the log simply stops mid-call.

## What the evidence shows

- Log is 12.4 MB / 78,926 lines, almost entirely CUGA `DEBUG`/`INFO` output
  from agent invocations running **in the parent process**.
- The run died inside **phase 9 (preference judging)** of an early round. The
  tail of the log is repeated `read_baseline()` / `submit_preference()`
  workspace-agent calls — one fresh CUGA agent per `(task, candidate)`
  comparison, each loading two full trajectories into context.
- `run_evolution` is no longer resident; 12 orphaned Playwright browser
  subprocesses (`firefox`/`webkit`/`playwright`) were still alive after the run.
- `cuga_workspace/` holds **368 directories / 9.5 GB** on disk — one workspace
  per agent invocation, never cleaned.

## Root causes (ranked)

### 1. Long-lived worker processes reuse one CUGA wrapper across every rollout, with no reset

`cuga_process_pool.py:632-671` — each of the 12 workers builds **one**
`CugaWrapper` once (`CugaWrapper.from_cuga(...)`) and then serves every task
with the same `wrapper.run_task(...)` in a loop. `run_task`
(`cuga_wrapper/__init__.py:1802-1817`) delegates to the CUGA SDK's full agent
graph (CugaLiteSubgraph → `call_model` ⇄ sandbox → FinalAnswerAgent) on **each**
call. Nothing is closed, reset, or GC'd between tasks, so the SDK's per-invocation
state — message histories, the in-memory `instructions_manager` cache (the log's
repeated `Loaded 'X' from in-memory cache` lines), context-summarizer buffers,
LangChain/LangGraph run trees, embedding/tokenizer state — accumulates
monotonically inside each worker for the whole run. 12 workers × hundreds of
rollouts is the dominant growth term.

### 2. The parent process constructs a fresh CUGA agent per RHO judge/optimizer/edit call, never closed

`cuga_workspace_agent.py:261-291` (`_run_real_agent`) does
`agent = CugaAgent(...)` → `asyncio.run(...)` with **no `close()`**. The same
pattern repeats in `cuga_editor.py:369-379`. The RHO round drives hundreds of
these in the parent process:

- 127 trajectory comprehensions + 127 difficulty verdicts (phase 2–3)
- 10 diagnoses (phase 6) + 3 optimizer proposals (phase 7)
- **N×k = 30 preference comparisons/round** (phase 9, `rounds.py:526-544`)

Each preference comparison runs a full agent whose `read_baseline()` /
`read_candidate()` tools pull complete trajectories (including up-to-4 MB
payload events, `max_observation_bytes=4_194_304`) into the agent's context, and
the model is prompted to iterate over `events` lists. Because the agent is never
`close()`d, its buffers and the `asyncio` event-loop state are not released
between calls. The 12.4 MB log being almost all phase-9 content confirms this is
where the run spent its hours and memory.

### 3. Orphaned browser subprocesses and unbounded workspace scratch

`cuga_workspace_agent.py:274` calls `prepare_workspace_environment(folder)`,
which binds `CUGA_FOLDER` to a fresh per-call directory. CUGA's browser-backed
tool (`web_fetch`/web search) spawns Playwright `firefox`/`webkit` processes
inside that directory, and they are never terminated — 12 were still resident
after the run. The scratch grows one directory per invocation (368 dirs, 9.5 GB).

## Secondary contributors (not the 90 GB, but worth noting)

- `load_history` (`core/rho/history.py:96-125`) keeps each trace's full
  `raw_trace` mapping in memory for the whole run (127 records) — bounded, but
  additive.
- `--capture-logs` + `tee` produce a 12.4 MB single log whose lines are giant
  single-line prompts; harmless to RAM but makes the run hard to monitor.

## Recommended fixes (in order of impact)

1. **Recycle worker processes.** Restart a worker after a bounded number of
   rollouts (or one rollout per process), or force `gc.collect()` / re-create the
   wrapper per task. This caps the dominant growth term at per-worker steady
   state instead of unbounded accumulation.
2. **Close every `CugaAgent` after use.** Add an explicit teardown/`close()`
   (and drop references) in `_run_real_agent`, `CugaEditorAgent`, and every RHO
   adapter that constructs an agent, plus `gc.collect()` after each phase.
3. **Kill browser subprocesses and prune `cuga_workspace/`** after each agent
   invocation (or at end-of-run); the orphaned Playwright processes are a real
   leak and a disk leak.
4. **Bound the preference-judge context.** Cap trajectory length passed to
   `read_baseline`/`read_candidate` (drop payload bodies, keep event
   signatures) — phase 9 is where the run died.
5. **Operational mitigations now:** fewer workers (`--max-workers 4`), fewer
   rounds, `--rho-candidate-rollouts 2` (not the issue here, but keeps entropy
   live), and drop `--capture-logs` (or narrow `--log-channels`) to reduce
   parent-side noise.

## Immediate cleanup for this machine

```bash
pkill -f 'ms-playwright' 2>/dev/null; pkill -f 'firefox' 2>/dev/null
rm -rf cuga_workspace          # 9.5 GB scratch, safe to delete
rm -f .cuga/knowledge/.lock    # stale lock, per USER-MANUAL §4a
```

## Does this also affect `genetic` and `rho-genetic`? — yes

Both dominant leaks live in code paths shared by every live mode. The modes
differ only in *which* parent-side agents run, not in *whether* the leaks occur.

| Leak surface | `genetic` | `rho` | `rho-genetic` |
|---|---|---|---|
| Rollout workers reuse one CUGA wrapper per process — the dominant 90 GB term. `CugaProcessPool` + `CugaRolloutRunner` are built once and reused for the whole run (`pipeline.py:955-967`, `orchestrator.py:1114-1136` → `rollout_batch.run_rollouts`) | yes | yes | yes |
| Editor constructs a fresh `CugaAgent` per `propose_edit`, never closed (`cuga_editor.py:369-393`) | yes | no (editor not run in rho-only) | yes (genetic phase runs every round) |
| RHO workspace agents (`run_workspace_agent` → `CugaAgent` per call, no close): diagnoser `cuga_rho_diagnoser.py:604`, optimizer `cuga_rho_optimizer.py:706`, preference judge `cuga_preference_judge.py:520` | no | yes | yes |
| Pure `litellm.completion` (Interface A) per call — no CUGA agent, lower risk but still a per-call client: analyzer `cuga_analyzer.py:471`, comprehender `cuga_rho_comprehender.py:548`, difficulty judge `cuga_rho_judge.py:276`, proxy validator `cuga_proxy_validator.py:319` | analyzer + proxy validator | comprehender + difficulty judge | all of the above |

Conclusions:

- **The worker-pool leak is mode-independent.** Any live run using
  `--max-workers > 1 --isolation process` leases N CUGA subprocesses once and
  reuses them for every rollout of every iteration/round, so the same monotonic
  growth will occur in `genetic` and `rho-genetic` as in `rho`.
- **`rho-genetic` is the worst case**: it stacks the RHO agents (diagnoser +
  optimizer + preference judge) *and* the genetic editor agents *and* the same
  worker pool, with the genetic phase repeated `--genetic-iterations-per-round`
  times per round.
- **`genetic` is lower than `rho`** in parent-side agent *count* (only the editor
  constructs CUGA agents; the analyzer is pure litellm), but it shares the
  identical worker leak, so at 12 process workers it will exhibit the same
  dominant growth. Its parent-side growth is governed by how many edits are
  attempted (`--max-attempts`, `--max-accepted-edits`) across `--iterations`.

The fix set is identical across modes: recycle worker processes after a bounded
number of rollouts, and `close()` every `CugaAgent` (editor + all
`run_workspace_agent` callers) after each invocation.
