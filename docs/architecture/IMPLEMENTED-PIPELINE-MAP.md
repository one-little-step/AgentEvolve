# Implemented Pipeline Map

**Verified against the codebase on 2026-08-19** (branch `dev7`, `HEAD` `0f72d98`,
suite `1825 passed, 1 skipped`). Derived by reading and executing the code, not
from the design docs. Where code and design doc disagree, the **code** is
reported here and the divergence is flagged.

Every box carries its `file:line` so you can cross-reference directly. Line
numbers drift as code moves — the accompanying grep anchor in each table is the
durable reference.

## Legend

| Colour | Meaning |
| --- | --- |
| 🟩 green | Implemented **and** wired into a real run |
| 🟦 blue | Implemented, wired, but semantics disputed / under active repair |
| 🟨 yellow | Partially implemented — capability exists, never exercised in practice |
| 🟥 red | **Not wired** — code exists but no production caller, or hardcoded stub |
| ⬜ grey | Formula / annotation, not an execution step |

---

## 0. Correction to a common premise

`select_champion` is **not** part of the genetic stage. It lives at
`core/pool.py:577` and has exactly three callers, none inside the genetic loop:

```
core/orchestrator.py:2160   SequentialGepaRunner.run()   <- NOT used by run_evolution.py
pipeline.py:606             champion_version()           <- reporting
pipeline.py:614             export_pool()                <- writes champion.json
```

It is a **post-hoc reporting/export selector over the whole pool**, shared by all
three modes. The genetic loop never consults it; `run_round` never calls it.
`rounds.py` is explicit: *"Rank orders the report and picks a champion; it never
decides survival."*

**Consequence:** champion defects change **which harness you export and carry into
the next run** via `--harness`, not who survives a round. In a chained multi-run
experiment that error compounds.

---

## 1. Three modes, one code path

`core/rho/rounds.py:69`

```python
PHASES: dict[str, tuple[str, ...]] = {
  "rho":         _RHO_PHASES,                              # 10 phases
  "genetic":     ("genetic_iterations",),                  # legacy GEPA loop only
  "rho-genetic": _RHO_PHASES + ("genetic_iterations",),    # RHO then genetic
}
```

```mermaid
flowchart TB
    CLI["scripts/run_evolution.py<br/>--mode {rho | genetic | rho-genetic}"]

    CLI -->|"mode != genetic"| RR["core/rho/rounds.py:335 run_rounds()<br/>-> :348 run_round()"]
    CLI -->|"mode == genetic<br/>run_evolution.py:1149"| GEN["pipeline.py:539<br/>stack.run_iterations(n)"]

    RR --> P1_10["RHO phases 1..10"]
    P1_10 -->|"only if 'genetic_iterations' in phases"| GHOOK["_run_genetic()<br/>rounds.py:740"]
    GHOOK -->|"hooks.run_genetic(coreset_tasks, iters)<br/>pipeline.py:1372"| GEN

    GEN --> POOL[("PersistentPool<br/>core/pool.py")]
    P1_10 --> POOL
    POOL --> EXPORT["pipeline.py:614 export_pool()<br/>-> pool.py:577 select_champion()"]
    EXPORT --> NEXT["champion.json<br/>--harness for the NEXT run"]

    style RR fill:#d6eaff
    style GEN fill:#ffe9cc
    style POOL fill:#e8ffe8
    style EXPORT fill:#d6eaff
```

**Key wiring fact:** the genetic phase is not a reimplementation.
`pipeline.py:1385` narrows `stack.tasks` to the coreset, calls the *same*
`run_iterations`, and restores it in `finally` — *"byte-for-byte the loop that
produced the measured baseline."*

---

## 2. RHO stage — 10 phases

```mermaid
flowchart TB
    subgraph RHO["RHO ROUND — core/rho/rounds.py:348 run_round()"]
      direction TB
      H1["<b>P1 history_load</b><br/>hooks.load_history()<br/>core/rho/history.py"]
      H2["<b>P2 trajectory_comprehension</b><br/>hooks.comprehend(record)<br/>adapters/cuga_rho_comprehender.py"]
      H3["<b>P3 difficulty_fingerprint</b><br/>hooks.judge(record, summary)<br/>adapters/cuga_rho_judge.py<br/><i>paper Listing 2</i>"]
      H4["<b>P4 coreset_selection</b><br/>core/rho/coreset.py:197 select_coreset()"]
      H5["<b>P5 group_rollouts</b><br/>k x G on INCUMBENT<br/>rounds.py:622 _rollout_grid()"]
      H6["<b>P6 group_diagnosis</b><br/>hooks.diagnose(task, traces)<br/>adapters/cuga_rho_diagnoser.py<br/><i>paper Listing 3</i>"]
      H7["<b>P7 candidate_proposal</b><br/>N independent invocations<br/>adapters/cuga_rho_optimizer.py<br/><i>paper Listing 4</i>"]
      H8["<b>P8 candidate_rollouts</b><br/>k x R per candidate"]
      H9["<b>P9 preference_judging</b><br/>compare_symmetric()<br/>adapters/cuga_preference_judge.py<br/><i>paper Listing 5</i>"]
      H10["<b>P10 pool_commit</b><br/><b>ALL N committed, never best-of-N</b>"]
      H1-->H2-->H3-->H4-->H5-->H6-->H7-->H8-->H9-->H10
    end

    H4 -.-> F4["<b>coreset.py:124</b><br/>normalized = max(difficulty/MAX_DIFFICULTY, score_floor)<br/>quality = normalized ** theta<br/>selector: dpp | difficulty_rank | random"]
    H5 -.-> F5["<b>SV-9 CLOSED</b> rounds.py:610 _answered()<br/>ANSWERED_TRACE_STATUSES gate<br/>crashed status='error' traces create NO cell"]
    H9 -.-> F9["<b>cuga_preference_judge.py:591</b><br/>score = (fwd - rev)/2<br/>position_bias = (fwd + rev)/2<br/><b>2 judge calls per pair</b>"]
    H10 -.-> F10["<b>pipeline.py:1491</b> _record_pool_score<br/>clamp(value, 0, 1)<br/>severity/confidence omitted -> default 1.0"]

    style H10 fill:#e8ffe8
    style F5 fill:#e8ffe8
    style F4 fill:#f0f0f0
    style F9 fill:#f0f0f0
    style F10 fill:#f0f0f0
```

### Phase-to-file table

| Phase | Anchor | Hook | Implementation | Paper | Status |
| --- | --- | --- | --- | --- | --- |
| 1 history_load | `rounds.py` `history_load` | `load_history` | `core/rho/history.py` | — | 🟩 |
| 2 trajectory_comprehension | `comprehend(` | `comprehend` | `adapters/cuga_rho_comprehender.py` | — | 🟩 |
| 3 difficulty_fingerprint | `hooks.judge(` | `judge` | `adapters/cuga_rho_judge.py` | Listing 2 | 🟩 |
| 4 coreset_selection | `coreset.py:197` | — (core) | `core/rho/coreset.py` | §4.1 | 🟩 |
| 5 group_rollouts | `rounds.py:622` | `rollout` | `adapters/cuga_adapter.py` | Listing 1 | 🟩 |
| 6 group_diagnosis | `diagnose(` | `diagnose` | `adapters/cuga_rho_diagnoser.py` | Listing 3 | 🟩 |
| 7 candidate_proposal | `propose(` | `propose` | `adapters/cuga_rho_optimizer.py` | Listing 4 | 🟨 SV-8 |
| 8 candidate_rollouts | `rounds.py:622` | `rollout` | same adapter, `R` per task | — | 🟩 |
| 9 preference_judging | `pipeline.py:1410` | `compare` | `cuga_preference_judge.compare_symmetric` | Listing 5 | 🟦 SV-7 |
| 10 pool_commit | `pipeline.py:1332` | `commit` | `core/pool.py:378 record_score` | Alg. 1 | 🟩 |

---

## 3. Genetic stage — the legacy GEPA loop

Lifecycle per `orchestrator.py:509 run_iteration`:

```
observe -> build_issues -> select_issues -> select_parent -> propose_edits
        -> validate -> commit_to_pool
```

```mermaid
flowchart TB
    subgraph GENETIC["GENETIC — pipeline.py:539 -> orchestrator.py:509 run_iteration"]
      direction TB
      G1["<b>observe</b> orchestrator.py:1304<br/>rollouts per task"]
      G2["<b>build_issues</b> :1427<br/>core/issues.py — trace-backed Issue"]
      G3["<b>select_issues</b> :1623 DPP<br/>quality = cross-candidate score VARIANCE"]
      G4["<b>select_parent</b> :1632<br/>pool.py:504 parent_frequencies()"]
      G5["<b>propose_edits</b> :1683<br/>core/editor.py -> adapters/cuga_editor.py"]
      G6["<b>validate</b> :478 _validate<br/>origin + worked + regression probes"]
      G7["<b>commit_to_pool</b><br/>only if accepted"]
      G1-->G2-->G3-->G4-->G5-->G6-->G7
    end

    G3 -.-> FG3["<b>coreset.py:11-13 WARNING</b><br/>RHO quality = judge difficulty<br/>GENETIC quality = score variance<br/><b>'Those two must not be unified'</b>"]
    G3 -.-> SV12["<b>SV-12 OPEN</b><br/>entropy starved: needs >=3 comparable<br/>candidates per cell; SV-11 prevents it"]
    G4 -.-> FG4["<b>pool.py:504</b> frequency(c) = SUM over won (t,m)<br/>of severity x confidence<br/>both inert 1.0 => counts cells won<br/>needs rollout_count >= min_comparable_rollouts"]
    G5 -.-> SV10["<b>SV-10 OPEN</b><br/>parent vulnerabilities never reach editor<br/>ParentContext.score_summary is lossy"]
    G6 -.-> FG6["<b>orchestrator.py:2043</b><br/>accept iff weighted_net_gain > net_gain_threshold (0.0)<br/>+ protected floors + retry budget<br/><b>SV-6 CLOSED:</b> retry budget now actually fires"]

    style FG3 fill:#f0f0f0
    style FG4 fill:#f0f0f0
    style FG6 fill:#f0f0f0
    style SV10 fill:#ffe6e6
    style SV12 fill:#ffe6e6
```

**Two different quality functions, deliberately.** `coreset.py:11-13` is explicit:
RHO's DPP quality is *judge-assigned difficulty*; genetic issue-selection quality
is *cross-candidate score variance*. They answer different questions and must not
be merged. This is also why `run_genetic` receives **coreset tasks only**:
variance needs a populated `(task, mechanism)` cell, and after a RHO round cells
exist only for the coreset. Off-coreset variance is *undefined*, not low.

---

## 4. The scoring tensor and every selection formula

```mermaid
flowchart TB
    subgraph CELL["ScoreCell — core/pool.py:90"]
      C1["scores: list[float] in [0,1]<br/>provenance: list[ScoreProvenance] (pool.py:53)"]
      C2["mean = sum(scores)/len(scores)  (0.0 if empty)<br/>severity = mean(p.severity)<br/>confidence = mean(p.confidence)"]
      C3["<b>weighted_score() = mean x severity x confidence</b><br/>pool.py:140"]
      C1-->C2-->C3
    end

    C3 --> INERT["<b>SV-1 RECLASSIFIED</b><br/>NO caller in src/ ever passes severity= or confidence=<br/>all 4 sites omit them: orchestrator.py:342, :1507, :1886, pipeline.py:1507<br/>frozen dataclass, no replace() path<br/><b>=> weighted_score() == mean, always</b>"]

    C3 --> OUT["<b>_champion_outcome</b> pool.py:563<br/>mean of per-task mean weighted scores<br/>rollout_count==0 cells SKIPPED (not zero)"]
    C3 --> PF["<b>parent_frequencies</b> pool.py:504"]
    C3 --> PAR["<b>pareto_frontier</b> :485 / <b>dominates</b> :462"]

    OUT --> AGG
    COV["<b>_champion_coverage</b> pool.py:570<br/>total = _observed_cells() :554<br/>coverage = |entry cells &cap; total| / |total|"] --> AGG
    STAB["<b>stability = 1.0</b> pool.py:657<br/>HARDCODED"] --> AGG
    RISK["<b>regression_risk = 0.0</b> pool.py:658<br/>HARDCODED"] --> AGG

    AGG["<b>select_champion</b> pool.py:577<br/>aggregate = 0.55*outcome + 0.20*coverage<br/>+ 0.15*stability - 0.10*regression_risk<br/>argmax, tie-break ascending candidate_id"]

    AGG --> EFF["<b>EFFECTIVE formula (SV-5)</b><br/>gamma*1.0 and delta*0.0 are CONSTANT for every entry<br/>=> they cancel in any comparison<br/><b>rank = 0.55*outcome + 0.20*coverage</b>"]

    style INERT fill:#ffe6e6
    style STAB fill:#ffdddd
    style RISK fill:#ffdddd
    style EFF fill:#ffdddd
    style C3 fill:#f0f0f0
```

### Eligibility gates applied BEFORE ranking — `pool.py:577`

```python
# 1. protected floors
if entry.candidate_id in protected_floor_violations:  continue

# 2. SV-4 RHO pairwise gate — ACTIVE BY DEFAULT
if gate_applied and not entry.is_base:
    if entry.preference is None or entry.preference <= 0.0:
        disqualified.add(entry.candidate_id); continue

# 3. coverage floor (default 0.0 = inactive)
if coverage < min_coverage_fraction:                  continue

if not scored: raise ValueError("no eligible candidates for champion selection")
```

**SV-4 gate semantics** (`config.py:146` `experimental_candidate_promotion=False`):

- **Strict `> 0`** — a measured tie is not evidence of improvement.
- **`preference is None` disqualifies** — no verdict means no evidence.
- **Base is exempt** — it is the comparison subject, and gating it would raise
  `ValueError` whenever nothing improved.
- **Promotion only** — pool membership untouched; all N candidates retained.
- `--experimental-candidate-promotion` disables it for an ablation arm.

### Where the preference score goes — SV-4 CLOSED

```mermaid
flowchart LR
    J["cuga_preference_judge.py:591<br/>compare_symmetric()<br/>2 judge calls/pair"]
    J --> CE["CandidateEvidence.mean_preference"]
    CE --> CM["pipeline.py:1332 commit()<br/>-> pool.py:349 record_preference()"]
    CM --> PE["PoolEntry.preference"]
    PE --> GATE["pool.py:577 select_champion()<br/><b>ELIGIBILITY GATE</b>"]
    GATE --> CR["ChampionReport.preference<br/>.preference_gate_applied (pool.py:288)"]
    J --> RS["RoundSummary.preference_mean<br/>rounds.py:306"]

    style GATE fill:#e8ffe8
    style CM fill:#e8ffe8
```

The paper's `S_j` now gates promotion. Judge wall-time is no longer spent on a
signal that only reached the report.

---

## 5. Wired vs NOT wired — verified by grep

```mermaid
flowchart LR
    subgraph WIRED["🟩 WIRED (reachable in a real run)"]
      W1["core/rho/* — all 10 phases"]
      W2["core/pool.py — record_score, record_preference,<br/>outcome, coverage, parent_frequencies,<br/>pareto_frontier, dominates, select_champion"]
      W3["core/entropy.py — PROTECTED, untouched"]
      W4["core/clustering.py + embeddings.py — coreset DPP"]
      W5["core/issues.py, blame.py, editor.py, evaluation.py"]
      W6["core/parallel.py + parallel_analysis.py"]
      W7["core/memory.py EditMemory — SV-6 fix<br/>pipeline.py:825, :1013"]
      W8["benchmarks/cuga_process_pool.py — worker recycling"]
      W9["benchmarks/cleanup.py — --cleanup-on-exit"]
    end

    subgraph DEAD["🟥 NOT WIRED (no production caller)"]
      D1["<b>core/merge.py</b> (393 L) = CROSSOVER<br/>plan_merge, compute_diff, ConflictReport,<br/>merge_respects_protected_floors<br/>only importer: tests/test_merge.py:8"]
      D2["<b>pool.prune()</b> pool.py:691<br/>only tests/test_pool.py:302,312,319"]
      D3["<b>stability / regression_risk</b> = SV-5<br/>specified in selection-algorithms.md:330-331<br/>hardcoded pool.py:657-658"]
      D4["<b>SequentialGepaRunner.run()</b> orchestrator.py:2160<br/>zero callers; run_evolution.py:1149 uses run_iterations"]
      D5["<b>ScoreProvenance.severity/.confidence</b> = SV-1<br/>never supplied by any production site"]
      D6["<b>X-AE-* correlation headers</b><br/>addon reads them; no caller emits them yet"]
    end

    style DEAD fill:#ffe6e6
    style WIRED fill:#e8ffe8
```

### Crossover is not implemented in any runnable path

`AGENTS.md` states crossover is *"provenance-preserving deterministic merge by
default."* `core/merge.py` implements exactly that. **Nothing in `src/` imports
it.** Verified:

```bash
grep -rn 'core\.merge' src/ tests/ scripts/ | grep -v 'core/merge.py'
#  tests/test_merge.py:8   <- only importer
```

So the genetic stage is **mutation-only**. `donor_count: int = 2`
(`orchestrator.py:957`) hands donors to an LLM editor as *context*
(`orchestrator.py:1683 propose_edits` → `select_parents(k=donor_count+1)`); it
does not run the deterministic merge planner.

---

## 6. Severe-issue register mapped onto the pipeline

Authoritative status: `docs/SEVERE-OPEN-ISSUES.md`.

```mermaid
flowchart TB
    subgraph CLOSED["🟩 CLOSED 2026-08-19"]
      S4["<b>SV-4</b> S_j > 0 gate active<br/>pool.py:577 + config.py:146<br/>tests/test_preference_gate.py (16)"]
      S6["<b>SV-6</b> runner owns EditMemory<br/>pipeline.py:825,1013<br/>tests/test_runner_edit_memory.py (13)"]
      S9["<b>SV-9</b> crashed traces excluded<br/>rounds.py:610 _answered()<br/>tests/test_crashed_rollout_exclusion.py (12)"]
      S1["<b>SV-1</b> RECLASSIFIED — not a perverse<br/>gradient; severity is inert<br/>pool.py:140 docs only, no behaviour change"]
    end

    subgraph OFFLINE["🟦 OPEN — offline-fixable"]
      S2["<b>SV-2</b> outcome averages over<br/>DIFFERENT task sets<br/>pool.py:563"]
      S3["<b>SV-3</b> coverage is not quality<br/>carries 27% of the decision<br/>pool.py:570"]
      S5["<b>SV-5</b> 2 of 4 objectives inert<br/>pool.py:657-658"]
      S10["<b>SV-10</b> parent vulnerabilities<br/>never reach editor<br/>orchestrator.py:1683"]
      S12["<b>SV-12</b> entropy structurally starved<br/>consequence of SV-11"]
    end

    subgraph GATED["🟨 OPEN — needs live proxy capture"]
      S7["<b>SV-7</b> NARROWED to MEDIUM<br/>judge + grid EXONERATED<br/>tests/test_judge_slot_distinctness.py (5)<br/>only upstream materialization remains"]
      S8["<b>SV-8</b> root cause FOUND<br/>list_artifacts() shows only existing surfaces<br/>bare HarnessVersion -> ['instructions'] only<br/>fix = multi-surface base seeding"]
      S11["<b>SV-11</b> SCOPED<br/>orchestrator.py:541 and :1441<br/>needs observation-budget decision"]
    end

    S11 --> S12
    S8 --> S7

    style CLOSED fill:#e8ffe8
    style OFFLINE fill:#d6eaff
    style GATED fill:#fff4cc
```

### The two live ranking defects, with reproduced numbers

| # | Issue | Locus | Evidence |
| --- | --- | --- | --- |
| **SV-2** | `outcome` averages over **different task sets** — no shared-cell restriction, so dropping a hard task raises your own mean | `pool.py:563` | base ran easy(0.9)+hard(0.1) → `0.500`; candA ran only easy(0.9) → `0.900`, **wins by skipping the hard task** |
| **SV-3** | `coverage` measures how much you *measured*, not how well. Exchange rate `cov 0.5→1.0` = `+0.100` aggregate = `0.100/0.55 = 0.18` of outcome | `pool.py:570` | `base-v0: cells=4 outcome=0.5500 coverage=1.0000 aggregate=0.6525`<br/>`cand-A:  cells=2 outcome=0.8500 coverage=0.5000 aggregate=0.7175` |

**SV-9 makes these more visible, correctly.** A crashed rollout now creates *no
cell* instead of bad evidence — so a candidate that crashes on hard tasks shrinks
its own outcome denominator. The fix is to repair outcome/coverage, **not** to
revert crash filtering.

**SV-1 correction to earlier drafts of this document.** Previous revisions listed
"severity is per-candidate" and "perverse gradient" as live defects A and B. Both
are **wrong in production**: `ScoreProvenance.severity` and `.confidence` are
never supplied by any of the four construction sites, so they hold `1.0` and
`weighted_score() == mean`. Two unrelated fields share the name `severity` —
`CausalAnalysis.severity` **does** influence diagnosis and targeting; the
`ScoreProvenance` one does not.

---

## 7. Paper-fidelity status of the prompts

`docs/research/rho-paper-prompt-fidelity.md`.

```mermaid
flowchart TB
    G1["🟩 GAP 1 — judge efficiency axis<br/>cuga_preference_judge.py"]
    G2["🟩 GAP 2 — consistency is reliability<br/>cuga_rho_diagnoser.py severity 0.4 band"]
    G3["🟩 GAP 3 — 4 named per-rollout findings<br/>cuga_rho_diagnoser.py"]
    G4["🟩 GAP 4 — 'with fewer wasted steps'<br/>cuga_rho_optimizer.py"]
    G5["🟨 GAP 5 OPEN — 1-call vs 2-call judge<br/>cost decision only"]
    G6["🟩 GAP 6 CLOSED — was S5-1/SV-4<br/>S_j > 0 gate now active"]
    G7["🟨 GAP 7 OPEN = SV-8<br/>only 'instructions' ever edited"]

    G2 -.-> NOTE["<b>Coupling now BENIGN</b><br/>earlier revisions warned that raising<br/>severity would raise a candidate's outcome<br/>SV-1 shows ScoreProvenance.severity is inert<br/>=> diagnoser severity cannot reach the aggregate"]

    style G1 fill:#e8ffe8
    style G2 fill:#e8ffe8
    style G3 fill:#e8ffe8
    style G4 fill:#e8ffe8
    style G6 fill:#e8ffe8
    style G5 fill:#fff4cc
    style G7 fill:#fff4cc
    style NOTE fill:#f0f0f0
```

**GAP 7 / SV-8 root cause, at code level.** `cuga_rho_optimizer.py:51` sets
`CREATABLE_PREFIX = "skills/generated-"` and the prompt documents all four
surfaces, so the *capability* exists. The blocker is upstream: `list_artifacts()`
exposes only **existing** artifacts, and a bare live
`HarnessVersion(instructions=...)` has empty `skills`, `memory`, `policies` — so
the optimizer is usually offered `["instructions"]` and nothing else. The offline
stack has multiple surfaces and therefore **does not reproduce** the live roster
condition. Correct first fix is multi-surface base seeding, then surface-history
awareness. There is also no executable-tool artifact class;
`cuga_editor_tools.py` is the *editor's own* toolset, not an evolvable surface.

`_CANDIDATE_FRAMINGS: tuple[str, ...] = ("",)` in `cuga_rho_optimizer.py` keeps
the N proposals prompt-identical and independent.

---

## 8. Operational subsystems (not in the algorithm, required to run it)

Built in response to the 90 GB memory-exhaustion incident
(`feedback/rho-memory-leak-report.md`) and the need for live interception.

```mermaid
flowchart TB
    subgraph MEM["🟩 Memory-exhaustion fixes — tests/test_memory_leak_fixes.py (19)"]
      M1["<b>Worker recycling</b> benchmarks/cuga_process_pool.py<br/>DEFAULT_MAX_ROLLOUTS_PER_WORKER = 25<br/>_recycle() replaces process<br/>CLI --max-rollouts-per-worker"]
      M2["<b>Agent teardown</b> adapters/cuga_workspace_agent.py:301-313<br/>await aclose() in finally; del agent; gc.collect()<br/>same in adapters/cuga_editor.py"]
      M3["<b>Bounded judge context</b> adapters/cuga_preference_judge.py<br/>_MAX_PAYLOAD_CHARS=2048, _MAX_RENDERED_EVENTS=120<br/>30 ev x 4 MB: 126 MB -> 63 KB (1954x)"]
      M4["<b>Out-of-heap cleanup</b> benchmarks/cleanup.py<br/>--cleanup-on-exit; dry-run by default<br/>Playwright-path processes only, never bare firefox"]
    end

    subgraph PROXY["🟩 Interactive proxy — docker/observability/"]
      P1["proxy.sh — up / run / tail / env / down"]
      P2["compose.yml — mitmproxy 11.0.0<br/>proxy 127.0.0.1:8082, UI 127.0.0.1:8083"]
      P3["addons/correlate.py — correlation, Authorization<br/>redaction, X-AE-* stripping, hot-reload mocks"]
      P4["mocks/rules.json (ignored)<br/>mocks/rules.example.json (committed)"]
      P5["🟥 X-AE-Candidate / -Task / -Rollout / -Phase / -Run<br/>addon reads them; NO caller emits them yet"]
    end

    style MEM fill:#e8ffe8
    style PROXY fill:#e8ffe8
    style P5 fill:#ffe6e6
```

**Rollout wrapper was never the agent leak.** `cuga_wrapper/__init__.py:2084`
already had `asyncio.run(agent.aclose())` in a `finally` before this work. The
worker path leaked by *reusing one wrapper across every rollout*, which is what
`--max-rollouts-per-worker` addresses. Note that site swallows exceptions
(`except Exception: pass`) — defensible, since cleanup must not mask rollout
evidence, but a consistently failing `aclose()` there would be invisible.

Current safe live-run shape:

```bash
set -a && . ./.env && set +a
rm -f .cuga/knowledge/.lock
./docker/observability/proxy.sh up          # UI http://127.0.0.1:8083 (pw: agentevolve)
./docker/observability/proxy.sh run -- python scripts/run_evolution.py --mode rho \
  --max-workers 6 --isolation process \
  --max-rollouts-per-worker 20 \
  --cleanup-on-exit
```

`--max-workers 10` or fewer, never 24. `--cleanup-on-exit` is destructive by
design: without it, cleanup only reports reclaimable resources.

---

## 9. End-to-end, one picture

```mermaid
flowchart TB
    START["scripts/run_evolution.py"] --> CFG["core/config.py ResolvedConfig<br/>alpha/beta/gamma/delta, k, G, N, R, rounds<br/>:146 experimental_candidate_promotion=False"]
    CFG --> MODE{"--mode"}

    MODE -->|rho| RHO10
    MODE -->|rho-genetic| RHO10
    MODE -->|genetic| GLOOP

    RHO10["core/rho/rounds.py:348 run_round<br/>P1..P10"] --> COMMIT["P10 pipeline.py:1332 commit<br/>ALL N committed, never best-of-N"]
    COMMIT --> POOL[("PersistentPool core/pool.py<br/>ScoreCell tensor (cand, task, mech)<br/>+ PoolEntry.preference")]
    COMMIT -->|"rho-genetic only<br/>coreset tasks only"| GLOOP

    GLOOP["orchestrator.py:509 run_iteration<br/>observe->issues->select->edit->validate"] --> POOL

    POOL --> ENT["core/entropy.py<br/>cross-candidate variance<br/>PROTECTED FILE"]
    ENT -.->|"feeds issue selection<br/>SV-12: starved"| GLOOP

    POOL --> G1{"protected floors?"}
    G1 -->|pass| G2{"SV-4 gate<br/>preference > 0?<br/>base exempt"}
    G2 -->|pass| G3{"coverage floor?"}
    G3 -->|pass| CH["pool.py:577 aggregate rank<br/>0.55*outcome + 0.20*coverage<br/>(+0.15 and -0.0 inert = SV-5)<br/><b>SV-2 / SV-3 live here</b>"]
    G2 -->|"None or <= 0"| DQ[["disqualified"]]

    CH --> EXP["pipeline.py:614 export_pool<br/>champion.json + candidate-*.json"]
    EXP --> NEXT["--harness for the NEXT run<br/>defects compound across chained runs"]

    XMERGE["core/merge.py CROSSOVER<br/>393 L, zero production callers"] -.->|NOT WIRED| GLOOP

    style POOL fill:#e8ffe8
    style COMMIT fill:#e8ffe8
    style G2 fill:#e8ffe8
    style CH fill:#d6eaff
    style ENT fill:#d6eaff
    style XMERGE fill:#ffe6e6
```

---

## 10. Summary — what to trust

**🟩 Solid and wired.** All 10 RHO phases. The score tensor and its provenance
discipline (`rollout_count==0` is not a zero; `rollout_seq` must be the next
slot). Persistent-pool retention of all N candidates. The two-quality-function
separation. Entropy. Coreset DPP with a documented quality-only fallback. The
SV-4 pairwise acceptance gate. Crash-filtered RHO evidence (SV-9). Runner-owned
edit memory with a live retry budget (SV-6). Worker recycling, agent teardown,
and bounded judge context.

**🟦 Wired but semantics disputed.** `select_champion` ranking. Two reproduced
defects remain — SV-2 (`outcome` averaged over different task sets) and SV-3
(`coverage` scored as quality). Both are *ranking* defects; acceptance is now
gated separately by SV-4. It decides which harness gets **exported and carried
into the next run**.

**🟨 Partially implemented.** Multi-surface harness editing (SV-8 — capability
present, roster starves it). Mechanism analysis of candidates (SV-11 — needs an
observation-budget decision). Judge trajectory distinctness (SV-7 — judge and
grid exonerated; only upstream materialization unresolved).

**🟥 Not wired at all.** `core/merge.py` — **crossover does not run**, despite
being an architecture decision in `AGENTS.md`. Also `pool.prune`,
`SequentialGepaRunner.run`, two of the four champion objectives (SV-5), the
`ScoreProvenance` severity/confidence weights (SV-1), and `X-AE-*` correlation
headers at call sites.

### Recommended order

1. **SV-8** multi-surface base seeding — unblocks SV-7's remaining branch and is
   a precondition for meaningful proposals. *Needs a decision on what to seed.*
2. **SV-11** observation budget — parent-only is cost-neutral. *Needs a decision.*
3. **Emit `X-AE-*` headers**, then one bounded live proxy-captured RHO smoke to
   settle SV-7 upstream materialization and the live roster claims.
4. **SV-2** then **SV-3** — shared comparable-cell denominator; coverage becomes
   evidence *eligibility*, not a quality reward.
5. **SV-5** — delete or implement `stability` / `regression_risk`.
6. **SV-10**, then **SV-12** (largely a consequence of SV-11).

All of 4–5 change selection semantics and none should be made without an explicit
decision.

---

## Appendix — durable grep anchors

| Concept | Anchor |
| --- | --- |
| Mode → phase table | `grep -n "^PHASES" src/agent_evolve/core/rho/rounds.py` |
| RHO round body | `grep -n "def run_round" src/agent_evolve/core/rho/rounds.py` |
| Crashed-trace filter | `grep -n "def _answered\|ANSWERED_TRACE_STATUSES" src/agent_evolve/core/rho/rounds.py` |
| Champion selection | `grep -n "def select_champion" src/agent_evolve/core/pool.py` |
| SV-4 gate | `grep -n "experimental_candidate_promotion" src/agent_evolve/core/{pool,config}.py` |
| Inert weights | `grep -rn "ScoreProvenance(" src/` |
| Outcome / coverage | `grep -n "_champion_outcome\|_champion_coverage\|_observed_cells" src/agent_evolve/core/pool.py` |
| Hardcoded objectives | `grep -n "stability = 1.0\|regression_risk = 0.0" src/agent_evolve/core/pool.py` |
| Genetic lifecycle | `grep -n "def run_iteration\|def observe\|def build_issues\|def select_issues\|def select_parent\|def propose_edits\|def _validate" src/agent_evolve/core/orchestrator.py` |
| Crossover deadness | `grep -rn "core\.merge" src/ tests/ scripts/` |
| Symmetric judge | `grep -n "def compare_symmetric" src/agent_evolve/adapters/cuga_preference_judge.py` |
| Judge context bounds | `grep -n "_MAX_PAYLOAD_CHARS\|_MAX_RENDERED_EVENTS" src/agent_evolve/adapters/cuga_preference_judge.py` |
| Worker recycling | `grep -n "MAX_ROLLOUTS_PER_WORKER\|def _recycle" src/agent_evolve/benchmarks/cuga_process_pool.py` |
| Agent teardown | `grep -n "aclose" src/agent_evolve/adapters/cuga_{workspace_agent,editor}.py` |
| Export path | `grep -n "def export_pool\|def champion_version" src/agent_evolve/pipeline.py` |
