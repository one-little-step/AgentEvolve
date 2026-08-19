# Implemented Pipeline Map — what is actually wired, 2026-08-19

Derived by reading the code and by executing it, not from the design docs. Where
code and design doc disagree, the **code** is reported here and the divergence is
flagged.

> **First, a correction to a premise.** `select_champion` is **not** part of the
> genetic stage. It lives in `core/pool.py:482` and is called from exactly three
> places, none of them inside the genetic loop:
>
> ```
> core/orchestrator.py:2117   SequentialGepaRunner.run()   <- multi-attempt entry, NOT used by run_evolution.py
> pipeline.py:609             champion_version()           <- reporting
> pipeline.py:632             export_pool()                <- writing champion.json
> ```
>
> It is a **post-hoc reporting/export selector over the whole pool**, shared by
> all three modes. The genetic loop never consults it, and `run_round` never calls
> it — `rounds.py:561` says *"Rank orders the report and picks a champion; it never
> decides survival."* So the S5-1 finding is about **which harness you export and
> carry into the next run**, not about survival inside a round.

---

## 1. Three modes, one code path

`core/rho/rounds.py:69`

```python
PHASES = {
  "rho":         _RHO_PHASES,                              # 10 phases
  "genetic":     ("genetic_iterations",),                  # legacy GEPA loop only
  "rho-genetic": _RHO_PHASES + ("genetic_iterations",),    # RHO then genetic
}
```

```mermaid
flowchart TB
    CLI["scripts/run_evolution.py<br/>--mode {rho | genetic | rho-genetic}"]

    CLI -->|"mode != genetic<br/>line 961-982"| RR["core/rho/rounds.py<br/>run_rounds() -> run_round()"]
    CLI -->|"mode == genetic<br/>line 1095"| GEN["pipeline.py:539<br/>stack.run_iterations(n)"]

    RR --> P1_10["RHO phases 1..10"]
    P1_10 -->|"only if 'genetic_iterations' in phases<br/>rounds.py:572"| GHOOK["_run_genetic()<br/>rounds.py:701"]
    GHOOK -->|"hooks.run_genetic(coreset_tasks, iters)"| GEN

    GEN --> POOL[("PersistentPool<br/>core/pool.py")]
    P1_10 --> POOL
    POOL --> EXPORT["pipeline.py:614 export_pool()<br/>calls select_champion()"]

    style RR fill:#d6eaff
    style GEN fill:#ffe9cc
    style POOL fill:#e8ffe8
    style EXPORT fill:#ffd6d6
```

**Key wiring fact:** the genetic phase is not a reimplementation. `pipeline.py:1334`
narrows `stack.tasks` to the coreset, calls the *same* `run_iterations`, and
restores it in `finally`. The comment states why: *"byte-for-byte the loop that
produced the measured baseline."*

---

## 2. RHO stage — 10 phases, with file mapping and formulas

```mermaid
flowchart TB
    subgraph RHO["RHO ROUND — core/rho/rounds.py:348 run_round()"]
      direction TB

      H1["<b>P1 history_load</b> :383<br/>hooks.load_history()<br/><i>adapters/cuga_rho_comprehender.py</i>"]
      H2["<b>P2 trajectory_comprehension</b> :406<br/>hooks.comprehend(record)<br/><i>adapters/cuga_rho_comprehender.py</i>"]
      H3["<b>P3 difficulty_fingerprint</b> :420<br/>hooks.judge(record, summary)<br/><i>adapters/cuga_rho_judge.py</i><br/>paper Listing 2"]
      H4["<b>P4 coreset_selection</b> :430<br/><i>core/rho/coreset.py:197 select_coreset()</i>"]
      H5["<b>P5 group_rollouts</b> :467<br/>k x G on INCUMBENT<br/><i>_rollout_grid :610</i>"]
      H6["<b>P6 group_diagnosis</b> :475<br/>hooks.diagnose(task, traces)<br/><i>adapters/cuga_rho_diagnoser.py</i><br/>paper Listing 3"]
      H7["<b>P7 candidate_proposal</b> :494<br/>N independent invocations<br/><i>adapters/cuga_rho_optimizer.py</i><br/>paper Listing 4"]
      H8["<b>P8 candidate_rollouts</b> :518<br/>k x R per candidate"]
      H9["<b>P9 preference_judging</b> :526<br/>compare_symmetric()<br/><i>adapters/cuga_preference_judge.py</i><br/>paper Listing 5"]
      H10["<b>P10 pool_commit</b> :560<br/><b>ALL N committed, never best-of-N</b>"]

      H1-->H2-->H3-->H4-->H5-->H6-->H7-->H8-->H9-->H10
    end

    H4 -.->|"quality x diversity"| F4["<b>coreset.py:124</b><br/>normalized = max(difficulty/MAX_DIFFICULTY, score_floor)<br/>quality = normalized ** theta<br/>selector: dpp | difficulty_rank | random"]
    H9 -.-> F9["<b>cuga_preference_judge.py:591 compare_symmetric</b><br/>score = (fwd - rev)/2<br/>position_bias = (fwd + rev)/2<br/><b>2 judge calls per pair</b>"]
    H10 -.-> F10["<b>pipeline.py:1466</b> _record_pool_score<br/>clamp(value, 0, 1)<br/>RHO path: severity=1.0 confidence=1.0 (defaults)<br/>blame_confidence=0.0 blame_stability=0.0 (honest zero)"]

    style F4 fill:#fff4cc
    style F9 fill:#fff4cc
    style F10 fill:#fff4cc
    style H10 fill:#e8ffe8
```

### Phase-to-file table

| Phase | rounds.py line | Hook | Implementation | Paper |
| --- | --- | --- | --- | --- |
| 1 history_load | 383 | `load_history` | `core/rho/history.py` + `adapters/cuga_rho_comprehender.py` | — |
| 2 trajectory_comprehension | 406 | `comprehend` | `adapters/cuga_rho_comprehender.py` (600 L) | — |
| 3 difficulty_fingerprint | 420 | `judge` | `adapters/cuga_rho_judge.py` (541 L) | Listing 2 |
| 4 coreset_selection | 430 | — (core) | `core/rho/coreset.py:197` | §4.1 |
| 5 group_rollouts | 467 | `rollout` | `adapters/cuga_adapter.py` | Listing 1 |
| 6 group_diagnosis | 475 | `diagnose` | `adapters/cuga_rho_diagnoser.py` (660 L) | Listing 3 |
| 7 candidate_proposal | 494 | `propose` | `adapters/cuga_rho_optimizer.py` (795 L) | Listing 4 |
| 8 candidate_rollouts | 518 | `rollout` | same adapter, `R` per task | — |
| 9 preference_judging | 526 | `compare` | `adapters/cuga_preference_judge.py` (703 L) | Listing 5 |
| 10 pool_commit | 560 | `commit` | `core/pool.py` `record_score` | Alg. 1 |

---

## 3. Genetic stage — the legacy GEPA loop

Lifecycle per `orchestrator.py:865`:

```
observe -> build_issues -> select_issues -> select_parent -> propose_edits
        -> validate -> commit_to_pool
```

```mermaid
flowchart TB
    subgraph GENETIC["GENETIC — pipeline.py:539 run_iterations -> orchestrator.py:509 run_iteration"]
      direction TB
      G1["<b>observe</b><br/>base rollouts G per task<br/><i>orchestrator.py:530-570</i>"]
      G2["<b>build_issues</b><br/><i>core/issues.py (741 L)</i><br/>trace-backed Issue"]
      G3["<b>select_issues</b> DPP<br/><i>core/issues.py</i><br/>quality = cross-candidate score VARIANCE"]
      G4["<b>select_parent</b><br/><i>pool.py parent_frequencies()</i><br/>seeded RNG, proportional"]
      G5["<b>propose_edits</b><br/><i>core/editor.py (579 L)</i><br/>-> adapters/cuga_editor.py (393 L)"]
      G6["<b>validate</b> :478 _validate<br/>origin + worked + regression probes"]
      G7["<b>commit_to_pool</b><br/>only if accepted"]
      G1-->G2-->G3-->G4-->G5-->G6-->G7
    end

    G3 -.-> FG3["<b>coreset.py:11-13 WARNING</b><br/>RHO quality = judge difficulty<br/>GENETIC quality = score variance<br/><b>'Those two must not be unified'</b>"]
    G4 -.-> FG4["<b>pool.py parent_frequencies</b><br/>frequency(c) = SUM over winning (t,m) of severity x confidence<br/>c wins (t,m) iff strict max comparable weighted_score<br/>ties award ALL tied winners"]
    G6 -.-> FG6["<b>orchestrator.py:2029</b><br/>accept iff weighted_net_gain > net_gain_threshold (default 0.0)<br/>+ protected floors + retry budget"]

    style FG3 fill:#ffdddd
    style FG4 fill:#fff4cc
    style FG6 fill:#fff4cc
```

**Two different quality functions, deliberately.** `coreset.py:11-13` is explicit:
RHO's DPP quality is *judge-assigned difficulty*; genetic issue-selection quality
is *cross-candidate score variance*. They answer different questions and must not
be merged. This is also why `run_genetic` receives **coreset tasks only**
(`rounds.py` docstring): variance needs a populated `(task, mechanism)` cell, and
after a RHO round cells exist only for the coreset. Off-coreset variance is
*undefined*, not low.

---

## 4. The scoring tensor and every selection formula

```mermaid
flowchart TB
    subgraph CELL["ScoreCell — core/pool.py:88"]
      C1["scores: list[float] in [0,1]<br/>provenance: list[ScoreProvenance]"]
      C2["mean = sum(scores)/len(scores)&nbsp;&nbsp;(0.0 if empty)<br/>severity = mean(p.severity)<br/>confidence = mean(p.confidence)"]
      C3["<b>weighted_score() = mean x severity x confidence</b><br/>pool.py:140"]
      C1-->C2-->C3
    end

    C3 --> OUT["<b>outcome</b> pool.py:466<br/>per task: mean of that task's cell weighted_scores<br/>outcome = mean of per-task values<br/>rollout_count==0 cells SKIPPED (not zero)"]
    C3 --> PF["<b>parent_frequencies</b><br/>needs rollout_count >= min_comparable_rollouts (2)"]
    C3 --> PAR["<b>pareto_frontier / dominates</b><br/>pool.py:390,407"]

    OUT --> AGG
    COV["<b>coverage</b> pool.py:474<br/>total_cells = union of ALL entries' cells with rollout_count>=1<br/>coverage = |entry cells &cap; total| / |total|"] --> AGG
    STAB["<b>stability = 1.0</b><br/>HARDCODED, never computed"] --> AGG
    RISK["<b>regression_risk = 0.0</b><br/>HARDCODED, never computed"] --> AGG

    AGG["<b>select_champion</b> pool.py:482<br/>aggregate = 0.55 x outcome + 0.20 x coverage<br/>&nbsp;&nbsp;&nbsp;&nbsp;+ 0.15 x stability - 0.10 x regression_risk<br/>argmax, tie-break ascending candidate_id"]

    AGG --> EFF["<b>EFFECTIVE formula</b><br/>gamma x 1.0 and delta x 0.0 are CONSTANT for every entry<br/>=> they cancel in any comparison<br/><b>rank = 0.55 x outcome + 0.20 x coverage</b>"]

    style STAB fill:#ffdddd
    style RISK fill:#ffdddd
    style EFF fill:#ffdddd
    style C3 fill:#fff4cc
```

### Disqualifiers applied BEFORE ranking

```python
# pool.py:482 select_champion
if entry.candidate_id in protected_floor_violations:  continue   # disqualified
if coverage < min_coverage_fraction:                  continue   # disqualified
if not scored: raise ValueError("no eligible candidates")
```

---

## 5. Wired vs NOT wired — verified by grep + execution

```mermaid
flowchart LR
    subgraph WIRED["WIRED (reachable in a real run)"]
      W1["core/rho/* — all 10 phases"]
      W2["core/pool.py — record_score, outcome, coverage,<br/>parent_frequencies, pareto_frontier, dominates"]
      W3["core/entropy.py — issues.py, rho/rounds.py, orchestrator.py"]
      W4["core/clustering.py + embeddings.py — coreset DPP"]
      W5["core/issues.py, blame.py, editor.py, evaluation.py"]
      W6["core/parallel.py + parallel_analysis.py — orchestrator:124"]
      W7["adapters/cuga_editor_tools.py + _skills.py<br/>via cuga_editor.py:368,371 (GENETIC editor only)"]
    end

    subgraph DEAD["NOT WIRED (no production caller)"]
      D1["<b>core/merge.py</b> (393 L)<br/>plan_merge, compute_diff, ConflictReport,<br/>mechanisms_are_complementary,<br/>merge_respects_protected_floors<br/>imported ONLY by tests/test_merge.py"]
      D2["<b>pool.prune()</b><br/>called ONLY by tests/test_pool.py"]
      D3["<b>stability / regression_risk</b><br/>specified in selection-algorithms.md:330-331<br/>hardcoded constants in code"]
      D4["<b>SequentialGepaRunner.run()</b> orchestrator:2104<br/>run_evolution.py uses run_iterations, not run()"]
      D5["<b>cache_hits['embedding']</b><br/>pipeline.py comment: 'nothing reads it yet'"]
    end

    style DEAD fill:#ffe6e6
    style WIRED fill:#e8ffe8
```

### CROSSOVER IS NOT IMPLEMENTED IN ANY RUNNABLE PATH

`AGENTS.md` states crossover is *"provenance-preserving deterministic merge by
default."* `core/merge.py` implements exactly that — 393 lines, `plan_merge`,
`ConflictResolution`, `merge_respects_protected_floors`. **Nothing in `src/`
imports it.** Verified:

```bash
grep -rn 'core.merge' src/ tests/ scripts/ | grep -v 'core/merge.py:'
#  tests/test_merge.py:8   <- only caller
```

`orchestrator.py:867` confirms scope: *"Merge, parallel batching, RHO proposal
generation, tracing, checkpoints, and replay are explicitly out of scope for
Phase 6."* So the genetic stage is **mutation-only**. The `donor_count: int = 2`
field (`orchestrator.py:947`, "Donor parents offered to the editor alongside the
primary") hands donors to an LLM editor as *context*; it does not run the
deterministic merge planner.

---

## 6. Known defects in the selection math

All four reproduced by executing the real `PersistentPool`.

| # | Issue | Evidence |
| --- | --- | --- |
| **A** | **Severity is per-candidate, contradicting the spec.** `selection-algorithms.md:295` defines `weighted = score(c,t,m) * severity(t,m) * confidence(c,t,m)` — severity indexed by `(t,m)` **only**. `ScoreCell.severity` docstring agrees: *"a property of the (task, mechanism) pair… constant within a cell."* But `orchestrator.py:462,1405` write `severity=analysis.severity`, the diagnoser's judgment **of that candidate**. | Two candidates both scoring a perfect `1.0`: `sev=0.2` → outcome `0.200`; `sev=0.9` → outcome `0.900`. **The more alarming-looking candidate wins on identical performance.** |
| **B** | **Perverse gradient.** Because severity multiplies the score and a *good* candidate makes the diagnoser report *low* severity, fixing a problem shrinks your own multiplier. | base fails all 3 (`0.0`, sev `1.0`) → outcome `0.0000`, agg `0.3500`. Candidate aces all 3 (`1.0`, sev `0.1`) → outcome `0.1000`, agg `0.4050`. A perfect candidate scores **0.1**, winning by 0.055 — reversible by one coverage cell. |
| **C** | **Coverage is not quality.** It measures how much you *measured*. Exchange rate: `cov 0.5→1.0` = `+0.100` aggregate = `0.100/0.55 = 0.18` of outcome. **A candidate can be 0.18 worse in weighted score and still win.** Structural, since base gets `G` rollouts and post-RHO candidates get `R` per task by design. | base `outcome=0.600 cov=0.500 agg=0.5800`; candB `outcome=0.550 cov=1.000 agg=0.6525` → **worse candidate exported as champion**. |
| **D** | **`outcome` averages over different task sets.** No shared-cell restriction. | base ran easy(0.9)+hard(0.1) → `outcome 0.500`. candA ran only easy(0.9), identical where compared → `outcome 0.900`, **wins by skipping the hard task**. |

Tracked as **S5-1** (no `S_j > 0` acceptance gate) and **S5-2** (crashed rollouts)
in `docs/OPEN-ISSUES.md`. Items A–D above are the fuller picture.

### Where the preference score goes

```mermaid
flowchart LR
    J["compare_symmetric()<br/>2 judge calls/pair"] --> RS["RoundSummary.preference_mean<br/>rounds.py:596"]
    RS --> REP["reporting / manifest"]
    J -.->|"NEVER"| SC["select_champion()"]
    SC -.- NOTE["signature has NO parameter for it<br/>verified: 'preference','mean_score',<br/>'is_base','base','S_j' all absent from source"]

    style SC fill:#ffdddd
    style NOTE fill:#ffe6e6
```

The paper's `S_j` — the mean oriented preference — **is computed** (`pipeline.py:1662`,
`rounds.py:596`) and **is never consulted by champion selection**. Roughly half the
aborted run's wall time went to judge calls that inform the RHO round's report but
not the exported champion.

---

## 7. Paper-fidelity status of the prompts

`docs/research/rho-paper-prompt-fidelity.md` — GAPs 1–4 implemented 2026-08-19.

```mermaid
flowchart TB
    G1["GAP 1 DONE — judge efficiency axis<br/>cuga_preference_judge.py<br/>correctness-gated, retries excluded"]
    G2["GAP 2 DONE — consistency is reliability<br/>cuga_rho_diagnoser.py severity 0.4 band"]
    G3["GAP 3 DONE — 4 named per-rollout findings<br/>cuga_rho_diagnoser.py"]
    G4["GAP 4 DONE — 'with fewer wasted steps'<br/>cuga_rho_optimizer.py"]
    G5["GAP 5 OPEN — 1-call vs 2-call judge<br/>cost decision"]
    G6["GAP 6 OPEN — S_j > 0 gate = S5-1"]
    G7["GAP 7 OPEN — full harness<br/>only 'instructions' ever edited"]

    G2 -.->|"severity flows into weighted_score<br/>=> into outcome => into champion"| WARN["<b>COUPLING</b><br/>raising severity for mere inconsistency<br/>now RAISES that candidate's outcome<br/>see defect A/B"]

    style G1 fill:#e8ffe8
    style G2 fill:#e8ffe8
    style G3 fill:#e8ffe8
    style G4 fill:#e8ffe8
    style G5 fill:#fff4cc
    style G6 fill:#ffdddd
    style G7 fill:#ffdddd
    style WARN fill:#ffdddd
```

**GAP 7 confirmed at the code level.** `cuga_rho_optimizer.py:51` sets
`CREATABLE_PREFIX = "skills/generated-"` and the prompt documents all four
surfaces (`instructions`, `skills/`, `policies/`, `memory/`), so the *capability*
exists. But no run has produced a candidate editing anything except
`instructions`, and there is **no executable-tool artifact class** — the paper's
harness contains real executables (`bin/repair-verify`,
`tools/validate_mask_csv.py`). `cuga_editor_tools.py` is the *editor's own*
toolset, not an evolvable surface.

---

## 8. End-to-end, one picture

```mermaid
flowchart TB
    START["scripts/run_evolution.py"] --> CFG["core/config.py ResolvedConfig<br/>alpha/beta/gamma/delta, k, G, N, R, rounds"]
    CFG --> MODE{"--mode"}

    MODE -->|rho| RHO10
    MODE -->|rho-genetic| RHO10
    MODE -->|genetic| GLOOP

    RHO10["core/rho/rounds.py run_round<br/>P1..P10"] --> COMMIT["P10: commit ALL N<br/>never best-of-N"]
    COMMIT --> POOL[("PersistentPool<br/>ScoreCell tensor<br/>(cand, task, mech)")]
    COMMIT -->|"rho-genetic only<br/>coreset tasks only"| GLOOP

    GLOOP["orchestrator.py run_iteration<br/>observe->issues->select->edit->validate"] --> POOL

    POOL --> ENT["core/entropy.py<br/>cross-candidate variance<br/>PROTECTED FILE"]
    ENT -.->|"feeds issue selection"| GLOOP

    POOL --> CH["select_champion pool.py:482<br/>0.55 outcome + 0.20 coverage<br/>(+0.15 and -0.0 inert)"]
    CH --> EXP["pipeline.py:614 export_pool<br/>champion.json + candidate-*.json"]
    EXP --> NEXT["--harness for the NEXT run"]

    JUDGE["preference S_j"] -.->|"reporting only"| RPT["RoundSummary"]
    JUDGE -.->|"NOT connected"| CH

    style POOL fill:#e8ffe8
    style CH fill:#ffdddd
    style ENT fill:#d6eaff
    style COMMIT fill:#e8ffe8
```

---

## 9. Summary — what to trust

**Solid and wired:** all 10 RHO phases; the score tensor and its provenance
discipline (`rollout_count==0` is not a zero; `rollout_seq` must be the next slot);
persistent-pool retention of all N candidates; the two-quality-function separation;
entropy; coreset DPP with a documented quality-only fallback.

**Wired but mathematically wrong:** `select_champion`. Four independent defects
(A–D), each reproduced. It decides which harness gets **exported and carried into
the next run**, so a chained multi-run experiment compounds the error.

**Not wired at all:** `core/merge.py` — i.e. **crossover does not run**, despite
being an architecture decision in `AGENTS.md`. Also `pool.prune`,
`SequentialGepaRunner.run`, and 2 of the 4 champion objectives.

**Recommended order:** (A) severity → per-task, smallest change and it removes the
perverse gradient; (D) restrict `outcome` to shared cells; (C) make coverage an
eligibility gate rather than a scored term — subsumes S5-1; (2) delete or implement
`stability`/`regression_risk`. All are changes to selection semantics and none
should be made without an explicit decision.
