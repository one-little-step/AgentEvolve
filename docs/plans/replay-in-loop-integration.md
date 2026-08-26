# Replay-in-Loop Integration — Wiring Plan & Tracking

**Status:** PLANNING COMPLETE 2026-08-26 · not yet started
**Purpose:** single source of truth for wiring the proven replay/tape/editor
capabilities into the *evolution pipeline itself* (`scripts/run_evolution.py`
→ `agent_evolve.pipeline` → `SequentialGepaRunner`). Written so a context
switch mid-ladder loses nothing: every fact below was verified in-session
against the installed tree (cuga 0.3.1, suite 2203).

**Companion documents:**
* [`docs/design/recorded-prefix-replay.md`](../design/recorded-prefix-replay.md)
  — settled design R1–R9 (incl. R5 volatility amendment)
* [`docs/plans/node-level-replay-implementation.md`](node-level-replay-implementation.md)
  — Phases 1–6, all DONE (✓86–✓93)

---

## 0. One-paragraph state of the world

Every capability needed for cheap editor experimentation exists and is
proven: complete trace capture (Phase 1–2), tape reload + re-drive verified
byte-faithful by the STRICT crown (`verified_normalized`, ~50 s vs 443 s,
zero spend), hybrid taped-prefix/live-tail execution, MUTATION-mode artifact
injection, and a REAL `CugaEditorAgent` turn producing a working skill
mutation. **None of it is reachable from the evolution pipeline**: all
orchestration lives in hand-driven scripts. This document is the bridge.

## 1. As-is pipeline map (verified against source)

```
scripts/run_evolution.py main()
 ├─ build_live_stack(args…)                    [pipeline.py — ALL wiring lives here]
 │    ├── CugaExecutor + workers               (traces written under --trace-root;
 │    │                                         workers rebuild their OWN wrapper,
 │    │                                         so env vars are the only config
 │    │                                         channel that reaches them)
 │    ├── analyzer/judge adapters              (_litellm_completion copies ×5)
 │    ├── CugaEditorAgent                      (wired incl. complement tool ✓81)
 │    └── SequentialGepaRunner                 (pool, score tensor, entropy, blame)
 ├─ stack.measure(base, prefix="before")       FULL-price rollouts
 ├─ stack.run_iterations(N)   or  _run_rho_rounds()
 │      iteration ≈ build_issues(diagnose) → editor.propose_edit()
 │                     → candidate workspace → validate() FULL rollouts
 │                     → commit_to_pool
 ├─ stack.measure(champion, prefix="after")
 └─ stack.export_pool(path)                    only survivor unless flagged
```

Already wired and working: capture-side tracing (every rollout emits
`causal-trace.json` + `payloads/` under the executor's trace root),
correlation headers, retry hardening Layers 1+2 (workers inherit via
`configure_cuga_environment`), response-cache guard, editor + complement
tooling, blame graphs, entropy selection, budgets, process-isolation rules.

## 2. Capability inventory (exists, tested, UNREACHABLE from the loop)

| Capability | Lives in | Proven by | Production callers |
|---|---|---|---|
| TapeIndex (load trace, resolve refs, lazy sha256) | `core/tape.py` | tests/test_tape_index.py 13 | **none** |
| TapeModel (full taped re-drive) | `cuga_wrapper/tape_replay.py` | test_tape_model.py 12 + crown run | **none** |
| HybridTapeModel (taped prefix → live tail; MUTATION gate-open; pointer accounting fixed) | same | test_hybrid_tape_model.py 7 + tail-experiment-v2 | **none** |
| Volatility scrub registry (R5 amendment) | driver-side regex tuple | TestVolatilityScrubbing | **none** |
| LLM seam: `LLMManager.set_llm(tape_model)` | cuga models.py:1383 honors pre-instantiated for EVERY agent | crown verdict `verified_normalized` | scripts only |
| Resume-point selection | manual (hand-picked boundary 3/4) | — | does not exist |
| Editor→mutation extraction | `EditorResponse.writes` dict | editor-experiment/editor-plan.json | wired for full-validation path only |

Structural warning: this is the exact situation `core/merge.py` lived in —
built, tested, unreachable. §11-style AST checks exist because this state is
easy to miss.

## 3. The five gaps

### Gap 1 — trace→pool provenance is dropped
The wrapper result carries `causal_trace_path`; the executor/pipeline discard
it. `ScoreProvenance` records `trace_id` (an id string), never a location.
After an iteration the loop cannot answer *"where is the parent's tape?"*
**Fix:** carry the path through the adapter into the rollout record /
provenance (plain string field; core stays neutral). Evolution rollouts
already persist under repo `data/` paths (tracked); ad-hoc script runs under
gitignored `terminal_output/` (see Open Q3 below).

### Gap 2 — tape machinery unused (covered by inventory table)
**Fix:** consumed automatically once Gaps 3–5 close.

### Gap 3 — no resume-point mapper
Resume selection is manual ("boundary 3 of 4" chosen by reading the trace).
The loop has the **blame graph** instead. Missing pure function:
`boundary_for_fault(tape_index, analysis) -> int` — walk `NodeStart`s, match
blamed actor's failing cycle, return preceding LLM-boundary index.
Edge cases to handle/test: multiple blamed nodes (take max-blame);
fault before first boundary (nothing to tape → fall through to full
validation); fault at last boundary (tail≈whole run → replay pointless, fall
through); subgraph nesting via `parent_event_id`.

### Gap 4 — no experiment entrypoint
Missing: `runner.experiment_with_replay(parent_trace_dir, artifacts, resume=None)`
returning a structured report:
`{verdict: cleared|persisted|inconclusive, gates:{raw,scrubbed}, final_output_diff_summary, live_calls, elapsed}`.
Internals (all already demonstrated manually): HybridTapeModel via
`LLMManager.set_llm`; gate **on** for control arms, deliberately **open**
for MUTATION arms (an injected skill legitimately changes prompts from call
one); fresh workspace per R7; scrub registry applied identically both sides;
`AE_TAPE_DEBUG=1` instrumentation available.

### Gap 5 — no pre-filter stage or policy hook between `propose_edit()` and `validate()`
See §5 below for the full policy design. Summary: the insertion point,
demote-only authority, budget counting, and evidence isolation all need to
exist as code, not convention.

## 4. Wiring ladder (dependency order; each step offline-testable)

| Step | Deliverable | Done when | Unblocks |
|---|---|---|---|
| W1 | Provenance plumbing: `trace_dir` persisted onto rollout record + cell provenance at capture time | red→green test: after `_record_rollout_score` of a traced rollout, cell exposes the trace location; AST/grep shows wrapper result path no longer discarded | everything |
| W2 | `boundary_for_fault` mapper in `cuga_wrapper` (pure fn) + unit tests incl. all four edge cases above | tests green against synthetic fixture mirroring crown-trace shape; real-trace integration test skipif absent | W3 |
| W3 | `experiment_with_replay(...)` on the runner (or adapter facade) returning the structured report; HybridTapeModel injected **inside whichever process executes**; correlation scope tagged `phase="replay"` | offline tests: control arm gates stay on, mutated arm gate-open, live_calls counted (pointer fix regression holds), report contract stable | W4 |
| W4 | Pre-filter hook in `run_iterations()` behind `AE_REPLAY_EXPERIMENTS=1`; env-channel propagation to process workers (precedent: `ALLOW_RESPONSE_CACHE_ENV` set pre-fork); budget counter extension | flag-off run byte-identical to today (pin test); flag-on dry-run exercises hook with fakes; budget refuses when exhausted | W5 |
| W5 | Live in-loop demonstration: F1 scenario triggered by the pipeline itself (editor turn → replay filter → invited validation), tee'd log + report artifacts | one end-to-end run with all reports persisted; comparison vs manual-run baseline | DONE |

Spend note: W1–W4 zero provider calls (fakes/synthetic). W5 small (~editor
turn + 1–2 live tail calls).

## 5. The pre-filter policy (Gap 5 in full)

Insertion point: inside `run_iterations()`, immediately after
`propose_edit()` returns writes, before candidate materialization/validation.

Decision procedure (the hook owns ALL of these):

1. **Eligibility.** Run an experiment only when: parent trace dir known
   (Gap 1), trace passes `verify_all_refs()` (integrity), classifier reports
   zero `unclassified` tools, mapper returns a usable boundary. Otherwise
   **fall through silently-but-logggedly to full validation** — honest
   fallback, never a skipped validation disguised as a skip.
2. **Run control + mutated arms.** Control (gate on) doubles as prefix
   verification; mutated (gate open) measures behavioral effect. Reports
   persisted next to the parent trace; correlation-tagged `phase="replay"`.
3. **Verdict mapping.**
   * target failure gone AND scrubbed gates pass AND node walk sane
     → `promising`
   * target failure reproduced in tail → `dead` (edit abandoned; saves the
     full validation spend)
   * anything else (divergence after scrubbing, tool-classification refusal,
     crash) → `inconclusive` → treated exactly like `promising` for safety:
     falls through to full validation.
4. **Demote-only authority.** A replay verdict may *skip* a validation that
   would have been bought, never *grant* acceptance. Full `validate()`
   remains the only path into the score tensor / pool. Replay reports live
   outside the score tensor entirely — they must never enter entropy or
   ranking denominators.
5. **Budget honesty.** Experiments spend 1–2 live calls + wall clock; they
   must count. New optional budget field `max_replay_experiments` alongside
   existing `BudgetLimits`; exhausted ⇒ hook self-disables for the rest of
   the run with a loud line in the summary.
6. **Flag discipline.** Default OFF; flag-off behavior pinned byte-identical
   by test. Workers receive the flag + parent-trace path via environment
   (pre-fork), never in-memory state.

Failure modes this design prevents: silent replacement of measurement
(R8 violation — blocked by rule 4); wasted spend on hopeless edits (rule 3);
unaccounted spend (rule 5); pool-evidence contamination (rule 4); parallel
runs that don't actually replay (env-channel requirement).

## 6. Open questions (register here during wiring; promote to ledger ?N when real)

* **OQ1** Acceptance threshold: is "cleared + gates pass" sufficient to
  *invite* full validation always, or rate-limit invitations per iteration?
* **OQ2** Should RHO rounds eventually use replay for re-solves too, or is
  genetic-loop pre-filtering the whole scope?
* **OQ3** Tape retention: parent traces must outlive the session. Benchmark
  traces already land under tracked `data/` roots; script-made traces under
  gitignored `terminal_output/` do not. Decide promotion policy before W5.
* **OQ4** Determinism caveat for mutated arms: `knowledge_search` re-executes
  live (Ollama embeddings). Scores drift is expected and harmless *because*
  tool outputs feed only the tail prompt — but confirm no taped-boundary
  symmetry check ever depends on it (gates are open in MUTATION mode, so OK
  by construction; keep it that way).
* **OQ5** Does `experiment_with_replay` belong on the runner (core-adjacent)
  or the adapter facade? Lean adapter facade: it needs cuga imports
  (tape_replay), violating core neutrality if placed on the runner.

## Decisions log (append-only)

| Date | Decision | Evidence |
|---|---|---|
| 2026-08-26 | Plan created; five gaps enumerated from source walkthrough of run_evolution.py/pipeline seams; ladder ordered W1→W5 with demote-only pre-filter policy | session walkthrough; capabilities table above |

## Progress log (append-only)

| Date | Entry |
|---|---|
| 2026-08-26 | Plan committed to docs; nothing wired yet. Prerequisite noted: user commits the standing pile (tape modules, retry hardening, editor experiment scripts) BEFORE W1 surgery begins |

## Lessons (append-only)

*(empty — record strikes here the moment they happen)*
