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

| Capability                                                                               | Lives in                                                    | Proven by                                        | Production callers                  |
| ---------------------------------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------ | ----------------------------------- |
| TapeIndex (load trace, resolve refs, lazy sha256)                                        | `core/tape.py`                                              | tests/test_tape_index.py 13                      | **none**                            |
| TapeModel (full taped re-drive)                                                          | `cuga_wrapper/tape_replay.py`                               | test_tape_model.py 12 + crown run                | **none**                            |
| HybridTapeModel (taped prefix → live tail; MUTATION gate-open; pointer accounting fixed) | same                                                        | test_hybrid_tape_model.py 7 + tail-experiment-v2 | **none**                            |
| Volatility scrub registry (R5 amendment)                                                 | driver-side regex tuple                                     | TestVolatilityScrubbing                          | **none**                            |
| LLM seam: `LLMManager.set_llm(tape_model)`                                               | cuga models.py:1383 honors pre-instantiated for EVERY agent | crown verdict `verified_normalized`              | scripts only                        |
| Resume-point selection                                                                   | manual (hand-picked boundary 3/4)                           | —                                                | does not exist                      |
| Editor→mutation extraction                                                               | `EditorResponse.writes` dict                                | editor-experiment/editor-plan.json               | wired for full-validation path only |

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

| Step | Deliverable                                                                                                                                                                                                | Done when                                                                                                                                                  | Unblocks   |
| ---- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| W1   | Provenance plumbing: `trace_dir` persisted onto rollout record + cell provenance at capture time                                                                                                           | red→green test: after `_record_rollout_score` of a traced rollout, cell exposes the trace location; AST/grep shows wrapper result path no longer discarded | everything |
| W2   | `boundary_for_fault` mapper in `cuga_wrapper` (pure fn) + unit tests incl. all four edge cases above                                                                                                       | tests green against synthetic fixture mirroring crown-trace shape; real-trace integration test skipif absent                                               | W3         |
| W3   | `experiment_with_replay(...)` on the runner (or adapter facade) returning the structured report; HybridTapeModel injected **inside whichever process executes**; correlation scope tagged `phase="replay"` | offline tests: control arm gates stay on, mutated arm gate-open, live_calls counted (pointer fix regression holds), report contract stable                 | W4         |
| W4   | Pre-filter hook in `run_iterations()` behind `AE_REPLAY_EXPERIMENTS=1`; env-channel propagation to process workers (precedent: `ALLOW_RESPONSE_CACHE_ENV` set pre-fork); budget counter extension          | flag-off run byte-identical to today (pin test); flag-on dry-run exercises hook with fakes; budget refuses when exhausted                                  | W5         |
| W5   | Live in-loop demonstration: F1 scenario triggered by the pipeline itself (editor turn → replay filter → invited validation), tee'd log + report artifacts                                                  | one end-to-end run with all reports persisted; comparison vs manual-run baseline                                                                           | DONE       |

Spend note: W1–W4 zero provider calls (fakes/synthetic). W5 small (~editor
turn + 1–2 live tail calls).

## 5. Architecture correction (2026-08-26, user) — replayTape is an EDITOR TOOL

**The earlier pre-filter plan drafted a pipeline-side "pre-filter hook" inside
`run_iterations()`. That design is WRONG for this house and is retracted.**
The settled architecture, matching how `list_complementary_parents` was
built (checkmarks 73 and 81):

* **replayTape is an editor tool, voluntary, invoked at the editor agent's
  own wish** — exactly like `list_complementary_parents`: registered through
  `EditorToolContext`, unavailable-by-default until the composition root
  attaches a provider factory (`attach_complement_provider` precedent → new
  `wire_editor_replays`), never raising into the agent.
* The editor decides WHEN to experiment (after diagnosing an issue, before
  committing to a revision), WHAT to apply (its staged/current writes),
  and reads RAW structured observations back — **interpretation belongs to
  editor wisdom, not a pipeline verdict enum**.
* The pipeline keeps exactly one monopoly it already has: `validate()` is
  and remains the only acceptance path (R8). No new policy stage needed —
  acceptance structure already enforces R8; replay results simply are not
  scores.
* There is NO fixed control-arm ritual. Tape fidelity was proven once by
  the crown (✓89); per-experiment verification is optional and, if ever
  wanted, is itself something an editor can request by running the tool
  twice. Gate on/off becomes an internal parameter of the facade, not a
  user-facing decision.
* Budget/cost discipline moves into the TOOL layer: per-invocation cost cap
  + per-attempt experiment limit as configuration of the provider factory
    (mirrors `retry_budget` patterns), reported honestly inside each tool
    result; plus the existing `--max-editor-calls` style accounting if we
    count experiments as editor-attempt work.

### Rewritten wiring ladder

| Step | Deliverable                                                                                                                                                                                                                                                                            | Notes                                                               |
| ---- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| W1   | trace→pool provenance plumbing (unchanged)                                                                                                                                                                                                                                             | prerequisite for any tape addressing                                |
| W2   | `boundary_for_fault(tape_index, analysis)` mapper (unchanged)                                                                                                                                                                                                                          | also directly useful to the EDITOR: surfaced inside the tool result |
| W3   | Replay facade replacing `experiment_with_replay(runner,…)` plan: `ReplayExperimentFacade` constructed at composition root, owning HybridTapeModel build/set_llm/fresh-workspace/scrub registry, accepting `(parent_trace_dir, artifacts, resume_hint?)`, returning the raw report JSON | editor-agnostic; no runner method                                   |
| W4′  | New editor tool `run_replay_experiment` (+ registration cluster) fed by a `replay_provider_factory` attached via `wire_editor_replays(agent, facade_factory)` at `build_live_stack`; cost caps as facade config                                                                        | mirrors attach_complement_provider exactly                          |
| W5   | Live demonstration INSIDE an iteration: real editor chooses to invoke the tool mid-decision                                                                                                                                                                                            | replaces the old pre-filter demo                                    |

### NEW ITEM discovered 2026-08-26 — Judge-2 positivity judge is DEAD on the live path

Verified: zero `CugaPositivityJudge(...)` construction sites in `src/`;
`build_live_stack` never passes `positivity_judge`; orchestrator default is
`None` with the loop commented "Off by default… never runs"
(orchestrator.py:1159/:1463). It ran ONLY in hand-built gauntlets (✓71/✓74).
Consequence: D5 strengths never enter TS2/index/complementary evidence in
production runs. **Wiring task (J-W):** construct `CugaPositivityJudge` in
`build_live_stack` alongside the analyzer adapters, pass into
`SequentialGepaRunner`, flag/config-gated with its spend counted under
judge budgets; tests: composition-root pin + one loop-level test proving

    CORRECTION (user caught my hallucination against docs/design/issue-lifecycle.md): the passes-only (task, trace) loop is a PARTIAL BUILD, not the design. Per D5.3 input = scorable traces of ANY score; D5.5 requires cross-candidate traces INCLUDING failures (least-bad degradation); D5.6 target flow: J2 reads STORED traces via TS2; Q8 defers WHICH traces get analyzed to an index-time policy (per-cluster top-k, recency; cache key candidate,task). Complementarity is mechanism-INFORMED targeting: selected-issue clusters drive whose traces get analyzed - user instinct confirmed by the doc, not blind scanning.
  Therefore J-W splits into JW-a (activate existing passes loop: construct judge in build_live_stack, pass to runner, pin) and JW-b (BUILD the Q8 targeting layer over traces_for() incl. failure-trace analysis and cache policy - genuinely new work, owner: this ladder before W-tools rely on complementary evidence).

## 6. Open questions

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

| Date       | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            | Evidence                                                                                                                              |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-08-26 | Plan created; five gaps enumerated from source walkthrough of run_evolution.py/pipeline seams; ladder ordered W1→W5 with demote-only pre-filter policy                                                                                                                                                                                                                                                                                                                                                                                                              | session walkthrough; capabilities table above                                                                                         |
| 2026-08-26 | RETRACTED the pipeline-side pre-filter (old section 5 / step W4) after user correction: replayTape is an EDITOR tool, voluntary, invoked at the editor's own wish; raw observations return to the editor for its own reflection - no pipeline verdict enum, no control-arm ritual, no run_iterations hook. Ladder rewritten as W1, W2, W3 facade, W4-prime editor tool via wire_editor_replays, W5 in-iteration demo. Acceptance monopoly of validate() unchanged (R8 already structural). Judge-2 live-path death discovered same turn (new item J-W in section 5) | user correction message; grep proof: zero CugaPositivityJudge constructions anywhere in src/, orchestrator 1159 and 1463 default-None |
| 2026-08-26 | Mutation semantics clarified for the editor tool: harness mutation under tape is sound because prompt-flowing artifacts (skills/instructions) are INERT before the resume point - taped responses cannot see changed prompts - and become behavioral exactly at the live tail where the experiment question lives; side-effect-flowing artifacts (memory via live knowledge_search, policy enactment) carry mid-prefix drift risk, mitigated by late resume points and disclosed in every report; instructions injection reaches the agent constructor not materialize_harness, so W3 must exercise it explicitly | demonstrated: editor-authored skill mutated arm (tail adopted hex-authoring, final output changed); mechanisms verified in-session |

## Progress log (append-only)

| Date       | Entry                                                                                                                                                                             |
| ---------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-08-26 | Plan committed to docs; nothing wired yet. Prerequisite noted: user commits the standing pile (tape modules, retry hardening, editor experiment scripts) BEFORE W1 surgery begins |

## Lessons (append-only)

*(empty — record strikes here the moment they happen)*
