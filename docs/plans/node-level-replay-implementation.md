# Node-Level Replay — Implementation Plan

**Status:** PLANNING COMPLETE 2026-08-25 · Phase 0 done (this document)
**Design:** [`docs/design/recorded-prefix-replay.md`](../design/recorded-prefix-replay.md)
**Discipline:** tests-first per phase · non-vacuity revert per behaviour ·
append-only Progress and Lessons sections · live phases behind `AE_LIVE_GO`
cost guard with tee'd logs in `terminal_output/live-run-prep/`.

---

## Phases

### Phase 1 — Capture-side: persist tool observations ✅ DONE 2026-08-25
Extend the graph-event collector/trace writer so tool invocations land in the
verbatim payload store: args digest on `graph_tool_start`, result ref on
`graph_tool_end` (`RQ1`/`RQ2` answered first against the installed SDK).
Classification table from R3 recorded alongside.
**Done when:** offline tests red→green covering start/end pairing, verbatim
result blob, withheld path; existing suite green except the 9 known Windows
platform failures.
**Result:** `on_tool_start` persists `args_ref` (structured `inputs` kwarg,
falling back to `input_str`); `on_tool_end` persists `output_ref`; absent
store ⇒ keys omitted (honest absence); capability `tool_observations`
flips to `captured` on graph-layer results alone.
Verified by `tests/test_tool_observation_capture.py` (6 tests red→green;
non-vacuity revert killed exactly the three result-dependent tests) +
full suite 2164 passed / same 9 platform failures.

### Phase 2 — Mint a reference trace with complete tapes
One paid re-run of `scripts/run_live_complex_query.py` (guard + tee).
**Spend estimate:** run-2-shaped ≈ 4 LLM calls × ≤100k cap, observed ~500 s;
expect a few thousand completion tokens plus reasoning burn.
**Done when:** new trace shows `tool_observations: captured`, non-empty
result refs; INDEX.md regenerated; key excerpts copied into the design doc's
Observed-basis section (terminal_output is gitignored).

### Phase 3 — TapeIndex + replayer core (offline)
Load a trace dir → index of LLM boundaries and classified tool results.
Synthetic two-step trace fixture for unit tests (stable texts; lesson ✓:
fixtures that test joining need authorship control over embedded strings).
**Done when:** TapeIndex resolves every ref; STRICT-mode dry classifies all
calls; no network anywhere in these tests.

### Phase 4 — LLM tape seam prototype (R9 decision)
Prototype the lightest seam that serves recorded responses to CUGA-internal
models. **Decision rule:** passes R5 oracle → becomes the settled seam,
recorded here.
**Done when:** chosen seam documented in §Decisions below with evidence.

### Phase 5 — STRICT crown on the real trace
Replay the full reference trace under STRICT; compare per-step state hashes
against source blobs.
**Done when:** byte-equal chain end-to-end (or every divergence explained by
an R3 classification bug fixed); capability `recorded_prefix_replay` declared
in trace capabilities with status `verified`.

### Phase 6 — LIVE-TAIL editor loop
Resume-point selection (RQ5), fresh-workspace isolation (R7), MUTATION mode
(RQ4). Integration point: the D5 editor experiment flow — diagnose fault at
node N → mutate skill → LIVE-TAIL from N.
**Done when:** one end-to-end editor experiment demonstrated on the F1 fault
(Windows file-authoring) without a full rollout.

## Decisions log (append-only)

| Date | Decision | Evidence |
|---|---|---|
| 2026-08-25 | R1 Design A over engine checkpointer | user choice; trace holds complete states |
| 2026-08-25 | Tool capture format: `args_ref` on start (structured `inputs`, fallback `input_str`), `output_ref` on end, verbatim; None ⇒ key omitted | langchain_core callback signatures inspected; tests/test_tool_observation_capture.py |
| 2026-08-25 | **?15 seam SELECTED (provisional until Phase-5 crown): inject a tape-backed `BaseChatModel` via CUGA's documented `LLMManager.set_llm` override** — refined from candidate (a): not an HTTP-client patch but the model-factory boundary. Evidence: `get_model` returns the pre-instantiated model for EVERY agent before any platform path (`cuga/backend/llm/models.py:1383`); recorded blobs are LangChain-native serializations (`LLMResult` dump tree / our `_json_safe` message projection), so serving in-process needs zero wire-format translation; local HTTP server would need a lossy LLMResult→wire→LLMResult round trip; client-patching couples to two libraries. Mechanics proven offline: symmetry gate (live prompts re-projected through the SAME `_json_safe` must byte-match recordings) + field-level verbatim reconstruction; settlement deferred: R5 hash-chain oracle on the real trace is the decision rule and runs in Phase 5 | tests/test_tape_model.py (10 tests incl real-trace load); probe output in-session showing `.generate()` packaging enriches messages identically on both sides and empties custom llm_output — envelope fidelity therefore scoped OUT of the unit contract |
| 2026-08-25 | **?15 seam SETTLED by the crown**: set_llm injection passed the R5 oracle on the real reference trace |
| 2026-08-25 | Phase 5 complete — **crown verdict `verified_normalized`** (scripts/verify_replay_strict_crown.py, offline ~50 s vs 443.7 s source, zero spend): tape consumed exactly 4/4 with no extra callers; prompt gates 4/4 under the volatility scrub registry and 1/4 raw (divergence is exclusively sandbox `Created:` wall-clock stamps, as predicted pre-build); node sequence fully aligned; final 3 KB answer byte-identical WITHOUT scrubbing. Declaration note: the capability lives in crown-report.json + docs rather than the per-trace capabilities block — capture-side traces cannot know they were replay-driven without threading replay context through the writer; deferred as cosmetic | terminal_output/node-replay/logs/crown-run2.log; replay trace cab73261 + crown-report.json |

## Progress log (append-only)

| Date | Entry |
|---|---|
| 2026-08-25 | Phase 0 complete: design + this plan committed to docs |
| 2026-08-25 | Phase 1 complete: tool args/results persisted verbatim at graph layer; capability flips on graph-layer results alone; suite 2164 passed / 9 known platform failures |
| 2026-08-25 | Phase 2 complete: reference trace `3306905e` minted (success, 443.7 s, self-healed F1); `tool_observations: captured`; args/output refs sha256 byte-verified; INDEX.md written; observed-basis updated in design doc incl. tool-tape scope correction (one LangChain tool pair per run; node states carry the rest) |
| 2026-08-25 | Seam evidence banked for R9/?15: CUGA-internal models construct via `langchain_openai.ChatOpenAI` (+ raw `openai`, + `OpenAIEmbeddings`), NOT our litellm wrappers; `_get_base_url` reads per-agent TOML `base_url` (our packaged profile) with `OPENAI_BASE_URL` env override → local tape server covers every caller uniformly; client-patch would couple to two libraries |
| 2026-08-25 | RQ1 closed by design choice, not a threshold: TapeIndex holds only metadata and lazily reads blobs on demand, re-verifying sha256 at every read — verbatim is never traded for memory; there is no truncation number to pick |
| 2026-08-25 | Phase 3 complete: `core/tape.py` (TapeIndex + ToolTapeClassifier, agent-neutral, generic `*_ref` discovery, run_id pairing as observed in production); fixture mirrors reference-trace serialized shape exactly; classification registry ships NO tool names in core (unknown ⇒ UNRECORDABLE with reason) |
| 2026-08-25 | Phase 4 complete: `cuga_wrapper/tape_replay.py` — TapeModel (BaseChatModel) serving recorded responses via the set_llm seam; symmetry gate + pointer discipline + field-level verbatim reconstruction; real reference trace loads (4 boundaries); suite 2187 passed / same 9 platform failures. No-op revert slip caught by green-stays-green (wrong indent in match pattern, second occurrence of this class of slip) |
| 2026-08-25 | Phase 6 complete (LIVE-TAIL demonstrated, one finding): `HybridTapeModel` (taped prefix → lazy live handoff, tools forwarded, MUTATION gate-open semantics) + `scripts/run_live_tail_experiment.py`; both arms ran end-to-end on F1 at resume=3 without a full paid rollout — prefix taped free, tails live, final outputs measurably differ (mutated arm's answer explicitly credits Windows file authoring). **Finding ?16**: late-stage nodes can construct models via `create_llm_from_config` (`graph.py:202`) bypassing the singleton override — hybrid served 3/5 boundaries; two calls per arm hit the real provider. Editor role was scripted (diagnosis + skill hand-authored), labelled as capability-demo | terminal_output/node-replay/tail-experiment/ (both reports, final outputs, summary) |
| 2026-08-25 | ?16 DISSOLVED — coverage was uniform all along; the reading was my bug (`_live_branch` never advanced the pointer ⇒ reports printed `live_calls=0` while provider calls fired). Instrumented rerun (`AE_TAPE_DEBUG=1`): single `TapeState` id across every invocation, 3 taped + 1 live exactly; `create_llm_from_config` unreachable in our runtime (`controller.py` passes `llm_config=None`). Fixed + regression-tested (`TestLiveAccounting`) | logs/tail-experiment-v2.log, tape-debug-cutoff4.log |
| 2026-08-25 | **REAL-EDITOR run complete** — the scripted-editor caveat is closed: `scripts/run_editor_replay_experiment.py` drove the actual `CugaEditorAgent.propose_edit` with the F1 `CausalAnalysis`; the editor executed 11 real tool calls (incl. `list_rollout_tools` independently confirming no write-file tool exists), staged and submitted its own 1472-char skill, outcome VALID (wrapper ledger + SDK tool-call record agree); LIVE-TAIL with the editor-authored skill: success 26.8 s, 3 taped + 1 live. Also registered **?17**: Console Go degrades under load — requests over ~24–32 KB die with 503 in windows while tiny ones pass (three editor attempts failed; 45 KB probe flipped fail→pass within minutes) | logs/editor-experiment-retry3.log; editor-experiment/{editor-plan.json, experiment-summary.json} |

## What went wrong / lessons (append-only)

* 2026-08-25 — Run 1 died on CUGA's default HTTP timeout (~2 min) long before
  any token limit mattered; rich reasoning calls take 165–210 s. Fix: per-agent
  `timeout=900` in packaged profile. **Trigger:** any "LLM did nothing" error —
  check timeout before tokens.
* 2026-08-25 — Provider returned 503 mid-trajectory killing run 2 at step ~9.
  **Trigger:** design replay resume so an aborted live run is itself a valid
  replay source (it was — run 2's tape ends honestly at its error).
* 2026-08-25 — Editing TOML via PowerShell string surgery corrupted the header
  comment (second strike for this lesson). **Trigger:** content edits use the
  Edit tool only; shell does execution, never authoring.
* 2026-08-25 — Live driver initially written into `tools/probes/` beside 39
  established scripts; relocated to `scripts/run_live_complex_query.py`; rule
  codified in AGENTS.md. **Trigger:** before creating ANY driver, read
  `scripts/` naming grammar first.
