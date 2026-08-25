# Recorded-Prefix Replay — Settled Design

**Status:** DESIGN SETTLED 2026-08-25 · NOT YET IMPLEMENTED
**Owner:** D5 follow-on ("node-level replay" work stream)
**Companion plan:** [`docs/plans/node-level-replay-implementation.md`](../plans/node-level-replay-implementation.md)

---

## 1. Problem

After an editor mutates an artifact, validating the edit costs a **full CUGA
rollout** — every node, every model call, minutes of wall time and real spend
(measured worst single call: 4281 completion tokens / 165 s). Experimentation
loops ("try a skill tweak, see if the failure clears") are unaffordable at that
price.

Most failures happen late in a trajectory whose early steps were fine. If we
can **re-drive the prefix deterministically and for free**, an editor can
experiment at the failure point directly.

## 2. Observed basis (live runs, 2026-08-25)

Two live runs through `CugaWrapper` (`scripts/run_live_complex_query.py`,
traces under `terminal_output/live-run-prep/traces/`, gitignored — key facts
recorded here):

| | Run 1 | Run 2 |
|---|---|---|
| Task | memory-doc analysis, 4-step deliverable | same |
| Outcome | `APITimeoutError` @ first big LLM call | reached step ~9/70, died on provider `503` |
| Duration | 171 s | 472 s |
| Fix between | per-agent `timeout=900` added to packaged profile | — |

Run 2 demonstrated genuine complex behaviour (knowledge retrieval, arithmetic
verification of a planted inconsistency: itemized 105000 vs stated-total
120000 → gap 15000) and surfaced **two real fault candidates** for future
evolution:

* F1: cuga_lite sandbox exposes no `write_file`; bash heredoc fallback fails
  under cmd.exe (`<< was unexpected at this time`) → agent cannot reliably
  author files on Windows.
* F2: provider 503 mid-trajectory kills the run (no retry visible).

### Trace-format facts the design rests on

* Every node start records its **complete input state** as
  `state_before_ref` (`store_payload(inputs)` — LangGraph hands each node the
  full current state); payload blobs are stored **verbatim** (`PayloadStore`
  docstring: *"reconstructing a subagent's exact pre/post state, prompt and
  response is the entire point"*).
* Every LLM boundary records `messages_ref` / `response_ref`.
* Routing decisions (`routed_to`) and tool invocations
  (`graph_tool_start/end` + `tool_name`) are captured.
* After-states are *derived* (`load_node_state`: most nodes return a LangGraph
  `Command`; after := Command.update applied onto before), provenance-flagged.
* **Engine checkpoints are NOT captured**: `"checkpoints": []`. The SDK
  attaches an in-process `MemorySaver` (`cuga/sdk.py:2296`) that dies with the
  process; our capability block honestly reports
  `graph_history: unavailable_no_checkpointer`.
* ~~Tool RESULTS are not yet persisted~~ **Superseded by Phase 1 wiring
  2026-08-25**: `graph_tool_start` carries `args_ref`, `graph_tool_end`
  carries `output_ref`, both sha256-verified verbatim against `payloads\`.

### Reference trace minted for this design (2026-08-25)

Run `3306905e` (`complex-20260825-193759`, same task as run 2): **status
success in 443.7 s** — first complete trajectory ever captured. It also
**self-healed F1**: hit the heredoc failure again, then pivoted unprompted to
`python -c` with hex-encoded content and produced+verified `report.md`.
Shape: 13 node starts / 4 LLM call boundaries / **exactly one LangChain tool
pair** (`knowledge_search_knowledge`, invoked inside the `sandbox` node) with
`args_ref` 142 B + `output_ref` 3103 B both byte-verifying; 35 payload blobs,
2.3 MiB total, max 164 KiB. Structural lesson: CUGA's heavy actions (code
execution rounds) happen *inside* nodes and are captured whole via
`state_before_ref` — the tool tape's scope is narrower than designed:
externals like `knowledge_search` plus any future registered tools, while
node states carry most of the fidelity burden.

## 3. Settled decisions

**R1 — Design A (recorded-prefix replay), not engine checkpointer.**
User-selected 2026-08-25. Re-drive the graph from step 1 serving each LLM call
its recorded response; continue with live calls from resume point N onward.
Engine-native checkpointing (SqliteSaver via `sdk.py:2257` hook) is **Phase-B,
deferred**, not cancelled.

**R2 — The verbatim payload store is the substrate.**
No new capture format for states/prompts/responses; replay reads existing
`payloads\` blobs. Content addressing (sha256) doubles as the fidelity oracle
(R5).

**R3 — A tool-tape middle layer is REQUIRED, not optional.**
Untaped external tools poison the contract: divergent `web_search` output at
step k invalidates the taped model responses that follow. Classification:

| Class | Examples | Replay behaviour |
|---|---|---|
| pure | `calculator`, fs ops in fresh workspace | re-execute |
| external | `web_search`, `web_fetch`, `wikipedia_search` | serve from tape |
| stateful-local | `knowledge_search` (corpus mutated earlier in-run) | tape under STRICT; configurable otherwise |
| unrecordable | oversized/streamed results | `withheld_reason`; prefix stops honestly before that step |

`core.trace.ToolObservation` already models this
(`canonical_arguments`, `result`, `replay_eligible`, `content_digest`,
`withheld_reason`) — the layer wires it, it does not invent a schema.

**R4 — Three strictness modes.**

* `STRICT` — LLM *and* classified tools all served from tape. Verification
  mode; must reproduce the original hash chain exactly.
* `LIVE-TAIL` — tape up to N, real execution after. The editor-experiment
  mode.
* `MUTATION` — LIVE-TAIL plus candidate artifacts re-materialized **edited**
  before the prefix re-runs, so nodes reading a mutated artifact during the
  prefix observe the edit.

**R5 — Fidelity oracle = hash-chain equality.**
A STRICT replay must reproduce byte-identical state hashes at every step vs
the source trace. Because payloads are content-addressed, equivalence checking
is free and non-vacuous. The new capability is declared **only** after this
test passes on a production-shaped trace (run 2's).

> **AMENDED 2026-08-25 (Phase 5 ground truth, measured before building the
> driver):** raw byte-equality is **unachievable on this trace shape** —
> 20/35 reference-trace payloads embed wall-clock traces (sandbox variable
> summaries render fresh ``Created:`` stamps into outputs on every execution)
> and 16/35 embed the task id. Faithful replay therefore cannot reproduce raw
> state bytes, and the byte oracle would fail even when behaviour is exactly
> reproduced. The oracle becomes **normalized chain verification**: a small,
> explicit volatility-scrub registry (ISO-like datetimes; the source task id)
> applied identically to both sides before comparison, at three checkpoints —
> (1) per-sequence prompt gate, (2) node-name/step sequence, (3) final
> output. Verdict levels, honestly labelled: ``verified_raw`` (no scrubbing
> needed anywhere — strongest), ``verified_normalized`` (all checkpoints
> pass under the declared registry), ``diverged`` (a checkpoint fails after
> scrubbing; first divergence excerpt reported). Every report states raw vs
> scrubbed failure counts so normalization can never silently absorb a real
> behavioural difference. The scrub registry is data next to the driver, not
> buried logic.

**R6 — Capability naming and the house invariant.**
This ships as facility `recorded_prefix_replay`. It does NOT flip
`supports_counterfactual_replay` — that flag means engine-state resume and
stays `False` until Design B earns its own verified reproduction.

**R7 — Side-effect isolation.** Prefix replay re-executes tools; file-writing
steps repeat their effects. Replays always target a **fresh workspace copy**,
never the original workspace.

**R8 — Semantic limit, stated up front.** Prefix replay answers *"does my fix
change behaviour from N onward?"*. An edit that would have altered an
**earlier** decision is invisible to the taped prefix; full re-roll via
`validate()` remains the gold standard for acceptance. Replay cheapens
experimentation; it never replaces measurement.

**R9 — LLM tape injection seam: SETTLED 2026-08-25 (crown-passed).**
Candidates were (a) patch the model client, (b) local OpenAI-compatible tape
server + base_url override, (c) mitmproxy addon. Evidence resolved it:
CUGA-internal models construct through `langchain_openai.ChatOpenAI` via the
`LLMManager` singleton, whose **documented `set_llm(BaseChatModel)` override
is returned to EVERY agent before any platform-specific path**
(`cuga/backend/llm/models.py:1383`) — and the recorded blobs are LangChain-
native serializations (`LLMResult` dump tree for responses; our `_json_safe`
projection for messages), so a tape-backed `BaseChatModel` serves them
in-process with **zero wire-format translation**, while candidate (b) would
need a lossy LLMResult→wire→LLMResult round trip and (a)/(c) couple to HTTP
client internals. Implementation: `cuga_wrapper/tape_replay.py`.
**Settlement evidence (Phase 5 crown on reference trace 3306905e):**
verdict `verified_normalized` — all four prompt gates pass under the R5
volatility scrub registry (1/4 raw; divergence exclusively sandbox wall-clock
stamps), node sequence aligned, final answer byte-identical raw, tape consumed
exactly 4/4, ~50 s offline vs 443.7 s source at zero provider spend.
RQ3 **ANSWERED**.

## 4. Open questions

* **RQ1** ~~Capture format~~ **ANSWERED 2026-08-25**: `args_ref` on
  `graph_tool_start` (structured `inputs`, falling back to `input_str`),
  `output_ref` on `graph_tool_end`, both verbatim via the payload store;
  unstoreable ⇒ key omitted. Remaining sub-question: truncation threshold for
  oversized results — deferred to Phase 3, where TapeIndex memory pressure
  makes the number real.
* **RQ2** ~~Which CUGA callback surfaces tool OUTPUTS~~ **ANSWERED
  2026-08-25**: LangChain hands the raw output to `on_tool_end(output=...)`
  and structured inputs to `on_tool_start(inputs=...)`; our handler was
  dropping both. Wired in Phase 1.
* **RQ3** LLM tape seam (see R9).
* **RQ4** ~~MUTATION mode~~ **ANSWERED 2026-08-25 (Phase 6)**: mutated
  surfaces are the harness-config dictionaries the wrapper already
  materializes (`skills` / `policies` / `memory` name→content maps); because
  an injected skill legitimately rewrites prompts from call one, MUTATION
  arms run with prefix gates **deliberately opened** (`gate_enabled=False`)
  and measure behavioural effect, never prefix fidelity — control arms keep
  gates on and double as verification.
* **RQ5** ~~Resume-point addressing~~ **ANSWERED 2026-08-25 (Phase 6)**: a
  boundary index into the trace-ordered `llm_boundaries` list
  (`--resume N`: boundaries `< N` taped, `>= N` live). Coarse but robust;
  event-id anchoring remains available if finer grain is ever needed.
* ~~RQ6 NEW from Phase 6 (?16)~~ **DISSOLVED 2026-08-25**: the suspected
  non-uniform seam coverage never existed. Instrumented rerun
  (`AE_TAPE_DEBUG=1`, stable `TapeState` id logged per call) proves EVERY
  model invocation flows through the injected model — 3 taped + 1 live per
  arm, zero bypass; `create_llm_from_config` (`graph.py:202`) is unreachable
  in our runtime (`controller.py` passes `llm_config=None`). Root cause of
  the false reading: `HybridTapeModel._live_branch` did not advance the
  shared pointer, so reports printed `live_calls=0` while provider calls
  fired — fixed and regression-tested (`TestLiveAccounting`). Residual
  observation: one duplicate `llm_call_start` under a single parent appeared
  once in v1 traces (likely a provider retry double-emission); benign for
  run_id-based pairing since identical retries produce identical message
  digests.

## 5. Trace-format quick reference (appendix)

Event grammar: kinds `graph_node_start/end/error`,
`llm_call_start/end/error`, `graph_tool_start/end`; `actor_id` names the graph
node; `sequence` is chronological; `routed_to` shows router decisions;
`*_ref` fields sha256-point into `payloads\`; `parent_event_id` encodes
subgraph nesting. Failure localization: read the **lowest** `*_error` event —
everything above it is propagation. See `INDEX.md` next to any trace for a
pre-rendered timeline.
