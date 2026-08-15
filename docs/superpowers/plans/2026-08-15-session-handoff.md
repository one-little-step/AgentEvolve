# Session Handoff — 2026-08-15 (Full-Fidelity Trajectory Graph)

**Read this first. Then `docs/superpowers/plans/2026-08-15-full-fidelity-trajectory-graph.md`.**

Branch `dev4`, HEAD `1b3df2e "phase7 v1"`. **Nothing committed this session.**
Suite: **637 passed, 1 skipped** (`terminal_output/cuga-tracing/pre-compaction-suite.log`).

---

## Status: complete graph capture DONE and live-verified

The reference trace — inspect this one, it exercises every facility:

```
data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781/
├── manifest.json          7 capabilities, all honest
├── causal-trace.json      full trace + 3 tool_observations
├── events.jsonl           56 events, 52 edges, contiguous sequence
├── graph-topology.json    10 nodes / 15 declared edges   (NEW)
├── payloads/              44 content-addressed blobs, 758 KB   (NEW)
├── observations/          per-tool JSON
└── checkpoints/           final state (state_keys only, replay_safe=False)
```

Capabilities in that run:
```
graph_topology     captured      node_payloads         captured
stream_events      captured      tool_observations     captured
graph_final_state  captured      graph_history         unavailable_no_checkpointer
external_correlation  unavailable_no_sdk_surface
```

Recovered routing loop: `prepare→call_model`, `call_model→sandbox` (x3),
`call_model→__end__`, `SDKCallback→FinalAnswerAgent→__end__`.
Recovered tool chain (proves real execution, not narration):
```
fetch_alpha_token       {}                                 -> ALPHA-7924786034
exchange_alpha_for_beta {"alpha_token":"ALPHA-7924786034"}  -> BETA-2779592008
checksum_beta           {"beta_token":"BETA-2779592008"}    -> 858
```

---

## Why this work happened (requirement, not preference)

CUGA cannot replay a subagent counterfactually. The trace is the substitute:
capture each node's exact pre-state, prompt and response so a failing component
can be **simulated in isolation via a single LLM call** as edit feedback. Full
rollout regression still runs, but only after a batch of edits. Phase 7 had
captured structure only, which cannot support this.

---

## What was implemented (all TDD, all live-verified)

1. **Real graph edges.** Collector records `run_id`/`parent_run_id`;
   `parent_event_id` resolved to the parent's event id; `actor_id` from node name;
   `on_chain_end` recovers its node via `node_for_run()`.
2. **`_events_from_dicts` fixed** — `parent_event_id`/`actor_id`/`timestamp`/
   `sequence` map to top-level `CausalEvent` fields instead of sinking into payload.
3. **Deferred edge resolution** (`resolved_events()`) — LangGraph's outermost
   chain reports LAST, so eager resolution dropped every edge into the root and
   split one trajectory into 5 fragments. Fixed: 5 roots -> 1.
4. **Full payload capture** — `chain_start.inputs`, `chain_end.outputs`,
   `on_chat_model_start.messages`, `on_llm_end.response` stored verbatim as
   `payloads/<sha256>.json`, referenced by `state_before_ref`/`state_after_ref`/
   `messages_ref`/`response_ref`. Largest blob **40,002 bytes intact**.
5. **Routing + derived post-state** — `Command` uses `__slots__` so `vars()`
   returned `{}` and lost the routing decision; `_json_safe` now reads declared
   slots. `routed_to` recorded on `graph_node_end`.
   `load_node_state(..., with_provenance=True)` reports `after_source` as
   `chain_end_outputs` | `command_update` | `unavailable`.
6. **Static topology** — `graph-topology.json` sidecar from `get_graph()`.
7. **Chronological sequencing** — `_sorted_events()` by timestamp, tie-break on
   arrival. Within-source ordering verified 25/25 non-decreasing.
8. **`tool_observations` implemented** — `ingest_sdk_tool_calls()` from
   `InvokeResult.tool_calls` (the only live surface). `wrap()` is dead code.

New config: `capture_node_payloads`. New API: `load_node_state`, `PayloadStore`,
`_graph_topology`, `_routing_target`, `_json_safe`, `_sorted_events`.

---

## Accepted limitations — do NOT "fix" these

- **Local vs UTC timestamps: ACCEPTED AS-IS (user decision, 2026-08-15).**
  CUGA tool timestamps are naive LOCAL (`09:52:18.118732`); callback events are
  UTC `Z` (`04:21:59Z`), +5:30 apart. So `tool_call` events sort AFTER the graph
  events they occurred within. Ordering within each source is correct.
  Three tests encoding TZ normalization were **deleted** as a rejected
  requirement — do not re-add them.
- **SDK tool reports carry no `run_id`**, so tool calls cannot be attributed to
  their issuing node by identity. They stay unattributed rather than guessing.
- **`get_state_history()` returns 0 entries** -> checkpoint replay unavailable.
  Callback capture is the only viable state source. `graph_history` stays
  `unavailable_no_checkpointer`.
- **`on_tool_start`/`on_tool_end` never fire** (CUGA's sandbox calls the bare
  callable). Tool I/O comes only from the SDK report.
- **`on_llm_start` never fires** — chat models use `on_chat_model_start`.
- **`get_graph(xray=True)` == `get_graph()`** on this build (10 nodes/15 edges,
  no expansion). Subgraph internals (`prepare`, `call_model`, `sandbox`) are NOT
  in the declared topology, so an anomaly detector must scope to top-level edges
  or it will flag every internal transition as illegal.
- **Post-state is directly observable only at subgraph boundaries.** Elsewhere it
  is derived from `Command.update`. Pre-state is complete for every node (31-64 keys).
- **3 LLM calls include CUGA's internal classifiers** (e.g. policy "classify this
  assistant output" -> `{"auto_continue":false}`), not just the main reasoning
  call. Check the prompt before treating an `llm_call_*` as the subagent decision.
- **`roots=4` in the reference trace is CORRECT**, not a bug: 1 true root
  (`parent_run_id=None`) + 3 unlinkable SDK tool calls.

---

## Security decision that must be gated later

User chose **RAW_OPT_IN, no scanning** and **capture everything verbatim**
(2026-08-15). Payload blobs deliberately bypass `sanitize_for_persistence`,
because it truncates strings at `MAX_STRING_LENGTH = 2000` and hard-fails on key
names like `token`/`label`/`raw_prompt` that occur naturally in agent state.

Requires `PayloadLevel.RAW_OPT_IN` + `allow_raw_payloads=True`.

**Consequence to gate:** if a task's `expected_answer` reaches captured state, an
evolver reading these blobs could cheat and silently invalidate results.
AGENTS.md scopes its prohibition to what the evolver reads, so **feeding payload
blobs to the analyzer/editor must remain a separate, explicit decision.**

---

## Corrections to earlier claims (do not regress to these)

- "16 starts / 17 ends cannot pair" — FALSE. They pair 10/10 by `run_id`.
- "`prepare` x4 is a duplicate bug" — FALSE. Genuine nesting depth, distinct run ids.
- "`run_id` dropped by the redaction gateway" — FALSE. 19/19 present inside
  `payload`; I had checked the wrong nesting level.
- "Edges appear only where LangGraph reported one" — true by accident; 6 real
  edges were being dropped by the forward-reference bug.
- **D4's "prompt wording deterministically controls tool execution" — OVERSTATED.**
  Each table arm was effectively ONE observation. Two later runs executed 0/3
  tools on wording that had previously worked 3/3, while the model *narrated*
  "All three tool calls completed successfully". **Model text is never evidence of
  execution** — only tool-body side effects and `tool_calls` are.
- `feedback/gpt_context/cuga_tracing_sdk.md` is partly web-derived and wrong for
  this build: it "strongly recommends" `xray=True` (no effect here) and lists ~17
  nodes (runtime compiles 10). Verify its claims individually.
- **qf24.md fix #2 must NOT be implemented**: it asks for a *synthetic* `llm_call`
  parent. `docs/architecture/data-contracts.md:103` forbids synthetic placeholder
  nodes; qf23 says this exact bug was already fixed in Phase 6. Real
  `parent_run_id` exists (24/26), so synthesis is unnecessary AND prohibited.
  qf24 fix #1 was real and IS applied.

---

## Next steps

1. **Commit this work** (not yet committed; requires explicit user approval).
2. Wire the **subagent simulation harness** that consumes `load_node_state`:
   pre-state + exact prompt -> single LLM call -> compare against recorded response.
3. Decide the gate for feeding payload blobs to analyzer/editor (see security).
4. Optional cleanup: delete dead `ToolObservationRecorder.wrap()` + its 3
   `FakeTool`-only tests, or document it as non-CUGA-only.
5. `.vscode/` is untracked — user asked earlier whether to gitignore it; unanswered.

## Verification commands

```bash
uv run pytest 2>&1 | tee terminal_output/cuga-tracing/<name>.log
uv run python /tmp/complete_trace_demo.py     # regenerates a complete trace
T=data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781
jq . $T/manifest.json; jq . $T/graph-topology.json
jq -c '{seq:.sequence,kind,node:.actor_id,parent:.parent_event_id,routed:.payload.routed_to}' $T/events.jsonl
uv run python -c "
from pathlib import Path
from agent_evolve.cuga_wrapper import load_node_state
b,a,p = load_node_state(Path('$T'), node='call_model', with_provenance=True)
print(p); print(len(b), len(a))"
```

## Key files

- `src/agent_evolve/cuga_wrapper/__init__.py` — collector, `PayloadStore`,
  `load_node_state`, `_json_safe`, `_graph_topology`, `_sorted_events`, `TraceWriter`.
- `src/agent_evolve/core/trace.py` — agent-neutral schema (`extra="forbid"`, which
  is why topology is a sidecar file).
- `src/agent_evolve/core/storage.py` — `sanitize_for_persistence`,
  `MAX_STRING_LENGTH=2000`, `_DENYLIST_FIELDS`, `_SECRET_PATTERNS`.
- `tests/test_cuga_wrapper.py` — all regressions.
- `docs/superpowers/plans/2026-08-15-full-fidelity-trajectory-graph.md` — the plan.
- `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md` — durable
  cross-project CUGA findings (785 lines).
- `/tmp/complete_trace_demo.py` — regenerates the reference trace (TEMP: copy into
  `scripts/` if it should survive).
