# Plan — Full-Fidelity Trajectory Graph (Phase 7 correction)

Status: verified surfaces, implementation not started.
Supersedes the Phase 7 assumption that traces record structure only.

## Why this exists

Phase 7 captured node names and tool-call summaries. The evolver needs more: to
attribute a failure to one subagent and then *simulate* that subagent in
isolation (single LLM call, as edit feedback), the trace must carry the exact
pre-state and post-state of that subagent plus the real prompt/response. Full
rollout regression still runs, but only after a batch of edits.

CUGA does not support subagent-level counterfactual replay. Our graph is the
substitute: reconstruct the state, re-issue one model call, compare.

## Requirement (from user, 2026-08-15)

- Complete graph: all edges, tool inputs AND outputs, agent input/output responses.
- Full response capture, so a subagent state can be rebuilt programmatically,
  lazily, on demand.
- If the graph payload already contains the subagent context, no rebuild layer is
  needed; keep the lazy reader only as fallback.

## Verified surfaces (live run, `terminal_output/cuga-tracing/payload-surface-probe.log`)

| Surface | Result |
|---|---|
| `chain_start.inputs` | full CUGA `AgentState`, 41 keys/node, 84 distinct across run |
| `chain_end.outputs` | post-state, node-scoped |
| start/end pairing | **10/10 exact by `run_id`** (never by node name) |
| `parent_run_id` | **24/26 populated**, correctly nested tree |
| `on_chat_model_start` | 3 calls, `SystemMessage`+`HumanMessage`, up to 39,506 B |
| `on_llm_end` | `generations[0][0].text` + `llm_output` |
| state keys present | `chat_messages`, `api_planner_history`, `variables_storage`, `instructions`, `skills_prompt_section`, `prepared_prompt`, `pi` |
| Total size | 254 KB for a 3-LLM-call run (`outputs` 123 KB, `inputs` 71 KB) |

Answer to "is the state already in the graph": **yes**. No reconstruction layer
required; persist payloads and read them lazily.

## Negative findings (do not retry these)

- `get_state_history()` exists but returned **`count: 0`**. Checkpoint-based
  replay is not viable; `graph_history: unavailable_no_checkpointer` stands.
  Callback-captured state is the only viable source.
- `on_llm_start` fires **0** times (chat models use `on_chat_model_start`).
- `on_tool_start` / `on_tool_end` fire **0** times — CUGA's sandbox invokes the
  bare callable. Tool I/O must come from the SDK `tool_calls` report.
- `get_graph(xray=True)` == `get_graph()` here: 10 nodes / 15 edges, no
  expansion. Do not rely on xray for subgraph internals in this config.

## Prohibition

`feedback/from_qwen/qf24.md` fix #2 asks for a **synthetic** `llm_call` parent
event. Do NOT implement. `docs/architecture/data-contracts.md:103` forbids
synthetic placeholder nodes; absence must be `insufficient_evidence`. qf23 notes
this exact bug was already fixed in Phase 6. Real `parent_run_id` exists, so
synthesis is also unnecessary. qf24 fix #1 (`_events_from_dicts` burying
`parent_event_id`/`actor_id`/`timestamp` in `payload`) is a real bug and is in
scope — but it is inert until the collector records the IDs.

## Tasks (TDD, tests first)

1. **DONE — Collector records identity + edges.** `run_id`/`parent_run_id`
   captured; `parent_event_id` resolved to the real parent's event id; `actor_id`
   set from node name; `on_chain_end` recovers its node via `node_for_run()`.
   No synthesis. Live-verified on `data/traces/9ce1e3b6-0cd0-42c6-b3e5-314de5b044b8`:
   `parent_event_id` 12/19, `actor_id` 18/19, `timestamp` 19/19,
   `run_id`/`parent_run_id` 19/19 (inside `payload`), `graph_node_end` named 9/10
   (was 0/17). Tree reconstructs: CugaLiteSubgraph > CugaLiteSubgraph > prepare
   (x3 nested) + call_model, then SDKCallback, FinalAnswerAgent.
2. **DONE — `_events_from_dicts` fixed.** `parent_event_id`, `actor_id`,
   `timestamp`, `sequence` now map to top-level `CausalEvent` fields instead of
   sinking into `payload`.
3. **DONE — Full-fidelity payload capture.** `chain_start.inputs`,
   `chain_end.outputs`, `on_chat_model_start.messages`, `on_llm_end.response`
   stored verbatim as content-addressed blobs under `<trace>/payloads/<sha256>.json`,
   referenced from events as `state_before_ref` / `state_after_ref` /
   `messages_ref` / `response_ref`. New `capture_node_payloads` flag and
   `node_payloads` capability. Live-verified
   (`data/traces/ced08fb7-91bc-41d6-82d3-e7cc8e1212c6`): 22 blobs, 245 KB,
   largest blob **40,002 bytes** intact.
3b. **DONE — Routing objects and derived post-state.** Most CUGA nodes return
   LangGraph `Command(goto=..., update=...)`, not full state. `Command` uses
   `__slots__` so a `vars()` projection yielded `{}` and lost the routing
   decision; `_json_safe` now reads declared slots. `routed_to` is recorded on
   `graph_node_end`. `load_node_state(..., with_provenance=True)` returns
   `after_source` of `chain_end_outputs` | `command_update` | `unavailable`, and
   never returns a raw routing object as a post-state. Live-verified: real deltas
   (`call_model`: chat_messages, execution_complete, step_count, final_answer) and
   the observed routing chain (`prepare -> call_model`,
   `SDKCallback -> FinalAnswerAgent`, `call_model -> __end__`).
4. **DONE — Static topology persisted.** `_graph_topology()` reads
   `agent.graph.get_graph()` and writes `<trace>/graph-topology.json` as a
   sidecar (kept out of `CausalTrace`, which forbids extra fields). New
   `graph_topology` capability. Live-verified: **10 nodes / 15 edges** with a
   `conditional` flag per edge.
5. **DONE — Chronological sequencing.** `_sorted_events()` orders by observed
   timestamp with arrival order as tie-break, applied after callback events and
   the SDK tool report are merged. Live-verified: **25/25 timestamps populated,
   non-decreasing, `sequence` contiguous 0..24**, with `llm_call_start`/`_end`
   correctly nested between the `prepare` nodes that wrapped them.
6. ~~Lazy state reader~~ — **DONE** as part of Task 3/3b (`load_node_state`).
7. **Honest capability statuses** — done for `tool_observations`, `node_payloads`,
   `graph_topology`. Remaining: revisit `graph_history` (see below).

## Validating observed routing against declared edges

Cross-checking `routed_to` against `graph-topology.json` separates two cases:

```
SDKCallback -> FinalAnswerAgent   declared
FinalAnswerAgent -> __end__       declared
prepare -> call_model             NOT in declared edges
call_model -> __end__             NOT in declared edges
```

`prepare` and `call_model` are **subgraph-internal** to `CugaLiteSubgraph`, which
is opaque at top level (and `xray=True` does not expand it on this build). So
"not declared" means internal, not illegal. Any anomaly detector must scope
itself to top-level edges, or it will flag every subgraph-internal transition.


## Surface notes discovered during Task 3

- **Pre-state is complete for every node** (`chain_start.inputs` = full
  `AgentState`, 31-64 keys). **Post-state is directly observable only at subgraph
  boundaries**; elsewhere it must be derived from `Command.update`.
- The 3 LLM calls include CUGA's **internal classifiers** (e.g. a policy
  "classify this assistant output" call returning `{"auto_continue":false}`), not
  only the main reasoning call. Do not assume an `llm_call_*` event is the
  subagent's decision - check the prompt.
- `PayloadLevel.RAW_OPT_IN` + `allow_raw_payloads=True` is required, and payload
  blobs deliberately bypass `sanitize_for_persistence` (2000-char truncation and
  a key-name denylist covering `token`/`label`/`raw_prompt` would otherwise
  destroy or reject legitimate agent state). User decision, 2026-08-15.
  **Consequence to gate later:** if a task's `expected_answer` reaches captured
  state, an evolver reading these blobs could cheat. Feeding payload blobs to the
  analyzer/editor must remain a separate, explicit decision.


## Open issue (separate from tracing)

Tool execution is not reliable across runs even with the "write and execute"
wording. Same code path, same prompt: 3/3 tools at 01:07, **0/3** at 08:54
(`data/traces/9ce1e3b6-...`, model narrated "the exact value was printed by the
executed code" without ever calling a tool). This contradicts the earlier
conclusion that phrasing alone determines execution — that conclusion was drawn
from single observations per arm. Prompt wording is *a* factor, not the whole
story; there is genuine run-to-run variance. Do not treat any wording as
guaranteed. This affects verification instruments only, not the tracing code.


## Guardrails

- Never persist credentials, expected answers, evaluator internals, or labels.
- `supports_counterfactual_replay()` stays False: simulation from captured state
  is not SDK state reconstruction. Snapshots keep `replay_safe=False`.
- Bound blob sizes; on overflow retain digest + truncation flag, never silent loss.
- Capture logs to `terminal_output/cuga-tracing/<name>.log`.
