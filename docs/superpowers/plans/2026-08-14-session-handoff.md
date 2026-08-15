# Session Handoff — 2026-08-14 (CUGA tool execution + harness injection verified)

This supersedes the earlier `2026-08-14-session-handoff.md` conclusions about
tool calling. Read this FIRST on resume.

## Status (read this first)

- D1 (skills), D2 (policies), D3 (stream_events + graph_final_state + model)
  are FIXED and VERIFIED against real runs.
- D4 is ROOT-CAUSED and CLOSED as a model property, not a defect: tool
  invocation is a deterministic function of prompt wording for
  `openai/azure/gpt-5.6-luna`. No wrapper code needs changing. User decision:
  "if it's model dependent, we don't need to bother about this".
- Suite: 608 passed, 1 skipped. Nothing committed. Branch `dev4`, HEAD `1b3df2e`.
- All CUGA learnings are consolidated in
  `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md` (785 lines)
  for reuse in other CUGA SDK wrapper projects. That file is the durable
  artifact; this handoff is the session log.

## Headline: the previous root cause was WRONG

The earlier session concluded "the reasoning model `azure/gpt-5.6-luna` cannot
emit executable code, so CUGA never runs tools". That is **disproved**.

Verified this session, with unguessable random-token probes:

| Test | Evidence | Result |
|---|---|---|
| Bare `CugaAgent` + probe | `extract_code_from_model_response` returned `sum_result = await diag_add(17, 25)` | tool body ran, `tool_calls=1` |
| Wrapper's exact kwargs (`enable_knowledge=True` + prose instructions) | random `TKN-*` token appeared in answer | tool body ran, `tool_calls=1` (2/2 trials) |
| Real `CugaWrapper.run_task`, 3-tool dependency chain | `chain_completed: true` | all 3 tools ran in correct order |

**Actual cause of the old empty `tool_calls`:** the probe tasks were *guessable*.
`1234 * 5678` and `17 + 25` are solvable mentally, so the model answered directly
and correctly without tools. `tool_calls: []` was truthful, not a defect.

**Lesson: any tool-execution probe MUST return a value the model cannot derive
or know** (random token, side-effect file). Ground truth is the recorded tool
function body execution, never the model's narrative claims.

The `Cannot connect to host localhost:8001` / `Error while calling registry to
get apps` lines are a harmless fallback; they appear in fully-working runs too.

## Confirmed working

- Live multistep tool execution through `CugaWrapper.run_task` (3 chained tools).
- `track_tool_calls=True` returns populated records: `name`, `arguments`,
  `result`, `app_name`, `operation_id`, `timestamp`, `duration_ms`, `error`.
- Trace persistence: manifest, `events.jsonl`, `causal-trace.json` written
  atomically; `thread_id_source="wrapper_generated_injected"`.
- Harness **structural** injection: skills discovered, policy loaded into
  storage, memory ingested + semantically searchable.
- Harness **memory behavioral** injection: injected random `MEM-*` token was
  retrieved from knowledge and used in the answer.
- Test suite: 600 passed, 1 skipped (at commit `1b3df2e`).

## Defect log (D1-D3 fixed & verified; D4 closed as model property)

### D1 — FIXED & VERIFIED: skills now reach the model
CUGA suppressed the entire skills prompt block unless the shell tool was on.

- Root cause: `cuga/backend/cuga_graph/nodes/cuga_lite/prompt_utils.py:682-689`
  sets `skills_enabled = False; skills_prompt_section = ""` when
  `not enable_shell_tool`. Live log confirmed:
  `"Skills are enabled but enable_shell_tool=False; the skills block will be suppressed."`
- **Fix applied:** `DYNACONF_ADVANCED_FEATURES__ENABLE_SHELL_TOOL=true` in `.env`
  (commented with the reason and the side effect).
- **Verified:** `load_skill("status-report")` executed in the sandbox, returned
  the skill body, and the injected `SKL-*` signature appeared in the answer.
- **Accepted side effect:** CUGA now injects a real sandbox `run_command` shell
  tool (log: `[NativeSandbox] Injected run_command`). Tested and accepted by the
  user; note it if the threat model changes.

### D2 — FIXED & VERIFIED: playbooks now match
An always-only playbook loaded and deserialized but was never evaluated.

- Root cause: in `cuga/backend/cuga_graph/policy/agent.py`, `_check_trigger`
  (line 167) handles `AlwaysTrigger` (line 178), but `match_policy` (line 929)
  builds candidates only from `_evaluate_keyword_triggered_policies` (685,
  filters `KeywordTrigger`) and `_evaluate_natural_language_policies` (767).
  **No evaluator selects `AlwaysTrigger`**, so an always-only policy can never win.
- Empirically confirmed: `always` -> `matched=False`; `keywords` -> `matched=True`;
  `natural_language` -> `matched=True` with `Playbook guidance will be injected`.
- **Fix applied** in `materialize_harness` (`cuga_wrapper/__init__.py`): emit a
  `natural_language` trigger derived from the policy body (`target: intent`,
  `threshold: 0.5`), keeping `always: true` as forward-compatible intent.
- **Two schema traps found the hard way (both now regression-tested):**
  1. The frontmatter key is **`keywords`** (plural). `keyword:` produces zero
     triggers and CUGA rejects the file: *"must have at least one trigger"*.
  2. The trigger phrase **must be a quoted YAML scalar**. Policy text usually
     contains `:` (e.g. "end with the line: MARKER"); unquoted it fails with
     *"Invalid YAML in frontmatter: mapping values are not allowed here"* and the
     policy is **silently dropped** — it looks configured but has no effect.
- Tests added: `test_materialized_playbook_uses_matchable_triggers`,
  `test_materialized_playbook_frontmatter_is_valid_yaml_with_colons`.

### Harness injection: ALL THREE CLASSES NOW VERIFIED
`scripts/verify_harness_behavioral.py` (unguessable tokens per class):
```
memory_token_in_answer: true
policy_token_in_answer: true
skill_token_in_answer:  true
all_three_influenced_behavior: true
```
Trace: `data/traces/f3fa9b6a-e10e-4760-89c4-e6dc02bc0b62`.
Full suite after the fixes: **602 passed, 1 skipped**.

Note on methodology: the pre-existing `verify_harness_injection.py` checks only
that CUGA *loaded* artifacts into its stores. That reported success while two of
three never reached the model. Structural loading is NOT evidence of behavioral
influence — always assert on an unguessable token in the output.

### D3 — FIXED & VERIFIED: stream_events + graph_final_state now captured
Real run manifest (`data/traces/9be8a129-f09f-4d8f-8a9b-a952a20198e5`):
```
"stream_events":     {"status": "captured"}     <- 19 node lifecycle events
"graph_final_state": {"status": "captured"}     <- 1 StateSnapshot, replay_safe=false
"model": "openai/azure/gpt-5.6-luna"            <- was null
"captured_event_count": 19                       <- was 3 (tool summaries only)
"files": {"events.jsonl", "checkpoints/", "causal-trace.json", "manifest.json"}
```
Mechanism (one execution only; `stream()` is never called):
- `GraphEventCollector` is an agent-neutral event sink holding no LangChain types.
- `build_graph_callback_handler(collector)` adapts it to LangChain by
  **subclassing `BaseCallbackHandler`**, then it is passed through
  `invoke(..., config={"callbacks": [handler]})`. CUGA merges caller callbacks
  into that single `graph.ainvoke` (`sdk.py:_apply_callbacks`), so node evidence
  needs no second run.
- `graph_final_state` reads `agent.graph.get_state({"configurable": {"thread_id": ...}})`
  **after** the run. `CugaAgent.graph` compiles with a `MemorySaver`
  (`sdk.py:2291-2301`), so this is real post-run state, not a re-execution.
  Reported `replay_safe=false`: reading final state is not state reconstruction.
- Node observed: `CugaLiteSubgraph`, `prepare`, `call_model`, `SDKCallback`,
  `FinalAnswerAgent`.
- Only structural identifiers persist (node name, step, tool name). Node inputs
  and outputs are deliberately NOT persisted — they can carry evaluator
  internals or expected answers.

**Two defects found and fixed while doing D3:**
1. A duck-typed callback handler is not enough. LangChain async dispatch reads
   `h.run_inline` (`langchain_core/callbacks/manager.py:471`), so a handler that
   merely implements `on_chain_start` raises
   `AttributeError: 'GraphEventCollector' object has no attribute 'run_inline'`
   mid-run. Regression test:
   `test_graph_event_collector_satisfies_langchain_callback_contract`.
2. `run_task` swallowed invoke exceptions into a bare `status="error"` with empty
   output and no reason — which is exactly how defect 1 first presented. The
   error is now persisted in the result dict and in `CausalTrace.error`.
   Regression test: `test_sdk_runtime_persists_invoke_exception_as_trace_evidence`.

Still open in D3 scope:
- `tool_observations`: `ToolObservationRecorder.wrap()` is still never called on
  live tools; honestly reported `unavailable_no_sdk_surface`.
- `graph_history`: still `unavailable_no_checkpointer`. Note `get_state_history`
  exists on the compiled graph and was verified to return 4 entries on a local
  LangGraph probe, so this facility is now plausibly implementable — but it has
  NOT been verified against CUGA's graph and must not be claimed until it is.

### D4 — CLOSED (model property, no fix needed): prompt wording controls tool use
Ruled out as a D3 regression by controlled A/B on the same prompt and agent:
```
A-baseline-no-config     (no callbacks): 0/4 trials executed the tool
B-config-with-callbacks  (D3 path):      0/4 trials executed the tool, 19 events each
```
Callbacks cost zero tool executions. Also unaffected by
`DYNACONF_ADVANCED_FEATURES__ENABLE_SHELL_TOOL=false`, by an empty
`SKILLS_ROOT`, and by quarantining the leaked `.cuga/playbooks` global policy.
- Failure mode: the model returns prose such as *"I'm unable to execute the tool
  call in the current interaction"*; `call_model` runs, `CodeAgent` never does.
- `scripts/bisect_instructions_contract.py` scored
  `wrapper_prose_instructions#1 RAN calls=1` while its other 3 cases did not run.
  That looked like intermittency at the time; it was actually per-prompt
  determinism across arms with different wording (see the table below).
- Earlier today the same three-tool chain completed
  (`data/traces/2b3ced93-6929-404b-803f-56a2a22cc003`, 21:39) — with different
  task text. Not drift; different prompt.
**Root cause (confirmed, all-or-nothing across 5 phrasings):** with identical
tools, config, and registered callable, whether the agent invokes the tool is
decided by incidental task wording. Ground truth = tool body executed:

```
(no suffix)                                 0/2
"Respond with only the value."              0/2
"Return just the value, nothing else."      2/2
"First call X. Then report..."              0/2
"Write and execute Python code that
 calls read_build_number(), then report"    2/2
```

Never 1/2 — so it is reproducible per prompt, NOT flaky. Failing runs emit no
``` fence, so `extract_code_from_model_response` returns "" and `call_model`
takes the no-code branch; the sandbox is never reached. The model then narrates
"I'm unable to call the tool", which is false — the tool was in the prompt and
registered in the sandbox.

Ruled out by direct experiment (each with its own log):
- D3 callbacks: 0/4 vs 0/4 with and without, 19 node events either way
- `enable_shell_tool=false`, empty `SKILLS_ROOT`, quarantined `.cuga/playbooks`
- tool construction (`@tool` over `@tracked_tool` vs post-hoc `__doc__`): 1/3
  vs 1/3, byte-identical tool metadata
- probe vocabulary alone: the arm that scored 3/3 scored 0/3 later once only its
  task suffix changed

**Methodology error this exposed (now in the learnings doc):** repeating an
identical prompt is not sampling. This reasoning model skips temperature, so
decoding is greedy and identical prompts give byte-identical output — my
"3 trials" were 1 observation reported 3 times. Vary the PROMPT, not the trial
index.

**Consequence for future work:** phrase any tool-exercising task as an explicit
code-execution instruction. Do not attribute non-execution to safety filters
without evidence — one arm did produce a real refusal
("I can't provide or reveal secret tokens", provoked by probe words
"secret"/"token"/"reveal"), but neutral-vocabulary arms failed too.

## Verified CUGA facts (cuga 0.3.1; `cuga.__version__` misreports 0.2.20)

- `CugaAgent.__init__` accepts: `tools`, `tool_provider`, `model`, `callbacks`,
  `policy_system`, `special_instructions`, `cuga_folder`, `auto_load_policies`,
  `reset_policy_storage`, `filesystem_sync`, `enable_knowledge`,
  `enable_citations`, `enable_skills`, `skills_folder`.
- `invoke(message, thread_id, config, action_response, user_context,
  track_tool_calls, variables) -> InvokeResult(answer, tool_calls, sources,
  thread_id, error, variables)`.
- `stream(message, thread_id, config, action_response)` — async generator of raw
  LangGraph state; has NO `track_tool_calls` parameter.
- `find_tools` shortlisting is OFF here: `enable_find_tools = total_tool_count >
  shortlisting_tool_threshold or _web_search_enabled()`; runtime values are
  threshold `35`, `enable_web_search=False`, and we pass ~5 tools. So our tools
  ARE exposed directly to the model. (Ruled out as a cause.)
- Effective settings: `force_autonomous_mode=True`,
  `cuga_lite_nl_auto_continue=True`, `enable_todos=False`,
  `features.code_generation="fast"`, `features.local_sandbox=True`,
  `enable_shell_tool=False`.
- `tracked_tool` records into `ToolCallTracker` contextvars; correct decorator
  order is `@tool` on top of `@tracked_tool(app_name=...)` (matches our
  `build_tools`, and the verified-working reference wrapper).
- `prepare_node` extracts `tool.coroutine` -> `.func` -> `._run` into
  `adapter._tools_context` (line 365/378); the sandbox awaits these directly,
  bypassing `args_schema`.
- Policy files need frontmatter `id` (e.g. `playbook_<name>`) or
  `filesystem_sync` deletes them from storage after load. `materialize_harness`
  already writes `id` correctly — confirmed by the `be-concise` vs
  `status-format` sync lines in the log.
- `skills_folder` must be the folder CONTAINING `skills/`; `materialize_harness`
  writes `<workspace>/skills/<name>/SKILL.md` and passes the workspace. Correct.

## Reproduction commands

```bash
uv run python -m scripts.diagnose_tool_prompt        # proves code emission + execution
uv run python -m scripts.bisect_instructions_contract # unguessable probe, 2 trials/config
uv run python -m scripts.verify_multistep_e2e        # 3-tool chain via real wrapper
uv run python -m scripts.verify_harness_injection    # structural only (can false-green)
uv run python -m scripts.verify_harness_behavioral   # AUTHORITATIVE: all 3 classes
uv run python -m scripts.test_policy_triggers        # keyword vs NL trigger matching
uv run pytest                                        # 608 passed, 1 skipped
uv run python -m scripts.diagnose_callback_config    # A/B: callbacks vs none, N trials
uv run python -m scripts.diagnose_run_task_error     # surfaces a swallowed invoke error
```
Logs land in `terminal_output/cuga-tracing/`. `2>&1 | tee` is required by AGENTS.md.
Note: macOS zsh has no `timeout`; do not use it.
Delete `data/workspaces/<task-id>` before re-running a harness test — a stale
workspace can leave a previously-written playbook in place.

## Evidence artifacts

- `terminal_output/cuga-tracing/tool-prompt-diagnosis.log` + `.json`
- `terminal_output/cuga-tracing/bisect-instructions-contract.log`
- `terminal_output/cuga-tracing/e2e-multistep.log`
- `terminal_output/cuga-tracing/e2e-tool-execution.jsonl` (ground-truth calls)
- `terminal_output/cuga-tracing/harness-injection-structural.log`
- `terminal_output/cuga-tracing/harness-behavioral.log`
- `terminal_output/cuga-tracing/callback-config-ab.log` (D4 A/B: 0/4 both arms)
- `terminal_output/cuga-tracing/d4-code-emission.log` (no ``` fence emitted)
- `terminal_output/cuga-tracing/d4-prompt-tools.log` (tool WAS in prompt+sandbox)
- `terminal_output/cuga-tracing/d4-vocabulary-ab.log` (0/3, 0/3, 3/3)
- `terminal_output/cuga-tracing/d4-framing-isolation.log` (refuted vocabulary)
- `terminal_output/cuga-tracing/d4-tool-construction.log` (1/3 vs 1/3)
- `terminal_output/cuga-tracing/d4-prompt-determinism.log` (all-or-nothing proof)
- `terminal_output/cuga-tracing/full-suite-d3.log` (608 passed, 1 skipped)
- `terminal_output/cuga-tracing/diagnose-run-task-error.log` (the `run_inline` bug)
- Traces: `data/traces/2b3ced93-6929-404b-803f-56a2a22cc003` (3-tool chain),
  `data/traces/70e605d6-698d-4665-82a3-7ea229691175` (harness behavioral),
  `data/traces/9be8a129-f09f-4d8f-8a9b-a952a20198e5` (D3: stream_events +
  graph_final_state captured, events.jsonl + checkpoints/ present)

## Git state

- Branch `dev4`, HEAD `1b3df2e "phase7 v1"`. Nothing committed this session.
- Modified this session:
  - `.env` — added `DYNACONF_ADVANCED_FEATURES__ENABLE_SHELL_TOOL=true` (D1 fix)
  - `src/agent_evolve/cuga_wrapper/__init__.py` — `materialize_harness` now emits
    a quoted `natural_language` trigger (D2 fix)
  - `tests/test_cuga_wrapper.py` — 2 playbook trigger/YAML tests, plus 6 D3 tests
    (callback node events, graph final state without a second execution, honest
    runtime_failure when no events arrive, model in manifest, LangChain callback
    contract, persisted invoke exception)
  - `src/agent_evolve/core/trace.py` — `CausalTrace.error` field added
  - `src/agent_evolve/cuga_wrapper/__init__.py` (D3) — `GraphEventCollector`,
    `build_graph_callback_handler`, `_final_state_snapshot`,
    `_stream_events_capability`, `CugaSdkRuntime(model=...)`, error persistence
  - `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md` —
    578 -> 785 lines, the durable cross-project artifact. Added this session:
    removed the disproved "`triggers: {always: true}` works" advice; the
    always-never-matches root cause; `keywords`-plural and quoted-scalar traps;
    the skills `enable_shell_tool` gate; the guessable-probe rule; a new
    "Node-Level Tracing From One `invoke()`" section (callbacks via
    `invoke(config=...)`, the `BaseCallbackHandler` subclassing requirement,
    `get_state` after invoke, honest capture-status taxonomy, never swallow the
    exception); and "Tool Invocation Can Be A Deterministic Function Of Prompt
    Wording" with the measured phrasing table and the identical-prompt
    non-sampling rule
  - `.gitignore` — ignore `cuga_workspace/`, `data/workspaces/`, and ingested
    knowledge files (all regenerated per run)
  - `docs/architecture/cuga-adapter/sdk-verification-matrix.md`
  - `docs/superpowers/plans/2026-08-14-session-handoff.md`
- Deleted `scripts/bisect_wrapper_config.py` (guessable probe produced a false
  negative; superseded by `bisect_instructions_contract.py`).
- Untracked diagnostics: `scripts/diagnose_tool_prompt.py`,
  `scripts/bisect_instructions_contract.py`, `scripts/verify_multistep_e2e.py`,
  `scripts/verify_harness_behavioral.py`, `scripts/test_policy_triggers.py`,
  `scripts/live_trace_smoke.py`, `scripts/probe_tool_tracing.py`,
  `scripts/diagnose_callback_config.py`, `scripts/diagnose_run_task_error.py`,
  `scripts/diagnose_code_emission.py`, `scripts/diagnose_prompt_tools.py`,
  `scripts/diagnose_probe_vocabulary.py`, `scripts/diagnose_framing_isolation.py`,
  `scripts/diagnose_tool_construction.py`, `scripts/diagnose_prompt_determinism.py`,
  `feedback/ganeral/`
- Do NOT commit without explicit user request.

## Next steps

D4 is closed; no wrapper fix is needed. Remaining work, in order:

1. **Re-phrase the verification scripts' task text** to the verified-executing
   form ("Write and execute Python code that calls `X()`, then report the exact
   value it returned"). `scripts/verify_multistep_e2e.py` and
   `scripts/verify_harness_behavioral.py` currently use wording that
   deterministically does NOT execute tools on this model, so they under-report.
   This is a test-instrument change, not a product change.
2. **`tool_observations`** — wire `ToolObservationRecorder.wrap()` into the live
   tool path, or delete the class and report the facility honestly. It is
   currently dead code: only `.replay()` is reachable. This is the last
   `unavailable_no_sdk_surface` facility that is implementable today.
3. **`graph_history`** via `get_state_history` — verified on a standalone
   LangGraph probe (4 entries with `MemorySaver`) but NOT against CUGA's graph.
   Verify before claiming. Keep `supports_counterfactual_replay()` false
   regardless: reading state is not state reconstruction.
4. Final acceptance: one complex multistep task exercising tools + skills +
   policies + memory with a complete saved, inspectable trace.

Do not re-open D4 by retrying identical prompts. If tools do not execute, first
check the task phrasing against the table above.
