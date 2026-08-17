# CUGA SDK Integration Learnings

This note records findings from integrating the installed CUGA SDK. It separates
verified behavior from hypotheses. Recheck source paths and behavior when
changing CUGA versions.

Validated against `cuga==0.3.1` (note: `cuga.__version__` misreports `0.2.20`),
model `openai/azure/gpt-5.6-luna` via a LiteLLM OpenAI-compatible endpoint,
`balanced` mode. Line numbers refer to that installed version.

Several sections below are marked **verified** because a live run produced the
quoted log line or observable side effect. Anything not so marked should be
treated as a hypothesis to retest.

## Environment And Configuration

### `.env` Is Loaded By The Wrapper, But Is Not Shell Environment

`printenv NAME` only displays variables exported into the current shell. It
does not parse the project's `.env` file. Therefore a blank result from:

```bash
printenv DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE
```

does not mean the setting is unavailable to `run2.py`.

This wrapper calls:

```python
load_dotenv(ROOT / ".env")
```

before importing CUGA. A clean normal invocation was verified to produce:

```text
os.getenv("DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE") == "true"
settings.advanced_features.force_autonomous_mode is True
```

The shell-prefixed form is not required when `.env` has the same value:

```bash
DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true uv run python run2.py "..."
```

and:

```bash
uv run python run2.py "..."
```

both enable autonomous mode when the repository `.env` contains:

```dotenv
DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true
```

`python-dotenv` uses `override=False` by default. A pre-existing exported
shell value takes precedence over `.env`; inspect it with `printenv` only when
debugging a possible conflicting shell value.

### Empty Configuration Directory Breaks CUGA Import

`CUGA_CONFIGURATIONS_DIR` is optional, but an empty environment value is not
treated as unset by CUGA. CUGA reads it with:

```python
CONFIGURATIONS_DIR = os.environ.get(
    "CUGA_CONFIGURATIONS_DIR",
    os.path.join(PACKAGE_ROOT, "configurations"),
)
```

Therefore `CUGA_CONFIGURATIONS_DIR=` resolves model files as relative paths:

```text
models/settings.openai.toml
```

instead of using CUGA's installed configuration directory. This fails during
the CUGA import, before an agent is created or a trace is produced.

**Required wrapper behavior:** remove blank or whitespace-only values before
importing `cuga`; preserve explicit non-empty values for a real custom CUGA
configuration tree.

```python
value = os.getenv("CUGA_CONFIGURATIONS_DIR")
if value is not None and not value.strip():
    os.environ.pop("CUGA_CONFIGURATIONS_DIR", None)
```

**Evidence:** `cuga/config.py:72-74`, `cuga/config.py:267-294`.

## Agent Execution APIs

### `stream()` And `invoke()` Are Separate Executions

`CugaAgent.stream(message)` runs `graph.astream(...)`. `CugaAgent.invoke(message)`
runs `graph.ainvoke(...)`. Calling both with the same prompt starts two graph
executions. They do not share a continuation unless the caller deliberately
reuses a thread and sends a new turn or resumes an interruption.

Do not use this pattern to obtain both a trajectory and `tool_calls`:

```python
async for event in agent.stream(prompt):
    ...
result = await agent.invoke(prompt, track_tool_calls=True)
```

It repeats model inference and may repeat external tool calls, writes, and
other side effects. In this project it could run `save_note` twice.

**Choose one primary API per task:**

- Use `invoke(..., track_tool_calls=True)` when the required output is the
  final answer plus SDK-provided aggregated tool calls.
- Use `stream(...)` when the required output is a node-level trajectory.
- If both are needed, **attach a callback handler to the single `invoke()`
  call** — see "Node-Level Tracing From One `invoke()`" below. This is verified
  and removes any reason to run the task twice.

**Evidence:** `cuga/sdk.py:2350-2793` and `cuga/sdk.py:2795-2934`.

### `track_tool_calls` Is An `invoke()` Parameter

The documented SDK surface for tool-call aggregation is:

```python
result = await agent.invoke(prompt, track_tool_calls=True)
print(result.tool_calls)
```

The SDK passes this flag in graph configuration and reads the completed graph
state's `tool_calls` field. `stream()` does not expose a matching public
parameter in this CUGA version.

**Evidence:** `cuga/sdk.py:2357`, `cuga/sdk.py:2445-2446`,
`cuga/sdk.py:2702-2703`.

## Node-Level Tracing From One `invoke()` (verified)

This is the answer to "I need a node-level trajectory AND `tool_calls` without
executing the task twice". It works, and it needs no `stream()` call.

### Pass callbacks through `invoke(config=...)`

CUGA merges caller-supplied handlers into the same `graph.ainvoke` execution.
`_apply_callbacks` (`cuga/sdk.py:1924-1962`) reads `run_config["callbacks"]`,
merges it with CUGA's built-ins (`TokenUsageTracker`), and writes the merged list
to **both** the top level and `configurable`:

```python
merged = built_callbacks + existing
run_config["callbacks"] = merged
run_config["configurable"]["callbacks"] = merged
```

So this yields node lifecycle events from the one execution that also returns
`tool_calls`:

```python
result = await agent.invoke(
    message,
    thread_id=thread_id,
    track_tool_calls=True,
    config={"configurable": {"thread_id": thread_id}, "callbacks": [handler]},
)
```

Verified node names seen in `metadata["langgraph_node"]` for a CugaLite run:
`CugaLiteSubgraph`, `prepare`, `call_model`, `SDKCallback`, `FinalAnswerAgent`
(19 start/end events for a single-tool task).

`CugaAgent(callbacks=[...])` also exists and is baked in as `base_callbacks` so
direct `agent.graph.ainvoke(...)` is instrumented too (`sdk.py:2039-2051`), but
per-call `config` is what `invoke()`/`stream()` actually use.

### The handler MUST subclass `BaseCallbackHandler` — duck typing breaks the run

A handler that merely implements `on_chain_start` is **not** enough. LangChain's
async dispatch reads handler attributes directly:

```python
# langchain_core/callbacks/manager.py:471
for handler in [h for h in handlers if h.run_inline]:
```

A duck-typed object raises mid-run, aborting the whole graph execution:

```text
AttributeError: 'GraphEventCollector' object has no attribute 'run_inline'
```

Keep the collector agent-neutral (no LangChain types) and adapt it at the edge:

```python
def build_graph_callback_handler(collector):
    from langchain_core.callbacks import BaseCallbackHandler   # lazy import

    class _Handler(BaseCallbackHandler):
        def on_chain_start(self, serialized=None, inputs=None, **kwargs):
            metadata = kwargs.get("metadata") or {}
            node = metadata.get("langgraph_node")
            if node:
                collector.record("graph_node_start", {"node": node})
        # on_chain_end / on_chain_error / on_tool_* likewise

    return _Handler()
```

Assert `isinstance(handler, BaseCallbackHandler)` in a test. That single
assertion catches this class of bug before a live run does.

### Persist only structural identifiers from callbacks

Node inputs and outputs pass through these callbacks and can contain evaluator
internals, expected answers, or raw payloads. Record the node name, step index,
and tool name — not `inputs`/`outputs` — unless the consuming project has
explicitly approved persisting payloads.

### Final graph state is readable after `invoke()` without re-execution

`CugaAgent.graph` always compiles with a `MemorySaver` checkpointer and
`interrupt_before=["WaitForResponse"]` (`sdk.py:2291-2301`). So after `invoke()`
returns, the state left by that run is directly readable:

```python
state = agent.graph.get_state({"configurable": {"thread_id": thread_id}})
state.values          # full CugaLite state dict (60+ keys)
state.next            # () when the graph completed, non-empty if interrupted
state.config["configurable"]["checkpoint_id"]
```

This is **not** a second execution and does not repeat side effects. Do not pass
the callback handler into `get_state` — only the `thread_id`.

Two cautions:

- Reading a final state is **not** a verified replay/state-reconstruction
  capability. Report any such snapshot as not replay-safe, and do not advertise
  counterfactual replay on the strength of it.
- `get_state_history(...)` also exists on the compiled graph (verified to return
  4 entries on a standalone LangGraph probe with `MemorySaver`), but that was not
  verified against CUGA's graph. Verify before claiming it.

### Report capture status honestly, distinguishing failure modes

"Not captured" has several distinct causes and collapsing them destroys
diagnosability. Distinguish at minimum:

- `disabled_by_config` — the caller turned capture off.
- `unavailable_no_sdk_surface` — e.g. `invoke()` has no `config` parameter.
- `runtime_failure` + reason `"callback handler attached but emitted no events"`
  — wired correctly, produced nothing. This is the state that reveals a broken
  handler contract.
- `captured` — only when real events exist.

### Never let a swallowed exception masquerade as a normal result

A wrapper that converts any invoke exception into `status="error"` with an empty
output and **no recorded reason** will hide exactly the bug above: the run looks
like an ordinary empty-answer failure. Persist the exception text (`repr`) into
the result and the trace. Cost of omitting it here: one full misdiagnosis cycle.

## Tool Execution And Trajectories

### Tool Calls Are Recorded In CUGA Lite State

When code reaches CUGA Lite's sandbox, it starts a tool tracker, executes the
script, stops tracking, and appends calls to `state.tool_calls` only when
`track_tool_calls` is enabled.

```python
execution_tool_calls = ToolCallTracker.stop_tracking()
accumulated_tool_calls = (state.tool_calls or []) + (
    execution_tool_calls if track_tool_calls else []
)
```

This means a stream trace can show sandbox execution in intermediate events,
while the final top-level state may not expose a useful `tool_calls` list if
tracking was not configured for that run.

**Evidence:**
`cuga/backend/cuga_graph/nodes/cuga_lite/adapter/sandbox_node.py:239-268`.

### A Registered Tool Is Not Necessarily A Working Tool

CUGA logging `Created DirectLangChainToolsProvider with 5 tools` proves tool
registration only. A tool can still fail when invoked due to code in the
wrapper. Verify tools directly and through an agent run.

Example failure found here:

```text
NameError: name 'quote_plus' is not defined
```

`wikipedia_search` was registered but failed at runtime because its wrapper
referenced `quote_plus` and `re` without module-level imports.

### Never Diagnose Tool Execution With A Guessable Task

This cost a full debugging session and produced a confidently wrong root cause,
so it is the single most important methodology rule here.

Symptom: a multi-step run returned a correct answer but
`InvokeResult.tool_calls == []`, and the causal trace recorded zero tool events.
The conclusion drawn was "this reasoning model cannot emit executable code, so
CUGA never routes to the sandbox". **That conclusion was wrong.**

The real explanation: the probe tasks were *mentally solvable*. Asked to compute
`1234 * 5678` "using the calculator tool", the model simply did the arithmetic
and answered correctly without calling anything. `tool_calls: []` was **truthful
reporting of a run in which no tool was needed** — not a tracing defect and not a
model limitation.

Re-tested with a tool returning a per-run random token that cannot be derived or
recalled, everything worked on the first attempt:

```text
extract_code_from_model_response -> "sum_result = await diag_add(17, 25)"
tool function body executed       -> True
InvokeResult.tool_calls           -> 1 fully populated record
```

Rules for any tool-execution probe:

- **The tool's return value must be unguessable** — a random token per run, or a
  value that exists only behind the tool (remote state, generated id).
- **Ground truth is the tool function body executing**, recorded by the tool
  itself (append to a list or a file). Never trust the model's narrative: a model
  will happily say "Calling the tool now..." in a turn where nothing ran.
- **Never assert on a correct final answer.** A correct answer proves only that
  the model produced the right string, which it may have done from prior
  knowledge or arithmetic.
- **Chain multiple tools with a data dependency** for multi-step verification, so
  step *n+1* is impossible without the real output of step *n*. Example that
  works well: `fetch_token()` -> `exchange(token)` -> `checksum(result)`, where
  each tool rejects an input it did not itself produce.
- **Repeat each configuration** before attributing a failure to a config change;
  a single run of a nondeterministic agent proves nothing.

Corollary for interpreting logs: `Error while calling registry to get apps` and
`Cannot connect to host localhost:8001` are **harmless fallbacks**. They appear
in fully successful runs that execute every tool. Do not treat them as the cause
of missing tool calls.

### Tool Invocation Can Be A Deterministic Function Of Prompt Wording (verified)

The companion rule to the one above, and it changes how you sample. With
*everything* held constant — same CUGA version, same tools, same agent config,
same registered callable — whether the agent invokes the tool at all can be
decided entirely by incidental task wording.

Measured on `openai/azure/gpt-5.6-luna`, one trivial single-tool task
(`read_build_number`), ground truth = tool function body executed:

```text
task suffix                              executed
(none)                                    0/2
"Respond with only the value."            0/2
"Return just the value, nothing else."    2/2
"First call X. Then report..."            0/2
"Write and execute Python code that
 calls read_build_number(), then..."      2/2
```

Every phrasing was **all-or-nothing** — never 1/2. Failing runs produced no
```` ``` ```` fence at all, so `extract_code_from_model_response` returned `""`,
`call_model` took the no-code branch, and the graph never reached the sandbox.
The model narrated *"I'm unable to call the tool in the current environment"*,
which is simply false: the tool was in the prompt and registered in the sandbox.

Consequences for methodology:

- **Repeating an identical prompt is not sampling.** Reasoning models often skip
  temperature (`Skipping temperature for reasoning model: ...` in CUGA's log), so
  decoding is effectively greedy: the same prompt gives byte-identical output.
  Three "trials" of one prompt is one observation reported three times. Vary the
  **prompt**, not the trial index, and treat identical-prompt repeats as one
  sample.
- **Do not call this "flaky".** It is reproducible per prompt. Calling it flaky
  invites a retry loop that will never converge.
- **A/B one variable at a time, and re-run the control in the same session.** A
  wording arm that passed 3/3 earlier scored 0/3 later once its task suffix
  changed — which is what exposed the suffix, not the vocabulary, as the cause.
- **Prefer an explicit code-execution instruction** for probes and for any task
  that must exercise tools: *"Write and execute Python code that calls `X()`,
  then report the exact value it returned"* was one of only two reliable
  phrasings. Vague "use the tool" phrasing is the weakest form.
- **Do not attribute this to safety filters without evidence.** One arm did
  produce a real refusal (*"I can't provide or reveal secret tokens"*, provoked
  by probe vocabulary such as "secret"/"token"/"reveal"), but neutral-vocabulary
  arms failed too. Avoid secrecy framing in probe names, docstrings, and tasks —
  it can trigger a genuine refusal and confound the measurement — but do not
  stop investigating once you find one refusal.
- **This is a model property, not a CUGA or wrapper defect.** Before changing
  wrapper code, prove the tool reaches the prompt and the sandbox. Instrument
  the prompt boundary (does the rendered prompt contain the tool name and the
  code-fence contract?) and the sandbox registration.

Useful boundary-instrumentation points for this class of investigation:

```text
cuga/.../graph/shared_nodes.py:197   extract_code_from_model_response(...)
                                     -> did the model emit extractable code?
cuga/.../graph/shared_nodes.py:233   if code: goto execute_node
                                     -> the exact routing decision
cuga/.../adapter/graph_adapter.py    AgentGraphAdapter.ainvoke_model
                                     -> the fully rendered prompt
cuga/.../adapter/prepare_node.py     make_tool_awaitable
                                     -> what was registered into the sandbox
```

## Why Multi-Step Runs Can End Early

### Verified Graph Routing

After each model response, CUGA does the following:

1. Extract Python code from model content and reasoning.
2. If code is extracted, route to the sandbox.
3. If no code is extracted, use a natural-language auto-continue classifier.
4. If the classifier says false, set the response as `final_answer` and route
   to graph end.

Simplified SDK behavior:

```python
if code:
    return Command(goto=adapter.execute_node_name, ...)

should_continue = await adapter.classify_auto_continue(
    state, active_model, content, reasoning
)
if should_continue:
    return Command(goto="call_model", ...)

return Command(goto=END, update={"final_answer": final_answer, ...})
```

**Evidence:**
`cuga/backend/cuga_graph/nodes/cuga_agent_core/graph/shared_nodes.py:196-284`.

### Observed Failure Mode

For a request to verify five tools, the agent executed `calculator`, then
produced narrative and several code blocks for the remaining tools. CUGA
routed to `FinalAnswerAgent` before executing those later blocks. The final
answer correctly admitted that only the calculator had been confirmed.

This is not `stream()` terminating early. The graph itself reached its final
answer route.

The same type of early finalization occurred with direct `invoke()` testing:
the agent returned an initial plan with no tool calls instead of executing the
requested tools.

### Verified Autonomous-Mode Outcome

For this installed CUGA version, enabling autonomous mode fixed the observed
multi-step execution failure. With:

```dotenv
DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true
```

and one `agent.stream(...)` execution, CUGA repeatedly followed this cycle:

```text
model response containing code
-> sandbox execution
-> execution output added to context
-> next model response
```

The tool-verification task completed real invocations of all five custom
tools before producing its final answer:

- `calculator`: `2 + 2` returned `4`.
- `web_search`: returned an OpenAI result and URL.
- `web_fetch`: returned HTTP 200 and page text.
- `wikipedia_search`: returned a Wikipedia result.
- `save_note`: wrote the verification note.

The relevant evidence is in `feedback/out3.txt` and `feedback/out4.txt`.
The normal `.env` invocation in `out4.txt` confirms that this outcome does
not depend on an inline shell environment assignment.

Autonomous mode is therefore the required default for this wrapper's
open-ended research and GAIA benchmark tasks. It does not guarantee every
model response will contain executable code, but it changes CUGA's routing so
completed substeps return to the planning loop instead of immediately ending
as a top-level final answer.

### Auto-Continue Classifier Hazard

CUGA's NL auto-continue classifier is intended to continue interim plans, but
it explicitly prefers finalization when text says the agent cannot continue or
tools are unavailable. A response containing both an intended plan and a
phrase such as "I cannot truthfully mark the remaining tools as working" can
therefore be classified as final even if work remains.

**Evidence:**
`cuga/backend/cuga_graph/nodes/cuga_lite/nl_auto_continue_classifier.py:12-47`
and `:182-220`.

## Top-Level Versus Autonomous Execution

CUGA treats a task as autonomous if either:

```python
settings.advanced_features.force_autonomous_mode
or state.sub_task is not None
```

Regular top-level CUGA Lite execution routes a successful result directly to
`FinalAnswerAgent`. Autonomous subtasks return to `PlanControllerAgent`, which
can schedule additional work.

**Evidence:**
`cuga/backend/cuga_graph/nodes/cuga_lite/cuga_lite_node.py:267-273` and
`:466-586`.

> **This routing does NOT apply to the SDK.** The code above lives in a file the
> SDK graph never imports. See the next section — reading `cuga_lite_node.py` and
> assuming it describes `CugaAgent` behavior cost a full debugging cycle.

## CUGA Ships TWO Graph Topologies; The SDK Gets The Simplified One (verified)

The single most consequential architectural fact found so far, and the one most
likely to make a wrapper author debug a non-existent bug. CUGA contains **one set
of agent implementations** but **two independent graph assemblers** that wire them
into fundamentally different topologies.

|                        | Server / full graph                        | SDK graph (what a wrapper gets)              |
| ---------------------- | ------------------------------------------ | -------------------------------------------- |
| Assembled by           | `backend/cuga_graph/graph.py:63` (`DynamicAgentGraph`) | `sdk.py:2014` (`_create_hitl_wrapper_graph`) |
| Reached via            | `DynamicAgentGraph(...)`                   | `CugaAgent(...)` → `_create_graph` (`:2006`) |
| Real nodes             | ~20                                        | 5 real + 3 dummy stubs                       |
| `PlanControllerAgent`? | **Yes** (`graph.py:164`)                   | **No — not even stubbed**                    |
| CugaLite callback      | `CugaLiteCallback` (`graph.py:282`)        | `sdk_callback_node` (inline in `sdk.py`)     |

### The symptom this explains

`PlanControllerAgent` appeared in **0 of 15** rollouts with
`force_autonomous_mode=True`, while `cuga_lite_node.py:529-571` plainly routes
success there when autonomous. Two plausible hypotheses — `_has_error` firing
every time, or early termination — were **both wrong**:

* `_has_error`: applying CUGA's own error-indicator list (`cuga_lite_node.py:229`)
  to `final_output` across 16 rollouts gave **0 hits**. And it would not have
  mattered: under autonomous mode `:498` (error) and `:571` (success) *both* go to
  `PlanControllerAgent`. If that code ran at all, it would be 15/15, not 0/15.
* early termination: all 104 traces reach `SDKCallback` **and**
  `FinalAnswerAgent`. Nothing terminated early.

The actual reason: **our graph never contains the node.** `CugaLiteNode` (which
holds that routing) is imported by exactly one non-test file — `graph.py:46`, the
server graph. `sdk.py` contains no `PlanControllerAgent` reference in its node
registration at all.

### What the SDK registers (`sdk.py:2144-2154`)

```python
wrapper.add_node("CugaLiteSubgraph", compiled_subgraph)
wrapper.add_node("SDKCallback", sdk_callback_node)
wrapper.add_node(suggest_actions.name, suggest_actions.node)      # HITL
wrapper.add_node(wait_for_response.name, wait_for_response.node)  # HITL
wrapper.add_node(final_answer_node.final_answer_agent.name, final_answer_node.node)
# Dummy nodes purely so internal CugaLiteSubgraph routing references resolve:
wrapper.add_node(NodeNames.API_PLANNER_AGENT, dummy_api_planner_node)
wrapper.add_node(NodeNames.CHAT_AGENT, dummy_chat_agent_node)
wrapper.add_node(NodeNames.CUGA_LITE, dummy_cuga_lite_node)
```

Note lines 2152-2154: `APIPlannerAgent` and `ChatAgent` exist only as **stubs
whose entire body routes back to `SDKCallback`**. Their presence in a node list
is not evidence they do anything. `sdk.py:2017` states the shape outright:
*"Graph structure (simplified for SDK): START -> CugaLiteSubgraph -> SDKCallback"*.

### The substitution that removes planning

The server graph uses `CugaLiteCallback` — the planning-capable callback in
`cuga_lite_node.py`. The SDK uses `sdk_callback_node`, a **separate
reimplementation defined inline in `sdk.py`**. It handles tool-approval HITL,
applies output-formatter policies, then unconditionally finalizes
(`sdk.py:2126-2134`):

```python
# Otherwise, route to FinalAnswerAgent
answer = state.final_answer or "No answer found"
state.sender = NodeNames.CUGA_LITE
return Command(update=state.model_dump(), goto=NodeNames.FINAL_ANSWER_AGENT)
```

It never calls `_has_error` and never reads `is_autonomous_subtask`. **There is no
code path from the SDK graph to `PlanControllerAgent`.**

### What the SDK path actually is

`create_cuga_lite_graph` (`cuga_lite_graph.py:159-214`) builds exactly three
nodes — `prepare_node`, `call_model_node`, `sandbox_node` (`:204-206`). Confirmed
by trace evidence: 104 traces, six distinct actors, **identical shape every
time**, terminal always `FinalAnswerAgent`:

```text
CugaLiteSubgraph → prepare → call_model ⇄ sandbox → SDKCallback → FinalAnswerAgent
```

Actor counts across 104 traces: `call_model` 518, `prepare` 448,
`CugaLiteSubgraph` 392, `sandbox` 248, `SDKCallback` 196, `FinalAnswerAgent` 196.
`PlanControllerAgent` 0, `CugaLiteCallback` 0.

So the SDK is a **single-agent ReAct/CodeAct loop**, not hierarchical
planner/executor decomposition. It *does* re-plan — empirically, inside the
`call_model ⇄ sandbox` loop, with observable mid-run correction ("The prior result
used an incorrect time (2:01:09). I'll verify…") — but there is no distinct
planner node and no inspectable plan artifact.

**`force_autonomous_mode` is still consumed on the SDK path**, just not for
routing: it changes *prompt content* (`prepare_node.py:630`, `sandbox_node.py:210`,
and 5 conditionals in `mcp_prompt.jinja2`). It is not being ignored.

### "Server CUGA" is prebuilt and local — not a hosted API

Worth stating plainly, because "server" invites the wrong guess:

* `DynamicAgentGraph` **ships in the same installed wheel**. There is nothing to
  build or download.
* It is not an IBM-hosted or online service. Its main consumer is a **local
  FastAPI app** (`backend/server/main.py:1992`, `app = FastAPI(lifespan=lifespan)`)
  with ~70 routes (`POST /stream`, `POST /reset`, `GET /api/agent/state`,
  `/api/config/policies`, `/api/skills`, a browser-extension channel), started via
  the `cuga` console script under `uvicorn` (`main.py:488`). Fully offline.
* **You do not need the server to use the full graph.** Two call sites build
  `DynamicAgentGraph` in-process with no HTTP at all:
  `backend/cuga_graph/utils/controller.py:193` and `:299`, plus
  `backend/cuga_graph/policy/tests/helpers.py:298`.

The real migration cost is therefore **not** construction — it is that the two
entry points take **different constructor surfaces**. `CugaAgent` takes
`cuga_folder`, `skills_folder`, `enable_skills`, `auto_load_policies`,
`reset_policy_storage`, `filesystem_sync`. `DynamicAgentGraph` takes
`policy_system`, `tool_provider`, `llm_config`, `enable_todos`,
`reflection_enabled`, `shortlisting_tool_threshold`, `cuga_lite_max_steps`
(`main.py:951-960`). Every artifact-injection finding in this document was verified
against the `CugaAgent` surface and **must be re-verified**, not assumed, before
trusting it on `DynamicAgentGraph`.

### Planning knobs available *without* changing graphs

`enable_todos` and `reflection_enabled` are read at runtime from
`config["configurable"]`, falling back to `settings.advanced_features`
(`cuga_lite_graph.py:173`). Both default `false`. Enabling them gives explicit
todo tracking (`create_update_todos`) and reflection inside the SDK graph — the
closest thing to planning without leaving `CugaAgent`. Caveat: it changes prompt
content, so **any baseline collected with them off is not comparable**.

### Methodology rules this establishes

* **Before believing a routing edge applies to you, prove the file containing it
  is imported by *your* graph.** `grep` for the node class in the assembler you
  actually construct. A routing block in the package is not a routing block in
  your run.
* **Enumerate node names from real traces before theorizing about a missing
  node.** A single scan of on-disk traces (six actors, stable across 104 runs)
  refuted two hypotheses at zero inference cost. Do this first; it is free.
* **A node present in `add_node` may be a dummy.** Read its body before counting
  it as a capability.
* **Never describe SDK-path results as hierarchical planning.** The numbers are
  valid; prose claiming a planner/executor split contradicts the trace and would
  not survive review.

## Recommended Investigation And Fix Order

1. Keep one agent execution per user task. Never restore a second same-prompt
   `invoke()` after `stream()` merely to obtain metadata.
2. Use `invoke(..., track_tool_calls=True)` for benchmark execution when its
   completed tool-call list is required. Save the returned final state as the
   trajectory, or attach SDK callbacks if node-level events are also required.
3. Test the exact multi-tool prompt with
   `DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true` set before CUGA is
   imported. This was verified for this installed version: it kept the
   five-tool verification task in the planning loop through all five real
   invocations.
4. If autonomous mode does not solve the issue, inspect the raw model response
   immediately before `extract_code_from_model_response` and the classifier
   decision. Determine whether failure is code extraction or false
   auto-continuation classification.
5. Prefer a CUGA policy/playbook or a focused prompt contract requiring one
   executable fenced Python block per turn, followed by a final answer only
   when all requested checks are complete. This reduces reliance on fragile
   mixed narrative-plus-code responses.

## Practical Wrapper Checklist

- **Know which graph you are on.** `CugaAgent` (SDK) builds a 5-node simplified
  graph with NO `PlanControllerAgent`; `DynamicAgentGraph` (server) builds the
  ~20-node hierarchical one. Routing code in `cuga_lite_node.py` applies only to
  the latter. Verify by enumerating node names in a real trace, not by reading
  package source.
- **`DynamicAgentGraph` is prebuilt and local**, not a hosted API, and is
  constructible in-process without the FastAPI server
  (`cuga_graph/utils/controller.py:193`). Migration cost is the *different
  constructor surface*, not building anything.
- Normalize blank optional CUGA environment variables before importing CUGA.
- Set `DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true` in `.env` for
  autonomous research and benchmark workloads.
- Do not use `printenv` to check whether `.env` was read; inspect the value
  inside the Python process and the resulting CUGA setting instead.
- Import every dependency used by custom tools at module scope.
- Test every custom tool directly with representative input.
- Execute each task once only.
- Use a stable `thread_id` only for a real continuation or HITL resume, not to
  re-run a finished prompt.
- Persist raw stream events or callback output when diagnosing tool routing.
- Record the CUGA version and source paths used to validate behavior. Note that
  `cuga.__version__` can be stale/misleading (reported `0.2.20` for an installed
  `cuga==0.3.1`); trust the distribution metadata, not the attribute.
- Treat an empty `result.tool_calls` as "no tracked calls surfaced", not proof
  that no tool code executed; inspect sandbox events before concluding. Equally,
  do not treat it as proof of a defect — verify with an unguessable probe first
  (see "Never Diagnose Tool Execution With A Guessable Task").
- Phrase any tool-exercising task as an explicit code-execution instruction
  ("write and execute Python code that calls `X()`, then report ..."). Vague
  "use the tool" wording deterministically fails on some models.
- Never repeat an identical prompt and call it N trials; reasoning models with
  temperature skipped decode greedily. Vary the prompt to get real samples.
- Attach tracing via `invoke(config={"callbacks": [handler]})`, and make the
  handler a real `BaseCallbackHandler` subclass. Assert `isinstance` in a test.
- Read final graph state with `agent.graph.get_state({"configurable":
  {"thread_id": ...}})` after `invoke()`; never re-run the task for state.
- Always persist the exception text when a run fails; an opaque `status="error"`
  hides handler/contract bugs and costs a full misdiagnosis cycle.
- Verify artifact injection **behaviorally**, not structurally. Checking that
  CUGA loaded a skill/policy into its stores can report success while the
  artifact never reaches the model. Assert that an unguessable marker from each
  artifact appears in the final output.
- When generating policy/skill files programmatically, parse the frontmatter you
  emit (`yaml.safe_load`) in a test; invalid YAML drops the policy silently.
- CUGA writes global state under `<cwd>/.cuga/` (playbooks, skills, knowledge,
  intent_guards, ...) and per-thread sandboxes under `<cwd>/cuga_workspace/`.
  Both are regenerated per run — gitignore them, and remember that a policy or
  skill written by an earlier run can persist into a later one.

## Injecting Editable Harness Artifacts (Skills, Policies, Memory)

These findings come from wiring the AgentEvolve wrapper's "harness"
(`instructions`, `tools`, `skills`, `policies`, `memory`) into a fresh
`CugaAgent` per run. The harness is the evolvable unit; each class below is a
separately verified CUGA surface and must NOT be generalized from another.

### Verified constructor surface

`CugaAgent` (installed 0.3.1) accepts all of `tools`, `tool_provider`, `model`,
`callbacks`, `policy_system`, `special_instructions`, `cuga_folder`,
`auto_load_policies`, `reset_policy_storage`, `filesystem_sync`,
`enable_knowledge`, `enable_citations`, `enable_skills`, `skills_folder`. They
exist in `inspect.signature(CugaAgent.__init__)` — do not treat them as
"undocumented" just because a docstring omits one.

### Skills (`enable_skills` + `skills_folder`)

- `skills_folder` is the folder that **contains** `skills/`, not the `skills/`
  directory itself. CUGA discovers `<skills_folder>/skills/**/SKILL.md`.
  (`cuga/backend/skills/loader.py:get_skill_root` resolves the root via
  `settings.skills.root`, default `"cuga"` → `<cuga_folder>/skills`.)
- Each `SKILL.md` needs YAML frontmatter with `name` **and** `description`
  (the loader raises "missing name or description" otherwise) plus a body.
- `settings.skills.enabled` **defaults to `False`** (`config.py` validator).
  Passing `enable_skills=True` sets `configurable["skills_enabled"]` in the
  graph config, which `prepare_node` reads; `DYNACONF_SKILLS__ENABLED=true` is
  the global fallback.
- The skill is surfaced to the model as a `load_skill` tool plus a
  "available skills" prompt block, built in
  `cuga/backend/cuga_graph/nodes/cuga_lite/adapter/prepare_node.py`. It runs
  only when `skills_enabled` is truthy in the node's `configurable`.

#### Skills also require `enable_shell_tool=true` (verified)

`enable_skills=True` + a discoverable `SKILL.md` is **not sufficient**. CUGA
silently discards the entire skills prompt block unless the shell tool is also
enabled, so `discover_skills` logs a successful load while the model never sees
the skill.

`cuga/backend/cuga_graph/nodes/cuga_lite/prompt_utils.py:682-689`:

```python
if not enable_shell_tool:
    if skills_enabled:
        logger.warning(
            "Skills are enabled but enable_shell_tool=False; the skills block will be suppressed. "
            "Set advanced_features.enable_shell_tool=true to activate skills."
        )
    skills_enabled = False
    skills_prompt_section = ""
```

`settings.advanced_features.enable_shell_tool` **defaults to `False`**, so this
is the default outcome. Watch for that exact warning line whenever a skill
"loads" but has no effect. Fix:

```dotenv
DYNACONF_SKILLS__ENABLED=true
DYNACONF_ADVANCED_FEATURES__ENABLE_SHELL_TOOL=true
```

Confirmed working after the change: the model emitted
`skill_instructions = await load_skill("status-report")`, the sandbox executed
it, the returned skill body appeared in the execution output, and the skill's
marker text appeared in the final answer.

**Security tradeoff to accept consciously:** this also injects a real sandbox
shell tool into the agent's execution context —
`[NativeSandbox] Injected run_command (thread_id=...)`. The agent can then run
shell commands in its sandbox. Decide deliberately rather than enabling it
reflexively to make a skill work.

### Policies (`cuga_folder` + `auto_load_policies`)

- Policies live in `<cuga_folder>/<type>/*.md` where `<type>` is one of
  `playbooks`, `output_formatters`, `tool_guides`, `intent_guards`,
  `tool_approvals` (`folder_loader.py:load_policies_from_folder`).
- **Critical:** each policy file MUST carry an `id` in its frontmatter matching
  CUGA's convention (`playbook_<name>`, `output_formatter_<name>`, ...). CUGA's
  `filesystem_sync` reads frontmatter `id` to reconcile storage with disk
  (`filesystem_sync.py:get_filesystem_policy_ids`); a file without `id` is seen
  as "not in filesystem" and the just-loaded policy is **deleted from storage**
  during the post-load sync. Symptom: `load_from_folder` logs
  "Loaded 1 policies" but `await agent.policies.list()` returns `[]`.
- A playbook needs `name` and at least one trigger.
- `settings.policy.enabled` defaults `True`; `auto_load_policies` and
  `filesystem_sync` also default `True`.

#### An `always: true` playbook loads but NEVER matches (verified)

**Do not use `triggers: {always: true}` as your only trigger.** An always-only
playbook loads cleanly, deserializes into an `AlwaysTrigger`, appears in
`agent.policies.list()` — and then never fires. The artifact looks configured
while having zero effect on the agent.

Root cause in cuga 0.3.1 (`cuga/backend/cuga_graph/policy/agent.py`):

- `PolicyAgent._check_trigger` (line 167) *does* handle `AlwaysTrigger`
  correctly (line 178, returns `True, 1.0`).
- But `match_policy` (line 929) only builds candidates from
  `_evaluate_keyword_triggered_policies` (line 685, filters
  `isinstance(t, KeywordTrigger)`) and `_evaluate_natural_language_policies`
  (line 767, filters NL triggers).
- **No evaluator ever selects an `AlwaysTrigger`**, so an always-only policy is
  never a candidate and can never win.

Observed, same policy body, single variable changed:

```text
triggers: {always: true}         -> matched=False, "No policies matched the current context"
triggers: {keywords: [...]}      -> matched=True
triggers: {natural_language: []} -> matched=True, "Playbook guidance will be injected: <name>"
```

Use a `natural_language` trigger (works for open-ended intents) or `keywords`
(works when you can guarantee a literal term). Keeping `always: true` alongside
them is harmless as forward-compatible intent:

```yaml
---
name: status-format
id: playbook_status-format
triggers:
  natural_language:
    - "user asks for a project status report"
  target: intent
  threshold: 0.5
  always: true
---
Body text with the actual guidance.
```

#### Two silent-failure traps in playbook frontmatter

Both produce a policy that looks present but has no effect. Both are worth an
automated test in any wrapper that generates policy files.

**Trap 1 — the key is `keywords`, plural.** `folder_loader.py:85` reads
`triggers_config.get('keywords', [])`. Writing `keyword:` (singular) yields zero
triggers, and CUGA then rejects the whole file:

```text
Failed to load .../status-format.md: Playbook status-format must have at least one trigger
```

**Trap 2 — the trigger phrase must be a quoted YAML scalar.** Policy text very
often contains a colon (for example `"end your reply with the exact line:
MARKER"`). Emitted unquoted into YAML, that colon makes the frontmatter invalid
and the entire policy is dropped:

```text
Failed to load .../status-format.md: Invalid YAML in frontmatter: mapping values are not allowed here
Failed to parse policy file .../status-format.md: mapping values are not allowed here
```

The failure is easy to miss because the run still succeeds — only the policy is
gone. When generating a trigger phrase from arbitrary policy text, always
double-quote it and escape/strip embedded quotes and backslashes:

```python
trigger_phrase = derived_text.replace("\\", " ").replace('"', "'")
frontmatter = f'  natural_language:\n    - "{trigger_phrase}"\n'
```

Verify by parsing the generated frontmatter with `yaml.safe_load` in a test,
rather than trusting that the file "looks fine".

### Memory / knowledge (`enable_knowledge` + `agent.knowledge`)

- `agent.knowledge` (the direct API) is a **separate** gate from the
  `enable_knowledge=True` constructor kwarg. The property builds
  `KnowledgeConfig.from_settings(settings)` and reads
  `settings.knowledge.enabled`, which **defaults to `False`**
  (`knowledge_settings.toml: enabled = false`). `enable_knowledge` only controls
  graph tool injection; it does NOT enable `agent.knowledge.ingest/search`.
- To use `agent.knowledge.ingest(...)`, set
  `DYNACONF_KNOWLEDGE__ENABLED=true`. Otherwise you get
  `ValueError('Agent-level knowledge is disabled for this agent')`.
- `ingest(file_path, scope="agent")` takes a file path, not a string; write
  memory entries to files and ingest them, then `search(...)` / `list_documents()`.
- There is no first-class "memory" constructor surface; knowledge ingestion is
  the retrieval path.

### Wrapper wiring bug (don't repeat)

When building the agent per-task, pass the **full harness** (all of
`instructions`, `tools`, `skills`, `policies`, `memory`) into the agent
constructor. Filtering the config to only `instructions`/`tools` (an old
"only verified surfaces" habit) silently drops `skills`/`policies`, so
`enable_skills`/`auto_load_policies` end up `False` and the skill/policy never
loads even though the files were materialized correctly. Verify by spying on
`cuga.backend.cuga_graph.nodes.cuga_lite.adapter.prepare_node.discover_skills`
to confirm it is actually called with your skills folder.

## Building A CUGA Agent As A Tool-Driven Worker (verified)

These findings come from building an *editor* agent: a CUGA agent whose whole
job is to read evidence through tools and write results back through tools,
with its prose answer deliberately ignored. Everything below was found by live
runs after a full offline test suite passed, which is the point — none of it is
reachable from unit tests that stub the agent.

### `@tool` requires a docstring on every callable

`langchain_core.tools.tool` raises at construction:

```text
Function must have a docstring if description not provided.
```

A tool body with no docstring fails the whole agent build, so the first live run
dies before inference. The docstring is not decoration: it is the **tool
description the model reads when deciding whether to call the tool**. Write it
as a usage trigger, not a restatement of the name.

Pin it with a test; it costs nothing and the failure is otherwise live-only:

```python
missing = [n for n, f in build_tool_callables(ctx).items() if not f.__doc__]
assert missing == []
```

### A wrapper around a tool body must use `functools.wraps`

Wrapping tool callables (for call recording, timing, auth) with a bare
`*args, **kwargs` function breaks two things at once:

- `__doc__` is lost, so `@tool` raises the error above.
- The signature is lost, so `@tool` builds an **empty args schema** and the
  model is told the tool takes no arguments.

`functools.wraps` carries `__doc__` and sets `__wrapped__`, which is what
`inspect.signature` follows to recover the real parameters:

```python
@functools.wraps(fn)
def recorded(*args, **kwargs):
    names.append(name)
    return fn(*args, **kwargs)
```

### Instrumented callables must actually reach the agent

A wrapper that builds recorded callables, then constructs its tools from the
*original* source instead of the recorded dict, reports zero tool calls on runs
where tools demonstrably executed. This is the dangerous class of bug: the
machinery reports success while measuring nothing. Assert that the supplied
callable is the one invoked:

```python
built = build_editor_tools(ctx, {"get_mechanism": recorded})
built[0].invoke({})
assert calls == ["get_mechanism"]
```

Use `invoke(..., track_tool_calls=True)` as independent corroboration of your
own ledger; two sources disagreeing is a signal worth investigating.

### Emit ONE fenced Python block per turn, and say so in the prompt

CUGA executes only the **first** fenced block in a model response and silently
discards the rest (`extract_code_from_model_response` → single `code` →
`goto execute_node`). Observed live: the model emitted 8 blocks in one turn,
only the first ran, and it then concluded from the missing variables that *"the
tool execution did not return the required results"* and refused to finish.

The failure is self-reinforcing — the model blames the tools and gives up — so
state the contract explicitly in the system instructions:

```text
* Emit exactly ONE fenced Python block per turn. CUGA executes only the first
  block in a response and discards the rest.
* Put every call you want executed in that single block, then print results.
* Wait for execution output before deciding the next step. A missing variable
  means the call did not run; re-issue it rather than concluding the tools
  are unavailable.
```

This single addition turned a `no_tool_call` run into a 7-tool run that
completed its terminal submit call.

### Also demand code on the *first* turn

Even with the one-block rule, a multi-step task invites the model to narrate a
plan first. A run that opened with seven prose steps and no fence produced zero
tool calls and was routed straight to `FinalAnswerAgent`. Adding an explicit
first-turn directive fixed it:

```text
Start now: make your very next message a single fenced Python block that awaits
the evidence tools you need first. Narration without a fenced block executes
nothing, so do not describe a plan before running it.
```

Keep the verified *"Write and execute Python code that calls ..."* phrasing as
well — the two are complementary, and removing the former regressed a run.

### `cuga_folder=None` silently loads *other people's* skills

This is the most damaging silent failure found. Passing `enable_skills=True`
with `cuga_folder=None` and no `skills_folder` does not disable skills — CUGA
resolves its skill root to `<cwd>/.cuga/skills` and loads whatever any previous
run, or any other component of the repo, left there:

```text
Loaded 1 agent skill(s) from /path/to/repo/.cuga/skills   # a stale, unrelated skill
```

The agent received an unrelated `web-research` skill and **none of its own
four**, while the log line looked like success. Always bind an explicit
workspace, and set all three surfaces (they are read by different consumers):

```python
kwargs = {"cuga_folder": ws, "skills_folder": ws, "enable_skills": True}
os.environ["CUGA_FOLDER"] = ws     # sandbox + prepare_node read the env var
```

Verify by asserting the **count and the path** in the log, not merely that some
skill loaded: `Loaded 4 agent skill(s) from <your temp ws>/skills`.

### A `#`-leading derived description yields `description: None`

Deriving a skill/policy description from the first line of a Markdown body is a
trap: the first line is usually `# Heading`, and an unquoted `#` in YAML starts
a comment. The frontmatter parses with `description: None`, and CUGA's loader
then rejects the skill for a missing description — file on disk, invisible to
the model.

Strip Markdown markers **and** emit a quoted scalar:

```python
first = line.strip().lstrip("#").strip()
frontmatter = f'---\nname: {name}\ndescription: "{safe(first)}"\n---\n'
```

Assert with `yaml.safe_load` that `description` is a non-empty string. A test
asserting the *unquoted* form (`'description: Use the catalog.' in text`) is
worse than no test: it locks in the shape that breaks on `#` and `:`.

### Skill descriptions are selection criteria, not titles

The model chooses a skill from its description alone. Passive titles
(`"Refining an existing artifact"`) leave it guessing; trigger-oriented
descriptions get invoked:

```text
"Use when blame points at an artifact the primary parent already owns."
```

Verified: with trigger-oriented descriptions the model called
`load_skill("refine-artifact")` unprompted, 8 times in one run.

### A capability absent from the prompt is a capability the agent will not use

Structural availability is not enough. Two live runs offered a *better* donor
candidate whose artifact already contained the missing logic, with working
`list_parents` / `read_parent_artifact` tools — and the agent never called
either, because nothing in the prompt said donors existed. It refined the
primary from scratch instead.

Adding one line of inventory to the prompt changed the behavior immediately:

```text
PARENTS: 1 donor parent(s) available: donor (scores {'task-token': 1.0}).
Inspect a donor's artifact before deciding to refine.
```

Generalization: when measuring whether an agent *can* do something, first prove
the option is stated in the rendered prompt. Otherwise a negative result
measures your prompt, not the model.

### Track per-tool reachability, not just success

An agent can return a valid result while never touching most of its toolset.
Measured across two scenarios: 11 of 16 tools reached; 5 never invoked. Report
which tools were never reached — that list is the honest scope of what a live
verification actually covered.

### Sandbox-authored text arrives with the code's indentation

An agent that writes file/artifact content from inside the Python sandbox writes
it as a string literal, and an indented literal carries its indentation into the
value. Observed: a Markdown skill body whose every line after the first began
with four spaces, because the model wrote it inside an indented block.

This is silent and consequential for Markdown, where uniformly indented lines
are a code block, so an instruction document degrades into a literal listing
that the consuming agent cannot follow.

Normalize at the single choke point where authored text enters your system, and
use `inspect.cleandoc`, **not** `textwrap.dedent`:

```python
import inspect

def normalize_authored_content(content: str) -> str:
    # dedent computes the common prefix across ALL lines. A triple-quoted
    # literal's first line is flush, so the common prefix is "" and dedent is
    # a no-op on exactly this shape. cleandoc ignores the first line when
    # computing the margin -- the docstring convention that caused the defect.
    return inspect.cleandoc(content) if content else content
```

`cleandoc` preserves *relative* indentation, so nested list items and fenced
code blocks inside the authored text keep their structure. Test that explicitly:
a normalizer that flattens everything trades one corruption for another.

### The same prompt is not the same run

Re-running an identical scenario against the same model after a no-op refactor
changed observed behavior: the agent consulted its history tool in one run and
skipped it in the next. Nothing in the input differed.

Consequence for verification: a single live run demonstrates a capability is
*reachable*, never that it is *reliable*. Report n, and do not upgrade "it
worked once" into "it works". Conversely, do not treat one negative run as proof
a path is broken — check reachability across varied prompts before concluding.

### Reaching the agent and improving the agent are different claims

It is worth verifying end-to-end that a generated artifact survives every stage
of the pipeline: authorization, adapter application, inventory, harness config,
and on-disk materialization into a loadable file with valid frontmatter. Each
stage can silently drop it (see the `description: None` entry above).

But note precisely what that proves: the artifact *reaches* the agent. Whether
it makes the agent better is a separate measurement requiring a rerun and a
score. Keep the two claims apart in reporting.
