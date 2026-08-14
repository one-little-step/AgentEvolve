# CUGA SDK Integration Learnings

This note records findings from integrating the installed CUGA SDK in this
repository. It separates verified behavior from hypotheses. Recheck source
paths and behavior when changing CUGA versions.

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
- If both are needed, collect the trajectory from a single execution and
  extract tool calls from the streamed CUGA state, or use an SDK-supported
  callback/tracker. Do not execute the task twice.

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
- Record the CUGA version and source paths used to validate behavior.
- Treat an empty `result.tool_calls` as "no tracked calls surfaced", not proof
  that no tool code executed; inspect sandbox events before concluding.

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
- A playbook needs `name` and at least one trigger; `triggers: {always: true}`
  is the simplest always-on trigger.
- `settings.policy.enabled` defaults `True`; `auto_load_policies` and
  `filesystem_sync` also default `True`.

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
