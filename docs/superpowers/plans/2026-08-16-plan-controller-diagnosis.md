# PlanControllerAgent Absence — Diagnosis

Date: 2026-08-16
Branch: dev4
Status: **RESOLVED — root cause proven from installed source + on-disk traces. No live rollout required.**

## Verdict (one line)

`PlanControllerAgent` is absent because **the code that routes to it is never
loaded on the SDK path we use.** `cuga_lite_node.py` belongs to the *server*
graph (`backend/cuga_graph/graph.py`). `CugaAgent` (the SDK class we call)
builds a completely different, smaller wrapper graph that has no
`PlanControllerAgent` node at all and substitutes a node literally named
`SDKCallback`. Neither `_has_error` nor early termination is responsible.

## 1. The exact routing condition (and why it is unreachable)

File: `.venv/lib/python3.14/site-packages/cuga/backend/cuga_graph/nodes/cuga_lite/cuga_lite_node.py`

Two `goto="PlanControllerAgent"` sites exist, both inside
`CugaLiteNode._process_results`:

| Line | Branch | Condition |
|---|---|---|
| `cuga_lite_node.py:470` | `has_error = self._has_error(answer)` | `_has_error` at `:218-231` matches any of `['Error during execution:', 'Error:', 'Exception:', 'Traceback', 'Failed to']` in `answer` |
| `cuga_lite_node.py:498` | error + autonomous | `has_error and is_autonomous_subtask` → `goto="PlanControllerAgent"` |
| `cuga_lite_node.py:510` | error + not autonomous | `has_error and not is_autonomous_subtask` → `goto="FinalAnswerAgent"` |
| `cuga_lite_node.py:571` | **success + autonomous** | `not has_error and is_autonomous_subtask` → `goto="PlanControllerAgent"` |
| `cuga_lite_node.py:586` | success + not autonomous | `not has_error and not is_autonomous_subtask` → `goto="FinalAnswerAgent"` |

`is_autonomous_subtask` is recomputed at `cuga_lite_node.py:466` as
`settings.advanced_features.force_autonomous_mode or (state.sub_task not empty)`
— so with `force_autonomous_mode=True` it is unconditionally `True`, and
**both** the success path (`:571`) and the error path (`:498`) would route to
`PlanControllerAgent`. There is no branch under autonomous mode that avoids it.

**That is the whole point: under autonomous mode the condition is
unconditionally satisfied, so if this function ran at all we would see
`PlanControllerAgent` in 15/15 rollouts, not 0/15.** The function does not run.

### Proof that `_process_results` never runs on our path

- `CugaLiteNode` is imported by exactly one non-test file:
  `backend/cuga_graph/graph.py:46`. That file is the **server** graph, which
  registers `PlanControllerAgent` at `graph.py:164` and `CugaLiteCallback` at
  `graph.py:282`.
- We do not use that graph. `src/agent_evolve/cuga_wrapper/__init__.py:131,142`
  constructs `cuga.CugaAgent`, and `__init__.py:307` calls `agent.invoke(...)`.
- `CugaAgent` (`sdk.py:1659`) builds its graph via
  `sdk.py:2293 → _create_graph (sdk.py:2006) → _create_hitl_wrapper_graph (sdk.py:2014)`.
- That wrapper graph's full node set (`sdk.py:2145-2154`) is:
  `CugaLiteSubgraph`, `SDKCallback`, `SuggestHumanActions`, `WaitForResponse`,
  `FinalAnswerAgent`, plus three **dummy** stubs (`APIPlannerAgent`,
  `ChatAgent`, `CugaLite`) that all just `goto="SDKCallback"`
  (`sdk.py:2058-2071`).
- **`PlanControllerAgent` does not appear anywhere in `sdk.py`** (grep for
  `PlanController|PLAN_CONTROLLER` over `sdk.py` → zero hits).
- Edges are static and terminal: `START → CugaLiteSubgraph` (`sdk.py:2157`),
  `CugaLiteSubgraph → SDKCallback` (`sdk.py:2158`),
  `FinalAnswerAgent → END` (`sdk.py:2165`).
- `sdk_callback_node` (`sdk.py:2074-2134`) is a *reimplementation* of the
  callback with the autonomous branch removed. Its only non-HITL exit is
  `sdk.py:2130-2134`: set `state.sender = NodeNames.CUGA_LITE`, then
  unconditionally `goto=NodeNames.FINAL_ANSWER_AGENT`. **It never calls
  `_has_error` and never consults `is_autonomous_subtask`.**

So the routing edge at `cuga_lite_node.py:571` is not merely un-taken — the
module containing it is never even imported into the running graph.

## 2. Observed node inventory (empirical)

Scanned **104 trace dirs** containing `events.jsonl` under `data/traces/`,
including all named roots: `tiny5-baseline-a/` (5 rollouts),
`tiny5-baseline-b/` (5), `tiny5-baseline-c/` (5), `tiny5-autoprobe/` (1), and
reference `5d434903-bc26-4dc4-9229-8d886d2c6781/`.

Complete `actor_id` inventory across **every** trace on disk — six values, no others:

| count | actor |
|---:|---|
| 518 | `call_model` |
| 448 | `prepare` |
| 392 | `CugaLiteSubgraph` |
| 248 | `sandbox` |
| 196 | `SDKCallback` |
| 196 | `FinalAnswerAgent` |

`PlanControllerAgent` count: **0**. `CugaLiteCallback` count: **0**.
`APIPlannerAgent` / `ChatAgent` / `CugaLite` (the dummy stubs): **0** — the
dummies are never reached either.

Every single trace has the identical shape, terminal node always
`FinalAnswerAgent`:

```
CugaLiteSubgraph → prepare → call_model ⇄ sandbox (loop) → SDKCallback → FinalAnswerAgent
```

Inner subgraph confirmed at
`backend/cuga_graph/nodes/cuga_agent_core/graph/shared_graph.py:50-59`:
`StateGraph` with exactly `prepare`, `call_model`, `<execute>` (= `sandbox`),
`START → prepare`, `sandbox → call_model`. Built by
`cuga_lite_graph.py:208-214` (`create_cuga_lite_graph`), which is what
`sdk.py:2047` instantiates.

**Answer to "is it named something else?"** Functionally, `SDKCallback`
(`sdk.py:2146`) occupies the slot that `CugaLiteCallback` (`graph.py:282`) fills
in the server graph. But it is **not** `PlanControllerAgent` under another
label — it is a different, planning-free implementation. There is no renamed
planner in the trace.

## 3. Which branch wins, and the evidence

Neither of the two hypothesized branches. The winner is a **third** path that
was not in the hypothesis space: `sdk.py:2131-2134`, an unconditional
`goto=FinalAnswerAgent` in `sdk_callback_node`.

Disconfirming evidence for the two stated hypotheses:

- **`_has_error` fires every time — REFUTED.** Extracted `final_output` from
  `causal-trace.json` for all 16 rollouts in the four tiny5 roots and applied
  CUGA's exact `error_indicators` list from `cuga_lite_node.py:229`.
  Result: **0 of 16** contain any indicator. All 16 are clean substantive
  answers (e.g. `tiny5-baseline-a/16deaea0…` → "The highest number of bird
  species on camera simultaneously is 3."). Even if `_has_error` *had* fired,
  line `:498` routes to `PlanControllerAgent` anyway under autonomous mode, so
  this hypothesis could not explain the absence in either polarity.
- **CugaLite terminates before the routing edge — REFUTED.** All 104 traces
  reach `SDKCallback` and then `FinalAnswerAgent`; `FinalAnswerAgent`
  start/end pairs are present in every trace (196 events = 98 start/end pairs
  across the traces that carry actors). The callback slot executes to
  completion every time. It simply is not the callback that contains the
  routing block.
- **Named something else — PARTIALLY TRUE but not the cause.** `SDKCallback`
  is the structural analogue, but it lacks the branch entirely.

`force_autonomous_mode` remains correctly set
(`DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE` present in `.env`; CUGA
default is `false` at `settings.toml:66`; our wrapper hard-fails without it at
`cuga_wrapper/__init__.py:90-98`). It is consumed on the SDK path — just not
for routing. Live consumers are:
- `prepare_node.py:630` — feeds `is_autonomous_subtask` into the prompt.
- `sandbox_node.py:210` — passed as sandbox context.
- `cuga_lite_node.py:269,466` — dead code on this path.

## 4. Does this matter for tomorrow's self-improvement delta?

**Largely a non-issue for the delta, with one real caveat.**

**Why it mostly does not matter:**

`PlanControllerAgent` in the server graph is a *multi-subtask decomposition
loop* — it exists to sequence `sub_task`s and re-dispatch CugaLite per subtask.
Our benchmark issues one task per rollout via `agent.invoke(message)` with no
`sub_task` decomposition, so there is nothing for a plan controller to
sequence. Routing to it would return control to a planner with an empty plan.

**The agent does get a planning pass**, just not a graph-node one. Planning is
in-context inside `call_model`:
- The autonomous-mode prompt branch is live and fires — `mcp_prompt.jinja2`
  has 5 `is_autonomous_subtask` conditionals, fed by `prepare_node.py:630-631`.
  These inject the "work independently, do not return until complete, make all
  decisions autonomously" instructions. This is the behavior we actually want
  from autonomous mode, and **we are getting it.**
- Iterative replanning is empirically present: the `call_model ⇄ sandbox` loop
  runs up to 12 `call_model` turns (`tiny5-baseline-c/aa6df264…`: 24
  `call_model` events, 20 `sandbox`), and rollout text shows explicit mid-run
  replanning — `tiny5-baseline-b/b85009a0…`: "The prior result used an
  incorrect time (2:01:09). I'll verify…"; `tiny5-autoprobe/e0526da0…`: "The
  failure was only a missing import. I'm re-running…". That is a planning pass
  in every meaningful sense for our purposes.

**Does the absence limit what our editor can improve? Mostly no.** Our editable
artifacts (instructions, skills, policies/playbooks, knowledge) all land on the
live path: `special_instructions` → `prepare_node` → prompt; skills → prompt
section; policies → `apply_output_formatter_policies` (`sdk.py:2124`) and
prepare-time playbook loading. The optimization surface is intact and the
`call_model`/`sandbox`/`prepare` nodes give the blame graph plenty of
attributable structure.

**The one real caveat (scope your claims, do not chase it):** `enable_todos`
and `reflection_enabled` are both **`false`** by default
(`settings.toml:34-35`) and nothing in our config overrides them. So the
explicit todo-list planning artifact (`create_update_todos`,
`cuga_agent_core/execution/todos.py`) is off, and it never appears in the
observed tool inventory (only `web_search` 64, `web_fetch` 31,
`knowledge_search_knowledge` 16, `calculator` 7, `wikipedia_search` 5, plus 3
alpha/beta chain tools). This is a *separate* knob from `PlanControllerAgent`
and is the only genuine planning capability we are leaving on the table.

**Do not claim in the writeup that the agent runs a hierarchical
planner/executor decomposition.** It runs a single-agent ReAct-style
`prepare → call_model ⇄ sandbox` loop under autonomous-mode prompting. The
numbers are valid; the architectural description must match the trace.

## 5. Fix options (NOT implemented — reporting only)

**Recommended for tomorrow: do nothing.** The absence is by SDK design, does
not degrade the delta, and any change perturbs a baseline we already paid 15+
rollouts for. Cost of change now exceeds the benefit.

If a planning capability is wanted *after* the deadline, ranked by risk:

1. **`enable_todos = true` (cheap, low risk, ~15 min).** Set
   `DYNACONF_ADVANCED_FEATURES__ENABLE_TODOS=true`. Read at runtime from
   `config["configurable"]` with fallback to `settings.advanced_features`
   (`cuga_lite_graph.py:173`), so no code change. Adds an explicit plan
   artifact to the prompt and a `create_update_todos` tool.
   *Risk:* changes prompt content → **invalidates the current baseline**; every
   number must be re-collected. Also adds tool-call turns (cost). Do not do
   this before tomorrow.
2. **Switch to the server graph (`backend/cuga_graph/graph.py`) to get a real
   `PlanControllerAgent`.** High risk: different entry point, different state
   contract, requires `sub_task` decomposition upstream, and would need our
   tracing adapter re-verified against a graph we have zero trace evidence for.
   Multi-day, not a fix.
3. **`reflection_enabled = true`.** Same invalidation risk as (1), less
   obviously useful, unmeasured on this build.

No behavior change has been made. No commit.

## Method note

All findings derived programmatically from the installed package and on-disk
traces; no new rollouts were purchased. The worker-stderr instrument
(`CugaProcessPool(log_capture=...)`, `src/agent_evolve/core/run_logging.py`)
was not needed — the static import graph plus the 104-trace actor inventory
were jointly conclusive, since `sdk.py` provably contains no
`PlanControllerAgent` node for a `Routing to:` line to ever name.
