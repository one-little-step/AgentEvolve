go through the cuga sdk docs .... and tell me how to get the complete trace (for our causal and counterfactual trajector analysis)...


Yes. I went through the **current CUGA SDK docs, settings reference, and repository documentation**, specifically looking at what you can obtain for **trace reconstruction, state capture, replay, and trajectory analysis**.

The key conclusion is:

> **CUGA's `track_tool_calls=True` is NOT the complete trace you need.**
> For your causal/counterfactual research, you should combine **CUGA's tracker/trajectory recording + Langfuse/OpenLit tracing + the underlying LangGraph state/checkpoint machinery**, and then add your own **state/environment recorder**.

CUGA explicitly exposes `agent.graph`, the underlying compiled LangGraph, which is the important escape hatch. ([CUGA Documentation][1])

---

# 1. First: what CUGA gives you out of the box

There are actually **four different levels** of observability in CUGA.

```text
                         CUGA
                          │
       ┌──────────────────┼──────────────────┐
       │                  │                  │
       ▼                  ▼                  ▼
  Tool tracking       Tracker          LLM tracing
       │                  │                  │
 track_tool_calls     tracker_enabled   Langfuse/OpenLit
       │                  │                  │
       └──────────────────┼──────────────────┘
                          ▼
                    Trajectory
                          │
                          ▼
                    CUGA Visualizer
```

And separately:

```text
                 agent.graph
                     │
                     ▼
                LangGraph
                     │
              state/checkpoints
```

That last part is the one you need for **counterfactual replay**.

---

# 2. `track_tool_calls=True` — useful, but NOT enough

CUGA's SDK gives you:

```python
result = await agent.invoke(
    task,
    track_tool_calls=True
)
```

The returned `InvokeResult.tool_calls` contains, for each tool invocation:

```text
name
operation_id
arguments
result
app_name
duration_ms
timestamp
error
```

CUGA explicitly documents these fields. ([CUGA Documentation][1])

So you get something like:

```json
{
  "name": "search_customer",
  "operation_id": "search_customer",
  "arguments": {
    "name": "Alice"
  },
  "result": "...",
  "app_name": "crm",
  "duration_ms": 423,
  "timestamp": "...",
  "error": null
}
```

### This is good for:

* tool trajectory
* tool arguments
* tool results
* latency
* failures
* auditing

### But it does NOT give you:

```text
❌ complete LLM calls
❌ complete prompts
❌ complete model outputs
❌ exact graph state
❌ every node transition
❌ checkpoint IDs
❌ state history
❌ environment snapshot
❌ replayable state
```

So **do not use `InvokeResult.tool_calls` as your research trace**.

---

# 3. `agent.stream()` gives you more

CUGA also exposes:

```python
async for state in agent.stream(task):
    print(state)
```

The documentation describes this as streaming the agent's execution **step-by-step**. ([CUGA Documentation][1])

This is useful because instead of:

```text
task → final answer
```

you can observe:

```text
task
 ↓
step
 ↓
step
 ↓
step
 ↓
...
 ↓
answer
```

But there is an important distinction:

> `agent.stream()` is an **execution stream**, not automatically a durable state ledger.

If you need every state to survive the run and be replayable later, **you should persist what you receive**.

---

# 4. CUGA has a dedicated tracker

This is more interesting.

The current CUGA settings reference exposes:

```toml
[advanced_features]
tracker_enabled = true
```

CUGA describes this as enabling:

> activity tracking and detailed execution logs

for debugging and analysis. ([CUGA Documentation][2])

So you should enable:

```toml
[advanced_features]
tracker_enabled = true
```

This is different from:

```python
track_tool_calls=True
```

Think of it as:

```text
track_tool_calls
      │
      ▼
individual tool-call metadata

tracker_enabled
      │
      ▼
broader CUGA execution tracking
```

---

# 5. CUGA also has trajectory persistence/visualization

This is important.

CUGA documents:

```bash
cuga viz
```

which launches a trajectory visualization server.

The docs say you can see:

* a list of trajectories
* individual trajectories
* execution details
* decision-making
* tool usage

after running CUGA. ([CUGA Documentation][3])

So CUGA clearly maintains an internal notion of:

```text
Trajectory
    │
    ├── task
    ├── execution
    ├── decisions
    └── tool usage
```

That's useful for us.

But again:

> **Visualization does not mean that the stored trajectory is a complete replayable state history.**

That's the distinction I'd maintain.

---

# 6. CUGA's Langfuse integration is much closer to "complete tracing"

The current settings expose:

```toml
[advanced_features]
langfuse_tracing = true
```

CUGA describes this as:

> comprehensive execution tracing and monitoring

and requires a Langfuse account. ([CUGA Documentation][2])

So you can have:

```text
                    CUGA
                     │
             Langfuse tracing
                     │
                     ▼
                  Langfuse
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
    LLM calls      tools        execution
```

This is much better for your **LLM-level causal analysis** than only tracking tools.

---

# 7. There is also OpenLit/OpenTelemetry

CUGA's current settings have:

```toml
[observability]
openlit = true
```

with:

```text
OTEL_EXPORTER_OTLP_ENDPOINT
```

CUGA describes this as:

> OpenLit LLM observability via OpenTelemetry

and provides a local OTel Collector + Tempo + Prometheus + Grafana stack. ([CUGA Documentation][2])

So another option is:

```text
CUGA
 │
 OpenTelemetry
 │
 ▼
OTel Collector
 │
 ├── Tempo
 ├── Prometheus
 └── Grafana
```

---

# 8. But Langfuse/OpenLit still aren't the thing we ultimately need

This is a subtle but **very important** point.

A trace like:

```text
LLM call
 ↓
tool call
 ↓
LLM call
 ↓
tool call
 ↓
answer
```

is an **event trace**.

Your research needs:

```text
               STATE
                 │
          ┌──────┴──────┐
          │             │
       action A       action B
          │             │
          ▼             ▼
       state A'       state B'
```

That requires **state snapshots/checkpoints**, not just observability spans.

So I'd use:

### Langfuse/OpenTelemetry

for:

```text
"What happened?"
```

and

### LangGraph checkpoints

for:

```text
"What was the agent's exact state at that point?"
```

---

# 9. The key CUGA API: `agent.graph`

This is the most important thing CUGA gives you.

The SDK explicitly exposes:

```python
agent.graph
```

and documents it as:

> Access the underlying compiled LangGraph `StateGraph`.

You can even invoke it directly:

```python
compiled_graph = agent.graph

await compiled_graph.ainvoke(...)
```

([CUGA Documentation][1])

So your architecture should **not stop at `CugaAgent`**.

You should go:

```text
CugaAgent
    │
    └── graph
          │
          ▼
      LangGraph
          │
          ▼
    state/checkpoints
```

---

# 10. Why this matters for replay

LangGraph's persistence system provides checkpoint-based state.

The important conceptual objects are:

```text
Thread
  │
  ├── checkpoint 0
  ├── checkpoint 1
  ├── checkpoint 2
  ├── checkpoint 3
  └── checkpoint 4
```

A checkpoint corresponds to a state snapshot in the graph execution.

LangGraph exposes APIs for retrieving historical state/checkpoints and replaying/branching execution from previous checkpoints. The underlying LangGraph implementation is therefore much closer to what you need than CUGA's high-level `InvokeResult`. ([GitHub][4])

---

# 11. So your research trace should look like this

I would create **one canonical trace object** in your own harness.

For example:

```text
RUN
│
├── run_id
├── thread_id
├── parent_run_id
├── task
│
├── configuration
│   ├── cuga_version
│   ├── model
│   ├── model_version
│   ├── system_prompt_version
│   ├── skill_version
│   ├── memory_version
│   ├── policy_version
│   └── tool_registry_version
│
├── checkpoints
│   │
│   ├── CP0
│   │   ├── graph_state
│   │   ├── node
│   │   └── metadata
│   │
│   ├── CP1
│   │
│   ├── CP2
│   │
│   └── CP3
│
├── events
│   ├── LLM_CALL
│   ├── TOOL_CALL
│   ├── TOOL_RESULT
│   ├── NODE_START
│   ├── NODE_END
│   ├── POLICY
│   └── ERROR
│
└── final_result
```

That becomes your **causal trace**.

---

# 12. The state snapshot is the most important part

At every meaningful decision point, you want:

```json
{
  "checkpoint_id": "cp_17",
  "node": "api_planner",

  "state": {
    "messages": [...],
    "plan": [...],
    "variables": {...},
    "tool_results": {...},
    "pending_actions": [...]
  }
}
```

And then:

```json
{
  "configuration": {
    "model": "...",
    "skill_version": "v12",
    "memory_version": "v8",
    "policy_version": "v4",
    "tools_version": "v19"
  }
}
```

This lets you say:

> At checkpoint 17, the agent was in **exactly this state**, with **exactly these skills, memories, policies and tools**.

That's what you need for scientific counterfactual analysis.

---

# 13. Then your counterfactual engine can branch

Suppose factual execution was:

```text
CP17
 │
 ▼
Planner chooses:
search_web()
 │
 ▼
CP18
 │
 ▼
answer
```

You preserve CP17.

Then your counterfactual engine does:

```text
                   CP17
                     │
            ┌────────┴─────────┐
            │                  │
        factual             intervention
            │                  │
     search_web()       search_filings()
            │                  │
            ▼                  ▼
         CP18-A              CP18-B
            │                  │
            ▼                  ▼
         outcome A           outcome B
```

Now you can calculate:

```text
effect =
    outcome(counterfactual)
    -
    outcome(factual)
```

---

# 14. You can also intervene on memory

This is where your self-evolution research gets really interesting.

Suppose:

```text
CP17
memory_version = M4
skill_version = S7
policy_version = P3
```

Your evolutionary engine proposes:

```text
M5
```

Then:

```text
                    CP17
                      │
             ┌────────┴─────────┐
             │                  │
            M4                 M5
             │                  │
             ▼                  ▼
         trajectory A       trajectory B
             │                  │
             ▼                  ▼
          score .72           score .84
```

Now you have evidence for the effect of the memory intervention.

---

# 15. Same for skills

```text
                   CP17
                     │
              ┌──────┴───────┐
              │              │
           Skill S7        Skill S8
              │              │
              ▼              ▼
          trajectory A    trajectory B
              │              │
              ▼              ▼
            .72            .88
```

This is exactly the kind of experiment I'd want your harness to automate.

---

# 16. But there's a huge missing piece: **external environment**

Even if you have perfect LangGraph checkpoints, replay isn't automatically deterministic.

Consider:

```text
Agent
 │
 ├── LLM
 ├── Web search
 ├── browser
 ├── MCP server
 ├── database
 ├── filesystem
 ├── current time
 └── random state
```

If the agent made:

```text
search_web("IBM CUGA")
```

at 10:00 AM and you replay it tomorrow, the web response may be different.

Therefore, **your trace must capture tool/environment observations too**.

---

# 17. I would record every tool boundary

For every tool:

```text
TOOL_CALL
│
├── tool_name
├── tool_version
├── arguments
├── timestamp
├── state_before
│
└── TOOL_RESULT
    ├── return_value
    ├── error
    ├── duration
    └── state_after
```

Then during counterfactual replay you can choose:

### Real environment

```text
call actual tool
```

or

### Recorded environment

```text
return recorded result
```

That gives you two modes:

```text
REPLAY
 └── recorded observations

COUNTERFACTUAL
 ├── replay unchanged parts
 └── execute intervention
```

---

# 18. Your trace architecture should therefore have 3 stores

I would use:

```text
┌──────────────────────────────┐
│        TRACE STORE           │
│                              │
│ LLM calls                    │
│ tool calls                   │
│ events                       │
│ timing                       │
│ metadata                     │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│       STATE STORE            │
│                              │
│ LangGraph checkpoints        │
│ graph state                  │
│ state history                │
│ branch relationships         │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│    ENVIRONMENT STORE         │
│                              │
│ tool inputs                  │
│ tool outputs                 │
│ HTTP responses               │
│ browser observations         │
│ filesystem effects           │
│ DB observations              │
└──────────────────────────────┘
```

Then your causal engine consumes all three.

---

# 19. Where each CUGA feature fits

Here's the mapping I'd use:

| CUGA facility           | Use in your harness                          |
| ----------------------- | -------------------------------------------- |
| `track_tool_calls=True` | Tool-level event capture                     |
| `agent.stream()`        | Runtime event/state observation              |
| `tracker_enabled=true`  | CUGA execution tracking                      |
| `cuga viz`              | Human inspection/debugging                   |
| `langfuse_tracing=true` | LLM/execution observability                  |
| `openlit=true`          | OpenTelemetry LLM traces                     |
| `agent.graph`           | **Exact graph-level access**                 |
| LangGraph checkpoints   | **State snapshots / replay**                 |
| CUGA `thread_id`        | Execution lineage                            |
| CUGA skills             | Versioned procedural interventions           |
| CUGA policies           | Versioned behavioral interventions           |
| CUGA knowledge          | Versioned knowledge interventions            |
| ToolProvider            | Instrumented/replayable tools                |
| Evolve                  | Existing trajectory-based learning, optional |

CUGA's current docs explicitly expose all of the first-level pieces above. ([CUGA Documentation][1])

---

# 20. And CUGA already has an Evolve integration

This is worth mentioning because it overlaps with what you're building.

Current CUGA supports an optional Evolve integration:

```toml
[evolve]
enabled = true
save_on_success = true
save_on_failure = true
async_save = true
```

The documented flow is:

```text
CUGA
 ↓
task
 ↓
CugaLite
 ↓
Evolve
 ↓
retrieve guidance
 ↓
execute
 ↓
save trajectory
```

CUGA says Evolve can save completed trajectories and retrieve task-specific guidance. ([GitHub][5])

**But don't confuse this with your causal engine.**

Evolve is basically:

```text
experience → guidance → reuse
```

Your research system needs:

```text
experience
   ↓
state reconstruction
   ↓
intervention
   ↓
counterfactual branch
   ↓
causal effect
   ↓
evolution
```

Those are substantially different.

---

# 21. The exact implementation direction I'd take

I would build a `CugaTraceAdapter` around CUGA:

```python
class CugaTraceAdapter:

    async def run(self, task):
        ...

    async def capture_event(self, event):
        ...

    async def capture_state(self, checkpoint):
        ...

    async def get_state(self, checkpoint_id):
        ...

    async def get_history(self, thread_id):
        ...

    async def replay(self, checkpoint_id):
        ...

    async def branch(self, checkpoint_id, intervention):
        ...
```

Internally:

```text
CugaTraceAdapter
       │
       ├── CugaAgent.invoke()
       ├── CugaAgent.stream()
       ├── CugaAgent.graph
       ├── CUGA tracker
       ├── Langfuse/OpenTelemetry
       └── LangGraph checkpoint APIs
```

---

# 22. One thing I would NOT do

Don't make your causal engine depend on the **CUGA Visualizer's stored trajectory format**.

The visualizer is excellent for:

```text
human debugging
```

but your research needs a canonical machine-readable format.

Create your own:

```text
CausalTrace v1
```

schema.

For example:

```text
CausalTrace
│
├── Run
├── StateSnapshot
├── NodeEvent
├── LLMEvent
├── ToolEvent
├── ObservationEvent
├── Intervention
├── Checkpoint
└── Outcome
```

Then CUGA is just one runtime that produces that schema.

This becomes extremely valuable later if you compare:

```text
CUGA
vs
Pi
vs
OpenCode
vs
your own agent
```

because all four can feed the **same causal-analysis engine**.

---

# 23. So, can CUGA give us the "complete trace"?

### **Not from one CUGA SDK call.**

This:

```python
result = await agent.invoke(
    task,
    track_tool_calls=True
)
```

is **not enough**. ([CUGA Documentation][1])

You need to combine:

```text
             CUGA
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
    tracker  Langfuse  stream
       │       │        │
       └───────┼────────┘
               ▼
          event trace
               │
               +
        agent.graph
               │
               ▼
       LangGraph state/
        checkpoints
               │
               +
       environment/tool
         recordings
               │
               ▼
       YOUR CausalTrace
```

### For your particular research, I would define **"complete trace"** as:

> **Every agent decision + every LLM interaction + every tool invocation/result + every graph state checkpoint + every configuration/version + every external observation required to reproduce or branch the execution.**

CUGA gives us **most of the observation hooks**, and its `agent.graph` gives us the critical path into LangGraph state. But **the final causal/counterfactual trace layer should belong to your own harness**, not CUGA.

That is actually a good thing: it means your research contribution remains **runtime-independent**, while CUGA is one instrumented execution backend. ([CUGA Documentation][1])

[1]: https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com "CugaAgent | CUGA AGENT"
[2]: https://docs.cuga.dev/docs/customization/settings-reference/?utm_source=chatgpt.com "Settings Reference | CUGA AGENT"
[3]: https://docs.cuga.dev/docs/getting-started/installation/?utm_source=chatgpt.com "Installation Guide | CUGA AGENT"
[4]: https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py?utm_source=chatgpt.com "langgraph/libs/langgraph/langgraph/pregel/main.py at main · langchain-ai/langgraph · GitHub"
[5]: https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com "GitHub - cuga-project/cuga-agent: CUGA is an open-source generalist agent harness for the enterprise, supporting complex task execution on web and APIs, OpenAPI/MCP integrations, composable architecture, reasoning modes, and policy-aware features. · GitHub"
