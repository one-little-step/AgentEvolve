now , go though cuga's SDK , .... write me how to get the causal comaple execution graph in cuga sdk .... and how this graph looks loke...

Yes. I went through the **current CUGA SDK/repository implementation**, not just the high-level docs. There is an important correction to what I said earlier:

> **`CugaAgent.graph` is the compiled LangGraph, and the actual CUGA execution graph is much richer than a simple Planner → Executor → Tools loop.**

The current implementation builds a `StateGraph(AgentState)`, adds a large set of CUGA nodes, adds subgraphs for CugaLite and optionally CugaSupervisor, and finally compiles it with a `MemorySaver` checkpoint and interrupts. 

---

# 1. The exact way to get CUGA's compiled execution graph

The public SDK exposes:

```python
agent.graph
```

The SDK documentation explicitly says that this property gives you the **underlying compiled LangGraph**. ([CUGA Documentation][1])

So:

```python
from cuga import CugaAgent

agent = CugaAgent(
    tools=MY_TOOLS,
)

graph = agent.graph
```

However, there is one important detail:

### `agent.graph` is the compiled graph

while:

```python
graph.get_graph()
```

returns LangGraph's **drawable graph representation**.

LangGraph's `CompiledStateGraph.get_graph()` is specifically designed to return a drawable representation of the computation graph. ([GitHub][2])

Therefore:

```python
compiled_graph = agent.graph

drawable_graph = compiled_graph.get_graph()

print(drawable_graph.draw_mermaid())
```

is the basic thing you want.

---

# 2. The most useful inspection code

I would use this:

```python
from cuga import CugaAgent

agent = CugaAgent(
    tools=MY_TOOLS,
)

compiled_graph = agent.graph

print("=== COMPILED GRAPH ===")
print(compiled_graph)

print("\n=== NODES ===")
for name, node in compiled_graph.nodes.items():
    print(name)

print("\n=== MERMAID ===")
print(
    compiled_graph
        .get_graph()
        .draw_mermaid()
)
```

And if you want to inspect the **subgraphs too**:

```python
print(
    compiled_graph
        .get_graph(xray=True)
        .draw_mermaid()
)
```

`xray=True` is important because LangGraph's `get_graph()` supports recursively exposing subgraphs. ([GitHub][2])

So you essentially have:

```text
agent.graph
     │
     ▼
CompiledStateGraph
     │
     ├── .nodes
     │
     ├── .get_graph()
     │
     │       └── top-level graph
     │
     └── .get_graph(xray=True)
             │
             └── top-level + nested subgraphs
```

---

# 3. But there's an important CUGA-specific issue

If you simply do:

```python
agent = CugaAgent(...)
print(agent.graph.get_graph().draw_mermaid())
```

you may **not see the complete CUGA architecture** that you are expecting.

Why?

Because CUGA's `DynamicAgentGraph` builds the graph asynchronously.

The current implementation does:

```python
graph = StateGraph(AgentState)

await self.add_nodes(graph)

self.add_edges(graph)

self.graph = graph.compile(...)
```

Specifically, it compiles with:

```python
self.graph = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_after=[
        self.action_agent.action_agent.name,
        self.interrupt_tool_node.name,
    ],
)
```



And `CugaAgent.graph` exposes that resulting compiled graph. ([CUGA Documentation][1])

So the graph is **constructed inside CUGA**, not something you manually assemble in your application.

---

# 4. What nodes does CUGA actually put into this graph?

This is where things get interesting.

The current `DynamicAgentGraph` constructs these major nodes:

```text
Chat
Task Decomposition
Plan Controller
Final Answer
Browser Planner
Action Agent
QA Agent
Task Analyzer
Interrupt Tool
API Code Planner
API Shortlister
API Coder
API Planner
CugaLite
CugaSupervisor
Save/Reuse
Human Follow-up
```

These aren't guesses—the current source explicitly instantiates these node classes. 

The relevant portion is essentially:

```text
TaskDecompositionNode
PlanControllerNode
FinalAnswerNode
PlannerNode
SuggestHumanActions
WaitForResponse
SaveReuseNode
ChatNode
QaNode
InterruptToolNode
TaskAnalyzer
ActionNode
ApiCodePlanner
ApiPlanner
ApiShortlister
ApiCoder
CugaLiteNode
CugaSupervisorNode
```



So the graph is **far more complicated than:**

```text
Planner → Executor → Tool
```

---

# 5. The top-level graph looks roughly like this

There is a subtlety here: CUGA uses **LangGraph `Command`-based routing inside nodes**, so not every logical transition appears as a simple `graph.add_edge()` in `add_edges()`.

But structurally, the current graph is approximately:

```text
                         ┌─────────────┐
                         │    START    │
                         └──────┬──────┘
                                │
                                ▼
                         ┌─────────────┐
                         │    CHAT     │
                         └──────┬──────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  TASK ANALYZER   │
                       └────────┬─────────┘
                                │
                         route by task type
                                │
              ┌─────────────────┼──────────────────┐
              │                 │                  │
              ▼                 ▼                  ▼
        API execution      Browser/Web        CugaLite
              │                 │                  │
              ▼                 ▼                  ▼
          API Planner       Browser Planner    CugaLite
              │                 │                  │
              ▼                 ▼                  ▼
        API execution        Action Agent      Callback
              │                 │
              │                 ▼
              │                QA
              │                 │
              │                 ▼
              │             Browser Planner
              │
              └────────────────┬─────────────────┐
                               │                 │
                               ▼                 ▼
                         Plan Controller     Supervisor
                               │                 │
                               └────────┬────────┘
                                        ▼
                                  Final Answer
                                        │
                                        ▼
                                       END
```

**But this is a conceptual rendering**, because the exact routing decisions are implemented inside the node logic/`Command` transitions rather than all being represented by explicit `add_edge()` statements.

---

# 6. Let's look at what CUGA explicitly wires

The source explicitly adds these edges:

```python
graph.add_edge(
    START,
    self.chat.chat_agent.name
)
```

So:

```text
START
  │
  ▼
CHAT
```

Then:

```python
graph.add_edge(
    task_decomposition,
    plan_controller
)
```

so:

```text
Task Decomposition
        │
        ▼
Plan Controller
```

And:

```python
graph.add_edge(
    interrupt_tool,
    plan_controller
)
```

so:

```text
Interrupt Tool
      │
      ▼
Plan Controller
```

Also:

```python
graph.add_edge(
    qa,
    planner
)
```

giving:

```text
QA
 │
 ▼
Browser Planner
```

And:

```python
graph.add_edge(
    action_agent,
    planner
)
```

giving:

```text
Action Agent
     │
     ▼
Browser Planner
```

Finally:

```python
graph.add_edge(
    final_answer,
    END
)
```

giving:

```text
Final Answer
     │
     ▼
    END
```

All of these are directly visible in CUGA's current `add_edges()` implementation. 

---

# 7. The really interesting part: CugaLite is a subgraph

This is **very important**.

CUGA does not just have a node called "CugaLite".

It actually constructs a **separate LangGraph subgraph**:

```python
cuga_lite_subgraph = create_cuga_lite_graph(...)
```

then:

```python
compiled_cuga_lite_subgraph = cuga_lite_subgraph.compile()
```

and then inserts that compiled graph into the main graph:

```python
graph.add_node(
    "CugaLiteSubgraph",
    compiled_cuga_lite_subgraph
)
```



So:

```text
                 CUGA MAIN GRAPH
                       │
                       ▼
                CugaLiteSubgraph
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
          internal             internal
          node A               node B
             │                   │
             └─────────┬─────────┘
                       ▼
                 CugaLiteCallback
```

This means **`get_graph()` vs `get_graph(xray=True)` matters a lot**.

---

# 8. Without `xray=True`

You are essentially looking at:

```text
CUGA
 │
 ├── Chat
 ├── TaskAnalyzer
 ├── CugaLiteSubgraph
 ├── ...
 └── FinalAnswer
```

The subgraph may appear as a single node.

---

# 9. With `xray=True`

You ask LangGraph:

> "Open up the subgraphs too."

So:

```python
graph.get_graph(xray=True)
```

can expose:

```text
CUGA
 │
 ├── Chat
 │
 ├── TaskAnalyzer
 │
 ├── CugaLiteSubgraph
 │      │
 │      ├── internal node
 │      ├── internal node
 │      ├── tool node
 │      └── ...
 │
 ├── ...
 │
 └── FinalAnswer
```

LangGraph's current implementation explicitly gathers subgraphs recursively when `xray=True`. ([GitHub][2])

---

# 10. And CugaSupervisor is ALSO a subgraph

This is even more relevant given your previous question about sub-agents.

When Supervisor is enabled, CUGA does:

```python
supervisor_subgraph = create_cuga_supervisor_graph(
    supervisor_model=supervisor_model,
    agents=agents,
    ...
)
```

then:

```python
compiled_supervisor_subgraph = supervisor_subgraph.compile()
```

and:

```python
graph.add_node(
    "CugaSupervisorSubgraph",
    compiled_supervisor_subgraph
)
```



So the architecture becomes:

```text
                       CUGA
                        │
                        ▼
                CugaSupervisorSubgraph
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
          Agent A    Agent B    Agent C
```

That is a **nested LangGraph**.

This is why I strongly recommend using:

```python
agent.graph.get_graph(xray=True)
```

when you're studying CUGA's architecture.

---

# 11. So the graph hierarchy is actually

Think of it as:

```text
                 ┌─────────────────────────┐
                 │      CUGA GRAPH         │
                 │                         │
                 │  StateGraph(AgentState) │
                 │                         │
                 │  Chat                   │
                 │  Task Analyzer          │
                 │  Planning               │
                 │  API                    │
                 │  Browser                │
                 │  Final Answer           │
                 │                         │
                 │  ┌───────────────────┐  │
                 │  │ CugaLiteSubgraph  │  │
                 │  │                   │  │
                 │  │ internal graph    │  │
                 │  └───────────────────┘  │
                 │                         │
                 │  ┌───────────────────┐  │
                 │  │ Supervisor        │  │
                 │  │ Subgraph          │  │
                 │  │                   │  │
                 │  │ Agent A           │  │
                 │  │ Agent B           │  │
                 │  │ Agent C           │  │
                 │  └───────────────────┘  │
                 └─────────────────────────┘
```

This is much closer to the **real architecture**.

---

# 12. How I would inspect it in your experiment

Since you're interested in the **execution graph itself**, I'd write a little inspection script.

```python
from cuga import CugaAgent

agent = CugaAgent(
    tools=MY_TOOLS,
)

g = agent.graph

print("TYPE:")
print(type(g))

print("\nTOP LEVEL NODES:")
for name in g.nodes:
    print(" -", name)

print("\nMERMAID:")
print(g.get_graph().draw_mermaid())

print("\nFULL GRAPH WITH SUBGRAPHS:")
print(
    g.get_graph(xray=True).draw_mermaid()
)
```

You can save it:

```python
with open("cuga_graph.mmd", "w") as f:
    f.write(
        agent.graph
            .get_graph(xray=True)
            .draw_mermaid()
    )
```

Then open the `.mmd` in a Mermaid viewer/editor.

---

# 13. You can also get the graph as PNG

LangGraph's drawable graph supports rendering methods such as `draw_mermaid_png()`. ([GitHub][2])

So:

```python
png = (
    agent.graph
    .get_graph(xray=True)
    .draw_mermaid_png()
)

with open("cuga_graph.png", "wb") as f:
    f.write(png)
```

That should give you a visual graph.

---

# 14. Even better: inspect nodes programmatically

For your self-evolution work, I'd actually do:

```python
g = agent.graph

for name, node in g.nodes.items():
    print("=" * 80)
    print("NODE:", name)
    print("TYPE:", type(node))
    print(node)
```

This lets you distinguish:

```text
Chat
TaskAnalyzer
TaskDecomposition
PlanController
ApiPlanner
ApiCoder
BrowserPlanner
ActionAgent
QA
CugaLiteSubgraph
CugaSupervisorSubgraph
FinalAnswer
```

rather than treating CUGA as one opaque agent.

---

# 15. There's another VERY important thing: `AgentState`

The graph is:

```python
StateGraph(AgentState)
```

not merely:

```python
StateGraph(MessagesState)
```

CUGA has its own `AgentState` implementation. 

This means the graph is effectively:

```text
                AgentState
                    │
        ┌───────────┼────────────┐
        │           │            │
     messages     task         plan
        │           │            │
        │           │            │
        ▼           ▼            ▼
      Node A      Node B       Node C
        │           │            │
        └───────────┼────────────┘
                    │
               updated state
```

Every node operates against the shared CUGA state.

That is the **data-flow backbone** of the graph.

---

# 16. So what does "execution graph" mean here?

This distinction is crucial for your research.

There are actually **three different graphs** you might be talking about.

### A. Static computational graph

This is:

```python
agent.graph.get_graph()
```

It tells you:

> What nodes and possible transitions exist?

---

### B. Expanded computational graph

This is:

```python
agent.graph.get_graph(xray=True)
```

It tells you:

> What does the graph look like when nested subgraphs are expanded?

---

### C. Actual execution trajectory

This is:

```text
Run 1:

START
 ↓
CHAT
 ↓
TASK_ANALYZER
 ↓
CUGA_LITE
 ↓
TOOL
 ↓
CUGA_LITE
 ↓
FINAL_ANSWER
 ↓
END
```

That is **not necessarily identical to the static graph**.

The static graph says:

```text
possible transitions
```

The trajectory says:

```text
what actually happened for this task
```

CUGA itself provides trajectory visualization via `cuga viz`, which is specifically intended to inspect execution trajectories, decision-making, and tool usage. ([GitHub][3])

---

# 17. This distinction is VERY important for your self-evolution work

Suppose your benchmark task produces:

```text
Static graph:

START
 │
 ▼
CHAT
 │
 ▼
TASK_ANALYZER
 │
 ├──── API ────► API_PLANNER
 │
 ├──── WEB ────► BROWSER_PLANNER
 │
 └──── GENERAL ► CUGA_LITE
```

But actual run:

```text
Trajectory:

CHAT
 ↓
TASK_ANALYZER
 ↓
CUGA_LITE
 ↓
tool_shortlist
 ↓
tool_call
 ↓
tool_result
 ↓
CUGA_LITE
 ↓
FINAL_ANSWER
```

The latter is what you want for **trajectory-based evolution/evaluation**.

---

# 18. And this gives you a very nice experimental representation

For your project, I'd represent CUGA as:

```text
                   ┌──────────────────────┐
                   │    STATIC HARNESS    │
                   │                      │
                   │  LangGraph topology  │
                   │  Nodes               │
                   │  Routing             │
                   │  Prompts             │
                   │  Skills              │
                   │  Tools               │
                   │  Policies            │
                   └──────────┬───────────┘
                              │
                              ▼
                         CUGA Runtime
                              │
                              ▼
                       Actual execution
                              │
                              ▼
                    ┌────────────────────┐
                    │    TRAJECTORY      │
                    │                    │
                    │ node₁              │
                    │ node₂              │
                    │ tool₁              │
                    │ observation₁       │
                    │ node₃              │
                    │ ...                │
                    └─────────┬──────────┘
                              │
                              ▼
                          Evaluator
                              │
                              ▼
                           Reward
```

Then your evolutionary algorithm can operate on **either**:

```text
static harness
```

or:

```text
runtime behavior / trajectory
```

or both.

---

# 19. One subtle but important correction from my previous answer

Earlier I described CUGA roughly as:

```text
Planner
   ↓
Executor
   ↓
Tools
```

That's useful as a conceptual abstraction, but after looking at the **actual current source**, the implementation is better described as:

```text
                        CUGA
                         │
                       CHAT
                         │
                    TASK ANALYZER
                         │
              ┌──────────┼───────────┐
              │          │           │
             API        WEB       CUGALITE
              │          │           │
              │       PLANNER        │
              │          │           │
              │       ACTION         │
              │          │           │
              │          QA           │
              │          │           │
              └──────┬───┴───────────┘
                     │
               PLAN CONTROLLER
                     │
              ┌──────┴──────┐
              │             │
           Supervisor     Reuse
              │
         sub-agent graph
              │
              ▼
          FINAL ANSWER
```

And underneath:

```text
       ┌──────────────────────┐
       │    AgentState        │
       │                      │
       │ shared state across  │
       │ graph execution      │
       └──────────────────────┘
```

That description is much closer to what the current source actually builds. 

---

## 20. The exact commands I'd use first

If you have your CUGA environment already set up, run:

```python
agent = CugaAgent(
    tools=CUSTOM_TOOLS,
    special_instructions=(
        "You are running inside an experimental CUGA wrapper. "
        "Use available tools when appropriate."
    ),
)

# 1. Actual compiled LangGraph
compiled = agent.graph

# 2. Top-level graph
top = compiled.get_graph()

# 3. Expanded graph
full = compiled.get_graph(xray=True)

# 4. Mermaid
print(full.draw_mermaid())

# 5. Nodes
print("\nNODES:")
for node in compiled.nodes:
    print(node)
```

**Start with `xray=True`.** That's the one I'd particularly recommend for what you're doing, because CUGA currently embeds **CugaLite and, when enabled, CugaSupervisor as compiled subgraphs**. 

And if your ultimate goal is to understand **exactly which node invokes which node, where the `Command(goto=...)` routing happens, and where parallel branches/sub-agent execution occur**, the next step is to trace the **individual node implementations and their routing functions**, rather than stopping at `DynamicAgentGraph.add_edges()`. That's where the actual CUGA control-flow logic lives.

[1]: https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com "CugaAgent | CUGA AGENT"
[2]: https://github.com/langchain-ai/langgraph/blob/main/libs/langgraph/langgraph/pregel/main.py?utm_source=chatgpt.com "langgraph/libs/langgraph/langgraph/pregel/main.py at main · langchain-ai/langgraph · GitHub"
[3]: https://github.com/cuga-project/cuga-agent "GitHub - cuga-project/cuga-agent: CUGA is an open-source generalist agent harness for the enterprise, supporting complex task execution on web and APIs, OpenAPI/MCP integrations, composable architecture, reasoning modes, and policy-aware features. · GitHub"
