**You:**

from the CUGA sdk docs , list me all the intergaces that ibm's cuga agent exposes in detail.....

* * *

**ChatGPT:**

Yes. I went through the **current CUGA SDK documentation and the IBM/CUGA repository** rather than just the high-level README. The important thing is that CUGA exposes several different kinds of interfaces: **agent lifecycle/execution, multi-agent orchestration, tool providers, policies, knowledge, graph-level access, callbacks/tracking, and external agent protocols**.

CUGA SDK documentation  
CUGA GitHub repository

* * *

# 1. The big picture

I would model CUGA's exposed interfaces like this:

```
                         ┌──────────────────────────┐
                         │       CugaAgent          │
                         │  single-agent interface  │
                         └────────────┬─────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
                 ▼                    ▼                    ▼
          Execution API         Policy API          Knowledge API
          invoke()              policies             knowledge
          stream()              ├─ Intent Guard       ├─ ingest
          aclose()              ├─ Playbook           ├─ search
                                ├─ Tool Guide         └─ list docs
                                ├─ Tool Approval
                                └─ Output Formatter

                                      │
                                      ▼
                            ┌──────────────────┐
                            │   Tool Provider  │
                            │      API         │
                            └────────┬─────────┘
                                     │
                 ┌───────────────────┼────────────────────┐
                 ▼                   ▼                    ▼
          Local tools          OpenAPI tools          MCP tools
          LangChain tools      REST APIs              MCP servers


                         ┌──────────────────────────┐
                         │    CugaSupervisor        │
                         │ multi-agent interface    │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────┴─────────────┐
                         ▼                          ▼
                   Local CugaAgents             A2A Agents
                                                   remote
```

And underneath all of this, CUGA exposes the **compiled LangGraph**, so you can bypass some of the high-level SDK abstraction when necessary. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

* * *

# 2. `CugaAgent` — the primary interface

This is the most important public SDK interface.

```Python
from cuga import CugaAgent

agent = CugaAgent(...)
```

CUGA describes `CugaAgent` as the main high-level interface for creating and running an agent. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

## Constructor interface

The documented parameters are:

```Python
CugaAgent(
    tools=None,
    tool_provider=None,
    model=None,
    callbacks=None,
    policy_system=None,
    special_instructions=None,
    enable_knowledge=True,
)
```

### `tools`

```Python
tools: List[BaseTool]
```

You can directly give CUGA LangChain tools.

```Python
agent = CugaAgent(
    tools=[search_tool, database_tool]
)
```

This is the simplest extension point.

* * *

### `tool_provider`

```Python
tool_provider: ToolProviderInterface
```

Instead of giving CUGA a static list of tools, you can give it a **tool provider abstraction**.

This is much more interesting for your self-evolving-agent research because tools can become dynamically discoverable rather than hard-coded. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/tool_provider)

* * *

### `model`

```Python
model: BaseChatModel
```

You can provide your own LangChain-compatible chat model.

CUGA also supports configured providers including OpenAI, IBM watsonx, Azure OpenAI, Groq, OpenRouter, etc. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

* * *

### `callbacks`

```Python
callbacks: List[BaseCallbackHandler]
```

This is CUGA's connection to the LangChain callback/observability ecosystem.

This is worth distinguishing from a dedicated CUGA "hook system": **the SDK docs expose callbacks, but they do not document a general `before_tool`, `after_tool`, `before_llm`, etc. hook interface comparable to some other agent frameworks.**

* * *

### `policy_system`

```Python
policy_system: PolicyConfigurable
```

Allows the policy subsystem to be configured/injected.

* * *

### `special_instructions`

```Python
special_instructions: Optional[str]
```

Additional instructions supplied to the agent.

Example:

```Python
agent = CugaAgent(
    special_instructions="Always respond using JSON."
)
```

* * *

### `enable_knowledge`

```Python
enable_knowledge: bool
```

Controls CUGA's built-in knowledge/RAG subsystem.

* * *

# 3. `agent.invoke()` — synchronous-style execution interface

This is the core execution interface.

```Python
result = await agent.invoke(message)
```

The documented signature is essentially:

```Python
await agent.invoke(
    message=None,
    thread_id=None,
    config=None,
    action_response=None,
    track_tool_calls=False,
)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

### Input can be

#### String

```Python
await agent.invoke(
    "Find the top customers"
)
```

#### LangChain messages

```Python
await agent.invoke([
    HumanMessage(...),
    AIMessage(...),
    HumanMessage(...)
])
```

#### `None`

This is particularly important.

CUGA allows:

```Python
await agent.invoke(
    None,
    thread_id="user-123",
    action_response=approval_response
)
```

That means `invoke()` isn't merely "send prompt → receive answer."

It is also a **resume execution interface**.

That becomes important for HITL/tool-approval workflows. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/sdk/policies/tool-approval)

* * *

# 4. `InvokeResult` — execution result interface

`invoke()` returns an `InvokeResult`.

Documented fields:

```Python
InvokeResult:
    answer
    tool_calls
    thread_id
    error
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

Conceptually:

```
InvokeResult
│
├── answer
│
├── tool_calls
│
├── thread_id
│
└── error
```

The `tool_calls` field becomes particularly valuable for evaluation and self-improvement because CUGA can expose execution information rather than only the final textual answer.

* * *

# 5. Tool-call tracking interface

CUGA provides explicit execution tracing through:

```Python
track_tool_calls=True
```

Example:

```Python
result = await agent.invoke(
    "Find the top account",
    track_tool_calls=True
)

for call in result.tool_calls:
    ...
```

Each tracked call can contain:

```
name
operation_id
arguments
result
app_name
duration_ms
timestamp
error
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

This is a **very important interface for your self-evolving-agent research**.

You can conceptually obtain:

```
Task
  ↓
CUGA execution
  ↓
tool_1(arguments)
  ↓
tool_2(arguments)
  ↓
tool_3(arguments)
  ↓
Final answer

         ↓

trajectory / tool-call metadata
```

So CUGA already gives you part of the instrumentation needed to build an evolution/evaluation harness.

* * *

# 6. `agent.stream()` — streaming execution interface

CUGA also exposes:

```Python
async for state in agent.stream(message):
    print(state)
```

It streams the execution **step by step** rather than waiting for the final result. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

Its documented arguments are essentially:

```Python
agent.stream(
    message=None,
    thread_id=None,
    config=None,
    action_response=None
)
```

This gives you a lower-latency execution observation interface.

For example:

```
stream()
   │
   ├── planning state
   ├── tool selection
   ├── tool execution
   ├── intermediate state
   ├── next action
   └── final state
```

The exact streamed state structure is lower-level than `InvokeResult`.

* * *

# 7. `thread_id` — state/session interface

CUGA supports state isolation through:

```Python
thread_id="user-123"
```

So the same agent can maintain separate conversational execution states:

```
CugaAgent
   │
   ├── thread A
   │     ├── state
   │     └── history
   │
   ├── thread B
   │     ├── state
   │     └── history
   │
   └── thread C
         ├── state
         └── history
```

This is important because `thread_id` is also used by the knowledge subsystem for **session-scoped knowledge**. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com)

* * *

# 8. `agent.graph` — low-level LangGraph interface

This is one of the most interesting interfaces.

CUGA exposes:

```Python
agent.graph
```

which gives access to the underlying compiled LangGraph.

For example:

```Python
compiled_graph = agent.graph

await compiled_graph.ainvoke(...)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

So CUGA isn't a completely opaque black box.

You effectively have:

```
                    CugaAgent
                       │
              ┌────────┴────────┐
              │                 │
         High-level API     graph property
              │                 │
        invoke()/stream()      ▼
                         Compiled LangGraph
```

### Why this matters

For your research, this means you could potentially:

* inspect the graph
* invoke the graph directly
* experiment with execution behavior
* add external orchestration around CUGA
* instrument graph execution
* build research/evolution mechanisms above the agent

without necessarily rewriting the whole CUGA harness.

* * *

# 9. `agent.policies` — policy-management interface

CUGA exposes:

```Python
agent.policies
```

which gives you the policy management interface.

The policy subsystem has **five documented policy types**:

```
PoliciesManager
│
├── Intent Guard
├── Playbook
├── Tool Guide
├── Tool Approval
└── Output Formatter
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/policies/?utm_source=chatgpt.com)

The general management API includes:

```Python
await agent.policies.list()

await agent.policies.get(policy_id)

await agent.policies.delete(policy_id)

await agent.policies.load_from_json(file_path)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/policies/?utm_source=chatgpt.com)

This is a real programmatic interface, not merely configuration files.

* * *

# 10. Intent Guard interface

```Python
await agent.policies.add_intent_guard(...)
```

Its purpose is:

> detect forbidden user intents and block them.

Example:

```Python
await agent.policies.add_intent_guard(
    name="Block Deletions",
    description="Prevent data deletion",
    keywords=["delete", "remove", "drop"],
    response="I cannot perform deletion operations."
)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/policies/intent-guard)

It can use:

* keyword matching
* natural-language/semantic matching
* custom responses
* priorities
* optional overrides

Conceptually:

```
user request
      ↓
Intent Guard
      │
      ├── forbidden → BLOCK
      │
      └── allowed → continue
```

* * *

# 11. Playbook interface

```Python
await agent.policies.add_playbook(...)
```

A Playbook is essentially **procedural knowledge injected into the agent when a trigger fires**.

Example:

```Python
await agent.policies.add_playbook(
    name="Onboarding Guide",
    keywords=["onboard", "new user"],
    content="""
    # Customer Onboarding
    1. Collect email
    2. Verify identity
    3. Create CRM account
    """
)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/policies/playbook)

It can also use semantic triggers.

An important feature is **playbook refinement based on conversation history**.

That means:

```
Playbook
   ↓
What has already been completed?
   ↓
Remove/ignore completed steps
   ↓
Focus on remaining procedure
```

This is actually quite relevant to agent evolution because it provides a primitive mechanism for **procedural adaptation**.

* * *

# 12. Tool Guide interface

```Python
await agent.policies.add_tool_guide(...)
```

Tool Guide modifies the tool description presented to the agent.

Example:

```Python
await agent.policies.add_tool_guide(
    name="Security Context",
    target_tools=["*"],
    content="All operations are logged.",
    prepend=True
)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/policies/tool-guide)

So:

```
Original tool description
        +
dynamic policy context
        ↓
modified tool description
        ↓
LLM
```

This is effectively a **dynamic tool-context injection interface**.

* * *

# 13. Tool Approval interface

```Python
await agent.policies.add_tool_approval(...)
```

This is CUGA's explicit HITL interface.

Example:

```Python
await agent.policies.add_tool_approval(
    name="Database Write Approval",
    required_tools=[
        "sql_query",
        "update_record"
    ],
    approval_message="Proceed with database modification?"
)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/policies/tool-approval)

You can control:

```
required_tools
required_apps
approval_message
show_code_preview
auto_approve_after
```

When approval is required:

```
Agent
  ↓
generates action
  ↓
Tool Approval policy
  ↓
PAUSE
  ↓
Human
  ↓
approval/rejection
  ↓
agent.invoke(
    None,
    thread_id=...,
    action_response=...
)
  ↓
resume
```

This is an actual **pause/resume execution interface**.

* * *

# 14. Output Formatter interface

```Python
await agent.policies.add_output_formatter(...)
```

It operates **after the agent generates its final answer**.

It supports three main formats:

### Markdown

```
LLM response
     ↓
formatter
     ↓
structured markdown
```

### JSON schema

```Python
format_type="json_schema"
```

### Direct replacement

```Python
format_type="direct"
```

The direct formatter doesn't require another LLM call; it replaces the output with the configured content. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/policies/output-formatter)

This gives CUGA an explicit **post-processing interface**.

* * *

# 15. Policy trigger interface

Policies can be activated using:

### Keyword triggers

```
"delete"
"payment"
"refund"
```

### Natural-language triggers

Semantic similarity against examples.

So the architecture is roughly:

```
                 Policy
                   │
          ┌────────┴────────┐
          │                 │
     keyword trigger    semantic trigger
          │                 │
          └────────┬────────┘
                   ▼
              policy fires
```

CUGA documents both exact keyword and natural-language trigger mechanisms. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/sdk/policies/?utm_source=chatgpt.com)

* * *

# 16. `agent.knowledge` — Knowledge/RAG interface

When knowledge is enabled:

```Python
agent.knowledge
```

gives access to CUGA's Knowledge Manager.

The documented operations include:

```Python
await agent.knowledge.ingest(...)
await agent.knowledge.search(...)
await agent.knowledge.list_documents()
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

For example:

```Python
await agent.knowledge.ingest(
    "/path/to/report.pdf"
)

results = await agent.knowledge.search(
    "Q4 revenue"
)

docs = await agent.knowledge.list_documents()
```

CUGA uses Docling for document ingestion and supports formats such as:

* PDF
* DOCX
* XLSX
* PPTX
* HTML
* Markdown
* images

[![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub](https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com)

* * *

# 17. Agent-level vs session-level knowledge

This is particularly interesting.

CUGA supports:

```Python
scope="agent"
```

and:

```Python
scope="session"
```

For example:

```Python
await agent.knowledge.ingest(
    "file.pdf",
    scope="session",
    thread_id="user-session-123"
)
```

So:

```
                   Knowledge
                      │
            ┌─────────┴─────────┐
            │                   │
       Agent scope         Session scope
            │                   │
       persistent          thread-specific
       shared              temporary
```

[![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub](https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com)

This is **not the same thing as a general long-term agent memory system**.

* * *

# 18. Important: CUGA's old Memory interface is deprecated

This is an important correction if you're comparing CUGA with Pi/Hermes/OpenCode for your self-evolving-agent work.

Older CUGA material mentions memory, including Mem0-style integration.

The **current settings reference explicitly says the legacy `memory_provider` was removed from CUGA classic**, and the old memory service port was removed/deprecated. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/customization/settings-reference/?utm_source=chatgpt.com)

So I would **not** currently classify CUGA as exposing a first-class generic:

```Python
agent.memory.store(...)
agent.memory.retrieve(...)
agent.memory.update(...)
```

interface.

Instead, current CUGA gives you:

```
conversation state
+
context summarization
+
knowledge/RAG
+
execution variables
+
saved workflows
```

rather than a generic long-term semantic memory API.

That's an important distinction.

* * *

# 19. `CugaSupervisor` — multi-agent interface

CUGA also exposes:

```Python
CugaSupervisor
```

This is the multi-agent orchestration layer.

```Python
supervisor = CugaSupervisor(
    agents={
        "crm": crm_agent,
        "email": email_agent,
    }
)
```

Then:

```Python
result = await supervisor.invoke(...)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_supervisor)

Its constructor accepts:

```Python
CugaSupervisor(
    agents,
    model=None,
    description=None,
    callbacks=None
)
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_supervisor)

* * *

# 20. Supervisor delegation interface

The supervisor can delegate work to specialized agents:

```
                    Supervisor
                        │
             ┌──────────┼──────────┐
             ▼          ▼          ▼
           CRM        Email     Analytics
          Agent       Agent       Agent
```

It can also pass variables between agents.

For example:

```
CRM Agent
   │
   │ top_customer = Alice
   ▼
Supervisor
   │
   │ passes variable
   ▼
Email Agent
   │
   └── send email to Alice
```

CUGA explicitly documents variable passing between agents. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com)

* * *

# 21. `CugaSupervisor.from_yaml()`

Another interface:

```Python
supervisor = await CugaSupervisor.from_yaml(
    "supervisor_config.yaml"
)
```

This allows the multi-agent topology to be represented declaratively rather than entirely in Python. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_supervisor)

So:

```
Python API
     OR
YAML configuration
     ↓
CugaSupervisor
```

* * *

# 22. A2A — external-agent interface

This is another major interface.

CUGA can combine:

```
local CugaAgent
```

with:

```
remote A2A agent
```

The configuration looks conceptually like:

```Python
{
    "name": "analytics",
    "type": "external",
    "description": "Remote analytics service",
    "config": {
        "a2a_protocol": {
            "endpoint": "...",
            "transport": "http"
        }
    }
}
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_supervisor)

So CUGA can participate in an agent network:

```
                  CugaSupervisor
                  /      |      \
                 /       |       \
                ▼        ▼        ▼
          Local CUGA   Local CUGA   Remote A2A
             Agent       Agent       Agent
                                      │
                                      ▼
                                another system
```

This is a **real external-agent interoperability interface**, not just internal multi-agent delegation.

* * *

# 23. `ToolProviderInterface` — custom tool abstraction

This is probably the most important _actual interface/protocol_ in CUGA's SDK.

CUGA documents:

```Python
class ToolProviderInterface
```

with four core methods:

```Python
async def initialize() -> None

async def get_apps() -> List[AppDefinition]

async def get_tools(
    app_name: str
) -> List[StructuredTool]

async def get_all_tools() -> List[StructuredTool]
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/tool_provider)

This means you can implement:

```Python
class MyToolProvider(ToolProviderInterface):
    ...
```

and plug your own tool ecosystem into CUGA.

* * *

# 24. `AppDefinition` interface/schema

Applications exposed through a tool provider have:

```Python
AppDefinition(
    name=...,
    url=...,
    description=...,
    type=...
)
```

The schema is:

```
AppDefinition
├── name: str
├── url: Optional[str]
├── description: Optional[str]
└── type: Optional[str]
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/tool_provider)

This gives CUGA a level of abstraction above individual tools:

```
Application
   │
   ├── Tool A
   ├── Tool B
   ├── Tool C
   └── Tool D
```

That application/tool hierarchy is important when designing dynamic tool discovery.

* * *

# 25. `CombinedToolProvider`

CUGA provides a built-in:

```Python
CombinedToolProvider
```

It combines:

```
Runtime tracker tools
        +
MCP registry tools
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/tool_provider)

The priority is roughly:

```
             Tool request
                  │
                  ▼
       Runtime/dynamic tools?
             /       \
           yes        no
            │          │
            ▼          ▼
       use runtime   Registry
                      │
                 ┌────┴────┐
                 ▼         ▼
              OpenAPI     MCP
```

This is a particularly useful interface if you're building a self-evolving system that may dynamically add tools.

* * *

# 26. OpenAPI interface

CUGA's registry can expose tools derived from OpenAPI specifications.

Conceptually:

```
OpenAPI specification
        ↓
CUGA registry
        ↓
ToolProvider
        ↓
CugaAgent
```

The registry handles things such as:

* authentication
* request construction
* response parsing
* tool discovery

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/tool_provider)

Therefore APIs can be converted into agent-callable tools without manually implementing every LangChain tool.

* * *

# 27. MCP interface

CUGA also exposes MCP servers through its tool ecosystem.

```
MCP Server
    ↓
MCP Registry
    ↓
ToolProvider
    ↓
CugaAgent
```

The agent can therefore consume external MCP tools.

The repository explicitly identifies **MCP + OpenAPI + LangChain** as supported tool integration mechanisms. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub](https://github.com/cuga-project/cuga-agent)

* * *

# 28. LangChain tool interface

CUGA directly accepts LangChain tools:

```Python
@tool
def my_tool(...):
    ...
```

and:

```Python
CugaAgent(
    tools=[my_tool]
)
```

So there are effectively three major tool ingestion paths:

```
                 CugaAgent
                    │
       ┌────────────┼────────────┐
       ▼            ▼            ▼
   LangChain      MCP         OpenAPI
     tools       servers        APIs
```

[![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/cuga-project/cuga-agent)

* * *

# 29. `@tracked_tool` — tool instrumentation interface

CUGA exposes a decorator:

```Python
from cuga import tracked_tool
```

Example:

```Python
@tracked_tool(app_name="calculator")
def add(a, b):
    return a + b
```

or combined with LangChain:

```Python
@tool
@tracked_tool(app_name="math")
def multiply(a, b):
    return a * b
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/tool_provider)

This is effectively an instrumentation wrapper around tools.

* * *

# 30. Callback interface

CUGA accepts:

```Python
callbacks=[
    ...
]
```

using LangChain's:

```Python
BaseCallbackHandler
```

interface.

This gives you an integration point for things like:

* logging
* tracing
* observability
* custom execution monitoring

The SDK exposes the callback parameter on both `CugaAgent` and `CugaSupervisor`. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

But again:

> **Callbacks ≠ a CUGA-specific hook lifecycle API.**

I would not claim that CUGA exposes a rich generic hooks system unless you're referring to internal implementation details.

* * *

# 31. `aclose()` — resource lifecycle interface

CUGA exposes:

```Python
await agent.aclose()
```

to release resources.

This matters particularly when knowledge/database/background resources are enabled. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

So the basic lifecycle is:

```
create
  ↓
invoke / stream
  ↓
...
  ↓
aclose
```

* * *

# 32. Context summarization interface

CUGA also has a configurable context-management layer.

It can automatically summarize older conversation messages when the context window gets too large.

Configuration includes:

```TOML
[context_summarization]

enabled = true
keep_last_n_messages = 10
trim_tokens_to_summarize = 500
summarization_model = "gpt-4o-mini"
trigger_fraction = 0.75
```

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/customization/context-summarization/?utm_source=chatgpt.com)

This isn't exposed as:

```Python
agent.memory
```

but it is nevertheless an important **state/context management interface**.

* * *

# 33. Execution-variable interface

CUGA's execution system also maintains intermediate variables.

Conceptually:

```
Tool A
  ↓
variable_1
  ↓
Tool B
  ↓
variable_2
  ↓
Tool C
```

These variables can be used for chaining and, in supervisor mode, passed between agents. The CUGA demo explicitly exposes the variables view and describes intermediate-result storage and chaining. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/integrations/huggingface-spaces/?utm_source=chatgpt.com)

This is different from memory:

```
variable ≠ memory
```

Variables are primarily **execution-state artifacts**.

* * *

# 34. Browser/computer-use interface

CUGA also exposes a task-execution mode interface:

```
API
WEB
HYBRID
```

### API

```
Agent
 ↓
APIs / tools
```

### Web

```
Agent
 ↓
Browser
 ↓
web page
```

### Hybrid

```
                Agent
               /     \
              ▼       ▼
            APIs    Browser
```

The hybrid mode is particularly interesting because CUGA can perform API operations and browser operations in the same workflow. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/cuga-project/cuga-agent)

* * *

# 35. Skills interface

Current CUGA also has an **Agent Skills** mechanism.

Skills are packaged as:

```
SKILL.md
```

files with frontmatter such as:

```YAML
name: ...
description: ...
```

CUGA discovers skills and exposes a:

```
load_skill
```

tool so that the agent can load the full instructions when needed. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub](https://github.com/cuga-project/cuga-agent)

So:

```
Skill repository
       ↓
SKILL.md
       ↓
skill discovery
       ↓
available skills
       ↓
load_skill
       ↓
full instructions
```

This is quite similar conceptually to the skill systems you've been looking at in OpenCode/Pi/Hermes.

* * *

# 36. Save & Reuse interface

CUGA has an experimental:

```
save_reuse_fast
```

mode.

It can capture and reuse successful execution paths involving things such as:

* plans
* code
* trajectories/workflows

The purpose is to avoid recomputing similar workflows. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/cuga-project/cuga-agent)

For your research, this is probably one of the **most relevant CUGA interfaces** because it starts approaching:

```
experience
   ↓
successful trajectory
   ↓
store
   ↓
retrieve
   ↓
reuse
```

However, it is better described as **workflow/trajectory reuse**, not autonomous self-improvement.

* * *

# 37. Reflection interface

CUGA has a configurable:

```TOML
reflection_enabled = true
```

capability.

The documented behavior is an additional reflection pass after planning/execution to detect and recover from errors. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/customization/settings-reference/?utm_source=chatgpt.com)

Conceptually:

```
Task
 ↓
Plan
 ↓
Execute
 ↓
Reflect
 ↓
Recover / finish
```

But this is primarily a **built-in execution feature**, not a general public:

```Python
agent.reflect(...)
```

API.

That distinction matters.

* * *

# 38. Observability interface

CUGA supports tracking and OpenTelemetry/OpenLIT-based observability.

Current settings expose:

```TOML
[observability]
openlit = true
```

and an OTLP endpoint.

[![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/customization/settings-reference/?utm_source=chatgpt.com)

There is also:

```TOML
tracker_enabled = true
```

for execution tracking.

So the observability stack can look like:

```
CUGA
 │
 ├── execution tracker
 │
 ├── tool-call tracking
 │
 └── OpenTelemetry/OpenLIT
           ↓
       OTLP collector
```

* * *

# 39. What CUGA **doesn't** expose as a first-class interface

This is arguably even more important for your research.

Based on the **current public SDK documentation**, I would **not** classify the following as first-class CUGA SDK interfaces:

| Interface | Current status |
| --- | --- |
| `agent.memory.store()` | ❌ Legacy memory removed/deprecated |
| `agent.memory.retrieve()` | ❌ Not current public SDK |
| `agent.hooks.before_tool()` | ❌ No documented generic CUGA hook API |
| `agent.hooks.after_tool()` | ❌ No documented generic CUGA hook API |
| `agent.learn()` | ❌ No |
| `agent.evolve()` | ❌ No |
| `agent.update_prompt()` as an evolution API | ❌ No |
| `agent.update_policy_from_experience()` | ❌ No |
| `agent.modify_trajectory()` | ❌ No |
| `agent.evaluate()` | ❌ Not a core agent SDK method |
| `agent.finetune()` | ❌ No |
| `agent.self_improve()` | ❌ No |

Instead, CUGA gives you **building blocks** from which you could build those systems:

```
CugaAgent
CugaSupervisor
ToolProviderInterface
InvokeResult
tool-call tracking
stream()
graph
policies
knowledge
skills
variables
reflection
save/reuse
callbacks
A2A
MCP
OpenAPI
```

* * *

# 40. The complete interface map

If I were documenting CUGA as an SDK for your research, I'd summarize its exposed interfaces as:

| Layer | Interface | What it gives you |
| --- | --- | --- |
| **Agent** | `CugaAgent` | Create/run one agent |
| **Execution** | `invoke()` | Execute task |
| **Execution** | `stream()` | Observe execution incrementally |
| **Execution** | `thread_id` | Persistent per-thread state |
| **Execution** | `action_response` | Resume paused execution |
| **Result** | `InvokeResult` | Answer + trajectory/tool metadata |
| **Graph** | `agent.graph` | Underlying LangGraph |
| **Lifecycle** | `aclose()` | Resource cleanup |
| **Tools** | `ToolProviderInterface` | Custom tool discovery/provider |
| **Tools** | `CombinedToolProvider` | Runtime + registry tools |
| **Tools** | LangChain tools | Direct Python tools |
| **Tools** | OpenAPI | REST API → tools |
| **Tools** | MCP | MCP server → tools |
| **Tools** | `@tracked_tool` | Tool instrumentation |
| **Tools** | `AppDefinition` | Application/tool namespace |
| **Policies** | `agent.policies` | Policy management |
| **Policies** | Intent Guard | Block intents |
| **Policies** | Playbook | Inject procedures |
| **Policies** | Tool Guide | Modify tool context |
| **Policies** | Tool Approval | HITL |
| **Policies** | Output Formatter | Post-process output |
| **Knowledge** | `agent.knowledge` | RAG/document store |
| **Knowledge** | `ingest()` | Add documents |
| **Knowledge** | `search()` | Retrieve knowledge |
| **Knowledge** | `list_documents()` | Manage corpus |
| **Multi-agent** | `CugaSupervisor` | Orchestrate agents |
| **Multi-agent** | `from_yaml()` | Declarative topology |
| **Multi-agent** | A2A | Remote agents |
| **Context** | Summarization | Compress old context |
| **Skills** | `SKILL.md` / `load_skill` | Procedural skill system |
| **Experience** | Save & Reuse | Reuse successful workflows |
| **Reasoning** | Reflection | Post-execution reflection |
| **Observation** | Callbacks | LangChain callbacks |
| **Observation** | Tracker | Execution/tool telemetry |
| **Observation** | OpenTelemetry/OpenLIT | External tracing |
| **Computer use** | API/Web/Hybrid modes | APIs + browser |
| **External interoperability** | MCP/A2A | CUGA ↔ agent/tool ecosystem |

* * *

# 41. The really important part for your **self-evolving-agent** research

Given what you've been investigating with **Pi Agent, CUGA, evolution harnesses, memory, skills, MCP, hooks, etc.**, I'd classify CUGA like this:

```
                     CUGA
                      │
       ┌──────────────┼────────────────┐
       │              │                │
       ▼              ▼                ▼
   EXECUTION       EXTENSION        GOVERNANCE
       │              │                │
   invoke()      ToolProvider       Policies
   stream()      MCP               HITL
   graph         OpenAPI            Intent
   variables     LangChain          Playbooks
       │              │                │
       └──────────────┼────────────────┘
                      │
                      ▼
                 OBSERVABILITY
                      │
              tool tracking
              trajectories
              callbacks
              OpenTelemetry
                      │
                      ▼
              EXPERIENCE LAYER
                      │
                save/reuse
                reflection
                knowledge
                skills
```

### But the missing piece is:

```
             ┌────────────────────────────┐
             │   SELF-EVOLUTION ENGINE    │
             │                            │
             │ evaluate trajectory        │
             │ diagnose failure           │
             │ generate improvement       │
             │ modify skill/prompt/tool   │
             │ test modification          │
             │ retain/reject change       │
             └──────────────┬─────────────┘
                            │
                            ▼
                         CUGA
```

**CUGA gives you a surprisingly good substrate for building that layer, but it does not itself provide the complete self-evolution loop.**

In particular, its **`InvokeResult` + tool tracking + `stream()` + `graph` + policies + skills + knowledge + save/reuse + reflection** give you many of the hooks you would need to build an external evolution harness. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+3![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+3![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+3](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

And one especially important conclusion from the current docs:

> **CUGA is much stronger on execution orchestration, tool integration, governance, and enterprise agent composition than on first-class long-term memory or autonomous self-modification.**

That distinction is important when comparing it against **Pi/OpenCode/Hermes** for your self-evolving-agent architecture.

CUGA SDK — CugaAgent  
CUGA SDK — CugaSupervisor  
CUGA SDK — Tool Provider  
CUGA SDK — Policies 

* * *

**You:**

should we use the SDK or clone it...?

* * *

**ChatGPT:**

For **your specific goal — building/researching a self-evolving agent — I would NOT clone/fork CUGA as the foundation.**

I would use a **hybrid approach: CUGA SDK as the execution substrate + your own external evolution harness**.

There is one important caveat: CUGA's current repository documentation says the project is **proprietary to IBM Corporation, all rights reserved**. So "clone it" for local inspection is technically possible and the docs themselves provide clone instructions, but **treating the source as a base to modify/redistribute is not something I'd assume you're licensed to do**. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/README/?utm_source=chatgpt.com)

* * *

# My recommendation

```
                 YOUR RESEARCH SYSTEM
                        │
              ┌─────────▼─────────┐
              │ Evolution Harness │  ← YOU BUILD
              │                   │
              │ • evaluator       │
              │ • trajectory DB   │
              │ • failure analysis│
              │ • mutation        │
              │ • selection       │
              │ • experiments     │
              └─────────┬─────────┘
                        │
              ┌─────────▼─────────┐
              │   CUGA SDK        │  ← USE
              │                   │
              │ CugaAgent         │
              │ invoke()         │
              │ stream()         │
              │ graph             │
              │ policies          │
              │ knowledge         │
              │ skills            │
              │ ToolProvider      │
              └─────────┬─────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
        Tools         MCP/API      Browser
```

This gives you the best separation of concerns.

* * *

# Why I would **not** clone CUGA as the core

The SDK already gives you unusually good extension points.

`CugaAgent` accepts:

* custom tools
* custom `ToolProvider`
* custom model
* callbacks
* policies
* special instructions
* knowledge configuration

and exposes:

* `invoke()`
* `stream()`
* `graph`
* `policies`
* `knowledge`
* `aclose()`

It also gives you **tool-call tracking**, including arguments, results, duration, errors, timestamps, etc. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

That's almost exactly the kind of instrumentation your evolution harness needs.

So instead of modifying:

```
CUGA internals
```

you can observe:

```
CUGA execution
      ↓
trajectory
      ↓
evaluator
      ↓
proposed improvement
      ↓
new agent configuration
      ↓
CUGA execution again
```

* * *

# The architecture I'd build

For your research, I'd make **CUGA the "body" and your evolution framework the "brain that evolves the body."**

### CUGA handles

```
┌─────────────────────────────┐
│            CUGA             │
├─────────────────────────────┤
│ Planning                    │
│ Task decomposition          │
│ Tool selection              │
│ Tool execution              │
│ Browser/API interaction     │
│ Multi-agent orchestration   │
│ Policies                    │
│ HITL                        │
│ Knowledge/RAG               │
│ Skills                      │
│ Context management          │
└─────────────────────────────┘
```

### Your system handles

```
┌─────────────────────────────┐
│      EVOLUTION HARNESS      │
├─────────────────────────────┤
│ Task generation             │
│ Benchmark                   │
│ Trajectory collection       │
│ Success evaluation          │
│ Failure classification      │
│ Reflection                  │
│ Experience selection        │
│ Skill mutation              │
│ Prompt mutation             │
│ Policy mutation             │
│ Tool-selection optimization │
│ Candidate generation        │
│ Candidate evaluation        │
│ Regression testing          │
│ Versioning                  │
│ Selection / rollback        │
└─────────────────────────────┘
```

This is much cleaner scientifically.

* * *

# And CUGA actually gives you the right observation points

For example:

```Python
result = await agent.invoke(
    task,
    track_tool_calls=True
)
```

You can capture:

```
task
 ↓
CUGA
 ↓
tool_1
 ├─ arguments
 ├─ result
 ├─ duration
 └─ error
 ↓
tool_2
 ├─ arguments
 ├─ result
 └─ ...
 ↓
answer
```

The current SDK explicitly exposes this tool-call tracking information. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

You can turn that into your own:

```Python
Trajectory(
    task=task,
    actions=result.tool_calls,
    final_answer=result.answer,
    success=score
)
```

Then your evolution algorithm operates **outside CUGA**.

* * *

# Even better: use `agent.graph`

This is where I would go one level deeper.

CUGA exposes:

```Python
agent.graph
```

which gives access to the underlying compiled LangGraph. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

So your architecture can have two levels:

### Level 1 — normal research

```
Evolution Harness
       ↓
CugaAgent.invoke()
       ↓
trajectory
```

### Level 2 — advanced research

```
Evolution Harness
       ↓
CugaAgent.graph
       ↓
LangGraph execution
       ↓
inspect / instrument / experiment
```

That means you don't have to immediately hack CUGA's internals.

* * *

# Where I WOULD clone the repository

I **would clone it locally for source-code research**.

In fact, for your project I'd maintain:

```
research/
│
├── cuga-agent/          ← local CUGA source, READ/UNDERSTAND
│
├── evolution-harness/   ← YOUR CODE
│
├── experiments/
│   ├── exp001/
│   ├── exp002/
│   └── ...
│
├── trajectories/
│
├── evaluators/
│
├── mutations/
│
└── benchmarks/
```

The CUGA docs explicitly provide a development setup based on cloning the repository and installing it with `uv`. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/getting-started/installation/?utm_source=chatgpt.com)

But I'd keep the conceptual dependency:

```
your_system
    ↓
CUGA public interfaces
```

rather than:

```
your_system
    ↓
random CUGA internal classes
```

* * *

# When would I actually modify CUGA itself?

Only if your research question requires changing the **agent's internal algorithm**.

For example, suppose your research hypothesis is:

> "Can an agent autonomously evolve its task decomposition algorithm?"

Then you may need:

```
CUGA planner
     ↓
modify planner
     ↓
new planner
     ↓
benchmark
```

At that point, an external wrapper isn't enough.

Similarly, if you want to study:

### Internal planning evolution

```
task
 ↓
planner
 ↓
plan
```

and evolve:

```
planner → planner'
```

### Internal tool-selection evolution

```
LLM
 ↓
tool selection
```

and evolve:

```
selection policy → selection policy'
```

### Internal state-machine evolution

```
LangGraph
   ↓
modify graph topology
```

then you'd need deeper source-level access.

**But that's a different research question from "self-evolving agent."**

* * *

# CUGA already has several things you don't need to reinvent

This is another reason I'd avoid rewriting it.

CUGA already provides a policy framework with:

```
Intent Guard
Playbook
Tool Guide
Tool Approval
Output Formatter
```

and policies can be managed programmatically through `agent.policies`. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/policies/?utm_source=chatgpt.com)

It also has a skill mechanism based around `SKILL.md`, discovery, `load_skill`, and sandboxed skill assets. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/guides/agent-skills/?utm_source=chatgpt.com)

And it has **Save & Reuse** execution modes intended for successful workflow persistence/reuse. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/usage/modes/?utm_source=chatgpt.com)

So rebuilding all of that would distract from your actual research contribution.

* * *

# The really interesting possibility

I'd actually make **skills the first evolutionary genome**.

For example:

```
              SKILL v1
                 │
                 ▼
             CUGA Agent
                 │
                 ▼
             trajectories
                 │
                 ▼
              evaluator
                 │
          failure analysis
                 │
                 ▼
        Skill Mutator / LLM
                 │
       ┌─────────┴─────────┐
       ▼                   ▼
   SKILL v2             SKILL v3
       │                   │
       └─────────┬─────────┘
                 ▼
              evaluate
                 │
        ┌────────┴────────┐
        ▼                 ▼
     retain             reject
```

This is especially attractive because CUGA's current skill system already supports reusable `SKILL.md` instruction packs and on-demand loading. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/guides/agent-skills/?utm_source=chatgpt.com)

Then your contribution becomes:

> **Can an agent autonomously discover, mutate, evaluate, and retain improved skills based on execution experience?**

That's a much cleaner research problem than modifying CUGA itself.

* * *

# SDK vs clone: my scorecard

| Criterion | CUGA SDK | Clone/modify CUGA |
| --- | --- | --- |
| Start quickly | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Stable experiments | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Tool integration | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| MCP | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Skills | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Policies | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Trajectory observation | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Modify planner | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Modify internal graph | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Build evolution harness | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Reproducibility | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Maintainability | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Risk of coupling to internals | **Low** | **High** |
| Legal/licensing simplicity | **Better** | **Requires careful review** |

* * *

# So my recommendation for **your project**

I would do:

```
              ┌──────────────────────┐
              │  Evolution Harness   │
              │      YOUR WORK       │
              └──────────┬───────────┘
                         │
             Public CUGA SDK interfaces
                         │
              ┌──────────▼───────────┐
              │       CUGA           │
              │                      │
              │ CugaAgent            │
              │ ToolProvider         │
              │ Skills               │
              │ Policies             │
              │ Knowledge            │
              │ Supervisor           │
              │ Graph                │
              └──────────────────────┘
```

And **also clone the CUGA repository locally for understanding/debugging**, but don't make your research system depend on private/internal CUGA implementation unless your experiment specifically requires it.

### In one sentence:

> **Use the SDK as the agent runtime; clone the repo as a source-code reference; build the self-evolution layer outside CUGA.**

That gives you a much stronger experimental story: **CUGA remains the fixed agent substrate, while your evolution harness is the independent variable.**

And importantly, because the current CUGA repository states **"proprietary to IBM Corporation. All rights reserved,"** I'd avoid treating the clone as an open-source fork without first checking the applicable IBM license/permission. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/README/?utm_source=chatgpt.com)

* * *

**You:**

then , can i provide it with custom skills (on runtime) and memories, ..... since our self evolving part/harness will also iterate on it......

* * *

**ChatGPT:**

Yes — **but with an important distinction**:

> **Custom skills: yes, very naturally. Custom memory: yes, but I would implement it outside CUGA's native knowledge/memory layer and inject/retrieve it through your harness.**

For your self-evolving architecture, this is actually a **very good design**.

* * *

# 1. Runtime skills — YES

CUGA's skill system is a good fit for this.

You can think of CUGA as receiving a **mutable skill set** from your evolution harness:

```
                 Evolution Harness
                       │
                       │ generates / mutates
                       ▼
                 ┌─────────────┐
                 │ Skill Store │
                 └──────┬──────┘
                        │
                 current skills
                        │
                        ▼
                  ┌───────────┐
                  │   CUGA    │
                  │  Agent     │
                  └─────┬─────┘
                        │
                    execution
                        │
                        ▼
                   trajectory
                        │
                        ▼
                 Evolution Harness
                        │
                   evaluate
                        │
                  mutate skills
                        │
                        └──────────→ v2
```

CUGA's current Agent Skills mechanism is based around `SKILL.md` packages and on-demand skill loading. [CUGA Agent Skills documentation](https://docs.cuga.dev/docs/guides/agent-skills/?utm_source=chatgpt.com)

So your harness could maintain:

```
skills/
├── web_research/
│   └── SKILL.md
├── database_analysis/
│   └── SKILL.md
├── debugging/
│   └── SKILL.md
└── report_generation/
    └── SKILL.md
```

and mutate them between runs.

* * *

# 2. But there's an important question: **can you change skills while an agent is already running?**

I would distinguish:

### Between runs

**Definitely the architecture I'd recommend.**

```
Run 1
  ↓
evaluate
  ↓
modify skill
  ↓
Run 2
  ↓
evaluate
  ↓
modify skill
  ↓
Run 3
```

### During a single run

I wouldn't design your system around dynamically rewriting the skill files while the same CUGA execution is already in progress.

Instead:

```
Evolution Harness
      │
      │ prepare version
      ▼
Skill snapshot v17
      │
      ▼
CUGA execution
      │
      ▼
trajectory
      │
      ▼
evaluation
```

That gives you **reproducibility**.

You can record:

```JSON
{
  "run_id": "run_183",
  "skill_version": "v17",
  "memory_version": "v42",
  "policy_version": "v8"
}
```

This is extremely valuable for your experiments.

* * *

# 3. Custom memory — YES, but don't confuse it with CUGA Knowledge

This is the part where I'd slightly change my earlier recommendation.

CUGA's current documentation distinguishes its **Knowledge/RAG** functionality from the old memory system. The old generic `memory_provider` interface was removed/deprecated.

So don't try to make:

```Python
agent.memory.store(...)
agent.memory.retrieve(...)
```

your primary architecture.

Instead, build:

```
                  Evolution Harness
                         │
              ┌──────────┴──────────┐
              │                     │
         Skill Store           Memory Store
              │                     │
              ▼                     ▼
         SKILL.md / DB       vector / graph / SQL
              │                     │
              └──────────┬──────────┘
                         │
                         ▼
                       CUGA
```

* * *

# 4. There are actually **three different kinds of memory** you should separate

For your research, I strongly recommend this distinction.

## A. Episodic memory

"What happened in previous runs?"

Example:

```JSON
{
  "task": "research competitor X",
  "actions": [
    "search_web",
    "open_page",
    "extract_data"
  ],
  "result": "successful",
  "score": 0.91
}
```

This is **experience**.

Store it in your evolution harness.

* * *

## B. Semantic memory

"What has the agent learned/established?"

For example:

```
Company X launched Product Y in 2025.
Product Y uses architecture Z.
```

This can live in:

```
vector DB
+
graph DB
+
document store
```

depending on your experiment.

* * *

## C. Procedural memory

"How should I perform this task?"

This is where **skills** belong.

```
Skill:
Research competitor

1. Search official sources
2. Cross-check claims
3. Extract evidence
4. Compare sources
5. Produce structured result
```

And this is exactly what your evolutionary system can modify.

* * *

# 5. This gives you a very powerful architecture

I'd structure your system as:

```
                         ┌──────────────────────┐
                         │   EVOLUTION HARNESS  │
                         └───────────┬──────────┘
                                     │
             ┌───────────────────────┼───────────────────────┐
             │                       │                       │
             ▼                       ▼                       ▼
       Skill Store             Memory Store           Experience DB
             │                       │                       │
        procedural              semantic/episodic      trajectories
             │                       │                       │
             └───────────────────────┼───────────────────────┘
                                     │
                             Runtime Builder
                                     │
                         ┌───────────▼───────────┐
                         │       CUGA Agent      │
                         │                       │
                         │ system instructions   │
                         │ skills                │
                         │ knowledge             │
                         │ tools                 │
                         │ policies              │
                         └───────────┬───────────┘
                                     │
                                  execute
                                     │
                                     ▼
                                Trajectory
                                     │
                                     ▼
                                Evaluator
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
                 success          failure         partial
                    │                │                │
                    └────────────────┼────────────────┘
                                     ▼
                               Evolution LLM
                                     │
                         ┌───────────┼───────────┐
                         ▼           ▼           ▼
                      Skill       Memory       Policy
                      mutation    update       mutation
                         │           │           │
                         └───────────┼───────────┘
                                     ▼
                                  next run
```

* * *

# 6. And you can inject memory into CUGA at runtime

There are several ways to do this.

### Option 1 — Context injection

Your harness retrieves relevant memories:

```Python
memories = memory.search(task)
```

Then constructs the runtime instructions:

```
Relevant previous experience:

- When researching financial data, official filings
  performed better than news articles.
- Avoid source X because it frequently returned stale data.
- Use tool Y before tool Z.
```

Then:

```Python
CugaAgent(
    special_instructions=runtime_context
)
```

Conceptually:

```
Task
 ↓
Memory retrieval
 ↓
Relevant memories
 ↓
CUGA special instructions/context
 ↓
Agent
```

* * *

# 7. Better: make memory a tool

I actually prefer this for your experiment.

Give CUGA tools like:

```
search_memory()
store_memory()
update_memory()
forget_memory()
```

Then the agent itself can decide when memory is useful.

For example:

```
CUGA
 │
 ├── search_web()
 ├── database()
 ├── search_memory()
 ├── store_memory()
 └── update_memory()
```

And your `ToolProviderInterface` is a natural extension point for this. CUGA's SDK explicitly allows custom tool providers. [CUGA Tool Provider documentation](https://docs.cuga.dev/docs/sdk/tool_provider/?utm_source=chatgpt.com)

This means your memory system can be completely external:

```
                  CUGA
                   │
          search_memory()
                   │
                   ▼
             Memory Service
                   │
         ┌─────────┼─────────┐
         ▼         ▼         ▼
       Vector     Graph      SQL
        DB         DB        DB
```

* * *

# 8. This is actually better for self-evolution

Suppose run #17 fails.

Your evaluator discovers:

```
Failure:
Agent trusted search snippets without opening
primary sources.
```

Your evolution system can generate:

### New skill

```
ResearchSkill v8

RULE:
Never use a search snippet as final evidence.
Open and verify the primary source.
```

### New episodic memory

```
Run 17:
Search snippets caused incorrect answer.
Primary-source verification fixed it.
```

### New policy

```
Tool Guide:
For research tasks, prefer primary-source tools.
```

Now:

```
                 Failure
                    │
          ┌─────────┼──────────┐
          ▼         ▼          ▼
        Skill     Memory     Policy
       mutation   update     update
          │         │          │
          └─────────┼──────────┘
                    ▼
                 Run 18
```

That's **much more interesting than simply adding RAG**.

* * *

# 9. You should version all three

This is extremely important for your research.

Don't just have:

```
skills/
memory/
```

Have:

```
evolution/
│
├── skills/
│   ├── research/
│   │   ├── v001/
│   │   ├── v002/
│   │   └── v003/
│
├── memories/
│   ├── v001/
│   ├── v002/
│   └── v003/
│
├── policies/
│   ├── v001/
│   └── v002/
│
└── runs/
    ├── run_001.json
    ├── run_002.json
    └── run_003.json
```

Each run records:

```JSON
{
  "run_id": "run_018",
  "agent_version": "agent_7",
  "skill_version": "skill_23",
  "memory_version": "memory_14",
  "policy_version": "policy_8",
  "tools_version": "tools_5",
  "trajectory": "...",
  "score": 0.93
}
```

Now you can actually answer:

> **"Why did v23 outperform v22?"**

That's essential for a research paper.

* * *

# 10. One thing I would NOT do

Don't let the evolution LLM directly rewrite everything:

```
LLM
 ↓
rewrite entire agent
```

That becomes impossible to evaluate.

Instead, constrain mutations:

```
Evolution Engine
      │
      ├── modify skill
      ├── add memory
      ├── delete memory
      ├── modify policy
      ├── modify tool description
      └── modify system instruction
```

Each mutation is a **candidate**.

Then:

```
candidate
   ↓
benchmark
   ↓
score
   ↓
better?
 ┌─┴─┐
yes no
 │   │
retain discard
```

This gives you an actual evolutionary loop.

* * *

# 11. So your CUGA setup becomes

I would create a **thin adapter around CUGA**:

```Python
class EvolvingCugaAgent:

    def __init__(
        self,
        skill_store,
        memory_store,
        evolution_db,
    ):
        ...

    async def run(self, task):

        skills = self.skill_store.get_current()
        memories = self.memory_store.retrieve(task)

        agent = build_cuga_agent(
            skills=skills,
            memories=memories
        )

        result = await agent.invoke(
            task,
            track_tool_calls=True
        )

        trajectory = collect_trajectory(result)

        score = evaluate(
            task,
            result,
            trajectory
        )

        return trajectory, score
```

Then:

```Python
while True:

    trajectory, score = await agent.run(task)

    candidates = evolution_engine.propose(
        trajectory,
        score
    )

    winner = evaluate_candidates(candidates)

    promote(winner)
```

* * *

# 12. The key architectural decision

I'd therefore revise the earlier **"SDK vs clone"** answer slightly:

### Don't clone CUGA to get custom memory/skills.

You **don't need to**.

Use:

```
CUGA SDK
   +
your Skill Store
   +
your Memory Store
   +
your Evolution Harness
```

and only dive into the cloned CUGA source when you need to understand or experiment with internals.

* * *

## In fact, this could become your research contribution

Your architecture could be framed as:

> **CUGA = fixed agent execution substrate**  
> **Evolution Harness = adaptive meta-controller**  
> **Skills = mutable procedural knowledge**  
> **Memory = mutable experience/semantic knowledge**  
> **Policies = mutable behavioral constraints**  
> **Trajectory Store = evolutionary fitness history**

Then the evolution loop becomes:

```
             ┌─────────────────────┐
             │       TASK          │
             └──────────┬──────────┘
                        ▼
             ┌─────────────────────┐
             │   Runtime Builder   │
             │                     │
             │ Skills + Memory +   │
             │ Policies + Tools    │
             └──────────┬──────────┘
                        ▼
                  ┌───────────┐
                  │   CUGA    │
                  └─────┬─────┘
                        ▼
                    TRAJECTORY
                        │
                        ▼
                    EVALUATOR
                        │
                        ▼
                 EVOLUTION ENGINE
                        │
              ┌─────────┼─────────┐
              ▼         ▼         ▼
            Skills    Memory    Policies
              │         │         │
              └─────────┼─────────┘
                        ▼
                  NEW AGENT STATE
                        │
                        └──────────→ next generation
```

**That's the architecture I'd recommend for your project.** It lets you use CUGA's substantial agent/tool/policy infrastructure without making CUGA itself the thing you're trying to evolve.

* * *

**You:**

we need causal and counter factual analysis of traces .... do CUGA SDK support it ? , .... along  with just the normal logging , we also need the exact agent state , so that we can replay from there ... for our counterfactual analysis......

* * *

**ChatGPT:**

Yes — and **this changes my recommendation somewhat**.

For your research, you need to distinguish **ordinary observability** from **causal replayability**. CUGA gives you a good amount of the former, but **the CUGA SDK does not appear to expose the full checkpoint/time-travel interface you need as a first-class SDK abstraction**.

The underlying LangGraph gives you the primitives, but you may need to go below CUGA's public SDK boundary — or add your own state/checkpoint layer.

## Short answer

| Requirement | CUGA SDK | Enough for your research? |
| --- | --- | --- |
| Normal logs | ✅ | Yes |
| Tool-call traces | ✅ | Yes |
| Streaming execution | ✅ | Yes |
| Final trajectory | ✅ | Yes |
| Agent graph access | ✅ `agent.graph` | Very useful |
| Exact intermediate state | ⚠️ Indirectly via underlying graph | **Not cleanly exposed by CUGA SDK** |
| Persistent checkpoints | ⚠️ Depends on underlying LangGraph setup | **Need to verify/configure** |
| Retrieve historical state | ⚠️ LangGraph supports it; CUGA doesn't prominently expose it | **Likely need lower-level access** |
| Replay from checkpoint | ⚠️ LangGraph supports this | **Need lower-level graph/checkpointer access** |
| Branch from checkpoint | ⚠️ LangGraph supports state updates/time travel | **Need lower-level access** |
| Counterfactual execution | ❌ Not a CUGA SDK feature | **You build it** |
| Causal analysis | ❌ Not a CUGA SDK feature | **You build it** |
| Exact environment snapshot | ❌ Not automatically guaranteed | **You must build it** |

CUGA explicitly exposes `agent.graph`, which is the underlying compiled LangGraph, and `agent.stream()` exposes execution step-by-step. It also has detailed tool-call tracking. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation+1](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

* * *

# 1. The crucial distinction: trace ≠ state

This is the most important point for your architecture.

Suppose CUGA produces:

```
Run 42

t0  user task
t1  planner generated plan
t2  tool A called
t3  tool A returned X
t4  planner generated next action
t5  tool B called
t6  final answer
```

Normal logging might give you:

```JSON
{
  "tool": "search",
  "arguments": {...},
  "result": "...",
  "duration_ms": 821
}
```

That's useful.

But **counterfactual analysis requires much more**.

You need something closer to:

```
Checkpoint C4
│
├── graph node = planner
├── state =
│     messages = [...]
│     variables = {...}
│     plan = [...]
│     current_agent = ...
│     tool_results = {...}
│     pending_actions = [...]
│
├── execution config
├── model configuration
├── policy state
├── skill version
├── memory version
├── tool registry version
├── random/seed information
└── environment snapshot
```

Then you want:

```
                         C4
                         │
                ┌────────┴────────┐
                │                 │
          factual action      counterfactual
             Tool A              Tool A'
                │                 │
                ▼                 ▼
             State 5'           State 5''
                │                 │
                ▼                 ▼
             Tool B             Tool B'
                │                 │
                ▼                 ▼
             Outcome A         Outcome B
```

**That's the level you need.**

* * *

# 2. CUGA does give you one very important piece

CUGA exposes:

```Python
agent.graph
```

and explicitly describes it as access to the underlying compiled LangGraph. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

That is extremely important.

Because LangGraph itself has a much richer concept of state persistence/checkpoints than the high-level CUGA API exposes.

LangGraph checkpoints represent a snapshot containing things such as:

```
config
metadata
state values
next nodes
pending tasks
```

and LangGraph supports retrieving state/history and replaying from a particular checkpoint. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/catuscio/wiki-as-readme/blob/main/examples/langgraph_readme_en.md?utm_source=chatgpt.com)

So conceptually:

```
                    CugaAgent
                       │
                       ▼
               ┌───────────────┐
               │ compiled graph│
               └───────┬───────┘
                       │
                LangGraph state
                       │
              ┌────────┴────────┐
              ▼                 ▼
         checkpoints         execution
              │                 │
              ▼                 ▼
       historical state       trace
```

This is why I **wouldn't clone CUGA immediately**.

First investigate whether you can access/configure the underlying LangGraph checkpoint machinery through `agent.graph`.

* * *

# 3. LangGraph is much closer to what you need

LangGraph's persistence model supports the concept of a **checkpoint per graph step**.

A checkpoint contains state such as:

```
StateSnapshot
│
├── config
├── metadata
├── values
├── next
└── tasks
```

Historical checkpoints can be retrieved, and a specific checkpoint can be selected using a `checkpoint_id`. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/catuscio/wiki-as-readme/blob/main/examples/langgraph_readme_en.md?utm_source=chatgpt.com)

That means the underlying substrate can support something like:

```Python
state = graph.get_state(
    {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_id": checkpoint_id
        }
    }
)
```

and history:

```Python
graph.get_state_history(...)
```

The LangGraph documentation also describes replaying execution from a checkpoint and updating state before continuing. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub](https://github.com/catuscio/wiki-as-readme/blob/main/examples/langgraph_readme_en.md?utm_source=chatgpt.com)

**That is almost exactly the primitive your counterfactual engine needs.**

* * *

# 4. But here's the catch with CUGA

CUGA's documented Python SDK gives you:

```Python
agent.invoke()
agent.stream()
agent.graph
agent.policies
agent.knowledge
```

and `invoke(..., track_tool_calls=True)` gives you detailed tool-call metadata. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

But I don't see the CUGA SDK documentation presenting first-class methods like:

```Python
agent.get_state()
agent.get_state_history()
agent.get_checkpoint()
agent.replay(checkpoint_id)
agent.branch(checkpoint_id)
agent.update_state()
```

as part of `CugaAgent`.

That's a **major distinction**.

So I would classify it as:

> **CUGA exposes the graph, but does not wrap the full LangGraph time-travel/checkpoint API as a CUGA-level research interface.**

* * *

# 5. And normal CUGA tracing isn't sufficient

CUGA's tool tracking is quite useful:

```Python
result = await agent.invoke(
    task,
    track_tool_calls=True
)
```

You can obtain:

```
tool name
arguments
result
duration
...
```

CUGA explicitly describes this as useful for debugging, observability and auditing. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

But imagine your counterfactual question is:

> "What would have happened if the planner had selected `search_company_filings` instead of `search_web` at step 7?"

A tool-call log alone cannot answer that.

You need:

```
State at step 7
       │
       ├── factual action
       │      ↓
       │   outcome A
       │
       └── alternative action
              ↓
           outcome B
```

Therefore:

**trajectory logging ≠ counterfactual replay.**

* * *

# 6. Your actual research system needs a "state ledger"

I would add this to the architecture we discussed earlier.

```
                         CUGA
                          │
            ┌─────────────┼─────────────┐
            │             │             │
            ▼             ▼             ▼
          Events        State       Tool calls
            │             │             │
            └─────────────┼─────────────┘
                          ▼
                  ┌───────────────┐
                  │ State Ledger  │  ← YOUR COMPONENT
                  └───────┬───────┘
                          │
             ┌────────────┼────────────┐
             ▼            ▼            ▼
        Checkpoint     Trace DB     Environment
           Store                     snapshots
             │
             ▼
      Counterfactual Engine
```

The **State Ledger** is arguably more important for your project than the ordinary trace database.

* * *

# 7. What exactly should you capture?

I'd capture **every meaningful state transition**.

Something like:

```JSON
{
  "run_id": "run_00182",
  "step_id": 17,
  "parent_step_id": 16,

  "checkpoint_id": "cp_17",

  "node": "api_planner",

  "state": {
    "messages": "...",
    "plan": "...",
    "variables": "...",
    "tool_results": "...",
    "pending_actions": "..."
  },

  "input": {
    "task": "..."
  },

  "decision": {
    "action": "search_web",
    "arguments": {
      "query": "IBM CUGA"
    }
  },

  "model": {
    "provider": "...",
    "model": "...",
    "temperature": 0
  },

  "agent_config": {
    "skill_version": "skill_17",
    "memory_version": "memory_23",
    "policy_version": "policy_9"
  },

  "environment": {
    "tool_registry_version": "tools_31"
  },

  "timestamp": "...",

  "outcome": "..."
}
```

Now you have a genuine **causal execution record**.

* * *

# 8. Then counterfactual analysis becomes possible

Suppose:

```
             checkpoint 17
                   │
          factual decision
                   │
             search_web
                   │
                   ▼
               outcome
                   │
                   ▼
                 0.71
```

Your evolutionary engine proposes:

```
"What if we used search_filings?"
```

Then:

```
             checkpoint 17
                   │
          ┌────────┴─────────┐
          │                  │
     factual              altered
   search_web          search_filings
          │                  │
          ▼                  ▼
      trajectory A       trajectory B
          │                  │
          ▼                  ▼
        score .71          score .86
```

You can estimate:

```
Δ = score_counterfactual - score_factual
  = .86 - .71
  = +.15
```

That's much more meaningful than simply saying:

> "The new prompt worked better."

* * *

# 9. You can go even further: intervention graphs

For your research, I would represent an execution as a DAG:

```
                  Task
                   │
                   ▼
                Planner
                   │
          ┌────────┴────────┐
          ▼                 ▼
       Tool A             Tool B
          │                 │
          ▼                 ▼
       Result A          Result B
          │                 │
          └────────┬────────┘
                   ▼
               Planner 2
                   │
                   ▼
               Final Answer
```

Then annotate edges:

```
node
state
action
observation
reward
```

Now you can ask:

### Intervention

```
do(Tool_A = Tool_C)
```

### State intervention

```
do(memory = memory_v17)
```

### Skill intervention

```
do(skill = skill_v22)
```

### Policy intervention

```
do(policy = policy_v8)
```

### Model intervention

```
do(model = model_B)
```

This is where your system starts becoming a **causal agent evaluation framework**, rather than just an agent benchmark.

* * *

# 10. There's another subtle problem: deterministic replay

Even if you capture the exact LangGraph state, you **still don't automatically have exact replay**.

Why?

Because the environment can change.

For example:

```
checkpoint
   │
   ├── LLM
   ├── web
   ├── database
   ├── filesystem
   ├── MCP server
   ├── current time
   ├── random numbers
   └── external APIs
```

If you replay tomorrow:

```
search_web("X")
```

you may get a completely different result.

So you need **environment recording** too.

* * *

# 11. I would therefore build four layers of recording

### Layer 1 — Agent state

```
LangGraph checkpoint
```

### Layer 2 — Agent decision trace

```
LLM input
LLM output
tool selection
arguments
reasoning metadata where legally/technically available
```

### Layer 3 — Tool/environment trace

```
API request
API response
browser observation
filesystem changes
database response
MCP response
```

### Layer 4 — Agent configuration

```
model version
system prompt version
skill version
memory version
policy version
tool registry version
CUGA version
```

Then:

```
                    RUN
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
      State        Events       Config
        │            │            │
        └────────────┼────────────┘
                     ▼
              Reproducible Run
```

* * *

# 12. This is where your own harness becomes essential

CUGA should **not** be responsible for all of this.

I'd make your evolution harness something like:

```
EvolutionHarness
│
├── RuntimeManager
│
├── CugaAdapter
│
├── StateRecorder
│
├── TraceRecorder
│
├── EnvironmentRecorder
│
├── CheckpointManager
│
├── ReplayEngine
│
├── CounterfactualEngine
│
├── CausalAnalyzer
│
├── Evaluator
│
├── MutationEngine
│
├── CandidateSelector
│
└── VersionManager
```

And CUGA is simply:

```
             CugaAdapter
                  │
                  ▼
               CUGA
```

* * *

# 13. Your `CugaAdapter` should expose YOUR interface

For example:

```Python
class AgentRuntime:

    async def run(self, task):
        ...

    async def get_state(self, checkpoint_id):
        ...

    async def get_history(self, run_id):
        ...

    async def replay(self, checkpoint_id):
        ...

    async def branch(self, checkpoint_id, intervention):
        ...
```

Then:

```
                 AgentRuntime
                      │
          ┌───────────┴────────────┐
          ▼                        ▼
     CugaAdapter             Future adapter
          │                   Pi/OpenCode/...
          ▼
        CUGA
```

This is **very important** if you're comparing CUGA vs Pi vs OpenCode.

Your causal/evolution framework shouldn't care which agent harness you're using.

* * *

# 14. And this changes my SDK-vs-clone recommendation

Earlier I said:

> Use CUGA SDK + external evolution harness.

I still recommend that **as the starting point**.

But now I'd make it:

```
                         YOUR HARNESS
                              │
          ┌───────────────────┼────────────────────┐
          │                   │                    │
       Evolution          Replay/Causal       Versioning
          │                   │                    │
          └───────────────────┼────────────────────┘
                              │
                        CUGA Adapter
                              │
                    ┌─────────▼─────────┐
                    │      CUGA          │
                    │                    │
                    │ CugaAgent          │
                    │ agent.graph        │
                    │ invoke             │
                    │ stream             │
                    │ tracking           │
                    └────────────────────┘
                              │
                         LangGraph
                              │
                    checkpoint/state APIs
```

### Start with SDK.

If you discover that the public CUGA SDK **doesn't let you install/configure the checkpoint saver or retrieve the exact historical state you need**, then:

> **Don't immediately fork all of CUGA. Instead, create a very thin source-level adapter around the underlying LangGraph/CUGA internals.**

Only if you need to change CUGA's internal state machine/planner should you maintain a fork.

* * *

# 15. One especially interesting discovery

The current CUGA repository actually advertises a **trajectory visualization/analysis tool**:

```Bash
cuga viz
```

and describes it as visualizing agent execution trajectories, decision-making, and tool usage. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub](https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com)

It also has an experimental **Save & Reuse** capability that captures successful execution paths, including plans, code, and trajectories. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub](https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com)

And CUGA now documents an optional **Evolve integration** that saves completed trajectories and retrieves task-specific guidance. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub](https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com)

That's useful for us because it confirms CUGA is already thinking in terms of:

```
trajectory
   ↓
experience
   ↓
reuse/evolve
```

But **that is still not equivalent to the causal replay infrastructure you're proposing**.

* * *

# 16. So the answer to your original question

### Does CUGA SDK support causal analysis?

**No, not as a first-class feature.**

You would build the causal analysis layer.

### Does CUGA SDK support counterfactual analysis?

**No, not as a first-class feature.**

You need your own intervention + branching + evaluation machinery.

### Does CUGA provide normal traces?

**Yes. Very reasonably.**

Tool-call tracking and streaming are explicitly supported. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

### Does it expose exact agent state?

**Partially.**

The SDK exposes the underlying compiled LangGraph via `agent.graph`, but the CUGA-level documentation does not present the full checkpoint/state-history/time-travel API as a first-class `CugaAgent` interface. [![](https://www.google.com/s2/favicons?domain=https://docs.cuga.dev&sz=128)CUGA Documentation](https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com)

### Does the underlying LangGraph support what you need?

**Yes — much closer.**

LangGraph's persistence/checkpoint model supports state snapshots, historical state, checkpoint IDs, replay and state updates. [![](https://www.google.com/s2/favicons?domain=https://github.com&sz=128)GitHub+1](https://github.com/catuscio/wiki-as-readme/blob/main/examples/langgraph_readme_en.md?utm_source=chatgpt.com)

### Can you therefore build your research system on CUGA?

**Yes.**

But I would make **checkpoint/state capture a first-class component of your own harness**, rather than relying on CUGA's ordinary logging.

* * *

## The architecture I'd now recommend for your project

```
                         ┌───────────────────────────┐
                         │     EVOLUTION ENGINE      │
                         │                           │
                         │ mutation / selection      │
                         │ causal analysis           │
                         │ counterfactual generation │
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │     EXPERIMENT MANAGER     │
                         │                           │
                         │ skills                    │
                         │ memories                  │
                         │ policies                  │
                         │ prompts                   │
                         │ tools                     │
                         └─────────────┬─────────────┘
                                       │
                              ┌────────▼────────┐
                              │  CUGA ADAPTER   │
                              └────────┬────────┘
                                       │
                   ┌───────────────────┼──────────────────┐
                   │                   │                  │
                   ▼                   ▼                  ▼
              CugaAgent           agent.graph        ToolProvider
                   │                   │                  │
                   ▼                   ▼                  ▼
              execution          LangGraph           tools/MCP/API
                   │                   │
                   │            ┌──────┴───────┐
                   │            ▼              ▼
                   │       checkpoints      state history
                   │            │              │
                   └────────────┴──────┬───────┘
                                       ▼
                              ┌──────────────────┐
                              │   STATE LEDGER   │
                              │                  │
                              │ checkpoints      │
                              │ events           │
                              │ tool I/O         │
                              │ configs          │
                              │ environment      │
                              └────────┬─────────┘
                                       │
                         ┌─────────────┴─────────────┐
                         ▼                           ▼
                   FACTUAL RUN                COUNTERFACTUAL
                         │                           │
                         └────────────┬──────────────┘
                                      ▼
                               CAUSAL ANALYZER
                                      │
                                      ▼
                               FITNESS / SCORE
                                      │
                                      ▼
                               EVOLUTION LOOP
```

**This is the architecture I would pursue.**

The key is: **don't treat a CUGA trace as your experimental record. Treat a versioned LangGraph checkpoint + environment snapshot + event trace as the experimental record.** That gives you the substrate necessary to ask genuinely causal questions such as _"Was this improvement caused by the new skill, the new memory, the changed tool-selection policy, or merely a different trajectory?"_