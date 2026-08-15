since cuga support skills , memories , policies , instructions .... all sounds similar , .... can you explain where they differ inside cuga egent ..... where / when what get injected , how they differ ?

Yes — **they look similar because all four ultimately influence what the LLM does**, but inside CUGA they operate at **different layers, times, and scopes**.

The cleanest way to understand them is:

> **Instructions define how an agent component should behave.**
> **Skills teach the agent how to perform a particular kind of task.**
> **Memory/knowledge gives it information from previous experience or stored data.**
> **Policies constrain or modify behavior dynamically at runtime.**

And crucially, **they are not all injected into the same prompt at the same time**.

---

# 1. First, the big picture

Think of a CUGA execution like this:

```text
User Request
     │
     ▼
┌──────────────────────────┐
│     CUGA Runtime         │
│                          │
│  Instructions ─────────┐ │
│  Policies ─────────────┤ │
│  Skills ───────────────┤ │
│  Memory / Knowledge ───┤ │
└────────────────────────┼─┘
                         ▼
                  Planner / Agent
                         │
                         ▼
                      Executor
                         │
                  Tool selection
                         │
                         ▼
                      Tool call
```

But the important thing is that those things enter the execution at **different points**.

---

# 2. Instructions — "How should this component behave?"

CUGA has a **special-instructions system** for its internal components.

The documented instruction sets include components such as:

```text
answer.md
api_planner.md
code_agent.md
plan_controller.md
reflection.md
shortlister.md
task_decomposition.md
```

These are not generic "memories". They are **component-level behavioral instructions**. ([CUGA Documentation][1])

For example:

```text
api_planner.md

You are an API planner.

- Prefer existing APIs
- Decompose complex API workflows
- Verify dependencies
- Avoid redundant calls
```

Conceptually:

```text
                   CUGA
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
      Planner     Executor    Answer
         │          │          │
         ▼          ▼          ▼
 api_planner.md  code_agent   answer.md
```

So an **instruction belongs to a particular CUGA component/node**.

### When is it injected?

When that component becomes active, its corresponding instructions are incorporated into that component's internal prompt. The documentation describes these files as being automatically integrated into the internal prompts when the component is active. ([CUGA Documentation][1])

So:

```text
Task arrives
   ↓
Planner activates
   ↓
planner instructions included
   ↓
LLM call
```

Later:

```text
Executor activates
   ↓
executor/code-agent instructions included
   ↓
LLM call
```

This gives you an important distinction:

> **Instructions are static behavioral configuration for an internal agent component.**

---

# 3. Skills — "How do I perform this particular task?"

Skills are quite different.

CUGA skills are packaged as:

```text
skill-name/
└── SKILL.md
```

with optional helper scripts/assets.

The crucial part is that **the entire skill is NOT placed in the initial prompt**.

Instead, CUGA does:

```text
Startup
   │
   ▼
Scan SKILL.md files
   │
   ▼
Read name + description
   │
   ▼
Short descriptions → system prompt
```

Then, during execution:

```text
User task
   │
   ▼
LLM notices:
"This task looks like the PPTX skill"
   │
   ▼
load_skill("pptx")
   │
   ▼
Full SKILL.md injected
   │
   ▼
Agent follows skill
```

That's explicitly how current CUGA skills work. ([CUGA Documentation][2])

---

# 4. So skills are basically lazy-loaded procedures

This is the easiest mental model:

```text
Instructions
    =
always relevant behavioral rules
```

versus:

```text
Skill
    =
on-demand procedural playbook
```

Example.

### Instruction

```text
Always verify calculations before answering.
```

This is broadly relevant.

### Skill

```text
Financial Analysis Skill

1. Load CSV
2. Normalize columns
3. Calculate revenue
4. Calculate YoY
5. Produce table
6. Check totals
```

You don't want all of that in every request.

So CUGA initially gives the LLM:

```text
Available skill:
Financial Analysis — use when analyzing financial datasets.
```

Only when needed:

```text
load_skill("financial-analysis")
```

and then the full instructions appear. ([CUGA Documentation][2])

---

# 5. Policies — "What am I allowed / required to do right now?"

This is where policies become fundamentally different.

Policies are **runtime behavioral controls**.

CUGA supports several policy types:

```text
Intent Guard
Playbook
Tool Guide
Tool Approval
Output Formatter
```

Policies can trigger from:

* user input
* conversation context
* tool usage

and they can use keyword or semantic triggers. ([CUGA Documentation][3])

So policies are much more **event-driven** than instructions.

---

# 6. Example: Intent Guard

Suppose the user says:

> Delete all customer records.

A policy could be:

```text
Intent Guard:
Prevent destructive database operations.
```

Execution:

```text
User request
    │
    ▼
Policy matching
    │
    ▼
"destructive database operation"
    │
    ▼
Intent Guard fires
    │
    ▼
BLOCK
```

The LLM isn't merely being "advised" not to do it.

The policy layer is an actual **control mechanism in the agent runtime**.

---

# 7. Tool Guide is even more interesting

This is probably the clearest example of why policies ≠ instructions.

Suppose the tool is:

```text
delete_customer(customer_id)
```

Normally its description may be:

```text
Delete a customer.
```

A policy can dynamically modify the tool description:

```text
delete_customer(customer_id)

IMPORTANT:
- Operation is irreversible.
- Requires approval.
- All operations are logged.
```

CUGA's Tool Guide explicitly works by **injecting additional context into tool descriptions at runtime**. ([CUGA Documentation][4])

So the flow becomes:

```text
Tool registry
     │
     ▼
delete_customer()
     │
     ▼
Policy evaluates context
     │
     ▼
Tool Guide applies
     │
     ▼
Modified tool description
     │
     ▼
LLM sees enriched tool
```

This is very different from a normal system instruction.

---

# 8. Playbook — policy + procedure

A Playbook is particularly confusing because it looks very similar to a Skill.

For example:

```text
Refund Playbook

1. Verify customer
2. Verify order
3. Check refund eligibility
4. Obtain approval
5. Issue refund
```

That sounds exactly like a Skill.

But the architectural distinction is:

### Skill

```text
Reusable capability/playbook
loaded on demand by the agent.
```

### Policy Playbook

```text
Policy-controlled workflow
activated when policy conditions are triggered.
```

CUGA explicitly categorizes Playbooks under its policy system. ([CUGA Documentation][3])

So:

```text
Skill:
"Here's how you can do X."

Policy Playbook:
"When situation Y occurs, you must follow workflow X."
```

That distinction is **very important for your self-evolving system**.

---

# 9. Memory / Knowledge — "What information do I know?"

This is different again.

CUGA's current `CugaAgent` exposes a `KnowledgeManager`, with operations such as:

```python
agent.knowledge.ingest(...)
agent.knowledge.search(...)
agent.knowledge.list_documents(...)
```

and knowledge is available through retrieval rather than being the same thing as the agent's behavioral instructions. ([CUGA Documentation][5])

So imagine:

```text
Knowledge base

Customer:
Alice → Enterprise → $1.2M ARR
Bob   → SMB        → $40k ARR
...
```

That's **information**, not behavior.

---

# 10. Memory and knowledge should be distinguished carefully

This is where terminology can get messy.

Historically, CUGA had a `memory_provider` setting around Mem0, but the current settings documentation says the legacy `memory_provider` key was removed from CUGA classic. ([CUGA Documentation][6])

So with **current CUGA**, I'd avoid assuming that "memory" always means a generic persistent autonomous memory subsystem.

Instead think in terms of:

```text
State / thread context
+
Knowledge retrieval
+
saved/reused workflows
```

depending on the feature you're using.

---

# 11. There are actually several kinds of "memory-like" information

This is important for your research.

### A. Conversation/thread state

```text
User: Find customer A.
Agent: ...
User: Now email them.
```

The second turn depends on the first.

That's **state**.

CUGA supports thread-based execution/state via `thread_id`. ([CUGA Documentation][5])

---

### B. Knowledge

```text
Company policy PDF
Customer database
Documentation
Manuals
```

The agent retrieves relevant information.

```text
Question
  ↓
Knowledge search
  ↓
Relevant documents
  ↓
Context
  ↓
LLM
```

That's **retrieval**, not procedural behavior. ([CUGA Documentation][5])

---

### C. Skill

```text
How to make a financial report
How to create a PPTX
How to process a CSV
```

That's **procedural knowledge**.

---

### D. Policy

```text
Never delete data without approval.
Always cite sources.
Don't expose PII.
```

That's **constraint/governance knowledge**.

---

# 12. The easiest way to remember the four

Imagine CUGA is hiring an employee.

### Instructions

> "This is how your job role works."

### Skill

> "Here's the procedure for performing this particular task."

### Knowledge / Memory

> "Here is information you need to know."

### Policy

> "These are rules you must obey."

That's the conceptual difference.

---

# 13. Now let's look at injection timing

This is the really important part.

Here's a simplified CUGA execution:

```text
                         USER
                          │
                          ▼
                ┌───────────────────┐
                │  Thread / State   │
                └─────────┬─────────┘
                          │
                          ▼
                 Policy evaluation
                          │
                          ▼
                  ┌──────────────┐
                  │   Planner    │
                  └──────┬───────┘
                         │
                 Planner instructions
                         │
                         ▼
                   Planning LLM
                         │
                         ▼
                Relevant Skill?
                    │       │
                   NO      YES
                    │       │
                    │       ▼
                    │   load_skill()
                    │       │
                    │       ▼
                    │   Full SKILL.md
                    │       │
                    └───────┘
                         │
                         ▼
                    Executor
                         │
                         ▼
                  Tool selection
                         │
                         ▼
                  Policy evaluation
                         │
                         ▼
             Tool Guide / Approval
                         │
                         ▼
                    Tool call
                         │
                         ▼
                    Observation
                         │
                         ▼
                  Knowledge lookup
                    if needed
                         │
                         ▼
                     LLM
                         │
                         ▼
                    Final answer
```

This diagram captures the **different moments at which these mechanisms participate**.

---

# 14. More precisely: what enters the prompt?

Here's the useful comparison.

| Mechanism         | What it contains               | When used             | Where it influences               |
| ----------------- | ------------------------------ | --------------------- | --------------------------------- |
| **Instructions**  | Behavioral directions          | Component activation  | Internal component prompt         |
| **Skill**         | Detailed procedural knowledge  | When task matches     | Loaded into agent context         |
| **Knowledge**     | Facts/documents                | When retrieved        | Retrieved context                 |
| **Policy**        | Constraints/workflows/controls | When trigger fires    | Runtime, prompt, tools, execution |
| **Thread state**  | Conversation/execution history | Across turns          | Agent context                     |
| **Tool Guide**    | Tool-specific guidance         | When policy applies   | Tool description                  |
| **Tool Approval** | Human gate                     | Before sensitive call | Execution control                 |

This is the distinction you were looking for.

---

# 15. The subtle distinction: prompt injection vs runtime control

Not everything is "injected into the prompt."

That's an easy mistake to make.

### Instructions

Mostly:

```text
prompt construction
```

### Skills

Mostly:

```text
dynamic context loading
```

### Knowledge

Mostly:

```text
retrieval → context
```

### Policy

Can be:

```text
prompt/context modification
+
tool modification
+
execution blocking
+
human approval
+
output transformation
```

That's why policy is architecturally stronger.

---

# 16. Example: same information expressed four ways

Suppose your company has a rule:

> "Refunds above $1,000 require manager approval."

You could encode related information in four different ways.

### As instruction

```text
Always consider refund approval requirements.
```

This merely tells the model what to think about.

### As skill

```text
Refund Processing Skill

1. Check refund amount
2. Check approval threshold
3. Request approval if needed
4. Process refund
```

This describes **how to execute**.

### As knowledge

```text
Refund policy document:
Refunds > $1,000 require manager approval.
```

The model retrieves it as factual information.

### As policy

```text
If refund > $1,000:
require human approval before execute_refund().
```

Now CUGA can actually enforce the workflow.

That is the key distinction.

---

# 17. Where this matters for your self-evolving agent

This is actually **very important for how you design your evolutionary search space**.

Don't treat everything as one giant "prompt."

Instead define:

```text
              CUGA HARNESS
                   │
     ┌─────────────┼─────────────┐
     │             │             │
 Instructions    Skills        Policies
     │             │             │
 behavior       procedures    constraints
     │             │             │
     └─────────────┼─────────────┘
                   │
              CUGA runtime
                   │
                   ▼
              Trajectory
```

Then your evolutionary algorithm can ask:

```text
What should evolve?
```

Maybe:

```text
Planner instructions
Executor instructions
Skill content
Skill selection descriptions
Policy rules
Tool guides
```

while **keeping knowledge and benchmark data fixed**.

That gives you a much more scientifically meaningful experiment.

---

# 18. A particularly important distinction for your baseline

Suppose CUGA gets:

```text
10 skills
20 policies
100 documents
```

You should not simply say:

> "CUGA has a huge prompt."

Because these aren't all present simultaneously.

A more accurate model is:

```text
STARTUP
│
├── Instructions → configured into components
│
├── Skill metadata → lightweight summaries
│
└── Policies → loaded/indexed
      │
      ▼
TASK
│
├── relevant policy triggers?
│
├── relevant skill?
│       │
│       └── load_skill()
│
├── relevant knowledge?
│       │
│       └── retrieve()
│
└── active component?
        │
        └── corresponding instructions
                │
                ▼
             LLM CALL
```

That makes their roles much clearer.

---

# 19. One more subtle distinction: "Skill selection" vs "Policy triggering"

They can look almost identical:

```text
Task:
"Create a PowerPoint"
```

Skill system:

```text
Description matching
       ↓
pptx skill
       ↓
load_skill()
```

Policy system:

```text
Policy trigger matching
       ↓
"presentation creation policy"
       ↓
policy activates
```

But the semantics are different:

```text
Skill selection:
"What capability should I load?"

Policy triggering:
"What rule/workflow should apply?"
```

This distinction is extremely useful when analyzing trajectories.

---

# 20. My mental model for CUGA

I'd summarize the architecture as:

```text
                  ┌──────────────────────┐
                  │      INSTRUCTIONS    │
                  │                      │
                  │ "BEHAVE LIKE THIS"  │
                  └──────────┬───────────┘
                             │
                             ▼
                    Agent components
                             │
                             ▼
                     ┌─────────────┐
                     │   AGENT     │
                     └──────┬──────┘
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
       SKILLS            KNOWLEDGE         POLICIES
   "HOW TO DO IT"     "WHAT IS TRUE"    "WHAT IS ALLOWED"
          │                 │                 │
          ▼                 ▼                 ▼
       load_skill()      retrieve()       trigger()
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
                       EXECUTION
                            │
                            ▼
                          TOOLS
```

And **policies are the odd one out** because they don't merely provide knowledge to the model; they can affect the runtime itself — including modifying tool descriptions, blocking actions, requiring approval, and formatting outputs. ([CUGA Documentation][3])

---

## The one-line distinction

> **Instructions shape the agent's general behavior; Skills provide on-demand procedures; Knowledge/Memory provides information; Policies dynamically govern what behavior is permitted or required.**

For your self-evolution work, I'd therefore treat them as **four separate mutation surfaces**, not one shared "prompt" surface. That separation will make your ablation study much cleaner.

[1]: https://docs.cuga.dev/docs/old-configuration/special-instructions/?utm_source=chatgpt.com "Special Instructions Configuration | CUGA AGENT"
[2]: https://docs.cuga.dev/docs/guides/agent-skills/?utm_source=chatgpt.com "Agent Skills | CUGA AGENT"
[3]: https://docs.cuga.dev/docs/sdk/policies/?utm_source=chatgpt.com "Overview | CUGA AGENT"
[4]: https://docs.cuga.dev/docs/sdk/policies/tool-guide/?utm_source=chatgpt.com "Tool Guide | CUGA AGENT"
[5]: https://docs.cuga.dev/docs/sdk/cuga_agent/?utm_source=chatgpt.com "CugaAgent | CUGA AGENT"
[6]: https://docs.cuga.dev/docs/customization/settings-reference/?utm_source=chatgpt.com "Settings Reference | CUGA AGENT"
