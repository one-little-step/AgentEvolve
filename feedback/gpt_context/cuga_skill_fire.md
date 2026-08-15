can you explain me what make a cuga agent skill's fire ? , ...ie., when and how a skill is loaded into context , how this loading is triggered...?

Yes. I checked the **current CUGA Agent Skills implementation/docs**, and the mechanism is quite clean. The key idea is **progressive disclosure**:

> **CUGA does NOT put the full contents of every `SKILL.md` into the context initially. It first advertises a compact skill catalog, and the LLM itself decides when to call `load_skill`.** ([CUGA Documentation][1])

This distinction is especially important for your self-evolution work.

---

# 1. The entire skill lifecycle

Think of a CUGA skill as going through **4 stages**:

```text
        Filesystem
            │
            │ startup
            ▼
     ┌──────────────┐
     │   DISCOVERY  │
     │              │
     │ scan SKILL.md│
     └──────┬───────┘
            │
            │ name + description
            ▼
     ┌──────────────┐
     │  ADVERTISE   │
     │              │
     │ system prompt│
     └──────┬───────┘
            │
            │ model recognizes relevance
            ▼
     ┌──────────────┐
     │  LOAD_SKILL  │◄──── LLM tool call
     │              │
     │ full SKILL.md│
     └──────┬───────┘
            │
            ▼
     ┌──────────────┐
     │   EXECUTE    │
     │              │
     │ follow skill │
     │ + tools      │
     │ + resources  │
     └──────────────┘
```

CUGA's documentation explicitly describes these stages as **Discovery → Prompt → `load_skill` → Sandbox copy → Execution**. ([CUGA Documentation][1])

---

# 2. Stage 1 — CUGA discovers the skill

Suppose you have:

```text
.cuga/
└── skills/
    ├── pptx/
    │   ├── SKILL.md
    │   ├── scripts/
    │   └── references/
    │
    └── data-analysis/
        └── SKILL.md
```

CUGA starts up.

It scans the configured skill root recursively for:

```text
**/SKILL.md
```

The default current configuration is:

```toml
[skills]
enabled = true
root = "cuga"
```

where `cuga` maps to:

```text
.cuga/skills/
```

Other supported roots include `.agents/skills`, global agent skills, and the legacy global CUGA path. ([CUGA Documentation][1])

---

# 3. It does NOT load the whole SKILL.md

This is the important part.

Suppose:

```text
pptx/SKILL.md
```

contains:

```yaml
---
name: pptx
description: Create and manipulate PowerPoint presentations.
---
```

followed by **5,000 tokens** of detailed instructions.

CUGA does **not** initially inject those 5,000 tokens.

Instead it extracts the frontmatter:

```text
name:
    pptx

description:
    Create and manipulate PowerPoint presentations.
```

and creates a compact catalog.

The documentation explicitly says:

> CUGA discovers skills at startup, reads `name` + `description`, and lists short descriptions in the agent prompt; the full body is not inlined. ([CUGA Documentation][1])

---

# 4. So what is actually put into the LLM context?

Conceptually something like:

```text
SYSTEM PROMPT
│
├── CUGA instructions
│
├── tool descriptions
│
├── policies
│
└── AVAILABLE SKILLS
      │
      ├── pptx
      │    "Create and manipulate PowerPoint presentations."
      │
      ├── data-analysis
      │    "Analyze datasets and generate reports."
      │
      └── web-research
           "Perform structured web research."
```

The **full skill bodies aren't there yet**.

This is why CUGA can have many skills without exploding the context window. ([CUGA Documentation][1])

---

# 5. This is where the "trigger" happens

This is perhaps the most important thing to understand:

## The skill description itself doesn't execute anything.

There isn't necessarily a deterministic rule like:

```python
if "PowerPoint" in user_message:
    load_skill("pptx")
```

Instead, the **LLM sees the skill catalog** and decides:

> "This task requires the PPTX skill."

Then the model generates a tool call:

```text
tool_call:
    name = "load_skill"
    arguments = {
        "skill_name": "pptx"
    }
```

CUGA's docs explicitly describe this as:

> the model calls the `load_skill` tool when the task matches a skill description. ([CUGA Documentation][1])

So the triggering chain is:

```text
User request
     │
     ▼
LLM sees skill catalog
     │
     │ reasoning
     ▼
"This looks like PPTX work"
     │
     ▼
LLM emits tool call
     │
     ▼
load_skill("pptx")
```

**The LLM is the skill router.**

---

# 6. What exactly does `load_skill` do?

Suppose the model calls:

```text
load_skill("pptx")
```

CUGA resolves:

```text
pptx
  ↓
.cuga/skills/pptx/SKILL.md
```

and reads the **full markdown body**.

For example:

```markdown
---
name: pptx
description: Create and manipulate PowerPoint presentations.
---

# PPTX Skill

When creating presentations:

1. Inspect the input data.
2. Determine slide structure.
3. Use the provided helper scripts.
4. Validate the generated presentation.
5. ...
```

The full body is then returned/injected into the agent's context. CUGA documents this as:

> "`load_skill` — Model calls the tool when the task matches a skill description; full markdown body is injected." ([CUGA Documentation][1])

---

# 7. So the context changes dynamically

Before loading:

```text
┌─────────────────────────────────────┐
│ SYSTEM PROMPT                       │
│                                     │
│ CUGA instructions                   │
│ tools                               │
│ policies                            │
│                                     │
│ Available Skills:                   │
│                                     │
│ pptx → Create presentations         │
│ data → Analyze datasets             │
│ web → Perform web research          │
│                                     │
│ ❌ full skill bodies                │
└─────────────────────────────────────┘
```

After:

```text
┌─────────────────────────────────────┐
│ SYSTEM PROMPT                       │
│                                     │
│ CUGA instructions                   │
│ tools                               │
│ policies                            │
│                                     │
│ Available Skills: ...               │
│                                     │
│ ┌─────────────────────────────────┐ │
│ │ FULL PPTX SKILL                 │ │
│ │                                 │ │
│ │ # PPTX Skill                   │ │
│ │                                 │ │
│ │ detailed workflow...            │ │
│ │ rules...                        │ │
│ │ scripts...                      │ │
│ │ validation...                   │ │
│ └─────────────────────────────────┘ │
└─────────────────────────────────────┘
```

So **skill loading is a context mutation during execution**.

---

# 8. The important thing: `load_skill` is itself a tool

This is a very useful mental model.

CUGA effectively gives the LLM something like:

```text
Tools:

search(...)
write_file(...)
run_command(...)
load_skill(...)
...
```

And:

```text
load_skill(...)
```

is not a normal business-domain tool.

It's a **meta-tool**.

It changes what the agent knows/how it should behave.

So:

```text
Normal tool:

search_customer()
       │
       ▼
   external world
```

whereas:

```text
load_skill()
       │
       ▼
agent context
       │
       ▼
new instructions
```

That's a very different category.

---

# 9. Skill loading is therefore a two-level reasoning process

You can think of it as:

```text
                 USER TASK
                     │
                     ▼
              ┌─────────────┐
              │ LLM         │
              │             │
              │ "What do I  │
              │ need?"      │
              └──────┬──────┘
                     │
               skill relevant?
                     │
              ┌──────┴──────┐
              │             │
             NO            YES
              │             │
              ▼             ▼
          continue      load_skill
                            │
                            ▼
                     skill instructions
                            │
                            ▼
                         LLM again
                            │
                            ▼
                         execute
```

So the skill itself doesn't "fire."

More accurately:

> **The model chooses to invoke the skill-loading tool because the advertised skill metadata indicates that the skill is relevant.**

---

# 10. Example: PPTX task

Suppose the user says:

> "Create a 10-slide presentation about our Q4 sales."

Initially CUGA has:

```text
Available skills:

pptx:
Create and manipulate PowerPoint presentations.

data-analysis:
Analyze datasets and generate reports.

web-research:
Perform structured web research.
```

The LLM reasons:

```text
User wants presentation
        ↓
pptx description matches
        ↓
load_skill("pptx")
```

Then:

```text
load_skill("pptx")
        │
        ▼
SKILL.md loaded
        │
        ▼
LLM now knows:
  - how to construct slides
  - what scripts exist
  - formatting rules
  - validation procedure
        │
        ▼
run_command(...)
        │
        ▼
create PPTX
```

CUGA's own example explicitly tells the model to create a sales deck, and says CUGA should call `load_skill("pptx")` and then use the bundled scripts from `/workspace/skills/pptx/`. ([CUGA Documentation][1])

---

# 11. What happens to scripts and other files?

This is another subtle distinction.

Suppose:

```text
pptx/
├── SKILL.md
├── scripts/
│   ├── generate.py
│   └── validate.py
└── references/
    └── style-guide.md
```

`load_skill("pptx")` primarily loads:

```text
SKILL.md
```

But CUGA also makes the **skill directory available inside the sandbox**:

```text
/workspace/skills/pptx/
```

according to the current docs. ([CUGA Documentation][1])

So the agent can subsequently execute:

```text
run_command
     │
     ▼
/workspace/skills/pptx/scripts/generate.py
```

The scripts themselves don't need to be dumped into the LLM context.

---

# 12. So there are actually three levels of loading

This is **very relevant to your research**.

### Level 0 — Discovery

```text
Filesystem
   ↓
SKILL.md frontmatter
   ↓
name + description
```

Tiny context cost.

---

### Level 1 — Skill activation

```text
load_skill("pptx")
       ↓
full SKILL.md
       ↓
LLM context
```

Potentially thousands of tokens.

---

### Level 2 — Resource/script loading

```text
SKILL.md
   │
   ├── reference.md
   ├── schema.json
   └── generate.py
          │
          ▼
     loaded/executed
     only if necessary
```

CUGA copies the skill folder into the sandbox, while the actual execution of helper scripts/resources happens later through tools. ([CUGA Documentation][1])

So:

```text
             SKILL
              │
       ┌──────┴───────┐
       │              │
     SKILL.md       resources
       │              │
       ▼              ▼
    context       on-demand
                     access
```

---

# 13. What actually determines whether a skill gets loaded?

There are **three important factors**.

### ① The skill must be discovered

If CUGA doesn't find:

```text
SKILL.md
```

there is nothing to load.

---

### ② Its `name` and `description` must be advertised

For example:

```yaml
---
name: database-migration
description: Safely create and execute database schema migrations.
---
```

The model sees that description.

---

### ③ The LLM must decide that it is relevant

This is the actual "trigger":

```text
user task
   +
skill descriptions
   ↓
LLM reasoning
   ↓
load_skill(skill_name)
```

There isn't necessarily a deterministic keyword matcher.

This is **LLM-driven routing**.

---

# 14. This means skill descriptions are VERY important

Suppose you write:

```yaml
description: Helps with PowerPoint.
```

That's weak.

The model has little information about **when it should activate**.

Compare:

```yaml
description: >
  Create, edit, and validate PowerPoint presentations.
  Use when the user asks to create or modify PPTX files,
  presentations, slide decks, or presentation slides.
```

Now the model has a much better routing signal.

So the skill description isn't merely documentation.

It is effectively:

> **a routing hint supplied to the LLM.**

That's a very important design point for your self-evolution work.

---

# 15. This also means a skill can fail to fire

For example:

```text
Skill:
database-migration

description:
"Safely migrate relational databases."
```

User:

> "Upgrade our PostgreSQL schema."

The LLM might correctly infer:

```text
PostgreSQL schema upgrade
        ↓
database-migration
        ↓
load_skill()
```

But if your description is:

```text
"Database utilities"
```

the skill might not be selected.

Therefore:

```text
Skill activation probability
      ↑
      │
      ├── description quality
      ├── task wording
      ├── competing skills
      ├── system prompt
      └── model reasoning
```

---

# 16. This is especially interesting for your self-evolving agent

You now have a **new evolvable parameter**.

You could evolve:

```text
SKILL.md
   │
   ├── name
   ├── description      ← routing behavior
   │
   └── instructions     ← execution behavior
```

Notice that there are actually **two separate optimization targets**.

### Routing optimization

> Does the correct skill get activated?

```text
task
 ↓
skill description
 ↓
load_skill?
```

### Execution optimization

Once loaded:

```text
SKILL.md
 ↓
agent behavior
 ↓
tool calls
 ↓
result
```

This gives you an interesting decomposition:

```text
                 Skill
                   │
        ┌──────────┴──────────┐
        │                     │
   ROUTING PART          EXECUTION PART
        │                     │
   description             body
        │                     │
        ▼                     ▼
 "Should I load?"       "How should I act?"
```

That is **very useful experimentally**.

---

# 17. You can even have skill chains

Suppose:

```text
User:
"Analyze this sales CSV and create a PowerPoint."
```

Potentially:

```text
                    USER
                      │
                      ▼
                    LLM
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
    load_skill(data)      load_skill(pptx)
            │                   │
            ▼                   ▼
      data instructions    pptx instructions
            │                   │
            └─────────┬─────────┘
                      ▼
                   execute
```

The model can load multiple skills when appropriate.

The CUGA documentation describes skills as reusable instruction packs and the `load_skill` mechanism as on-demand loading; it does not impose a single-skill-per-run restriction. ([CUGA Documentation][1])

---

# 18. And this is different from CUGA's normal node instructions

This distinction is **very important**.

CUGA has internal instructions like:

```text
configurations/instructions/
├── api_planner.md
├── code_agent.md
├── plan_controller.md
├── task_decomposition.md
└── ...
```

Those are **CUGA node-level instructions** and are automatically integrated when the corresponding component is active. ([GitHub][2])

Skills are different:

```text
CUGA internal instructions
        │
        ▼
deterministic component configuration
```

versus:

```text
Agent Skill
        │
        ▼
LLM decides whether to load
```

So:

| Mechanism            | Who triggers it?         | When?                    |
| -------------------- | ------------------------ | ------------------------ |
| `api_planner.md`     | CUGA runtime             | API planner executes     |
| `plan_controller.md` | CUGA runtime             | plan controller executes |
| Skill metadata       | CUGA startup             | skill discovery          |
| Full `SKILL.md`      | **LLM via `load_skill`** | when relevant            |
| Skill script         | **LLM via tool**         | when needed              |

That's a crucial distinction.

---

# 19. The complete CUGA skill flow

Putting everything together:

```text
                         START CUGA
                             │
                             ▼
                    Scan skills directory
                             │
                             ▼
                     Find SKILL.md files
                             │
                             ▼
                    Parse frontmatter
                      /            \
                     /              \
                  name          description
                    \              /
                     \            /
                      ▼          ▼
                     Skill Catalog
                           │
                           ▼
                     Inject into prompt
                           │
                           ▼
                      ┌──────────┐
                      │   LLM    │
                      └────┬─────┘
                           │
                     user task arrives
                           │
                           ▼
                  "Is a skill relevant?"
                           │
                    ┌──────┴──────┐
                    │             │
                   NO            YES
                    │             │
                    │             ▼
                    │       load_skill("X")
                    │             │
                    │             ▼
                    │        read SKILL.md
                    │             │
                    │             ▼
                    │       inject full body
                    │             │
                    └──────┬──────┘
                           ▼
                      continue LLM
                           │
                           ▼
                    follow skill
                           │
               ┌───────────┼───────────┐
               ▼           ▼           ▼
             tools       scripts    resources
               │           │           │
               └───────────┼───────────┘
                           ▼
                         result
```

---

# 20. The most important takeaway for your research

I would model CUGA's skill system as:

```text
             ┌─────────────────────────────┐
             │        SKILL CATALOG        │
             │                             │
             │ name + description          │
             └──────────────┬──────────────┘
                            │
                            ▼
                       LLM ROUTER
                            │
                     ┌──────┴──────┐
                     │             │
                  no skill       skill X
                     │             │
                     │             ▼
                     │      load_skill(X)
                     │             │
                     │             ▼
                     │       FULL SKILL
                     │             │
                     │             ▼
                     │        LLM EXECUTION
                     │             │
                     └─────────────┤
                                   ▼
                              TOOL ACTIONS
```

So **the skill itself does not autonomously "fire."**

The precise mechanism is:

> **CUGA discovers the skill → advertises its metadata → the LLM sees the metadata → the LLM decides the task requires the skill → the LLM calls `load_skill` → CUGA reads/injects the full `SKILL.md` → execution continues with those instructions available.** ([CUGA Documentation][1])

And for your self-evolution work, I'd pay particular attention to the **skill description as a routing policy**. You can separately measure:

```text
                    Skill evolution
                          │
             ┌────────────┴────────────┐
             ▼                         ▼
      Activation accuracy       Execution quality
      ──────────────────        ─────────────────
      "Did correct skill        "Did loaded skill
       fire?"                    produce good behavior?"
             │                         │
             ▼                         ▼
       description                SKILL.md body
```

That separation could give you a surprisingly clean experiment: **evolve only skill routing descriptions first, then evolve skill bodies, then evolve both**, while keeping the underlying CUGA runtime unchanged.

[1]: https://docs.cuga.dev/docs/guides/agent-skills/?utm_source=chatgpt.com "Agent Skills | CUGA AGENT"
[2]: https://github.com/cuga-project/cuga-agent?utm_source=chatgpt.com "GitHub - cuga-project/cuga-agent: CUGA is an open-source generalist agent harness for the enterprise, supporting complex task execution on web and APIs, OpenAPI/MCP integrations, composable architecture, reasoning modes, and policy-aware features. · GitHub"
