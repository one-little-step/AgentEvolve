# System Architecture — AgentEvolve

**What this file is.** The complete system, end to end, in diagrams: how an agent's
harness is evolved, how an issue is born and dies, and how the LLM layer is
intercepted so a *builder* can transfer wanted behaviour into a complex agent it
does not control.

**Companion files.** For *"is that line actually wired?"* read
[`IMPLEMENTED-PIPELINE-MAP.md`](IMPLEMENTED-PIPELINE-MAP.md), which anchors every
claim to `file:line`. For the defects that make a number untrustworthy read
[`../SEVERE-OPEN-ISSUES.md`](../SEVERE-OPEN-ISSUES.md). This file is the shape of
the system; those two are its errata.

**Honesty marker used throughout.** Green = LIVE, amber = GATED or partial,
red = DEAD/ABSENT. Nothing here is aspirational unless the box says so.

---

## 1. The thesis in one diagram

An agent fails a task. Something in its *harness* — the instructions, skills,
policies and memory it carries — is responsible. Find which, edit it, prove the
edit helped, keep the evidence.

```mermaid
flowchart LR
    subgraph IN["the harness — what we evolve"]
        I["instructions"]:::art
        S["skills/"]:::art
        P["policies/"]:::art
        M["memory/"]:::art
    end
    AG["the agent<br/>(CUGA, via SDK)"]:::ext
    T["task"]:::ext
    TR["trace<br/>events · tool calls · answer"]:::ev
    SC["score<br/>(grader)"]:::ev
    AN["causal analyzer<br/>WHY did it fail"]:::live
    ED["editor agent<br/>rewrite the blamed artifact"]:::live
    PO["persistent pool<br/>every candidate + its evidence"]:::live

    IN --> AG
    T --> AG
    AG --> TR --> SC
    TR --> AN
    AN -->|"mechanism + blamed artifacts"| ED
    ED -->|"new candidate"| PO
    PO -->|"breed from a parent"| IN

    classDef art fill:#cfe8ff,stroke:#036,color:#000
    classDef ext fill:#eee,stroke:#666,color:#000
    classDef ev fill:#fff4cc,stroke:#a70,color:#000
    classDef live fill:#0b6,stroke:#053,color:#fff
```

**Why this is not prompt engineering.** The edit is *attributed*: an issue only
exists if the analyzer's blame lands on an artifact the adapter declares writable.
The single most consequential line in the system, `issues.py:186-188`:

```python
write_set = tuple(sorted(aid for aid in attributed if aid in writable_ids))
if not write_set:
    return None          # issues.py:188 — no writable attribution, no issue
```

So the surface offered to the editor is **analyzer blame ∩ declared-writable**,
never "whatever the harness happens to contain."

---

## 2. Layered architecture, and the boundary that makes it agent-neutral

```mermaid
flowchart TB
    subgraph L0["entry"]
        CLI["scripts/run_evolution.py<br/>--mode --profile --iterations"]:::live
    end
    subgraph L1["the ONLY wiring seam"]
        PIPE["pipeline.py<br/>build_live_stack :1199<br/>build_rho_hooks :1478"]:::live
    end
    subgraph L2["core/ — agent-neutral · 35 files · 0 forbidden imports"]
        ORC["orchestrator.py<br/>SequentialGepaRunner :1022"]:::live
        RHO["rho/rounds.py<br/>10 phases, 17 injected hooks"]:::live
        POOL["pool.py<br/>score tensor · Pareto · champion"]:::live
        ISS["issues.py<br/>build · DPP select"]:::live
        ENT["entropy.py<br/>H(t,m) + evidence floors"]:::live
        CLU["clustering.py<br/>mechanism identity"]:::live
        MRG["merge.py — crossover<br/>0 importers"]:::dead
        PAR["parallel.py<br/>test-only"]:::gated
    end
    subgraph L3["adapters/ — the only place core meets a real agent"]
        AN["cuga_analyzer"]:::live
        ED["cuga_editor"]:::live
        JG["cuga_preference_judge"]:::live
        OPT["cuga_rho_optimizer"]:::live
        ADJ["cuga_mechanism_adjudicator"]:::live
    end
    subgraph L4["the agent + the model"]
        CUGA["CUGA SDK 0.2.20"]:::ext
        LLM["LiteLLM gateway -> models"]:::ext
    end

    CLI --> PIPE --> ORC & RHO
    ORC --> POOL & ISS & ENT
    ISS --> CLU
    PIPE --> AN & ED & JG & OPT & ADJ
    AN & ED & JG & OPT & ADJ --> CUGA --> LLM

    classDef live fill:#0b6,stroke:#053,color:#fff
    classDef dead fill:#e55,stroke:#900,color:#fff
    classDef gated fill:#fd7,stroke:#a70,color:#000
    classDef ext fill:#eee,stroke:#666,color:#000
```

**The boundary is enforced, not aspirational.** `core/` must not import `cuga`,
`litellm`, `openai`, `httpx`, `requests`, or `agent_evolve.adapters` — currently
**35 files, 0 violations**, verified by AST (a substring grep gives false
positives; several `core/` docstrings mention adapters in prose).

**Why it matters commercially:** the evolution engine is not a CUGA product. Swap
the adapter package and the same engine evolves a different agent. CUGA is the
reference adapter because the research needs exact state tracing, not because the
core depends on it.

---

## 3. Modes — data, not branches

`PHASES` in `core/rho/rounds.py` maps a mode name to an ordered phase tuple.

```mermaid
flowchart TB
    subgraph RHOP["mode rho — 10 phases, cold-start safe"]
        direction LR
        R1["1 history<br/>load"]:::live --> R2["2 trajectory<br/>comprehension"]:::live --> R3["3 difficulty<br/>fingerprint"]:::live
        R3 --> R4["4 coreset<br/>DPP"]:::live --> R5["5 group<br/>rollouts"]:::live --> R6["6 group<br/>diagnosis"]:::live
        R6 --> R7["7 N candidate<br/>proposals"]:::live --> R8["8 candidate<br/>rollouts"]:::live
        R8 --> R9["9 pairwise<br/>judging"]:::live --> R10["10 pool<br/>commit ALL N"]:::live
    end
    subgraph GEN["mode genetic — the attempt loop"]
        direction LR
        G1["observe<br/>parent"]:::live --> G2["build<br/>issues"]:::live --> G3["select<br/>issue"]:::live
        G3 --> G4["propose<br/>edit"]:::live --> G5["validate"]:::live --> G6["commit +<br/>retire"]:::live
    end
    R10 -->|"mode rho-genetic"| G1
    classDef live fill:#0b6,stroke:#053,color:#fff
```

Cost model, RHO: **rollouts per round = `k × (G + N×R)`**. Paper defaults
(k=10, G=3, N=3, R=2) → **90 per round**. This dominates a run; budget it first.

---

## 4. The genetic attempt loop, in detail

```mermaid
sequenceDiagram
    participant R as SequentialGepaRunner
    participant A as Adapter (CUGA)
    participant N as Analyzer
    participant E as Editor agent
    participant P as Persistent pool

    R->>A: rollout_group(parent, tasks)
    A-->>R: traces
    R->>R: score (grader)
    Note over R,N: only ANSWERED FAILURES are analyzed<br/>orchestrator.py:1401
    R->>N: analyze(failing traces)
    N-->>R: mechanism · blame graph · severity
    R->>R: build_issues -> write_set = blame ∩ writable
    R->>R: select_issues (DPP: quality × diversity)
    R->>R: select_parent (frequency-proportional)
    R->>E: EditorRequest(issue, parent faults, donors, history)
    E-->>R: staged edits (one artifact or more)
    R->>A: apply + materialize
    R->>A: validate: origin probe + regression probes
    A-->>R: scores
    alt accepted
        R->>P: commit candidate + score cells
        R->>P: maybe soft-retire the parent (judge decides)
    else rejected
        R->>R: record in edit memory (failures are the useful history)
    end
```

**Two design properties worth naming.**

- **Rejections are recorded too.** The point of history is *"do not repeat a
  strategy that already failed"*, so failed attempts are the load-bearing entries.
- **Retirement is soft.** When the pairwise judge prefers a child over its parent,
  the parent leaves *breeding* — parent sampling, Pareto frontier, champion
  selection — but keeps every score cell, lineage link and preference record. Hard
  deletion would destroy the comparable cells cross-candidate entropy needs.

---

## 5. Issue lifecycle — birth to death

This is the heart of the method. An *issue* is a trace-backed, attributed,
selectable unit of work.

```mermaid
flowchart TB
    RO["rollout"]:::ev
    PASS{"scored<br/>and passed?"}:::dec
    NOAN{"produced<br/>an answer?"}:::dec
    DROP1["DROPPED — unscorable<br/>a broken harness is NOT a wrong answer"]:::drop
    NOISS["no issue<br/>(nothing to fix)"]:::drop
    ANL["ANALYZER<br/>mechanism · blame graph · severity · confidence"]:::live
    CLU["MECHANISM CLUSTERING<br/>cosine pre-filter -> dedup LLM in the ambiguous band"]:::live
    ID{"cluster id<br/>assigned?"}:::dec
    UNAV["entropy UNAVAILABLE<br/>reason recorded, never substituted"]:::gated
    ATTR["ATTRIBUTION<br/>blamed artifacts ∩ declared-writable"]:::live
    WS{"write_set<br/>empty?"}:::dec
    DROP2["DROPPED — issues.py:188<br/>no writable surface = not actionable"]:::drop
    ISSUE["ISSUE born<br/>severity · confidence · entropy · coverage · pareto"]:::live
    DPP["DPP SELECTION<br/>quality × diversity, theta"]:::live
    EDIT["EDITOR acts on it"]:::live
    VAL{"validation<br/>net gain > 0?"}:::dec
    COMMIT["candidate committed<br/>issue RESOLVED"]:::live
    RETRY["edit memory + retry budget<br/>3 per (issue, artifacts, lineage)"]:::gated
    DEAD["issue ABANDONED<br/>budget exhausted"]:::drop

    RO --> NOAN
    NOAN -->|no| DROP1
    NOAN -->|yes| PASS
    PASS -->|yes| NOISS
    PASS -->|no| ANL --> CLU --> ID
    ID -->|no| UNAV
    ID -->|yes| ATTR
    UNAV -.->|"issue still built,<br/>entropy term = 0"| ATTR
    ATTR --> WS
    WS -->|yes| DROP2
    WS -->|no| ISSUE --> DPP --> EDIT --> VAL
    VAL -->|yes| COMMIT
    VAL -->|no| RETRY --> DPP
    RETRY -->|exhausted| DEAD

    classDef live fill:#0b6,stroke:#053,color:#fff
    classDef ev fill:#fff4cc,stroke:#a70,color:#000
    classDef dec fill:#cfe8ff,stroke:#036,color:#000
    classDef drop fill:#e55,stroke:#900,color:#fff
    classDef gated fill:#fd7,stroke:#a70,color:#000
```

### 5.1 Why mechanism identity is the hard part

Two analyzer descriptions of *the same* fault must land in the same cluster, or
their evidence fragments and variance becomes meaningless. **Cosine alone provably
cannot do this.** Measured over 66 live pairs across 4 fault families:

| | range | mean |
| --- | --- | --- |
| same fault | 0.466 – 0.851 | **0.728** |
| different fault | 0.244 – 0.502 | **0.393** |

The distributions **overlap** — separation `-0.036`. No single threshold separates
analyzer paraphrase from a genuinely different fault. Hence:

```mermaid
flowchart LR
    M["new mechanism<br/>description"]:::ev --> E["embed (768-dim)"]:::live
    E --> C{"cosine vs<br/>nearest cluster"}:::dec
    C -->|"< 0.45"| SEP["DISTINCT — free"]:::live
    C -->|"0.45 .. 0.75"| ADJ["ask the DEDUP LLM<br/>load-bearing, not an optimisation"]:::live
    C -->|">= 0.75"| JOIN["SAME — free"]:::live
    ADJ --> V{"verdict"}:::dec
    V -->|same| JOIN
    V -->|different| SEP
    V -->|unavailable| FALL["cosine stands<br/>reason recorded"]:::gated
    classDef live fill:#0b6,stroke:#053,color:#fff
    classDef ev fill:#fff4cc,stroke:#a70,color:#000
    classDef dec fill:#cfe8ff,stroke:#036,color:#000
    classDef gated fill:#fd7,stroke:#a70,color:#000
```

The shipped band `[0.45, 0.75)` is the smallest measured band that silently splits
**zero** true paraphrase pairs; the previous `0.60–0.85` split 2 of 12. A live
adjudicator probe in the newly reached `0.45–0.60` range scored **12/12** correct.

### 5.2 Two key policies that must stay separate

A subtle, load-bearing asymmetry. The same `(task, mechanism)` pair is keyed
**differently** in two structures because they answer different questions:

```mermaid
flowchart TB
    OBS["one scored rollout"]:::ev
    OBS --> PK["POOL score tensor<br/>key = (task, CONSTANT mechanism)"]:::live
    OBS --> TK["ENTROPY tracker<br/>key = (task, REAL mechanism)"]:::live
    PK --> Q1["asks: is c1 better than base?<br/>champion selection intersects on the<br/>EXACT key -> needs SHARED keys"]:::note
    TK --> Q2["asks: how much do candidates<br/>disagree on this mechanism?<br/>-> needs SEPARATED keys"]:::note
    Q1 --> W1["mechanism-keyed pool cells<br/>=> empty intersection<br/>=> ranking regresses SILENTLY"]:::bad
    Q2 --> W2["constant-keyed tracker cells<br/>=> unrelated faults pool<br/>=> fake 'a fix is reachable'"]:::bad
    classDef live fill:#0b6,stroke:#053,color:#fff
    classDef ev fill:#fff4cc,stroke:#a70,color:#000
    classDef note fill:#cfe8ff,stroke:#036,color:#000
    classDef bad fill:#e55,stroke:#900,color:#fff
```

Collapsing these into one key looks like a tidy-up and breaks selection without
raising an error. This is the single easiest way to silently ruin the system.

### 5.3 Entropy: honest unavailability

`H(t,m) = Var(scores) × max(max_score, floor)`, gated on **≥3 comparable
candidates** and **≥2 rollouts each**. Below the floor the cell is **unavailable
with a reason from a stable category set** — never a substituted zero.

```
fallback_rate = None    for 0 observed cells   (never 0.0 — zero would claim
                                                perfect availability for a run
                                                that measured nothing)
```

Measured on an offline loop: `3/3 cells unavailable = 100% fallback
(floor_unmet=3)`. That is the report working, not the system working — and it is
exactly the distinction most frameworks hide.

### 5.4 The known gap: we only autopsy failures

Today only *failing* rollouts are analyzed. A candidate can score `1.0` on every
task and hold **zero** mechanism ids — invisible to any mechanism-keyed lookup. So
the compass compares **bad vs less-bad** and never reads the survivors.

The planned fix (two polarity-isolated judges, one shared cluster namespace,
signed valence) is specified as a **future directive** in
[`../design/issue-lifecycle.md`](../design/issue-lifecycle.md) §6 D5.6. It is
entirely unbuilt.

---

## 6. Selection: how work is chosen

```mermaid
flowchart LR
    POOLC["all candidates<br/>+ score cells"]:::live
    CMP["comparable_cells<br/>the cells BOTH measured"]:::live
    DOM["dominates"]:::live
    PF["pareto_frontier"]:::live
    CH["select_champion<br/>king-of-the-hill, PAIRWISE"]:::live
    GATE["S_j > 0 preference gate<br/>pairwise judge, ON by default"]:::live
    AGG["weighted aggregate<br/>REPORTED DIAGNOSTIC ONLY"]:::gated
    POOLC --> CMP --> DOM --> PF --> CH
    GATE --> CH
    CH -.->|"printed, never decides"| AGG
    classDef live fill:#0b6,stroke:#053,color:#fff
    classDef gated fill:#fd7,stroke:#a70,color:#000
```

**Why pairwise, not a scalar.** A candidate that *skipped* the hard task once won
champion selection by averaging over a smaller task set:

```text
base   ran easy(0.9) + hard(0.1)  -> outcome 0.500  agg 0.6250
candA  ran easy(0.9) only         -> outcome 0.900  agg 0.7450  <== WON
```

`candA` was identical to base on the only task both attempted. Ranking is now
pairwise over shared cells, so skipping cannot win. Four of the five champion-math
defects (SV-2..SV-5) were exactly this class: **arithmetic that ran without error
and ranked the wrong harness.**

---

## 7. LLM call topology — three routes, one observer

**The section most likely to surprise you.** There are three distinct ways this
system reaches a model, and they differ in whether a call can be *labelled*.

```mermaid
flowchart TB
    subgraph R1["Route 1 — direct LiteLLM wrapper"]
        A1["cuga_analyzer"]:::live
        A2["cuga_mechanism_adjudicator"]:::live
        A3["cuga_rho_comprehender"]:::live
        A4["cuga_rho_judge"]:::live
    end
    subgraph R2["Route 2 — via CugaAgent"]
        B1["preference_judge"]:::gated
        B2["rho_optimizer"]:::gated
    end
    subgraph R3["Route 3 — editor via CugaAgent"]
        C1["CugaEditorAgent"]:::gated
    end
    HDR["correlation_headers()<br/>emits X-AE-*"]:::live
    SCOPE["correlation_scope()<br/>ZERO production callers"]:::dead
    PX["mitmproxy<br/>captures ALL THREE routes"]:::live
    A1 & A2 & A3 & A4 --> HDR
    SCOPE -.->|"never set =><br/>headers render EMPTY"| HDR
    A1 & A2 & A3 & A4 --> PX
    B1 & B2 --> PX
    C1 --> PX
    classDef live fill:#0b6,stroke:#053,color:#fff
    classDef gated fill:#fd7,stroke:#a70,color:#000
    classDef dead fill:#e55,stroke:#900,color:#fff
```

| Fact | Status |
| --- | --- |
| CUGA-**internal** calls are captured (it honours `HTTPS_PROXY`) | **verified** — one editor run → 3 flows |
| 4 adapters emit `X-AE-*` on the wire | **LIVE** |
| production ever *sets* a correlation label | **DEAD** — 0 callers in `src/`, 0 in `scripts/` |
| routes 2 and 3 can carry labels | **ABSENT by construction** — they bypass our wrappers |

**Hold onto this:** interception is complete; *correlation* is half-wired. Every
captured flow today is unlabelled and must be grouped by timestamp and body. That
is a labelling gap, not a visibility gap — see the next document.

---

## 8. What is not wired — the honest list

| Item | Status | Consequence |
| --- | --- | --- |
| Crossover / merge (`core/merge.py`, 393 lines) | **DEAD**, 0 importers | no recombination runs at all |
| Parallel batch execution | **TEST-ONLY** | sequential in production |
| `Orchestrator.run_iteration` | **TEST-ONLY**, 0 `src/` callers | read `SequentialGepaRunner` instead |
| `correlation_scope` | **DEAD in production** | captures unlabelled |
| Checkpoint / counterfactual replay | **ABSENT** | never assume a trace is replayable |
| Cross-task mechanism identity | **ABSENT, deferred by decision** | needs content-addressed ids, not a patch |
| Executable-tool artifact class | **ABSENT** | we edit text surfaces, not runnable scripts |
| Two-judge positivity (D5) | **ABSENT, specified** | survivors are never studied |

**Why we publish this list.** A framework that cannot name its own unwired paths
cannot be trusted on the paths it claims. Across 2105 passing tests, two fully
tested modules are unreachable in production — a green suite proves code runs, not
that it runs *here*.

---

## 9. Reading order

1. This file — the shape.
2. [`IMPLEMENTED-PIPELINE-MAP.md`](IMPLEMENTED-PIPELINE-MAP.md) — `file:line` truth.
3. [`../SEVERE-OPEN-ISSUES.md`](../SEVERE-OPEN-ISSUES.md) — where the instrument lies.
4. [`../design/issue-lifecycle.md`](../design/issue-lifecycle.md) — clustering decisions + D5 directive.
5. [`LLM-INTERCEPTION-AND-REFLECTION.md`](LLM-INTERCEPTION-AND-REFLECTION.md) — the builder's workflow.
6. [`selection-algorithms.md`](selection-algorithms.md) — the formulas.
7. [`../USER-MANUAL.md`](../USER-MANUAL.md) — every flag.
