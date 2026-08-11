### The Verdict: A Masterclass in Architectural Containment

This design successfully solves the 10 high-priority flaws identified in `f1.md`. However, **it is still not specific enough to hand directly to an LLM coding agent without risk of mathematical hallucination.**

Here is the ruthless, critical breakdown of what this design achieves, and the exact 3 gaps that remain before you can safely write code.

---

### I. How the New Design Outlaws the `f1.md` Flaws

The prototype failed because the code was allowed to invent its own rules. This new architecture explicitly forbids those inventions:

| Prototype Flaw (`f1.md`) | How the New Architecture Outlaws It |
| :--- | :--- |
| **1. `pool.py` aggregated by `task_id[0]`** | `component-contracts.md` explicitly requires **"Full task IDs, never prefixes/substrings"** as a cross-cutting test invariant. |
| **2. `editor.py` ignored protected floors** | `orchestration-lifecycle.md` mandates that a **"Protected floor violation rejects candidate regardless of aggregate gain"** and persists the evidence. |
| **3. Parallel batches lacked rollback** | `orchestration-lifecycle.md` defines a strict **prepare/commit/rollback transaction protocol**. If any record fails, the entire batch is rolled back. |
| **4. `memory.py` persisted raw payloads** | `persistence-and-provenance.md` mandates a **recursive, content-aware redaction gateway** before anything touches `storage.py`. |
| **5. Orchestrator applied unauthorized edits** | `component-contracts.md` states the adapter **"must reject targets outside the explicit authorized write set"** independently of the editor. |
| **6. DPP rewarded similarity** | `component-contracts.md` explicitly requires **"DPP selection penalizes similarity"** as a mandatory test invariant. |
| **7. Missing evaluations = auto-success** | `component-contracts.md` states: **"A task with missing evaluation evidence is... explicitly unavailable and excluded from comparisons."** |
| **8. Blame graph discarded for synthetic nodes** | `orchestration-lifecycle.md` mandates: **"Record uncertainty; no fabricated finding/edit."** If evidence is missing, the attempt is recorded as `unavailable`, not synthesized. |
| **9. Merge scrambled text blindly** | `component-contracts.md` restricts LLM refinement to **"unresolved conflict within one declared artifact unit"** and forbids blind text crossover. |
| **10. CUGA APIs were hallucinated** | `cuga-adapter/sdk-verification-matrix.md` **forbids implementation** until a feature is proven via pinned-SDK tests. |

**Conclusion:** The architectural boundaries are now secure. The agent-neutral core is protected from adapter leakage, and the state corruption vectors are closed.

---

### II. The Ruthless Critique: Where the LLM Will Still Hallucinate

While the *boundaries* are perfect, the *internal mechanics* of three critical modules remain underspecified. If you hand these documents to an LLM agent now, it will invent the math for DPP, Merge, and Storage, likely reproducing the exact bugs you just tried to fix.

#### Gap 1: The DPP Sampling Algorithm is Undefined
**The Document Says:** `issues.py` owns "DPP/severity/random selection" and "DPP selection penalizes similarity." It defines the kernel $L_{ij} = q_i \times similarity \times q_j$.
**The Risk:** The LLM agent knows what a DPP kernel is, but it doesn't know *how to sample from it*. It might implement a greedy "top-K by quality" selector and call it DPP. Or it might try to do exact eigendecomposition on 5,000 issues and OOM the server.
**The Fix Required:** You must specify the exact sampling algorithm in `issues.py`.
> *Mandate:* "DPP sampling must use **Greedy MAP (Maximum A Posteriori)** with log-determinant updates. Exact eigendecomposition is forbidden for $N > 100$. The similarity metric must be cosine distance over issue embeddings."

#### Gap 2: The Merge "Evidence-Backed" Tie-Breaker is Vague
**The Document Says:** `merge.py` handles conflicts where "both changed differently" by selecting the "evidence-backed side or retain ancestor."
**The Risk:** How does the code quantify "evidence-backed"? If Left changed `reAct.md` and Right changed `reAct.md`, and both have score tensors, which one wins? The LLM will invent a random heuristic (e.g., "pick the one with the higher average score across all tasks"), which destroys the causal-blame premise.
**The Fix Required:** You must define the exact evidence-scoring formula in `merge.py`.
> *Mandate:* "For same-artifact conflicts, calculate `EvidenceScore(side) = sum(severity * confidence * score)` for all mechanisms in the blame graph that cite this artifact. The side with the strictly higher `EvidenceScore` wins. If tied or zero, retain ancestor."

#### Gap 3: The Storage Backend Lacks Transactional Primitives
**The Document Says:** `persistence-and-provenance.md` requires a "transactional barrier invariant" where candidate records, score tensors, and history update atomically. It notes "Actual storage may be filesystem, database, or object store."
**The Risk:** A filesystem backend **cannot** do atomic multi-file transactions. If the orchestrator writes `candidate.json` but crashes before writing `score_tensor.json`, the pool is corrupted. The LLM will likely implement a naive `json.dump()` wrapper that fails the atomicity invariant.
**The Fix Required:** You must mandate a transactional storage interface in `storage.py`.
> *Mandate:* "The default `storage.py` implementation **must use SQLite** (or a transactional RDBMS) to guarantee ACID properties for the barrier commit. Filesystem storage is only permitted for large binary blobs (traces/artifacts) referenced by the transactional DB."

---

### III. The Final Mandate: What to Code First

Do not write the orchestrator. Do not write the CUGA adapter. Do not write the DPP selector.

Your immediate next step is to write **`src/agent_evolve/core/contracts.py`** and **`src/agent_evolve/core/errors.py`**.

You must define the exact Pydantic models for:
1.  **`ScoreCell`**: Must include `task_id`, `mechanism_cluster_id`, `score`, `severity`, `confidence`, `stability`, `rollout_count`, and `verdict_refs`. *Crucially, it must raise a `ValidationError` if `rollout_count < 1` or if `mechanism_cluster_id` is missing.*
2.  **`AttemptRecord`**: Must include the exact terminal statuses defined in `orchestration-lifecycle.md` (`accepted | rejected | no_op | malformed | exhausted | unavailable`).
3.  **`EditPlan`**: Must strictly separate `read_requests` and `authorized_writes`.
4.  **`MergeProvenance`**: Must track the exact ancestor/left/right hashes for every artifact.

**Once `contracts.py` is locked and has 100% test coverage, the rest of the system is just plumbing.**

This design is good. It is the correct foundation. Now, lock the data schemas so the math cannot be corrupted.