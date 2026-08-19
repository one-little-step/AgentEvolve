# RHO Paper-Fidelity Delta — Prompt Rubrics

> **STATUS 2026-08-19 — GAPs 1, 2, 3, 4 implemented.** Concepts were migrated, not
> transcribed: process-over-outcome, consistency-as-reliability, and the four named
> per-rollout findings. Wording is ours and is CUGA-specific where our live data
> required it. GAP 6 investigated, divergence confirmed, decision pending. GAPs 5
> and 7 still open. See "What shipped" at the bottom.
>
> **No unit tests accompany the GAP 1-4 changes, deliberately.** These are prompt
> strings consumed by an LLM. An offline test can only assert that a substring is
> present, which is satisfied by any wording -- including a wrong one -- and is
> made green by pasting the asserted sentence into the source. That is a tautology,
> not verification. Efficacy is only measurable live; see "How to verify".

Source: `reference/rho_ref/RHO_2606.05922.pdf` — *"Evolving Agents in the Dark:
Retrospective Harness Optimization via Self-Preference"* (Pan, Liu, Lin, Zeng,
Tang, Zhou, Lu, Jia). 41 pages. Prompts are Appendix B, Listings 1-5, pages
11-16. Harness examples are Listings 6-13, pages 24-38.

**Why this file exists.** Our prompts reward *outcome* more than the paper does.
The paper rewards the **process / trajectory** — explicitly, structurally, and in
its severity anchors. Fixing this is a prerequisite for a credible delta, because
the reward signal determines what evolution optimizes toward. This is not an
"open issue" to log and defer; it is the measurement instrument itself.

Extracted 2026-08-18 with `uv run --with pypdf`. Quotes are verbatim from the PDF.

---

## The five paper prompts

| Listing | Page | Role | Our module |
| --- | --- | --- | --- |
| 1 | 11 | `solve` wrapper | rollout path |
| 2 | 12 | difficulty judge (coreset selection) | `cuga_rho_judge.py` |
| 3 | 13-15 | **diagnosis** | `cuga_rho_diagnoser.py` |
| 4 | 15-16 | **optimize** | `cuga_rho_optimizer.py` |
| 5 | 16 | **rank** (Best-of-N acceptance) | `cuga_preference_judge.py` |

---

## GAP 1 (P0) — The judge has no efficiency axis

**Paper, Listing 5 rubric:**

```text
+10: A -> B is a change from unacceptable to excellent; B's trajectory is
     EFFICIENT and its answer is correct.
  0: A and B perform comparably, or it is not possible to determine which is
     better.
-10: A -> B is a severe regression; B's trajectory is INEFFICIENT and its
     answer is wrong.
```

Efficiency is **conjoined with correctness at both poles**.

**Ours:** `cuga_preference_judge.py` — `JUDGE_INSTRUCTIONS` (line 129),
`_PROMPT_TEMPLATE` (line 167). Verified by grep:

```bash
grep -cin "efficien|wasted step|unnecessary work|fewer step" \
  src/agent_evolve/adapters/cuga_preference_judge.py
# 0
```

**Zero mentions.** Our `WHAT YOU SHOULD REWARD` covers reaching the answer,
executing tools, verifying, committing — all good, none about waste. Consequence:
a candidate that reaches the right answer in 30 thrashing steps currently ties
with one that reaches it in 5. Under the paper's rubric that is a clear positive
for the efficient side.

Note our judge *does* say "Ignore length." That is correct anti-verbosity
guidance and must **not** be confused with efficiency — length is output prose,
efficiency is wasted *steps/tool calls*. Keep the former, add the latter, and say
explicitly that they are different.

---

## GAP 2 (P0) — Cross-trajectory inconsistency is not first-class

**Paper, Listing 3 Step 3:**

```text
## Step 3: Analyze inconsistency
Compare the three event sequences and final answers. Identify whether there are
inconsistencies among them: where and why the trajectories diverged, and how
those differences affected the behavior.
```

**Paper severity anchors (Listing 3 Step 5):**

```text
0.0     : no meaningful issue; all trajectories answered accurately and efficiently
0.1-0.3 : minor inefficiency or weak concern; do not optimize from this alone
0.4-0.7 : mixed success, INCONSISTENCY, or a plausible harness gap
0.8-1.0 : clear failure, missing information, or a high-confidence harness issue
```

**Divergence alone earns 0.4-0.7 even when nothing failed.** The paper's rationale
(§4.2): `I_t = rank_val(t,{τ_g}) ∪ rank_con(t,{τ_g})` — self-validation UNION
self-consistency. Inconsistency is half the diagnostic signal, and Table 4 ablates
`−self-consistency` as its own arm.

**Ours:** `cuga_rho_diagnoser.py` anchors (lines 212-223):

```text
0.0  nothing recurs; the rollouts differ only harmlessly
0.2  a minor inefficiency; the task was still handled
0.4  one rollout was derailed by it; the others absorbed it
0.6  it recurs and degrades results, but some rollout still recovered
0.8  it recurs in most rollouts and none recovered
1.0  it recurs in every rollout and blocks the task outright
```

Every band above 0.2 is keyed to **recurrence**. Ours does collect
`disagreements` and instructs "Report only the consequential ones" (Step 3, lines
177-181) — the mechanism exists — but severity never rewards divergence on its
own. A task where rollouts diverge badly yet none outright fails scores low here
and mid-band in the paper.

**Newly relevant:** before the response-cache fix (2026-08-18) trajectories were
often byte-identical, so this signal was structurally zero. It only became
available now. Do not port this change without re-reading §1.1 of
`RESUME-HERE-2026-08-18-CACHE-FIX.md`.

---

## GAP 3 (P1) — Per-trajectory `quality_analysis` slots

**Paper, Listing 3 Step 1.4** binds four named findings to a fixed output slot:

```text
In quality_analysis, note what evidence, files, tools, or reasoning steps the
trajectory relied on, and whether there was UNNECESSARY WORK, MISSED
INFORMATION, MISLEADING EVIDENCE, or an INCORRECT DECISION.
```

Plus `issues` per trajectory: *"any missed information, misleading evidence,
inefficiency, or incorrect decision; empty string if none"*.

**Ours** (diagnoser lines 170-176) asks about verification and "what went wrong:
wrong assumption, tool misuse, missing evidence, unfinished work, stopped too
early, wasted steps". Overlapping but not bound to a structured per-trajectory
slot, and `wasted steps` appears once in a list rather than as a first-class
field. The paper's JSON binds each field "to a fixed slot so `optimize` can attend
by severity" — the structure is load-bearing for the next stage.

---

## GAP 4 (P1) — Optimizer's definition of "better"

**Paper, Listing 4 opening sentence:**

```text
Based on the per-task diagnoses in diagnoses/, analyze and optimize the current
harness/ to improve performance on future tasks. "Better performance" means the
agent's final answer more directly and correctly answers what each task asks,
WITH FEWER WASTED STEPS.
```

**Ours:** `cuga_rho_optimizer.py` — 0 efficiency mentions (same grep). Our
`OPTIMIZER_INSTRUCTIONS` is strong on *surface selection* (which artifact actually
reaches the model) — genuinely better than the paper here, and worth keeping — but
it never states that fewer wasted steps is part of the objective.

Paper's steps 2-5 we already match well: severity as "a soft attention weight,
not as ground truth", cross-task pattern matching, "Low-severity tasks usually
should not cause a harness edit by themselves unless the same issue motif recurs",
and no task-specific hardcoded fixes.

---

## GAP 5 (P2) — Position-bias strategy: DECIDE, do not blindly change

**Paper (p16 prose):** candidate is presented **first** as `trajectory_A`, baseline
as `trajectory_B`, and "the orchestrator negates the returned integer so the scalar
score is oriented as baseline → candidate regardless of presentation order."
Rationale: *"Presenting the candidate first reduces a later-option preference bias
we observed in pilot runs."* **One judge call per pair.**

**Ours:** `compare_symmetric` (`cuga_preference_judge.py:559`) runs BOTH orders,
reports `(fwd - rev)/2` as score and `(fwd + rev)/2` as an observable
`position_bias`. **Two judge calls per pair.**

Ours is arguably more rigorous — it *measures* bias instead of mitigating it by
ordering. But it doubles judge spend, and judge cost dominated the aborted run
(log lines 13,050-25,593 = ~half the run). Given budget pressure this is a
deliberate trade, not an obvious fix. Do not "correct" ours to match the paper
without deciding.

---

## GAP 6 (P2) — Score scale and acceptance gate

- Paper: integer `[-10, +10]`; "downstream only the sign and relative magnitude
  of the score are used".
- Ours: float `[-1.0, +1.0]` (`MIN_SCORE`/`MAX_SCORE`, line 73-74).

Functionally equivalent if only sign+magnitude matter, but the integer scale gives
the judge coarser, better-anchored buckets. Low priority.

**Acceptance gate to verify:** paper Algorithm 1 accepts candidate `j` only when
`S_j > 0`, "otherwise the harness remains at `h_0`", where `S_j` averages the
oriented integer over the coreset `D_core`. Confirm our champion selection matches
this and does not promote on ties or on thin evidence (cf. S1-5).

---

## GAP 7 (P0, the biggest — and NOT a prompt fix)

Paper Table 5 (p11) scores methods on three axes, one being:

> **Full harness:** edits executable **tools and skills**, not memory or prompt
> text alone.

RHO claims this axis. Its harness is a **directory of files** — Listing 4:
*"any type of file - helper scripts, artifacts, environment setup, documentation
with relevant context, and workflows to follow."* The concrete examples are real
executables:

- Listing 8 (p25): `bin/repair-verify` — SWE-Bench Pro tool
- Listing 11 (p32): `tools/validate_mask_csv.py` — Terminal-Bench 2 tool
- Listing 13 (p38): `are_helper.py` — GAIA-2 tool
- Listings 6/9/12: `README.md` = instructions
- Listing 7: `checklists/contract-v...` = workflow artifact

**Ours:** every candidate in every run to date has edited **only `instructions`**.
`skills`, `policies`, `memory` and the `skills/generated-<name>` creation path have
never once been exercised — including in a run whose candidate framing explicitly
invited creating a skill. We have no editable executable-tool surface at all.

So on the paper's own axis we are currently "prompt text alone", i.e. the
**partial** mark, not the satisfied one. This is a larger fidelity gap than any
prompt wording and cannot be closed by editing a rubric. Needs investigation of
editor-surface reachability, and possibly a tools artifact class.

---

## What NOT to change (ours is better or deliberately different)

- **Surface-selection doctrine** in `OPTIMIZER_INSTRUCTIONS` (which artifact
  actually reaches the model, and why a fix on the wrong surface silently does
  nothing). The paper has no equivalent; ours was earned from live failures.
- **`MECHANISM, NOT SYMPTOM`** with banned phrases and the two-way test
  (diagnoser lines ~186-200). Sharper than the paper's "faithful and actionable".
- **Anti-sycophancy / label-blindness** block in `JUDGE_INSTRUCTIONS`. The paper
  relies on presentation order; ours states the contract.
- **Ground-truth-is-a-regex** warning (`_GT_PRESENT`, line 218) and the
  vacuous-regex filter. Specific to our GAIA splits.
- **`compare_symmetric`** — see GAP 5; a cost decision, not a defect.

---

## Suggested order

1. GAP 1 (judge efficiency axis) — smallest change, largest signal effect.
2. GAP 2 (inconsistency severity) — now measurable post-cache-fix.
3. GAP 4 (optimizer "fewer wasted steps") — one sentence.
4. GAP 3 (structured per-trajectory slots) — schema change, needs tests.
5. GAP 6 verify acceptance gate `S_j > 0`.
6. GAP 5 decide with the user (cost).
7. GAP 7 investigate separately; it is a capability gap, not a rubric gap.

Every prompt change alters measured results, so no baseline collected before these
edits is comparable to one after. Record which rubric version produced any number.

---

## What shipped (2026-08-19)

Evidence that shaped the wording, from the 29 real traces in `data/cachefix_traces/`:

```text
event_count over 29 rollouts: min=13 median=31 max=127
same task, 9 rollouts:  gaia-e1fc63a2 -> [13,13,85,91,95,97,97,112,127]
ALL SIX 13-event rollouts: status=error, graph_node_error=4, answer=NONE
all 23 successful rollouts: 25..127 events, llm_call_start 3..17
```

**This killed the naive rubric.** "Fewer steps is better" would have ranked the six
crashes as the most efficient trajectories in the corpus — a rubric that rewards
crashing. Among *successful* rollouts the signal is real and large (5.1x events,
5.7x model calls for the same task), so efficiency is worth rewarding, but only
gated on success.

| GAP | Where | Concept migrated |
| --- | --- | --- |
| 1 | `cuga_preference_judge.py` `JUDGE_INSTRUCTIONS` + `_PROMPT_TEMPLATE` | Efficiency as a ranked axis **subordinate to correctness**. "Fewer steps never redeems a wrong answer"; a short errored/uncommitted trajectory is the worst case, not the most efficient. Tool-forced retries excluded as infrastructure noise. Length vs wasted steps explicitly separated. Points the judge at `event_count`, already supplied by `_render_trace` (line 344). |
| 2 | `cuga_rho_diagnoser.py` severity anchors | **Consistency is reliability.** 0.4 band now reads "one rollout derailed, OR the rollouts disagreed consequentially even though none failed" — divergence alone earns mid-severity, because it means the harness underdetermined the outcome. Excludes harmless variation and crash-induced divergence. |
| 3 | `cuga_rho_diagnoser.py` analysis order | Four named findings bound per rollout: UNNECESSARY WORK / MISSED INFORMATION / MISLEADING EVIDENCE / INCORRECT DECISION, so the optimizer can attend by category across tasks. Step 1 now tells the diagnoser to *exclude* crashed rollouts rather than read them as lean. |
| 4 | `cuga_rho_optimizer.py` `OPTIMIZER_INSTRUCTIONS` | New `WHAT "BETTER" MEANS`: "more directly and correctly ... WITH FEWER WASTED STEPS", reliability as an objective, and an explicit anti-goal — do not cut steps by stopping before verifying. |

Deliberately not transcribed from the paper: its `+10/0/-10` integer anchors (ours
is `[-1,1]`, GAP 6), and its exact Listing 3 phrasing where our surface-selection
and MECHANISM-not-SYMPTOM doctrine is sharper.

Suite after these edits: `1757 passed, 1 skipped` — unchanged, no test asserts on
this wording.

### GAP 6 — investigated, divergence confirmed, decision needed

`core/pool.py:482` `select_champion` is **argmax over an aggregate**, with base as
one ranked entry among many:

```python
aggregate = alpha*outcome + beta*coverage + gamma*stability - delta*regression_risk
#           0.55            0.20            0.15 (uniform)    0.10 (always 0.0)
```

`stability` is 1.0 and `regression_risk` 0.0 for every entry, so both cancel; the
effective ranking is `0.55*outcome + 0.20*coverage`. Consequences:

- There is **no `S_j > 0` gate**. The paper keeps `h_0` unless the candidate's mean
  oriented score is strictly positive; we promote whoever tops the aggregate.
- A candidate with **equal outcome but broader coverage can displace the base**,
  which the paper's rule would refuse.

Changing this alters selection semantics and interacts with protected floors and
entropy safeguards, so it is left as-is pending an explicit decision.

### How to verify the GAP 1 change for real

Offline tests cannot do it. The honest check is a live pair comparison on rollouts
that both succeeded and both answered, where event counts differ sharply:

```text
gaia-ec09fa32   25 events  vs   55 events
gaia-e1fc63a2   85 events  vs  127 events
```

Run `compare_symmetric` on those pairs and check the sign favours the leaner side
while `position_bias` stays near zero. Costs judge calls; not yet run.
