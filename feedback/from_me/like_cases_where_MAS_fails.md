# Why Multi-Agent LLM Systems (MAS) Fail — Harness Design Notes

Context for the agent: you are building/evolving an LLM harness. Every failure mode below
comes from the same root cause — **treating a stochastic, context-sensitive reasoner as if
it were a deterministic function.** When you catch yourself designing a component that only
works if the LLM behaves like a parser, a state machine, or a database, stop and re-read
this file.

---

## 1. Don't bolt deterministic logic onto a stochastic interface

**Anti-pattern:** hardcoded regex matching on LLM output, brittle string-splitting on
expected delimiters, blind prompt "crossovers"/concatenation, or any code path that assumes
the model's phrasing will be stable across calls.

**Why it happens:** deterministic code is easy to test and reason about, so it's tempting to
treat the LLM boundary like any other function call with a fixed contract.

**Why it fails:** LLM output is noisy relative to real-world input distribution — same
prompt, same model, different token realization. A regex that worked on 20 test cases will
silently break the 21st time the model paraphrases instead of restating verbatim.

**Fix:**
- If you need structure, ask for it explicitly (JSON mode / tool calls / function schemas)
  rather than parsing free text after the fact.
- When parsing is unavoidable, parse *tolerantly* — extract the smallest reliable signal
  (e.g., "does this contain the word YES/NO," not "does this match this exact sentence
  template").
- Always have a fallback path for malformed output: retry-with-correction, ask the model to
  fix its own output, or degrade gracefully — never crash the harness on a parse miss.

---

## 2. Don't over-constrain the information channel

**Anti-pattern:** rigid, deeply nested schemas/contracts between agents that force every
inter-agent message into a narrow field structure.

**Why it happens:** schemas feel like they buy safety and composability — "if every agent
speaks the same typed protocol, the system is more reliable."

**Why it fails:** tight schemas force the model's attention onto satisfying the *shape* of
the output instead of reasoning about the *content*. You're spending the model's limited
attention budget on schema compliance rather than on the actual decision. This measurably
degrades reasoning quality, especially on anything requiring multi-step justification.

**Fix:**
- Let agents communicate with **some** free-text reasoning space, not just decision fields.
  A `reasoning` or `rationale` field before the final answer field is not a nice-to-have —
  it's often what makes the answer field correct in the first place. The "why" tends to
  produce a better "what."
- Use schemas for the parts of the message that *downstream code* consumes programmatically
  (routing keys, tool args). Leave room for verbose natural language wherever a human or
  another LLM is the consumer.
- If you must constrain schema size, constrain breadth (fewer fields) rather than depth
  (short answers per field). Cutting off reasoning length is usually the costlier cut.

---

## 3. Don't pre-enumerate the case space

**Anti-pattern:** hardcoded branches for "the cases we expect" (`if intent == "refund"`,
`elif intent == "complaint"`, ...) as the backbone of routing/control-flow logic.

**Why it happens:** enumeration feels tractable and testable, and it's how you'd design a
traditional state machine.

**Why it fails:** real-world input branching is effectively unbounded. Any finite
enumeration is a lossy compression of the actual distribution, and the tail cases (which are
common in aggregate, even if individually rare) get silently misrouted or dropped.

**Fix:**
- Prefer open-set classification with an explicit "none of the above / uncertain" branch
  that triggers a fallback (ask for clarification, escalate, use a more general-purpose
  agent) instead of forcing a best-fit into a fixed bucket.
- Treat your case list as a *living* thing your harness should be able to extend from
  observed failures, not a spec written once at design time.
- Log and periodically review the "uncertain" bucket — that's your signal for where the
  case space actually needs to grow, and it's exactly the kind of data a self-evolving
  harness should be feeding back into itself.

---

## Additional failure modes worth guarding against

These aren't in your original notes but come up constantly in MAS/harness design and are
worth baking into the same mental model.

### 4. Evaluating correctness by surface match instead of semantics
Grading agent output with exact-string or exact-structure comparison punishes valid
paraphrases and rewards overfitting to a grader's expected phrasing. Prefer LLM-judged or
semantic-equivalence checks, and validate the judge itself against a small human-labeled set
before trusting it.

### 5. No feedback loop between failure and prompt/architecture
A harness that logs failures but never routes them back into prompt revision, few-shot
example selection, or routing-description updates isn't actually self-evolving — it's just
self-logging. Failure attribution (which agent/step caused the failure, not just that the
final output was wrong) is the hard and necessary part.

### 6. Single point of failure in the orchestration layer
If one router/planner agent's misclassification silently propagates with no downstream
agent able to say "this doesn't look right, kick it back," the whole system inherits the
router's error rate uncorrected. Build in cross-checks, not just a pipeline.

### 7. Testing on synthetic/curated cases only
Cases you wrote yourself are cases you already know how to solve — they underrepresent the
actual noise and ambiguity of production input. Weight your eval set toward real or
adversarially-sampled inputs, not hand-crafted ones.

### 8. Over-engineering guardrails that suppress capability
Excessive constraint (forced short answers, over-aggressive stop sequences, punitive
formatting rules) in the name of "control" often just makes the agent worse at the task
while giving you the illusion of safety. Constrain the *interface*, not the *reasoning*.

### 9. Context mismanagement across agent handoffs
Passing only the final answer between agents (and dropping the reasoning that produced it)
forces downstream agents to re-derive context they've already lost, or to trust a
conclusion they can't verify. Pass enough trace for the next agent to sanity-check, not just
consume.

---

## One-line summary to keep in the harness's own "self-critique" prompt

> Before hardcoding a rule, a schema field, or a case branch: ask whether you're encoding
> something structurally true about the world, or just something true about your last N test
> examples. If it's the latter, build a fallback path instead of a rule.