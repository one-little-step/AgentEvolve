# Session Handoff: Real CUGA Editor Complete, Rigorous Verification In Progress

**Date:** 2026-08-15
**Branch:** `dev4`
**HEAD:** `f3c21aa wiring cuga editor 1`
**Suite:** 798 passed, 1 skipped, 0 failures (`uv run pytest`)

---

## 1. What Is Done

All 11 tasks of `docs/superpowers/plans/2026-08-15-unified-cuga-editor-agent.md`
are implemented. `FakeEditor` is no longer the only editor: a real
CUGA-agent-backed editor produces evidence-grounded artifact edits and has been
proven live.

| Task | Component | File |
|---|---|---|
| 1 | `max_editor_calls` budget cap | `core/config.py` |
| 2 | `EditStagingArea` (authorization, caps, parent ledger) | `adapters/cuga_editor_state.py` |
| 3 | `EvidenceView` + contamination guard | `adapters/cuga_editor_evidence.py` |
| 4 | adapter `create` op, `creatable_prefix`, `created_artifact_count` | `adapters/cuga_adapter.py` |
| 5 | `ParentContext`, `EditorOutcome`, extended `EditorRequest` | `core/editor.py` |
| 6 | `EDITOR_INSTRUCTIONS`, 4 skills, `build_editor_prompt` | `adapters/cuga_editor_skills.py` |
| 7 | 16 request-scoped tools | `adapters/cuga_editor_tools.py` |
| 8 | `CugaEditorAgent.propose_edit` | `adapters/cuga_editor.py` |
| 9 | `select_parents`, `donor_count`, `extra_parent_ids` lineage | `core/orchestrator.py` |
| 10 | live verification scripts | `scripts/verify_editor_*.py` |
| 11 | `propose_edits` wiring, `run_attempt` arity | `core/orchestrator.py` |

Uncommitted (staged) at handoff time: `scripts/verify_editor_rigorous.py`,
plus modifications to `adapters/cuga_editor.py`,
`adapters/cuga_editor_skills.py`, `cuga_wrapper/__init__.py`,
`tests/test_cuga_editor_skills.py`, `tests/test_harness_materialization.py`,
and `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md`.

---

## 2. Live Verification Status (the important part)

### Verified working, with evidence

`terminal_output/cuga-editor/live/rigorous-report.json` is the canonical record.
Model: `openai/azure/gpt-5.6-luna`. Real `JSONFileStorage`-backed edit history.

| scenario | outcome | tool calls | consulted history | read donor |
|---|---|---|---|---|
| `history` | `valid` | 7-8 | yes (varies, see below) | n/a |
| `crossover` | `valid` | 7-8 | yes | **yes** (`parents_read: ['donor']`) |
| `creation` | `valid` | 8-9 | yes | n/a (no donors offered) |

Both token scenarios produced edits that mention verification, are prescriptive
rather than advisory, and preserve the pre-existing steps. In `history` the
editor's rationale explicitly cited the seeded prior rejection ("advisory wording
did not change behavior") and responded with numbered prescriptive steps. In
`crossover` it justified transplanting from the donor by citing the 1.0-vs-0.0
score.

`creation` (added after the first handoff, `rigorous-creation2.log`) staged
`skills/generated-pagination` against a mechanism describing an *absent*
capability, and left the unrelated auth skill untouched. Its created artifact was
then proven to survive the whole downstream path, not merely to be staged:
adapter accepted it, it appears in `artifact_inventory`, `created_artifact_count`
counts it, it lands in the `skills` group of `_harness_config`, and
`materialize_harness` writes a loadable `SKILL.md` with a populated (non-`None`)
description. That last check exists because bug 8 below made a skill file exist
on disk yet be silently rejected by CUGA's loader.

Ground truth is tool-body execution, never model prose: log lines show
`"accepted": true, "reason": "staged replacement for 'skills/token-workflow'"`.

Skills are confirmed reaching the model: `Loaded 4 agent skill(s) from
<temp ws>/skills` and the model called `load_skill("refine-artifact")` 8 times.

**Run-to-run variation is real and already observed.** Re-running `history` after
a pure refactor produced `consulted_history=False` where the earlier run had
`True`, with the same prompt and model. This is direct evidence that single live
runs do not establish behavior, and is exactly why step 2 (prompt-variation
sampling) matters. Do not treat any n=1 result here as a capability claim.

### NOT verified — do not claim these

- **5 of 16 tools never reached** across all live runs to date:
  `get_attempt_outcome`, `get_task_input`, `list_trace_actors`, `stage_replace`
  in the creation scenario, `unstage`. Creation itself is now verified;
  `stage_create` has been exercised live twice.
- **n=1-2 per scenario, and results already vary between runs** (see above).
  Greedy decoding means identical prompts are not independent samples. No
  variance estimate exists.
- **Edit-history retrieval is exact-key lookup, NOT embeddings RAG.**
  `EditMemory.retrieve()` (`core/memory.py:372`) matches `issue_fingerprint`
  strings. An Ollama embedder exists (`core/embeddings.py`) but the editor does
  not use it. A semantically similar past attempt under a different issue id
  will not be found.
- **No delta measurement.** It has never been confirmed that an edited harness
  scores better on a rerun. This is the number that matters and it is unproven.
  Note the creation check proves an artifact *reaches* CUGA, which is necessary
  but not sufficient: reaching the agent is not improving the score.
- Multi-parent crossover has one live sample; `plan_merge` in `core/merge.py`
  still has zero production callers.
- `supports_counterfactual_replay()` remains `False`.

---

## 3. Immediate Next Steps (user-approved order)

1. ~~**Creation scenario** forcing `stage_create`~~ **DONE.** Third scenario in
   `scripts/verify_editor_rigorous.py`, verified live twice, plus downstream
   survival through `materialize_harness`. It found bug 11 (below).
2. **Prompt-variation sampling**, 3-4 phrasings x scenarios, for a real
   tool-reachability estimate. Vary the *prompt*, never repeat an identical one
   and call it N trials. Now higher priority than before: the `history` scenario
   already flipped `consulted_history` between two runs of identical input.
3. **Wire the embedder into `EditMemory.retrieve()`** so history generalizes
   across issue fingerprints. Requires a decision on the similarity floor and on
   whether lexical fallback is acceptable when Ollama is unavailable.
4. **The delta loop:** edit -> rerun -> score. Until this exists, no claim about
   harness improvement is supported.

Then: replace the judge with a CUGA agent (share the editor's substrate per
`feedback/from_qwen/qf31.md`: shared construction, separate roles), and RHO seed
generation.

---

## 4. Bugs Found And Fixed This Session (11 total)

Every one was invisible to the offline suite. Each now has a regression test.

**Live-only defects in the editor path:**

1. All 16 tool bodies lacked docstrings -> LangChain `@tool` refuses to build.
2. Editor never configured the model env -> CUGA defaulted to `gpt-4o` against
   api.openai.com -> "Missing credentials". Fixed by calling
   `RuntimeSettings.from_env().configure_cuga_environment()` in
   `prepare_editor_environment`.
3. `_run_cuga_agent` ignored the recorded `callables` and rebuilt from `ctx` ->
   tool ledger structurally always empty on the real path. **Same class as the
   retracted Phase 8 PASS: machinery reporting success while measuring nothing.**
4. `_recording_wrapper` dropped `__doc__` and the signature -> re-broke (1) and
   would have produced an empty args schema. Fixed with `functools.wraps`.
5. Model emitted 8 code fences per turn; CUGA runs only the first -> added a
   one-block-per-turn contract to `EDITOR_INSTRUCTIONS`.
6. Multi-step narration with no fence -> 0 tool calls -> added a first-turn
   "make your very next message a fenced block" directive to the prompt.

**Skills pipeline (the two most damaging):**

7. `EDITOR_SKILLS` were **never materialized**. `enable_skills=True` with
   `cuga_folder=None` made CUGA load `<cwd>/.cuga/skills`, so a live run got a
   stale `web-research` skill and none of its own four. Fixed by
   `materialize_editor_skills()` + binding `cuga_folder`, `skills_folder`, and
   `CUGA_FOLDER` to an isolated temp workspace.
8. `_derive_description` returned `# Heading`; unquoted `#` is a YAML comment ->
   `description: None` -> CUGA's loader rejects the skill. Fixed by stripping
   Markdown markers and emitting a quoted scalar. This also correctly broke
   `test_materialize_harness_writes_skill_with_frontmatter`, which had asserted
   the *unquoted* form — a test that locked in the broken shape.
9. Skill descriptions were passive titles; the description is the only text the
   model uses to select a skill. Rewritten as usage triggers.
10. **Crossover was unreachable**: nothing in the prompt said donors existed, so
    two runs with a strictly better donor never called `list_parents`. Fixed by
    adding `_parent_summary()` to the prompt plus an instruction to inspect
    donors before refining.

**Found by the creation scenario:**

11. **Authored artifact bodies carried the agent's source indentation.** The
    editor writes artifact content inside a Python string literal in the
    sandbox; the first live creation produced a skill whose every line after the
    first began with four spaces. Markdown reads uniformly indented lines as a
    code block, so the skill would have been materialized as a literal listing
    instead of instructions — a silent capability loss of exactly the kind bug 8
    caused. Fixed with `normalize_authored_content()` applied at both staging
    entry points in `cuga_editor_state.py`.
    `inspect.cleandoc`, not `textwrap.dedent`: dedent takes the common prefix
    over *all* lines, and the literal's first line is flush, so the prefix is
    `""` and dedent is a **no-op on exactly the observed shape**. cleandoc
    ignores the first line when computing the margin. Relative indentation is
    preserved, so nested list items and fenced code blocks keep their structure
    (regression-tested).

**Contamination guard hardening (Task 3):** `contamination_terms_from` scanned
only top-level `.values()`, so `{"expected_any": ["tok"]}` and
`{"grader": {"expected": "tok"}}` yielded **zero terms** — and a guard with zero
terms passes every payload through. Now recurses. Two residual, deliberate
non-leaks remain undecided by the user: non-string contract values
(`{"expected_value": 42}`) and terms under 3 characters.

---

## 5. Key Files

**Editor implementation:**
- `src/agent_evolve/adapters/cuga_editor.py` — `CugaEditorAgent.propose_edit`,
  `materialize_editor_skills`, `editor_agent_kwargs`,
  `prepare_editor_environment`, `_parent_summary`, outcome classification.
- `src/agent_evolve/adapters/cuga_editor_tools.py` — `build_tool_callables`
  (16 pure-Python bodies, no SDK) and `build_editor_tools` (the only
  CUGA/LangChain import; accepts supplied callables).
- `src/agent_evolve/adapters/cuga_editor_skills.py` — `EDITOR_INSTRUCTIONS`,
  `EDITOR_SKILLS` (4), `build_editor_prompt`. **Prompt text here is
  empirically load-bearing; change it only with a live run to confirm.**
- `src/agent_evolve/adapters/cuga_editor_state.py` — `EditStagingArea`.
- `src/agent_evolve/adapters/cuga_editor_evidence.py` — `EvidenceView`, guard.

**Verification:**
- `scripts/verify_editor_rigorous.py` — the one to extend. Scenarios `history`,
  `crossover`; reports per-tool reachability and substantive edit-quality flags.
- `scripts/verify_editor_against_live_trace.py` — original single-parent check.
- `scripts/probe_editor_evidence_guard.py`,
  `scripts/probe_contamination_term_shapes.py` — adversarial guard probes.
- `terminal_output/cuga-editor/live/` — all live logs and reports.

**Reference:**
- `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md` — updated
  this session with a new section, "Building A CUGA Agent As A Tool-Driven
  Worker". Read before any CUGA work; it is shared with the other CUGA repo.
- `.superpowers/sdd/progress.md` — task-by-task ledger.
- `feedback/from_qwen/qf31.md` — decoupling and analyzer/editor role separation.

---

## 6. Methodology Rules Earned The Hard Way

- **Green offline tests prove almost nothing about a live agent.** Ten defects
  survived a passing 790-test suite. Every capability claim needs a live run.
- **Never trust a passing test you have not seen fail.** Mutation-verify guards
  by breaking the thing they protect. Two guards this session were vacuous until
  checked: the contamination guard (flat-shape-only tests) and an import blocker
  using the removed `find_module` hook, which Python 3.14 silently ignores.
- **Ground truth is the tool body executing.** Never the model's narration.
- **A capability the prompt does not mention will not be used.** Prove the option
  appears in the rendered prompt before concluding the model won't use it.
- **Report unreached tools.** "Valid outcome" plus 5/16 tools untouched is a
  much weaker claim than it looks.
- Prefer the plan's exact code, but the plan is not authoritative: its test
  fixtures had a repeated `CugaWrapper(runtime=...)` error (missing `settings`),
  a dead `_test_tasks` seam, and prompt phrasing contradicted by measured data.

---

## 7. Environment

- `uv run pytest` / `uv run python` — system Python lacks deps.
- `cuga==0.3.1` (`cuga.__version__` misreports `0.2.20`).
- `.env` supplies `CUGA_MODEL`, `CUGA_BASE_URL`, autonomous mode,
  `DYNACONF_SKILLS__ENABLED=true`,
  `DYNACONF_ADVANCED_FEATURES__ENABLE_SHELL_TOOL=true` (required for skills;
  consciously injects a sandbox shell tool).
- Alternative model for ablations: `openai/azure/gpt-5.6-terra` (more expensive;
  avoid during development).
- `rg` is gitignore-aware and skips `.venv`; use `grep -r` for SDK spelunking.
- `timeout` is unavailable on this macOS shell.
- Never `git commit` without explicit user approval.

**Known repo contamination, still unresolved:** `.cuga/skills/web_research/`,
`.cuga/playbooks/`, `.cuga/knowledge/` are git-tracked and were the source of
bug 7. Decide whether to remove them. `.vscode/` gitignore is also undecided.
