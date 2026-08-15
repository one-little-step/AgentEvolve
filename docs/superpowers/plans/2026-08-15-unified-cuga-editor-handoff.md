# Session Handoff — Unified CUGA Editor Agent (2026-08-15)

**Session type: design + planning only. Zero production code written.**

## Next action

The user was asked to choose an execution mode (subagent-driven vs inline) for the
11-task plan and asked for compaction prep instead. So: **ask again which execution
mode, then execute the plan task by task.** Do not re-design. Do not re-plan.

Plan: `docs/superpowers/plans/2026-08-15-unified-cuga-editor-agent.md` (3746 lines, 11 tasks)
Spec: `docs/superpowers/specs/2026-08-15-unified-cuga-editor-agent-design.md` (472 lines, 16 sections)

Both are self-reviewed and approved. The spec governs where the plan disagrees.

## Verified git state

- Branch `dev4`, HEAD `8e29846 "wiring fix ,using fake editor , phase 8 fix1"`
- Phase 8 adapter wiring was committed in `8e29846` (prior session)
- Uncommitted from this session: the two docs above, plus `feedback/from_qwen/qf28.md`
  (modified), `qf29.md`, `qf30.md`, `feedback/gpt_context/cuga_skills_polices_etc.md`
- Suite baseline: **664 passed, 1 skipped** — unchanged this session

## Why this work exists

`core/fake_editor.py` is the only `Editor` implementation and it cheats:
`fake_editor.py:72` reads `request.task.expected_contract.get("expected_substring", "")`
and pastes the expected answer into the artifact. Answer injection, not optimization.
Valid as a fixture; must never produce a reported research result.

Phase 8 proved the adapter carries edits to CUGA and the causal DAG back to the
analyzer. The editor is the last inert component.

## User decisions locked (do not relitigate)

1. **CUGA agent, multi-turn** editor — not a single LLM call. User's reasoning: a single
   call's attention degrades handling evidence + history + mutation + crossover at once.
2. **One unified call** for mutation AND crossover. No mode flag. The agent sees all
   available parents and picks its own strategy; prompting nudges balance.
3. **Free-form crossover.** Permissive conflict filter accepted as a "dummy proxy".
4. **Same process + explicit trace detachment** (not subprocess isolation).
5. **Evidence scope:** blame + artifacts + history + task `input_text`. Never
   `expected_contract`, never `final_output`, never payload blobs.
6. **Trace payloads:** event metadata + `tool_call` payloads (the richer option) — which
   is exactly why the contamination guard exists.
7. **Terminal submit tool**, adapter-interfaced, clustered via `tracked_tool(app_name=...)`.
   The agent's prose answer is ignored.
8. **Creation allowed**, namespaced and capped.
9. **Parent set:** primary sample + K−1 Pareto donors, K=3.
10. Editor gets **skills/tips** teaching effective harness evolution (spec §6).

## Critical correction to qf30

qf30 proposed a flat `generated/` creation prefix. That breaks: `_harness_slot`
(`cuga_adapter.py:122-138`) accepts only `instructions` or a
`skills|policies|memory/<name>` prefix, so `generated/foo` raises `ValueError` at
registration and the creation path is dead on arrival.

Correct scheme is CUGA-group-first: **`skills/generated-<name>`**, per-attempt cap 2,
pool-wide cap 10, skills-only for now. A test pins this specifically.

## qf30 review outcome

Verdict **approved**, 3 required actions, 6 claims marked unverified. All 6 verified:

| Claim | Evidence |
|---|---|
| `lineage_of` accepts multi-parent | `editor.py:490-491` joins sorted; `record_attempt:499` threads it |
| `commit_to_pool` hardcodes 1 parent | `orchestrator.py:1221` |
| `merge.py:335` never compares left to right | bare `else:` after three `changed` tests |
| `merge.py:250` sums raw blame | `total += n.blame` |
| `plan_merge:305` drops absent artifacts | `continue` |
| `plan_merge` zero production callers | 15 refs, all in `tests/test_merge.py` |
| `editor_calls` uncapped | `config.py:49` is the only occurrence |

Action 1 (budget) → spec §12 + Task 1. Action 2 (namespace) → spec §5 + Tasks 2/4, with
the prefix correction. Action 3 (verify `lineage_of`) → already satisfied by the lines
qf30 itself cited; regression test added anyway.

qf30 inaccuracy (spec §15): its §V table says `EditMemory.retrieve` is bounded "via
`max_records` parameter". Conclusion right, mechanism conflated — `retrieve()`'s
*parameter* `max_records` (`memory.py:373`) is unrelated to the class *field*
`max_records` (`memory.py:251`), which `__post_init__` forces to `None`
(`memory.py:271`). The live bound is `max_history_records`.

## Plan structure

| Task | Deliverable |
|---|---|
| 1 | `max_editor_calls` → `BudgetLimits` + `reserve()` + `manifest_payload` |
| 2 | `adapters/cuga_editor_state.py` — `EditStagingArea`, no CUGA import, 20 tests |
| 3 | `adapters/cuga_editor_evidence.py` — `EvidenceView` + contamination guard, 16 tests |
| 4 | `cuga_adapter.py` — `create` op, `creatable_prefix`, `created_artifact_count`, 8 tests |
| 5 | `core/editor.py` — `ParentContext`, `EditorOutcome`, request fields, 9 tests |
| 6 | `adapters/cuga_editor_skills.py` — instructions + 4 skills, 9 tests |
| 7 | `adapters/cuga_editor_tools.py` — 5 tool clusters, 27 tests |
| 8 | `adapters/cuga_editor.py` — `CugaEditorAgent.propose_edit`, 15 tests |
| 9 | `orchestrator.py` — `select_parents`, observed lineage, 11 tests |
| 10 | `scripts/verify_editor_against_live_trace.py` — one live inference |
| 11 | `orchestrator.py` — **wiring**; without it the editor is unreachable dead code |

Dependencies: 1-6 parallelizable · 7 needs 2,3,5 · 8 needs 6,7 · 9 needs 5 · 10 needs 8 ·
11 needs 8,9.

`manifest_payload` (`config.py:156-166`) enumerates budget keys explicitly and no test
asserts completeness, so omitting the new key would pass silently.

## Architecture invariants

- `core/` never imports `cuga`. `CugaEditorAgent.propose_edit(request) -> EditorResponse`
  satisfies the existing protocol; the multi-turn loop lives entirely inside that call.
  `repair_once_then_classify` works unmodified.
- Tools close over the request, not global state. Authorization enforced in the tool body
  (the real write boundary) **and** re-checked by `validate_editor_plan`. Two layers.
- Tool bodies **return** rejections, never raise — an exception inside a CUGA tool body
  can abort the whole agent run.
- Every tool returns a JSON string (CUGA requires string returns).
- `EditorResponse.__post_init__` requires non-empty edits (`editor.py:103-104`), so a
  decline cannot be an `EditorResponse`. `propose_edit` raises `EditorDeclined`, which
  `_propose_safely` (`editor.py:374-378`) already converts to a recorded non-promotion.
  The distinct outcome survives on `.last_outcome`.
- `EditorRequest.__post_init__` requires `current_artifacts ⊆ write_set`
  (`editor.py:77-82`), pinned by `tests/test_editor.py:68`. Guard stays intact: donor
  content is fetched via tool at runtime, never placed in `current_artifacts`.
- **`no_tool_call` must stay distinct from `no_op`.** Collapsing them would let "the agent
  did not engage" masquerade as "the agent judged no edit warranted" — the same error
  class that produced the retracted Phase 8 E2E PASS.
- `parent_ids` comes from the `read_parent_artifact` tool-execution ledger, not agent
  prose. Same principle that makes `ingest_sdk_tool_calls` correct and
  `ToolObservationRecorder.wrap` dead on live CUGA.
- `supports_counterfactual_replay()` stays `False`.

## Self-review caught 3 defects in my own draft (fixed)

1. **Missing Task 11** — Tasks 1-10 built an editor nothing called; 5/8/9 would be dead code.
2. **`EditorToolContext` mutability contradiction** — test used `object.__setattr__`, impl
   was a mutable slots dataclass, Interfaces block said "frozen".
3. **Task 8 test had a stray `}`** and I'd added a step telling the implementer to
   hand-repair it — a plan defect wearing a step's clothing. Fixed at source.

Also: the File Structure table had omitted `cuga_editor_evidence.py` entirely.

## Known limitations (spec §13) — state honestly

- Editor's own skills are **hand-authored**, so edit quality is bounded by our prompting.
- **Tool-invocation reliability is unproven for this model.** In isolated live runs it
  stopped at ~92 completion tokens with `finish_reason:"stop"` and never called
  `load_skill`. It may not call editor tools either. Task 10 reveals this immediately.
  Fallbacks: better tool-calling model, structured-output mode, simpler single-call editor.
- `instructions` is **one flat scalar** (`cuga_adapter.py:24`) but CUGA instructions are
  per-component, so the editor cannot target the planner separately from the answer node
  even though blame graphs name those actors distinctly.
- CUGA's singleton `ActivityTracker` (`tracker.py:92-94`) and process-global policy DB
  remain shared between editor and rollout in-process. Accepted; guarded by an isolation
  regression test asserting editor LLM calls never appear in a rollout trace.

## Deviations (spec §14) — deliberate

- Free-form crossover bypasses `merge-resolution.md:96-104`. `MergeProvenance` and
  `ArtifactMergeDecision` (`contracts.py:326-406`) are unused on this path; replaced by
  observed-parent lineage.
- `core/merge.py` not fixed: 3 defects plus a **lossy** hole (deletion silently dropped —
  lossy, not merely permissive). Zero production callers, so unreachable. Phase 5 merge
  activation requires fixing these first.

## Retracted claim — do not re-assert

An earlier live E2E was reported PASS with exclusive tokens. That PASS was **contaminated**
by a stale global CUGA playbook (`POL-8078061184`) from a previous session which coerced
skill execution. Wiring is proven; **behavioral proof that an edit changes CUGA behavior is
not established.**

Related: CUGA supports skills *and* policies together (one answer carried both markers).
The narrower true finding is that skill execution isn't reliably triggered by a skill
merely being offered.

## Unresolved user questions (asked, never answered)

1. `.cuga/playbooks/playbook_status-format.md` and `.cuga/skills/web_research/SKILL.md`
   are **git-tracked** from `890927b` and contaminate every future run on any clone.
   Backed up outside the workspace. Delete or keep? Also tracked:
   `.cuga/knowledge/favorite-color.md`, `project-clearance-code.md`.
2. Should `.vscode/` be gitignored? (asked 3×)
3. The user answered "Revise something first" at the design-summary gate, then said
   "continue" without stating the revision. Worth re-asking before implementing.

## Facts not to rediscover

- `cuga==0.3.1` installed (`cuga.__version__` misreports `0.2.20`). Model
  `openai/azure/gpt-5.6-luna`, balanced mode, forced autonomous, native sandbox, shell tool on.
- Editable artifacts: `skills/<name>` → `<ws>/skills/<name>/SKILL.md`;
  `policies/<name>` → `<ws>/playbooks/playbook_<name>.md`;
  `memory/<name>` → `<ws>/memory/<name>.md`; `instructions` → constructor arg.
  Only `replace` existed before Task 4.
- `CugaAgent.invoke()` does not initialize the policy system; `await initialize()` first.
- CUGA needs **both** `cuga_folder=` and env `CUGA_FOLDER` (sandbox and `prepare_node`
  read the env var).
- Reference live trace: `data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781/` — 56 events,
  52 parent edges, 6 real actors, blame mass 1.0, zero payload leaks. Used by Task 10.
- `rg` is gitignore-aware and will not search `.venv`; use `grep -r` or `rg --no-ignore`.
- Run tests with `uv run pytest`; capture to `terminal_output/cuga-editor/<name>.log`.
- The gpt_context doc is at `feedback/gpt_context/cuga_skills_polices_etc.md` (not the
  path originally given).
- CUGA's four surfaces are architecturally distinct mutation surfaces: instructions =
  per-component always-on behavior; skills = lazy `load_skill` procedures; knowledge =
  retrieval; policies = runtime control (can rewrite tool descriptions, block, gate).

## Standing user rules

- **Never `git commit` without explicit request.** Plan steps stage only.
- Tests before implementation.
- Capture every test/smoke/verification run to `terminal_output/<topic>/<name>.log`.
- Never persist credentials, expected answers, evaluator internals, labels, or regexes.
- Prefer structured hardcoded runner config over CLI args.
- Payload storage: `PayloadLevel.RAW_OPT_IN`, `allow_raw_payloads=True`, no redaction.
