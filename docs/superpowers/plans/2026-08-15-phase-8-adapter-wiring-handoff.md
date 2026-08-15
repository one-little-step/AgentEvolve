# Session Handoff — Phase 8 CUGA Adapter Wiring (2026-08-15)

## Verified state at handoff

- Branch `dev4`, HEAD `890927b "phase7 fix2"`.
- **Nothing committed this session.** All work uncommitted.
- Suite: **664 passed, 1 skipped, 1 warning** (was 637+1 at session start; +27 tests, 0 regressions).
- Evidence: `terminal_output/cuga-adapter/` (16 logs) and `terminal_output/cuga-adapter/e2e/`
  (`e2e-report.json`, full `trace-A/`, `trace-B/`).

### Modified (3 files, +207/-23)
- `src/agent_evolve/adapters/cuga_adapter.py` — Fixes 1-3 (+174)
- `src/agent_evolve/cuga_wrapper/__init__.py` — isolation fixes (+49)
- `tests/test_cuga_adapter.py` — corrected a test that **pinned the bug** (+7)

### New (untracked)
- `tests/test_cuga_adapter_wiring.py` — 12 tests (harness delivery, DAG mapping, inventory, leak audit, live-trace regression)
- `tests/test_cuga_agent_construction.py` — 11 tests (cuga_folder / CUGA_FOLDER / policy reset isolation)
- `tests/test_cuga_execute_lifecycle.py` — 4 tests (initialize-before-invoke, fail-closed)
- `scripts/verify_adapter_against_live_trace.py` — replays adapter over real 56-event trace
- `scripts/verify_adapter_e2e_live.py` — differential live E2E (2 real inferences)
- `scripts/diagnose_skill_root_resolution.py` — proves skill-root fallback bug
- `feedback/from_qwen/qf25.md`, `qf26.md`, `qf27.md`

---

## What was fixed and PROVEN

### Fix 1 — edits reach the agent (`cuga_adapter.py`)
`run_full_rollout` previously passed only `{"input": task.input_text}`. Now builds a real harness
config via `_harness_config()`, mapping artifact ids onto CUGA groups.

### Fix 2 — the DAG reaches the analyzer
`capture_trace` previously hardcoded `actor_id=None, parent_event_id=None` and read the thin
2-event list. Now reads `causal_trace_path` and maps the rich persisted trace, falling back to
thin events when absent.

### Fix 3 — inventory is real
Added `register_candidate(version, artifacts)` as the seeding seam (for the future
SeedGenerator / RHO stage), with per-version storage so N sibling candidates stay independent.

### Anti-inert guard
`_harness_slot()` raises on any artifact id CUGA cannot receive, at **both** registration and
rollout. Silent dropping was the fabricated-evidence risk.

### Verified against the REAL 56-event live trace
`data/traces/5d434903-bc26-4dc4-9229-8d886d2c6781` via
`scripts/verify_adapter_against_live_trace.py` → **PASS**:
```
mapped 56/56 events · 52 parent edges · 6 distinct actors
actors: CugaLiteSubgraph, FinalAnswerAgent, SDKCallback, call_model, prepare, sandbox
blame nodes 6 · blame mass 1.0 · 'unknown' placeholder: False
blob bodies leaked into core trace: 0
```

---

## SECURITY GATE — RESOLVED FAVORABLY (was open from last session)

Checked empirically, not assumed: **rich event payloads contain only content-addressed refs.**
Every `*_ref` is a bare 64-char SHA-256 hash. The 40 KB prompts and full AgentState live in
`payloads/` blobs that `capture_trace` NEVER opens.

The only >64-char payloads are the 3 `tool_call` observations (tool args/results) — environment
evidence that blame needs, not model state.

**Conclusion:** the DAG reaches the analyzer with zero raw-payload exposure. The broader gate
decision is only needed if something later chooses to dereference blobs. Pinned by
`test_capture_trace_never_dereferences_payload_blobs` and the live-trace regression test.

---

## FOUR ADDITIONAL REAL BUGS found by the live E2E (all in `cuga_wrapper/__init__.py`)

The first live E2E run FAILED. Investigating was worth the cost — each of these alone would make
the loop measure nothing while appearing to run.

1. **`cuga_folder` bound only when policies existed.** CUGA resolves the skills dir from
   `cuga_folder` (`skills.loader.get_skill_root`), NOT `skills_folder`. A skills-only candidate got
   `cuga_folder=None` → fell back to `<cwd>/.cuga/skills`. Fix: bind whenever a workspace exists.
2. **The constructor arg never reaches the sandbox or `prepare_node`.** `build_runtime_tools`
   (`runtime_tools.py:147`) calls `create_sandbox_tools(thread_id=...)` without `cuga_folder`;
   `prepare_node.py:321` reads `os.getenv("CUGA_FOLDER", ...)`. Fix: export `CUGA_FOLDER`, and pop
   it when no workspace so a previous candidate can't leak.
3. **CUGA persists policies in a PROCESS-GLOBAL store** at
   `.venv/lib/python3.14/site-packages/cuga/dbs/cuga.db` (`config.DBS_DIR`), shared by all
   candidates, ignoring `cuga_folder` entirely. A playbook from a *previous session*
   (`POL-8078061184`) contaminated every run. Fix: `reset_policy_storage=bool(workspace_dir)`.
4. **`reset_policy_storage` was silently ignored.** `CugaAgent.invoke()` never initializes the
   policy system — that lazy init lives in `CugaSupervisor.invoke()` (`sdk.py:3326`), a DIFFERENT
   class. Fix: `_execute` now awaits `initialize()` before `invoke()`, fail-closed on error.

---

## RETRACTED CLAIM (important — do not re-assert)

I earlier reported the live E2E as PASS with exclusive tokens (A had only token A, B only token B).
**That PASS was contaminated and must not be cited as proof.**

Trace diff proved why (all traces preserved on disk):
- **PASS run** `data/traces/bee316b5-d209-434d-b639-2550191ffdb5` — first LLM call is only
  **1595 bytes, contains `POLICY-MARKER`, no `load_skill`**. A policy pre-pass matched the STALE
  global playbook and injected guidance that pushed the model through
  `sandbox` → `load_skill` → skill body → signature. 5 llm_calls, sandbox reached.
- **FAIL runs** `bb9c2eab-...` (12:15) and `8becce68-...` (12:19) — correctly isolated, no stale
  policy. Straight to a 42826-byte prompt. 2 llm_calls, **sandbox never reached**,
  `finish_reason:"stop"` at 92 completion tokens. Byte-identical A/B responses (2428/1914).

So the earlier success was caused by the very contamination the isolation fixes remove. I credited
the fix when stale state deserved the credit.

**Answer to "does CUGA support either policies or skills, not both?" — NO, it supports both.**
The passing answer contained BOTH `SKILL-SIGNATURE` and `POLICY-MARKER` simultaneously. The real
finding is narrower and is an AGENT property, not an adapter bug:
**CUGA skill execution is not reliably triggered by a skill merely being offered; it needed policy
pressure.** User decided this class of model behavior is a target for evolution, not a blocker.

**Live proof that an edit changes real behavior: NOT established.** Wiring is proven; behavioral
proof is not.

---

## WHAT THE EDITOR ACTUALLY EDITS (user's last question)

Editable artifact types — `cuga_adapter.py:23-24`:

| artifact_id | CUGA harness key | On disk |
|---|---|---|
| `skills/<name>` | `skills` | `<ws>/skills/<name>/SKILL.md` |
| `policies/<name>` | `policies` | `<ws>/playbooks/playbook_<name>.md` |
| `memory/<name>` | `memory` | `<ws>/memory/<name>.md` |
| `instructions` | `special_instructions` | constructor arg |

Only operation: **`replace`** (whole-artifact swap). No section-level or diff editing. Cannot touch
CUGA code, tools, graph, or model config.

Target selection: `build_issues` (`orchestrator.py:960`) observes base per task; for failures
`finding_from_analysis` sorts blame descending and attaches the write set to the **top-blamed
actor** (`orchestrator.py:932`). Editor returns structured edits; the ORCHESTRATOR applies them —
the editor never mutates the workspace.

### CRITICAL: there is no real editor
Only `FakeEditor` exists (`core/fake_editor.py`); `grep propose_edit` → one class. Its docstring
says real ones "will be backed by LLMs."

Worse, it **cheats by design**:
```python
expected = request.task.expected_contract.get("expected_substring", "")
```
It reads the expected answer from the task contract and pastes it into the artifact. That is
answer injection, not optimization. Acceptable as a deterministic fixture for pool/validation
mechanics; **must never produce a reported result.**

Note the asymmetry: `EditorResponse` runs `sanitize_payload` and `finding_from_analysis` keeps
`expected_contract` out of rationales — the guardrails exist, but this fixture sits inside them.

---

## NEXT STEPS (agreed order)

1. **Build a real LLM editor** — reads blame + current artifact content, proposes a rewrite,
   and MUST NOT receive `expected_contract`. This is the piece that makes the system
   self-improving rather than self-copying. User was asked and this is the pending decision point.
2. **Seed generator / RHO stage** (qf27) — needs the real editor first, else N candidates are
   meaningless. qf27 Option A = deterministic mutation seeds (interim, label honestly, NOT
   "RHO proposals"); Option C = full RHO pipeline.
3. Optional clean differential re-run: give each candidate its OWN policy (supplying the trigger
   the agent needs) sourced from the candidate workspace, varying only the skill token. Two
   inferences.
4. Commit only on explicit user request.

---

## OPEN ITEMS / HAZARDS

- **`.cuga/playbooks/playbook_status-format.md` and `.cuga/skills/web_research/SKILL.md` are
  git-tracked** (committed in `890927b`) and are LIVE CONTAMINATION SOURCES for every future run.
  I recommended deleting them; user has not decided. Backed up at
  `/var/folders/.../T/opencode/cuga-stale-backup/`. Also `.cuga/knowledge/` holds a tracked
  `favorite-color.md` + `project-clearance-code.md`.
- The stale `POL-8078061184` still exists in `cuga/dbs/cuga.db`; the reset clears it per-run but
  the file is regenerated. `CUGA_DBS_DIR` is read at import time (per-process only), so it cannot
  isolate candidates within one process.
- `.vscode/` untracked; gitignore question asked twice, still unanswered.
- `ToolObservationRecorder.wrap()` is dead on live CUGA (documented in prior handoff).
- `supports_counterfactual_replay()` must stay `False`.

## KEY FACTS TO NOT REDISCOVER

- Installed `cuga==0.3.1` (`cuga.__version__` misreports `0.2.20`). Model `openai/azure/gpt-5.6-luna`,
  balanced mode, forced autonomous, native sandbox, shell tool enabled.
- Workspace root is `data/workspaces/<task_id>` (NOT `data/cuga_workspaces`).
- `rg` is gitignore-aware and will NOT search `.venv`; use `grep -r` or `rg --no-ignore`.
- `materialize_harness` returns `None` when no skills/policies/memory present.
- `CausalTrace` uses `extra="forbid"`; static topology is a sidecar `graph-topology.json`.
- Core consumes `actor_id` at `analyzer.py:88-90` and `orchestrator.py:240` via `if e.actor_id`.
