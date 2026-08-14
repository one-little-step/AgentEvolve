# Session Handoff — 2026-08-14 (Phase 6 done + qf22 fix pushed; Phase 7 next)

This is the durable resume point. Read this file FIRST after compaction/resume.
It supersedes `2026-08-13-session-handoff.md`.

## One-line current state

`dev4` @ `438a37b "phase-6 actor_id fix1"`, in sync with `origin/dev4`.
570 tests pass. Phase 4.5 + Phase 6 + qf22 Part I (synthetic-blame-node fix) are
committed and pushed. Next: Phase 7 (CUGA Tracing). qf22 Part II (delete legacy
Orchestrator) still pending.

## Git State (IMPORTANT: work is on dev4, not dev3)

- Branches: dev1, dev2, dev3, **dev4** (working). dev3 is at b0dbb06 (synced
  with origin/dev3). dev2 is ahead of origin/dev2 by 27.
- dev4 commit history (linear):
  - `b0dbb06` rho-gepa till phase-4 v1 (base)
  - `340d450` feat(core): phase 4.5 cleanup — ChampionReport, min-coverage gate, frontier-weight
  - `b4fba0c` feat(core): phase 6 sequential GEPA orchestrator + B1 runner (Ollama embeddings)
  - `438a37b` phase-6 actor_id fix1 (qf22 Part I fix + .db/.lock purge)
- Remote: `https://github.com/one-little-step/AgentEvolve`, branch dev4.
  HEAD == origin/dev4 (verified `git rev-list --left-right --count HEAD...origin/dev4` = `0 0`).
- **Discrepancy note:** the earlier handoff assumed dev3, but the Phase 4.5/6
  commits actually landed on dev4 (the working dir was on dev4). The user chose
  to push dev4. Do not "fix" this without asking.

### Working tree (uncommitted)
- `?? feedback/from_qwen/qf22.md` (untracked only). Everything else committed.

## What is committed & pushed (dev4)

1. **Phase 4.5** (3 gaps): `config.py` (+champion_min_coverage_fraction),
   `issues.py` (Issue.entropy_tier + frontier-weight wiring), `pool.py`
   (ChampionReport + min-coverage gate), `tests/test_issues.py`,
   `tests/test_pool.py`.
2. **Phase 6**: `core/embeddings.py` (NEW), `core/orchestrator.py`
   (SequentialGepaRunner + GepaAttemptOutcome + GepaRunResult),
   `examples/run_phase_6_b1.py` (NEW), `tests/test_embeddings.py`,
   `tests/test_phase_6_orchestrator.py`, `tests/test_phase_6_b1.py`.
3. **qf22 Part I**: `finding_from_analysis()` returns
   `status="insufficient_evidence"` (empty blame graph) when no actor is blamed;
   `build_issues()` skips `insufficient_evidence`; 3 new tests.
4. **.db/.lock purge**: removed `.cuga/knowledge/*.db`, `.cuga/knowledge/.lock`,
   `reference/cuga_example_wrapper/.cuga/knowledge/{metadata.db,.lock}` from
   history; added `.gitignore` rules `**/.cuga/knowledge/*.db` and
   `**/.cuga/knowledge/.lock`.

## Test state

`uv run pytest -p no:cacheprovider` → **570 passed, 1 skipped, 1 warning**.
(Note: `pyproject.toml` has `addopts = "-q"`; running `pytest -q` becomes
`-qq` and suppresses the summary line — use `-p no:cacheprovider` without `-q`.)
Logs: `terminal_output/phase-6/full-suite.log` (567, pre-fix),
`terminal_output/phase-6/full-suite-fixed.log` (570), `b1-smoke.log`.

B1 smoke: `OLLAMA_EMBEDDING_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=embeddinggemma uv run python examples/run_phase_6_b1.py`
→ seed 0/1/2: 4 attempts, 2 accepted, 2 rejected, pool 3→5, frontier 4,
coverage 1.00, embedding=ollama, redacted=True.

## Phase status

- Phase 1-4: committed (`bc77a5f` merged).
- Phase 4.5: DONE (committed 340d450).
- Phase 6 (sequential GEPA orchestrator + B1 runner): DONE (b4fba0c + 438a37b).
- qf22 Part I (synthetic blame node): DONE (438a37b).
- qf22 Part II (delete legacy Orchestrator): PENDING — see below.
- Phase 5 (merge/parallel): DEFERRED, do not touch.
- Phase 7 (CUGA Tracing): NEXT.

## Pending: qf22 Part II — delete legacy Orchestrator

Delete the legacy `Orchestrator` class at top of `core/orchestrator.py`, plus
`Profile`, `IterationResult`, and `MINIMAL`/`RESEARCH_SEQUENTIAL`/
`RESEARCH_PARALLEL`/`FULL_ABLATION` constants. The legacy class still has a
synthetic `BlameNode(actor_id="agent", blame=1.0, artifacts=())` at ~line 508
(its `run_iteration` fallback path). It uses deprecated `entropy.Issue`.
NOT truly dead code: `tests/test_orchestrator.py` (~20 tests) and
`examples/run_orchestrator_demo.py` import it — delete/rewrite those too.
`SequentialGepaRunner` does NOT depend on it. Do this via TDD; verify full suite.

## Phase 7 (CUGA Tracing) — starting context

- Goal: exact agent-state tracing, artifact provenance, optionally valid
  checkpoint replay (the reason CUGA is the reference adapter). CUGA SDK is a
  pinned dependency (`cuga>=0.3.1` in pyproject). Source NOT vendored — do not
  invent CUGA APIs/trace fields/artifact types/checkpoint behavior.
- `core/` must remain agent-neutral: never import `cuga` in core. Adapter
  boundary is `src/agent_evolve/adapters/` (abstract contract + future CUGA/Pi/
  Gaia adapters). Replay available only when an adapter reports a valid
  checkpoint/state-reconstruction capability.

## Critical architecture facts (verified — do not re-derive)

- **Two `Issue` classes**: legacy `entropy.Issue` vs target `issues.Issue`
  (issue_id, task_id, mechanism_cluster_id, severity, confidence, entropy,
  coverage_need, pareto_relevance, raw_quality, embedding,
  writable_artifact_ids, evidence_refs, lineage, entropy_tier). Phase 6 uses
  target.
- **Two selectors**: legacy `entropy.HierarchicalDPPSelector` (returns
  tuple[Issue]) vs target `issues.HierarchicalDPPSelector`
  (`select(issues, k=k) -> IssueSelectionReport` with `.items`/`.selected`).
  Phase 6 uses target (aliased in orchestrator.py as `TargetIssueSelector`,
  `TargetIssue`, `TargetIssueSelectionReport`, `build_target_issue`,
  `TARGET_THETA`, `TARGET_SCORE_FLOOR`).
- **Two `AcceptanceDecision`**: `editor.py` (accepted, status, reason,
  weighted_net_gain, protected_floors_violated) is what `decide_acceptance`
  returns; `evaluation.py` has a different shape (unused by orchestrator).
- `FakeAnalyzerJudge.analyze` returns blame nodes with `artifacts=()`. On
  FAILURE it returns nodes (actors from trace, e.g. actor "agent"); on SUCCESS
  returns `empty_analysis()` (no nodes). So `finding_from_analysis` normally
  sees non-empty `blamed`; the empty case is the defensive
  `insufficient_evidence` path.
- `CausalFinding` (pydantic, blame.py): required verdict_id, candidate_id,
  task_id, trace_id, status, rationale. `status="observed"` also requires
  mechanism_description, mechanism_cluster_id, severity, confidence,
  evidence_refs, and every blame-node artifact in evidence_refs. Other statuses
  (uncertain/insufficient_evidence/malformed) may omit those.
- `build_issue(finding, inventory, ...)` rejects (returns None) findings with no
  writable artifact attribution. Empty embedding forces DPP into
  `incompatible_embeddings` fallback (never silently).
- `FakeAdapter`: base artifacts `skills/retrieval`, `policies/execution`,
  `prompts/system` (all writable). Scores by substring match against
  `task.expected_contract["expected_substring"]`.
- `FakeEditor.propose_edit`: injects expected_substring into the highest-blame
  writable artifact (or write_set[0] on empty blame).
- `PersistentPool`: add_base, add_candidate, record_score(score, ScoreProvenance),
  parent_frequencies() (frequency = sum severity*confidence over winning cells),
  select_champion() -> ChampionReport (with .candidate_id alias), pareto_frontier.
- `SequentialGepaRunner` methods: `observe(entry, task) -> (trace, analysis)`,
  `finding_from_analysis(...)`, `build_issues(tasks)`, `select_issues(issues, k)`,
  `select_parent()`, `propose_edits(parent, issue, task, analysis, attempt_id)`,
  `validate(workspace, origin_task, regression_tasks)`,
  `commit_to_pool(parent, workspace, attempt_id, report, analysis)`,
  `run_attempt(tasks)`, `run(tasks, n_attempts)`. Fields: adapter, pool,
  analyzer_judge, editor, embedder, storage, config, mechanism_cluster_id="c0",
  seed=0, protected_floors=(), net_gain_threshold=0.0.
- Parent sampling: `random.Random(seed)` proportional to parent_frequencies;
  zero mass → `pool.base`. Runs via `_rng.random() * total` cumulative walk.
- `GepaAttemptOutcome`: attempt_id, issue_id, parent_candidate_id,
  result_candidate_id (None if rejected/no-issue), status (AttemptStatus),
  accepted, weighted_net_gain, reason, artifact_ids, fallback_reason.
  `GepaRunResult`: attempts, champion (ChampionReport|None), pool_size,
  pareto_frontier; props attempts_run/accepted_count/rejected_count/no_issue_count.

## Embeddings (new this phase)

- `core/embeddings.py`: `OllamaEmbedder(url, model, dim=768, timeout=30,
  transport=None)` — POST `/api/embed` with `{model, input}`, parses
  `embeddings[0]` or `embedding`, caches per text, raises
  `EmbeddingProviderUnavailable` on any transport/malformed/dim error.
  `FallbackEmbedder(primary, fallback)` records `fallback_reason` (never silent),
  requires matching dim. `build_embedder(config, dim, timeout, transport)` —
  provider "ollama"+fallback "lexical" → FallbackEmbedder; "lexical" →
  LexicalEmbedder(dim); "none" → bare OllamaEmbedder.
- Live service verified: `embeddinggemma:latest` at localhost:11434, `/api/embed`,
  768-dim, L2-normalized (norm=1.0), deterministic. `qwen3.5:0.8b` also present.
- `EmbeddingConfig` (config.py): url/model/provider/fallback. resolve_profile
  reads `OLLAMA_EMBEDDING_URL`/`OLLAMA_EMBEDDING_MODEL`.
- Unit tests stay OFFLINE: `OllamaEmbedder(transport=_RecordingTransport(...))`.
  Live test gated by env `AGENT_EVOLVE_LIVE_EMBEDDINGS=1`.

## Binding constraints / non-negotiables (AGENTS.md + qf21/qf22)

- `src/agent_evolve/core/` is agent-neutral: never import `cuga`/Gaia/runtime.
- TDD: write failing tests first, then implement. Run
  `uv run pytest -p no:cacheprovider`.
- Capture commands with `2>&1 | tee terminal_output/<topic>/<name>.log`.
- NEVER persist credentials, expected answers, evaluator internals, labels,
  regexes, raw prompts/responses/traces. `JSONFileStorage` redacts
  `expected_*`, label, regex, secret, token, raw_* fields.
- **qf22 mandate**: synthetic placeholder blame nodes are forbidden; absence of
  evidence = `status="insufficient_evidence"` (empty blame graph).
- Phase 5 merge.py/parallel.py: DO NOT TOUCH. RHO outer-stage proposal
  generation: deferred (use deterministic fake candidates from `_build_harness`).
- Do not commit unless explicitly asked.
- `.env` is gitignored; requires CUGA/LiteLLM config + `DYNACONF_*` vars +
  `SEARXNG_URL="http://localhost:8080"`.
- CUGA wrapper stays OUTSIDE core/ (`src/agent_evolve/cuga_wrapper/`), deferred
  imports, no top-level `import cuga`.
- `.gitignore` now ignores `**/.cuga/knowledge/*.db` and `**/.cuga/knowledge/.lock`.

## Key decisions / gotchas this session

- Earlier assumption "work on dev3" was wrong; commits are on dev4. Keep dev4.
- `.env.example` and `config/settings.openai.toml` are blank placeholders (no
  real secrets) — kept intentionally.
- `uv.lock` files (root + reference/) are dependency lockfiles — KEEP (dev).
- The GAIA datasets / terminal outputs under `reference/cuga_example_wrapper/`
  were deliberately KEPT at the user's instruction ("keep all dev things").
- `ctx_index`/context-mode KB writes have historically failed with
  "disk I/O error"; the durable handoff file is the reliable resume path.

## Relevant files

- `feedback/from_qwen/qf21.md` — binding directive (Phase 4.5 + 6; skip Phase 5).
- `feedback/from_qwen/qf22.md` — Part I done; Part II (legacy Orchestrator) pending.
- `src/agent_evolve/core/embeddings.py` — Ollama/Fallback/build_embedder.
- `src/agent_evolve/core/orchestrator.py` — SequentialGepaRunner (target) + legacy Orchestrator (to delete in qf22 Part II).
- `src/agent_evolve/core/issues.py` — target Issue/selector/build_issue.
- `src/agent_evolve/core/pool.py` — ChampionReport, parent_frequencies.
- `src/agent_evolve/core/blame.py` — CausalFinding/CausalAnalysis/BlameGraph/BlameNode.
- `src/agent_evolve/core/editor.py` — decide_acceptance, AcceptanceDecision, repair_once_then_classify.
- `src/agent_evolve/core/fake_editor.py`, `core/analyzer.py` — FakeEditor/FakeAnalyzerJudge.
- `src/agent_evolve/core/storage.py` — JSONFileStorage + redaction.
- `src/agent_evolve/core/config.py` — ResolvedConfig, resolve_profile.
- `examples/fake_adapter.py`, `examples/run_phase_6_b1.py`, `examples/run_phase_1_4_smoke.py`.
- `tests/test_embeddings.py`, `tests/test_phase_6_orchestrator.py`, `tests/test_phase_6_b1.py`.
- `tests/test_orchestrator.py`, `examples/run_orchestrator_demo.py` — to remove in qf22 Part II.
- `terminal_output/phase-6/` — full-suite-fixed.log (570), b1-smoke.log.
- AGENTS.md — non-negotiables + required reading order.

## Next steps (in order)

1. qf22 Part II: TDD-delete legacy `Orchestrator` + `Profile` + `IterationResult`
   + MINIMAL/RESEARCH_* + `tests/test_orchestrator.py` +
   `examples/run_orchestrator_demo.py`; verify 570→(570 - deleted tests) green;
   commit + push dev4.
2. Phase 7 (CUGA Tracing): read `docs/migration/cuga-sdk-integration-notes.md`,
   `docs/migration/cuga-adaptation-guide.md`,
   `reference/cuga_example_wrapper/docs/cuga-integration-learnings.md`; design
   tracing/provenance/checkpoint adapter boundary; TDD. Do NOT invent CUGA APIs.
3. Only after tests + adapters prove them, claim tracing/replay/parallel GEPA.
