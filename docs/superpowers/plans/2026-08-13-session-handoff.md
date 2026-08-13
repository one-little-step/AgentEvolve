# Session Handoff — 2026-08-13 (Phase 4.5 + Phase 6 complete)

This file is the durable resume point. It supersedes any earlier handoff.
Read `feedback/from_qwen/qf21.md` for the binding directive (skip Phase 5; do
Phase 4.5 cleanup then Phase 6 orchestrator).

## Git State

- Branch: `dev3`.
- Phase 1-4 research core committed/merged (`bc77a5f`). Tests green.
- Phase 4.5 and Phase 6 are committed (see commit list below).

### Committed this session
- Phase 4.5 cleanup (3 architectural gaps): `config.py`, `issues.py`,
  `pool.py`, `tests/test_issues.py`, `tests/test_pool.py`.
- Phase 6 sequential GEPA orchestrator + B1 runner: `core/embeddings.py`,
  `core/orchestrator.py` (`SequentialGepaRunner`), `examples/run_phase_6_b1.py`,
  `tests/test_embeddings.py`, `tests/test_phase_6_orchestrator.py`,
  `tests/test_phase_6_b1.py`.

### NOT committed (intentionally — prior-session CUGA work + runtime data)
- `src/agent_evolve/cuga_wrapper/__init__.py`, `tests/test_cuga_wrapper.py`,
  `src/agent_evolve/cuga_wrapper/tools.py`, CUGA tests/scripts/specs.
- `.cuga/`, `data/`, `reference/cuga_example_wrapper/`, `feedback/`, CUGA spec
  docs. These are runtime/experiment artifacts; never commit them.
- `.env` changes are hidden by `.gitignore` (must contain CUGA/LiteLLM config +
  `DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true`,
  `DYNACONF_ADVANCED_FEATURES__CUGA_LITE_NL_AUTO_CONTINUE=true`,
  `DYNACONF_KNOWLEDGE__ENABLED=true`, `DYNACONF_SKILLS__ENABLED=true`,
  `SEARXNG_URL="http://localhost:8080"`).

## Progress

### Done — Phase 4.5 cleanup (Task A of qf21), all three gaps closed
1. **ChampionReport** (`pool.py`): `select_champion()` returns a frozen
   `ChampionReport(entry, outcome, coverage, stability, regression_risk,
   aggregate, tie_breaker="ascending_candidate_id", disqualifications=())`.
2. **Min-coverage gate** (`pool.py` + `config.py`): `champion_min_coverage_fraction`.
3. **Frontier-weight wiring** (`issues.py`): `Issue.entropy_tier`,
   `raw_issue_quality(..., frontier_weight=0.30, entropy_tier=...)`, selector
   threads tier through `_final_raw_quality`.

### Done — Phase 6 (Task B of qf21): sequential orchestrator + B1 runner
- **`core/embeddings.py`** (new): `OllamaEmbedder` (real `/api/embed`, 768-dim,
  injectable transport, per-text cache, raises `EmbeddingProviderUnavailable`),
  `FallbackEmbedder` (records `fallback_reason`, never silent), and
  `build_embedder(config)`. NOTE: no Ollama embedder existed before; without it
  the DPP kernel sat permanently in `incompatible_embeddings` fallback.
- **`core/orchestrator.py`**: added `SequentialGepaRunner` (the target-correct
  loop `observe -> build_issues -> select_issues -> select_parent ->
  propose_edits -> validate -> commit_to_pool`), plus `GepaAttemptOutcome` and
  `GepaRunResult`. Uses the TARGET `issues.py` `Issue`/`HierarchicalDPPSelector`
  (NOT legacy `entropy.py`), synthesizes a trace-backed `CausalFinding` from the
  analyzer's actor-only blame (attributing the adapter writable set), seeded
  frequency-proportional parent sampling, and optional redacted JSON storage.
  The legacy `Orchestrator.run_iteration` is UNCHANGED.
- **`examples/run_phase_6_b1.py`** (new): `run_b1_experiment(seed, storage_root,
  n_attempts, embedder=None)` seeds base+c1+c2, records comparable scores, runs
  N sequential GEPA attempts, selects a champion with full manifest, persists
  redacted records, and verifies no evaluator token leaked. Defaults to the live
  `embeddinggemma` embedder; tests pass a `LexicalEmbedder(dim=32)` to stay
  offline and fast.

**Test count: 567 passed, 1 skipped, 1 warning** (`uv run pytest -p
no:cacheprovider`). Logs: `terminal_output/phase-6/full-suite.log` (567),
`terminal_output/phase-6/b1-smoke.log` (live embeddinggemma smoke).

### Blocked / Deferred
- Phase 5 `merge.py` / `parallel.py` — DO NOT touch.
- RHO outer-stage proposal generation — DO NOT build; use deterministic fake
  candidates from `_build_harness` in `examples/run_phase_6_b1.py`.
- CUGA tracing/checkpoints/replay/counterfactual branching — not Phase 6.

## Phase 6 Implementation Notes (verified — do not re-derive)

- Live Ollama `embeddinggemma:latest` at `http://localhost:11434`: `/api/embed`
  (and `/api/embeddings`) both work, 768-dim, already L2-normalized (norm=1.0),
  deterministic across calls.
- `FakeAnalyzerJudge.analyze` returns blame nodes with `artifacts=()`. The
  runner's `finding_from_analysis` attributes the adapter's writable set to the
  top-blame actor and lists each artifact as an `evidence_ref`, so `build_issue`
  accepts it and the DPP selector runs the joint quality+diversity objective.
- Parent sampling uses `random.Random(seed)` with `parent_frequencies()` mass;
  zero mass falls back to `pool.base`.
- `_persist_attempt` writes only references/decisions (`attempt_id`, `issue_id`,
  `parent/result_candidate_id`, `status`, `accepted`, `weighted_net_gain`,
  `reason`, `artifact_ids`, `mechanism_cluster_id`, `selection_fallback_reason`)
  — never task inputs, expected contracts, editor payloads, or traces.

## Next Steps
1. (Optional) Phase 5 `merge.py` / `parallel.py` — only if re-prioritized.
2. (Optional) RHO outer stage to seed real proposals.
3. (Optional) CUGA adapter tracing/checkpoints/replay.
4. Do not commit CUGA runtime data (`.cuga/`, `data/`, `reference/`).

## Relevant Files
- `feedback/from_qwen/qf21.md`: binding directive (Task A done, Task B done).
- `src/agent_evolve/core/embeddings.py`: Ollama + lexical fallback embedders.
- `src/agent_evolve/core/orchestrator.py`: `SequentialGepaRunner` + outcome/run result.
- `src/agent_evolve/core/issues.py`: target `Issue`, `build_issue`, `HierarchicalDPPSelector`.
- `src/agent_evolve/core/pool.py`: `ChampionReport`, `parent_frequencies`.
- `src/agent_evolve/core/storage.py`: `JSONFileStorage`.
- `src/agent_evolve/core/blame.py`: `CausalFinding`, `CausalAnalysis`.
- `src/agent_evolve/core/fake_editor.py`: `FakeEditor`.
- `src/agent_evolve/core/analyzer.py`: `FakeAnalyzerJudge`.
- `examples/fake_adapter.py`: `FakeAdapter`.
- `examples/run_phase_6_b1.py`: B1 experiment runner (`run_b1_experiment`).
- `tests/test_embeddings.py`, `tests/test_phase_6_orchestrator.py`, `tests/test_phase_6_b1.py`.
- `terminal_output/phase-6/full-suite.log`, `terminal_output/phase-6/b1-smoke.log`.
