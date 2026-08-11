# 11 - Configuration Reference

All tunables live in dataset/evolve_run.py and dataset/batch_run.py.

## dataset/evolve_run.py

| Constant | Default | Effect |
|----------|---------|--------|
| SOURCE_RUNS | ["gaia_l1_validation_tiny5_20260723_055623"] | Historical runs used as the evolution corpus. All are combined each round. |
| INITIAL_HARNESS | "base" | Parent version for round 1. |
| TARGET_HARNESS_NAME_PREFIX | "rho-gaia" | Prefix for accepted version names. |
| ROUND_COUNT | 1 | Number of sequential rounds. Rejection stops the chain unless experimental promotion is on. |
| MODEL | "rits/openai/gpt-oss-120b-a100" | Model for rollouts, diagnosis, optimization, and judging. |
| JUDGE_MODEL | None | Sets the LLM client; currently also used for diagnosis and optimization. None means use MODEL. |
| SELECTOR | "dpp" | Coreset selector: dpp, random, difficulty, coverage. |
| THETA | 0.7 | DPP quality/diversity tradeoff. 0 = pure diversity, 1 = pure quality. |
| SCORE_FLOOR | 0.1 | Minimum normalized quality for DPP. Must be > 0. |
| SEED | 0 | Seed for stochastic selectors. DPP is deterministic. |
| CANDIDATE_COUNT | 1 | Number of candidate bundles generated per round. |
| OPTIMIZE_SAMPLES | CANDIDATE_COUNT | Must equal CANDIDATE_COUNT in this implementation. |
| CORESET_SIZE | 2 | Number of tasks selected for diagnosis and evaluation. |
| GROUP_ROLLOUTS_PER_TASK | 1 | Number of fresh parent and candidate rollouts per selected task. |
| MAX_RERUN_WORKERS | 4 | Maximum selected task groups admitted concurrently during one fresh evaluation phase. |
| MAX_ROLLOUT_WORKERS | 2 | Maximum concurrent repeated rollouts for one admitted task group. |
| GLOBAL_MAX_WORKERS | 6 | Hard cap on all concurrent fresh Gaia runs. Must be no greater than `MAX_RERUN_WORKERS * MAX_ROLLOUT_WORKERS`. |
| MAX_TRAJECTORY_WORKERS | 10 | Maximum parallel historical summary/embedding preparations. Valid range is 1 to 10; independent of fresh rollout workers. |
| TRAJECTORY_SUMMARY_CACHE_DIR | None | Optional override for summary cache root. Default is `<DATASET_RUNS_ROOT>/cache_trajectory_summaries`. |
| TRAJECTORY_EMBEDDING_CACHE_DIR | None | Optional override for embedding cache root. Default is `<DATASET_RUNS_ROOT>/cache_trajectory_embeddings`. |
| ACCEPTANCE_THRESHOLD | 0.0 | Minimum average pairwise score for normal promotion. |
| EVALUATION_TIMEOUT_SECONDS | 300.0 | Metadata only; no separate timeout is enforced. |
| CACHE_MODE | "off" | Only "off" is implemented. Other values raise. |
| CACHE_DIR | None | Reserved for future response-cache support. |
| ENABLE_WISDOM_EDITING | True | If False, candidates are copies of the parent. |
| EXPERIMENTAL_PROMOTE_CANDIDATE | False | Bypasses the acceptance gate; see 08-acceptance-and-promotion.md. |
| REPO_ROOT, DATASET_RUNS_ROOT, WISDOM_ROOT, EVOLUTION_ARTIFACT_ROOT | auto | Filesystem paths. Usually do not need editing. |

## dataset/batch_run.py

| Constant | Default | Effect |
|----------|---------|--------|
| DEFAULT_MODEL | env-driven | Model passed to each Gaia worker. |
| CONFIG_TEMPLATE | hardcoded YAML | Template for per-task gaia configs. |
| DEFAULT_EXPERIMENT_FILES | ["dataset/experiments/gaia_l1_validation_tiny5.json"] | Experiments to run. |
| EXPERIMENT_FILES | env or default | Active experiment list. |
| TARGET_DIR | "dataset/runs" | Root for batch run outputs. |
| MAX_WORKERS | 20 | Parallel worker processes. |
| CLEANUP_CHUNKS | False | If True, deletes per-task chunk folders after merging. |
| WISDOM_VERSION | None | Selects an evolved bundle. None = vanilla. |
| WISDOM_ROOT | "policies/evolved_context" | Root directory containing versioned bundles. |

## Trajectory-summary configuration

`Config.trajectory_summary` controls task-local summary generation. Its defaults
are a 10,000-token soft budget, 12,000-token hard budget, a 25,000-token target
for the optional failure-narrative packet, and a 30,000-token hard packet limit.
New inference summaries are task-local and naturally parallel under batch
workers. The optional narrative uses at most one LLM call and only for negative
or unresolved runs.

## Command-line usage

Evolution:

```bash
uv run python dataset/evolve_run.py
```

Batch evaluation:

```bash
uv run python dataset/batch_run.py
```

Single manual run:

```bash
uv run gaia -a gaia_lg_react --config path/to/config.yaml
```
# Web Provider Configuration

Run the local first-tier search service from the repository root:

```bash
cd searxng
docker compose up
```

Then configure the agent process with `SEARXNG_URL=http://localhost:8080`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `WEB_TRANSPORT_MAX_CONCURRENT_REQUESTS` | `8` | Process-wide cap for simultaneous outbound web requests. |
| `WEB_TRANSPORT_MAX_RETRY_AFTER_SECONDS` | `30` | Maximum per-host 429 cooldown in seconds. |
| `WEB_SEARCH_PROVIDERS` | `server,searxng,tavily,jina,ddgs` | Ordered provider chain. |
| `WEB_SEARCH_URL` | unset | Existing compatible search proxy. |
| `SEARXNG_URL` | unset | Local SearXNG base URL. |
| `TAVILY_API_KEYS` / `TAVILY_API_KEY` | unset | Comma-separated or single Tavily API keys. |
| `JINA_API_KEYS` / `JINA_API_KEY` | unset | Comma-separated or single Jina API keys. Each key is attempted before one keyless Jina fallback. |

Provider keys are read from environment variables, rotated safely under concurrent calls, and never written to output, artifacts, or tool errors. When Jina is reached without working configured keys, it makes one keyless request before falling through.
