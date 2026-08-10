# 10 - Runner and Batch Integration

This page explains how to run the evolution pipeline and how to evaluate any
produced version with the batch runner.

## Running evolution

Edit dataset/evolve_run.py, then run:

```bash
uv run python dataset/evolve_run.py
```

Key configuration constants:

```python
SOURCE_RUNS = ["gaia_l1_validation_tiny5_20260723_055623"]
INITIAL_HARNESS = "base"
TARGET_HARNESS_NAME_PREFIX = "rho-gaia"
ROUND_COUNT = 1
MODEL = "rits/openai/gpt-oss-120b-a100"
SELECTOR = "dpp"
CORESET_SIZE = 2
CANDIDATE_COUNT = 1
MAX_RERUN_WORKERS = 4
MAX_ROLLOUT_WORKERS = 2
GLOBAL_MAX_WORKERS = 6
EXPERIMENTAL_PROMOTE_CANDIDATE = True
```

Fresh parent and candidate reruns use a bounded shared scheduler. At most
`MAX_RERUN_WORKERS` selected task groups are admitted at once, and at most
`MAX_ROLLOUT_WORKERS` repeated rollouts of one task run simultaneously.
`GLOBAL_MAX_WORKERS` is the hard cap on every simultaneous Gaia rerun, so it
must satisfy:

```text
GLOBAL_MAX_WORKERS <= MAX_RERUN_WORKERS * MAX_ROLLOUT_WORKERS
```

The runner prints the round status and artifact directory:

```text
Round 1: base -> rho-gaia-1
  status: completed
  artifacts: dataset/runs/evolution/20260723_034742_dee5b1
```

## Evaluating a version in batch

Edit dataset/batch_run.py:

```python
WISDOM_VERSION = "rho-gaia-1"
WISDOM_ROOT = "policies/evolved_context"
```

Then run:

```bash
uv run python dataset/batch_run.py
```

The generated per-task config will contain:

```yaml
wisdom_version: rho-gaia-1
evolution:
  wisdom_root: policies/evolved_context
```

The merged result.json will contain:

```json
{
  "wisdom_version": "rho-gaia-1",
  "wisdom_root": "policies/evolved_context"
}
```

To use the vanilla base harness:

```python
WISDOM_VERSION = None
```

## How the agent loads a version

In agent/gaia_lg_react/runner.py:

```python
def _resolve_wisdom_bundle(cfg: Config) -> WisdomBundle | None:
    if not cfg.wisdom_version:
        return None
    root = Path(cfg.evolution.wisdom_root)
    bundle = WisdomBundle.load(root, version=cfg.wisdom_version)
    bundle.validate()
    return bundle
```

If cfg.wisdom_version is None, the agent runs without a wisdom bundle, using
the base policy files only.

If cfg.wisdom_version is set, the agent loads the six phase files from the
specified version directory and passes them into the graph as wisdom_bundle.

## How the graph uses wisdom

The wisdom_bundle is placed into the initial AgentState. Individual graph nodes
can read the phase files and incorporate them into prompts. The exact injection
points are outside the evolution package, but the contract is:

```text
intent_planner.md -> planning/intent extraction
reAct.md          -> reasoning/action loop
critic.md         -> candidate evidence review
consolidator.md   -> source-output briefing
scratchpad.md     -> goal-aware scratchpad compaction
synthesis.md      -> final answer synthesis
```

## Running a single task manually

Create a config YAML:

```yaml
agent_type: gaia_lg_react

agent:
  model: rits/openai/gpt-oss-120b-a100
  max_steps: 16
  timeout_seconds: 1000
  policy_path:
    - policies/default_policy.md
    - policies/anti_contamination.md
  tools: null
  telemetry:
    enabled: true
    exporter: file
    app_name: gaia-agent

name: my_manual_run

wisdom_version: rho-gaia-1

evolution:
  wisdom_root: policies/evolved_context

experiment:
  source: dataset/experiments/gaia_l1_validation_tiny5.json
  task_ids: ["gaia-e1fc63a2"]
  limit: null
```

Then:

```bash
uv run gaia -a gaia_lg_react --config config.yaml
```
