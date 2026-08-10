# RHO-GEPA Population Evolution

> For the complete implementation map, end-to-end generation sequence,
> candidate-editor behavior, artifact inspection map, failure playbook, and
> improvement backlog, see
> [RHO-GEPA Architecture And Debugging Dossier](15-rho-gepa-architecture-and-debugging.md).

## Scope And Boundaries

`agent/evolution_core/` implements an offline, agent-neutral population
evolution loop. It owns generation directories, immutable version naming,
parent/child lifecycle, LLM operator dispatch, rollout-score collection, Pareto
selection, lineage sidecars, population manifests, and edit-history writes. It
does not import Gaia runtime code or write an agent's policy format directly.

An `AgentEvolutionAdapter` owns all agent-specific behavior: bundle loading and
materialization, rollout execution, scoring, diagnosis, phase evidence, and the
editor used by operators. The core creates a candidate directory and passes it
[Agent Integration And History RAG](14-agent-integration-and-history-rag.md)
for the exact adapter and Gaia implementation.

The population path is opt-in. `dataset/evolve_run.py` uses the existing
sequential `EvolutionRound` loop when `GEPA_ENABLED = False`; it constructs
`PopulationEvolution` only when `GEPA_ENABLED = True`.

## Generation Data And Control Flow

`PopulationEvolution.run_generation()` takes an initial version, output prefix,
sequence of `NormalizedTrajectory` tasks.

1. It validates `generation >= 1`, `elite_count >= 1`,
   `offspring_count >= elite_count`, and `0 <= crossover_count <= offspring_count`.
2. It preflights every elite and champion target, then rejects an existing
   `<artifact-root>/evolution/g<generation>` directory. Neither artifacts nor
   versions are overwritten.
3. Generation 1 loads only `initial_version`. Later generations load
   `<prefix>-g<generation - 1>-elite-<rank>` for every requested rank.
4. Every retained parent is rolled out and scored before children are created.
   Parents have no reference bundle, so the Gaia adapter returns `0.0` for each
   requested task; this is a baseline convention, not an inferred quality
   result.
5. The engine reserves the first `crossover_count` child slots. An eligible
   pair consists of two retained candidates whose lineage sidecars name the
   same non-null `ancestor`. Eligible pairs are cycled in deterministic list
   order. If there is no eligible pair, or a crossover call raises, the slot is
   filled by mutation instead.
6. Remaining slots are mutations. Parent selection cycles through retained
   parents, and the target module cycles through `adapter.module_names` using
   the child index. `offspring_count` is exactly the number of new children;
   parents are additional evaluated candidates.
7. Each child is rolled out, then scored against its direct mutation parent or
   the left crossover parent. The adapter returns one `float | None` score per
   requested task. `None` is preserved for unavailable judgments and excluded
   from the candidate average.
8. Parents and children are selected together. The non-dominated frontier is
   ranked by descending available-score average and candidate ID; if it has
   fewer than `elite_count` entries, dominated candidates fill the remainder by
   that same ordering. The first selected elite is the champion.
9. Selected bundles are materialized as immutable elites and as a separate
   champion copy. Child history records are appended, then `population.json` is
   written.

The core's Pareto relation compares only task IDs shared by both candidates for
which both scores are available. One candidate dominates another when all such
scores are at least as large and at least one is larger. A candidate with no
comparable scores does not dominate another candidate.

## LLM Operators And Editor Safety

Mutation and crossover both use the neutral two-string protocol:

```python
class EvolutionLLM(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str: ...
```

Each operator makes one `complete()` call and requires model output containing
JSON with an `edits` list. An edit has `operation`, `filename`, `heading`, and
optional `content`. Supported operations are `append_section`,
`replace_section`, and `delete_section`.

`filename` must be a module in the bundle supplied to the operator. The core
never accepts paths, raw file contents, shell commands, or arbitrary writes.
It delegates accepted operations to the adapter editor and always calls
`editor.close()`. Invalid JSON, a non-list `edits` value, unsupported
operations, disallowed modules, or an editor exception are recorded as skipped
edits. An LLM exception at the population layer creates an evaluated no-op
mutation child; a crossover exception causes that reserved slot to fall back to
mutation. A syntactically valid no-edit response is also evaluated normally.

The mutation prompt contains the complete parent module mapping, selected target
module, sanitized diagnoses, up to 20 sanitized events for that module, and a
history packet. The crossover prompt contains the ancestor mapping, both parent
mappings, both per-task score mappings, sanitized diagnoses, up to 30 sanitized
events from all supplied trajectories, and the same history packet. The prompt
sanitizer drops dictionary keys containing `api_key`, `token`, `secret`,
`expected`, `evaluator`, `regex`, or `label`, recursively processes lists, and
truncates string values to 4,000 characters. Independently, the generic history
store removes those prohibited fields and inline key/value text before record
persistence, query embedding, retrieval, or history-packet rendering. This
history guarantee does not sanitize non-history adapter inputs or artifacts.

## Scoring And Selection

The core calls `adapter.run_rollouts()` with the configured number of rollouts
and `RolloutLimits(rerun_workers, rollout_workers, global_workers)`. It caches
rollouts by bundle version for the lifetime of the `PopulationEvolution`
instance. It then calls `adapter.score_rollouts()` and normalizes the result to
the supplied task IDs, preserving absent scores as `None`.

The reusable core does not calculate terminal-status quality, pairwise
preference, changed-module evaluations, acceptance thresholds, or global
per-task scores itself. Those are adapter responsibilities. Gaia compares
paired candidate and reference rollouts with `pairwise_preference`, averaging
available normalized judgments for each task; a task with no available
comparison is `None`. Its parent baseline scores are `0.0` per task. There is
no generic module-scoring call in `PopulationEvolution`.

## Artifacts, Versions, And Lineage

Given prefix `rho`, generation `2`, and two elites, materialized versions are:

```text
<version-root>/rho-g2-elite-1
<version-root>/rho-g2-elite-2
<version-root>/rho-g2-champion
```

Every materialized elite and champion receives
`.rho-gepa-lineage.json`:

```json
{
  "ancestor": "...",
  "candidate_id": "...",
  "parents": ["..."],
  "schema_version": "1"
}
```

The sidecar is private lineage metadata used by later crossover eligibility. A
missing, unreadable, or structurally invalid sidecar is treated as
`{"ancestor": <version>, "parents": []}`. The reader validates only that the
stored ancestor is a string and that parents is a list of strings.

Each generation writes:

```text
<artifact-root>/evolution/g<generation>/
  parents/<bundle-version>/...
  candidates/<candidate-id>/...
  population.json
```

Candidate and rollout sub-artifacts are adapter-defined. `population.json`
contains the adapter name; generation configuration; source parent versions;
child count; history mode, path, and fallback reasons; every parent and child
candidate's identity, parent IDs, ancestor, operator, changed modules, task
scores, average score, and artifact directory; selected IDs and paths; champion
details; and accumulated operator errors. Paths are serialized as strings.

History lives under `<artifact-root>/history/<agent-name>/`; its exact layout
and retrieval behavior are documented in
[Agent Integration And History RAG](14-agent-integration-and-history-rag.md).

## Running Gaia

Edit `dataset/evolve_run.py`, then run from the repository root:

```bash
uv run python dataset/evolve_run.py
```

For GEPA, configure at least:

```python
GEPA_ENABLED = True
INITIAL_HARNESS = "base"
TARGET_HARNESS_NAME_PREFIX = "rho-gaia"
ROUND_COUNT = 5
ELITE_COUNT = 3
OFFSPRING_COUNT = 6
MERGE_OFFSPRING_COUNT = 1
GROUP_ROLLOUTS_PER_TASK = 3
```

`ROUND_COUNT` becomes the number of generations. The runner loads every entry
in `SOURCE_RUNS`, selects one coreset once with `CORESET_SIZE`, `SELECTOR`,
`THETA`, `SCORE_FLOOR`, and `SEED`, and reuses that task tuple for all
generations. `OPTIMIZE_SAMPLES` must equal `CANDIDATE_COUNT` even though the
GEPA branch does not pass `CANDIDATE_COUNT` into `run_generation()`.

The runner enforces `ELITE_COUNT >= 1`, `OFFSPRING_COUNT >= ELITE_COUNT`,
`0 <= MERGE_OFFSPRING_COUNT <= OFFSPRING_COUNT`, worker counts of at least one,
`CACHE_MODE` must remain `"off"`; response caching is not implemented. The
legacy-only settings such as acceptance threshold and experimental promotion do
not alter the GEPA branch.

## History Ablations And Ollama

The GEPA runner passes these independent controls to `EditHistoryStore`:

```python
EDIT_HISTORY_RETRIEVAL_ENABLED = True
EDIT_HISTORY_SEMANTIC_ENABLED = True
```

The actual manifest history mode is:

| Retrieval | Semantic | Result |
| --- | --- | --- |
| `False` | either | `off`; no records are returned to operators |
| `True` | `False` | `lexical`; deterministic term-overlap ranking |
| `True` | `True` with usable provider | `semantic`; cosine ranking then lexical and record-ID ties |
| `True` | `True` with missing/failing provider | `lexical`; fallback reason is collected when a retrieval embedding call fails |

`dataset/evolve_run.py` calls `load_config()` and resolves an embedding provider
only when semantic history is enabled. `resolve_embedding_provider()` first
probes Ollama by embedding a fixed startup string. It uses
`OLLAMA_EMBEDDING_URL` and `OLLAMA_EMBEDDING_MODEL`, whose defaults are
`http://localhost:11434` and `embeddinggemma`. The provider posts
`{"model": ..., "input": ...}` to `<base-url>/api/embed` with a 10-second
timeout. A failed startup probe returns no provider, so the history store uses
lexical retrieval; this specific no-provider case has no per-retrieval fallback
reason.

## Troubleshooting And Limits

| Symptom | Current behavior and action |
| --- | --- |
| `FileExistsError` before work begins | Remove or choose new immutable elite/champion names and a new generation artifact root; the engine deliberately does not resume or overwrite. |
| No crossover children | Generation 1 has one parent. Later generations need at least two retained candidates with the same lineage `ancestor`; otherwise every reserved slot mutates. |
| Crossover slot mutated | The selected crossover raised an exception. Inspect `errors` in `population.json`; the fallback mutation is expected. |
| All task scores are `null` | Gaia found no available pairwise judgments. The average is `null`; ranking then uses candidate ID among otherwise tied candidates. |
| History unexpectedly lexical | Confirm both ablation flags, `GAIA_SEMANTIC_ENABLED`, Ollama reachability, `OLLAMA_EMBEDDING_URL`, and `OLLAMA_EMBEDDING_MODEL`. Inspect manifest fallback reasons for retrieval-time errors. |
| Candidate has no changes | Inspect its raw candidate directory and manifest. Invalid output, rejected edits, unavailable LLM calls, and empty edit lists all remain evaluated no-op candidates. |

Known limitations of the current implementation:

- `EvolutionBundle` is frozen as a dataclass but contains a mutable `dict`.
- The core does not call `phase_evidence()`; it filters the supplied trajectory
  events itself by phase name.
- Crossover eligibility only compares the immediate stored `ancestor` values;
  it does not search an arbitrary lineage graph or exclude direct siblings
  beyond that equality rule.
- The `minimum_records` value used by population retrieval is fixed at `1`.
- History retrieval queries combine the target module with available diagnosis
  failure mode, root cause, proposed fix, evidence, and phase; they do not use
  arbitrary raw trajectory content.
- JSONL history loading does not recover from malformed record lines.
- Selection does not use module-level scores, and history outcomes are based on
  non-negative average score rather than selected-elite status.
- The core does not persist raw operator output or skipped-edit details in
  `population.json`.

## Test Coverage

Focused coverage is in `tests/unit/test_evolution_core_contracts.py`,
`tests/unit/test_evolution_operators.py`, `tests/unit/test_evolution_history.py`,
`tests/unit/test_evolution_population.py`,
`tests/integration/test_evolution_core_population.py`,
`tests/unit/test_gaia_adapter.py`, and the GEPA dispatch tests
`tests/unit/test_evolve_run.py` and `tests/unit/test_evolve_run_gepa_dispatch.py`.
They cover neutral contracts, editor gating and prompt context, history modes and
fallback, generation-one and generation-two lifecycle behavior, immutable
targets, lineage-driven crossover, Gaia mapping, and runner dispatch. They use
fake adapters/LLMs and do not perform a live Ollama request or a production
Gaia rollout.
