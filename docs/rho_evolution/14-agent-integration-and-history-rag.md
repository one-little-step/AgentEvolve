# Agent Integration And History RAG

> This page is the focused adapter and history reference. For the full runner to
> population to Gaia control flow, candidate editor diagnostics, artifact map,
> and verified field findings, see
> [RHO-GEPA Architecture And Debugging Dossier](15-rho-gepa-architecture-and-debugging.md).

## Integration Boundary

The reusable RHO-GEPA core in `agent/evolution_core/` is Python-only and agent
neutral. An agent integrates by implementing the structural
`AgentEvolutionAdapter` protocol. The core owns population lifecycle but not an
agent's bundle format, rollout implementation, policy semantics, scoring, or
file editor.

```python
class AgentEvolutionAdapter(Protocol):
    agent_name: str
    module_names: tuple[str, ...]

    def load_bundle(self, version: str) -> EvolutionBundle: ...
    def materialize_bundle(self, bundle: EvolutionBundle, target: Path) -> None: ...
    def run_rollouts(
        self,
        bundle: EvolutionBundle,
        tasks: Sequence[NormalizedTrajectory],
        *, rollout_count: int, limits: RolloutLimits, artifact_dir: Path,
    ) -> Mapping[str, Sequence[NormalizedTrajectory]]: ...
    def score_rollouts(
        self,
        bundle: EvolutionBundle,
        tasks: Sequence[NormalizedTrajectory],
        rollouts: Mapping[str, Sequence[NormalizedTrajectory]],
        *, reference_bundle: EvolutionBundle | None,
        reference_rollouts: Mapping[str, Sequence[NormalizedTrajectory]] | None,
        artifact_dir: Path,
    ) -> Mapping[str, float | None]: ...
    def diagnose(
        self, tasks: Sequence[NormalizedTrajectory], parent: EvolutionBundle,
    ) -> Sequence[DiagnosisRecord]: ...
    def phase_evidence(
        self, trajectory: NormalizedTrajectory, module: str,
    ) -> Sequence[Mapping[str, object]]: ...
    def open_editor(self, candidate_dir: Path, candidate_id: str) -> CandidateEditor: ...
```

`EvolutionBundle(version, modules)` carries a version string and a mapping of
permitted module names to text. `NormalizedTrajectory` contains `task_id`,
`input_text`, `output_text`, `status`, tuple-valued dictionary `events`, and
optional source-path strings. `DiagnosisRecord` contains failure mode, root
cause, proposed fix, severity, phase, and evidence. `RolloutLimits` carries
rerun, per-task rollout, and global worker limits.

The editor protocol is deliberately narrow:

```python
append_section(filename, heading, content)
replace_section(filename, heading, content)
delete_section(filename, heading)
close()
```

The operator allows a filename only when it is a key in the parent or ancestor
bundle given to that operation. An adapter must still enforce its own editor
policy, because `module_names` determines the core's target-module cycle and
candidate reload set but is not passed directly into the operator's allowed-file
check. The current core calls `phase_evidence()` nowhere; adapters may provide
it for compatibility, but prompt evidence is selected directly from normalized
trajectory events.

## Gaia Mapping

`agent/gaia_lg_react/evolution/gaia_adapter.py` provides the current adapter.
It declares:

```python
agent_name = "gaia_lg_react"
module_names = (
    "intent_planner.md", "reAct.md", "critic.md", "consolidator.md",
    "scratchpad.md", "synthesis.md",
)
```

| Neutral operation | Gaia implementation |
| --- | --- |
| `load_bundle` | `WisdomBundle.load(wisdom_root, version)`, converted to `EvolutionBundle` |
| `materialize_bundle` | `WisdomBundle(...).materialize(target)` |
| `run_rollouts` | Builds `TrajectoryRecord` values and calls `EvolutionRound._run_rollouts()` with the supplied limits |
| `score_rollouts` | Calls `pairwise_preference()` for zipped reference/candidate rollouts and averages available normalized scores per task |
| `diagnose` | Calls `EvolutionRound._diagnose_selected()` and converts its entries to `DiagnosisRecord` |
| `phase_evidence` | Selects events whose `phase` equals the module stem or filename |
| `open_editor` | `WisdomEditRegistry.create(candidate_dir, candidate_id, True)` |

For a candidate directory, Gaia materializes the parent bundle before opening
the editor. During candidate rollouts, it uses the candidate directory as the
wisdom root only if every module in the bundle is present as a file there;
otherwise it uses the adapter's configured wisdom root. This allows the editor
to remain the write boundary while reusing `EvolutionRound` rollout machinery.

`GaiaEvolutionLLM` adapts the neutral `complete(system_prompt, user_prompt)`
call to Gaia's `LLMClient`: it emits one system and one user `LLMMessage` at
temperature `0.0` and returns an empty string when the client response has no
content.

In `dataset/evolve_run.py`, the GEPA branch creates `EvolutionRound`, then
`GaiaEvolutionAdapter`, `PopulationEvolution`, and `EditHistoryStore`. It loads
offline historical runs with `TrajectoryRunLoader`, selects a coreset through
the existing Gaia selector, and converts selected records to
`NormalizedTrajectory`. The legacy branch is separate and remains the path when
`GEPA_ENABLED` is false.

## Minimal Adapter Shape

An integrating agent needs a stable bundle mapping, an editor that writes only
declared policy files, and scoring that returns every requested task key when
possible. A minimal test adapter can look like this:

```python
class DemoAdapter:
    agent_name = "demo"
    module_names = ("policy.md",)

    def load_bundle(self, version):
        return EvolutionBundle(version, {"policy.md": "# Policy\n"})

    def materialize_bundle(self, bundle, target):
        target.mkdir(parents=True, exist_ok=False)
        for name, text in bundle.modules.items():
            (target / name).write_text(text, encoding="utf-8")

    # run_rollouts, score_rollouts, diagnose, phase_evidence, and open_editor
    # implement the protocol above for the agent's runtime and policy format.
```

Use a new immutable `version_root` for materialized elites/champion and a
separate artifact root for generation data. The core assumes an adapter can
later load the elite names it writes. See
[RHO-GEPA Population Evolution](13-rho-gepa-population-evolution.md) for the
lifecycle, naming, selection, and runner configuration.

## Edit-History Record Format

`EditHistoryStore(root, agent_name, ...)` stores records at:

```text
<root>/history/<agent-name>/records.jsonl
```

The current record schema is intentionally small:

```json
{
  "record_id": "2-g2-mutation-0-reAct_md",
  "lineage_id": "rho-g1-elite-1",
  "module": "reAct.md",
  "text": "mutation reAct.md score=0.25",
  "outcome": "helpful"
}
```

All five fields are required by `EditHistoryRecord`. Population persistence
writes one record per changed module; a no-change child writes one record for
`<generation>-<candidate-id>-<module>`, the lineage ID is the candidate
ancestor or candidate ID, text is `"<operator> <module> score=<average>"`, and
outcome is `helpful` when the average is non-negative (including `None` via the
current expression) and `harmful` otherwise. Before appending, the generic
store removes dictionary fields and inline key/value text involving `api_key`,
`token`, `secret`, `expected`, `evaluator`, `regex`, or `label`. It applies the
same redaction when loading legacy records, before embedding document/query
text, and when rendering the history packet for an LLM. The store does not
deduplicate, validate remaining field content, or classify `rejected`/`inconclusive`.

Appending reads the existing JSONL, writes the entire sequence to a uniquely
named temporary sibling file, then atomically replaces `records.jsonl`. A
crash before replacement leaves the old records file intact, but concurrent
writers are not synchronized and malformed existing JSONL causes reads to
raise. After each append, the store atomically writes `manifest.json` with
`schema_version`, `record_count`, and, when configured, `embedding_model`.

## Embedding Cache And Validation

Semantic retrieval caches each record under:

```text
<root>/history/<agent-name>/embeddings/<sanitized-record-id>.json
```

An embedding cache entry is:

```json
{
  "schema_version": "1",
  "record_id": "...",
  "text_sha256": "...",
  "embedding_model": "...",
  "dimension": 768,
  "vector": [0.1, 0.2]
}
```

Before reuse, the store validates schema version `"1"`, SHA-256 of the current
record text, embedding model identity, a non-empty list vector, and that every
vector item is numeric. A missing, malformed, stale, or model-mismatched cache
is regenerated through `embedder.embed_document()` and atomically replaced.
The regenerated vector must be non-empty.

`dimension` is written but is not checked against vector length during cache
reuse. The cache also does not validate that query and document vector lengths
match; cosine calculation uses `zip()`. These are current limitations, not
guarantees supplied by the format.

## Retrieval Cascade And Modes

`retrieve(query, lineage_id, module, minimum_records)` applies this cascade:

1. Select records matching both `lineage_id` and `module`.
2. If fewer than `minimum_records`, add all remaining records matching
   `module`, including records from other lineages.
3. If still fewer than `minimum_records`, add all remaining agent-scoped
   records regardless of module.

The population engine calls it with `minimum_records=1` and a query that combines
the target module with available diagnosis failure mode, root cause, proposed
fix, evidence, and phase. It therefore normally uses just the first non-empty
stage while ranking against meaningful failure context; the store supports
broader fallback for other callers.

With history retrieval disabled, result mode is `off` and no records are
returned. With retrieval enabled but semantic ranking disabled or no embedder,
mode is `lexical`: records sort by descending overlap between lowercased,
whitespace-split query terms and record text terms, then by record ID. With an
embedder, mode is `semantic`: records sort by descending cosine similarity,
then lexical overlap, then record ID. An embedding error changes that retrieval
to lexical and exposes the exception string as `fallback_reason`.

The LLM history packet separates `outcome == "helpful"`, outcomes `"harmful"`
is prompt presentation, not a separate retrieval filter.

## Ollama Configuration

`dataset/evolve_run.py` uses the standard Gaia config resolver when
`EDIT_HISTORY_SEMANTIC_ENABLED` is true. Relevant environment variables are:

```bash
export GAIA_SEMANTIC_ENABLED=true
export OLLAMA_EMBEDDING_URL=http://localhost:11434
export OLLAMA_EMBEDDING_MODEL=embeddinggemma
```

`GAIA_SEMANTIC_ENABLED` controls whether `resolve_embedding_provider()` attempts
to construct a provider. The resolver probes `<url>/api/embed`; it returns
`None` on any failure, printing a warning for a failed probe. The subsequent
history store then uses lexical ranking. Set
`EDIT_HISTORY_RETRIEVAL_ENABLED = False` for the `off` ablation, or leave
retrieval enabled and set `EDIT_HISTORY_SEMANTIC_ENABLED = False` for lexical
ranking without any Ollama call.

## Safety And Operational Notes

The system is offline policy evolution, not an online agent feature. The
generic core has no Gaia imports and no direct policy-file writer. Generic
history redaction removes the specified prohibited key/value fields before
history persistence, embedding, retrieval, and LLM history packets; operator
prompts also remove specified sensitive field names and truncate string values,
while the editor blocks raw arbitrary-write instructions. These protections do
not prove that non-history adapter rollouts, artifacts, or an editor
implementation cannot expose sensitive data.

For operation, ensure the initial Gaia wisdom version exists below
`WISDOM_ROOT`, all source runs exist below `DATASET_RUNS_ROOT`, and every later
generation's elite versions remain available under `WISDOM_ROOT`. Choose a
previously unused target prefix/generation or clean only intentionally discarded
outputs before retrying: existing target versions and generation directories are
errors by design.

Relevant tests are `tests/unit/test_evolution_history.py` for record storage,
history redaction across persistence, embedding, retrieval, and packets, mode
selection, cache creation, and semantic fallback;
`tests/unit/test_evolution_operators.py` for prompt safety and editor gating;
`tests/unit/test_gaia_adapter.py` for Gaia mapping; and
`tests/integration/test_evolution_core_population.py` for population lifecycle,
lineage, crossover, and manifest coverage. The tests use fake embedding
providers and do not validate live Ollama connectivity, concurrent history
writes, malformed-history recovery, or cache dimension mismatch handling.
