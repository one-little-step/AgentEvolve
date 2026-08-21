# SV-12 / Clustering — Compaction Anchor

**Written 2026-08-20 immediately before a context compaction. Revised 2026-08-20
(§5 corrected, §§9-12 design added) and again 2026-08-21** for the second
compaction, after step 3 completed, the first live model calls were made, and step
4 was re-scoped. Every fact below was re-derived mechanically, not recalled. Where
a line says *measured*, the command is given (§6) so a fresh reader can re-run it
rather than trust it.

**Why this file exists.** In the three turns before it, I stated three mutually
contradictory things about `EntropyTracker` — unwired, then wired, then write-only.
Only the last is true. Then §5 itself turned out to be wrong: it asserted a global
score-before-diagnose ordering constraint that holds only for RHO. A fresh agent
must not inherit either narrative. Re-run the commands.

**Reading order — 2026-08-21.**

1. §1 for state.
2. **`docs/design/issue-lifecycle.md`** — the authoritative design now. Whole issue
   lifecycle, mermaid, module map, decisions D1-D4 with live measurements.
3. §15 for what changed on 2026-08-21 (step 3 done, env defect, live calibration).
4. §16 for the next task (step 4, re-scoped).
5. §5 for the corrected ordering rule; §11 for build order and the **pool/tracker
   key separation — never merge those two keyspaces**.
6. §7 and §10 are the traps. §§9-10 measurements are the original evidence trail —
   read their STATUS notes, not just the numbers.

---

## 1. State at compaction

| Fact | Value | How to re-derive |
| --- | --- | --- |
| Branch / HEAD | `dev7` / `8d48a8f` | `git rev-parse --short HEAD` |
| Working tree | **33 modified, 32 untracked** | `git status --porcelain` |
| Suite | **2106 collected: 2105 passed, 1 skipped, 0 failed** | see §6 for the counting command |
| `core/entropy.py` diff | **0 lines — do not modify without asking** | `git diff --numstat src/agent_evolve/core/entropy.py` |
| Core purity | 35 files, 0 forbidden imports | AST scan, §6 |
| Nothing committed | all work is in the working tree | — |

**SV-12 IS CLOSED. All four steps of §11 are DONE and green**, plus the fallback-rate
aggregation. Suite reconciles as `2040 + 12 + 10 + 17 + 6 + 9 + 12 = 2106 collected` (2105 passed, 1 skipped):

| File | Tests | What it pins |
| --- | --- | --- |
| `tests/test_dedup_band_defaults.py` | 12 | step 4: the band, the four-defaults collapse, the `base_url`/`url` defect |
| `tests/test_correlation_context.py` | 10 | SV-7: the `X-AE-*` correlation scope |
| `tests/test_correlation_headers_wired.py` | 17 | SV-7: all four adapter wrappers emit it |
| `tests/test_sv7_materialization_distinctness.py` | 6 | SV-7: upstream materialization exonerated |
| `tests/test_entropy_availability_report.py` | 9 | the fallback-rate arithmetic |
| `tests/test_entropy_availability_wired.py` | 12 | that the runner and pipeline actually populate it |

Earlier files from steps 1-3: `test_embedder_wiring.py` (15),
`test_mechanism_adjudicator.py` (13), `test_cuga_mechanism_adjudicator.py` (27),
`test_genetic_entropy_tracker.py` (13), `test_env_reaches_config.py` (4).

**Step 4 is the next task, RE-SCOPED** — see §16. It is no longer "task-agnostic
mechanism identity"; it is widening the dedup adjudicator band so analyzer
paraphrase stops fragmenting inside a single task.

**Open issues: SV-7 (live-proxy, user APPROVED) and SV-12 (this file, step 4).**
Closed: SV-2, SV-3, SV-4, SV-5, SV-6, SV-8, SV-9, SV-10, SV-11, SV-13.
SV-1 is RECLASSIFIED (a third state — not open, not closed).

**Live models are now configured and working** (user set these in `.env`; dev
credentials, fine to use):

| Role | Model | Verified |
| --- | --- | --- |
| Mechanism embedding | `embeddinggemma` via Ollama `localhost:11434` | 12 live embeds, §15.2 |
| Dedup adjudicator | `openai/aws/gpt-oss-120b` via IBM LiteLLM | 6 of 6 correct, §15.2 |

---

## 2. The corrected picture: two half-wired entropy paths

This is the load-bearing section. Measured, not recalled.

```
                 writes tracker   reads tracker   own entropy        does DPP issue selection
  RHO path       YES (ph 5, 8)    NO              --                 NO
  GENETIC path   NO               NO              _cell_entropy      YES
```

**RHO is a write-only entropy sink.** It pays for the bookkeeping and no phase
consumes the number.

Write path — each hop below was resolved to a real call site by ripgrep over `src/`
and `scripts/`; coverage is the call chain only, and excludes any live run:

```
scripts/run_evolution.py:1048   tracker = EntropyTracker()      <- the ONLY construction site
  -> rounds.py:343              run_round(..., tracker=tracker)
      -> rounds.py:473          _record_scores(base)        phase 5  group_rollouts
      -> rounds.py:524          _record_scores(candidate)   phase 8  candidate_rollouts
          -> rounds.py:717      tracker.record_score(task_id, cluster, version, value)
          -> rounds.py:719      tracker.mark_comparable(...)   only once the candidate
                                                               clears min_rollouts_per_candidate
```

**Read path: none.** All six public read methods have **zero** callers anywhere in
`src/` or `scripts/` outside the defining module:

```
.cell_entropy()  .entropy()  .classify()
.top_entropy_cells()  .all_cells()  .entropy_weighted_with_freshness()
                                            -> 0 callers each
```

**The genetic path never touches the tracker.** `self.entropy` exists as a field
(`orchestrator.py:227`, `default_factory=EntropyTracker`) and has exactly **one**
use in the whole file:

```
orchestrator.py:531   self.entropy.refresh_at_barrier(self._iteration)
```

— and line 531 is inside `run_iteration`, which is **dead code** (§3). `pipeline.py`
never supplies a tracker, so the genetic runner's tracker is default-constructed and
never touched. Instead the genetic path recomputes variance inline:

```
orchestrator.py:1541   entropy=self._cell_entropy(rollout.task.task_id)
orchestrator.py:1546   entropy_tier=self._entropy_tier(rollout.task.task_id)
orchestrator.py:1616   def _cell_entropy(...)   <- filters m_id == self.mechanism_cluster_id
```

---

## 3. `run_iteration` is dead code — and it holds the only clusterer wiring

```
orchestrator.py:222   cluster_registry: ClusterRegistry = field(...)   <- declaration
orchestrator.py:530   self.cluster_registry.begin_iteration(...)       \  both inside
orchestrator.py:544   self.cluster_registry.clusterer_for(task_id)     /  run_iteration()
```

`rg 'run_iteration\(' src/ scripts/` returns **only its own `def` at :510**. Zero
callers. Production uses `run_attempt`.

**Consequence: `MechanismClusterer` is a fully implemented, independently tested
component wired only into dead code.** `tests/test_clustering.py` holds 24 tests, all
passing — coverage is the clusterer's own join/spawn/anchor behaviour, and it does not
touch the production scoring path, which is precisely the gap. It is not broken; it is
unreached.

(Do not confuse with `stack.run_iterations(...)` in `pipeline.py`/tests — different
name, different thing, genuinely live.)

---

## 4. Why the variance is not what the spec defines

Spec (`docs/architecture/selection-algorithms.md:37`):

```
H(t, m) = variance * max(max_score, GEPA_ENTROPY_SCORE_FLOOR)
floors: comparable candidates >= 3, rollouts per candidate >= 2
```

`_cell_entropy` filters on `m_id == self.mechanism_cluster_id`, and that is a
**constant**:

```
pipeline.py:143    DEFAULT_MECHANISM_CLUSTER = "mechanism-default"
pipeline.py:977    mechanism_cluster_id=DEFAULT_MECHANISM_CLUSTER
pipeline.py:1167   mechanism_cluster_id=DEFAULT_MECHANISM_CLUSTER
```

Measured over `mechanism_cluster_id=` at every pool-write site in `orchestrator.py`:
**5 sites pass the constant `self.mechanism_cluster_id`; 1 site (`:345`) passes a
real `cluster_id`** — and `:345` is reached only from dead `run_iteration`.

So the variance is **the spread of one mean score per candidate inside a single
synthetic bucket**, pooling candidates that failed for unrelated reasons. That is
cross-candidate score spread, not per-mechanism variance.

**The user's framing, which corrected mine:** clustering is not for parent/candidate
identity. It exists so the DPP issue/mechanism selector has a well-defined grouping
to compute variance *within*. With LLM non-determinism, run-to-run score noise is
otherwise **indistinguishable from genuine mechanism diversity**, so the DPP
diversity term may be steering on sampling noise.

---

## 5. The ordering constraint — RHO ONLY (this section was wrong before)

**Correction, measured 2026-08-20.** An earlier revision of this section claimed
"you score before you diagnose" as a *global* constraint and proposed retroactive
re-filing to work around it. That is true for RHO and **false for the genetic
path**. The wrong version argued against the simple fix and pointed at the
complicated one, stated as confidently as the parts that are true.

### Where it holds: RHO

`rounds.py:86-96`, `rho_cluster_id` returns `f"rho-task:{task_id}"` — task-local but
**not** mechanism-local, deliberately:

> Base rollouts happen in **phase 5** (`group_rollouts`), before any diagnosis exists
> (**phase 6**, `group_diagnosis`), so a diagnosis-derived cluster id would file base
> evidence in a different cell from candidate evidence and
> `min_comparable_candidates` could never be met.

The 10 RHO phases, in order — note **none** is an entropy or DPP issue-selection
phase; RHO picks its coreset by difficulty fingerprint:

```
history_load, trajectory_comprehension, difficulty_fingerprint,
coreset_selection, group_rollouts, group_diagnosis,
candidate_proposal, candidate_rollouts, preference_judging, pool_commit
```

### Where it does NOT hold: the genetic path

`_rollout` (`orchestrator.py:296-330`) runs the analyzer *inside* the rollout and
returns trace and diagnosis **as one pair**:

```python
def _rollout(self, workspace, task, rollout_id) -> tuple[ExecutionTrace, CausalAnalysis]:
    result   = self.adapter.run_full_rollout(workspace, task, rollout_id)
    trace    = self.adapter.capture_trace(result)
    if self.profile.use_causal_blame:
        analysis = self.resolved_analyzer.analyze(task, trace)   # diagnosis HERE
    ...
    return trace, analysis                                       # TOGETHER
```

`ObservedRollout.analysis` (`evaluation.py:251`) is therefore already populated
before scoring. Measured by monkey-patching `_record_rollout_score` during a real
`run_attempt` and inspecting the rollout at the instant of recording:

```
record-time observations (task, analysis_present, has_blame_graph, passed):
    ('task-a', True, True, False)
    ('task-b', True, True, False)
RESULT: 2/2 score-record calls had a diagnosis ALREADY attached
```

**Consequence: on the genetic path the mechanism can be keyed correctly on the
first write.** No retroactive re-filing, no barrier bookkeeping, no provisional
keys. And because `EntropyTracker.record_score` is a plain
`dict[CellKey, _Cell]` insert, mechanism-keyed cells need **no change to
`entropy.py`** — the 0-diff constraint (§1) holds.

---

## 6. Commands (copy-paste; several are traps if done naively)

```bash
# Suite. `-q` suppresses the summary line on this machine, so count collection
# and grep FAILED. Do NOT regex for "N passed" — it reports 0 and looks catastrophic.
python -m pytest -p no:warnings --no-header -q --tb=short 2>&1 | tail -20 > /tmp/s.txt
grep -c '^FAILED' /tmp/s.txt          # expect 0
python -m pytest -p no:warnings --co -q 2>/dev/null \
  | awk -F': ' '/: [0-9]+$/{s+=$NF} END{print s}'   # expect 2106 (2105 pass + 1 skip)
```

### 6.0 The 2026-08-21 checks (run these too)

```bash
# Env now actually reaches the config. Was the defect of §15.2.
python3 -c "
import sys; sys.path.insert(0,'src')
from dotenv import load_dotenv; load_dotenv('.env')   # PASS THE PATH, see §15.2 trap
from agent_evolve.core.config import resolve_profile
c=resolve_profile('research_sequential',seed=0); d=c.mechanism_dedup
print('dedup enabled',d.enabled,'model',d.model)       # expect True, gpt-oss-120b
print('embed',c.embedding.url,c.embedding.model)       # expect localhost:11434 embeddinggemma
print('key in manifest?', bool(d.api_key) and d.api_key in repr(c.manifest_payload()))  # expect False
"

# Entropy value and tier must come from the SAME cell (§15.1 defect).
python -m pytest tests/test_genetic_entropy_tracker.py -p no:warnings -q 2>&1 | tail -3

# Live embedder calibration — reproduces the OVERLAP finding of §15.3.
# Needs Ollama up. ~2s. Both scripts are kept next to their logs.
python3 terminal_output/calibration/embedder_calibration.py  # expect separation -0.036
python3 terminal_output/calibration/adjudicator_probe.py     # expect 6/6 (LiteLLM creds needed)

# Dead read API: which EntropyTracker read methods still have no caller?
for m in cell_entropy entropy classify top_entropy_cells all_cells \
         entropy_weighted_with_freshness; do
  printf '%-34s %s\n' ".$m(" \
    "$(rg -c "\.${m}\(" src/ scripts/ --glob '!entropy.py' --glob '!__pycache__' 2>/dev/null | wc -l | tr -d ' ')"
done
# expect: entropy/classify/all_cells have callers; the other three are 0
```

```bash
# Core purity — use AST, not ripgrep. `rg 'adapters'` false-positives on prose.
python -c "
import ast,pathlib
bad=[]
for p in pathlib.Path('src/agent_evolve/core').rglob('*.py'):
    for n in ast.walk(ast.parse(p.read_text())):
        if isinstance(n,(ast.Import,ast.ImportFrom)):
            mods=[a.name for a in n.names] if isinstance(n,ast.Import) else [n.module or '']
            for m in mods:
                if any(k in (m or '') for k in ('cuga','litellm','adapters')): bad.append((p.name,m))
print(bad or 'clean')"

# The decisive read-consumer check that settles what the tracker is for
for m in cell_entropy entropy classify top_entropy_cells all_cells; do
  echo -n "$m: "; rg -c "\.$m\(" src/ scripts/ --type py 2>/dev/null \
    | grep -v 'core/entropy.py' | wc -l
done

# Reproduce the starvation (offline; note the confound in §7)
python - <<'EOF'
import sys; from pathlib import Path
R=Path('.').resolve(); sys.path[:0]=[str(R/'src'),str(R),str(R/'tests')]
from test_phase_6_orchestrator import _runner,_task
r=_runner(seed=0); t=(_task("task-a"),_task("task-b"))
for _ in range(6): r.run_attempt(t)
cells={}
for e in r.pool.all_entries():
    for k,c in e.score_tensor.items(): cells.setdefault(k,[]).append(c.rollout_count)
for k,v in sorted(cells.items()):
    print(k, "candidates=",len(v), "comparable(>=2)=",sum(1 for n in v if n>=2), "floor=3")
EOF

# §9.1 The embedder WAS unwired (step 1 fixed it). Expect NOW:
#   build_embedder has exactly one caller: pipeline.embedder_for_config
#   the dim=32 hits at the two stack-build sites REMAIN (adjudicator not wired yet)
#   cluster_registry still absent from pipeline.py
rg -n 'build_embedder' src/ scripts/ --type py | rg -v 'core/embeddings.py'
rg -n 'LexicalEmbedder' src/agent_evolve/pipeline.py
rg -n 'cluster_registry' src/agent_evolve/pipeline.py     # expect NO hits

# Steps 1-2 regression check. Expect 55 passed, 0 failed.
python -m pytest tests/test_embedder_wiring.py \
  tests/test_mechanism_adjudicator.py \
  tests/test_cuga_mechanism_adjudicator.py \
  -p no:warnings --no-header -q --tb=short

# The at_cap fix (§10.3). Expect cluster_id='' and a reason naming the cap.
python - <<'EOF'
import sys; from pathlib import Path
R=Path('.').resolve(); sys.path[:0]=[str(R/'src'),str(R),str(R/'tests')]
from agent_evolve.core.clustering import MechanismClusterer, LexicalEmbedder
from agent_evolve.core.blame import CausalAnalysis, BlameGraph, BlameNode
def A(m): return CausalAnalysis(mechanism=m, severity=0.6, score=0.2,
    blame_graph=BlameGraph(nodes=(BlameNode(actor_id="agent", artifacts=("skills/a.md",), blame=0.9),)))
cl=MechanismClusterer(task_id="t", embedder=LexicalEmbedder(dim=768),
                      max_clusters_per_task=2, join_threshold=0.95)
cl.begin_iteration(1)
cl.assign(A("alpha alpha alpha")); cl.assign(A("beta beta beta"))
out=cl.assign(A("zeta completely unrelated wording here"))
print(f"cluster_id={out.cluster_id!r} sim={out.similarity:.3f}")
print(f"unassigned_reason={out.unassigned_reason!r}")
EOF

# The dedup config surface. Expect disabled by default, enabled only with BOTH vars.
python - <<'EOF'
import sys, json; sys.path[:0]=['src']
from agent_evolve.core.config import resolve_profile
print("default enabled:", resolve_profile('research_sequential', {}).mechanism_dedup.enabled)
print("model only     :", resolve_profile('research_sequential', {'AE_MECHANISM_DEDUP_MODEL':'m'}).mechanism_dedup.enabled)
c=resolve_profile('research_sequential', {'AE_MECHANISM_DEDUP_MODEL':'m','AE_MECHANISM_DEDUP_BASE_URL':'http://x','AE_MECHANISM_DEDUP_API_KEY':'SECRET'})
blob=json.dumps(c.manifest_payload(), default=str)
print("both -> enabled:", c.mechanism_dedup.enabled, "| SECRET in manifest:", 'SECRET' in blob)
EOF

# §5 correction. Expect: 2/2 score-record calls already carry a diagnosis.
python - <<'EOF'
import sys; from pathlib import Path
R=Path('.').resolve(); sys.path[:0]=[str(R/'src'),str(R),str(R/'tests')]
from test_phase_6_orchestrator import _runner,_task
r=_runner(seed=0); obs=[]; orig=type(r)._record_rollout_score
def spy(self, cid, rollout):
    obs.append((rollout.task.task_id, rollout.analysis is not None))
    return orig(self, cid, rollout)
type(r)._record_rollout_score = spy
try: r.run_attempt((_task("task-a"),_task("task-b")))
finally: type(r)._record_rollout_score = orig
print(f"{sum(1 for o in obs if o[1])}/{len(obs)} score-records had a diagnosis attached")
EOF

# §11 The pool/tracker key separation. Expect CONSTANT: 2 shared cells, dominates True.
#                                     Expect MECHANISM: 0 shared cells, dominates False.
python - <<'EOF'
import sys; from pathlib import Path
R=Path('.').resolve(); sys.path[:0]=[str(R/'src'),str(R),str(R/'tests')]
from agent_evolve.core.pool import PersistentPool
from test_pool import _candidate, _prov
def build(mech_of):
    p=PersistentPool(); p.add_base(_candidate("base"))
    p.add_candidate(_candidate("c1", parents=("base",)))
    for cid,score in (("base",0.5),("c1",0.9)):
        for task in ("task-a","task-b"):
            for r in range(2):
                s,prov=_prov(task, mech_of(cid,task), rollout=r, score=score)
                p.record_score(cid,s,prov)
    return p
for label,fn in (("CONSTANT ",lambda c,t:"mechanism-default"),
                 ("MECHANISM",lambda c,t:f"{t}:mech-of-{c}")):
    p=build(fn)
    print(label, "shared=",len(p.comparable_cells("base","c1")),
          "dominates=",p.dominates("c1","base"), "frontier=",p.pareto_frontier())
EOF

# §10 Clustering failure modes: drift, order dependence, at_cap forced merge.
python - <<'EOF'
import sys; from pathlib import Path
R=Path('.').resolve(); sys.path[:0]=[str(R/'src'),str(R),str(R/'tests')]
from agent_evolve.core.clustering import MechanismClusterer, LexicalEmbedder
from agent_evolve.core.blame import CausalAnalysis, BlameGraph, BlameNode
def A(m): return CausalAnalysis(mechanism=m, severity=0.6, score=0.2,
    blame_graph=BlameGraph(nodes=(BlameNode(actor_id="agent", artifacts=("skills/a.md",), blame=0.9),)))
cl=MechanismClusterer(task_id="t", embedder=LexicalEmbedder(dim=32), max_clusters_per_task=2)
cl.begin_iteration(1)
for m in ["alpha alpha alpha","beta beta beta","zeta completely unrelated wording here"]:
    a=cl.assign(A(m)); print(f"at_cap {m[:30]:30s} -> {a.cluster_id} sim={a.similarity:+.3f} new={a.is_new_cluster}")
cl=MechanismClusterer(task_id="t", embedder=LexicalEmbedder(dim=32)); cl.begin_iteration(1)
for m in ["date filter missing","date filter absent","filter for dates omitted","dates unfiltered entirely"]:
    a=cl.assign(A(m)); print(f"drift  {m:26s} -> {a.cluster_id} sim={a.similarity:.3f}")
EOF
```

---

## 7. Traps that have already cost time

1. **The offline suite cannot demonstrate the floor being *cleared*.** Measured: the
   fake becomes perfect after attempt 1, so attempts 2-6 all return
   `"no evidence-backed work item available"` and generate no mechanism diversity.
   Offline proves the floor is *unmet*; it cannot prove a fix clears it. Live-shaped
   tasks are required for that claim.
2. **`git stash` reverts modified *test* files too.** A before/after comparison via
   stash silently compares different assertions. This produced one wrong conclusion
   already. Prefer editing a copy, or verify with `diff` after restoring.
3. **A `Fix direction:` in the register is a hypothesis, not an instruction.** SV-10's
   was wrong on both halves (schema-derived, not execution-derived). Re-derive any such
   prescription from running code before spending a day on it.
4. **Two unrelated fields named `severity`.** `CausalAnalysis.severity` /
   `Issue.severity` are genuinely written: `cuga_analyzer.py:590` parses the value
   from the analyzer's JSON response and range-checks it to `[0,1]`, rejecting a
   finding that omits it. `ScoreProvenance.severity` is **never** written by any
   production caller, so `ScoreCell.weighted_score() == mean` exactly. Do not reach
   for the inert one.
5. **Known-stale LSP diagnostics** — do not "fix" these; validate against source:
   `pipeline.py` `python_executable`; `cuga_editor.py:~440` CugaAgent kwargs;
   `cuga_editor_tools.py` `read_artifacts` on `object`.
6. **`jspace ship` false positives** on house style, left deliberately: long
   section-divider comment runs, repeated closing parens, and an accurate in-code
   comment that points at where a dataclass checks its own fields. Note the checker
   also flags *this very list* and any imperative telling a reader to check something
   — if `ship` reports line ~236 of this file, that is the flag reading its own
   example text, not a defect. Do not delete the warning to satisfy the checker.
7. **`rg -r` silently corrupts every quote you gather.** In ripgrep `-r` is
   `--replace`, not "recursive" (recursion is the default). `rg -rn 'task-local|cross-task' docs/`
   returned doc text with every match replaced by the literal `n`, yielding
   fabricated-looking quotes such as *"mechanisms align through n semantic
   clusters"*. This produced wrong quotations of `AGENTS.md` and
   `hypotheses-and-validation.md` in one turn. **Any doc quote gathered with `-r` is
   unreliable** — re-read with Grep or Read before quoting a document.

---

## 8. Decisions already recorded — do not re-litigate

- **SV-5 / SV-1:** inert champion terms **stay**, documentation-only. SV-5 closed on
  that basis. SV-1's register "fix direction" (remove severity from `weighted_score`)
  would **reopen SV-5** — user has not approved it.
- **Champion ranking:** pairwise over shared comparable cells. Aggregate is
  report-only. Coverage is a floor, never a reward. Do not reintroduce aggregate
  ranking.
- **SV-10 delivery:** `ParentContext.issues` only — no `get_parent_issues` tool.
- **Parent draw:** ONE `select_parent()` per attempt. Deeper sampling redesign deferred.
- **Judge:** required in every mode, including `genetic` and
  `--experimental-candidate-promotion` (which disables the SV-4 gate only).
- **No prose across the editor boundary** — cluster ids, numbers, `evidence_refs` only.
  Holds by construction: `Issue` has no prose field.
- **SV-7 live proxy run: APPROVED by the user.** Needs `X-AE-*` headers emitted from
  the call sites first (`X-AE-Candidate/Task/Rollout/Phase/Run`); the proxy addon
  already lifts and strips them (`docker/observability/addons/correlate.py:146-152`).
- Never commit/stash/discard without explicit approval. macOS: no `timeout`.

---

## 9. SV-12 is THREE defects, not one

Established 2026-08-20 with the user. Every claim here is measured; commands in §6
and §10.

> **STATUS: 9.1 FIXED (step 1). 9.2 and 9.3 still open — steps 3 and 4.** The
> measurements below are the *original* ones and are retained as the evidence
> trail; do not read them as the current state. See §12 for what changed.

### 9.1 The embedder is unwired — this is the blocking defect

**FIXED in step 1** — `pipeline.embedder_for_config()` now exists and honours
`config.embedding`; default dim is `DEFAULT_EMBEDDING_DIM` (768), not 32.

`config.embedding` defaults to `provider="ollama"` (`config.py:316`) and a real
semantic embedder exists (`core/embeddings.py`: `OllamaEmbedder`,
`FallbackEmbedder`, `build_embedder`). None of it ran:

- `build_embedder` has **zero** callers in `src/` or `scripts/` — only `tests/test_embeddings.py`.
- `config.embedding` is never read to construct anything, only serialized (`config.py:197-200`).
- `pipeline.py:974` and `:1164` hardcode `LexicalEmbedder(dim=32)`; `cluster_registry` is never passed, so the orchestrator's `default_factory` lexical embedder is what production uses.

**So the config advertises semantic embeddings while production runs hashed-token
cosine.** `LexicalEmbedder` measures *word overlap, not meaning*. Measured on four
descriptions of one identical fault using different vocabulary:

```
dim= 32 (production) -> 4 clusters / 4 findings   largest=1  floor(>=3) STARVES
dim= 64              -> 4 clusters / 4 findings   largest=1  floor(>=3) STARVES
dim=256              -> 4 clusters / 4 findings   largest=1  floor(>=3) STARVES
```

The clusterer's 24 passing tests use vocabulary-*sharing* paraphrases, which do
join (measured sim 0.847 and 0.769 against the 0.75 threshold — note how thin that
margin is). Both facts are true; the tests simply do not cover wording variation.

### 9.2 The genetic path recomputes variance over a constant bucket

Per §2 and §4: `_cell_entropy` filters `m_id == self.mechanism_cluster_id`, a
constant. It yields cross-candidate score spread in one synthetic bucket.

### 9.3 Cluster identity is task-local, multiplying the evidence cost

`ClusterRegistry` holds one clusterer per task and namespaces ids
`f"{task_id}:{cluster_id}"` (`clustering.py:345`), so the *same* mechanism on two
tasks is structurally unable to share a cluster. `SEVERE-OPEN-ISSUES.md:1058-1066`
already records this limitation. Measured, one systemic fault across 4 tasks:

```
TASK-LOCAL (today):  4 cells, each needing >=3 comparable candidates independently
TASK-AGNOSTIC:       1 cell, evidence pooled -> same entropy at 1/4 the evidence cost
```

`AGENTS.md:73` named the intended mechanism: *"task-local semantic clusters
anchored by base-harness observations"* — the anchoring (`add_anchor`,
`force_new=True`) is implemented but never called in production.

**STATUS 2026-08-21 — SUPERSEDED. The anchoring does not work, and step 4 is
re-scoped.** Measured (see `docs/design/issue-lifecycle.md` D1/D2):

- Anchors embed **bare mechanism text**; observations embed mechanism **plus actor
  plus artifacts**. An identical mechanism scored only **0.756** against its own
  anchor, and 2 anchors plus their 2 matching observations gave **4 clusters, not
  2** — anchors do not absorb the observations they exist to attract.
- Cluster ids are a per-task counter, therefore **order-dependent**: the same fault
  came out `c2` in one task and `c3` in another purely from arrival order. No
  amount of anchoring makes a counter content-addressed.
- `similarity` is `1.0` by construction on the `force_new` path (it skips
  `_best_match`), so the number reported for an anchor says nothing about
  proximity.

`AGENTS.md:73` has been rewritten accordingly. **Cross-task pooling is deferred**;
the table above remains valid as a statement of what pooling *would* save, not as a
plan. The live priority is instead **within-task dedup quality**, because measurement
shows the fragmentation is happening inside a single task: at the current
`join_threshold=0.75` only **6 of 12** analyzer paraphrase pairs merge, so half of
all rephrasings split into one-candidate cells that can never meet the floors.

---

## 10. Measured weaknesses of the current clustering algorithm

> **STATUS: 10.3 (`at_cap`) FIXED in step 2. 10.1 (order dependence) and 10.2
> (centroid drift) are now *mitigable* when an adjudicator is configured, but the
> underlying cosine-only path still exhibits them.** Measurements below are the
> originals, retained as the evidence trail.

The algorithm is **online single-pass nearest-centroid with a similarity
threshold** (sequential leader clustering), `clustering.py:_add`:

```python
best_id, best_sim = self._best_match(vec)              # cosine vs each centroid
at_cap = len(self._clusters) >= self.max_clusters_per_task
if best_sim >= self.join_threshold or at_cap:          # 0.75 default
    self._update_cluster(best_id, vec)                 # centroid := running mean
else:
    cluster_id = f"c{self._next_id}"                   # spawn
```

The vocabulary is **emergent, not fixed** — clusters are created on demand. A fixed
taxonomy would contradict `component-architecture.md:155` (mechanisms are
"free-form, task-local, and may be uncertain") and AGENTS.md's ban on a fixed
failure taxonomy.

Three measured failure modes:

1. **Order dependence.** Same 4 mechanisms, two arrival orders:
   ```
   arrival [0,1,2,3] -> ['c0','c0','c1','c2']
   arrival [0,2,1,3] -> ['c0','c1','c0','c2']
   ```
2. **Centroid drift splits one fault.** The running-mean centroid moves, so later
   phrasings of the same fault fall below threshold:
   ```
   "date filter missing"       -> c0 (size 1)
   "date filter absent"        -> c0 (size 2)
   "filter for dates omitted"  -> c1   <- same fault, 2nd cluster
   "dates unfiltered entirely" -> c2   <- same fault, 3rd cluster
   ```
3. **`at_cap` forces false merges.** The `or at_cap` short-circuits the similarity
   check entirely: at the cap, *every* mechanism joins the nearest cluster
   regardless of similarity.

   > **Correction, 2026-08-20 (mine).** I originally cited the `0.822` figure below
   > as the forced-merge evidence. **It is not.** `0.822 >= 0.75` (the default
   > `join_threshold`), so that join was *legitimate by the threshold* — the defect
   > it actually demonstrates is `dim=32` hash collision making unrelated text
   > score 0.822, which is the **embedder** problem (§9.1, step 1), not the
   > `at_cap` problem. The genuine `at_cap` defect needs a similarity *below* the
   > threshold, which the §6 probe now shows at `0.612` against a `0.950`
   > threshold. Two real defects, and I conflated their evidence.

   Original measurement, cap=2, default threshold 0.75 (now understood as the
   embedder defect):
   ```
   "zeta completely unrelated wording" -> c0  sim=+0.822  new=False
   ```
   The genuine cap-forced merge, threshold 0.95, **fixed in step 2**:
   ```
   before: cluster_id='c0'  sim=0.612   <- absorbed below threshold
   after:  cluster_id=''    sim=0.612   unassigned_reason='cluster cap reached (2)
                                         and nearest cluster similarity 0.612 is
                                         below the join threshold 0.950'
   ```
   Absorbing below threshold is worse than starvation because it is **confidently
   wrong**. Two unrelated faults in one cell produce a *high* variance reading that
   says "a fix is reachable here" when no single mechanism exists to fix:
   ```
   FORCED MERGE (one cell, two faults): entropy=0.07200  classify=recombination_target
   REFUSED  (two honest cells):         entropy=None     classify=skip
   ```
   The spec already mandates the second behaviour for under-evidenced cells:
   *"A cell failing the floors is marked `entropy_unavailable` with a reason and
   falls back to severity/coverage quality. It must never contribute a
   high-variance signal derived from a single sample."*

---

## 11. Approved design and build order

**User-approved 2026-08-20.** Sequencing matters: mechanism-keying the cells
*before* the embedder works would replace one always-available-but-meaningless
number with a mostly-`entropy_unavailable` one, which reads as a regression.

**Step 1 — wire the embedder. DONE (§13).** Make the pipeline honour
`config.embedding` via `build_embedder` (Ollama primary, lexical fallback, fallback
reason reported rather than silently substituted).

**Step 2 — cosine pre-filter + small-LLM adjudicator. DONE (§13).** Keep embedding
cosine as the cheap pre-filter. Call a small dedup/categorization model **only** in
the ambiguous band around `join_threshold` and on every `at_cap` decision — precisely
where cosine is measurably unreliable. This is order-independent and can re-merge
drifted clusters at the existing `refresh_at_barrier`.
  - **Its own env vars, for cost control** (user's explicit requirement), following
    the established `LITELLM_MODEL` / `LITELLM_BASE_URL` / `LITELLM_API_KEY` shape:
    a dedicated small-model id, endpoint, and API key, separate from the rollout,
    analyzer, judge and editor roles.
  - `core/clustering.py` is agent-neutral, so the adjudicator enters through an
    **injected protocol** (as `MechanismEmbedder` does), never a direct model import.

**Step 3 — mechanism-key the entropy tracker on the genetic path. DONE (§14)**,
keyed at write time per §5. Report coarse-vs-fine mode; never silently substitute.

**Step 4 — RE-SCOPED 2026-08-21: within-task dedup quality, not cross-task
identity.** This step originally read *"task-agnostic mechanism identity, cells
still indexed per task"*, on the assumption that `m0` could be made to mean the
same fault everywhere via the base-harness anchoring of §9.3. That anchoring is
defective (see the STATUS note in §9.3), and cluster ids are an order-dependent
counter, so cross-task identity does not follow from it.

The measured priority is **inside a single task**: at `join_threshold=0.75` only
**6 of 12** analyzer paraphrase pairs merge, and same-fault versus different-fault
cosine distributions **overlap** (separation `-0.036`), so no threshold alone
separates them. Step 4 is therefore: widen the adjudicator band to roughly
`0.45-0.75` so no true pair is decided by cosine alone, and let the dedup model
settle the ambiguous ones (probed live, 6 of 6 correct).

Cross-task pooling stays **deferred**: it would cut evidence cost on systemic
faults, but it needs order-independent, content-derived ids, which is a larger
change. Full design and measurements: `docs/design/issue-lifecycle.md`.

**Also approved:** remove `task_id` from the text `_embed_finding` embeds. The task
name is not evidence about the failure mechanism, and with a real semantic embedder
it biases toward same-task grouping — the opposite of the pooling in step 4. (With
today's lexical embedder it measurably changed nothing: similarity 1.000 either
way.)

### The pool/tracker key separation — DO NOT MERGE THESE

Two structures key on `(task_id, mechanism_cluster_id)` and want **opposite** things
from the second slot:

| structure | question it answers | needs |
| --- | --- | --- |
| `EntropyTracker` cells | *where is variance high, so a fix is reachable?* | mechanism-**keyed** |
| Pool `score_tensor` | *is c1 better than base?* | **shared** keys |

Champion comparison intersects on the exact full key (`pool.py:449-451`, set `&`).
Mechanism ids come from diagnosis, and an offspring exists *because* it fixed its
parent's fault, so their mechanisms are supposed to differ. Measured with c1 scoring
**0.9 on every task** against base's **0.5** — c1 unambiguously better:

```
POOL KEY = CONSTANT (today)      shared cells: 2   dominates: True    frontier: ('c1',)
POOL KEY = MECHANISM             shared cells: 0   dominates: False   frontier: ('base','c1')
                                 exclusions: 4 x ['missing for c1', 'missing for base']
```

It fails **silently**: no exception, `dominates()` correctly returns `False` on an
empty overlap, and a frontier containing everything looks like healthy diversity
while actually meaning *nothing could be compared to anything*. SV-2 is closed and
its tests construct matching keys, so they would not catch it. **Keep the pool's
mechanism id constant; mechanism-key only the tracker.** A guard test locking this
is approved (§12).

**Open question, reframed by the user 2026-08-20 — and the reframing is correct.**
This was originally written as "one shared tracker or two?", which implied RHO is a
*consumer* whose needs must be balanced against the genetic path's. It is not.
**`mode='rho'` never reads entropy at all** — measured: all six read methods
(`.cell_entropy()`, `.entropy()`, `.classify()`, `.top_entropy_cells()`,
`.all_cells()`, `.entropy_weighted_with_freshness()`) have **0 callers**, and
`tracker` appears in `run_evolution.py` only 4 times: the comment, the constructor
at `:1048`, and the `run_rounds(...)` hand-off at `:1051`. It is never read,
printed, persisted or returned. RHO selects its coreset by difficulty fingerprint;
no RHO phase is an entropy phase.

So there is exactly **one** consumer (the genetic DPP) and **one** producer (RHO),
and the real question is narrower and low-risk:

> **Should the genetic path read the cells RHO already fills?**

RHO cannot regress from sharing — it writes the same cells either way and reads
nothing. This also drains the force from the comment at `run_evolution.py:1045-1047`
("mixing them ... would make two different mechanisms share one cell"): a real
hazard, but one that has only ever corrupted a number **nobody reads**, and it
disappears once cells are mechanism-keyed.

**The actual work is not the sharing — it is key agreement.** RHO keys cells
`rho-task:{task_id}` (task-local, not mechanism-local); the genetic side uses the
constant `mechanism-default`. Two mutually meaningless keyspaces today. For the
genetic DPP to *use* RHO's evidence, both must key by mechanism **and mean the same
thing by a given mechanism id** — the same cross-source identity problem as §9.3's
task-agnostic point, one level up.

Why it is worth doing: the genetic path reaches only 1-2 comparable candidates
against a floor of 3 (§6 starvation command), while RHO performs real rollouts in
phases 5 and 8 whose evidence is currently discarded.

---

## 12. Tests: what is written, what remains

Steps 1-2 wrote 55 tests across three files. Items 1, 5 (partly), 6, 7, 8 and 9 are
**done**; 2, 3 and 4 belong to step 3.

| # | Test | Status |
| --- | --- | --- |
| 1 | Pool key guard — production pool writes use a constant mechanism id | **done** (`test_embedder_wiring.py`) |
| 2 | Unrelated mechanisms do not inflate each other's DPP variance | step 3 |
| 3 | Same-mechanism evidence across candidates forms one reliable cell | step 3 |
| 4 | Floors keep sparse cells out of DPP (`entropy_unavailable` + reason) | step 3 |
| 5 | Coarse-vs-fine fallback is reported, never silently substituted | partial — `adjudication_unavailable_reason` and `embedding_fallback_reason` exist; the *entropy-side* report is step 3 |
| 6 | Wording variation does not fragment one fault | **done** (`test_mechanism_adjudicator.py`) |
| 7 | `at_cap` does not force a below-threshold merge | **done** |
| 8 | Assignment is order-independent | **done** (adjudicator cache, `test_cuga_mechanism_adjudicator.py`) |
| 9 | `_embed_finding` does not embed `task_id` | **done** |

---

## 13. What steps 1 and 2 actually changed (read before touching anything)

### Step 1 — the embedder seam and the dedup config

- **`pipeline.embedder_for_config(config, *, dim=None, timeout=None, transport=None)`**
  is new and is the only intended way to build a mechanism embedder from a resolved
  config. It calls `build_embedder`, which previously had **zero** callers in `src/`.
  Default `dim` is `DEFAULT_EMBEDDING_DIM` (768); the old hardcoded value was 32.
- **`MechanismDedupConfig`** in `core/config.py`, reachable as
  `config.mechanism_dedup`. Env vars, all optional:
  `AE_MECHANISM_DEDUP_MODEL`, `AE_MECHANISM_DEDUP_BASE_URL`,
  `AE_MECHANISM_DEDUP_API_KEY`, `AE_MECHANISM_DEDUP_BAND_LOW` (0.60),
  `AE_MECHANISM_DEDUP_BAND_HIGH` (0.85).
  - `enabled` is `True` only when **both** model and url are set. A partial config
    never half-enables.
  - `api_key` is excluded from `manifest_payload()` by construction. Note
    `manifest_payload()` is the real serializer — there is **no** `as_dict()`; I
    checked the wrong name once and that check proved nothing.
  - Malformed band values raise; they do not fall back to the default.
  - `_DEFAULT_DEDUP_BAND_LOW/HIGH` are **module-level constants** because
    `slots=True` turns a dataclass field into a `member_descriptor`, so
    `MechanismDedupConfig.band_low` is a descriptor object, not a float. Reading it
    as a default raised `TypeError` inside every `resolve_profile` call.
- **`_embed_finding` no longer embeds `task_id`** (`orchestrator.py`). The `task`
  argument is retained because cells stay indexed per task.
- `DEFAULT_MECHANISM_CLUSTER` in `pipeline.py` now carries the §11 rationale.

### Step 2 — the adjudicator

- **`MechanismAdjudicator`** Protocol in `core/clustering.py`
  (`same_mechanism(left, right) -> bool | None`). Injected, never imported by core.
- **`ClusterAssignment` gained two fields**: `unassigned_reason` and
  `adjudication_unavailable_reason`.
- **`MechanismClusterer` gained** `adjudicator`, `band_low`, `band_high`, plus an
  internal `_exemplars` map (one representative text per cluster, so the
  adjudicator can be asked about *text* rather than a centroid it cannot read).
- **The `at_cap` defect is fixed.** `best_sim >= join_threshold or at_cap` used to
  discard the similarity check entirely at the cap. Now a below-threshold
  observation at the cap returns `cluster_id=""` with an `unassigned_reason`.
  **Consequence step 3 must handle: a finding can now come back unassigned.** A
  caller that writes `assignment.cluster_id` into a cell key without checking will
  create a cell keyed by the empty string.
- **`ClusterRegistry` gained** `adjudicator`, `band_low`, `band_high`, and passes
  them to every per-task clusterer. Its `embedder_factory` annotation was tightened
  from `"callable"` to `Callable[[], MechanismEmbedder]`.
- **`CugaMechanismAdjudicator`** in `adapters/cuga_mechanism_adjudicator.py`: uses
  the `AE_MECHANISM_DEDUP_*` vars, parses strictly (`same` / `different` /
  anything-else-abstains), caches order-independently so a repeated pair costs one
  call, and **never raises** — every failure is an abstention.

### What is NOT done, and must not be claimed

- **No live model call has ever been made** through either the semantic embedder or
  the adjudicator. Every test injects a fake transport or `completion_fn`.
  Therefore: **no claim that clustering quality improves on real analyzer output is
  supported yet.** Wiring makes it possible; it does not demonstrate it.
- The adjudicator is **not wired into the production stack**. `pipeline.py` does not
  construct one, and `SequentialGepaRunner` is still handed
  `LexicalEmbedder(dim=32)` at `pipeline.py:1022` and `:1212`. Steps 1-2 built the
  seams and covered them with 55 offline tests over config resolution, the embedder
  builder, the clustering decision path and the adapter's parsing, caching and
  failure behaviour; that coverage excludes every live model call and the live
  genetic path. Connecting the seams to production is part of step 3.
- `EntropyTracker` is still **write-only by RHO and unread by anyone** (§2).

---

## 14. Step 3 — DONE 2026-08-20

**Mechanism-key the entropy tracker on the genetic path, keyed at write time.**

**STATUS: complete and green.** Full suite 2035 collected, 2034 passed, 1 skipped,
0 failed (`terminal_output/sv12/12-step3-final.log`), reconciling as 2023 + 12 new
tests in `tests/test_genetic_entropy_tracker.py`. `entropy.py` needed **no
change** and remains 0-diff, as predicted below. Core purity holds at 34 files, 0
forbidden imports.

All four required parts landed:

| # | Required | Implemented as |
| --- | --- | --- |
| 1 | A producer on the genetic path | `_record_entropy_evidence`, called from `_record_rollout_score`; promotes to comparable only after `min_rollouts_per_candidate` |
| 2 | A consumer | `_cell_entropy` and `_entropy_tier` now read the tracker; the inline duplicate is deleted |
| 3 | Mechanism-keyed cells | `_entropy_cluster_id`, which returns `""` and records a reason rather than substituting a placeholder |
| 4 | Report coarse-vs-fine | `entropy_unavailable_reason(task_id)` plus `_last_entropy_unavailable_reasons` |

Plus the production wiring that steps 1-2 left disconnected:
`pipeline.cluster_registry_for_config` builds the registry and attaches
`CugaMechanismAdjudicator` only when dedup is fully configured, importing it
lazily at the composition boundary so `core/` still imports no adapter. Both
runner construction sites now receive a real registry.

**Measured, in one run:** tracker cells `('task-a', 'task-a:c0')` and
`('task-b', 'task-b:c0')` while pool mechanism ids stayed `['mechanism-default']`
— the two keyspaces diverged exactly as intended, and the SV-2 guard test still
passes.

### Two defects found while implementing, neither previously visible

1. **`ClusterRegistry.assign` laundered refusals.** It built
   `f"{task_id}:{assignment.cluster_id}"` unconditionally, so a refusal (inner
   id `""`) became `"task-a:"` — **non-empty, therefore truthy**. `CellKey`
   rejects only a falsy mechanism id, so the refusal passed the guard and would
   have been filed as a real mechanism; a caller writing the obvious
   `if assignment.cluster_id:` was defeated by the namespacing alone. Both
   `unassigned_reason` and `adjudication_unavailable_reason` were also dropped at
   this hop. Measured value `'task-a:'` is what the red test reports.
2. **I made the offline stack do network I/O.** Wiring `embedder_for_config` into
   `build_offline_stack` honoured a config whose default provider is `ollama`;
   with a local daemon running that is a real HTTP call per embed, measured
   ~0.18s each, which stalled the suite at 92% and made offline results depend on
   whether a daemon happened to be up. The offline builder now keeps a
   deterministic `LexicalEmbedder(dim=DEFAULT_EMBEDDING_DIM)` — 768, not the old
   colliding 32 — and two tests lock it. `build_live_stack` still gets
   `embedder_for_config`.

### Non-vacuity, proven separately per half

* Disabling the producer failed exactly 3 tests.
* Restoring the registry laundering failed exactly 1, reporting `'task-a:'`.
* Both files then restored byte-identical by string compare.

### The honest outcome, unchanged from the prediction below

Mechanism-keyed cells report `skip`/unavailable **more often** than the old
constant-bucket version, because the `>=3` comparable-candidate floor is genuinely
harder to clear per mechanism than across one pooled cell. Measured on the offline
fake: `tier=skip`, `H=None` after 4 attempts. Correct-but-unavailable beats
wrong-and-confident, but it is not a throughput win and must not be presented as
one.

Floors were measured directly against the tracker: 3 comparable candidates with 2
rollouts each gave `H=0.109` and `recombination_target`; one candidate with one
rollout gave `None` and `skip`. That covers the arithmetic and the floor branch,
**not** a live run clearing the floor through real rollouts — trap 1 in section 7
explains why the offline fake structurally cannot show that.

### What is still NOT done

* **No live model call** has run through the semantic embedder or the dedup
  adjudicator. Every test injects a fake, so **no claim that clustering quality
  improves on real analyzer wording is supported.**
* Mechanism identity is still **task-local** (`task-a:c0`), so one fault on two
  tasks occupies two cells — that is step 4.
* The `entropy_unavailable` fallback rate is reportable per task but not yet
  aggregated into the run summary.

---

## 14b. The original step-3 brief (kept as the rationale)

Why it was safe to key at write time: section 5. The genetic path has its diagnosis
*before* it records a score (measured 2/2), so no retroactive re-filing is needed.
`EntropyTracker.record_score` is a plain `dict[CellKey, _Cell]` insert, so **no
change to `entropy.py`** should be required — and it is under a 0-diff constraint,
so **ask before editing it**.

The four things step 3 must do:

1. **A producer on the genetic path.** Today only RHO writes to a tracker. The
   genetic path must `record_score(...)` and `mark_comparable(...)` as its rollouts
   land.
2. **A consumer.** `_cell_entropy` / `_entropy_tier` (`orchestrator.py`) must read
   the tracker instead of recomputing variance inline over the constant
   `mechanism-default` bucket.
3. **Mechanism-keyed cells** — from the clusterer, *not* the constant. Handle the
   unassigned case from section 13.
4. **Report coarse-vs-fine.** When a mechanism cell misses the floors and the
   selector falls back, that must be visible, never a silent substitution. The spec
   already mandates `entropy_unavailable` with a reason.

**Hard constraint, do not violate:** the *pool*'s `mechanism_cluster_id` stays
constant (section 11). Only the tracker gets mechanism-keyed. The guard test in
`test_embedder_wiring.py` enforces this and will fail loudly.

**Expected honest outcome:** with mechanism-keyed cells the `>=3` comparable-candidate
floor will be *harder* to clear, so entropy will report `unavailable` more often than
today's always-available-but-meaningless number. That is the correct direction —
correct-but-unavailable beats wrong-and-confident — but say so plainly rather than
presenting it as a straightforward win. Section 7 trap 1 explains why the offline suite
cannot demonstrate the floor being *cleared*.

Then step 4 (task-agnostic mechanism identity, cells still per task) and finally
SV-7's approved live proxy run.

---

## 15. Session of 2026-08-21 — what changed

### 15.1 Step 3 landed, plus one defect in it found by a user question

Step 3 (§14) completed: the genetic path now both **writes** mechanism-keyed cells
into `EntropyTracker` and **reads** them back for the DPP.

Then the user asked *"how are you handling those skip / unavailable cases?"* and the
answer exposed a real defect in the code I had just called complete:

**`_cell_entropy` and `_entropy_tier` resolved their mechanism cell
independently.** With two qualifying cells on one task:

```text
m1: H=0.001050  recombination_target   (low variance, high score)
m2: H=0.002904  frontier_exploration   (high variance, low score)

_cell_entropy -> 0.002904  (max, from m2)
_entropy_tier -> recombination_target   (from m1)   <-- MISMATCH
```

The tier is an instruction about *that specific number* — `raw_issue_quality`
(`issues.py:140-145`) damps `frontier_exploration` to `frontier_weight` (0.30) and
zeroes `skip`. So m2's value inherited m1's weight: a frontier signal **silently
promoted from 30% to 100%**. No crash, no failing test, just a wrong weight.

Fixed with `_entropy_cell_for(task_id)` as the single resolution point both methods
read. Non-vacuity: reverting only the tier resolution failed exactly that one test.

**Also verified while there:** `issues.py:557` adds `.entropy` into the DPP ordering
key **ungated by tier**. A 300-trial randomized sweep found **0** cases where
`tier == "skip"` coexisted with nonzero entropy, so it contributes exactly `0.0` in
every skip case — safe by invariant, not by construction. Worth knowing if the
entropy source ever changes.

The three entropy states are distinct and all reachable:

| State | value | tier | reason |
| --- | --- | --- | --- |
| floors met, real variance | `0.109` | `recombination_target` | — |
| floors met, **zero** variance | `0.0` | `skip` | `None` (genuinely measured) |
| floors **unmet** | `0.0` | `skip` | explicit string |

### 15.2 The environment was never reaching the config

`resolve_profile(name, environ=None)` defaulted `environ` to an **empty dict**, not
`os.environ`. Neither production call site passes it (`pipeline.py` in
`build_offline_stack` and `build_live_stack`), so **every** `env.get(...)` in that
function was dead in production — including the `AE_MECHANISM_DEDUP_*` vars added in
step 1 and documented in `docs/USER-MANUAL.md` as working. `enabled=bool(model and
url)` could never be true.

Silent because every var has a default, so the config resolved fine. **Hidden
further by coincidence:** `OLLAMA_EMBEDDING_URL` defaults to
`http://localhost:11434` and the model to `embeddinggemma`, which happen to match a
stock local Ollama — so earlier "live embedding" appeared to honour the environment
while actually using defaults.

Fixed: `environ` defaults to `os.environ`; an explicitly passed mapping (including
`{}`) still wins, so tests stay deterministic and ambient shell state cannot leak
into offline runs. 4 tests in `tests/test_env_reaches_config.py`, 2 proven red.

**Trap for a fresh agent:** `load_dotenv()` with no argument raises
`AssertionError` when run from `python3 - <<'PY'` (stdin), because `find_dotenv()`
walks the call stack. Pass the path: `load_dotenv('.env')`, or write to a temp file
and run that.

### 15.3 First live model calls in this repo — cosine alone provably fails

Full detail and logs in `docs/design/issue-lifecycle.md` §3. Headline, measured with
`embeddinggemma` over 4 fault families and all 66 unique pairs
(`terminal_output/calibration/live-embedder-calibration.log`):

```text
SAME fault   min=0.466  mean=0.728  max=0.851
DIFF fault   min=0.244  mean=0.393  max=0.502
separation (min_same - max_diff) = -0.036      <- NEGATIVE, distributions OVERLAP
```

**No single cosine threshold separates analyzer paraphrase from a genuinely
different fault.** At today's `join_threshold=0.75` only **6 of 12** true pairs
merge, so half of all rephrasings fragment into one-candidate cells that can never
meet the 3-comparable-candidate floor. That is the SV-12 starvation, reproduced at
its root cause.

The dedup adjudicator (`openai/aws/gpt-oss-120b`) answered **6 of 6** probe pairs
correctly including both refusals
(`terminal_output/calibration/live-adjudicator-probe.log`, probe script kept
alongside). The two-stage design is load-bearing, not an optimisation.

**Caveat that must travel with these numbers:** all 12 strings are synthetic
phrasings written by me, not real CUGA analyzer output. The overlap finding is
robust; the exact thresholds are indicative, not tuned.

### 15.4 Documents updated, per the user's standing instruction

The user directed: when a decision contradicts a file a future session relies on,
**update that file** so the next agent does not inherit a false premise. Done:

| File | Change |
| --- | --- |
| `AGENTS.md:73` | "anchored by base-harness observations" replaced with dynamic cluster formation plus cosine-and-adjudicator identity; old wording preserved in a dated superseded note |
| `docs/SEVERE-OPEN-ISSUES.md` | cross-task constraint section now says the anchoring **exists and is defective**, not merely unbuilt |
| `docs/COMPACTION-ANCHOR-SV12.md` §9.3 | STATUS block: anchoring superseded, with the 0.756 measurement |
| this file's step-4 line (§11) | re-scoped from task-agnostic identity to within-task dedup quality |
| `docs/design/issue-lifecycle.md` | **NEW** — the authoritative design |

---

## 16. Step 4 — DONE 2026-08-21, and what it cost

**Not** "task-agnostic mechanism identity". That framing rested on base-harness
anchoring, which is defective (§9.3 STATUS). Cross-task pooling is **deferred by
decision** (design doc D1), not open.

Step 4 was: **make within-task dedup actually work**, because measurement shows the
fragmentation happens inside a single task. All four parts landed, plus a
prerequisite defect nobody had spotted.

**0. THE PREREQUISITE — the band was going to be inert.**
`cluster_registry_for_config` passed `base_url=dedup.base_url`, but the field is
`url` (fields are `url/model/api_key/enabled/band_low/band_high`;
`hasattr(MechanismDedupConfig, "base_url")` is `False`). The broad
`except Exception` caught the `AttributeError` and degraded to cosine-only, so the
adjudicator had **never attached in production** despite `.env` resolving
`enabled=True`. The band is only consulted when an adjudicator exists. Now read
behind an explicit guard that raises rather than degrades.

**1. Band widened** `0.60-0.85` -> `0.45-0.75`. Chosen by measurement over 66 live
pairs, scored by *silent splits* (true pairs decided against merging by cosine
alone, no model call): `0.60-0.85` split **2**, `0.45-0.75` splits **0** at 16/66
adjudicated, `0.40-0.75` also 0 but at 35/66. Note the anchor previously said "3
true pairs below 0.60" and the design doc said "~43/66 adjudicated" — both were
wrong; the measured figures are 2 and 16/66.

**2. FOUR default pairs, not two.** The anchor undercounted. `core/config.py:37-38`,
`clustering.py` on `MechanismClusterer`, `clustering.py` on `ClusterRegistry`, and
`pipeline.py:164-165`. Now **one** definition in `core/clustering.py`
(`DEFAULT_JOIN_THRESHOLD`, `DEFAULT_BAND_LOW`, `DEFAULT_BAND_HIGH`), re-exported by
the other two. Drift is now impossible by construction, not by discipline.

**3. A NEW INVARIANT fell out of the measurement:** `band_high >= join_threshold`.
Below it, the span `[band_high, join_threshold)` is neither ambiguous nor joining,
so cosine decides it alone — measured stranding true pairs at `0.718`, `0.749`,
`0.726`. **Scoped to "an adjudicator is attached"**: my first unscoped version broke
7 existing tests, and all 7 raise the join threshold with *no* adjudicator, where
the band is never read. An unscoped raise rejects legitimate cosine-only configs.

**4. Live verification, first time possible.** `registry.adjudicator` is a real
`CugaMechanismAdjudicator` with stderr silent, and on the 12 live pairs in the
newly-reached `0.45-0.60` window the model scored **12/12** — merged both true
paraphrases, refused all 10 distinct pairs.

**Then (also done):** the `entropy_unavailable` fallback rate is now aggregated —
`EntropyAvailabilityReport` with per-category tallies, reaching `GepaRunResult` and
the per-iteration audit record. **SV-12 is CLOSED.**

**Hard constraints, held:** pool mechanism id stays constant (§11);
`core/entropy.py` is 0-diff; `core/` imports no adapter (35 files, 0 forbidden).

### Dead read API still unused (invariant 2)

`EntropyTracker.cell_entropy`, `.top_entropy_cells`, and
`.entropy_weighted_with_freshness` have **zero callers** after step 3. Step 3 wired
`.entropy()`, `.classify()` and `.all_cells()` only. Three of six read methods
remain dead code.

---

## 17. Nothing is committed — the untracked inventory

**HEAD is `8d48a8f` and every change from the last several sessions is in the
working tree.** 33 modified, 32 untracked. `git stash`, `git checkout .`, or a
careless `git clean` would destroy load-bearing source, not just scratch files.
**Do not run any destructive git command.** Re-derive with
`git status --porcelain`.

Untracked, and load-bearing:

| Path | What it is |
| --- | --- |
| `src/agent_evolve/adapters/cuga_mechanism_adjudicator.py` | the dedup adjudicator (live 12/12 on the rescued band) |
| `src/agent_evolve/core/correlation.py` | **SV-7: the `X-AE-*` correlation scope** — without it every proxy capture is uncorrelated |
| `src/agent_evolve/core/resolution.py` | champion resolution (SV-2/3/5) |
| `src/agent_evolve/core/retirement.py` | SV-13 generational retirement |
| `docs/design/issue-lifecycle.md` | **the authoritative design doc** |
| `docs/COMPACTION-ANCHOR-SV12.md` | this file |
| `docs/SESSION-HANDOFF.md` | handoff notes |
| `.jspace/` | the durable ledger — 45 checkpoints of session history |
| 24 `tests/test_*.py` files | the six from §1's table, plus `test_genetic_entropy_tracker.py` (13), `test_env_reaches_config.py` (4), `test_embedder_wiring.py` (15), `test_mechanism_adjudicator.py` (13), `test_cuga_mechanism_adjudicator.py` (27) |

Also **gitignored**, not merely untracked: `terminal_output/` (`.gitignore:13`)
holds the two live measurement scripts and their logs. §6.0 re-runs them, and §15.3
quotes their numbers, but `git` will never preserve them — if they are lost, the
calibration must be re-measured from the recipe in the design doc §3.
