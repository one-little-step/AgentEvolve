# AgentEvolve Instructions

## Mission

AgentEvolve is an independent, agent-neutral RHO-Parallel-GEPA project. It uses
IBM CUGA **through its SDK**, never a fork.

The generic evolution core and the adapter boundary both exist and run. The
reference adapter is CUGA because the research phase requires exact agent-state
tracing and artifact provenance. Gaia is historical context only, not a runtime
dependency.

**CUGA is installed and in use** — `cuga 0.2.20`, imported from `.venv`. This
supersedes an earlier instruction in this file that said CUGA was "not yet
vendored or installed" and told you not to rely on its APIs. You may now read the
installed package to answer questions about CUGA. What remains forbidden is
*inventing* an API: check `.venv/.../cuga/` or the SDK docs, and if a surface is
not there, say so rather than assuming.

> **Open discrepancy, unresolved.** `pyproject.toml:12` declares `cuga>=0.3.1`,
> but the installed version is `0.2.20` — **below its own declared floor**. Nobody
> has established which is correct. Do not "fix" this by editing the pin or by
> upgrading: the whole suite and every measurement to date ran against 0.2.20, so
> either change invalidates existing evidence. Raise it and let the user decide.

## Non-Negotiable Boundaries

- `src/agent_evolve/core/` is agent-neutral and must never import `cuga`,
  `litellm`, `openai`, `httpx`, a request library, or `agent_evolve.adapters`.
  Currently **35 files, 0 violations**. Verify with the AST check in
  `docs/architecture/IMPLEMENTED-PIPELINE-MAP.md` §11 — a substring grep gives
  false positives, because several `core/` docstrings mention
  `agent_evolve.adapters` in prose.
- `src/agent_evolve/adapters/` is the only place allowed to bind core to a
  concrete agent. `pipeline.py` is the single wiring seam
  (`build_rho_hooks`, `pipeline.py:1478`).
- Keep any CUGA clone read-only under ignored `vendor/`.
- Never assume a generic trace can be replayed. Replay exists only where an
  adapter explicitly reports a valid checkpoint/state-reconstruction capability.
  **No adapter currently does.**
- Artifacts are adapter-declared editable units — instructions, skills, memory,
  policies, workflows, and others. Never hardcode Gaia wisdom filenames or
  Markdown section editing in the generic core.
- Never persist credentials, expected answers, evaluator internals, labels, or
  regexes into edit memory, embeddings, prompts, manifests, or terminal logs.
- Add tests before implementation changes. Keep current implementation, research
  hypothesis, and target architecture clearly distinct — this file's own
  structure below is the model for that separation.
- Capture long test/smoke/verification runs with
  `2>&1 | tee terminal_output/<topic>/<name>.log`. **Be aware `terminal_output/`
  is gitignored (`.gitignore:13`)**, so anything there is unprotected by a commit;
  copy artefacts worth keeping somewhere tracked (see `tools/probes/`).

### Environment traps that have each cost real time

- `pytest -q` suppresses the summary line on this machine. Run pytest through
  `subprocess` and print the matching line yourself.
- macOS has no `timeout` command.
- **`rg -r` means `--replace`, not "recursive", and can modify files.** Never use
  it to search.
- `load_dotenv()` from a stdin snippet can raise `AssertionError`; use
  `load_dotenv('.env')`.
- `str(dict)` escapes newlines, so `"text" in str(payload)` can report `False` for
  content that is present. Compare structures, not reprs.

## What Is Actually Built

Verified 2026-08-21 by static analysis. Suite: **2106 collected, 2105 passed,
1 skipped, 0 failed** across 107 test files. The authoritative, `file:line`
annotated map is `docs/architecture/IMPLEMENTED-PIPELINE-MAP.md`.

| Area | Status |
| --- | --- |
| Persistent pool, score tensor, pairwise champion selection | **LIVE** |
| Causal blame graphs (replacing a fixed failure taxonomy) | **LIVE** |
| Task-local semantic mechanism clustering + dedup adjudicator | **LIVE** |
| RHO 10-phase rounds; modes `rho` / `genetic` / `rho-genetic` | **LIVE** |
| Genetic GEPA attempt loop with edit memory | **LIVE** |
| Four editable surfaces: instructions, skills, policies, memory | **LIVE** |
| Generational soft retirement (SV-13) | **LIVE** |
| Entropy tracker + availability/fallback reporting | **LIVE**, but see below |
| Proxy interception of LLM calls, including CUGA-internal | **LIVE** |
| Crossover / merge | **built, entirely unwired** |
| Parallel batch execution | **implemented, test-only** |
| Checkpoint / counterfactual replay | **absent** |

**The production runner is `SequentialGepaRunner`** (`orchestrator.py:1022`).
`Orchestrator.run_iteration` (`orchestrator.py:510`) has **zero callers in
`src/`** — reading it to understand a live run will mislead you.

**Nothing has been observed end to end.** No correlation-captured live run has
been performed, so no claim of *behavioural gain* is supported. Entropy has been
seen honestly reporting `3/3 cells unavailable = 100% fallback (floor_unmet=3)`
on an offline loop; it is not known to clear its floors in practice.

## Architecture Decisions Already Made

These are durable decisions. Where one describes an unbuilt target, it says so.

- **Persistent pool.** Base plus every initial RHO proposal are retained. Every
  score cell ever recorded stays in the tensor; nothing is deleted mid-run.
- **Generational retirement (SV-13).** Retention is about *evidence*, not breeding
  rights. When an accepted offspring is preferred over its parent by the RHO
  symmetric pairwise judge, the parent is **soft-retired**: excluded from parent
  sampling, the Pareto frontier and champion selection, while its score cells,
  lineage and preference record are all kept. An offspring exists to fix its
  parent's diagnosed faults, so breeding further from a version its own descendant
  improved on spends rollouts re-deriving a fix that already exists.
  - Soft, never pruned: hard deletion would destroy the comparable cells
    cross-candidate entropy needs, and the negative evidence a later analysis
    wants. `pool.prune()` remains ablation-only and has no caller.
  - The judge decides, not the arithmetic. Numeric dominance cannot see whether a
    child solved the parent's failure *mechanism*; the pairwise judge reads
    trajectories. One instrument governs retirement, promotion (SV-4) and final
    resolution, so a candidate can never be retired by one standard and promoted
    by another.
  - Conservative on missing evidence: no judge, an unavailable verdict, a tie, an
    incomplete trace pair, or a raising judge all leave the parent alive. A judge
    outage must never silently shrink the breeding population.
  - The live population is never emptied; the base is retired only if a descendant
    supersedes it while another live entry remains.
  - Terminal condition: if the live pool shrinks to one, that candidate wins
    outright. Otherwise survivors are resolved by symmetric pairwise preference
    over the coreset.
  - Costs judge calls only. Both trace sets already exist at commit time — the
    parent's from `build_issues`, the child's from `validate` — so retirement adds
    `2k` model calls and **zero** rollouts.
- **Rollout budget.** Base receives `G` rollout-group evidence; post-RHO
  candidates initially receive one rollout per selected task, preserving RHO-scale
  cost.
- **Model roles.** Default: rollout, analyzer+judge, editor. Specialized roles are
  optional ablation overrides. Mechanism dedup is independently addressed so a
  small cheap model can serve it (`AE_MECHANISM_DEDUP_*`).
- **Causal blame graphs replace a fixed failure taxonomy.**
- **Cross-candidate entropy requires comparable evidence floors** before it drives
  selection. Mechanisms align through task-local semantic clusters formed
  dynamically as mechanisms arrive: an embedding cosine pre-filter decides the
  clear cases for free, and a dedicated small dedup model adjudicates only the
  ambiguous band `[0.45, 0.75)`. This is because **measurement shows cosine alone
  cannot separate analyzer paraphrase from a genuinely different fault** — the
  same-fault and different-fault similarity distributions overlap (separation
  `-0.036` over 66 live pairs). The dedup model is therefore load-bearing, not a
  cost optimisation. See `docs/design/issue-lifecycle.md`.
  - Mechanism identity is deliberately **task-local**: variance is computed within
    one task across candidates, so an id never needs meaning outside its task.
    Cross-task pooling is **deferred by design decision**, not merely unbuilt.
  - Superseded 2026-08-21: this decision previously read *"anchored by base-harness
    observations"*. `MechanismClusterer.add_anchor(force_new=True)`
    (`clustering.py:304`) has never had a caller in `src/`, and as built it does
    not work — anchors embed bare mechanism text while observations embed mechanism
    plus actor plus artifacts, so an identical mechanism scored only `0.756`
    against its own anchor, and two anchors plus their two matching observations
    produced four clusters rather than two.
  - **Two separate key policies, and they must stay separate.** The pool's score
    tensor asks *"is c1 better than base?"* and needs **shared** keys, because
    champion selection intersects on the exact full key — mechanism-keyed pool
    cells would yield an empty intersection and regress ranking **silently**. The
    entropy tracker asks *"how much do candidates disagree on this mechanism?"* and
    needs **separated** keys, or unrelated faults pool into one cell and their
    score spread reads as within-mechanism variance.
- **Entropy availability is never silently substituted.** Floors unmet means
  unavailable, with an explicit reason from a stable category set; genuine zero
  variance is a real measured zero. `fallback_rate` is `None` for no observed
  cells, never `0.0`.
- **Edit validation** uses origin cases, worked sets, regression probes, deferred
  cluster-level generalization probes, retry exhaustion, and protected floors.
- **Correlation is ambient, via `contextvars`** (`core/correlation.py`), never a
  module global and never threaded parameters. A global would let one worker's
  candidate id label another worker's calls under parallel execution —
  misattributing evidence, which is unrecoverable after the fact and worse than
  having no correlation at all. Absent facts are **omitted, never blanked**: a
  capture that is honestly silent about the candidate is recognisable as
  uncorrelated, whereas `candidate=""` looks like data.
- **Target, not built — parallel batches** are to use immutable snapshots,
  exclusive artifact write leases, and coordinator-only shared-state commits.
  `core/parallel.py` exists; `use_parallel_batch` defaults `False` and
  `config.py _PROFILES` lists `parallel_execution` as *deferred*.
- **Target, not built — crossover** is to be provenance-preserving deterministic
  merge by default, with an editor resolving only documented same-artifact
  conflicts. `core/merge.py` is complete and has **zero importers in `src/`**.

## Observability

`docker/observability/` runs a mitmproxy interceptor in **regular proxy mode**
(not reverse), because CUGA ships its own per-agent model config and reverse mode
would capture our calls while silently missing CUGA-internal ones.

```bash
./docker/observability/proxy.sh up                  # proxy :8082, UI :8083
./docker/observability/proxy.sh run -- <command>     # run through the proxy
./docker/observability/proxy.sh down
```

- **Mock rules make live-path testing free.** `mocks/rules.json` is hot-reloaded
  on mtime change with no restart, and a request-hook mock never reaches upstream.
  First match wins, so when driving a multi-turn agent a **terminate rule must
  precede the drive rule**, or the agent is handed the same block forever.
  **Always restore `rules.json` from `rules.example.json` afterwards.**
- Captures redact `Authorization`, and `X-AE-*` headers are lifted into the
  capture record then **stripped before the request goes upstream**, so no vendor
  ever receives internal experiment identifiers.
- **Correlation is half-wired.** The *emit* side is live in four LiteLLM wrappers.
  But `correlation_scope` (`core/correlation.py:103`) has **zero callers in
  `src/` and `scripts/`**, so in production the context is never set, headers
  render empty, and **every capture is currently unlabelled**. Two adapter routes
  (`run_workspace_agent` and `CugaEditorAgent`) bypass the wrappers entirely by
  design and can never carry headers; group their traffic by timestamp and body.
  Wiring the set side is the prerequisite for the live run being worth its cost.

## Required Reading Order

Start here — these three describe the code as it is:

1. `docs/architecture/IMPLEMENTED-PIPELINE-MAP.md` — what is wired, what is not,
   with `file:line` anchors and diagrams
2. `docs/SEVERE-OPEN-ISSUES.md` — defects where the *measurement instrument* is
   itself untrustworthy; read before trusting any number
3. `docs/COMPACTION-ANCHOR-SV12.md` — session history; **read §20 first**, then
   §20.9 for the current job order (§19 is superseded)

Then, for design intent and rationale:

4. `docs/design/issue-lifecycle.md` — clustering decisions D1–D4, open Q2–Q5
5. `docs/architecture/selection-algorithms.md` — the selection formulas
6. `docs/USER-MANUAL.md` — flags, env vars, operational defaults
7. `docs/OPEN-ISSUES.md` — ordinary issues
8. `docker/observability/README.md` — interception and mocking
9. `docs/architecture/target-rho-parallel-gepa.md` — the target, deliberately
   ahead of the code
10. `docs/research/hypotheses-and-validation.md`
11. `docs/vision-and-decision-record.md`

Historical, useful but Gaia-specific in its paths and runtime assumptions:

12. `docs/rho_evolution/README.md` and `18-`/`19-` in that directory — the
    authoritative RHO-GEPA rationale, schemas and debugging evidence; there is no
    need to rediscover that design from scratch
13. `docs/migration/` — `cuga-sdk-integration-notes.md`,
    `cuga-adaptation-guide.md`, `gaia-baseline-and-gap-audit.md`,
    `self-contained-migration-inventory.md`
14. `docs/START_HERE.md`, `docs/plans/rho-parallel-gepa-completion.md`,
    `reference/gaia_evolution_core/README.md`

## Working Rules

- **Do not claim something works because tests pass.** A green suite proves code
  *runs*, never that it runs *in production*: across all 2106 collected cases,
  `core/merge.py` and `core/parallel.py` are both fully tested and unreachable on
  the live path. When reporting a verification, name the **files, modules or cases**
  it covered **and** what it excluded.
- **A reproduction is only evidence if it uses production-shaped inputs.** SV-1's
  original reproduction passed a value into a helper by hand that no production
  call site ever writes; the arithmetic was real and the scenario unreachable.
- **A mocked arm never measures preference.** If a mock rule dictated the output,
  the result establishes capability, not choice. Label it as such.
- **Prefer AST over grep** for any structural question about imports or callers.
- **Do not start cross-task mechanism pooling as a patch.** It needs
  content-addressed identity; the obvious anchoring approach is recorded as
  defective.
- Do not alter git state — no commit, stash, restore, clean, or discard — unless
  explicitly asked. The user commits.
