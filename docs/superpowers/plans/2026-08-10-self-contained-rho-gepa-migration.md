# Self-Contained RHO-Parallel-GEPA Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make AgentEvolve a self-contained RHO-Parallel-GEPA continuation repository with the full historical documentation, portable generic baseline code, and explicit CUGA adaptation guidance.

**Architecture:** Preserve the original RHO evolution documents as an internally linked historical archive under `docs/rho_evolution/`; keep active target summaries as navigational entry points. Copy the old generic evolution core into a non-importable `reference/gaia_evolution_core/` boundary, then provide a vision/decision record and adaptation guide that direct future work to the active, agent-neutral `src/agent_evolve/` package.

**Tech Stack:** Python 3.11+, Markdown, `uv`, `pytest`, standard-library structural validation.

## Global Constraints

- AgentEvolve must be comprehensible without the Gaia repository or this conversation.
- `src/agent_evolve/core/` must never import `cuga`, Gaia, or `reference` modules.
- CUGA integration is SDK-only after official API/source inspection; do not invent API names, artifact types, trace fields, checkpoint behavior, or package names.
- Preserve complete historical documentation with original filenames and working internal relative links.
- `reference/gaia_evolution_core/` is read-only baseline code and must never be imported by active code.
- Do not copy Gaia runtime, datasets, generated runs, feedback artifacts, model credentials, evaluator internals, expected answers, labels, or regexes.
- Capture every verification command using `2>&1 | tee terminal_output/migration/<name>.log`.
- Add structural tests before the migration implementation and run the full AgentEvolve test suite before completion.

---

### Task 1: Add Self-Containment Structural Tests

**Files:**
- Create: `tests/test_self_contained_migration.py`
- Read: `docs/superpowers/specs/2026-08-10-self-contained-rho-gepa-migration-design.md`
- Read: `docs/rho_evolution/README.md`
- Read: `reference/gaia_evolution_core/README.md`

**Interfaces:**
- Consumes: repository root resolved from `Path(__file__).parents[1]`.
- Produces: tests that verify required archive files, reference code, decision documents, local documentation links, and active-package import boundaries.

- [ ] **Step 1: Write the failing migration tests**

```python
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_complete_rho_evolution_archive_is_present() -> None:
    archive = ROOT / "docs" / "rho_evolution"
    expected = {
        "README.md",
        *{f"{index:02d}-{name}.md" for index, name in [
            (1, "overview"),
            (2, "data-model"),
            (3, "control-flow"),
            (4, "coreset-selection"),
            (5, "diagnosis"),
            (6, "candidate-generation"),
            (7, "pairwise-judging"),
            (8, "acceptance-and-promotion"),
            (9, "artifacts-and-versioning"),
            (10, "runner-and-batch-integration"),
            (11, "configuration-reference"),
            (12, "tracing-and-debugging"),
            (13, "rho-gepa-population-evolution"),
            (14, "agent-integration-and-history-rag"),
            (15, "rho-gepa-architecture-and-debugging"),
            (16, "rho-gepa-execution-atlas"),
            (17, "rho-gepa-prompts-and-data-contracts"),
            (18, "rho-parallel-gepa-target-architecture"),
            (19, "rho-parallel-gepa-research-hypotheses"),
        ]},
        "selection_algo_explaination.md",
    }
    assert {path.name for path in archive.glob("*.md")} == expected


def test_reference_baseline_is_complete_and_explicitly_non_importable() -> None:
    reference = ROOT / "reference" / "gaia_evolution_core"
    assert {path.name for path in reference.glob("*.py")} == {
        "__init__.py", "contracts.py", "history.py", "operators.py", "population.py"
    }
    readme = (reference / "README.md").read_text(encoding="utf-8")
    assert "read-only historical baseline" in readme
    assert "must never be imported" in readme


def test_continuation_brief_names_vision_decisions_and_cuga_boundary() -> None:
    brief = (ROOT / "docs" / "vision-and-decision-record.md").read_text(encoding="utf-8")
    for phrase in (
        "evolves externally configurable agent harnesses, not model weights",
        "persistent GEPA pool",
        "base plus every RHO candidate",
        "CUGA SDK",
        "do not invent CUGA APIs",
        "minimal",
    ):
        assert phrase in brief


def test_active_package_has_no_legacy_or_adapter_runtime_imports() -> None:
    forbidden = re.compile(r"^\s*(?:from|import)\s+(?:cuga|agent\.|dataset\.|reference\.|gaia)", re.MULTILINE)
    violations = [
        str(path.relative_to(ROOT))
        for path in (ROOT / "src" / "agent_evolve").rglob("*.py")
        if forbidden.search(path.read_text(encoding="utf-8"))
    ]
    assert violations == []


def test_copied_documentation_has_no_broken_relative_markdown_links() -> None:
    broken: list[str] = []
    for path in (ROOT / "docs" / "rho_evolution").glob("*.md"):
        text = path.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", text):
            if not (path.parent / target).resolve().exists():
                broken.append(f"{path.relative_to(ROOT)} -> {target}")
    assert broken == []
```

- [ ] **Step 2: Run the migration tests and verify failure**

Run:

```bash
uv run pytest tests/test_self_contained_migration.py -v 2>&1 | tee terminal_output/migration/05_self_contained_tests_red.log
```

Expected: FAIL because `docs/rho_evolution/`, `reference/gaia_evolution_core/`, and the continuation brief do not yet exist.

- [ ] **Step 3: Commit the failing tests**

```bash
git add tests/test_self_contained_migration.py
git commit -m "test: define self-contained migration contract"
```

### Task 2: Preserve Complete Historical Documentation Archive

**Files:**
- Create: `docs/rho_evolution/README.md`
- Create: `docs/rho_evolution/01-overview.md` through `docs/rho_evolution/19-rho-parallel-gepa-research-hypotheses.md`
- Create: `docs/rho_evolution/selection_algo_explaination.md`
- Modify: `docs/START_HERE.md`
- Modify: `README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: source archive at `../docs/rho_evolution/` relative to AgentEvolve.
- Produces: `docs/rho_evolution/` as the complete, internally linked historical documentation archive.

- [ ] **Step 1: Copy only the source Markdown archive**

Run:

```bash
mkdir -p docs/rho_evolution
cp ../docs/rho_evolution/*.md docs/rho_evolution/
```

Expected: `docs/rho_evolution/` contains exactly the 21 Markdown files asserted in `test_complete_rho_evolution_archive_is_present`.

- [ ] **Step 2: Add historical-status context without changing source technical content**

Prepend this block to `docs/rho_evolution/README.md` after its title:

```markdown
> **Historical archive status:** These documents preserve the detailed Gaia-era
> RHO and RHO-GEPA analysis that informed AgentEvolve. They are authoritative for
> established rationale, schemas, debugging evidence, and target architecture;
> Gaia-specific paths and runtime assumptions are historical examples, not active
> AgentEvolve dependencies. Read `../vision-and-decision-record.md` and
> `../migration/cuga-adaptation-guide.md` before implementing against CUGA.
```

Expected: the archive retains its original filenames, internal links, and detailed content while its role is clear to a fresh agent.

- [ ] **Step 3: Update entry points to require the complete archive**

Add this required-reading sequence to `AGENTS.md`, `README.md`, and `docs/START_HERE.md`:

```markdown
1. `docs/vision-and-decision-record.md`
2. `docs/rho_evolution/README.md`
3. `docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md`
4. `docs/rho_evolution/19-rho-parallel-gepa-research-hypotheses.md`
5. `docs/migration/cuga-adaptation-guide.md`
6. `reference/gaia_evolution_core/README.md`
```

Also state that the historical archive eliminates any need to rediscover the RHO-GEPA design from scratch.

- [ ] **Step 4: Run archive tests and verify pass**

Run:

```bash
uv run pytest tests/test_self_contained_migration.py::test_complete_rho_evolution_archive_is_present tests/test_self_contained_migration.py::test_copied_documentation_has_no_broken_relative_markdown_links -v 2>&1 | tee terminal_output/migration/06_archive_tests.log
```

Expected: PASS.

- [ ] **Step 5: Commit the documentation archive**

```bash
git add AGENTS.md README.md docs/START_HERE.md docs/rho_evolution
git commit -m "docs: preserve complete RHO-GEPA archive"
```

### Task 3: Preserve Generic Legacy Core as a Read-Only Reference

**Files:**
- Create: `reference/gaia_evolution_core/README.md`
- Create: `reference/gaia_evolution_core/__init__.py`
- Create: `reference/gaia_evolution_core/contracts.py`
- Create: `reference/gaia_evolution_core/history.py`
- Create: `reference/gaia_evolution_core/operators.py`
- Create: `reference/gaia_evolution_core/population.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: source generic baseline at `../agent/evolution_core/` relative to AgentEvolve.
- Produces: a non-importable reference snapshot for selective future porting.

- [ ] **Step 1: Copy the complete generic baseline source**

Run:

```bash
mkdir -p reference/gaia_evolution_core
cp ../agent/evolution_core/__init__.py ../agent/evolution_core/contracts.py ../agent/evolution_core/history.py ../agent/evolution_core/operators.py ../agent/evolution_core/population.py reference/gaia_evolution_core/
```

Expected: reference contains exactly the five Python modules asserted by `test_reference_baseline_is_complete_and_explicitly_non_importable`.

- [ ] **Step 2: Add an explicit reference boundary README**

Create `reference/gaia_evolution_core/README.md` with this content:

```markdown
# Gaia Evolution-Core Reference

This directory is a **read-only historical baseline** copied from the Gaia RHO-
GEPA effort. It preserves reusable implementation ideas and known behavioral
limitations for a future CUGA-neutral implementation. It is not production code,
is not part of the `src` package, and **must never be imported** by active
AgentEvolve code.

## What It Preserves

- Initial agent-neutral bundle, trajectory, adapter, editor, and LLM contracts.
- Append-only redacted edit history with lexical/semantic retrieval fallback.
- Editor-gated mutation and LLM-synthesis crossover protocols.
- Immutable generation artifacts, lineage manifests, rollout caching, and simple
  task-score Pareto selection.

## Why It Is Not The Target Implementation

The baseline has known gaps: parent-relative and synthetic score comparability,
elite-only retention instead of a persistent pool, round-robin target selection,
coarse edit-history outcomes, LLM-first rather than deterministic merge, and
Gaia-shaped module/Markdown assumptions. The approved target is documented in
`../../docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md`.

## Porting Rule

Port behavior selectively into `src/agent_evolve/` only after writing tests
against the active artifact and adapter contracts. Replace Gaia-shaped types with
declared artifact capabilities. Do not patch this snapshot and do not let it set
the CUGA API boundary.
```

- [ ] **Step 3: Ignore reference bytecode and retain source snapshot only**

Ensure `.gitignore` contains:

```gitignore
reference/**/__pycache__/
reference/**/*.py[cod]
```

- [ ] **Step 4: Run reference-boundary tests and verify pass**

Run:

```bash
uv run pytest tests/test_self_contained_migration.py::test_reference_baseline_is_complete_and_explicitly_non_importable -v 2>&1 | tee terminal_output/migration/07_reference_tests.log
```

Expected: PASS.

- [ ] **Step 5: Commit the baseline reference snapshot**

```bash
git add .gitignore reference/gaia_evolution_core
git commit -m "docs: retain Gaia evolution-core baseline reference"
```

### Task 4: Add Vision Record, CUGA Adaptation Guide, and Inventory

**Files:**
- Create: `docs/vision-and-decision-record.md`
- Create: `docs/migration/cuga-adaptation-guide.md`
- Create: `docs/migration/self-contained-migration-inventory.md`
- Modify: `docs/START_HERE.md`
- Modify: `AGENTS.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: complete archive, reference baseline, active `EvolutionAdapter` contract in `src/agent_evolve/adapters/base.py`, and active data contracts in `src/agent_evolve/core/contracts.py`.
- Produces: a self-contained decision record that maps historical RHO/GEPA concepts into active CUGA-neutral capabilities without claiming CUGA APIs.

- [ ] **Step 1: Create `docs/vision-and-decision-record.md`**

Include these mandatory sections and literal statements:

```markdown
# AgentEvolve Vision And Decision Record

## Vision

AgentEvolve evolves externally configurable agent harnesses, not model weights.
It improves reusable agent behavior by analyzing execution evidence, preserving
promising candidate variants, editing declared artifacts such as skills, memory,
policies, prompts, and workflows, and validating improvements against regressions.

## Approved Target

The approved target is RHO-Parallel-GEPA, not a request to reinvent an evolution
algorithm. Its pipeline is historical trajectories and DPP coreset -> base plus
every RHO candidate in a persistent GEPA pool -> provenance-bearing evaluation ->
causal blame and artifact-targeted edits -> structured edit memory and focused
regression validation -> entropy/DPP selection, deterministic merge, and optional
safe parallelism.

## CUGA Boundary

CUGA is the intended reference adapter through its SDK. Do not invent CUGA APIs,
artifact types, trace fields, checkpoint behavior, replay semantics, or package
names. Inspect official SDK documentation and source first; then map only proven
public capabilities to the active adapter contract.

## First Implementation Path

Implement and evaluate the `minimal` profile first: persistent pool, common
outcome-score provenance, base plus every RHO proposal, fixed historical coreset,
and sequential editing. Compare B0 and B1 under matched budget before enabling
causal blame, edit memory, entropy, merge, or parallelism.
```

Then enumerate the settled decisions, links to detailed source documents, what is historical versus active, explicit deferrals, and the “do not redo” instruction.

- [ ] **Step 2: Create `docs/migration/cuga-adaptation-guide.md`**

Create a table with these rows:

| Historical concept | Target AgentEvolve capability | CUGA investigation required |
| --- | --- | --- |
| `EvolutionBundle.modules` | `EvolutionCandidate` plus adapter `ArtifactDescriptor` inventory | Exact CUGA artifact grouping, versioning, and write policy |
| Gaia wisdom module | Any declared artifact kind: skill, memory, policy, prompt, workflow, or adapter-defined unit | CUGA artifact metadata and edit surface |
| `NormalizedTrajectory.events` | `ExecutionTrace.events` with immutable provenance | CUGA event, tool, subagent, artifact-read, and final-output data |
| `run_rollouts` | `run_full_rollout` then `capture_trace` | Public task execution and trace retrieval APIs |
| legacy `open_editor` section operations | `apply_structured_edits` in a candidate workspace | Artifact mutation/override and lifecycle APIs |
| legacy replay absence | optional `discover_checkpoints` and `replay_from_checkpoint` | Valid checkpoint/state reconstruction and artifact dependency boundary |
| legacy score map | common provenance-bearing score tensor | CUGA evaluator/task-contract integration |

After the table, list reference-module reuse and limitations from the reference README, then the CUGA inspection checklist: package/version/license, artifacts, traces, tool/subagent provenance, candidate workspaces, state checkpoints, replay validity, concurrency behavior, and error semantics.

- [ ] **Step 3: Create migration inventory**

Create `docs/migration/self-contained-migration-inventory.md` with sections:

```markdown
# Self-Contained Migration Inventory

## Included Documentation

`docs/rho_evolution/` preserves the complete 21-file source archive, including
the execution atlas, prompt/data contracts, target architecture, research
hypotheses, current implementation analysis, and debugging record.

## Included Baseline Code

`reference/gaia_evolution_core/` preserves `contracts.py`, `history.py`,
`operators.py`, `population.py`, and `__init__.py` as read-only reference.

## Intentionally Excluded Material

- Gaia runtime adapters and agent implementation.
- Datasets, task fixtures, generated artifacts, run outputs, and caches.
- Feedback inputs, credentials, secrets, expected answers, evaluator internals,
  labels, and regexes.
- Any CUGA source or guessed SDK dependency.

## Active Implementation Location

New implementation belongs only in `src/agent_evolve/` and must pass active
tests. Reference code and historical documentation are inputs to design and
selective porting, never runtime dependencies.
```

- [ ] **Step 4: Update all entry documents**

Update `docs/START_HERE.md`, `AGENTS.md`, and `README.md` to link to the vision record, full archive, adaptation guide, reference baseline, and inventory. State that a fresh agent must use these local materials before changing architecture or attempting CUGA integration.

- [ ] **Step 5: Run decision and import-boundary tests**

Run:

```bash
uv run pytest tests/test_self_contained_migration.py::test_continuation_brief_names_vision_decisions_and_cuga_boundary tests/test_self_contained_migration.py::test_active_package_has_no_legacy_or_adapter_runtime_imports -v 2>&1 | tee terminal_output/migration/08_decision_boundary_tests.log
```

Expected: PASS.

- [ ] **Step 6: Commit the self-contained continuation guidance**

```bash
git add AGENTS.md README.md docs/START_HERE.md docs/vision-and-decision-record.md docs/migration/cuga-adaptation-guide.md docs/migration/self-contained-migration-inventory.md
git commit -m "docs: add self-contained CUGA continuation guidance"
```

### Task 5: Verify Full Migration and Commit the Foundation

**Files:**
- Modify: `docs/superpowers/specs/2026-08-10-self-contained-rho-gepa-migration-design.md`
- Modify: `docs/superpowers/plans/2026-08-10-self-contained-rho-gepa-migration.md`

**Interfaces:**
- Consumes: all copied documents, reference code, continuation documents, and tests from Tasks 1-4.
- Produces: verified self-contained AgentEvolve migration with recorded test evidence.

- [x] **Step 1: Run the full test suite**

Run:

```bash
uv run pytest -q 2>&1 | tee terminal_output/migration/09_full_test_suite.log
```

Expected: all contract and migration tests PASS.

- [x] **Step 2: Run a source-boundary and link validation script**

Run:

```bash
uv run python - <<'PY' 2>&1 | tee terminal_output/migration/10_structural_validation.log
from pathlib import Path
import re

root = Path.cwd()
archive = root / "docs" / "rho_evolution"
reference = root / "reference" / "gaia_evolution_core"
errors = []

for path in archive.glob("*.md"):
    for target in re.findall(r"\[[^]]+\]\(([^)#]+)(?:#[^)]+)?\)", path.read_text(encoding="utf-8")):
        if not (path.parent / target).resolve().exists():
            errors.append(f"broken link: {path.relative_to(root)} -> {target}")

for path in (root / "src" / "agent_evolve").rglob("*.py"):
    text = path.read_text(encoding="utf-8")
    if re.search(r"^\s*(?:from|import)\s+(?:cuga|agent\.|dataset\.|reference\.|gaia)", text, re.MULTILINE):
        errors.append(f"active import boundary violation: {path.relative_to(root)}")

for required in (
    root / "docs" / "vision-and-decision-record.md",
    root / "docs" / "migration" / "cuga-adaptation-guide.md",
    root / "docs" / "migration" / "self-contained-migration-inventory.md",
    reference / "README.md",
):
    if not required.exists():
        errors.append(f"missing required file: {required.relative_to(root)}")

if errors:
    raise SystemExit("\n".join(errors))
print("Self-contained migration structural validation: PASS")
PY
```

Expected: `Self-contained migration structural validation: PASS`.

- [x] **Step 3: Check patch hygiene and inspect the staged change set**

Run:

```bash
git diff --check
git status --short
git diff --stat
```

Expected: no whitespace errors; only intended AgentEvolve source, documentation, tests, lockfile, and configuration changes appear.

- [ ] **Step 4: Commit remaining migration metadata and verification updates**

```bash
git add docs/superpowers/specs/2026-08-10-self-contained-rho-gepa-migration-design.md docs/superpowers/plans/2026-08-10-self-contained-rho-gepa-migration.md
git commit -m "docs: record self-contained migration plan"
```

- [ ] **Step 5: Verify final repository state before optional push**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: all migration commits are visible on `main`; do not push unless explicitly requested.
