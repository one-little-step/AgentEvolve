# Task 4 Report: Self-Contained CUGA Continuation Guidance

## Status

Completed.

All required documents were created or updated, the specified decision and
import-boundary tests were run with tee capture, and the changes were committed.

## Scope Executed

### Documents Created

1. `docs/vision-and-decision-record.md`
   - Includes the mandatory literal sections: Vision, Approved Target,
     CUGA Boundary, and First Implementation Path.
   - Enumerates 12 settled decisions with rationale and source links.
   - Distinguishes historical versus active material.
   - Lists explicit deferrals.
   - States the "do not redo" instruction.
   - Provides links to all relevant entry, architecture, research, migration,
     and reference documents.

2. `docs/migration/cuga-adaptation-guide.md`
   - Contains the required historical-to-active mapping table.
   - Describes reference-module reuse and limitations from
     `reference/gaia_evolution_core/README.md`.
   - Provides a detailed CUGA inspection checklist covering package/version/license,
     artifacts, traces, tool/subagent provenance, candidate workspaces, state
     checkpoints, replay validity, concurrency behavior, and error semantics.
   - Includes an empty adapter-mapping table to be filled after SDK inspection.

3. `docs/migration/self-contained-migration-inventory.md`
   - Lists the 21-file `docs/rho_evolution/` archive.
   - Lists the five read-only `reference/gaia_evolution_core/` modules.
   - Enumerates intentionally excluded material (Gaia runtime, datasets, evaluator
     internals, credentials, CUGA source).
   - Identifies `src/agent_evolve/` as the only active implementation location.

### Documents Modified

1. `docs/START_HERE.md`
   - Expanded the decision record description to mention the CUGA boundary.
   - Added `migration/self-contained-migration-inventory.md` to required reading.
   - Clarified that the adaptation guide maps historical concepts to CUGA-neutral
     capabilities without inventing SDK APIs.
   - Added a fresh-agent instruction to read local materials before changing
     architecture or attempting CUGA integration.
   - Added two fresh-agent first actions: read the vision record and inventory.

2. `AGENTS.md`
   - Added `docs/migration/self-contained-migration-inventory.md` to the required
     reading order.
   - Added a fresh-agent instruction to use local materials before CUGA integration
     and not to invent CUGA APIs, artifact types, trace fields, checkpoint
     behavior, replay semantics, or package names.

3. `README.md`
   - Added `docs/migration/self-contained-migration-inventory.md` to the Start Here
     list.
   - Added the same fresh-agent instruction about local materials and not inventing
     CUGA APIs.

## Test Execution

Command run:

```bash
uv run pytest tests/test_self_contained_migration.py::test_continuation_brief_names_vision_decisions_and_cuga_boundary tests/test_self_contained_migration.py::test_active_package_has_no_legacy_or_adapter_runtime_imports -v 2>&1 | tee terminal_output/migration/08_decision_boundary_tests.log
```

Result: **2 passed**.

Additional sanity check:

```bash
uv run pytest tests/test_self_contained_migration.py -v
```

Result: **5 passed** (archive presence, reference baseline, vision/boundary,
import boundary, and broken-link checks).

## Commit

```text
73053aca0e468d5117a6f39a65e84edc258a6416
```

Message: `docs: add self-contained CUGA continuation guidance`

Files in commit:

- `AGENTS.md`
- `README.md`
- `docs/START_HERE.md`
- `docs/vision-and-decision-record.md`
- `docs/migration/cuga-adaptation-guide.md`
- `docs/migration/self-contained-migration-inventory.md`

## Concerns

- The test `test_continuation_brief_names_vision_decisions_and_cuga_boundary`
  checks literal substrings (`base plus every RHO candidate`,
  `do not invent CUGA APIs`) that cross line boundaries in the source markdown.
  Initial drafts split these phrases across lines, causing two test failures.
  The final document keeps each required phrase contiguous on a single wrapped
  line. Future edits to this file should preserve these literal phrases exactly.

- `terminal_output/migration/08_decision_boundary_tests.log` was captured as
  required by `AGENTS.md`, but the brief's explicit commit command did not
  include it. The `terminal_output/` directory appears to be untracked and
  likely gitignored, so it was left out of the commit per the brief's
  instruction.

- The repository contains many untracked files and directories that predate this
  task (e.g., `src/`, `tests/`, `docs/architecture/`, `pyproject.toml`). Only the
  six files specified in the brief were committed.

- No source code outside `AgentEvolve/` was modified.

## Review Follow-Up

Reviewer finding: the CUGA Boundary section must reproduce the required imperative
sentence literally. Replaced the wording drift (`We do not invent`) with:

```text
Do not invent CUGA APIs, artifact types, trace fields, checkpoint behavior, replay semantics, or package names.
```

Test command:

```bash
uv run pytest tests/test_self_contained_migration.py::test_continuation_brief_names_vision_decisions_and_cuga_boundary -v 2>&1 | tee terminal_output/migration/08_decision_boundary_tests.log
```

Initial result: **1 failed**. The existing test requires the lowercase substring
`do not invent CUGA APIs`, while the review requires the uppercase imperative
sentence. The boundary now contains both the mandated sentence and a no-scope-
expansion restatement needed for the existing assertion.

Final result: **1 passed**.
