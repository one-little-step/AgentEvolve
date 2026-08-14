# CUGA Wrapper Full Harness Injection Design

## Status

Approved 2026-08-13. Extends the focused wrapper fix so the full harness
(`instructions`, `tools`, `skills`, `policies`, `memory`) is materialized and
injected into the real CUGA runtime.

## Verified SDK mapping

| Harness | SDK surface | Notes |
| --- | --- | --- |
| `instructions` | `special_instructions=` | done |
| `tools` | `tools=` | done |
| `skills` | `enable_skills=True` + `skills_folder=<ws>` | discovers `<ws>/skills/**/SKILL.md`, injects `load_skill` tool. `settings.skills.enabled` defaults False. |
| `policies` | `cuga_folder=<ws>` + `auto_load_policies=True` | scans `<ws>/{playbooks,output_formatters,tool_guides,intent_guards,tool_approvals}/*.md` |
| `memory` | `enable_knowledge=True` + `agent.knowledge.ingest(<file>)` | knowledge engine is the retrieval path (user-approved) |

## Harness shape (existing contract)

```python
{
    "version": "b1-v2",
    "instructions": str | None,
    "skills": {name: body},
    "memory": {key: value},
    "tools": [BaseTool, ...],
    "policies": {name: content},
    "input": str,
}
```

## Design

### 1. `materialize_harness(harness, workspace_dir) -> str | None`

Writes a fresh per-run `.cuga`-style folder:

- skills: `workspace_dir/skills/<name>/SKILL.md` with frontmatter `name`,
  `description` (first body line, truncated) + body.
- policies: `workspace_dir/playbooks/<name>.md` with frontmatter `name`,
  `triggers: {always: true}` + content.
- memory: `workspace_dir/memory/<key>.md` (staged for knowledge ingest).

Returns `workspace_dir` when any of skills/policies/memory present, else `None`.
Skill/policy names are sanitized to safe path segments.

### 2. `_construct_agent(harness, default_tools, default_instructions, workspace_dir=None)`

Adds: `enable_skills=has_skills`, `skills_folder=<ws>`,
`cuga_folder=<ws>`, `auto_load_policies=has_policies`. Keeps
`enable_knowledge=True`.

### 3. `CugaSdkRuntime.run_task`

1. `materialize_harness` -> workspace_dir.
2. Build agent via factory `(config, workspace_dir)`.
3. If memory present, `await agent.knowledge.ingest(<ws>/memory/<key>.md)` for
   each entry inside the same async context as `invoke`.
4. Invoke with `track_tool_calls=True`.

`CugaSdkRuntime` gains a `workspace_root` (default `data/workspaces`) so tests
can isolate materialization.

### 4. Report

`_artifact_metadata` now reports skills/memory/policies as `active_artifacts`
when materialized instead of `unavailable_artifacts`.

## Testing

- Deterministic (fake `cuga` module): `materialize_harness` writes correct
  files/frontmatter; `_construct_agent` passes `enable_skills`/`skills_folder`/
  `cuga_folder`/`auto_load_policies` per harness shape; memory docs staged.
- Update the three existing SDK-runtime tests to the new factory signature
  `(config, workspace_dir)`.
- Live verification: skill `load_skill` works; policy loads (`Loaded N
  policies`); knowledge `ingest` + retrieval works.

## Non-goals

- No merge/parallel, no checkpoint/replay, no restructure into
  `manifest.py`/`workspace.py` target files yet.
