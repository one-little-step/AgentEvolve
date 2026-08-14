# CUGA Wrapper Focused Fix Design

## Status

Approved 2026-08-13. Scope: focused fix of the existing
`src/agent_evolve/cuga_wrapper/` package so its CUGA SDK path is verified and
working, followed by a live verification run. No package restructure; the
architecture target shape (SDK-independent `cuga_wrapper` + `adapters/cuga.py`)
remains deferred.

## Goal

Make `CugaWrapper` + `CugaSdkRuntime` actually drive a live CUGA SDK agent in
autonomous mode, following the verified pattern recorded in
`reference/cuga_example_wrapper/run2.py` and
`reference/cuga_example_wrapper/docs/cuga-integration-learnings.md`. Then verify
end-to-end by asking a real agent to invoke all five custom tools.

## Non-Goals

- No `merge.py` / `parallel.py` work (Phase 5 deferred).
- No CUGA checkpoint/replay, counterfactual replay, or tracing claims.
- No restructure into the SDK-independent target package shape.
- No change to `adapters/cuga_adapter.py` (already delegates to `CugaWrapper`).

## Constraints

- `src/agent_evolve/core/` must never import `cuga`.
- Do not invent CUGA APIs; use only the verified `CugaAgent(tools=...,
  special_instructions=..., enable_knowledge=True)` constructor and
  `invoke(..., track_tool_calls=True)`.
- Never persist credentials; `RuntimeSettings.public_config()` excludes the key.
- TDD: add tests before implementation; capture commands with `2>&1 | tee`.

## Design

### 1. `.env` additions

Append CUGA integration variables, mirroring the existing `LITELLM_*` values:

```dotenv
CUGA_MODEL=openai/azure/gpt-5.6-luna
CUGA_BASE_URL=https://ete-litellm.ai-models.vpc-int.res.ibm.com
CUGA_API_KEY=<same value as LITELLM_API_KEY>
AGENT_SETTING_CONFIG=settings.openai.toml
SKILLS_ROOT=cuga
DYNACONF_ADVANCED_FEATURES__FORCE_AUTONOMOUS_MODE=true
DYNACONF_ADVANCED_FEATURES__CUGA_LITE_NL_AUTO_CONTINUE=true
```

`SKILLS_ROOT=cuga` maps to a project-owned `.cuga/skills` directory.

### 2. Wrapper environment setup (refinement 1)

A pre-import setup function performs, in order, before any `cuga` import:

1. `load_dotenv(<project>/.env)`.
2. Remove blank/whitespace-only `CUGA_CONFIGURATIONS_DIR` (treat as unset).
3. Resolve `AGENT_SETTING_CONFIG` to `settings.openai.toml`.
4. Resolve `SKILLS_ROOT` (`cuga` -> project `.cuga/skills`), fail if not a dir.
5. Map model/base_url/api_key onto `MODEL_NAME`, `OPENAI_BASE_URL`,
   `OPENAI_API_KEY`.

`CUGA` imports are forbidden at module top level in `cuga_wrapper/__init__.py`
and `cuga_wrapper/tools.py`. All CUGA imports (including `from cuga.config
import settings`) are deferred into `CugaSdkRuntime` methods so environment is
resolved before the SDK reads its Dynaconf/settings state.

### 3. `RuntimeSettings` env source

`from_env()` reads `CUGA_MODEL` / `CUGA_BASE_URL` / `CUGA_API_KEY` first, then
falls back to `LITELLM_MODEL` / `LITELLM_BASE_URL` / `LITELLM_API_KEY`. Keeps
backward compatibility with the pre-existing project `.env`.

### 4. Autonomous-mode guard

After the deferred CUGA import, if `settings.advanced_features
.force_autonomous_mode` is not `True`, raise `RuntimeError` directing the user
to set the `DYNACONF_*` variable.

### 5. Verified agent construction

`CugaSdkRuntime.from_settings` builds the agent with only the verified surface:

```python
CugaAgent(tools=<5 tools>, special_instructions=<str or None>, enable_knowledge=True)
```

Drops the invented `auto_load_policies`, `filesystem_sync`, `enable_skills`
kwargs. Uses `invoke(message, track_tool_calls=True)`.

### 6. Custom tools (`cuga_wrapper/tools.py`, refinement 3)

Five tools, each with module-scope imports and a try/except returning a
structured JSON error string rather than raising:

- `calculator(expression)`
- `web_search(query, max_results=5)`
- `web_fetch(url, max_chars=12000)`
- `wikipedia_search(query, max_results=5)`
- `save_note(note)`

On exception, return `{"status": "error", "message": "<reason>"}` so one failed
tool does not crash the whole agent execution.

### 7. Project-owned skills dir

Create `.cuga/skills/web_research/SKILL.md` (adapted from the reference's valid
skill) so `SKILLS_ROOT` resolves.

### 8. Verification

- Deterministic tests (no real CUGA/network): env setup + blank-config
  normalization, `CUGA_*` -> `LITELLM_*` fallback, autonomous guard raising,
  `build_agent` receiving verified kwargs via a fake `cuga` module, and tool
  error handling.
- Live verification prompts (refinement 2):
  1. "What tools and skills do you have?"
  2. "Execute the following 5 steps in order and report the result of each:
     1. Use the calculator tool to multiply 17 by 24.
     2. Use the web_search tool to find the current capital of Australia.
     3. Use the wikipedia_search tool to find the summary for 'Albert Einstein'.
     4. Use the web_fetch tool to retrieve the content of 'http://example.com'.
     5. Use the save_note tool to save the text 'All 5 tools verified
        successfully'. Report the exact output or success status for each of
        the 5 steps."

  Capture to `terminal_output/cuga_wrapper/verify-<timestamp>.log` with secrets
  redacted; report a per-tool success summary (not raw model text).

## Testing

TDD order: write failing tests first, then implement. Verify with
`uv run pytest -p no:cacheprovider`.
