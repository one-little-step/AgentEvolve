# CUGA SDK Integration Notes

## Decision

Use the official CUGA SDK as a version-pinned dependency. Do not fork CUGA.

```text
AgentEvolve core -> adapter protocol -> CUGAAdapter -> import cuga
```

The generic core must never import CUGA modules. Only
`src/agent_evolve/adapters/cuga.py` may do so after the SDK is inspected.

## Why SDK Instead Of Fork

- Preserves agent-neutral architecture and multi-adapter support.
- Avoids ownership of CUGA internals and upstream merge maintenance.
- Permits version pinning for reproducible experiments.
- Keeps CUGA-specific trace, skill, memory, and checkpoint mapping inside one
  adapter boundary.

## Inspection Checklist Before Coding

Do not write `CUGAAdapter` until the following are verified from official CUGA
documentation/source:

```text
SDK package and supported version
agent initialization API
task execution API
artifact types: skills, policies, memory, workflows, tools
artifact read/write/materialization APIs
trace/event export format and stable identifiers
state transition and subagent/tool provenance fields
checkpoint/resume API and its public support status
artifact override semantics during resume
thread safety and concurrent-agent execution limitations
license and test fixtures
```

## Adapter Mapping Required

| AgentEvolve capability | CUGA evidence needed |
| --- | --- |
| `artifact_inventory()` | Public listing of candidate-version skills/policies/memory/workflows |
| `read_artifacts()` | Stable content/read API and version/hash information |
| `materialize_candidate()` | Isolated candidate version/workspace mechanism |
| `apply_structured_edits()` | Safe structured artifact update API |
| `run_full_rollout()` | Public task execution interface |
| `capture_trace()` | Exact state, tool, subagent, and artifact-read provenance |
| `discover_checkpoints()` | Public checkpoint descriptors associated with trace state |
| `replay_from_checkpoint()` | Supported resume with artifact overrides |

## Replay Rule

Counterfactual replay is disabled unless CUGA exposes a valid public checkpoint
and state reconstruction contract. The adapter must return:

```python
supports_counterfactual_replay() -> False
```

until this is proven. The generic core then uses full rollouts for deferred
generalization probes.

## Read-Only Source Inspection

If SDK documentation is insufficient, clone CUGA under ignored `vendor/`:

```bash
git clone --depth=1 <official-cuga-url> vendor/cuga-agent
```

Do not edit this clone. Do not add it to this repository. Forking is a last
resort only when public SDK extension and upstream contribution paths cannot
provide a necessary capability.
