# CUGA Wrapper And Adapter Architecture

## Purpose

The intended CUGA integration is an internal package hierarchy, not a CUGA fork

```text
AgentEvolve core
  -> CUGAAdapter
  -> cuga_wrapper immutable manifest/workspace
  -> fresh CUGA runtime built from that manifest
```

`cuga_wrapper` stores AgentEvolve-owned harness configuration and editable

## Reading Order

1. [Wrapper Architecture](wrapper-architecture.md) describes internal package
   ownership, immutable manifests, workspaces, and runtime construction.
2. [Trace And Provenance Mapping](trace-and-provenance-mapping.md) describes
   the neutral observation boundary without assuming undocumented trace fields.
3. [SDK Verification Matrix](sdk-verification-matrix.md) is a development-time
   evidence record required before adapter implementation.

## Target Package Shape

```text
src/agent_evolve/
  adapters/
    base.py
    cuga.py                    # future; only CUGA SDK import location
  cuga_wrapper/
    __init__.py
    manifest.py
    artifacts.py
    workspace.py
    runtime_spec.py
    observations.py
    builder.py
```

These files are target design. They do not imply that CUGA API names, artifact
