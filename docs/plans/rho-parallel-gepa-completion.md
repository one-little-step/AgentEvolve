# RHO-Parallel-GEPA Completion Roadmap

## Preconditions

- CUGA SDK/docs must be inspected before any CUGA adapter code is written.
- Core modules must remain free of CUGA imports.
- Every stage must have tests and tee-captured verification output.

## Phases

1. **Core contract and CUGA reconnaissance**
   - Confirm public SDK package, artifacts, traces, checkpoints, and concurrency.
   - Implement a fake adapter test suite before the real adapter.

2. **Minimal profile / B0 versus B1**
   - Persistent pool, common outcome score provenance, base plus all RHO proposals.
   - RHO severity/self-consistency editor, sequential execution, fixed coreset.

3. **Causal and memory profile / B2-B3**
   - Analyzer+judge causal graphs, mechanism clustering, structured attempts,
     worked/regression/failed state, retry budget, focused validation.

4. **Entropy and merge profile / B4-B5**
   - Comparable-score floors, incremental entropy heap, hierarchical DPP,
     deterministic provenance merge, deferred probes.

5. **Parallel profile / B6**
   - Snapshot/read-write lease manager, batch coordinator, concurrency tests,
     barrier-only entropy refresh.

6. **Ablations and CUGA validation**
   - Run profiles under matched budgets, publish manifests and metrics, then
     decide whether to add Pi or Gaia compatibility adapters.
