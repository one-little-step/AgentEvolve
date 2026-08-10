# Self-Contained Migration Inventory

## Included Documentation

`docs/rho_evolution/` preserves the complete 21-file source archive, including
the execution atlas, prompt/data contracts, target architecture, research
hypotheses, current implementation analysis, and debugging record.

Specifically, the archive contains:

- `docs/rho_evolution/README.md` — archive entry point.
- `docs/rho_evolution/01-overview.md` through `16-rho-gepa-execution-atlas.md` — full RHO-GEPA design and operation record.
- `docs/rho_evolution/17-rho-gepa-prompts-and-data-contracts.md` — prompt and data contracts.
- `docs/rho_evolution/18-rho-parallel-gepa-target-architecture.md` — historical target architecture.
- `docs/rho_evolution/19-rho-parallel-gepa-research-hypotheses.md` — research hypotheses and validation criteria.
- `docs/rho_evolution/selection_algo_explaination.md` — selection-algorithm rationale.

## Included Baseline Code

`reference/gaia_evolution_core/` preserves `contracts.py`, `history.py`,
`operators.py`, `population.py`, and `__init__.py` as read-only reference.

These modules are explicitly documented as a historical baseline and must never
be imported by active AgentEvolve code.

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

The active package structure is:

```text
src/agent_evolve/
  core/
    contracts.py          # agent-neutral data and adapter contracts
  adapters/
    base.py               # runtime adapter validation
    cuga.py               # future CUGA SDK adapter (after inspection)
```

All new capabilities must be tested against these contracts before any claim of
implementation is made.
