"""Ambient run correlation for LLM call capture.

The observability proxy records every LLM call verbatim, but mitmproxy sees only
bytes on a socket: it cannot know which candidate, task or rollout produced a
call. The addon lifts ``X-AE-*`` request headers into the capture record and
**strips them before the request goes upstream**, so no vendor endpoint ever
receives internal experiment identifiers
(``docker/observability/addons/correlate.py:139-152``).

This module is the label side. The capture side has been working and verified.
As of 2026-08-24 (?03) the production call sites OPEN scopes: every fault
diagnosis opens ``phase="diagnose"`` (legacy sequential AND the parallel
fan-out, whose labels travel into worker threads via
``ParallelAnalysisRunner.run(labels=...)`` because pool threads do not inherit
the submitting thread's context), and every Judge-2 success analysis opens
``phase="positivity"``. The dedup adjudicator inherits ambient scope during
clustering. Still unlabelled by design: adapter routes that bypass the LiteLLM
wrappers entirely (`run_workspace_agent`, `CugaEditorAgent`) and the RHO path's
two adapters.

**Why ambient rather than an explicit parameter.** The four
``_litellm_completion`` wrappers sit at the bottom of long call chains, while the
correlation facts are known only at the top. Threading five parameters through
every intermediate signature would touch far more code than the problem warrants
and would still silently lose correlation on any path that forgot to forward
them.

**Why ``contextvars`` rather than a module global.** ``parallel_execution`` is a
supported feature gate. Under a global, one worker's candidate id would label
another worker's calls -- silently attributing evidence to the wrong candidate,
which is unrecoverable after the fact and worse than having no correlation at
all. A context variable is naturally per-task and per-thread.

**Absent facts are omitted, never blanked.** An empty header value is
indistinguishable from a genuinely empty identifier. A capture that is honestly
silent about the candidate can be recognised as uncorrelated; one that says
``candidate=""`` looks like data.

Agent-neutral by construction: this module builds a header mapping and knows
nothing about any provider or adapter.
"""
from __future__ import annotations

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

#: The five keys the proxy addon recognises. It reads
#: ``X-AE-{key.capitalize()}``, so this ordering also fixes the header spelling.
#: A header the addon does not recognise is dropped silently, with no error
#: anywhere, so these names are pinned by test rather than left to convention.
CORRELATION_KEYS = ("run", "candidate", "task", "rollout", "phase")


@dataclass(frozen=True, slots=True)
class CorrelationContext:
    """Which run, candidate, task, rollout and phase a call belongs to.

    Frozen: a correlation label must not be editable after the scope is entered,
    or a call could be attributed to a candidate that did not make it.
    """

    run: str | None = None
    candidate: str | None = None
    task: str | None = None
    rollout: int | str | None = None
    phase: str | None = None

    def headers(self) -> dict[str, str]:
        """Render as ``X-AE-*`` headers, omitting absent facts.

        ``rollout`` is naturally an ``int`` while HTTP header values must be
        strings, so every value is coerced. ``rollout=0`` is a real rollout
        index and must survive, so the test is ``is not None`` rather than
        truthiness.
        """
        out: dict[str, str] = {}
        for key in CORRELATION_KEYS:
            value = getattr(self, key)
            if value is None or value == "":
                continue
            out[f"X-AE-{key.capitalize()}"] = str(value)
        return out


_CURRENT: contextvars.ContextVar[CorrelationContext | None] = contextvars.ContextVar(
    "ae_correlation", default=None
)


def current_correlation() -> CorrelationContext | None:
    """The correlation in force, or ``None`` outside any scope."""
    return _CURRENT.get()


def correlation_headers() -> dict[str, str]:
    """Headers for the current scope; empty outside one.

    Returning ``{}`` rather than inventing placeholders keeps an uncorrelated
    call recognisable as uncorrelated.
    """
    ctx = _CURRENT.get()
    return ctx.headers() if ctx is not None else {}


@contextmanager
def correlation_scope(
    *,
    run: str | None = None,
    candidate: str | None = None,
    task: str | None = None,
    rollout: int | str | None = None,
    phase: str | None = None,
) -> Iterator[CorrelationContext]:
    """Label every LLM call made inside this block.

    Nests: an inner scope replaces the outer one for its duration and the outer
    is restored on exit, including when the body raises -- a failed rollout must
    not leave its label attached to subsequent calls.
    """
    ctx = CorrelationContext(
        run=run, candidate=candidate, task=task, rollout=rollout, phase=phase
    )
    token = _CURRENT.set(ctx)
    try:
        yield ctx
    finally:
        _CURRENT.reset(token)
