"""Post-hoc detection of ground-truth literals inside evolved artifacts.

Ground truth reaches the preference judge and the editor by explicit decision
(spec section 7), overriding the ``AGENTS.md`` no-labels rule. Containment is by
prompting, which means nothing in the data path prevents a candidate from
improving its score by carrying an answer in an artifact rather than by improving
procedure.

This module changes none of that. It runs AFTER a run and reports which exported
artifacts contain literals drawn from the dataset's answer keys, so a contaminated
harness is discovered here rather than by a reviewer at the worst possible moment.
It is observational only: it mutates nothing, blocks nothing, and restricts no
prompt.

Confidence is reported, not decided: a long distinctive phrase appearing in an
artifact is almost certainly memorization, whereas ``17`` legitimately appears in
ordinary prose.

Callers own the literals. This module deliberately contains no answer-key
constants, so importing it leaks nothing. Recovered literals must never be
written to a persisted log; report artifact ids and confidences instead.
"""
from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: Regexes that are placeholders rather than answer keys.
_PLACEHOLDERS = frozenset({r"(?i)\?", r"\?", r"(?i)\\?"})

#: At or above this length, a matched literal is distinctive enough that an
#: incidental occurrence is implausible.
HIGH_CONFIDENCE_MIN_LENGTH = 8

#: Leading inline-flag group, e.g. ``(?i)``.
_INLINE_FLAGS = re.compile(r"^\(\?[a-zA-Z]+\)")

#: Any backslash escape, e.g. ``\.`` or ``\ ``.
_ESCAPE = re.compile(r"\\(.)")


@dataclass(frozen=True, slots=True)
class ContaminationHit:
    """One ground-truth literal found inside one artifact."""

    artifact_id: str
    literal: str
    confidence: str


def literals_from_regexes(regexes: Sequence[str]) -> tuple[str, ...]:
    r"""Recover plain literals from ``\b``-delimited answer-key regexes.

    The dataset stores answers as word-boundary regexes with inline flags.
    Stripping the flags, the boundaries, and the backslash escapes recovers the
    literal that a memorizing artifact would contain.

    Placeholder patterns (a bare ``?``, meaning "no answer recorded") are
    dropped, as is anything that reduces to the empty string.
    """
    literals: list[str] = []
    for raw in regexes:
        if raw in _PLACEHOLDERS:
            continue
        text = _INLINE_FLAGS.sub("", raw)
        text = text.replace(r"\b", "")
        text = _ESCAPE.sub(r"\1", text)  # \. -> .  and  \<space> -> space
        text = text.strip()
        if not text or text == "?":
            continue
        literals.append(text)
    return tuple(dict.fromkeys(literals))


def scan_artifacts(
    artifacts: Mapping[str, str], literals: Sequence[str]
) -> tuple[ContaminationHit, ...]:
    """Report every (artifact, literal) co-occurrence, case-insensitively.

    Artifact ids are visited in sorted order so the report is deterministic.
    Returns an empty tuple when there is nothing to look for.
    """
    if not literals:
        return ()
    hits: list[ContaminationHit] = []
    for artifact_id in sorted(artifacts):
        haystack = artifacts[artifact_id].lower()
        for literal in literals:
            needle = literal.lower().strip()
            if not needle or needle not in haystack:
                continue
            confidence = (
                "high" if len(needle) >= HIGH_CONFIDENCE_MIN_LENGTH else "low"
            )
            hits.append(
                ContaminationHit(
                    artifact_id=artifact_id,
                    literal=literal,
                    confidence=confidence,
                )
            )
    return tuple(hits)
