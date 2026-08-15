"""Staged-write boundary for the CUGA editor agent.

Pure Python: no CUGA, no filesystem, no network. The editor agent stages edits
incrementally through tool bodies and finalizes once, so every authorization
and cap rule is enforced at staging time with per-artifact feedback.

Rejections are RETURNED, never raised. An exception inside a CUGA tool body can
abort the whole agent run, which would turn a recoverable authorization mistake
into a lost attempt.
"""
from __future__ import annotations

import inspect
from dataclasses import dataclass, field

from agent_evolve.core.contracts import ArtifactEdit


def normalize_authored_content(content: str) -> str:
    """Strip the indentation an agent's source formatting adds to an artifact.

    The editor authors artifact bodies inside Python string literals in a
    sandbox. A body written in an indented block arrives with that indentation
    baked in: observed live, every line after the first carried four leading
    spaces. Markdown reads uniformly indented lines as a code block, so the
    skill would be materialized as a literal listing rather than instructions,
    and the agent that loads it gets prose it cannot follow.

    ``inspect.cleandoc`` rather than ``textwrap.dedent``: dedent computes the
    common prefix across *all* lines, and the first line of a triple-quoted
    literal is flush, so the common prefix is "" and dedent is a no-op on
    exactly the shape observed. cleandoc ignores the first line when computing
    the margin, which is the docstring convention that produced the defect.

    Relative indentation is preserved, so nested list items and fenced code
    blocks keep their structure.
    """
    if not content:
        return content
    return inspect.cleandoc(content)

# Created ids must carry the CUGA group first. ``_harness_slot`` in
# cuga_adapter.py accepts only ``instructions`` or a
# ``skills|policies|memory/<name>`` prefix, so a flat ``generated/<name>`` would
# raise ValueError at registration and the creation path would be dead code.
DEFAULT_CREATABLE_PREFIX = "skills/generated-"
DEFAULT_PER_ATTEMPT_CREATE_CAP = 2
DEFAULT_POOL_CREATE_CAP = 10


@dataclass(frozen=True, slots=True)
class StageOutcome:
    """Result of one staging operation."""

    accepted: bool
    reason: str = ""


@dataclass(slots=True)
class EditStagingArea:
    """Accumulates authorized edits for one editor attempt."""

    write_set: tuple[str, ...]
    creatable_prefix: str = DEFAULT_CREATABLE_PREFIX
    per_attempt_create_cap: int = DEFAULT_PER_ATTEMPT_CREATE_CAP
    pool_created_count: int = 0
    pool_create_cap: int = DEFAULT_POOL_CREATE_CAP
    _replaced: dict[str, str] = field(default_factory=dict)
    _created: dict[str, str] = field(default_factory=dict)
    _parents_read: set[str] = field(default_factory=set)

    # -------------------------------------------------------------- #
    # Writes
    # -------------------------------------------------------------- #
    def stage_replace(self, artifact_id: str, content: str) -> StageOutcome:
        if artifact_id not in self.write_set:
            return StageOutcome(
                False,
                f"{artifact_id!r} is not in the authorized write set "
                f"{sorted(self.write_set)}",
            )
        if not isinstance(content, str):
            return StageOutcome(False, "content must be a string")
        self._replaced[artifact_id] = normalize_authored_content(content)
        return StageOutcome(True, f"staged replacement for {artifact_id!r}")

    def stage_create(self, artifact_id: str, content: str) -> StageOutcome:
        if artifact_id in self.write_set:
            return StageOutcome(
                False,
                f"{artifact_id!r} already exists; use stage_replace instead",
            )
        if not artifact_id.startswith(self.creatable_prefix):
            return StageOutcome(
                False,
                f"created artifacts must start with {self.creatable_prefix!r}; "
                f"got {artifact_id!r}",
            )
        if len(artifact_id) <= len(self.creatable_prefix):
            return StageOutcome(False, "created artifact needs a name after the prefix")
        if not isinstance(content, str):
            return StageOutcome(False, "content must be a string")
        if artifact_id not in self._created and (
            len(self._created) >= self.per_attempt_create_cap
        ):
            return StageOutcome(
                False,
                f"per-attempt creation cap of {self.per_attempt_create_cap} reached",
            )
        if artifact_id not in self._created and (
            self.pool_created_count + len(self._created) >= self.pool_create_cap
        ):
            return StageOutcome(
                False,
                f"pool-wide creation cap of {self.pool_create_cap} reached",
            )
        self._created[artifact_id] = normalize_authored_content(content)
        return StageOutcome(True, f"staged creation of {artifact_id!r}")

    def unstage(self, artifact_id: str) -> StageOutcome:
        if self._replaced.pop(artifact_id, None) is not None:
            return StageOutcome(True, f"unstaged {artifact_id!r}")
        if self._created.pop(artifact_id, None) is not None:
            return StageOutcome(True, f"unstaged {artifact_id!r}")
        return StageOutcome(False, f"{artifact_id!r} is not staged")

    # -------------------------------------------------------------- #
    # Reads
    # -------------------------------------------------------------- #
    @property
    def created_count(self) -> int:
        return len(self._created)

    def staged_ids(self) -> tuple[str, ...]:
        return tuple(sorted({*self._replaced, *self._created}))

    def edits(self) -> tuple[ArtifactEdit, ...]:
        staged: list[ArtifactEdit] = [
            ArtifactEdit(
                artifact_id=artifact_id,
                operation="replace",
                payload={"content": content},
            )
            for artifact_id, content in self._replaced.items()
        ]
        staged.extend(
            ArtifactEdit(
                artifact_id=artifact_id,
                operation="create",
                payload={"content": content},
            )
            for artifact_id, content in self._created.items()
        )
        return tuple(sorted(staged, key=lambda e: e.artifact_id))

    # -------------------------------------------------------------- #
    # Provenance ledger
    # -------------------------------------------------------------- #
    def record_parent_read(self, parent_id: str) -> None:
        """Record that donor ``parent_id`` was actually read.

        This ledger -- not the agent's prose -- is the provenance source for
        ``parent_ids`` (spec §9), following the same tool-execution-over-
        narration principle that makes ``ingest_sdk_tool_calls`` correct.
        """
        if parent_id:
            self._parents_read.add(parent_id)

    def parents_read(self) -> tuple[str, ...]:
        return tuple(sorted(self._parents_read))
