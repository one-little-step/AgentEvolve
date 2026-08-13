"""Typed domain errors for agent-neutral evolution contracts."""
from __future__ import annotations


class EvolutionContractError(ValueError):
    """Base class for invalid agent-neutral evolution records."""


class IdentityError(EvolutionContractError):
    """Raised when an identifier or content hash is invalid."""


class ScoreProvenanceError(EvolutionContractError):
    """Raised when score provenance cannot support a valid evidence cell."""


class ScoreRangeError(EvolutionContractError):
    """Raised when a normalized score-like value is outside [0, 1]."""


class AttemptRecordError(EvolutionContractError):
    """Raised when an attempt record has inconsistent terminal state."""


class PersistenceSafetyError(EvolutionContractError):
    """A value cannot be safely persisted after recursive redaction."""


class BudgetExceededError(EvolutionContractError):
    """A requested operation exceeds a resolved experiment budget."""


class WriteAuthorizationError(EvolutionContractError):
    """Raised when an edit targets an artifact outside its write authorization."""


class MergeProvenanceError(EvolutionContractError):
    """Raised when immutable merge provenance is internally inconsistent."""


class ValidationResultError(EvolutionContractError):
    """Raised when a validation result cannot support its decision."""


class MemoryRecordError(EvolutionContractError):
    """Raised when an append-only memory record is invalid or unsafe."""
