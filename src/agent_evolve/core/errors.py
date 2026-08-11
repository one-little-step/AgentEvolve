"""Typed domain errors for agent-neutral evolution contracts."""
from __future__ import annotations


class EvolutionContractError(ValueError):
    """Base class for invalid agent-neutral evolution records."""


class ScoreProvenanceError(EvolutionContractError):
    """Raised when score provenance cannot support a valid evidence cell."""


class ScoreRangeError(EvolutionContractError):
    """Raised when a normalized score-like value is outside [0, 1]."""


class AttemptRecordError(EvolutionContractError):
    """Raised when an attempt record has inconsistent terminal state."""


class WriteAuthorizationError(EvolutionContractError):
    """Raised when an edit targets an artifact outside its write authorization."""
