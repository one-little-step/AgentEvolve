"""Marginal-gain proof tests for the bounded greedy-MAP DPP selector."""
from __future__ import annotations

import numpy as np

from agent_evolve.core.entropy import greedy_map_dpp


def test_dpp_penalizes_similarity_and_promotes_diversity() -> None:
    """Two near-duplicates lose to an equally valuable dissimilar item."""
    kernel = np.array(
        [[0.810, 0.801, 0.000], [0.801, 0.810, 0.000], [0.000, 0.000, 0.810]]
    )

    selected = greedy_map_dpp(kernel, k=2)

    assert 2 in selected
    assert not (0 in selected and 1 in selected)


def test_dpp_uses_ascending_index_for_equal_marginal_gains() -> None:
    """Identical gains have a stable, ascending-ID tie break."""
    selected = greedy_map_dpp(np.eye(3), k=2)

    assert selected == (0, 1)


def test_dpp_stops_when_no_remaining_gain_exceeds_minimum() -> None:
    selected = greedy_map_dpp(np.zeros((2, 2)), k=1)

    assert selected == ()
