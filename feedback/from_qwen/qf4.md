### DIRECTIVE: FIX DPP SELECTION MATH AND LOGIC

**CONTEXT & FAILURE ANALYSIS:**
Your current implementation of the DPP (Determinantal Point Process) selector in `core/issues.py` and `core/entropy.py` is mathematically inverted. You implemented a greedy "top-K quality" or "similarity-rewarding" clusterer. A true DPP **penalizes** similarity to promote diversity. 

You are strictly bound by `docs/architecture/selection-algorithms.md` and the foundational math in `docs/rho_evolution/RHO_2606.05922.pdf` (Section 4.1). You will delete your current selection logic and replace it with the **Greedy MAP (Maximum A Posteriori) inference with Cholesky-style incremental log-determinant updates** defined below.

#### 1. THE KERNEL CONSTRUCTION ($L$)
The DPP operates on a positive semi-definite kernel matrix $L$, where the diagonal represents item quality and the off-diagonals represent quality-weighted similarity.

*   **Similarity:** $\text{sim}(i, j) = \max(0.0, \text{cosine\_similarity}(e_i, e_j))$
*   **Quality ($q_i$):** Derived from RHO's difficulty scaling: $q_i = \tilde{r}_i = \left( \frac{\max(r_i, \epsilon)}{\max_j(r_j, \epsilon)} \right)^\alpha$ (where $\alpha = \frac{\theta}{2(1-\theta)}$), combined with entropy and coverage weights as defined in `selection-algorithms.md`.
*   **Off-Diagonal:** $L_{ij} = q_i \times \text{sim}(i, j) \times q_j$
*   **Diagonal:** $L_{ii} = q_i^2 + \text{JITTER}$ (Default JITTER = $1e^{-9}$)

**CRITICAL RULE:** The kernel $L$ encodes that the probability of selecting a subset $Y$ is proportional to $\det(L_Y)$. Because the determinant of a matrix with highly similar (collinear) vectors approaches zero, **the DPP mathematically penalizes redundant items.** If your code adds similarity to quality, you are violating the architecture.

#### 2. THE ALGORITHM: GREEDY MAP WITH CHOLESKY UPDATES
Exact eigendecomposition is **FORBIDDEN** for $N > 100$. You must implement the $O(N^2)$ Greedy MAP approximation using incremental Schur complement updates (Cholesky-style). 

Here is the exact Python logic you must implement:

```python
import numpy as np
from typing import List, Set

def greedy_map_dpp(L: np.ndarray, k: int, min_gain: float = 1e-12) -> List[int]:
    """
    Selects k items from the kernel matrix L using Greedy MAP DPP.
    Selecting an item MUST reduce the marginal gain of similar items.
    
    Args:
        L: (N x N) positive semi-definite kernel matrix.
        k: Number of items to select.
        min_gain: Minimum marginal gain threshold to continue selection.
        
    Returns:
        List of selected indices.
    """
    N = L.shape[0]
    if k >= N:
        return list(range(N))

    selected: List[int] = []
    remaining: Set[int] = set(range(N))
    
    # d2[i] stores the current marginal gain (diagonal of the Schur complement)
    # Initialize with the diagonal of L (L[i][i])
    d2 = np.diag(L).copy() 
    
    # c[i] stores the projection coefficients for the Cholesky update
    c = [[] for _ in range(N)] 
    
    for _ in range(k):
        if not remaining:
            break
            
        # 1. Select the item 'j' with the maximum marginal gain
        # Break ties by ascending stable issue/attempt ID (deterministic)
        j = max(remaining, key=lambda i: (d2[i], -i)) 
        
        # 2. Check termination condition
        if d2[j] <= min_gain:
            break
            
        selected.append(j)
        remaining.remove(j)
        
        # 3. Update marginal gains for all remaining items (Cholesky Update)
        dj = np.sqrt(d2[j])
        for i in remaining:
            # Calculate the dot product of the projection coefficients
            dot_c = sum(a * b for a, b in zip(c[i], c[j]))
            
            # Calculate the new projection coefficient 'e'
            e = (L[i, j] - dot_c) / dj
            
            # Append to the projection history
            c[i].append(e)
            
            # UPDATE RULE: Selecting 'j' REDUCES the marginal gain of 'i'
            # This is the mathematical penalty for similarity.
            d2[i] = d2[i] - (e * e)
            
    return selected
```

#### 3. HIERARCHICAL EXECUTION & PREFILTERING
As mandated by `selection-algorithms.md`:
1.  **Prefilter:** You must prefilter the candidate pool to at most `GEPA_DPP_MAX_ITEMS` (default 100) using the entropy heap and quality ranking before constructing the $N \times N$ kernel matrix.
2.  **Hierarchical Selection:** 
    *   **Stage 1:** Select tasks using aggregate task entropy and task embeddings.
    *   **Stage 2:** Within each selected task, select mechanism clusters using mechanism entropy and mechanism embeddings.
3.  **Forbidden Implementations:**
    *   Do NOT use exact eigendecomposition or dense kernel factorization when $N > 100$.
    *   Do NOT implement any selector that adds similarity to quality (which rewards redundancy).
    *   Do NOT implement a selector that ignores `sim` and returns top-K by quality while naming it "DPP".
    *   Do NOT allow the output to depend on unseeded randomness.

#### 4. MANDATORY UNIT TESTS (The "Marginal-Gain" Proof)
You must write and pass the following unit test in `tests/test_issues.py` or `tests/test_entropy.py` to prove your implementation is correct. If this test fails, your DPP is broken.

```python
def test_dpp_penalizes_similarity_and_promotes_diversity():
    """
    REQUIRED TEST: Proves the DPP implementation penalizes similarity.
    Given two near-duplicate high-quality issues and one dissimilar high-quality issue,
    the selector MUST return one duplicate plus the dissimilar issue, NOT both duplicates.
    """
    # Setup: 3 items. 
    # Item 0 and Item 1 are near-duplicates (cosine sim = 0.99) with high quality (0.9).
    # Item 2 is completely dissimilar (cosine sim = 0.0) with high quality (0.9).
    
    # The kernel L will look approximately like this:
    # [[0.81, 0.80, 0.00],
    #  [0.80, 0.81, 0.00],
    #  [0.00, 0.00, 0.81]]
    
    L = np.array([
        [0.810, 0.801, 0.000],
        [0.801, 0.810, 0.000],
        [0.000, 0.000, 0.810]
    ])
    
    selected = greedy_map_dpp(L, k=2)
    
    # ASSERTION: The selector MUST pick Item 2 (the dissimilar one).
    # It MUST NOT pick both Item 0 and Item 1.
    assert 2 in selected, "DPP failed to select the dissimilar item. It is rewarding similarity!"
    assert not (0 in selected and 1 in selected), "DPP selected redundant duplicates. The Cholesky update is missing or inverted."
```

**EXECUTION ORDER:**
1. Delete the current flawed selection logic in `core/issues.py` and `core/entropy.py`.
2. Implement the `greedy_map_dpp` function exactly as specified above.
3. Wire it into the prefilter and hierarchical selection pipeline.
4. Run `pytest tests/test_issues.py tests/test_entropy.py`. 
5. Do not proceed to the orchestrator or parallel batch logic until the `test_dpp_penalizes_similarity_and_promotes_diversity` test passes. 

Acknowledge this directive and execute.