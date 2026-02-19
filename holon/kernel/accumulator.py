"""
Accumulator: Frequency-preserving streaming operations.

Mirrors holon-rs/src/accumulator.rs for cross-language parity.

Accumulators are essential for building baselines from streaming data
where frequency information matters. Unlike simple bundling, accumulators
preserve the fact that a pattern seen 100 times should contribute 100x
more than a pattern seen once.

## Key Insight

The critical difference between accumulator and bundle:
- `bundle([a, a, a, a, a])` = `a` (idempotent after threshold)
- `accumulator.add(a) * 5` preserves that `a` was seen 5 times

This matters for anomaly detection: common patterns should dominate
the baseline, so rare anomalies have low similarity.

## Quick Reference

- `create_accumulator(dimensions)` → Initialize empty accumulator
- `accumulate(accum, vec)` → Add vector (frequency-preserving)
- `accumulate_weighted(accum, vec, weight)` → Add with weight
- `decay(accum, factor)` → Apply exponential decay
- `normalize_accumulator(accum)` → Get unit vector for similarity
- `threshold_accumulator(accum)` → Convert to bipolar
- `merge_accumulators(a, b)` → Combine accumulators
- `capacity(accum, codebook_size)` → Remaining capacity estimate
- `purity(accum)` → Concentration measure (quantum-inspired)
- `participation_ratio(accum)` → Effective number of active dimensions
"""

import numpy as np


def create_accumulator(dimensions: int) -> np.ndarray:
    """
    Create a new empty accumulator.

    Args:
        dimensions: Vector dimensionality

    Returns:
        Zero-initialized float64 accumulator
    """
    return np.zeros(dimensions, dtype=np.float64)


def accumulate(accumulator: np.ndarray, example: np.ndarray) -> np.ndarray:
    """
    Add example to a running sum WITHOUT thresholding.

    Unlike bundling which thresholds after each addition (losing
    frequency information), accumulate() preserves the actual frequency
    signal by keeping a float sum.

    Args:
        accumulator: Running float sum (np.float64)
        example: New vector to add (bipolar int8)

    Returns:
        Updated accumulator (float64)
    """
    return accumulator + example.astype(np.float64)


def accumulate_weighted(
    accumulator: np.ndarray, example: np.ndarray, weight: float
) -> np.ndarray:
    """
    Add example with a specific weight.

    Args:
        accumulator: Running float sum
        example: New vector to add
        weight: Weight to apply to example

    Returns:
        Updated accumulator
    """
    return accumulator + weight * example.astype(np.float64)


def decay(accumulator: np.ndarray, factor: float = 0.99) -> np.ndarray:
    """
    Apply exponential decay to an accumulator.

    Useful for time-weighted baselines where recent patterns
    should have more influence than older ones.

    Args:
        accumulator: Float accumulator
        factor: Decay multiplier (0.99 = 1% decay, 0.9 = 10% decay)

    Returns:
        Decayed accumulator
    """
    return accumulator * factor


def normalize_accumulator(accumulator: np.ndarray) -> np.ndarray:
    """
    Normalize an accumulator for similarity queries.

    Returns a unit-normalized float vector suitable for cosine similarity.
    This preserves the frequency weighting: dimensions with high agreement
    (many +1s or many -1s) have larger magnitudes.

    Args:
        accumulator: Float accumulator from accumulate()

    Returns:
        Unit-normalized float32 vector
    """
    norm = np.linalg.norm(accumulator)
    if norm < 1e-10:
        return np.zeros(len(accumulator), dtype=np.float32)
    return (accumulator / norm).astype(np.float32)


def threshold_accumulator(accumulator: np.ndarray) -> np.ndarray:
    """
    Threshold an accumulator to bipolar {-1, 0, 1}.

    Converts the float accumulator back to a standard bipolar vector.
    Use this if you need to compose with other VSA operations.

    Note: This loses some frequency information compared to using
    normalize_accumulator() for similarity queries.

    Args:
        accumulator: Float accumulator from accumulate()

    Returns:
        Bipolar int8 vector
    """
    return np.where(accumulator > 0, 1, np.where(accumulator < 0, -1, 0)).astype(
        np.int8
    )


def merge_accumulators(accum_a: np.ndarray, accum_b: np.ndarray) -> np.ndarray:
    """
    Merge two accumulators into one.

    Useful for parallel processing: accumulate in separate threads,
    then merge results.

    Args:
        accum_a: First accumulator
        accum_b: Second accumulator

    Returns:
        Merged accumulator
    """
    return accum_a + accum_b


def clear_accumulator(accumulator: np.ndarray) -> np.ndarray:
    """
    Clear an accumulator to start fresh.

    Args:
        accumulator: Accumulator to clear

    Returns:
        Zero-initialized accumulator
    """
    return np.zeros_like(accumulator)


def capacity(accumulator: np.ndarray, codebook_size: int) -> float:
    """
    Estimate how close the accumulator is to saturation.

    As more vectors are added, the accumulator's ability to distinguish
    individual components degrades. This estimates the remaining capacity
    as a fraction of theoretical maximum.

    Based on the capacity bound: N ≤ d / (2 * ln(k)) for d-dimensional
    vectors with k codebook items, where N is the max number of items
    that can be reliably retrieved.

    Args:
        accumulator: Float accumulator from accumulate()
        codebook_size: Number of distinct vectors that may be stored

    Returns:
        Remaining capacity as a fraction in [0.0, 1.0]
        - 1.0 = completely empty
        - 0.0 = fully saturated (no further items can be reliably stored)
    """
    d = len(accumulator)
    if codebook_size < 2:
        return 1.0

    # Theoretical max items for reliable retrieval
    max_items = d / (2.0 * np.log(codebook_size))

    # Estimate current load from the magnitude
    # Each accumulated vector contributes ~sqrt(d) to the norm
    norm = np.linalg.norm(accumulator)
    estimated_items = (norm**2) / d

    remaining = max(0.0, 1.0 - estimated_items / max_items)
    return float(remaining)


def purity(accumulator: np.ndarray) -> float:
    """
    Quantum-inspired purity measure: how concentrated is the accumulator?

    Purity indicates whether the accumulator represents a single concept
    (high purity) or a diffuse superposition of many (low purity).

    For a single bipolar vector of dimension d, sum(v_i^2) = d, so
    purity = d / d = 1.0. For N random bipolar vectors accumulated,
    sum(v_i^2) ≈ N*d, so purity ≈ 1/N.

    Analogous to Tr(ρ²) from quantum mechanics where a pure state has
    purity 1 and a maximally mixed state has purity 1/N.

    Args:
        accumulator: Float accumulator from accumulate()

    Returns:
        Purity score in (0.0, 1.0]
        - 1.0 = single clean vector
        - ~1/N = N dissimilar vectors accumulated
    """
    d = len(accumulator)
    if d == 0:
        return 0.0

    v = accumulator.astype(np.float64)
    l2_sq = np.sum(v**2)

    if l2_sq < 1e-10:
        return 0.0

    return float(min(d / l2_sq, 1.0))


def participation_ratio(accumulator: np.ndarray) -> float:
    """
    Participation ratio: effective number of active dimensions.

    A baseline-free measure of how many dimensions contribute meaningfully
    to the accumulator's energy. Reciprocal of purity.

    PR = (sum v_i^2)^2 / sum(v_i^4)

    For a single bipolar vector of dimension d: PR = d (all dimensions contribute).
    As structure concentrates into fewer dimensions, PR decreases.

    Args:
        accumulator: Float accumulator from accumulate()

    Returns:
        Participation ratio (1.0 to d). Higher = more diffuse/uniform.
    """
    v = accumulator.astype(np.float64)
    l2_sq = np.sum(v**2)

    if l2_sq < 1e-10:
        return 0.0

    l4_sum = np.sum(v**4)

    if l4_sum < 1e-10:
        return 0.0

    return float(l2_sq**2 / l4_sum)
