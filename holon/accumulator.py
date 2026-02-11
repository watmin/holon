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
