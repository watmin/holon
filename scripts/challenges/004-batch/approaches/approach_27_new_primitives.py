#!/usr/bin/env python3
"""
Approach 27: Exploring New VSA Primitives

Current VSA operations:
- Bind (⊙):    AND-like, creates associations
- Bundle (+):  OR-like, creates superpositions
- Permute (ρ): Creates sequences/ordering
- Negate (-):  NOT-like, removes components [NEW]

What else could we add?

This exploration tests potential new primitives for Holon.
"""

import sys
sys.path.insert(0, "/home/watmin/work/holon")

from typing import List, Optional, Set, Tuple, Dict
import numpy as np
import time

from holon import CPUStore

from common import similarity


def create_store(dimensions: int = 16384):
    return CPUStore(dimensions=dimensions)


# =============================================================================
# PRIMITIVE 1: AMPLIFY / RESONATE
# =============================================================================

def amplify(superposition: np.ndarray, component: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Strengthen a component's presence in a superposition.

    Opposite of negate - makes a component MORE prominent.

    amplify(A+B+C, B, 2.0) → B is now 3x stronger than A and C
    """
    result = superposition.astype(float) + strength * component.astype(float)
    return np.where(result > 0, 1, np.where(result < 0, -1, 0)).astype(np.int8)


def test_amplify():
    print("=" * 70)
    print("PRIMITIVE 1: AMPLIFY")
    print("=" * 70)

    store = create_store()
    A = store.vector_manager.get_vector("A")
    B = store.vector_manager.get_vector("B")
    C = store.vector_manager.get_vector("C")

    ABC = store.bundle([A, B, C])

    print("\nOriginal bundle:")
    print(f"  sim(ABC, A) = {similarity(ABC, A):.4f}")
    print(f"  sim(ABC, B) = {similarity(ABC, B):.4f}")
    print(f"  sim(ABC, C) = {similarity(ABC, C):.4f}")

    amplified = amplify(ABC, B, strength=2.0)
    print("\nAfter amplify(ABC, B, 2.0):")
    print(f"  sim(result, A) = {similarity(amplified, A):.4f}")
    print(f"  sim(result, B) = {similarity(amplified, B):.4f}  ← Should be higher!")
    print(f"  sim(result, C) = {similarity(amplified, C):.4f}")


# =============================================================================
# PRIMITIVE 2: CLEAN / SHARPEN
# =============================================================================

def clean(vec: np.ndarray, threshold_percentile: float = 50.0) -> np.ndarray:
    """
    Remove noise by keeping only the most significant components.

    Useful after many operations accumulate noise.
    """
    vec_float = vec.astype(float)
    abs_vec = np.abs(vec_float)

    # Keep only values above percentile threshold
    threshold = np.percentile(abs_vec, threshold_percentile)
    mask = abs_vec >= threshold

    result = np.zeros_like(vec_float)
    result[mask] = vec_float[mask]

    return np.where(result > 0, 1, np.where(result < 0, -1, 0)).astype(np.int8)


def test_clean():
    print("\n" + "=" * 70)
    print("PRIMITIVE 2: CLEAN / SHARPEN")
    print("=" * 70)

    store = create_store()

    # Create a noisy superposition
    vecs = [store.vector_manager.get_vector(f"v{i}") for i in range(20)]
    noisy = store.bundle(vecs)

    # Query for first vector
    v0 = vecs[0]
    print(f"\nNoisy bundle of 20 vectors:")
    print(f"  sim(noisy, v0) = {similarity(noisy, v0):.4f}")

    cleaned = clean(noisy, threshold_percentile=75)
    print(f"\nAfter clean(75th percentile):")
    print(f"  sim(cleaned, v0) = {similarity(cleaned, v0):.4f}")
    print(f"  Non-zero elements: {np.sum(cleaned != 0)} / {len(cleaned)}")


# =============================================================================
# PRIMITIVE 3: BLEND / INTERPOLATE
# =============================================================================

def blend(vec1: np.ndarray, vec2: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Weighted combination of two vectors.

    blend(A, B, 0.0) = A
    blend(A, B, 0.5) = midpoint
    blend(A, B, 1.0) = B
    """
    result = (1 - alpha) * vec1.astype(float) + alpha * vec2.astype(float)
    return np.where(result > 0, 1, np.where(result < 0, -1, 0)).astype(np.int8)


def test_blend():
    print("\n" + "=" * 70)
    print("PRIMITIVE 3: BLEND / INTERPOLATE")
    print("=" * 70)

    store = create_store()
    A = store.vector_manager.get_vector("concept_A")
    B = store.vector_manager.get_vector("concept_B")

    print("\nBlending A → B:")
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        blended = blend(A, B, alpha)
        print(f"  α={alpha:.2f}: sim(A)={similarity(blended, A):.4f}, sim(B)={similarity(blended, B):.4f}")


# =============================================================================
# PRIMITIVE 4: XOR / SYMMETRIC DIFFERENCE
# =============================================================================

def xor(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Symmetric difference: what's unique to each vector.

    XOR(A, B) = (A AND NOT B) OR (B AND NOT A)
             = what's in A but not B, plus what's in B but not A
    """
    # Where they differ, keep the stronger signal
    v1 = vec1.astype(float)
    v2 = vec2.astype(float)

    # XOR: where signs differ, keep; where same, zero out
    same_sign = (v1 * v2) > 0
    result = v1.copy()
    result[same_sign] = 0  # Zero out where they agree

    return np.where(result > 0, 1, np.where(result < 0, -1, 0)).astype(np.int8)


def test_xor():
    print("\n" + "=" * 70)
    print("PRIMITIVE 4: XOR / SYMMETRIC DIFFERENCE")
    print("=" * 70)

    store = create_store()
    A = store.vector_manager.get_vector("A")
    B = store.vector_manager.get_vector("B")
    C = store.vector_manager.get_vector("C")

    AB = store.bundle([A, B])
    BC = store.bundle([B, C])

    xor_result = xor(AB, BC)

    print("\nAB = bundle([A, B])")
    print("BC = bundle([B, C])")
    print("\nXOR(AB, BC) should be similar to A and C (the unique parts):")
    print(f"  sim(xor, A) = {similarity(xor_result, A):.4f}")
    print(f"  sim(xor, B) = {similarity(xor_result, B):.4f}  ← Should be lower (common)")
    print(f"  sim(xor, C) = {similarity(xor_result, C):.4f}")


# =============================================================================
# PRIMITIVE 5: PROTOTYPE / CONSENSUS
# =============================================================================

def prototype(vectors: List[np.ndarray], threshold: float = 0.5) -> np.ndarray:
    """
    Extract the common pattern from a set of vectors.

    Only keeps dimensions where majority agree.
    """
    if not vectors:
        return np.zeros_like(vectors[0])

    # Sum all vectors
    total = np.sum([v.astype(float) for v in vectors], axis=0)

    # Threshold: keep only where absolute majority agrees
    n = len(vectors)
    agreement_threshold = n * threshold

    result = np.zeros_like(total)
    result[total > agreement_threshold] = 1
    result[total < -agreement_threshold] = -1

    return result.astype(np.int8)


def test_prototype():
    print("\n" + "=" * 70)
    print("PRIMITIVE 5: PROTOTYPE / CONSENSUS")
    print("=" * 70)

    store = create_store()

    # Create vectors that share a common component
    common = store.vector_manager.get_vector("common")
    unique1 = store.vector_manager.get_vector("unique1")
    unique2 = store.vector_manager.get_vector("unique2")
    unique3 = store.vector_manager.get_vector("unique3")

    # Each vector = common + something unique
    v1 = store.bundle([common, unique1])
    v2 = store.bundle([common, unique2])
    v3 = store.bundle([common, unique3])

    proto = prototype([v1, v2, v3], threshold=0.5)

    print("\nThree vectors, each contains 'common' + unique:")
    print(f"  sim(prototype, common)  = {similarity(proto, common):.4f}  ← Should be high")
    print(f"  sim(prototype, unique1) = {similarity(proto, unique1):.4f}")
    print(f"  sim(prototype, unique2) = {similarity(proto, unique2):.4f}")
    print(f"  sim(prototype, unique3) = {similarity(proto, unique3):.4f}")


# =============================================================================
# PRIMITIVE 6: DIFFERENCE / DELTA
# =============================================================================

def difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """
    Compute what changed between two states.

    Returns a vector representing the change direction.
    """
    delta = after.astype(float) - before.astype(float)
    return np.where(delta > 0, 1, np.where(delta < 0, -1, 0)).astype(np.int8)


def test_difference():
    print("\n" + "=" * 70)
    print("PRIMITIVE 6: DIFFERENCE / DELTA")
    print("=" * 70)

    store = create_store()
    A = store.vector_manager.get_vector("A")
    B = store.vector_manager.get_vector("B")
    C = store.vector_manager.get_vector("C")

    before = store.bundle([A, B])
    after = store.bundle([A, B, C])

    delta = difference(before, after)

    print("\nBefore: bundle([A, B])")
    print("After:  bundle([A, B, C])")
    print("\nDelta (what was added):")
    print(f"  sim(delta, A) = {similarity(delta, A):.4f}")
    print(f"  sim(delta, B) = {similarity(delta, B):.4f}")
    print(f"  sim(delta, C) = {similarity(delta, C):.4f}  ← Should be highest (added)")


# =============================================================================
# PRIMITIVE 7: MASK / ATTENTION
# =============================================================================

def mask(vec: np.ndarray, attention: np.ndarray) -> np.ndarray:
    """
    Apply attention mask to focus on specific dimensions.

    Dimensions where attention is positive are kept.
    """
    # Use element-wise multiplication as mask
    result = vec.astype(float) * np.abs(attention.astype(float))
    return np.where(result > 0, 1, np.where(result < 0, -1, 0)).astype(np.int8)


def test_mask():
    print("\n" + "=" * 70)
    print("PRIMITIVE 7: MASK / ATTENTION")
    print("=" * 70)

    store = create_store()

    # Create a vector and a partial attention mask
    full = store.vector_manager.get_vector("full_concept")

    # Create attention that focuses on first half of dimensions
    attention = np.zeros(store.dimensions, dtype=np.int8)
    attention[:store.dimensions // 2] = 1

    masked = mask(full, attention)

    print(f"\nOriginal vector norm: {np.linalg.norm(full):.1f}")
    print(f"Masked vector norm:   {np.linalg.norm(masked):.1f}")
    print(f"sim(masked, full) = {similarity(masked, full):.4f}")


# =============================================================================
# PRIMITIVE 8: SIMILARITY THRESHOLD
# =============================================================================

def threshold_similarity(vec: np.ndarray, reference: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    Keep only dimensions that contribute to similarity with reference.

    Useful for extracting the "relevant" part of a vector.
    """
    v = vec.astype(float)
    r = reference.astype(float)

    # Where they agree (same sign), keep the value
    agree = (v * r) > 0
    result = np.zeros_like(v)
    result[agree] = v[agree]

    return np.where(result > 0, 1, np.where(result < 0, -1, 0)).astype(np.int8)


def test_threshold_similarity():
    print("\n" + "=" * 70)
    print("PRIMITIVE 8: SIMILARITY THRESHOLD")
    print("=" * 70)

    store = create_store()
    A = store.vector_manager.get_vector("A")
    B = store.vector_manager.get_vector("B")

    AB = store.bundle([A, B])

    # Extract part of AB that's similar to A
    a_part = threshold_similarity(AB, A)

    print("\nAB = bundle([A, B])")
    print("Extract part similar to A:")
    print(f"  sim(a_part, A) = {similarity(a_part, A):.4f}  ← Should be higher than original")
    print(f"  sim(a_part, B) = {similarity(a_part, B):.4f}")
    print(f"  sim(AB, A)     = {similarity(AB, A):.4f}  (original)")


# =============================================================================
# PRIMITIVE 9: NORMALIZE / UNIT
# =============================================================================

def normalize(vec: np.ndarray) -> np.ndarray:
    """
    Normalize to unit vector (for continuous operations).

    Note: This returns float, not bipolar.
    """
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec.astype(float)
    return vec.astype(float) / norm


def test_normalize():
    print("\n" + "=" * 70)
    print("PRIMITIVE 9: NORMALIZE")
    print("=" * 70)

    store = create_store()
    v = store.vector_manager.get_vector("test")

    print(f"\nOriginal norm: {np.linalg.norm(v):.4f}")
    normed = normalize(v)
    print(f"Normalized norm: {np.linalg.norm(normed):.4f}")


# =============================================================================
# PRIMITIVE 10: COMPOSE / CHAIN
# =============================================================================

def compose(operations: List[tuple]) -> np.ndarray:
    """
    Apply a sequence of operations.

    Each operation is (op_name, args) tuple.
    """
    # This is more of a utility than a primitive
    # But shows how primitives can be composed
    pass


# =============================================================================
# SUMMARY
# =============================================================================

def summarize_primitives():
    print("\n" + "=" * 70)
    print("NEW PRIMITIVE SUMMARY")
    print("=" * 70)
    print("""
EXISTING VSA PRIMITIVES:
========================
| Name      | Symbol | Operation           | Purpose                    |
|-----------|--------|---------------------|----------------------------|
| Bind      | ⊙      | A * B               | Create associations (AND)  |
| Bundle    | +      | A + B → threshold   | Create superpositions (OR) |
| Permute   | ρ      | rotate(A, k)        | Create sequences           |
| Negate    | -      | A - B → threshold   | Remove components (NOT)    |

PROPOSED NEW PRIMITIVES:
========================
| Name       | Operation              | Purpose                         |
|------------|------------------------|---------------------------------|
| Amplify    | A + α*B                | Strengthen a component          |
| Clean      | threshold(A, pct)      | Remove noise                    |
| Blend      | (1-α)*A + α*B          | Interpolate between concepts    |
| XOR        | (A AND NOT B) OR (B AND NOT A) | Symmetric difference   |
| Prototype  | consensus(A,B,C,...)   | Extract common pattern          |
| Difference | B - A                  | Compute what changed            |
| Mask       | A * |attention|        | Focus on specific dimensions    |
| Threshold  | keep_agreeing(A, ref)  | Extract relevant part           |
| Normalize  | A / ||A||              | Unit vector (for math ops)      |

MOST VALUABLE FOR HOLON:
========================
1. AMPLIFY   - Strengthen weak signals, useful for boosting matches
2. PROTOTYPE - Extract common patterns from examples
3. DIFFERENCE - Track changes between states
4. CLEAN     - Remove accumulated noise after many operations
""")


# =============================================================================
# MAIN
# =============================================================================

def main():
    test_amplify()
    test_clean()
    test_blend()
    test_xor()
    test_prototype()
    test_difference()
    test_mask()
    test_threshold_similarity()
    test_normalize()
    summarize_primitives()


if __name__ == "__main__":
    main()
