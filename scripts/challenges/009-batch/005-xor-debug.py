#!/usr/bin/env python3
"""
Debug script: Why doesn't bind() help with XOR?

This investigates why the (feature_a, feature_b) interaction
doesn't improve XOR classification.
"""

import random
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from holon import CPUStore


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity."""
    dot = np.dot(a.astype(float), b.astype(float))
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def main():
    print("=" * 60)
    print("Debug: XOR Interaction Analysis")
    print("=" * 60)

    store = CPUStore(dimensions=4096)
    encoder = store.encoder

    # Encode the four XOR patterns
    print("\n1. Encoding XOR patterns...")

    # Feature vectors
    a0 = encoder.encode_data({"feature_a": "A0"})
    a1 = encoder.encode_data({"feature_a": "A1"})
    b0 = encoder.encode_data({"feature_b": "B0"})
    b1 = encoder.encode_data({"feature_b": "B1"})

    # Convert to numpy
    for name, vec in [("a0", a0), ("a1", a1), ("b0", b0), ("b1", b1)]:
        if hasattr(vec, 'cpu'):
            vec = vec.cpu().numpy()

    print(f"   a0 shape: {a0.shape}")

    # Check orthogonality of base vectors
    print("\n2. Base vector similarities (should be ~0 for orthogonal):")
    print(f"   sim(a0, a1) = {cosine_similarity(a0, a1):.3f}")
    print(f"   sim(b0, b1) = {cosine_similarity(b0, b1):.3f}")
    print(f"   sim(a0, b0) = {cosine_similarity(a0, b0):.3f}")
    print(f"   sim(a1, b1) = {cosine_similarity(a1, b1):.3f}")

    # Compute bindings
    print("\n3. Computing bindings (interactions)...")

    # XOR = 1 (positive class): A0B1 or A1B0
    bind_a0b1 = (a0 * b1).astype(np.int8)
    bind_a1b0 = (a1 * b0).astype(np.int8)

    # XOR = 0 (negative class): A0B0 or A1B1
    bind_a0b0 = (a0 * b0).astype(np.int8)
    bind_a1b1 = (a1 * b1).astype(np.int8)

    print("\n4. Binding similarities:")
    print("\n   Within positive class (XOR=1):")
    print(f"   sim(A0⊙B1, A1⊙B0) = {cosine_similarity(bind_a0b1, bind_a1b0):.3f}")

    print("\n   Within negative class (XOR=0):")
    print(f"   sim(A0⊙B0, A1⊙B1) = {cosine_similarity(bind_a0b0, bind_a1b1):.3f}")

    print("\n   Across classes:")
    print(f"   sim(A0⊙B1, A0⊙B0) = {cosine_similarity(bind_a0b1, bind_a0b0):.3f}")
    print(f"   sim(A0⊙B1, A1⊙B1) = {cosine_similarity(bind_a0b1, bind_a1b1):.3f}")
    print(f"   sim(A1⊙B0, A0⊙B0) = {cosine_similarity(bind_a1b0, bind_a0b0):.3f}")
    print(f"   sim(A1⊙B0, A1⊙B1) = {cosine_similarity(bind_a1b0, bind_a1b1):.3f}")

    # The problem: prototypes
    print("\n5. Creating class prototypes...")

    # Positive prototype: average of A0B1 and A1B0
    positive_sum = bind_a0b1.astype(float) + bind_a1b0.astype(float)
    positive_proto = np.where(positive_sum > 0, 1, np.where(positive_sum < 0, -1, 0)).astype(np.int8)

    # Negative prototype: average of A0B0 and A1B1
    negative_sum = bind_a0b0.astype(float) + bind_a1b1.astype(float)
    negative_proto = np.where(negative_sum > 0, 1, np.where(negative_sum < 0, -1, 0)).astype(np.int8)

    print(f"\n   Positive prototype: bundling A0⊙B1 and A1⊙B0")
    print(f"   Negative prototype: bundling A0⊙B0 and A1⊙B1")

    print("\n6. Prototype discriminability:")
    print(f"   sim(positive_proto, negative_proto) = {cosine_similarity(positive_proto, negative_proto):.3f}")

    print("\n7. Classification accuracy of prototypes:")
    test_cases = [
        ("A0⊙B1 (positive)", bind_a0b1, "positive"),
        ("A1⊙B0 (positive)", bind_a1b0, "positive"),
        ("A0⊙B0 (negative)", bind_a0b0, "negative"),
        ("A1⊙B1 (negative)", bind_a1b1, "negative"),
    ]

    correct = 0
    for name, vec, true_class in test_cases:
        sim_pos = cosine_similarity(vec, positive_proto)
        sim_neg = cosine_similarity(vec, negative_proto)
        pred_class = "positive" if sim_pos > sim_neg else "negative"
        is_correct = pred_class == true_class
        correct += is_correct
        mark = "✓" if is_correct else "✗"
        print(f"   {mark} {name}: sim(pos)={sim_pos:.3f}, sim(neg)={sim_neg:.3f} → {pred_class}")

    print(f"\n   Accuracy: {correct}/4 = {correct/4:.0%}")

    # Analysis
    print("\n" + "=" * 60)
    print("ANALYSIS")
    print("=" * 60)

    within_class_sim = (
        cosine_similarity(bind_a0b1, bind_a1b0) +
        cosine_similarity(bind_a0b0, bind_a1b1)
    ) / 2

    across_class_sim = (
        cosine_similarity(bind_a0b1, bind_a0b0) +
        cosine_similarity(bind_a0b1, bind_a1b1) +
        cosine_similarity(bind_a1b0, bind_a0b0) +
        cosine_similarity(bind_a1b0, bind_a1b1)
    ) / 4

    print(f"""
Within-class similarity (should be HIGH): {within_class_sim:.3f}
Across-class similarity (should be LOW):  {across_class_sim:.3f}

The issue: Within-class similarity is {"HIGH ✓" if within_class_sim > 0.5 else "LOW ✗"}
           Across-class similarity is {"LOW ✓" if across_class_sim < 0.3 else "HIGH ✗"}

For XOR to work with VSA prototypes, we need:
- A0⊙B1 similar to A1⊙B0 (both are positive class)
- A0⊙B0 similar to A1⊙B1 (both are negative class)
- Positive patterns dissimilar from negative patterns

If within-class similarity is low, the prototypes become "washed out"
and lose discriminability.
    """)

    # Alternative: use XOR-specific encoding
    print("\n" + "=" * 60)
    print("ALTERNATIVE: Explicit XOR encoding")
    print("=" * 60)

    print("""
Instead of bind(A, B), we could encode the XOR relationship explicitly:
- Create a "match" indicator: A[0] == B[0]
- This directly encodes whether values match or not
    """)

    # Create match/mismatch vectors
    match_vec = encoder.encode_data({"relation": "match"})
    mismatch_vec = encoder.encode_data({"relation": "mismatch"})

    print(f"   sim(match, mismatch) = {cosine_similarity(match_vec, mismatch_vec):.3f}")

    # This IS what we want - clear separation

    # Now test: what happens when we bundle interactions WITH individual fields?
    print("\n" + "=" * 60)
    print("TEST: Bundling interactions with individual fields")
    print("=" * 60)

    print("\nThe issue might be dilution - individual fields overwhelm the interaction.")

    # Full encoding with interaction
    def encode_full(a_val, b_val, interaction_weight=1.0):
        """Encode with both individual fields and interaction."""
        a_vec = encoder.encode_data({"feature_a": a_val})
        b_vec = encoder.encode_data({"feature_b": b_val})
        interaction = (a_vec * b_vec).astype(np.float32)

        # Bundle: individual fields + weighted interaction
        total = a_vec.astype(np.float32) + b_vec.astype(np.float32) + interaction_weight * interaction
        return np.where(total > 0, 1, np.where(total < 0, -1, 0)).astype(np.int8)

    # Test with different interaction weights
    for interaction_weight in [0.0, 1.0, 2.0, 5.0, 10.0]:
        print(f"\nInteraction weight = {interaction_weight}:")

        # Create prototypes
        pos_examples = [
            encode_full("A0", "B1", interaction_weight),
            encode_full("A1", "B0", interaction_weight),
        ]
        neg_examples = [
            encode_full("A0", "B0", interaction_weight),
            encode_full("A1", "B1", interaction_weight),
        ]

        pos_proto = np.where(
            np.sum(pos_examples, axis=0) > 0, 1,
            np.where(np.sum(pos_examples, axis=0) < 0, -1, 0)
        ).astype(np.int8)
        neg_proto = np.where(
            np.sum(neg_examples, axis=0) > 0, 1,
            np.where(np.sum(neg_examples, axis=0) < 0, -1, 0)
        ).astype(np.int8)

        # Test each case
        correct = 0
        for a, b, true_class in [("A0", "B1", "pos"), ("A1", "B0", "pos"),
                                  ("A0", "B0", "neg"), ("A1", "B1", "neg")]:
            vec = encode_full(a, b, interaction_weight)
            sim_pos = cosine_similarity(vec, pos_proto)
            sim_neg = cosine_similarity(vec, neg_proto)
            pred = "pos" if sim_pos > sim_neg else "neg"
            if pred == true_class:
                correct += 1

        print(f"   Accuracy: {correct}/4 = {correct*25}%")

    # The fix
    print("\n" + "=" * 60)
    print("SOLUTION: Interaction-only encoding for XOR")
    print("=" * 60)

    print("""
For XOR, the solution is to ONLY use the interaction, not the individual fields.

When individual fields provide zero information (like XOR), including them
actually HURTS because they add noise that drowns out the interaction signal.

The InteractionEncoder should learn to set individual field weights to 0
when interactions dominate.
    """)


if __name__ == "__main__":
    main()
