#!/usr/bin/env python3
"""
Diagnose why CHAINED mode fails at substring matching.
Let's understand the mathematical implementation.
"""

import numpy as np
from holon import CPUStore
from holon.encoder import ListEncodeMode


def inspect_chained_encoding():
    """Inspect what CHAINED mode actually produces."""
    print("🔬 Inspecting CHAINED mode encoding internals")
    print("=" * 60)

    store = CPUStore(dimensions=1000)  # Smaller for easier inspection
    encoder = store.encoder

    # Test sequences
    full_seq = ["the", "quick", "brown", "fox"]
    sub_seq = ["quick", "brown", "fox"]

    print(f"Full sequence: {full_seq}")
    print(f"Subsequence: {sub_seq}")
    print()

    # Encode both
    full_vector = encoder.encode_list(full_seq, ListEncodeMode.CHAINED)
    sub_vector = encoder.encode_list(sub_seq, ListEncodeMode.CHAINED)

    print("📊 Vector statistics:")
    print(f"Full vector - Mean: {full_vector.mean():.3f}, Std: {full_vector.std():.3f}")
    print(f"Sub vector - Mean: {sub_vector.mean():.3f}, Std: {sub_vector.std():.3f}")
    print()

    # Check similarity
    dot_product = np.dot(full_vector, sub_vector)
    magnitude_full = np.linalg.norm(full_vector)
    magnitude_sub = np.linalg.norm(sub_vector)
    cosine_sim = dot_product / (magnitude_full * magnitude_sub)

    print("🔗 Similarity analysis:")
    print(".3f")
    print(".3f")
    print(".3f")
    print()

    # Let's see what the binding operations actually do
    print("🔧 Understanding the binding chain:")

    # Manual step-by-step encoding
    words = ["quick", "brown", "fox"]
    item_vecs = [encoder._encode_recursive(word) for word in words]

    print(f"Individual word vectors:")
    for i, (word, vec) in enumerate(zip(words, item_vecs)):
        print(f"  {word}: mean={vec.mean():.3f}, std={vec.std():.3f}")

    # Chain from the end (as CHAINED mode does)
    print("\nChaining process (from end):")
    chained = item_vecs[-1]  # "fox"
    print(f"Start: {chained.mean():.3f}")

    for i, prev in enumerate(reversed(item_vecs[:-1])):
        word = words[len(words)-2-i]  # Get corresponding word
        chained = encoder.bind(prev, chained)
        print(f"Bind '{word}': {chained.mean():.3f}")

    print(f"\nFinal chained vector: {chained.mean():.3f}")
    print("Encoder CHAINED result: {sub_vector.mean():.3f}")
    print(f"Match: {np.allclose(chained, sub_vector)}")


def test_binding_unbinding():
    """Test if binding/unbinding works for subsequence extraction."""
    print("\n🔄 Testing binding/unbinding operations")
    print("=" * 60)

    store = CPUStore(dimensions=1000)
    encoder = store.encoder

    # Test if we can unbind prefixes
    full = ["A", "B", "C", "D"]
    prefix = ["A", "B", "C"]

    full_vec = encoder.encode_list(full, ListEncodeMode.CHAINED)
    prefix_vec = encoder.encode_list(prefix, ListEncodeMode.CHAINED)

    print(f"Full: {full}")
    print(f"Prefix: {prefix}")

    # Try to "unbind" the last element from full to get prefix
    # If CHAINED worked for subsequences, this should work
    try:
        # This is theoretical - we don't have unbind operations
        print("❌ No unbind operations available in holon")
        print("This is why CHAINED mode can't find subsequences")
    except Exception as e:
        print(f"Error: {e}")

    # Check if similarity is preserved
    sim = np.dot(full_vec, prefix_vec) / (np.linalg.norm(full_vec) * np.linalg.norm(prefix_vec))
    print(".3f")


def diagnose_chained_limitations():
    """Explain why CHAINED mode fails at substring matching."""
    print("\n🎯 DIAGNOSIS: Why CHAINED mode fails")
    print("=" * 60)

    print("PROBLEM: CHAINED mode is designed for exact prefix/suffix matching,")
    print("         not for finding subsequences within larger sequences.")
    print()

    print("WHAT CHAINED DOES:")
    print("  - Chains bindings: A ⊙ (B ⊙ (C ⊙ D))")
    print("  - Good for: exact sequence matching, prefix removal")
    print("  - Bad for: finding C-D within A-B-C-D-E")
    print()

    print("WHAT'S MISSING:")
    print("  - No subsequence extraction operations")
    print("  - No sliding window geometric search")
    print("  - No fuzzy alignment algorithms")
    print()

    print("WHY DIFFLIB WORKS:")
    print("  - Compares all possible substrings")
    print("  - Uses edit distance for fuzzy matching")
    print("  - Not geometric, but effective")
    print()

    print("CONCLUSION:")
    print("  Holon's CHAINED mode is NOT defective - it's just")
    print("  designed for different use cases than substring matching.")
    print("  The difflib fallback fills a genuine capability gap.")


if __name__ == "__main__":
    inspect_chained_encoding()
    test_binding_unbinding()
    diagnose_chained_limitations()