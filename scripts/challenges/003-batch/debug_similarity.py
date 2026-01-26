#!/usr/bin/env python3
"""
Debug script to test VSA/HDC similarity calculations for n-gram encoding.
"""

import json
from holon import CPUStore
from holon.encoder import ListEncodeMode


def test_bundle_encoding():
    """Test if simple bundling works better than n-gram for text similarity."""
    print("📦 Testing Bundle Encoding (simpler approach)")
    print("=" * 48)

    store = CPUStore(dimensions=16000)

    # Test sequence
    words = ["everything", "depends", "upon", "relative", "minuteness"]

    # Create encodings with different modes
    bundle_data = {"words": {"_encode_mode": "bundle", "sequence": words}}
    ngram_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

    # Insert bundle version
    bundle_id = store.insert(json.dumps(bundle_data), "json")

    # Query with bundle
    results = store.query(
        probe=json.dumps(bundle_data),
        data_type="json",
        top_k=1,
        threshold=0.0
    )

    if results:
        score = results[0][1]
        print(f"   Bundle self-similarity: {score:.4f}")

    # Insert ngram version
    ngram_id = store.insert(json.dumps(ngram_data), "json")

    # Query bundle against ngram
    results = store.query(
        probe=json.dumps(bundle_data),
        data_type="json",
        top_k=5,
        threshold=0.0
    )

    print(f"   Bundle vs N-gram similarity: {results[0][1]:.4f} (same data, different encoding)")


def test_ngram_similarity():
    """Test if n-gram encoded vectors have high similarity for identical sequences."""
    print("🔍 Testing N-gram Similarity Calculation")
    print("=" * 50)

    store = CPUStore(dimensions=16000)

    # Test sequence
    words = ["everything", "depends", "upon", "relative", "minuteness"]

    # Create two identical encodings
    probe_data1 = {"words": {"_encode_mode": "ngram", "sequence": words}}
    probe_data2 = {"words": {"_encode_mode": "ngram", "sequence": words}}

    # Insert one as data
    data_id = store.insert(json.dumps(probe_data1), "json")
    print(f"✅ Inserted test data with ID: {data_id}")

    # Query with the identical probe
    results = store.query(
        probe=json.dumps(probe_data2),
        data_type="json",
        top_k=5,
        threshold=0.0
    )

    print(f"🔍 Query results: {len(results)} matches")
    if results:
        score = results[0][1]
        print(f"   Similarity score: {score:.4f}")
        # Expected: should be very close to 1.0 for identical sequences
        if score > 0.9:
            print("✅ Similarity test PASSED - identical sequences have high similarity")
        elif score > 0.5:
            print("⚠️  Similarity test PARTIAL - moderate similarity")
        else:
            print("❌ Similarity test FAILED - low similarity for identical sequences")
    else:
        print("❌ Similarity test FAILED - no results returned")

    print()


def test_different_sequences():
    """Test similarity between different sequences."""
    print("🔄 Testing Different Sequence Similarity")
    print("-" * 40)

    store = CPUStore(dimensions=16000)

    # Original sequence
    words1 = ["everything", "depends", "upon", "relative", "minuteness"]
    data = {"words": {"_encode_mode": "ngram", "sequence": words1}}
    data_id = store.insert(json.dumps(data), "json")

    # Similar but different sequence
    words2 = ["depends", "on", "relative", "smallness"]
    probe = {"words": {"_encode_mode": "ngram", "sequence": words2}}

    results = store.query(
        probe=json.dumps(probe),
        data_type="json",
        top_k=5,
        threshold=0.0
    )

    print(f"🔍 Query results: {len(results)} matches")
    if results:
        score = results[0][1]
        print(f"   Similarity score: {score:.4f}")
        print("   Expected: lower similarity due to different sequences")
    else:
        print("❌ No results for different sequence")


def test_raw_vector_similarity():
    """Test similarity at the raw vector level."""
    print("🔬 Testing Raw Vector Similarity")
    print("-" * 35)

    store = CPUStore(dimensions=16000)

    words = ["everything", "depends", "upon", "relative", "minuteness"]
    data = {"words": {"_encode_mode": "ngram", "sequence": words}}

    # Get raw vector
    vector = store.encoder.encode_data(data)
    cpu_vector = store.vector_manager.to_cpu(vector)

    # Query with same data
    results = store.query(
        probe=json.dumps(data),
        data_type="json",
        top_k=1,
        threshold=0.0
    )

    if results:
        score = results[0][1]
        print(f"   Similarity score: {score:.4f}")
        print(f"   Vector magnitude: {cpu_vector.shape}")
        print(f"   Non-zero elements: {np.count_nonzero(cpu_vector)}")


def test_manual_similarity():
    """Manually compute similarity between identical vectors."""
    print("🧮 Manual Similarity Calculation")
    print("-" * 32)

    store = CPUStore(dimensions=16000)

    words = ["everything", "depends", "upon", "relative", "minuteness"]
    data = {"words": {"_encode_mode": "ngram", "sequence": words}}

    # Get two identical vectors
    vector1 = store.encoder.encode_data(data)
    vector2 = store.encoder.encode_data(data)

    cpu_vec1 = store.vector_manager.to_cpu(vector1)
    cpu_vec2 = store.vector_manager.to_cpu(vector2)

    # Manual dot product similarity
    dot_product = np.dot(cpu_vec1.astype(float), cpu_vec2.astype(float))
    normalized_similarity = dot_product / len(cpu_vec1)

    print(f"   Manual dot product: {dot_product}")
    print(f"   Manual normalized similarity: {normalized_similarity:.4f}")
    print(f"   Vectors are identical: {np.array_equal(cpu_vec1, cpu_vec2)}")

    # Check if vectors are actually the same
    if np.array_equal(cpu_vec1, cpu_vec2):
        print("   ✅ Vectors are identical - similarity should be 1.0")

        # But let's check why similarity isn't 1.0
        print(f"   Vector dtype: {cpu_vec1.dtype}")
        unique_vals = np.unique(cpu_vec1)
        print(f"   Unique values in vector: {unique_vals}")

        # Count each value
        for val in unique_vals:
            count = np.sum(cpu_vec1 == val)
            print(f"     {val}: {count} positions")

        # Expected dot product for identical bipolar vector
        expected_dot = np.sum(cpu_vec1 * cpu_vec1)  # Should be dimension if bipolar
        print(f"   Self-dot product (v·v): {expected_dot}")
        print(f"   Expected similarity: {expected_dot / len(cpu_vec1):.4f}")

    else:
        diff_count = np.sum(cpu_vec1 != cpu_vec2)
        print(f"   ⚠️  Vectors differ in {diff_count} positions")


if __name__ == "__main__":
    import numpy as np

    test_bundle_encoding()
    test_ngram_similarity()
    test_different_sequences()
    test_raw_vector_similarity()
    test_manual_similarity()
