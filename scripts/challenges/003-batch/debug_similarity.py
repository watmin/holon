#!/usr/bin/env python3
"""
Debug script to test VSA/HDC similarity calculations for n-gram encoding.
"""

import json
from holon import CPUStore, HolonClient
from holon.encoder import ListEncodeMode


def test_bundle_encoding():
    """Test if simple bundling works better than n-gram for text similarity."""
    print("📦 Testing Bundle Encoding (simpler approach)")
    print("=" * 48)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)
    client = HolonClient(local_store=store)

    # Test sequence
    words = ["everything", "depends", "upon", "relative", "minuteness"]

    # Create encodings with different modes
    bundle_data = {"words": {"_encode_mode": "bundle", "sequence": words}}
    ngram_data = {"words": {"_encode_mode": "ngram", "sequence": words}}

    # Insert bundle version
    bundle_id = client.insert_json(bundle_data)

    # Query with bundle
    results = client.search_json(
        bundle_data,
        top_k=1,
        threshold=0.0
    )

    if results:
        score = results[0]["score"]
        print(f"   Bundle self-similarity: {score:.4f}")

    # Insert ngram version
    ngram_id = client.insert_json(ngram_data)

    # Query bundle against ngram
    results = client.search_json(
        bundle_data,
        top_k=5,
        threshold=0.0
    )

    print(f"   Bundle vs N-gram similarity: {results[0]['score']:.4f} (same data, different encoding)")


def test_ngram_similarity():
    """Test if n-gram encoded vectors have high similarity for identical sequences."""
    print("🔍 Testing N-gram Similarity Calculation")
    print("=" * 50)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Test sequence
    words = ["everything", "depends", "upon", "relative", "minuteness"]

    # Create two identical encodings
    probe_data1 = {"words": {"_encode_mode": "ngram", "sequence": words}}
    probe_data2 = {"words": {"_encode_mode": "ngram", "sequence": words}}

    # Insert one as data
    data_id = client.insert_json(probe_data1)
    print(f"✅ Inserted test data with ID: {data_id}")

    # Query with the identical probe
    results = client.search_json(
        probe_data2,
        top_k=5,
        threshold=0.0
    )

    print(f"🔍 Query results: {len(results)} matches")
    if results:
        score = results[0]["score"]
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
    client = HolonClient(local_store=store)

    # Original sequence
    words1 = ["everything", "depends", "upon", "relative", "minuteness"]
    data = {"words": {"_encode_mode": "ngram", "sequence": words1}}
    data_id = client.insert_json(data)

    # Similar but different sequence
    words2 = ["depends", "on", "relative", "smallness"]
    probe = {"words": {"_encode_mode": "ngram", "sequence": words2}}

    results = client.search_json(
        probe,
        top_k=5,
        threshold=0.0
    )

    print(f"🔍 Query results: {len(results)} matches")
    if results:
        score = results[0]["score"]
        print(f"   Similarity score: {score:.4f}")
        print("   Expected: lower similarity due to different sequences")
    else:
        print("❌ No results for different sequence")


def test_raw_vector_similarity():
    """Test similarity at the raw vector level."""
    print("🔬 Testing Raw Vector Similarity")
    print("-" * 35)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    words = ["everything", "depends", "upon", "relative", "minuteness"]
    data = {"words": {"_encode_mode": "ngram", "sequence": words}}

    # Get raw vector
    vector = client.encode_vectors_json(data)

    # Query with same data
    results = client.search_json(
        data,
        top_k=1,
        threshold=0.0
    )

    if results:
        score = results[0]["score"]
        print(f"   Similarity score: {score:.4f}")
        print(f"   Vector magnitude: {cpu_vector.shape}")
        print(f"   Non-zero elements: {np.count_nonzero(cpu_vector)}")


def test_manual_similarity():
    """Manually compute similarity between identical vectors."""
    print("🧮 Manual Similarity Calculation")
    print("-" * 32)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    words = ["everything", "depends", "upon", "relative", "minuteness"]
    data = {"words": {"_encode_mode": "ngram", "sequence": words}}

    # Get two identical vectors
    vector1 = client.encode_vectors_json(data)
    vector2 = client.encode_vectors_json(data)

    # Manual dot product similarity
    dot_product = sum(a * b for a, b in zip(vector1, vector2))
    normalized_similarity = dot_product / len(vector1)

    print(f"   Manual dot product: {dot_product}")
    print(f"   Manual normalized similarity: {normalized_similarity:.4f}")
    print(f"   Vectors are identical: {vector1 == vector2}")

    # Check if vectors are actually the same
    if vector1 == vector2:
        print("   ✅ Vectors are identical - similarity should be 1.0")

        # But let's check why similarity isn't 1.0
        unique_vals = set(vector1)
        print(f"   Unique values in vector: {sorted(unique_vals)}")

        # Count each value
        for val in sorted(unique_vals):
            count = vector1.count(val)
            print(f"     {val}: {count} positions")

        # Expected dot product for identical bipolar vector
        expected_dot = sum(v * v for v in vector1)  # Should be dimension if bipolar
        print(f"   Self-dot product (v·v): {expected_dot}")
        print(f"   Expected similarity: {expected_dot / len(vector1):.4f}")

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
