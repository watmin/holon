#!/usr/bin/env python3
"""
Detailed investigation: Why does CHAINED mode give 0 results for substring matching?
"""

import numpy as np
from holon import CPUStore, HolonClient
from holon.encoder import ListEncodeMode


def test_chained_similarity_thresholds():
    """Test CHAINED mode with different similarity thresholds."""
    print("🔍 Testing CHAINED mode similarity thresholds")
    print("=" * 50)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Store full sequence
    full_seq = ["the", "quick", "brown", "fox", "jumps"]
    full_unit = {"text": {"_encode_mode": "chained", "sequence": full_seq}}
    full_id = client.insert_json(full_unit)
    print(f"Stored full sequence: {full_seq} → {full_id}")

    # Query various subsequences
    test_queries = [
        (["quick", "brown", "fox"], "exact middle subsequence"),
        (["the", "quick", "brown"], "prefix subsequence"),
        (["brown", "fox", "jumps"], "suffix subsequence"),
        (["fox", "jumps"], "short suffix"),
        (["the", "quick"], "short prefix"),
    ]

    for query_seq, description in test_queries:
        print(f"\n🔎 Query: {query_seq} ({description})")

        query_unit = {"text": {"_encode_mode": "chained", "sequence": query_seq}}

        # Test with different thresholds
        for threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
            results = client.search_json(
                probe=query_unit,
                top_k=5,
                threshold=threshold
            )
            if results:
                print(".3f")
                break
        else:
            print("  ❌ No results at any threshold")
def test_chained_vector_similarity():
    """Test actual vector similarity between chained encodings."""
    print("\n🔬 Testing vector similarities directly")
    print("=" * 50)

    store = CPUStore(dimensions=16000)
    encoder = store.encoder

    full_seq = ["the", "quick", "brown", "fox", "jumps"]
    sub_seq = ["quick", "brown", "fox"]

    full_vec = encoder.encode_list(full_seq, ListEncodeMode.CHAINED)
    sub_vec = encoder.encode_list(sub_seq, ListEncodeMode.CHAINED)

    # Test similarity
    dot_product = np.dot(full_vec, sub_vec)
    cosine_sim = dot_product / (np.linalg.norm(full_vec) * np.linalg.norm(sub_vec))

    print(f"Full sequence: {full_seq}")
    print(f"Subsequence: {sub_seq}")
    print(".3f")
    print(".3f")

    # Let's also check what happens with NGRAM
    print("\n📊 Comparing with NGRAM mode:")
    full_ngram = encoder.encode_list(full_seq, ListEncodeMode.NGRAM)
    sub_ngram = encoder.encode_list(sub_seq, ListEncodeMode.NGRAM)

    ngram_cosine = np.dot(full_ngram, sub_ngram) / (np.linalg.norm(full_ngram) * np.linalg.norm(sub_ngram))

    print(".3f")
    print(".3f")

    # And POSITIONAL
    print("\n📊 Comparing with POSITIONAL mode:")
    full_pos = encoder.encode_list(full_seq, ListEncodeMode.POSITIONAL)
    sub_pos = encoder.encode_list(sub_seq, ListEncodeMode.POSITIONAL)

    pos_cosine = np.dot(full_pos, sub_pos) / (np.linalg.norm(full_pos) * np.linalg.norm(sub_pos))

    print(".3f")
    print(".3f")


def test_theoretical_chained_substring():
    """Test if CHAINED can theoretically find substrings with different approach."""
    print("\n🧠 Theoretical CHAINED substring test")
    print("=" * 50)

    store = CPUStore(dimensions=16000)
    encoder = store.encoder

    # Idea: Maybe CHAINED works better if we query with overlapping sequences?
    full_seq = ["A", "B", "C", "D", "E"]
    queries = [
        ["B", "C", "D"],  # Exact middle
        ["C", "D", "E"],  # Suffix
        ["A", "B", "C"],  # Prefix
    ]

    print(f"Full sequence: {full_seq}")

    # Store full sequence
    full_vec = encoder.encode_list(full_seq, ListEncodeMode.CHAINED)

    for query in queries:
        query_vec = encoder.encode_list(query, ListEncodeMode.CHAINED)
        sim = np.dot(full_vec, query_vec) / (np.linalg.norm(full_vec) * np.linalg.norm(query_vec))
        print(".3f")

    # What if we manually create a "sliding window" search?
    print("\n🔄 Manual sliding window simulation:")
    window_size = 3
    for start in range(len(full_seq) - window_size + 1):
        window = full_seq[start:start + window_size]
        window_vec = encoder.encode_list(window, ListEncodeMode.CHAINED)
        sim = np.dot(full_vec, window_vec) / (np.linalg.norm(full_vec) * np.linalg.norm(window_vec))
        print(".3f")


def test_chained_vs_ngram_realistic():
    """Test CHAINED vs NGRAM with realistic text sequences."""
    print("\n📖 Realistic text comparison")
    print("=" * 50)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # More realistic text
    full_text = "calculus made easy integration is reverse of differentiation"
    full_words = full_text.split()

    # Store full text
    full_unit = {"text": {"_encode_mode": "chained", "sequence": full_words}}
    full_id = client.insert_json(full_unit)

    # Query a clear substring
    query_text = "integration is reverse"
    query_words = query_text.split()

    print(f"Full text: '{full_text}'")
    print(f"Query: '{query_text}'")

    # Test CHAINED
    chained_query = {"text": {"_encode_mode": "chained", "sequence": query_words}}
    chained_results = client.search_json(probe=chained_query, top_k=3, threshold=0.0)

    # Test NGRAM
    ngram_query = {"text": {"_encode_mode": "ngram", "sequence": query_words}}
    ngram_results = client.search_json(probe=ngram_query, top_k=3, threshold=0.0)

    print(f"\nCHAINED results: {len(chained_results)}")
    for result in chained_results[:2]:
        print(".3f")

    print(f"\nNGRAM results: {len(ngram_results)}")
    for result in ngram_results[:2]:
        print(".3f")


if __name__ == "__main__":
    test_chained_similarity_thresholds()
    test_chained_vector_similarity()
    test_theoretical_chained_substring()
    test_chained_vs_ngram_realistic()

    print("\n🎯 CONCLUSION:")
    print("If CHAINED gives 0 results even at threshold 0.0,")
    print("there's a fundamental issue with how similarity is calculated,")
    print("not just the encoding itself.")
