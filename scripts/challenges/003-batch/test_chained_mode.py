#!/usr/bin/env python3
"""
Test Holon's CHAINED mode for substring matching capabilities.
Does it actually work for fuzzy subsequence matching as advertised?
"""

import json
import re
from typing import List

from holon import CPUStore, HolonClient
from holon.encoder import ListEncodeMode


def normalize_text(text: str) -> List[str]:
    """Normalize text like in the quote finder."""
    normalized = re.sub(r"[^\w\s]", "", text.lower())
    words = [word for word in normalized.split() if word]
    return words


def test_chained_vs_ngram_substring_matching():
    """Test if CHAINED mode can find substrings within larger sequences."""
    print("🧪 Testing Holon's CHAINED vs NGRAM mode for substring matching")
    print("=" * 70)

    # Test data: quotes and larger containing texts
    test_cases = [
        {
            "substring": "everything depends upon relative minuteness",
            "containing_text": "philosophical quote that everything depends upon relative minuteness in mathematics",
            "description": "Exact substring within larger text"
        },
        {
            "substring": "depends on relative smallness",
            "containing_text": "the concept that everything depends upon relative minuteness",
            "description": "Fuzzy substring match"
        },
        {
            "substring": "integration is reverse",
            "containing_text": "fundamental calculus principle that integration is the reverse of differentiation",
            "description": "Partial phrase within sentence"
        }
    ]

    # Test both modes
    modes_to_test = [
        ("chained", ListEncodeMode.CHAINED),
        ("ngram", ListEncodeMode.NGRAM),
    ]

    for mode_name, mode_enum in modes_to_test:
        print(f"\n🔍 Testing {mode_name.upper()} mode:")
        print("-" * 40)

        store = CPUStore(dimensions=16000)
        client = HolonClient(local_store=store)

        # Insert containing texts
        print("📝 Ingesting containing texts...")
        for i, case in enumerate(test_cases):
            words = normalize_text(case["containing_text"])
            unit_data = {
                "text": {"_encode_mode": mode_name, "sequence": words},
                "metadata": {
                    "case_id": i,
                    "description": case["description"],
                    "full_text": case["containing_text"]
                }
            }
            unit_id = client.insert_json(unit_data)
            print(f"  • Case {i}: '{case['containing_text'][:50]}...' → {unit_id}")

        # Test substring queries
        print("\n🔎 Testing substring queries...")
        for i, case in enumerate(test_cases):
            words = normalize_text(case["substring"])
            probe_data = {"text": {"_encode_mode": mode_name, "sequence": words}}

            print(f"\n  Query {i}: '{case['substring']}'")
            print(f"  Expected: Should find case {i}")

            try:
                results = client.search_json(
                    probe=probe_data,
                    top_k=3,
                    threshold=0.0  # Show all results
                )

                if results:
                    print(f"  ✅ Found {len(results)} results:")
                    for j, result in enumerate(results):
                        score = result["score"]
                        metadata = result["data"]["metadata"]
                        found_case = metadata["case_id"]
                        match_indicator = "🎯" if found_case == i else "❌"
                        print(".3f")
                else:
                    print("  ❌ No results found")

                # Also test with higher threshold
                threshold_results = client.search_json(
                    probe=probe_data,
                    top_k=3,
                    threshold=0.5
                )
                print(f"  Threshold 0.5: {len(threshold_results)} results")

            except Exception as e:
                print(f"  💥 Error: {e}")


def test_chained_theoretical_capabilities():
    """Test theoretical CHAINED mode capabilities with controlled sequences."""
    print("\n🧮 Testing CHAINED mode theoretical capabilities")
    print("=" * 70)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Test 1: Exact subsequence matching
    print("\n📋 Test 1: Exact subsequence matching")

    # Store: [A, B, C, D, E]
    # Query: [B, C, D]
    full_sequence = ["the", "quick", "brown", "fox", "jumps"]
    subsequence = ["quick", "brown", "fox"]

    # Insert full sequence
    full_unit = {"text": {"_encode_mode": "chained", "sequence": full_sequence}}
    full_id = client.insert_json(full_unit)

    # Query subsequence
    sub_probe = {"text": {"_encode_mode": "chained", "sequence": subsequence}}
    results = client.search_json(probe=sub_probe, top_k=5, threshold=0.0)

    print(f"Full sequence: {full_sequence}")
    print(f"Subsequence query: {subsequence}")
    print(f"Results: {len(results)} found")
    for result in results:
        print(".3f")

    # Test 2: Fuzzy subsequence matching
    print("\n📋 Test 2: Fuzzy subsequence matching")

    # Store: [A, B, C, D, E, F]
    # Query: [B, X, D] (where X is similar to C)
    fuzzy_full = ["integration", "is", "reverse", "of", "differentiation"]
    fuzzy_query = ["is", "the", "reverse"]  # "the" vs "reverse" - should still match

    fuzzy_unit = {"text": {"_encode_mode": "chained", "sequence": fuzzy_full}}
    fuzzy_id = client.insert_json(fuzzy_unit)

    fuzzy_probe = {"text": {"_encode_mode": "chained", "sequence": fuzzy_query}}
    fuzzy_results = client.search_json(probe=fuzzy_probe, top_k=5, threshold=0.0)

    print(f"Fuzzy full: {fuzzy_full}")
    print(f"Fuzzy query: {fuzzy_query}")
    print(f"Fuzzy results: {len(fuzzy_results)} found")
    for result in fuzzy_results:
        print(".3f")


if __name__ == "__main__":
    test_chained_vs_ngram_substring_matching()
    test_chained_theoretical_capabilities()

    print("\n🎯 Analysis:")
    print("If CHAINED mode shows poor substring matching performance,")
    print("it reveals a gap in holon's geometric fuzzy matching capabilities.")
    print("The difflib fallback is working around this fundamental limitation.")
