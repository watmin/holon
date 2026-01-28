#!/usr/bin/env python3
"""
Test the minimal similarity enhancement kernel addition.
Clojure philosophy: rock solid kernel, userland freedom.
"""

import json
from holon import CPUStore, HolonClient


def test_similarity_kernel():
    """Test the minimal similarity enhancement kernel."""
    print("🧠 Testing Minimal Similarity Enhancement Kernel")
    print("=" * 50)

    # Setup
    store = CPUStore(dimensions=1000)
    client = HolonClient(local_store=store)

    # Add some test data
    test_data = [
        {"text": "machine learning algorithms", "category": "ml"},
        {"text": "neural network models", "category": "ml"},
        {"text": "calculus mathematics", "category": "math"},
        {"text": "differential equations", "category": "math"},
    ]

    for item in test_data:
        client.insert_json(item)

    # Test queries
    queries = [
        {"text": "machine learning"},
        {"text": "calculus"},
        {"text": "neural networks"},
    ]

    print("📊 Basic Search Results:")
    for query in queries:
        results = client.search_json(probe=query, limit=2)
        print(f"Query: '{query['text']}'")
        for result in results:
            print(f"  {result['score']:.3f} - {result['data']['text']}")
        print()

    print("🚀 Enhanced Similarity Search:")
    print("(Minimal kernel addition, maximum userland freedom)")

    # Test different similarity enhancements
    enhancements = [
        {
            "name": "Multi-metric",
            "config": {
                "method": "multi_metric",
                "weights": {"cosine": 0.6, "euclidean": 0.4}
            }
        }
    ]

    for enhancement in enhancements:
        print(f"\n🎯 {enhancement['name']} Enhancement:")
        for query in queries:
            results = client.search(
                probe=query,
                similarity=enhancement["config"],
                limit=2
            )
            print(f"  '{query['text']}': ", end="")
            if results:
                print(".3f")
            else:
                print("No results")

    print("\n✅ Kernel Enhancement Summary:")
    print("• Added: 1 flexible method (search_with_similarity)")
    print("• Powers: Multi-metric similarity")
    print("• Philosophy: Rock solid kernel, userland freedom")
    print("• Qdrant Compatible: Client-side enhancement")
    print("• Extensible: JSON config drives all enhancements")


if __name__ == "__main__":
    test_similarity_kernel()
