#!/usr/bin/env python3
"""
Vector Bootstrapping Example: Custom Similarity Operations
Demonstrates how to encode custom vectors and perform similarity operations.
"""

from holon import CPUStore, HolonClient


def main():
    store = CPUStore()
    client = HolonClient(local_store=store)

    # Sample data
    data = [
        {"type": "authentication", "method": "password", "security": "basic"},
        {"type": "authentication", "method": "oauth", "security": "high"},
        {"type": "authorization", "method": "jwt", "security": "high"},
        {"type": "authorization", "method": "api_key", "security": "medium"},
        {"type": "logging", "method": "structured", "security": "low"},
        {"type": "logging", "method": "unstructured", "security": "low"},
    ]

    # Insert data
    for item in data:
        client.insert_json(item)

    print("🧬 Vector Bootstrapping Examples")
    print("=" * 50)

    # Example 1: Custom vector encoding
    print("\n1. Custom Vector Encoding")
    custom_vector = client.encode_vectors_json(
        {"type": "authentication", "method": "biometric", "security": "very_high"}
    )
    print(f"   Encoded custom vector: {len(custom_vector)}D")

    # Example 2: Similarity between custom vectors
    print("\n2. Custom Vector Similarity")
    vector1 = client.encode_vectors_json({"type": "authentication", "security": "high"})
    vector2 = client.encode_vectors_json({"type": "authorization", "security": "high"})

    # Compute similarity (vectors are normalized, so dot product gives cosine similarity)
    similarity = sum(a * b for a, b in zip(vector1, vector2))
    print(f"   Similarity: {similarity:.4f}")

    # Example 3: Bootstrapping for custom search terms
    print("\n3. Bootstrapping Custom Search Terms")
    search_terms = [
        {"concept": "login", "category": "authentication"},
        {"concept": "access_control", "category": "authorization"},
        {"concept": "audit_trail", "category": "logging"},
    ]

    # Encode search concepts
    search_vectors = {}
    for term in search_terms:
        vector = client.encode_vectors_json(term)
        search_vectors[term["concept"]] = vector
        print(f"   Encoded '{term['concept']}': {len(vector)}D")

    # Example 4: Finding similar stored data to custom concepts
    print("\n4. Finding Similar Data to Custom Concepts")

    for concept, search_vector in search_vectors.items():
        print(f"\n   Similar to '{concept}':")

        # Use client search with the concept data (this demonstrates the typical usage)
        # Note: In practice, you'd search for similar items using the stored data
        similar_results = client.search_json(
            {"type": concept.split("_")[0] if "_" in concept else concept}, top_k=2
        )

        for i, result in enumerate(similar_results):
            data = result["data"]
            score = result["score"]
            print(f"   {i+1}. {data}: {score:.4f}")

    # Example 5: Bulk encoding for efficiency
    print("\n5. Bulk Vector Operations")

    bulk_data = [
        {"operation": "encrypt", "level": "field"},
        {"operation": "hash", "level": "record"},
        {"operation": "mask", "level": "display"},
    ]

    print("   Bulk encoding multiple items:")
    encoded_vectors = []
    for item in bulk_data:
        vector = client.encode_vectors_json(item)
        encoded_vectors.append(vector)
        print(f"   → {item['operation']}: {len(vector)}D")

    # Compare bulk encoded vectors
    if len(encoded_vectors) >= 2:
        sim = sum(a * b for a, b in zip(encoded_vectors[0], encoded_vectors[1]))
        print(f"   Bulk vector similarity: {sim:.4f}")

    print("\n✅ Vector bootstrapping examples completed!")


if __name__ == "__main__":
    main()
