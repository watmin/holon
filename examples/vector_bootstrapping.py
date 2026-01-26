#!/usr/bin/env python3
"""
Vector Bootstrapping Example: Custom Similarity Operations
Demonstrates how to encode custom vectors and perform similarity operations.
"""

import json

import numpy as np

from holon import CPUStore
from holon.similarity import find_similar_vectors, normalized_dot_similarity


def main():
    store = CPUStore()

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
        store.insert(json.dumps(item))

    print("🧬 Vector Bootstrapping Examples")
    print("=" * 50)

    # Example 1: Custom vector encoding
    print("\n1. Custom Vector Encoding")
    custom_vector = store.encoder.encode_data(
        {"type": "authentication", "method": "biometric", "security": "very_high"}
    )
    print(f"   Encoded custom vector: shape {custom_vector.shape}")

    # Example 2: Similarity between custom vectors
    print("\n2. Custom Vector Similarity")
    vector1 = store.encoder.encode_data({"type": "authentication", "security": "high"})
    vector2 = store.encoder.encode_data({"type": "authorization", "security": "high"})

    similarity = normalized_dot_similarity(vector1, vector2)
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
        vector = store.encoder.encode_data(term)
        search_vectors[term["concept"]] = vector
        print(f"   Encoded '{term['concept']}': {vector.shape}")

    # Example 4: Finding similar stored data to custom concepts
    print("\n4. Finding Similar Data to Custom Concepts")

    for concept, search_vector in search_vectors.items():
        print(f"\n   Similar to '{concept}':")

        # Find similar stored vectors
        similar_results = find_similar_vectors(
            search_vector, store.stored_vectors, top_k=2
        )

        for i, (data_id, score) in enumerate(similar_results):
            data = store.stored_data[data_id]
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
        vector = store.encoder.encode_data(item)
        encoded_vectors.append(vector)
        print(f"   → {item['operation']}: {vector.shape}")

    # Compare bulk encoded vectors
    if len(encoded_vectors) >= 2:
        sim = normalized_dot_similarity(encoded_vectors[0], encoded_vectors[1])
        print(f"   Bulk vector similarity: {sim:.4f}")

    print("\n✅ Vector bootstrapping examples completed!")


if __name__ == "__main__":
    main()
