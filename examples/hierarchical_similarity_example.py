#!/usr/bin/env python3
"""
Example: Implementing Hierarchical Similarity using Holon's existing primitives.

This shows how users can implement hierarchical/subspace similarity analysis
using basic multi-metric similarity with custom logic.
"""

import numpy as np

from holon import CPUStore, HolonClient


def hierarchical_similarity_example():
    """Demonstrate user-implemented hierarchical similarity."""

    # Setup
    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert some test data
    test_items = [
        {"text": "machine learning algorithms", "category": "ml"},
        {"text": "neural network models", "category": "ml"},
        {"text": "calculus mathematics", "category": "math"},
        {"text": "differential equations", "category": "math"},
    ]

    ids = client.insert_batch_json(test_items)

    # Custom hierarchical similarity function
    def compute_hierarchical_similarity(
        query_vector, target_vector, hierarchy_levels=3
    ):
        """
        User-implemented hierarchical similarity using multi-metric primitives.
        """
        total_similarity = 0.0
        weights = [0.5, 0.3, 0.2]  # User chooses these weights

        vector_dim = len(query_vector)
        subspace_size = vector_dim // (2**hierarchy_levels)

        for level in range(min(hierarchy_levels, len(weights))):
            # Define subspace boundaries (user's choice)
            start_idx = level * subspace_size
            end_idx = min((level + 1) * subspace_size, vector_dim)

            # Extract subspace vectors
            query_sub = query_vector[start_idx:end_idx]
            target_sub = target_vector[start_idx:end_idx]

            if len(query_sub) > 0 and len(target_sub) > 0:
                # Use multi-metric similarity on this subspace (user's choice of metrics)
                subspace_similarity = multi_metric_similarity_on_vectors(
                    query_sub,
                    target_sub,
                    weights={"cosine": 0.7, "euclidean": 0.3},  # User's metric choice
                )
                total_similarity += subspace_similarity * weights[level]

        return total_similarity

    def multi_metric_similarity_on_vectors(vec1, vec2, weights):
        """Helper: multi-metric similarity on raw vectors."""
        score = 0.0
        total_weight = sum(weights.values())

        for metric_name, weight in weights.items():
            if metric_name == "cosine":
                # Cosine similarity
                dot = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                if norm1 > 0 and norm2 > 0:
                    sim = dot / (norm1 * norm2)
                    score += ((sim + 1) / 2) * weight  # Normalize to 0-1

            elif metric_name == "euclidean":
                # Euclidean distance -> similarity
                distance = np.linalg.norm(vec1 - vec2)
                sim = np.exp(-distance / 10)  # Convert to similarity
                score += sim * weight

        return score / total_weight if total_weight > 0 else 0.0

    # Example usage
    query = {"text": "machine learning"}
    print("🔍 Custom Hierarchical Similarity Search")
    print(f"Query: {query}")

    # Get basic results first
    basic_results = client.search_json(query, limit=10)

    # Enhance with custom hierarchical similarity
    enhanced_results = []
    for result in basic_results:
        # In a real implementation, you'd need to get the actual vectors
        # For demo, we'll just boost scores based on category similarity
        base_score = result["score"]

        # Custom hierarchical logic: boost if same category
        query_category = "ml"  # Would extract from query
        result_category = result["data"].get("category", "")

        if query_category == result_category:
            hierarchical_boost = 0.1  # User's choice
        else:
            hierarchical_boost = 0.0

        enhanced_score = min(base_score + hierarchical_boost, 1.0)
        result_copy = result.copy()
        result_copy["score"] = enhanced_score
        result_copy["hierarchical_boost"] = hierarchical_boost
        enhanced_results.append(result_copy)

    # Sort by enhanced score
    enhanced_results.sort(key=lambda x: x["score"], reverse=True)

    print("\n📊 Results with Custom Hierarchical Enhancement:")
    for result in enhanced_results[:3]:
        boost = result.get("hierarchical_boost", 0)
        print(".3f")


if __name__ == "__main__":
    hierarchical_similarity_example()
