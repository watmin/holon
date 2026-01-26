#!/usr/bin/env python3
"""
Statistical Validation for Challenge 2 Graph Matching
Tests VSA/HDC geometric graph similarity against known graph structures
"""

import json
import time
from holon import CPUStore

# Copy the graph generation from the solution
def generate_test_graphs():
    """Generate test graphs for validation"""
    graphs = []

    def add_graph(name, nodes, edges, graph_type="undirected", attributes=None, description=""):
        graph = {
            "graph-id": f"test-{len(graphs)+1:02d}",
            "name": name,
            "description": description,
            "nodes": set(nodes),
            "edges": [],
            "type": graph_type,
            "attributes": attributes or {},
        }

        for edge in edges:
            if isinstance(edge, tuple):
                if len(edge) == 3:
                    graph["edges"].append({
                        "from": edge[0],
                        "to": edge[1],
                        "label": edge[2]
                    })
                else:
                    graph["edges"].append({
                        "from": edge[0],
                        "to": edge[1],
                        "label": "connects"
                    })
            else:
                graph["edges"].append(edge)

        graphs.append(graph)
        return graph

    # Create test graphs with known structures
    # Star graphs (should be very similar to each other)
    add_graph("star_4", ["A", "B", "C", "D"], [
        ("A", "B", "connects"), ("A", "C", "connects"), ("A", "D", "connects")
    ], description="4-node star")

    add_graph("star_5", ["A", "B", "C", "D", "E"], [
        ("A", "B", "connects"), ("A", "C", "connects"), ("A", "D", "connects"), ("A", "E", "connects")
    ], description="5-node star")

    # Cycle graphs (should be similar to each other)
    add_graph("cycle_3", ["A", "B", "C"], [
        ("A", "B", "connects"), ("B", "C", "connects"), ("C", "A", "connects")
    ], description="3-node cycle")

    add_graph("cycle_4", ["A", "B", "C", "D"], [
        ("A", "B", "connects"), ("B", "C", "connects"), ("C", "D", "connects"), ("D", "A", "connects")
    ], description="4-node cycle")

    # Tree graphs (should be similar to each other)
    add_graph("tree_binary", ["A", "B", "C", "D", "E"], [
        ("A", "B", "connects"), ("A", "C", "connects"), ("B", "D", "connects"), ("B", "E", "connects")
    ], description="binary tree")

    add_graph("tree_chain", ["A", "B", "C", "D", "E"], [
        ("A", "B", "connects"), ("B", "C", "connects"), ("C", "D", "connects"), ("D", "E", "connects")
    ], description="chain tree")

    # Different structures (should NOT be similar)
    add_graph("complete_4", ["A", "B", "C", "D"], [
        ("A", "B", "connects"), ("A", "C", "connects"), ("A", "D", "connects"),
        ("B", "C", "connects"), ("B", "D", "connects"), ("C", "D", "connects")
    ], description="complete graph")

    add_graph("random_4", ["A", "B", "C", "D"], [
        ("A", "B", "connects"), ("A", "D", "connects"), ("B", "C", "connects")
    ], description="random graph")

    return graphs

def run_graph_validation():
    """Run comprehensive graph matching validation"""
    print("🕸️ Graph Matching VSA/HDC Validation")
    print("=" * 45)

    # Generate test graphs
    graphs = generate_test_graphs()
    print(f"Generated {len(graphs)} test graphs")

    # Initialize store and ingest
    store = CPUStore(dimensions=16000)
    print("Ingesting graphs into Holon...")

    for graph in graphs:
        # Convert to JSON and ingest
        graph_json = json.dumps(graph, default=str)
        store.insert(graph_json, data_type="json")

    print(f"✅ Ingested {len(graphs)} graphs")

    # Test cases
    test_cases = [
        {
            "name": "Star Similarity",
            "query": {"name": "star_4"},
            "expected_similar": ["star_5"],  # Should be very similar
            "expected_dissimilar": ["cycle_3", "complete_4"]  # Should be different
        },
        {
            "name": "Cycle Similarity",
            "query": {"name": "cycle_3"},
            "expected_similar": ["cycle_4"],  # Should be similar
            "expected_dissimilar": ["star_4", "tree_binary"]  # Should be different
        },
        {
            "name": "Tree Similarity",
            "query": {"name": "tree_binary"},
            "expected_similar": ["tree_chain"],  # Should be similar
            "expected_dissimilar": ["cycle_3", "complete_4"]  # Should be different
        }
    ]

    results = {
        "total_tests": len(test_cases),
        "similarity_correct": 0,
        "dissimilarity_correct": 0,
        "response_times": []
    }

    for test in test_cases:
        print(f"\n🧪 Testing: {test['name']}")
        print(f"Query: {test['query']}")

        # Run similarity query
        start_time = time.time()
        query_results = store.query(
            json.dumps(test["query"]),
            data_type="json",
            top_k=10,
            threshold=0.0
        )
        response_time = time.time() - start_time
        results["response_times"].append(response_time)

        # Extract similarity scores
        similarities = {}
        for graph_id, score, _ in query_results:
            # Find the graph name
            for graph in graphs:
                if graph["graph-id"] == graph_id:
                    similarities[graph["name"]] = score
                    break

        print("Top similar graphs:")
        for i, (name, score) in enumerate(list(similarities.items())[:5]):
            print(f"  {i+1}. {name}: {score:.4f}")

        # Evaluate results
        # Check if expected similar graphs have higher similarity than expected dissimilar
        expected_similar_scores = [similarities.get(name, 0) for name in test["expected_similar"]]
        expected_dissimilar_scores = [similarities.get(name, 0) for name in test["expected_dissimilar"]]

        if expected_similar_scores and expected_dissimilar_scores:
            max_similar = max(expected_similar_scores)
            min_dissimilar = min(expected_dissimilar_scores)

            similarity_correct = max_similar > min_dissimilar
            if similarity_correct:
                results["similarity_correct"] += 1
                print("✅ Similarity ranking CORRECT")
            else:
                print("❌ Similarity ranking INCORRECT")

    # Calculate final metrics
    similarity_accuracy = results["similarity_correct"] / results["total_tests"]
    avg_response_time = sum(results["response_times"]) / len(results["response_times"])

    print("\n📊 VALIDATION RESULTS:")
    print(f"   Similarity Accuracy: {similarity_accuracy:.1%}")
    print(f"   Avg Response Time: {avg_response_time:.4f}s")

    if similarity_accuracy >= 0.7:
        assessment = "✅ GOOD - Effective geometric graph similarity"
    elif similarity_accuracy >= 0.5:
        assessment = "⚠️ FAIR - Some geometric matching capability"
    else:
        assessment = "❌ POOR - Geometric similarity not effective"

    print(f"\n🏆 Assessment: {assessment}")

    return similarity_accuracy

if __name__ == "__main__":
    accuracy = run_graph_validation()
    print(f"\nFinal Similarity Accuracy: {accuracy:.1%}")
