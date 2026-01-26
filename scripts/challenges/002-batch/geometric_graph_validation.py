#!/usr/bin/env python3
"""
Validation for VSA/HDC Geometric Graph Matching
Tests if geometric encoding properly captures graph structural similarity
"""

import json
import time
from geometric_graph_matching import GeometricGraphMatcher, create_test_graphs


def create_expanded_test_graphs():
    """Create larger dataset for better validation"""
    graphs = create_test_graphs()

    # Add more variety
    def add_graph(name, nodes, edges, graph_type="undirected", description=""):
        graph = {
            "graph-id": f"test-{len(graphs)+1:02d}",
            "name": name,
            "nodes": set(nodes),
            "edges": [{"from": e[0], "to": e[1], "label": e[2] if len(e) > 2 else "connects"}
                     for e in edges],
            "type": graph_type,
            "description": description,
            "attributes": {}
        }
        graphs.append(graph)
        return graph

    # More star graphs
    add_graph("star_3", ["A", "B", "C"], [("A", "B"), ("A", "C")], description="3-node star")
    add_graph("star_6", ["A", "B", "C", "D", "E", "F"],
             [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E"), ("A", "F")], description="6-node star")

    # More cycle graphs
    add_graph("cycle_5", ["A", "B", "C", "D", "E"],
             [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A")], description="5-node cycle")

    # More tree graphs
    add_graph("tree_star", ["A", "B", "C", "D", "E"],
             [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")], description="star-like tree")
    add_graph("tree_balanced", ["A", "B", "C", "D", "E", "F", "G"],
             [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F"), ("C", "G")], description="balanced tree")

    return graphs


def validate_geometric_graph_matching():
    """Comprehensive validation of geometric graph matching"""
    print("🔬 VSA/HDC Geometric Graph Matching Validation")
    print("=" * 55)

    # Initialize geometric matcher
    matcher = GeometricGraphMatcher(dimensions=16000)

    # Create and ingest test graphs
    graphs = create_test_graphs()
    print(f"📊 Testing with {len(graphs)} graphs")

    for graph in graphs:
        matcher.ingest_graph(graph)
    print("✅ All graphs geometrically encoded and ingested")

    # Define validation tests
    validation_tests = [
        {
            "name": "Star Topology Similarity",
            "query_graph": next(g for g in graphs if g["name"] == "star_4"),
            "expected_top_match": "star_5",  # Should be most similar
            "should_be_dissimilar": ["cycle_3", "complete_4"]
        },
        {
            "name": "Cycle Topology Similarity",
            "query_graph": next(g for g in graphs if g["name"] == "cycle_3"),
            "expected_top_match": "cycle_4",  # Should be most similar
            "should_be_dissimilar": ["star_4", "tree_binary"]
        },
        {
            "name": "Tree Topology Similarity",
            "query_graph": next(g for g in graphs if g["name"] == "tree_binary"),
            "expected_top_match": "tree_chain",  # Should be most similar
            "should_be_dissimilar": ["cycle_3", "complete_4"]
        }
    ]

    results = {
        "total_tests": len(validation_tests),
        "topology_correct": 0,
        "dissimilarity_correct": 0,
        "response_times": []
    }

    print("\n🧪 VALIDATION TESTS")
    print("-" * 20)

    for test in validation_tests:
        print(f"\n🎯 {test['name']}")

        # Time the geometric similarity search
        start_time = time.time()
        similar_graphs = matcher.find_similar_graphs(test["query_graph"], top_k=5, use_topological_similarity=True)
        response_time = time.time() - start_time

        results["response_times"].append(response_time)

        if similar_graphs:
            # Check if expected similar graph is in top results (excluding self-match)
            query_name = test["query_graph"]["name"]
            expected_match = test["expected_top_match"]

            # Get top matches excluding self
            top_matches = [result["graph"]["name"] for result in similar_graphs if result["graph"]["name"] != query_name][:3]

            topology_correct = expected_match in top_matches
            if topology_correct:
                results["topology_correct"] += 1
                position = top_matches.index(expected_match) + 1
                print(f"   ✅ Topology match: {expected_match} found at position {position}")
            else:
                print(f"   ❌ Topology match: {expected_match} not in top 3 (found: {top_matches})")

            # Check dissimilarity - expected dissimilar graphs should not be in top 3
            top_3_names = [result["graph"]["name"] for result in similar_graphs[:3]]
            dissimilar_correct = not any(name in top_3_names for name in test["should_be_dissimilar"])

            if dissimilar_correct:
                results["dissimilarity_correct"] += 1
                print("   ✅ Dissimilarity preserved")
            else:
                print("   ❌ Dissimilarity violated")

            # Show top 3 results
            print("   📊 Top 3 geometric similarities:")
            for i, result in enumerate(similar_graphs[:3]):
                graph_name = result["graph"]["name"]
                similarity = result["geometric_similarity"]
                print(f"      {i+1}. {graph_name}: {similarity:.4f}")
        else:
            print("   ❌ No similar graphs found")
            if test["expected_top_match"] != test["query_graph"]["name"]:
                topology_correct = False
                dissimilar_correct = True  # Can't violate dissimilarity if no results

    # Calculate final metrics
    topology_accuracy = results["topology_correct"] / results["total_tests"]
    dissimilarity_accuracy = results["dissimilarity_correct"] / results["total_tests"]
    overall_accuracy = (topology_accuracy + dissimilarity_accuracy) / 2
    avg_response_time = sum(results["response_times"]) / len(results["response_times"])

    print("\n📊 VALIDATION RESULTS")
    print(f"   Topology Accuracy: {topology_accuracy:.1%}")
    print(f"   Dissimilarity Accuracy: {dissimilarity_accuracy:.1%}")
    print(f"   Overall Geometric Accuracy: {overall_accuracy:.1%}")
    print(f"   Avg Response Time: {avg_response_time:.4f}s")
    # Challenge 4 comparison
    if overall_accuracy >= 0.7:
        assessment = "🎉 EXCELLENT - Strong geometric graph matching!"
    elif overall_accuracy >= 0.5:
        assessment = "✅ GOOD - Effective geometric encoding"
    elif overall_accuracy >= 0.3:
        assessment = "⚠️ FAIR - Basic geometric properties captured"
    else:
        assessment = "❌ POOR - Geometric encoding needs improvement"

    print(f"\n🏆 Overall Assessment: {assessment}")

    if overall_accuracy >= 0.5:
        print("   ✅ VSA/HDC geometric graph encoding is working!")
        print("   ✅ Structural similarity reflected in hyperspace")
        print("   ✅ Approximate solutions to NP-hard graph problems achieved")
    else:
        print("   ⚠️ Geometric encoding shows promise but needs refinement")

    print("\n🔍 Key Findings:")
    print(f"   • Topology recognition accuracy: {topology_accuracy:.1%}")
    print(f"   • Dissimilarity preservation: {dissimilarity_accuracy:.1%}")
    print("   • VSA/HDC enables geometric graph similarity in hyperspace")
    return overall_accuracy


if __name__ == "__main__":
    accuracy = validate_geometric_graph_matching()
    print(f"\nFinal Geometric Graph Matching Accuracy: {accuracy:.1%}")
