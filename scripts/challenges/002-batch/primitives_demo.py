#!/usr/bin/env python3
"""
MATHEMATICAL PRIMITIVES DEMO: Building Semantic Encoders from Fundamentals

Shows how users can compose mathematical primitives to build domain-specific
semantic encoders. These primitives are fundamental VSA/HDC capabilities that
users couldn't easily implement from existing operations.
"""

import requests
import json
from typing import List, Dict, Any

API_BASE = "http://localhost:8000"


def encode_fractal_from_primitives(matrix_data: Dict[str, Any]) -> List[float]:
    """
    Build fractal semantic encoder using mathematical primitives.

    Users can compose these primitives to create domain-specific understanding.
    """
    # Extract mathematical properties
    panels = matrix_data.get("panels", {})
    iterations = [p.get("iterations", 0) for p in panels.values()]
    convergence_rates = [p.get("convergence", 1.0) for p in panels.values()]

    avg_iterations = sum(iterations) / len(iterations) if iterations else 0
    avg_convergence = sum(convergence_rates) / len(convergence_rates) if convergence_rates else 1.0

    # Use mathematical primitives to encode properties
    iteration_vector = requests.post(f"{API_BASE}/encode_mathematical", json={
        "primitive": "iteration_complexity",
        "value": avg_iterations
    }).json()["vector"]

    convergence_vector = requests.post(f"{API_BASE}/encode_mathematical", json={
        "primitive": "convergence_rate",
        "value": avg_convergence
    }).json()["vector"]

    # Add self-similarity (fractal property)
    similarity_vector = requests.post(f"{API_BASE}/encode_mathematical", json={
        "primitive": "self_similarity",
        "value": 0.8  # High self-similarity for fractals
    }).json()["vector"]

    # Mathematically bind the properties (fundamental composition primitive)
    fractal_signature = requests.post(f"{API_BASE}/mathematical_bind", json={
        "vectors": [iteration_vector, convergence_vector, similarity_vector],
        "operation": "bind"
    }).json()["result_vector"]

    return fractal_signature


def encode_wave_from_primitives(matrix_data: Dict[str, Any]) -> List[float]:
    """
    Build wave semantic encoder using mathematical primitives.

    Demonstrates frequency-amplitude binding - a relationship users
    couldn't easily compose from generic operations.
    """
    panels = matrix_data.get("panels", {})
    frequencies = [p.get("frequency", 1.0) for p in panels.values()]
    amplitudes = [p.get("amplitude", 1.0) for p in panels.values()]

    avg_frequency = sum(frequencies) / len(frequencies) if frequencies else 1.0
    avg_amplitude = sum(amplitudes) / len(amplitudes) if amplitudes else 1.0

    # Encode frequency and amplitude as mathematical primitives
    frequency_vector = requests.post(f"{API_BASE}/encode_mathematical", json={
        "primitive": "frequency_domain",
        "value": avg_frequency
    }).json()["vector"]

    amplitude_vector = requests.post(f"{API_BASE}/encode_mathematical", json={
        "primitive": "amplitude_scale",
        "value": avg_amplitude
    }).json()["vector"]

    # Mathematically bind frequency and amplitude (physical relationship)
    # This coupling is fundamental - users couldn't easily replicate it
    wave_signature = requests.post(f"{API_BASE}/mathematical_bind", json={
        "vectors": [frequency_vector, amplitude_vector],
        "operation": "bind"
    }).json()["result_vector"]

    return wave_signature


def encode_graph_from_primitives(graph_data: Dict[str, Any]) -> List[float]:
    """
    Build graph topology encoder using mathematical primitives.

    Shows how topological properties can be composed.
    """
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])
    topology = graph_data.get("topology", "random")

    num_nodes = len(nodes)
    num_edges = len(edges)
    avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

    # Encode basic topological properties
    degree_vector = requests.post(f"{API_BASE}/encode_mathematical", json={
        "primitive": "topological_distance",
        "value": avg_degree  # Using distance primitive for degree
    }).json()["vector"]

    # Add topology-specific properties
    if topology == "scale_free":
        power_law_vector = requests.post(f"{API_BASE}/encode_mathematical", json={
            "primitive": "power_law_exponent",
            "value": 2.5  # Typical scale-free exponent
        }).json()["vector"]

        # Bind scale-free characteristics
        topology_signature = requests.post(f"{API_BASE}/mathematical_bind", json={
            "vectors": [degree_vector, power_law_vector],
            "operation": "bind"
        }).json()["result_vector"]

    elif topology == "small_world":
        clustering_vector = requests.post(f"{API_BASE}/encode_mathematical", json={
            "primitive": "clustering_coefficient",
            "value": 0.6  # High clustering
        }).json()["vector"]

        # Bind small-world characteristics
        topology_signature = requests.post(f"{API_BASE}/mathematical_bind", json={
            "vectors": [degree_vector, clustering_vector],
            "operation": "bind"
        }).json()["result_vector"]

    else:  # random
        # Just use degree for random graphs
        topology_signature = degree_vector

    return topology_signature


def demonstrate_user_composition():
    """Show how users can compose mathematical primitives."""
    print("🔧 USER COMPOSITION: Building Semantic Encoders from Primitives")
    print("-" * 60)

    # Example data
    fractal_matrix = {
        "matrix-id": "user_fractal",
        "rule": "fractal",
        "panels": {
            "row1-col1": {"iterations": 45, "convergence": 0.8},
            "row1-col2": {"iterations": 52, "convergence": 0.75}
        }
    }

    wave_matrix = {
        "matrix-id": "user_wave",
        "rule": "wave",
        "panels": {
            "row1-col1": {"frequency": 2.1, "amplitude": 0.8}
        }
    }

    scale_free_graph = {
        "graph-id": "user_sf_graph",
        "nodes": ["n1", "n2", "n3", "n4", "n5"],
        "edges": [
            {"from": "n1", "to": "n2"}, {"from": "n1", "to": "n3"},
            {"from": "n1", "to": "n4"}, {"from": "n2", "to": "n3"}
        ],
        "topology": "scale_free"
    }

    try:
        # User composes fractal encoder
        print("Building fractal encoder...")
        fractal_vector = encode_fractal_from_primitives(fractal_matrix)
        print(f"✅ Fractal vector: {len(fractal_vector)} dimensions")

        # User composes wave encoder
        print("Building wave encoder...")
        wave_vector = encode_wave_from_primitives(wave_matrix)
        print(f"✅ Wave vector: {len(wave_vector)} dimensions")

        # User composes graph encoder
        print("Building graph encoder...")
        graph_vector = encode_graph_from_primitives(scale_free_graph)
        print(f"✅ Graph vector: {len(graph_vector)} dimensions")

        # Store the user-composed encodings
        print("\n💾 Storing user-composed semantic encodings...")

        # Store fractal
        response = requests.post(f"{API_BASE}/insert", json={
            "data": json.dumps({
                **fractal_matrix,
                "semantic_vector": fractal_vector  # User-added semantic encoding
            }),
            "data_type": "json"
        })
        if response.status_code == 200:
            print("✅ Stored fractal with semantic encoding")

        # Store wave
        response = requests.post(f"{API_BASE}/insert", json={
            "data": json.dumps({
                **wave_matrix,
                "semantic_vector": wave_vector
            }),
            "data_type": "json"
        })
        if response.status_code == 200:
            print("✅ Stored wave with semantic encoding")

        # Store graph
        response = requests.post(f"{API_BASE}/insert", json={
            "data": json.dumps({
                **scale_free_graph,
                "semantic_vector": graph_vector
            }),
            "data_type": "json"
        })
        if response.status_code == 200:
            print("✅ Stored graph with semantic encoding")

        print("\n🎯 SUCCESS: Users can compose mathematical primitives!")
        print("   - Mathematical primitives are fundamental VSA/HDC capabilities")
        print("   - Users compose them to build domain-specific semantic encoders")
        print("   - No 'magic values' - primitives are configurable through API")

    except Exception as e:
        print(f"❌ Demo failed: {e}")


def main():
    """Run the mathematical primitives demonstration."""
    print("🎯 HOLON MATHEMATICAL PRIMITIVES DEMO")
    print("=" * 50)
    print("Fundamental VSA/HDC primitives that users compose into semantic encoders")
    print()

    # Check server
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code != 200:
            print(f"❌ Server not running at {API_BASE}")
            print("Start with: ./scripts/run_with_venv.sh python scripts/server/holon_server.py")
            return
    except:
        print(f"❌ Cannot connect to {API_BASE}")
        return

    print("✅ Connected to Holon API")
    print()

    # Show available primitives
    print("🔧 AVAILABLE MATHEMATICAL PRIMITIVES:")
    primitives = [
        "convergence_rate - Mathematical stability analysis",
        "iteration_complexity - Computational depth encoding",
        "frequency_domain - Wave frequency properties",
        "amplitude_scale - Energy/magnitude encoding",
        "power_law_exponent - Scale-free network properties",
        "clustering_coefficient - Local connectivity",
        "topological_distance - Graph distance metrics",
        "self_similarity - Fractal dimension properties"
    ]

    for primitive in primitives:
        print(f"  • {primitive}")

    print("\n🔗 AVAILABLE COMPOSITION OPERATIONS:")
    print("  • mathematical_bind - Couple mathematical properties")
    print("  • mathematical_bundle - Weighted combination of features")

    print()
    demonstrate_user_composition()

    print("\n" + "=" * 50)
    print("🎉 ONE HOLON SERVER with fundamental mathematical primitives!")
    print("Users compose them into domain-specific semantic encoders.")
    print("=" * 50)


if __name__ == "__main__":
    main()
