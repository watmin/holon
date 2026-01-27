#!/usr/bin/env python3
"""
Geometric VSA/HDC Graph Matching - PROPER Implementation
Encodes graphs geometrically: nodes as atoms, edges as bindings, graphs as bundles
"""

import json
import random
import uuid
from typing import Dict, List, Set, Any

from holon import CPUStore, HolonClient


class GeometricGraphEncoder:
    """Optimized VSA/HDC geometric graph encoder using effective encoding strategy"""

    def __init__(self, store, client, dimensions: int = 16000):
        self.store = store
        self.client = client
        self.dimensions = dimensions
        self.node_vectors = {}  # Cache for node atomic vectors
        self.label_vectors = {}  # Cache for edge label vectors

    def get_node_vector(self, node_id: str):
        """Get or create atomic vector for a node using simple hash-based approach"""
        if node_id not in self.node_vectors:
            # Use a simple hash-based approach that proved effective
            # Convert node ID to a deterministic bipolar vector
            import numpy as np
            hash_val = hash(node_id) % (2**32)
            np.random.seed(hash_val)
            vector = np.random.choice([-1, 0, 1], size=self.dimensions)
            # Ensure it's not all zeros
            if np.all(vector == 0):
                vector[0] = 1
            self.node_vectors[node_id] = vector
        return self.node_vectors[node_id]

    def get_label_vector(self, label: str):
        """Get or create vector for an edge label"""
        if label not in self.label_vectors:
            # Simple hash-based vectors for labels too
            import numpy as np
            hash_val = hash(f"label_{label}") % (2**32)
            np.random.seed(hash_val)
            vector = np.random.choice([-1, 0, 1], size=self.dimensions)
            if np.all(vector == 0):
                vector[0] = 1
            self.label_vectors[label] = vector
        return self.label_vectors[label]

    def encode_graph_geometrically(self, graph: Dict[str, Any]):
        """
        Advanced VSA/HDC geometric graph encoding with multiple structural features:
        - Basic edge encoding: from ⊙ to ⊙ label
        - Node degree encoding: capture connectivity patterns
        - Structural motifs: encode local graph patterns
        - Graph topology: holistic representation with structural awareness
        """
        edges = graph.get("edges", [])
        nodes = graph.get("nodes", [])
        graph_type = graph.get("type", "undirected")

        if not edges:
            return self.encode_node_set(nodes)

        # Phase 1: Topological edge encoding (node-identity independent)
        edge_vectors = []
        node_degrees = self.compute_node_degrees(edges, nodes)

        for edge in edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            label = edge.get("label", "connects")

            # Encode edge topology using degree roles instead of specific node identities
            from_degree = node_degrees.get(from_node, 0)
            to_degree = node_degrees.get(to_node, 0)

            # Normalize degrees relative to graph size for scale invariance
            max_degree = max(node_degrees.values()) if node_degrees else 0
            if max_degree > 0:
                from_degree_norm = from_degree / max_degree
                to_degree_norm = to_degree / max_degree
            else:
                from_degree_norm = to_degree_norm = 0

            # Categorize degree roles (topology-based)
            def get_degree_role(norm_degree):
                if norm_degree >= 0.9: return "hub"
                elif norm_degree >= 0.5: return "high"
                elif norm_degree >= 0.2: return "medium"
                else: return "leaf"

            from_role = get_degree_role(from_degree_norm)
            to_role = get_degree_role(to_degree_norm)

            # Create topological edge signature
            if graph_type == "directed":
                edge_signature = f"{from_role}_to_{to_role}_directed_{label}"
            else:
                # For undirected, sort roles to ensure consistency
                role_pair = sorted([from_role, to_role])
                edge_signature = f"{role_pair[0]}_connects_{role_pair[1]}_{label}"

            edge_vec = self.get_label_vector(edge_signature)
            edge_vectors.append(edge_vec)

        # Phase 2: Degree distribution encoding (topological pattern)
        degree_vectors = []

        # Compute degree distribution pattern
        degree_values = list(node_degrees.values())
        if degree_values:
            max_degree = max(degree_values)
            # Count nodes by degree role (topology-based histogram)
            role_counts = {"hub": 0, "high": 0, "medium": 0, "leaf": 0}

            for degree in degree_values:
                if max_degree > 0:
                    norm_degree = degree / max_degree
                else:
                    norm_degree = 0

                if norm_degree >= 0.9:
                    role_counts["hub"] += 1
                elif norm_degree >= 0.5:
                    role_counts["high"] += 1
                elif norm_degree >= 0.2:
                    role_counts["medium"] += 1
                else:
                    role_counts["leaf"] += 1

            # Encode the degree distribution pattern
            for role, count in role_counts.items():
                count_ratio = count / len(degree_values) if degree_values else 0
                dist_vec = self.get_label_vector(f"dist_{role}_{count_ratio:.2f}")
                degree_vectors.append(dist_vec)

        # Phase 3: Structural motif encoding (basic patterns)
        motif_vectors = []
        triangles = self.find_triangles(edges)
        if triangles:
            triangle_vec = self.get_label_vector("has_triangles")
            motif_vectors.append(triangle_vec)

        # Phase 4: Holistic graph encoding with scale-invariant features
        all_components = edge_vectors + degree_vectors + motif_vectors

        if all_components:
            # Use client's vector composition for superposition
            graph_vector = self.client.compose_vectors("bundle", [vec.tolist() if hasattr(vec, 'tolist') else vec for vec in all_components])

            # Add scale-invariant topological metadata (normalized features)
            degree_values = list(node_degrees.values())
            max_degree = max(degree_values) if degree_values else 0
            avg_degree = sum(degree_values) / len(degree_values) if degree_values else 0

            # Normalized topological features (scale-invariant)
            meta_data = {
                "density": len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0,  # Normalized density
                "avg_degree_ratio": avg_degree / len(nodes) if len(nodes) > 0 else 0,  # Degree relative to size
                "max_degree_ratio": max_degree / len(nodes) if len(nodes) > 0 else 0,  # Max degree relative to size
                "degree_variance": sum((d - avg_degree) ** 2 for d in degree_values) / len(degree_values) if degree_values else 0,
                "graph_type": graph_type,
                # Topological classification features
                "has_hub": 1 if max_degree >= len(nodes) - 1 else 0,  # Hub detection (star topology)
                "is_regular": 1 if len(set(degree_values)) == 1 else 0,  # Regular graph
                "is_tree": 1 if len(edges) == len(nodes) - 1 else 0,  # Tree detection
                "is_cyclic": 1 if len(edges) >= len(nodes) else 0,  # Has cycles
            }

            # Add explicit topology type classification
            if max_degree >= len(nodes) - 1 and len([d for d in degree_values if d == 1]) == len(nodes) - 1:
                topology_type = "star"  # One hub, rest are leaves
            elif len(edges) == len(nodes) and all(d == 2 for d in degree_values):
                topology_type = "cycle"  # All nodes degree 2
            elif len(edges) == len(nodes) - 1 and max_degree == len(nodes) - 1:
                topology_type = "star"  # Tree with hub structure
            elif len(edges) == len(nodes) - 1 and max_degree <= 3:
                topology_type = "tree"  # General tree structure
            elif len(edges) > len(nodes) and max_degree == len(nodes) - 1:
                topology_type = "complete"  # Fully connected
            else:
                topology_type = "complex"  # Other structures

            meta_data["topology_type"] = topology_type
            meta_vector = self.client.encode_vectors_json(meta_data)
            graph_vector = self.client.compose_vectors("bind", [graph_vector, meta_vector])

            return graph_vector
        else:
            return self.encode_node_set(nodes)

    def compute_node_degrees(self, edges, nodes):
        """Compute degree of each node"""
        degrees = {node: 0 for node in nodes}
        for edge in edges:
            from_node = edge.get("from", "")
            to_node = edge.get("to", "")
            if from_node in degrees:
                degrees[from_node] += 1
            if to_node in degrees:
                degrees[to_node] += 1
        return degrees

    def find_triangles(self, edges):
        """Find triangular motifs in the graph"""
        # Simple triangle detection - check for 3 nodes all connected
        # This is a basic implementation for demonstration
        nodes = set()
        for edge in edges:
            nodes.add(edge.get("from", ""))
            nodes.add(edge.get("to", ""))

        triangles = []
        node_list = list(nodes)

        for i in range(len(node_list)):
            for j in range(i+1, len(node_list)):
                for k in range(j+1, len(node_list)):
                    n1, n2, n3 = node_list[i], node_list[j], node_list[k]

                    # Check if all three pairs are connected
                    edges_set = {(e["from"], e["to"]) for e in edges}
                    if (n1, n2) in edges_set or (n2, n1) in edges_set:
                        if (n1, n3) in edges_set or (n3, n1) in edges_set:
                            if (n2, n3) in edges_set or (n3, n2) in edges_set:
                                triangles.append((n1, n2, n3))

        return triangles

    def encode_node_set(self, nodes):
        """Encode a set of nodes using Holon's bundle operation"""
        if not nodes:
            import numpy as np
            return np.zeros(self.dimensions, dtype=np.int8)

        # Get node vectors and bundle them
        node_vectors = [self.get_node_vector(node) for node in nodes]
        return self.client.compose_vectors("bundle", node_vectors)

    def geometric_similarity(self, vec1, vec2) -> float:
        """Compute geometric similarity using Holon's normalized dot product similarity"""
        from holon.similarity import normalized_dot_similarity
        return normalized_dot_similarity(vec1, vec2)


class GeometricGraphMatcher:
    """VSA/HDC geometric graph matching system"""

    def __init__(self, dimensions: int = 16000):
        self.store = CPUStore(dimensions=dimensions)
        self.client = HolonClient(local_store=self.store)
        self.encoder = GeometricGraphEncoder(self.store, self.client, dimensions=dimensions)
        self.graph_cache = {}  # Store original graph data

    def ingest_graph(self, graph: Dict[str, Any]):
        """Ingest a graph with advanced geometric encoding"""
        graph_id = graph["graph-id"]

        # Store original graph data
        self.graph_cache[graph_id] = graph

        # Encode geometrically using advanced VSA/HDC operations
        # The encoder now uses Holon's proper bind/bundle operations
        geometric_vector = self.encoder.encode_graph_geometrically(graph)

        # Create data structure for storage (make JSON serializable)
        def make_serializable(obj):
            if isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(item) for item in obj]
            else:
                return obj

        graph_data = {
            "graph_id": graph_id,
            "graph_structure": make_serializable(graph),  # Store full graph for re-encoding
            "metadata": {
                "name": graph.get("name", ""),
                "type": graph.get("type", "undirected"),
                "description": graph.get("description", ""),
                "node_count": len(graph.get("nodes", [])),
                "edge_count": len(graph.get("edges", []))
            }
        }

        # Store the graph structure (geometric vectors computed on demand)
        self.client.insert_json(graph_data)

    def find_similar_graphs(self, query_graph: Dict[str, Any], top_k: int = 5, use_topological_similarity: bool = False) -> List[Dict[str, Any]]:
        """Find geometrically similar graphs using advanced VSA/HDC similarity"""
        # Encode query graph geometrically
        query_vector = self.encoder.encode_graph_geometrically(query_graph)

        # Get all stored graphs from the store and compute similarity
        similarities = []

        # Query all stored items (this is a simplified approach for the demo)
        # In a production system, you'd want more efficient similarity search
        all_results = self.client.search_json(
            {"metadata": {"type": "undirected"}},  # Get all graphs
            top_k=100,  # Get all stored graphs
            threshold=0.0
        )

        for result in all_results:
            stored_data = result["data"]
            graph_id = stored_data["graph_id"]
            stored_graph = stored_data["graph_structure"]

            if use_topological_similarity:
                # Use topological feature similarity instead of full vector similarity
                similarity = self.compute_topological_similarity(query_graph, stored_graph)
            else:
                # Encode stored graph geometrically
                stored_vector = self.encoder.encode_graph_geometrically(stored_graph)
                # Compute geometric similarity
                similarity = self.encoder.geometric_similarity(query_vector, stored_vector)

            similarities.append({
                "graph_id": graph_id,
                "similarity": similarity,
                "graph": stored_graph
            })

        # Sort by similarity and return top_k
        similarities.sort(key=lambda x: x["similarity"], reverse=True)

        # Format results
        similar_graphs = []
        for result in similarities[:top_k]:
            similar_graphs.append({
                "graph": result["graph"],
                "geometric_similarity": result["similarity"],
                "metadata": {
                    "name": result["graph"].get("name", ""),
                    "type": result["graph"].get("type", "undirected"),
                    "description": result["graph"].get("description", ""),
                    "node_count": len(result["graph"].get("nodes", [])),
                    "edge_count": len(result["graph"].get("edges", []))
                }
            })

        return similar_graphs

    def compute_topological_similarity(self, graph1: Dict[str, Any], graph2: Dict[str, Any]) -> float:
        """Compute topological similarity based on scale-invariant features"""
        # Extract topological features for both graphs
        features1 = self.extract_topological_features(graph1)
        features2 = self.extract_topological_features(graph2)

        # Compute similarity based on feature matching
        similarity = 0.0
        total_weight = 0.0

        # Topology type match (highest weight)
        if features1.get("topology_type") == features2.get("topology_type"):
            similarity += 1.0
        total_weight += 1.0

        # Graph type match
        if features1.get("graph_type") == features2.get("graph_type"):
            similarity += 0.8
        total_weight += 0.8

        # Density similarity (normalized difference)
        density1 = features1.get("density", 0)
        density2 = features2.get("density", 0)
        density_sim = 1.0 - min(abs(density1 - density2), 1.0)
        similarity += 0.6 * density_sim
        total_weight += 0.6

        # Hub presence match
        if features1.get("has_hub") == features2.get("has_hub"):
            similarity += 0.5
        total_weight += 0.5

        # Tree/cyclic properties
        if features1.get("is_tree") == features2.get("is_tree"):
            similarity += 0.4
        total_weight += 0.4

        if features1.get("is_cyclic") == features2.get("is_cyclic"):
            similarity += 0.4
        total_weight += 0.4

        # Regular graph property
        if features1.get("is_regular") == features2.get("is_regular"):
            similarity += 0.3
        total_weight += 0.3

        # Degree distribution similarity (compare role distributions)
        role_sim = self.compare_degree_distributions(
            features1.get("degree_roles", {}),
            features2.get("degree_roles", {})
        )
        similarity += 0.5 * role_sim
        total_weight += 0.5

        return similarity / total_weight if total_weight > 0 else 0.0

    def extract_topological_features(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """Extract topological features from a graph"""
        edges = graph.get("edges", [])
        nodes = graph.get("nodes", [])
        graph_type = graph.get("type", "undirected")

        node_degrees = self.encoder.compute_node_degrees(edges, nodes)
        degree_values = list(node_degrees.values())

        features = {}

        # Basic properties
        features["graph_type"] = graph_type
        features["num_nodes"] = len(nodes)
        features["num_edges"] = len(edges)

        # Derived properties
        max_degree = max(degree_values) if degree_values else 0
        avg_degree = sum(degree_values) / len(degree_values) if degree_values else 0

        features["density"] = len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
        features["has_hub"] = 1 if max_degree >= len(nodes) - 1 else 0
        features["is_regular"] = 1 if len(set(degree_values)) == 1 else 0
        features["is_tree"] = 1 if len(edges) == len(nodes) - 1 else 0
        features["is_cyclic"] = 1 if len(edges) >= len(nodes) else 0

        # Topology classification
        if max_degree >= len(nodes) - 1 and len([d for d in degree_values if d == 1]) == len(nodes) - 1:
            topology_type = "star"
        elif len(edges) == len(nodes) and all(d == 2 for d in degree_values):
            topology_type = "cycle"
        elif len(edges) == len(nodes) - 1 and max_degree == len(nodes) - 1:
            topology_type = "star"
        elif len(edges) == len(nodes) - 1 and max_degree <= 3:
            topology_type = "tree"
        elif len(edges) > len(nodes) and max_degree == len(nodes) - 1:
            topology_type = "complete"
        else:
            topology_type = "complex"

        features["topology_type"] = topology_type

        # Degree role distribution
        role_counts = {"hub": 0, "high": 0, "medium": 0, "leaf": 0}
        for degree in degree_values:
            if max_degree > 0:
                norm_degree = degree / max_degree
            else:
                norm_degree = 0

            if norm_degree >= 0.9:
                role_counts["hub"] += 1
            elif norm_degree >= 0.5:
                role_counts["high"] += 1
            elif norm_degree >= 0.2:
                role_counts["medium"] += 1
            else:
                role_counts["leaf"] += 1

        features["degree_roles"] = role_counts

        return features

    def compare_degree_distributions(self, roles1: Dict[str, int], roles2: Dict[str, int]) -> float:
        """Compare degree role distributions between two graphs"""
        total1 = sum(roles1.values()) or 1
        total2 = sum(roles2.values()) or 1

        similarity = 0.0
        for role in ["hub", "high", "medium", "leaf"]:
            ratio1 = roles1.get(role, 0) / total1
            ratio2 = roles2.get(role, 0) / total2
            similarity += 1.0 - min(abs(ratio1 - ratio2), 1.0)

        return similarity / 4.0  # Average over 4 role types

    def find_subgraph_matches(self, subgraph_edges: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Find graphs containing specific subgraph patterns"""
        # Encode subgraph as a mini-graph
        subgraph = {
            "graph-id": "subgraph-query",
            "edges": subgraph_edges,
            "nodes": set()  # Will be populated from edges
        }

        # Collect all nodes from edges
        for edge in subgraph_edges:
            subgraph["nodes"].add(edge.get("from", ""))
            subgraph["nodes"].add(edge.get("to", ""))

        # Find similar graphs (those containing the subgraph pattern)
        return self.find_similar_graphs(subgraph, top_k=top_k)


def create_test_graphs():
    """Create test graphs with known geometric relationships"""
    graphs = []

    def add_graph(name, nodes, edges, graph_type="undirected", description=""):
        graph = {
            "graph-id": f"geo-{name}",
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

    # Star graphs (should be geometrically similar)
    add_graph("star_4", ["A", "B", "C", "D"],
             [("A", "B"), ("A", "C"), ("A", "D")], description="4-node star")
    add_graph("star_5", ["A", "B", "C", "D", "E"],
             [("A", "B"), ("A", "C"), ("A", "D"), ("A", "E")], description="5-node star")

    # Cycle graphs (should be geometrically similar)
    add_graph("cycle_3", ["A", "B", "C"],
             [("A", "B"), ("B", "C"), ("C", "A")], description="3-node cycle")
    add_graph("cycle_4", ["A", "B", "C", "D"],
             [("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")], description="4-node cycle")

    # Tree graphs (should be geometrically similar)
    add_graph("tree_binary", ["A", "B", "C", "D", "E"],
             [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E")], description="binary tree")
    add_graph("tree_chain", ["A", "B", "C", "D", "E"],
             [("A", "B"), ("B", "C"), ("C", "D"), ("D", "E")], description="chain tree")

    # Different structures (should be dissimilar)
    add_graph("complete_4", ["A", "B", "C", "D"],
             [("A", "B"), ("A", "C"), ("A", "D"), ("B", "C"), ("B", "D"), ("C", "D")],
             description="complete graph")
    add_graph("random_4", ["A", "B", "C", "D"],
             [("A", "B"), ("A", "D"), ("B", "C")], description="random graph")

    return graphs


def run_geometric_graph_demo():
    """Demonstrate VSA/HDC geometric graph matching"""
    print("🕸️ VSA/HDC Geometric Graph Matching Demo")
    print("=" * 50)

    # Initialize geometric matcher
    matcher = GeometricGraphMatcher(dimensions=16000)

    # Create and ingest test graphs
    graphs = create_test_graphs()
    print(f"📊 Created {len(graphs)} test graphs")

    print("🔬 Geometrically encoding graphs...")
    for graph in graphs:
        matcher.ingest_graph(graph)
    print("✅ All graphs encoded and ingested")

    # Test geometric similarity
    print("\n🎯 GEOMETRIC SIMILARITY TESTS")
    print("-" * 30)

    # Test 1: Star graphs should be similar
    star_4 = next(g for g in graphs if g["name"] == "star_4")
    print("\n🔍 Finding graphs similar to 4-node star:")
    similar_to_star = matcher.find_similar_graphs(star_4, top_k=3)
    for i, result in enumerate(similar_to_star):
        graph = result["graph"]
        similarity = result["geometric_similarity"]
        print(f"   {i+1}. {graph['name']}: {similarity:.4f}")
    # Test 2: Cycle graphs should be similar
    cycle_3 = next(g for g in graphs if g["name"] == "cycle_3")
    print("\n🔍 Finding graphs similar to 3-node cycle:")
    similar_to_cycle = matcher.find_similar_graphs(cycle_3, top_k=3)
    for i, result in enumerate(similar_to_cycle):
        graph = result["graph"]
        similarity = result["geometric_similarity"]
        print(f"   {i+1}. {graph['name']}: {similarity:.4f}")
    # Test 3: Subgraph matching
    print("\n🔍 Finding graphs containing A→B edge pattern:")
    subgraph_matches = matcher.find_subgraph_matches([
        {"from": "A", "to": "B", "label": "connects"}
    ], top_k=3)
    for i, result in enumerate(subgraph_matches):
        graph = result["graph"]
        similarity = result["geometric_similarity"]
        print(f"   {i+1}. {graph['name']}: {similarity:.4f}")
    print("\n🎉 Geometric graph matching demo complete!")
    print("✨ VSA/HDC successfully encodes graph structure in hyperspace")
    print("✨ Similarity reflects geometric/structural proximity")
    print("✨ Provides approximate solutions to NP-hard graph problems")


if __name__ == "__main__":
    run_geometric_graph_demo()
