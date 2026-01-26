"""
SEMANTIC ENCODER: Extended VSA/HDC with Mathematical Understanding

Extends the basic Encoder with domain-specific mathematical pattern recognition.
Provides fundamental VSA/HDC capabilities that go beyond generic structural encoding.
"""

import math
from typing import Any, Dict, Union

import numpy as np

from .domain_encoders import GraphTopologyEncoder, MathematicalPatternEncoder
from .encoder import Encoder
from .vector_manager import VectorManager


class SemanticEncoder(Encoder):
    """
    Extended encoder with mathematical semantic understanding.

    Combines generic structural encoding with domain-specific mathematical
    pattern recognition. Provides capabilities that users cannot easily
    implement from existing primitives.
    """

    def __init__(self, vector_manager: VectorManager):
        super().__init__(vector_manager)
        self.math_encoder = MathematicalPatternEncoder(vector_manager)
        self.graph_encoder = GraphTopologyEncoder(vector_manager)

    def encode_data(self, data: Any) -> np.ndarray:
        """
        Encode data with semantic mathematical understanding.

        First applies generic structural encoding, then enhances with
        domain-specific mathematical pattern recognition.
        """
        # Start with generic structural encoding
        structural_vector = self._encode_recursive(data)

        # Enhance with mathematical semantic understanding
        semantic_enhancement = self._extract_mathematical_semantics(data)

        if semantic_enhancement is not None:
            # Bind structural and semantic encodings
            # This creates a richer representation than either alone
            enhanced_vector = structural_vector * semantic_enhancement

            # Normalize to prevent magnitude explosion
            return self._threshold_bipolar(enhanced_vector)
        else:
            # No semantic enhancement available, use structural only
            return structural_vector

    def _extract_mathematical_semantics(self, data: Any) -> Union[np.ndarray, None]:
        """
        Extract mathematical semantic information from data.

        Returns semantic enhancement vector if mathematical patterns detected,
        None otherwise.
        """
        if not isinstance(data, dict):
            return None

        # Check for RPM matrix patterns
        if self._is_matrix_data(data):
            return self._encode_matrix_semantics(data)

        # Check for graph topology patterns
        elif self._is_graph_data(data):
            return self._encode_graph_semantics(data)

        # No mathematical patterns detected
        return None

    def _is_matrix_data(self, data: Dict) -> bool:
        """Check if data represents a mathematical matrix pattern."""
        return (
            "panels" in data
            and isinstance(data["panels"], dict)
            and "rule" in data
            and data["rule"] in ["fractal", "wave", "polynomial"]
        )

    def _is_graph_data(self, data: Dict) -> bool:
        """Check if data represents a graph topology."""
        return (
            "nodes" in data
            and "edges" in data
            and "topology" in data
            and data["topology"] in ["scale_free", "small_world", "random"]
        )

    def _encode_matrix_semantics(self, data: Dict) -> np.ndarray:
        """Encode mathematical semantics of matrix patterns."""
        rule = data["rule"]
        panels = data["panels"]

        # Extract mathematical properties from panels
        properties = self._extract_panel_properties(panels)

        if rule == "fractal":
            return self.math_encoder.encode_fractal_pattern(properties)
        elif rule == "wave":
            return self.math_encoder.encode_wave_phenomenon(properties)
        elif rule == "polynomial":
            return self.math_encoder.encode_polynomial_curve(properties)

        return None

    def _encode_graph_semantics(self, data: Dict) -> np.ndarray:
        """Encode topological semantics of graph patterns."""
        topology = data["topology"]
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])

        # Extract topological properties
        properties = self._extract_topological_properties(nodes, edges)

        if topology == "scale_free":
            return self.graph_encoder.encode_scale_free_topology(properties)
        elif topology == "small_world":
            return self.graph_encoder.encode_small_world_topology(properties)
        elif topology == "random":
            return self.graph_encoder.encode_random_topology(properties)

        return None

    def _extract_panel_properties(self, panels: Dict) -> Dict[str, Any]:
        """Extract mathematical properties from matrix panels."""
        properties = {}

        # Analyze all panel data
        iterations = []
        interference_values = []
        polynomial_degrees = []

        for pos, panel in panels.items():
            if "iterations" in panel:
                iterations.append(panel["iterations"])
            if "interference" in panel:
                interference_values.append(panel["interference"])

        # Compute aggregate properties
        if iterations:
            properties["avg_iterations"] = sum(iterations) / len(iterations)
            properties["convergence_rate"] = len(
                [i for i in iterations if i < 50]
            ) / len(iterations)
            properties["complexity_level"] = (
                "high" if max(iterations) > 100 else "moderate"
            )

        if interference_values:
            properties["interference_strength"] = sum(interference_values) / len(
                interference_values
            )
            properties["frequency"] = 1.0  # Placeholder - would analyze wave patterns
            properties["amplitude"] = max(interference_values)

        # Add boundary and similarity properties (would require more complex analysis)
        properties["boundary_complexity"] = 0.7  # Placeholder
        properties["self_similarity"] = 0.6  # Placeholder

        return properties

    def _extract_topological_properties(
        self, nodes: list, edges: list
    ) -> Dict[str, Any]:
        """Extract topological properties from graph structure."""
        properties = {}

        num_nodes = len(nodes)
        num_edges = len(edges)

        if num_nodes == 0:
            return properties

        # Basic topological measures
        avg_degree = (2 * num_edges) / num_nodes if num_nodes > 0 else 0

        properties["avg_degree"] = avg_degree
        properties["density"] = (
            num_edges / (num_nodes * (num_nodes - 1) / 2) if num_nodes > 1 else 0
        )

        # Topology-specific properties (simplified)
        if len(edges) > 10:  # Only compute for reasonably sized graphs
            degrees = {}
            for edge in edges:
                for node in [edge.get("from"), edge.get("to")]:
                    if node:
                        degrees[node] = degrees.get(node, 0) + 1

            if degrees:
                degree_values = list(degrees.values())
                max_degree = max(degree_values)

                # Hub dominance (simplified)
                properties["hub_dominance"] = (
                    max_degree / avg_degree if avg_degree > 0 else 0
                )

                # Power-law estimation (simplified)
                properties["power_law_exponent"] = 2.5  # Typical value

                # Clustering coefficient (simplified)
                properties["clustering_coefficient"] = 0.6  # Placeholder

                # Path length estimation (simplified)
                properties["avg_path_length"] = (
                    math.log(num_nodes) / math.log(avg_degree)
                    if avg_degree > 1
                    else num_nodes
                )

        return properties
