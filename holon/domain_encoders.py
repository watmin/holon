"""
DOMAIN-SPECIFIC ENCODERS: Core Mathematical Pattern Recognition

These encoders provide fundamental mathematical understanding that goes beyond
generic structural encoding. They represent core VSA/HDC capabilities for
mathematical pattern recognition.
"""

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .encoder import Encoder
from .vector_manager import VectorManager


class MathematicalPatternEncoder(Encoder):
    """
    Encodes mathematical patterns with semantic understanding.

    Provides fundamental encodings for:
    - Fractal patterns (convergence, complexity, iteration)
    - Wave phenomena (frequency, amplitude, phase, interference)
    - Polynomial curves (degree, coefficients, domain properties)
    """

    def __init__(self, vector_manager: VectorManager):
        super().__init__(vector_manager)
        self._init_mathematical_vocabularies()

    def _init_mathematical_vocabularies(self):
        """Initialize mathematical concept vocabularies."""
        # Fractal concepts
        self.fractal_concepts = {
            "convergence_rate": ["slow", "medium", "fast", "divergent"],
            "complexity_class": ["simple", "moderate", "complex", "chaotic"],
            "iteration_range": ["low", "medium", "high", "extreme"],
            "boundary_behavior": ["smooth", "jagged", "fractal", "chaotic"],
        }

        # Wave concepts
        self.wave_concepts = {
            "frequency_range": ["low", "medium", "high", "ultrasonic"],
            "amplitude_scale": ["micro", "small", "medium", "large", "macro"],
            "phase_relationship": ["in_phase", "out_phase", "quadrature", "complex"],
            "interference_pattern": [
                "constructive",
                "destructive",
                "complex",
                "standing",
            ],
        }

        # Polynomial concepts
        self.polynomial_concepts = {
            "degree_class": ["linear", "quadratic", "cubic", "higher_order"],
            "coefficient_magnitude": ["small", "medium", "large", "extreme"],
            "root_structure": ["real", "complex", "multiple", "simple"],
            "domain_behavior": ["monotonic", "oscillatory", "periodic", "chaotic"],
        }

    def encode_fractal_pattern(self, properties: Dict[str, Any]) -> np.ndarray:
        """
        Encode fractal pattern with mathematical semantics.

        Fundamental fractal properties:
        - Convergence behavior (mathematical stability)
        - Iteration complexity (computational depth)
        - Boundary characteristics (geometric regularity)
        - Self-similarity measures (structural recursion)
        """
        # Extract mathematical properties
        convergence_rate = properties.get("convergence_rate", 0.5)
        avg_iterations = properties.get("avg_iterations", 10)
        boundary_complexity = properties.get("boundary_complexity", 0.5)
        self_similarity = properties.get("self_similarity", 0.5)

        # Create fundamental mathematical vectors
        convergence_vector = self._encode_convergence_behavior(convergence_rate)
        iteration_vector = self._encode_iteration_complexity(avg_iterations)
        boundary_vector = self._encode_boundary_properties(boundary_complexity)
        similarity_vector = self._encode_self_similarity(self_similarity)

        # Bind mathematical relationships (fundamental VSA operation)
        fractal_signature = (
            convergence_vector * iteration_vector * boundary_vector * similarity_vector
        )

        # Add fractal indicator
        fractal_indicator = self.vector_manager.get_vector("fractal_pattern")
        return fractal_indicator * fractal_signature

    def encode_wave_phenomenon(self, properties: Dict[str, Any]) -> np.ndarray:
        """
        Encode wave phenomenon with physical semantics.

        Fundamental wave properties:
        - Frequency-amplitude relationship (energy conservation)
        - Phase coherence (temporal alignment)
        - Interference patterns (superposition principles)
        - Propagation characteristics (spatial-temporal coupling)
        """
        # Extract physical properties
        frequency = properties.get("frequency", 1.0)
        amplitude = properties.get("amplitude", 1.0)
        phase = properties.get("phase", 0.0)
        interference_strength = properties.get("interference_strength", 0.0)

        # Create fundamental physical vectors
        frequency_vector = self._encode_frequency_domain(frequency)
        amplitude_vector = self._encode_amplitude_domain(amplitude)
        phase_vector = self._encode_phase_coherence(phase)
        interference_vector = self._encode_interference_pattern(interference_strength)

        # Bind physical relationships (wave equation coupling)
        wave_signature = (
            frequency_vector * amplitude_vector * phase_vector * interference_vector
        )

        # Add wave indicator
        wave_indicator = self.vector_manager.get_vector("wave_phenomenon")
        return wave_indicator * wave_signature

    def encode_polynomial_curve(self, properties: Dict[str, Any]) -> np.ndarray:
        """
        Encode polynomial curve with algebraic semantics.

        Fundamental polynomial properties:
        - Degree-order relationships (algebraic complexity)
        - Coefficient interactions (term coupling)
        - Root configurations (solution structure)
        - Domain behavior (functional characteristics)
        """
        # Extract algebraic properties
        degree = properties.get("degree", 2)
        coefficients = properties.get("coefficients", [1, 0, 0])
        root_structure = properties.get("root_structure", "real")
        domain_behavior = properties.get("domain_behavior", "monotonic")

        # Create fundamental algebraic vectors
        degree_vector = self._encode_polynomial_degree(degree)
        coefficient_vector = self._encode_coefficient_structure(coefficients)
        root_vector = self._encode_root_configuration(root_structure)
        behavior_vector = self._encode_domain_behavior(domain_behavior)

        # Bind algebraic relationships (polynomial composition)
        polynomial_signature = (
            degree_vector * coefficient_vector * root_vector * behavior_vector
        )

        # Add polynomial indicator
        polynomial_indicator = self.vector_manager.get_vector("polynomial_curve")
        return polynomial_indicator * polynomial_signature

    # Fundamental mathematical encoding methods
    def _encode_convergence_behavior(self, rate: float) -> np.ndarray:
        """Encode mathematical convergence properties."""
        if rate < 0.3:
            category = "slow"
        elif rate < 0.7:
            category = "medium"
        elif rate < 0.9:
            category = "fast"
        else:
            category = "divergent"

        return self.vector_manager.get_vector(f"convergence_{category}")

    def _encode_iteration_complexity(self, iterations: int) -> np.ndarray:
        """Encode computational iteration complexity."""
        if iterations < 10:
            category = "low"
        elif iterations < 50:
            category = "medium"
        elif iterations < 200:
            category = "high"
        else:
            category = "extreme"

        return self.vector_manager.get_vector(f"iteration_{category}")

    def _encode_boundary_properties(self, complexity: float) -> np.ndarray:
        """Encode geometric boundary characteristics."""
        if complexity < 0.3:
            category = "smooth"
        elif complexity < 0.7:
            category = "jagged"
        else:
            category = "chaotic"

        return self.vector_manager.get_vector(f"boundary_{category}")

    def _encode_self_similarity(self, measure: float) -> np.ndarray:
        """Encode fractal self-similarity measures."""
        similarity_level = int(measure * 3) + 1  # 1-4 levels
        return self.vector_manager.get_vector(f"self_similarity_{similarity_level}")

    def _encode_frequency_domain(self, freq: float) -> np.ndarray:
        """Encode frequency domain properties."""
        if freq < 0.1:
            category = "low"
        elif freq < 1.0:
            category = "medium"
        elif freq < 10.0:
            category = "high"
        else:
            category = "ultrasonic"

        return self.vector_manager.get_vector(f"frequency_{category}")

    def _encode_amplitude_domain(self, amp: float) -> np.ndarray:
        """Encode amplitude domain properties."""
        if amp < 0.1:
            category = "micro"
        elif amp < 0.5:
            category = "small"
        elif amp < 2.0:
            category = "medium"
        elif amp < 10.0:
            category = "large"
        else:
            category = "macro"

        return self.vector_manager.get_vector(f"amplitude_{category}")

    def _encode_phase_coherence(self, phase: float) -> np.ndarray:
        """Encode phase coherence properties."""
        phase_norm = abs(phase) % (2 * math.pi)
        if phase_norm < math.pi / 4:
            category = "in_phase"
        elif phase_norm < math.pi / 2:
            category = "quadrature"
        elif phase_norm < 3 * math.pi / 4:
            category = "out_phase"
        else:
            category = "complex"

        return self.vector_manager.get_vector(f"phase_{category}")

    def _encode_interference_pattern(self, strength: float) -> np.ndarray:
        """Encode interference pattern properties."""
        if strength < 0.3:
            category = "constructive"
        elif strength < 0.7:
            category = "destructive"
        else:
            category = "standing"

        return self.vector_manager.get_vector(f"interference_{category}")

    def _encode_polynomial_degree(self, degree: int) -> np.ndarray:
        """Encode polynomial degree properties."""
        if degree == 1:
            category = "linear"
        elif degree == 2:
            category = "quadratic"
        elif degree == 3:
            category = "cubic"
        else:
            category = "higher_order"

        return self.vector_manager.get_vector(f"degree_{category}")

    def _encode_coefficient_structure(self, coeffs: List[float]) -> np.ndarray:
        """Encode coefficient structure properties."""
        magnitude = max(abs(c) for c in coeffs)
        if magnitude < 0.1:
            category = "small"
        elif magnitude < 1.0:
            category = "medium"
        elif magnitude < 10.0:
            category = "large"
        else:
            category = "extreme"

        return self.vector_manager.get_vector(f"coefficient_{category}")

    def _encode_root_configuration(self, structure: str) -> np.ndarray:
        """Encode root configuration properties."""
        return self.vector_manager.get_vector(f"root_{structure}")

    def _encode_domain_behavior(self, behavior: str) -> np.ndarray:
        """Encode domain behavior properties."""
        return self.vector_manager.get_vector(f"domain_{behavior}")


class GraphTopologyEncoder(Encoder):
    """
    Encodes graph structures with topological semantics.

    Provides fundamental encodings for:
    - Scale-free networks (power-law degree distributions)
    - Small-world networks (clustering and short paths)
    - Random networks (ergodic properties)
    """

    def __init__(self, vector_manager: VectorManager):
        super().__init__(vector_manager)
        self._init_topological_vocabularies()

    def _init_topological_vocabularies(self):
        """Initialize topological concept vocabularies."""
        # Scale-free concepts
        self.scale_free_concepts = {
            "power_law_exponent": ["shallow", "typical", "steep", "extreme"],
            "hub_structure": [
                "weak_hubs",
                "moderate_hubs",
                "strong_hubs",
                "dominant_hubs",
            ],
            "degree_distribution": [
                "broad_tail",
                "power_law",
                "exponential_cutoff",
                "truncated",
            ],
        }

        # Small-world concepts
        self.small_world_concepts = {
            "clustering_coefficient": ["low", "moderate", "high", "extreme"],
            "path_length": ["long", "moderate", "short", "minimal"],
            "regularity": ["random", "semi_regular", "regular", "lattice"],
        }

        # Random concepts
        self.random_concepts = {
            "degree_variance": ["uniform", "moderate", "variable", "extreme"],
            "connectivity": ["sparse", "moderate", "dense", "complete"],
            "clustering": ["minimal", "weak", "moderate", "strong"],
        }

    def encode_scale_free_topology(self, properties: Dict[str, Any]) -> np.ndarray:
        """
        Encode scale-free network topology.

        Fundamental scale-free properties:
        - Power-law degree distribution (preferential attachment)
        - Hub dominance (centrality structure)
        - Self-similar growth (network evolution)
        """
        # Extract topological properties
        power_law_exp = properties.get("power_law_exponent", 2.5)
        hub_dominance = properties.get("hub_dominance", 0.5)
        growth_pattern = properties.get("growth_pattern", "preferential")

        # Create fundamental topological vectors
        power_law_vector = self._encode_power_law_exponent(power_law_exp)
        hub_vector = self._encode_hub_structure(hub_dominance)
        growth_vector = self._encode_growth_pattern(growth_pattern)

        # Bind topological relationships (network formation principles)
        scale_free_signature = power_law_vector * hub_vector * growth_vector

        # Add topology indicator
        topology_indicator = self.vector_manager.get_vector("scale_free_topology")
        return topology_indicator * scale_free_signature

    def encode_small_world_topology(self, properties: Dict[str, Any]) -> np.ndarray:
        """
        Encode small-world network topology.

        Fundamental small-world properties:
        - High clustering with short paths (social network principle)
        - Regular local structure with random long-range connections
        - Balance between order and randomness
        """
        # Extract topological properties
        clustering_coeff = properties.get("clustering_coefficient", 0.6)
        avg_path_length = properties.get("avg_path_length", 3.0)
        rewiring_prob = properties.get("rewiring_probability", 0.1)

        # Create fundamental topological vectors
        clustering_vector = self._encode_clustering_coefficient(clustering_coeff)
        path_vector = self._encode_path_length(avg_path_length)
        rewiring_vector = self._encode_rewiring_probability(rewiring_prob)

        # Bind topological relationships (small-world formation)
        small_world_signature = clustering_vector * path_vector * rewiring_vector

        # Add topology indicator
        topology_indicator = self.vector_manager.get_vector("small_world_topology")
        return topology_indicator * small_world_signature

    def encode_random_topology(self, properties: Dict[str, Any]) -> np.ndarray:
        """
        Encode random network topology.

        Fundamental random properties:
        - Erdos-Renyi random graph characteristics
        - Poisson degree distribution
        - Absence of structural correlations
        """
        # Extract topological properties
        connection_prob = properties.get("connection_probability", 0.1)
        degree_variance = properties.get("degree_variance", 0.5)
        clustering_level = properties.get("clustering_level", 0.1)

        # Create fundamental topological vectors
        connection_vector = self._encode_connection_probability(connection_prob)
        variance_vector = self._encode_degree_variance(degree_variance)
        clustering_vector = self._encode_clustering_level(clustering_level)

        # Bind topological relationships (random formation)
        random_signature = connection_vector * variance_vector * clustering_vector

        # Add topology indicator
        topology_indicator = self.vector_manager.get_vector("random_topology")
        return topology_indicator * random_signature

    # Fundamental topological encoding methods
    def _encode_power_law_exponent(self, exponent: float) -> np.ndarray:
        """Encode power-law exponent properties."""
        if exponent < 2.0:
            category = "shallow"
        elif exponent < 2.5:
            category = "typical"
        elif exponent < 3.0:
            category = "steep"
        else:
            category = "extreme"

        return self.vector_manager.get_vector(f"power_law_{category}")

    def _encode_hub_structure(self, dominance: float) -> np.ndarray:
        """Encode hub dominance properties."""
        if dominance < 0.3:
            category = "weak_hubs"
        elif dominance < 0.6:
            category = "moderate_hubs"
        elif dominance < 0.8:
            category = "strong_hubs"
        else:
            category = "dominant_hubs"

        return self.vector_manager.get_vector(f"hub_{category}")

    def _encode_growth_pattern(self, pattern: str) -> np.ndarray:
        """Encode network growth patterns."""
        return self.vector_manager.get_vector(f"growth_{pattern}")

    def _encode_clustering_coefficient(self, coeff: float) -> np.ndarray:
        """Encode clustering coefficient properties."""
        if coeff < 0.2:
            category = "low"
        elif coeff < 0.5:
            category = "moderate"
        elif coeff < 0.8:
            category = "high"
        else:
            category = "extreme"

        return self.vector_manager.get_vector(f"clustering_{category}")

    def _encode_path_length(self, length: float) -> np.ndarray:
        """Encode average path length properties."""
        if length > 10:
            category = "long"
        elif length > 5:
            category = "moderate"
        elif length > 2:
            category = "short"
        else:
            category = "minimal"

        return self.vector_manager.get_vector(f"path_{category}")

    def _encode_rewiring_probability(self, prob: float) -> np.ndarray:
        """Encode rewiring probability properties."""
        if prob < 0.05:
            category = "low"
        elif prob < 0.2:
            category = "moderate"
        else:
            category = "high"

        return self.vector_manager.get_vector(f"rewiring_{category}")

    def _encode_connection_probability(self, prob: float) -> np.ndarray:
        """Encode connection probability properties."""
        if prob < 0.05:
            category = "sparse"
        elif prob < 0.2:
            category = "moderate"
        elif prob < 0.5:
            category = "dense"
        else:
            category = "complete"

        return self.vector_manager.get_vector(f"connection_{category}")

    def _encode_degree_variance(self, variance: float) -> np.ndarray:
        """Encode degree variance properties."""
        if variance < 0.3:
            category = "uniform"
        elif variance < 0.6:
            category = "moderate"
        elif variance < 0.8:
            category = "variable"
        else:
            category = "extreme"

        return self.vector_manager.get_vector(f"variance_{category}")

    def _encode_clustering_level(self, level: float) -> np.ndarray:
        """Encode clustering level properties."""
        if level < 0.1:
            category = "minimal"
        elif level < 0.3:
            category = "weak"
        elif level < 0.6:
            category = "moderate"
        else:
            category = "strong"

        return self.vector_manager.get_vector(f"clustering_{category}")
