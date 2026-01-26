"""
Unit tests for mathematical primitives in Holon VSA/HDC system.

Tests the fundamental mathematical encoding capabilities added to the Encoder class.
These primitives provide mathematical understanding that users cannot easily compose
from existing generic operations.
"""

import numpy as np
import pytest

from holon.encoder import Encoder, MathematicalPrimitive
from holon.vector_manager import VectorManager


class TestMathematicalPrimitives:
    """Test mathematical primitive encoding functionality."""

    @pytest.fixture
    def vector_manager(self):
        """Create a vector manager for testing."""
        return VectorManager(dimensions=1000)  # Smaller for faster tests

    @pytest.fixture
    def encoder(self, vector_manager):
        """Create an encoder with mathematical primitives."""
        return Encoder(vector_manager)

    def test_convergence_rate_encoding(self, encoder):
        """Test convergence rate mathematical primitive."""
        # Test different convergence rates
        slow_vec = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 0.2
        )
        fast_vec = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 0.9
        )
        divergent_vec = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 1.5
        )

        # Verify they're different vectors
        assert not np.array_equal(slow_vec, fast_vec)
        assert not np.array_equal(fast_vec, divergent_vec)
        assert not np.array_equal(slow_vec, divergent_vec)

        # Verify they're bipolar vectors
        assert np.all(np.isin(slow_vec, [-1, 0, 1]))
        assert np.all(np.isin(fast_vec, [-1, 0, 1]))
        assert np.all(np.isin(divergent_vec, [-1, 0, 1]))

    def test_iteration_complexity_encoding(self, encoder):
        """Test iteration complexity mathematical primitive."""
        low_complexity = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.ITERATION_COMPLEXITY, 5
        )
        high_complexity = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.ITERATION_COMPLEXITY, 150
        )

        # Should be different
        assert not np.array_equal(low_complexity, high_complexity)

        # Should be bipolar
        assert np.all(np.isin(low_complexity, [-1, 0, 1]))
        assert np.all(np.isin(high_complexity, [-1, 0, 1]))

    def test_frequency_domain_encoding(self, encoder):
        """Test frequency domain mathematical primitive."""
        # Use values that definitely fall into different categories
        low_freq = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.FREQUENCY_DOMAIN, 0.05  # medium_low_frequency
        )
        med_freq = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.FREQUENCY_DOMAIN, 5.0  # medium_frequency
        )
        high_freq = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.FREQUENCY_DOMAIN, 50.0  # high_frequency
        )

        # Different frequencies should produce different vectors
        assert not np.array_equal(low_freq, med_freq)
        assert not np.array_equal(med_freq, high_freq)
        assert np.all(np.isin(low_freq, [-1, 0, 1]))
        assert np.all(np.isin(med_freq, [-1, 0, 1]))
        assert np.all(np.isin(high_freq, [-1, 0, 1]))

    def test_power_law_exponent_encoding(self, encoder):
        """Test power-law exponent mathematical primitive."""
        shallow = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.POWER_LAW_EXPONENT, 2.1
        )
        steep = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.POWER_LAW_EXPONENT, 3.5
        )

        assert not np.array_equal(shallow, steep)
        assert np.all(np.isin(shallow, [-1, 0, 1]))
        assert np.all(np.isin(steep, [-1, 0, 1]))

    def test_clustering_coefficient_encoding(self, encoder):
        """Test clustering coefficient mathematical primitive."""
        low_clustering = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CLUSTERING_COEFFICIENT, 0.1
        )
        high_clustering = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CLUSTERING_COEFFICIENT, 0.8
        )

        assert not np.array_equal(low_clustering, high_clustering)
        assert np.all(np.isin(low_clustering, [-1, 0, 1]))
        assert np.all(np.isin(high_clustering, [-1, 0, 1]))

    def test_mathematical_bind_operation(self, encoder, vector_manager):
        """Test mathematical binding operation."""
        vec1 = vector_manager.get_vector("test_vec_1")
        vec2 = vector_manager.get_vector("test_vec_2")
        vec3 = vector_manager.get_vector("test_vec_3")

        # Bind two vectors
        bound = encoder.mathematical_bind(vec1, vec2)
        assert bound.shape == vec1.shape
        assert np.all(np.isin(bound, [-1, 0, 1]))

        # Bind three vectors
        bound3 = encoder.mathematical_bind(vec1, vec2, vec3)
        assert bound3.shape == vec1.shape
        assert np.all(np.isin(bound3, [-1, 0, 1]))

        # Should be different from inputs
        assert not np.array_equal(bound, vec1)
        assert not np.array_equal(bound, vec2)

    def test_mathematical_bundle_operation(self, encoder, vector_manager):
        """Test mathematical bundling operation."""
        vec1 = vector_manager.get_vector("bundle_test_1")
        vec2 = vector_manager.get_vector("bundle_test_2")
        vec3 = vector_manager.get_vector("bundle_test_3")

        vectors = [vec1, vec2, vec3]

        # Bundle without weights
        bundled = encoder.mathematical_bundle(vectors)
        assert bundled.shape == vec1.shape
        assert np.all(np.isin(bundled, [-1, 0, 1]))

        # Bundle with weights
        weights = [0.5, 1.0, 1.5]
        bundled_weighted = encoder.mathematical_bundle(vectors, weights)
        assert bundled_weighted.shape == vec1.shape
        assert np.all(np.isin(bundled_weighted, [-1, 0, 1]))

        # Weighted and unweighted should be different
        assert not np.array_equal(bundled, bundled_weighted)

    def test_mathematical_primitive_consistency(self, encoder):
        """Test that same inputs produce same outputs."""
        # Same convergence rate should produce same vector
        vec1 = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 0.7
        )
        vec2 = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 0.7
        )

        assert np.array_equal(vec1, vec2)

    def test_mathematical_primitive_different_inputs(self, encoder):
        """Test that different inputs produce different outputs."""
        vec1 = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.FREQUENCY_DOMAIN, 0.05  # medium_low_frequency
        )
        vec2 = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.FREQUENCY_DOMAIN, 50.0  # high_frequency
        )

        assert not np.array_equal(vec1, vec2)

    def test_mathematical_bind_empty_input(self, encoder):
        """Test mathematical bind with empty input."""
        result = encoder.mathematical_bind()
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(result == 0)  # Should return zero vector

    def test_mathematical_bundle_empty_input(self, encoder):
        """Test mathematical bundle with empty input."""
        result = encoder.mathematical_bundle([])
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(result == 0)  # Should be zero vector

    def test_invalid_mathematical_primitive(self, encoder):
        """Test invalid mathematical primitive raises error."""
        with pytest.raises(ValueError):
            encoder.encode_mathematical_primitive("invalid_primitive", 1.0)


class TestMathematicalPrimitiveDomains:
    """Test mathematical primitives for different domains."""

    @pytest.fixture
    def vector_manager(self):
        """Create a vector manager for testing."""
        from holon.vector_manager import VectorManager

        return VectorManager(dimensions=1000)

    @pytest.fixture
    def encoder(self, vector_manager):
        return Encoder(vector_manager)

    def test_fractal_domain_primitives(self, encoder):
        """Test primitives relevant to fractal patterns."""
        convergence = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 0.8
        )
        complexity = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.ITERATION_COMPLEXITY, 50
        )
        similarity = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.SELF_SIMILARITY, 0.85
        )

        # Bind fractal properties
        fractal_signature = encoder.mathematical_bind(
            convergence, complexity, similarity
        )

        assert fractal_signature.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(fractal_signature, [-1, 0, 1]))

    def test_wave_domain_primitives(self, encoder):
        """Test primitives relevant to wave phenomena."""
        frequency = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.FREQUENCY_DOMAIN, 2.5
        )
        amplitude = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.AMPLITUDE_SCALE, 0.8
        )
        # Using topological distance as phase coherence approximation
        coherence = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.TOPOLOGICAL_DISTANCE, 0.9
        )

        # Bind wave properties
        wave_signature = encoder.mathematical_bind(frequency, amplitude, coherence)

        assert wave_signature.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(wave_signature, [-1, 0, 1]))

    def test_graph_domain_primitives(self, encoder):
        """Test primitives relevant to graph topologies."""
        power_law = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.POWER_LAW_EXPONENT, 2.5
        )
        clustering = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CLUSTERING_COEFFICIENT, 0.7
        )
        degree_dist = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.TOPOLOGICAL_DISTANCE, 2.1
        )

        # Bind graph properties
        graph_signature = encoder.mathematical_bind(power_law, clustering, degree_dist)

        assert graph_signature.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(graph_signature, [-1, 0, 1]))


class TestSemanticEncoderIntegration:
    """Test semantic encoder integration with mathematical primitives."""

    @pytest.fixture
    def vector_manager(self):
        """Create a vector manager for testing."""
        from holon.vector_manager import VectorManager

        return VectorManager(dimensions=1000)

    @pytest.fixture
    def semantic_encoder(self, vector_manager):
        """Create semantic encoder for testing."""
        from holon.semantic_encoder import SemanticEncoder

        return SemanticEncoder(vector_manager)

    def test_semantic_encoder_creation(self, semantic_encoder):
        """Test semantic encoder can be created."""
        assert semantic_encoder is not None
        assert hasattr(semantic_encoder, "math_encoder")
        assert hasattr(semantic_encoder, "graph_encoder")

    def test_semantic_encoder_inherits_encoder(self, semantic_encoder):
        """Test semantic encoder inherits from base Encoder."""
        from holon.encoder import Encoder

        assert isinstance(semantic_encoder, Encoder)

    def test_semantic_encoder_has_mathematical_primitives(self, semantic_encoder):
        """Test semantic encoder has access to mathematical primitives."""
        # Should have the mathematical primitive methods
        assert hasattr(semantic_encoder, "encode_mathematical_primitive")
        assert hasattr(semantic_encoder, "mathematical_bind")
        assert hasattr(semantic_encoder, "mathematical_bundle")
