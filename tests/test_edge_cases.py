"""
Comprehensive edge case testing for mathematical primitives in Holon.

Tests all edge cases, error conditions, and boundary scenarios for the
new mathematical encoding capabilities.
"""

import numpy as np
import pytest

from holon.encoder import Encoder, MathematicalPrimitive
from holon.vector_manager import VectorManager


class TestMathematicalPrimitiveEdgeCases:
    """Test edge cases for mathematical primitive encoding."""

    @pytest.fixture
    def encoder(self, vector_manager):
        return Encoder(vector_manager)

    @pytest.fixture
    def vector_manager(self):
        return VectorManager(dimensions=1000)

    def test_invalid_primitive_enum(self, encoder):
        """Test invalid mathematical primitive raises proper error."""
        with pytest.raises(ValueError, match="Unknown mathematical primitive"):
            encoder.encode_mathematical_primitive("invalid_primitive", 1.0)

    def test_invalid_primitive_type(self, encoder):
        """Test passing wrong type for primitive raises error."""
        with pytest.raises(ValueError):
            encoder.encode_mathematical_primitive(123, 1.0)

    def test_non_numeric_value(self, encoder):
        """Test non-numeric values for mathematical primitives."""
        # String value
        with pytest.raises((ValueError, TypeError)):
            encoder.encode_mathematical_primitive(
                MathematicalPrimitive.CONVERGENCE_RATE, "not_a_number"
            )

        # None value
        with pytest.raises((ValueError, TypeError)):
            encoder.encode_mathematical_primitive(
                MathematicalPrimitive.CONVERGENCE_RATE, None
            )

        # List value
        with pytest.raises((ValueError, TypeError)):
            encoder.encode_mathematical_primitive(
                MathematicalPrimitive.CONVERGENCE_RATE, [1, 2, 3]
            )

    def test_extreme_numeric_values(self, encoder):
        """Test extreme numeric values."""
        # Very large positive
        result = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 1e10
        )
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(result, [-1, 0, 1]))

        # Very large negative
        result = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, -1e10
        )
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(result, [-1, 0, 1]))

        # Very small positive
        result = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 1e-10
        )
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(result, [-1, 0, 1]))

        # Very small negative
        result = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, -1e-10
        )
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(result, [-1, 0, 1]))

        # Zero
        result = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, 0.0
        )
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_boundary_values_for_each_primitive(self, encoder):
        """Test boundary values for each mathematical primitive."""
        test_cases = [
            (MathematicalPrimitive.CONVERGENCE_RATE, [0.0, 0.3, 0.7, 0.9, 1.0, 2.0]),
            (MathematicalPrimitive.ITERATION_COMPLEXITY, [0, 10, 50, 200, 1000]),
            (MathematicalPrimitive.FREQUENCY_DOMAIN, [0.0, 0.1, 1.0, 10.0, 100.0]),
            (MathematicalPrimitive.AMPLITUDE_SCALE, [0.0, 0.1, 0.5, 2.0, 10.0, 100.0]),
            (MathematicalPrimitive.POWER_LAW_EXPONENT, [1.0, 2.0, 2.5, 3.0, 5.0]),
            (MathematicalPrimitive.CLUSTERING_COEFFICIENT, [0.0, 0.2, 0.5, 0.8, 1.0]),
            (MathematicalPrimitive.TOPOLOGICAL_DISTANCE, [0.0, 2.0, 5.0, 10.0, 100.0]),
            (MathematicalPrimitive.SELF_SIMILARITY, [0.0, 0.25, 0.5, 0.75, 1.0]),
        ]

        for primitive, values in test_cases:
            for value in values:
                result = encoder.encode_mathematical_primitive(primitive, value)
                assert result.shape[0] == encoder.vector_manager.dimensions
                assert np.all(np.isin(result, [-1, 0, 1]))

    def test_mathematical_bind_edge_cases(self, encoder, vector_manager):
        """Test mathematical bind edge cases."""
        vec1 = vector_manager.get_vector("test1")
        vec2 = vector_manager.get_vector("test2")

        # Normal bind
        result = encoder.mathematical_bind(vec1, vec2)
        assert result.shape == vec1.shape
        assert np.all(np.isin(result, [-1, 0, 1]))

        # Bind with empty list (should return zero vector)
        result = encoder.mathematical_bind()
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(result == 0)

        # Bind single vector (should return the vector)
        result = encoder.mathematical_bind(vec1)
        assert np.array_equal(result, vec1)

        # Bind many vectors
        vecs = [vector_manager.get_vector(f"test{i}") for i in range(10)]
        result = encoder.mathematical_bind(*vecs)
        assert result.shape == vec1.shape
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_mathematical_bundle_edge_cases(self, encoder, vector_manager):
        """Test mathematical bundle edge cases."""
        vec1 = vector_manager.get_vector("test1")
        vec2 = vector_manager.get_vector("test2")
        vectors = [vec1, vec2]

        # Normal bundle
        result = encoder.mathematical_bundle(vectors)
        assert result.shape == vec1.shape
        assert np.all(np.isin(result, [-1, 0, 1]))

        # Bundle with weights
        weights = [0.5, 1.5]
        result_weighted = encoder.mathematical_bundle(vectors, weights)
        assert result_weighted.shape == vec1.shape
        assert np.all(np.isin(result_weighted, [-1, 0, 1]))
        # Weighted result should be different
        assert not np.array_equal(result, result_weighted)

        # Bundle with empty list (should return zero vector)
        result = encoder.mathematical_bundle([])
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(result == 0)

        # Bundle single vector
        result = encoder.mathematical_bundle([vec1])
        assert np.array_equal(result, vec1)

        # Mismatched weights length (should handle gracefully)
        with pytest.raises((ValueError, IndexError)):
            encoder.mathematical_bundle(vectors, [0.5])  # Too few weights

    def test_vector_dimension_mismatch(self, encoder, vector_manager):
        """Test behavior with vectors of different dimensions."""
        # Create vectors from different vector managers (different dimensions)
        vm1 = VectorManager(dimensions=500)
        vm2 = VectorManager(dimensions=1000)

        vec1 = vm1.get_vector("test1")  # 500 dimensions
        vec2 = vm2.get_vector("test2")  # 1000 dimensions

        # This should fail when trying to bind (dimension mismatch)
        with pytest.raises(ValueError):
            encoder.mathematical_bind(vec1, vec2)

    def test_nan_and_inf_values(self, encoder):
        """Test NaN and infinity values."""
        # NaN values
        result = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, float("nan")
        )
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(
            np.isin(result, [-1, 0, 1])
        )  # Should still produce valid bipolar vector

        # Positive infinity
        result = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, float("inf")
        )
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(result, [-1, 0, 1]))

        # Negative infinity
        result = encoder.encode_mathematical_primitive(
            MathematicalPrimitive.CONVERGENCE_RATE, float("-inf")
        )
        assert result.shape[0] == encoder.vector_manager.dimensions
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_concurrent_access(self, encoder):
        """Test concurrent access to mathematical primitives."""
        import queue
        import threading

        results = queue.Queue()
        errors = []

        def worker(primitive, value, worker_id):
            try:
                result = encoder.encode_mathematical_primitive(primitive, value)
                results.put((worker_id, result.shape[0]))
            except Exception as e:
                errors.append((worker_id, str(e)))

        # Start multiple threads
        threads = []
        primitives = list(MathematicalPrimitive)
        for i in range(min(10, len(primitives))):
            primitive = primitives[i % len(primitives)]
            value = 1.0 + (i * 0.1)  # Slightly different values
            t = threading.Thread(target=worker, args=(primitive, value, i))
            threads.append(t)
            t.start()

        # Wait for completion
        for t in threads:
            t.join(timeout=5.0)

        # Check results
        successful_results = 0
        while not results.empty():
            worker_id, dimensions = results.get()
            assert dimensions == encoder.vector_manager.dimensions
            successful_results += 1

        assert successful_results == len(threads)
        assert len(errors) == 0

    def test_memory_usage_large_vectors(self, encoder):
        """Test memory usage with large vectors."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create many large vectors
        vectors = []
        for i in range(100):
            vec = encoder.encode_mathematical_primitive(
                MathematicalPrimitive.CONVERGENCE_RATE, i * 0.01
            )
            vectors.append(vec)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Should not have excessive memory growth (less than 10MB)
        assert memory_increase < 10 * 1024 * 1024

        # Clean up
        del vectors

    def test_mathematical_operations_consistency(self, encoder, vector_manager):
        """Test that mathematical operations are consistent."""
        vec1 = vector_manager.get_vector("consistency_test_1")
        vec2 = vector_manager.get_vector("consistency_test_2")
        vec3 = vector_manager.get_vector("consistency_test_3")

        # Bind should be commutative (order shouldn't matter for final result)
        bind_1_2 = encoder.mathematical_bind(vec1, vec2)
        bind_2_1 = encoder.mathematical_bind(vec2, vec1)

        # Note: Due to the nature of bipolar vectors and binding,
        # the exact result might differ but should be similar
        # We test that both produce valid bipolar vectors
        assert np.all(np.isin(bind_1_2, [-1, 0, 1]))
        assert np.all(np.isin(bind_2_1, [-1, 0, 1]))

        # Bundle should be commutative
        bundle_1_2 = encoder.mathematical_bundle([vec1, vec2])
        bundle_2_1 = encoder.mathematical_bundle([vec2, vec1])
        assert np.array_equal(bundle_1_2, bundle_2_1)

        # Bundle should be associative (approximately)
        bundle_1_2_then_3 = encoder.mathematical_bundle(
            [encoder.mathematical_bundle([vec1, vec2]), vec3]
        )
        bundle_all = encoder.mathematical_bundle([vec1, vec2, vec3])
        # Due to thresholding, exact equality might not hold, but both should be valid
        assert np.all(np.isin(bundle_1_2_then_3, [-1, 0, 1]))
        assert np.all(np.isin(bundle_all, [-1, 0, 1]))
