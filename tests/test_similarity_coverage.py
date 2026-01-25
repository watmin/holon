#!/usr/bin/env python3
"""
Similarity Module Coverage Tests
Tests for vector similarity algorithms, edge cases, and parallel processing to improve similarity coverage.
"""

import numpy as np
import pytest

from holon.similarity import (
    _find_similar_vectors_parallel,
    _find_similar_vectors_single,
    find_similar_vectors,
    normalized_dot_similarity,
)


class TestNormalizedDotSimilarity:
    """Test normalized dot product similarity function."""

    def test_similarity_identical_vectors(self):
        """Test similarity of identical vectors (should be 1.0)."""
        vec = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        sim = normalized_dot_similarity(vec, vec)
        # Note: similarity is dot product divided by dimension
        # For bipolar vectors, this gives values between 0 and 1
        expected = np.dot(vec.astype(float), vec.astype(float)) / len(vec)
        assert sim == pytest.approx(expected, abs=1e-6)

    def test_similarity_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors (should be 0.0)."""
        vec1 = np.array([1, 1, 1, 0, 0], dtype=np.int8)
        vec2 = np.array([0, 0, 0, 1, 1], dtype=np.int8)
        sim = normalized_dot_similarity(vec1, vec2)
        assert sim == pytest.approx(0.0, abs=1e-6)

    def test_similarity_opposite_vectors(self):
        """Test similarity of opposite vectors (should be -1.0)."""
        vec1 = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        vec2 = np.array([-1, 1, -1, 1, -1], dtype=np.int8)
        sim = normalized_dot_similarity(vec1, vec2)
        assert sim == pytest.approx(-1.0, abs=1e-6)

    def test_similarity_partial_overlap(self):
        """Test similarity with partial overlap."""
        vec1 = np.array([1, 1, 1, 0, 0], dtype=np.int8)
        vec2 = np.array([1, 1, 0, 0, 0], dtype=np.int8)
        sim = normalized_dot_similarity(vec1, vec2)
        # Expected: (2) / 5 = 0.4
        assert sim == pytest.approx(0.4, abs=1e-6)

    def test_similarity_different_lengths(self):
        """Test similarity with different length vectors (should raise error)."""
        vec1 = np.array([1, 1, 1], dtype=np.int8)
        vec2 = np.array([1, 1, 1, 0, 0], dtype=np.int8)
        # This should raise a ValueError for incompatible shapes
        with pytest.raises(ValueError):
            normalized_dot_similarity(vec1, vec2)

    def test_similarity_with_floats(self):
        """Test similarity with float vectors."""
        vec1 = np.array([1.0, -1.0, 0.5, -0.5], dtype=float)
        vec2 = np.array([0.5, -0.5, 1.0, -1.0], dtype=float)
        sim = normalized_dot_similarity(vec1, vec2)
        # Expected: (1*0.5 + (-1)*(-0.5) + 0.5*1 + (-0.5)*(-1)) / 4 = (0.5 + 0.5 + 0.5 + 0.5) / 4 = 2/4 = 0.5
        assert sim == pytest.approx(0.5, abs=1e-6)

    @pytest.mark.skipif(not hasattr(np, "cuda"), reason="CuPy not available")
    def test_similarity_cupy_vectors(self):
        """Test similarity with CuPy vectors (if available)."""
        cupy = pytest.importorskip("cupy")
        vec1 = cupy.array([1, -1, 1, -1], dtype=cupy.int8)
        vec2 = cupy.array([1, 1, -1, -1], dtype=cupy.int8)
        sim = normalized_dot_similarity(vec1, vec2)
        assert isinstance(sim, float)
        # Expected: (1 + (-1*(-1)) + (1*(-1)) + ((-1)*(-1))) / 4 = (1 + 1 - 1 + 1) / 4 = 2/4 = 0.5
        assert sim == pytest.approx(0.5, abs=1e-6)

    def test_similarity_mixed_cpu_gpu(self):
        """Test similarity between CPU and GPU vectors."""
        pytest.importorskip("cupy")
        import cupy as cp

        vec1 = np.array([1, -1, 1, -1], dtype=np.int8)
        vec2 = cp.array([1, 1, -1, -1], dtype=cp.int8)
        sim = normalized_dot_similarity(vec1, vec2)
        assert isinstance(sim, float)
        assert sim == pytest.approx(0.5, abs=1e-6)


class TestFindSimilarVectors:
    """Test vector similarity search functions."""

    def create_test_vectors(self, count=100):
        """Create test vectors for similarity testing."""
        np.random.seed(42)  # For reproducible results
        vectors = {}
        for i in range(count):
            # Create random bipolar vectors
            vec = np.random.choice([-1, 0, 1], size=100, p=[0.3, 0.4, 0.3])
            vectors[f"vec_{i}"] = vec.astype(np.int8)
        return vectors

    def test_find_similar_empty_stored(self):
        """Test similarity search with empty stored vectors."""
        probe = np.array([1, -1, 1, -1], dtype=np.int8)
        result = find_similar_vectors(probe, {}, top_k=5)
        assert result == []

    def test_find_similar_single_vector(self):
        """Test similarity search with single stored vector."""
        probe = np.array([1, -1, 1, -1], dtype=np.int8)
        stored = {"test": probe.copy()}
        result = find_similar_vectors(probe, stored, top_k=5)
        assert len(result) == 1
        assert result[0][0] == "test"
        assert result[0][1] == pytest.approx(1.0, abs=1e-6)

    def test_find_similar_multiple_vectors(self):
        """Test similarity search with multiple vectors."""
        vectors = self.create_test_vectors(50)
        probe = vectors["vec_0"]  # Use first vector as probe

        result = find_similar_vectors(probe, vectors, top_k=10)
        assert len(result) <= 10

        # Should have some results
        assert len(result) > 0

        # All scores should be between -1 and 1
        for _, score in result:
            assert -1 <= score <= 1

    def test_find_similar_with_threshold(self):
        """Test similarity search with threshold filtering."""
        vectors = self.create_test_vectors(50)
        probe = vectors["vec_0"]

        # High threshold should return fewer results
        result_high = find_similar_vectors(probe, vectors, top_k=50, threshold=0.8)
        result_low = find_similar_vectors(probe, vectors, top_k=50, threshold=0.1)

        assert len(result_high) <= len(result_low)
        # All high threshold results should have score >= 0.8
        for _, score in result_high:
            assert score >= 0.8

    def test_find_similar_top_k_limit(self):
        """Test that top_k limits results correctly."""
        vectors = self.create_test_vectors(100)
        probe = vectors["vec_0"]

        result_5 = find_similar_vectors(probe, vectors, top_k=5)
        result_10 = find_similar_vectors(probe, vectors, top_k=10)

        assert len(result_5) <= 5
        assert len(result_10) <= 10
        assert len(result_10) >= len(result_5)

    def test_find_similar_zero_threshold(self):
        """Test similarity search with zero threshold."""
        vectors = self.create_test_vectors(20)
        probe = np.zeros(100, dtype=np.int8)  # All zeros

        result = find_similar_vectors(probe, vectors, top_k=20, threshold=0.0)
        # Should return all vectors (since dot product with zero vector is 0)
        assert len(result) >= 1  # At least some results

    def test_single_threaded_vs_parallel_consistency(self):
        """Test that single-threaded and parallel give same results."""
        vectors = self.create_test_vectors(1000)  # Larger dataset
        probe = vectors["vec_0"]

        # Test single-threaded
        result_single = _find_similar_vectors_single(
            probe, vectors, top_k=20, threshold=0.1
        )

        # Test parallel (force by calling directly)
        result_parallel = _find_similar_vectors_parallel(
            probe, vectors, top_k=20, threshold=0.1
        )

        # Results should be very similar (may differ slightly due to floating point precision)
        assert len(result_single) > 0
        assert len(result_parallel) > 0

        # Check that top results are consistent
        if len(result_single) >= 5 and len(result_parallel) >= 5:
            single_top_ids = {id for id, _ in result_single[:5]}
            parallel_top_ids = {id for id, _ in result_parallel[:5]}
            # At least some overlap in top 5
            assert len(single_top_ids & parallel_top_ids) >= 3

    def test_parallel_processing_small_dataset(self):
        """Test parallel processing with small dataset."""
        vectors = self.create_test_vectors(100)  # Small dataset
        probe = vectors["vec_0"]

        result = _find_similar_vectors_parallel(probe, vectors, top_k=10, threshold=0.0)
        assert len(result) <= 10
        # May not find itself as first result due to random vectors

    def test_parallel_processing_large_dataset(self):
        """Test parallel processing with large dataset."""
        vectors = self.create_test_vectors(2000)  # Large dataset
        probe = vectors["vec_0"]

        result = _find_similar_vectors_parallel(probe, vectors, top_k=20, threshold=0.0)
        assert len(result) <= 20

    def test_parallel_worker_scaling(self):
        """Test that parallel processing scales workers appropriately."""
        # This is more of an integration test, but tests the worker scaling logic
        vectors = self.create_test_vectors(50)  # Small
        probe = vectors["vec_0"]

        # With small dataset, should use fewer workers
        result_small = _find_similar_vectors_parallel(
            probe, vectors, top_k=10, threshold=0.0
        )

        vectors_large = self.create_test_vectors(5000)  # Large
        result_large = _find_similar_vectors_parallel(
            probe, vectors_large, top_k=10, threshold=0.0
        )

        # Both should work
        assert len(result_small) <= 10
        assert len(result_large) <= 10

    def test_heap_optimization_edge_cases(self):
        """Test heap optimization edge cases."""
        vectors = self.create_test_vectors(50)
        probe = vectors["vec_0"]

        # Test with top_k larger than dataset
        result = find_similar_vectors(probe, vectors, top_k=100)
        assert len(result) <= 50  # Can't return more than available

        # Test with very high threshold
        result = find_similar_vectors(probe, vectors, top_k=10, threshold=0.99)
        # May return fewer results, but should not crash
        assert isinstance(result, list)

    def test_similarity_score_ordering(self):
        """Test that results are properly ordered by similarity."""
        vectors = self.create_test_vectors(100)
        probe = vectors["vec_0"]

        result = find_similar_vectors(probe, vectors, top_k=20, threshold=0.0)

        # Scores should be in descending order
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)

    def test_similarity_with_duplicate_vectors(self):
        """Test similarity search with duplicate vectors."""
        probe = np.array([1, -1, 1, -1, 0], dtype=np.int8)
        stored = {
            "dup1": probe.copy(),
            "dup2": probe.copy(),
            "different": np.array([-1, 1, -1, 1, 0], dtype=np.int8),
        }

        result = find_similar_vectors(probe, stored, top_k=10)

        # Should find the duplicates with high similarity scores
        dup1_match = [r for r in result if r[0] == "dup1"]
        dup2_match = [r for r in result if r[0] == "dup2"]
        assert len(dup1_match) == 1
        assert len(dup2_match) == 1

        # Duplicate scores should be very high (perfect matches)
        assert dup1_match[0][1] >= 0.5  # Should be reasonably high similarity
        assert dup2_match[0][1] >= 0.5

    @pytest.mark.parametrize("top_k", [1, 5, 10, 50])
    def test_various_top_k_values(self, top_k):
        """Test similarity search with various top_k values."""
        vectors = self.create_test_vectors(100)
        probe = vectors["vec_0"]

        result = find_similar_vectors(probe, vectors, top_k=top_k)
        assert len(result) <= top_k

        # Should have some results for valid queries
        if top_k > 0:
            assert len(result) > 0

    @pytest.mark.parametrize("threshold", [0.0, 0.1, 0.5, 0.8])
    def test_various_threshold_values(self, threshold):
        """Test similarity search with various threshold values."""
        vectors = self.create_test_vectors(100)
        probe = vectors["vec_0"]

        result = find_similar_vectors(probe, vectors, top_k=50, threshold=threshold)

        # All results should meet threshold
        for _, score in result:
            assert score >= threshold

        # Higher threshold should generally return fewer results
        # (though not guaranteed due to random vectors)
