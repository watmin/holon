"""
Unit tests for holon/distance.py

Tests all distance and similarity metrics with:
- Mathematical correctness
- Edge cases (zero vectors, identical, opposite)
- Expected properties (symmetry, triangle inequality where applicable)
- DistanceEngine unified interface
"""

import math

import numpy as np
import pytest

from holon.distance import (  # Core functions; VSA-specific; Advanced; Engine and utilities
    DistanceEngine,
    DistanceMetric,
    agreement_similarity,
    chebyshev_distance,
    chebyshev_similarity,
    compare_metrics,
    cosine_similarity,
    dot_product_similarity,
    euclidean_distance,
    euclidean_similarity,
    get_recommended_metric,
    hamming_distance,
    hamming_similarity,
    manhattan_distance,
    manhattan_similarity,
    minkowski_distance,
    overlap_count,
    overlap_similarity,
    weighted_cosine_similarity,
    weighted_euclidean_distance,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def bipolar_vectors():
    """Create test bipolar vectors (-1, 0, +1)."""
    np.random.seed(42)
    D = 1000

    # Random bipolar
    vec_a = np.random.choice([-1, 0, 1], size=D, p=[0.33, 0.34, 0.33]).astype(np.int8)

    # Identical
    vec_identical = vec_a.copy()

    # Similar (90% same)
    vec_similar = vec_a.copy()
    flip_idx = np.random.choice(D, size=int(D * 0.1), replace=False)
    vec_similar[flip_idx] *= -1

    # Opposite
    vec_opposite = -vec_a

    # Orthogonal (independent random)
    vec_orthogonal = np.random.choice([-1, 0, 1], size=D, p=[0.33, 0.34, 0.33]).astype(
        np.int8
    )

    # Zero vector
    vec_zero = np.zeros(D, dtype=np.int8)

    return {
        "a": vec_a,
        "identical": vec_identical,
        "similar": vec_similar,
        "opposite": vec_opposite,
        "orthogonal": vec_orthogonal,
        "zero": vec_zero,
    }


@pytest.fixture
def simple_vectors():
    """Create simple test vectors for exact calculations."""
    return {
        "ones": np.ones(100, dtype=np.float64),
        "neg_ones": -np.ones(100, dtype=np.float64),
        "zeros": np.zeros(100, dtype=np.float64),
        "half": np.array([1, 1, -1, -1] * 25, dtype=np.float64),
    }


# =============================================================================
# Cosine Similarity Tests
# =============================================================================


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self, bipolar_vectors):
        """Identical vectors should have cosine similarity of 1.0."""
        sim = cosine_similarity(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_opposite_vectors(self, bipolar_vectors):
        """Opposite vectors should have cosine similarity of -1.0."""
        sim = cosine_similarity(bipolar_vectors["a"], bipolar_vectors["opposite"])
        assert sim == pytest.approx(-1.0, abs=1e-6)

    def test_similar_vectors(self, bipolar_vectors):
        """90% similar vectors should have high positive similarity."""
        sim = cosine_similarity(bipolar_vectors["a"], bipolar_vectors["similar"])
        assert sim > 0.7  # Should be high

    def test_orthogonal_vectors(self, bipolar_vectors):
        """Orthogonal random vectors should have similarity near 0."""
        sim = cosine_similarity(bipolar_vectors["a"], bipolar_vectors["orthogonal"])
        assert abs(sim) < 0.2  # Should be near 0

    def test_zero_vector(self, bipolar_vectors):
        """Zero vector should return 0 similarity."""
        sim = cosine_similarity(bipolar_vectors["a"], bipolar_vectors["zero"])
        assert sim == 0.0

    def test_symmetry(self, bipolar_vectors):
        """Cosine similarity should be symmetric."""
        sim1 = cosine_similarity(bipolar_vectors["a"], bipolar_vectors["similar"])
        sim2 = cosine_similarity(bipolar_vectors["similar"], bipolar_vectors["a"])
        assert sim1 == pytest.approx(sim2, abs=1e-10)

    def test_range(self, bipolar_vectors):
        """Cosine similarity should be in [-1, 1]."""
        for name, vec in bipolar_vectors.items():
            if name == "zero":
                continue
            sim = cosine_similarity(bipolar_vectors["a"], vec)
            assert -1.0 <= sim <= 1.0


# =============================================================================
# Dot Product Similarity Tests
# =============================================================================


class TestDotProductSimilarity:
    """Tests for dot_product_similarity function."""

    def test_identical_vectors(self, simple_vectors):
        """Identical vectors should have positive dot product."""
        sim = dot_product_similarity(simple_vectors["ones"], simple_vectors["ones"])
        assert sim == 100.0  # 100 ones * 100 ones

    def test_opposite_vectors(self, simple_vectors):
        """Opposite vectors should have negative dot product."""
        sim = dot_product_similarity(simple_vectors["ones"], simple_vectors["neg_ones"])
        assert sim == -100.0

    def test_orthogonal(self, simple_vectors):
        """Orthogonal vectors should have zero dot product."""
        sim = dot_product_similarity(simple_vectors["ones"], simple_vectors["half"])
        assert sim == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Euclidean Distance Tests
# =============================================================================


class TestEuclideanDistance:
    """Tests for euclidean_distance function."""

    def test_identical_vectors(self, bipolar_vectors):
        """Identical vectors should have zero distance."""
        dist = euclidean_distance(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert dist == 0.0

    def test_opposite_vectors(self, bipolar_vectors):
        """Opposite bipolar vectors should have large distance."""
        dist = euclidean_distance(bipolar_vectors["a"], bipolar_vectors["opposite"])
        # For bipolar, diff is 0 or 2, so max distance is 2*sqrt(non_zeros)
        assert dist > 0

    def test_non_negative(self, bipolar_vectors):
        """Euclidean distance should be non-negative."""
        for name, vec in bipolar_vectors.items():
            dist = euclidean_distance(bipolar_vectors["a"], vec)
            assert dist >= 0

    def test_symmetry(self, bipolar_vectors):
        """Euclidean distance should be symmetric."""
        dist1 = euclidean_distance(bipolar_vectors["a"], bipolar_vectors["similar"])
        dist2 = euclidean_distance(bipolar_vectors["similar"], bipolar_vectors["a"])
        assert dist1 == pytest.approx(dist2, abs=1e-10)

    def test_triangle_inequality(self, bipolar_vectors):
        """Euclidean distance should satisfy triangle inequality."""
        d_ab = euclidean_distance(bipolar_vectors["a"], bipolar_vectors["similar"])
        d_bc = euclidean_distance(
            bipolar_vectors["similar"], bipolar_vectors["orthogonal"]
        )
        d_ac = euclidean_distance(bipolar_vectors["a"], bipolar_vectors["orthogonal"])
        assert d_ac <= d_ab + d_bc + 1e-10  # Allow small floating point error


class TestEuclideanSimilarity:
    """Tests for euclidean_similarity (distance → similarity conversion)."""

    def test_identical_has_max_similarity(self, bipolar_vectors):
        """Identical vectors should have similarity of 1.0."""
        sim = euclidean_similarity(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert sim == 1.0

    def test_range(self, bipolar_vectors):
        """Euclidean similarity should be in (0, 1]."""
        for name, vec in bipolar_vectors.items():
            sim = euclidean_similarity(bipolar_vectors["a"], vec)
            assert 0 < sim <= 1.0


# =============================================================================
# Manhattan Distance Tests
# =============================================================================


class TestManhattanDistance:
    """Tests for manhattan_distance function."""

    def test_identical_vectors(self, bipolar_vectors):
        """Identical vectors should have zero distance."""
        dist = manhattan_distance(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert dist == 0.0

    def test_simple_calculation(self):
        """Test with simple vectors for exact calculation."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 2, 1])
        # |1-4| + |2-2| + |3-1| = 3 + 0 + 2 = 5
        assert manhattan_distance(v1, v2) == 5.0

    def test_non_negative(self, bipolar_vectors):
        """Manhattan distance should be non-negative."""
        for name, vec in bipolar_vectors.items():
            dist = manhattan_distance(bipolar_vectors["a"], vec)
            assert dist >= 0


# =============================================================================
# Hamming Distance Tests (VSA-specific)
# =============================================================================


class TestHammingDistance:
    """Tests for hamming_distance function (VSA-specific)."""

    def test_identical_vectors(self, bipolar_vectors):
        """Identical vectors should have zero Hamming distance."""
        dist = hamming_distance(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert dist == 0

    def test_opposite_vectors(self):
        """Opposite bipolar vectors should have max Hamming distance for non-zeros."""
        v1 = np.array([1, 1, -1, -1, 0], dtype=np.int8)
        v2 = np.array([-1, -1, 1, 1, 0], dtype=np.int8)
        # All 4 non-zero positions differ, zeros match
        dist = hamming_distance(v1, v2)
        assert dist == 4

    def test_partial_match(self):
        """Test partial matching."""
        v1 = np.array([1, 1, 1, -1, -1], dtype=np.int8)
        v2 = np.array([1, -1, 1, -1, 1], dtype=np.int8)
        # Positions: match, diff, match, match, diff = 2 differences
        dist = hamming_distance(v1, v2)
        assert dist == 2

    def test_zeros_match(self):
        """Zeros should match with zeros."""
        v1 = np.array([0, 0, 1, -1], dtype=np.int8)
        v2 = np.array([0, 0, 1, -1], dtype=np.int8)
        dist = hamming_distance(v1, v2)
        assert dist == 0

    def test_integer_result(self, bipolar_vectors):
        """Hamming distance should always be an integer."""
        dist = hamming_distance(bipolar_vectors["a"], bipolar_vectors["similar"])
        assert isinstance(dist, int)


class TestHammingSimilarity:
    """Tests for hamming_similarity function."""

    def test_identical_vectors(self, bipolar_vectors):
        """Identical vectors should have similarity of 1.0."""
        sim = hamming_similarity(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert sim == 1.0

    def test_range(self, bipolar_vectors):
        """Hamming similarity should be in [0, 1]."""
        for name, vec in bipolar_vectors.items():
            sim = hamming_similarity(bipolar_vectors["a"], vec)
            assert 0.0 <= sim <= 1.0


# =============================================================================
# Overlap Tests (VSA-specific)
# =============================================================================


class TestOverlap:
    """Tests for overlap_count and overlap_similarity functions."""

    def test_overlap_count_identical(self):
        """Identical vectors should have full overlap."""
        v = np.array([1, 1, -1, -1, 1], dtype=np.int8)
        count = overlap_count(v, v)
        assert count == 5  # All 5 positions match

    def test_overlap_count_opposite(self):
        """Opposite vectors should have zero overlap."""
        v1 = np.array([1, 1, -1, -1], dtype=np.int8)
        v2 = -v1
        count = overlap_count(v1, v2)
        assert count == 0

    def test_overlap_similarity_identical(self, bipolar_vectors):
        """Identical vectors should have overlap similarity of 1.0."""
        sim = overlap_similarity(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert sim == 1.0

    def test_overlap_similarity_opposite(self):
        """Opposite vectors should have overlap similarity of 0."""
        v1 = np.array([1, 1, -1, -1], dtype=np.int8)
        v2 = -v1
        sim = overlap_similarity(v1, v2)
        assert sim == 0.0


# =============================================================================
# Agreement Similarity Tests (VSA-specific)
# =============================================================================


class TestAgreementSimilarity:
    """Tests for agreement_similarity function."""

    def test_identical_vectors(self):
        """Identical non-zero vectors should have high agreement."""
        v = np.array([1, 1, -1, -1, 1], dtype=np.int8)
        sim = agreement_similarity(v, v)
        # 5 agreements, 0 disagreements, D=5 → (5-0)/5 = 1.0
        assert sim == 1.0

    def test_opposite_vectors(self):
        """Opposite vectors should have negative agreement."""
        v1 = np.array([1, 1, -1, -1], dtype=np.int8)
        v2 = -v1
        sim = agreement_similarity(v1, v2)
        # 0 agreements, 4 disagreements, D=4 → (0-4)/4 = -1.0
        assert sim == -1.0

    def test_mixed(self):
        """Test with mixed agreements/disagreements."""
        v1 = np.array([1, 1, 1, -1], dtype=np.int8)
        v2 = np.array([1, 1, -1, -1], dtype=np.int8)
        # Agreements: pos 0, 1, 3 = 3
        # Disagreements: pos 2 = 1
        # (3-1)/4 = 0.5
        sim = agreement_similarity(v1, v2)
        assert sim == pytest.approx(0.5, abs=1e-6)

    def test_range(self, bipolar_vectors):
        """Agreement similarity should be in [-1, 1]."""
        for name, vec in bipolar_vectors.items():
            if name == "zero":
                continue
            sim = agreement_similarity(bipolar_vectors["a"], vec)
            assert -1.0 <= sim <= 1.0


# =============================================================================
# Chebyshev Distance Tests
# =============================================================================


class TestChebyshevDistance:
    """Tests for chebyshev_distance function."""

    def test_identical_vectors(self, bipolar_vectors):
        """Identical vectors should have zero Chebyshev distance."""
        dist = chebyshev_distance(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert dist == 0.0

    def test_simple_calculation(self):
        """Test with simple vectors."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([1, 2, 10])
        # max(|1-1|, |2-2|, |3-10|) = max(0, 0, 7) = 7
        assert chebyshev_distance(v1, v2) == 7.0

    def test_bipolar_max_distance(self):
        """For bipolar, max Chebyshev distance is 2."""
        v1 = np.array([1, 1, 1], dtype=np.int8)
        v2 = np.array([-1, -1, -1], dtype=np.int8)
        dist = chebyshev_distance(v1, v2)
        assert dist == 2.0


class TestChebyshevSimilarity:
    """Tests for chebyshev_similarity function."""

    def test_identical_has_max_similarity(self, bipolar_vectors):
        """Identical vectors should have similarity of 1.0."""
        sim = chebyshev_similarity(bipolar_vectors["a"], bipolar_vectors["identical"])
        assert sim == 1.0

    def test_opposite_bipolar(self):
        """Opposite bipolar vectors should have similarity of 0."""
        v1 = np.array([1, 1, 1], dtype=np.int8)
        v2 = np.array([-1, -1, -1], dtype=np.int8)
        sim = chebyshev_similarity(v1, v2)
        assert sim == 0.0


# =============================================================================
# Minkowski Distance Tests
# =============================================================================


class TestMinkowskiDistance:
    """Tests for minkowski_distance function."""

    def test_p1_equals_manhattan(self):
        """Minkowski with p=1 should equal Manhattan distance."""
        v1 = np.array([1, 2, 3, 4])
        v2 = np.array([5, 6, 7, 8])
        mink = minkowski_distance(v1, v2, p=1.0)
        manh = manhattan_distance(v1, v2)
        assert mink == pytest.approx(manh, abs=1e-6)

    def test_p2_equals_euclidean(self):
        """Minkowski with p=2 should equal Euclidean distance."""
        v1 = np.array([1, 2, 3, 4])
        v2 = np.array([5, 6, 7, 8])
        mink = minkowski_distance(v1, v2, p=2.0)
        eucl = euclidean_distance(v1, v2)
        assert mink == pytest.approx(eucl, abs=1e-6)

    def test_identical_vectors(self, bipolar_vectors):
        """Identical vectors should have zero distance for any p."""
        for p in [1.0, 2.0, 3.0]:
            dist = minkowski_distance(
                bipolar_vectors["a"], bipolar_vectors["identical"], p=p
            )
            assert dist == 0.0


# =============================================================================
# Weighted Metrics Tests
# =============================================================================


class TestWeightedCosineSimilarity:
    """Tests for weighted_cosine_similarity function."""

    def test_uniform_weights_equals_cosine(self, bipolar_vectors):
        """Uniform weights should give same result as regular cosine."""
        weights = np.ones(len(bipolar_vectors["a"]), dtype=np.float64)
        weighted = weighted_cosine_similarity(
            bipolar_vectors["a"], bipolar_vectors["similar"], weights
        )
        regular = cosine_similarity(bipolar_vectors["a"], bipolar_vectors["similar"])
        assert weighted == pytest.approx(regular, abs=1e-6)

    def test_zero_weights_for_dimensions(self):
        """Zeroing weights should exclude those dimensions."""
        v1 = np.array([1, 1, 1, 1], dtype=np.float64)
        v2 = np.array([1, 1, -1, -1], dtype=np.float64)
        # Full cosine: (1+1-1-1) / (2*2) = 0

        # Weight only matching dimensions
        weights = np.array([1, 1, 0, 0], dtype=np.float64)
        weighted = weighted_cosine_similarity(v1, v2, weights)
        # Now only considers first 2 dims: (1+1) / (sqrt(2)*sqrt(2)) = 1.0
        assert weighted == pytest.approx(1.0, abs=1e-6)

    def test_high_weights_emphasize_dimensions(self):
        """Higher weights should emphasize those dimensions."""
        v1 = np.array([1, 1, -1], dtype=np.float64)
        v2 = np.array([1, -1, -1], dtype=np.float64)
        # Regular: match at 0, diff at 1, match at 2

        # Emphasize the matching dimensions
        weights_match = np.array([10, 1, 10], dtype=np.float64)
        sim_match = weighted_cosine_similarity(v1, v2, weights_match)

        # Emphasize the differing dimension
        weights_diff = np.array([1, 10, 1], dtype=np.float64)
        sim_diff = weighted_cosine_similarity(v1, v2, weights_diff)

        # Emphasizing matches should give higher similarity
        assert sim_match > sim_diff


class TestWeightedEuclideanDistance:
    """Tests for weighted_euclidean_distance function."""

    def test_uniform_weights_equals_euclidean(self, bipolar_vectors):
        """Uniform weights should give same result as regular Euclidean."""
        weights = np.ones(len(bipolar_vectors["a"]), dtype=np.float64)
        weighted = weighted_euclidean_distance(
            bipolar_vectors["a"], bipolar_vectors["similar"], weights
        )
        regular = euclidean_distance(bipolar_vectors["a"], bipolar_vectors["similar"])
        assert weighted == pytest.approx(regular, abs=1e-6)

    def test_zero_weights_exclude_dimensions(self):
        """Zero weights should exclude dimensions from distance."""
        v1 = np.array([0, 0, 0], dtype=np.float64)
        v2 = np.array([3, 4, 100], dtype=np.float64)
        # Regular: sqrt(9+16+10000) = sqrt(10025) ≈ 100.12

        # Zero out the big difference
        weights = np.array([1, 1, 0], dtype=np.float64)
        weighted = weighted_euclidean_distance(v1, v2, weights)
        # Now: sqrt(9+16) = 5.0
        assert weighted == pytest.approx(5.0, abs=1e-6)


# =============================================================================
# DistanceEngine Tests
# =============================================================================


class TestDistanceEngine:
    """Tests for the unified DistanceEngine class."""

    def test_default_metric(self):
        """Engine should use default metric when none specified."""
        engine = DistanceEngine(default_metric=DistanceMetric.COSINE)
        v1 = np.array([1, 1, 1], dtype=np.float64)
        v2 = np.array([1, 1, 1], dtype=np.float64)
        sim = engine.similarity(v1, v2)
        assert sim == pytest.approx(1.0, abs=1e-6)

    def test_explicit_metric(self):
        """Engine should use explicitly specified metric."""
        engine = DistanceEngine()
        v1 = np.array([1, 1, -1, -1], dtype=np.int8)
        v2 = np.array([1, -1, -1, 1], dtype=np.int8)

        # Hamming: 2 differences out of 4 → similarity = 0.5
        sim = engine.similarity(v1, v2, DistanceMetric.HAMMING)
        assert sim == 0.5

    def test_all_metrics_work(self, bipolar_vectors):
        """All metrics should run without error."""
        engine = DistanceEngine()
        metrics = [
            DistanceMetric.COSINE,
            DistanceMetric.DOT_PRODUCT,
            DistanceMetric.EUCLIDEAN,
            DistanceMetric.MANHATTAN,
            DistanceMetric.HAMMING,
            DistanceMetric.OVERLAP,
            DistanceMetric.AGREEMENT,
            DistanceMetric.CHEBYSHEV,
        ]
        for metric in metrics:
            sim = engine.similarity(
                bipolar_vectors["a"], bipolar_vectors["similar"], metric
            )
            assert isinstance(sim, float)

    def test_distance_method(self, bipolar_vectors):
        """Distance method should work correctly."""
        engine = DistanceEngine()
        dist = engine.distance(
            bipolar_vectors["a"],
            bipolar_vectors["identical"],
            DistanceMetric.EUCLIDEAN,
        )
        assert dist == 0.0

    def test_weighted_requires_weights(self, bipolar_vectors):
        """Weighted metrics should require weights argument."""
        engine = DistanceEngine()
        with pytest.raises(ValueError, match="requires 'weights'"):
            engine.similarity(
                bipolar_vectors["a"],
                bipolar_vectors["similar"],
                DistanceMetric.WEIGHTED_COSINE,
            )

    def test_weighted_with_weights(self, bipolar_vectors):
        """Weighted metrics should work with weights."""
        engine = DistanceEngine()
        weights = np.ones(len(bipolar_vectors["a"]), dtype=np.float64)
        sim = engine.similarity(
            bipolar_vectors["a"],
            bipolar_vectors["similar"],
            DistanceMetric.WEIGHTED_COSINE,
            weights=weights,
        )
        assert isinstance(sim, float)

    def test_is_qdrant_native(self):
        """Should correctly identify Qdrant-native metrics."""
        engine = DistanceEngine()
        assert engine.is_qdrant_native(DistanceMetric.COSINE) is True
        assert engine.is_qdrant_native(DistanceMetric.DOT_PRODUCT) is True
        assert engine.is_qdrant_native(DistanceMetric.EUCLIDEAN) is True
        assert engine.is_qdrant_native(DistanceMetric.MANHATTAN) is True
        assert engine.is_qdrant_native(DistanceMetric.HAMMING) is False
        assert engine.is_qdrant_native(DistanceMetric.OVERLAP) is False

    def test_get_qdrant_distance(self):
        """Should return correct Qdrant distance names."""
        engine = DistanceEngine()
        assert engine.get_qdrant_distance(DistanceMetric.COSINE) == "Cosine"
        assert engine.get_qdrant_distance(DistanceMetric.DOT_PRODUCT) == "Dot"
        assert engine.get_qdrant_distance(DistanceMetric.EUCLIDEAN) == "Euclid"
        assert engine.get_qdrant_distance(DistanceMetric.HAMMING) is None

    def test_find_nearest(self, bipolar_vectors):
        """find_nearest should return correct ordering."""
        engine = DistanceEngine()
        candidates = [
            ("identical", bipolar_vectors["identical"]),
            ("similar", bipolar_vectors["similar"]),
            ("opposite", bipolar_vectors["opposite"]),
        ]
        results = engine.find_nearest(
            bipolar_vectors["a"], candidates, k=3, metric=DistanceMetric.COSINE
        )

        # Should be ordered: identical > similar > opposite
        assert results[0][0] == "identical"
        assert results[1][0] == "similar"
        assert results[2][0] == "opposite"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_get_recommended_metric(self):
        """Should return appropriate metrics for use cases."""
        assert get_recommended_metric("semantic") == DistanceMetric.COSINE
        assert get_recommended_metric("bipolar") == DistanceMetric.HAMMING
        assert get_recommended_metric("normalized") == DistanceMetric.DOT_PRODUCT
        assert get_recommended_metric("geometric") == DistanceMetric.EUCLIDEAN
        assert get_recommended_metric("outlier") == DistanceMetric.CHEBYSHEV
        assert get_recommended_metric("weighted") == DistanceMetric.WEIGHTED_COSINE
        # Unknown should default to COSINE
        assert get_recommended_metric("unknown") == DistanceMetric.COSINE

    def test_compare_metrics(self, bipolar_vectors):
        """compare_metrics should return dict with all metrics."""
        results = compare_metrics(bipolar_vectors["a"], bipolar_vectors["similar"])
        assert isinstance(results, dict)
        assert "cosine" in results
        assert "hamming" in results
        assert "agreement" in results
        # All values should be floats
        for key, value in results.items():
            if not isinstance(value, str):  # Skip error messages
                assert isinstance(value, float)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_dimension(self):
        """Should work with single-dimension vectors."""
        v1 = np.array([1])
        v2 = np.array([-1])
        assert cosine_similarity(v1, v2) == -1.0
        assert hamming_distance(v1, v2) == 1

    def test_large_dimension(self):
        """Should work with high-dimensional vectors."""
        np.random.seed(42)
        D = 16000
        v1 = np.random.choice([-1, 1], size=D).astype(np.int8)
        v2 = np.random.choice([-1, 1], size=D).astype(np.int8)

        sim = cosine_similarity(v1, v2)
        assert isinstance(sim, float)
        assert -1.0 <= sim <= 1.0

    def test_all_zeros_hamming(self):
        """Hamming with zero vectors should be zero (all match)."""
        v1 = np.zeros(100, dtype=np.int8)
        v2 = np.zeros(100, dtype=np.int8)
        dist = hamming_distance(v1, v2)
        assert dist == 0

    def test_mixed_dtypes(self):
        """Should handle mixed dtypes."""
        v1 = np.array([1, 2, 3], dtype=np.int32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        sim = cosine_similarity(v1, v2)
        assert sim == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Correlation Tests (Cosine vs Agreement)
# =============================================================================


class TestMetricCorrelation:
    """Tests verifying known correlations between metrics."""

    def test_cosine_agreement_correlation(self):
        """Cosine and Agreement should be perfectly correlated for bipolar vectors."""
        np.random.seed(42)
        D = 1000

        correlations = []
        for _ in range(10):
            v1 = np.random.choice([-1, 1], size=D).astype(np.int8)
            v2 = np.random.choice([-1, 1], size=D).astype(np.int8)

            cos = cosine_similarity(v1, v2)
            agr = agreement_similarity(v1, v2)
            correlations.append((cos, agr))

        # For pure bipolar (no zeros), cosine == agreement
        for cos, agr in correlations:
            assert cos == pytest.approx(agr, abs=1e-6)
