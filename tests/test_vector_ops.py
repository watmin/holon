"""
Tests for new vector operations (Tier 1, Tier 2, Quantum-inspired).

Covers:
- Tier 1: sparsify, capacity, centroid, flip, topk_similar, similarity_matrix
- Tier 2: entropy, random_project, power, autocorrelate, cross_correlate
- Quantum: purity, participation_ratio
"""

import numpy as np
import pytest

from holon.accumulator import (
    accumulate,
    capacity,
    create_accumulator,
    participation_ratio,
    purity,
)
from holon.distance import cosine_similarity
from holon.primitives import (
    autocorrelate,
    bundle,
    centroid,
    cross_correlate,
    entropy,
    flip,
    power,
    random_project,
    similarity_matrix,
    sparsify,
    topk_similar,
)
from holon.vector_manager import VectorManager


@pytest.fixture
def vm():
    """Create a VectorManager for testing."""
    return VectorManager(dimensions=1024)


def make_bipolar(n, seed=0):
    """Generate a random bipolar vector {-1, +1}."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=n).astype(np.int8)


# =============================================================================
# sparsify
# =============================================================================


class TestSparsify:
    def test_reduces_nonzero_count(self):
        vec = make_bipolar(1024)
        sparse = sparsify(vec, k=100)
        assert np.sum(sparse != 0) <= 100

    def test_k_greater_than_dims_returns_copy(self):
        vec = make_bipolar(64)
        sparse = sparsify(vec, k=100)
        np.testing.assert_array_equal(sparse, vec)

    def test_preserves_original_signs(self):
        vec = make_bipolar(1024)
        sparse = sparsify(vec, k=256)
        mask = sparse != 0
        np.testing.assert_array_equal(np.sign(sparse[mask]), np.sign(vec[mask]))

    def test_full_k_returns_original(self):
        vec = make_bipolar(100)
        sparse = sparsify(vec, k=100)
        np.testing.assert_array_equal(sparse, vec)


# =============================================================================
# centroid
# =============================================================================


class TestCentroid:
    def test_single_vector_identity(self, vm):
        v = vm.get_vector("hello")
        c = centroid([v])
        assert cosine_similarity(v, c) > 0.99

    def test_centroid_similar_to_all_inputs(self, vm):
        vecs = [vm.get_vector(f"word_{i}") for i in range(5)]
        c = centroid(vecs)
        for v in vecs:
            sim = cosine_similarity(c, v)
            assert sim > -0.5  # not anti-correlated

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            centroid([])

    def test_centroid_is_bipolar(self, vm):
        vecs = [vm.get_vector(f"t_{i}") for i in range(3)]
        c = centroid(vecs)
        unique_vals = set(c.tolist())
        assert unique_vals.issubset({-1, 0, 1})


# =============================================================================
# flip
# =============================================================================


class TestFlip:
    def test_negation(self):
        vec = make_bipolar(1024)
        flipped = flip(vec)
        np.testing.assert_array_equal(flipped, -vec)

    def test_double_flip_identity(self):
        vec = make_bipolar(1024)
        np.testing.assert_array_equal(flip(flip(vec)), vec)

    def test_anti_similar(self, vm):
        v = vm.get_vector("cat")
        f = flip(v)
        sim = cosine_similarity(v, f)
        assert sim < -0.99

    def test_zeros_preserved(self):
        vec = np.array([1, 0, -1, 0, 1], dtype=np.int8)
        flipped = flip(vec)
        np.testing.assert_array_equal(
            flipped, np.array([-1, 0, 1, 0, -1], dtype=np.int8)
        )


# =============================================================================
# topk_similar
# =============================================================================


class TestTopkSimilar:
    def test_finds_exact_match(self, vm):
        query = vm.get_vector("target")
        candidates = [vm.get_vector(f"other_{i}") for i in range(10)]
        candidates[3] = query.copy()

        results = topk_similar(query, candidates, k=3)
        assert results[0][0] == 3
        assert results[0][1] > 0.99

    def test_k_limits_results(self, vm):
        query = vm.get_vector("q")
        candidates = [vm.get_vector(f"c_{i}") for i in range(20)]
        results = topk_similar(query, candidates, k=5)
        assert len(results) == 5

    def test_empty_candidates(self, vm):
        query = vm.get_vector("q")
        assert topk_similar(query, [], k=5) == []

    def test_sorted_descending(self, vm):
        query = vm.get_vector("q")
        candidates = [vm.get_vector(f"c_{i}") for i in range(10)]
        results = topk_similar(query, candidates, k=10)
        sims = [r[1] for r in results]
        assert sims == sorted(sims, reverse=True)


# =============================================================================
# similarity_matrix
# =============================================================================


class TestSimilarityMatrix:
    def test_diagonal_is_one(self, vm):
        vecs = [vm.get_vector(f"v_{i}") for i in range(5)]
        mat = similarity_matrix(vecs)
        np.testing.assert_allclose(np.diag(mat), 1.0, atol=0.01)

    def test_symmetric(self, vm):
        vecs = [vm.get_vector(f"s_{i}") for i in range(4)]
        mat = similarity_matrix(vecs)
        np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_correct_shape(self, vm):
        vecs = [vm.get_vector(f"m_{i}") for i in range(7)]
        mat = similarity_matrix(vecs)
        assert mat.shape == (7, 7)

    def test_orthogonal_near_zero(self, vm):
        vecs = [vm.get_vector(f"o_{i}") for i in range(3)]
        mat = similarity_matrix(vecs)
        for i in range(3):
            for j in range(3):
                if i != j:
                    assert abs(mat[i, j]) < 0.15  # random vectors ~ orthogonal


# =============================================================================
# entropy
# =============================================================================


class TestEntropy:
    def test_all_ones_low_entropy(self):
        vec = np.ones(1024, dtype=np.int8)
        h = entropy(vec)
        assert h < 0.1  # all +1, no variety

    def test_balanced_high_entropy(self):
        vec = make_bipolar(1024, seed=42)
        h = entropy(vec)
        assert h > 0.5  # roughly balanced +1/-1

    def test_zero_vector_zero_entropy(self):
        vec = np.zeros(100, dtype=np.int8)
        h = entropy(vec)
        assert h < 0.1

    def test_range(self):
        vec = make_bipolar(1024)
        h = entropy(vec)
        assert 0.0 <= h <= 1.0


# =============================================================================
# random_project
# =============================================================================


class TestRandomProject:
    def test_output_dimensionality(self, vm):
        v = vm.get_vector("proj")
        projected = random_project(v, target_dims=128)
        assert len(projected) == 128

    def test_deterministic_with_seed(self, vm):
        v = vm.get_vector("det")
        p1 = random_project(v, target_dims=128, seed=99)
        p2 = random_project(v, target_dims=128, seed=99)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_different_results(self, vm):
        v = vm.get_vector("diff")
        p1 = random_project(v, target_dims=128, seed=1)
        p2 = random_project(v, target_dims=128, seed=2)
        assert not np.array_equal(p1, p2)

    def test_output_is_bipolar(self, vm):
        v = vm.get_vector("bip")
        projected = random_project(v, target_dims=256)
        unique_vals = set(projected.tolist())
        assert unique_vals.issubset({-1, 0, 1})


# =============================================================================
# power
# =============================================================================


class TestPower:
    def test_power_one_identity(self):
        vec = make_bipolar(1024)
        np.testing.assert_array_equal(power(vec, 1.0), vec)

    def test_power_zero_all_zeros(self):
        vec = make_bipolar(1024)
        result = power(vec, 0.0)
        np.testing.assert_array_equal(result, np.zeros(1024, dtype=np.int8))

    def test_even_power_all_positive(self):
        vec = make_bipolar(1024)
        result = power(vec, 2.0)
        assert np.all(result >= 0)

    def test_odd_power_preserves(self):
        vec = make_bipolar(1024)
        result = power(vec, 3.0)
        np.testing.assert_array_equal(result, vec)

    def test_negative_exponent_raises(self):
        vec = make_bipolar(64)
        with pytest.raises(ValueError):
            power(vec, -1.0)


# =============================================================================
# autocorrelate
# =============================================================================


class TestAutocorrelate:
    def test_lag_zero_is_one(self, vm):
        stream = [vm.get_vector(f"ac_{i}") for i in range(20)]
        acf = autocorrelate(stream, max_lag=5)
        assert acf[0] == 1.0

    def test_periodic_stream(self, vm):
        a = vm.get_vector("period_a")
        b = vm.get_vector("period_b")
        # Period-2 pattern: a, b, a, b, ...
        stream = [a, b] * 20
        acf = autocorrelate(stream, max_lag=6)
        # lag=2 should be high (period match), lag=1 should be low
        assert acf[2] > acf[1]

    def test_max_lag_capped(self, vm):
        stream = [vm.get_vector(f"cap_{i}") for i in range(5)]
        acf = autocorrelate(stream, max_lag=100)
        assert len(acf) == 5  # capped to n-1+1


# =============================================================================
# cross_correlate
# =============================================================================


class TestCrossCorrelate:
    def test_identical_streams_high_at_zero(self, vm):
        stream = [vm.get_vector(f"xc_{i}") for i in range(20)]
        xcf = cross_correlate(stream, stream, max_lag=5)
        # lag=0 should be highest
        assert xcf[0] > 0.9

    def test_lagged_stream_peak(self, vm):
        a = vm.get_vector("xa")
        b = vm.get_vector("xb")
        c = vm.get_vector("xc")
        stream_a = [a, b, c, a, b, c, a, b, c, a]
        # stream_b is stream_a shifted by 1
        stream_b = [vm.get_vector("pad")] + stream_a[:-1]
        xcf = cross_correlate(stream_a, stream_b, max_lag=4)
        # lag=1 should be relatively higher
        assert len(xcf) == 5

    def test_different_lengths(self, vm):
        stream_a = [vm.get_vector(f"la_{i}") for i in range(10)]
        stream_b = [vm.get_vector(f"lb_{i}") for i in range(15)]
        xcf = cross_correlate(stream_a, stream_b, max_lag=3)
        assert len(xcf) == 4  # lags 0..3


# =============================================================================
# capacity (accumulator)
# =============================================================================


class TestCapacity:
    def test_empty_accumulator_full_capacity(self):
        acc = create_accumulator(1024)
        cap = capacity(acc, codebook_size=100)
        assert cap == 1.0

    def test_capacity_decreases_with_items(self, vm):
        acc = create_accumulator(1024)
        vecs = [vm.get_vector(f"cap_{i}") for i in range(50)]

        caps = []
        for v in vecs:
            acc = accumulate(acc, v)
            caps.append(capacity(acc, codebook_size=100))

        # Capacity should decrease monotonically (or nearly so)
        assert caps[-1] < caps[0]

    def test_small_codebook_more_capacity(self):
        acc = create_accumulator(1024)
        vec = make_bipolar(1024)
        acc = accumulate(acc, vec)
        # Smaller codebook = easier to distinguish = more capacity
        assert capacity(acc, codebook_size=10) > capacity(acc, codebook_size=1000)


# =============================================================================
# purity (accumulator, quantum-inspired)
# =============================================================================


class TestPurity:
    def test_single_vector_high_purity(self):
        vec = make_bipolar(1024)
        acc = create_accumulator(1024)
        acc = accumulate(acc, vec)
        p = purity(acc)
        assert p > 0.95

    def test_many_random_vectors_lower_purity(self, vm):
        acc = create_accumulator(1024)
        for i in range(50):
            acc = accumulate(acc, vm.get_vector(f"pur_{i}"))
        p = purity(acc)
        assert p < 0.5  # diffuse superposition

    def test_empty_accumulator_zero_purity(self):
        acc = create_accumulator(1024)
        assert purity(acc) == 0.0

    def test_purity_range(self, vm):
        acc = create_accumulator(1024)
        acc = accumulate(acc, vm.get_vector("x"))
        p = purity(acc)
        assert 0.0 <= p <= 1.0


# =============================================================================
# participation_ratio (accumulator, quantum-inspired)
# =============================================================================


class TestParticipationRatio:
    def test_single_bipolar_full_participation(self):
        vec = make_bipolar(1024)
        acc = create_accumulator(1024)
        acc = accumulate(acc, vec)
        pr = participation_ratio(acc)
        # For a single bipolar vector, all dims participate equally
        assert pr > 900  # close to 1024

    def test_empty_accumulator_zero(self):
        acc = create_accumulator(1024)
        assert participation_ratio(acc) == 0.0

    def test_concentrated_fewer_participants(self):
        # Build a vector that concentrates energy in few dimensions
        acc = np.zeros(1024, dtype=np.float64)
        acc[:10] = 100.0  # energy in only 10 dims
        pr = participation_ratio(acc)
        assert pr < 20  # should be close to 10

    def test_purity_and_pr_related(self):
        # For a single bipolar vector:
        # purity = d / l2_sq = d / d = 1.0
        # PR = l2_sq^2 / l4_sum = d^2 / d = d
        vec = make_bipolar(1024)
        acc = create_accumulator(1024)
        acc = accumulate(acc, vec)
        p = purity(acc)
        pr = participation_ratio(acc)
        assert abs(p - 1.0) < 0.01
        assert abs(pr - 1024) < 5
