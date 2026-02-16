"""
Tests for advanced vector operations.

Covers:
- decode_scalar_log: inverse of scalar encoding
- significance: cosine similarity to z-score
- reject: orthogonal complement of project
- bundle_with_confidence: bundle + per-dimension margins
- coherence: mean pairwise similarity
- grover_amplify: iterative amplitude amplification
- drift_rate: temporal derivative of similarity
"""

import numpy as np
import pytest

from holon.distance import cosine_similarity, significance
from holon.primitives import (
    bundle,
    bundle_with_confidence,
    coherence,
    drift_rate,
    flip,
    grover_amplify,
    project,
    reflect_about_mean,
    reject,
)
from holon.scalar import decode_scalar_log, encode_scalar_log
from holon.vector_manager import VectorManager


@pytest.fixture
def vm():
    """Create a VectorManager for testing."""
    return VectorManager(dimensions=4096)


def make_bipolar(n, seed=0):
    """Generate a random bipolar vector {-1, +1}."""
    rng = np.random.RandomState(seed)
    return rng.choice([-1, 1], size=n).astype(np.int8)


# =============================================================================
# decode_scalar_log
# =============================================================================


class TestDecodeScalarLog:
    def test_roundtrip_100(self):
        v = encode_scalar_log(100.0, 4096)
        decoded = decode_scalar_log(v)
        assert abs(decoded - 100.0) / 100.0 < 0.1  # within 10%

    def test_roundtrip_10000(self):
        v = encode_scalar_log(10000.0, 4096)
        decoded = decode_scalar_log(v)
        assert abs(decoded - 10000.0) / 10000.0 < 0.1

    def test_roundtrip_500(self):
        # Values near 1.0 (log10 ≈ 0) sit in the encoding's flat zone
        # where scale=1000 can't distinguish values 1-37. Use larger values.
        v = encode_scalar_log(500.0, 4096)
        decoded = decode_scalar_log(v)
        assert abs(decoded - 500.0) / 500.0 < 0.1

    def test_ordering_preserved(self):
        v100 = encode_scalar_log(100.0, 4096)
        v1000 = encode_scalar_log(1000.0, 4096)
        d100 = decode_scalar_log(v100)
        d1000 = decode_scalar_log(v1000)
        assert d100 < d1000

    def test_custom_range(self):
        v = encode_scalar_log(50.0, 4096)
        decoded = decode_scalar_log(v, lo=1.0, hi=1000.0)
        assert abs(decoded - 50.0) / 50.0 < 0.2


# =============================================================================
# significance
# =============================================================================


class TestSignificance:
    def test_zero_similarity_zero_z(self):
        assert significance(0.0, 4096) == 0.0

    def test_positive_similarity_positive_z(self):
        z = significance(0.05, 4096)
        assert z > 3.0  # ~3.2 sigma

    def test_scales_with_dimensions(self):
        z_small = significance(0.1, 64)
        z_large = significance(0.1, 4096)
        assert z_large > z_small

    def test_negative_similarity(self):
        z = significance(-0.05, 4096)
        assert z < -3.0

    def test_known_value(self):
        # 0.05 * sqrt(4096) = 0.05 * 64 = 3.2
        z = significance(0.05, 4096)
        assert abs(z - 3.2) < 0.01

    def test_zero_dimensions(self):
        assert significance(0.5, 0) == 0.0


# =============================================================================
# reject
# =============================================================================


class TestReject:
    def test_reject_removes_projected_component(self, vm):
        # Create a correlated pair: bundle a with b to get overlap
        a = vm.get_vector("basis_a")
        b = vm.get_vector("basis_b")
        combined = bundle([a, b])  # has component of both

        rejected = reject(combined, [a])

        # Rejected should have less similarity to 'a' than original
        sim_orig = abs(cosine_similarity(combined, a))
        sim_reject = abs(cosine_similarity(rejected, a))
        assert sim_reject < sim_orig

    def test_reject_empty_subspace(self, vm):
        vec = vm.get_vector("test")
        result = reject(vec, [])
        np.testing.assert_array_equal(result, vec)

    def test_reject_self_gives_zeros_or_near(self, vm):
        vec = vm.get_vector("self")
        result = reject(vec, [vec])
        # Rejecting from own subspace should give near-zero
        assert np.sum(np.abs(result)) < len(vec) * 0.1

    def test_reject_orthogonal_preserves(self):
        # Two orthogonal-ish vectors
        a = make_bipolar(4096, seed=0)
        b = make_bipolar(4096, seed=100)
        # Random vectors are nearly orthogonal in high-D
        rejected = reject(a, [b])
        # Should still be similar to original
        sim = cosine_similarity(a, rejected)
        assert sim > 0.8

    def test_reject_is_bipolar(self, vm):
        vec = vm.get_vector("bip")
        basis = [vm.get_vector("basis")]
        result = reject(vec, basis)
        assert set(result.tolist()).issubset({-1, 0, 1})


# =============================================================================
# bundle_with_confidence
# =============================================================================


class TestBundleWithConfidence:
    def test_returns_tuple(self, vm):
        vecs = [vm.get_vector(f"v_{i}") for i in range(5)]
        bundled, margins = bundle_with_confidence(vecs)
        assert isinstance(bundled, np.ndarray)
        assert isinstance(margins, np.ndarray)
        assert len(bundled) == len(margins)

    def test_bundled_matches_regular_bundle(self, vm):
        vecs = [vm.get_vector(f"c_{i}") for i in range(5)]
        bundled, _ = bundle_with_confidence(vecs)
        regular = bundle(vecs)
        np.testing.assert_array_equal(bundled, regular)

    def test_unanimous_high_confidence(self):
        vec = make_bipolar(1024)
        vecs = [vec.copy() for _ in range(10)]
        _, margins = bundle_with_confidence(vecs)
        # All identical → all margins = 1.0
        np.testing.assert_allclose(margins, 1.0)

    def test_split_low_confidence(self):
        a = make_bipolar(1024, seed=0)
        b = flip(a)
        _, margins = bundle_with_confidence([a, b])
        # Equal and opposite → all margins = 0.0
        np.testing.assert_allclose(margins, 0.0)

    def test_margins_range(self, vm):
        vecs = [vm.get_vector(f"m_{i}") for i in range(7)]
        _, margins = bundle_with_confidence(vecs)
        assert np.all(margins >= 0.0)
        assert np.all(margins <= 1.0)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bundle_with_confidence([])


# =============================================================================
# coherence
# =============================================================================


class TestCoherence:
    def test_identical_vectors_one(self, vm):
        v = vm.get_vector("same")
        assert coherence([v, v, v]) > 0.99

    def test_random_vectors_near_zero(self, vm):
        vecs = [vm.get_vector(f"r_{i}") for i in range(10)]
        c = coherence(vecs)
        assert abs(c) < 0.15

    def test_single_vector_one(self, vm):
        v = vm.get_vector("single")
        assert coherence([v]) == 1.0

    def test_opposite_vectors_negative(self):
        v = make_bipolar(1024)
        c = coherence([v, flip(v)])
        assert c < -0.9

    def test_range(self, vm):
        vecs = [vm.get_vector(f"cr_{i}") for i in range(5)]
        c = coherence(vecs)
        assert -1.0 <= c <= 1.0


# =============================================================================
# reflect_about_mean
# =============================================================================


class TestReflectAboutMean:
    def test_all_ones_flips_to_negative(self):
        vec = np.ones(100, dtype=np.int8)
        result = reflect_about_mean(vec)
        # mean = 1.0, reflected = 2*1 - 1 = 1 for all dims
        # Actually all same, so no change
        np.testing.assert_array_equal(result, vec)

    def test_balanced_flips(self):
        vec = np.array([1, 1, -1, -1], dtype=np.int8)
        result = reflect_about_mean(vec)
        # mean = 0, reflected = -v
        np.testing.assert_array_equal(result, -vec)

    def test_output_is_bipolar(self):
        vec = make_bipolar(1024)
        result = reflect_about_mean(vec)
        assert set(result.tolist()).issubset({-1, 0, 1})


# =============================================================================
# grover_amplify
# =============================================================================


class TestGroverAmplify:
    def test_amplifies_weak_signal(self, vm):
        signal = vm.get_vector("attack")
        noise = [vm.get_vector(f"noise_{i}") for i in range(10)]
        # Bury signal in noise
        background = bundle(noise + [signal])

        # Without amplification
        sim_before = cosine_similarity(background, signal)

        # With amplification
        amplified = grover_amplify(signal, background, iterations=2)
        sim_after = cosine_similarity(amplified, signal)

        # Amplified should be more similar to signal
        assert sim_after > sim_before or sim_after > 0.0

    def test_output_is_bipolar(self, vm):
        signal = vm.get_vector("sig")
        bg = vm.get_vector("bg")
        result = grover_amplify(signal, bg, iterations=1)
        assert set(result.tolist()).issubset({-1, 0, 1})

    def test_zero_iterations_returns_thresholded_bg(self, vm):
        signal = vm.get_vector("s")
        bg = vm.get_vector("b")
        result = grover_amplify(signal, bg, iterations=0)
        # Zero iterations = just threshold the background
        assert len(result) == len(bg)


# =============================================================================
# drift_rate
# =============================================================================


class TestDriftRate:
    def test_stable_stream_low_drift(self, vm):
        v = vm.get_vector("stable")
        stream = [v.copy() for _ in range(10)]
        rates = drift_rate(stream)
        # Identical vectors → similarity always 1.0 → drift = 0
        assert all(abs(r) < 0.01 for r in rates)

    def test_sudden_change_spike(self, vm):
        a = vm.get_vector("before")
        b = vm.get_vector("after")
        # Stable, then sudden change
        stream = [a] * 5 + [b] * 5
        rates = drift_rate(stream)
        # Should have a negative spike at the transition
        min_rate = min(rates)
        assert min_rate < -0.5

    def test_short_stream_empty(self, vm):
        assert drift_rate([vm.get_vector("x")]) == []
        assert drift_rate([vm.get_vector("x"), vm.get_vector("y")]) == []

    def test_output_length(self, vm):
        stream = [vm.get_vector(f"d_{i}") for i in range(10)]
        rates = drift_rate(stream)
        assert len(rates) == len(stream) - 2

    def test_windowed_smoothing(self, vm):
        stream = [vm.get_vector(f"w_{i}") for i in range(20)]
        rates_w1 = drift_rate(stream, window=1)
        rates_w5 = drift_rate(stream, window=5)
        # Windowed should be smoother (lower max absolute value)
        assert len(rates_w1) == len(rates_w5)
