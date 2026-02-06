"""
Tests for continuous scalar encoding methods.

These methods encode continuous values where nearby values have similar vectors,
unlike structured data encoding where {"x": 5} is unrelated to {"x": 6}.
"""

import numpy as np
import pytest

from holon import CPUStore


class TestEncodeScalarLinear:
    """Tests for encode_scalar with mode='linear'."""

    @pytest.fixture
    def store(self):
        return CPUStore(dimensions=4096)

    def test_returns_bipolar_vector(self, store):
        """Result should be a bipolar vector."""
        vec = store.encode_scalar(100, mode="linear")
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.int8
        assert set(np.unique(vec)).issubset({-1, 0, 1})

    def test_correct_dimensions(self, store):
        """Vector should match store dimensions."""
        vec = store.encode_scalar(100, mode="linear")
        assert len(vec) == store.dimensions

    def test_nearby_values_similar(self, store):
        """Nearby values should have high similarity."""
        v100 = store.encode_scalar(100, mode="linear")
        v110 = store.encode_scalar(110, mode="linear")
        v200 = store.encode_scalar(200, mode="linear")

        sim_near = store.similarity(v100, v110)
        sim_far = store.similarity(v100, v200)

        assert sim_near > sim_far, "Nearby values should be more similar"
        assert sim_near > 0.5, "100 and 110 should have decent similarity"

    def test_identical_values_identical_vectors(self, store):
        """Same value should produce same vector."""
        v1 = store.encode_scalar(42.5, mode="linear")
        v2 = store.encode_scalar(42.5, mode="linear")

        assert np.array_equal(v1, v2)
        assert store.similarity(v1, v2) == 1.0

    def test_scale_affects_similarity_decay(self, store):
        """Larger scale should mean slower similarity decay."""
        v0_small = store.encode_scalar(0, mode="linear", scale=100)
        v100_small = store.encode_scalar(100, mode="linear", scale=100)

        v0_large = store.encode_scalar(0, mode="linear", scale=100000)
        v100_large = store.encode_scalar(100, mode="linear", scale=100000)

        sim_small = store.similarity(v0_small, v100_small)
        sim_large = store.similarity(v0_large, v100_large)

        assert (
            sim_large > sim_small
        ), "Larger scale should have higher similarity for same difference"


class TestEncodeScalarCircular:
    """Tests for encode_scalar with mode='circular'."""

    @pytest.fixture
    def store(self):
        return CPUStore(dimensions=4096)

    def test_returns_bipolar_vector(self, store):
        """Result should be a bipolar vector."""
        vec = store.encode_scalar(12, mode="circular", period=24)
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.int8

    def test_requires_period(self, store):
        """Circular mode should require period parameter."""
        with pytest.raises(ValueError, match="period is required"):
            store.encode_scalar(12, mode="circular")

    def test_wrapping_similarity(self, store):
        """Values near period boundary should be similar to values near 0."""
        h0 = store.encode_scalar(0, mode="circular", period=24)
        h1 = store.encode_scalar(1, mode="circular", period=24)
        h23 = store.encode_scalar(23, mode="circular", period=24)
        h12 = store.encode_scalar(12, mode="circular", period=24)

        sim_0_1 = store.similarity(h0, h1)
        sim_0_23 = store.similarity(h0, h23)
        sim_0_12 = store.similarity(h0, h12)

        # 0 and 1 are adjacent
        assert sim_0_1 > 0.7, "Adjacent values should be similar"

        # 0 and 23 are also adjacent (wrapping)
        assert sim_0_23 > 0.7, "Values should wrap around period"

        # 0 and 12 are opposite
        assert sim_0_12 < sim_0_1, "Opposite values should be less similar"
        assert (
            sim_0_12 < sim_0_23
        ), "Opposite values should be less similar than adjacent"

    def test_period_equivalence(self, store):
        """Value 0 should equal value at period."""
        v0 = store.encode_scalar(0, mode="circular", period=24)
        v24 = store.encode_scalar(24, mode="circular", period=24)

        sim = store.similarity(v0, v24)
        assert sim > 0.99, "0 and period should be nearly identical"


class TestEncodeScalarLog:
    """Tests for encode_scalar_log (logarithmic scaling)."""

    @pytest.fixture
    def store(self):
        return CPUStore(dimensions=4096)

    def test_returns_bipolar_vector(self, store):
        """Result should be a bipolar vector."""
        vec = store.encode_scalar_log(100)
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.int8

    def test_equal_ratios_equal_similarity(self, store):
        """Equal ratios should have approximately equal similarity."""
        v10 = store.encode_scalar_log(10)
        v100 = store.encode_scalar_log(100)
        v1000 = store.encode_scalar_log(1000)
        v10000 = store.encode_scalar_log(10000)

        sim_10_100 = store.similarity(v10, v100)  # 10x ratio
        sim_100_1000 = store.similarity(v100, v1000)  # 10x ratio
        sim_1000_10000 = store.similarity(v1000, v10000)  # 10x ratio

        # All 10x ratios should have similar similarity
        assert (
            abs(sim_10_100 - sim_100_1000) < 0.1
        ), "Equal ratios should have similar similarity"
        assert (
            abs(sim_100_1000 - sim_1000_10000) < 0.1
        ), "Equal ratios should have similar similarity"

    def test_larger_ratio_lower_similarity(self, store):
        """Larger ratios should have lower similarity."""
        v100 = store.encode_scalar_log(100)
        v1000 = store.encode_scalar_log(1000)  # 10x ratio
        v100000 = store.encode_scalar_log(100000)  # 1000x ratio

        sim_10x = store.similarity(v100, v1000)
        sim_1000x = store.similarity(v100, v100000)

        assert sim_10x > sim_1000x, "Larger ratio should have lower similarity"

    def test_handles_small_values(self, store):
        """Should handle values close to zero."""
        v_small = store.encode_scalar_log(0.001)
        v_one = store.encode_scalar_log(1)

        assert isinstance(v_small, np.ndarray)
        assert len(v_small) == store.dimensions

    def test_handles_zero(self, store):
        """Should handle zero gracefully."""
        v_zero = store.encode_scalar_log(0)
        assert isinstance(v_zero, np.ndarray)


class TestEncodeScalarViaEncoder:
    """Test that encoder methods work directly."""

    @pytest.fixture
    def store(self):
        return CPUStore(dimensions=4096)

    def test_encoder_encode_scalar(self, store):
        """encoder.encode_scalar should work."""
        vec = store.encoder.encode_scalar(100, mode="linear")
        assert isinstance(vec, np.ndarray)
        assert len(vec) == store.dimensions

    def test_encoder_encode_scalar_log(self, store):
        """encoder.encode_scalar_log should work."""
        vec = store.encoder.encode_scalar_log(100)
        assert isinstance(vec, np.ndarray)
        assert len(vec) == store.dimensions

    def test_store_and_encoder_produce_same_result(self, store):
        """Store and encoder methods should produce identical results."""
        store_vec = store.encode_scalar(100, mode="linear")
        encoder_vec = store.encoder.encode_scalar(100, mode="linear")

        assert np.array_equal(store_vec, encoder_vec)


class TestEncodeScalarInvalidInput:
    """Test error handling."""

    @pytest.fixture
    def store(self):
        return CPUStore(dimensions=4096)

    def test_invalid_mode(self, store):
        """Invalid mode should raise error."""
        with pytest.raises(ValueError, match="Unknown mode"):
            store.encode_scalar(100, mode="invalid")

    def test_circular_without_period(self, store):
        """Circular mode without period should raise error."""
        with pytest.raises(ValueError, match="period is required"):
            store.encode_scalar(100, mode="circular")
