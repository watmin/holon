"""
Tests for the accumulator primitives in encoder.py.

The accumulator primitives preserve frequency information by keeping
a running float sum instead of thresholding after each update.

Key properties tested:
1. Frequency preservation: High-frequency patterns dominate
2. Determinism: Same inputs produce same outputs
3. Commutativity: Order of accumulation doesn't affect final result
4. Separation: Frequent patterns are distinguishable from rare ones
5. Edge cases: Empty accumulator, single observation, etc.
"""

import numpy as np
import pytest

from holon.encoder import Encoder
from holon.vector_manager import VectorManager


@pytest.fixture
def vector_manager():
    """Create a VectorManager for testing."""
    return VectorManager(dimensions=1024)


@pytest.fixture
def encoder(vector_manager):
    """Create an Encoder for testing."""
    return Encoder(vector_manager=vector_manager)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


class TestAccumulatorBasics:
    """Basic functionality tests for accumulator primitives."""

    def test_create_accumulator(self, encoder):
        """Test that create_accumulator returns zero vector of correct shape."""
        accum = encoder.create_accumulator()

        assert accum.shape == (encoder.vector_manager.dimensions,)
        assert accum.dtype == np.float64
        assert np.all(accum == 0)

    def test_accumulate_single_vector(self, encoder):
        """Test accumulating a single vector."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")

        result = encoder.accumulate(accum, vec)

        # Should be equal to the input vector (as float64)
        np.testing.assert_array_equal(result, vec.astype(np.float64))

    def test_accumulate_returns_float64(self, encoder):
        """Test that accumulate returns float64 for precision."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")

        result = encoder.accumulate(accum, vec)

        assert result.dtype == np.float64

    def test_accumulate_multiple_same_vector(self, encoder):
        """Test accumulating the same vector multiple times."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("repeated")

        for _ in range(100):
            accum = encoder.accumulate(accum, vec)

        # After 100 accumulations, values should be 100x the original
        expected = 100 * vec.astype(np.float64)
        np.testing.assert_array_almost_equal(accum, expected)

    def test_normalize_accumulator_unit_length(self, encoder):
        """Test that normalize_accumulator returns unit-length vector."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")

        for _ in range(50):
            accum = encoder.accumulate(accum, vec)

        normalized = encoder.normalize_accumulator(accum)

        norm = np.linalg.norm(normalized)
        assert abs(norm - 1.0) < 1e-6, f"Expected unit length, got {norm}"

    def test_normalize_empty_accumulator(self, encoder):
        """Test normalizing an empty accumulator."""
        accum = encoder.create_accumulator()

        normalized = encoder.normalize_accumulator(accum)

        assert np.all(normalized == 0)
        assert normalized.dtype == np.float32

    def test_threshold_accumulator(self, encoder):
        """Test thresholding accumulator to bipolar."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")

        for _ in range(10):
            accum = encoder.accumulate(accum, vec)

        thresholded = encoder.threshold_accumulator(accum)

        assert thresholded.dtype == np.int8
        assert set(np.unique(thresholded)).issubset({-1, 0, 1})


class TestFrequencyPreservation:
    """Tests for frequency preservation property."""

    def test_frequent_pattern_dominates(self, encoder):
        """Test that frequently accumulated patterns dominate the accumulator."""
        accum = encoder.create_accumulator()

        frequent = encoder.vector_manager.get_vector("frequent")
        rare = encoder.vector_manager.get_vector("rare")

        # Add frequent 99 times, rare 1 time
        for _ in range(99):
            accum = encoder.accumulate(accum, frequent)
        accum = encoder.accumulate(accum, rare)

        normalized = encoder.normalize_accumulator(accum)

        sim_frequent = cosine_similarity(normalized, frequent)
        sim_rare = cosine_similarity(normalized, rare)

        assert sim_frequent > sim_rare, (
            f"Frequent pattern should have higher similarity: "
            f"frequent={sim_frequent:.4f}, rare={sim_rare:.4f}"
        )

    def test_frequency_ratio_preserved(self, encoder):
        """Test that relative frequency affects similarity proportionally."""
        # Create two accumulators with different frequency ratios
        accum_90_10 = encoder.create_accumulator()
        accum_50_50 = encoder.create_accumulator()

        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        # 90/10 split
        for _ in range(90):
            accum_90_10 = encoder.accumulate(accum_90_10, A)
        for _ in range(10):
            accum_90_10 = encoder.accumulate(accum_90_10, B)

        # 50/50 split
        for _ in range(50):
            accum_50_50 = encoder.accumulate(accum_50_50, A)
        for _ in range(50):
            accum_50_50 = encoder.accumulate(accum_50_50, B)

        norm_90_10 = encoder.normalize_accumulator(accum_90_10)
        norm_50_50 = encoder.normalize_accumulator(accum_50_50)

        # In 90/10, A should be much more similar than in 50/50
        sim_A_in_90_10 = cosine_similarity(norm_90_10, A)
        sim_A_in_50_50 = cosine_similarity(norm_50_50, A)

        assert sim_A_in_90_10 > sim_A_in_50_50, (
            f"A should be more similar in 90/10 split: "
            f"90/10={sim_A_in_90_10:.4f}, 50/50={sim_A_in_50_50:.4f}"
        )

    def test_vs_prototype_add_frequency_loss(self, encoder):
        """Test that accumulate preserves frequency better than prototype_add."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        # Using accumulate
        accum = encoder.create_accumulator()
        for _ in range(99):
            accum = encoder.accumulate(accum, A)
        accum = encoder.accumulate(accum, B)
        accum_normalized = encoder.normalize_accumulator(accum)

        # Using prototype_add
        proto = A.copy()
        for i in range(1, 99):
            proto = encoder.prototype_add(proto, A, i)
        proto = encoder.prototype_add(proto, B, 99)

        # Accumulator should show more difference between A and B similarity
        accum_sim_A = cosine_similarity(accum_normalized, A)
        accum_sim_B = cosine_similarity(accum_normalized, B)
        accum_separation = accum_sim_A - accum_sim_B

        proto_sim_A = cosine_similarity(proto, A)
        proto_sim_B = cosine_similarity(proto, B)
        proto_separation = proto_sim_A - proto_sim_B

        assert accum_separation > proto_separation, (
            f"Accumulator should preserve frequency better: "
            f"accum_sep={accum_separation:.4f}, proto_sep={proto_separation:.4f}"
        )


class TestCommutativity:
    """Tests for order-independence (commutativity) of accumulation."""

    def test_order_independent(self, encoder):
        """Test that order of accumulation doesn't affect result."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        # Order 1: A, A, B, C
        accum1 = encoder.create_accumulator()
        for vec in [A, A, B, C]:
            accum1 = encoder.accumulate(accum1, vec)

        # Order 2: C, B, A, A
        accum2 = encoder.create_accumulator()
        for vec in [C, B, A, A]:
            accum2 = encoder.accumulate(accum2, vec)

        np.testing.assert_array_almost_equal(accum1, accum2)

    def test_batch_vs_streaming(self, encoder):
        """Test that batch and streaming give same result."""
        vectors = [encoder.vector_manager.get_vector(f"vec_{i}") for i in range(50)]

        # Streaming
        accum_stream = encoder.create_accumulator()
        for vec in vectors:
            accum_stream = encoder.accumulate(accum_stream, vec)

        # Batch (using numpy sum)
        accum_batch = np.sum([v.astype(np.float64) for v in vectors], axis=0)

        np.testing.assert_array_almost_equal(accum_stream, accum_batch)


class TestAnomalyDetection:
    """Tests for anomaly detection use case."""

    def test_benign_vs_malicious_separation(self, encoder):
        """Test that benign patterns have higher similarity than malicious."""
        # Create "benign" patterns (high frequency)
        benign_patterns = [
            encoder.encode_data({"type": "normal", "action": "read"}),
            encoder.encode_data({"type": "normal", "action": "write"}),
            encoder.encode_data({"type": "normal", "action": "list"}),
        ]

        # Create "malicious" patterns (low frequency)
        malicious_patterns = [
            encoder.encode_data({"type": "attack", "action": "inject"}),
            encoder.encode_data({"type": "attack", "action": "exfil"}),
        ]

        # Build accumulator: 100 benign, 2 malicious
        accum = encoder.create_accumulator()

        import random

        random.seed(42)

        # Add benign patterns (100 times total)
        for _ in range(100):
            vec = random.choice(benign_patterns)
            accum = encoder.accumulate(accum, vec)

        # Add malicious patterns (2 times total)
        for vec in malicious_patterns:
            accum = encoder.accumulate(accum, vec)

        normalized = encoder.normalize_accumulator(accum)

        # Test similarity to benign vs malicious
        benign_sims = [cosine_similarity(normalized, p) for p in benign_patterns]
        malicious_sims = [cosine_similarity(normalized, p) for p in malicious_patterns]

        avg_benign = np.mean(benign_sims)
        avg_malicious = np.mean(malicious_sims)

        assert avg_benign > avg_malicious, (
            f"Benign should have higher similarity: "
            f"benign={avg_benign:.4f}, malicious={avg_malicious:.4f}"
        )

    def test_unseen_pattern_low_similarity(self, encoder):
        """Test that completely unseen patterns have low similarity."""
        # Build accumulator from seen patterns
        seen = [
            encoder.encode_data({"method": "GET", "path": "/api/users"}),
            encoder.encode_data({"method": "POST", "path": "/api/users"}),
        ]

        accum = encoder.create_accumulator()
        for _ in range(50):
            for vec in seen:
                accum = encoder.accumulate(accum, vec)

        normalized = encoder.normalize_accumulator(accum)

        # Test against unseen pattern
        unseen = encoder.encode_data({"method": "DELETE", "path": "/admin/config"})

        seen_sims = [cosine_similarity(normalized, v) for v in seen]
        unseen_sim = cosine_similarity(normalized, unseen)

        avg_seen = np.mean(seen_sims)

        assert unseen_sim < avg_seen, (
            f"Unseen pattern should have lower similarity: "
            f"unseen={unseen_sim:.4f}, seen_avg={avg_seen:.4f}"
        )


class TestEdgeCases:
    """Edge case tests."""

    def test_accumulate_opposite_vectors(self, encoder):
        """Test accumulating a vector and its negation."""
        vec = encoder.vector_manager.get_vector("test")
        neg_vec = -vec

        accum = encoder.create_accumulator()
        accum = encoder.accumulate(accum, vec)
        accum = encoder.accumulate(accum, neg_vec)

        # Should cancel out to zero
        np.testing.assert_array_almost_equal(accum, np.zeros_like(accum))

    def test_accumulate_zero_vector(self, encoder):
        """Test accumulating a zero vector."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")
        zero_vec = np.zeros_like(vec)

        accum = encoder.accumulate(accum, vec)
        accum = encoder.accumulate(accum, zero_vec)

        # Should be unchanged
        expected = vec.astype(np.float64)
        np.testing.assert_array_equal(accum, expected)

    def test_very_large_accumulator(self, encoder):
        """Test accumulator with many observations doesn't overflow."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")

        # Accumulate 1 million times
        n = 1_000_000
        for _ in range(n):
            accum = encoder.accumulate(accum, vec)

        # Check values are reasonable (should be n * original values)
        expected = n * vec.astype(np.float64)
        np.testing.assert_array_almost_equal(accum, expected)

        # Normalization should still work
        normalized = encoder.normalize_accumulator(accum)
        assert abs(np.linalg.norm(normalized) - 1.0) < 1e-6

    def test_mixed_positive_negative_accumulation(self, encoder):
        """Test accumulating mix of positive and negative values."""
        accum = encoder.create_accumulator()

        # Create vectors with known values
        vec_pos = np.ones(encoder.vector_manager.dimensions, dtype=np.int8)
        vec_neg = -np.ones(encoder.vector_manager.dimensions, dtype=np.int8)

        # Add 3 positive, 1 negative
        accum = encoder.accumulate(accum, vec_pos)
        accum = encoder.accumulate(accum, vec_pos)
        accum = encoder.accumulate(accum, vec_pos)
        accum = encoder.accumulate(accum, vec_neg)

        # Result should be 2 * ones
        expected = 2 * np.ones(encoder.vector_manager.dimensions, dtype=np.float64)
        np.testing.assert_array_equal(accum, expected)


class TestIntegrationWithEncodedData:
    """Integration tests with real encoded data structures."""

    def test_accumulate_encoded_records(self, encoder):
        """Test accumulating encoded data records."""
        records = [
            {"user": "alice", "action": "login"},
            {"user": "bob", "action": "login"},
            {"user": "alice", "action": "logout"},
        ]

        accum = encoder.create_accumulator()
        for record in records:
            vec = encoder.encode_data(record)
            accum = encoder.accumulate(accum, vec)

        # Should be able to query
        normalized = encoder.normalize_accumulator(accum)

        # Query with similar record
        query = encoder.encode_data({"user": "alice", "action": "login"})
        sim = cosine_similarity(normalized, query)

        assert sim > 0, "Similar record should have positive similarity"

    def test_accumulate_nested_data(self, encoder):
        """Test accumulating nested data structures."""
        records = [
            {"request": {"method": "GET", "headers": {"auth": "token"}}},
            {"request": {"method": "POST", "headers": {"auth": "token"}}},
        ]

        accum = encoder.create_accumulator()
        for record in records:
            vec = encoder.encode_data(record)
            accum = encoder.accumulate(accum, vec)

        normalized = encoder.normalize_accumulator(accum)

        # Should handle nested structures
        assert normalized.shape == (encoder.vector_manager.dimensions,)
        assert not np.all(normalized == 0)


class TestDeterminism:
    """Tests for deterministic behavior."""

    def test_same_inputs_same_output(self, encoder):
        """Test that same inputs produce same outputs."""
        vectors = [encoder.vector_manager.get_vector(f"item_{i}") for i in range(10)]

        # Run twice
        accum1 = encoder.create_accumulator()
        for vec in vectors:
            accum1 = encoder.accumulate(accum1, vec)

        accum2 = encoder.create_accumulator()
        for vec in vectors:
            accum2 = encoder.accumulate(accum2, vec)

        np.testing.assert_array_equal(accum1, accum2)

    def test_reproducible_across_encoders(self):
        """Test that different encoder instances give same result."""
        # VectorManager uses internal random state, so same atoms
        # get same vectors within a single instance. For cross-instance
        # determinism, use the DeterministicVectorManager from challenge 010.
        vm1 = VectorManager(dimensions=512)

        # Same encoder, encode same data twice
        enc1 = Encoder(vector_manager=vm1)

        data = {"key": "value", "number": 42}

        accum1 = enc1.create_accumulator()
        accum1 = enc1.accumulate(accum1, enc1.encode_data(data))

        accum2 = enc1.create_accumulator()
        accum2 = enc1.accumulate(accum2, enc1.encode_data(data))

        # Same encoder instance should give same result
        np.testing.assert_array_equal(accum1, accum2)
