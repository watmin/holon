#!/usr/bin/env python3
"""
Tests for the enhanced HolonClient.

This validates the client as the primary interface for Holon,
covering all VSA primitives, streaming operations, and encoding.
"""

import numpy as np
import pytest

from holon import CPUStore, HolonClient, create_client


class TestClientInitialization:
    """Test client creation modes."""

    def test_standalone_client(self):
        """Client can be created without arguments."""
        client = HolonClient()
        assert client.dimensions == 4096
        assert client._mode == "local"

    def test_standalone_with_dimensions(self):
        """Client respects custom dimensions."""
        client = HolonClient(dimensions=2048)
        assert client.dimensions == 2048

    def test_with_existing_store(self):
        """Client works with existing CPUStore."""
        store = CPUStore(dimensions=8192)
        client = HolonClient(local_store=store)
        assert client.dimensions == 8192
        assert client._store is store

    def test_create_client_convenience(self):
        """create_client() convenience function works."""
        client = create_client(dimensions=2048)
        assert isinstance(client, HolonClient)
        assert client.dimensions == 2048

    def test_cannot_specify_both(self):
        """Cannot specify both local_store and remote_url."""
        store = CPUStore()
        with pytest.raises(ValueError, match="Cannot specify both"):
            HolonClient(local_store=store, remote_url="http://localhost:8000")


class TestCoreEncoding:
    """Test the encode() method."""

    def test_encode_dict(self):
        """Encode a dict directly."""
        client = HolonClient()
        vec = client.encode({"type": "billing", "amount": 100})
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (4096,)
        assert vec.dtype == np.int8

    def test_encode_json_string(self):
        """Encode a JSON string."""
        client = HolonClient()
        vec = client.encode('{"type": "technical"}')
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (4096,)

    def test_encode_deterministic(self):
        """Same data produces same vector."""
        client = HolonClient()
        vec1 = client.encode({"type": "billing"})
        vec2 = client.encode({"type": "billing"})
        assert np.array_equal(vec1, vec2)

    def test_different_data_different_vectors(self):
        """Different data produces different vectors."""
        client = HolonClient()
        vec1 = client.encode({"type": "billing"})
        vec2 = client.encode({"type": "technical"})
        assert not np.array_equal(vec1, vec2)


class TestVSAPrimitives:
    """Test VSA/HDC operations."""

    @pytest.fixture
    def client(self):
        return HolonClient()

    @pytest.fixture
    def sample_vectors(self, client):
        A = client.encode({"item": "A"})
        B = client.encode({"item": "B"})
        C = client.encode({"item": "C"})
        return A, B, C

    def test_bind(self, client, sample_vectors):
        """Bind creates association."""
        A, B, _ = sample_vectors
        AB = client.bind(A, B)
        assert isinstance(AB, np.ndarray)
        # Binding is dissimilar to both inputs
        assert abs(client.similarity(AB, A)) < 0.3
        assert abs(client.similarity(AB, B)) < 0.3

    def test_unbind(self, client, sample_vectors):
        """Unbind reverses bind."""
        A, B, _ = sample_vectors
        AB = client.bind(A, B)
        B_recovered = client.unbind(AB, A)
        # Should be similar to B
        sim = client.similarity(B_recovered, B)
        assert sim > 0.5

    def test_bundle(self, client, sample_vectors):
        """Bundle creates superposition."""
        A, B, C = sample_vectors
        ABC = client.bundle([A, B, C])
        assert isinstance(ABC, np.ndarray)
        # Bundle is similar to all inputs
        assert client.similarity(ABC, A) > 0.3
        assert client.similarity(ABC, B) > 0.3
        assert client.similarity(ABC, C) > 0.3

    def test_negate(self, client, sample_vectors):
        """Negate removes component."""
        A, B, C = sample_vectors
        ABC = client.bundle([A, B, C])
        AC = client.negate(ABC, B)
        # B should have lower (possibly negative) similarity
        orig_sim = client.similarity(ABC, B)
        new_sim = client.similarity(AC, B)
        assert new_sim < orig_sim

    def test_amplify(self, client, sample_vectors):
        """Amplify strengthens component."""
        A, B, C = sample_vectors
        ABC = client.bundle([A, B, C])
        boosted = client.amplify(ABC, B, strength=2.0)
        # B should have higher similarity after amplification
        orig_sim = client.similarity(ABC, B)
        new_sim = client.similarity(boosted, B)
        assert new_sim > orig_sim

    def test_prototype(self, client):
        """Prototype extracts common pattern."""
        # Create vectors with a common component
        base = client.encode({"common": "shared"})
        v1 = client.bundle([base, client.encode({"unique": "1"})])
        v2 = client.bundle([base, client.encode({"unique": "2"})])
        v3 = client.bundle([base, client.encode({"unique": "3"})])

        proto = client.prototype([v1, v2, v3])
        assert isinstance(proto, np.ndarray)
        # Prototype should be similar to the base
        assert client.similarity(proto, base) > 0.3

    def test_difference(self, client, sample_vectors):
        """Difference captures what changed."""
        A, B, _ = sample_vectors
        AB = client.bundle([A, B])
        ABC = client.bundle([A, B, client.encode({"item": "C_new"})])
        delta = client.difference(AB, ABC)
        assert isinstance(delta, np.ndarray)

    def test_blend(self, client, sample_vectors):
        """Blend interpolates between vectors."""
        A, B, _ = sample_vectors
        mid = client.blend(A, B, alpha=0.5)
        assert isinstance(mid, np.ndarray)
        # Midpoint should be somewhat similar to both
        assert client.similarity(mid, A) > 0.2
        assert client.similarity(mid, B) > 0.2

    def test_resonance(self, client, sample_vectors):
        """Resonance extracts agreeing parts."""
        A, B, _ = sample_vectors
        AB = client.bundle([A, B])
        resonant = client.resonance(AB, A)
        assert isinstance(resonant, np.ndarray)

    def test_permute(self, client, sample_vectors):
        """Permute shifts vector."""
        A, _, _ = sample_vectors
        shifted = client.permute(A, 1)
        assert isinstance(shifted, np.ndarray)
        # Shifted should be different from original
        assert not np.array_equal(A, shifted)

    def test_cleanup(self, client, sample_vectors):
        """Cleanup finds closest codebook match."""
        A, B, C = sample_vectors
        # Add noise to A
        noisy = A + np.random.randint(-1, 2, size=A.shape, dtype=np.int8)
        cleaned = client.cleanup(noisy, [A, B, C])
        # Should return A (closest match)
        assert np.array_equal(cleaned, A)


class TestAccumulators:
    """Test streaming/accumulator operations."""

    @pytest.fixture
    def client(self):
        return HolonClient()

    def test_create_accumulator(self, client):
        """Create accumulator produces zero vector."""
        accum = client.create_accumulator()
        assert isinstance(accum, np.ndarray)
        assert accum.dtype == np.float64
        assert accum.shape == (client.dimensions,)
        assert np.all(accum == 0)

    def test_accumulate(self, client):
        """Accumulate adds to running sum."""
        accum = client.create_accumulator()
        vec = client.encode({"type": "test"})
        accum = client.accumulate(accum, vec)
        assert not np.all(accum == 0)

    def test_accumulate_frequency(self, client):
        """Accumulate preserves frequency."""
        accum = client.create_accumulator()
        vec_common = client.encode({"type": "common"})
        vec_rare = client.encode({"type": "rare"})

        # Add common 10 times, rare once
        for _ in range(10):
            accum = client.accumulate(accum, vec_common)
        accum = client.accumulate(accum, vec_rare)

        norm = client.normalize_accumulator(accum)
        # Common should have higher similarity
        sim_common = client.similarity(vec_common, norm)
        sim_rare = client.similarity(vec_rare, norm)
        assert sim_common > sim_rare

    def test_normalize_accumulator(self, client):
        """Normalize produces unit vector."""
        accum = client.create_accumulator()
        vec = client.encode({"type": "test"})
        accum = client.accumulate(accum, vec)
        norm = client.normalize_accumulator(accum)
        assert isinstance(norm, np.ndarray)
        assert norm.dtype == np.float32
        # Should be approximately unit length
        length = np.linalg.norm(norm)
        assert abs(length - 1.0) < 0.01

    def test_threshold_accumulator(self, client):
        """Threshold produces bipolar vector."""
        accum = client.create_accumulator()
        vec = client.encode({"type": "test"})
        for _ in range(5):
            accum = client.accumulate(accum, vec)
        thresholded = client.threshold_accumulator(accum)
        assert isinstance(thresholded, np.ndarray)
        assert thresholded.dtype == np.int8
        # Values should be in {-1, 0, 1}
        unique = np.unique(thresholded)
        assert set(unique).issubset({-1, 0, 1})


class TestContinuousEncoding:
    """Test scalar encoding."""

    @pytest.fixture
    def client(self):
        return HolonClient()

    def test_encode_scalar_linear(self, client):
        """Linear scalar encoding produces vector."""
        vec = client.encode_scalar(100, mode="linear")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (client.dimensions,)

    def test_encode_scalar_nearby_similar(self, client):
        """Nearby linear values are similar."""
        v100 = client.encode_scalar(100)
        v110 = client.encode_scalar(110)
        v1000 = client.encode_scalar(1000)

        sim_close = client.similarity(v100, v110)
        sim_far = client.similarity(v100, v1000)
        assert sim_close > sim_far

    def test_encode_scalar_circular(self, client):
        """Circular encoding wraps."""
        h0 = client.encode_scalar(0, mode="circular", period=24)
        h23 = client.encode_scalar(23, mode="circular", period=24)
        h12 = client.encode_scalar(12, mode="circular", period=24)

        # 0 and 23 should be similar (close on circle)
        # 0 and 12 should be less similar (opposite on circle)
        sim_0_23 = client.similarity(h0, h23)
        sim_0_12 = client.similarity(h0, h12)
        assert sim_0_23 > sim_0_12

    def test_encode_scalar_log(self, client):
        """Log scale encoding."""
        v100 = client.encode_scalar_log(100)
        v1000 = client.encode_scalar_log(1000)
        v10000 = client.encode_scalar_log(10000)

        # Equal ratios should have similar similarity drops
        sim_100_1000 = client.similarity(v100, v1000)
        sim_1000_10000 = client.similarity(v1000, v10000)
        # Both are 10x ratio, should be similar drop
        assert abs(sim_100_1000 - sim_1000_10000) < 0.1


class TestSequenceEncoding:
    """Test sequence encoding."""

    @pytest.fixture
    def client(self):
        return HolonClient()

    def test_encode_sequence_positional(self, client):
        """Positional encoding preserves order."""
        seq1 = client.encode_sequence(["A", "B", "C"], mode="positional")
        seq2 = client.encode_sequence(["C", "B", "A"], mode="positional")
        # Different orders should produce different vectors
        sim = client.similarity(seq1, seq2)
        assert sim < 0.9  # Not identical

    def test_encode_sequence_bundle(self, client):
        """Bundle mode ignores order."""
        seq1 = client.encode_sequence(["A", "B", "C"], mode="bundle")
        seq2 = client.encode_sequence(["C", "B", "A"], mode="bundle")
        # Same elements, same vector (order-independent)
        sim = client.similarity(seq1, seq2)
        assert sim > 0.95  # Nearly identical

    def test_encode_sequence_ngram(self, client):
        """N-gram mode captures local patterns."""
        seq = client.encode_sequence(["quick", "brown", "fox"], mode="ngram")
        assert isinstance(seq, np.ndarray)


class TestSimilarity:
    """Test similarity computation."""

    @pytest.fixture
    def client(self):
        return HolonClient()

    def test_similarity_cosine(self, client):
        """Cosine similarity works."""
        v1 = client.encode({"type": "billing"})
        v2 = client.encode({"type": "billing"})
        sim = client.similarity(v1, v2, metric="cosine")
        assert abs(sim - 1.0) < 1e-10  # Identical vectors

    def test_similarity_hamming(self, client):
        """Hamming similarity works."""
        v1 = client.encode({"type": "billing"})
        v2 = client.encode({"type": "technical"})
        sim = client.similarity(v1, v2, metric="hamming")
        assert 0 <= sim <= 1

    def test_similarity_multiple_metrics(self, client):
        """Multiple metrics available."""
        v1 = client.encode({"type": "billing"})
        v2 = client.encode({"type": "technical"})

        metrics = ["cosine", "dot", "euclidean", "manhattan", "hamming"]
        for metric in metrics:
            sim = client.similarity(v1, v2, metric=metric)
            assert isinstance(sim, float)


class TestGetVector:
    """Test direct atom vector access."""

    def test_get_vector(self):
        """Get base vector for atom."""
        client = HolonClient()
        v1 = client.get_vector("billing")
        v2 = client.get_vector("billing")
        assert np.array_equal(v1, v2)

    def test_get_vector_different(self):
        """Different atoms produce different vectors."""
        client = HolonClient()
        v1 = client.get_vector("billing")
        v2 = client.get_vector("technical")
        assert not np.array_equal(v1, v2)


class TestDataOperations:
    """Test insert/search/get operations."""

    def test_insert_and_get(self):
        """Insert and retrieve data."""
        client = HolonClient()
        item_id = client.insert({"type": "test", "value": 42})
        retrieved = client.get(item_id)
        assert retrieved["type"] == "test"
        assert retrieved["value"] == 42

    def test_insert_batch(self):
        """Batch insert multiple items."""
        client = HolonClient()
        items = [{"type": "test", "idx": i} for i in range(10)]
        ids = client.insert_batch(items)
        assert len(ids) == 10

    def test_search(self):
        """Search finds similar items."""
        client = HolonClient()
        client.insert({"type": "billing", "amount": 100})
        client.insert({"type": "billing", "amount": 200})
        client.insert({"type": "technical", "issue": "bug"})

        results = client.search(probe={"type": "billing"}, limit=10)
        assert len(results) > 0
        # Top results should be billing-related
        assert "billing" in str(results[0])


class TestProperties:
    """Test property access."""

    def test_dimensions_property(self):
        """Dimensions property works."""
        client = HolonClient(dimensions=2048)
        assert client.dimensions == 2048

    def test_encoder_property(self):
        """Encoder property works in local mode."""
        client = HolonClient()
        encoder = client.encoder
        assert encoder is not None

    def test_vector_manager_property(self):
        """Vector manager property works in local mode."""
        client = HolonClient()
        vm = client.vector_manager
        assert vm is not None


class TestIntegration:
    """Integration tests for common workflows."""

    def test_anomaly_detection_workflow(self):
        """Complete anomaly detection workflow."""
        client = HolonClient()

        # Build baseline from "normal" data
        accum = client.create_accumulator()
        for i in range(100):
            vec = client.encode({"type": "normal", "value": i % 10})
            accum = client.accumulate(accum, vec)
        baseline = client.normalize_accumulator(accum)

        # Test normal data - should have high similarity
        normal_vec = client.encode({"type": "normal", "value": 5})
        normal_sim = client.similarity(normal_vec, baseline)

        # Test anomalous data - should have lower similarity
        anomaly_vec = client.encode({"type": "attack", "payload": "malicious"})
        anomaly_sim = client.similarity(anomaly_vec, baseline)

        assert normal_sim > anomaly_sim

    def test_rate_encoding_workflow(self):
        """Rate-based detection workflow."""
        client = HolonClient()

        # Build baseline from normal rate
        accum = client.create_accumulator()
        for _ in range(50):
            rate_vec = client.encode_scalar_log(100)  # 100 pps normal
            accum = client.accumulate(accum, rate_vec)
        baseline = client.normalize_accumulator(accum)

        # Normal rate should have high similarity
        normal_rate = client.encode_scalar_log(120)
        normal_sim = client.similarity(normal_rate, baseline)

        # Attack rate (1000x) should have low similarity
        attack_rate = client.encode_scalar_log(100000)
        attack_sim = client.similarity(attack_rate, baseline)

        assert normal_sim > attack_sim

    def test_multi_signal_detection(self):
        """Multi-signal detection combining pattern and rate."""
        client = HolonClient()

        # Pattern baseline
        pattern_accum = client.create_accumulator()
        for _ in range(50):
            vec = client.encode({"protocol": "TCP", "dst_port": 80})
            pattern_accum = client.accumulate(pattern_accum, vec)
        pattern_baseline = client.normalize_accumulator(pattern_accum)

        # Rate baseline
        rate_accum = client.create_accumulator()
        for _ in range(50):
            vec = client.encode_scalar_log(100)
            rate_accum = client.accumulate(rate_accum, vec)
        rate_baseline = client.normalize_accumulator(rate_accum)

        # Attack: different pattern AND high rate
        attack_pattern = client.encode({"protocol": "UDP", "src_port": 53})
        attack_rate = client.encode_scalar_log(100000)

        pattern_sim = client.similarity(attack_pattern, pattern_baseline)
        rate_sim = client.similarity(attack_rate, rate_baseline)

        # Both should indicate anomaly
        assert pattern_sim < 0.5  # Different pattern
        assert rate_sim < 0.8  # Abnormal rate


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
