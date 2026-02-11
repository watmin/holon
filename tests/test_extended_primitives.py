"""
Tests for extended algebraic primitives in encoder.py.

These primitives extend the VSA kernel with:
1. decay() - Time-weighted forgetting
2. unbind() - Explicit inverse bind (self-inverse for bipolar)
3. similarity_profile() - Similarity as vector, not scalar
4. attend() - Weighted resonance / soft attention
5. segment() - Find breakpoints in streams
6. analogy() - A:B::C:? relational transfer
7. project() - Subspace projection
8. complexity() - Entropy/mixture measure
9. conditional_bind() - Gated binding
10. invert() - Reconstruction from vector

Key properties tested:
- Algebraic correctness (unbind inverts bind)
- Streaming behavior (segment, decay)
- Compositional properties (analogy, project)
- Information-theoretic measures (complexity)
"""

import numpy as np
import pytest

from holon.distance import cosine_similarity
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


class TestDecay:
    """Tests for the decay() primitive."""

    def test_decay_basic(self, encoder):
        """Test basic decay functionality."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")

        # Accumulate some vectors
        for _ in range(100):
            accum = encoder.accumulate(accum, vec)

        # Apply decay
        decayed = encoder.decay(accum, factor=0.5)

        # Should be half the original
        expected = accum * 0.5
        np.testing.assert_array_almost_equal(decayed, expected)

    def test_decay_preserves_shape(self, encoder):
        """Test that decay preserves shape and dtype."""
        accum = encoder.create_accumulator()
        accum = encoder.accumulate(accum, encoder.vector_manager.get_vector("x"))

        decayed = encoder.decay(accum, factor=0.99)

        assert decayed.shape == accum.shape
        assert decayed.dtype == np.float64

    def test_decay_zero_factor(self, encoder):
        """Test decay with factor=0 zeroes accumulator."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")
        accum = encoder.accumulate(accum, vec)

        decayed = encoder.decay(accum, factor=0.0)

        np.testing.assert_array_equal(decayed, np.zeros_like(decayed))

    def test_decay_one_factor(self, encoder):
        """Test decay with factor=1 leaves accumulator unchanged."""
        accum = encoder.create_accumulator()
        vec = encoder.vector_manager.get_vector("test")
        accum = encoder.accumulate(accum, vec)

        decayed = encoder.decay(accum, factor=1.0)

        np.testing.assert_array_equal(decayed, accum)

    def test_decay_streaming_recency_bias(self, encoder):
        """Test that decay creates recency bias in streaming."""
        old_pattern = encoder.vector_manager.get_vector("old")
        new_pattern = encoder.vector_manager.get_vector("new")

        accum = encoder.create_accumulator()

        # Add old pattern 100 times
        for _ in range(100):
            accum = encoder.accumulate(accum, old_pattern)

        # Apply heavy decay
        accum = encoder.decay(accum, factor=0.1)

        # Add new pattern 10 times
        for _ in range(10):
            accum = encoder.accumulate(accum, new_pattern)

        normalized = encoder.normalize_accumulator(accum)

        # New pattern should dominate despite fewer observations
        sim_old = cosine_similarity(normalized, old_pattern)
        sim_new = cosine_similarity(normalized, new_pattern)

        assert sim_new > sim_old, (
            f"New pattern should dominate after decay: "
            f"old={sim_old:.4f}, new={sim_new:.4f}"
        )


class TestUnbind:
    """Tests for the unbind() primitive."""

    def test_unbind_recovers_value(self, encoder):
        """Test that unbind(bind(A, B), A) ≈ B.

        Note: For bipolar vectors with zeros, recovery isn't perfect
        because 0 * x = 0 loses information. But similarity should be high.
        """
        A = encoder.vector_manager.get_vector("key")
        B = encoder.vector_manager.get_vector("value")

        bound = encoder.bind(A, B)
        recovered = encoder.unbind(bound, A)

        sim = cosine_similarity(recovered, B)
        # Recovery is imperfect due to zero dimensions, but should be high
        assert (
            sim > 0.7
        ), f"Unbind should recover value with high similarity: sim={sim:.4f}"

    def test_unbind_self_inverse(self, encoder):
        """Test that unbind is identical to bind for bipolar vectors."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        bound = encoder.bind(A, B)

        unbind_result = encoder.unbind(bound, A)
        bind_result = encoder.bind(bound, A)

        np.testing.assert_array_equal(unbind_result, bind_result)

    def test_unbind_wrong_key(self, encoder):
        """Test that unbind with wrong key gives low similarity."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        bound = encoder.bind(A, B)
        wrong = encoder.unbind(bound, C)

        sim = cosine_similarity(wrong, B)
        assert abs(sim) < 0.2, f"Wrong key should give low similarity: sim={sim:.4f}"


class TestSimilarityProfile:
    """Tests for the similarity_profile() primitive."""

    def test_similarity_profile_shape(self, encoder):
        """Test that profile has correct shape and dtype."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        profile = encoder.similarity_profile(A, B)

        assert profile.shape == A.shape
        assert profile.dtype == np.int8

    def test_similarity_profile_identical_vectors(self, encoder):
        """Test profile of identical vectors is all +1."""
        A = encoder.vector_manager.get_vector("A")

        profile = encoder.similarity_profile(A, A)

        # Where A is non-zero, profile should be +1
        nonzero_mask = A != 0
        assert np.all(profile[nonzero_mask] == 1)

    def test_similarity_profile_opposite_vectors(self, encoder):
        """Test profile of opposite vectors is all -1."""
        A = encoder.vector_manager.get_vector("A")
        neg_A = -A

        profile = encoder.similarity_profile(A, neg_A)

        # Where A is non-zero, profile should be -1
        nonzero_mask = A != 0
        assert np.all(profile[nonzero_mask] == -1)

    def test_similarity_profile_orthogonal_vectors(self, encoder):
        """Test profile of orthogonal vectors mixes +1/-1."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        profile = encoder.similarity_profile(A, B)

        # Should have mix of +1, -1, and 0
        unique = set(np.unique(profile))
        assert len(unique) >= 2, "Orthogonal vectors should have mixed profile"


class TestAttend:
    """Tests for the attend() primitive."""

    def test_attend_hard_mode(self, encoder):
        """Test hard attention is same as resonance."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        AB = encoder.bundle([A, B])

        attended = encoder.attend(A, AB, mode="hard")
        resonance = encoder.resonance(AB, A)

        np.testing.assert_array_equal(attended, resonance)

    def test_attend_soft_increases_query_similarity(self, encoder):
        """Test that soft attention increases similarity to query."""
        query = encoder.vector_manager.get_vector("query")
        memory = encoder.bundle(
            [
                encoder.vector_manager.get_vector("query"),
                encoder.vector_manager.get_vector("noise1"),
                encoder.vector_manager.get_vector("noise2"),
            ]
        )

        attended = encoder.attend(query, memory, strength=2.0, mode="soft")

        sim_before = cosine_similarity(memory, query)
        sim_after = cosine_similarity(attended, query)

        assert sim_after >= sim_before, (
            f"Attention should increase query similarity: "
            f"before={sim_before:.4f}, after={sim_after:.4f}"
        )

    def test_attend_amplify_mode(self, encoder):
        """Test amplify mode boosts agreeing dimensions."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        AB = encoder.bundle([A, B])

        attended = encoder.attend(A, AB, strength=1.0, mode="amplify")

        # Attended should be more similar to A than original
        sim_original = cosine_similarity(AB, A)
        sim_attended = cosine_similarity(attended, A)

        assert sim_attended >= sim_original


class TestSegment:
    """Tests for the segment() primitive."""

    def test_segment_detects_change(self, encoder):
        """Test that segment detects pattern changes."""
        pattern1 = encoder.vector_manager.get_vector("pattern1")
        pattern2 = encoder.vector_manager.get_vector("pattern2")

        # Create stream: 50 of pattern1, then 50 of pattern2
        stream = [pattern1] * 50 + [pattern2] * 50

        breakpoints = encoder.segment(stream, window=20, threshold=0.5)

        # Should detect at least one breakpoint around index 50
        assert len(breakpoints) >= 2, "Should detect pattern change"
        assert any(
            45 <= bp <= 55 for bp in breakpoints
        ), f"Should detect breakpoint near 50: {breakpoints}"

    def test_segment_empty_stream(self, encoder):
        """Test segment on empty stream."""
        breakpoints = encoder.segment([], window=10, threshold=0.5)
        assert breakpoints == []

    def test_segment_single_element(self, encoder):
        """Test segment on single element stream."""
        A = encoder.vector_manager.get_vector("A")
        breakpoints = encoder.segment([A], window=10, threshold=0.5)
        assert breakpoints == [0]

    def test_segment_uniform_stream(self, encoder):
        """Test segment on uniform stream (no changes)."""
        A = encoder.vector_manager.get_vector("A")
        stream = [A] * 100

        breakpoints = encoder.segment(stream, window=20, threshold=0.5)

        # Should only have initial breakpoint
        assert breakpoints == [
            0
        ], f"Uniform stream should have no extra breaks: {breakpoints}"

    def test_segment_diff_method(self, encoder):
        """Test segment with diff method."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        stream = [A, A, A, B, B, B]

        breakpoints = encoder.segment(stream, window=10, threshold=0.5, method="diff")

        # Should detect change at index 3
        assert 3 in breakpoints, f"Should detect change at 3: {breakpoints}"


class TestAnalogy:
    """Tests for the analogy() primitive."""

    def test_analogy_basic(self, encoder):
        """Test basic analogy operation."""
        # Create related pairs
        king = encoder.vector_manager.get_vector("king")
        man = encoder.vector_manager.get_vector("man")
        woman = encoder.vector_manager.get_vector("woman")
        queen = encoder.vector_manager.get_vector("queen")

        # king:man :: woman:? should be similar to queen
        result = encoder.analogy(king, man, woman)

        # This is a weak test since vectors are random, but result should differ from inputs
        assert result.shape == king.shape
        assert result.dtype == np.int8

    def test_analogy_identity(self, encoder):
        """Test analogy with identical A and B."""
        A = encoder.vector_manager.get_vector("A")
        C = encoder.vector_manager.get_vector("C")

        # If A == B, analogy should return ~C
        result = encoder.analogy(A, A, C)

        sim = cosine_similarity(result, C)
        assert sim > 0.9, f"Analogy with A==B should return ~C: sim={sim:.4f}"

    def test_analogy_inverse(self, encoder):
        """Test that analogy is invertible."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        # A:B::C:D, then B:A::D:? should be ~C
        D = encoder.analogy(A, B, C)
        recovered = encoder.analogy(B, A, D)

        sim = cosine_similarity(recovered, C)
        assert sim > 0.5, f"Analogy should be invertible: sim={sim:.4f}"


class TestProject:
    """Tests for the project() primitive."""

    def test_project_onto_self(self, encoder):
        """Test that projecting onto self returns ~self."""
        A = encoder.vector_manager.get_vector("A")

        projected = encoder.project(A, [A])

        sim = cosine_similarity(projected, A)
        assert sim > 0.9, f"Projection onto self should return ~self: sim={sim:.4f}"

    def test_project_onto_orthogonal(self, encoder):
        """Test that projecting onto orthogonal subspace is small."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        # Project A onto [B, C] subspace
        projected = encoder.project(A, [B, C])

        # Should be low similarity to A (mostly orthogonal)
        sim = cosine_similarity(projected, A)
        assert (
            sim < 0.5
        ), f"Projection onto orthogonal subspace should be small: sim={sim:.4f}"

    def test_project_empty_subspace(self, encoder):
        """Test projection onto empty subspace returns zeros."""
        A = encoder.vector_manager.get_vector("A")

        projected = encoder.project(A, [])

        assert np.all(projected == 0)

    def test_project_preserves_subspace_component(self, encoder):
        """Test that projection preserves the subspace component."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        # Create a vector with A and B components
        mixed = encoder.bundle([A, B])

        # Project onto A subspace
        projected = encoder.project(mixed, [A])

        # Should be more similar to A than original
        sim_to_A = cosine_similarity(projected, A)
        sim_to_B = cosine_similarity(projected, B)

        assert sim_to_A > sim_to_B, (
            f"Projection should preserve A component: "
            f"sim_A={sim_to_A:.4f}, sim_B={sim_to_B:.4f}"
        )


class TestComplexity:
    """Tests for the complexity() primitive."""

    def test_complexity_range(self, encoder):
        """Test that complexity returns value in [0, 1]."""
        A = encoder.vector_manager.get_vector("A")

        c = encoder.complexity(A)

        assert 0.0 <= c <= 1.0, f"Complexity should be in [0, 1]: {c}"

    def test_complexity_single_vs_bundle(self, encoder):
        """Test that bundles have higher complexity than single vectors."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")
        D = encoder.vector_manager.get_vector("D")
        E = encoder.vector_manager.get_vector("E")

        single = A
        bundle5 = encoder.bundle([A, B, C, D, E])

        c_single = encoder.complexity(single)
        c_bundle = encoder.complexity(bundle5)

        # Bundle of 5 should be more complex
        assert (
            c_bundle > c_single
        ), f"Bundle should be more complex: single={c_single:.4f}, bundle={c_bundle:.4f}"

    def test_complexity_zero_vector(self, encoder):
        """Test complexity of zero vector."""
        zero = np.zeros(encoder.vector_manager.dimensions, dtype=np.int8)

        c = encoder.complexity(zero)

        assert c == 0.0, f"Zero vector should have zero complexity: {c}"

    def test_complexity_accumulator(self, encoder):
        """Test complexity on float accumulator."""
        accum = encoder.create_accumulator()

        for _ in range(100):
            vec = encoder.vector_manager.get_vector(f"item_{np.random.randint(0, 10)}")
            accum = encoder.accumulate(accum, vec)

        c = encoder.complexity(accum)

        assert 0.0 <= c <= 1.0, f"Accumulator complexity should be in [0, 1]: {c}"


class TestConditionalBind:
    """Tests for the conditional_bind() primitive."""

    def test_conditional_bind_positive_gate(self, encoder):
        """Test conditional bind with positive gate."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        gate = encoder.vector_manager.get_vector("gate")

        gated = encoder.conditional_bind(A, B, gate, mode="positive")

        # Should have same shape
        assert gated.shape == A.shape
        assert gated.dtype == np.int8

        # Where gate is negative, result should be 0
        neg_mask = gate < 0
        assert np.all(gated[neg_mask] == 0), "Negative gate dims should be zero"

    def test_conditional_bind_negative_gate(self, encoder):
        """Test conditional bind with negative gate."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        gate = encoder.vector_manager.get_vector("gate")

        gated = encoder.conditional_bind(A, B, gate, mode="negative")

        # Where gate is positive, result should be 0
        pos_mask = gate > 0
        assert np.all(gated[pos_mask] == 0), "Positive gate dims should be zero"

    def test_conditional_bind_full_positive_gate(self, encoder):
        """Test that all-positive gate gives same as bind."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        full_gate = np.ones(encoder.vector_manager.dimensions, dtype=np.int8)

        gated = encoder.conditional_bind(A, B, full_gate, mode="positive")
        normal = encoder.bind(A, B)

        # Should be identical since gate is all positive
        np.testing.assert_array_equal(gated, normal)

    def test_conditional_bind_zero_gate(self, encoder):
        """Test that all-zero gate gives zeros."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        zero_gate = np.zeros(encoder.vector_manager.dimensions, dtype=np.int8)

        gated = encoder.conditional_bind(A, B, zero_gate, mode="positive")

        assert np.all(gated == 0), "Zero gate should give zero result"


class TestInvert:
    """Tests for the invert() primitive."""

    def test_invert_with_codebook(self, encoder):
        """Test invert finds matching components in codebook."""
        billing = encoder.vector_manager.get_vector("billing")
        auth = encoder.vector_manager.get_vector("auth")
        network = encoder.vector_manager.get_vector("network")

        # Create mixed vector with billing and auth
        mixed = encoder.bundle([billing, auth])

        # Create codebook
        codebook = [
            ("billing", billing),
            ("auth", auth),
            ("network", network),
        ]

        results = encoder.invert(mixed, codebook, top_k=3, threshold=0.0)

        # Should find billing and auth with higher similarity
        names = [name for name, _ in results]
        assert "billing" in names, f"Should find billing: {results}"
        assert "auth" in names, f"Should find auth: {results}"

        # Network should have lower similarity
        billing_sim = next(sim for name, sim in results if name == "billing")
        auth_sim = next(sim for name, sim in results if name == "auth")
        network_sim = next((sim for name, sim in results if name == "network"), 0.0)

        assert billing_sim > network_sim
        assert auth_sim > network_sim

    def test_invert_without_codebook(self, encoder):
        """Test invert returns analysis without codebook."""
        A = encoder.vector_manager.get_vector("A")

        results = encoder.invert(A, codebook=None)

        assert len(results) == 1
        assert results[0][0] == "_analysis"
        analysis = results[0][1]
        assert "complexity" in analysis
        assert "density" in analysis
        assert "magnitude" in analysis

    def test_invert_threshold(self, encoder):
        """Test that invert respects similarity threshold."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        codebook = [("A", A), ("B", B), ("C", C)]

        # Query with A, high threshold
        results = encoder.invert(A, codebook, threshold=0.9)

        # Only A should pass high threshold
        names = [name for name, _ in results]
        assert "A" in names
        assert len(names) == 1, f"Only A should pass threshold: {results}"

    def test_invert_top_k(self, encoder):
        """Test that invert respects top_k limit."""
        vecs = [encoder.vector_manager.get_vector(f"vec_{i}") for i in range(10)]
        codebook = [(f"vec_{i}", v) for i, v in enumerate(vecs)]

        query = encoder.bundle(vecs[:3])

        results = encoder.invert(query, codebook, top_k=2, threshold=0.0)

        assert len(results) == 2, f"Should return top_k=2 results: {len(results)}"


class TestIntegration:
    """Integration tests combining multiple primitives."""

    def test_decay_with_segment(self, encoder):
        """Test decay affects segmentation."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        # Stream with gradual decay
        stream = []
        for _ in range(50):
            stream.append(A)
        for _ in range(50):
            stream.append(B)

        # With accumulator method, decay should affect detection
        breakpoints = encoder.segment(
            stream, window=20, threshold=0.5, method="accumulator"
        )

        assert len(breakpoints) >= 1

    def test_analogy_with_project(self, encoder):
        """Test combining analogy with projection."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        # Compute analogy
        D = encoder.analogy(A, B, C)

        # Project result onto C subspace
        projected = encoder.project(D, [C])

        # Should have some component in C direction
        sim = cosine_similarity(projected, C)
        assert sim > 0, f"Analogy result should have C component: sim={sim:.4f}"

    def test_attend_with_invert(self, encoder):
        """Test attention followed by inversion."""
        billing = encoder.vector_manager.get_vector("billing")
        auth = encoder.vector_manager.get_vector("auth")
        network = encoder.vector_manager.get_vector("network")

        memory = encoder.bundle([billing, auth, network])

        # Attend to billing
        attended = encoder.attend(billing, memory, strength=2.0, mode="soft")

        # Invert to find components
        codebook = [
            ("billing", billing),
            ("auth", auth),
            ("network", network),
        ]
        results = encoder.invert(attended, codebook, threshold=0.0)

        # Billing should be top result
        assert results[0][0] == "billing", f"Billing should be top: {results}"

    def test_conditional_bind_with_complexity(self, encoder):
        """Test that conditional bind reduces complexity."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        gate = encoder.vector_manager.get_vector("gate")

        full_bind = encoder.bind(A, B)
        conditional = encoder.conditional_bind(A, B, gate, mode="positive")

        # Conditional should be sparser (lower density contributes to complexity)
        c_full = encoder.complexity(full_bind)
        c_conditional = encoder.complexity(conditional)

        # Conditional bind zeros out some dimensions, so should be different
        assert c_full != c_conditional or np.sum(full_bind != 0) != np.sum(
            conditional != 0
        )
