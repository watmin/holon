#!/usr/bin/env python3
"""
Encoder Coverage Tests
Tests for different encoding modes and error handling to improve encoder coverage.
"""

import edn_format
import numpy as np
import pytest

from holon.encoder import Encoder, ListEncodeMode
from holon.vector_manager import VectorManager


class TestEncoderModes:
    """Test different encoding modes and error handling."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance for testing."""
        vm = VectorManager(dimensions=1000)  # Small for testing
        return Encoder(vm)

    def test_encode_data_basic(self, encoder):
        """Test basic data encoding."""
        data = {"name": "Alice", "age": 30}
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert result.shape == (1000,)
        assert result.dtype == np.int8
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_empty_dict(self, encoder):
        """Test empty dictionary encoding."""
        result = encoder.encode_data({})
        assert np.all(result == 0)

    def test_encode_empty_list(self, encoder):
        """Test empty list encoding."""
        result = encoder.encode_data([])
        assert np.all(result == 0)

    def test_encode_empty_set(self, encoder):
        """Test empty set encoding."""
        result = encoder.encode_data(set())
        assert np.all(result == 0)

    def test_encode_nested_structures(self, encoder):
        """Test nested data structures."""
        data = {
            "user": {
                "name": "Alice",
                "profile": {"age": 30, "skills": ["python", "ml"]},
            }
        }
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_list_positional_mode(self, encoder):
        """Test positional list encoding (default)."""
        data = ["a", "b", "c"]
        result = encoder.encode_list(data, mode=ListEncodeMode.POSITIONAL)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_list_bundle_mode(self, encoder):
        """Test bundle list encoding."""
        data = ["a", "b", "c"]
        result = encoder.encode_list(data, mode=ListEncodeMode.BUNDLE)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_list_chained_mode(self, encoder):
        """Test chained list encoding."""
        data = ["a", "b", "c"]
        result = encoder.encode_list(data, mode=ListEncodeMode.CHAINED)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_list_ngram_mode(self, encoder):
        """Test n-gram list encoding."""
        data = ["a", "b", "c", "d"]
        result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_list_ngram_single_item(self, encoder):
        """Test n-gram encoding with single item."""
        data = ["single"]
        result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_list_ngram_empty(self, encoder):
        """Test n-gram encoding with empty list."""
        data = []
        result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM)
        assert np.all(result == 0)

    def test_encode_mode_hint_in_dict(self, encoder):
        """Test encoding mode hints in dictionaries."""
        data = {"sequence": {"$mode": "ngram", "data": ["word1", "word2", "word3"]}}
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_mode_hint_chained(self, encoder):
        """Test chained encoding mode hint."""
        data = {
            "words": {
                "$mode": "chained",
                "sequence": ["first", "second", "third"],
            }
        }
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_mode_hint_bundle(self, encoder):
        """Test bundle encoding mode hint."""
        data = {"tags": {"$mode": "bundle", "items": ["tag1", "tag2", "tag3"]}}
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_invalid_mode_hint(self, encoder):
        """Test invalid encoding mode hint (should ignore and use default)."""
        data = {"sequence": {"$mode": "invalid_mode", "data": ["a", "b", "c"]}}
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        # Should still work with default positional encoding

    def test_encode_list_invalid_mode(self, encoder):
        """Test invalid encoding mode raises ValueError."""
        with pytest.raises(ValueError, match="is not a valid ListEncodeMode"):
            encoder.encode_list(["a", "b"], mode="invalid_mode")

    def test_encode_list_string_mode(self, encoder):
        """Test encoding mode as string."""
        data = ["a", "b", "c"]
        result = encoder.encode_list(data, mode="ngram")
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_scalar_string(self, encoder):
        """Test string scalar encoding."""
        result = encoder._encode_scalar("test")
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_scalar_number(self, encoder):
        """Test numeric scalar encoding."""
        result = encoder._encode_scalar(42)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_scalar_float(self, encoder):
        """Test float scalar encoding."""
        result = encoder._encode_scalar(3.14)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_scalar_none(self, encoder):
        """Test None scalar encoding."""
        result = encoder._encode_scalar(None)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_scalar_boolean(self, encoder):
        """Test boolean scalar encoding."""
        result_true = encoder._encode_scalar(True)
        result_false = encoder._encode_scalar(False)
        assert isinstance(result_true, np.ndarray)
        assert isinstance(result_false, np.ndarray)
        assert np.all(np.isin(result_true, [-1, 0, 1]))
        assert np.all(np.isin(result_false, [-1, 0, 1]))

    def test_encode_scalar_edn_keyword(self, encoder):
        """Test EDN keyword encoding."""
        keyword = edn_format.Keyword("test")
        result = encoder._encode_scalar(keyword)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_scalar_edn_symbol(self, encoder):
        """Test EDN symbol encoding."""
        symbol = edn_format.Symbol("test-symbol")
        result = encoder._encode_scalar(symbol)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_scalar_edn_char(self, encoder):
        """Test EDN character encoding."""
        char = edn_format.Char("a")
        result = encoder._encode_scalar(char)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_scalar_unknown_type(self, encoder):
        """Test unknown type fallback encoding."""

        class UnknownClass:
            def __str__(self):
                return "unknown"

        obj = UnknownClass()
        result = encoder._encode_scalar(obj)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_bind_vectors(self, encoder):
        """Test vector binding operation."""
        vec1 = encoder._encode_scalar("a")
        vec2 = encoder._encode_scalar("b")
        result = encoder.bind(vec1, vec2)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_bundle_vectors(self, encoder):
        """Test vector bundling operation."""
        vecs = [
            encoder._encode_scalar("a"),
            encoder._encode_scalar("b"),
            encoder._encode_scalar("c"),
        ]
        result = encoder.bundle(vecs)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_bundle_empty_list(self, encoder):
        """Test bundling empty vector list."""
        result = encoder.bundle([])
        assert np.all(result == 0)

    def test_encode_set_basic(self, encoder):
        """Test basic set encoding."""
        data = {"a", "b", "c"}
        result = encoder._encode_set(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_set_empty(self, encoder):
        """Test empty set encoding."""
        result = encoder._encode_set(set())
        assert np.all(result == 0)

    def test_encode_set_frozenset(self, encoder):
        """Test frozenset encoding."""
        data = frozenset(["x", "y", "z"])
        result = encoder._encode_set(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_threshold_bipolar(self, encoder):
        """Test bipolar thresholding."""
        # Test with positive values
        vec = np.array([2, -1, 0, 3, -2], dtype=np.int8)
        result = encoder._threshold_bipolar(vec)
        expected = np.array([1, -1, 0, 1, -1], dtype=np.int8)
        np.testing.assert_array_equal(result, expected)

    def test_default_list_mode(self, encoder):
        """Test default list encoding mode."""
        assert encoder.default_list_mode == ListEncodeMode.POSITIONAL

    def test_custom_default_mode(self):
        """Test custom default list mode."""
        vm = VectorManager(dimensions=1000)
        encoder = Encoder(vm, default_list_mode=ListEncodeMode.NGRAM)
        assert encoder.default_list_mode == ListEncodeMode.NGRAM

    def test_encode_tuple(self, encoder):
        """Test tuple encoding (should use list encoding)."""
        data = ("a", "b", "c")
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    # ==========================================================================
    # Negation Primitive Tests
    # ==========================================================================

    def test_negate_subtract_basic(self, encoder):
        """Test basic negation via subtraction."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        ABC = encoder.bundle([A, B, C])
        AC = encoder.negate(ABC, B, method="subtract")

        # B should now have negative similarity
        sim_B = np.dot(AC.astype(float), B.astype(float))
        sim_A = np.dot(AC.astype(float), A.astype(float))
        sim_C = np.dot(AC.astype(float), C.astype(float))

        assert sim_B < 0, "Negated component should have negative similarity"
        assert sim_A > 0, "Non-negated component A should remain positive"
        assert sim_C > 0, "Non-negated component C should remain positive"

    def test_negate_project_method(self, encoder):
        """Test negation via orthogonal projection."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        AB = encoder.bundle([A, B])
        result = encoder.negate(AB, B, method="project")

        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_negate_flip_method(self, encoder):
        """Test negation via sign flipping."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        AB = encoder.bundle([A, B])
        result = encoder.negate(AB, B, method="flip")

        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_negate_invalid_method(self, encoder):
        """Test negation with invalid method raises error."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        with pytest.raises(ValueError, match="Unknown negation method"):
            encoder.negate(A, B, method="invalid")

    def test_negate_preserves_other_components(self, encoder):
        """Test that negation preserves similarity to other components."""
        vecs = [encoder.vector_manager.get_vector(f"v{i}") for i in range(5)]
        superpos = encoder.bundle(vecs)

        # Negate v2
        result = encoder.negate(superpos, vecs[2])

        # v2 should be diminished
        sim_v2_before = np.dot(superpos.astype(float), vecs[2].astype(float))
        sim_v2_after = np.dot(result.astype(float), vecs[2].astype(float))
        assert sim_v2_after < sim_v2_before, "Negated component should be diminished"

        # Other vectors should still have positive similarity
        for i in [0, 1, 3, 4]:
            sim = np.dot(result.astype(float), vecs[i].astype(float))
            assert sim > 0, f"v{i} should still have positive similarity"

    def test_remove_component_alias(self, encoder):
        """Test remove_component is alias for negate with subtract."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        AB = encoder.bundle([A, B])
        result1 = encoder.negate(AB, B, method="subtract")
        result2 = encoder.remove_component(AB, B)

        np.testing.assert_array_equal(result1, result2)

    # ==========================================================================
    # Additional Primitive Tests
    # ==========================================================================

    def test_amplify_increases_similarity(self, encoder):
        """Test that amplify increases similarity to target component."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        ABC = encoder.bundle([A, B, C])
        amplified = encoder.amplify(ABC, B, strength=2.0)

        sim_before = np.dot(ABC.astype(float), B.astype(float))
        sim_after = np.dot(amplified.astype(float), B.astype(float))

        assert sim_after > sim_before, "Amplify should increase similarity"

    def test_prototype_extracts_common(self, encoder):
        """Test that prototype extracts common pattern."""
        common = encoder.vector_manager.get_vector("common")
        u1 = encoder.vector_manager.get_vector("unique1")
        u2 = encoder.vector_manager.get_vector("unique2")
        u3 = encoder.vector_manager.get_vector("unique3")

        v1 = encoder.bundle([common, u1])
        v2 = encoder.bundle([common, u2])
        v3 = encoder.bundle([common, u3])

        proto = encoder.prototype([v1, v2, v3])

        sim_common = np.dot(proto.astype(float), common.astype(float))
        sim_unique = np.dot(proto.astype(float), u1.astype(float))

        assert (
            sim_common > sim_unique
        ), "Prototype should be more similar to common pattern"

    def test_prototype_empty_list(self, encoder):
        """Test prototype with empty list."""
        result = encoder.prototype([])
        assert np.all(result == 0)

    def test_difference_identifies_added(self, encoder):
        """Test that difference identifies added components."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")
        C = encoder.vector_manager.get_vector("C")

        before = encoder.bundle([A, B])
        after = encoder.bundle([A, B, C])

        delta = encoder.difference(before, after)

        sim_C = np.dot(delta.astype(float), C.astype(float))
        sim_A = np.dot(delta.astype(float), A.astype(float))

        assert sim_C > sim_A, "Difference should highlight added component"

    def test_blend_interpolates(self, encoder):
        """Test that blend interpolates between vectors."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        # At alpha=0, should be similar to A
        blend_0 = encoder.blend(A, B, alpha=0.0)
        np.testing.assert_array_equal(blend_0, A)

        # At alpha=1, should be similar to B
        blend_1 = encoder.blend(A, B, alpha=1.0)
        np.testing.assert_array_equal(blend_1, B)

        # At alpha=0.5, should be similar to both
        blend_half = encoder.blend(A, B, alpha=0.5)
        sim_A = np.dot(blend_half.astype(float), A.astype(float))
        sim_B = np.dot(blend_half.astype(float), B.astype(float))
        assert abs(sim_A - sim_B) < 0.1 * max(
            sim_A, sim_B
        ), "Midpoint should be similar to both"

    def test_resonance_extracts_relevant(self, encoder):
        """Test that resonance extracts relevant part."""
        A = encoder.vector_manager.get_vector("A")
        B = encoder.vector_manager.get_vector("B")

        AB = encoder.bundle([A, B])
        a_part = encoder.resonance(AB, A)

        # Resonance should preserve or increase similarity to reference
        sim_to_A = np.dot(a_part.astype(float), A.astype(float))
        sim_to_B = np.dot(a_part.astype(float), B.astype(float))

        # The resonance with A should keep more of A than B
        # (relative similarity to A should be higher than to B)
        assert sim_to_A >= sim_to_B, "Resonance should favor the reference pattern"
        assert isinstance(a_part, np.ndarray)
        assert np.all(np.isin(a_part, [-1, 0, 1]))

    # ==========================================================================
    # Numeric Scalar Marker Tests ($log, $linear)
    # ==========================================================================

    def test_log_marker_basic(self, encoder):
        """Test basic $log marker encoding produces valid vector."""
        result = encoder.encode_data({"$log": 1000})
        assert isinstance(result, np.ndarray)
        assert result.shape == (1000,)
        assert result.dtype == np.int8
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_log_marker_magnitude_similarity(self, encoder):
        """Test that similar magnitudes have high similarity with $log."""
        v100 = encoder.encode_data({"$log": 100})
        v200 = encoder.encode_data({"$log": 200})
        v10000 = encoder.encode_data({"$log": 10000})

        sim_close = np.dot(v100.astype(float), v200.astype(float)) / 1000
        sim_far = np.dot(v100.astype(float), v10000.astype(float)) / 1000

        assert sim_close > sim_far, "Closer magnitudes should have higher similarity"
        assert sim_close > 0.9, "2x ratio should have high similarity"

    def test_log_marker_equal_ratios(self, encoder):
        """Test that equal ratios produce similar similarity drops."""
        v100 = encoder.encode_data({"$log": 100})
        v1000 = encoder.encode_data({"$log": 1000})
        v10000 = encoder.encode_data({"$log": 10000})

        # 100→1000 is 10x, 1000→10000 is also 10x
        sim_100_1000 = np.dot(v100.astype(float), v1000.astype(float)) / 1000
        sim_1000_10000 = np.dot(v1000.astype(float), v10000.astype(float)) / 1000

        # Should be approximately equal (within 10%)
        ratio = sim_100_1000 / sim_1000_10000
        assert 0.9 < ratio < 1.1, "Equal ratios should give similar similarity"

    def test_log_marker_in_record(self, encoder):
        """Test $log marker works correctly in record context."""
        record1 = {"rate_pps": {"$log": 1000}, "src_ip": "10.0.0.1"}
        record2 = {"rate_pps": {"$log": 1100}, "src_ip": "10.0.0.1"}
        record3 = {"rate_pps": {"$log": 100000}, "src_ip": "10.0.0.1"}

        v1 = encoder.encode_data(record1)
        v2 = encoder.encode_data(record2)
        v3 = encoder.encode_data(record3)

        sim_close = np.dot(v1.astype(float), v2.astype(float)) / 1000
        sim_far = np.dot(v1.astype(float), v3.astype(float)) / 1000

        assert sim_close > sim_far, "Similar rates should match better"
        # In record context, similarity is diluted by other fields
        assert sim_close > 0.5, "Similar rates should have reasonable similarity"

    def test_log_marker_with_scale(self, encoder):
        """Test $log marker with custom $scale parameter."""
        # Smaller scale = faster similarity decay
        v100_small = encoder.encode_data({"$log": 100, "$scale": 100})
        v1000_small = encoder.encode_data({"$log": 1000, "$scale": 100})

        v100_large = encoder.encode_data({"$log": 100, "$scale": 5000})
        v1000_large = encoder.encode_data({"$log": 1000, "$scale": 5000})

        sim_small_scale = np.dot(v100_small.astype(float), v1000_small.astype(float))
        sim_large_scale = np.dot(v100_large.astype(float), v1000_large.astype(float))

        assert (
            sim_large_scale > sim_small_scale
        ), "Larger scale should give higher similarity"

    def test_log_marker_handles_small_values(self, encoder):
        """Test $log marker handles small positive values."""
        v_small = encoder.encode_data({"$log": 0.001})
        v_one = encoder.encode_data({"$log": 1})

        assert isinstance(v_small, np.ndarray)
        assert np.all(np.isin(v_small, [-1, 0, 1]))

        # They should still have reasonable similarity structure
        sim = np.dot(v_small.astype(float), v_one.astype(float)) / 1000
        assert sim > 0, "Should have positive similarity"

    def test_log_marker_handles_zero(self, encoder):
        """Test $log marker handles zero (edge case)."""
        # Zero should not crash - uses small epsilon internally
        v_zero = encoder.encode_data({"$log": 0})
        assert isinstance(v_zero, np.ndarray)
        assert np.all(np.isin(v_zero, [-1, 0, 1]))

    def test_linear_marker_basic(self, encoder):
        """Test basic $linear marker encoding produces valid vector."""
        result = encoder.encode_data({"$linear": 100})
        assert isinstance(result, np.ndarray)
        assert result.shape == (1000,)
        assert result.dtype == np.int8
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_linear_marker_equal_differences(self, encoder):
        """Test that closer values have higher similarity with $linear."""
        v0 = encoder.encode_data({"$linear": 0})
        v10 = encoder.encode_data({"$linear": 10})
        v100 = encoder.encode_data({"$linear": 100})

        sim_close = np.dot(v0.astype(float), v10.astype(float)) / 1000
        sim_far = np.dot(v0.astype(float), v100.astype(float)) / 1000

        # Closer values should have higher similarity
        assert sim_close > sim_far, "Closer values should have higher similarity"
        assert sim_close > 0, "Close values should have positive similarity"

    def test_linear_marker_in_record(self, encoder):
        """Test $linear marker works correctly in record context."""
        record = {"temperature": {"$linear": 72.5}, "unit": "fahrenheit"}
        result = encoder.encode_data(record)

        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_string_encoding_vs_log_encoding(self, encoder):
        """Test that string encoding has no magnitude relationship."""
        # String encoding (default for numbers)
        s100 = encoder.encode_data(100)
        s200 = encoder.encode_data(200)

        # Log encoding
        l100 = encoder.encode_data({"$log": 100})
        l200 = encoder.encode_data({"$log": 200})

        sim_string = np.dot(s100.astype(float), s200.astype(float)) / 1000
        sim_log = np.dot(l100.astype(float), l200.astype(float)) / 1000

        # String encoding should be near-orthogonal
        assert abs(sim_string) < 0.1, "String encoding should be orthogonal"
        # Log encoding should have high similarity
        assert sim_log > 0.9, "Log encoding should preserve magnitude similarity"

    def test_is_numeric_scalar_marker_detection(self, encoder):
        """Test _is_numeric_scalar_marker correctly identifies markers."""
        assert encoder._is_numeric_scalar_marker({"$log": 100}) is True
        assert encoder._is_numeric_scalar_marker({"$linear": 100}) is True
        assert encoder._is_numeric_scalar_marker({"$log": 100, "$scale": 500}) is True
        assert encoder._is_numeric_scalar_marker({"foo": "bar"}) is False
        assert encoder._is_numeric_scalar_marker({"$time": 12345}) is False
        assert encoder._is_numeric_scalar_marker(100) is False
        assert encoder._is_numeric_scalar_marker("string") is False

    def test_encode_numeric_scalar_invalid(self, encoder):
        """Test _encode_numeric_scalar raises on invalid input."""
        with pytest.raises(ValueError, match="without \\$log or \\$linear"):
            encoder._encode_numeric_scalar({"foo": "bar"})
