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
        data = {
            "sequence": {"_encode_mode": "ngram", "data": ["word1", "word2", "word3"]}
        }
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_mode_hint_chained(self, encoder):
        """Test chained encoding mode hint."""
        data = {
            "words": {
                "_encode_mode": "chained",
                "sequence": ["first", "second", "third"],
            }
        }
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_mode_hint_bundle(self, encoder):
        """Test bundle encoding mode hint."""
        data = {"tags": {"_encode_mode": "bundle", "items": ["tag1", "tag2", "tag3"]}}
        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_invalid_mode_hint(self, encoder):
        """Test invalid encoding mode hint (should ignore and use default)."""
        data = {"sequence": {"_encode_mode": "invalid_mode", "data": ["a", "b", "c"]}}
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
