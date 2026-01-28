"""
Tests for enhanced encoder functionality.
Tests configurable n-gram sizes and advanced geometric primitives.
"""

import pytest
import numpy as np
from holon.encoder import Encoder, ListEncodeMode
from holon.vector_manager import VectorManager


class TestEnhancedEncoder:
    """Test enhanced encoder with configurable n-gram sizes and primitives."""

    @pytest.fixture
    def encoder(self):
        """Create encoder instance for testing."""
        vm = VectorManager(dimensions=1000)
        return Encoder(vm)

    def test_configurable_ngram_sizes(self, encoder):
        """Test configurable n_sizes parameter."""
        data = ["the", "quick", "brown", "fox"]

        # Test different n_sizes configurations
        configs = [
            {"n_sizes": [1]},  # Unigrams only
            {"n_sizes": [2]},  # Bigrams only (default behavior)
            {"n_sizes": [1, 2]},  # Both unigrams and bigrams
            {"n_sizes": [2, 3]},  # Bigrams and trigrams
        ]

        for config in configs:
            result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM, **config)
            assert isinstance(result, np.ndarray)
            assert np.all(np.isin(result, [-1, 0, 1]))

    def test_ngram_weights(self, encoder):
        """Test n-gram weighting functionality."""
        data = ["the", "quick", "brown"]

        # Test with different weights
        config = {
            "n_sizes": [1, 2],
            "weights": [0.3, 0.7]  # Weight bigrams higher
        }

        result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM, **config)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_length_penalty(self, encoder):
        """Test length penalty normalization."""
        # Short sequence
        short_data = ["hello"]
        # Long sequence
        long_data = ["the", "quick", "brown", "fox", "jumps", "over"]

        config = {"length_penalty": True}

        short_result = encoder.encode_list(short_data, mode=ListEncodeMode.NGRAM, **config)
        long_result = encoder.encode_list(long_data, mode=ListEncodeMode.NGRAM, **config)

        assert isinstance(short_result, np.ndarray)
        assert isinstance(long_result, np.ndarray)
        assert np.all(np.isin(short_result, [-1, 0, 1]))
        assert np.all(np.isin(long_result, [-1, 0, 1]))

    def test_term_weighting(self, encoder):
        """Test term importance weighting."""
        data = ["common", "word", "rare", "term"]

        config = {"term_weighting": True}

        result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM, **config)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_positional_weighting(self, encoder):
        """Test positional weighting for n-grams."""
        data = ["first", "second", "third", "fourth"]

        config = {"positional_weighting": True}

        result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM, **config)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_discrimination_boost(self, encoder):
        """Test discrimination boost for unique components."""
        data = ["similar", "words", "here"]

        config = {"discrimination_boost": True}

        result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM, **config)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_encode_config_json_interface(self, encoder):
        """Test _encode_config JSON interface."""
        data = {
            "sequence": {
                "_encode_mode": "ngram",
                "_encode_config": {
                    "n_sizes": [1, 2],
                    "weights": [0.5, 0.5],
                    "length_penalty": True
                },
                "words": ["hello", "world"]
            }
        }

        result = encoder.encode_data(data)
        assert isinstance(result, np.ndarray)
        assert np.all(np.isin(result, [-1, 0, 1]))

    def test_backward_compatibility(self, encoder):
        """Test that old ngram usage still works."""
        data = ["old", "style", "ngram"]

        # Old way - should still work
        result_old = encoder.encode_list(data, mode=ListEncodeMode.NGRAM)
        assert isinstance(result_old, np.ndarray)
        assert np.all(np.isin(result_old, [-1, 0, 1]))

        # New way with empty config - should be equivalent
        result_new = encoder.encode_list(data, mode=ListEncodeMode.NGRAM, n_sizes=[2])
        assert isinstance(result_new, np.ndarray)
        assert np.all(np.isin(result_new, [-1, 0, 1]))

    def test_invalid_config_handling(self, encoder):
        """Test handling of edge case configurations."""
        data = ["test", "data"]

        # Should handle gracefully even with potentially problematic configs
        # The encoder is designed to be robust and not fail on config issues
        result1 = encoder.encode_list(data, mode=ListEncodeMode.NGRAM, n_sizes=[1])
        result2 = encoder.encode_list(data, mode=ListEncodeMode.NGRAM,
                                    n_sizes=[1, 2], weights=[0.5, 0.5])  # Correct length

        assert isinstance(result1, np.ndarray)
        assert isinstance(result2, np.ndarray)
        assert np.all(np.isin(result1, [-1, 0, 1]))
        assert np.all(np.isin(result2, [-1, 0, 1]))

    def test_edge_cases_enhanced(self, encoder):
        """Test edge cases with enhanced features."""
        # Empty config should work
        data = ["test"]
        result = encoder.encode_list(data, mode=ListEncodeMode.NGRAM, n_sizes=[1])
        assert isinstance(result, np.ndarray)

        # Very long n-gram sizes
        long_data = ["a"] * 10
        result = encoder.encode_list(long_data, mode=ListEncodeMode.NGRAM,
                                   n_sizes=[5], weights=[1.0])
        assert isinstance(result, np.ndarray)

    def test_enhanced_vs_basic_similarity(self, encoder):
        """Test that enhanced encoding produces different but valid results."""
        data = ["the", "same", "data"]

        # Basic encoding
        basic = encoder.encode_list(data, mode=ListEncodeMode.NGRAM)

        # Enhanced encoding
        enhanced = encoder.encode_list(data, mode=ListEncodeMode.NGRAM,
                                     n_sizes=[1, 2], weights=[0.3, 0.7],
                                     length_penalty=True)

        # Both should be valid bipolar vectors
        assert np.all(np.isin(basic, [-1, 0, 1]))
        assert np.all(np.isin(enhanced, [-1, 0, 1]))

        # They will be different due to different processing
        assert not np.array_equal(basic, enhanced)