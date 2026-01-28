"""
Enhanced Encoder Primitives for Advanced Geometric Operations
Extends holon kernel with primitives for better substring matching and ranking.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np

from .encoder import Encoder, ListEncodeMode
from .vector_manager import VectorManager


class EnhancedListEncodeMode(str, Enum):
    """Extended encoding modes with advanced primitives."""

    # Existing modes
    POSITIONAL = "positional"
    CHAINED = "chained"
    NGRAM = "ngram"
    BUNDLE = "bundle"

    # New enhanced modes
    NGRAM_CONFIGURABLE = "ngram_configurable"  # Configurable N-gram sizes
    NGRAM_WEIGHTED = "ngram_weighted"  # TF-IDF style bigram weighting
    SUBSEQUENCE_ALIGNED = "subsequence_aligned"  # Advanced subsequence matching


class EnhancedEncoder(Encoder):
    """
    Enhanced encoder with advanced geometric primitives.

    Provides the "rock solid kernel" primitives that empower userland developers
    to build sophisticated geometric applications without traditional fallbacks.
    """

    def __init__(self, vector_manager: VectorManager):
        super().__init__(vector_manager)

    def encode_list(
        self, seq: List[Any], mode: Union[str, EnhancedListEncodeMode] = None, **config
    ) -> np.ndarray:
        """
        Enhanced list encoding with configurable primitives.

        Args:
            seq: Sequence to encode
            mode: Encoding mode
            **config: Mode-specific configuration options
        """
        if mode is None:
            mode = self.default_list_mode
        elif isinstance(mode, str):
            mode = EnhancedListEncodeMode(mode)

        if not seq:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

        item_vecs = [self._encode_recursive(item) for item in seq]

        # Handle enhanced modes
        if mode == EnhancedListEncodeMode.NGRAM_CONFIGURABLE:
            return self._encode_ngram_configurable(item_vecs, **config)

        elif mode == EnhancedListEncodeMode.NGRAM_WEIGHTED:
            return self._encode_ngram_weighted(item_vecs, **config)

        elif mode == EnhancedListEncodeMode.SUBSEQUENCE_ALIGNED:
            return self._encode_subsequence_aligned(item_vecs, **config)

        else:
            # Fall back to base encoder for standard modes
            return super().encode_list(seq, mode)

    def _encode_ngram_configurable(
        self,
        item_vecs: List[np.ndarray],
        n_sizes: List[int] = None,
        weights: List[float] = None,
    ) -> np.ndarray:
        """
        Configurable N-gram encoding - choose which N values to use.

        Args:
            item_vecs: Encoded item vectors
            n_sizes: List of N-gram sizes (e.g., [1, 2, 3] for unigrams, bigrams, trigrams)
            weights: Weights for each N-gram size
        """
        if n_sizes is None:
            n_sizes = [2]  # Default to bigrams only

        if weights is None:
            weights = [1.0] * len(n_sizes)

        if len(weights) != len(n_sizes):
            raise ValueError("weights must match n_sizes length")

        all_ngrams = []

        for n, weight in zip(n_sizes, weights):
            if n == 1:
                # Unigrams
                weighted_unigrams = [weight * vec for vec in item_vecs]
                all_ngrams.extend(weighted_unigrams)
            else:
                # N-grams
                for i in range(len(item_vecs) - n + 1):
                    ngram_vecs = item_vecs[i : i + n]
                    # Chain the n-gram
                    chained = ngram_vecs[0]
                    for vec in ngram_vecs[1:]:
                        chained = self.bind(chained, vec)
                    weighted_ngram = weight * chained
                    all_ngrams.append(weighted_ngram)

        # Bundle all n-grams
        if all_ngrams:
            bundled = np.sum(all_ngrams, axis=0)
            return self._threshold_bipolar(bundled)
        else:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

    def _encode_ngram_weighted(
        self,
        item_vecs: List[np.ndarray],
        corpus_stats: Optional[Dict[str, float]] = None,
        length_penalty: bool = True,
    ) -> np.ndarray:
        """
        TF-IDF style weighted N-gram encoding.

        Args:
            item_vecs: Encoded item vectors
            corpus_stats: Pre-computed corpus statistics for IDF weighting
            length_penalty: Apply length normalization
        """
        if len(item_vecs) < 2:
            # Fall back to bundling for short sequences
            bundled = np.sum(item_vecs, axis=0)
            return self._threshold_bipolar(bundled)

        # Create bigrams with optional weighting
        bigrams = []
        for i in range(len(item_vecs) - 1):
            bigram = self.bind(item_vecs[i], item_vecs[i + 1])

            # Apply IDF-style weighting if corpus stats available
            if corpus_stats:
                bigram_key = f"{i}:{i+1}"  # Simple key for demo
                idf_weight = corpus_stats.get(bigram_key, 1.0)
                bigram = bigram * min(idf_weight, 3.0)  # Cap at 3x boost

            bigrams.append(bigram)

        # Add unigrams with lower weight
        unigrams = [0.5 * vec for vec in item_vecs]  # Lower weight for singles

        # Combine
        all_components = bigrams + unigrams

        # Apply length penalty if requested
        if length_penalty and len(all_components) > 0:
            length_factor = 1.0 / np.sqrt(len(all_components))
            all_components = [length_factor * comp for comp in all_components]

        bundled = np.sum(all_components, axis=0)
        return self._threshold_bipolar(bundled)

    def _encode_subsequence_aligned(
        self,
        item_vecs: List[np.ndarray],
        alignment_mode: str = "sliding_window",
        window_size: int = 3,
    ) -> np.ndarray:
        """
        Advanced subsequence alignment encoding.

        Args:
            item_vecs: Encoded item vectors
            alignment_mode: Type of alignment ('sliding_window', 'best_match')
            window_size: Size of sliding window
        """
        if len(item_vecs) < window_size:
            # Too short, fall back to bundling
            bundled = np.sum(item_vecs, axis=0)
            return self._threshold_bipolar(bundled)

        if alignment_mode == "sliding_window":
            # Create overlapping windows and encode each
            windows = []
            for i in range(len(item_vecs) - window_size + 1):
                window_vecs = item_vecs[i : i + window_size]

                # Chain the window
                chained = window_vecs[0]
                for vec in window_vecs[1:]:
                    chained = self.bind(chained, vec)

                # Weight by position (earlier windows more important)
                position_weight = 1.0 / (i + 1)  # Decay with position
                windows.append(position_weight * chained)

            # Bundle all windows
            bundled = np.sum(windows, axis=0)
            return self._threshold_bipolar(bundled)

        elif alignment_mode == "best_match":
            # Find "best" subsequence pattern (simplified)
            # This could be extended with more sophisticated geometric alignment
            if len(item_vecs) >= 3:
                # Prefer trigrams over bigrams for longer sequences
                trigram = self.bind(self.bind(item_vecs[0], item_vecs[1]), item_vecs[2])
                return trigram
            else:
                # Fall back to bigram
                return self.bind(item_vecs[0], item_vecs[1])

        else:
            raise ValueError(f"Unknown alignment_mode: {alignment_mode}")

    def compute_enhanced_similarity(
        self,
        query_vector: np.ndarray,
        target_vector: np.ndarray,
        similarity_mode: str = "cosine",
        **params,
    ) -> float:
        """
        Enhanced similarity computation with advanced scoring.

        Args:
            query_vector: Query vector
            target_vector: Target vector
            similarity_mode: Type of similarity ('cosine', 'length_normalized', 'contiguous_bonus')
            **params: Additional parameters for similarity computation
        """
        if similarity_mode == "cosine":
            return self._cosine_similarity(query_vector, target_vector)

        elif similarity_mode == "length_normalized":
            # Normalize by vector magnitudes to reduce length bias
            query_norm = np.linalg.norm(query_vector)
            target_norm = np.linalg.norm(target_vector)

            if query_norm == 0 or target_norm == 0:
                return 0.0

            # Length-normalized similarity
            length_factor = min(query_norm, target_norm) / max(query_norm, target_norm)
            cosine_sim = np.dot(query_vector, target_vector) / (
                query_norm * target_norm
            )

            return cosine_sim * length_factor

        elif similarity_mode == "contiguous_bonus":
            # Base cosine similarity
            base_sim = self._cosine_similarity(query_vector, target_vector)

            # Add bonus for contiguous matches (simplified geometric approach)
            # This could be enhanced with actual bigram overlap detection
            contiguous_bonus = params.get("contiguous_bonus", 0.1)
            overlap_factor = params.get("overlap_factor", 0.5)

            # Simplified: assume some overlap exists and apply bonus
            enhanced_sim = base_sim + (contiguous_bonus * overlap_factor)
            return min(enhanced_sim, 1.0)  # Cap at 1.0

        else:
            raise ValueError(f"Unknown similarity_mode: {similarity_mode}")

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)
