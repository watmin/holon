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

    # Enhanced modes
    NGRAM_CONFIGURABLE = "ngram_configurable"  # Configurable N-gram sizes
    NGRAM_WEIGHTED = "ngram_weighted"  # TF-IDF style bigram weighting
    SUBSEQUENCE_ALIGNED = "subsequence_aligned"  # Advanced subsequence matching
    # SEMANTIC_FIELD = "semantic_field"  # Disabled - domain-specific semantic clustering
    # CONTEXT_WINDOW = "context_window"  # Disabled - context-aware window encoding


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

        # elif mode == EnhancedListEncodeMode.SEMANTIC_FIELD:
        #     return self._encode_semantic_field(item_vecs, **config)

        # elif mode == EnhancedListEncodeMode.CONTEXT_WINDOW:
        #     return self._encode_context_window(item_vecs, **config)

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

    def _encode_semantic_field(
        self,
        item_vecs: List[np.ndarray],
        semantic_groups: Optional[Dict[str, List[str]]] = None,
        field_weights: Optional[Dict[str, float]] = None,
    ) -> np.ndarray:
        """
        Semantic field encoding - group related terms into geometric fields.

        This primitive enables domain-specific term relationships, allowing
        synonyms and related concepts to cluster geometrically in hyperspace.

        Args:
            item_vecs: Encoded item vectors
            semantic_groups: Dict mapping field names to lists of related terms
            field_weights: Weights for different semantic fields

        Returns:
            Vector representing semantic field relationships
        """
        if not semantic_groups:
            # Default semantic groups for demonstration
            semantic_groups = {
                "mathematical": [
                    "calculus",
                    "differential",
                    "integral",
                    "derivative",
                    "function",
                ],
                "logical": ["therefore", "hence", "thus", "consequently", "follows"],
                "temporal": ["before", "after", "then", "now", "when", "time"],
                "spatial": ["above", "below", "beside", "between", "around"],
            }

        if not field_weights:
            field_weights = {field: 1.0 for field in semantic_groups.keys()}

        # Create semantic field vectors
        field_vectors = []
        for field_name, related_terms in semantic_groups.items():
            # Get vectors for all terms in this semantic field
            field_term_vecs = []
            for term in related_terms:
                term_vec = self.vector_manager.get_vector(term)
                field_term_vecs.append(term_vec)

            if field_term_vecs:
                # Bundle all terms in this field
                field_bundle = np.sum(field_term_vecs, axis=0)
                field_bundle = self._threshold_bipolar(field_bundle)

                # Weight the field
                weight = field_weights.get(field_name, 1.0)
                field_vectors.append(weight * field_bundle)

        # Bundle all semantic fields
        if field_vectors:
            semantic_bundle = np.sum(field_vectors, axis=0)
            semantic_bundle = self._threshold_bipolar(semantic_bundle)

            # Bind with original sequence for context
            if item_vecs:
                sequence_bundle = np.sum(item_vecs, axis=0)
                sequence_bundle = self._threshold_bipolar(sequence_bundle)
                return self.bind(sequence_bundle, semantic_bundle)
            else:
                return semantic_bundle
        else:
            # Fallback to regular bundling
            if item_vecs:
                return np.sum(item_vecs, axis=0)
            else:
                return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

    def _encode_context_window(
        self,
        item_vecs: List[np.ndarray],
        window_size: int = 3,
        stride: int = 1,
        window_weighting: str = "uniform",
    ) -> np.ndarray:
        """
        Context window encoding - encode sequences with surrounding context.

        This primitive creates overlapping windows of vectors, allowing geometric
        similarity to capture contextual relationships for better substring matching.

        Args:
            item_vecs: Encoded item vectors
            window_size: Size of context window
            stride: Step size for sliding window
            window_weighting: How to weight vectors in window ('uniform', 'gaussian', 'positional')

        Returns:
            Vector representing contextual encoding
        """
        if len(item_vecs) < window_size:
            # Too short, fall back to bundling
            return np.sum(item_vecs, axis=0)

        windows = []

        # Create sliding windows
        for i in range(0, len(item_vecs) - window_size + 1, stride):
            window_vecs = item_vecs[i : i + window_size]

            # Apply window weighting
            if window_weighting == "uniform":
                weighted_vecs = window_vecs
            elif window_weighting == "gaussian":
                # Gaussian weighting - center vectors more important
                weights = self._gaussian_weights(window_size)
                weighted_vecs = [w * v for w, v in zip(weights, window_vecs)]
            elif window_weighting == "positional":
                # Position-based weighting - earlier positions more important
                weights = [1.0 / (j + 1) for j in range(window_size)]
                weighted_vecs = [w * v for w, v in zip(weights, window_vecs)]
            else:
                weighted_vecs = window_vecs

            # Chain the window vectors geometrically
            if len(weighted_vecs) == 1:
                window_encoding = weighted_vecs[0]
            else:
                window_encoding = weighted_vecs[0]
                for vec in weighted_vecs[1:]:
                    window_encoding = self.bind(window_encoding, vec)

            windows.append(window_encoding)

        # Bundle all windows
        if windows:
            result = np.sum(windows, axis=0)
            return self._threshold_bipolar(result)
        else:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

    def _gaussian_weights(self, size: int) -> List[float]:
        """Generate Gaussian weights for window."""
        import math

        sigma = size / 4.0  # Spread weight across window
        center = (size - 1) / 2.0

        weights = []
        for i in range(size):
            x = i - center
            weight = math.exp(-(x**2) / (2 * sigma**2))
            weights.append(weight)

        # Normalize
        total = sum(weights)
        return [w / total for w in weights]

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
