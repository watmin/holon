from enum import Enum
from typing import Any, Dict, List, Sequence, Set, Union

import edn_format
import numpy as np
from edn_format.immutable_dict import ImmutableDict

from .vector_manager import VectorManager

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class ListEncodeMode(str, Enum):
    """Encoding modes for sequences/lists."""

    POSITIONAL = "positional"  # Absolute position binding (default, current behavior)
    CHAINED = (
        "chained"  # Relative chained binding for suffix operations and prefix unbinding
    )
    NGRAM = "ngram"  # N-gram pairs/triples for local order preservation
    BUNDLE = "bundle"  # Pure bundling (multiset, no order)


class MathematicalPrimitive(str, Enum):
    """Fundamental mathematical encoding primitives."""

    CONVERGENCE_RATE = "convergence_rate"  # Mathematical stability analysis
    ITERATION_COMPLEXITY = "iteration_complexity"  # Computational depth encoding
    FREQUENCY_DOMAIN = "frequency_domain"  # Wave frequency properties
    AMPLITUDE_SCALE = "amplitude_scale"  # Energy/magnitude encoding
    POWER_LAW_EXPONENT = "power_law_exponent"  # Scale-free network properties
    CLUSTERING_COEFFICIENT = "clustering_coefficient"  # Local connectivity
    TOPOLOGICAL_DISTANCE = "topological_distance"  # Graph distance metrics
    SELF_SIMILARITY = "self_similarity"  # Fractal dimension properties


class Encoder:
    def __init__(
        self,
        vector_manager: VectorManager,
        default_list_mode: ListEncodeMode = ListEncodeMode.POSITIONAL,
    ):
        self.vector_manager = vector_manager
        self.backend = vector_manager.backend
        self.default_list_mode = default_list_mode

    def encode_data(self, data: Any) -> np.ndarray:
        """
        Encode a data structure into a single vector using binding and bundling,
        preserving structural relationships.

        :param data: Parsed data structure.
        :return: Encoded vector.
        """
        return self._encode_recursive(data)

    def _encode_recursive(self, data: Any, list_mode=None, **kwargs) -> np.ndarray:
        """
        Recursively encode data structures with proper binding for relationships.
        Supports encoding mode hints via _encode_mode key in dicts.
        """
        if isinstance(data, (dict, ImmutableDict)):
            return self._encode_map(data, list_mode=list_mode, **kwargs)
        elif isinstance(data, (list, tuple)):
            # Use provided list_mode or default
            mode = list_mode if list_mode is not None else self.default_list_mode
            return self.encode_list(data, mode=mode, **kwargs)
        elif isinstance(data, (frozenset, set)):
            return self._encode_set(data)
        else:
            # Scalar value
            return self._encode_scalar(data)

    def _encode_map(
        self, data: Union[dict, ImmutableDict], list_mode=None, **kwargs
    ) -> np.ndarray:
        """Encode a map by binding keys to values. Supports encoding mode hints."""
        if not data:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

        bound_vectors = []
        for key, value in data.items():
            # Check for encoding mode and config hints
            effective_list_mode = list_mode
            encode_config = {}
            if isinstance(value, dict):
                if "_encode_mode" in value:
                    mode_str = value["_encode_mode"]
                    if mode_str in [m.value for m in ListEncodeMode]:
                        effective_list_mode = ListEncodeMode(mode_str)
                    # Remove the hint from the value for encoding
                    value = {k: v for k, v in value.items() if k != "_encode_mode"}

                if "_encode_config" in value:
                    encode_config = value["_encode_config"]
                    # Remove the config from the value for encoding
                    value = {k: v for k, v in value.items() if k != "_encode_config"}

            key_vector = self._encode_scalar(key)
            value_vector = self._encode_recursive(
                value, list_mode=effective_list_mode, **encode_config, **kwargs
            )
            # Bind key and value
            bound = key_vector * value_vector
            bound_vectors.append(bound)

        # Bundle all key-value bindings
        bundled = np.sum(bound_vectors, axis=0)
        return self._threshold_bipolar(bundled)

    def encode_list(
        self, seq: Sequence[Any], mode: ListEncodeMode | str | None = None, **config
    ) -> np.ndarray:
        """
        Encode a sequence with configurable encoding mode.

        :param seq: Sequence to encode
        :param mode: Encoding mode (positional, chained, ngram, bundle)
        :param **config: Additional configuration for enhanced modes
        :return: Encoded vector
        """
        if mode is None:
            mode = self.default_list_mode
        elif isinstance(mode, str):
            mode = ListEncodeMode(mode)

        if not seq:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

        item_vecs = [self._encode_recursive(item) for item in seq]

        if mode == ListEncodeMode.BUNDLE:
            # Pure bundling (multiset, no order)
            bundled = np.sum(item_vecs, axis=0)
            return self._threshold_bipolar(bundled)

        elif mode == ListEncodeMode.POSITIONAL:
            # Absolute position binding (original behavior)
            bound_vectors = []
            for i, item_vector in enumerate(item_vecs):
                pos_vector = self.vector_manager.get_position_vector(i)
                bound = item_vector * pos_vector
                bound_vectors.append(bound)
            bundled = np.sum(bound_vectors, axis=0)
            return self._threshold_bipolar(bundled)

        elif mode == ListEncodeMode.CHAINED:
            # Relative chained binding for suffix operations and prefix unbinding
            # Creates: itemN ⊙ (itemN-1 ⊙ (... ⊙ item1))
            # Useful for: suffix matching, prefix removal, sequence reversal operations
            if len(item_vecs) == 0:
                return np.zeros(self.vector_manager.dimensions, dtype=np.int8)
            # Chain from the end for easy unbinding of prefixes
            chained = item_vecs[-1]
            for prev in reversed(item_vecs[:-1]):
                chained = self.bind(prev, chained)
            return chained

        elif mode == ListEncodeMode.NGRAM:
            # Enhanced N-gram encoding with configurable primitives
            return self._encode_ngram_enhanced(item_vecs, **config)

        else:
            raise ValueError(f"Unknown encoding mode: {mode}")

    def _encode_ngram_enhanced(
        self, item_vecs: List[np.ndarray], **config
    ) -> np.ndarray:
        """
        Enhanced N-gram encoding with advanced kernel-level primitives.

        Supports configurable geometric operations for optimal substring matching.
        """
        if len(item_vecs) < 2:
            # For short sequences, apply enhanced single-term processing
            bundled = (
                self.bundle(item_vecs)
                if item_vecs
                else np.zeros(self.vector_manager.dimensions, dtype=np.int8)
            )
            if config.get("length_penalty", False):
                # Apply length normalization for short queries
                length_factor = 1.0 / np.sqrt(len(item_vecs)) if item_vecs else 1.0
                bundled = bundled * length_factor

            # Apply term importance weighting for single terms
            if config.get("term_weighting", False):
                # Weight based on vector magnitude (important terms have stronger vectors)
                magnitudes = [np.linalg.norm(vec) for vec in item_vecs]
                avg_magnitude = np.mean(magnitudes) if magnitudes else 1.0
                if avg_magnitude > 0:
                    importance_factor = min(avg_magnitude / 2.0, 2.0)  # Cap at 2x boost
                    bundled = bundled * importance_factor

            return self._threshold_bipolar(bundled)

        # Extract configuration options
        n_sizes = config.get("n_sizes", [1, 2])  # Individual items + pairs
        weights = config.get("weights", [1.0] * len(n_sizes))
        length_penalty = config.get("length_penalty", False)
        idf_weighting = config.get("idf_weighting", False)
        corpus_stats = config.get("corpus_stats", None)
        term_weighting = config.get("term_weighting", False)  # New primitive

        # Generate n-grams of specified sizes
        all_ngrams = []

        for n_size, weight in zip(n_sizes, weights):
            if n_size == 1:
                # Enhanced unigrams with term importance weighting
                for vec in item_vecs:
                    weighted_vec = weight * vec

                    # Apply term importance weighting
                    if term_weighting:
                        # Weight based on vector density/magnitude
                        magnitude = np.linalg.norm(vec)
                        density = np.sum(np.abs(vec)) / len(vec)
                        importance_score = (
                            magnitude * density
                        ) / 1000.0  # Normalized metric
                        importance_factor = min(
                            max(importance_score, 0.5), 2.0
                        )  # 0.5x to 2x
                        weighted_vec = weighted_vec * importance_factor

                    if idf_weighting and corpus_stats:
                        weighted_vec = weighted_vec * 0.8  # Reduce unigram weight

                    all_ngrams.append(weighted_vec)
            else:
                # Multi-item patterns with enhanced weighting
                for i in range(len(item_vecs) - n_size + 1):
                    # Chain the pattern
                    chained = item_vecs[i]
                    for j in range(1, n_size):
                        chained = self.bind(chained, item_vecs[i + j])

                    # Apply base weighting
                    weighted_pattern = weight * chained

                    # Apply positional weighting (earlier patterns more important)
                    if config.get("positional_weighting", False):
                        position_factor = 1.0 / (i + 1)  # Decay with position
                        weighted_pattern = weighted_pattern * position_factor

                    # Apply IDF weighting if available
                    if idf_weighting and corpus_stats:
                        pattern_key = f"ngram_{n_size}_{i}"
                        idf_factor = corpus_stats.get(pattern_key, 1.0)
                        weighted_pattern = weighted_pattern * min(idf_factor, 2.0)

                    all_ngrams.append(weighted_pattern)

        # Apply sequence-level enhancements
        if length_penalty and all_ngrams:
            # Enhanced length normalization
            seq_length = len(item_vecs)
            length_factor = 1.0 / np.sqrt(seq_length)

            # Apply different normalization for different pattern sizes
            normalized_patterns = []
            for i, pattern in enumerate(all_ngrams):
                # Individual items get slight boost, patterns get slight reduction
                if i < len(item_vecs):  # Individual items
                    pattern_length_factor = length_factor * 1.2
                else:  # Multi-item patterns
                    pattern_length_factor = length_factor * 0.8

                normalized_patterns.append(pattern_length_factor * pattern)

            all_ngrams = normalized_patterns

        # Apply discrimination enhancement
        if config.get("discrimination_boost", False):
            # Boost components that are more unique (higher variance vectors)
            enhanced_patterns = []
            for pattern in all_ngrams:
                variance = np.var(pattern)
                uniqueness_factor = min(variance / 0.1, 1.5)  # Cap at 1.5x boost
                enhanced_patterns.append(pattern * uniqueness_factor)
            all_ngrams = enhanced_patterns

        # Bundle all enhanced components
        if all_ngrams:
            bundled = np.sum(all_ngrams, axis=0)

            return self._threshold_bipolar(bundled)
        else:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Bind two vectors using element-wise multiplication."""
        return vec1 * vec2

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple vectors by summing and thresholding."""
        if not vectors:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)
        bundled = np.sum(vectors, axis=0)
        return self._threshold_bipolar(bundled)

    def _encode_sequence(self, data: Union[list, tuple]) -> np.ndarray:
        """Encode a sequence by binding items to positional vectors and bundling."""
        return self.encode_list(data, mode=ListEncodeMode.POSITIONAL)

    def _encode_set(self, data: Union[frozenset, set]) -> np.ndarray:
        """Encode a set by bundling items with set indicator."""
        if not data:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

        set_indicator = self.vector_manager.get_vector("set_indicator")
        item_vectors = [self._encode_recursive(item) for item in data]
        bundled_items = np.sum(item_vectors, axis=0)
        bundled_items = self._threshold_bipolar(bundled_items)
        # Bind set indicator to bundled items
        return set_indicator * bundled_items

    # Mathematical Primitive Encoding Methods
    def encode_mathematical_primitive(
        self, primitive: MathematicalPrimitive, value: Union[int, float]
    ) -> np.ndarray:
        """
        Encode fundamental mathematical properties.

        These are core VSA/HDC primitives that provide mathematical understanding
        beyond generic structural encoding. Users can compose these to build
        domain-specific semantic encoders.
        """
        if primitive == MathematicalPrimitive.CONVERGENCE_RATE:
            return self._encode_convergence_rate(value)
        elif primitive == MathematicalPrimitive.ITERATION_COMPLEXITY:
            return self._encode_iteration_complexity(value)
        elif primitive == MathematicalPrimitive.FREQUENCY_DOMAIN:
            return self._encode_frequency_domain(value)
        elif primitive == MathematicalPrimitive.AMPLITUDE_SCALE:
            return self._encode_amplitude_scale(value)
        elif primitive == MathematicalPrimitive.POWER_LAW_EXPONENT:
            return self._encode_power_law_exponent(value)
        elif primitive == MathematicalPrimitive.CLUSTERING_COEFFICIENT:
            return self._encode_clustering_coefficient(value)
        elif primitive == MathematicalPrimitive.TOPOLOGICAL_DISTANCE:
            return self._encode_topological_distance(value)
        elif primitive == MathematicalPrimitive.SELF_SIMILARITY:
            return self._encode_self_similarity(value)
        else:
            raise ValueError(f"Unknown mathematical primitive: {primitive}")

    def _encode_convergence_rate(self, rate: float) -> np.ndarray:
        """Encode mathematical convergence properties (fundamental primitive)."""
        # Create more granular categories with specific thresholds
        if rate < 0.2:
            category = "very_slow_convergence"
        elif rate < 0.4:
            category = "slow_convergence"
        elif rate < 0.6:
            category = "moderate_slow_convergence"
        elif rate < 0.8:
            category = "moderate_convergence"
        elif rate < 0.9:
            category = "fast_convergence"
        elif rate < 0.95:
            category = "very_fast_convergence"
        else:
            category = "divergent"

        return self.vector_manager.get_vector(category)

    def _encode_iteration_complexity(self, iterations: int) -> np.ndarray:
        """Encode computational iteration complexity (fundamental primitive)."""
        if iterations < 10:
            category = "low_complexity"
        elif iterations < 50:
            category = "moderate_complexity"
        elif iterations < 200:
            category = "high_complexity"
        else:
            category = "extreme_complexity"

        return self.vector_manager.get_vector(category)

    def _encode_frequency_domain(self, freq: float) -> np.ndarray:
        """Encode frequency domain properties (fundamental primitive)."""
        if freq < 0.01:
            category = "very_low_frequency"
        elif freq < 0.1:
            category = "low_frequency"
        elif freq < 1.0:
            category = "medium_low_frequency"
        elif freq < 10.0:
            category = "medium_frequency"
        elif freq < 100.0:
            category = "high_frequency"
        else:
            category = "ultrasonic_frequency"

        return self.vector_manager.get_vector(category)

    def _encode_amplitude_scale(self, amp: float) -> np.ndarray:
        """Encode amplitude/energy scale (fundamental primitive)."""
        if amp < 0.1:
            category = "micro_scale"
        elif amp < 0.5:
            category = "small_scale"
        elif amp < 2.0:
            category = "medium_scale"
        elif amp < 10.0:
            category = "large_scale"
        else:
            category = "macro_scale"

        return self.vector_manager.get_vector(category)

    def _encode_power_law_exponent(self, exponent: float) -> np.ndarray:
        """Encode power-law scaling properties (fundamental primitive)."""
        if exponent < 2.0:
            category = "shallow_power_law"
        elif exponent < 2.5:
            category = "typical_power_law"
        elif exponent < 3.0:
            category = "steep_power_law"
        else:
            category = "extreme_power_law"

        return self.vector_manager.get_vector(category)

    def _encode_clustering_coefficient(self, coeff: float) -> np.ndarray:
        """Encode local clustering/connectivity (fundamental primitive)."""
        if coeff < 0.2:
            category = "low_clustering"
        elif coeff < 0.5:
            category = "moderate_clustering"
        elif coeff < 0.8:
            category = "high_clustering"
        else:
            category = "extreme_clustering"

        return self.vector_manager.get_vector(category)

    def _encode_topological_distance(self, distance: float) -> np.ndarray:
        """Encode network distance/path properties (fundamental primitive)."""
        if distance > 10:
            category = "long_distance"
        elif distance > 5:
            category = "moderate_distance"
        elif distance > 2:
            category = "short_distance"
        else:
            category = "minimal_distance"

        return self.vector_manager.get_vector(category)

    def _encode_self_similarity(self, measure: float) -> np.ndarray:
        """Encode fractal self-similarity properties (fundamental primitive)."""
        similarity_level = int(measure * 3) + 1  # 1-4 levels
        return self.vector_manager.get_vector(
            f"self_similarity_level_{similarity_level}"
        )

    # Mathematical Composition Primitives
    def mathematical_bind(self, *vectors: np.ndarray) -> np.ndarray:
        """
        Bind mathematical properties together (fundamental composition primitive).

        This provides the mathematical coupling operations needed for semantic encoding,
        such as frequency-amplitude binding in waves or convergence-iteration binding in fractals.
        """
        if not vectors:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

        result = vectors[0]
        for vec in vectors[1:]:
            result = result * vec  # Mathematical binding

        return self._threshold_bipolar(result)

    def mathematical_bundle(
        self, vectors: List[np.ndarray], weights: List[float] = None
    ) -> np.ndarray:
        """
        Bundle mathematical properties with optional weighting (fundamental composition primitive).

        This enables weighted combination of mathematical features, such as
        prioritizing certain properties in similarity calculations.
        """
        if not vectors:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

        if weights is None:
            weights = [1.0] * len(vectors)
        elif len(weights) != len(vectors):
            raise ValueError(
                f"Weights length ({len(weights)}) must match vectors length ({len(vectors)})"
            )

        weighted_sum = np.zeros(self.vector_manager.dimensions, dtype=np.float32)
        for vec, weight in zip(vectors, weights):
            weighted_sum += weight * vec.astype(np.float32)

        return self._threshold_bipolar(weighted_sum)

    def _encode_scalar(self, data: Any) -> np.ndarray:
        """Encode a scalar value."""
        if isinstance(data, str):
            return self.vector_manager.get_vector(data)
        elif isinstance(data, (int, float)):
            return self.vector_manager.get_vector(str(data))
        elif isinstance(data, edn_format.Keyword):
            return self.vector_manager.get_vector(f":{data.name}")
        elif isinstance(data, edn_format.Symbol):
            return self.vector_manager.get_vector(data.name)
        elif data is None:
            return self.vector_manager.get_vector("nil")
        elif isinstance(data, bool):
            return self.vector_manager.get_vector("true" if data else "false")
        elif isinstance(data, edn_format.Char):
            return self.vector_manager.get_vector(str(data))
        else:
            # Fallback for unknown types
            return self.vector_manager.get_vector(str(data))

    def _threshold_bipolar(self, vector) -> Union[np.ndarray, "cp.ndarray"]:
        """Threshold summed vector to bipolar {-1, 0, 1}."""
        if self.backend == "gpu" and CUPY_AVAILABLE:
            return cp.where(vector > 0, 1, cp.where(vector < 0, -1, 0)).astype(cp.int8)
        else:
            return np.where(vector > 0, 1, np.where(vector < 0, -1, 0)).astype(np.int8)
