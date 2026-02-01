from datetime import datetime
from enum import Enum
from typing import Any, List, Sequence, Union

import edn_format
import numpy as np
from edn_format.immutable_dict import ImmutableDict

from .vector_manager import VectorManager


class TimeResolution(str, Enum):
    """Resolution levels for time encoding."""

    SECOND = "second"  # High-frequency logs, events
    MINUTE = "minute"  # Transactions, API calls
    HOUR = "hour"  # Business data, orders (default)
    DAY = "day"  # Reports, aggregates


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
        marker_prefix: str = "$",
    ):
        self.vector_manager = vector_manager
        self.backend = vector_manager.backend
        self.default_list_mode = default_list_mode
        self.marker_prefix = marker_prefix
        
        # Derived marker names (user-configurable to avoid conflicts with data)
        self._time_marker = f"{marker_prefix}time"
        self._time_resolution_marker = f"{marker_prefix}time_resolution"
        self._any_marker = f"{marker_prefix}any"
        self._not_marker = f"{marker_prefix}not"
        self._or_marker = f"{marker_prefix}or"
        self._gt_marker = f"{marker_prefix}gt"
        self._gte_marker = f"{marker_prefix}gte"
        self._lt_marker = f"{marker_prefix}lt"
        self._lte_marker = f"{marker_prefix}lte"
        self._in_marker = f"{marker_prefix}in"
        self._contains_marker = f"{marker_prefix}contains"
        self._exists_marker = f"{marker_prefix}exists"

    @property
    def xp(self):
        """Get the appropriate array module (numpy or cupy) for the backend."""
        return self.vector_manager.np

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
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        # Check for $time marker at top level of this dict
        if self._time_marker in data:
            return self._encode_time(data)

        bound_vectors = []
        for key, value in data.items():
            # Check for encoding mode and config hints
            effective_list_mode = list_mode
            encode_config = {}
            if isinstance(value, dict):
                # Check for $time marker in nested value
                if self._time_marker in value:
                    value_vector = self._encode_time(value)
                    key_vector = self._encode_scalar(key)
                    bound = key_vector * value_vector
                    bound_vectors.append(bound)
                    continue

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
        bundled = self.xp.sum(self.xp.stack(bound_vectors), axis=0)
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
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        item_vecs = [self._encode_recursive(item) for item in seq]

        if mode == ListEncodeMode.BUNDLE:
            # Pure bundling (multiset, no order)
            bundled = self.xp.sum(self.xp.stack(item_vecs), axis=0)
            return self._threshold_bipolar(bundled)

        elif mode == ListEncodeMode.POSITIONAL:
            # Absolute position binding (original behavior)
            bound_vectors = []
            for i, item_vector in enumerate(item_vecs):
                pos_vector = self.vector_manager.get_position_vector(i)
                bound = item_vector * pos_vector
                bound_vectors.append(bound)
            bundled = self.xp.sum(self.xp.stack(bound_vectors), axis=0)
            return self._threshold_bipolar(bundled)

        elif mode == ListEncodeMode.CHAINED:
            # Relative chained binding for suffix operations and prefix unbinding
            # Creates: itemN ⊙ (itemN-1 ⊙ (... ⊙ item1))
            # Useful for: suffix matching, prefix removal, sequence reversal operations
            if len(item_vecs) == 0:
                return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)
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
                else self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)
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
            bundled = self.xp.sum(self.xp.stack(all_ngrams), axis=0)

            return self._threshold_bipolar(bundled)
        else:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Bind two vectors using element-wise multiplication."""
        return vec1 * vec2

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple vectors by summing and thresholding."""
        if not vectors:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)
        bundled = self.xp.sum(self.xp.stack(vectors), axis=0)
        return self._threshold_bipolar(bundled)

    def negate(
        self, superposition: np.ndarray, component: np.ndarray, method: str = "subtract"
    ) -> np.ndarray:
        """
        Remove a component's influence from a superposition (NOT operation).

        This extends VSA with negation capability. Traditional VSA only has:
        - Binding (AND): A ⊙ B
        - Bundling (OR): A + B

        Negation provides:
        - NOT: A - B (removes B's influence from A)

        Args:
            superposition: The vector to remove from (e.g., bundle([A, B, C]))
            component: The vector to remove (e.g., B)
            method: "subtract" (default), "project", or "flip"
                - subtract: Simple subtraction, fast, effective
                - project: Orthogonal projection, mathematically cleaner
                - flip: Flip signs where component is strong

        Returns:
            Vector with component's influence removed

        Example:
            >>> A, B, C = [encoder.vector_manager.get_vector(x) for x in "ABC"]
            >>> ABC = encoder.bundle([A, B, C])
            >>> AC = encoder.negate(ABC, B)  # Removes B's influence
            >>> similarity(AC, B) < 0  # B now has negative similarity
            >>> similarity(AC, A) > 0  # A is preserved
        """
        sup = superposition.astype(float)
        comp = component.astype(float)

        if method == "subtract":
            # Simple subtraction - most effective for VSA
            result = sup - comp

        elif method == "project":
            # Orthogonal projection - mathematically cleaner
            comp_norm = np.linalg.norm(comp)
            if comp_norm < 1e-10:
                return superposition
            comp_unit = comp / comp_norm
            projection = np.dot(sup, comp_unit) * comp_unit
            result = sup - projection

        elif method == "flip":
            # Flip signs where component is strong
            result = sup.copy()
            mask = comp > 0
            result[mask] = -result[mask]

        else:
            raise ValueError(f"Unknown negation method: {method}")

        return self._threshold_bipolar(result)

    def remove_component(
        self, superposition: np.ndarray, component: np.ndarray
    ) -> np.ndarray:
        """
        Alias for negate() with default subtract method.

        Removes a component from a superposition.
        """
        return self.negate(superposition, component, method="subtract")

    def amplify(
        self, superposition: np.ndarray, component: np.ndarray, strength: float = 1.0
    ) -> np.ndarray:
        """
        Strengthen a component's presence in a superposition.

        Opposite of negate - makes a component MORE prominent.

        Args:
            superposition: The vector containing multiple components
            component: The component to amplify
            strength: How much to boost (1.0 = double, 2.0 = triple, etc.)

        Returns:
            Vector with component's influence strengthened

        Example:
            >>> ABC = encoder.bundle([A, B, C])
            >>> sim(ABC, B) = 0.53
            >>> amplified = encoder.amplify(ABC, B, strength=2.0)
            >>> sim(amplified, B) = 0.87  # 64% boost!
        """
        result = superposition.astype(float) + strength * component.astype(float)
        return self._threshold_bipolar(result)

    def prototype(
        self, vectors: List[np.ndarray], threshold: float = 0.5
    ) -> np.ndarray:
        """
        Extract the common pattern from a set of vectors.

        Keeps only dimensions where a majority of vectors agree.
        Useful for finding what's shared across examples.

        Args:
            vectors: List of vectors to find consensus from
            threshold: Fraction of vectors that must agree (0.5 = majority)

        Returns:
            Vector representing the common pattern

        Example:
            >>> # Three vectors, each with 'common' component plus unique parts
            >>> v1 = bundle([common, unique1])
            >>> v2 = bundle([common, unique2])
            >>> v3 = bundle([common, unique3])
            >>> proto = encoder.prototype([v1, v2, v3])
            >>> sim(proto, common) = 0.79  # High - shared pattern
            >>> sim(proto, unique1) = 0.28  # Low - not shared
        """
        if not vectors:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        # Sum all vectors (convert to float for precision)
        stacked = self.xp.stack([v.astype(self.xp.float32) for v in vectors])
        total = self.xp.sum(stacked, axis=0)

        # Threshold: keep only where absolute majority agrees
        n = len(vectors)
        agreement_threshold = n * threshold

        result = self.xp.zeros_like(total)
        result[total > agreement_threshold] = 1
        result[total < -agreement_threshold] = -1

        return result.astype(self.xp.int8)

    def difference(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Compute what changed between two states.

        Returns a vector representing the change/delta.

        Args:
            before: The original state
            after: The new state

        Returns:
            Vector highlighting what was added (positive) or removed (negative)

        Example:
            >>> before = bundle([A, B])
            >>> after = bundle([A, B, C])
            >>> delta = encoder.difference(before, after)
            >>> sim(delta, C) = 0.74  # High - C was added
            >>> sim(delta, A) = -0.15  # Negative - A was already there
        """
        delta = after.astype(float) - before.astype(float)
        return self._threshold_bipolar(delta)

    def blend(
        self, vec1: np.ndarray, vec2: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Weighted interpolation between two vectors.

        Creates a smooth transition between concepts.

        Args:
            vec1: First vector (alpha=0 returns this)
            vec2: Second vector (alpha=1 returns this)
            alpha: Interpolation factor (0.0 to 1.0)

        Returns:
            Interpolated vector

        Example:
            >>> blend(A, B, 0.0)  # Returns A
            >>> blend(A, B, 0.5)  # Midpoint, similar to both
            >>> blend(A, B, 1.0)  # Returns B
        """
        result = (1 - alpha) * vec1.astype(float) + alpha * vec2.astype(float)
        return self._threshold_bipolar(result)

    def resonance(self, vec: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Extract the part of vec that resonates with reference.

        Keeps only dimensions where both vectors agree.
        Useful for extracting the "relevant" part of a signal.

        Args:
            vec: Vector to filter
            reference: Reference pattern to resonate with

        Returns:
            Vector containing only the resonating components

        Example:
            >>> AB = bundle([A, B])
            >>> a_part = encoder.resonance(AB, A)
            >>> sim(a_part, A) = 0.82  # Higher than original
            >>> sim(AB, A) = 0.67      # Original similarity
        """
        v = vec.astype(float)
        r = reference.astype(float)

        # Where they agree (same sign), keep the value
        agree = (v * r) > 0
        result = np.zeros_like(v)
        result[agree] = v[agree]

        return self._threshold_bipolar(result)

    def _encode_sequence(self, data: Union[list, tuple]) -> np.ndarray:
        """Encode a sequence by binding items to positional vectors and bundling."""
        return self.encode_list(data, mode=ListEncodeMode.POSITIONAL)

    def _encode_set(self, data: Union[frozenset, set]) -> np.ndarray:
        """Encode a set by bundling items with set indicator."""
        if not data:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        set_indicator = self.vector_manager.get_vector("set_indicator")
        item_vectors = [self._encode_recursive(item) for item in data]
        bundled_items = self.xp.sum(self.xp.stack(item_vectors), axis=0)
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

    # ==================== TIME ENCODING ====================

    def _encode_time(self, value: dict) -> np.ndarray:
        """
        Encode a $time marked value with circular + positional components.

        Circular encoding captures periodic patterns:
        - Hour of day (24-hour cycle)
        - Day of week (7-day cycle)
        - Month of year (12-month cycle)

        Positional encoding captures linear time progression.

        Usage:
            {"$time": 1706500000}  # Unix timestamp
            {"$time": "2024-01-29T10:30:00Z"}  # ISO string
            {"$time": 1706500000, "$time_resolution": "minute"}
            
        Note: The "$time" marker prefix is configurable via marker_prefix.
        """
        timestamp = value[self._time_marker]
        resolution_str = value.get(self._time_resolution_marker, "hour")
        resolution = TimeResolution(resolution_str)

        # Parse timestamp if string
        if isinstance(timestamp, str):
            # Handle ISO format strings
            ts_clean = timestamp.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(ts_clean)
                timestamp = dt.timestamp()
            except ValueError:
                # Fallback: try parsing common formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        timestamp = dt.timestamp()
                        break
                    except ValueError:
                        continue
                else:
                    # Can't parse, encode as string
                    return self._encode_scalar(str(timestamp))

        dt = datetime.fromtimestamp(timestamp)
        dim = self.vector_manager.dimensions
        components = []

        # Circular components (periodic patterns)
        # Hour of day with fractional minutes
        hour_frac = dt.hour + dt.minute / 60 + dt.second / 3600
        components.append(("hour", self._encode_circular(hour_frac, 24, dim)))

        # Day of week (0=Monday, 6=Sunday)
        components.append(("dow", self._encode_circular(dt.weekday(), 7, dim)))

        # Month of year with fractional days
        month_frac = (dt.month - 1) + (dt.day - 1) / 30
        components.append(("month", self._encode_circular(month_frac, 12, dim)))

        # Positional component (linear time)
        if resolution == TimeResolution.SECOND:
            position = timestamp
        elif resolution == TimeResolution.MINUTE:
            position = timestamp / 60
        elif resolution == TimeResolution.HOUR:
            position = timestamp / 3600
        else:  # DAY
            position = timestamp / 86400

        components.append(("position", self._encode_positional(position, dim)))

        # Bind each component with its role vector and bundle
        result = np.zeros(dim, dtype=np.float64)
        for role_name, vec in components:
            role_vec = self.vector_manager.get_vector(f"__time_role_{role_name}__")
            # Convert to numpy for consistent math (role_vec may be cupy)
            role_np = role_vec.get() if hasattr(role_vec, 'get') else role_vec
            result += role_np.astype(np.float64) * vec.astype(np.float64)

        # Convert result to backend type before thresholding
        if self.backend == "gpu" and CUPY_AVAILABLE:
            result = cp.asarray(result)
        return self._threshold_bipolar(result)

    def _encode_circular(
        self, value: float, period: float, dim: int, seed: int = 42
    ) -> np.ndarray:
        """
        Encode a value on a circle with given period.

        Values that are close on the circle will have similar encodings.
        The encoding wraps: value 0 is similar to value `period`.
        """
        rng = np.random.default_rng(seed + int(period * 1000))
        angle = 2 * np.pi * value / period

        # Random phase offsets for each dimension
        phases = rng.uniform(0, 2 * np.pi, dim)

        # Project angle onto random directions
        return np.sign(np.cos(angle + phases)).astype(np.int8)

    def _encode_positional(
        self, position: float, dim: int, scale: float = 10000
    ) -> np.ndarray:
        """
        Transformer-style positional encoding for linear time.

        Nearby positions have similar encodings, with gradual decay.
        """
        indices = np.arange(dim)
        freqs = 1 / (scale ** (indices / dim))

        # Alternate sin/cos
        values = np.where(
            indices % 2 == 0,
            np.sin(position * freqs),
            np.cos(position * freqs),
        )

        return np.sign(values).astype(np.int8)

    def _threshold_bipolar(self, vector) -> Union[np.ndarray, "cp.ndarray"]:
        """Threshold summed vector to bipolar {-1, 0, 1}."""
        if self.backend == "gpu" and CUPY_AVAILABLE:
            return cp.where(vector > 0, 1, cp.where(vector < 0, -1, 0)).astype(cp.int8)
        else:
            return np.where(vector > 0, 1, np.where(vector < 0, -1, 0)).astype(np.int8)

    # =========================================================================
    # Additional Primitives
    # =========================================================================

    def permute(self, vec: np.ndarray, k: int) -> np.ndarray:
        """
        Circular shift (permutation) of vector dimensions.
        
        Used for positional encoding in sequences:
            sequence = bundle([permute(A, 0), permute(B, 1), permute(C, 2)])
        
        And for "what comes after X?" queries:
            # If sequence = A + permute(B, 1) + permute(C, 2)
            # unbind with permute(X, -1) to get "what follows X"
        
        :param vec: Input vector
        :param k: Shift amount (positive = right, negative = left)
        :return: Shifted vector
        """
        if self.backend == "gpu" and CUPY_AVAILABLE:
            return cp.roll(vec, k)
        return np.roll(vec, k)

    def cleanup(self, noisy: np.ndarray, codebook: List[np.ndarray]) -> np.ndarray:
        """
        Find the closest vector in codebook to the noisy input.
        
        Useful for denoising composed vectors before further operations:
            query = bundle([signal1, signal2, noise])
            clean = cleanup(query, [proto_a, proto_b, proto_c])
            result = amplify(base, clean, 0.5)
        
        :param noisy: Noisy or composed input vector
        :param codebook: List of clean/known vectors to match against
        :return: The codebook vector with highest similarity to noisy
        """
        if not codebook:
            return noisy
        
        best_vec = codebook[0]
        best_sim = -float('inf')
        
        for vec in codebook:
            # Normalized dot product similarity
            noisy_norm = noisy / (np.linalg.norm(noisy) + 1e-10)
            vec_norm = vec / (np.linalg.norm(vec) + 1e-10)
            sim = float(np.dot(noisy_norm, vec_norm))
            
            if sim > best_sim:
                best_sim = sim
                best_vec = vec
        
        return best_vec

    def prototype_add(
        self, prototype: np.ndarray, example: np.ndarray, count: int
    ) -> np.ndarray:
        """
        Incrementally update a prototype with a new example.
        
        Instead of re-computing prototype([all_examples]), you can:
            proto = prototype([ex1, ex2, ex3])  # Initial, n=3
            proto = prototype_add(proto, ex4, 3)  # Now n=4
            proto = prototype_add(proto, ex5, 4)  # Now n=5
        
        :param prototype: Existing prototype vector
        :param example: New example to incorporate
        :param count: Number of examples already in prototype (before this one)
        :return: Updated prototype incorporating the new example
        """
        # Weighted average: (proto * count + example) / (count + 1)
        # Then threshold back to bipolar
        weighted = prototype.astype(np.float32) * count + example.astype(np.float32)
        averaged = weighted / (count + 1)
        return self._threshold_bipolar(averaged)
