"""
VSA/HDC Encoder for structured data.

This module provides the Encoder class which combines:
- Data encoding (JSON, dicts, lists, walkable objects)
- Algebraic primitives (delegated to primitives.py)
- Streaming operations (delegated to accumulator.py)
- Scalar encoding (delegated to scalar.py)

The Encoder class is the main entry point for most operations.
For lower-level access, import directly from:
- holon.primitives - Core VSA algebra
- holon.accumulator - Streaming operations
- holon.scalar - Continuous value encoding
"""

from datetime import datetime
from enum import Enum
from typing import Any, List, Sequence, Union

import edn_format
import numpy as np
from edn_format.immutable_dict import ImmutableDict

# Import from reorganized modules
from . import accumulator as accum_module
from . import primitives as prim
from . import scalar as scalar_module
from .distance import cosine_similarity
from .vector_manager import VectorManager
from .walkable import Walkable, WalkType, as_walkable


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

    POSITIONAL = "positional"  # Absolute position binding (default)
    CHAINED = "chained"  # Relative chained binding
    NGRAM = "ngram"  # N-gram pairs/triples
    BUNDLE = "bundle"  # Pure bundling (multiset, no order)


class MathematicalPrimitive(str, Enum):
    """Fundamental mathematical encoding primitives."""

    CONVERGENCE_RATE = "convergence_rate"
    ITERATION_COMPLEXITY = "iteration_complexity"
    FREQUENCY_DOMAIN = "frequency_domain"
    AMPLITUDE_SCALE = "amplitude_scale"
    POWER_LAW_EXPONENT = "power_law_exponent"
    CLUSTERING_COEFFICIENT = "clustering_coefficient"
    TOPOLOGICAL_DISTANCE = "topological_distance"
    SELF_SIMILARITY = "self_similarity"


class Encoder:
    """
    VSA/HDC Encoder with algebraic primitives for vector operations.

    The Encoder provides:
    1. **Data Encoding**: Convert structured data to hyperdimensional vectors
    2. **Algebraic Primitives**: Core VSA operations (delegated to primitives.py)
    3. **Streaming Primitives**: Accumulators and decay (delegated to accumulator.py)
    4. **Scalar Encoding**: Continuous values (delegated to scalar.py)

    ## Quick Reference: All Primitives

    ### Core Algebra (Binding & Bundling) - see primitives.py
    - `bind(A, B)` → Element-wise multiply
    - `unbind(AB, A)` → Retrieve B from bound vector
    - `bundle([A, B, C])` → Sum + threshold
    - `negate(ABC, B)` → Remove B's influence
    - `amplify(ABC, B, strength)` → Strengthen B's presence

    ### Pattern Extraction - see primitives.py
    - `prototype([v1, v2, v3])` → Extract consensus pattern
    - `resonance(vec, ref)` → Extract agreeing dimensions
    - `difference(before, after)` → Compute delta
    - `blend(A, B, alpha)` → Weighted interpolation

    ### Streaming & Decay - see accumulator.py
    - `create_accumulator()` → Initialize accumulator
    - `accumulate(accum, vec)` → Add vector
    - `decay(accum, factor)` → Apply exponential decay
    - `normalize_accumulator(accum)` → Get unit vector

    ### Extended Algebra - see primitives.py
    - `similarity_profile(A, B)` → Similarity as vector
    - `attend(query, memory)` → Soft attention
    - `analogy(A, B, C)` → A:B::C:? transfer
    - `project(vec, subspace)` → Subspace projection
    - `segment(stream, window)` → Find breakpoints
    - `invert(vec, codebook)` → Reconstruct components
    - `complexity(vec)` → Entropy measure

    ### Data Encoding
    - `encode_data(data)` → Encode JSON/dict/list
    - `encode_walkable(obj)` → Zero-serialization encoding
    - `encode_scalar(val, mode)` → Continuous values
    - `encode_list(seq, mode)` → Sequence encoding
    """

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

        # Derived marker names
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
        self._mode_marker = f"{marker_prefix}mode"
        self._mode_config_marker = f"{marker_prefix}mode_config"
        # Numeric scalar markers (for magnitude-aware encoding)
        self._log_marker = f"{marker_prefix}log"
        self._linear_marker = f"{marker_prefix}linear"
        self._scale_marker = f"{marker_prefix}scale"

    @property
    def xp(self):
        """Get the appropriate array module (numpy or cupy)."""
        return self.vector_manager.np

    # =========================================================================
    # Data Encoding
    # =========================================================================

    def encode_data(self, data: Any) -> np.ndarray:
        """Encode a data structure into a single vector."""
        return self._encode_recursive(data)

    def encode_walkable(self, data: Any) -> np.ndarray:
        """
        Encode any in-memory data structure using the Walkable interface.

        This is the zero-serialization path: objects don't need to be
        converted to JSON/EDN strings first.
        """
        return self._encode_walkable_recursive(data)

    def _encode_walkable_recursive(
        self, data: Any, list_mode=None, **kwargs
    ) -> np.ndarray:
        """Recursively encode using the Walkable interface."""
        walkable = as_walkable(data)
        wtype = walkable.walk_type()

        if wtype == WalkType.MAP:
            return self._encode_walkable_map(walkable, list_mode=list_mode, **kwargs)
        elif wtype == WalkType.LIST:
            mode = list_mode if list_mode is not None else self.default_list_mode
            return self._encode_walkable_list(walkable, mode=mode, **kwargs)
        elif wtype == WalkType.SET:
            return self._encode_walkable_set(walkable)
        else:  # WalkType.SCALAR
            return self._encode_walkable_scalar(walkable)

    def _encode_walkable_map(
        self, walkable: Walkable, list_mode=None, **kwargs
    ) -> np.ndarray:
        """Encode a map-type walkable by binding keys to values."""
        bound_vectors = []

        for key, value in walkable.walk_map_items():
            effective_list_mode = list_mode
            encode_config = {}

            if isinstance(value, dict):
                if self._time_marker in value:
                    value_vector = self._encode_time(value)
                    key_vector = self._encode_walkable_scalar(as_walkable(key))
                    bound = key_vector * value_vector
                    bound_vectors.append(bound)
                    continue

                if self._is_numeric_scalar_marker(value):
                    value_vector = self._encode_numeric_scalar(value)
                    key_vector = self._encode_walkable_scalar(as_walkable(key))
                    bound = key_vector * value_vector
                    bound_vectors.append(bound)
                    continue

                if self._mode_marker in value:
                    mode_str = value[self._mode_marker]
                    if mode_str in [m.value for m in ListEncodeMode]:
                        effective_list_mode = ListEncodeMode(mode_str)
                    value = {k: v for k, v in value.items() if k != self._mode_marker}

                if self._mode_config_marker in value:
                    encode_config = value[self._mode_config_marker]
                    value = {
                        k: v for k, v in value.items() if k != self._mode_config_marker
                    }

            key_vector = self._encode_walkable_scalar(as_walkable(key))
            value_vector = self._encode_walkable_recursive(
                value, list_mode=effective_list_mode, **encode_config, **kwargs
            )
            bound = key_vector * value_vector
            bound_vectors.append(bound)

        if not bound_vectors:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        bundled = self.xp.sum(self.xp.stack(bound_vectors), axis=0)
        return self._threshold_bipolar(bundled)

    def _encode_walkable_list(
        self, walkable: Walkable, mode: ListEncodeMode, **config
    ) -> np.ndarray:
        """Encode a list-type walkable using the specified mode."""
        items = list(walkable.walk_list_items())
        if not items:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        item_vecs = [self._encode_walkable_recursive(item) for item in items]

        if mode == ListEncodeMode.BUNDLE:
            bundled = self.xp.sum(self.xp.stack(item_vecs), axis=0)
            return self._threshold_bipolar(bundled)

        elif mode == ListEncodeMode.POSITIONAL:
            bound_vectors = []
            for i, item_vector in enumerate(item_vecs):
                pos_vector = self.vector_manager.get_position_vector(i)
                bound = item_vector * pos_vector
                bound_vectors.append(bound)
            bundled = self.xp.sum(self.xp.stack(bound_vectors), axis=0)
            return self._threshold_bipolar(bundled)

        elif mode == ListEncodeMode.CHAINED:
            if len(item_vecs) == 0:
                return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)
            chained = item_vecs[-1]
            for prev in reversed(item_vecs[:-1]):
                chained = self.bind(prev, chained)
            return chained

        elif mode == ListEncodeMode.NGRAM:
            return self._encode_ngram_enhanced(item_vecs, **config)

        else:
            raise ValueError(f"Unknown encoding mode: {mode}")

    def _encode_walkable_set(self, walkable: Walkable) -> np.ndarray:
        """Encode a set-type walkable by bundling items with set indicator."""
        items = list(walkable.walk_set_items())
        if not items:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        set_indicator = self.vector_manager.get_vector("set_indicator")
        item_vectors = [self._encode_walkable_recursive(item) for item in items]
        bundled_items = self.xp.sum(self.xp.stack(item_vectors), axis=0)
        bundled_items = self._threshold_bipolar(bundled_items)
        return set_indicator * bundled_items

    def _encode_walkable_scalar(self, walkable: Walkable) -> np.ndarray:
        """Encode a scalar-type walkable."""
        value = walkable.walk_scalar_value()
        return self._encode_scalar(value)

    def _encode_recursive(self, data: Any, list_mode=None, **kwargs) -> np.ndarray:
        """Recursively encode data structures."""
        if isinstance(data, (dict, ImmutableDict)):
            return self._encode_map(data, list_mode=list_mode, **kwargs)
        elif isinstance(data, (list, tuple)):
            mode = list_mode if list_mode is not None else self.default_list_mode
            return self.encode_list(data, mode=mode, **kwargs)
        elif isinstance(data, (frozenset, set)):
            return self._encode_set(data)
        else:
            return self._encode_scalar(data)

    def _encode_map(
        self, data: Union[dict, ImmutableDict], list_mode=None, **kwargs
    ) -> np.ndarray:
        """Encode a map by binding keys to values."""
        if not data:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        if self._time_marker in data:
            return self._encode_time(data)

        if self._is_numeric_scalar_marker(data):
            return self._encode_numeric_scalar(data)

        bound_vectors = []
        for key, value in data.items():
            effective_list_mode = list_mode
            encode_config = {}
            if isinstance(value, dict):
                if self._time_marker in value:
                    value_vector = self._encode_time(value)
                    key_vector = self._encode_scalar(key)
                    bound = key_vector * value_vector
                    bound_vectors.append(bound)
                    continue

                if self._is_numeric_scalar_marker(value):
                    value_vector = self._encode_numeric_scalar(value)
                    key_vector = self._encode_scalar(key)
                    bound = key_vector * value_vector
                    bound_vectors.append(bound)
                    continue

                if self._mode_marker in value:
                    mode_str = value[self._mode_marker]
                    if mode_str in [m.value for m in ListEncodeMode]:
                        effective_list_mode = ListEncodeMode(mode_str)
                    value = {k: v for k, v in value.items() if k != self._mode_marker}

                if self._mode_config_marker in value:
                    encode_config = value[self._mode_config_marker]
                    value = {
                        k: v for k, v in value.items() if k != self._mode_config_marker
                    }

            key_vector = self._encode_scalar(key)
            value_vector = self._encode_recursive(
                value, list_mode=effective_list_mode, **encode_config, **kwargs
            )
            bound = key_vector * value_vector
            bound_vectors.append(bound)

        bundled = self.xp.sum(self.xp.stack(bound_vectors), axis=0)
        return self._threshold_bipolar(bundled)

    def encode_list(
        self, seq: Sequence[Any], mode: ListEncodeMode | str | None = None, **config
    ) -> np.ndarray:
        """Encode a sequence with configurable encoding mode."""
        if mode is None:
            mode = self.default_list_mode
        elif isinstance(mode, str):
            mode = ListEncodeMode(mode)

        if not seq:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        item_vecs = [self._encode_recursive(item) for item in seq]

        if mode == ListEncodeMode.BUNDLE:
            bundled = self.xp.sum(self.xp.stack(item_vecs), axis=0)
            return self._threshold_bipolar(bundled)

        elif mode == ListEncodeMode.POSITIONAL:
            bound_vectors = []
            for i, item_vector in enumerate(item_vecs):
                pos_vector = self.vector_manager.get_position_vector(i)
                bound = item_vector * pos_vector
                bound_vectors.append(bound)
            bundled = self.xp.sum(self.xp.stack(bound_vectors), axis=0)
            return self._threshold_bipolar(bundled)

        elif mode == ListEncodeMode.CHAINED:
            if len(item_vecs) == 0:
                return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)
            chained = item_vecs[-1]
            for prev in reversed(item_vecs[:-1]):
                chained = self.bind(prev, chained)
            return chained

        elif mode == ListEncodeMode.NGRAM:
            return self._encode_ngram_enhanced(item_vecs, **config)

        else:
            raise ValueError(f"Unknown encoding mode: {mode}")

    def _encode_ngram_enhanced(
        self, item_vecs: List[np.ndarray], **config
    ) -> np.ndarray:
        """Enhanced N-gram encoding with configurable primitives."""
        if len(item_vecs) < 2:
            bundled = (
                self.bundle(item_vecs)
                if item_vecs
                else self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)
            )
            if config.get("length_penalty", False):
                length_factor = 1.0 / np.sqrt(len(item_vecs)) if item_vecs else 1.0
                bundled = bundled * length_factor

            if config.get("term_weighting", False):
                magnitudes = [np.linalg.norm(vec) for vec in item_vecs]
                avg_magnitude = np.mean(magnitudes) if magnitudes else 1.0
                if avg_magnitude > 0:
                    importance_factor = min(avg_magnitude / 2.0, 2.0)
                    bundled = bundled * importance_factor

            return self._threshold_bipolar(bundled)

        n_sizes = config.get("n_sizes", [1, 2])
        weights = config.get("weights", [1.0] * len(n_sizes))
        length_penalty = config.get("length_penalty", False)
        idf_weighting = config.get("idf_weighting", False)
        corpus_stats = config.get("corpus_stats", None)
        term_weighting = config.get("term_weighting", False)

        all_ngrams = []

        for n_size, weight in zip(n_sizes, weights):
            if n_size == 1:
                for vec in item_vecs:
                    weighted_vec = weight * vec

                    if term_weighting:
                        magnitude = np.linalg.norm(vec)
                        density = np.sum(np.abs(vec)) / len(vec)
                        importance_score = (magnitude * density) / 1000.0
                        importance_factor = min(max(importance_score, 0.5), 2.0)
                        weighted_vec = weighted_vec * importance_factor

                    if idf_weighting and corpus_stats:
                        weighted_vec = weighted_vec * 0.8

                    all_ngrams.append(weighted_vec)
            else:
                for i in range(len(item_vecs) - n_size + 1):
                    chained = item_vecs[i]
                    for j in range(1, n_size):
                        chained = self.bind(chained, item_vecs[i + j])

                    weighted_pattern = weight * chained

                    if config.get("positional_weighting", False):
                        position_factor = 1.0 / (i + 1)
                        weighted_pattern = weighted_pattern * position_factor

                    if idf_weighting and corpus_stats:
                        pattern_key = f"ngram_{n_size}_{i}"
                        idf_factor = corpus_stats.get(pattern_key, 1.0)
                        weighted_pattern = weighted_pattern * min(idf_factor, 2.0)

                    all_ngrams.append(weighted_pattern)

        if length_penalty and all_ngrams:
            seq_length = len(item_vecs)
            length_factor = 1.0 / np.sqrt(seq_length)

            normalized_patterns = []
            for i, pattern in enumerate(all_ngrams):
                if i < len(item_vecs):
                    pattern_length_factor = length_factor * 1.2
                else:
                    pattern_length_factor = length_factor * 0.8

                normalized_patterns.append(pattern_length_factor * pattern)

            all_ngrams = normalized_patterns

        if config.get("discrimination_boost", False):
            enhanced_patterns = []
            for pattern in all_ngrams:
                variance = np.var(pattern)
                uniqueness_factor = min(variance / 0.1, 1.5)
                enhanced_patterns.append(pattern * uniqueness_factor)
            all_ngrams = enhanced_patterns

        if all_ngrams:
            bundled = self.xp.sum(self.xp.stack(all_ngrams), axis=0)
            return self._threshold_bipolar(bundled)
        else:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

    def _encode_sequence(self, data: Union[list, tuple]) -> np.ndarray:
        """Encode a sequence using positional mode."""
        return self.encode_list(data, mode=ListEncodeMode.POSITIONAL)

    def _encode_set(self, data: Union[frozenset, set]) -> np.ndarray:
        """Encode a set by bundling items with set indicator."""
        if not data:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)

        set_indicator = self.vector_manager.get_vector("set_indicator")
        item_vectors = [self._encode_recursive(item) for item in data]
        bundled_items = self.xp.sum(self.xp.stack(item_vectors), axis=0)
        bundled_items = self._threshold_bipolar(bundled_items)
        return set_indicator * bundled_items

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
            return self.vector_manager.get_vector(str(data))

    # =========================================================================
    # Numeric Scalar Encoding (magnitude-aware)
    # =========================================================================

    def _encode_numeric_scalar(self, value: dict) -> np.ndarray:
        """
        Encode a numeric value with magnitude-aware encoding.

        Supports two markers:
        - {"$log": 1000} → Log10 encoding (equal ratios = equal similarity)
        - {"$linear": 1000} → Linear positional encoding (equal differences = equal similarity)

        Optional scale parameter:
        - {"$log": 1000, "$scale": 500} → Custom scale for similarity decay

        Args:
            value: Dict with $log or $linear marker

        Returns:
            Bipolar vector where similar magnitudes have similar encodings
        """
        scale = value.get(self._scale_marker, 1000.0)

        if self._log_marker in value:
            num = value[self._log_marker]
            return scalar_module.encode_scalar_log(
                num, self.vector_manager.dimensions, scale
            )
        elif self._linear_marker in value:
            num = value[self._linear_marker]
            return scalar_module.encode_positional(
                num, self.vector_manager.dimensions, scale
            )
        else:
            raise ValueError("_encode_numeric_scalar called without $log or $linear")

    def _is_numeric_scalar_marker(self, value: Any) -> bool:
        """Check if value is a numeric scalar marker dict."""
        if not isinstance(value, dict):
            return False
        return self._log_marker in value or self._linear_marker in value

    # =========================================================================
    # Time Encoding
    # =========================================================================

    def _encode_time(self, value: dict) -> np.ndarray:
        """Encode a $time marked value with circular + positional components."""
        timestamp = value[self._time_marker]
        resolution_str = value.get(self._time_resolution_marker, "hour")
        resolution = TimeResolution(resolution_str)

        if isinstance(timestamp, str):
            ts_clean = timestamp.replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(ts_clean)
                timestamp = dt.timestamp()
            except ValueError:
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d"]:
                    try:
                        dt = datetime.strptime(timestamp, fmt)
                        timestamp = dt.timestamp()
                        break
                    except ValueError:
                        continue
                else:
                    return self._encode_scalar(str(timestamp))

        dt = datetime.fromtimestamp(timestamp)
        dim = self.vector_manager.dimensions
        components = []

        hour_frac = dt.hour + dt.minute / 60 + dt.second / 3600
        components.append(("hour", scalar_module.encode_circular(hour_frac, 24, dim)))

        components.append(("dow", scalar_module.encode_circular(dt.weekday(), 7, dim)))

        month_frac = (dt.month - 1) + (dt.day - 1) / 30
        components.append(("month", scalar_module.encode_circular(month_frac, 12, dim)))

        if resolution == TimeResolution.SECOND:
            position = timestamp
        elif resolution == TimeResolution.MINUTE:
            position = timestamp / 60
        elif resolution == TimeResolution.HOUR:
            position = timestamp / 3600
        else:  # DAY
            position = timestamp / 86400

        components.append(("position", scalar_module.encode_positional(position, dim)))

        result = np.zeros(dim, dtype=np.float64)
        for role_name, vec in components:
            role_vec = self.vector_manager.get_vector(f"__time_role_{role_name}__")
            role_np = role_vec.get() if hasattr(role_vec, "get") else role_vec
            result += role_np.astype(np.float64) * vec.astype(np.float64)

        if self.backend == "gpu" and CUPY_AVAILABLE:
            result = cp.asarray(result)
        return self._threshold_bipolar(result)

    # =========================================================================
    # Mathematical Primitive Encoding
    # =========================================================================

    def encode_mathematical_primitive(
        self, primitive: MathematicalPrimitive, value: Union[int, float]
    ) -> np.ndarray:
        """Encode fundamental mathematical properties."""
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
        similarity_level = int(measure * 3) + 1
        return self.vector_manager.get_vector(
            f"self_similarity_level_{similarity_level}"
        )

    def mathematical_bind(self, *vectors: np.ndarray) -> np.ndarray:
        """Bind mathematical properties together."""
        if not vectors:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)
        result = vectors[0]
        for vec in vectors[1:]:
            result = result * vec
        return self._threshold_bipolar(result)

    def mathematical_bundle(
        self, vectors: List[np.ndarray], weights: List[float] = None
    ) -> np.ndarray:
        """Bundle mathematical properties with optional weighting."""
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

    # =========================================================================
    # Utility
    # =========================================================================

    def _threshold_bipolar(self, vector) -> Union[np.ndarray, "cp.ndarray"]:
        """Threshold summed vector to bipolar {-1, 0, 1}."""
        if self.backend == "gpu" and CUPY_AVAILABLE:
            return cp.where(vector > 0, 1, cp.where(vector < 0, -1, 0)).astype(cp.int8)
        else:
            return np.where(vector > 0, 1, np.where(vector < 0, -1, 0)).astype(np.int8)

    # =========================================================================
    # Scalar Encoding (delegated to scalar.py)
    # =========================================================================

    def encode_scalar(
        self,
        value: float,
        mode: str = "linear",
        scale: float = 10000.0,
        period: float = None,
        seed: int = 42,
    ) -> np.ndarray:
        """Encode a continuous scalar value. See scalar.py for details."""
        return scalar_module.encode_scalar(
            value, self.vector_manager.dimensions, mode, scale, period, seed
        )

    def encode_scalar_log(self, value: float, scale: float = 1000.0) -> np.ndarray:
        """Encode a scalar on log scale. See scalar.py for details."""
        return scalar_module.encode_scalar_log(
            value, self.vector_manager.dimensions, scale
        )

    # =========================================================================
    # Core Algebra (delegated to primitives.py)
    # =========================================================================

    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Bind two vectors. See primitives.py for details."""
        return prim.bind(vec1, vec2)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbind a key from bound vector. See primitives.py for details."""
        return prim.unbind(bound, key)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle vectors. See primitives.py for details."""
        if not vectors:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)
        return prim.bundle(vectors)

    def negate(
        self, superposition: np.ndarray, component: np.ndarray, method: str = "subtract"
    ) -> np.ndarray:
        """Remove component from superposition. See primitives.py for details."""
        return prim.negate(superposition, component, method)

    def remove_component(
        self, superposition: np.ndarray, component: np.ndarray
    ) -> np.ndarray:
        """Alias for negate() with default subtract method."""
        return self.negate(superposition, component, method="subtract")

    def amplify(
        self, superposition: np.ndarray, component: np.ndarray, strength: float = 1.0
    ) -> np.ndarray:
        """Strengthen component in superposition. See primitives.py for details."""
        return prim.amplify(superposition, component, strength)

    def prototype(
        self, vectors: List[np.ndarray], threshold: float = 0.5
    ) -> np.ndarray:
        """Extract common pattern. See primitives.py for details."""
        if not vectors:
            return self.xp.zeros(self.vector_manager.dimensions, dtype=self.xp.int8)
        return prim.prototype(vectors, threshold)

    def prototype_add(
        self, proto: np.ndarray, example: np.ndarray, count: int
    ) -> np.ndarray:
        """Incrementally update prototype. See primitives.py for details."""
        return prim.prototype_add(proto, example, count)

    def resonance(self, vec: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Extract agreeing dimensions. See primitives.py for details."""
        return prim.resonance(vec, reference)

    def difference(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Compute delta between states. See primitives.py for details."""
        return prim.difference(before, after)

    def blend(
        self, vec1: np.ndarray, vec2: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Weighted interpolation. See primitives.py for details."""
        return prim.blend(vec1, vec2, alpha)

    def permute(self, vec: np.ndarray, k: int) -> np.ndarray:
        """Circular shift. See primitives.py for details."""
        if self.backend == "gpu" and CUPY_AVAILABLE:
            return cp.roll(vec, k)
        return prim.permute(vec, k)

    def cleanup(self, noisy: np.ndarray, codebook: List[np.ndarray]) -> np.ndarray:
        """Find closest in codebook. See primitives.py for details."""
        return prim.cleanup(noisy, codebook)

    # =========================================================================
    # Extended Algebra (delegated to primitives.py)
    # =========================================================================

    def similarity_profile(self, vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
        """Similarity as vector. See primitives.py for details."""
        return prim.similarity_profile(vec_a, vec_b)

    def attend(
        self,
        query: np.ndarray,
        memory: np.ndarray,
        strength: float = 1.0,
        mode: str = "soft",
    ) -> np.ndarray:
        """Soft attention. See primitives.py for details."""
        return prim.attend(query, memory, strength, mode)

    def analogy(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        """A:B::C:? transfer. See primitives.py for details."""
        return prim.analogy(a, b, c)

    def project(
        self,
        vec: np.ndarray,
        subspace: List[np.ndarray],
        orthogonalize: bool = True,
    ) -> np.ndarray:
        """Subspace projection. See primitives.py for details."""
        return prim.project(vec, subspace, orthogonalize)

    def conditional_bind(
        self,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
        gate: np.ndarray,
        mode: str = "positive",
    ) -> np.ndarray:
        """Gated binding. See primitives.py for details."""
        return prim.conditional_bind(vec_a, vec_b, gate, mode)

    def segment(
        self,
        stream: List[np.ndarray],
        window: int = 100,
        threshold: float = 0.3,
        method: str = "prototype",
    ) -> List[int]:
        """Find structural breakpoints. See primitives.py for details."""
        return prim.segment(stream, window, threshold, method)

    def complexity(self, vec: np.ndarray) -> float:
        """Entropy measure. See primitives.py for details."""
        return prim.complexity(vec)

    def invert(
        self,
        vec: np.ndarray,
        codebook: List = None,
        top_k: int = 5,
        threshold: float = 0.3,
    ) -> List:
        """Reconstruct components. See primitives.py for details."""
        return prim.invert(vec, codebook, top_k, threshold)

    # =========================================================================
    # Streaming (delegated to accumulator.py)
    # =========================================================================

    def create_accumulator(self) -> np.ndarray:
        """Create empty accumulator. See accumulator.py for details."""
        return accum_module.create_accumulator(self.vector_manager.dimensions)

    def accumulate(self, accumulator: np.ndarray, example: np.ndarray) -> np.ndarray:
        """Add to accumulator. See accumulator.py for details."""
        return accum_module.accumulate(accumulator, example)

    def decay(self, accumulator: np.ndarray, factor: float = 0.99) -> np.ndarray:
        """Apply decay. See accumulator.py for details."""
        return accum_module.decay(accumulator, factor)

    def normalize_accumulator(self, accumulator: np.ndarray) -> np.ndarray:
        """Normalize for queries. See accumulator.py for details."""
        return accum_module.normalize_accumulator(accumulator)

    def threshold_accumulator(self, accumulator: np.ndarray) -> np.ndarray:
        """Convert to bipolar. See accumulator.py for details."""
        return accum_module.threshold_accumulator(accumulator)
