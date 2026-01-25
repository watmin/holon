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
    CHAINED = "chained"  # Relative chained binding for fuzzy subsequence matching
    NGRAM = "ngram"  # N-gram pairs/triples for local order preservation
    BUNDLE = "bundle"  # Pure bundling (multiset, no order)


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
            return self.encode_list(data, mode=mode)
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
            # Check for encoding mode hint
            effective_list_mode = list_mode
            if isinstance(value, dict) and "_encode_mode" in value:
                mode_str = value["_encode_mode"]
                if mode_str in [m.value for m in ListEncodeMode]:
                    effective_list_mode = ListEncodeMode(mode_str)
                # Remove the hint from the value for encoding
                value = {k: v for k, v in value.items() if k != "_encode_mode"}

            key_vector = self._encode_scalar(key)
            value_vector = self._encode_recursive(
                value, list_mode=effective_list_mode, **kwargs
            )
            # Bind key and value
            bound = key_vector * value_vector
            bound_vectors.append(bound)

        # Bundle all key-value bindings
        bundled = np.sum(bound_vectors, axis=0)
        return self._threshold_bipolar(bundled)

    def encode_list(
        self, seq: Sequence[Any], mode: ListEncodeMode | str | None = None
    ) -> np.ndarray:
        """
        Encode a sequence with configurable encoding mode.

        :param seq: Sequence to encode
        :param mode: Encoding mode (positional, chained, ngram, bundle)
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
            # Relative chained binding for subsequence matching
            if len(item_vecs) == 0:
                return np.zeros(self.vector_manager.dimensions, dtype=np.int8)
            # Chain from the end for easy unbinding of prefixes
            chained = item_vecs[-1]
            for prev in reversed(item_vecs[:-1]):
                chained = self.bind(prev, chained)
            return chained

        elif mode == ListEncodeMode.NGRAM:
            # N-gram encoding (pairs by default, with singles for robustness)
            if len(seq) < 2:
                return (
                    self.bundle(item_vecs)
                    if item_vecs
                    else np.zeros(self.vector_manager.dimensions, dtype=np.int8)
                )

            # Create bigram bindings
            pairs = []
            for i in range(len(item_vecs) - 1):
                pair_bound = self.bind(item_vecs[i], item_vecs[i + 1])
                pairs.append(pair_bound)

            # Bundle pairs + singles for better recall
            all_components = pairs + item_vecs
            bundled = np.sum(all_components, axis=0)
            return self._threshold_bipolar(bundled)

        else:
            raise ValueError(f"Unknown encoding mode: {mode}")

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
