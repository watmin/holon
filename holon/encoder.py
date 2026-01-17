import numpy as np
import edn_format
from edn_format.immutable_dict import ImmutableDict
from typing import Dict, Any, Set, Union
from .vector_manager import VectorManager

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class Encoder:
    def __init__(self, vector_manager: VectorManager):
        self.vector_manager = vector_manager
        self.backend = vector_manager.backend

    def encode_data(self, data: Any) -> np.ndarray:
        """
        Encode a data structure into a single vector using binding and bundling,
        preserving structural relationships.

        :param data: Parsed data structure.
        :return: Encoded vector.
        """
        return self._encode_recursive(data)

    def _encode_recursive(self, data: Any) -> np.ndarray:
        """
        Recursively encode data structures with proper binding for relationships.
        """
        if isinstance(data, (dict, ImmutableDict)):
            return self._encode_map(data)
        elif isinstance(data, (list, tuple)):
            return self._encode_sequence(data)
        elif isinstance(data, (frozenset, set)):
            return self._encode_set(data)
        else:
            # Scalar value
            return self._encode_scalar(data)

    def _encode_map(self, data: Union[dict, ImmutableDict]) -> np.ndarray:
        """Encode a map by binding keys to values."""
        if not data:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

        bound_vectors = []
        for key, value in data.items():
            key_vector = self._encode_scalar(key)
            value_vector = self._encode_recursive(value)
            # Bind key and value
            bound = key_vector * value_vector
            bound_vectors.append(bound)

        # Bundle all key-value bindings
        bundled = np.sum(bound_vectors, axis=0)
        return self._threshold_bipolar(bundled)

    def _encode_sequence(self, data: Union[list, tuple]) -> np.ndarray:
        """Encode a sequence by bundling the encoded items."""
        if not data:
            return np.zeros(self.vector_manager.dimensions, dtype=np.int8)

        item_vectors = [self._encode_recursive(item) for item in data]
        # Bundle all item vectors
        bundled = np.sum(item_vectors, axis=0)
        return self._threshold_bipolar(bundled)

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

    def _threshold_bipolar(self, vector) -> Union[np.ndarray, 'cp.ndarray']:
        """Threshold summed vector to bipolar {-1, 0, 1}."""
        if self.backend == 'gpu' and CUPY_AVAILABLE:
            return cp.where(vector > 0, 1, cp.where(vector < 0, -1, 0)).astype(cp.int8)
        else:
            return np.where(vector > 0, 1, np.where(vector < 0, -1, 0)).astype(np.int8)