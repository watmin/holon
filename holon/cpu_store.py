import json
import logging
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

from .atomizer import parse_data
from .encoder import Encoder
from .similarity import find_similar_vectors
from .store import Store
from .vector_manager import VectorManager

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

ANN_THRESHOLD = 1000  # Switch to ANN when > 1000 items


class CPUStore(Store):
    def __init__(self, dimensions: int = 16000, backend: str = "auto"):
        self.dimensions = dimensions

        # Auto-select backend
        if backend == "auto":
            try:
                import cupy as cp

                try:
                    cp.cuda.runtime.getDeviceCount()  # Check GPU availability
                    self.backend = "gpu"
                    print("🎮 Auto-selected GPU backend")
                except cp.cuda.runtime.CUDARuntimeError:
                    self.backend = "cpu"
                    print("💻 Auto-selected CPU backend (no GPU available)")
            except ImportError:
                self.backend = "cpu"
                print("💻 Auto-selected CPU backend (cupy not available)")
        else:
            self.backend = backend

        self.vector_manager = VectorManager(dimensions, self.backend)
        self.encoder = Encoder(self.vector_manager)
        self.stored_data: Dict[str, Dict[str, Any]] = {}  # id -> original data dict
        self.stored_vectors: Dict[str, Any] = {}  # id -> encoded vector

        # ANN indexing
        self.ann_index = None
        self.ann_ids: List[str] = []  # Ordered list of IDs for FAISS index mapping

        # Bulk insert optimization
        self.bulk_mode = False  # When True, defer ANN rebuilds during rapid insertions
        self.ann_vectors = None  # Numpy array for FAISS

    def _build_ann_index(self):
        """Build FAISS ANN index when dataset grows large."""
        if not FAISS_AVAILABLE or len(self.stored_vectors) <= ANN_THRESHOLD:
            return

        # Convert stored vectors to numpy array
        vectors_list = []
        ids_list = []
        for data_id, vec in self.stored_vectors.items():
            if isinstance(vec, np.ndarray):
                vectors_list.append(vec.astype(np.float32))
            else:
                # Handle cupy, convert to numpy
                vectors_list.append(vec.get().astype(np.float32))
            ids_list.append(data_id)

        self.ann_vectors = np.stack(vectors_list)
        self.ann_ids = ids_list

        # Create FAISS index for inner product (dot product)
        dim = self.dimensions
        self.ann_index = faiss.IndexFlatIP(dim)
        self.ann_index.add(self.ann_vectors)

        logging.info(f"ANN index built with {len(self.ann_ids)} vectors")

    def insert(self, data: str, data_type: str = "json") -> str:
        import time

        start = time.time()
        parsed = parse_data(data, data_type)
        parse_time = time.time() - start

        start = time.time()
        encoded_vector = self.encoder.encode_data(parsed)
        encode_time = time.time() - start

        data_id = str(uuid.uuid4())
        self.stored_data[data_id] = parsed
        self.stored_vectors[data_id] = encoded_vector

        # Invalidate ANN index if it exists (unless in bulk mode)
        if self.ann_index is not None and not self.bulk_mode:
            self.ann_index = None
            self.ann_vectors = None
            self.ann_ids = []

        # Log timing for first few inserts
        if len(self.stored_data) <= 5:
            logging.info(
                f"INSERT_TIMING: parse={parse_time:.4f}s, "
                f"encode={encode_time:.4f}s, total={parse_time+encode_time:.4f}s"
            )

        return data_id

    def query(
        self,
        probe: str,
        data_type: str = "json",
        top_k: int = 10,
        threshold: float = 0.0,
        guard=None,
        negations=None,
        any_marker="$any",
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        parsed_probe = parse_data(probe, data_type)

        # Handle $or disjunctions
        if "$or" in parsed_probe and isinstance(parsed_probe["$or"], list):
            all_results = []
            seen_ids = set()
            for sub_probe in parsed_probe["$or"]:
                sub_results = self.query(
                    json.dumps(sub_probe), data_type, top_k, threshold, guard, negations
                )
                for res in sub_results:
                    if res[0] not in seen_ids:
                        all_results.append(res)
                        seen_ids.add(res[0])
            return all_results[:top_k]  # Limit to top_k

        # Handle user-specified any wildcards
        clean_probe = {}
        for k, v in parsed_probe.items():
            if isinstance(v, dict) and any_marker in v:
                continue  # Skip for encoding
            clean_probe[k] = v

        try:
            probe_vector = self.encoder.encode_data(clean_probe)
        except Exception as e:
            raise ValueError(
                f"Failed to encode query probe: {e}. "
                "Check that your query structure matches the expected data format (JSON or EDN)."
            )

        # Parse user-specified $not markers in negations
        negation_specs = []
        cleaned_negations = {}

        def parse_negations(neg, cleaned, path=""):
            for k, v in neg.items():
                if isinstance(v, dict) and "$not" in v:
                    not_val = v["$not"]
                    if isinstance(not_val, list):
                        for val in not_val:
                            negation_specs.append((path + k, val))
                        cleaned[k] = not_val[0]  # For vector, use first
                    else:
                        negation_specs.append((path + k, not_val))
                        cleaned[k] = not_val
                elif isinstance(v, dict):
                    sub_clean = {}
                    parse_negations(v, sub_clean, path + k + ".")
                    if sub_clean:
                        cleaned[k] = sub_clean
                else:
                    cleaned[k] = v

        if negations:
            parse_negations(negations, cleaned_negations)

        # Vector-level negation via subtraction (encode cleaned)
        if cleaned_negations:
            neg_vector = self.encoder.encode_data(cleaned_negations)
            probe_vector = probe_vector - neg_vector

        # Data-based negation check
        def matches_negation(data, specs):
            for path, value in specs:
                keys = path.split(".")
                current = data
                try:
                    for key in keys:
                        current = current[key]
                    if current == value:
                        return True
                except (KeyError, TypeError):
                    pass
            return False

        negation_filters = negation_specs

        # Helper for guard matching
        def is_subset(guard, data):
            # Handle top-level $or in guards for powerful OR logic
            if "$or" in guard and isinstance(guard["$or"], list):
                # Any of the OR conditions must match
                return any(
                    is_subset(or_condition, data) for or_condition in guard["$or"]
                )

            for key, value in guard.items():
                if key not in data:
                    return False
                if isinstance(value, dict):
                    # Handle nested $or
                    if "$or" in value and isinstance(value["$or"], list):
                        # For nested $or, any of the conditions for this key must match
                        if not any(
                            is_subset({key: or_val}, data) for or_val in value["$or"]
                        ):
                            return False
                    elif not isinstance(data[key], dict) or not is_subset(
                        value, data[key]
                    ):
                        return False
                elif isinstance(value, list):
                    # Support OR logic: if guard has a list and data has a scalar that's IN
                    # the list,
                    # treat it as "match any of these values" (backward compatibility)
                    data_value = data[key]
                    if isinstance(data_value, list):
                        # Exact array matching for array-to-array comparison
                        # (backward compatibility)
                        if len(value) != len(data_value):
                            return False
                        for g_item, d_item in zip(value, data_value):
                            if isinstance(g_item, dict) and any_marker in g_item:
                                continue
                            elif g_item != d_item:
                                return False
                    else:
                        # OR logic: scalar data value must be IN the guard list
                        if data_value not in value:
                            return False
                elif value is not None and data[key] != value:
                    return False
            return True

        # Use ANN if available and dataset is large
        if FAISS_AVAILABLE and len(self.stored_vectors) > ANN_THRESHOLD:
            if self.ann_index is None:
                self._build_ann_index()

            if self.ann_index is not None:
                # Ensure probe_vector is numpy float32
                if isinstance(probe_vector, np.ndarray):
                    query_vec = probe_vector.astype(np.float32).reshape(1, -1)
                else:
                    query_vec = probe_vector.get().astype(np.float32).reshape(1, -1)

                # FAISS search returns scores and indices
                scores, indices = self.ann_index.search(query_vec, top_k)

                similar_ids_scores = []
                for i, idx in enumerate(indices[0]):
                    if idx != -1:  # Valid index
                        score = (
                            float(scores[0][i]) / self.dimensions
                        )  # Normalize like dot similarity
                        if score >= threshold:
                            data_id = self.ann_ids[idx]
                            similar_ids_scores.append((data_id, score))

                similar_ids_scores.sort(key=lambda x: x[1], reverse=True)
            else:
                # Fallback to brute-force
                similar_ids_scores = find_similar_vectors(
                    probe_vector, self.stored_vectors, top_k, threshold
                )
        else:
            # Use brute-force for small datasets
            similar_ids_scores = find_similar_vectors(
                probe_vector, self.stored_vectors, top_k, threshold
            )

        results = []
        for data_id, score in similar_ids_scores:
            data_dict = self.stored_data[data_id]
            # Apply negations
            if negation_filters and matches_negation(data_dict, negation_filters):
                continue
            # Apply guard if provided (data structure matching)
            if guard and not is_subset(guard, data_dict):
                continue  # Skip if guard fails
            results.append((data_id, score, data_dict))
        return results

    def get(self, data_id: str) -> Dict[str, Any]:
        if data_id not in self.stored_data:
            raise KeyError(
                f"Data ID '{data_id}' not found. "
                "Make sure the data was previously inserted and not deleted."
            )
        return self.stored_data[data_id]

    def delete(self, data_id: str) -> bool:
        if data_id in self.stored_data:
            del self.stored_data[data_id]
            del self.stored_vectors[data_id]
            return True
        return False

    def clear(self):
        """Clear all stored data (for testing)."""
        self.stored_data.clear()
        self.stored_vectors.clear()
        self.ann_index = None
        self.ann_ids.clear()
        self.ann_vectors = None

    def start_bulk_insert(self):
        """Enter bulk insert mode: defer ANN index rebuilding for faster insertions."""
        self.bulk_mode = True
        # Invalidate any existing index to ensure consistency
        self.ann_index = None
        self.ann_vectors = None
        self.ann_ids = []

    def end_bulk_insert(self):
        """Exit bulk insert mode and rebuild ANN index if needed."""
        self.bulk_mode = False
        # Force rebuild on next query if above threshold
        if len(self.stored_data) >= ANN_THRESHOLD:
            self._build_ann_index()

    def batch_insert(self, items: List[str], data_type: str = "json") -> List[str]:
        """
        Insert multiple items efficiently by deferring ANN rebuilds.

        :param items: List of data strings to insert.
        :param data_type: 'json' or 'edn'.
        :return: List of IDs for the inserted items.
        """
        self.start_bulk_insert()
        ids = []
        for item in items:
            id_ = self.insert(item, data_type)
            ids.append(id_)
        self.end_bulk_insert()
        return ids
