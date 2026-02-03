import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from .atomizer import parse_data
from .encoder import Encoder
from .similarity import find_similar_vectors
from .vector_manager import VectorManager

# Optional TorchHD backend
try:
    from .torchhd_encoder import TorchHDEncoder

    TORCHHD_AVAILABLE = True
except ImportError:
    TorchHDEncoder = None
    TORCHHD_AVAILABLE = False

# =============================================================================
# Store Base Class (ABC)
# =============================================================================


class Store(ABC):
    """Abstract base class for Holon storage backends."""

    @abstractmethod
    def insert(self, data: str, data_type: str = "json") -> str:
        """
        Insert a data blob (JSON or EDN string) into the store.
        Returns a unique ID for the inserted data.

        :param data: The data blob as a string.
        :param data_type: 'json' or 'edn'.
        :return: Unique identifier for the data.
        """
        pass

    @abstractmethod
    def query(
        self,
        probe: str,
        data_type: str = "json",
        top_k: int = 10,
        threshold: float = 0.0,
        guard: Callable = None,
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Query the store with a probe data blob.
        Returns a list of (id, similarity_score, original_data) tuples for top matches
        that pass the guard.

        :param probe: The query probe as a string.
        :param data_type: 'json' or 'edn'.
        :param top_k: Number of top results to return.
        :param threshold: Minimum similarity score to include in results.
        :param guard: Optional dict for subset matching to filter results post-similarity.
        :return: List of tuples (data_id, score, data_dict).
        """
        pass

    @abstractmethod
    def get(self, data_id: str) -> Dict[str, Any]:
        """
        Retrieve original data by ID.

        :param data_id: Unique identifier.
        :return: Original data as a dictionary.
        """
        pass

    @abstractmethod
    def delete(self, data_id: str) -> bool:
        """
        Delete data by ID.

        :param data_id: Unique identifier.
        :return: True if deleted, False otherwise.
        """
        pass


# =============================================================================
# CPU/GPU Store Implementation
# =============================================================================

try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False

ANN_THRESHOLD = 1000  # Switch to ANN when > 1000 items


class CPUStore(Store):
    def __init__(
        self, dimensions: int = 16000, backend: str = "auto", marker_prefix: str = "$"
    ):
        """
        Initialize a CPU-based vector store.

        :param dimensions: Vector dimensionality (4096 recommended for most use cases)
        :param backend: "cpu", "gpu", "torchhd", or "auto"
        :param marker_prefix: Prefix for special markers like $time, $any, $gt, etc.
                             Change this if your data legitimately contains keys like "$time".
                             Example: marker_prefix="$$" makes the time marker "$$time"
        """
        self.dimensions = dimensions
        self._use_torchhd = False
        self.marker_prefix = marker_prefix

        # Environment variable override for testing
        env_backend = os.environ.get("HOLON_BACKEND")
        if env_backend:
            backend = env_backend

        # Auto-select backend
        if backend == "torchhd":
            if not TORCHHD_AVAILABLE:
                raise ImportError(
                    "TorchHD backend requested but torch-hd not installed. Run: pip install torch-hd"
                )
            self.backend = "torchhd"
            self._use_torchhd = True
            self.encoder = TorchHDEncoder(
                dimensions=dimensions, marker_prefix=marker_prefix
            )
            self.vector_manager = None  # Not used with TorchHD
            print("🔥 Using TorchHD backend")
        elif backend == "auto":
            # Default to CPU - fastest for individual operations
            # GPU backends have transfer overhead that kills throughput for typical workloads
            # Use backend="gpu" or backend="torchhd" explicitly when needed
            self.backend = "cpu"
            self.vector_manager = VectorManager(dimensions, self.backend)
            self.encoder = Encoder(self.vector_manager, marker_prefix=marker_prefix)
        else:
            self.backend = backend
            self.vector_manager = VectorManager(dimensions, self.backend)
            self.encoder = Encoder(self.vector_manager, marker_prefix=marker_prefix)

        self.stored_data: Dict[str, Dict[str, Any]] = {}  # id -> original data dict
        self.stored_vectors: Dict[str, Any] = {}  # id -> encoded vector (numpy arrays)

        # ANN indexing options
        self.ann_index = None
        self.ann_ids: List[str] = []  # Ordered list of IDs for FAISS index mapping
        self.ann_vectors = None  # Numpy array for FAISS

        # ANN control flags
        self.bulk_mode = False  # When True, defer ANN rebuilds during rapid insertions
        self.ann_enabled = True  # Master switch to enable/disable ANN indexing
        self.ann_auto_rebuild = False  # If True, rebuild on every insert (slow); if False, lazy rebuild on query

    def _build_ann_index(self):
        """Build FAISS ANN index when dataset grows large."""
        if not FAISS_AVAILABLE or not self.ann_enabled:
            return
        if len(self.stored_vectors) <= ANN_THRESHOLD:
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

        # Convert TorchHD tensors to numpy for storage
        if self._use_torchhd and hasattr(encoded_vector, "cpu"):
            encoded_vector = encoded_vector.cpu().numpy()

        data_id = str(uuid.uuid4())
        # Store original string and type for accurate retrieval
        self.stored_data[data_id] = {
            "_raw": data,
            "_type": data_type,
            "_parsed": parsed,
        }
        self.stored_vectors[data_id] = encoded_vector

        # Invalidate ANN index if it exists (unless in bulk mode or lazy mode)
        # With ann_auto_rebuild=False (default), we just invalidate and rebuild on next query
        if self.ann_index is not None and not self.bulk_mode:
            if self.ann_auto_rebuild:
                # Expensive: rebuild immediately (old behavior)
                self.ann_index = None
                self.ann_vectors = None
                self.ann_ids = []
                self._build_ann_index()
            else:
                # Cheap: just invalidate, rebuild lazily on next query
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
        *,
        probe: str,
        data_type: str = "json",
        top_k: int = 10,
        threshold: float = 0.0,
        guard=None,
        negations=None,
        any_marker="$any",
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        parsed_probe = parse_data(probe, data_type)

        # Handle $or disjunctions using VSA superposition (bundling)
        # Instead of N separate queries, encode all branches and bundle into one probe
        if "$or" in parsed_probe and isinstance(parsed_probe["$or"], list):
            or_branches = parsed_probe["$or"]
            if or_branches:
                # Helper to strip $any markers from a branch before encoding
                def clean_branch(branch):
                    if not isinstance(branch, dict):
                        return branch
                    cleaned = {}
                    for k, v in branch.items():
                        if isinstance(v, dict) and any_marker in v:
                            continue  # Skip $any fields
                        elif isinstance(v, dict):
                            cleaned[k] = clean_branch(v)  # Recurse
                        else:
                            cleaned[k] = v
                    return cleaned

                # Encode each branch (after cleaning)
                branch_vectors = []
                for branch in or_branches:
                    try:
                        clean = clean_branch(branch)
                        vec = self.encoder.encode_data(clean)
                        branch_vectors.append(vec)
                    except Exception:
                        continue  # Skip branches that fail to encode

                if branch_vectors:
                    # Bundle via superposition (sum + normalize) - the VSA way!
                    # Convert TorchHD tensors to numpy if needed
                    if self._use_torchhd:
                        branch_vectors = [
                            v.cpu().numpy() if hasattr(v, "cpu") else v
                            for v in branch_vectors
                        ]
                    bundled = sum(branch_vectors)
                    bundled = bundled / (np.linalg.norm(bundled) + 1e-10)

                    # Now search with single bundled probe
                    # Build a modified parsed_probe for guard matching
                    # (guards still need the original structure for filtering)
                    parsed_probe = {"_bundled_or": True}  # Mark as bundled

                    # Continue with normal query flow using bundled vector
                    probe_vector = bundled

                    # Skip the normal encoding below
                    skip_encoding = True
                else:
                    return []
            else:
                return []
        else:
            skip_encoding = False

        # Handle user-specified any wildcards
        clean_probe = {}
        for k, v in parsed_probe.items():
            if isinstance(v, dict) and any_marker in v:
                continue  # Skip for encoding
            clean_probe[k] = v

        # Skip encoding if we already bundled $or branches above
        if not skip_encoding:
            try:
                probe_vector = self.encoder.encode_data(clean_probe)
                # Convert TorchHD tensors to numpy if needed
                if self._use_torchhd and hasattr(probe_vector, "cpu"):
                    probe_vector = probe_vector.cpu().numpy()
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
            # Convert TorchHD tensors to numpy if needed
            if self._use_torchhd and hasattr(neg_vector, "cpu"):
                neg_vector = neg_vector.cpu().numpy()
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
                # Handle $exists operator
                if key == "$exists":
                    if isinstance(value, dict):
                        for field, should_exist in value.items():
                            exists = field in data
                            if exists != should_exist:
                                return False
                    continue

                if key not in data:
                    return False

                data_value = data[key]

                if isinstance(value, dict):
                    # Handle comparison operators
                    if "$gt" in value:
                        if not (
                            isinstance(data_value, (int, float))
                            and data_value > value["$gt"]
                        ):
                            return False
                    if "$gte" in value:
                        if not (
                            isinstance(data_value, (int, float))
                            and data_value >= value["$gte"]
                        ):
                            return False
                    if "$lt" in value:
                        if not (
                            isinstance(data_value, (int, float))
                            and data_value < value["$lt"]
                        ):
                            return False
                    if "$lte" in value:
                        if not (
                            isinstance(data_value, (int, float))
                            and data_value <= value["$lte"]
                        ):
                            return False
                    # Handle $contains for substring matching
                    if "$contains" in value:
                        if not (
                            isinstance(data_value, str)
                            and value["$contains"] in data_value
                        ):
                            return False
                    # Handle $in for membership
                    if "$in" in value:
                        if data_value not in value["$in"]:
                            return False
                    # Handle nested $or
                    if "$or" in value and isinstance(value["$or"], list):
                        # For nested $or, any of the conditions for this key must match
                        if not any(
                            is_subset({key: or_val}, data) for or_val in value["$or"]
                        ):
                            return False
                    # Skip if we handled special operators
                    elif any(
                        k.startswith("$")
                        for k in value.keys()
                        if k in ("$gt", "$gte", "$lt", "$lte", "$contains", "$in")
                    ):
                        pass  # Already handled above
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

        # Use ANN if available, enabled, and dataset is large
        if (
            FAISS_AVAILABLE
            and self.ann_enabled
            and len(self.stored_vectors) >= ANN_THRESHOLD
        ):
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
            stored = self.stored_data[data_id]
            # Handle new storage format with _raw, _type, _parsed
            if isinstance(stored, dict) and "_parsed" in stored:
                data_dict = stored["_parsed"]
                # Return raw string for HTTP serialization
                return_data = stored["_raw"]
            else:
                # Legacy format
                data_dict = stored
                return_data = stored
            # Apply negations
            if negation_filters and matches_negation(data_dict, negation_filters):
                continue
            # Apply guard if provided (data structure matching)
            if guard and not is_subset(guard, data_dict):
                continue  # Skip if guard fails
            results.append((data_id, score, return_data))
        return results

    def get(self, data_id: str) -> Dict[str, Any]:
        if data_id not in self.stored_data:
            raise KeyError(
                f"Data ID '{data_id}' not found. "
                "Make sure the data was previously inserted and not deleted."
            )
        stored = self.stored_data[data_id]
        # Handle new storage format with _raw, _type, _parsed
        if isinstance(stored, dict) and "_parsed" in stored:
            return stored["_parsed"]
        return stored

    def get_raw(self, data_id: str) -> tuple:
        """Get raw data string and type for a stored item."""
        if data_id not in self.stored_data:
            raise KeyError(f"Data ID '{data_id}' not found.")
        stored = self.stored_data[data_id]
        if isinstance(stored, dict) and "_raw" in stored:
            return stored["_raw"], stored["_type"]
        # Legacy format - return as JSON
        return json.dumps(stored), "json"

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

    # ==========================================================================
    # VSA Kernel Primitives - Direct Access
    # ==========================================================================

    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Bind two vectors (AND-like operation).

        Creates an association between two concepts.
        bind(A, B) is similar to neither A nor B individually.
        """
        return self.encoder.bind(vec1, vec2)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple vectors (OR-like operation).

        Creates a superposition representing all input concepts.
        bundle([A, B, C]) is similar to A, B, and C.
        """
        return self.encoder.bundle(vectors)

    def negate(
        self, superposition: np.ndarray, component: np.ndarray, method: str = "subtract"
    ) -> np.ndarray:
        """
        Remove a component's influence from a superposition (NOT operation).

        This is a novel VSA primitive that extends traditional VSA operations.

        Args:
            superposition: The vector to remove from
            component: The vector to remove
            method: "subtract" (default), "project", or "flip"

        Returns:
            Vector with component's influence diminished

        Example:
            >>> A = store.vector_manager.get_vector("A")
            >>> B = store.vector_manager.get_vector("B")
            >>> C = store.vector_manager.get_vector("C")
            >>> ABC = store.bundle([A, B, C])
            >>> AC = store.negate(ABC, B)  # Removes B
            >>> similarity(AC, B) < 0  # B has negative similarity now
        """
        return self.encoder.negate(superposition, component, method)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind to retrieve associated value (inverse of bind).

        If bound = bind(key, value), then unbind(bound, key) ≈ value.
        """
        # For bipolar vectors, unbinding is the same as binding (self-inverse)
        return self.encoder.bind(bound, key)

    def amplify(
        self, superposition: np.ndarray, component: np.ndarray, strength: float = 1.0
    ) -> np.ndarray:
        """
        Strengthen a component's presence in a superposition.

        Boosts the specified component's influence.
        """
        return self.encoder.amplify(superposition, component, strength)

    def prototype(
        self, vectors: List[np.ndarray], threshold: float = 0.5
    ) -> np.ndarray:
        """
        Extract the common pattern from a set of vectors.

        Returns what's shared across all examples.
        """
        return self.encoder.prototype(vectors, threshold)

    def difference(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Compute what changed between two states.

        Returns a vector highlighting additions and removals.
        """
        return self.encoder.difference(before, after)

    def blend(
        self, vec1: np.ndarray, vec2: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """
        Weighted interpolation between two vectors.

        Creates smooth transitions between concepts.
        """
        return self.encoder.blend(vec1, vec2, alpha)

    def resonance(self, vec: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """
        Extract the part of vec that resonates with reference.

        Keeps only dimensions where both agree.
        """
        return self.encoder.resonance(vec, reference)

    def permute(self, vec: np.ndarray, k: int) -> np.ndarray:
        """
        Circular shift (permutation) of vector dimensions.

        Used for positional encoding in sequences and "what comes after?" queries.
        """
        return self.encoder.permute(vec, k)

    def cleanup(self, noisy: np.ndarray, codebook: List[np.ndarray]) -> np.ndarray:
        """
        Find the closest vector in codebook to the noisy input.

        Returns a vector (the closest match from codebook), not data.
        Useful for denoising composed vectors before further operations.
        """
        return self.encoder.cleanup(noisy, codebook)

    def prototype_add(
        self, prototype: np.ndarray, example: np.ndarray, count: int
    ) -> np.ndarray:
        """
        Incrementally update a prototype with a new example.

        Avoids re-computing prototype([all_examples]) from scratch.
        Pass the current count (before adding this example).
        """
        return self.encoder.prototype_add(prototype, example, count)

    def encode_sequence(
        self, items: List[Any], mode: str = "positional", **config
    ) -> np.ndarray:
        """
        Encode a sequence of items into a single vector.

        Modes:
            - "positional": Bind each item to position vector (default)
                           Good for: ordered lists, event sequences
            - "chained": Relative binding for suffix/prefix operations
                        Good for: prefix matching, sequence reversal
            - "ngram": N-gram pairs for fuzzy substring matching
                      Good for: text search, partial phrase matching
                      Config: n_sizes=[1,2] (unigrams + bigrams)
            - "bundle": Pure superposition, no order preserved
                       Good for: bag-of-words, unordered sets

        Examples:
            # Event sequence (order matters)
            store.encode_sequence(["login", "view", "purchase"])

            # Text search (fuzzy matching)
            store.encode_sequence(["quick", "brown", "fox"], mode="ngram")

            # Tags (order doesn't matter)
            store.encode_sequence(["python", "ml", "api"], mode="bundle")

        :param items: List of items to encode
        :param mode: Encoding mode
        :param config: Mode-specific options (e.g., n_sizes for ngram)
        :return: Encoded vector
        """
        return self.encoder.encode_list(items, mode=mode, **config)

    def similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: str = "cosine",
        **kwargs,
    ) -> float:
        """
        Compute similarity between two vectors using the specified metric.

        Metrics:
            - "cosine": Cosine similarity (default, Qdrant-native)
            - "dot": Dot product similarity (Qdrant-native)
            - "euclidean": Euclidean distance as similarity (Qdrant-native)
            - "manhattan": Manhattan distance as similarity (Qdrant-native)
            - "hamming": Hamming similarity (optimal for bipolar vectors)
            - "overlap": Overlap similarity (count of matching positions)
            - "agreement": Agreement similarity (balanced view)
            - "weighted_cosine": Weighted cosine (requires weights kwarg)

        Examples:
            >>> store = CPUStore()
            >>> vec1 = store.encode({"type": "billing"})
            >>> vec2 = store.encode({"type": "technical"})
            >>> store.similarity(vec1, vec2)  # default cosine
            0.35
            >>> store.similarity(vec1, vec2, metric="hamming")
            0.42

        :param vec1: First vector
        :param vec2: Second vector
        :param metric: Similarity metric name
        :param kwargs: Additional arguments (e.g., weights for weighted_cosine)
        :return: Similarity score
        """
        from .distance import DistanceEngine, DistanceMetric

        # Map string to enum
        metric_map = {
            "cosine": DistanceMetric.COSINE,
            "dot": DistanceMetric.DOT_PRODUCT,
            "euclidean": DistanceMetric.EUCLIDEAN,
            "manhattan": DistanceMetric.MANHATTAN,
            "hamming": DistanceMetric.HAMMING,
            "overlap": DistanceMetric.OVERLAP,
            "agreement": DistanceMetric.AGREEMENT,
            "chebyshev": DistanceMetric.CHEBYSHEV,
            "weighted_cosine": DistanceMetric.WEIGHTED_COSINE,
            "weighted_euclidean": DistanceMetric.WEIGHTED_EUCLIDEAN,
        }

        if metric not in metric_map:
            raise ValueError(
                f"Unknown metric: {metric}. Available: {list(metric_map.keys())}"
            )

        engine = DistanceEngine()
        return engine.similarity(vec1, vec2, metric_map[metric], **kwargs)
