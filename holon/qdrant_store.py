"""
Qdrant-backed storage for Holon.

Provides persistent vector storage with:
- Collections as namespaces (easy deletion/partitioning)
- Server-side ANN search
- Payload storage for original data

Usage:
    from holon import QdrantStore, HolonClient

    store = QdrantStore(
        collection="my_app",
        dimensions=4096,
        url="http://localhost:6333"
    )
    client = HolonClient(local_store=store)

    # Use like CPUStore
    client.insert_json({"name": "Alice", "role": "developer"})
    results = client.search_json(probe={"role": "developer"})

    # Easy cleanup
    store.drop_collection()  # Wipe everything in this namespace
"""

import json
import logging
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .atomizer import parse_data
from .cpu_store import Store
from .encoder import Encoder
from .vector_manager import VectorManager

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import Distance, PointStruct, VectorParams

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None


def _encode_chunk(
    work_items: List[Tuple[int, str, str, int]]
) -> List[Tuple[int, List[float], Any]]:
    """
    Encode a chunk of items in a worker process.

    Args:
        work_items: List of (index, data_string, data_type, dimensions)

    Returns:
        List of (index, vector_as_list, parsed_data)
    """
    # Create encoder for this worker (each process needs its own)
    if not work_items:
        return []

    dimensions = work_items[0][3]
    vm = VectorManager(dimensions, backend="cpu")
    encoder = Encoder(vm)

    results = []
    for idx, data_str, data_type, _ in work_items:
        parsed = parse_data(data_str, data_type)
        vector = encoder.encode_data(parsed)

        if hasattr(vector, "get"):
            vector = vector.get()
        vector = vector.astype(np.float32)

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        results.append((idx, vector.tolist(), parsed))

    return results


class QdrantStore(Store):
    """
    Qdrant-backed Holon store.

    Each instance connects to a single collection (namespace).
    Multiple QdrantStore instances can share a Qdrant server
    with different collections for data isolation.
    """

    def __init__(
        self,
        collection: str = "holon_default",
        dimensions: int = 4096,
        url: str = "http://localhost:6333",
        api_key: Optional[str] = None,
        recreate_collection: bool = False,
        marker_prefix: str = "$",
    ):
        """
        Initialize Qdrant store.

        Args:
            collection: Collection name (namespace). Use different names
                        for isolated data partitions.
            dimensions: Vector dimensions (must match data complexity).
            url: Qdrant server URL.
            api_key: Optional API key for Qdrant Cloud.
            recreate_collection: If True, drop and recreate collection on init.
            marker_prefix: Prefix for special markers like $time, $any, $gt, etc.
                          Change this if your data legitimately contains keys like "$time".
        """
        if not QDRANT_AVAILABLE:
            raise ImportError(
                "qdrant-client not installed. Run: pip install qdrant-client"
            )

        self.collection = collection
        self.dimensions = dimensions
        self.url = url
        self.marker_prefix = marker_prefix

        # Initialize Qdrant client
        self.client = QdrantClient(url=url, api_key=api_key)

        # Initialize encoder (CPU-based, vectors sent to Qdrant)
        self.vector_manager = VectorManager(dimensions, backend="cpu")
        self.encoder = Encoder(self.vector_manager, marker_prefix=marker_prefix)

        # Setup collection
        if recreate_collection:
            self.drop_collection()

        self._ensure_collection()

        logging.info(
            f"QdrantStore initialized: collection={collection}, dim={dimensions}"
        )

    def _ensure_collection(self):
        """Create collection if it doesn't exist."""
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=self.dimensions,
                    distance=Distance.DOT,  # Matches Holon's normalized_dot_similarity
                ),
            )
            logging.info(f"Created collection: {self.collection}")

    # =========================================================================
    # Core Store Interface
    # =========================================================================

    def insert(self, data: str, data_type: str = "json") -> str:
        """
        Insert data into Qdrant.

        Args:
            data: JSON or EDN string.
            data_type: 'json' or 'edn'.

        Returns:
            UUID string for the inserted point.
        """
        # Parse and encode
        parsed = parse_data(data, data_type)
        vector = self.encoder.encode_data(parsed)

        # Convert to float32 for Qdrant
        if hasattr(vector, "get"):  # cupy
            vector = vector.get()
        vector = vector.astype(np.float32)

        # Normalize for dot product similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Generate ID
        point_id = str(uuid.uuid4())

        # Store in Qdrant with payload
        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload={
                        "_raw": data,
                        "_type": data_type,
                        "_parsed": parsed if isinstance(parsed, dict) else str(parsed),
                    },
                )
            ],
        )

        return point_id

    def query(
        self,
        *,
        probe: str,
        data_type: str = "json",
        top_k: int = 10,
        threshold: float = 0.0,
        guard: Callable = None,
        negations: Optional[Dict] = None,
        any_marker: str = "$any",
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Query Qdrant with similarity search.

        Args:
            probe: Query probe as JSON/EDN string.
            data_type: 'json' or 'edn'.
            top_k: Number of results to return.
            threshold: Minimum similarity score.
            guard: Optional filter function or dict for guard matching.
            negations: Optional negation specs.
            any_marker: Marker for wildcards (default "$any").

        Returns:
            List of (id, score, data) tuples.
        """
        # Parse probe
        parsed = parse_data(probe, data_type)

        # Handle $or disjunctions using VSA superposition (bundling)
        if "$or" in parsed and isinstance(parsed["$or"], list):
            or_branches = parsed["$or"]
            if or_branches:
                branch_vectors = []
                for branch in or_branches:
                    try:
                        clean = self._clean_any_markers(branch, any_marker)
                        vec = self.encoder.encode_data(clean)
                        branch_vectors.append(vec)
                    except Exception:
                        continue

                if branch_vectors:
                    bundled = sum(branch_vectors)
                    bundled = bundled / (np.linalg.norm(bundled) + 1e-10)
                    vector = bundled
                else:
                    return []
            else:
                return []
        else:
            # Clean any markers from probe
            clean_probe = self._clean_any_markers(parsed, any_marker)
            vector = self.encoder.encode_data(clean_probe)

        # Handle negations
        if negations:
            neg_parsed = (
                parse_data(json.dumps(negations), "json")
                if isinstance(negations, dict)
                else negations
            )
            cleaned_negations = self._extract_not_values(neg_parsed)
            if cleaned_negations:
                neg_vector = self.encoder.encode_data(cleaned_negations)
                vector = vector - neg_vector

        if hasattr(vector, "get"):
            vector = vector.get()
        vector = vector.astype(np.float32)

        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm

        # Search in Qdrant
        # Request more results if guard filtering might reduce count
        search_limit = top_k * 3 if guard else top_k

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=search_limit,
            score_threshold=threshold if threshold > 0 else None,
            with_payload=True,
        )
        results = response.points

        # Build guard function if guard is a dict
        guard_fn = self._make_guard_fn(guard) if isinstance(guard, dict) else guard

        # Process results
        output = []
        for hit in results:
            payload = hit.payload
            data = payload.get("_parsed", payload)

            # Apply guard filter
            if guard_fn and not guard_fn(data):
                continue

            # Apply negation filter
            if negations and self._matches_negation(data, negations):
                continue

            output.append((hit.id, hit.score, data))

            if len(output) >= top_k:
                break

        return output

    def _clean_any_markers(self, data: Any, any_marker: str) -> Any:
        """Remove $any markers from data structure."""
        if not isinstance(data, dict):
            return data
        cleaned = {}
        for k, v in data.items():
            if isinstance(v, dict) and any_marker in v:
                continue
            elif isinstance(v, dict):
                cleaned[k] = self._clean_any_markers(v, any_marker)
            else:
                cleaned[k] = v
        return cleaned

    def _extract_not_values(self, negations: Dict) -> Dict:
        """Extract values from $not specs for vector encoding."""
        cleaned = {}
        for k, v in negations.items():
            if isinstance(v, dict) and "$not" in v:
                not_val = v["$not"]
                if isinstance(not_val, list):
                    cleaned[k] = not_val[0]
                else:
                    cleaned[k] = not_val
            elif isinstance(v, dict):
                sub = self._extract_not_values(v)
                if sub:
                    cleaned[k] = sub
        return cleaned

    def _matches_negation(self, data: Dict, negations: Dict) -> bool:
        """Check if data matches any negation spec."""
        for k, v in negations.items():
            if isinstance(v, dict) and "$not" in v:
                not_val = v["$not"]
                if k in data:
                    if isinstance(not_val, list):
                        if data[k] in not_val:
                            return True
                    elif data[k] == not_val:
                        return True
            elif isinstance(v, dict) and k in data:
                if self._matches_negation(data[k], v):
                    return True
        return False

    def _make_guard_fn(self, guard: Dict) -> Callable:
        """Create a guard function from a guard dict."""

        def guard_fn(data: Dict) -> bool:
            return self._guard_matches(guard, data)

        return guard_fn

    def _guard_matches(self, guard: Dict, data: Dict) -> bool:
        """Check if data matches guard conditions."""
        if "$or" in guard:
            return any(self._guard_matches(c, data) for c in guard["$or"])

        for key, value in guard.items():
            if key == "$exists":
                for field, should_exist in value.items():
                    if (field in data) != should_exist:
                        return False
                continue

            if key not in data:
                return False

            data_value = data[key]

            if isinstance(value, dict):
                if "$gt" in value and not (
                    isinstance(data_value, (int, float)) and data_value > value["$gt"]
                ):
                    return False
                if "$gte" in value and not (
                    isinstance(data_value, (int, float)) and data_value >= value["$gte"]
                ):
                    return False
                if "$lt" in value and not (
                    isinstance(data_value, (int, float)) and data_value < value["$lt"]
                ):
                    return False
                if "$lte" in value and not (
                    isinstance(data_value, (int, float)) and data_value <= value["$lte"]
                ):
                    return False
                if "$contains" in value and not (
                    isinstance(data_value, str) and value["$contains"] in data_value
                ):
                    return False
                if "$in" in value and data_value not in value["$in"]:
                    return False
                # Nested dict matching
                if not any(k.startswith("$") for k in value.keys()):
                    if not isinstance(data_value, dict) or not self._guard_matches(
                        value, data_value
                    ):
                        return False
            elif isinstance(value, list):
                if data_value not in value:
                    return False
            else:
                if data_value != value:
                    return False

        return True

    def get(self, data_id: str) -> Dict[str, Any]:
        """Retrieve data by ID."""
        result = self.client.retrieve(
            collection_name=self.collection,
            ids=[data_id],
        )

        if not result:
            raise KeyError(
                f"ID '{data_id}' not found in collection '{self.collection}'"
            )

        payload = result[0].payload
        return payload.get("_parsed", payload)

    def get_raw(self, data_id: str) -> Tuple[str, str]:
        """Get raw data string and type."""
        result = self.client.retrieve(
            collection_name=self.collection,
            ids=[data_id],
        )

        if not result:
            raise KeyError(f"ID '{data_id}' not found")

        payload = result[0].payload
        return payload.get("_raw", ""), payload.get("_type", "json")

    def delete(self, data_id: str) -> bool:
        """Delete a point by ID."""
        try:
            self.client.delete(
                collection_name=self.collection,
                points_selector=models.PointIdsList(points=[data_id]),
            )
            return True
        except Exception:
            return False

    # =========================================================================
    # Collection Management (Namespace Operations)
    # =========================================================================

    def drop_collection(self):
        """
        Drop the entire collection.

        This is the fastest way to wipe all data in a namespace.
        """
        try:
            self.client.delete_collection(collection_name=self.collection)
            logging.info(f"Dropped collection: {self.collection}")
        except Exception:
            pass  # Collection didn't exist

    def clear(self):
        """Clear all data (drop and recreate collection)."""
        self.drop_collection()
        self._ensure_collection()

    def count(self) -> int:
        """Get number of points in collection."""
        info = self.client.get_collection(collection_name=self.collection)
        return info.points_count

    # =========================================================================
    # Bulk Operations
    # =========================================================================

    def batch_insert(
        self, items: List[str], data_type: str = "json", batch_size: int = 200
    ) -> List[str]:
        """
        Insert multiple items efficiently in chunks.

        Note: The bottleneck is Qdrant HTTP upload (~320-335 items/sec),
        not encoding. Larger batch sizes help slightly but hit payload
        limits above 200 for 4096d vectors.

        Args:
            items: List of data strings.
            data_type: 'json' or 'edn'.
            batch_size: Number of items per batch (default 200).

        Returns:
            List of generated IDs.
        """
        all_ids = []

        # Process in chunks to avoid payload size limits
        for chunk_start in range(0, len(items), batch_size):
            chunk = items[chunk_start : chunk_start + batch_size]
            points = []
            chunk_ids = []

            for data in chunk:
                parsed = parse_data(data, data_type)
                vector = self.encoder.encode_data(parsed)

                if hasattr(vector, "get"):
                    vector = vector.get()
                vector = vector.astype(np.float32)

                norm = np.linalg.norm(vector)
                if norm > 0:
                    vector = vector / norm

                point_id = str(uuid.uuid4())
                chunk_ids.append(point_id)

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={
                            "_raw": data,
                            "_type": data_type,
                            "_parsed": parsed
                            if isinstance(parsed, dict)
                            else str(parsed),
                        },
                    )
                )

            # Batch upsert chunk
            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
            all_ids.extend(chunk_ids)

        return all_ids

    def parallel_batch_insert(
        self,
        items: List[str],
        data_type: str = "json",
        batch_size: int = 200,
        num_workers: int = None,
    ) -> List[str]:
        """
        Insert multiple items with parallel encoding.

        Uses multiprocessing to encode vectors in parallel, then batch uploads.

        Note: Provides only ~1.2x speedup because Qdrant HTTP upload
        (~320 items/sec) is the bottleneck, not encoding (~1300 items/sec).
        Use regular batch_insert() for simplicity unless encoding is your
        bottleneck (e.g., very complex documents or higher dimensions).

        Args:
            items: List of data strings.
            data_type: 'json' or 'edn'.
            batch_size: Number of items per Qdrant batch (default 200).
            num_workers: Number of parallel workers (default: CPU count).

        Returns:
            List of generated IDs.
        """
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        if num_workers is None:
            num_workers = mp.cpu_count()

        # Prepare work items with indices for ordering
        work_items = [
            (i, data, data_type, self.dimensions) for i, data in enumerate(items)
        ]

        # Process in parallel
        encoded_results = [None] * len(items)

        # Use smaller chunks for parallel processing
        chunk_size = max(1, len(items) // (num_workers * 4))

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {}
            for i in range(0, len(work_items), chunk_size):
                chunk = work_items[i : i + chunk_size]
                future = executor.submit(_encode_chunk, chunk)
                futures[future] = i

            for future in as_completed(futures):
                chunk_start = futures[future]
                results = future.result()
                for idx, vector, parsed in results:
                    encoded_results[idx] = (vector, parsed)

        # Now batch upload to Qdrant
        all_ids = []

        for chunk_start in range(0, len(items), batch_size):
            chunk_end = min(chunk_start + batch_size, len(items))
            points = []
            chunk_ids = []

            for i in range(chunk_start, chunk_end):
                vector, parsed = encoded_results[i]
                point_id = str(uuid.uuid4())
                chunk_ids.append(point_id)

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "_raw": items[i],
                            "_type": data_type,
                            "_parsed": parsed
                            if isinstance(parsed, dict)
                            else str(parsed),
                        },
                    )
                )

            self.client.upsert(
                collection_name=self.collection,
                points=points,
            )
            all_ids.extend(chunk_ids)

        return all_ids

    def start_bulk_insert(self):
        """No-op for Qdrant (handles batching internally)."""
        pass

    def end_bulk_insert(self):
        """No-op for Qdrant."""
        pass

    # =========================================================================
    # VSA Primitives (delegated to encoder)
    # =========================================================================

    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """Bind two vectors."""
        return self.encoder.bind(vec1, vec2)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """Bundle multiple vectors."""
        return self.encoder.bundle(vectors)

    def negate(
        self, superposition: np.ndarray, component: np.ndarray, method: str = "subtract"
    ) -> np.ndarray:
        """Negate a component from a superposition."""
        return self.encoder.negate(superposition, component, method)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """Unbind to retrieve value."""
        return self.encoder.bind(bound, key)

    def amplify(
        self, superposition: np.ndarray, component: np.ndarray, strength: float = 1.0
    ) -> np.ndarray:
        """Amplify a component."""
        return self.encoder.amplify(superposition, component, strength)

    def prototype(
        self, vectors: List[np.ndarray], threshold: float = 0.5
    ) -> np.ndarray:
        """Extract common pattern."""
        return self.encoder.prototype(vectors, threshold)

    def difference(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """Compute difference between states."""
        return self.encoder.difference(before, after)

    def blend(
        self, vec1: np.ndarray, vec2: np.ndarray, alpha: float = 0.5
    ) -> np.ndarray:
        """Blend two vectors."""
        return self.encoder.blend(vec1, vec2, alpha)

    def resonance(self, vec: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Extract resonating parts."""
        return self.encoder.resonance(vec, reference)

    def permute(self, vec: np.ndarray, k: int) -> np.ndarray:
        """Circular shift (permutation) of vector dimensions."""
        return self.encoder.permute(vec, k)

    def cleanup(self, noisy: np.ndarray, codebook: List[np.ndarray]) -> np.ndarray:
        """Find the closest vector in codebook to the noisy input."""
        return self.encoder.cleanup(noisy, codebook)

    def prototype_add(
        self, prototype: np.ndarray, example: np.ndarray, count: int
    ) -> np.ndarray:
        """Incrementally update a prototype with a new example."""
        return self.encoder.prototype_add(prototype, example, count)

    def encode_sequence(
        self, items: List[Any], mode: str = "positional", **config
    ) -> np.ndarray:
        """
        Encode a sequence of items into a single vector.
        
        Modes: "positional", "chained", "ngram", "bundle"
        """
        return self.encoder.encode_list(items, mode=mode, **config)

    # =========================================================================
    # Info
    # =========================================================================

    def info(self) -> Dict[str, Any]:
        """Get collection info."""
        collection_info = self.client.get_collection(collection_name=self.collection)
        return {
            "collection": self.collection,
            "dimensions": self.dimensions,
            "url": self.url,
            "points_count": collection_info.points_count,
            "status": collection_info.status.value,
        }
