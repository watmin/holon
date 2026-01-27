#!/usr/bin/env python3
"""
Holon Unified Client

Provides a unified interface for interacting with Holon whether it's a local
CPUStore instance or a remote HTTP API. Users don't need to know or care
about the underlying implementation.

The client abstracts all vector operations - users work with data, not vectors.
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import requests

if TYPE_CHECKING:
    from .cpu_store import CPUStore


class HolonClient:
    """
    Unified client for Holon operations.

    Can work with either:
    - Local CPUStore instance (direct method calls)
    - Remote HTTP API (REST requests)

    Users don't need to know the difference - same interface either way.
    """

    def __init__(
        self,
        *,
        local_store: Optional["CPUStore"] = None,
        remote_url: Optional[str] = None,
    ):
        """
        Initialize client with either a local store or remote URL.

        Use keyword arguments to clearly specify the connection type.

        Args:
            local_store: CPUStore instance for local operations
            remote_url: URL string (e.g., "http://localhost:8000") for remote operations

        Raises:
            ValueError: If neither or both local_store and remote_url are provided

        Examples:
            # Local usage
            client = HolonClient(local_store=my_store)

            # Remote usage
            client = HolonClient(remote_url="http://localhost:8000")
        """
        if local_store is not None and remote_url is not None:
            raise ValueError("Cannot specify both local_store and remote_url")
        if local_store is None and remote_url is None:
            raise ValueError("Must specify either local_store or remote_url")

        if remote_url is not None:
            # Remote HTTP client
            self._mode = "http"
            self._base_url = remote_url.rstrip("/")
            self._session = requests.Session()
        else:
            # Local store client
            self._mode = "local"
            self._store = local_store

    def health(self) -> Dict[str, Any]:
        """Get system health and statistics."""
        if self._mode == "http":
            response = self._session.get(f"{self._base_url}/api/v1/health")
            response.raise_for_status()
            return response.json()
        else:
            # Local mode
            return {
                "status": "healthy",
                "backend": self._store.backend,
                "items_count": len(self._store.stored_data),
            }

    def insert(self, data: Union[str, Dict], data_type: str = "json") -> str:
        """
        Insert a single item.

        Args:
            data: The data to insert (dict or JSON string)
            data_type: "json" or "edn"

        Returns:
            Unique ID of inserted item
        """
        if isinstance(data, dict):
            data_str = json.dumps(data)
        else:
            data_str = data

        if self._mode == "http":
            response = self._session.post(
                f"{self._base_url}/api/v1/items",
                json={"data": data_str, "data_type": data_type},
            )
            response.raise_for_status()
            return response.json()["id"]
        else:
            return self._store.insert(data_str, data_type)

    def insert_batch(
        self, items: List[Union[str, Dict]], data_type: str = "json"
    ) -> List[str]:
        """
        Insert multiple items efficiently.

        Args:
            items: List of data items (dicts or JSON strings)
            data_type: "json" or "edn"

        Returns:
            List of unique IDs for inserted items
        """
        # Convert dicts to JSON strings if needed
        items_str = []
        for item in items:
            if isinstance(item, dict):
                items_str.append(json.dumps(item))
            else:
                items_str.append(item)

        if self._mode == "http":
            response = self._session.post(
                f"{self._base_url}/api/v1/items/batch",
                json={"items": items_str, "data_type": data_type},
            )
            response.raise_for_status()
            return response.json()["ids"]
        else:
            return self._store.batch_insert(items_str, data_type)

    def get(self, item_id: str) -> Optional[Dict]:
        """
        Retrieve an item by ID.

        Args:
            item_id: The unique ID of the item

        Returns:
            Item data dict, or None if not found
        """
        if self._mode == "http":
            response = self._session.get(f"{self._base_url}/api/v1/items/{item_id}")
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()["data"]
        else:
            return self._store.get(item_id)

    def search(
        self,
        probe: Union[str, Dict],
        data_type: str = "json",
        top_k: int = 10,
        threshold: float = 0.0,
        guard: Optional[Dict] = None,
        negations: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items using vector similarity.

        Args:
            probe: Search probe (dict or JSON string)
            data_type: "json" or "edn"
            top_k: Maximum number of results
            threshold: Similarity threshold (0.0-1.0)
            guard: Guard conditions
            negations: Negation filters

        Returns:
            List of results with id, score, and data
        """
        if isinstance(probe, dict):
            probe_str = json.dumps(probe)
        else:
            probe_str = probe

        payload = {
            "probe": probe_str,
            "data_type": data_type,
            "top_k": top_k,
            "threshold": threshold,
            "any_marker": "$any",
        }
        if guard:
            payload["guard"] = guard
        if negations:
            payload["negations"] = negations

        if self._mode == "http":
            response = self._session.post(
                f"{self._base_url}/api/v1/search", json=payload
            )
            response.raise_for_status()
            return response.json()["results"]
        else:
            results = self._store.query(
                probe_str, data_type, top_k, threshold, guard=guard, negations=negations
            )
            # Convert to same format as HTTP API
            return [
                {"id": item_id, "score": score, "data": data}
                for item_id, score, data in results
            ]

    def encode_vectors(
        self, data: Union[str, Dict], data_type: str = "json"
    ) -> List[float]:
        """
        Encode data to a vector without storing it.

        Useful for vector bootstrapping and custom similarity operations.

        Args:
            data: Data to encode (dict or JSON string)
            data_type: "json" or "edn"

        Returns:
            Encoded vector as list of floats
        """
        if isinstance(data, dict):
            data_str = json.dumps(data)
        else:
            data_str = data

        if self._mode == "http":
            response = self._session.post(
                f"{self._base_url}/api/v1/vectors/encode",
                json={"data": data_str, "data_type": data_type},
            )
            response.raise_for_status()
            return response.json()["vector"]
        else:
            from .atomizer import parse_data

            parsed = parse_data(data_str, data_type)
            vector = self._store.encoder.encode_data(parsed)
            cpu_vector = self._store.vector_manager.to_cpu(vector)
            return cpu_vector.tolist()

    def encode_mathematical(
        self, primitive: str, value: Union[int, float]
    ) -> List[float]:
        """
        Encode mathematical primitives.

        Args:
            primitive: Mathematical primitive name (e.g., "addition")
            value: Value for the primitive

        Returns:
            Encoded vector as list of floats
        """
        if self._mode == "http":
            response = self._session.post(
                f"{self._base_url}/api/v1/vectors/encode/mathematical",
                json={"primitive": primitive, "value": value},
            )
            response.raise_for_status()
            return response.json()["vector"]
        else:
            from .encoder import MathematicalPrimitive

            prim = MathematicalPrimitive(primitive)
            vector = self._store.encoder.encode_mathematical_primitive(prim, value)
            cpu_vector = self._store.vector_manager.to_cpu(vector)
            return cpu_vector.tolist()

    def compose_vectors(
        self, operation: str, vectors: List[List[float]]
    ) -> List[float]:
        """
        Compose vectors using mathematical operations.

        Args:
            operation: "bind" or "bundle"
            vectors: List of vectors to compose

        Returns:
            Composed vector as list of floats
        """
        if self._mode == "http":
            response = self._session.post(
                f"{self._base_url}/api/v1/vectors/compose",
                json={"operation": operation, "vectors": vectors},
            )
            response.raise_for_status()
            return response.json()["vector"]
        else:
            import numpy as np

            np_vectors = [np.array(vec, dtype=np.int8) for vec in vectors]

            if operation == "bind":
                result = self._store.encoder.mathematical_bind(*np_vectors)
            elif operation == "bundle":
                result = self._store.encoder.mathematical_bundle(np_vectors)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            cpu_result = self._store.vector_manager.to_cpu(result)
            return cpu_result.tolist()

    # Convenience methods for common operations

    def insert_json(self, data: Dict) -> str:
        """Insert a JSON dict (convenience method)."""
        return self.insert(data, "json")

    def insert_batch_json(self, items: List[Dict]) -> List[str]:
        """Insert multiple JSON dicts (convenience method)."""
        return self.insert_batch(items, "json")

    def search_json(self, probe: Dict, **kwargs) -> List[Dict[str, Any]]:
        """Search with JSON dict probe (convenience method)."""
        return self.search(probe, "json", **kwargs)

    def encode_vectors_json(self, data: Dict) -> List[float]:
        """Encode JSON dict to vector (convenience method)."""
        return self.encode_vectors(data, "json")
