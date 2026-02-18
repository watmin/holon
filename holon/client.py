#!/usr/bin/env python3
"""
Holon Unified Client

The primary interface for Holon - a programmatic neural memory system using
Vector Symbolic Architectures (VSA) / Hyperdimensional Computing (HDC).

This client provides:
- Data storage and similarity search
- VSA primitives (bind, bundle, negate, amplify, etc.)
- Streaming operations (accumulators for continuous learning)
- Online subspace learning (anomaly detection via CCIPCA)
- Continuous scalar encoding (rates, frequencies, etc.)
- Multiple similarity metrics

The same interface works with:
- Local CPUStore instance (direct, fastest)
- Remote HTTP API (REST requests)

Designed for clean portability to Rust (`struct Holon { ... }`).
"""

import json
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Union

import numpy as np
import requests

if TYPE_CHECKING:
    from .cpu_store import CPUStore
    from .subspace import OnlineSubspace


class HolonClient:
    """
    Unified client for Holon operations.

    This is the primary interface for working with Holon. It provides:

    **Data Operations**:
    - insert(), insert_batch() - Store data as vectors
    - search() - Find similar data
    - get() - Retrieve by ID

    **Vector Encoding**:
    - encode() - Encode structured data to vector
    - encode_scalar(), encode_scalar_log() - Encode continuous values
    - encode_sequence() - Encode ordered/unordered sequences

    **VSA Primitives** (the core operations):
    - bind() - Create associations (AND-like)
    - bundle() - Create superpositions (OR-like)
    - negate() - Remove component from superposition (NOT)
    - amplify() - Strengthen a component
    - prototype() - Extract common pattern
    - difference() - Compute what changed
    - blend() - Interpolate between vectors
    - resonance() - Extract agreeing parts

    **Streaming Operations** (for continuous learning):
    - create_accumulator() - Initialize for streaming
    - accumulate() - Add observation
    - normalize_accumulator() - Get unit vector for queries
    - threshold_accumulator() - Convert back to bipolar

    **Online Subspace Learning** (anomaly detection):
    - create_subspace() - Initialize CCIPCA subspace tracker
    - surprise_fingerprint() - Per-field anomaly attribution

    **Similarity**:
    - similarity() - Compare vectors with various metrics

    Can work with either:
    - Local CPUStore instance (direct method calls)
    - Remote HTTP API (REST requests)

    Examples:
        # Standalone client (creates its own store)
        client = HolonClient()

        # With existing store
        from holon import CPUStore
        store = CPUStore(dimensions=4096)
        client = HolonClient(local_store=store)

        # Remote API
        client = HolonClient(remote_url="http://localhost:8000")
    """

    def __init__(
        self,
        *,
        local_store: Optional["CPUStore"] = None,
        remote_url: Optional[str] = None,
        dimensions: int = 4096,
    ):
        """
        Initialize client.

        Args:
            local_store: Optional CPUStore instance for local operations.
                        If not provided and no remote_url, creates a new store.
            remote_url: Optional URL string for remote operations.
            dimensions: Vector dimensions (only used when creating new store).

        Raises:
            ValueError: If both local_store and remote_url are provided.

        Examples:
            # Standalone (creates new store)
            client = HolonClient()
            client = HolonClient(dimensions=8192)

            # With existing store
            client = HolonClient(local_store=my_store)

            # Remote usage
            client = HolonClient(remote_url="http://localhost:8000")
        """
        if local_store is not None and remote_url is not None:
            raise ValueError("Cannot specify both local_store and remote_url")

        if remote_url is not None:
            # Remote HTTP client
            self._mode = "http"
            self._base_url = remote_url.rstrip("/")
            self._session = requests.Session()
            self._dimensions = dimensions  # May not match server
        elif local_store is not None:
            # Use provided store
            self._mode = "local"
            self._store = local_store
            self._dimensions = local_store.dimensions
        else:
            # Create new local store
            from .cpu_store import CPUStore

            self._mode = "local"
            self._store = CPUStore(dimensions=dimensions)
            self._dimensions = dimensions

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def dimensions(self) -> int:
        """Vector dimensionality."""
        return self._dimensions

    @property
    def encoder(self):
        """Access the underlying encoder (local mode only)."""
        if self._mode != "local":
            raise RuntimeError("encoder is only available in local mode")
        return self._store.encoder

    @property
    def vector_manager(self):
        """Access the vector manager for atom lookups (local mode only)."""
        if self._mode != "local":
            raise RuntimeError("vector_manager is only available in local mode")
        return self._store.vector_manager

    def _to_numpy(self, vector) -> np.ndarray:
        """Convert vector to numpy array, handling TorchHD tensors."""
        # Check if it's already a numpy array
        if isinstance(vector, np.ndarray):
            return vector
        # Check if it's a torch tensor
        if hasattr(vector, "cpu") and callable(vector.cpu):
            return vector.cpu().numpy()
        # Try vector_manager if available
        if self._mode == "local" and self._store.vector_manager is not None:
            return self._store.vector_manager.to_cpu(vector)
        return vector

    def _ensure_local(self, operation: str):
        """Raise if operation requires local mode but we're remote."""
        if self._mode != "local":
            raise RuntimeError(
                f"{operation}() requires local mode. "
                "Create client with HolonClient() or HolonClient(local_store=...)"
            )

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
        *,
        probe: Union[str, Dict],
        data_type: str = "json",
        limit: int = 10,
        threshold: float = 0.0,
        guard: Optional[Dict] = None,
        negations: Optional[Dict] = None,
        similarity: Optional[Union[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items using vector similarity.

        Args:
            probe: Search probe (dict or JSON string)
            data_type: "json" or "edn"
            limit: Maximum number of results
            threshold: Similarity threshold (0.0-1.0)
            guard: Guard conditions
            negations: Negation filters
            similarity: Distance metric for Qdrant
                - None: Basic cosine similarity
                - "cosine": Cosine similarity (Qdrant native)
                - "euclidean": Euclidean distance (Qdrant native)
                - "manhattan": Manhattan distance (Qdrant native)
                - "dot_product": Dot product (Qdrant native)

        Returns:
            List of results with id, score, and data

        Examples:
            # Basic search (cosine similarity)
            results = client.search({"text": "machine learning"})

            # Named similarity methods
            results = client.search({"text": "machine learning"}, similarity="euclidean")
            results = client.search({"text": "machine learning"}, similarity="manhattan")
            results = client.search({"text": "machine learning"}, similarity="dot_product")

            # Different distance metrics
            results = client.search({"text": "machine learning"}, similarity="euclidean")

            # Full query with all options
            results = client.search(
                {"text": "machine learning"},
                limit=20,
                threshold=0.1,
                guard={"category": "ml"},
                similarity="euclidean"
            )
        """
        if isinstance(probe, dict):
            probe_str = json.dumps(probe)
        else:
            probe_str = probe

        payload = {
            "probe": probe_str,
            "data_type": data_type,
            "top_k": limit,  # Use limit parameter, but API expects top_k
            "threshold": threshold,
            "any_marker": "$any",
        }
        if guard:
            payload["guard"] = guard
        if negations:
            payload["negations"] = negations

        # Perform basic search first
        if self._mode == "http":
            response = self._session.post(
                f"{self._base_url}/api/v1/search", json=payload
            )
            response.raise_for_status()
            basic_results = response.json()["results"]
        else:
            basic_results = self._store.query(
                probe=probe_str,
                data_type=data_type,
                top_k=limit,  # API still uses top_k internally for compatibility
                threshold=threshold,
                guard=guard,
                negations=negations,
            )
            # Convert to same format as HTTP API
            # Parse JSON strings to dicts for local mode consistency
            parsed_results = []
            for item_id, score, data in basic_results:
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except (json.JSONDecodeError, TypeError):
                        pass  # Keep as string if not valid JSON
                parsed_results.append({"id": item_id, "score": score, "data": data})
            basic_results = parsed_results

        # For Qdrant-native similarity methods, results are already computed with the correct metric
        # No client-side enhancement needed - similarity parameter controls Qdrant distance metric
        return basic_results

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
            cpu_vector = self._to_numpy(vector)
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
            cpu_vector = self._to_numpy(vector)
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
            np_vectors = [np.array(vec, dtype=np.int8) for vec in vectors]

            if operation == "bind":
                result = self._store.encoder.mathematical_bind(*np_vectors)
            elif operation == "bundle":
                result = self._store.encoder.mathematical_bundle(np_vectors)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            cpu_result = self._to_numpy(result)
            return cpu_result.tolist()

    # Advanced Similarity (Minimal Kernel Addition)

    # Convenience methods for common operations

    def insert_json(self, data: Dict) -> str:
        """Insert a JSON dict (convenience method)."""
        return self.insert(data, "json")

    def insert_batch_json(self, items: List[Dict]) -> List[str]:
        """Insert multiple JSON dicts (convenience method)."""
        return self.insert_batch(items, "json")

    def search_json(self, probe: Dict, **kwargs) -> List[Dict[str, Any]]:
        """Search with JSON dict probe (convenience method)."""
        return self.search(probe=probe, data_type="json", **kwargs)

    def encode_vectors_json(self, data: Dict) -> List[float]:
        """Encode JSON dict to vector (convenience method)."""
        return self.encode_vectors(data, "json")

    # =========================================================================
    # Core Encoding (Returns numpy arrays for local operations)
    # =========================================================================

    def encode(self, data: Union[str, Dict], data_type: str = "json") -> np.ndarray:
        """
        Encode structured data to a vector.

        This is the primary encoding method for local operations. Returns
        a numpy array that can be used with VSA primitives.

        Args:
            data: Data to encode (dict or JSON/EDN string)
            data_type: "json" or "edn"

        Returns:
            Encoded vector as numpy array (int8 bipolar: {-1, 0, 1})

        Example:
            >>> client = HolonClient()
            >>> vec = client.encode({"type": "billing", "amount": 100})
            >>> vec.shape
            (4096,)
        """
        self._ensure_local("encode")

        if isinstance(data, dict):
            data_str = json.dumps(data)
        else:
            data_str = data

        from .atomizer import parse_data

        parsed = parse_data(data_str, data_type)
        return self._store.encoder.encode_data(parsed)

    def encode_walkable(self, data: Any) -> np.ndarray:
        """
        Encode any in-memory data structure using the Walkable interface.

        This is the zero-serialization path: your objects don't need to be
        converted to JSON/EDN strings first. Any object implementing the
        Walkable protocol can be encoded directly.

        Native Python types (dict, list, set, scalars) work automatically.
        Custom types can implement Walkable for custom traversal.

        Args:
            data: Any data structure (Walkable, dict, list, set, or scalar)

        Returns:
            Encoded vector as numpy array (int8 bipolar: {-1, 0, 1})

        Example:
            >>> client = HolonClient()
            >>> vec = client.encode_walkable({"type": "billing", "amount": 100})
            >>> vec.shape
            (4096,)

            # Custom types work too
            >>> class Person(Walkable):
            ...     def walk_type(self): return WalkType.MAP
            ...     def walk_map_items(self):
            ...         yield "name", self.name
            ...         yield "age", self.age
            >>> vec = client.encode_walkable(Person("Alice", 30))
        """
        self._ensure_local("encode_walkable")
        return self._store.encoder.encode_walkable(data)

    # =========================================================================
    # VSA Primitives - The Core Operations
    # =========================================================================

    def bind(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Bind two vectors (AND-like association).

        Creates a vector representing the association between two concepts.
        bind(A, B) is similar to neither A nor B individually, but
        unbind(bind(A, B), A) ≈ B.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Bound vector

        Example:
            >>> A = client.encode({"role": "user"})
            >>> B = client.encode({"name": "alice"})
            >>> AB = client.bind(A, B)  # "user named alice"
        """
        self._ensure_local("bind")
        return self._store.bind(vec1, vec2)

    def bundle(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple vectors (OR-like superposition).

        Creates a vector similar to ALL input vectors simultaneously.
        Useful for representing sets or combining multiple attributes.

        Args:
            vectors: List of vectors to combine

        Returns:
            Bundled vector (similar to all inputs)

        Example:
            >>> tags = [client.encode({"tag": t}) for t in ["python", "ml", "api"]]
            >>> combined = client.bundle(tags)
            >>> # combined is similar to all three tag vectors
        """
        self._ensure_local("bundle")
        return self._store.bundle(vectors)

    def negate(
        self, superposition: np.ndarray, component: np.ndarray, method: str = "subtract"
    ) -> np.ndarray:
        """
        Remove a component's influence from a superposition (NOT operation).

        This extends traditional VSA with negation capability.

        Args:
            superposition: Vector to remove from (e.g., bundle([A, B, C]))
            component: Vector to remove (e.g., B)
            method: "subtract" (default), "project", or "flip"

        Returns:
            Vector with component's influence diminished

        Example:
            >>> ABC = client.bundle([A, B, C])
            >>> AC = client.negate(ABC, B)  # Removes B's influence
            >>> client.similarity(AC, B) < 0  # B now has negative similarity
        """
        self._ensure_local("negate")
        return self._store.negate(superposition, component, method)

    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind to retrieve associated value (inverse of bind).

        If bound = bind(key, value), then unbind(bound, key) ≈ value.

        Args:
            bound: Previously bound vector
            key: Key to unbind with

        Returns:
            Approximate value vector

        Example:
            >>> AB = client.bind(A, B)
            >>> B_recovered = client.unbind(AB, A)
            >>> client.similarity(B_recovered, B)  # High similarity
        """
        self._ensure_local("unbind")
        return self._store.unbind(bound, key)

    def amplify(
        self, superposition: np.ndarray, component: np.ndarray, strength: float = 1.0
    ) -> np.ndarray:
        """
        Strengthen a component's presence in a superposition.

        Opposite of negate - makes a component MORE prominent.

        Args:
            superposition: Vector containing multiple components
            component: Component to amplify
            strength: How much to boost (1.0 = double, 2.0 = triple, etc.)

        Returns:
            Vector with component's influence strengthened

        Example:
            >>> ABC = client.bundle([A, B, C])
            >>> boosted = client.amplify(ABC, B, strength=2.0)
            >>> # similarity(boosted, B) > similarity(ABC, B)
        """
        self._ensure_local("amplify")
        return self._store.amplify(superposition, component, strength)

    def prototype(
        self, vectors: List[np.ndarray], threshold: float = 0.5
    ) -> np.ndarray:
        """
        Extract the common pattern from a set of vectors.

        Keeps only dimensions where a majority of vectors agree.
        Useful for finding what's shared across examples.

        Args:
            vectors: List of vectors to find consensus from
            threshold: Fraction that must agree (0.5 = majority)

        Returns:
            Vector representing the common pattern

        Example:
            >>> # Three billing records with different details
            >>> vecs = [client.encode(r) for r in billing_records]
            >>> proto = client.prototype(vecs)
            >>> # proto captures what's common to all billing records
        """
        self._ensure_local("prototype")
        return self._store.prototype(vectors, threshold)

    def difference(self, before: np.ndarray, after: np.ndarray) -> np.ndarray:
        """
        Compute what changed between two states.

        Returns a vector highlighting additions (positive) and removals (negative).

        Args:
            before: Original state
            after: New state

        Returns:
            Delta vector showing what changed

        Example:
            >>> baseline = client.normalize_accumulator(normal_traffic)
            >>> current = client.normalize_accumulator(attack_traffic)
            >>> delta = client.difference(baseline, current)
            >>> # delta highlights what's different in attack traffic
        """
        self._ensure_local("difference")
        return self._store.difference(before, after)

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
            >>> midpoint = client.blend(A, B, 0.5)  # Halfway between A and B
        """
        self._ensure_local("blend")
        return self._store.blend(vec1, vec2, alpha)

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
            >>> # Extract billing-related aspects from a mixed signal
            >>> billing_part = client.resonance(mixed_vec, billing_prototype)
        """
        self._ensure_local("resonance")
        return self._store.resonance(vec, reference)

    def permute(self, vec: np.ndarray, k: int) -> np.ndarray:
        """
        Circular shift (permutation) of vector dimensions.

        Used for positional encoding in sequences and "what comes after?" queries.

        Args:
            vec: Input vector
            k: Shift amount (positive = right, negative = left)

        Returns:
            Shifted vector

        Example:
            >>> # Sequence encoding: A at position 0, B at position 1
            >>> seq = client.bundle([A, client.permute(B, 1)])
        """
        self._ensure_local("permute")
        return self._store.permute(vec, k)

    def cleanup(self, noisy: np.ndarray, codebook: List[np.ndarray]) -> np.ndarray:
        """
        Find the closest vector in codebook to the noisy input.

        Useful for denoising composed vectors before further operations.

        Args:
            noisy: Noisy or composed input vector
            codebook: List of clean/known vectors to match against

        Returns:
            The codebook vector with highest similarity to noisy
        """
        self._ensure_local("cleanup")
        return self._store.cleanup(noisy, codebook)

    def prototype_add(
        self, prototype: np.ndarray, example: np.ndarray, count: int
    ) -> np.ndarray:
        """
        Incrementally update a prototype with a new example.

        Use this for batch updates. For streaming with frequency preservation,
        use accumulate() instead.

        Args:
            prototype: Existing prototype
            example: New example to incorporate
            count: Number of examples already in prototype (before this one)

        Returns:
            Updated prototype

        Note:
            For streaming anomaly detection, prefer create_accumulator() +
            accumulate() + normalize_accumulator() which preserve frequency.
        """
        self._ensure_local("prototype_add")
        return self._store.prototype_add(prototype, example, count)

    # =========================================================================
    # Accumulator Operations (Streaming / Continuous Learning)
    # =========================================================================

    def create_accumulator(self) -> np.ndarray:
        """
        Create a new empty accumulator for streaming operations.

        Accumulators preserve frequency information (unlike prototype_add),
        making them ideal for anomaly detection where high-frequency patterns
        should dominate.

        Returns:
            Zero-initialized float64 accumulator

        Example:
            >>> accum = client.create_accumulator()
            >>> for record in stream:
            ...     vec = client.encode(record)
            ...     accum = client.accumulate(accum, vec)
            >>> # Query using normalized accumulator
            >>> baseline = client.normalize_accumulator(accum)
        """
        self._ensure_local("create_accumulator")
        return self._store.encoder.create_accumulator()

    def accumulate(self, accumulator: np.ndarray, example: np.ndarray) -> np.ndarray:
        """
        Add an example to a running accumulator WITHOUT thresholding.

        This preserves frequency information: patterns seen 99 times contribute
        99x more than patterns seen once. Essential for streaming anomaly detection.

        Args:
            accumulator: Running float sum from create_accumulator()
            example: New vector to add (bipolar int8)

        Returns:
            Updated accumulator (float64)

        Example:
            >>> accum = client.create_accumulator()
            >>> for packet in stream:
            ...     vec = client.encode(packet)
            ...     accum = client.accumulate(accum, vec)
        """
        self._ensure_local("accumulate")
        return self._store.encoder.accumulate(accumulator, example)

    def normalize_accumulator(self, accumulator: np.ndarray) -> np.ndarray:
        """
        Normalize an accumulator for similarity queries.

        Returns a unit-normalized float vector suitable for cosine similarity.
        Frequency weighting is preserved: dimensions with high agreement have
        larger magnitudes.

        Args:
            accumulator: Float accumulator from accumulate()

        Returns:
            Unit-normalized float32 vector

        Example:
            >>> baseline = client.normalize_accumulator(warmup_accum)
            >>> for packet in stream:
            ...     vec = client.encode(packet)
            ...     sim = client.similarity(vec, baseline)
            ...     if sim < threshold:
            ...         print("Anomaly detected!")
        """
        self._ensure_local("normalize_accumulator")
        return self._store.encoder.normalize_accumulator(accumulator)

    def threshold_accumulator(self, accumulator: np.ndarray) -> np.ndarray:
        """
        Threshold an accumulator to bipolar {-1, 0, 1}.

        Use this if you need to compose with other VSA operations.
        Note: This loses some frequency information compared to
        normalize_accumulator().

        Args:
            accumulator: Float accumulator from accumulate()

        Returns:
            Bipolar int8 vector
        """
        self._ensure_local("threshold_accumulator")
        return self._store.encoder.threshold_accumulator(accumulator)

    # =========================================================================
    # Online Subspace Learning
    # =========================================================================

    def create_subspace(
        self,
        k: int = 64,
        amnesia: float = 2.0,
        sigma_mult: float = 3.5,
        ema_alpha: float = 0.01,
        reorth_interval: int = 500,
    ) -> "OnlineSubspace":
        """
        Create an OnlineSubspace for streaming anomaly detection.

        Learns the low-dimensional manifold that familiar vectors occupy
        via CCIPCA (Candid Covariance-free Incremental PCA). Vectors that
        don't project cleanly onto the manifold have high residuals.

        Args:
            k: Number of principal components to track. 32-128 typical.
                Lower = faster, higher = tighter boundary.
            amnesia: Forgetting exponent. 2.0 = moderate, 3.0 = aggressive.
                Higher values adapt faster to concept drift.
            sigma_mult: Threshold sensitivity (number of std devs).
                3.5 = conservative, 2.0 = aggressive.
            ema_alpha: EMA decay for threshold tracking.
            reorth_interval: Re-orthogonalize basis every N updates.

        Returns:
            OnlineSubspace configured with this client's dimensionality.

        Example:
            >>> sub = client.create_subspace(k=64)
            >>> for record in normal_stream:
            ...     sub.update(client.encode(record))
            >>> for record in mixed_stream:
            ...     if sub.residual(client.encode(record)) > sub.threshold:
            ...         print("anomaly")
        """
        self._ensure_local("create_subspace")
        from .subspace import OnlineSubspace

        return OnlineSubspace(
            dim=self._dimensions,
            k=k,
            amnesia=amnesia,
            ema_alpha=ema_alpha,
            sigma_mult=sigma_mult,
            reorth_interval=reorth_interval,
        )

    def surprise_fingerprint(
        self,
        vec: np.ndarray,
        subspace: "OnlineSubspace",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Compute per-field anomaly attribution from the subspace residual.

        Extracts the anomalous component (the part of vec that doesn't
        project onto the learned subspace), then unbinds each field's
        role vector to measure that field's contribution to the anomaly.

        Higher magnitude = more surprising = better rule predicate candidate.

        Args:
            vec: Encoded vector to analyze.
            subspace: Trained OnlineSubspace.
            fields: Field names to attribute. If None, returns the raw
                anomalous component norm instead.

        Returns:
            Dict mapping field name to anomaly magnitude. Fields are
            sorted by magnitude (highest = most surprising).

        Example:
            >>> sub = client.create_subspace()
            >>> # ... train on normal traffic ...
            >>> fp = client.surprise_fingerprint(attack_vec, sub,
            ...     fields=["src_ip", "dst_port", "proto", "ttl"])
            >>> # fp = {"dst_port": 44.2, "ttl": 43.8, "proto": 42.1, ...}
        """
        self._ensure_local("surprise_fingerprint")
        anomaly = subspace.anomalous_component(vec)

        if fields is None:
            return {"_total": float(np.linalg.norm(anomaly))}

        scores = {}
        for field in fields:
            role_vec = self.get_vector(field)
            field_anomaly = role_vec * anomaly  # unbind
            scores[field] = float(np.linalg.norm(field_anomaly))

        return dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    # =========================================================================
    # Continuous Scalar Encoding
    # =========================================================================

    def encode_scalar(
        self,
        value: float,
        mode: str = "linear",
        scale: float = 10000.0,
        period: float = None,
        seed: int = 42,
    ) -> np.ndarray:
        """
        Encode a continuous scalar value into a vector.

        Unlike structured data encoding (where {"x": 5} is unrelated to {"x": 6}),
        continuous encoding creates vectors where NEARBY VALUES are SIMILAR.

        Args:
            value: The scalar value to encode
            mode: "linear" or "circular"
                - "linear": Nearby values similar, no wrapping (rate, temperature)
                - "circular": Values wrap at period (angle, hour of day)
            scale: For linear mode, controls similarity decay rate
            period: For circular mode, the period of wrapping (required)
            seed: Random seed for circular mode

        Returns:
            Bipolar vector encoding

        Example:
            >>> # Rate encoding - similar rates have similar vectors
            >>> v100 = client.encode_scalar(100)
            >>> v110 = client.encode_scalar(110)
            >>> client.similarity(v100, v110)  # ~0.95 (high)

            >>> # Hour of day - wraps (23:00 similar to 00:00)
            >>> h23 = client.encode_scalar(23, mode="circular", period=24)
            >>> h0 = client.encode_scalar(0, mode="circular", period=24)
            >>> client.similarity(h23, h0)  # High - they're close on the circle
        """
        self._ensure_local("encode_scalar")
        return self._store.encode_scalar(value, mode, scale, period, seed)

    def encode_scalar_log(self, value: float, scale: float = 1000.0) -> np.ndarray:
        """
        Encode a scalar on log scale.

        Equal ratios have equal similarity:
        - 100 → 1000 is same "distance" as 1000 → 10000

        Perfect for rates, frequencies, and other multiplicative quantities.

        Args:
            value: The scalar value (must be > 0)
            scale: Controls similarity decay rate

        Returns:
            Bipolar vector encoding of log10(value)

        Example:
            >>> v100 = client.encode_scalar_log(100)    # log10 = 2
            >>> v1000 = client.encode_scalar_log(1000)  # log10 = 3
            >>> v10000 = client.encode_scalar_log(10000)  # log10 = 4
            >>> # similarity(v100, v1000) ≈ similarity(v1000, v10000)
        """
        self._ensure_local("encode_scalar_log")
        return self._store.encode_scalar_log(value, scale)

    # =========================================================================
    # Sequence Encoding
    # =========================================================================

    def encode_sequence(
        self, items: Sequence[Any], mode: str = "positional", **config
    ) -> np.ndarray:
        """
        Encode a sequence of items into a single vector.

        Args:
            items: Sequence of items to encode
            mode: Encoding mode
                - "positional": Bind each item to position vector (default)
                              Good for: ordered lists, event sequences
                - "chained": Relative binding for suffix/prefix operations
                            Good for: prefix matching, sequence reversal
                - "ngram": N-gram pairs for fuzzy substring matching
                          Good for: text search, partial phrase matching
                          Config: n_sizes=[1,2] (unigrams + bigrams)
                - "bundle": Pure superposition, no order preserved
                           Good for: bag-of-words, unordered sets
            **config: Mode-specific options

        Returns:
            Encoded vector

        Example:
            >>> # Event sequence (order matters)
            >>> events = client.encode_sequence(["login", "view", "purchase"])

            >>> # Text search (fuzzy matching)
            >>> text = client.encode_sequence(["quick", "brown", "fox"], mode="ngram")

            >>> # Tags (order doesn't matter)
            >>> tags = client.encode_sequence(["python", "ml", "api"], mode="bundle")
        """
        self._ensure_local("encode_sequence")
        return self._store.encode_sequence(items, mode, **config)

    # =========================================================================
    # Similarity
    # =========================================================================

    def similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: str = "cosine",
        **kwargs,
    ) -> float:
        """
        Compute similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector
            metric: Similarity metric
                - "cosine": Cosine similarity (default)
                - "dot": Dot product
                - "euclidean": Euclidean distance as similarity
                - "manhattan": Manhattan distance as similarity
                - "hamming": Hamming similarity (optimal for bipolar)
                - "overlap": Count of matching positions
                - "agreement": Balanced view
                - "weighted_cosine": Weighted cosine (requires weights kwarg)
            **kwargs: Metric-specific options (e.g., weights)

        Returns:
            Similarity score (higher = more similar)

        Example:
            >>> vec1 = client.encode({"type": "billing"})
            >>> vec2 = client.encode({"type": "technical"})
            >>> client.similarity(vec1, vec2)  # Default cosine
            0.35
            >>> client.similarity(vec1, vec2, metric="hamming")
            0.42
        """
        self._ensure_local("similarity")
        return self._store.similarity(vec1, vec2, metric, **kwargs)

    # =========================================================================
    # Vector Retrieval
    # =========================================================================

    def get_vector(self, atom: str) -> np.ndarray:
        """
        Get the base vector for an atomic value.

        Useful for working directly with primitive atoms.

        Args:
            atom: String atom (e.g., "billing", "user123")

        Returns:
            Base vector for the atom

        Example:
            >>> v = client.get_vector("billing")
            >>> # Same atom always produces same vector (deterministic)
        """
        self._ensure_local("get_vector")
        return self._store.vector_manager.get_vector(atom)
