import hashlib
import warnings
from typing import Dict, List, Union

import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False


class VectorManager:
    """
    Vector manager for generating and caching bipolar vectors.

    By default, uses deterministic hash-based seeding so that:
    - Same atom → same vector, always, regardless of call order
    - Multiple instances with same global_seed produce identical vectors
    - Enables distributed consensus without coordination

    Args:
        dimensions: Vector dimensionality (default 16000)
        backend: "cpu" or "gpu" (requires CuPy)
        deterministic: If True (default), use hash-based seeding for
            order-independent vector generation. If False, use legacy
            sequential random state (order-dependent, deprecated).
        global_seed: Base seed for deterministic generation. All nodes
            must use the same value for distributed consensus.

    Example:
        # Two nodes with same global_seed get identical vectors
        node_a = VectorManager(global_seed=42)
        node_b = VectorManager(global_seed=42)

        # Request atoms in different order
        node_a.get_vector("billing")
        node_a.get_vector("technical")

        node_b.get_vector("technical")
        node_b.get_vector("billing")

        # Vectors are identical!
        assert np.array_equal(
            node_a.get_vector("billing"),
            node_b.get_vector("billing")
        )
    """

    def __init__(
        self,
        dimensions: int = 16000,
        backend: str = "cpu",
        deterministic: bool = True,
        global_seed: int = 42,
    ):
        self.dimensions = dimensions
        self.backend = backend
        self.deterministic = deterministic
        self.global_seed = global_seed

        self.atom_vectors: Dict[str, Union[np.ndarray, "cp.ndarray"]] = {}
        self.position_vectors: Dict[int, Union[np.ndarray, "cp.ndarray"]] = {}

        # Stats for deterministic mode
        self._cache_hits = 0
        self._cache_misses = 0

        if backend == "cpu":
            self.np = np
            if not deterministic:
                self.rng = np.random.RandomState(global_seed)
        elif backend == "gpu" and CUPY_AVAILABLE:
            self.np = cp
            if not deterministic:
                self.rng = cp.random.RandomState(global_seed)
        else:
            raise ValueError(f"Backend {backend} not supported or CuPy not available")

        if not deterministic:
            warnings.warn(
                "VectorManager(deterministic=False) is deprecated. "
                "Order-dependent vector generation will be removed in a future version. "
                "Use deterministic=True (default) for reproducible, distributed-safe vectors.",
                DeprecationWarning,
                stacklevel=2,
            )

    def _atom_to_seed(self, atom: str) -> int:
        """Convert atom string to deterministic seed using SHA-256."""
        atom_hash = hashlib.sha256(atom.encode("utf-8")).digest()
        atom_int = int.from_bytes(atom_hash[:8], "big")
        return atom_int ^ self.global_seed

    def _position_to_seed(self, position: int) -> int:
        """Convert position to deterministic seed."""
        pos_hash = hashlib.sha256(f"__pos__{position}".encode("utf-8")).digest()
        pos_int = int.from_bytes(pos_hash[:8], "big")
        return pos_int ^ self.global_seed

    def _generate_deterministic_vector(self, seed: int) -> np.ndarray:
        """Generate a deterministic bipolar vector from seed.

        Uses the same distribution as legacy mode: equal probability
        for -1, 0, 1 (1/3 each) to ensure backwards compatibility.
        """
        rng = np.random.RandomState(seed & 0xFFFFFFFF)

        # Use choice([-1, 0, 1]) to match legacy VectorManager distribution
        vector = rng.choice([-1, 0, 1], size=self.dimensions).astype(np.int8)

        # Convert to backend if needed
        if self.backend == "gpu" and CUPY_AVAILABLE:
            vector = cp.asarray(vector)

        return vector

    def get_vector(self, atom: str) -> Union[np.ndarray, "cp.ndarray"]:
        """
        Get or create a vector for an atom.

        In deterministic mode (default): same atom → same vector, always.
        In legacy mode: vectors depend on call order (deprecated).

        Args:
            atom: The atom string.

        Returns:
            Bipolar vector in {-1, 0, 1}.
        """
        if atom in self.atom_vectors:
            self._cache_hits += 1
            return self.atom_vectors[atom]

        self._cache_misses += 1

        if self.deterministic:
            seed = self._atom_to_seed(atom)
            vector = self._generate_deterministic_vector(seed)
        else:
            # Legacy order-dependent generation
            vector = self.rng.choice([-1, 0, 1], size=self.dimensions).astype(
                self.np.int8
            )

        self.atom_vectors[atom] = vector
        return vector

    def get_position_vector(self, position: int) -> Union[np.ndarray, "cp.ndarray"]:
        """Get or create a position vector for sequences."""
        if position in self.position_vectors:
            return self.position_vectors[position]

        if self.deterministic:
            seed = self._position_to_seed(position)
            vector = self._generate_deterministic_vector(seed)
        else:
            vector = self.rng.choice([-1, 0, 1], size=self.dimensions).astype(
                self.np.int8
            )

        self.position_vectors[position] = vector
        return vector

    def to_cpu(self, vector: Union[np.ndarray, "cp.ndarray"]) -> np.ndarray:
        """Convert vector to CPU numpy array."""
        if self.backend == "gpu" and CUPY_AVAILABLE:
            return cp.asnumpy(vector)
        return vector

    def to_backend(self, vector: np.ndarray) -> Union[np.ndarray, "cp.ndarray"]:
        """Convert CPU vector to backend array."""
        if self.backend == "gpu" and CUPY_AVAILABLE:
            return cp.asarray(vector)
        return vector

    def clear(self):
        """Clear all stored vectors."""
        self.atom_vectors.clear()
        self.position_vectors.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def get_stats(self) -> Dict[str, Union[int, float]]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        return {
            "atoms_cached": len(self.atom_vectors),
            "positions_cached": len(self.position_vectors),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, total),
        }

    def export_codebook(self) -> Dict[str, bytes]:
        """Export codebook for persistence."""
        return {
            atom: self.to_cpu(vec).tobytes() for atom, vec in self.atom_vectors.items()
        }

    def import_codebook(self, codebook: Dict[str, bytes]):
        """Import a previously exported codebook."""
        for atom, vec_bytes in codebook.items():
            vec = np.frombuffer(vec_bytes, dtype=np.int8).copy()
            self.atom_vectors[atom] = self.to_backend(vec)

    def verify_determinism(self, atoms: List[str], n_trials: int = 3) -> bool:
        """
        Verify that vector generation is truly order-independent.

        Creates multiple instances and verifies they produce identical vectors.
        Only meaningful when deterministic=True.
        """
        if not self.deterministic:
            return False

        import random

        results = []
        for trial in range(n_trials):
            vm = VectorManager(
                dimensions=self.dimensions,
                global_seed=self.global_seed,
                deterministic=True,
            )

            # Request atoms in random order
            shuffled = atoms.copy()
            random.shuffle(shuffled)

            vectors = {atom: vm.get_vector(atom).tobytes() for atom in shuffled}
            results.append(vectors)

        # Compare all trials
        for i in range(1, len(results)):
            for atom in atoms:
                if results[0][atom] != results[i][atom]:
                    return False

        return True


class DeterministicVectorManager(VectorManager):
    """
    Backwards compatibility alias for VectorManager with deterministic=True.

    This class exists for backwards compatibility with code written before
    VectorManager gained the deterministic parameter. New code should use
    VectorManager directly.

    Note: Default dimensions is 4096 to match original DeterministicVectorManager.
    VectorManager default is 10000.
    """

    def __init__(
        self,
        dimensions: int = 4096,
        global_seed: int = 42,
    ):
        super().__init__(
            dimensions=dimensions,
            backend="cpu",
            deterministic=True,
            global_seed=global_seed,
        )
