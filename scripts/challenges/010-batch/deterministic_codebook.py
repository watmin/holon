#!/usr/bin/env python3
"""
Deterministic Codebook for Distributed Consensus

The key insight: if every node generates the SAME vector for the SAME atom,
they can process data independently and still reach consensus.

Current VectorManager problem:
- Uses sequential RandomState
- Order of atom requests affects codebook
- Node A requesting "billing" then "technical" ≠ Node B requesting "technical" then "billing"

Solution:
- Hash each atom to get its unique seed
- Vector generation is order-independent
- Same atom → same vector, always

This enables:
- Sharded stream processing without sync points
- Parallel encoding across nodes
- Deterministic "AI" - reproducible, no magic weights
"""

import hashlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


class DeterministicVectorManager:
    """
    Order-independent vector generation for distributed consensus.

    Key property: get_vector("X") returns the SAME vector regardless of:
    - What other atoms have been requested
    - What order atoms were requested
    - Which process/node is requesting

    This means N nodes processing sharded data independently will
    generate identical vectors for shared atoms → consensus without coordination.
    """

    def __init__(
        self,
        dimensions: int = 4096,
        global_seed: int = 42,
        sparsity: float = 0.33,  # Fraction of zeros in bipolar vector
    ):
        """
        Initialize the deterministic vector manager.

        Args:
            dimensions: Vector dimensionality
            global_seed: Base seed (all nodes must use same value!)
            sparsity: Fraction of zero elements (0.33 means ~33% zeros)
        """
        self.dimensions = dimensions
        self.global_seed = global_seed
        self.sparsity = sparsity

        # Compatibility with Holon's Encoder (expects these attributes)
        self.backend = "cpu"
        self.np = np

        # Caches (optional - for performance, not correctness)
        self.atom_vectors: Dict[str, np.ndarray] = {}
        self.position_vectors: Dict[int, np.ndarray] = {}

        # Stats
        self._cache_hits = 0
        self._cache_misses = 0

    def _atom_to_seed(self, atom: str) -> int:
        """
        Convert atom string to deterministic seed.

        Uses SHA-256 for good distribution, then combines with global_seed.
        """
        # Hash the atom to get bytes
        atom_hash = hashlib.sha256(atom.encode('utf-8')).digest()
        # Take first 8 bytes as int
        atom_int = int.from_bytes(atom_hash[:8], 'big')
        # Combine with global seed (XOR preserves determinism)
        return atom_int ^ self.global_seed

    def _position_to_seed(self, position: int) -> int:
        """Convert position to deterministic seed."""
        # Use a different mixing function for positions
        # This ensures position vectors are different from atom vectors
        # even if position number matches an atom's hash
        pos_hash = hashlib.sha256(f"__pos__{position}".encode('utf-8')).digest()
        pos_int = int.from_bytes(pos_hash[:8], 'big')
        return pos_int ^ self.global_seed

    def _generate_bipolar_vector(self, seed: int) -> np.ndarray:
        """
        Generate a deterministic bipolar vector from seed.

        Returns vector in {-1, 0, 1} with controlled sparsity.
        """
        rng = np.random.RandomState(seed & 0xFFFFFFFF)  # Ensure valid seed range

        # Generate with sparsity
        # Method: generate uniform, then threshold
        raw = rng.uniform(-1, 1, self.dimensions)

        # Create bipolar with sparsity
        # Values near 0 become 0, others become -1 or 1
        threshold = self.sparsity / 2  # Half on each side of zero
        vector = np.where(
            raw > threshold, 1,
            np.where(raw < -threshold, -1, 0)
        ).astype(np.int8)

        return vector

    def get_vector(self, atom: str) -> np.ndarray:
        """
        Get deterministic vector for an atom.

        Same atom → same vector, always, regardless of call order.
        """
        if atom in self.atom_vectors:
            self._cache_hits += 1
            return self.atom_vectors[atom]

        self._cache_misses += 1
        seed = self._atom_to_seed(atom)
        vector = self._generate_bipolar_vector(seed)
        self.atom_vectors[atom] = vector
        return vector

    def get_position_vector(self, position: int) -> np.ndarray:
        """Get deterministic position vector."""
        if position in self.position_vectors:
            return self.position_vectors[position]

        seed = self._position_to_seed(position)
        vector = self._generate_bipolar_vector(seed)
        self.position_vectors[position] = vector
        return vector

    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "atoms_cached": len(self.atom_vectors),
            "positions_cached": len(self.position_vectors),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": self._cache_hits / max(1, self._cache_hits + self._cache_misses),
        }

    def clear_cache(self):
        """Clear caches (vectors can be regenerated)."""
        self.atom_vectors.clear()
        self.position_vectors.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    def export_codebook(self) -> Dict[str, bytes]:
        """
        Export codebook for persistence.

        While vectors CAN be regenerated, exporting saves computation.
        """
        return {
            atom: vec.tobytes()
            for atom, vec in self.atom_vectors.items()
        }

    def import_codebook(self, codebook: Dict[str, bytes]):
        """Import a previously exported codebook."""
        for atom, vec_bytes in codebook.items():
            self.atom_vectors[atom] = np.frombuffer(vec_bytes, dtype=np.int8)

    def verify_determinism(self, atoms: List[str], n_trials: int = 3) -> bool:
        """
        Verify that vector generation is truly order-independent.

        Creates multiple instances and verifies they produce identical vectors.
        """
        import random

        results = []
        for trial in range(n_trials):
            # Create fresh instance
            vm = DeterministicVectorManager(
                dimensions=self.dimensions,
                global_seed=self.global_seed,
                sparsity=self.sparsity,
            )

            # Request atoms in random order
            shuffled = atoms.copy()
            random.shuffle(shuffled)

            # Generate vectors
            vectors = {atom: vm.get_vector(atom).tobytes() for atom in shuffled}
            results.append(vectors)

        # Compare all trials
        for i in range(1, len(results)):
            for atom in atoms:
                if results[0][atom] != results[i][atom]:
                    return False

        return True


def demo_distributed_consensus():
    """
    Demonstrate order-independent vector generation.

    Simulates multiple "nodes" processing atoms in different orders
    and verifying they reach consensus.
    """
    print("=" * 70)
    print("Deterministic Codebook - Distributed Consensus Demo")
    print("=" * 70)

    # Shared parameters (all nodes must agree on these)
    DIMENSIONS = 4096
    GLOBAL_SEED = 42

    # Sample atoms (would come from streaming data)
    atoms = [
        "billing", "technical", "shipping", "account",
        "user_123", "order_456", "product_789",
        "api_request", "log_entry", "metric",
        "high", "low", "medium", "critical",
    ]

    print(f"\nTest atoms: {len(atoms)}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Global seed: {GLOBAL_SEED}")

    # Simulate Node A processing in one order
    print("\n--- Node A ---")
    node_a = DeterministicVectorManager(DIMENSIONS, GLOBAL_SEED)
    order_a = atoms.copy()
    import random
    random.seed(111)
    random.shuffle(order_a)
    print(f"Processing order: {order_a[:5]}...")
    vectors_a = {atom: node_a.get_vector(atom) for atom in order_a}

    # Simulate Node B processing in different order
    print("\n--- Node B ---")
    node_b = DeterministicVectorManager(DIMENSIONS, GLOBAL_SEED)
    order_b = atoms.copy()
    random.seed(222)
    random.shuffle(order_b)
    print(f"Processing order: {order_b[:5]}...")
    vectors_b = {atom: node_b.get_vector(atom) for atom in order_b}

    # Verify consensus
    print("\n--- Consensus Check ---")
    matches = 0
    mismatches = 0
    for atom in atoms:
        if np.array_equal(vectors_a[atom], vectors_b[atom]):
            matches += 1
        else:
            mismatches += 1
            print(f"  MISMATCH: {atom}")

    print(f"Matches: {matches}/{len(atoms)}")
    print(f"Mismatches: {mismatches}")

    if mismatches == 0:
        print("\n✓ CONSENSUS ACHIEVED - All nodes agree on all vectors!")
    else:
        print("\n✗ CONSENSUS FAILED - Nodes disagree!")

    # Verify formal property
    print("\n--- Formal Verification ---")
    vm = DeterministicVectorManager(DIMENSIONS, GLOBAL_SEED)
    verified = vm.verify_determinism(atoms, n_trials=5)
    print(f"Order-independence verified: {verified}")

    # Show vector properties
    print("\n--- Vector Properties ---")
    sample_vec = vectors_a["billing"]
    print(f"Sample vector (billing):")
    print(f"  Shape: {sample_vec.shape}")
    print(f"  Dtype: {sample_vec.dtype}")
    print(f"  Values: {set(sample_vec)}")
    print(f"  Sparsity: {np.sum(sample_vec == 0) / len(sample_vec):.1%}")
    print(f"  Positive: {np.sum(sample_vec == 1) / len(sample_vec):.1%}")
    print(f"  Negative: {np.sum(sample_vec == -1) / len(sample_vec):.1%}")

    # Similarity between different atoms
    print("\n--- Atom Similarities ---")
    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    pairs = [
        ("billing", "billing"),  # Self
        ("billing", "technical"),  # Different
        ("high", "low"),  # Opposites
        ("user_123", "user_456"),  # Similar names but different atoms
    ]
    for a, b in pairs:
        vec_a = node_a.get_vector(a)
        vec_b = node_a.get_vector(b) if b in atoms else node_a.get_vector(b)
        sim = cosine_sim(vec_a, vec_b)
        print(f"  sim({a}, {b}) = {sim:.4f}")


def demo_persistence():
    """Demonstrate codebook export/import for persistence."""
    print("\n" + "=" * 70)
    print("Codebook Persistence Demo")
    print("=" * 70)

    # Create and populate
    vm1 = DeterministicVectorManager(dimensions=4096, global_seed=42)
    atoms = ["billing", "technical", "shipping"]
    for atom in atoms:
        vm1.get_vector(atom)

    print(f"Original vectors generated: {len(vm1.atom_vectors)}")

    # Export
    codebook = vm1.export_codebook()
    print(f"Exported codebook size: {sum(len(v) for v in codebook.values())} bytes")

    # Import into new instance
    vm2 = DeterministicVectorManager(dimensions=4096, global_seed=42)
    vm2.import_codebook(codebook)
    print(f"Imported vectors: {len(vm2.atom_vectors)}")

    # Verify
    matches = all(
        np.array_equal(vm1.get_vector(a), vm2.get_vector(a))
        for a in atoms
    )
    print(f"Import verified: {matches}")

    # Show that regeneration also works
    vm3 = DeterministicVectorManager(dimensions=4096, global_seed=42)
    regenerated_match = all(
        np.array_equal(vm1.get_vector(a), vm3.get_vector(a))
        for a in atoms
    )
    print(f"Regeneration also works (no import needed): {regenerated_match}")


def benchmark_generation():
    """Benchmark vector generation speed."""
    import time

    print("\n" + "=" * 70)
    print("Generation Benchmark")
    print("=" * 70)

    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)

    # Generate many unique atoms
    n_atoms = 100000
    atoms = [f"atom_{i}" for i in range(n_atoms)]

    print(f"Generating {n_atoms:,} unique vectors...")
    start = time.time()
    for atom in atoms:
        vm.get_vector(atom)
    elapsed = time.time() - start

    print(f"Time: {elapsed:.2f}s ({n_atoms/elapsed:,.0f} vectors/sec)")
    print(f"Stats: {vm.get_stats()}")

    # Cache hit performance
    print("\nCache hit benchmark...")
    start = time.time()
    for atom in atoms:
        vm.get_vector(atom)
    elapsed_cached = time.time() - start
    print(f"Cached time: {elapsed_cached:.3f}s ({n_atoms/elapsed_cached:,.0f} vectors/sec)")
    print(f"Speedup from cache: {elapsed/elapsed_cached:.1f}x")


if __name__ == "__main__":
    demo_distributed_consensus()
    demo_persistence()
    benchmark_generation()
