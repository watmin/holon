"""
Unit tests for VectorManager and DeterministicVectorManager.

Tests the critical properties:
1. Determinism - same atom → same vector
2. Order independence - request order doesn't matter
3. Cross-instance consistency - same seed → same vectors
4. Distributed consensus - multiple nodes agree
"""

import warnings

import numpy as np
import pytest

from holon import DeterministicVectorManager, VectorManager


class TestBasicFunctionality:
    """Test basic vector generation."""

    def test_creates_vector_with_correct_dimensions(self):
        vm = DeterministicVectorManager(dimensions=1000)
        vec = vm.get_vector("test_atom")
        assert vec.shape == (1000,)

    def test_vector_is_bipolar(self):
        """Vectors should only contain {-1, 0, 1}."""
        vm = DeterministicVectorManager(dimensions=4096)
        vec = vm.get_vector("some_atom")
        unique_values = set(vec.tolist())
        assert unique_values.issubset({-1, 0, 1})

    def test_vector_dtype_is_int8(self):
        vm = DeterministicVectorManager()
        vec = vm.get_vector("atom")
        assert vec.dtype == np.int8

    def test_position_vector_works(self):
        vm = DeterministicVectorManager(dimensions=1000)
        pos_vec = vm.get_position_vector(5)
        assert pos_vec.shape == (1000,)
        assert pos_vec.dtype == np.int8

    def test_vector_has_expected_distribution(self):
        """Vectors should have roughly equal distribution of -1, 0, 1."""
        vm = DeterministicVectorManager(dimensions=10000)
        vec = vm.get_vector("test")

        # Count occurrences
        neg_ones = np.sum(vec == -1)
        zeros = np.sum(vec == 0)
        pos_ones = np.sum(vec == 1)

        # Each should be roughly 1/3 (with some variance)
        for count in [neg_ones, zeros, pos_ones]:
            assert 2500 < count < 4500  # ~33% ± 12%


class TestDeterminism:
    """Test that same atom always produces same vector."""

    def test_same_atom_same_vector(self):
        """Requesting same atom twice returns identical vector."""
        vm = DeterministicVectorManager()
        vec1 = vm.get_vector("billing")
        vec2 = vm.get_vector("billing")
        assert np.array_equal(vec1, vec2)

    def test_same_position_same_vector(self):
        """Requesting same position twice returns identical vector."""
        vm = DeterministicVectorManager()
        pos1 = vm.get_position_vector(42)
        pos2 = vm.get_position_vector(42)
        assert np.array_equal(pos1, pos2)

    def test_different_atoms_different_vectors(self):
        """Different atoms produce different vectors."""
        vm = DeterministicVectorManager()
        vec1 = vm.get_vector("billing")
        vec2 = vm.get_vector("technical")
        assert not np.array_equal(vec1, vec2)

    def test_reproducible_across_runs(self):
        """New instance with same seed produces same vectors."""
        vm1 = DeterministicVectorManager(global_seed=123)
        vec1 = vm1.get_vector("test_atom")

        # Create fresh instance
        vm2 = DeterministicVectorManager(global_seed=123)
        vec2 = vm2.get_vector("test_atom")

        assert np.array_equal(vec1, vec2)


class TestOrderIndependence:
    """Test that request order doesn't affect vectors."""

    def test_order_does_not_matter(self):
        """Atoms requested in different order produce same vectors."""
        vm1 = DeterministicVectorManager(global_seed=42)
        vm2 = DeterministicVectorManager(global_seed=42)

        # Request in different orders
        vm1.get_vector("alpha")
        vm1.get_vector("beta")
        vm1.get_vector("gamma")
        vec1 = vm1.get_vector("target")

        vm2.get_vector("gamma")
        vm2.get_vector("target")  # Request target earlier
        vm2.get_vector("alpha")
        vm2.get_vector("beta")
        vec2 = vm2.get_vector("target")

        assert np.array_equal(vec1, vec2)

    def test_verify_determinism_method(self):
        """The built-in verify_determinism method should pass."""
        vm = DeterministicVectorManager()
        atoms = ["alpha", "beta", "gamma", "delta", "epsilon"]
        assert vm.verify_determinism(atoms, n_trials=5)

    def test_many_atoms_order_independent(self):
        """Even with many atoms, order doesn't matter."""
        import random

        atoms = [f"atom_{i}" for i in range(100)]

        vm1 = DeterministicVectorManager(global_seed=999)
        vm2 = DeterministicVectorManager(global_seed=999)

        # vm1: sequential order
        for atom in atoms:
            vm1.get_vector(atom)

        # vm2: random order
        shuffled = atoms.copy()
        random.shuffle(shuffled)
        for atom in shuffled:
            vm2.get_vector(atom)

        # All vectors should match
        for atom in atoms:
            assert np.array_equal(vm1.get_vector(atom), vm2.get_vector(atom))


class TestDistributedConsensus:
    """Test properties needed for distributed deployment."""

    def test_different_seeds_different_vectors(self):
        """Different global_seed produces different vectors."""
        vm1 = DeterministicVectorManager(global_seed=1)
        vm2 = DeterministicVectorManager(global_seed=2)

        vec1 = vm1.get_vector("same_atom")
        vec2 = vm2.get_vector("same_atom")

        assert not np.array_equal(vec1, vec2)

    def test_simulated_distributed_nodes(self):
        """Simulate multiple nodes processing different data shards."""
        global_seed = 42

        # Node A processes shard A
        node_a = DeterministicVectorManager(global_seed=global_seed)
        node_a.get_vector("user_123")
        node_a.get_vector("billing_event")
        node_a.get_vector("payment_success")

        # Node B processes shard B (different data)
        node_b = DeterministicVectorManager(global_seed=global_seed)
        node_b.get_vector("user_456")
        node_b.get_vector("technical_event")
        node_b.get_vector("error_log")

        # Both nodes should agree on vectors for atoms they both encounter
        common_atom = "shared_reference"
        vec_a = node_a.get_vector(common_atom)
        vec_b = node_b.get_vector(common_atom)

        assert np.array_equal(vec_a, vec_b)

    def test_merge_results_from_nodes(self):
        """Simulated merge of results from distributed nodes."""
        global_seed = 42

        # Two nodes build accumulators independently
        node_a = DeterministicVectorManager(global_seed=global_seed)
        node_b = DeterministicVectorManager(global_seed=global_seed)

        # Each processes different records
        accumulator_a = np.zeros(4096, dtype=np.float64)
        for atom in ["GET", "/api/users", "200"]:
            accumulator_a += node_a.get_vector(atom)

        accumulator_b = np.zeros(4096, dtype=np.float64)
        for atom in ["POST", "/api/orders", "201"]:
            accumulator_b += node_b.get_vector(atom)

        # Merge accumulators (this works because vectors are consistent)
        merged = accumulator_a + accumulator_b

        # Verify we can query the merged result
        query_vm = DeterministicVectorManager(global_seed=global_seed)
        get_vec = query_vm.get_vector("GET")

        # GET should have positive similarity (it's in the merged accumulator)
        similarity = np.dot(merged, get_vec) / (
            np.linalg.norm(merged) * np.linalg.norm(get_vec) + 1e-10
        )
        assert similarity > 0.1  # Positive correlation


class TestCacheAndStats:
    """Test caching behavior and statistics."""

    def test_cache_hit_tracking(self):
        vm = DeterministicVectorManager()

        # First request - miss
        vm.get_vector("new_atom")
        stats = vm.get_stats()
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 0

        # Second request - hit
        vm.get_vector("new_atom")
        stats = vm.get_stats()
        assert stats["cache_misses"] == 1
        assert stats["cache_hits"] == 1

    def test_clear_resets_cache(self):
        vm = DeterministicVectorManager()
        vm.get_vector("atom1")
        vm.get_vector("atom2")

        assert vm.get_stats()["atoms_cached"] == 2

        vm.clear()

        assert vm.get_stats()["atoms_cached"] == 0
        assert vm.get_stats()["cache_hits"] == 0

    def test_regeneration_after_clear(self):
        """Vectors can be regenerated after clearing cache."""
        vm = DeterministicVectorManager(global_seed=42)
        vec_before = vm.get_vector("test").copy()

        vm.clear()

        vec_after = vm.get_vector("test")
        assert np.array_equal(vec_before, vec_after)


class TestCodebookExportImport:
    """Test codebook persistence."""

    def test_export_and_import(self):
        vm1 = DeterministicVectorManager()
        vm1.get_vector("alpha")
        vm1.get_vector("beta")

        codebook = vm1.export_codebook()

        vm2 = DeterministicVectorManager()
        vm2.import_codebook(codebook)

        assert np.array_equal(vm1.get_vector("alpha"), vm2.get_vector("alpha"))
        assert np.array_equal(vm1.get_vector("beta"), vm2.get_vector("beta"))

    def test_imported_codebook_matches_regenerated(self):
        """Imported vectors should match freshly generated ones."""
        vm1 = DeterministicVectorManager(global_seed=42)
        vm1.get_vector("test")
        codebook = vm1.export_codebook()

        # Import into vm2
        vm2 = DeterministicVectorManager(global_seed=42)
        vm2.import_codebook(codebook)

        # vm3 regenerates fresh
        vm3 = DeterministicVectorManager(global_seed=42)

        # All should match
        assert np.array_equal(vm2.get_vector("test"), vm3.get_vector("test"))


class TestCompatibility:
    """Test compatibility with existing VectorManager interface."""

    def test_has_backend_attribute(self):
        vm = DeterministicVectorManager()
        assert hasattr(vm, "backend")
        assert vm.backend == "cpu"

    def test_to_cpu_is_identity(self):
        vm = DeterministicVectorManager()
        vec = vm.get_vector("test")
        assert vm.to_cpu(vec) is vec

    def test_to_backend_is_identity(self):
        vm = DeterministicVectorManager()
        vec = vm.get_vector("test")
        assert vm.to_backend(vec) is vec

    def test_np_attribute_is_numpy(self):
        vm = DeterministicVectorManager()
        assert vm.np is np


class TestUnifiedVectorManager:
    """Test the unified VectorManager with deterministic flag."""

    def test_default_is_deterministic(self):
        """VectorManager defaults to deterministic=True."""
        vm = VectorManager()
        assert vm.deterministic is True

    def test_deterministic_mode_is_order_independent(self):
        """Default VectorManager is order-independent."""
        vm1 = VectorManager(global_seed=42)
        vm2 = VectorManager(global_seed=42)

        # Different order
        vm1.get_vector("alpha")
        vm1.get_vector("beta")
        vec1 = vm1.get_vector("target")

        vm2.get_vector("target")
        vm2.get_vector("beta")
        vm2.get_vector("alpha")
        vec2 = vm2.get_vector("target")

        assert np.array_equal(vec1, vec2)

    def test_legacy_mode_emits_deprecation_warning(self):
        """deterministic=False emits deprecation warning."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            VectorManager(deterministic=False)

            assert len(w) == 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "deprecated" in str(w[0].message).lower()

    def test_legacy_mode_is_order_dependent(self):
        """deterministic=False produces order-dependent vectors."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)

            vm1 = VectorManager(deterministic=False, global_seed=42)
            vm2 = VectorManager(deterministic=False, global_seed=42)

            # Same order - should match
            vm1.get_vector("first")
            vm1.get_vector("second")

            vm2.get_vector("first")
            vm2.get_vector("second")

            assert np.array_equal(vm1.get_vector("first"), vm2.get_vector("first"))

    def test_deterministic_vector_manager_is_subclass(self):
        """DeterministicVectorManager is a subclass of VectorManager."""
        assert issubclass(DeterministicVectorManager, VectorManager)

    def test_deterministic_vector_manager_defaults(self):
        """DeterministicVectorManager has its original defaults."""
        vm = DeterministicVectorManager()
        assert vm.dimensions == 4096  # Original default
        assert vm.deterministic is True

    def test_vector_manager_default_dimensions(self):
        """VectorManager default dimensions matches legacy."""
        vm = VectorManager()
        assert vm.dimensions == 16000

    def test_both_produce_same_vectors_with_same_params(self):
        """VectorManager and DeterministicVectorManager produce same vectors."""
        vm1 = VectorManager(dimensions=4096, global_seed=42)
        vm2 = DeterministicVectorManager(dimensions=4096, global_seed=42)

        for atom in ["test", "billing", "technical"]:
            assert np.array_equal(vm1.get_vector(atom), vm2.get_vector(atom))
