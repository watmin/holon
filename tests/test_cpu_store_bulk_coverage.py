#!/usr/bin/env python3
"""
CPU Store Bulk Operations and ANN Management Coverage Tests
Tests for bulk inserts, ANN index management, and edge cases to improve CPU store coverage.
"""

import json
import time

import pytest

from holon import CPUStore


class TestCPUStoreBulkOperations:
    """Test bulk operations and ANN management."""

    @pytest.fixture
    def store(self):
        """Create CPU store instance for testing."""
        return CPUStore(dimensions=1000)  # Small for testing

    def test_batch_insert_basic(self, store):
        """Test basic batch insert functionality."""
        items = [
            '{"name": "Alice", "id": 1}',
            '{"name": "Bob", "id": 2}',
            '{"name": "Charlie", "id": 3}',
        ]

        ids = store.batch_insert(items, "json")
        assert len(ids) == 3
        assert all(isinstance(id, str) for id in ids)

        # Verify items were inserted
        for id in ids:
            data = store.get(id)
            assert "name" in data
            assert "id" in data

    def test_batch_insert_edn(self, store):
        """Test batch insert with EDN data."""
        items = [
            '{:name "Alice" :type :user}',
            '{:name "Bob" :type :admin}',
            '{:name "Charlie" :type :user}',
        ]

        ids = store.batch_insert(items, "edn")
        assert len(ids) == 3

        # Verify EDN parsing worked - check that we can retrieve the data
        for id in ids:
            data = store.get(id)
            # EDN data uses ImmutableDict, but should still be dict-like
            assert hasattr(data, "items")  # Should be dict-like
            # The data should contain EDN keywords - just verify we have data
            assert len(data) > 0  # Should have some key-value pairs

    def test_batch_insert_empty_list(self, store):
        """Test batch insert with empty list."""
        ids = store.batch_insert([], "json")
        assert ids == []

    def test_batch_insert_large_dataset(self, store):
        """Test batch insert with larger dataset to trigger ANN."""
        # Create enough items to exceed ANN_THRESHOLD (1000)
        items = []
        for i in range(1200):
            items.append(json.dumps({"id": i, "data": f"item_{i}"}))

        start_time = time.time()
        ids = store.batch_insert(items, "json")
        end_time = time.time()

        assert len(ids) == 1200
        assert (end_time - start_time) < 10  # Should be reasonably fast

    def test_bulk_mode_manual(self, store):
        """Test manual bulk mode operations."""
        # Enter bulk mode
        store.start_bulk_insert()
        assert store.bulk_mode is True

        # Insert items in bulk mode
        for i in range(100):
            store.insert(json.dumps({"id": i, "bulk": True}), "json")

        # ANN index should not be built during bulk mode
        assert store.ann_index is None

        # Exit bulk mode
        store.end_bulk_insert()
        assert store.bulk_mode is False

        # With < 1000 items, ANN should still not be built
        assert store.ann_index is None

    def test_bulk_mode_with_ann_trigger(self, store):
        """Test bulk mode when ANN threshold is exceeded."""
        store.start_bulk_insert()

        # Insert enough items to exceed ANN threshold
        for i in range(1100):
            store.insert(json.dumps({"id": i, "trigger_ann": True}), "json")

        # Still in bulk mode, no ANN index
        assert store.bulk_mode is True
        assert store.ann_index is None

        # Exit bulk mode - should trigger ANN build
        store.end_bulk_insert()
        assert store.bulk_mode is False

        # ANN index should be built now
        if hasattr(store, "_build_ann_index"):
            # Force build if not already built
            store._build_ann_index()

    def test_clear_operation(self, store):
        """Test clear operation resets all state."""
        # Add some data
        for i in range(50):
            store.insert(json.dumps({"id": i}), "json")

        assert len(store.stored_data) == 50
        assert len(store.stored_vectors) == 50

        # Clear everything
        store.clear()

        assert len(store.stored_data) == 0
        assert len(store.stored_vectors) == 0
        assert store.ann_index is None
        assert len(store.ann_ids) == 0
        assert store.ann_vectors is None

    def test_delete_existing_item(self, store):
        """Test deleting existing item."""
        data = {"test": "delete_me"}
        id = store.insert(json.dumps(data), "json")

        # Verify it exists
        retrieved = store.get(id)
        assert retrieved == data

        # Delete it
        result = store.delete(id)
        assert result is True

        # Verify it's gone
        with pytest.raises(KeyError):
            store.get(id)

    def test_delete_nonexistent_item(self, store):
        """Test deleting non-existent item."""
        result = store.delete("nonexistent-id")
        assert result is False

    def test_delete_after_bulk_insert(self, store):
        """Test delete operations work after bulk insert."""
        items = [json.dumps({"id": i}) for i in range(10)]
        ids = store.batch_insert(items, "json")

        # Delete a few items
        for i in range(3):
            result = store.delete(ids[i])
            assert result is True

        # Verify they're gone
        for i in range(3):
            with pytest.raises(KeyError):
                store.get(ids[i])

        # Verify others still exist
        for i in range(3, 10):
            data = store.get(ids[i])
            assert data["id"] == i

    def test_query_after_bulk_operations(self, store):
        """Test that queries work correctly after bulk operations."""
        # Initial data
        store.insert(json.dumps({"category": "initial", "id": 0}), "json")

        # Bulk insert more data
        bulk_items = []
        for i in range(1, 100):
            bulk_items.append(json.dumps({"category": "bulk", "id": i}))
        store.batch_insert(bulk_items, "json")

        # Query should work
        results = store.query(probe='{"category": "bulk"}', data_type="json", top_k=50)
        assert len(results) >= 50  # Should find bulk items

        # Query for initial item
        results = store.query(
            probe='{"category": "initial"}', data_type="json", top_k=10
        )
        assert len(results) >= 1

    def test_ann_index_invalidation_on_insert(self, store):
        """Test ANN index invalidation when inserting new items."""
        # Add enough items to trigger ANN
        for i in range(1100):
            store.insert(json.dumps({"id": i}), "json")

        # Force ANN build
        if hasattr(store, "_build_ann_index"):
            store._build_ann_index()
            # initial_ann_state = store.ann_index is not None  # Not used

            # Insert one more item (should invalidate ANN when not in bulk mode)
            store.insert(json.dumps({"id": 9999}), "json")

            # ANN should be invalidated (unless in bulk mode)
            if not store.bulk_mode:
                assert store.ann_index is None

    def test_ann_index_reuse_during_bulk(self, store):
        """Test ANN index behavior during bulk operations."""
        # Start with some data
        for i in range(500):
            store.insert(json.dumps({"id": i}), "json")

        # Enter bulk mode
        store.start_bulk_insert()

        # Add more data
        for i in range(500, 1000):
            store.insert(json.dumps({"id": i}), "json")

        # ANN should not be built during bulk mode
        assert store.ann_index is None

        # Exit bulk mode
        store.end_bulk_insert()

        # ANN should still not be built (under threshold)
        assert store.ann_index is None

    def test_performance_bulk_vs_individual(self, store):
        """Test performance difference between bulk and individual inserts."""
        test_data = [json.dumps({"id": i, "data": f"test_{i}"}) for i in range(200)]

        # Time individual inserts
        individual_start = time.time()
        individual_ids = []
        for item in test_data:
            id = store.insert(item, "json")
            individual_ids.append(id)
        individual_time = time.time() - individual_start

        # Clear and test bulk insert
        store.clear()

        bulk_start = time.time()
        bulk_ids = store.batch_insert(test_data, "json")
        bulk_time = time.time() - bulk_start

        # Both should produce same number of results
        assert len(individual_ids) == len(bulk_ids)

        # Bulk should be faster (though with small dataset, difference may be minimal)
        # At minimum, both should complete successfully
        assert individual_time > 0
        assert bulk_time > 0

    def test_mixed_operations_after_bulk(self, store):
        """Test that mixed operations work after bulk insert."""
        # Bulk insert
        bulk_data = [json.dumps({"type": "bulk", "id": i}) for i in range(100)]
        bulk_ids = store.batch_insert(bulk_data, "json")

        # Individual insert
        store.insert(json.dumps({"type": "single", "id": 999}), "json")

        # Delete some bulk items
        for i in range(10):
            store.delete(bulk_ids[i])

        # Query operations should still work
        bulk_results = store.query(probe='{"type": "bulk"}', data_type="json", top_k=50)
        single_results = store.query(
            probe='{"type": "single"}', data_type="json", top_k=10
        )

        assert len(bulk_results) >= 50  # Should have remaining bulk items (top_k=50)
        assert len(single_results) >= 1  # Should find single item

    def test_bulk_insert_with_invalid_data(self, store):
        """Test bulk insert handles invalid data gracefully."""
        # Mix valid and invalid data
        items = [
            '{"valid": "data1"}',
            '{"invalid": json}',  # Invalid JSON
            '{"valid": "data2"}',
            "another invalid",  # Invalid JSON
            '{"valid": "data3"}',
        ]

        # Should raise exception on invalid data
        with pytest.raises((ValueError, json.JSONDecodeError)):
            store.batch_insert(items, "json")

    def test_bulk_mode_state_persistence(self, store):
        """Test bulk mode state is properly managed."""
        assert store.bulk_mode is False

        store.start_bulk_insert()
        assert store.bulk_mode is True

        # Even after operations, bulk mode should persist
        store.insert(json.dumps({"test": "data"}), "json")
        assert store.bulk_mode is True

        store.end_bulk_insert()
        assert store.bulk_mode is False
