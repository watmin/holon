"""
Unit tests for QdrantStore.

Requires Qdrant running at localhost:6333.
Tests are skipped if Qdrant is not available.
"""

import json

import pytest

# Skip all tests if Qdrant is not available
try:
    from qdrant_client import QdrantClient

    client = QdrantClient(url="http://localhost:6333", timeout=2)
    client.get_collections()
    QDRANT_AVAILABLE = True
except Exception:
    QDRANT_AVAILABLE = False

pytestmark = pytest.mark.skipif(not QDRANT_AVAILABLE, reason="Qdrant not available")


from holon import HolonClient, QdrantStore


class TestQdrantStoreBasic:
    """Basic QdrantStore operations."""

    def setup_method(self):
        """Create fresh collection for each test."""
        self.store = QdrantStore(
            collection="test_basic",
            dimensions=1024,
            recreate_collection=True,
        )
        self.client = HolonClient(local_store=self.store)

    def teardown_method(self):
        """Clean up collection."""
        self.store.drop_collection()

    def test_insert_and_count(self):
        """Test inserting documents."""
        self.client.insert_json({"name": "Alice"})
        self.client.insert_json({"name": "Bob"})
        assert self.store.count() == 2

    def test_search_basic(self):
        """Test basic similarity search."""
        self.client.insert_json({"role": "developer", "name": "Alice"})
        self.client.insert_json({"role": "designer", "name": "Bob"})

        results = self.client.search_json(probe={"role": "developer"}, limit=2)

        assert len(results) == 2
        # Developer should rank first
        assert results[0]["data"]["role"] == "developer"

    def test_get_by_id(self):
        """Test retrieving by ID."""
        doc_id = self.client.insert_json({"name": "Test", "value": 123})
        data = self.store.get(doc_id)
        assert data["name"] == "Test"
        assert data["value"] == 123

    def test_delete_by_id(self):
        """Test deleting by ID."""
        doc_id = self.client.insert_json({"name": "ToDelete"})
        assert self.store.count() == 1

        result = self.store.delete(doc_id)
        assert result is True
        assert self.store.count() == 0


class TestQdrantStoreNamespacing:
    """Test collection-based namespacing."""

    def test_isolated_collections(self):
        """Collections should be isolated from each other."""
        store1 = QdrantStore(
            collection="test_ns1", dimensions=1024, recreate_collection=True
        )
        store2 = QdrantStore(
            collection="test_ns2", dimensions=1024, recreate_collection=True
        )

        client1 = HolonClient(local_store=store1)
        client2 = HolonClient(local_store=store2)

        client1.insert_json({"ns": "one"})
        client2.insert_json({"ns": "two"})
        client2.insert_json({"ns": "two"})

        assert store1.count() == 1
        assert store2.count() == 2

        # Clean up
        store1.drop_collection()
        store2.drop_collection()


class TestQdrantStoreClear:
    """Test collection clearing."""

    def test_clear_wipes_all_data(self):
        """Clear should remove all data."""
        store = QdrantStore(
            collection="test_clear",
            dimensions=1024,
            recreate_collection=True,
        )
        client = HolonClient(local_store=store)

        for i in range(10):
            client.insert_json({"id": i})

        assert store.count() == 10

        store.clear()

        assert store.count() == 0

        store.drop_collection()


class TestQdrantStoreBatch:
    """Test batch operations."""

    def setup_method(self):
        self.store = QdrantStore(
            collection="test_batch",
            dimensions=1024,
            recreate_collection=True,
        )

    def teardown_method(self):
        self.store.drop_collection()

    def test_batch_insert(self):
        """Test batch insertion."""
        items = [json.dumps({"id": i}) for i in range(100)]
        ids = self.store.batch_insert(items)

        assert len(ids) == 100
        assert self.store.count() == 100


class TestQdrantStoreGuards:
    """Test guard filtering."""

    def setup_method(self):
        self.store = QdrantStore(
            collection="test_guards",
            dimensions=1024,
            recreate_collection=True,
        )
        self.client = HolonClient(local_store=self.store)

        # Insert test data
        self.client.insert_json({"name": "Alice", "age": 30, "role": "admin"})
        self.client.insert_json({"name": "Bob", "age": 25, "role": "user"})
        self.client.insert_json({"name": "Charlie", "age": 35, "role": "user"})

    def teardown_method(self):
        self.store.drop_collection()

    def test_guard_exact_match(self):
        """Test exact value guard."""
        results = self.client.search_json(
            probe={"name": "Alice"},
            guard={"role": "admin"},
            limit=3,
        )
        assert len(results) == 1
        assert results[0]["data"]["name"] == "Alice"

    def test_guard_comparison(self):
        """Test comparison operators in guards."""
        results = self.client.search_json(
            probe={"role": "user"},
            guard={"age": {"$gt": 30}},  # Greater than 30
            limit=3,
        )
        # Only Charlie (age 35) matches (age > 30)
        assert len(results) == 1
        assert results[0]["data"]["name"] == "Charlie"


class TestQdrantStoreNegations:
    """Test negation filtering."""

    def setup_method(self):
        self.store = QdrantStore(
            collection="test_negations",
            dimensions=1024,
            recreate_collection=True,
        )
        self.client = HolonClient(local_store=self.store)

        self.client.insert_json({"type": "task", "status": "open"})
        self.client.insert_json({"type": "task", "status": "closed"})
        self.client.insert_json({"type": "bug", "status": "open"})

    def teardown_method(self):
        self.store.drop_collection()

    def test_negation_excludes_value(self):
        """Test that negations exclude matching values."""
        results = self.client.search_json(
            probe={"type": "task"},
            negations={"status": {"$not": "closed"}},
            limit=3,
        )
        # Should exclude the closed task
        for r in results:
            if r["data"]["type"] == "task":
                assert r["data"]["status"] != "closed"


class TestQdrantStoreTimeEncoding:
    """Test time encoding with Qdrant."""

    def setup_method(self):
        self.store = QdrantStore(
            collection="test_time",
            dimensions=1024,
            recreate_collection=True,
        )
        self.client = HolonClient(local_store=self.store)

    def teardown_method(self):
        self.store.drop_collection()

    def test_time_marker_persists(self):
        """Test that time markers work with Qdrant."""
        from datetime import datetime

        base_ts = datetime(2024, 6, 15, 12, 0).timestamp()

        self.client.insert_json(
            {
                "event": "test",
                "ts": {"$time": base_ts},
            }
        )

        results = self.client.search_json(
            probe={"event": "test", "ts": {"$time": base_ts + 3600}},
            limit=1,
        )

        assert len(results) == 1
        assert results[0]["data"]["event"] == "test"
