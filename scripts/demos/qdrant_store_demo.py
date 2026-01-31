#!/usr/bin/env python3
"""
Demo: QdrantStore - Persistent vector storage for Holon.

Shows:
1. Basic insert/query with Qdrant backend
2. Collection management (namespacing)
3. Easy collection wipe
4. Batch operations
5. Time encoding with persistence
"""

import sys
import time

sys.path.insert(0, ".")

from holon import HolonClient, QdrantStore


def demo_basic_usage():
    """Basic insert and query operations."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Usage")
    print("=" * 60)

    # Create store with a specific collection (namespace)
    store = QdrantStore(
        collection="demo_basic",
        dimensions=4096,
        recreate_collection=True,  # Start fresh
    )
    client = HolonClient(local_store=store)

    # Insert some documents
    docs = [
        {"name": "Alice", "role": "developer", "skills": ["python", "ml"]},
        {"name": "Bob", "role": "designer", "skills": ["figma", "css"]},
        {"name": "Charlie", "role": "developer", "skills": ["java", "spring"]},
        {"name": "Diana", "role": "manager", "skills": ["leadership", "planning"]},
    ]

    print(f"Inserting {len(docs)} documents...")
    for doc in docs:
        client.insert_json(doc)

    print(f"Collection count: {store.count()}")

    # Query
    print("\nSearching for developers...")
    results = client.search_json(probe={"role": "developer"}, limit=5)

    for r in results:
        print(f"  {r['score']:.3f}: {r['data']['name']} - {r['data']['role']}")


def demo_namespacing():
    """Show collection-based namespacing."""
    print("\n" + "=" * 60)
    print("DEMO 2: Namespacing with Collections")
    print("=" * 60)

    # Create two separate namespaces
    store_orders = QdrantStore(collection="ns_orders", dimensions=4096, recreate_collection=True)
    store_users = QdrantStore(collection="ns_users", dimensions=4096, recreate_collection=True)

    client_orders = HolonClient(local_store=store_orders)
    client_users = HolonClient(local_store=store_users)

    # Insert into each namespace
    client_orders.insert_json({"order_id": "123", "total": 99.99})
    client_orders.insert_json({"order_id": "456", "total": 149.99})

    client_users.insert_json({"user_id": "alice", "tier": "premium"})
    client_users.insert_json({"user_id": "bob", "tier": "basic"})

    print(f"Orders collection: {store_orders.count()} items")
    print(f"Users collection: {store_users.count()} items")

    # Each namespace is isolated
    order_results = client_orders.search_json(probe={"order_id": "123"}, limit=2)
    print(f"\nOrder search found: {len(order_results)} results")
    for r in order_results:
        print(f"  {r['data']}")


def demo_collection_wipe():
    """Show easy collection deletion."""
    print("\n" + "=" * 60)
    print("DEMO 3: Collection Wipe")
    print("=" * 60)

    store = QdrantStore(collection="demo_wipe", dimensions=4096, recreate_collection=True)
    client = HolonClient(local_store=store)

    # Insert data
    for i in range(100):
        client.insert_json({"id": i, "data": f"record_{i}"})

    print(f"Inserted: {store.count()} records")

    # Wipe it
    print("Wiping collection...")
    store.clear()

    print(f"After wipe: {store.count()} records")


def demo_batch_insert():
    """Show efficient batch operations."""
    print("\n" + "=" * 60)
    print("DEMO 4: Batch Insert")
    print("=" * 60)

    store = QdrantStore(collection="demo_batch", dimensions=4096, recreate_collection=True)

    # Prepare batch
    import json
    items = [json.dumps({"id": i, "category": f"cat_{i % 10}"}) for i in range(1000)]

    # Time batch insert
    start = time.perf_counter()
    ids = store.batch_insert(items)
    elapsed = time.perf_counter() - start

    print(f"Batch inserted {len(ids)} items in {elapsed:.2f}s")
    print(f"Rate: {len(ids) / elapsed:.0f} items/sec")
    print(f"Collection count: {store.count()}")


def demo_time_encoding():
    """Show time encoding with Qdrant persistence."""
    print("\n" + "=" * 60)
    print("DEMO 5: Time Encoding with Persistence")
    print("=" * 60)

    store = QdrantStore(collection="demo_time", dimensions=4096, recreate_collection=True)
    client = HolonClient(local_store=store)

    from datetime import datetime

    base = datetime(2024, 6, 15, 14, 0)
    base_ts = base.timestamp()

    # Insert events at different times
    events = [
        {"event": "login", "user": "alice", "ts": {"$time": base_ts}},
        {"event": "purchase", "user": "alice", "ts": {"$time": base_ts + 3600}},  # 1 hour later
        {"event": "login", "user": "bob", "ts": {"$time": base_ts + 7200}},  # 2 hours later
        {"event": "logout", "user": "alice", "ts": {"$time": base_ts + 86400}},  # 1 day later
    ]

    for event in events:
        client.insert_json(event)

    print(f"Inserted {len(events)} time-tagged events")

    # Query for events around base time
    probe = {"event": "login", "ts": {"$time": base_ts + 1800}}  # 30 min later
    results = client.search_json(probe=probe, limit=4)

    print("\nSearching for logins around base time:")
    for r in results:
        print(f"  {r['score']:.3f}: {r['data']['event']} by {r['data']['user']}")


def demo_info():
    """Show collection info."""
    print("\n" + "=" * 60)
    print("DEMO 6: Collection Info")
    print("=" * 60)

    store = QdrantStore(collection="demo_info", dimensions=4096, recreate_collection=True)
    client = HolonClient(local_store=store)

    for i in range(10):
        client.insert_json({"id": i})

    info = store.info()
    print("Collection info:")
    for key, value in info.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    demo_basic_usage()
    demo_namespacing()
    demo_collection_wipe()
    demo_batch_insert()
    demo_time_encoding()
    demo_info()

    print("\n" + "=" * 60)
    print("All demos completed!")
    print("=" * 60)
