#!/usr/bin/env python3
"""
Test bulk insert performance improvements.
"""

import json
import time

from holon import CPUStore, HolonClient


def generate_test_data(count: int) -> list:
    """Generate test data."""
    data = []
    for i in range(count):
        item = {
            "id": i,
            "user": f"user_{i % 10}",
            "action": "login" if i % 2 == 0 else "logout",
            "status": "success" if i % 3 != 0 else "failed",
            "timestamp": time.time() + i,
        }
        data.append(item)  # Return dicts, not JSON strings
    return data


def test_individual_inserts(client, items):
    """Test inserting one by one."""
    start = time.time()
    ids = []
    for item in items:
        id_ = client.insert_json(item)
        ids.append(id_)
    end = time.time()
    return end - start, ids


def test_bulk_insert(client, items):
    """Test bulk insert."""
    start = time.time()
    ids = client.insert_batch_json(items)
    end = time.time()
    return end - start, ids


def main():
    print("🧪 Testing Bulk Insert Performance\n")

    store = CPUStore()
    client = HolonClient(local_store=store)
    test_items = generate_test_data(200)  # Small batch for demo

    # Test individual inserts
    print("Testing individual inserts...")
    time_ind, ids_ind = test_individual_inserts(client, test_items)
    print(f"  Time: {time_ind:.3f}s for {len(test_items)} inserts")

    # Create new client for bulk test (can't clear through client)
    store = CPUStore()
    client = HolonClient(local_store=store)
    print("\nTesting bulk insert...")
    time_bulk, ids_bulk = test_bulk_insert(client, test_items)
    print(f"  Time: {time_bulk:.3f}s for {len(test_items)} inserts")

    speedup = time_ind / time_bulk if time_bulk > 0 else float("inf")
    print(f"  Speedup: {speedup:.2f}x")
    # Verify same data
    assert len(ids_ind) == len(ids_bulk)
    print("✅ Both methods produced same number of inserts")

    # Test query works
    results = store.query('{"user": "user_0"}', top_k=5)
    print(f"✅ Query works: {len(results)} results found")


if __name__ == "__main__":
    main()
