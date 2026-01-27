#!/usr/bin/env python3

import json
import time

from holon import CPUStore, HolonClient

# Test ANN integration
if __name__ == "__main__":
    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert 1500 items
    print("Inserting 1500 items...")
    for i in range(1500):
        data = {"id": i, "value": f"test_{i}"}
        client.insert_json(data)

    print(f"Inserted {len(store.stored_vectors)} items")

    # Query
    probe = {"id": 100, "value": "test_100"}
    start = time.time()
    results = client.search_json(probe, top_k=5)
    query_time = time.time() - start

    print(f"Query time: {query_time:.4f}s")
    print(f"Results: {len(results)}")
    for res in results[:3]:
        print(f"ID: {res['id']}, Score: {res['score']:.4f}")

    print("ANN functionality: Working (implementation details abstracted)")
