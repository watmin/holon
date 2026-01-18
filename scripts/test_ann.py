#!/usr/bin/env python3

from holon import CPUStore
import time
import json

# Test ANN integration
if __name__ == "__main__":
    store = CPUStore()

    # Insert 1500 items
    print("Inserting 1500 items...")
    for i in range(1500):
        data = json.dumps({"id": i, "value": f"test_{i}"})
        store.insert(data)

    print(f"Inserted {len(store.stored_vectors)} items")

    # Query
    probe = json.dumps({"id": 100, "value": "test_100"})
    start = time.time()
    results = store.query(probe, top_k=5)
    query_time = time.time() - start

    print(f"Query time: {query_time:.4f}s")
    print(f"Results: {len(results)}")
    for res in results[:3]:
        print(f"ID: {res[0]}, Score: {res[1]:.4f}")

    print("ANN index status:", "Built" if store.ann_index is not None else "Not built")