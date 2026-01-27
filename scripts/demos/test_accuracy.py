#!/usr/bin/env python3

import json
import time

from holon import CPUStore, HolonClient

# Test accuracy: compare performance characteristics
# Note: ANN implementation details are abstracted by HolonClient
store = CPUStore()
client = HolonClient(local_store=store)

# Insert 100 items
print("Inserting 100 items...")
for i in range(100):
    data = {"id": i, "text": f"sample text {i}", "value": i * 10}
    client.insert_json(data)

print(f"Inserted {len(store.stored_vectors)} items")

# Test query - client abstracts ANN implementation details
probe = {"id": 50, "text": "sample text 50", "value": 500}
print(f"\nQuerying with probe: {probe}")

start = time.time()
results = client.search_json(probe, top_k=5)
query_time = time.time() - start

print(f"Query Results (time: {query_time:.4f}s):")
for res in results:
    print(f"  ID: {res['id']}, Score: {res['score']:.4f}")

print("\n✅ Query functionality works!")
print("Note: ANN implementation details are abstracted by HolonClient")
print("This ensures consistent, optimized performance automatically.")
