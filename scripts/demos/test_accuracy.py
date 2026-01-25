#!/usr/bin/env python3

import json
import time

from holon import CPUStore

# Test accuracy: compare brute-force vs ANN results
store = CPUStore()

# Temporarily set threshold low to force ANN
import holon.cpu_store

holon.cpu_store.ANN_THRESHOLD = 5  # Force ANN after 5 items

# Insert 100 items
print("Inserting 100 items...")
for i in range(100):
    data = json.dumps({"id": i, "text": f"sample text {i}", "value": i * 10})
    store.insert(data)

print(f"Inserted {len(store.stored_vectors)} items")

# Test query
probe = json.dumps({"id": 50, "text": "sample text 50", "value": 500})
print(f"\nQuerying with probe: {probe}")

# Force ANN
if store.ann_index is None:
    store._build_ann_index()
    print("ANN index built")

# Get ANN results
start = time.time()
ann_results = store.query(probe, top_k=5)
ann_time = time.time() - start

print(f"ANN Results (time: {ann_time:.4f}s):")
for res in ann_results:
    print(f"  ID: {res[0]}, Score: {res[1]:.4f}")

# Now disable ANN temporarily to get brute-force
holon.cpu_store.FAISS_AVAILABLE = False
store.ann_index = None  # Clear ANN

start = time.time()
brute_results = store.query(probe, top_k=5)
brute_time = time.time() - start

print(f"\nBrute-force Results (time: {brute_time:.4f}s):")
for res in brute_results:
    print(f"  ID: {res[0]}, Score: {res[1]:.4f}")

# Compare
ann_scores = [(r[0], r[1]) for r in ann_results]
brute_scores = [(r[0], r[1]) for r in brute_results]

if ann_scores == brute_scores:
    print("\n✅ ACCURACY: ANN and brute-force results are identical!")
else:
    print("\n❌ ACCURACY: Results differ!")
    print("ANN:", ann_scores)
    print("Brute:", brute_scores)

print(f"\nPerformance: ANN {ann_time:.4f}s vs Brute {brute_time:.4f}s")
