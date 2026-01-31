#!/usr/bin/env python3
"""
Test ANN vs Brute Force performance.
"""

import sys
import os
import time
import json
import random
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from holon.cpu_store import CPUStore, FAISS_AVAILABLE, ANN_THRESHOLD
from holon.client import HolonClient
import numpy as np


def main():
    print("="*60)
    print("ANN vs BRUTE FORCE PERFORMANCE TEST")
    print("="*60)

    print(f"\nFAISS Available: {FAISS_AVAILABLE}")
    print(f"ANN Threshold: {ANN_THRESHOLD} items")

    if not FAISS_AVAILABLE:
        print("ERROR: FAISS not available!")
        return

    # Test at different scales
    scales = [500, 1000, 2000, 5000, 10000, 20000]

    print("\n" + "-"*60)
    print(f"{'Items':>10} | {'ANN Index':>10} | {'Search (ms)':>12} | {'Mode':>15}")
    print("-"*60)

    for n in scales:
        gc.collect()

        store = CPUStore(dimensions=16000)
        client = HolonClient(local_store=store)

        # Insert items
        for i in range(n):
            record = {
                "id": i,
                "type": random.choice(["task", "event", "note"]),
                "value": f"item_{i}",
            }
            client.insert_json(record)

        # Insert target
        target = {"id": 99999, "type": "target", "special": "findme"}
        client.insert_json(target)

        # Check ANN status before query
        has_ann_before = store.ann_index is not None

        # Search (this should trigger ANN build if > threshold)
        start = time.time()
        results = client.search_json(probe={"type": "target", "special": "findme"}, limit=10)
        search_time = (time.time() - start) * 1000

        # Check ANN status after query
        has_ann_after = store.ann_index is not None

        # Determine mode
        if n <= ANN_THRESHOLD:
            mode = "Brute Force"
        elif has_ann_after:
            mode = "ANN (FAISS)"
        else:
            mode = "Brute Force (ANN failed?)"

        found = any("findme" in str(r.get("data", "")) for r in results)

        print(f"{n:>10} | {has_ann_after!s:>10} | {search_time:>12.2f} | {mode:>15} | Found: {found}")

        del store, client

    # Direct comparison at 10k items
    print("\n" + "="*60)
    print("DIRECT COMPARISON: Brute Force vs ANN at 10k items")
    print("="*60)

    n = 10000

    # Brute force (disable ANN)
    store_bf = CPUStore(dimensions=16000)
    client_bf = HolonClient(local_store=store_bf)

    for i in range(n):
        client_bf.insert_json({"id": i, "value": f"item_{i}"})
    client_bf.insert_json({"id": 99999, "special": "target"})

    # Force no ANN
    store_bf.ann_index = None
    original_faiss = sys.modules.get('holon.cpu_store', None)

    # Measure brute force
    bf_times = []
    for _ in range(5):
        store_bf.ann_index = None  # Keep clearing it
        start = time.time()
        # Force brute force by temporarily setting threshold very high
        results = []
        probe_vec = np.array(client_bf.encode_vectors({"special": "target"}, "json"))
        for item_id, vec_data in store_bf.stored_vectors.items():
            vec = vec_data["vector"]
            sim = np.dot(probe_vec, vec) / (np.linalg.norm(probe_vec) * np.linalg.norm(vec) + 1e-10)
            results.append((item_id, sim))
        results.sort(key=lambda x: x[1], reverse=True)
        bf_times.append((time.time() - start) * 1000)

    avg_bf = np.mean(bf_times)
    print(f"\nBrute Force: {avg_bf:.2f}ms (avg of 5)")

    # ANN
    store_ann = CPUStore(dimensions=16000)
    client_ann = HolonClient(local_store=store_ann)

    for i in range(n):
        client_ann.insert_json({"id": i, "value": f"item_{i}"})
    client_ann.insert_json({"id": 99999, "special": "target"})

    # Force ANN build
    store_ann._build_ann_index()
    print(f"ANN Index built: {store_ann.ann_index is not None}")

    ann_times = []
    for _ in range(5):
        start = time.time()
        results = client_ann.search_json(probe={"special": "target"}, limit=10)
        ann_times.append((time.time() - start) * 1000)

    avg_ann = np.mean(ann_times)
    print(f"ANN (FAISS): {avg_ann:.2f}ms (avg of 5)")

    if avg_bf > 0:
        speedup = avg_bf / avg_ann
        print(f"\nSpeedup: {speedup:.1f}x")

    # Check if target found
    found = any("target" in str(r.get("data", "")) for r in results)
    print(f"Target found: {found}")


if __name__ == "__main__":
    main()
