#!/usr/bin/env python3
"""
Qdrant Scale Stress Test

Tests Qdrant-backed Holon at scale:
1. Insert rate with increasing data
2. Query performance at various scales
3. Time encoding accuracy at scale
4. Memory usage (client-side only, Qdrant handles storage)

Usage:
    ./scripts/run_with_venv.sh python scripts/stress_tests/qdrant_scale_stress_test.py
"""

import gc
import json
import sys
import time
from datetime import datetime

import psutil

sys.path.insert(0, ".")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from holon import HolonClient, QdrantStore

# Configuration
DIMENSIONS = 4096  # Recommended dimension
BATCH_SIZE = 100  # Items per batch for Qdrant
COLLECTION_NAME = "stress_test"

# Scale targets
SCALE_CHECKPOINTS = [1_000, 10_000, 50_000, 100_000, 500_000, 1_000_000]


def get_memory_mb():
    """Get current process memory in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


def generate_document(i: int, base_ts: float) -> dict:
    """Generate a test document with time encoding."""
    category = f"cat_{i % 100}"
    subcategory = f"sub_{i % 20}"
    tier = ["bronze", "silver", "gold", "platinum"][i % 4]

    # Time spread over 1 year
    time_offset = (i % 365) * 86400 + (i % 24) * 3600

    return {
        "id": i,
        "category": category,
        "subcategory": subcategory,
        "customer": {"tier": tier, "region": ["us", "eu", "asia"][i % 3]},
        "value": i % 1000,
        "ts": {"$time": base_ts + time_offset},
    }


def run_insert_benchmark(store: QdrantStore, target_count: int, base_ts: float):
    """Insert documents up to target count."""
    current_count = store.count()
    to_insert = target_count - current_count

    if to_insert <= 0:
        return 0, 0

    print(f"  Inserting {to_insert:,} documents...")

    items = []
    start = time.perf_counter()

    for i in range(current_count, target_count):
        doc = generate_document(i, base_ts)
        items.append(json.dumps(doc))

        if len(items) >= BATCH_SIZE:
            store.batch_insert(items)
            items = []

    if items:
        store.batch_insert(items)

    elapsed = time.perf_counter() - start
    rate = to_insert / elapsed

    return elapsed, rate


def run_query_benchmark(client: HolonClient, num_queries: int = 100):
    """Run query benchmark."""
    start = time.perf_counter()

    for i in range(num_queries):
        # Vary the queries
        if i % 3 == 0:
            probe = {"category": f"cat_{i % 100}"}
        elif i % 3 == 1:
            probe = {"customer": {"tier": "platinum"}}
        else:
            probe = {"category": f"cat_{i % 50}", "customer": {"tier": "gold"}}

        client.search_json(probe=probe, limit=10)

    elapsed = time.perf_counter() - start
    return elapsed, num_queries / elapsed


def run_time_query_benchmark(client: HolonClient, base_ts: float, num_queries: int = 50):
    """Run time-based query benchmark."""
    start = time.perf_counter()

    for i in range(num_queries):
        # Query for specific time windows
        query_ts = base_ts + (i * 7 * 86400)  # Weekly intervals
        probe = {
            "customer": {"tier": "platinum"},
            "ts": {"$time": query_ts},
        }
        client.search_json(probe=probe, limit=10)

    elapsed = time.perf_counter() - start
    return elapsed, num_queries / elapsed


def run_stress_test():
    """Run the full stress test."""
    print("=" * 70)
    print("QDRANT SCALE STRESS TEST")
    print("=" * 70)
    print(f"Dimensions: {DIMENSIONS}")
    print(f"Scale targets: {SCALE_CHECKPOINTS}")
    print()

    # Create fresh store
    store = QdrantStore(
        collection=COLLECTION_NAME,
        dimensions=DIMENSIONS,
        recreate_collection=True,
    )
    client = HolonClient(local_store=store)

    base_ts = datetime(2024, 1, 1).timestamp()

    results = []

    for target in SCALE_CHECKPOINTS:
        print(f"\n{'=' * 70}")
        print(f"CHECKPOINT: {target:,} records")
        print("=" * 70)

        gc.collect()
        mem_before = get_memory_mb()

        # Insert
        insert_time, insert_rate = run_insert_benchmark(store, target, base_ts)
        actual_count = store.count()

        print(f"  Inserted in {insert_time:.1f}s ({insert_rate:.0f} items/sec)")
        print(f"  Collection count: {actual_count:,}")

        # Query benchmark
        print("  Running query benchmark...", end=" ", flush=True)
        query_time, query_rate = run_query_benchmark(client, num_queries=100)
        print(f"{query_rate:.1f} q/s")

        # Time query benchmark
        print("  Running time query benchmark...", end=" ", flush=True)
        time_query_time, time_query_rate = run_time_query_benchmark(
            client, base_ts, num_queries=50
        )
        print(f"{time_query_rate:.1f} q/s")

        mem_after = get_memory_mb()

        result = {
            "target": target,
            "actual": actual_count,
            "insert_time": insert_time,
            "insert_rate": insert_rate,
            "query_rate": query_rate,
            "time_query_rate": time_query_rate,
            "client_memory_mb": mem_after,
            "memory_delta_mb": mem_after - mem_before,
        }
        results.append(result)

        print(f"  Client memory: {mem_after:.0f} MB (delta: {mem_after - mem_before:+.0f} MB)")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Records':>12} | {'Insert':>10} | {'Query':>8} | {'Time Q':>8} | {'Client MB':>10}")
    print("-" * 60)

    for r in results:
        print(
            f"{r['actual']:>12,} | "
            f"{r['insert_rate']:>10.0f}/s | "
            f"{r['query_rate']:>8.1f}/s | "
            f"{r['time_query_rate']:>8.1f}/s | "
            f"{r['client_memory_mb']:>10.0f}"
        )

    # Get Qdrant server info
    print("\n" + "=" * 70)
    print("QDRANT COLLECTION INFO")
    print("=" * 70)
    info = store.info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Cleanup option
    print("\n" + "=" * 70)
    final_count = store.count()
    print(f"Final collection size: {final_count:,} records")
    print(f"Collection: {COLLECTION_NAME}")
    print("To clean up: store.drop_collection()")


if __name__ == "__main__":
    run_stress_test()
