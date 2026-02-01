#!/usr/bin/env python3
"""
Benchmark: Parallel vs Sequential Encoding

Compares insert performance with:
1. Sequential batch_insert (single-threaded encoding)
2. Parallel parallel_batch_insert (multi-core encoding)

Usage:
    ./scripts/run_with_venv.sh python scripts/benchmarks/parallel_encoding_benchmark.py
"""

import json
import multiprocessing as mp
import sys
import time
from datetime import datetime

sys.path.insert(0, ".")

from holon import QdrantStore

DIMENSIONS = 4096
NUM_ITEMS = 10000  # Items per test


def generate_documents(count: int) -> list:
    """Generate test documents."""
    base_ts = datetime(2024, 1, 1).timestamp()
    items = []

    for i in range(count):
        doc = {
            "id": i,
            "category": f"cat_{i % 100}",
            "customer": {"tier": ["bronze", "silver", "gold", "platinum"][i % 4]},
            "value": i % 1000,
            "ts": {"$time": base_ts + i * 3600},
        }
        items.append(json.dumps(doc))

    return items


def benchmark_sequential(items: list) -> float:
    """Benchmark sequential batch_insert."""
    store = QdrantStore(
        collection="bench_sequential",
        dimensions=DIMENSIONS,
        recreate_collection=True,
    )

    start = time.perf_counter()
    store.batch_insert(items)
    elapsed = time.perf_counter() - start

    count = store.count()
    store.drop_collection()

    return elapsed, count


def benchmark_parallel(items: list, num_workers: int) -> float:
    """Benchmark parallel_batch_insert."""
    store = QdrantStore(
        collection="bench_parallel",
        dimensions=DIMENSIONS,
        recreate_collection=True,
    )

    start = time.perf_counter()
    store.parallel_batch_insert(items, num_workers=num_workers)
    elapsed = time.perf_counter() - start

    count = store.count()
    store.drop_collection()

    return elapsed, count


def main():
    print("=" * 60)
    print("PARALLEL ENCODING BENCHMARK")
    print("=" * 60)
    print(f"Items: {NUM_ITEMS:,}")
    print(f"Dimensions: {DIMENSIONS}")
    print(f"CPU cores: {mp.cpu_count()}")
    print()

    # Generate test data
    print("Generating test documents...", end=" ", flush=True)
    items = generate_documents(NUM_ITEMS)
    print(f"done ({len(items):,} items)")

    # Sequential benchmark
    print("\nSequential batch_insert...", end=" ", flush=True)
    seq_time, seq_count = benchmark_sequential(items)
    seq_rate = NUM_ITEMS / seq_time
    print(f"{seq_time:.1f}s ({seq_rate:.0f} items/sec)")

    # Parallel benchmarks with different worker counts
    print("\nParallel batch_insert:")
    results = []

    for workers in [2, 4, 8, mp.cpu_count()]:
        print(f"  {workers} workers...", end=" ", flush=True)
        par_time, par_count = benchmark_parallel(items, workers)
        par_rate = NUM_ITEMS / par_time
        speedup = seq_time / par_time
        results.append((workers, par_time, par_rate, speedup))
        print(f"{par_time:.1f}s ({par_rate:.0f} items/sec, {speedup:.1f}x speedup)")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Workers':>8} | {'Time':>8} | {'Rate':>12} | {'Speedup':>8}")
    print("-" * 45)
    print(f"{'1 (seq)':>8} | {seq_time:>7.1f}s | {seq_rate:>10.0f}/s | {'1.0x':>8}")

    for workers, par_time, par_rate, speedup in results:
        print(f"{workers:>8} | {par_time:>7.1f}s | {par_rate:>10.0f}/s | {speedup:>7.1f}x")

    # Best result
    best = max(results, key=lambda x: x[3])
    print(f"\nBest: {best[0]} workers ({best[3]:.1f}x speedup, {best[2]:.0f} items/sec)")


if __name__ == "__main__":
    main()
