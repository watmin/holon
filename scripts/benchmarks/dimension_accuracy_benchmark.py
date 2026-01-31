#!/usr/bin/env python3
"""
Benchmark to find optimal vector dimensions for accuracy vs performance.

Tests various dimension sizes against:
1. Discrimination accuracy - can we tell similar from different?
2. Time encoding accuracy - circular and positional patterns
3. Structure complexity - nested data discrimination
4. Bundle capacity - how many items before interference?
5. Memory and speed

Theory:
- VSA capacity scales roughly with sqrt(dimensions)
- Higher dimensions = more orthogonal random vectors = less interference
- But: more memory, slower computation

Usage:
    ./scripts/run_with_venv.sh python scripts/benchmarks/dimension_accuracy_benchmark.py
"""

import gc
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, ".")

from holon import CPUStore, HolonClient
from holon.similarity import normalized_dot_similarity

# Dimensions to test
DIMENSIONS = [512, 1024, 2048, 4096, 8192, 16384]

# Test parameters
NUM_RECORDS = 1000  # Records per test
NUM_QUERIES = 100  # Queries per metric


def measure_memory_per_record(dim: int) -> float:
    """Measure bytes per record at given dimension."""
    store = CPUStore(dimensions=dim)
    client = HolonClient(local_store=store)

    # Insert 100 records
    for i in range(100):
        client.insert_json({"id": i, "value": f"test_{i}"})

    # Rough estimate: vector size + overhead
    # int8 vectors = dim bytes, plus dict storage
    vector_bytes = dim  # int8 = 1 byte per dimension
    overhead = 200  # Approximate dict/metadata overhead
    return vector_bytes + overhead


def test_discrimination_accuracy(dim: int) -> dict:
    """
    Test: Can we distinguish similar items from different items?
    Insert items with varying similarity, query, and check ranking.
    """
    store = CPUStore(dimensions=dim)
    client = HolonClient(local_store=store)

    # Create categories
    categories = ["alpha", "beta", "gamma", "delta"]

    # Insert items in each category
    for cat in categories:
        for i in range(50):
            client.insert_json({
                "category": cat,
                "subcategory": f"{cat}_{i % 5}",
                "value": i,
            })

    # Test: query for category, expect same-category items to rank higher
    correct = 0
    total = 0

    for cat in categories:
        probe = {"category": cat}
        results = client.search_json(probe=probe, limit=10, threshold=0.0)

        # Count how many of top 10 are from the right category
        for r in results:
            total += 1
            if r["data"]["category"] == cat:
                correct += 1

    accuracy = correct / total if total > 0 else 0
    return {"category_accuracy": accuracy}


def test_time_discrimination(dim: int) -> dict:
    """
    Test: Can we distinguish near times from far times?
    """
    store = CPUStore(dimensions=dim)
    client = HolonClient(local_store=store)

    base = datetime(2024, 6, 15, 12, 0)
    base_ts = base.timestamp()

    # Insert events spread across time
    for hour_offset in range(-12, 13):  # -12h to +12h
        for day_offset in range(7):  # Across a week
            ts = base_ts + hour_offset * 3600 + day_offset * 86400
            client.insert_json({
                "event": "test",
                "hour_offset": hour_offset,
                "day_offset": day_offset,
                "ts": {"$time": ts},
            })

    # Query for base time, check if near times rank higher
    probe = {"event": "test", "ts": {"$time": base_ts}}
    results = client.search_json(probe=probe, limit=20, threshold=0.0)

    # Calculate average hour_offset in top results (should be close to 0)
    if results:
        top_offsets = [abs(r["data"]["hour_offset"]) for r in results[:10]]
        avg_offset = sum(top_offsets) / len(top_offsets)
        # Score: lower avg_offset is better (0 = perfect)
        time_score = max(0, 1 - avg_offset / 12)
    else:
        time_score = 0

    return {"time_proximity_score": time_score}


def test_structure_complexity(dim: int) -> dict:
    """
    Test: Can we distinguish complex nested structures?
    """
    store = CPUStore(dimensions=dim)
    client = HolonClient(local_store=store)

    # Create deeply nested structures
    for tier in ["platinum", "gold", "silver", "bronze"]:
        for region in ["us", "eu", "asia"]:
            for dept in ["sales", "eng", "support"]:
                for i in range(5):
                    client.insert_json({
                        "customer": {
                            "tier": tier,
                            "region": region,
                            "account": {
                                "dept": dept,
                                "id": i,
                            },
                        },
                    })

    # Query for specific nested structure
    correct_tier = 0
    correct_region = 0
    correct_dept = 0
    total = 0

    test_cases = [
        ("platinum", "us", "eng"),
        ("gold", "eu", "sales"),
        ("silver", "asia", "support"),
    ]

    for tier, region, dept in test_cases:
        probe = {
            "customer": {
                "tier": tier,
                "region": region,
                "account": {"dept": dept},
            },
        }
        results = client.search_json(probe=probe, limit=10, threshold=0.0)

        for r in results:
            total += 1
            cust = r["data"]["customer"]
            if cust["tier"] == tier:
                correct_tier += 1
            if cust["region"] == region:
                correct_region += 1
            if cust["account"]["dept"] == dept:
                correct_dept += 1

    return {
        "tier_accuracy": correct_tier / total if total else 0,
        "region_accuracy": correct_region / total if total else 0,
        "dept_accuracy": correct_dept / total if total else 0,
    }


def test_bundle_capacity(dim: int) -> dict:
    """
    Test: How many items can we bundle before interference degrades quality?
    Bundle increasing numbers of vectors and measure discrimination.
    """
    store = CPUStore(dimensions=dim)
    client = HolonClient(local_store=store)

    # Create prototype from bundled items
    bundle_sizes = [5, 10, 25, 50, 100]
    results = {}

    for size in bundle_sizes:
        # Insert items for this bundle
        store_test = CPUStore(dimensions=dim)
        client_test = HolonClient(local_store=store_test)

        # Insert "in-class" items (similar structure)
        for i in range(size):
            client_test.insert_json({"class": "A", "id": i, "value": f"a_{i}"})

        # Insert "out-of-class" items
        for i in range(size):
            client_test.insert_json({"class": "B", "id": i, "value": f"b_{i}"})

        # Query for class A pattern
        probe = {"class": "A"}
        search_results = client_test.search_json(probe=probe, limit=size, threshold=0.0)

        # Count correct classifications in top-size results
        correct = sum(1 for r in search_results if r["data"]["class"] == "A")
        accuracy = correct / size if size > 0 else 0
        results[f"bundle_{size}"] = accuracy

    return results


def test_query_speed(dim: int) -> dict:
    """
    Test: Query speed at this dimension.
    """
    store = CPUStore(dimensions=dim)
    client = HolonClient(local_store=store)

    # Insert records
    for i in range(NUM_RECORDS):
        client.insert_json({"id": i, "type": "record", "value": i % 100})

    # Measure query time
    start = time.perf_counter()
    for i in range(NUM_QUERIES):
        client.search_json(probe={"type": "record", "value": i % 100}, limit=10)
    elapsed = time.perf_counter() - start

    return {
        "queries_per_sec": NUM_QUERIES / elapsed,
        "avg_query_ms": (elapsed / NUM_QUERIES) * 1000,
    }


def run_benchmark():
    """Run full benchmark across all dimensions."""
    print("=" * 70)
    print("DIMENSION ACCURACY BENCHMARK")
    print("=" * 70)
    print(f"Testing dimensions: {DIMENSIONS}")
    print(f"Records per test: {NUM_RECORDS}")
    print()

    all_results = {}

    for dim in DIMENSIONS:
        print(f"\n{'=' * 70}")
        print(f"TESTING DIMENSION: {dim}")
        print("=" * 70)

        gc.collect()

        results = {"dimension": dim}

        # Memory estimate
        mem = measure_memory_per_record(dim)
        results["bytes_per_record"] = mem
        results["records_per_gb"] = int(1e9 / mem)
        print(f"Memory: ~{mem} bytes/record, ~{results['records_per_gb']:,} records/GB")

        # Discrimination
        print("Testing category discrimination...", end=" ", flush=True)
        disc = test_discrimination_accuracy(dim)
        results.update(disc)
        print(f"Accuracy: {disc['category_accuracy']:.1%}")

        # Time encoding
        print("Testing time discrimination...", end=" ", flush=True)
        time_res = test_time_discrimination(dim)
        results.update(time_res)
        print(f"Score: {time_res['time_proximity_score']:.3f}")

        # Structure complexity
        print("Testing nested structure discrimination...", end=" ", flush=True)
        struct = test_structure_complexity(dim)
        results.update(struct)
        print(
            f"Tier: {struct['tier_accuracy']:.1%}, "
            f"Region: {struct['region_accuracy']:.1%}, "
            f"Dept: {struct['dept_accuracy']:.1%}"
        )

        # Bundle capacity
        print("Testing bundle capacity...", end=" ", flush=True)
        bundle = test_bundle_capacity(dim)
        results.update(bundle)
        print(f"Bundle@100: {bundle.get('bundle_100', 0):.1%}")

        # Speed
        print("Testing query speed...", end=" ", flush=True)
        speed = test_query_speed(dim)
        results.update(speed)
        print(f"{speed['queries_per_sec']:.0f} q/s, {speed['avg_query_ms']:.2f} ms/query")

        all_results[dim] = results

    # Summary table
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Header
    print(f"{'Dim':>6} | {'Cat':>5} | {'Time':>5} | {'Tier':>5} | "
          f"{'B@100':>5} | {'q/s':>7} | {'Rec/GB':>10}")
    print("-" * 70)

    for dim in DIMENSIONS:
        r = all_results[dim]
        print(
            f"{dim:>6} | "
            f"{r['category_accuracy']:>5.1%} | "
            f"{r['time_proximity_score']:>5.3f} | "
            f"{r['tier_accuracy']:>5.1%} | "
            f"{r.get('bundle_100', 0):>5.1%} | "
            f"{r['queries_per_sec']:>7.0f} | "
            f"{r['records_per_gb']:>10,}"
        )

    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    # Find sweet spot - best accuracy with reasonable speed
    best_dim = max(
        DIMENSIONS,
        key=lambda d: (
            all_results[d]["category_accuracy"] * 0.3
            + all_results[d]["time_proximity_score"] * 0.2
            + all_results[d]["tier_accuracy"] * 0.3
            + all_results[d].get("bundle_100", 0) * 0.2
        ),
    )

    # Find most memory efficient with acceptable accuracy
    efficient_dims = [
        d for d in DIMENSIONS
        if all_results[d]["category_accuracy"] >= 0.8
        and all_results[d]["tier_accuracy"] >= 0.5
    ]
    most_efficient = min(efficient_dims, key=lambda d: d) if efficient_dims else DIMENSIONS[0]

    print(f"Best overall accuracy: {best_dim} dimensions")
    print(f"Most memory efficient (>=80% category): {most_efficient} dimensions")

    # Capacity planning
    print("\nCapacity Planning:")
    for target in [1_000_000, 10_000_000, 100_000_000, 1_000_000_000]:
        print(f"\n  {target:,} records:")
        for dim in [1024, 4096, 16384]:
            if dim in all_results:
                r = all_results[dim]
                gb_needed = target / r["records_per_gb"]
                print(f"    {dim}d: ~{gb_needed:.1f} GB, cat_acc={r['category_accuracy']:.1%}")


if __name__ == "__main__":
    run_benchmark()
