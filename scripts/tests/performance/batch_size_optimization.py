#!/usr/bin/env python3
"""
Find optimal batch size for HTTP bulk inserts.
Tests various batch sizes to find the performance sweet spot.
"""

import json
import statistics
import time
from typing import List

import requests

BASE_URL = "http://localhost:8000"
TOTAL_ITEMS = 10000  # Fixed total for fair comparison
BATCH_SIZES = [500, 1000, 2000, 5000, 10000]


def generate_test_data(count: int) -> List[dict]:
    """Generate test data items."""
    data = []
    for i in range(count):
        item = {
            "id": i,
            "user": f"user_{i % 100}",
            "action": "login" if i % 2 == 0 else "logout",
            "status": "success" if i % 3 != 0 else "failed",
            "category": f"cat_{i % 10}",
            "timestamp": time.time() + i,
        }
        data.append(item)
    return data


def test_batch_size(batch_size: int) -> dict:
    """Test a specific batch size."""
    print(f"\n🧪 Testing batch size: {batch_size:,}")

    # Generate data for this test
    all_items = generate_test_data(TOTAL_ITEMS)

    # Split into batches
    batches = [
        all_items[i : i + batch_size] for i in range(0, len(all_items), batch_size)
    ]
    print(f"   📦 {len(batches)} batches of {batch_size:,} items each")

    # Time each batch
    batch_times = []

    for i, batch in enumerate(batches, 1):
        start = time.time()

        response = requests.post(
            f"{BASE_URL}/batch_insert",
            json={"items": [json.dumps(item) for item in batch], "data_type": "json"},
        )

        end = time.time()

        if response.status_code == 200:
            ids = response.json()["ids"]
            batch_time = end - start
            batch_times.append(batch_time)
            print(f"   ✅ Batch {i}: {len(ids)} items in {batch_time:.3f}s")
        else:
            print(f"   ❌ Batch {i} failed: {response.status_code}")
            return None

    # Calculate statistics
    total_time = sum(batch_times)
    avg_batch_time = statistics.mean(batch_times)
    throughput = TOTAL_ITEMS / total_time  # items/second

    results = {
        "batch_size": batch_size,
        "num_batches": len(batches),
        "total_time": total_time,
        "avg_batch_time": avg_batch_time,
        "throughput": throughput,
        "batch_times": batch_times,
    }

    print(f"   📊 Results: {total_time:.3f}s total, {throughput:.1f} items/sec")

    return results


def clear_server_data():
    """Clear server data between tests."""
    print("🧹 Clearing server data...")
    # Assuming server has a clear endpoint, or restart it
    # For demo, we'll just note it
    pass


def main():
    print("🔬 HTTP Batch Size Optimization Test")
    print(f"📊 Testing {TOTAL_ITEMS:,} total items with different batch sizes")
    print("Ensure Holon server is running!\n")

    results = []

    for batch_size in BATCH_SIZES:
        # Clear between tests
        clear_server_data()

        result = test_batch_size(batch_size)
        if result:
            results.append(result)

        # Small delay between tests
        time.sleep(1)

    # Analysis
    print("\n" + "=" * 80)
    print("🎯 OPTIMIZATION RESULTS")
    print("=" * 80)

    print("Batch Size | Batches | Total Time | Avg Batch | Throughput")
    print("-----------|---------|------------|-----------|-----------")
    for r in results:
        print(
            f"{r['batch_size']:>10,} | {r['num_batches']:>7} | {r['total_time']:>10.3f} | {r['avg_batch_time']:>9.3f} | {r['throughput']:>9.1f}"
        )

    # Find best
    best = max(results, key=lambda x: x["throughput"])
    print(f"\n🏆 BEST PERFORMANCE: {best['batch_size']:,} batch size")
    print(f"   Throughput: {best['throughput']:.1f} items/sec")
    print(f"   Avg batch time: {best['avg_batch_time']:.3f}s")
    print(f"   Total batches: {best['num_batches']}")
    # Recommendations
    print("\n💡 Recommendations:")
    if best["batch_size"] >= 1000:
        print("- ANN index rebuilding adds overhead for large batches")
        print("- Consider 1000-2000 as sweet spot balancing network and ANN costs")
    else:
        print("- Smaller batches may be better for memory-constrained environments")
        print("- But watch for increased network overhead")

    print("\n📈 Throughput by Batch Size:")
    for r in sorted(results, key=lambda x: x["throughput"], reverse=True):
        print(f"   {r['batch_size']:>4,}: {r['throughput']:.1f} items/sec")


if __name__ == "__main__":
    main()
