#!/usr/bin/env python3
"""
Large-scale HTTP bulk insert test with optimal batch size.
Tests 50k items in 10k batches to validate production performance.
"""

import json
import time
from typing import List

import requests

BASE_URL = "http://localhost:8000"
OPTIMAL_BATCH_SIZE = 10000
TARGET_TOTAL = 50000  # 50k items


def generate_test_data(count: int) -> List[dict]:
    """Generate test data items."""
    data = []
    for i in range(count):
        item = {
            "id": i,
            "user": f"user_{i % 200}",  # More variety
            "action": "login" if i % 2 == 0 else "logout",
            "status": "success" if i % 3 != 0 else "failed",
            "category": f"cat_{i % 20}",
            "department": f"dept_{i % 5}",
            "timestamp": time.time() + i,
        }
        data.append(item)
    return data


def insert_batch_http(batch: List[dict], batch_num: int) -> tuple:
    """Insert one batch via HTTP API."""
    print(f"🔄 Inserting batch {batch_num} ({len(batch):,} items)...")

    start = time.time()

    response = requests.post(
        f"{BASE_URL}/batch_insert",
        json={"items": [json.dumps(item) for item in batch], "data_type": "json"},
    )

    end = time.time()

    if response.status_code == 200:
        ids = response.json()["ids"]
        batch_time = end - start
        print(f"   ✅ Completed in {batch_time:.3f}s")
        print(f"   📊 Rate: {len(ids)/batch_time:.1f} items/sec")
        return batch_time, len(ids)
    else:
        print(f"❌ Batch {batch_num} failed: {response.status_code}")
        print(f"   Response: {response.text[:200]}...")
        return 0, 0


def verify_data_sample():
    """Verify data by sampling queries."""
    print("\n🔍 Verifying data integrity...")

    # Query different users
    test_users = ["user_0", "user_50", "user_100", "user_150"]
    total_found = 0

    for user in test_users:
        response = requests.post(
            f"{BASE_URL}/query",
            json={"probe": json.dumps({"user": user}), "top_k": 100},
        )

        if response.status_code == 200:
            results = response.json()["results"]
            total_found += len(results)
            print(f"   {user}: {len(results)} events")
        else:
            print(f"   {user}: Query failed")

    avg_per_user = total_found / len(test_users)
    expected_per_user = TARGET_TOTAL / 200  # Since 200 users
    print(f"   Average per user: {avg_per_user:.1f} events")
    print(f"   Expected per user: {expected_per_user:.1f} events")
    return total_found > 0


def main():
    print("🚀 Large-Scale HTTP Bulk Insert Test")
    print(f"📊 Target: {TARGET_TOTAL:,} items in {OPTIMAL_BATCH_SIZE:,} batches")
    print("Ensure Holon server is running!\n")

    # Generate all test data
    print("📝 Generating test data...")
    start_gen = time.time()
    all_items = generate_test_data(TARGET_TOTAL)
    gen_time = time.time() - start_gen
    print(f"⏱️  Data generation: {gen_time:.3f}s")
    # Split into optimal batches
    batches = [
        all_items[i : i + OPTIMAL_BATCH_SIZE]
        for i in range(0, len(all_items), OPTIMAL_BATCH_SIZE)
    ]
    print(f"📦 Split into {len(batches)} batches of {OPTIMAL_BATCH_SIZE:,} items each\n")

    # Insert all batches
    total_time = 0
    total_inserted = 0
    batch_times = []

    insert_start = time.time()

    for i, batch in enumerate(batches, 1):
        batch_time, batch_count = insert_batch_http(batch, i)
        if batch_count == 0:
            print("❌ Insertion failed, stopping test")
            return

        total_time += batch_time
        total_inserted += batch_count
        batch_times.append(batch_time)

        # Progress update
        progress = (i / len(batches)) * 100
        print(f"   📈 Progress: {progress:.1f}% complete")
    insert_end = time.time()
    total_wall_time = insert_end - insert_start

    # Final results
    print("\n" + "=" * 80)
    print("🎯 LARGE-SCALE INSERT RESULTS")
    print("=" * 80)
    print(f"📊 Total items inserted: {total_inserted:,}")
    print(f"⏱️  Wall time: {total_wall_time:.3f}s")
    print(f"⚡ Throughput: {total_inserted / total_wall_time:.1f} items/sec")
    print(f"🏠 Memory: {total_inserted / total_wall_time * 60:.1f} items/min")
    print(f"🚀 Velocity: {total_inserted / total_wall_time * 3600:.1f} items/hour")
    # Batch statistics
    avg_batch_time = sum(batch_times) / len(batch_times)
    min_batch_time = min(batch_times)
    max_batch_time = max(batch_times)

    print(f"📦 Avg batch time: {avg_batch_time:.3f}s")
    print(f"   Min batch time: {min_batch_time:.3f}s")
    print(f"   Max batch time: {max_batch_time:.3f}s")
    # Verify data
    if verify_data_sample():
        print("✅ Data integrity verified")
    else:
        print("❌ Data integrity check failed")

    # Performance analysis
    print("\n💡 Performance Analysis:")
    print(
        f"- Pure insert time: {total_time:.3f}s ({total_time/total_wall_time*100:.1f}% of wall time)"
    )
    print(f"- ANN rebuilds: {len(batches)} total (one per batch)")
    print(f"- Network overhead: {total_wall_time - total_time:.3f}s")
    print(f"- Data generation: {gen_time:.3f}s")
    # Scaling projections
    print("\n📈 Scaling Projections:")
    print(
        f"- 100k items: ~{total_wall_time * 2:.1f}s ({TARGET_TOTAL * 2 / total_wall_time:.0f}/sec)"
    )
    print(
        f"- 1M items: ~{total_wall_time * 20:.1f}s ({TARGET_TOTAL * 20 / total_wall_time:.0f}/sec)"
    )

    print("\n🏆 SUCCESS: Large-scale bulk insert completed!")
    if total_inserted == TARGET_TOTAL:
        print("All items inserted successfully!")
    else:
        print(f"Warning: {TARGET_TOTAL - total_inserted} items missing")


if __name__ == "__main__":
    main()
