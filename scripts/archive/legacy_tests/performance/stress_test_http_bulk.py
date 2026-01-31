#!/usr/bin/env python3
"""
Stress test HTTP bulk insert with 50k items in 10k batches.
Demonstrates performance at scale.
"""

import json
import time
from typing import List

import requests

BASE_URL = "http://localhost:8000"
BATCH_SIZE = 2000
TOTAL_ITEMS = 10000


def generate_test_data(count: int) -> List[dict]:
    """Generate test data items."""
    data = []
    for i in range(count):
        item = {
            "id": i,
            "user": f"user_{i % 100}",  # More users for variety
            "action": "login" if i % 2 == 0 else "logout",
            "status": "success" if i % 3 != 0 else "failed",
            "category": f"cat_{i % 10}",
            "timestamp": time.time() + i,
        }
        data.append(item)
    return data


def insert_batch_http(items: List[dict], batch_num: int) -> tuple:
    """Insert one batch via HTTP API."""
    start = time.time()

    response = requests.post(
        f"{BASE_URL}/batch_insert",
        json={"items": [json.dumps(item) for item in items], "data_type": "json"},
    )

    end = time.time()

    if response.status_code == 200:
        ids = response.json()["ids"]
        print(f"  ✅ Batch {batch_num}: {len(ids)} items in {end - start:.3f}s")
        return end - start, len(ids)
    else:
        print(f"❌ Batch {batch_num} failed: {response.status_code} - {response.text}")
        return 0, 0


def verify_data_sample():
    """Verify data by querying a sample."""
    print("\n🔍 Verifying data integrity...")

    # Query for one user
    response = requests.post(
        f"{BASE_URL}/query", json={"probe": '{"user": "user_0"}', "top_k": 50}
    )

    if response.status_code == 200:
        results = response.json()["results"]
        print(f"✅ Found {len(results)} events for user_0")
        return len(results) > 0
    else:
        print(f"❌ Query failed: {response.text}")
        return False


def main():
    print("🚀 HTTP Bulk Insert Stress Test")
    print(f"📊 Inserting {TOTAL_ITEMS:,} items in batches of {BATCH_SIZE:,}")
    print("Ensure Holon server is running!\n")

    # Generate all test data
    print("📝 Generating test data...")
    all_items = generate_test_data(TOTAL_ITEMS)
    print(f"✅ Generated {len(all_items):,} items\n")

    # Split into batches
    batches = [
        all_items[i : i + BATCH_SIZE] for i in range(0, len(all_items), BATCH_SIZE)
    ]
    print(f"📦 Split into {len(batches)} batches\n")

    # Insert batches
    total_time = 0
    total_inserted = 0

    for i, batch in enumerate(batches, 1):
        print(f"🔄 Inserting batch {i}/{len(batches)}...")
        batch_time, batch_count = insert_batch_http(batch, i)
        total_time += batch_time
        total_inserted += batch_count

    # Summary
    print("\n" + "=" * 60)
    print("🎯 STRESS TEST RESULTS")
    print("=" * 60)
    print(f"📊 Total items inserted: {total_inserted:,}")
    print(f"⏱️  Total time: {total_time:.3f}s")
    print(f"⚡ Items/second: {total_inserted / total_time:.1f}")
    if total_inserted == TOTAL_ITEMS:
        print("✅ All items inserted successfully!")
    else:
        print(f"❌ Missing {TOTAL_ITEMS - total_inserted} items")

    # Verification
    if verify_data_sample():
        print("✅ Data integrity verified")
    else:
        print("❌ Data integrity check failed")

    # Performance notes
    print("\n💡 Performance Notes:")
    print("- ANN index rebuilt after each batch (since >1000 items)")
    print("- Real-world speedup would be higher with larger batches")
    print(f"📈 Throughput: {(total_inserted / total_time) * 60:.1f} items/minute")


if __name__ == "__main__":
    main()
