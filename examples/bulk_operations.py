#!/usr/bin/env python3
"""
Bulk Operations Example: Efficient Large-Scale Data Ingestion
Demonstrates batch inserts, bulk queries, and performance optimizations.
"""

import json
import time

from holon import CPUStore


def generate_sample_data(count=100):
    """Generate sample user activity data."""
    activities = ["login", "logout", "view_profile", "edit_settings", "upload_file"]
    devices = ["desktop", "mobile", "tablet"]
    statuses = ["success", "failed", "pending"]

    data = []
    for i in range(count):
        item = {
            "user_id": f"user_{i % 20}",  # 20 different users
            "activity": activities[i % len(activities)],
            "device": devices[i % len(devices)],
            "status": statuses[i % len(statuses)],
            "timestamp": f"2024-01-{str((i % 28) + 1).zfill(2)}",
            "session_id": f"session_{i % 10}",
        }
        data.append(json.dumps(item))

    return data


def main():
    store = CPUStore()

    print("📦 Bulk Operations Examples")
    print("=" * 50)

    # Example 1: Individual vs Bulk Insert Performance
    print("\n1. Insert Performance Comparison")

    test_data = generate_sample_data(50)

    # Individual inserts
    print("   Individual inserts:")
    start_time = time.time()
    individual_ids = []
    for item in test_data:
        id_ = store.insert(item)
        individual_ids.append(id_)
    individual_time = time.time() - start_time
    print(f"   Individual inserts: {individual_time:.3f}s")

    # Clear and test bulk inserts
    store.clear()

    print("   Bulk inserts:")
    start_time = time.time()
    bulk_ids = store.batch_insert(test_data)
    bulk_time = time.time() - start_time
    print(f"   Bulk inserts: {bulk_time:.3f}s")

    speedup = individual_time / bulk_time
    print(f"   Speedup: {speedup:.1f}x")

    # Example 2: Bulk Query Operations
    print("\n2. Bulk Query Performance")

    # Add more data for meaningful queries
    more_data = generate_sample_data(200)
    store.batch_insert(more_data)

    print(f"   Total items in store: {len(store.stored_data)}")

    # Query for different activity types
    activities = ["login", "logout", "view_profile"]

    print("   Querying different activities:")
    for activity in activities:
        start_time = time.time()
        results = store.query(f'{{"activity": "{activity}"}}', top_k=10)
        query_time = time.time() - start_time
        print(f"   {activity}: {query_time:.4f}s")

    # Example 3: Complex Bulk Queries with Guards
    print("\n3. Complex Queries on Bulk Data")

    # Query successful logins on mobile devices
    results = store.query(
        '{"activity": "login"}',
        guard={"status": "success", "device": "mobile"},
        top_k=5,
    )

    print(f"   Successful mobile logins: {len(results)}")
    for result in results[:3]:
        data = result[2]
        print(f"   → User {data['user_id']} on {data['device']}")

    # Example 4: Bulk Operations with Negations
    print("\n4. Bulk Queries with Negations")

    # Find all activities except failed ones
    results = store.query("{}", negations={"status": {"$not": "failed"}}, top_k=10)

    print(f"   Non-failed activities: {len(results)}")
    status_counts = {}
    for result in results:
        status = result[2]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in status_counts.items():
        print(f"   → {status}: {count}")

    # Example 5: Memory and Performance Monitoring
    print("\n5. Performance Monitoring")

    print(f"   Items stored: {len(store.stored_data)}")
    print(f"   Vectors cached: {len(store.stored_vectors)}")

    # Check if ANN index is active
    if hasattr(store, "ann_index") and store.ann_index is not None:
        print("   ANN index: Active (optimized for queries)")
    else:
        print("   ANN index: Inactive (brute-force mode)")

    # Estimate memory usage
    estimated_memory = len(store.stored_data) * 70  # ~70KB per item
    print(f"   Estimated memory: ~{estimated_memory}KB")

    print("\n✅ Bulk operations examples completed!")


if __name__ == "__main__":
    main()
