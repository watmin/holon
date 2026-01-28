#!/usr/bin/env python3
"""
Bulk Operations Example: Efficient Large-Scale Data Ingestion
Demonstrates batch inserts, bulk queries, and performance optimizations using HolonClient.
"""

import json
import time

from holon import CPUStore, HolonClient


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
        data.append(item)  # Return dicts, not JSON strings

    return data


def main():
    store = CPUStore()
    client = HolonClient(local_store=store)

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
        id_ = client.insert_json(item)
        individual_ids.append(id_)
    individual_time = time.time() - start_time
    print(f"   Individual inserts: {individual_time:.3f}s")

    # Clear and test bulk inserts
    # Note: We can't directly clear through client, so we'll create a new store
    store = CPUStore()
    client = HolonClient(local_store=store)

    print("   Bulk inserts:")
    start_time = time.time()
    bulk_ids = client.insert_batch_json(test_data)
    bulk_time = time.time() - start_time
    print(f"   Bulk inserts: {bulk_time:.3f}s")

    speedup = individual_time / bulk_time
    print(f"   Speedup: {speedup:.1f}x")

    # Example 2: Bulk Query Operations
    print("\n2. Bulk Query Performance")

    # Add more data for meaningful queries
    more_data = generate_sample_data(200)
    client.insert_batch_json(more_data)

    # Query for different activity types
    activities = ["login", "logout", "view_profile"]

    print("   Querying different activities:")
    for activity in activities:
        start_time = time.time()
        results = client.search_json(probe={"activity": activity}, top_k=10)
        query_time = time.time() - start_time
        print(f"   {activity}: {query_time:.4f}s")

    # Example 3: Complex Bulk Queries with Guards
    print("\n3. Complex Queries on Bulk Data")

    # Query successful logins on mobile devices
    results = client.search_json(
        probe={"activity": "login"},
        guard={"status": "success", "device": "mobile"},
        top_k=5,
    )

    print(f"   Successful mobile logins: {len(results)}")
    for result in results[:3]:
        data = result["data"]
        print(f"   → User {data['user_id']} on {data['device']}")

    # Example 4: Bulk Operations with Negations
    print("\n4. Bulk Queries with Negations")

    # Find all activities except failed ones
    results = client.search_json(
        probe={}, negations={"status": {"$not": "failed"}}, top_k=10
    )

    print(f"   Non-failed activities: {len(results)}")
    status_counts = {}
    for result in results:
        status = result["data"]["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

    for status, count in status_counts.items():
        print(f"   → {status}: {count}")

    # Example 5: Performance Monitoring via Client
    print("\n5. Performance Monitoring")

    # Get health info from client
    health = client.health()
    print(f"   Items stored: {health['items_count']}")
    print(f"   Backend: {health['backend']}")

    # Note: Internal details like ANN index status are abstracted away
    # Users work with data, not implementation details
    print("   Vector operations: Handled internally by Holon")
    print("   Memory management: Optimized automatically")

    print("\n✅ Bulk operations examples completed!")


if __name__ == "__main__":
    main()
