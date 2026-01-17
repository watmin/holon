#!/usr/bin/env python3
"""
Extreme HTTP Test: Host Holon in-memory like Redis, hammer with API client insertions/queries.
"""

import json
import time
import random
import subprocess
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_extreme_data(num_items=1000):
    """Generate data for extreme test."""
    data = []
    for i in range(num_items):
        blob = {
            "id": i,
            "user": f"user_{i % 100}",
            "action": random.choice(["login", "logout", "view", "edit"]),
            "status": random.choice(["success", "failed", "pending"]),
            "meta": {
                "timestamp": i * 1000,
                "value": random.random()
            }
        }
        data.append(blob)
    return data

def insert_data_http(base_url, data, max_workers=10):
    """Insert data via HTTP with concurrency."""
    def insert_item(item):
        response = requests.post(f"{base_url}/insert", json={
            "data": json.dumps(item),
            "data_type": "json"
        })
        return response.status_code == 200

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(insert_item, item) for item in data]
        results = [f.result() for f in as_completed(futures)]
    return all(results)

def generate_queries(num_queries=500):
    """Generate queries for test."""
    queries = []
    for _ in range(num_queries):
        query_type = random.choice(["fuzzy", "guard", "negation"])
        if query_type == "fuzzy":
            queries.append({
                "probe": json.dumps({"user": f"user_{random.randint(0,99)}"}),
                "top_k": 5
            })
        elif query_type == "guard":
            queries.append({
                "probe": json.dumps({"action": random.choice(["login", "view"])}),
                "guard": json.dumps({"status": "success"}),
                "top_k": 5
            })
        elif query_type == "negation":
            queries.append({
                "probe": json.dumps({"action": random.choice(["edit", "logout"])}),
                "negations": {"status": {"$not": "failed"}},
                "top_k": 5
            })
    return queries

def query_http(base_url, queries, max_workers=20):
    """Query via HTTP with concurrency."""
    def run_query(q):
        response = requests.post(f"{base_url}/query", json=q)
        if response.status_code == 200:
            return len(response.json()["results"])
        return 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_query, q) for q in queries]
        results = [f.result() for f in as_completed(futures)]
    return results

def main():
    base_url = "http://localhost:8000"
    num_items = 1000
    num_queries = 500

    print("🚀 Extreme HTTP Test: Holon as In-Memory API Service")
    print("=" * 60)

    # Start server
    print("Starting Holon server...")
    server = subprocess.Popen(["python", "scripts/holon_server.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3)  # Wait for startup

    try:
        # Generate data
        data = generate_extreme_data(num_items)
        print(f"📦 Generated {num_items} data blobs")

        # Insert data
        print("📤 Inserting data via HTTP...")
        start = time.time()
        success = insert_data_http(base_url, data)
        insert_time = time.time() - start
        print(f"  Insert Time: {insert_time:.2f}s ({num_items/insert_time:.1f} items/sec)")
        if not success:
            print("❌ Insertions failed!")
            return

        # Generate queries
        queries = generate_queries(num_queries)
        print(f"🎯 Generated {num_queries} queries")

        # Run queries
        print("⚡ Executing queries via HTTP...")
        start = time.time()
        results = query_http(base_url, queries)
        query_time = time.time() - start
        total_results = sum(results)
        avg_results = total_results / len(results)

        print("\n📊 Results:")
        print(f"  Queries Executed: {len(results)}")
        print(f"  Total Results: {total_results}")
        print(f"  Avg Results/Query: {avg_results:.2f}")
        print(f"  Query Time: {query_time:.2f}s")
        print(f"  Queries/sec: {len(results)/query_time:.2f}")

        if all(r > 0 for r in results):
            print("✅ All queries returned results!")
        else:
            failed = sum(1 for r in results if r == 0)
            print(f"⚠️  {failed} queries returned no results")

    finally:
        server.terminate()
        server.wait()
        print("Server stopped.")

if __name__ == "__main__":
    main()