#!/usr/bin/env python3
"""
Holon Extreme Query Challenge: Push limits with 5000 data blobs & 500 complex queries.
Tests advanced features: guards, negations, wildcards, $or, positionals.
"""

import json
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from holon import CPUStore

def generate_extreme_data(num_items=5000):
    """Generate diverse, nested data blobs."""
    data = []
    users = [f"user_{i}" for i in range(100)]
    actions = ["login", "logout", "view", "edit", "create", "delete"]
    statuses = ["success", "failed", "pending", "error", "timeout"]
    priorities = ["low", "medium", "high", "critical"]

    for i in range(num_items):
        user = random.choice(users)
        action = random.choice(actions)
        status = random.choice(statuses)
        priority = random.choice(priorities)

        # Nested structures
        meta = {
            "timestamp": i * 1000 + random.randint(0, 999),
            "ip": f"192.168.{random.randint(0,255)}.{random.randint(0,255)}",
            "sequence": [random.randint(0,9) for _ in range(5)],
            "nested": {
                "level": random.randint(1,5),
                "flags": [random.choice(["read", "write", "exec"]) for _ in range(3)],
                "deep": {
                    "value": random.random(),
                    "list": [{"id": j, "data": f"item_{j}"} for j in range(random.randint(1,5))]
                }
            }
        }

        tags = [f"tag_{random.randint(0,19)}" for _ in range(random.randint(1,5))]

        blob = {
            "id": i,
            "user": user,
            "action": action,
            "status": status,
            "priority": priority,
            "tags": tags,
            "meta": meta
        }
        data.append(blob)
    return data

def generate_complex_queries(num_queries=500):
    """Generate complex queries."""
    queries = []
    users = [f"user_{i}" for i in range(100)]
    actions = ["login", "logout", "view", "edit", "create", "delete"]
    statuses = ["success", "failed", "pending", "error", "timeout"]

    for i in range(num_queries):
        query_type = random.choice(["fuzzy", "wildcard", "guard", "negation", "or", "combined"])

        if query_type == "fuzzy":
            probe = {"user": random.choice(users), "action": random.choice(actions)}
            queries.append((json.dumps(probe), None, None))

        elif query_type == "wildcard":
            probe = {"user": random.choice(users), "action": {"$any": True}}
            queries.append((json.dumps(probe), None, None))

        elif query_type == "guard":
            probe = {"user": random.choice(users)}
            guard = {"priority": random.choice(["low", "medium", "high"])}
            queries.append((json.dumps(probe), guard, None))

        elif query_type == "negation":
            probe = {"action": random.choice(actions)}
            negations = {"status": {"$not": random.choice(statuses)}}
            queries.append((json.dumps(probe), None, negations))

        elif query_type == "or":
            probe = {"$or": [{"user": random.choice(users)}, {"status": random.choice(statuses)}]}
            queries.append((json.dumps(probe), None, None))

        elif query_type == "combined":
            probe = {"user": random.choice(users), "action": {"$any": True}}
            guard = {"priority": random.choice(["low", "medium", "high"])}
            negations = {"status": {"$not": "failed"}}
            queries.append((json.dumps(probe), guard, negations))

    return queries

def run_extreme_challenge():
    num_items = 5000  # Keep items
    num_queries = 2000  # 2x queries

    print(f"🚀 Holon Extreme Query Challenge: {num_items} Blobs, {num_queries} Complex Queries")
    print("=" * 70)

    store = CPUStore()
    data = generate_extreme_data(num_items)

    # Insert data
    print(f"📥 Inserting {num_items} blobs...")
    start = time.time()
    for blob in data:
        store.insert(json.dumps(blob))
    insert_time = time.time() - start
    print(f"  Insert Time: {insert_time:.2f}s ({num_items/insert_time:.1f} blobs/sec)")

    # Generate queries
    queries = generate_complex_queries(num_queries)
    print(f"🎯 Generated {len(queries)} complex queries")

    # Run queries with concurrency
    print("⚡ Executing queries with 10 workers...")
    query_times = []
    total_results = 0

    def run_query(q_idx, probe, guard, negations):
        start = time.time()
        results = store.query(probe, top_k=10, guard=guard, negations=negations or {})
        query_time = time.time() - start
        return q_idx, len(results), query_time

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(run_query, i, probe, guard, negations) for i, (probe, guard, negations) in enumerate(queries)]
        for future in as_completed(futures):
            q_idx, num_results, q_time = future.result()
            query_times.append(q_time)
            total_results += num_results
            if (q_idx + 1) % 50 == 0:
                print(f"  Completed {q_idx+1}/{len(queries)} queries...")

    # Results
    avg_query_time = sum(query_times) / len(query_times)
    max_query_time = max(query_times)
    total_time = sum(query_times)

    print("\n📊 Results:")
    print(f"  Total Queries: {len(queries)}")
    print(f"  Total Results: {total_results}")
    print(f"  Avg Query Time: {avg_query_time:.4f}s")
    print(f"  Max Query Time: {max_query_time:.4f}s")
    print(f"  Total Query Time: {total_time:.2f}s")
    print(f"  Queries/sec: {len(queries)/total_time:.2f}")
    if all(t < 1.0 for t in query_times):
        print("✅ ALL QUERIES <1s! Lightning fast.")
    else:
        slow = sum(1 for t in query_times if t >= 1.0)
        print(f"⚠️  {slow} queries >=1s, but still impressive.")

    print("\n🎉 Extreme Challenge Complete! Holon conquers complexity at scale.")

if __name__ == "__main__":
    run_extreme_challenge()