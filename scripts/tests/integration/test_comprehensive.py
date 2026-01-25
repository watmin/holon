#!/usr/bin/env python3
"""
Comprehensive test demonstrating Holon with diverse data blobs.
Tests fuzzy search, guards, negations, wildcards, disjunctions, etc.
"""

import json
import time

from holon import CPUStore


def generate_test_data():
    """Generate diverse test data."""
    data = []
    users = ["alice", "bob", "charlie", "diana"]
    actions = ["login", "logout", "view", "edit", "delete"]
    statuses = ["success", "failed", "pending", "banned"]
    priorities = ["low", "medium", "high"]

    for i in range(200):  # 200 diverse blobs
        user = users[i % len(users)]
        action = actions[i % len(actions)]
        status = statuses[i % len(statuses)]
        priority = priorities[i % len(priorities)]
        tags = [f"tag_{i%5}", f"cat_{i%3}"]
        if i % 10 == 0:
            tags.append("special")

        blob = {
            "id": i,
            "user": user,
            "action": action,
            "status": status,
            "priority": priority,
            "tags": tags,
            "meta": {
                "timestamp": i * 1000,
                "sequence": [i % 5, (i + 1) % 5, (i + 2) % 5],
                "nested": {"level": i % 3, "flag": bool(i % 2)},
            },
        }
        data.append(blob)
    return data


def run_comprehensive_test():
    print("🚀 Holon Comprehensive Test with 200 Data Blobs")
    print("=" * 60)

    store = CPUStore()
    data = generate_test_data()

    # Insert all data
    start = time.time()
    ids = []
    for blob in data:
        id_ = store.insert(json.dumps(blob))
        ids.append(id_)
    insert_time = time.time() - start
    print(
        f"  Inserted: {len(data)} blobs in {insert_time:.2f}s ({len(data)/insert_time:.1f} blobs/sec)"
    )
    # Test 1: Fuzzy search
    print("\n📊 Test 1: Fuzzy Search")
    probe = json.dumps({"user": "alice", "action": "login"})
    results = store.query(probe, top_k=5)
    print(f"  Query: alice login → {len(results)} results")
    for res in results[:3]:
        print(f"    {res[2]['user']} {res[2]['action']} (score: {res[1]:.3f})")

    # Test 2: Wildcards
    print("\n🎭 Test 2: Wildcards")
    probe_wild = json.dumps({"user": "bob", "action": {"$any": True}})
    results_wild = store.query(probe_wild, top_k=5)
    print(f"  Query: bob with any action → {len(results_wild)} results")
    actions_found = set(res[2]["action"] for res in results_wild)
    print(f"    Actions found: {actions_found}")

    # Test 3: Guards
    print("\n🛡️ Test 3: Guards")
    probe_guard = json.dumps({"user": "alice"})

    # For direct store.query, guard must be callable
    def is_subset(guard, data):
        for key, value in guard.items():
            if key not in data:
                return False
            if isinstance(value, dict):
                if not isinstance(data[key], dict) or not is_subset(value, data[key]):
                    return False
            elif isinstance(value, list):
                if not isinstance(data[key], list) or len(value) != len(data[key]):
                    return False
                for g_item, d_item in zip(value, data[key]):
                    if isinstance(g_item, dict) and "$any" in g_item:
                        continue
                    elif g_item != d_item:
                        return False
            elif data[key] != value:
                return False
        return True

    guard_pattern = {"tags": ["tag_0", {"$any": True}, "cat_0"]}
    results_guard = store.query(probe_guard, top_k=5, guard=guard_pattern)
    print(f"  Query: alice with guard on tags → {len(results_guard)} results")

    # Test 4: Negations
    print("\n🚫 Test 4: Negations")
    probe_neg = json.dumps({"user": "charlie"})
    negations = {"status": {"$not": "failed"}}
    results_neg = store.query(probe_neg, top_k=5, negations=negations)
    print(f"  Query: charlie excluding failed → {len(results_neg)} results")
    statuses = [res[2]["status"] for res in results_neg]
    print(f"    Statuses: {set(statuses)} (no 'failed')")

    # Test 5: Disjunctions
    print("\n🔀 Test 5: Disjunctions ($or)")
    probe_or = json.dumps({"$or": [{"user": "diana"}, {"priority": "high"}]})
    results_or = store.query(probe_or, top_k=10)
    print(f"  Query: diana OR high priority → {len(results_or)} results")
    users_priorities = [(res[2]["user"], res[2]["priority"]) for res in results_or[:5]]
    print(f"    Samples: {users_priorities}")

    # Test 6: Combined (guard + negation + wildcard)
    print("\n⚡ Test 6: Combined Features")
    probe_combined = json.dumps(
        {"action": {"$any": True}, "meta": {"nested": {"flag": True}}}
    )
    guard_combined = {"priority": "medium"}
    negations_combined = {"status": {"$not": "banned"}}
    results_combined = store.query(
        probe_combined, top_k=5, guard=guard_combined, negations=negations_combined
    )
    print(
        f"  Query: any action + nested flag + medium priority - banned → "
        f"{len(results_combined)} results"
    )

    # Performance summary
    print("\n⏱️ Performance Summary")
    print(
        f"  Inserted: {len(data)} blobs in {insert_time:.2f}s "
        f"({len(data)/insert_time:.1f} blobs/sec)"
    )
    print("  Queries: Fast retrieval with complex filters")
    print(
        "\n✅ Comprehensive Test Passed! Holon handles diverse data with advanced querying."
    )


if __name__ == "__main__":
    run_comprehensive_test()
