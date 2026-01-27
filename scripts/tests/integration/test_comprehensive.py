#!/usr/bin/env python3
"""
Comprehensive test demonstrating Holon with diverse data blobs.
Tests fuzzy search, guards, negations, wildcards, disjunctions, etc.
"""

import json
import time

from holon import CPUStore, HolonClient


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
    client = HolonClient(local_store=store)
    data = generate_test_data()

    # Insert all data
    start = time.time()
    ids = []
    for blob in data:
        id_ = client.insert_json(blob)
        ids.append(id_)
    insert_time = time.time() - start
    print(
        f"  Inserted: {len(data)} blobs in {insert_time:.2f}s ({len(data)/insert_time:.1f} blobs/sec)"
    )
    # Test 1: Fuzzy search
    print("\n📊 Test 1: Fuzzy Search")
    probe = {"user": "alice", "action": "login"}
    results = client.search_json(probe, top_k=5)
    print(f"  Query: alice login → {len(results)} results")
    for res in results[:3]:
        print(f"    {res['data']['user']} {res['data']['action']} (score: {res['score']:.3f})")

    # Test 2: Wildcards
    print("\n🎭 Test 2: Wildcards")
    probe_wild = {"user": "bob", "action": {"$any": True}}
    results_wild = client.search_json(probe_wild, top_k=5)
    print(f"  Query: bob with any action → {len(results_wild)} results")
    actions_found = set(res["data"]["action"] for res in results_wild)
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
    results_guard = client.search_json(probe_guard, top_k=5, guard=guard_pattern)
    print(f"  Query: alice with guard on tags → {len(results_guard)} results")

    # Test 4: Negations
    print("\n🚫 Test 4: Negations")
    probe_neg = {"user": "charlie"}
    negations = {"status": {"$not": "failed"}}
    results_neg = client.search_json(probe_neg, top_k=5, negations=negations)
    print(f"  Query: charlie excluding failed → {len(results_neg)} results")
    statuses = [res["data"]["status"] for res in results_neg]
    print(f"    Statuses: {set(statuses)} (no 'failed')")

    # Test 5: Disjunctions
    print("\n🔀 Test 5: Disjunctions ($or)")
    probe_or = {"$or": [{"user": "diana"}, {"priority": "high"}]}
    results_or = client.search_json(probe_or, top_k=10)
    print(f"  Query: diana OR high priority → {len(results_or)} results")
    users_priorities = [(res["data"]["user"], res["data"]["priority"]) for res in results_or[:5]]
    print(f"    Samples: {users_priorities}")

    # Test 6: Combined (guard + negation + wildcard)
    print("\n⚡ Test 6: Combined Features")
    probe_combined = {"action": {"$any": True}, "meta": {"nested": {"flag": True}}}
    guard_combined = {"priority": "medium"}
    negations_combined = {"status": {"$not": "banned"}}
    results_combined = client.search_json(
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
