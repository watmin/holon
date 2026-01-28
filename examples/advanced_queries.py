#!/usr/bin/env python3
"""
Advanced Queries Example: Complex Guards, Negations, and $or Logic
Demonstrates sophisticated query patterns with compound conditions using HolonClient.
"""

from holon import CPUStore, HolonClient


def main():
    # Initialize store and create unified client
    store = CPUStore()
    client = HolonClient(local_store=store)

    # Create diverse test data
    data = [
        {
            "user": "alice",
            "role": "developer",
            "status": "active",
            "priority": "high",
            "project": "web",
        },
        {
            "user": "bob",
            "role": "designer",
            "status": "active",
            "priority": "medium",
            "project": "mobile",
        },
        {
            "user": "charlie",
            "role": "developer",
            "status": "inactive",
            "priority": "low",
            "project": "web",
        },
        {
            "user": "diana",
            "role": "manager",
            "status": "active",
            "priority": "high",
            "project": "backend",
        },
        {
            "user": "eve",
            "role": "developer",
            "status": "active",
            "priority": "medium",
            "project": "mobile",
        },
        {
            "user": "frank",
            "role": "analyst",
            "status": "inactive",
            "priority": "low",
            "project": "data",
        },
    ]

    # Insert data using client (convenience method for JSON)
    for item in data:
        client.insert_json(item)

    print("🔍 Advanced Query Examples")
    print("=" * 50)

    # Example 1: Complex $or guards
    print("\n1. Complex OR Conditions in Guards")
    print("   Query: High priority developers OR active managers")

    results = client.search_json(
        probe={},
        guard={
            "$or": [
                {"priority": "high", "role": "developer"},
                {"status": "active", "role": "manager"},
            ]
        },
    )

    for result in results:
        data = result["data"]
        print(
            f"   → {data['user']}: {data['role']} ({data['priority']} priority, {data['status']})"
        )

    # Example 2: Nested OR logic
    print("\n2. Nested OR Logic")
    print("   Query: Web project with high priority OR active mobile developers")

    results = client.search_json(
        probe={"project": "web"},
        guard={
            "$or": [{"priority": "high"}, {"project": "mobile", "status": "active"}]
        },
    )

    for result in results:
        data = result["data"]
        print(
            f"   → {data['user']}: {data['project']} project ({data['priority']} priority)"
        )

    # Example 3: Combined guards and negations
    print("\n3. Guards + Negations")
    print("   Query: Developers who are NOT inactive")

    results = client.search_json(
        probe={"role": "developer"},
        guard={"status": "active"},
        negations={"user": {"$not": "charlie"}},  # Exclude charlie specifically
    )

    for result in results:
        data = result["data"]
        print(f"   → {data['user']}: {data['role']} ({data['status']})")

    # Example 4: Multiple negations
    print("\n4. Multiple Negation Patterns")
    print("   Query: Exclude low priority AND inactive status")

    results = client.search_json(
        probe={},
        negations={"priority": {"$not": "low"}, "status": {"$not": "inactive"}},
    )

    for result in results:
        data = result["data"]
        print(
            f"   → {data['user']}: {data['role']} ({data['priority']} priority, {data['status']})"
        )

    # Example 5: Wildcards with guards
    print("\n5. Wildcards in Probes")
    print("   Query: Any priority level, but must be active developers")

    results = client.search_json(
        probe={"priority": {"$any": True}},  # Match any priority
        guard={"role": "developer", "status": "active"},
    )

    for result in results:
        data = result["data"]
        print(f"   → {data['user']}: {data['role']} ({data['priority']} priority)")

    print("\n✅ Advanced query examples completed!")


if __name__ == "__main__":
    main()
