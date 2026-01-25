#!/usr/bin/env python3
"""
Advanced Queries Example: Complex Guards, Negations, and $or Logic
Demonstrates sophisticated query patterns with compound conditions.
"""

import json

from holon import CPUStore


def main():
    store = CPUStore()

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

    for item in data:
        store.insert(json.dumps(item))

    print("🔍 Advanced Query Examples")
    print("=" * 50)

    # Example 1: Complex $or guards
    print("\n1. Complex OR Conditions in Guards")
    print("   Query: High priority developers OR active managers")

    results = store.query(
        "{}",
        guard={
            "$or": [
                {"priority": "high", "role": "developer"},
                {"status": "active", "role": "manager"},
            ]
        },
    )

    for result in results:
        data = result[2]
        print(
            f"   → {data['user']}: {data['role']} ({data['priority']} priority, {data['status']})"
        )

    # Example 2: Nested OR logic
    print("\n2. Nested OR Logic")
    print("   Query: Web project with high priority OR active mobile developers")

    results = store.query(
        '{"project": "web"}',
        guard={
            "$or": [{"priority": "high"}, {"project": "mobile", "status": "active"}]
        },
    )

    for result in results:
        data = result[2]
        print(
            f"   → {data['user']}: {data['project']} project ({data['priority']} priority)"
        )

    # Example 3: Combined guards and negations
    print("\n3. Guards + Negations")
    print("   Query: Developers who are NOT inactive")

    results = store.query(
        '{"role": "developer"}',
        guard={"status": "active"},
        negations={"user": {"$not": "charlie"}},  # Exclude charlie specifically
    )

    for result in results:
        data = result[2]
        print(f"   → {data['user']}: {data['role']} ({data['status']})")

    # Example 4: Multiple negations
    print("\n4. Multiple Negation Patterns")
    print("   Query: Exclude low priority AND inactive status")

    results = store.query(
        "{}", negations={"priority": {"$not": "low"}, "status": {"$not": "inactive"}}
    )

    for result in results:
        data = result[2]
        print(
            f"   → {data['user']}: {data['role']} ({data['priority']} priority, {data['status']})"
        )

    # Example 5: Wildcards with guards
    print("\n5. Wildcards in Probes")
    print("   Query: Any priority level, but must be active developers")

    results = store.query(
        '{"priority": {"$any": true}}',  # Match any priority
        guard={"role": "developer", "status": "active"},
    )

    for result in results:
        data = result[2]
        print(f"   → {data['user']}: {data['role']} ({data['priority']} priority)")

    print("\n✅ Advanced query examples completed!")


if __name__ == "__main__":
    main()
