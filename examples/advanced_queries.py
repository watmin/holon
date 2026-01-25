#!/usr/bin/env python3
"""
Advanced Queries Example: Guards, Negations, Wildcards, Disjunctions
"""

import json

from holon import CPUStore


def main():
    store = CPUStore()

    # Sample data
    data = [
        {"name": "Alice", "role": "developer", "status": "active"},
        {"name": "Bob", "role": "designer", "status": "inactive"},
        {"name": "Charlie", "role": "developer", "status": "active"},
        {"name": "Diana", "role": "manager", "status": "active"},
    ]

    for item in data:
        store.insert(json.dumps(item))

    print("=== Advanced Queries Examples ===\n")

    # Wildcards
    print("1. Wildcards ($any):")
    results = store.query('{"role": {"$any": true}}')
    print(f"   Query: Any role → {len(results)} results")
    for r in results[:2]:
        print(f"   - {r[2]['name']}: {r[2]['role']}")

    # Guards
    print("\n2. Guards (exact filter):")
    results = store.query('{"role": "developer"}', guard={"status": "active"})
    print(f"   Query: Developers with active status → {len(results)} results")
    for r in results:
        print(f"   - {r[2]['name']}")

    # Negations
    print("\n3. Negations (exclude):")
    results = store.query(
        '{"role": "developer"}', negations={"name": {"$not": "Alice"}}
    )
    print(f"   Query: Developers except Alice → {len(results)} results")
    for r in results:
        print(f"   - {r[2]['name']}")

    # Disjunctions
    print("\n4. Disjunctions ($or):")
    results = store.query('{"$or": [{"role": "developer"}, {"role": "manager"}]}')
    print(f"   Query: Developers OR Managers → {len(results)} results")
    for r in results:
        print(f"   - {r[2]['name']}: {r[2]['role']}")

    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    main()
