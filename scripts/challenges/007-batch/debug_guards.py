#!/usr/bin/env python3
"""
Debug guard filter issues in batch 007 solutions.

This script tests whether guard filters work correctly with nested data.
"""

import json
from holon import CPUStore, HolonClient


def test_simple_guard():
    """Test: Simple guard filter on flat data."""
    print("=" * 70)
    print("TEST 1: Simple Guard Filter (Flat Data)")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert test data
    data = [
        {"name": "alice", "score": 85},
        {"name": "bob", "score": 90},
        {"name": "charlie", "score": 75},
    ]

    for item in data:
        client.insert_json(item)

    # Query with guard
    print("\n🔍 Query: find items with score >= 80")
    results = client.search_json(
        probe={},
        guard={"score": {"$gte": 80}},
        limit=10
    )

    print(f"   Found {len(results)} items:")
    for r in results:
        print(f"   - {r['data']['name']}: {r['data']['score']}")

    expected = 2  # alice, bob
    status = "✅" if len(results) == expected else "❌"
    print(f"\n{status} Expected {expected}, got {len(results)}")
    return len(results) == expected


def test_nested_guard():
    """Test: Guard filter on nested data."""
    print("\n" + "=" * 70)
    print("TEST 2: Guard Filter on Nested Object")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert test data
    data = [
        {"name": "alice", "stats": {"score": 85, "level": 5}},
        {"name": "bob", "stats": {"score": 90, "level": 6}},
        {"name": "charlie", "stats": {"score": 75, "level": 4}},
    ]

    for item in data:
        client.insert_json(item)
        print(f"   Inserted: {item}")

    # Query with nested guard
    print("\n🔍 Query: find items with stats.score >= 80")
    results = client.search_json(
        probe={},
        guard={"stats": {"score": {"$gte": 80}}},
        limit=10
    )

    print(f"   Found {len(results)} items:")
    for r in results:
        print(f"   - {r['data']['name']}: {r['data']['stats']['score']}")

    expected = 2  # alice, bob
    status = "✅" if len(results) == expected else "❌"
    print(f"\n{status} Expected {expected}, got {len(results)}")
    return len(results) == expected


def test_array_guard():
    """Test: Guard filter on array elements."""
    print("\n" + "=" * 70)
    print("TEST 3: Guard Filter on Array Elements")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert test data (like medical records)
    data = [
        {
            "patient": "alice",
            "diagnoses": [
                {"condition": "flu", "severity": 5},
                {"condition": "cold", "severity": 3}
            ]
        },
        {
            "patient": "bob",
            "diagnoses": [
                {"condition": "pneumonia", "severity": 8}
            ]
        },
        {
            "patient": "charlie",
            "diagnoses": [
                {"condition": "cold", "severity": 2}
            ]
        },
    ]

    for item in data:
        client.insert_json(item)
        print(f"   Inserted: {item['patient']} with {item['diagnoses']}")

    # Try different guard approaches
    print("\n🔍 Approach 1: Guard on array (diagnoses.severity)")
    results = client.search_json(
        probe={},
        guard={"diagnoses": {"severity": {"$gte": 7}}},
        limit=10
    )
    print(f"   Found {len(results)} items")

    print("\n🔍 Approach 2: Guard on array with list syntax")
    results = client.search_json(
        probe={},
        guard={"diagnoses": [{"severity": {"$gte": 7}}]},
        limit=10
    )
    print(f"   Found {len(results)} items")

    print("\n🔍 Approach 3: Probe with guard")
    results = client.search_json(
        probe={"diagnoses": [{"condition": {"$any": True}}]},
        guard={"diagnoses": [{"severity": {"$gte": 7}}]},
        limit=10
    )
    print(f"   Found {len(results)} items")

    # Try manual filtering
    print("\n🔍 Approach 4: Manual filtering (get all, filter in Python)")
    all_results = client.search_json(probe={}, limit=10)
    filtered = []
    for r in all_results:
        patient_data = r['data']
        for diagnosis in patient_data.get('diagnoses', []):
            if diagnosis.get('severity', 0) >= 7:
                filtered.append(r)
                break

    print(f"   Manual filter found {len(filtered)} items:")
    for r in filtered:
        print(f"   - {r['data']['patient']}: max severity = {max(d['severity'] for d in r['data']['diagnoses'])}")

    expected = 1  # bob
    status = "✅" if len(filtered) == expected else "❌"
    print(f"\n{status} Manual filtering works: Expected {expected}, got {len(filtered)}")
    return len(filtered) == expected


def test_coverage_scenario():
    """Test: Exact scenario from code understanding challenge."""
    print("\n" + "=" * 70)
    print("TEST 4: Code Coverage Scenario (Real Use Case)")
    print("=" * 70)

    store = CPUStore()
    client = HolonClient(local_store=store)

    # Insert functions with coverage
    functions = [
        {"_type": "function", "name": "test_add", "coverage": 85},
        {"_type": "function", "name": "test_subtract", "coverage": 90},
        {"_type": "function", "name": "parse_data", "coverage": 60},
        {"_type": "function", "name": "format_output", "coverage": 65},
    ]

    for func in functions:
        client.insert_json(func)
        print(f"   Inserted: {func['name']} (coverage={func['coverage']}%)")

    # Query with guard
    print("\n🔍 Query: find functions with coverage >= 80")
    results = client.search_json(
        probe={"_type": "function"},
        guard={"coverage": {"$gte": 80}},
        limit=10
    )

    print(f"   Found {len(results)} functions:")
    for r in results:
        data = r['data']
        print(f"   - {data['name']}: {data['coverage']}%")

    expected = 2  # test_add, test_subtract
    status = "✅" if len(results) == expected else "❌"
    print(f"\n{status} Expected {expected}, got {len(results)}")

    if len(results) == 0:
        print("\n⚠️  DEBUGGING: Let's try without guard...")
        all_results = client.search_json(probe={"_type": "function"}, limit=10)
        print(f"   Without guard, found {len(all_results)} functions")
        for r in all_results:
            print(f"   - {r['data']['name']}: coverage={r['data']['coverage']}, score={r['score']:.4f}")

    return len(results) == expected


def main():
    print("\n🔬 DEBUGGING GUARD FILTERS\n")

    results = []

    results.append(("Simple Guard (Flat)", test_simple_guard()))
    results.append(("Nested Guard", test_nested_guard()))
    results.append(("Array Guard", test_array_guard()))
    results.append(("Coverage Scenario", test_coverage_scenario()))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for name, passed in results:
        status = "✅" if passed else "❌"
        print(f"{status} {name}")

    passed_count = sum(1 for _, passed in results if passed)
    total = len(results)
    print(f"\n{passed_count}/{total} tests passed")

    if passed_count < total:
        print("\n💡 CONCLUSION: Guard filters may have limitations with certain data structures.")
        print("   Consider using manual filtering as a workaround for complex cases.")


if __name__ == "__main__":
    main()
