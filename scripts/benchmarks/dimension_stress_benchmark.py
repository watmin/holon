#!/usr/bin/env python3
"""
Harder benchmark to find where dimensions matter.

Tests scenarios that push VSA capacity limits:
1. High bundle counts (where interference appears)
2. Near-duplicate discrimination
3. Many simultaneous categories
4. Deep nesting with many fields

Usage:
    ./scripts/run_with_venv.sh python scripts/benchmarks/dimension_stress_benchmark.py
"""

import gc
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, ".")

from holon import CPUStore, HolonClient
from holon.similarity import normalized_dot_similarity

DIMENSIONS = [512, 1024, 2048, 4096, 8192, 16384]


def test_high_bundle_capacity(dim: int) -> dict:
    """
    Test: At what bundle size does accuracy degrade?
    VSA capacity theory: sqrt(d) items can be bundled cleanly.
    """
    results = {}
    bundle_sizes = [10, 50, 100, 200, 500, 1000]

    for size in bundle_sizes:
        store = CPUStore(dimensions=dim)
        client = HolonClient(local_store=store)

        # Insert "in-class" and "out-of-class" items
        for i in range(size):
            client.insert_json({"class": "target", "id": i})
        for i in range(size):
            client.insert_json({"class": "other", "id": i})

        # Query for target class
        probe = {"class": "target"}
        search_results = client.search_json(probe=probe, limit=size, threshold=0.0)

        correct = sum(1 for r in search_results if r["data"]["class"] == "target")
        results[size] = correct / size if size > 0 else 0

    return results


def test_many_categories(dim: int) -> dict:
    """
    Test: How many categories can be discriminated?
    """
    category_counts = [10, 25, 50, 100, 200]
    results = {}

    for num_cats in category_counts:
        store = CPUStore(dimensions=dim)
        client = HolonClient(local_store=store)

        # Insert items in each category
        items_per_cat = 20
        for cat_id in range(num_cats):
            for i in range(items_per_cat):
                client.insert_json({
                    "category": f"cat_{cat_id}",
                    "item": i,
                })

        # Test random categories
        correct = 0
        total = 0
        test_cats = list(range(0, num_cats, max(1, num_cats // 10)))  # Sample 10

        for cat_id in test_cats:
            probe = {"category": f"cat_{cat_id}"}
            search_results = client.search_json(probe=probe, limit=10, threshold=0.0)

            for r in search_results:
                total += 1
                if r["data"]["category"] == f"cat_{cat_id}":
                    correct += 1

        results[num_cats] = correct / total if total > 0 else 0

    return results


def test_near_duplicates(dim: int) -> dict:
    """
    Test: Can we distinguish items that differ by only one field?
    """
    store = CPUStore(dimensions=dim)
    client = HolonClient(local_store=store)

    # Insert items that differ minimally
    base = {"type": "order", "status": "pending", "priority": "high", "region": "us"}

    # Variations - only one field different
    variations = [
        {"type": "order", "status": "pending", "priority": "high", "region": "us"},  # exact
        {"type": "order", "status": "complete", "priority": "high", "region": "us"},  # status
        {"type": "order", "status": "pending", "priority": "low", "region": "us"},  # priority
        {"type": "order", "status": "pending", "priority": "high", "region": "eu"},  # region
        {"type": "invoice", "status": "pending", "priority": "high", "region": "us"},  # type
    ]

    for i, var in enumerate(variations):
        var["id"] = i
        client.insert_json(var)

    # Query for exact match
    results = client.search_json(probe=base, limit=5, threshold=0.0)

    # Check if exact match is first
    exact_first = results[0]["data"]["id"] == 0 if results else False

    # Check score gaps
    scores = [r["score"] for r in results]
    score_gap = scores[0] - scores[-1] if len(scores) > 1 else 0

    return {
        "exact_match_first": 1.0 if exact_first else 0.0,
        "score_gap": score_gap,
    }


def test_field_count_scaling(dim: int) -> dict:
    """
    Test: How does accuracy change with number of fields?
    """
    field_counts = [5, 10, 20, 50, 100]
    results = {}

    for num_fields in field_counts:
        store = CPUStore(dimensions=dim)
        client = HolonClient(local_store=store)

        # Create documents with many fields
        for doc_id in range(100):
            doc = {"id": doc_id}
            for f in range(num_fields):
                doc[f"field_{f}"] = f"value_{doc_id % 10}_{f}"
            client.insert_json(doc)

        # Query with partial match
        probe = {"field_0": "value_5_0", "field_1": "value_5_1"}
        search_results = client.search_json(probe=probe, limit=10, threshold=0.0)

        # Check if id=5, 15, 25, etc. are in results (they match the pattern)
        expected_ids = {5, 15, 25, 35, 45, 55, 65, 75, 85, 95}
        found_ids = {r["data"]["id"] for r in search_results}

        precision = len(found_ids & expected_ids) / len(found_ids) if found_ids else 0
        results[num_fields] = precision

    return results


def test_nesting_depth(dim: int) -> dict:
    """
    Test: How deep can nesting go before accuracy degrades?
    """
    depths = [2, 4, 6, 8, 10]
    results = {}

    for depth in depths:
        store = CPUStore(dimensions=dim)
        client = HolonClient(local_store=store)

        # Create nested documents
        def make_nested(d: int, val: str) -> dict:
            if d <= 0:
                return {"leaf": val}
            return {"level": d, "child": make_nested(d - 1, val)}

        for i in range(50):
            client.insert_json(make_nested(depth, f"val_{i % 5}"))

        # Query for specific leaf value
        probe = make_nested(depth, "val_2")
        search_results = client.search_json(probe=probe, limit=10, threshold=0.0)

        # Check if results have val_2 at the leaf
        def get_leaf(doc: dict) -> str:
            while "child" in doc:
                doc = doc["child"]
            return doc.get("leaf", "")

        correct = sum(1 for r in search_results if get_leaf(r["data"]) == "val_2")
        results[depth] = correct / len(search_results) if search_results else 0

    return results


def run_benchmark():
    """Run the stress benchmark."""
    print("=" * 70)
    print("DIMENSION STRESS BENCHMARK")
    print("Finding where dimensions actually matter")
    print("=" * 70)

    all_results = {}

    for dim in DIMENSIONS:
        print(f"\n{'=' * 70}")
        print(f"DIMENSION: {dim}")
        print("=" * 70)

        gc.collect()
        results = {"dimension": dim}

        # High bundle capacity
        print("Testing high bundle capacity...", flush=True)
        bundle = test_high_bundle_capacity(dim)
        results["bundle"] = bundle
        # Find degradation point
        for size, acc in sorted(bundle.items()):
            if acc < 0.95:
                print(f"  Degradation at bundle size {size}: {acc:.1%}")
                break
        else:
            print(f"  No degradation up to 1000 bundles")

        # Many categories
        print("Testing many categories...", flush=True)
        cats = test_many_categories(dim)
        results["categories"] = cats
        for count, acc in sorted(cats.items()):
            if acc < 0.90:
                print(f"  Degradation at {count} categories: {acc:.1%}")
                break
        else:
            print(f"  All category counts at 90%+")

        # Near duplicates
        print("Testing near-duplicate discrimination...", flush=True)
        dups = test_near_duplicates(dim)
        results["near_duplicates"] = dups
        print(f"  Exact match first: {dups['exact_match_first']:.0%}, "
              f"Score gap: {dups['score_gap']:.3f}")

        # Field count
        print("Testing field count scaling...", flush=True)
        fields = test_field_count_scaling(dim)
        results["field_count"] = fields
        for count, acc in sorted(fields.items()):
            print(f"  {count} fields: {acc:.1%}")

        # Nesting depth
        print("Testing nesting depth...", flush=True)
        nesting = test_nesting_depth(dim)
        results["nesting"] = nesting
        for depth, acc in sorted(nesting.items()):
            print(f"  Depth {depth}: {acc:.1%}")

        all_results[dim] = results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Where dimensions matter")
    print("=" * 70)

    print("\nBundle capacity (accuracy at size 1000):")
    for dim in DIMENSIONS:
        acc = all_results[dim]["bundle"].get(1000, 0)
        print(f"  {dim:>6}d: {acc:.1%}")

    print("\nCategory discrimination (200 categories):")
    for dim in DIMENSIONS:
        acc = all_results[dim]["categories"].get(200, 0)
        print(f"  {dim:>6}d: {acc:.1%}")

    print("\nNear-duplicate score gap:")
    for dim in DIMENSIONS:
        gap = all_results[dim]["near_duplicates"]["score_gap"]
        print(f"  {dim:>6}d: {gap:.3f}")

    print("\n100-field document precision:")
    for dim in DIMENSIONS:
        prec = all_results[dim]["field_count"].get(100, 0)
        print(f"  {dim:>6}d: {prec:.1%}")

    print("\nDeep nesting (depth 10):")
    for dim in DIMENSIONS:
        acc = all_results[dim]["nesting"].get(10, 0)
        print(f"  {dim:>6}d: {acc:.1%}")


if __name__ == "__main__":
    run_benchmark()
