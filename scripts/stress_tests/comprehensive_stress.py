#!/usr/bin/env python3
"""
Comprehensive Stress Test Suite for Holon

Tests the limits of:
1. Scale - How many items before retrieval degrades
2. Similarity Collapse - When near-identical items become indistinguishable
3. Prototype Saturation - How many categories before prototypes overlap
4. N-gram Dilution - When long sequences lose signal
5. Query Complexity - When guards/negations break
6. Noise Tolerance - How robust is fuzzy matching
7. Dimensionality - Sweet spot for dimensions

Run: ./scripts/run_with_venv.sh python scripts/stress_tests/comprehensive_stress.py
"""

import sys
import os
import time
import json
import random
import string
import gc
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from holon.cpu_store import CPUStore
from holon.client import HolonClient
from holon.similarity import normalized_dot_similarity


@dataclass
class TestResult:
    test_name: str
    parameter: str
    value: float
    metric: str
    result: float
    notes: str = ""


class StressTestSuite:
    def __init__(self):
        self.results: List[TestResult] = []

    def add_result(self, test_name: str, parameter: str, value: float,
                   metric: str, result: float, notes: str = ""):
        self.results.append(TestResult(test_name, parameter, value, metric, result, notes))

    def print_summary(self):
        print("\n" + "="*80)
        print("STRESS TEST SUMMARY")
        print("="*80)

        current_test = None
        for r in self.results:
            if r.test_name != current_test:
                current_test = r.test_name
                print(f"\n--- {current_test} ---")
            print(f"  {r.parameter}={r.value}: {r.metric}={r.result:.4f} {r.notes}")


def generate_random_record(complexity: int = 3) -> dict:
    """Generate a random JSON record with given complexity."""
    record = {
        "id": ''.join(random.choices(string.ascii_lowercase, k=8)),
        "type": random.choice(["task", "event", "note", "record", "item"]),
        "priority": random.randint(1, 10),
        "tags": random.sample(["urgent", "low", "medium", "high", "critical",
                               "review", "done", "pending", "blocked"], k=random.randint(1, 4)),
    }
    if complexity >= 2:
        record["content"] = ' '.join(random.choices(
            ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
             "data", "system", "process", "memory", "vector", "search", "query"],
            k=random.randint(5, 15)
        ))
    if complexity >= 3:
        record["metadata"] = {
            "created": f"2024-{random.randint(1,12):02d}-{random.randint(1,28):02d}",
            "author": random.choice(["alice", "bob", "charlie", "diana"]),
            "version": random.randint(1, 5)
        }
    return record


# =============================================================================
# TEST 1: SCALE
# =============================================================================

def test_scale(suite: StressTestSuite):
    """Test retrieval quality and speed at different scales."""
    print("\n" + "="*80)
    print("TEST 1: SCALE - Retrieval at 100, 1K, 10K, 50K, 100K items")
    print("="*80)

    scales = [100, 1000, 10000, 50000, 100000]

    for n in scales:
        print(f"\n--- Scale: {n} items ---")
        gc.collect()

        store = CPUStore(dimensions=16000)
        client = HolonClient(local_store=store)

        # Insert items
        start = time.time()
        for i in range(n):
            record = generate_random_record()
            record["target_id"] = i  # Add unique identifier
            client.insert_json(record)
            if (i + 1) % 10000 == 0:
                print(f"  Inserted {i+1}/{n}...")

        insert_time = time.time() - start
        insert_rate = n / insert_time
        print(f"  Insert: {insert_time:.2f}s ({insert_rate:.0f} items/sec)")
        suite.add_result("Scale", "items", n, "insert_rate", insert_rate)

        # Insert a known target we'll search for
        target = {"type": "target", "name": "findme", "special": True}
        client.insert_json(target)

        # Search for target
        start = time.time()
        results = client.search_json(probe={"type": "target", "special": True}, limit=10)
        search_time = time.time() - start

        found = any("findme" in str(r.get("data", "")) for r in results)
        rank = -1
        for i, r in enumerate(results):
            if "findme" in str(r.get("data", "")):
                rank = i + 1
                break

        print(f"  Search: {search_time*1000:.2f}ms, Found: {found}, Rank: {rank}")
        suite.add_result("Scale", "items", n, "search_ms", search_time * 1000)
        suite.add_result("Scale", "items", n, "target_found", 1 if found else 0)
        suite.add_result("Scale", "items", n, "target_rank", rank if rank > 0 else 999)

        # Memory usage
        import psutil
        process = psutil.Process()
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f"  Memory: {mem_mb:.0f} MB")
        suite.add_result("Scale", "items", n, "memory_mb", mem_mb)

        del store, client
        gc.collect()


# =============================================================================
# TEST 2: SIMILARITY COLLAPSE
# =============================================================================

def test_similarity_collapse(suite: StressTestSuite):
    """Test when near-identical items become indistinguishable."""
    print("\n" + "="*80)
    print("TEST 2: SIMILARITY COLLAPSE - Near-identical items")
    print("="*80)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Base record
    base = {"type": "user", "name": "alice", "role": "developer", "team": "backend"}

    # Insert variations with increasing similarity
    variations = []
    for i in range(100):
        var = base.copy()
        var["id"] = f"user_{i}"
        # Only change one small thing
        if i % 10 == 0:
            var["variation_marker"] = f"unique_{i}"
        variations.append(var)
        client.insert_json(var)

    # Try to find the ones with variation_marker
    print("\n--- Finding marked variations among 100 near-identical records ---")
    results = client.search_json(probe={"variation_marker": "unique_50"}, limit=20)

    found_correct = False
    similarities = []
    for r in results:
        data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
        sim = r.get("similarity", 0)
        similarities.append(sim)
        if data.get("variation_marker") == "unique_50":
            found_correct = True
            print(f"  Found target at rank {results.index(r)+1} with similarity {sim:.4f}")

    if similarities:
        print(f"  Top-20 similarity range: {min(similarities):.4f} - {max(similarities):.4f}")
        spread = max(similarities) - min(similarities)
        suite.add_result("SimilarityCollapse", "near_identical", 100, "spread", spread)
        suite.add_result("SimilarityCollapse", "near_identical", 100, "found_target", 1 if found_correct else 0)

    # Now test with MORE near-identical items
    print("\n--- Testing with 1000 near-identical items ---")
    store2 = CPUStore(dimensions=16000)
    client2 = HolonClient(local_store=store2)

    for i in range(1000):
        var = base.copy()
        var["id"] = f"user_{i}"
        if i == 500:
            var["special_marker"] = "target"
        client2.insert_json(var)

    results = client2.search_json(probe={"special_marker": "target"}, limit=20)
    found = any("target" in str(r.get("data", "")) for r in results)
    rank = -1
    for i, r in enumerate(results):
        if "target" in str(r.get("data", "")):
            rank = i + 1
            break

    print(f"  Found target in 1000 near-identical: {found}, Rank: {rank}")
    suite.add_result("SimilarityCollapse", "near_identical", 1000, "found_target", 1 if found else 0)
    suite.add_result("SimilarityCollapse", "near_identical", 1000, "target_rank", rank if rank > 0 else 999)


# =============================================================================
# TEST 3: PROTOTYPE SATURATION
# =============================================================================

def test_prototype_saturation(suite: StressTestSuite):
    """Test how many categories prototypes can distinguish."""
    print("\n" + "="*80)
    print("TEST 3: PROTOTYPE SATURATION - Category limits")
    print("="*80)

    category_counts = [5, 10, 25, 50, 100, 200]

    for num_categories in category_counts:
        print(f"\n--- Testing {num_categories} categories ---")

        store = CPUStore(dimensions=16000)
        client = HolonClient(local_store=store)

        # Generate distinct category prototypes
        category_vectors = {}
        examples_per_category = 10

        for cat_id in range(num_categories):
            cat_name = f"category_{cat_id}"
            examples = []

            for ex in range(examples_per_category):
                # Each category has distinct structure
                record = {
                    "category": cat_name,
                    "cat_id": cat_id,
                    "feature_a": f"a_{cat_id}_{ex}",
                    "feature_b": f"b_{cat_id % 10}",  # Some shared features
                    "feature_c": cat_id * 100 + ex,
                }
                vec = np.array(client.encode_vectors(record, "json"))
                examples.append(vec)

            # Create prototype
            prototype = store.prototype(examples, threshold=0.3)
            category_vectors[cat_name] = prototype

        # Test classification accuracy
        correct = 0
        total = num_categories * 5  # 5 test examples per category

        for cat_id in range(num_categories):
            cat_name = f"category_{cat_id}"

            for test_ex in range(5):
                # Create test example
                test_record = {
                    "category": cat_name,
                    "cat_id": cat_id,
                    "feature_a": f"a_{cat_id}_test_{test_ex}",
                    "feature_b": f"b_{cat_id % 10}",
                    "feature_c": cat_id * 100 + 50 + test_ex,
                }
                test_vec = np.array(client.encode_vectors(test_record, "json"))

                # Find best matching prototype
                best_cat = None
                best_sim = -1
                for name, proto in category_vectors.items():
                    sim = normalized_dot_similarity(test_vec, proto)
                    if sim > best_sim:
                        best_sim = sim
                        best_cat = name

                if best_cat == cat_name:
                    correct += 1

        accuracy = correct / total
        print(f"  Classification accuracy: {accuracy:.1%} ({correct}/{total})")
        suite.add_result("PrototypeSaturation", "categories", num_categories, "accuracy", accuracy)

        # Check prototype overlap
        overlaps = []
        proto_list = list(category_vectors.values())
        for i in range(min(len(proto_list), 50)):  # Sample pairs
            for j in range(i+1, min(len(proto_list), 50)):
                sim = normalized_dot_similarity(proto_list[i], proto_list[j])
                overlaps.append(sim)

        if overlaps:
            avg_overlap = np.mean(overlaps)
            max_overlap = np.max(overlaps)
            print(f"  Prototype overlap: avg={avg_overlap:.4f}, max={max_overlap:.4f}")
            suite.add_result("PrototypeSaturation", "categories", num_categories, "avg_overlap", avg_overlap)
            suite.add_result("PrototypeSaturation", "categories", num_categories, "max_overlap", max_overlap)

        del store, client
        gc.collect()


# =============================================================================
# TEST 4: N-GRAM DILUTION
# =============================================================================

def test_ngram_dilution(suite: StressTestSuite):
    """Test when long sequences lose signal in n-gram encoding."""
    print("\n" + "="*80)
    print("TEST 4: N-GRAM DILUTION - Sequence length limits")
    print("="*80)

    word_pool = ["the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog",
                 "data", "system", "vector", "search", "query", "memory", "store",
                 "encode", "decode", "match", "find", "retrieve", "index", "hash"]

    sequence_lengths = [10, 50, 100, 500, 1000, 5000]

    for seq_len in sequence_lengths:
        print(f"\n--- Sequence length: {seq_len} words ---")

        store = CPUStore(dimensions=16000)
        client = HolonClient(local_store=store)

        # Create a long sequence with a known phrase embedded
        target_phrase = ["FINDME", "TARGET", "PHRASE"]
        insert_position = seq_len // 2

        # Generate random sequence
        sequence = random.choices(word_pool, k=seq_len)
        # Insert target phrase
        for i, word in enumerate(target_phrase):
            if insert_position + i < len(sequence):
                sequence[insert_position + i] = word

        # Insert the long sequence
        record = {
            "content": {
                "_encode_mode": "ngram",
                "sequence": sequence
            },
            "seq_len": seq_len
        }
        client.insert_json(record)

        # Also insert some shorter sequences as distractors
        for _ in range(20):
            distractor = {
                "content": {
                    "_encode_mode": "ngram",
                    "sequence": random.choices(word_pool, k=random.randint(10, 50))
                }
            }
            client.insert_json(distractor)

        # Search for the target phrase
        probe = {
            "content": {
                "_encode_mode": "ngram",
                "sequence": target_phrase
            }
        }
        results = client.search_json(probe=probe, limit=10)

        # Check if we found the long sequence
        found = False
        rank = -1
        target_sim = 0
        for i, r in enumerate(results):
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            if data.get("seq_len") == seq_len:
                found = True
                rank = i + 1
                target_sim = r.get("similarity", 0)
                break

        print(f"  Found target: {found}, Rank: {rank}, Similarity: {target_sim:.4f}")
        suite.add_result("NgramDilution", "seq_length", seq_len, "found", 1 if found else 0)
        suite.add_result("NgramDilution", "seq_length", seq_len, "rank", rank if rank > 0 else 999)
        suite.add_result("NgramDilution", "seq_length", seq_len, "similarity", target_sim)

        del store, client
        gc.collect()


# =============================================================================
# TEST 5: DIMENSIONALITY
# =============================================================================

def test_dimensionality(suite: StressTestSuite):
    """Test the impact of vector dimensions."""
    print("\n" + "="*80)
    print("TEST 5: DIMENSIONALITY - Optimal dimension count")
    print("="*80)

    dimensions = [1000, 2000, 4000, 8000, 16000, 32000]

    for dim in dimensions:
        print(f"\n--- Dimensions: {dim} ---")

        store = CPUStore(dimensions=dim)
        client = HolonClient(local_store=store)

        # Insert test data
        categories = ["alpha", "beta", "gamma", "delta", "epsilon"]
        category_records = {cat: [] for cat in categories}

        for cat in categories:
            for i in range(20):
                record = {
                    "category": cat,
                    "value": f"{cat}_{i}",
                    "features": [f"f{cat}{j}" for j in range(5)]
                }
                client.insert_json(record)
                vec = np.array(client.encode_vectors(record, "json"))
                category_records[cat].append(vec)

        # Test classification accuracy
        correct = 0
        total = len(categories) * 10

        for cat in categories:
            # Create prototype from first 10 examples
            prototype = store.prototype(category_records[cat][:10], threshold=0.3)

            # Test on remaining examples
            for test_vec in category_records[cat][10:]:
                best_cat = None
                best_sim = -1
                for other_cat in categories:
                    other_proto = store.prototype(category_records[other_cat][:10], threshold=0.3)
                    sim = normalized_dot_similarity(test_vec, other_proto)
                    if sim > best_sim:
                        best_sim = sim
                        best_cat = other_cat

                if best_cat == cat:
                    correct += 1

        accuracy = correct / total
        print(f"  Classification accuracy: {accuracy:.1%}")
        suite.add_result("Dimensionality", "dimensions", dim, "accuracy", accuracy)

        # Memory per vector
        vec = np.array(client.encode_vectors({"test": 1}, "json"))
        vec_bytes = vec.nbytes
        print(f"  Vector size: {vec_bytes} bytes ({vec_bytes/1024:.1f} KB)")
        suite.add_result("Dimensionality", "dimensions", dim, "vec_bytes", vec_bytes)

        del store, client
        gc.collect()


# =============================================================================
# TEST 6: QUERY COMPLEXITY
# =============================================================================

def test_query_complexity(suite: StressTestSuite):
    """Test performance with complex queries."""
    print("\n" + "="*80)
    print("TEST 6: QUERY COMPLEXITY - Guards, negations, $or")
    print("="*80)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Insert diverse data
    for i in range(1000):
        record = {
            "id": i,
            "type": random.choice(["task", "event", "note"]),
            "priority": random.randint(1, 5),
            "status": random.choice(["active", "pending", "done", "blocked"]),
            "author": random.choice(["alice", "bob", "charlie", "diana"]),
            "tags": random.sample(["urgent", "low", "medium", "high"], k=random.randint(1, 3)),
        }
        client.insert_json(record)

    # Insert specific target
    target = {
        "id": 9999,
        "type": "task",
        "priority": 5,
        "status": "active",
        "author": "alice",
        "tags": ["urgent", "high"],
        "special": "target"
    }
    client.insert_json(target)

    # Test queries of increasing complexity
    queries = [
        ("simple", {"type": "task"}, {}),
        ("two_fields", {"type": "task", "priority": 5}, {}),
        ("three_fields", {"type": "task", "priority": 5, "status": "active"}, {}),
        ("with_guard", {"type": "task"}, {"guard": {"author": "alice"}}),
        ("with_negation", {"type": "task"}, {"negations": {"status": {"$not": "done"}}}),
        ("complex", {"type": "task", "priority": 5, "author": "alice", "special": "target"}, {}),
    ]

    for name, probe, extra_args in queries:
        start = time.time()
        results = client.search_json(probe=probe, limit=20, **extra_args)
        query_time = time.time() - start

        found_target = any("target" in str(r.get("data", "")) for r in results)
        rank = -1
        for i, r in enumerate(results):
            if "target" in str(r.get("data", "")):
                rank = i + 1
                break

        print(f"  {name}: {query_time*1000:.2f}ms, found={found_target}, rank={rank}")
        suite.add_result("QueryComplexity", "query", hash(name) % 1000, "time_ms", query_time * 1000, name)
        suite.add_result("QueryComplexity", "query", hash(name) % 1000, "found", 1 if found_target else 0, name)


# =============================================================================
# TEST 7: NOISE TOLERANCE
# =============================================================================

def test_noise_tolerance(suite: StressTestSuite):
    """Test robustness to noisy/imperfect queries."""
    print("\n" + "="*80)
    print("TEST 7: NOISE TOLERANCE - Fuzzy matching robustness")
    print("="*80)

    store = CPUStore(dimensions=16000)
    client = HolonClient(local_store=store)

    # Insert clean records
    target = {
        "name": "Alice Johnson",
        "email": "alice.johnson@example.com",
        "department": "Engineering",
        "role": "Senior Developer",
        "skills": ["python", "machine learning", "databases"]
    }
    client.insert_json(target)

    # Add other records
    for i in range(100):
        record = {
            "name": f"Person {i}",
            "email": f"person{i}@example.com",
            "department": random.choice(["Sales", "Marketing", "HR", "Finance"]),
            "role": random.choice(["Manager", "Associate", "Director"]),
            "skills": random.sample(["excel", "powerpoint", "sales", "marketing"], k=2)
        }
        client.insert_json(record)

    # Test with increasingly noisy probes
    noise_levels = [
        ("exact", {"name": "Alice Johnson", "department": "Engineering"}),
        ("typo_name", {"name": "Alce Johnson", "department": "Engineering"}),
        ("typo_dept", {"name": "Alice Johnson", "department": "Enginering"}),
        ("partial", {"name": "Alice", "skills": ["python"]}),
        ("wrong_case", {"name": "alice johnson", "department": "engineering"}),
        ("extra_fields", {"name": "Alice Johnson", "department": "Engineering", "bogus": "field"}),
        ("minimal", {"skills": ["python", "machine learning"]}),
    ]

    for name, probe in noise_levels:
        results = client.search_json(probe=probe, limit=10)

        found = False
        rank = -1
        sim = 0
        for i, r in enumerate(results):
            data = json.loads(r["data"]) if isinstance(r["data"], str) else r["data"]
            if data.get("name") == "Alice Johnson":
                found = True
                rank = i + 1
                sim = r.get("similarity", 0)
                break

        print(f"  {name}: found={found}, rank={rank}, similarity={sim:.4f}")
        suite.add_result("NoiseTolerance", "noise_type", hash(name) % 1000, "found", 1 if found else 0, name)
        suite.add_result("NoiseTolerance", "noise_type", hash(name) % 1000, "rank", rank if rank > 0 else 999, name)
        suite.add_result("NoiseTolerance", "noise_type", hash(name) % 1000, "similarity", sim, name)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*80)
    print("HOLON COMPREHENSIVE STRESS TEST")
    print("="*80)
    print("Testing limits of fuzzy retrieval, prototypes, n-grams, and scale")
    print()

    suite = StressTestSuite()

    try:
        test_scale(suite)
        test_similarity_collapse(suite)
        test_prototype_saturation(suite)
        test_ngram_dilution(suite)
        test_dimensionality(suite)
        test_query_complexity(suite)
        test_noise_tolerance(suite)
    except KeyboardInterrupt:
        print("\n\nInterrupted! Showing partial results...")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()

    suite.print_summary()

    # Export results
    results_file = os.path.join(os.path.dirname(__file__), "stress_results.json")
    with open(results_file, "w") as f:
        json.dump([{
            "test": r.test_name,
            "param": r.parameter,
            "value": r.value,
            "metric": r.metric,
            "result": r.result,
            "notes": r.notes
        } for r in suite.results], f, indent=2)
    print(f"\nResults saved to: {results_file}")


if __name__ == "__main__":
    main()
