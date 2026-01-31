#!/usr/bin/env python3
"""
Challenge 007-007: Scale & Noise Limit Experiments

Systematically tests Holon's limits using HTTP-compatible operations:
1. Category Saturation - how many categories before k-NN fails?
2. Similar Item Density - finding targets among near-duplicates
3. Binding Depth - how deep can nesting go?
4. Field Count Dilution - do many fields drown important ones?
5. Sequence Length Limits - how long can sequences be?

All operations use search_json() - no local vector operations.
Works identically in local and HTTP modes.

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/007-scale-experiments-solution.py
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/007-scale-experiments-solution.py --http
"""

import argparse
import json
import random
import time
from typing import Any, Dict, List

from holon import CPUStore, HolonClient


def parse_data(data):
    """Parse data field - may be string in HTTP mode."""
    if isinstance(data, str):
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return data
    return data


class ScaleExperiments:
    """Run scale and limit experiments - HTTP compatible."""

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.results = {}

    def _reset_store(self):
        """Reset store for next experiment (local mode only)."""
        if hasattr(self, 'store'):
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

    def experiment_1_category_saturation(
        self, num_categories: int = 50, examples_per_category: int = 5
    ) -> Dict:
        """
        Test: At how many categories does k-NN classification fail?

        HTTP-Compatible: Uses search_json() to find nearest neighbors,
        then votes based on their labels.
        """
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: Category Saturation (k-NN Classification)")
        print("=" * 70)
        print(f"\nTesting {num_categories} categories with {examples_per_category} examples each...")

        self._reset_store()
        start_time = time.time()

        # Insert training examples for each category
        print("   Inserting training data...")
        for cat_id in range(num_categories):
            for ex_id in range(examples_per_category):
                example = {
                    "_category": cat_id,
                    "cat_name": f"category_{cat_id}",
                    "feature_a": cat_id * 10 + random.randint(0, 5),
                    "feature_b": f"type_{cat_id % 10}",
                    "feature_c": cat_id % 5,
                    "noise": random.random(),
                }
                self.client.insert_json(example)

        total_examples = num_categories * examples_per_category
        print(f"   Inserted {total_examples} training examples")

        # Test classification using k-NN
        print("   Testing k-NN classification...")
        k = 5
        correct = 0
        total = 0

        # Test 20 random categories
        test_categories = random.sample(range(num_categories), min(20, num_categories))

        for cat_id in test_categories:
            # Create a test example for this category
            test_example = {
                "feature_a": cat_id * 10 + random.randint(0, 5),
                "feature_b": f"type_{cat_id % 10}",
                "feature_c": cat_id % 5,
                "noise": random.random(),
            }

            # Find k nearest neighbors via Holon search
            results = self.client.search_json(probe=test_example, limit=k, threshold=0.0)

            # Vote based on neighbor labels
            votes = {}
            for r in results:
                data = parse_data(r["data"])
                label = data.get("_category") if isinstance(data, dict) else None
                if label is not None:
                    votes[label] = votes.get(label, 0) + 1

            # Predict most common label
            if votes:
                predicted = max(votes, key=votes.get)
                if predicted == cat_id:
                    correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        elapsed = time.time() - start_time

        result = {
            "categories": num_categories,
            "examples_per_category": examples_per_category,
            "k_neighbors": k,
            "test_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "elapsed": elapsed,
        }

        print(f"   ✅ Accuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"   ⏱️  Elapsed: {elapsed:.2f}s")

        self.results["category_saturation"] = result
        return result

    def experiment_2_similar_item_density(self, num_similar: int = 500) -> Dict:
        """Test: Can we find a specific target among near-duplicates?"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Similar Item Density")
        print("=" * 70)
        print(f"\nTesting with {num_similar} items 95% similar to target...")

        self._reset_store()
        start_time = time.time()

        # Create target with unique marker
        target = {
            "id": "TARGET",
            "unique_marker": "FIND_ME",
            "common_data": list(range(50)),
            "description": "This is the target item we want to find",
        }
        self.client.insert_json(target)

        # Create similar decoy items
        for i in range(num_similar):
            decoy = {
                "id": f"decoy_{i}",
                "unique_marker": "decoy",
                "common_data": list(range(50)),  # Same as target
                "description": "This is a decoy item very similar to target",
            }
            self.client.insert_json(decoy)

        # Search for target using its unique marker
        results = self.client.search_json(
            probe={"unique_marker": "FIND_ME"}, limit=10, threshold=0.0
        )

        # Check if target is in results
        found_rank = None
        for rank, result in enumerate(results, 1):
            data = parse_data(result["data"])
            if isinstance(data, dict) and data.get("id") == "TARGET":
                found_rank = rank
                break

        elapsed = time.time() - start_time

        result = {
            "num_similar_items": num_similar,
            "found_rank": found_rank,
            "found_in_top_10": found_rank is not None and found_rank <= 10,
            "elapsed": elapsed,
        }

        status = "✅" if found_rank and found_rank <= 10 else "❌"
        print(f"   {status} Target found at rank: {found_rank if found_rank else 'NOT FOUND'}")
        print(f"   ⏱️  Elapsed: {elapsed:.2f}s")

        self.results["similar_item_density"] = result
        return result

    def experiment_3_binding_depth(self, max_depth: int = 6) -> Dict:
        """Test: How deep can nesting go before signal is lost?"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 3: Binding Depth")
        print("=" * 70)
        print(f"\nTesting nesting up to {max_depth} levels...")

        self._reset_store()
        start_time = time.time()

        depth_scores = []

        for depth in range(1, max_depth + 1):
            # Create nested structure
            nested = {"marker": f"depth_{depth}"}
            current = nested
            for i in range(depth - 1):
                current["child"] = {"level": i + 2}
                current = current["child"]
            current["deepest_value"] = "FIND_THIS"

            # Insert
            self.client.insert_json(nested)

            # Query for deepest value
            query = {"deepest_value": "FIND_THIS"}
            results = self.client.search_json(probe=query, limit=5, threshold=0.0)

            # Get score
            score = results[0]["score"] if results else 0.0
            depth_scores.append((depth, score))

            print(f"   Depth {depth}: score = {score:.4f}")

        elapsed = time.time() - start_time

        # Find where degradation starts (30% drop from previous)
        degradation_depth = None
        for i, (depth, score) in enumerate(depth_scores):
            if i > 0 and depth_scores[i - 1][1] > 0:
                if score < depth_scores[i - 1][1] * 0.7:
                    degradation_depth = depth
                    break

        result = {
            "max_depth_tested": max_depth,
            "depth_scores": depth_scores,
            "degradation_starts_at": degradation_depth,
            "elapsed": elapsed,
        }

        if degradation_depth:
            print(f"\n   ⚠️  Signal degradation starts at depth {degradation_depth}")
        else:
            print(f"\n   ✅ No significant degradation up to depth {max_depth}")
        print(f"   ⏱️  Elapsed: {elapsed:.2f}s")

        self.results["binding_depth"] = result
        return result

    def experiment_4_field_count_dilution(self, max_fields: int = 100) -> Dict:
        """Test: Do many fields drown important ones?"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 4: Field Count Dilution")
        print("=" * 70)
        print(f"\nTesting up to {max_fields} fields per record...")

        self._reset_store()
        start_time = time.time()

        field_count_scores = []
        test_counts = [10, 20, 30, 50, 75, 100]

        for field_count in test_counts:
            if field_count > max_fields:
                break

            # Create record with important field + noise
            record = {"important_field": "CRITICAL_VALUE"}
            for i in range(field_count - 1):
                record[f"noise_field_{i}"] = f"noise_value_{random.randint(0, 1000)}"

            self.client.insert_json(record)

            # Query for important field
            results = self.client.search_json(
                probe={"important_field": "CRITICAL_VALUE"}, limit=5, threshold=0.0
            )

            score = results[0]["score"] if results else 0.0
            field_count_scores.append((field_count, score))

            print(f"   {field_count} fields: score = {score:.4f}")

        elapsed = time.time() - start_time

        # Calculate precision retention relative to baseline
        if field_count_scores:
            baseline = field_count_scores[0][1]
            retention = [
                (count, score / baseline if baseline > 0 else 0)
                for count, score in field_count_scores
            ]
        else:
            retention = []

        result = {
            "max_fields_tested": max_fields,
            "field_count_scores": field_count_scores,
            "precision_retention": retention,
            "elapsed": elapsed,
        }

        print(f"\n   📊 Precision retention:")
        for count, ratio in retention:
            print(f"      {count} fields: {ratio:.1%}")
        print(f"   ⏱️  Elapsed: {elapsed:.2f}s")

        self.results["field_count_dilution"] = result
        return result

    def experiment_5_sequence_length(self, max_length: int = 1000) -> Dict:
        """Test: How long can sequences be before n-gram signal weakens?"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: Sequence Length Limits")
        print("=" * 70)
        print(f"\nTesting sequences up to {max_length} elements...")

        self._reset_store()
        start_time = time.time()

        sequence_scores = []
        test_lengths = [10, 50, 100, 200, 500, 1000]

        for length in test_lengths:
            if length > max_length:
                break

            # Create sequence with marker in the middle
            marker_pos = length // 2
            sequence = [f"word_{i}" for i in range(length)]
            sequence[marker_pos] = "SPECIAL_MARKER"

            record = {
                "sequence": {"_encode_mode": "ngram", "sequence": sequence}
            }
            self.client.insert_json(record)

            # Query for subsequence containing marker
            query_seq = ["SPECIAL_MARKER", f"word_{marker_pos + 1}"]
            query = {
                "sequence": {"_encode_mode": "ngram", "sequence": query_seq}
            }

            results = self.client.search_json(probe=query, limit=5, threshold=0.0)

            score = results[0]["score"] if results else 0.0
            sequence_scores.append((length, score))

            print(f"   Length {length}: score = {score:.4f}")

        elapsed = time.time() - start_time

        result = {
            "max_length_tested": max_length,
            "sequence_scores": sequence_scores,
            "elapsed": elapsed,
        }

        print(f"   ⏱️  Elapsed: {elapsed:.2f}s")

        self.results["sequence_length"] = result
        return result


def main():
    parser = argparse.ArgumentParser(description="Scale & Noise Limit Experiments")
    parser.add_argument("--http", action="store_true", help="Use HTTP API")
    parser.add_argument("--url", default="http://localhost:8000", help="Server URL")
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Which experiments to run (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SCALE & NOISE LIMIT EXPERIMENTS (HTTP-Compatible)")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    experiments_to_run = args.experiments or [1, 2, 3, 4, 5]

    start_time = time.time()

    exp = ScaleExperiments(use_http=args.http, base_url=args.url)

    if 1 in experiments_to_run:
        exp.experiment_1_category_saturation(num_categories=50, examples_per_category=5)

    if 2 in experiments_to_run:
        exp.experiment_2_similar_item_density(num_similar=500)

    if 3 in experiments_to_run:
        exp.experiment_3_binding_depth(max_depth=6)

    if 4 in experiments_to_run:
        exp.experiment_4_field_count_dilution(max_fields=100)

    if 5 in experiments_to_run:
        exp.experiment_5_sequence_length(max_length=1000)

    total_elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 70)

    for exp_name, result in exp.results.items():
        print(f"\n📊 {exp_name.replace('_', ' ').title()}:")
        for key, value in result.items():
            if key not in ["depth_scores", "field_count_scores", "sequence_scores", "precision_retention"]:
                print(f"   {key}: {value}")

    print(f"\n⏱️  Total time: {total_elapsed:.2f}s")

    print("""
    ✅ HTTP-Compatible Implementation:
       - All similarity via search_json() (no local numpy)
       - k-NN classification instead of local prototype math
       - Works identically in local and HTTP modes

    Scale experiments reveal Holon's practical limits!
    """)


if __name__ == "__main__":
    main()
