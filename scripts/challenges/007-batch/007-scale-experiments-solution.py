#!/usr/bin/env python3
"""
Challenge 007-007: Scale & Noise Limit Experiments

Systematically tests Holon's limits:
1. Category Saturation - how many categories before overlap?
2. Similar Item Density - finding targets among near-duplicates
3. Binding Depth - how deep can nesting go?
4. Field Count Dilution - do many fields drown important ones?
5. Sequence Length Limits - how long can sequences be?

Usage:
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/007-scale-experiments-solution.py
    ./scripts/run_with_venv.sh python scripts/challenges/007-batch/007-scale-experiments-solution.py --http
"""

import argparse
import random
import time
import uuid
from typing import Any, Dict, List, Tuple

import numpy as np

from holon import CPUStore, HolonClient


class ScaleExperiments:
    """Run scale and limit experiments."""

    def __init__(self, use_http: bool = False, base_url: str = "http://localhost:8000"):
        if use_http:
            self.client = HolonClient(remote_url=base_url)
        else:
            self.store = CPUStore()
            self.client = HolonClient(local_store=self.store)

        self.results = {}

    def experiment_1_category_saturation(
        self, max_categories: int = 100, examples_per_category: int = 5
    ) -> Dict:
        """Test: At how many categories do prototypes start overlapping?"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 1: Category Saturation")
        print("=" * 70)
        print(
            f"\nTesting {max_categories} categories with {examples_per_category} examples each..."
        )

        start_time = time.time()

        # Generate categories
        categories = []
        prototypes = {}

        for cat_id in range(max_categories):
            # Create examples for this category
            examples = []
            for ex_id in range(examples_per_category):
                example = {
                    "category_id": cat_id,
                    "features": {
                        f"feature_{i}": random.random()
                        for i in range(10)  # 10 features per example
                    },
                    "label": f"category_{cat_id}",
                }
                examples.append(example)

            # Learn prototype
            vectors = []
            for example in examples:
                vec = self.client.encode_vectors(example)
                if isinstance(vec, list):
                    vec = np.array(vec)
                vectors.append(vec)

            prototype = np.mean(vectors, axis=0)
            prototypes[cat_id] = prototype
            categories.append({"id": cat_id, "examples": examples})

        # Test classification accuracy
        print("   Testing classification accuracy...")
        correct = 0
        total = 0

        # Test 5 random examples from each category
        for category in categories[:20]:  # Test first 20 categories
            cat_id = category["id"]
            test_example = {
                "category_id": cat_id,
                "features": {
                    f"feature_{i}": random.random() for i in range(10)
                },
                "label": f"category_{cat_id}",
            }

            test_vec = self.client.encode_vectors(test_example)
            if isinstance(test_vec, list):
                test_vec = np.array(test_vec)

            # Find closest prototype
            best_match = -1
            best_similarity = -1

            for proto_id, proto_vec in prototypes.items():
                similarity = float(
                    np.dot(test_vec, proto_vec)
                    / (np.linalg.norm(test_vec) * np.linalg.norm(proto_vec) + 1e-10)
                )
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = proto_id

            if best_match == cat_id:
                correct += 1
            total += 1

        accuracy = correct / total if total > 0 else 0
        elapsed = time.time() - start_time

        result = {
            "categories": max_categories,
            "examples_per_category": examples_per_category,
            "test_samples": total,
            "correct": correct,
            "accuracy": accuracy,
            "elapsed": elapsed,
        }

        print(f"   ✅ Accuracy: {correct}/{total} = {accuracy:.1%}")
        print(f"   ⏱️  Elapsed: {elapsed:.2f}s")

        self.results["category_saturation"] = result
        return result

    def experiment_2_similar_item_density(
        self, num_similar: int = 1000
    ) -> Dict:
        """Test: Can we find a specific target among near-duplicates?"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 2: Similar Item Density")
        print("=" * 70)
        print(f"\nTesting with {num_similar} items 95% similar to target...")

        start_time = time.time()

        # Create target
        target = {
            "id": "TARGET",
            "unique_marker": "FIND_ME",
            "data": list(range(50)),
        }
        target_id = self.client.insert_json(target)

        # Create similar items
        for i in range(num_similar):
            similar = {
                "id": f"similar_{i}",
                "unique_marker": "decoy",
                "data": list(range(50)),  # Same data
            }
            self.client.insert_json(similar)

        # Search for target
        results = self.client.search_json(
            probe={"unique_marker": "FIND_ME"}, limit=10
        )

        # Check if target is in top results
        found_rank = None
        for rank, result in enumerate(results, 1):
            if result["data"].get("id") == "TARGET":
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
            item_id = self.client.insert_json(nested)

            # Query for deepest value
            query = {"deepest_value": "FIND_THIS"}
            results = self.client.search_json(probe=query, limit=5)

            # Get score
            score = results[0]["score"] if results else 0.0
            depth_scores.append((depth, score))

            print(f"   Depth {depth}: score = {score:.4f}")

        elapsed = time.time() - start_time

        # Find where degradation starts
        degradation_depth = None
        for i, (depth, score) in enumerate(depth_scores):
            if i > 0 and score < depth_scores[i - 1][1] * 0.7:  # 30% drop
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

        start_time = time.time()

        field_count_scores = []

        for field_count in [10, 20, 30, 50, 75, 100]:
            if field_count > max_fields:
                break

            # Create record with many fields
            record = {"important_field": "CRITICAL_VALUE"}

            # Add noise fields
            for i in range(field_count - 1):
                record[f"noise_field_{i}"] = f"noise_value_{random.randint(0, 1000)}"

            # Insert
            self.client.insert_json(record)

            # Query for important field
            results = self.client.search_json(
                probe={"important_field": "CRITICAL_VALUE"}, limit=5
            )

            score = results[0]["score"] if results else 0.0
            field_count_scores.append((field_count, score))

            print(f"   {field_count} fields: score = {score:.4f}")

        elapsed = time.time() - start_time

        # Calculate precision retention
        if field_count_scores:
            baseline_score = field_count_scores[0][1]
            retention = [(count, score / baseline_score if baseline_score > 0 else 0) for count, score in field_count_scores]
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
        """Test: How long can sequences be?"""
        print("\n" + "=" * 70)
        print("EXPERIMENT 5: Sequence Length Limits")
        print("=" * 70)
        print(f"\nTesting sequences up to {max_length} elements...")

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

            # Insert
            self.client.insert_json(record)

            # Query for subsequence containing marker
            query_seq = ["SPECIAL_MARKER", f"word_{marker_pos + 1}"]
            query = {
                "sequence": {"_encode_mode": "ngram", "sequence": query_seq}
            }

            results = self.client.search_json(probe=query, limit=5)

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
    parser.add_argument("--http", action="store_true")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument(
        "--experiments",
        nargs="+",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Which experiments to run (default: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SCALE & NOISE LIMIT EXPERIMENTS")
    print("=" * 70)

    mode = "HTTP" if args.http else "Local"
    print(f"\n🔧 Mode: {mode}")

    experiments_to_run = args.experiments or [1, 2, 3, 4, 5]

    start_time = time.time()

    # Create experiment runner
    exp = ScaleExperiments(use_http=args.http, base_url=args.url)

    # Run experiments
    if 1 in experiments_to_run:
        exp.experiment_1_category_saturation(max_categories=50, examples_per_category=3)

    if 2 in experiments_to_run:
        exp.experiment_2_similar_item_density(num_similar=500)

    if 3 in experiments_to_run:
        exp.experiment_3_binding_depth(max_depth=6)

    if 4 in experiments_to_run:
        exp.experiment_4_field_count_dilution(max_fields=100)

    if 5 in experiments_to_run:
        exp.experiment_5_sequence_length(max_length=1000)

    total_elapsed = time.time() - start_time

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY OF ALL EXPERIMENTS")
    print("=" * 70)

    for exp_name, result in exp.results.items():
        print(f"\n📊 {exp_name.replace('_', ' ').title()}:")
        for key, value in result.items():
            if key not in ["depth_scores", "field_count_scores", "sequence_scores", "precision_retention"]:
                print(f"   {key}: {value}")

    print(f"\n⏱️  Total time: {total_elapsed:.2f}s")

    print(
        """
    ✅ Scale experiments reveal:
       - Category limits (prototype overlap)
       - Noise tolerance (finding needles in haystacks)
       - Binding depth constraints
       - Field dilution effects
       - Sequence length handling

    These findings guide optimal Holon usage patterns!
    """
    )


if __name__ == "__main__":
    main()
