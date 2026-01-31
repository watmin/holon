#!/usr/bin/env python3
"""
Time Encoding Quality Test

Tests if time encoding maintains accuracy/discriminative power at scale.

Key questions:
1. Can we find documents similar in structure AND time?
2. Does the time encoding still discriminate at 1M+ records?
3. Are false positives minimal?

Uses 1M records at 1024 dims (~3.5GB) for reasonable test size.
"""

import random
import sys
import time
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np

sys.stdout.reconfigure(line_buffering=True)

from holon import CPUStore, HolonClient


def create_test_data(
    client: HolonClient,
    num_records: int,
    num_services: int = 20,
    num_regions: int = 5,
    time_span_days: int = 365,
    rng: random.Random = None,
) -> Tuple[List[dict], datetime]:
    """Create test data and return inserted records with base time."""
    if rng is None:
        rng = random.Random(42)

    base_time = datetime(2024, 1, 1)
    inserted = []

    print(f"\n📥 Inserting {num_records:,} records...")
    start = time.time()

    for i in range(num_records):
        offset_seconds = rng.randint(0, time_span_days * 86400)
        event_time = base_time + timedelta(seconds=offset_seconds)

        record = {
            "id": i,
            "svc": f"svc-{rng.randint(0, num_services-1):02d}",
            "rgn": f"rgn-{rng.randint(0, num_regions-1):02d}",
            "err": rng.random() < 0.1,
            "ts": {"$time": event_time.timestamp()},
        }

        client.insert_json(record)
        inserted.append(record)

        if (i + 1) % 100000 == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            print(f"   {i+1:,} records - {rate:,.0f}/s")

    elapsed = time.time() - start
    print(f"   ✅ Done in {elapsed:.1f}s ({num_records/elapsed:,.0f}/s)")

    return inserted, base_time


def test_time_discrimination(
    client: HolonClient,
    inserted: List[dict],
    base_time: datetime,
    num_tests: int = 100,
    rng: random.Random = None,
) -> dict:
    """Test if time encoding discriminates between near and far times."""
    if rng is None:
        rng = random.Random(42)

    print(f"\n🔬 Test 1: Time Discrimination ({num_tests} tests)")
    print("   Query for specific time, check if results are temporally close...")

    time_errors = []
    near_scores = []
    far_scores = []

    for i in range(num_tests):
        # Pick a random target record
        target = rng.choice(inserted)
        target_ts = target["ts"]["$time"]
        target_dt = datetime.fromtimestamp(target_ts)

        # Query with this time
        probe = {
            "svc": target["svc"],
            "rgn": target["rgn"],
            "ts": {"$time": target_ts},
        }

        results = client.search_json(probe=probe, limit=20, threshold=0.0)

        if not results:
            continue

        # Analyze time distances of results
        for r in results:
            result_ts = r["data"]["ts"]["$time"]
            time_diff_hours = abs(target_ts - result_ts) / 3600

            if time_diff_hours < 24:  # Within 24 hours
                near_scores.append(r["score"])
            elif time_diff_hours > 24 * 30:  # More than 30 days away
                far_scores.append(r["score"])

            time_errors.append(time_diff_hours)

    # Calculate metrics
    avg_time_error = np.mean(time_errors) if time_errors else float("inf")
    median_time_error = np.median(time_errors) if time_errors else float("inf")
    avg_near_score = np.mean(near_scores) if near_scores else 0
    avg_far_score = np.mean(far_scores) if far_scores else 0

    results = {
        "avg_time_error_hours": avg_time_error,
        "median_time_error_hours": median_time_error,
        "avg_near_score": avg_near_score,
        "avg_far_score": avg_far_score,
        "score_gap": avg_near_score - avg_far_score,
    }

    print(f"   Avg time error: {avg_time_error:.1f} hours")
    print(f"   Median time error: {median_time_error:.1f} hours")
    print(f"   Avg score (near <24h): {avg_near_score:.4f}")
    print(f"   Avg score (far >30d): {avg_far_score:.4f}")
    print(f"   Score gap: {results['score_gap']:.4f} {'✅' if results['score_gap'] > 0 else '❌'}")

    return results


def test_structure_plus_time(
    client: HolonClient,
    inserted: List[dict],
    num_tests: int = 100,
    rng: random.Random = None,
) -> dict:
    """Test if structure + time queries work correctly."""
    if rng is None:
        rng = random.Random(42)

    print(f"\n🔬 Test 2: Structure + Time Accuracy ({num_tests} tests)")
    print("   Query for specific service+region+time, measure precision...")

    precision_scores = []
    exact_matches = 0
    total_results = 0

    for i in range(num_tests):
        # Pick a random target
        target = rng.choice(inserted)

        probe = {
            "svc": target["svc"],
            "rgn": target["rgn"],
            "err": target["err"],
            "ts": {"$time": target["ts"]["$time"]},
        }

        results = client.search_json(probe=probe, limit=10, threshold=0.0)

        if not results:
            continue

        # Count how many results match structure
        matches = 0
        for r in results:
            data = r["data"]
            if data["svc"] == target["svc"] and data["rgn"] == target["rgn"]:
                matches += 1
            if data["id"] == target["id"]:
                exact_matches += 1

        precision = matches / len(results)
        precision_scores.append(precision)
        total_results += len(results)

    avg_precision = np.mean(precision_scores) if precision_scores else 0
    exact_match_rate = exact_matches / num_tests

    results = {
        "avg_precision": avg_precision,
        "exact_match_rate": exact_match_rate,
        "total_results": total_results,
    }

    print(f"   Avg precision (struct match): {avg_precision:.1%}")
    print(f"   Exact match rate: {exact_match_rate:.1%}")

    return results


def test_time_window_ranking(
    client: HolonClient,
    inserted: List[dict],
    base_time: datetime,
    num_tests: int = 50,
    rng: random.Random = None,
) -> dict:
    """Test if closer times rank higher than farther times."""
    if rng is None:
        rng = random.Random(42)

    print(f"\n🔬 Test 3: Time Ranking Quality ({num_tests} tests)")
    print("   Verify that closer times rank higher than farther times...")

    correct_rankings = 0
    total_comparisons = 0

    for i in range(num_tests):
        target = rng.choice(inserted)
        target_ts = target["ts"]["$time"]

        probe = {
            "svc": target["svc"],
            "ts": {"$time": target_ts},
        }

        results = client.search_json(probe=probe, limit=20, threshold=0.0)

        if len(results) < 2:
            continue

        # Check if results are ordered by time proximity
        for j in range(len(results) - 1):
            r1, r2 = results[j], results[j + 1]
            time_diff_1 = abs(target_ts - r1["data"]["ts"]["$time"])
            time_diff_2 = abs(target_ts - r2["data"]["ts"]["$time"])

            # If same service, closer time should have higher score
            if r1["data"]["svc"] == r2["data"]["svc"] == target["svc"]:
                if r1["score"] >= r2["score"]:  # Higher score for closer time
                    if time_diff_1 <= time_diff_2:
                        correct_rankings += 1
                total_comparisons += 1

    ranking_accuracy = correct_rankings / total_comparisons if total_comparisons > 0 else 0

    results = {
        "ranking_accuracy": ranking_accuracy,
        "total_comparisons": total_comparisons,
    }

    print(f"   Ranking accuracy: {ranking_accuracy:.1%}")
    print(f"   Total comparisons: {total_comparisons}")

    return results


def test_false_positive_rate(
    client: HolonClient,
    inserted: List[dict],
    num_tests: int = 100,
    rng: random.Random = None,
) -> dict:
    """Test false positive rate with impossible queries."""
    if rng is None:
        rng = random.Random(42)

    print(f"\n🔬 Test 4: False Positive Analysis ({num_tests} tests)")
    print("   Query for non-existent combinations, check score distribution...")

    # Query for combinations that don't exist
    fake_scores = []

    for i in range(num_tests):
        # Create a fake service that doesn't exist
        fake_probe = {
            "svc": f"svc-99",  # Non-existent service
            "rgn": f"rgn-99",  # Non-existent region
            "ts": {"$time": datetime(2024, 6, 15, 12, 0).timestamp()},
        }

        results = client.search_json(probe=fake_probe, limit=10, threshold=0.0)

        for r in results:
            fake_scores.append(r["score"])

    # Compare with real queries
    real_scores = []
    for i in range(num_tests):
        target = rng.choice(inserted)
        probe = {
            "svc": target["svc"],
            "rgn": target["rgn"],
            "ts": {"$time": target["ts"]["$time"]},
        }
        results = client.search_json(probe=probe, limit=10, threshold=0.0)
        for r in results:
            real_scores.append(r["score"])

    avg_fake = np.mean(fake_scores) if fake_scores else 0
    avg_real = np.mean(real_scores) if real_scores else 0
    separation = avg_real - avg_fake

    results = {
        "avg_fake_score": avg_fake,
        "avg_real_score": avg_real,
        "score_separation": separation,
    }

    print(f"   Avg fake query score: {avg_fake:.4f}")
    print(f"   Avg real query score: {avg_real:.4f}")
    print(f"   Separation: {separation:.4f} {'✅' if separation > 0.1 else '⚠️'}")

    return results


def main():
    print("=" * 70)
    print("TIME ENCODING QUALITY TEST")
    print("Testing accuracy at scale")
    print("=" * 70)

    # Configuration
    NUM_RECORDS = 1_000_000  # 1M records at 1024 dims = ~3.5GB
    DIMENSIONS = 1024

    print(f"\nConfiguration:")
    print(f"   Records: {NUM_RECORDS:,}")
    print(f"   Dimensions: {DIMENSIONS}")

    # Initialize
    store = CPUStore(dimensions=DIMENSIONS)
    store.ann_enabled = False  # Disable ANN for consistent results
    client = HolonClient(local_store=store)
    rng = random.Random(42)

    # Create test data
    inserted, base_time = create_test_data(
        client, NUM_RECORDS, rng=rng
    )

    # Run quality tests
    results = {}

    results["time_discrimination"] = test_time_discrimination(
        client, inserted, base_time, num_tests=100, rng=rng
    )

    results["structure_plus_time"] = test_structure_plus_time(
        client, inserted, num_tests=100, rng=rng
    )

    results["time_ranking"] = test_time_window_ranking(
        client, inserted, base_time, num_tests=50, rng=rng
    )

    results["false_positives"] = test_false_positive_rate(
        client, inserted, num_tests=100, rng=rng
    )

    # Summary
    print("\n" + "=" * 70)
    print("QUALITY SUMMARY")
    print("=" * 70)

    time_disc = results["time_discrimination"]
    struct_time = results["structure_plus_time"]
    ranking = results["time_ranking"]
    fps = results["false_positives"]

    print(f"""
    Time Discrimination:
    - Score gap (near vs far): {time_disc['score_gap']:.4f} {'✅' if time_disc['score_gap'] > 0 else '❌'}
    - Median time error: {time_disc['median_time_error_hours']:.1f} hours

    Structure + Time Accuracy:
    - Precision: {struct_time['avg_precision']:.1%} {'✅' if struct_time['avg_precision'] > 0.5 else '❌'}
    - Exact match rate: {struct_time['exact_match_rate']:.1%}

    Time Ranking Quality:
    - Ranking accuracy: {ranking['ranking_accuracy']:.1%} {'✅' if ranking['ranking_accuracy'] > 0.5 else '❌'}

    False Positive Separation:
    - Real vs Fake gap: {fps['score_separation']:.4f} {'✅' if fps['score_separation'] > 0.1 else '⚠️'}

    Overall Assessment:
    """)

    # Calculate overall score
    passes = 0
    if time_disc["score_gap"] > 0:
        passes += 1
    if struct_time["avg_precision"] > 0.3:
        passes += 1
    if ranking["ranking_accuracy"] > 0.5:
        passes += 1
    if fps["score_separation"] > 0.05:
        passes += 1

    if passes == 4:
        print("    🎉 EXCELLENT - Time encoding works well at 1M scale!")
    elif passes >= 3:
        print("    ✅ GOOD - Time encoding mostly works at scale")
    elif passes >= 2:
        print("    ⚠️  FAIR - Time encoding has some issues at scale")
    else:
        print("    ❌ POOR - Time encoding degrades significantly at scale")


if __name__ == "__main__":
    main()
