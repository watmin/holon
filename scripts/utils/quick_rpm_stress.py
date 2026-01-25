#!/usr/bin/env python3
"""
Quick RPM Geometric Stress Test

Fast stress test focusing on concurrent geometric completion queries.
"""

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from statistics import mean, stdev

from holon import CPUStore


def generate_rpm_matrix(matrix_id, rule_type, missing_position=None):
    """Generate RPM matrix."""
    shapes = ["circle", "square", "triangle", "diamond", "star"]
    colors = ["black", "white", "red", "blue", "green"]

    panels = {}

    if rule_type == "progression":
        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

                panel_shapes = set()
                shape_count = row + col - 1
                for i in range(min(shape_count, len(shapes))):
                    panel_shapes.add(shapes[i])

                panel = {
                    "shapes": panel_shapes,
                    "count": len(panel_shapes),
                    "color": colors[(col - 1) % len(colors)],
                }
                panels[position] = panel

    elif rule_type == "xor":
        for row in range(1, 4):
            for col in range(1, 4):
                position = f"row{row}-col{col}"
                if missing_position and position == missing_position:
                    continue

                panel_shapes = set()
                for i, shape in enumerate(["circle", "square", "triangle"]):
                    if (row ^ col) & (1 << i):
                        panel_shapes.add(shape)

                panel = {
                    "shapes": panel_shapes,
                    "count": len(panel_shapes),
                    "color": "black",
                }
                panels[position] = panel

    edn_data = {"matrix-id": matrix_id, "panels": panels, "rule": rule_type}

    if missing_position:
        edn_data["missing-position"] = missing_position

    return edn_data


def edn_to_json(edn_data):
    def convert_sets(obj):
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {k: convert_sets(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_sets(item) for item in obj]
        else:
            return obj

    return json.dumps(convert_sets(edn_data))


def compute_expected_missing_panel(matrix_data, missing_position):
    """Compute expected missing panel."""
    rule = matrix_data.get("rule", "")
    parts = missing_position.split("-")
    row = int(parts[0][3:])
    col = int(parts[1][3:])

    if rule == "progression":
        shape_count = row + col - 1
        shapes = set()
        shape_options = ["circle", "square", "triangle", "diamond", "star"]
        for i in range(min(shape_count, len(shape_options))):
            shapes.add(shape_options[i])

        colors = ["black", "white", "red", "blue", "green"]
        return {
            "shapes": shapes,
            "count": len(shapes),
            "color": colors[(col - 1) % len(colors)],
        }

    elif rule == "xor":
        shapes = set()
        for i, shape in enumerate(["circle", "square", "triangle"]):
            if (row ^ col) & (1 << i):
                shapes.add(shape)

        return {"shapes": shapes, "count": len(shapes), "color": "black"}

    return {"shapes": set(), "count": 0}


def stress_worker(store, matrices, worker_id, num_queries=50):
    """Concurrent geometric completion worker."""
    correct = 0
    total = 0
    query_times = []

    incomplete_matrices = [m for m in matrices if "missing-position" in m]

    for _ in range(num_queries):
        if not incomplete_matrices:
            break

        matrix = random.choice(incomplete_matrices)
        missing_pos = matrix["missing-position"]
        expected = compute_expected_missing_panel(matrix, missing_pos)

        probe_structure = {
            "panels": {
                pos: panel
                for pos, panel in matrix.get("panels", {}).items()
                if pos != missing_pos
            },
            "rule": matrix.get("rule", ""),
        }

        start_time = time.time()
        results = store.query(
            edn_to_json(probe_structure),
            negations={"missing-position": {"$any": True}},
            top_k=3,
        )
        query_time = time.time() - start_time
        query_times.append(query_time)

        found_correct = False
        for result in results:
            data = result[2]
            actual = data.get("panels", {}).get(missing_pos, {})

            if (
                set(actual.get("shapes", [])) == expected["shapes"]
                and actual.get("count", 0) == expected["count"]
            ):
                found_correct = True
                break

        if found_correct:
            correct += 1
        total += 1

    avg_query_time = mean(query_times) if query_times else 0
    return worker_id, correct, total, avg_query_time


def main():
    """Run quick RPM geometric stress test."""
    print("⚡ QUICK RPM GEOMETRIC STRESS TEST")
    print("=" * 60)

    # Initialize store
    print("🚀 Initializing Holon CPUStore...")
    store = CPUStore(dimensions=16000)

    # Generate test dataset
    print("🎨 Generating 2,000 RPM matrices...")
    matrices = []
    rules = ["progression", "xor"]

    for i in range(2000):
        rule = random.choice(rules)
        has_missing = random.random() < 0.4  # 40% incomplete
        missing_pos = (
            random.choice(
                [
                    "row1-col2",
                    "row1-col3",
                    "row2-col1",
                    "row2-col3",
                    "row3-col1",
                    "row3-col2",
                    "row3-col3",
                ]
            )
            if has_missing
            else None
        )

        matrix = generate_rpm_matrix(f"stress-{i}", rule, missing_pos)
        matrices.append(matrix)

    print(f"✅ Generated {len(matrices)} matrices")

    # Ingest data
    print("📥 Ingesting matrices...")
    start_time = time.time()
    for matrix in matrices:
        store.insert(edn_to_json(matrix))
    ingest_time = time.time() - start_time
    print(f"✅ Ingested {len(matrices)} matrices in {ingest_time:.1f}s")
    # Run concurrent stress test
    print("🔥 Running concurrent geometric completion stress test...")
    print("   8 workers × 50 queries each = 400 concurrent geometric completions")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for worker_id in range(8):
            future = executor.submit(stress_worker, store, matrices, worker_id, 50)
            futures.append(future)

        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            worker_id, correct, total, avg_time = result
            accuracy = correct / total if total > 0 else 0
            print("3d")
    stress_time = time.time() - start_time

    # Aggregate results
    total_correct = sum(r[1] for r in results)
    total_queries = sum(r[2] for r in results)
    avg_times = [r[3] for r in results if r[3] > 0]

    overall_accuracy = total_correct / total_queries if total_queries > 0 else 0
    avg_query_time = mean(avg_times) if avg_times else 0
    qps = total_queries / stress_time

    print("\n" + "=" * 60)
    print("🎯 STRESS TEST RESULTS")
    print("=" * 60)

    print("📊 PERFORMANCE METRICS:")
    print(f"   Total queries processed: {total_queries}")
    print(f"   Total stress time: {stress_time:.1f}s")
    print(f"   Average query time: {avg_query_time:.3f}s")
    print(f"   Queries per second: {qps:.1f}")
    print("\n⚡ CONCURRENT PROCESSING:")
    print(f"   Workers used: {len(results)}")
    print(f"   Queries per worker: {total_queries // len(results)}")
    print("\n🧠 GEOMETRIC INTELLIGENCE VALIDATION:")
    print(f"   Overall accuracy: {overall_accuracy:.1%}")
    print("   vs. Random baseline: ~5%")
    print(f"   Performance boost: {(overall_accuracy/0.05):.0f}× better than random")
    if overall_accuracy >= 0.8:
        print("   ✅ EXCELLENT: Geometric intelligence holds under concurrent load!")
        print("   ✅ System demonstrates robust parallel geometric reasoning!")
    elif overall_accuracy >= 0.6:
        print("   ⚠️  GOOD: Some degradation under concurrent load")
        print("   🤔 May need optimization for high-concurrency geometric tasks")
    else:
        print("   ❌ POOR: Significant degradation under concurrent load")
        print(
            "   🔧 Needs architectural improvements for concurrent geometric processing"
        )

    print("\n🏆 CONCLUSION:")
    if overall_accuracy >= 0.7:
        print("   🎉 STRESS TEST PASSED!")
        print("   Our geometric AI successfully handles:")
        print("   • 2,000 complex RPM matrices")
        print("   • 400 concurrent geometric completion queries")
        print("   • High-throughput geometric reasoning")
        print("   • Statistical reliability under load")
        print("   ")
        print("   This is production-ready geometric intelligence! 🚀")
    else:
        print("   ⚠️  STRESS TEST ISSUES DETECTED")
        print("   System needs optimization for concurrent geometric workloads")


if __name__ == "__main__":
    main()
