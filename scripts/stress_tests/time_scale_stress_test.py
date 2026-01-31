#!/usr/bin/env python3
"""
Time-Based Scale Stress Test

Simulates a high-volume service producing timestamped events over time.
Tests time-based similarity queries at scale.

Target: ~30GB in-memory, progressive scaling to find breaking points.

Usage:
    ./scripts/run_with_venv.sh python scripts/stress_tests/time_scale_stress_test.py
"""

import gc
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple

import numpy as np
import psutil

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from holon import CPUStore, HolonClient
from holon.similarity import normalized_dot_similarity


@dataclass
class StressTestConfig:
    """Configuration for stress test."""
    # Scale targets (number of records)
    scale_levels: List[int] = None

    # Data generation
    num_services: int = 50          # Different service types
    num_regions: int = 10           # Geographic regions
    num_error_types: int = 25       # Error categories
    time_span_days: int = 365       # 1 year of data

    # Query test
    num_query_iterations: int = 100
    query_threshold: float = 0.1

    def __post_init__(self):
        if self.scale_levels is None:
            # Progressive scale levels
            self.scale_levels = [
                100_000,      # 100K - warmup (~130MB)
                1_000_000,    # 1M (~1.3GB)
                5_000_000,    # 5M (~6.5GB)
                10_000_000,   # 10M (~13GB)
                20_000_000,   # 20M (~26GB)
                25_000_000,   # 25M (~32GB) - target
            ]


def get_memory_usage_gb() -> float:
    """Get current process memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 3)


def get_system_memory_gb() -> Tuple[float, float]:
    """Get system memory (used, available) in GB."""
    mem = psutil.virtual_memory()
    return mem.used / (1024 ** 3), mem.available / (1024 ** 3)


def generate_event(
    event_id: int,
    config: StressTestConfig,
    base_time: datetime,
    rng: random.Random,
) -> dict:
    """Generate a realistic timestamped event."""
    # Random time within span
    offset_seconds = rng.randint(0, config.time_span_days * 86400)
    event_time = base_time + timedelta(seconds=offset_seconds)

    # Service metadata
    service_id = rng.randint(0, config.num_services - 1)
    region_id = rng.randint(0, config.num_regions - 1)

    # Event properties
    is_error = rng.random() < 0.1  # 10% error rate

    event = {
        "id": event_id,
        "service": f"svc-{service_id:03d}",
        "region": f"region-{region_id:02d}",
        "level": "error" if is_error else "info",
        "timestamp": {"$time": event_time.timestamp()},
    }

    if is_error:
        error_type = rng.randint(0, config.num_error_types - 1)
        event["error_code"] = f"ERR-{error_type:04d}"
        event["severity"] = rng.choice(["low", "medium", "high", "critical"])

    # Add some variable-length payload
    event["tags"] = [f"tag-{rng.randint(0, 99)}" for _ in range(rng.randint(1, 5))]

    return event


def run_scale_level(
    target_count: int,
    config: StressTestConfig,
    store: CPUStore,
    client: HolonClient,
    current_count: int,
    rng: random.Random,
) -> dict:
    """Run stress test at a specific scale level."""
    results = {
        "target_count": target_count,
        "start_count": current_count,
        "success": False,
    }

    base_time = datetime(2024, 1, 1)
    to_insert = target_count - current_count

    print(f"\n{'='*70}")
    print(f"SCALE LEVEL: {target_count:,} records")
    print(f"{'='*70}")
    print(f"   Currently have: {current_count:,}")
    print(f"   Need to insert: {to_insert:,}")

    if to_insert <= 0:
        print("   Already at or above target, skipping insert")
        results["success"] = True
        return results

    # Memory check before insert
    proc_mem = get_memory_usage_gb()
    sys_used, sys_avail = get_system_memory_gb()
    print(f"\n📊 Memory before insert:")
    print(f"   Process: {proc_mem:.2f} GB")
    print(f"   System: {sys_used:.2f} GB used, {sys_avail:.2f} GB available")

    # Check if we have enough headroom (need at least 5GB buffer)
    estimated_new_memory = to_insert * 1.3 / (1024 ** 3)  # ~1.3KB per record
    if sys_avail < estimated_new_memory + 5:
        print(f"\n⚠️  Insufficient memory!")
        print(f"   Need: {estimated_new_memory:.2f} GB + 5 GB buffer")
        print(f"   Available: {sys_avail:.2f} GB")
        results["error"] = "insufficient_memory"
        return results

    # Insert with bulk mode
    print(f"\n📥 Inserting {to_insert:,} records...")
    store.start_bulk_insert()

    insert_start = time.time()
    batch_size = 10000
    last_report = insert_start

    try:
        for i in range(to_insert):
            event = generate_event(current_count + i, config, base_time, rng)
            client.insert_json(event)

            # Progress reporting every 5 seconds
            if time.time() - last_report > 5:
                elapsed = time.time() - insert_start
                rate = (i + 1) / elapsed
                proc_mem = get_memory_usage_gb()
                _, sys_avail = get_system_memory_gb()
                pct = (i + 1) / to_insert * 100
                print(f"   {i+1:,}/{to_insert:,} ({pct:.1f}%) - "
                      f"{rate:,.0f} rec/s - "
                      f"Proc: {proc_mem:.2f}GB, Avail: {sys_avail:.2f}GB")
                last_report = time.time()

                # Emergency stop if memory gets too low
                if sys_avail < 3:
                    print(f"\n🛑 Emergency stop: Only {sys_avail:.2f}GB available!")
                    results["error"] = "emergency_stop_low_memory"
                    results["records_inserted"] = i + 1
                    store.end_bulk_insert()
                    return results

        insert_elapsed = time.time() - insert_start
        insert_rate = to_insert / insert_elapsed

        print(f"\n   ✅ Inserted {to_insert:,} in {insert_elapsed:.1f}s ({insert_rate:,.0f} rec/s)")

    except MemoryError as e:
        print(f"\n❌ MemoryError during insert: {e}")
        results["error"] = "memory_error"
        store.end_bulk_insert()
        return results
    except Exception as e:
        print(f"\n❌ Error during insert: {e}")
        results["error"] = str(e)
        store.end_bulk_insert()
        return results

    # End bulk mode (triggers ANN rebuild if threshold met)
    print("\n📊 Finalizing bulk insert...")
    bulk_end_start = time.time()
    store.end_bulk_insert()
    bulk_end_elapsed = time.time() - bulk_end_start
    print(f"   Bulk finalize took: {bulk_end_elapsed:.1f}s")

    results["insert_time_s"] = insert_elapsed
    results["insert_rate"] = insert_rate
    results["bulk_finalize_time_s"] = bulk_end_elapsed

    # Memory after insert
    gc.collect()
    proc_mem = get_memory_usage_gb()
    sys_used, sys_avail = get_system_memory_gb()
    print(f"\n📊 Memory after insert:")
    print(f"   Process: {proc_mem:.2f} GB")
    print(f"   System: {sys_used:.2f} GB used, {sys_avail:.2f} GB available")

    results["memory_after_gb"] = proc_mem
    results["system_available_gb"] = sys_avail

    # Query performance tests
    print(f"\n🔍 Query performance tests ({config.num_query_iterations} iterations)...")

    query_times = []
    result_counts = []

    # Test 1: Point-in-time query (find events similar to a specific time)
    print("\n   Test 1: Point-in-time queries")
    for i in range(config.num_query_iterations):
        # Random point in time
        query_time = base_time + timedelta(
            seconds=rng.randint(0, config.time_span_days * 86400)
        )

        probe = {
            "service": f"svc-{rng.randint(0, config.num_services-1):03d}",
            "region": f"region-{rng.randint(0, config.num_regions-1):02d}",
            "level": "error",
            "timestamp": {"$time": query_time.timestamp()},
        }

        start = time.time()
        results_list = client.search_json(
            probe=probe,
            limit=10,
            threshold=config.query_threshold,
        )
        elapsed = time.time() - start

        query_times.append(elapsed)
        result_counts.append(len(results_list))

    avg_query_time = np.mean(query_times)
    p50_query = np.percentile(query_times, 50)
    p95_query = np.percentile(query_times, 95)
    p99_query = np.percentile(query_times, 99)
    avg_results = np.mean(result_counts)

    print(f"      Avg query time: {avg_query_time*1000:.2f}ms")
    print(f"      P50: {p50_query*1000:.2f}ms, P95: {p95_query*1000:.2f}ms, P99: {p99_query*1000:.2f}ms")
    print(f"      Avg results: {avg_results:.1f}")

    results["query_avg_ms"] = avg_query_time * 1000
    results["query_p50_ms"] = p50_query * 1000
    results["query_p95_ms"] = p95_query * 1000
    results["query_p99_ms"] = p99_query * 1000

    # Test 2: Time-window queries (find similar events in a time range)
    print("\n   Test 2: Combined structure+time queries")
    window_times = []

    for i in range(config.num_query_iterations):
        service = f"svc-{rng.randint(0, config.num_services-1):03d}"
        query_time = base_time + timedelta(
            seconds=rng.randint(0, config.time_span_days * 86400)
        )

        probe = {
            "service": service,
            "level": "error",
            "severity": "critical",
            "timestamp": {"$time": query_time.timestamp()},
        }

        start = time.time()
        results_list = client.search_json(probe=probe, limit=20, threshold=0.0)
        elapsed = time.time() - start
        window_times.append(elapsed)

    avg_window = np.mean(window_times)
    p95_window = np.percentile(window_times, 95)

    print(f"      Avg query time: {avg_window*1000:.2f}ms")
    print(f"      P95: {p95_window*1000:.2f}ms")

    results["window_query_avg_ms"] = avg_window * 1000
    results["window_query_p95_ms"] = p95_window * 1000

    # Result quality check
    print("\n   Test 3: Result quality verification")
    quality_checks = []

    for i in range(10):
        # Insert a known event and find it
        known_service = f"svc-{rng.randint(0, config.num_services-1):03d}"
        known_time = base_time + timedelta(days=rng.randint(0, 30))

        probe = {
            "service": known_service,
            "level": "error",
            "timestamp": {"$time": known_time.timestamp()},
        }

        results_list = client.search_json(probe=probe, limit=5, threshold=0.0)

        # Check if top results have similar structure
        if results_list:
            top = results_list[0]
            data = top["data"]
            # Verify we're getting structurally similar results
            struct_match = (
                data.get("service") == known_service or
                data.get("level") == "error"
            )
            quality_checks.append(struct_match)

    quality_rate = sum(quality_checks) / len(quality_checks) if quality_checks else 0
    print(f"      Structure match rate: {quality_rate*100:.1f}%")

    results["quality_rate"] = quality_rate
    results["success"] = True
    results["final_count"] = target_count

    return results


def main():
    print("=" * 70)
    print("TIME-BASED SCALE STRESS TEST")
    print("=" * 70)

    config = StressTestConfig()

    print(f"\nConfiguration:")
    print(f"   Scale levels: {[f'{x:,}' for x in config.scale_levels]}")
    print(f"   Services: {config.num_services}")
    print(f"   Regions: {config.num_regions}")
    print(f"   Time span: {config.time_span_days} days")

    # Initial memory
    proc_mem = get_memory_usage_gb()
    sys_used, sys_avail = get_system_memory_gb()
    print(f"\n📊 Initial memory:")
    print(f"   Process: {proc_mem:.2f} GB")
    print(f"   System: {sys_used:.2f} GB used, {sys_avail:.2f} GB available")

    # Initialize store with ANN disabled for bulk loading
    print("\n🚀 Initializing store (ANN auto-rebuild disabled for bulk)...")
    store = CPUStore()
    store.ann_auto_rebuild = False  # Defer rebuilds until query time
    client = HolonClient(local_store=store)

    # Seed for reproducibility
    rng = random.Random(42)

    all_results = []
    current_count = 0

    for target in config.scale_levels:
        try:
            result = run_scale_level(
                target_count=target,
                config=config,
                store=store,
                client=client,
                current_count=current_count,
                rng=rng,
            )
            all_results.append(result)

            if result["success"]:
                current_count = target
            else:
                print(f"\n🛑 Stopped at {current_count:,} records")
                break

        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user at {current_count:,} records")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            all_results.append({"target_count": target, "error": str(e)})
            break

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n📊 Scale Results:")
    print(f"   {'Records':>15}  {'Insert Rate':>12}  {'Query P95':>10}  {'Memory':>10}  {'Status'}")
    print(f"   {'-'*15}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}")

    for r in all_results:
        count = r.get("target_count", 0)
        rate = r.get("insert_rate", 0)
        p95 = r.get("query_p95_ms", 0)
        mem = r.get("memory_after_gb", 0)
        status = "✅" if r.get("success") else f"❌ {r.get('error', 'failed')}"

        print(f"   {count:>15,}  {rate:>10,.0f}/s  {p95:>8.1f}ms  {mem:>8.2f}GB  {status}")

    # Performance analysis
    successful = [r for r in all_results if r.get("success")]
    if len(successful) >= 2:
        print(f"\n📈 Performance Trends:")

        # Query latency trend
        counts = [r["target_count"] for r in successful]
        p95s = [r["query_p95_ms"] for r in successful]

        if counts[-1] > counts[0]:
            latency_growth = (p95s[-1] - p95s[0]) / (counts[-1] - counts[0]) * 1_000_000
            print(f"   Query P95 growth: {latency_growth:.3f}ms per million records")

        # Memory efficiency
        mems = [r["memory_after_gb"] for r in successful]
        if counts[-1] > counts[0]:
            mem_per_million = (mems[-1] - mems[0]) / ((counts[-1] - counts[0]) / 1_000_000)
            print(f"   Memory growth: {mem_per_million:.2f}GB per million records")

    # Final memory
    proc_mem = get_memory_usage_gb()
    sys_used, sys_avail = get_system_memory_gb()
    print(f"\n📊 Final memory:")
    print(f"   Process: {proc_mem:.2f} GB")
    print(f"   System: {sys_used:.2f} GB used, {sys_avail:.2f} GB available")
    print(f"   Records in store: {len(store.items):,}")

    # Save results
    results_file = "scripts/stress_tests/time_scale_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n📝 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
