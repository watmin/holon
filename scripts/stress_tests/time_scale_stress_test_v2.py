#!/usr/bin/env python3
"""
Time-Based Scale Stress Test v2

Key changes from v1:
- Reduced dimensions (1024 instead of 16000) for memory efficiency
- Simplified data payload
- Skip FAISS for maximum insert count test

Target: Find the practical limit of in-memory storage with time queries.

Usage:
    ./scripts/run_with_venv.sh python scripts/stress_tests/time_scale_stress_test_v2.py
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


@dataclass
class StressTestConfig:
    """Configuration for stress test."""
    # Target memory usage (GB)
    target_memory_gb: float = 30.0

    # Dimensions (smaller = more records)
    dimensions: int = 1024

    # Data generation
    num_services: int = 50
    num_regions: int = 10
    time_span_days: int = 365

    # Batch size for progress reporting
    batch_size: int = 50000

    # Query test
    num_query_iterations: int = 100

    # Use FAISS?
    use_faiss: bool = False  # Disable for max capacity test


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
    """Generate a realistic timestamped event - minimal payload."""
    offset_seconds = rng.randint(0, config.time_span_days * 86400)
    event_time = base_time + timedelta(seconds=offset_seconds)

    return {
        "id": event_id,
        "svc": rng.randint(0, config.num_services - 1),
        "rgn": rng.randint(0, config.num_regions - 1),
        "err": rng.random() < 0.1,
        "ts": {"$time": event_time.timestamp()},
    }


def run_insert_test(config: StressTestConfig) -> dict:
    """Run insert test until memory limit reached."""
    print("\n" + "=" * 70)
    print("PHASE 1: Maximum Insert Capacity")
    print("=" * 70)

    print(f"\nConfiguration:")
    print(f"   Dimensions: {config.dimensions}")
    print(f"   Target memory: {config.target_memory_gb} GB")
    print(f"   FAISS enabled: {config.use_faiss}")

    # Initial memory
    proc_mem = get_memory_usage_gb()
    sys_used, sys_avail = get_system_memory_gb()
    print(f"\n📊 Initial memory:")
    print(f"   Process: {proc_mem:.2f} GB")
    print(f"   Available: {sys_avail:.2f} GB")

    # Initialize store
    print(f"\n🚀 Initializing store with {config.dimensions} dimensions...")
    store = CPUStore(dimensions=config.dimensions)
    store.ann_enabled = config.use_faiss
    client = HolonClient(local_store=store)

    rng = random.Random(42)
    base_time = datetime(2024, 1, 1)

    # Bulk insert mode
    store.start_bulk_insert()

    results = {
        "dimensions": config.dimensions,
        "use_faiss": config.use_faiss,
        "checkpoints": [],
    }

    insert_start = time.time()
    last_report = insert_start
    record_count = 0
    memory_limit_reached = False

    print(f"\n📥 Inserting records until {config.target_memory_gb}GB limit...")

    try:
        while not memory_limit_reached:
            # Insert batch
            for _ in range(config.batch_size):
                event = generate_event(record_count, config, base_time, rng)
                client.insert_json(event)
                record_count += 1

            # Check memory
            proc_mem = get_memory_usage_gb()
            _, sys_avail = get_system_memory_gb()
            elapsed = time.time() - insert_start
            rate = record_count / elapsed

            checkpoint = {
                "records": record_count,
                "memory_gb": proc_mem,
                "available_gb": sys_avail,
                "insert_rate": rate,
                "elapsed_s": elapsed,
            }
            results["checkpoints"].append(checkpoint)

            print(f"   {record_count:,} records - "
                  f"{rate:,.0f} rec/s - "
                  f"Proc: {proc_mem:.2f}GB, Avail: {sys_avail:.2f}GB")

            # Check limits
            if proc_mem >= config.target_memory_gb:
                print(f"\n✅ Reached target memory limit: {proc_mem:.2f}GB")
                memory_limit_reached = True
            elif sys_avail < 5:
                print(f"\n⚠️  Low system memory: {sys_avail:.2f}GB available")
                memory_limit_reached = True

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except MemoryError as e:
        print(f"\n❌ MemoryError: {e}")
        results["error"] = "memory_error"

    store.end_bulk_insert()

    # Final stats
    total_elapsed = time.time() - insert_start
    final_rate = record_count / total_elapsed

    results["final_count"] = record_count
    results["final_memory_gb"] = get_memory_usage_gb()
    results["total_time_s"] = total_elapsed
    results["avg_insert_rate"] = final_rate

    print(f"\n📊 Insert Complete:")
    print(f"   Total records: {record_count:,}")
    print(f"   Total time: {total_elapsed:.1f}s")
    print(f"   Avg rate: {final_rate:,.0f} rec/s")
    print(f"   Final memory: {results['final_memory_gb']:.2f}GB")
    print(f"   Bytes per record: {results['final_memory_gb'] * 1024**3 / record_count:,.0f}")

    return results, store, client, rng, base_time


def run_query_test(
    config: StressTestConfig,
    store: CPUStore,
    client: HolonClient,
    rng: random.Random,
    base_time: datetime,
    record_count: int,
) -> dict:
    """Run query performance test."""
    print("\n" + "=" * 70)
    print("PHASE 2: Query Performance")
    print("=" * 70)

    print(f"\n🔍 Running {config.num_query_iterations} queries against {record_count:,} records...")
    print(f"   (Note: Without FAISS, this is brute-force O(n) similarity)")

    query_times = []
    result_counts = []

    for i in range(config.num_query_iterations):
        query_time = base_time + timedelta(
            seconds=rng.randint(0, config.time_span_days * 86400)
        )

        probe = {
            "svc": rng.randint(0, config.num_services - 1),
            "rgn": rng.randint(0, config.num_regions - 1),
            "err": True,
            "ts": {"$time": query_time.timestamp()},
        }

        start = time.time()
        results = client.search_json(probe=probe, limit=10, threshold=0.0)
        elapsed = time.time() - start

        query_times.append(elapsed)
        result_counts.append(len(results))

        if (i + 1) % 20 == 0:
            print(f"   {i+1}/{config.num_query_iterations} - "
                  f"Avg: {np.mean(query_times[-20:])*1000:.1f}ms")

    results = {
        "record_count": record_count,
        "num_queries": config.num_query_iterations,
        "avg_ms": np.mean(query_times) * 1000,
        "p50_ms": np.percentile(query_times, 50) * 1000,
        "p95_ms": np.percentile(query_times, 95) * 1000,
        "p99_ms": np.percentile(query_times, 99) * 1000,
        "min_ms": np.min(query_times) * 1000,
        "max_ms": np.max(query_times) * 1000,
    }

    print(f"\n📊 Query Performance:")
    print(f"   Avg: {results['avg_ms']:.1f}ms")
    print(f"   P50: {results['p50_ms']:.1f}ms")
    print(f"   P95: {results['p95_ms']:.1f}ms")
    print(f"   P99: {results['p99_ms']:.1f}ms")
    print(f"   Min: {results['min_ms']:.1f}ms, Max: {results['max_ms']:.1f}ms")

    # Calculate ops/sec
    ops_per_sec = 1000 / results['avg_ms']
    print(f"   Throughput: {ops_per_sec:.1f} queries/sec")

    return results


def main():
    print("=" * 70)
    print("TIME-BASED SCALE STRESS TEST v2")
    print("Optimized for maximum record count")
    print("=" * 70)

    config = StressTestConfig()

    # Phase 1: Insert
    insert_results, store, client, rng, base_time = run_insert_test(config)

    if insert_results.get("final_count", 0) > 0:
        # Phase 2: Query
        gc.collect()
        query_results = run_query_test(
            config, store, client, rng, base_time,
            insert_results["final_count"]
        )

        # Combined results
        all_results = {
            "insert": insert_results,
            "query": query_results,
        }
    else:
        all_results = {"insert": insert_results}

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    record_count = insert_results.get("final_count", 0)
    memory_gb = insert_results.get("final_memory_gb", 0)
    bytes_per_record = memory_gb * 1024**3 / record_count if record_count > 0 else 0

    print(f"""
    Configuration:
    - Dimensions: {config.dimensions}
    - FAISS: {'Enabled' if config.use_faiss else 'Disabled'}

    Results:
    - Total records: {record_count:,}
    - Memory usage: {memory_gb:.2f} GB
    - Bytes per record: {bytes_per_record:,.0f}
    - Insert rate: {insert_results.get('avg_insert_rate', 0):,.0f} rec/s
    """)

    if "query" in all_results:
        q = all_results["query"]
        print(f"""    Query Performance:
    - Avg latency: {q['avg_ms']:.1f}ms
    - P95 latency: {q['p95_ms']:.1f}ms
    - Throughput: {1000/q['avg_ms']:.1f} queries/sec
    """)

    # Extrapolation for Qdrant
    print(f"""
    📊 Extrapolation for Qdrant (with {config.dimensions} dims):
    - Records per GB: ~{record_count / memory_gb:,.0f}
    - For 1TB storage: ~{record_count / memory_gb * 1000:,.0f} records
    - For 1 year @ 1000 events/sec: {365 * 24 * 3600 * 1000:,} records
    - Required storage: ~{365 * 24 * 3600 * 1000 * bytes_per_record / (1024**4):.1f} TB
    """)

    # Save results
    results_file = "scripts/stress_tests/time_scale_results_v2.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"📝 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
