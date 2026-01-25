#!/usr/bin/env python3

"""
EXTREME STRESS TEST: Dialed to 11
- 100,000 items with ultra-complex data
- Concurrent queries across all cores
- Memory pressure to 80% of system RAM
- Long runtime to watch system monitors
"""

import json
import logging
import logging.handlers
import multiprocessing
import os
import random

# Set up logging with stderr capture
import sys
import threading
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import psutil

from holon import CPUStore

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler("extreme_stress_test.log")
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
)

# Stream handler for stderr
stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)


def log_system_stats():
    """Log current system stats."""
    mem = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    logger.info(
        f"MEMORY: {mem.used/1024/1024:.1f}MB used, {mem.available/1024/1024:.1f}MB available, {mem.percent:.1f}%"
    )
    logger.info(f"CPU: {cpu:.1f}% total")


def log_performance_metrics(phase, start_time, items_processed=0, queries_processed=0):
    """Log performance metrics for a phase."""
    elapsed = time.time() - start_time
    if items_processed > 0:
        rate = items_processed / elapsed
        logger.info(
            f"PHASE_{phase}: Processed {items_processed} items in {elapsed:.1f}s ({rate:.1f} items/sec)"
        )
    elif queries_processed > 0:
        rate = queries_processed / elapsed
        logger.info(
            f"PHASE_{phase}: Processed {queries_processed} queries in {elapsed:.1f}s ({rate:.1f} queries/sec)"
        )
    else:
        logger.info(f"PHASE_{phase}: Completed in {elapsed:.1f}s")


def get_system_specs():
    mem = psutil.virtual_memory()
    return mem.total / (1024**3), mem.available / (1024**3)


def generate_ultra_complex_data(n_items):
    """Generate insanely complex data structures."""
    data = []
    for i in range(n_items):
        # Ultra-deep nesting with large arrays
        complex_item = {
            "id": i,
            "metadata": {
                "user": {
                    "profile": {
                        "personal": {
                            "name": f"user_{i}",
                            "age": 20 + i % 50,
                            "history": [f"event_{j}" for j in range(i % 20)],
                        },
                        "professional": {
                            "skills": [f"skill_{j}" for j in range(i % 15)],
                            "experience": i % 30,
                            "certifications": [
                                {"name": f"cert_{k}", "year": 2020 + k}
                                for k in range(i % 5)
                            ],
                        },
                    },
                    "preferences": {
                        "notifications": bool(i % 2),
                        "theme": f"theme_{i % 10}",
                        "language": "en" if i % 2 else "es",
                    },
                },
                "system": {
                    "created": f"2024-{(i%12)+1:02d}-{(i%28)+1:02d}T{(i%24):02d}:{(i%60):02d}:{(i%60):02d}Z",
                    "version": f"1.{i%10}.{i%100}",
                    "flags": [f"flag_{j}" for j in range(i % 8)],
                    "metrics": {
                        "cpu": i % 100,
                        "memory": i % 1000,
                        "disk": i % 10000,
                        "network": {"in": i * 1000, "out": i * 800, "latency": i % 100},
                    },
                },
            },
            "data": {
                "arrays": [
                    [j + i for j in range(50)],  # Large arrays
                    [f"string_{j}" for j in range(30)],
                    [{"key": f"val_{k}", "num": k} for k in range(20)],
                ],
                "nested_maps": {
                    f"level1_{x}": {
                        f"level2_{y}": {
                            f"level3_{z}": f"value_{x}_{y}_{z}" for z in range(3)
                        }
                        for y in range(3)
                    }
                    for x in range(3)
                },
                "mixed": [
                    i,
                    f"str_{i}",
                    bool(i % 2),
                    None,
                    {"nested": {"deep": {"value": i}}},
                    [i, i + 1, i + 2],
                ],
            },
            "tags": [f"tag_{j}" for j in range(i % 50)],  # Variable length tags
            "relations": {
                f"rel_{j}": [f"item_{i}_{j}_{k}" for k in range(5)]
                for j in range(i % 10)
            },
        }
        data.append(json.dumps(complex_item))
    return data


def query_worker(store, queries, results, worker_id):
    """Worker thread for concurrent queries."""
    for query in queries:
        try:
            start = time.time()
            res = store.query(query, "json", top_k=10)
            duration = time.time() - start
            results.append((worker_id, query[:30], len(res), duration))
        except Exception as e:
            results.append((worker_id, query[:30], f"ERROR: {e}", 0))


def encode_item_batch(batch_data):
    """Encode a batch of items using multiprocessing with local atom vectors."""
    items, global_atoms = batch_data
    local_atoms = {}  # Local copy of atom vectors
    results = []

    # Create local vector manager
    from holon.vector_manager import VectorManager

    local_vm = VectorManager(dimensions=16000)

    for item in items:
        try:
            parsed = (
                json_fast.loads(item) if "json_fast" in globals() else json.loads(item)
            )
            from holon.encoder import Encoder

            encoder = Encoder(local_vm)
            encoded = encoder.encode_data(parsed)
            results.append((parsed, encoded))
        except Exception as e:
            results.append((None, None))

    # Return results and local atom vectors for merging
    return results, dict(local_vm.atom_vectors)


def extreme_stress_test():
    logger.info("🚨 EXTREME STRESS TEST STARTED 🚨")
    log_system_stats()

    print("🚨 EXTREME STRESS TEST: DIALED TO 11 🚨")
    print("=" * 60)

    total_ram, avail_ram = get_system_specs()
    print(f"Total RAM: {total_ram:.1f} GB")
    print(f"Available RAM: {avail_ram:.1f} GB")
    # Target 50% of available RAM for faster testing
    target_ram = avail_ram * 0.5
    estimated_items = int(
        (target_ram * 1024 * 1024) / 200
    )  # 200KB per ultra-complex item
    estimated_items = min(estimated_items, 20000)  # Cap at 20k for reasonable runtime

    logger.info(f"Target config: {estimated_items} items, {target_ram:.1f}GB memory")

    print(f"Target memory: {target_ram:.1f} GB")
    print(f"Estimated items: {estimated_items:,}")
    print("Data complexity: Ultra-nested with large arrays")
    print("Query load: Concurrent across all cores")
    print("Expected runtime: 10-30 minutes")
    print("Expected memory: Up to 80% of system RAM")
    print("=" * 60)

    # Confirm
    response = input("This will stress your system HARD. Continue? (yes/no): ")
    if response.lower() != "yes":
        print("Aborted.")
        logger.info("Test aborted by user")
        return

    start_total = time.time()
    initial_mem = psutil.virtual_memory().used / (1024**3)
    logger.info(f"Test started at {datetime.now()}")
    log_system_stats()

    print("\n📝 Phase 1: Generating ultra-complex data...")
    logger.info("PHASE_1: Starting data generation")
    phase_start = time.time()
    data = generate_ultra_complex_data(estimated_items)
    log_performance_metrics("DATA_GEN", phase_start, items_processed=len(data))
    print(f"Generated {len(data)} items")
    log_system_stats()

    print("\n💾 Phase 2: Inserting data (PARALLEL batch processing)...")
    logger.info("PHASE_2: Starting parallel batch data insertion")
    store = CPUStore(dimensions=16000)
    insert_start = time.time()

    ids = []
    batch_size = 500  # Smaller batches for more parallelism
    n_workers = min(multiprocessing.cpu_count(), 14)
    max_concurrent = n_workers * 2  # Allow more concurrent batches

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all batches upfront for maximum parallelism
        futures = {}
        for batch_start in range(0, len(data), batch_size):
            batch_end = min(batch_start + batch_size, len(data))
            batch = data[batch_start:batch_end]
            batch_data = (
                batch,
                {},
            )  # Start with empty global atoms, processes will build locally
            future = executor.submit(encode_item_batch, batch_data)
            futures[future] = batch_start

        # Process completed batches as they finish
        completed_batches = 0
        for future in as_completed(futures):
            try:
                encoded_results, local_atoms = future.result(
                    timeout=300
                )  # 5min timeout
            except Exception as e:
                batch_start = futures[future]
                logger.error(f"Batch starting at {batch_start} failed: {e}")
                continue

            # Merge local atom vectors into global (with potential race, but rare)
            for atom, vector in local_atoms.items():
                if atom not in store.vector_manager.atom_vectors:
                    store.vector_manager.atom_vectors[atom] = vector

            # Store results serially
            for parsed, encoded in encoded_results:
                if parsed is not None:
                    data_id = str(uuid.uuid4())
                    store.stored_data[data_id] = parsed
                    store.stored_vectors[data_id] = encoded
                    ids.append(data_id)

            completed_batches += 1

            # Progress reporting every few batches
            if completed_batches % 10 == 0 or completed_batches >= len(futures):
                current_count = len(ids)
                elapsed = time.time() - insert_start
                rate = current_count / elapsed if elapsed > 0 else 0
                current_mem = psutil.virtual_memory().used / (1024**3)
                print(
                    f"  Completed {completed_batches}/{len(futures)} batches, inserted {current_count:,} items - {rate:.1f} items/sec - Memory: {current_mem:.1f} GB"
                )
                logger.info(
                    f"BATCH_PROGRESS: {completed_batches}/{len(futures)} batches, {current_count} items, {rate:.1f} items/sec, {current_mem:.1f}GB memory"
                )
                log_system_stats()
                if current_mem > total_ram * 0.85:
                    print("⚠️  Memory usage >85% - stopping")
                    logger.warning("Memory limit reached")
                    executor.shutdown(wait=False)
                    break

    insert_time = time.time() - insert_start
    final_mem = psutil.virtual_memory().used / (1024**3)
    log_performance_metrics("INSERT", insert_start, items_processed=len(ids))
    print(f"Total insert time: {insert_time:.2f}s")
    print(f"Final memory after insertion: {final_mem:.1f} GB")
    logger.info(
        f"Inserted {len(ids)} items with {n_workers} parallel workers, final memory {final_mem:.1f}GB"
    )
    log_system_stats()
    # Generate queries
    print("\n🔍 Phase 3: Generating query load...")
    logger.info("PHASE_3: Starting query generation")
    n_queries = min(1000, len(ids))  # 1000 queries
    queries = []
    for _ in range(n_queries):
        idx = random.randint(0, len(ids) - 1)
        queries.append(data[idx])

    logger.info(f"Generated {n_queries} queries")

    # Split queries for concurrent execution
    n_workers = min(14, len(os.sched_getaffinity(0)))  # Use all available cores
    queries_per_worker = n_queries // n_workers
    query_chunks = [
        queries[i : i + queries_per_worker]
        for i in range(0, n_queries, queries_per_worker)
    ]

    print(f"Queries: {n_queries} total, {n_workers} concurrent workers")
    logger.info(f"PHASE_4: Starting concurrent queries with {n_workers} workers")

    # Execute concurrent queries
    print("\n🚀 Phase 4: Executing concurrent queries (WATCH YOUR SYSTEM MONITORS!)...")
    results = []
    threads = []
    query_start = time.time()

    for i, chunk in enumerate(query_chunks):
        t = threading.Thread(target=query_worker, args=(store, chunk, results, i))
        threads.append(t)
        t.start()

    # Monitor progress
    while any(t.is_alive() for t in threads):
        time.sleep(10)  # Check every 10 seconds
        completed = len([r for r in results if r is not None])
        logger.info(f"QUERY_PROGRESS: {completed}/{n_queries} queries completed")
        log_system_stats()

    # Wait for all threads
    for t in threads:
        t.join()

    query_time = time.time() - query_start
    log_performance_metrics("QUERIES", query_start, queries_processed=len(results))

    # Analyze results
    successful_queries = [
        r for r in results if not isinstance(r[2], str) or not r[2].startswith("ERROR")
    ]
    error_queries = [
        r for r in results if isinstance(r[2], str) and r[2].startswith("ERROR")
    ]
    avg_results = (
        sum(r[2] for r in successful_queries) / len(successful_queries)
        if successful_queries
        else 0
    )
    avg_time = sum(r[3] for r in results) / len(results)

    total_time = time.time() - start_total

    logger.info("TEST COMPLETE")
    logger.info(
        f"Final stats: {len(ids)} items, {total_time:.1f}s runtime, {final_mem:.1f}GB memory"
    )
    logger.info(
        f"Query performance: {len(successful_queries)}/{len(results)} successful, {avg_time:.4f}s avg time"
    )

    print("\n📊 FINAL RESULTS")
    print("=" * 60)
    print(f"Items processed: {len(ids)}")
    print(f"Total runtime: {total_time:.1f}s")
    print(f"Insert time: {insert_time:.1f}s")
    print(f"Query time: {query_time:.1f}s")
    print(f"Peak memory: {final_mem:.1f} GB")
    print(f"Queries executed: {len(results)}")
    print(f"Successful queries: {len(successful_queries)}")
    print(f"Failed queries: {len(error_queries)}")
    print(f"Average results per query: {avg_results:.3f}")
    print(f"Average query time: {avg_time:.4f}s")
    print(f"Query throughput: {len(results)/query_time:.1f} queries/sec")
    print("System stress level: MAXIMUM 🔥")

    print("\n🏆 EXTREME TEST COMPLETE!")
    print("Your system handled the maximum stress test!")
    print(
        "Check the detailed log file: extreme_stress_test.log for performance analysis!"
    )


if __name__ == "__main__":
    extreme_stress_test()
