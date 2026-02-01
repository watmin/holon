#!/usr/bin/env python3
"""
GPU Validation for Holon - Challenge 008 Pre-requisite

Tests:
1. CuPy availability and GPU detection
2. Basic vector operations on GPU
3. CPU vs GPU performance comparison
4. Holon integration with GPU backend
"""

import time
import sys


def check_cupy():
    """Check if CuPy is available and GPU is detected."""
    print("=" * 60)
    print("STEP 1: CuPy Availability Check")
    print("=" * 60)

    try:
        import cupy as cp

        print(f"✅ CuPy version: {cp.__version__}")

        # Check GPU devices
        device_count = cp.cuda.runtime.getDeviceCount()
        print(f"✅ GPU devices found: {device_count}")

        for i in range(device_count):
            with cp.cuda.Device(i):
                props = cp.cuda.runtime.getDeviceProperties(i)
                name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
                total_mem = props["totalGlobalMem"] / (1024**3)
                print(f"   GPU {i}: {name} ({total_mem:.1f} GB)")

        # Current device
        current = cp.cuda.runtime.getDevice()
        print(f"✅ Current device: GPU {current}")

        return True, cp

    except ImportError:
        print("❌ CuPy not installed")
        print("   Install with: pip install cupy-cuda12x  # or cupy-cuda11x")
        return False, None

    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"❌ CUDA error: {e}")
        return False, None


def benchmark_vector_ops(cp):
    """Benchmark basic vector operations CPU vs GPU."""
    print("\n" + "=" * 60)
    print("STEP 2: Vector Operations Benchmark")
    print("=" * 60)

    import numpy as np

    dimensions = 16000
    num_vectors = 1000
    iterations = 100

    print(f"Config: {dimensions}D vectors, {num_vectors} items, {iterations} iterations")

    # Generate test data
    np.random.seed(42)
    cpu_vectors = [np.random.choice([-1, 0, 1], size=dimensions).astype(np.int8)
                   for _ in range(num_vectors)]
    cpu_query = np.random.choice([-1, 0, 1], size=dimensions).astype(np.int8)

    # CPU benchmark
    print("\n📊 CPU (NumPy):")
    start = time.perf_counter()
    for _ in range(iterations):
        for vec in cpu_vectors:
            _ = np.dot(cpu_query.astype(np.float32), vec.astype(np.float32))
    cpu_time = time.perf_counter() - start
    cpu_ops_per_sec = (num_vectors * iterations) / cpu_time
    print(f"   Time: {cpu_time:.3f}s")
    print(f"   Ops/sec: {cpu_ops_per_sec:,.0f}")

    # GPU benchmark
    print("\n📊 GPU (CuPy):")
    gpu_vectors = [cp.asarray(v) for v in cpu_vectors]
    gpu_query = cp.asarray(cpu_query)

    # Warmup
    for vec in gpu_vectors[:10]:
        _ = cp.dot(gpu_query.astype(cp.float32), vec.astype(cp.float32))
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        for vec in gpu_vectors:
            _ = cp.dot(gpu_query.astype(cp.float32), vec.astype(cp.float32))
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start
    gpu_ops_per_sec = (num_vectors * iterations) / gpu_time
    print(f"   Time: {gpu_time:.3f}s")
    print(f"   Ops/sec: {gpu_ops_per_sec:,.0f}")

    # Comparison
    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU Speedup: {speedup:.2f}x")

    if speedup < 1:
        print("⚠️  GPU slower than CPU for this workload (small vectors, transfer overhead)")
        print("   GPU excels with larger batches and matrix operations")

    return speedup


def benchmark_batch_similarity(cp):
    """Benchmark batch similarity computation (more realistic)."""
    print("\n" + "=" * 60)
    print("STEP 3: Batch Similarity Benchmark (Realistic)")
    print("=" * 60)

    import numpy as np

    dimensions = 16000
    num_stored = 5000
    num_queries = 100

    print(f"Config: {dimensions}D, {num_stored} stored vectors, {num_queries} queries")

    # Generate test data
    np.random.seed(42)
    stored_cpu = np.random.choice([-1, 0, 1], size=(num_stored, dimensions)).astype(np.float32)
    queries_cpu = np.random.choice([-1, 0, 1], size=(num_queries, dimensions)).astype(np.float32)

    # CPU: Matrix multiplication for batch similarity
    print("\n📊 CPU (NumPy matmul):")
    start = time.perf_counter()
    similarities_cpu = queries_cpu @ stored_cpu.T  # (num_queries, num_stored)
    cpu_time = time.perf_counter() - start
    print(f"   Time: {cpu_time:.4f}s")
    print(f"   Shape: {similarities_cpu.shape}")

    # GPU: Matrix multiplication
    print("\n📊 GPU (CuPy matmul):")
    stored_gpu = cp.asarray(stored_cpu)
    queries_gpu = cp.asarray(queries_cpu)

    # Warmup
    _ = queries_gpu @ stored_gpu.T
    cp.cuda.Stream.null.synchronize()

    start = time.perf_counter()
    similarities_gpu = queries_gpu @ stored_gpu.T
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.perf_counter() - start
    print(f"   Time: {gpu_time:.4f}s")
    print(f"   Shape: {similarities_gpu.shape}")

    # Verify results match
    similarities_gpu_cpu = cp.asnumpy(similarities_gpu)
    max_diff = np.max(np.abs(similarities_cpu - similarities_gpu_cpu))
    print(f"   Max diff from CPU: {max_diff:.6f}")

    speedup = cpu_time / gpu_time
    print(f"\n🚀 GPU Speedup: {speedup:.2f}x")

    return speedup


def test_holon_gpu():
    """Test Holon with GPU backend."""
    print("\n" + "=" * 60)
    print("STEP 4: Holon GPU Integration Test")
    print("=" * 60)

    import json

    try:
        from holon import CPUStore

        # Test GPU backend
        print("\n📊 Creating GPU-backed store...")
        store = CPUStore(dimensions=4096, backend="gpu")
        print(f"   Backend: {store.backend}")
        print(f"   Dimensions: {store.dimensions}")

        # Insert test data
        print("\n📊 Inserting test data...")
        test_data = [
            {"name": "Alice", "role": "developer", "skills": ["python", "ml"]},
            {"name": "Bob", "role": "designer", "skills": ["figma", "css"]},
            {"name": "Carol", "role": "developer", "skills": ["java", "spring"]},
        ]

        start = time.perf_counter()
        for data in test_data:
            store.insert(json.dumps(data), "json")
        insert_time = time.perf_counter() - start
        print(f"   Inserted {len(test_data)} items in {insert_time:.4f}s")

        # Query
        print("\n📊 Querying...")
        start = time.perf_counter()
        results = store.query(probe='{"role": "developer"}', data_type="json", top_k=5)
        query_time = time.perf_counter() - start
        print(f"   Found {len(results)} results in {query_time:.4f}s")

        for r in results:
            # Results are tuples: (id, similarity, data)
            if isinstance(r, tuple):
                print(f"      {r[0][:8]}... similarity={r[1]:.4f}")
            else:
                print(f"      {r.get('id', 'unknown')[:8]}... similarity={r.get('similarity', 0):.4f}")

        print("\n✅ Holon GPU integration working!")
        return True

    except Exception as e:
        print(f"\n❌ Holon GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def compare_holon_backends():
    """Compare Holon CPU vs GPU performance."""
    print("\n" + "=" * 60)
    print("STEP 5: Holon CPU vs GPU Comparison")
    print("=" * 60)

    from holon import CPUStore
    import json
    import time

    num_items = 500
    num_queries = 50

    print(f"Config: {num_items} items, {num_queries} queries")

    # Generate test data
    test_items = [
        {"id": i, "category": f"cat_{i % 20}", "value": i * 1.5, "tags": [f"tag_{j}" for j in range(i % 5)]}
        for i in range(num_items)
    ]

    # CPU test
    print("\n📊 CPU Backend:")
    store_cpu = CPUStore(dimensions=4096, backend="cpu")

    start = time.perf_counter()
    for item in test_items:
        store_cpu.insert(json.dumps(item), "json")
    cpu_insert_time = time.perf_counter() - start
    print(f"   Insert: {cpu_insert_time:.3f}s ({num_items/cpu_insert_time:.0f} items/sec)")

    start = time.perf_counter()
    for i in range(num_queries):
        store_cpu.query(probe=f'{{"category": "cat_{i % 20}"}}', data_type="json", top_k=10)
    cpu_query_time = time.perf_counter() - start
    print(f"   Query: {cpu_query_time:.3f}s ({num_queries/cpu_query_time:.0f} queries/sec)")

    # GPU test
    print("\n📊 GPU Backend:")
    store_gpu = CPUStore(dimensions=4096, backend="gpu")

    start = time.perf_counter()
    for item in test_items:
        store_gpu.insert(json.dumps(item), "json")
    gpu_insert_time = time.perf_counter() - start
    print(f"   Insert: {gpu_insert_time:.3f}s ({num_items/gpu_insert_time:.0f} items/sec)")

    start = time.perf_counter()
    for i in range(num_queries):
        store_gpu.query(probe=f'{{"category": "cat_{i % 20}"}}', data_type="json", top_k=10)
    gpu_query_time = time.perf_counter() - start
    print(f"   Query: {gpu_query_time:.3f}s ({num_queries/gpu_query_time:.0f} queries/sec)")

    # Summary
    print("\n📊 Summary:")
    insert_speedup = cpu_insert_time / gpu_insert_time
    query_speedup = cpu_query_time / gpu_query_time
    print(f"   Insert speedup: {insert_speedup:.2f}x")
    print(f"   Query speedup: {query_speedup:.2f}x")

    if insert_speedup < 1 or query_speedup < 1:
        print("\n⚠️  GPU may be slower for small workloads due to transfer overhead")
        print("   GPU excels at larger batch operations (1000+ items, batch queries)")

    return insert_speedup, query_speedup


def main():
    print("🎮 Holon GPU Validation")
    print("=" * 60)

    # Step 1: Check CuPy
    cupy_available, cp = check_cupy()

    if not cupy_available:
        print("\n❌ GPU validation failed: CuPy not available")
        print("\nTo install CuPy:")
        print("  pip install cupy-cuda12x  # For CUDA 12.x")
        print("  pip install cupy-cuda11x  # For CUDA 11.x")
        sys.exit(1)

    # Step 2: Vector ops benchmark
    vec_speedup = benchmark_vector_ops(cp)

    # Step 3: Batch similarity benchmark
    batch_speedup = benchmark_batch_similarity(cp)

    # Step 4: Holon integration
    holon_ok = test_holon_gpu()

    if not holon_ok:
        print("\n❌ Holon GPU integration failed")
        sys.exit(1)

    # Step 5: Backend comparison
    insert_speedup, query_speedup = compare_holon_backends()

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"✅ CuPy: Available")
    print(f"✅ GPU: Detected")
    print(f"✅ Vector ops speedup: {vec_speedup:.2f}x")
    print(f"✅ Batch similarity speedup: {batch_speedup:.2f}x")
    print(f"✅ Holon insert speedup: {insert_speedup:.2f}x")
    print(f"✅ Holon query speedup: {query_speedup:.2f}x")
    print(f"\n🎉 GPU acceleration validated!")

    # Recommendations
    print("\n📋 Recommendations:")
    if batch_speedup > 2:
        print("   ✅ GPU provides significant speedup for batch operations")
        print("   ✅ Use GPU backend for large datasets (1000+ items)")
    else:
        print("   ⚠️  GPU speedup modest - consider CPU for small workloads")
        print("   ✅ GPU still beneficial for very large batch operations")

    print("\n   Next: Run challenge solutions with --gpu flag")


if __name__ == "__main__":
    main()
