#!/usr/bin/env python3

"""
Test GPU acceleration with Holon.
"""

try:
    import cupy as cp

    print("✅ CuPy available - GPU support enabled")
    print(f"GPU devices: {cp.cuda.runtime.getDeviceCount()}")
    print(f"Current GPU: {cp.cuda.runtime.getDevice()}")
except ImportError:
    print("❌ CuPy not available - install with: pip install cupy-cuda12x")
    exit(1)

from holon import CPUStore


def test_gpu():
    print("\n🚀 Testing GPU acceleration...")

    # Test GPU store
    store = CPUStore(dimensions=1000, backend="gpu")

    # Test basic operations
    data = '{"test": "gpu_accelerated"}'
    id = store.insert(data, "json")
    print(f"✅ Inserted on GPU: {id}")

    # Test query
    results = store.query(data, "json", top_k=5)
    print(f"✅ Queried on GPU: {len(results)} results")

    # Test vector types
    sample_vec = store.vector_manager.get_vector("test")
    print(f"✅ Vector type: {type(sample_vec)}")
    print(f"✅ Vector shape: {sample_vec.shape}")

    print("\n🎉 GPU acceleration working!")


if __name__ == "__main__":
    test_gpu()
