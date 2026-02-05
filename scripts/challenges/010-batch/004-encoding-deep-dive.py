#!/usr/bin/env python3
"""
Challenge 010-004: Deep Dive into Encoding Behavior

Why do two records with path="/api/orders" have sim=0.0091?
They should be nearly identical!
"""

import sys
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    print("=" * 70)
    print("Deep Dive: Encoding Behavior")
    print("=" * 70)

    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    # Test 1: Encode the EXACT same dict twice
    print("\n--- Test 1: Exact Same Dict ---")
    record1 = {"path": "/api/orders", "method": "GET"}
    record2 = {"path": "/api/orders", "method": "GET"}

    vec1 = encoder.encode_data(record1)
    vec2 = encoder.encode_data(record2)

    print(f"record1: {record1}")
    print(f"record2: {record2}")
    print(f"Similarity: {cosine(vec1, vec2):.6f}")
    print(f"Exact match: {np.array_equal(vec1, vec2)}")

    # Test 2: Encode different method
    print("\n--- Test 2: Different Method ---")
    record3 = {"path": "/api/orders", "method": "POST"}
    vec3 = encoder.encode_data(record3)

    print(f"GET vs POST: {cosine(vec1, vec3):.6f}")

    # Test 3: Encode just the path string
    print("\n--- Test 3: Just Path String ---")
    path_vec = vm.get_vector("/api/orders")
    path_vec2 = vm.get_vector("/api/orders")

    print(f"Same path atom twice: {cosine(path_vec, path_vec2):.6f}")
    print(f"Exact match: {np.array_equal(path_vec, path_vec2)}")

    # Test 4: Encode dict with one field
    print("\n--- Test 4: Single Field Dict ---")
    single1 = {"path": "/api/orders"}
    single2 = {"path": "/api/orders"}

    s1 = encoder.encode_data(single1)
    s2 = encoder.encode_data(single2)

    print(f"Single field dicts: {cosine(s1, s2):.6f}")
    print(f"Exact match: {np.array_equal(s1, s2)}")

    # Test 5: What does encode_data actually produce?
    print("\n--- Test 5: Encoding Structure ---")
    print("For {'path': '/api/orders'}:")
    print(f"  encode_data produces vector with {np.sum(s1 != 0)} non-zero elements")

    # The encoding binds key to value: key_vec * value_vec
    key_vec = vm.get_vector("path")
    val_vec = vm.get_vector("/api/orders")
    bound = key_vec * val_vec  # Element-wise multiply

    print(f"  key_vec has {np.sum(key_vec != 0)} non-zero elements")
    print(f"  val_vec has {np.sum(val_vec != 0)} non-zero elements")
    print(f"  bound (key*val) has {np.sum(bound != 0)} non-zero elements")

    # Check if encoding matches expected
    print(f"  encode_data == bound? {cosine(s1, bound):.6f}")

    # Test 6: Two different paths
    print("\n--- Test 6: Different Paths ---")
    orders_vec = encoder.encode_data({"path": "/api/orders"})
    users_vec = encoder.encode_data({"path": "/api/users"})

    print(f"  /api/orders vs /api/users: {cosine(orders_vec, users_vec):.6f}")

    # Just the raw atoms
    orders_atom = vm.get_vector("/api/orders")
    users_atom = vm.get_vector("/api/users")
    print(f"  Raw atoms /api/orders vs /api/users: {cosine(orders_atom, users_atom):.6f}")

    # Test 7: What about bundled vectors?
    print("\n--- Test 7: Multi-Field Encoding ---")
    multi1 = {"path": "/api/orders", "method": "GET", "status": 200}
    multi2 = {"path": "/api/orders", "method": "GET", "status": 200}
    multi3 = {"path": "/api/orders", "method": "POST", "status": 201}

    m1 = encoder.encode_data(multi1)
    m2 = encoder.encode_data(multi2)
    m3 = encoder.encode_data(multi3)

    print(f"  Identical multi-field: {cosine(m1, m2):.6f}")
    print(f"  Different method/status: {cosine(m1, m3):.6f}")

    # Test 8: Build a prototype from identical records
    print("\n--- Test 8: Prototype from Identical Records ---")
    identical = [{"path": "/api/orders", "method": "GET"} for _ in range(10)]
    vecs = [encoder.encode_data(r) for r in identical]

    proto = encoder.prototype(vecs, threshold=0.3)

    print(f"  Prototype from 10 identical records")
    print(f"  Similarity of first record to prototype: {cosine(vecs[0], proto):.6f}")

    # Test 9: Prototype from varied records (same path, different method)
    print("\n--- Test 9: Prototype from Varied Records ---")
    varied = [
        {"path": "/api/orders", "method": "GET"},
        {"path": "/api/orders", "method": "POST"},
        {"path": "/api/orders", "method": "PUT"},
        {"path": "/api/orders", "method": "DELETE"},
    ]
    varied_vecs = [encoder.encode_data(r) for r in varied]
    varied_proto = encoder.prototype(varied_vecs, threshold=0.3)

    for i, r in enumerate(varied):
        sim = cosine(varied_vecs[i], varied_proto)
        print(f"  {r['method']}: sim to prototype = {sim:.6f}")

    # Different path
    diff_path = encoder.encode_data({"path": "/api/users", "method": "GET"})
    print(f"  Different path (/api/users): sim to prototype = {cosine(diff_path, varied_proto):.6f}")

    # Attack path
    attack = encoder.encode_data({"path": "/api/../../../etc/passwd", "method": "GET"})
    print(f"  Attack path: sim to prototype = {cosine(attack, varied_proto):.6f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The encoding IS deterministic and working correctly:
- Same input → same output (exact match)
- Similar inputs → low but non-zero similarity

The issue is that VSA/HDC uses RANDOM orthogonal vectors for atoms.
Any two atoms (like "/api/orders" and "/api/users") are nearly
orthogonal by design (similarity ≈ 0).

This is a FEATURE, not a bug:
- It allows millions of atoms without interference
- But it means "similarity" is based on SHARED ATOMS, not semantics

For anomaly detection to work, you need:
1. Attacks to produce DIFFERENT atoms than normal requests
2. Normal requests to share COMMON atoms

The path "/api/../../../etc/passwd" produces completely different
atoms than "/api/orders", so they're orthogonal - which is what
we want for detection!
""")


if __name__ == "__main__":
    main()
