#!/usr/bin/env python3
"""
Challenge 010-003: Why Is Baseline Similarity So Low?

Investigation into why normal records have near-zero similarity
to the baseline prototype.

Hypothesis: The records vary too much in OTHER fields (latency_ms,
timestamp, ip_address, etc.) that aren't being normalized.
"""

import sys
from collections import Counter

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from realistic_data_generator import RealisticDataGenerator

from holon.encoder import Encoder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def main():
    print("=" * 70)
    print("Investigation: Why Is Baseline Similarity Low?")
    print("=" * 70)

    # Setup
    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    gen = RealisticDataGenerator(seed=42, cardinality=10000)

    # Generate some api_request records
    records = []
    for i in range(100):
        record, _, _ = gen.generate_record("api_request", i)
        records.append(record)

    # Encode all records
    vectors = [encoder.encode_data(r) for r in records]

    # Check similarity between random pairs
    print("\n--- Similarity Between Random Record Pairs ---")
    pairs = [(0, 1), (0, 10), (0, 50), (10, 20), (25, 75)]
    for i, j in pairs:
        sim = cosine(vectors[i], vectors[j])
        print(f"  records[{i}] vs records[{j}]: {sim:.4f}")

    # Check what fields vary
    print("\n--- Field Variance Analysis ---")
    all_fields = set()
    for r in records:
        all_fields.update(r.keys())

    for field in sorted(all_fields):
        values = [str(r.get(field, "MISSING")) for r in records]
        unique = len(set(values))
        print(f"  {field}: {unique} unique values")

    # The problem: each record encodes MANY different atoms
    # Let's see what atoms are in each record
    print("\n--- Atoms Per Record ---")
    def count_atoms(data, prefix=""):
        atoms = []
        if isinstance(data, dict):
            for k, v in data.items():
                atoms.append(f"{prefix}{k}")  # The key
                atoms.extend(count_atoms(v, f"{prefix}{k}."))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                atoms.extend(count_atoms(item, f"{prefix}[{i}]."))
        else:
            atoms.append(f"{prefix}={data}")  # The value
        return atoms

    for i in range(3):
        atoms = count_atoms(records[i])
        print(f"  Record {i}: {len(atoms)} atoms")
        # Show first few
        print(f"    Sample: {atoms[:5]}...")

    # The insight: with so many varying atoms, similarity is diluted
    # Let's try encoding JUST the normalized path
    print("\n--- Similarity with ONLY Path Field ---")

    def normalize_path(path):
        import re
        if not isinstance(path, str):
            return str(path)
        # Replace numeric IDs with placeholder
        result = re.sub(r'/\d+', '/{id}', path)
        result = re.sub(r'/usr_[a-f0-9]+', '/{user_id}', result)
        return result

    path_only_records = [{"path": normalize_path(r.get("path", ""))} for r in records]
    path_vectors = [encoder.encode_data(r) for r in path_only_records]

    # Now check similarity
    print("  Using only normalized path:")
    for i, j in pairs:
        sim = cosine(path_vectors[i], path_vectors[j])
        path_i = path_only_records[i]["path"]
        path_j = path_only_records[j]["path"]
        match = "SAME" if path_i == path_j else "DIFF"
        print(f"    [{i}] vs [{j}]: sim={sim:.4f} ({match}: {path_i} vs {path_j})")

    # Group by path and check within-group similarity
    print("\n--- Within-Group Similarity (Same Normalized Path) ---")
    path_groups = {}
    for i, r in enumerate(path_only_records):
        path = r["path"]
        if path not in path_groups:
            path_groups[path] = []
        path_groups[path].append(i)

    print(f"  Found {len(path_groups)} unique normalized paths")
    for path, indices in list(path_groups.items())[:3]:
        if len(indices) >= 2:
            sim = cosine(path_vectors[indices[0]], path_vectors[indices[1]])
            print(f"    {path}: {len(indices)} records, sim between first two: {sim:.4f}")

    # The solution: encode only RELEVANT fields for the detection task
    print("\n--- Solution: Selective Field Encoding ---")

    def extract_relevant_fields(record):
        """Extract only the fields relevant for anomaly detection."""
        return {
            "path": normalize_path(record.get("path", "")),
            "method": record.get("method", ""),
            "status_code": record.get("status_code", ""),
            # Ignore: timestamp, latency, user_id, session_id, etc.
        }

    relevant_records = [extract_relevant_fields(r) for r in records]
    relevant_vectors = [encoder.encode_data(r) for r in relevant_records]

    # Now check similarity
    print("  Using only relevant fields (path, method, status_code):")
    for i, j in pairs:
        sim = cosine(relevant_vectors[i], relevant_vectors[j])
        print(f"    [{i}] vs [{j}]: sim={sim:.4f}")

    # Build prototype from relevant-only encoding
    print("\n--- Prototype with Relevant Fields ---")
    prototype = encoder.prototype(relevant_vectors, threshold=0.3)

    sims = [cosine(v, prototype) for v in relevant_vectors]
    print(f"  Similarity to prototype: min={min(sims):.4f}, max={max(sims):.4f}, mean={np.mean(sims):.4f}")

    # Now test with attacks
    print("\n--- Attack Detection with Relevant Fields ---")
    attacks = [
        {"path": "/api/../../../etc/passwd", "method": "GET", "status_code": 200},
        {"path": "/api/users/' OR 1=1--", "method": "GET", "status_code": 200},
        {"path": "/api/exec?cmd=ls", "method": "GET", "status_code": 200},
        {"path": "/api/users", "method": "TRACE", "status_code": 200},  # Unusual method
    ]

    for attack in attacks:
        attack_relevant = extract_relevant_fields(attack)
        attack_vec = encoder.encode_data(attack_relevant)
        sim = cosine(attack_vec, prototype)

        # Compare to normal sims
        percentile = sum(1 for s in sims if s < sim) / len(sims) * 100

        print(f"  {attack['path'][:40]:40} sim={sim:.4f} (percentile: {percentile:.1f}%)")

    print("\n" + "=" * 70)
    print("FINDINGS")
    print("=" * 70)
    print("""
The Problem:
  When encoding FULL records, each record has many varying fields
  (timestamp, latency_ms, ip_address, etc.) that create unique vectors.
  Similarity between any two records is near zero because they differ
  in too many dimensions.

The Solution:
  Encode only the RELEVANT fields for your detection task:
  - For API anomaly detection: path, method, status_code
  - Ignore: timestamp, latency, user_id, etc.

This is domain knowledge! You must decide:
  1. Which fields define "similarity" for your use case
  2. Which fields are noise that should be ignored
  3. Which fields need normalization (like path)

VSA/HDC encodes EVERYTHING in the input. If you give it noise,
it encodes noise. Garbage in, garbage out.
""")


if __name__ == "__main__":
    main()
