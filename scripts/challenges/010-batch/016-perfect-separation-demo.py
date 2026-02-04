#!/usr/bin/env python3
"""
Challenge 010-016: Perfect Separation Demo

Replicates the successful approach from 013 but using official Holon primitives.

Key factors for perfect separation:
1. Clean benign templates (few patterns, high frequency)
2. Distinct malicious templates (clearly different structure)
3. Simple normalization (reduce noise)
4. Large training set to build strong benign signal
"""

import sys
import time
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon.encoder import Encoder
from holon.vector_manager import VectorManager


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


def generate_data(
    n_benign_train: int,
    n_benign_test: int,
    n_malicious_test: int,
    seed: int = 42
) -> Tuple[List[dict], List[dict], List[dict]]:
    """Generate benign training, benign test, and malicious test data."""
    import random

    # Benign templates - common API patterns
    benign_templates = [
        {"method": "GET", "path": "/api/users", "headers": {"Content-Type": "json"}, "_w": 100},
        {"method": "GET", "path": "/api/users/{id}", "headers": {"Content-Type": "json", "Auth": "y"}, "_w": 80},
        {"method": "POST", "path": "/api/users", "headers": {"Content-Type": "json"}, "body": {"name": "x"}, "_w": 40},
        {"method": "GET", "path": "/api/orders", "headers": {"Content-Type": "json"}, "query": {"page": "1"}, "_w": 90},
        {"method": "POST", "path": "/api/orders", "headers": {"Content-Type": "json"}, "body": {"items": []}, "_w": 50},
    ]

    # Malicious templates - clearly different patterns
    malicious_templates = [
        {"method": "GET", "path": "/api/../../../etc/passwd", "headers": {}},
        {"method": "GET", "path": "/api/users/' OR 1=1--", "headers": {}},
        {"method": "POST", "path": "/api/users", "headers": {}, "body": {"evil": True, "sql": "x"}},
        {"method": "TRACE", "path": "/api/users", "headers": {}},
        {"method": "GET", "path": "/.git/config", "headers": {}},
    ]

    def generate_benign(n: int, rng_seed: int) -> List[dict]:
        rng = random.Random(rng_seed)
        total_w = sum(t["_w"] for t in benign_templates)
        records = []
        for _ in range(n):
            r = rng.random() * total_w
            c = 0
            for t in benign_templates:
                c += t["_w"]
                if r <= c:
                    records.append({k: v for k, v in t.items() if k != "_w"})
                    break
        return records

    def generate_malicious(n: int, rng_seed: int) -> List[dict]:
        rng = random.Random(rng_seed)
        return [rng.choice(malicious_templates).copy() for _ in range(n)]

    train = generate_benign(n_benign_train, seed)
    test_benign = generate_benign(n_benign_test, seed + 1000)
    test_malicious = generate_malicious(n_malicious_test, seed + 2000)

    return train, test_benign, test_malicious


def normalize(record: dict) -> dict:
    """Simple normalization - keep structure, remove noise."""
    result = {}
    result["method"] = record.get("method", "")
    result["path"] = record.get("path", "")

    headers = record.get("headers", {})
    if headers:
        result["headers"] = sorted(headers.keys())

    body = record.get("body")
    if body and isinstance(body, dict):
        result["body_keys"] = sorted(body.keys())

    query = record.get("query")
    if query:
        result["query_keys"] = sorted(query.keys())

    return result


def main():
    print("=" * 80)
    print("Challenge 010-016: Perfect Separation Demo")
    print("=" * 80)
    print("""
Setup:
- 10,000 benign training (5 patterns, high frequency)
- 1,000 benign test
- 100 malicious test (5 clearly different patterns)

Using Holon's accumulator primitives.
""")

    # Setup
    vm = VectorManager(dimensions=4096)
    encoder = Encoder(vector_manager=vm)

    # Generate data
    train, test_benign, test_malicious = generate_data(
        n_benign_train=10000,
        n_benign_test=1000,
        n_malicious_test=100,
        seed=42
    )

    print(f"Training: {len(train)} benign")
    print(f"Test: {len(test_benign)} benign + {len(test_malicious)} malicious")

    # Show normalized samples
    print("\n--- Normalized Samples ---")
    print("Benign:")
    for r in train[:3]:
        print(f"  {normalize(r)}")

    print("\nMalicious:")
    for r in test_malicious[:3]:
        print(f"  {normalize(r)}")

    # Build accumulator using Holon primitives
    print("\n--- Building Accumulator (Holon Primitives) ---")

    accum = encoder.create_accumulator()  # ← Holon primitive

    start = time.time()
    for record in train:
        normalized = normalize(record)
        vec = encoder.encode_data(normalized)
        accum = encoder.accumulate(accum, vec)  # ← Holon primitive

    train_time = time.time() - start
    print(f"Trained in {train_time:.2f}s ({len(train)/train_time:.0f} obs/sec)")

    # Get normalized accumulator for queries
    proto = encoder.normalize_accumulator(accum)  # ← Holon primitive

    # Evaluate
    print("\n--- Evaluation ---")

    benign_sims = []
    malicious_sims = []

    for record in test_benign:
        normalized = normalize(record)
        vec = encoder.encode_data(normalized)
        sim = cosine_similarity(vec, proto)
        benign_sims.append(sim)

    for record in test_malicious:
        normalized = normalize(record)
        vec = encoder.encode_data(normalized)
        sim = cosine_similarity(vec, proto)
        malicious_sims.append(sim)

    # Statistics
    print(f"\nBenign:    mean={np.mean(benign_sims):+.4f}, range=[{min(benign_sims):+.4f}, {max(benign_sims):+.4f}]")
    print(f"Malicious: mean={np.mean(malicious_sims):+.4f}, range=[{min(malicious_sims):+.4f}, {max(malicious_sims):+.4f}]")

    separation = np.mean(benign_sims) - np.mean(malicious_sims)
    print(f"\nSEPARATION: {separation:+.4f}")

    # Check for perfect separation
    benign_min = min(benign_sims)
    malicious_max = max(malicious_sims)

    if benign_min > malicious_max:
        print("\n" + "=" * 50)
        print("✓ PERFECT SEPARATION - F1 = 1.000")
        print("=" * 50)
        print(f"\nBenign minimum:   {benign_min:+.4f}")
        print(f"Malicious maximum: {malicious_max:+.4f}")
        print(f"Gap:              {benign_min - malicious_max:+.4f}")

        optimal_threshold = (benign_min + malicious_max) / 2
        print(f"\nOptimal threshold: {optimal_threshold:.4f}")
        print(f"Any threshold in [{malicious_max:.4f}, {benign_min:.4f}] works perfectly.")

        print(f"\nAt threshold={optimal_threshold:.4f}:")
        print(f"  True Positives:  {len(test_malicious)} (all malicious detected)")
        print(f"  False Positives: 0 (no benign flagged)")
        print(f"  Precision: 100%")
        print(f"  Recall: 100%")
        print(f"  F1 Score: 1.000")
    else:
        # Find best threshold if not perfect
        best_f1 = 0
        for threshold in np.linspace(0, 1, 201):
            tp = sum(1 for s in malicious_sims if s < threshold)
            fp = sum(1 for s in benign_sims if s < threshold)
            fn = sum(1 for s in malicious_sims if s >= threshold)

            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = 2 * p * r / max(0.001, p + r)

            if f1 > best_f1:
                best_f1 = f1

        print(f"\nBest F1: {best_f1:.3f}")

    # Show examples
    print("\n--- Examples ---")

    print("\nMost anomalous (lowest similarity):")
    mal_sorted = sorted(zip(test_malicious, malicious_sims), key=lambda x: x[1])
    for r, s in mal_sorted[:5]:
        print(f"  sim={s:+.4f} | {r.get('method'):6} {r.get('path')[:35]}")

    print("\nMost normal (highest similarity):")
    ben_sorted = sorted(zip(test_benign, benign_sims), key=lambda x: -x[1])
    for r, s in ben_sorted[:5]:
        print(f"  sim={s:+.4f} | {r.get('method'):6} {r.get('path')[:35]}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The accumulator primitive achieves perfect separation when:
1. Benign patterns are consistent (few templates, high frequency)
2. Malicious patterns are structurally different
3. Normalization removes noise while keeping signal

Holon primitives used:
- encoder.create_accumulator()      - Initialize float64 sum
- encoder.accumulate(accum, vec)    - Add without thresholding
- encoder.normalize_accumulator()   - Unit normalize for queries

This validates the new accumulator primitive for anomaly detection!
""")


if __name__ == "__main__":
    main()
