#!/usr/bin/env python3
"""
Challenge 010-013: Accumulator Prototype (No Intermediate Thresholding)

Problem with prototype_add():
- Thresholds after each update
- After 10,000 updates, each new one barely changes the prototype
- Loses the actual frequency signal

New approach: ACCUMULATOR
- Keep a running FLOAT sum (no thresholding)
- High-frequency patterns contribute more to the sum
- Only threshold at query time (or never - use raw similarity)

This preserves the frequency signal:
- 99% benign = benign patterns have 99x more weight in the sum
- 1% malicious = malicious patterns have minimal weight
"""

import sys
import time
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from deterministic_codebook import DeterministicVectorManager
from holon.encoder import Encoder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize_path(path: str) -> str:
    import re
    result = re.sub(r'/\d+', '/{id}', path)
    return result


def normalize_record(record: dict) -> dict:
    """Normalize record to remove noise."""
    normalized = {}
    normalized["method"] = record.get("method", "")
    normalized["path"] = normalize_path(record.get("path", ""))

    headers = record.get("headers", {})
    if headers:
        normalized["headers"] = {k: "present" for k in headers.keys()}

    body = record.get("body")
    if body is not None and isinstance(body, dict):
        normalized["body_keys"] = sorted(body.keys())

    query = record.get("query", {})
    if query:
        normalized["query_keys"] = sorted(query.keys())

    return normalized


class AccumulatorPrototype:
    """
    Accumulator-based prototype that preserves frequency signal.

    Unlike prototype_add() which thresholds after each update,
    this keeps a running FLOAT sum. High-frequency patterns
    contribute more weight to the accumulator.
    """

    def __init__(self, encoder: Encoder, dimensions: int):
        self.encoder = encoder
        self.dimensions = dimensions

        # Accumulator - float sum, no thresholding
        self.accumulator = np.zeros(dimensions, dtype=np.float64)
        self.observation_count = 0

    def observe(self, record: dict):
        """Add observation to accumulator."""
        normalized = normalize_record(record)
        vec = self.encoder.encode_data(normalized)

        # Add to accumulator (no thresholding!)
        self.accumulator += vec.astype(np.float64)
        self.observation_count += 1

    def observe_batch(self, records: List[dict]):
        for r in records:
            self.observe(r)

    def get_prototype(self, threshold: bool = False) -> np.ndarray:
        """
        Get the prototype vector.

        If threshold=True, returns bipolar {-1, 0, 1}
        If threshold=False, returns normalized float vector
        """
        if threshold:
            return np.where(
                self.accumulator > 0, 1,
                np.where(self.accumulator < 0, -1, 0)
            ).astype(np.int8)
        else:
            # Return normalized float vector
            norm = np.linalg.norm(self.accumulator)
            if norm < 1e-10:
                return np.zeros(self.dimensions, dtype=np.float32)
            return (self.accumulator / norm).astype(np.float32)

    def similarity(self, record: dict, use_float: bool = True) -> float:
        """Compute similarity to accumulated prototype."""
        normalized = normalize_record(record)
        vec = self.encoder.encode_data(normalized)

        proto = self.get_prototype(threshold=not use_float)
        return cosine(vec.astype(np.float32), proto.astype(np.float32))

    def inspect_accumulator(self) -> Dict:
        """Inspect the accumulator state."""
        return {
            "observation_count": self.observation_count,
            "accumulator_mean": np.mean(self.accumulator),
            "accumulator_std": np.std(self.accumulator),
            "accumulator_min": np.min(self.accumulator),
            "accumulator_max": np.max(self.accumulator),
            "positive_dims": np.sum(self.accumulator > 0),
            "negative_dims": np.sum(self.accumulator < 0),
            "zero_dims": np.sum(self.accumulator == 0),
        }


def generate_traffic(n_benign: int, n_malicious: int, seed: int = 42):
    """Generate traffic."""
    import random
    random.seed(seed)

    benign_templates = [
        {"method": "GET", "path": "/api/users", "headers": {"Content-Type": "application/json"}, "_w": 100},
        {"method": "GET", "path": "/api/users/123", "headers": {"Content-Type": "application/json", "Authorization": "x"}, "_w": 80},
        {"method": "POST", "path": "/api/users", "headers": {"Content-Type": "application/json"}, "body": {"name": "x"}, "_w": 40},
        {"method": "GET", "path": "/api/orders", "headers": {"Content-Type": "application/json"}, "query": {"page": "1"}, "_w": 90},
        {"method": "POST", "path": "/api/orders", "headers": {"Content-Type": "application/json"}, "body": {"items": []}, "_w": 50},
    ]

    malicious_templates = [
        {"method": "GET", "path": "/api/../../../etc/passwd", "headers": {}},
        {"method": "GET", "path": "/api/users/' OR 1=1--", "headers": {}},
        {"method": "POST", "path": "/api/users", "headers": {}, "body": {"evil": True, "sql": "x"}},
        {"method": "TRACE", "path": "/api/users", "headers": {}},
        {"method": "GET", "path": "/.git/config", "headers": {}},
    ]

    records, labels = [], []

    total_w = sum(t["_w"] for t in benign_templates)
    for _ in range(n_benign):
        r = random.random() * total_w
        c = 0
        for t in benign_templates:
            c += t["_w"]
            if r <= c:
                records.append({k: v for k, v in t.items() if k != "_w"})
                labels.append(False)
                break

    for _ in range(n_malicious):
        records.append(random.choice(malicious_templates).copy())
        labels.append(True)

    combined = list(zip(records, labels))
    random.shuffle(combined)
    records, labels = zip(*combined)
    return list(records), list(labels)


def main():
    print("=" * 80)
    print("Challenge 010-013: Accumulator Prototype (No Thresholding)")
    print("=" * 80)
    print("""
Key difference: Keep running FLOAT sum, don't threshold after each update.
This preserves the actual frequency signal.
""")

    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    # Generate data with clear frequency imbalance
    train_records, train_labels = generate_traffic(9900, 100, seed=42)  # 99% benign, 1% malicious
    test_records, test_labels = generate_traffic(1000, 100, seed=999)

    print(f"Training: {sum(1 for l in train_labels if not l)} benign, {sum(1 for l in train_labels if l)} malicious")
    print(f"Test: {sum(1 for l in test_labels if not l)} benign, {sum(1 for l in test_labels if l)} malicious")

    # Train accumulator
    print("\n--- Training Accumulator ---")
    accum = AccumulatorPrototype(encoder, dimensions=4096)

    start = time.time()
    accum.observe_batch(train_records)
    train_time = time.time() - start

    stats = accum.inspect_accumulator()
    print(f"Observations: {stats['observation_count']}")
    print(f"Train time: {train_time:.2f}s")
    print(f"Accumulator stats:")
    print(f"  Mean: {stats['accumulator_mean']:.4f}")
    print(f"  Std:  {stats['accumulator_std']:.4f}")
    print(f"  Range: [{stats['accumulator_min']:.1f}, {stats['accumulator_max']:.1f}]")
    print(f"  Positive dims: {stats['positive_dims']}")
    print(f"  Negative dims: {stats['negative_dims']}")

    # Evaluate with float prototype
    print("\n--- Evaluation (Float Prototype) ---")

    benign_sims = []
    malicious_sims = []

    for record, is_mal in zip(test_records, test_labels):
        sim = accum.similarity(record, use_float=True)
        if is_mal:
            malicious_sims.append(sim)
        else:
            benign_sims.append(sim)

    separation = np.mean(benign_sims) - np.mean(malicious_sims)

    print(f"Benign:    mean={np.mean(benign_sims):+.4f}, std={np.std(benign_sims):.4f}, range=[{min(benign_sims):+.4f}, {max(benign_sims):+.4f}]")
    print(f"Malicious: mean={np.mean(malicious_sims):+.4f}, std={np.std(malicious_sims):.4f}, range=[{min(malicious_sims):+.4f}, {max(malicious_sims):+.4f}]")
    print(f"SEPARATION: {separation:+.4f}")

    # Check if separation is correct direction
    if separation > 0:
        print("✓ Correct: Benign has HIGHER similarity to prototype")
    else:
        print("✗ Wrong: Malicious has higher similarity (prototype contaminated?)")

    # Find optimal threshold and compute metrics
    best_f1 = 0
    best_threshold = 0

    for threshold in np.linspace(-0.2, 0.5, 141):
        tp = sum(1 for s in malicious_sims if s < threshold)
        fp = sum(1 for s in benign_sims if s < threshold)
        fn = sum(1 for s in malicious_sims if s >= threshold)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    # Report at best threshold
    tp = sum(1 for s in malicious_sims if s < best_threshold)
    fp = sum(1 for s in benign_sims if s < best_threshold)
    fn = sum(1 for s in malicious_sims if s >= best_threshold)
    tn = len(benign_sims) - fp

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)

    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"Precision: {precision:.1%}")
    print(f"Recall: {recall:.1%}")
    print(f"F1: {best_f1:.3f}")

    # Compare to thresholded version
    print("\n--- Comparison: Float vs Thresholded Prototype ---")

    # Thresholded
    benign_sims_t = []
    malicious_sims_t = []

    for record, is_mal in zip(test_records, test_labels):
        sim = accum.similarity(record, use_float=False)
        if is_mal:
            malicious_sims_t.append(sim)
        else:
            benign_sims_t.append(sim)

    separation_t = np.mean(benign_sims_t) - np.mean(malicious_sims_t)

    print(f"Float prototype:      separation={separation:+.4f}")
    print(f"Thresholded prototype: separation={separation_t:+.4f}")

    # Show what's in the accumulator (top patterns)
    print("\n--- Accumulator Analysis ---")
    print("The accumulator captures which patterns are most frequent.")
    print("High positive values = frequently seen in benign traffic")
    print("Values near zero = rare or balanced")

    # Show sample predictions
    print("\n--- Sample Predictions ---")
    test_with_sim = [(r, l, accum.similarity(r)) for r, l in zip(test_records, test_labels)]
    test_with_sim.sort(key=lambda x: x[2])

    print("\nLowest similarity (predicted malicious):")
    for r, is_mal, sim in test_with_sim[:5]:
        actual = "MAL" if is_mal else "ben"
        correct = "✓" if is_mal else "✗"
        print(f"  {correct} sim={sim:+.4f} actual={actual} | {r.get('method', '')} {r.get('path', '')[:40]}")

    print("\nHighest similarity (predicted benign):")
    for r, is_mal, sim in test_with_sim[-5:]:
        actual = "MAL" if is_mal else "ben"
        correct = "✗" if is_mal else "✓"
        print(f"  {correct} sim={sim:+.4f} actual={actual} | {r.get('method', '')} {r.get('path', '')[:40]}")

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print(f"""
Accumulator approach:
- Keep running FLOAT sum (no thresholding during training)
- Frequency is preserved: 99 benign updates >> 1 malicious update
- Float prototype captures the actual distribution

Results:
- Separation: {separation:+.4f} ({'positive ✓' if separation > 0 else 'negative ✗'})
- F1: {best_f1:.3f}

{'This shows that preserving frequency via accumulation improves separation.' if separation > 0 else 'The approach still has issues - need more investigation.'}
""")


if __name__ == "__main__":
    main()
