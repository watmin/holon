#!/usr/bin/env python3
"""
Challenge 010-011: Streaming Prototype with Full Structure

Previous approach was too simplified:
- Encoded pattern as single atom (lost structural richness)
- Used pattern counting instead of streaming updates
- Didn't use prototype_add() as intended

This demo:
1. Uses FULL structural encoding (all fields, nested data)
2. Uses prototype_add() for streaming updates
3. Frequent patterns naturally dominate via repeated updates
4. Rare patterns (anomalies) have minimal impact

The key insight: If 99% of observations are benign, the prototype
is updated 99 times with benign data for every 1 malicious update.
The malicious patterns get "drowned out" naturally.
"""

import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

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


class StreamingPrototype:
    """
    A prototype that updates incrementally via streaming observations.

    Uses Holon's prototype_add() primitive to:
    - Incrementally update with each observation
    - Naturally weight by frequency (more observations = more influence)
    - Support optional decay for recency weighting

    Encodes the FULL structure of each record, not simplified patterns.
    """

    def __init__(
        self,
        encoder: Encoder,
        decay: float = 1.0,  # 1.0 = no decay, <1.0 = older observations matter less
    ):
        self.encoder = encoder
        self.decay = decay

        self.prototype = None
        self.observation_count = 0

    def observe(self, record: dict):
        """
        Update the streaming prototype with a new observation.

        Each observation:
        1. Encodes the FULL record structure
        2. Updates the prototype using prototype_add()
        3. High-frequency patterns naturally dominate
        """
        # Encode the full record structure
        vec = self.encoder.encode_data(record)

        if self.prototype is None:
            # First observation
            self.prototype = vec.copy()
            self.observation_count = 1
        else:
            # Apply decay if configured
            if self.decay < 1.0:
                # Exponential moving average style
                # new_proto = decay * old_proto + (1 - decay) * new_observation
                self.prototype = self._ema_update(vec)
            else:
                # Standard prototype_add (equal weight to all observations)
                self.prototype = self.encoder.prototype_add(
                    self.prototype, vec, self.observation_count
                )
            self.observation_count += 1

    def _ema_update(self, vec: np.ndarray) -> np.ndarray:
        """Exponential moving average update."""
        # Weighted combination
        alpha = 1.0 - self.decay  # Higher alpha = new observations matter more
        combined = self.decay * self.prototype.astype(np.float32) + alpha * vec.astype(np.float32)
        # Threshold back to bipolar
        return np.where(
            combined > 0, 1,
            np.where(combined < 0, -1, 0)
        ).astype(np.int8)

    def observe_batch(self, records: List[dict]):
        """Observe a batch of records in sequence."""
        for record in records:
            self.observe(record)

    def similarity(self, record: dict) -> float:
        """Check similarity of a record to the prototype."""
        if self.prototype is None:
            return 0.0
        vec = self.encoder.encode_data(record)
        return cosine(vec, self.prototype)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "observation_count": self.observation_count,
            "decay": self.decay,
            "prototype_density": np.sum(self.prototype != 0) / len(self.prototype) if self.prototype is not None else 0,
        }


def generate_rich_traffic(n_benign: int, n_malicious: int, seed: int = 42) -> Tuple[List[dict], List[bool]]:
    """
    Generate RICH traffic data with full structure.

    Not just method+path, but headers, bodies, metadata, etc.
    """
    import random
    random.seed(seed)

    # Benign request templates with realistic structure
    benign_templates = [
        {
            "method": "GET",
            "path": "/api/users",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Mozilla/5.0",
            },
            "query": {},
            "body": None,
            "_weight": 100,
        },
        {
            "method": "GET",
            "path": "/api/users/{id}",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": "Bearer token",
            },
            "query": {},
            "body": None,
            "_weight": 80,
        },
        {
            "method": "POST",
            "path": "/api/users",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            "body": {
                "name": "user",
                "email": "user@example.com",
            },
            "_weight": 40,
        },
        {
            "method": "GET",
            "path": "/api/orders",
            "headers": {
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            "query": {"page": "1", "limit": "20"},
            "body": None,
            "_weight": 90,
        },
        {
            "method": "POST",
            "path": "/api/orders",
            "headers": {
                "Content-Type": "application/json",
            },
            "body": {
                "items": ["item1", "item2"],
                "total": "99.99",
            },
            "_weight": 50,
        },
        {
            "method": "GET",
            "path": "/api/products",
            "headers": {
                "Accept": "application/json",
            },
            "query": {"category": "electronics"},
            "body": None,
            "_weight": 70,
        },
        {
            "method": "POST",
            "path": "/api/auth/login",
            "headers": {
                "Content-Type": "application/json",
            },
            "body": {
                "username": "user",
                "password": "***",
            },
            "_weight": 100,
        },
    ]

    # Malicious request templates with suspicious structure
    malicious_templates = [
        {
            "method": "GET",
            "path": "/api/../../../etc/passwd",
            "headers": {
                "Content-Type": "application/json",
            },
            "query": {},
            "body": None,
        },
        {
            "method": "GET",
            "path": "/api/users/' OR 1=1--",
            "headers": {
                "Content-Type": "application/json",
            },
            "query": {},
            "body": None,
        },
        {
            "method": "POST",
            "path": "/api/users",
            "headers": {
                "Content-Type": "application/json",
            },
            "body": {
                "name": "'; DROP TABLE users;--",
                "email": "attacker@evil.com",
            },
        },
        {
            "method": "GET",
            "path": "/admin/config",
            "headers": {
                "X-Forwarded-For": "127.0.0.1",
            },
            "query": {},
            "body": None,
        },
        {
            "method": "TRACE",
            "path": "/api/users",
            "headers": {},
            "query": {},
            "body": None,
        },
        {
            "method": "POST",
            "path": "/api/upload",
            "headers": {
                "Content-Type": "multipart/form-data",
            },
            "body": {
                "file": "../../etc/passwd",
                "content": "malicious",
            },
        },
        {
            "method": "GET",
            "path": "/api/search",
            "headers": {},
            "query": {
                "q": "<script>alert('xss')</script>",
            },
            "body": None,
        },
    ]

    records = []
    labels = []

    # Generate benign with weighted distribution
    total_weight = sum(t["_weight"] for t in benign_templates)
    for _ in range(n_benign):
        r = random.random() * total_weight
        cumulative = 0
        for template in benign_templates:
            cumulative += template["_weight"]
            if r <= cumulative:
                # Copy template without _weight
                record = {k: v for k, v in template.items() if k != "_weight"}
                # Add some variation
                record["timestamp"] = f"2026-01-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z"
                record["request_id"] = f"req_{random.randint(10000, 99999)}"
                records.append(record)
                labels.append(False)
                break

    # Generate malicious
    for _ in range(n_malicious):
        template = random.choice(malicious_templates)
        record = template.copy()
        record["timestamp"] = f"2026-01-{random.randint(1,28):02d}T{random.randint(0,23):02d}:{random.randint(0,59):02d}:00Z"
        record["request_id"] = f"req_{random.randint(10000, 99999)}"
        records.append(record)
        labels.append(True)

    # Shuffle
    combined = list(zip(records, labels))
    random.shuffle(combined)
    records, labels = zip(*combined)

    return list(records), list(labels)


def main():
    print("=" * 80)
    print("Challenge 010-011: Streaming Prototype with Full Structure")
    print("=" * 80)
    print("""
Key changes from previous approach:
1. FULL structure encoding (headers, body, query, etc.)
2. Using prototype_add() for streaming updates
3. Frequency weighting via repeated observations (not counting)
4. Testing different decay factors
""")

    # Setup
    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    # Generate rich traffic
    print("\n--- Generating Rich Traffic ---")
    train_records, train_labels = generate_rich_traffic(
        n_benign=10000, n_malicious=100, seed=42
    )
    test_records, test_labels = generate_rich_traffic(
        n_benign=1000, n_malicious=50, seed=999
    )

    benign_train = sum(1 for l in train_labels if not l)
    malicious_train = sum(1 for l in train_labels if l)
    print(f"Training: {benign_train} benign, {malicious_train} malicious "
          f"({100*malicious_train/(benign_train+malicious_train):.1f}% contamination)")

    benign_test = sum(1 for l in test_labels if not l)
    malicious_test = sum(1 for l in test_labels if l)
    print(f"Test: {benign_test} benign, {malicious_test} malicious")

    # Show sample record structure
    print("\n--- Sample Record Structure ---")
    sample = train_records[0]
    for key, value in sample.items():
        if isinstance(value, dict):
            print(f"  {key}: {{...}} ({len(value)} keys)")
        elif value is None:
            print(f"  {key}: null")
        else:
            print(f"  {key}: {str(value)[:50]}")

    # Test different decay factors
    print("\n" + "=" * 80)
    print("Testing Different Decay Factors")
    print("=" * 80)

    decay_values = [1.0, 0.999, 0.99, 0.95, 0.9]
    results = {}

    for decay in decay_values:
        print(f"\n--- Decay = {decay} ---")

        # Create streaming prototype
        streamer = StreamingPrototype(encoder, decay=decay)

        # Stream training data
        start = time.time()
        streamer.observe_batch(train_records)
        train_time = time.time() - start

        stats = streamer.get_stats()
        print(f"Trained on {stats['observation_count']} observations in {train_time:.2f}s")
        print(f"Prototype density: {stats['prototype_density']:.1%}")

        # Evaluate
        benign_sims = []
        malicious_sims = []

        for record, is_malicious in zip(test_records, test_labels):
            sim = streamer.similarity(record)
            if is_malicious:
                malicious_sims.append(sim)
            else:
                benign_sims.append(sim)

        separation = np.mean(benign_sims) - np.mean(malicious_sims)

        print(f"Benign similarity:    mean={np.mean(benign_sims):+.4f}, std={np.std(benign_sims):.4f}")
        print(f"Malicious similarity: mean={np.mean(malicious_sims):+.4f}, std={np.std(malicious_sims):.4f}")
        print(f"Separation: {separation:+.4f}")

        # Find best threshold
        best_f1 = 0
        best_threshold = 0

        for threshold in np.linspace(-0.1, 0.2, 61):
            tp = sum(1 for s in malicious_sims if s < threshold)
            fp = sum(1 for s in benign_sims if s < threshold)
            fn = sum(1 for s in malicious_sims if s >= threshold)

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(0.001, precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        # Report
        tp = sum(1 for s in malicious_sims if s < best_threshold)
        fp = sum(1 for s in benign_sims if s < best_threshold)
        fn = sum(1 for s in malicious_sims if s >= best_threshold)
        tn = len(benign_sims) - fp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        print(f"Best threshold: {best_threshold:.4f}")
        print(f"Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {best_f1:.3f}")

        results[decay] = {
            "separation": separation,
            "f1": best_f1,
            "precision": precision,
            "recall": recall,
            "benign_mean": np.mean(benign_sims),
            "malicious_mean": np.mean(malicious_sims),
        }

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n{'Decay':<10} {'Separation':>12} {'Precision':>12} {'Recall':>12} {'F1':>10}")
    print("-" * 60)
    for decay, r in sorted(results.items(), key=lambda x: -x[1]["f1"]):
        print(f"{decay:<10} {r['separation']:>+12.4f} {r['precision']:>12.1%} {r['recall']:>12.1%} {r['f1']:>10.3f}")

    best_decay = max(results.items(), key=lambda x: x[1]["f1"])
    print(f"\nBest: decay={best_decay[0]} with F1={best_decay[1]['f1']:.3f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print("""
Using prototype_add() for streaming:
- Each observation updates the running prototype
- High-frequency patterns naturally dominate (99 benign updates vs 1 malicious)
- Malicious patterns get "drowned out" by benign ones

Full structure encoding:
- All fields (method, path, headers, body, query) contribute to the vector
- More information = potentially better discrimination

Decay factor:
- decay=1.0: All observations weighted equally (standard prototype_add)
- decay<1.0: Recent observations matter more (EMA-style)

The question: Does this approach beat simple pattern membership?
""")

    # Compare to pattern membership
    print("\n--- Comparison to Pattern Membership ---")

    # For pattern membership, we need to define what a "pattern" is
    def extract_pattern(record):
        return f"{record.get('method', '')}|{record.get('path', '')}"

    from collections import Counter
    pattern_counts = Counter(extract_pattern(r) for r in train_records)
    known_patterns = {p for p, c in pattern_counts.items() if c >= 5}

    # Membership detection
    tp = sum(1 for r, l in zip(test_records, test_labels) if l and extract_pattern(r) not in known_patterns)
    fp = sum(1 for r, l in zip(test_records, test_labels) if not l and extract_pattern(r) not in known_patterns)
    fn = sum(1 for r, l in zip(test_records, test_labels) if l and extract_pattern(r) in known_patterns)

    membership_precision = tp / max(1, tp + fp)
    membership_recall = tp / max(1, tp + fn)
    membership_f1 = 2 * membership_precision * membership_recall / max(0.001, membership_precision + membership_recall)

    print(f"Pattern membership: Precision={membership_precision:.1%}, Recall={membership_recall:.1%}, F1={membership_f1:.3f}")
    print(f"Best streaming:     Precision={best_decay[1]['precision']:.1%}, Recall={best_decay[1]['recall']:.1%}, F1={best_decay[1]['f1']:.3f}")

    if best_decay[1]["f1"] > membership_f1:
        print("\n✓ Streaming prototype BEATS pattern membership!")
    else:
        print("\n✗ Pattern membership still wins.")


if __name__ == "__main__":
    main()
