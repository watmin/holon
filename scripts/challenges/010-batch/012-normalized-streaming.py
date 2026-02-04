#!/usr/bin/env python3
"""
Challenge 010-012: Normalized Streaming Prototype

Previous test showed streaming > membership, but separation was small.

Issue: High-variance fields (request_id, timestamp) add noise that
drowns out the structural differences we care about.

Solution: Normalize/remove high-variance fields before encoding.
Keep the fields that define "normal" vs "malicious" structure:
- method, path (normalized)
- headers (keys, not values)
- body structure (keys, not values)

This is domain knowledge: we choose what's relevant for detection.
"""

import sys
import time
from collections import Counter
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


def normalize_path(path: str) -> str:
    """Normalize path by replacing variable segments."""
    import re
    result = re.sub(r'/\d+', '/{id}', path)
    result = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', result)
    return result


def normalize_record(record: dict) -> dict:
    """
    Normalize a record to remove high-variance noise.

    Keep:
    - method (as-is)
    - path (normalized)
    - headers (keys only, values normalized)
    - body structure (keys and value types, not actual values)
    - query structure (keys only)

    Remove:
    - request_id (unique per request)
    - timestamp (unique per request)
    - actual values in body/query
    """
    normalized = {}

    # Method - important signal
    normalized["method"] = record.get("method", "")

    # Path - normalize to remove IDs
    normalized["path"] = normalize_path(record.get("path", ""))

    # Headers - keep keys and normalize common values
    headers = record.get("headers", {})
    if headers:
        header_info = {}
        for key, value in headers.items():
            # Keep key
            # Normalize common headers
            if key.lower() == "content-type":
                header_info[key] = value  # Keep content-type value
            elif key.lower() == "authorization":
                header_info[key] = "present"  # Just note it exists
            else:
                header_info[key] = "present"  # Note presence, not value
        normalized["headers"] = header_info

    # Body - keep structure, not values
    body = record.get("body")
    if body is not None:
        if isinstance(body, dict):
            # Keep keys and value types
            body_structure = {k: type(v).__name__ for k, v in body.items()}
            normalized["body_structure"] = body_structure
        else:
            normalized["body_type"] = type(body).__name__

    # Query - keep keys only
    query = record.get("query", {})
    if query:
        normalized["query_keys"] = list(query.keys())

    return normalized


class NormalizedStreamingPrototype:
    """Streaming prototype with field normalization."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self.prototype = None
        self.observation_count = 0

    def observe(self, record: dict):
        """Observe a record (after normalization)."""
        normalized = normalize_record(record)
        vec = self.encoder.encode_data(normalized)

        if self.prototype is None:
            self.prototype = vec.copy()
            self.observation_count = 1
        else:
            self.prototype = self.encoder.prototype_add(
                self.prototype, vec, self.observation_count
            )
            self.observation_count += 1

    def observe_batch(self, records: List[dict]):
        for r in records:
            self.observe(r)

    def similarity(self, record: dict) -> float:
        if self.prototype is None:
            return 0.0
        normalized = normalize_record(record)
        vec = self.encoder.encode_data(normalized)
        return cosine(vec, self.prototype)


def generate_traffic(n_benign: int, n_malicious: int, seed: int = 42):
    """Generate rich traffic."""
    import random
    random.seed(seed)

    benign_templates = [
        {"method": "GET", "path": "/api/users", "headers": {"Content-Type": "application/json"}, "body": None, "_w": 100},
        {"method": "GET", "path": "/api/users/123", "headers": {"Content-Type": "application/json", "Authorization": "Bearer xxx"}, "body": None, "_w": 80},
        {"method": "POST", "path": "/api/users", "headers": {"Content-Type": "application/json"}, "body": {"name": "x", "email": "y"}, "_w": 40},
        {"method": "GET", "path": "/api/orders", "headers": {"Content-Type": "application/json"}, "query": {"page": "1"}, "body": None, "_w": 90},
        {"method": "POST", "path": "/api/orders", "headers": {"Content-Type": "application/json"}, "body": {"items": [], "total": 0}, "_w": 50},
        {"method": "GET", "path": "/api/products", "headers": {"Accept": "application/json"}, "query": {"cat": "x"}, "body": None, "_w": 70},
        {"method": "POST", "path": "/api/auth/login", "headers": {"Content-Type": "application/json"}, "body": {"user": "x", "pass": "y"}, "_w": 100},
        {"method": "GET", "path": "/api/search", "headers": {"Accept": "application/json"}, "query": {"q": "normal"}, "body": None, "_w": 60},
    ]

    malicious_templates = [
        {"method": "GET", "path": "/api/../../../etc/passwd", "headers": {"Content-Type": "application/json"}, "body": None},
        {"method": "GET", "path": "/api/users/' OR 1=1--", "headers": {}, "body": None},
        {"method": "POST", "path": "/api/users", "headers": {"Content-Type": "application/json"}, "body": {"name": "'; DROP TABLE--", "evil": True}},
        {"method": "GET", "path": "/admin/config", "headers": {"X-Forwarded-For": "127.0.0.1"}, "body": None},
        {"method": "TRACE", "path": "/api/users", "headers": {}, "body": None},
        {"method": "POST", "path": "/api/upload", "headers": {"Content-Type": "multipart/form-data"}, "body": {"file": "x", "path": "../.."}},
        {"method": "GET", "path": "/api/search", "headers": {}, "query": {"q": "<script>alert(1)</script>"}, "body": None},
        {"method": "GET", "path": "/.git/config", "headers": {}, "body": None},
        {"method": "GET", "path": "/.env", "headers": {}, "body": None},
    ]

    records = []
    labels = []

    total_w = sum(t["_w"] for t in benign_templates)
    for _ in range(n_benign):
        r = random.random() * total_w
        cumulative = 0
        for t in benign_templates:
            cumulative += t["_w"]
            if r <= cumulative:
                record = {k: v for k, v in t.items() if k != "_w"}
                record["timestamp"] = f"2026-01-{random.randint(1,28):02d}"
                record["request_id"] = f"req_{random.randint(10000, 99999)}"
                records.append(record)
                labels.append(False)
                break

    for _ in range(n_malicious):
        t = random.choice(malicious_templates)
        record = t.copy()
        record["timestamp"] = f"2026-01-{random.randint(1,28):02d}"
        record["request_id"] = f"req_{random.randint(10000, 99999)}"
        records.append(record)
        labels.append(True)

    combined = list(zip(records, labels))
    random.shuffle(combined)
    records, labels = zip(*combined)
    return list(records), list(labels)


def main():
    print("=" * 80)
    print("Challenge 010-012: Normalized Streaming Prototype")
    print("=" * 80)

    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    # Generate data
    train_records, train_labels = generate_traffic(10000, 100, seed=42)
    test_records, test_labels = generate_traffic(1000, 100, seed=999)

    print(f"\nTraining: {sum(1 for l in train_labels if not l)} benign, {sum(1 for l in train_labels if l)} malicious")
    print(f"Test: {sum(1 for l in test_labels if not l)} benign, {sum(1 for l in test_labels if l)} malicious")

    # Show normalization example
    print("\n--- Normalization Example ---")
    sample = train_records[0]
    print("Original:")
    for k, v in sample.items():
        print(f"  {k}: {v}")

    normalized = normalize_record(sample)
    print("\nNormalized:")
    for k, v in normalized.items():
        print(f"  {k}: {v}")

    # Train streaming prototype
    print("\n--- Training ---")
    streamer = NormalizedStreamingPrototype(encoder)

    start = time.time()
    streamer.observe_batch(train_records)
    train_time = time.time() - start

    print(f"Trained on {streamer.observation_count} observations in {train_time:.2f}s")

    # Evaluate
    print("\n--- Evaluation ---")
    benign_sims = []
    malicious_sims = []

    for record, is_malicious in zip(test_records, test_labels):
        sim = streamer.similarity(record)
        if is_malicious:
            malicious_sims.append(sim)
        else:
            benign_sims.append(sim)

    separation = np.mean(benign_sims) - np.mean(malicious_sims)

    print(f"Benign similarity:    mean={np.mean(benign_sims):+.4f}, std={np.std(benign_sims):.4f}, range=[{min(benign_sims):+.4f}, {max(benign_sims):+.4f}]")
    print(f"Malicious similarity: mean={np.mean(malicious_sims):+.4f}, std={np.std(malicious_sims):.4f}, range=[{min(malicious_sims):+.4f}, {max(malicious_sims):+.4f}]")
    print(f"Separation: {separation:+.4f}")

    # Find optimal threshold
    best_f1 = 0
    best_threshold = 0
    best_precision = 0
    best_recall = 0

    for threshold in np.linspace(-0.1, 0.3, 81):
        tp = sum(1 for s in malicious_sims if s < threshold)
        fp = sum(1 for s in benign_sims if s < threshold)
        fn = sum(1 for s in malicious_sims if s >= threshold)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    print(f"\nBest threshold: {best_threshold:.4f}")
    print(f"Precision: {best_precision:.1%}")
    print(f"Recall: {best_recall:.1%}")
    print(f"F1: {best_f1:.3f}")

    # Show some examples
    print("\n--- Sample Predictions ---")
    test_with_sim = [(r, l, streamer.similarity(r)) for r, l in zip(test_records, test_labels)]

    # Sort by similarity
    test_with_sim.sort(key=lambda x: x[2])

    print("\nLowest similarity (should be malicious):")
    for record, is_mal, sim in test_with_sim[:10]:
        actual = "MAL" if is_mal else "ben"
        predicted = "MAL" if sim < best_threshold else "ben"
        correct = "✓" if (sim < best_threshold) == is_mal else "✗"
        path = record.get("path", "")[:35]
        print(f"  {correct} sim={sim:+.4f} actual={actual} pred={predicted} | {record.get('method', '')[:6]} {path}")

    print("\nHighest similarity (should be benign):")
    for record, is_mal, sim in test_with_sim[-10:]:
        actual = "MAL" if is_mal else "ben"
        predicted = "MAL" if sim < best_threshold else "ben"
        correct = "✓" if (sim < best_threshold) == is_mal else "✗"
        path = record.get("path", "")[:35]
        print(f"  {correct} sim={sim:+.4f} actual={actual} pred={predicted} | {record.get('method', '')[:6]} {path}")

    # Compare to pattern membership
    print("\n--- Comparison to Baselines ---")

    # Pattern membership (normalized pattern)
    def get_pattern(r):
        return f"{r.get('method', '')}|{normalize_path(r.get('path', ''))}"

    pattern_counts = Counter(get_pattern(r) for r in train_records)
    known_patterns = {p for p, c in pattern_counts.items() if c >= 5}

    membership_tp = sum(1 for r, l in zip(test_records, test_labels) if l and get_pattern(r) not in known_patterns)
    membership_fp = sum(1 for r, l in zip(test_records, test_labels) if not l and get_pattern(r) not in known_patterns)
    membership_fn = sum(1 for r, l in zip(test_records, test_labels) if l and get_pattern(r) in known_patterns)

    membership_precision = membership_tp / max(1, membership_tp + membership_fp)
    membership_recall = membership_tp / max(1, membership_tp + membership_fn)
    membership_f1 = 2 * membership_precision * membership_recall / max(0.001, membership_precision + membership_recall)

    print(f"\nPattern membership (normalized):  P={membership_precision:.1%}, R={membership_recall:.1%}, F1={membership_f1:.3f}")
    print(f"Normalized streaming prototype:   P={best_precision:.1%}, R={best_recall:.1%}, F1={best_f1:.3f}")

    if best_f1 > membership_f1:
        print("\n✓ Streaming prototype WINS!")
        improvement = (best_f1 - membership_f1) / max(0.001, membership_f1) * 100
        print(f"  Improvement: {improvement:.1f}% F1")
    else:
        print("\n✗ Pattern membership wins")

    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    print("""
Normalization removes noise and keeps structural signal:
- Remove: request_id, timestamp, specific values
- Keep: method, normalized path, header keys, body structure

The prototype captures "normal structure":
- Common headers (Content-Type: application/json)
- Normal paths (/api/users, /api/orders)
- Typical body shapes (name+email, items+total)

Malicious requests differ in structure:
- Unusual paths (/../../../etc/passwd)
- Missing headers
- Weird body fields (evil: True)

This is the value of full structure encoding:
We can detect structural anomalies, not just pattern novelty.
""")


if __name__ == "__main__":
    main()
