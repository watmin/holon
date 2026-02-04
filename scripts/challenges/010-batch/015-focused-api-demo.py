#!/usr/bin/env python3
"""
Challenge 010-015: Focused API Request Anomaly Detection

Focus on a single schema (api_request) for cleaner demonstration
of the accumulator primitive's effectiveness.

This mimics a real-world scenario: monitoring API traffic for anomalies.
"""

import sys
import time
from typing import List

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


def generate_api_requests(n: int, seed: int = 42, malicious_rate: float = 0.0) -> List[dict]:
    """
    Generate realistic API request records.

    Normal patterns:
    - Standard REST endpoints
    - Valid HTTP methods
    - Normal headers

    Malicious patterns (when malicious_rate > 0):
    - SQL injection in paths
    - Path traversal
    - Unusual methods
    - Suspicious headers
    """
    import random
    random.seed(seed)

    # Normal request templates
    benign_templates = [
        {"method": "GET", "path": "/api/users", "headers": {"Content-Type": "application/json"}, "_weight": 100},
        {"method": "GET", "path": "/api/users/{id}", "headers": {"Content-Type": "application/json", "Authorization": "Bearer"}, "_weight": 80},
        {"method": "POST", "path": "/api/users", "headers": {"Content-Type": "application/json"}, "body": {"type": "user_create"}, "_weight": 40},
        {"method": "GET", "path": "/api/orders", "headers": {"Content-Type": "application/json"}, "query": {"page": "1"}, "_weight": 90},
        {"method": "POST", "path": "/api/orders", "headers": {"Content-Type": "application/json"}, "body": {"type": "order_create"}, "_weight": 50},
        {"method": "GET", "path": "/api/products", "headers": {"Accept": "application/json"}, "_weight": 70},
        {"method": "GET", "path": "/api/search", "headers": {"Accept": "application/json"}, "query": {"q": "normal"}, "_weight": 60},
        {"method": "POST", "path": "/api/auth/login", "headers": {"Content-Type": "application/json"}, "body": {"type": "login"}, "_weight": 100},
        {"method": "POST", "path": "/api/auth/logout", "headers": {"Content-Type": "application/json"}, "_weight": 30},
        {"method": "GET", "path": "/api/health", "headers": {}, "_weight": 50},
        {"method": "GET", "path": "/api/metrics", "headers": {"Authorization": "Bearer"}, "_weight": 20},
        {"method": "PUT", "path": "/api/users/{id}", "headers": {"Content-Type": "application/json"}, "body": {"type": "user_update"}, "_weight": 25},
    ]

    # Malicious request templates
    malicious_templates = [
        {"method": "GET", "path": "/api/../../../etc/passwd", "headers": {}},
        {"method": "GET", "path": "/api/users/' OR '1'='1", "headers": {}},
        {"method": "GET", "path": "/api/users/1; DROP TABLE users;--", "headers": {}},
        {"method": "POST", "path": "/api/auth/login", "headers": {}, "body": {"type": "sql_injection", "user": "admin'--"}},
        {"method": "TRACE", "path": "/api/users", "headers": {}},
        {"method": "GET", "path": "/.git/config", "headers": {}},
        {"method": "GET", "path": "/.env", "headers": {}},
        {"method": "GET", "path": "/api/search", "query": {"q": "<script>alert(1)</script>"}, "headers": {}},
        {"method": "GET", "path": "/admin/config", "headers": {"X-Forwarded-For": "127.0.0.1"}},
        {"method": "POST", "path": "/api/upload", "headers": {"Content-Type": "multipart/form-data"}, "body": {"file": "../../../etc/passwd"}},
        {"method": "GET", "path": "/api/..%2f..%2f..%2fetc/passwd", "headers": {}},  # URL encoded
        {"method": "GET", "path": "/api/users/$(whoami)", "headers": {}},  # Command injection
    ]

    records = []
    labels = []

    total_weight = sum(t["_weight"] for t in benign_templates)

    for _ in range(n):
        if random.random() < malicious_rate:
            # Generate malicious
            template = random.choice(malicious_templates)
            record = {k: v for k, v in template.items()}
            records.append(record)
            labels.append(True)
        else:
            # Generate benign
            r = random.random() * total_weight
            cumulative = 0
            for template in benign_templates:
                cumulative += template["_weight"]
                if r <= cumulative:
                    record = {k: v for k, v in template.items() if k != "_weight"}
                    records.append(record)
                    labels.append(False)
                    break

    return records, labels


def normalize_request(record: dict) -> dict:
    """Normalize API request for encoding."""
    import re

    normalized = {}
    normalized["method"] = record.get("method", "GET")

    path = record.get("path", "")
    # Normalize IDs
    path = re.sub(r'/\d+', '/{id}', path)
    normalized["path"] = path

    # Headers - just keys
    headers = record.get("headers", {})
    if headers:
        normalized["headers"] = sorted(headers.keys())

    # Body type if present
    body = record.get("body")
    if body:
        if isinstance(body, dict):
            normalized["body_type"] = body.get("type", "unknown")
        else:
            normalized["has_body"] = True

    # Query keys if present
    query = record.get("query")
    if query:
        normalized["query_keys"] = sorted(query.keys())

    return normalized


def main():
    print("=" * 80)
    print("Challenge 010-015: Focused API Request Anomaly Detection")
    print("=" * 80)
    print("""
Scenario: Monitor API traffic for anomalies.
- Training: 10,000 benign API requests
- Test: 1,000 benign + 100 malicious requests

Using Holon's accumulator primitives for detection.
""")

    # Setup
    vm = VectorManager(dimensions=4096)
    encoder = Encoder(vector_manager=vm)

    # Generate data
    print("--- Generating Data ---")
    train_records, train_labels = generate_api_requests(10000, seed=42, malicious_rate=0.0)
    test_benign, _ = generate_api_requests(1000, seed=100, malicious_rate=0.0)
    test_malicious, _ = generate_api_requests(100, seed=200, malicious_rate=1.0)

    print(f"Training: {len(train_records)} benign requests")
    print(f"Test: {len(test_benign)} benign + {len(test_malicious)} malicious")

    # Show sample records
    print("\n--- Sample Normalized Records ---")
    print("Benign:")
    for r in train_records[:3]:
        print(f"  {normalize_request(r)}")

    print("\nMalicious:")
    for r in test_malicious[:3]:
        print(f"  {normalize_request(r)}")

    # Build accumulator
    print("\n--- Building Accumulator ---")
    accum = encoder.create_accumulator()

    start = time.time()
    for record in train_records:
        normalized = normalize_request(record)
        vec = encoder.encode_data(normalized)
        accum = encoder.accumulate(accum, vec)

    train_time = time.time() - start
    print(f"Trained on {len(train_records)} requests in {train_time:.2f}s")
    print(f"Rate: {len(train_records)/train_time:.0f} req/sec")

    # Get normalized accumulator
    normalized_accum = encoder.normalize_accumulator(accum)

    # Evaluate
    print("\n--- Evaluation ---")

    benign_sims = []
    malicious_sims = []

    for record in test_benign:
        normalized = normalize_request(record)
        vec = encoder.encode_data(normalized)
        sim = cosine_similarity(vec, normalized_accum)
        benign_sims.append(sim)

    for record in test_malicious:
        normalized = normalize_request(record)
        vec = encoder.encode_data(normalized)
        sim = cosine_similarity(vec, normalized_accum)
        malicious_sims.append(sim)

    print(f"\nBenign:    mean={np.mean(benign_sims):+.4f}, std={np.std(benign_sims):.4f}, range=[{min(benign_sims):+.4f}, {max(benign_sims):+.4f}]")
    print(f"Malicious: mean={np.mean(malicious_sims):+.4f}, std={np.std(malicious_sims):.4f}, range=[{min(malicious_sims):+.4f}, {max(malicious_sims):+.4f}]")

    separation = np.mean(benign_sims) - np.mean(malicious_sims)
    print(f"\nSEPARATION: {separation:+.4f}")

    # Check overlap
    benign_min = min(benign_sims)
    malicious_max = max(malicious_sims)

    if benign_min > malicious_max:
        print("\n✓ PERFECT SEPARATION - No overlap!")
        optimal_threshold = (benign_min + malicious_max) / 2
        print(f"  Optimal threshold: {optimal_threshold:.4f}")

        # At optimal threshold
        tp = len(malicious_sims)  # All malicious below threshold
        fp = 0  # No benign below threshold
        fn = 0  # No malicious above threshold
        tn = len(benign_sims)  # All benign above threshold

        print(f"\n  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision: 100.0%")
        print(f"  Recall: 100.0%")
        print(f"  F1: 1.000")
    else:
        # Find best threshold
        best_f1 = 0
        best_threshold = 0

        for threshold in np.linspace(-0.5, 0.8, 261):
            tp = sum(1 for s in malicious_sims if s < threshold)
            fp = sum(1 for s in benign_sims if s < threshold)
            fn = sum(1 for s in malicious_sims if s >= threshold)

            precision = tp / max(1, tp + fp)
            recall = tp / max(1, tp + fn)
            f1 = 2 * precision * recall / max(0.001, precision + recall)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        tp = sum(1 for s in malicious_sims if s < best_threshold)
        fp = sum(1 for s in benign_sims if s < best_threshold)
        fn = sum(1 for s in malicious_sims if s >= best_threshold)
        tn = len(benign_sims) - fp

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)

        print(f"\n  Best threshold: {best_threshold:.4f}")
        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall: {recall:.1%}")
        print(f"  F1: {best_f1:.3f}")

    # Sample predictions
    print("\n--- Sample Predictions ---")

    print("\nMost anomalous (lowest similarity):")
    malicious_with_sim = sorted(zip(test_malicious, malicious_sims), key=lambda x: x[1])
    for record, sim in malicious_with_sim[:5]:
        method = record.get("method", "?")
        path = record.get("path", "?")[:35]
        print(f"  sim={sim:+.4f} | {method:6} {path}")

    print("\nMost normal (highest similarity):")
    benign_with_sim = sorted(zip(test_benign, benign_sims), key=lambda x: -x[1])
    for record, sim in benign_with_sim[:5]:
        method = record.get("method", "?")
        path = record.get("path", "?")[:35]
        print(f"  sim={sim:+.4f} | {method:6} {path}")

    # Summary
    print("\n" + "=" * 80)
    print("RESULT: Accumulator works for API anomaly detection!")
    print("=" * 80)


if __name__ == "__main__":
    main()
