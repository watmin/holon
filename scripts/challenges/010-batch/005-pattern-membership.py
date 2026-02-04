#!/usr/bin/env python3
"""
Challenge 010-005: Pattern Membership for Anomaly Detection

Key insight: VSA/HDC atoms are random and orthogonal.
There's no semantic similarity - all different atoms are equally distant.

Instead of "how similar is this to normal?", we should ask:
"Does this MATCH any known pattern?"

Approach:
1. Build a SET of known good patterns (normalized templates)
2. For new input, check if it matches any known pattern
3. If it doesn't match → anomaly

This is membership testing, not similarity search.
"""

import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from deterministic_codebook import DeterministicVectorManager
from realistic_data_generator import RealisticDataGenerator

from holon.encoder import Encoder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize_path(path: str) -> str:
    """Normalize a RESTful path by replacing variable segments."""
    import re
    if not isinstance(path, str):
        return str(path)

    result = path
    # Replace numeric IDs
    result = re.sub(r'/\d+', '/{id}', result)
    # Replace UUIDs
    result = re.sub(r'/[a-f0-9-]{36}', '/{uuid}', result)
    # Replace our generated IDs
    result = re.sub(r'/usr_[a-f0-9]+', '/{user_id}', result)
    result = re.sub(r'/ord_\d+', '/{order_id}', result)
    result = re.sub(r'/prod_\d+', '/{product_id}', result)

    return result


@dataclass
class PatternMatch:
    """Result of pattern matching."""
    input_data: dict
    normalized_pattern: str
    matched: bool
    best_match: Optional[str]
    similarity: float


class PatternMembershipDetector:
    """
    Anomaly detection via pattern membership testing.

    Instead of "similarity to baseline", we check if the input
    MATCHES any known pattern exactly (after normalization).

    Unknown patterns → anomaly
    """

    def __init__(
        self,
        encoder: Encoder,
        pattern_fields: List[str],
        normalizers: Dict[str, callable] = None,
    ):
        """
        Args:
            encoder: Holon encoder
            pattern_fields: Which fields define the "pattern" (e.g., ["path", "method"])
            normalizers: Functions to normalize specific fields
        """
        self.encoder = encoder
        self.pattern_fields = pattern_fields
        self.normalizers = normalizers or {}

        # Known patterns: normalized pattern string → count
        self.known_patterns: Dict[str, int] = defaultdict(int)

        # Pattern vectors (for fuzzy matching if needed)
        self.pattern_vectors: Dict[str, np.ndarray] = {}

        # Stats
        self.total_trained = 0
        self.total_checked = 0
        self.anomalies_found = 0

    def _extract_pattern(self, record: dict) -> str:
        """Extract and normalize the pattern from a record."""
        parts = []
        for field in self.pattern_fields:
            value = record.get(field, "")

            # Apply normalizer if available
            if field in self.normalizers:
                value = self.normalizers[field](value)

            parts.append(f"{field}={value}")

        return "|".join(parts)

    def train(self, records: List[dict]):
        """Learn known patterns from training data."""
        for record in records:
            pattern = self._extract_pattern(record)
            self.known_patterns[pattern] += 1

            # Store vector for first occurrence
            if pattern not in self.pattern_vectors:
                pattern_dict = {}
                for field in self.pattern_fields:
                    value = record.get(field, "")
                    if field in self.normalizers:
                        value = self.normalizers[field](value)
                    pattern_dict[field] = value
                self.pattern_vectors[pattern] = self.encoder.encode_data(pattern_dict)

            self.total_trained += 1

        print(f"Trained on {self.total_trained} records")
        print(f"Discovered {len(self.known_patterns)} unique patterns")

    def check(self, record: dict, fuzzy_threshold: float = 0.9) -> PatternMatch:
        """
        Check if a record matches any known pattern.

        First tries exact match. If no exact match, tries fuzzy match
        against known pattern vectors.
        """
        self.total_checked += 1

        pattern = self._extract_pattern(record)

        # Exact match
        if pattern in self.known_patterns:
            return PatternMatch(
                input_data=record,
                normalized_pattern=pattern,
                matched=True,
                best_match=pattern,
                similarity=1.0,
            )

        # Fuzzy match against known patterns
        pattern_dict = {}
        for field in self.pattern_fields:
            value = record.get(field, "")
            if field in self.normalizers:
                value = self.normalizers[field](value)
            pattern_dict[field] = value

        input_vec = self.encoder.encode_data(pattern_dict)

        best_sim = -1.0
        best_pattern = None

        for known_pattern, known_vec in self.pattern_vectors.items():
            sim = cosine(input_vec, known_vec)
            if sim > best_sim:
                best_sim = sim
                best_pattern = known_pattern

        # Even with fuzzy matching, if similarity is low, it's still a mismatch
        # But we can use the similarity to understand HOW different it is
        matched = best_sim >= fuzzy_threshold

        if not matched:
            self.anomalies_found += 1

        return PatternMatch(
            input_data=record,
            normalized_pattern=pattern,
            matched=matched,
            best_match=best_pattern,
            similarity=best_sim,
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "patterns_known": len(self.known_patterns),
            "total_trained": self.total_trained,
            "total_checked": self.total_checked,
            "anomalies_found": self.anomalies_found,
            "top_patterns": sorted(
                self.known_patterns.items(),
                key=lambda x: -x[1]
            )[:10],
        }


def main():
    print("=" * 70)
    print("Challenge 010-005: Pattern Membership for Anomaly Detection")
    print("=" * 70)

    # Setup
    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    # Create detector
    detector = PatternMembershipDetector(
        encoder=encoder,
        pattern_fields=["method", "path"],  # Only these fields define the pattern
        normalizers={"path": normalize_path},
    )

    # Generate training data
    print("\n--- Generating Training Data ---")
    gen = RealisticDataGenerator(seed=42, cardinality=10000)

    train_records = []
    for i in range(1000):
        record, _, _ = gen.generate_record("api_request", i)
        train_records.append(record)

    # Train
    print("\n--- Training ---")
    detector.train(train_records)

    # Show discovered patterns
    print("\n--- Discovered Patterns ---")
    stats = detector.get_stats()
    for pattern, count in stats["top_patterns"]:
        print(f"  {count:4d}x  {pattern}")

    # Test on normal data
    print("\n--- Testing on Normal Data ---")
    test_normal = []
    for i in range(1000, 1100):
        record, _, _ = gen.generate_record("api_request", i)
        test_normal.append(record)

    normal_results = [detector.check(r) for r in test_normal]
    normal_matched = sum(1 for r in normal_results if r.matched)

    print(f"Normal records tested: {len(test_normal)}")
    print(f"Matched known pattern: {normal_matched}")
    print(f"Unknown patterns (false positives): {len(test_normal) - normal_matched}")

    # Show some that didn't match
    unmatched = [r for r in normal_results if not r.matched]
    if unmatched:
        print("\nSample unmatched patterns:")
        for r in unmatched[:5]:
            print(f"  Pattern: {r.normalized_pattern}")
            print(f"  Best match: {r.best_match} (sim={r.similarity:.4f})")

    # Test on attacks
    print("\n--- Testing on Attack Data ---")
    attacks = [
        # Path traversal
        {"method": "GET", "path": "/api/../../../etc/passwd", "status_code": 200},
        {"method": "GET", "path": "/api/users/../../admin/config", "status_code": 200},
        # SQL injection
        {"method": "GET", "path": "/api/users/' OR 1=1--", "status_code": 200},
        {"method": "POST", "path": "/api/users/1; DROP TABLE users;--", "status_code": 200},
        # Command injection
        {"method": "GET", "path": "/api/exec?cmd=ls%20-la", "status_code": 200},
        # Unusual methods
        {"method": "TRACE", "path": "/api/users", "status_code": 200},
        {"method": "OPTIONS", "path": "/api/admin", "status_code": 200},
        # Encoded attacks
        {"method": "GET", "path": "/api/%2e%2e/%2e%2e/etc/passwd", "status_code": 200},
    ]

    print(f"\nResults for {len(attacks)} attack requests:")
    print("-" * 70)

    attack_results = []
    for attack in attacks:
        result = detector.check(attack)
        attack_results.append(result)

        status = "✓ DETECTED" if not result.matched else "✗ MISSED"
        method = attack.get("method", "")
        path = attack.get("path", "")[:40]
        print(f"{status} | {method:6} | {path}")
        if not result.matched:
            print(f"          Pattern: {result.normalized_pattern}")

    attacks_detected = sum(1 for r in attack_results if not r.matched)
    print(f"\nAttacks detected: {attacks_detected}/{len(attacks)}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    false_positives = len(test_normal) - normal_matched
    true_positives = attacks_detected

    print(f"""
Approach: Pattern Membership Testing

Key insight:
  VSA/HDC atoms are random and orthogonal - no semantic similarity.
  Instead of "how similar?", ask "does this match a known pattern?"

How it works:
  1. Extract pattern from record (method + normalized path)
  2. Check if pattern exists in known set
  3. Unknown pattern → ANOMALY

Results:
  Normal data:    {len(test_normal)} tested, {false_positives} false positives ({100*false_positives/len(test_normal):.1f}%)
  Attack data:    {len(attacks)} tested, {true_positives} detected ({100*true_positives/len(attacks):.1f}%)

Known patterns:  {len(detector.known_patterns)}

Key advantage:
  ZERO similarity computation needed for exact matching!
  Just a hash lookup: pattern in known_patterns → O(1)

  VSA vectors are only needed for:
  - Building the pattern set (encode once)
  - Fuzzy matching for similar-but-not-exact patterns
""")


if __name__ == "__main__":
    main()
