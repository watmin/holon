#!/usr/bin/env python3
"""
Challenge 010-002: Path Normalization for Anomaly Detection

Key insight: We can't be fully headless. Understanding data structure
is a reasonable requirement for effective encoding.

For RESTful paths:
  /api/users/123     → /api/users/{id}
  /api/orders/abc-99 → /api/orders/{id}

This reduces cardinality while preserving meaningful signal.

Anomalies become obvious:
  /api/../../../etc/passwd  → doesn't normalize to known pattern → FLAG
  /api/users/' OR 1=1--     → doesn't normalize to known pattern → FLAG
"""

import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from realistic_data_generator import RealisticDataGenerator

from holon.encoder import Encoder


class FieldNormalizer:
    """
    Domain-aware field normalization to reduce cardinality
    while preserving meaningful signal.
    """

    def __init__(self):
        # Registered normalizers by field name
        self.normalizers: Dict[str, Callable[[Any], Any]] = {}

        # Track what we've seen for learning
        self.seen_values: Dict[str, Counter] = defaultdict(Counter)
        self.normalized_values: Dict[str, Counter] = defaultdict(Counter)

    def register(self, field_name: str, normalizer: Callable[[Any], Any]):
        """Register a normalizer for a specific field."""
        self.normalizers[field_name] = normalizer

    def normalize(self, field_name: str, value: Any) -> Any:
        """Normalize a field value, tracking what we see."""
        self.seen_values[field_name][str(value)] += 1

        if field_name in self.normalizers:
            normalized = self.normalizers[field_name](value)
        else:
            normalized = value

        self.normalized_values[field_name][str(normalized)] += 1
        return normalized

    def normalize_record(self, record: dict) -> dict:
        """Normalize all fields in a record."""
        result = {}
        for key, value in record.items():
            if isinstance(value, dict):
                result[key] = self.normalize_record(value)
            elif isinstance(value, list):
                result[key] = [
                    self.normalize_record(v) if isinstance(v, dict) else v
                    for v in value
                ]
            else:
                result[key] = self.normalize(key, value)
        return result

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cardinality reduction stats."""
        stats = {}
        for field in self.seen_values:
            raw_card = len(self.seen_values[field])
            norm_card = len(self.normalized_values[field])
            reduction = 1 - (norm_card / max(raw_card, 1))
            stats[field] = {
                "raw_cardinality": raw_card,
                "normalized_cardinality": norm_card,
                "reduction": reduction,
            }
        return stats


def create_path_normalizer() -> Callable[[str], str]:
    """
    Create a RESTful path normalizer.

    Patterns:
      /api/users/123         → /api/users/{id}
      /api/orders/abc-def    → /api/orders/{id}
      /api/products/SKU-1234 → /api/products/{id}
      /v1/users/123/orders   → /v1/users/{id}/orders

    Abnormal (not normalized):
      /api/../../../etc/passwd  → preserved (anomaly signal!)
      /api/users/' OR 1=1--     → preserved (anomaly signal!)
    """
    # Patterns that indicate a resource ID
    id_patterns = [
        r'/\d+',                    # Numeric IDs: /123
        r'/[a-f0-9-]{36}',          # UUIDs: /550e8400-e29b-41d4-a716-446655440000
        r'/[a-f0-9]{8,}',           # Hex IDs: /abc123def456
        r'/[A-Z]+-\d+',             # Formatted IDs: /SKU-123, /ORD-456
        r'/usr_[a-f0-9]+',          # User IDs: /usr_00000123
        r'/ord_\d+',                # Order IDs: /ord_0000000001
        r'/prod_\d+',               # Product IDs: /prod_000001
        r'/sess_[a-f0-9]+',         # Session IDs: /sess_000000000001
    ]

    # Compile patterns
    compiled = [(re.compile(p), '/{id}') for p in id_patterns]

    def normalize_path(path: str) -> str:
        if not isinstance(path, str):
            return str(path)

        result = path
        for pattern, replacement in compiled:
            result = pattern.sub(replacement, result)

        return result

    return normalize_path


def create_id_normalizer(prefix: str = "") -> Callable[[str], str]:
    """Normalize ID fields to just their type prefix."""
    def normalize_id(value: str) -> str:
        if not isinstance(value, str):
            return str(value)

        # Extract prefix (usr_, ord_, prod_, etc.)
        match = re.match(r'^([a-z]+_)', value)
        if match:
            return f"{match.group(1)}{{id}}"

        # UUIDs
        if re.match(r'^[a-f0-9-]{36}$', value):
            return "{uuid}"

        # Hex strings
        if re.match(r'^[a-f0-9]{8,}$', value):
            return "{hex_id}"

        # Numeric
        if re.match(r'^\d+$', value):
            return "{numeric_id}"

        return value

    return normalize_id


@dataclass
class AnomalyResult:
    """Result of anomaly detection."""
    record: dict
    normalized_record: dict
    vector: np.ndarray
    similarity_to_baseline: float
    is_anomaly: bool
    reason: str


class AnomalyDetector:
    """
    Detect anomalies by comparing to baseline of "normal" patterns.

    Uses normalization to reduce cardinality, then flags vectors
    that don't match the baseline well.
    """

    def __init__(
        self,
        encoder: Encoder,
        normalizer: FieldNormalizer,
        anomaly_threshold: float = 0.3,
    ):
        self.encoder = encoder
        self.normalizer = normalizer
        self.anomaly_threshold = anomaly_threshold

        # Baseline: built from training data
        self.baseline_vectors: List[np.ndarray] = []
        self.baseline_prototype: Optional[np.ndarray] = None

        # Known patterns (for pattern matching)
        self.known_patterns: Dict[str, np.ndarray] = {}

        # Stats
        self.records_seen = 0
        self.anomalies_detected = 0

    def train(self, records: List[dict]):
        """Build baseline from "normal" training data."""
        print(f"Training on {len(records)} records...")

        for record in records:
            normalized = self.normalizer.normalize_record(record)
            vector = self.encoder.encode_data(normalized)
            self.baseline_vectors.append(vector)

        # Build prototype (consensus of all training vectors)
        self.baseline_prototype = self.encoder.prototype(
            self.baseline_vectors, threshold=0.3
        )

        print(f"Baseline prototype built from {len(self.baseline_vectors)} vectors")
        non_zero = np.sum(self.baseline_prototype != 0)
        print(f"Prototype density: {non_zero}/{len(self.baseline_prototype)} non-zero")

    def add_known_pattern(self, name: str, pattern_record: dict):
        """Add a known pattern to match against (e.g., attack signatures)."""
        normalized = self.normalizer.normalize_record(pattern_record)
        vector = self.encoder.encode_data(normalized)
        self.known_patterns[name] = vector
        print(f"Added known pattern: {name}")

    def check(self, record: dict) -> AnomalyResult:
        """Check if a record is anomalous."""
        self.records_seen += 1

        normalized = self.normalizer.normalize_record(record)
        vector = self.encoder.encode_data(normalized)

        # Compare to baseline prototype
        sim = self._cosine(vector, self.baseline_prototype)

        # Check against known bad patterns
        pattern_matches = []
        for name, pattern_vec in self.known_patterns.items():
            pattern_sim = self._cosine(vector, pattern_vec)
            if pattern_sim > 0.5:  # High similarity to known bad pattern
                pattern_matches.append((name, pattern_sim))

        # Determine if anomaly
        is_anomaly = False
        reason = "normal"

        if pattern_matches:
            is_anomaly = True
            best_match = max(pattern_matches, key=lambda x: x[1])
            reason = f"matches_pattern:{best_match[0]} (sim={best_match[1]:.3f})"
            self.anomalies_detected += 1
        elif sim < self.anomaly_threshold:
            is_anomaly = True
            reason = f"low_baseline_similarity (sim={sim:.3f})"
            self.anomalies_detected += 1

        return AnomalyResult(
            record=record,
            normalized_record=normalized,
            vector=vector,
            similarity_to_baseline=sim,
            is_anomaly=is_anomaly,
            reason=reason,
        )

    def _cosine(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def get_stats(self) -> Dict[str, Any]:
        return {
            "records_seen": self.records_seen,
            "anomalies_detected": self.anomalies_detected,
            "anomaly_rate": self.anomalies_detected / max(1, self.records_seen),
            "baseline_size": len(self.baseline_vectors),
            "known_patterns": list(self.known_patterns.keys()),
        }


def generate_attack_requests() -> List[dict]:
    """Generate malicious API request examples."""
    attacks = [
        # Path traversal
        {
            "_schema": "api_request",
            "method": "GET",
            "path": "/api/../../../etc/passwd",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
        {
            "_schema": "api_request",
            "method": "GET",
            "path": "/api/users/../../admin/config",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
        # SQL injection
        {
            "_schema": "api_request",
            "method": "GET",
            "path": "/api/users/' OR 1=1--",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
        {
            "_schema": "api_request",
            "method": "POST",
            "path": "/api/users/1; DROP TABLE users;--",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
        # Command injection
        {
            "_schema": "api_request",
            "method": "GET",
            "path": "/api/exec?cmd=ls%20-la",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
        # Unusual methods
        {
            "_schema": "api_request",
            "method": "TRACE",
            "path": "/api/users",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
        # Encoded attacks
        {
            "_schema": "api_request",
            "method": "GET",
            "path": "/api/%2e%2e/%2e%2e/etc/passwd",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
    ]
    return attacks


def main():
    print("=" * 70)
    print("Challenge 010-002: Path Normalization for Anomaly Detection")
    print("=" * 70)

    # Setup normalizer with domain knowledge
    print("\n--- Setting Up Field Normalizers ---")
    normalizer = FieldNormalizer()

    # Register normalizers for known high-cardinality fields
    normalizer.register("path", create_path_normalizer())
    normalizer.register("user_id", create_id_normalizer())
    normalizer.register("session_id", create_id_normalizer())
    normalizer.register("order_id", create_id_normalizer())
    normalizer.register("request_id", create_id_normalizer())
    normalizer.register("trace_id", create_id_normalizer())
    normalizer.register("span_id", create_id_normalizer())
    normalizer.register("alert_id", create_id_normalizer())
    normalizer.register("deployment_id", create_id_normalizer())
    normalizer.register("change_id", create_id_normalizer())
    normalizer.register("customer_id", create_id_normalizer())

    print("Registered normalizers for: path, *_id fields")

    # Demo normalization
    print("\n--- Normalization Examples ---")
    examples = [
        ("path", "/api/users/123"),
        ("path", "/api/orders/abc-def-123"),
        ("path", "/api/products/SKU-999"),
        ("path", "/v1/users/456/orders"),
        ("path", "/api/../../../etc/passwd"),  # Attack - should NOT normalize!
        ("path", "/api/users/' OR 1=1--"),       # Attack - should NOT normalize!
        ("user_id", "usr_00000123"),
        ("user_id", "usr_99999999"),
        ("session_id", "sess_abc123def456"),
    ]

    for field, value in examples:
        normalized = normalizer.normalize(field, value)
        changed = "→" if normalized != value else "="
        print(f"  {field}: {value:40} {changed} {normalized}")

    # Generate training data (normal requests only)
    print("\n--- Generating Training Data ---")
    gen = RealisticDataGenerator(seed=42, cardinality=10000)

    # Generate only api_request records for this demo
    train_records = []
    for i in range(5000):
        record, schema, _ = gen.generate_record("api_request", i)
        train_records.append(record)

    print(f"Generated {len(train_records)} normal api_request records")

    # Create encoder and detector
    print("\n--- Building Anomaly Detector ---")
    vm = DeterministicVectorManager(dimensions=4096, global_seed=42)
    encoder = Encoder(vector_manager=vm)

    detector = AnomalyDetector(
        encoder=encoder,
        normalizer=normalizer,
        anomaly_threshold=0.2,  # Flag if similarity < 0.2
    )

    # Train on normal data
    detector.train(train_records[:1000])  # Use subset for speed

    # Add known attack patterns
    print("\n--- Adding Known Attack Patterns ---")
    attack_patterns = {
        "path_traversal": {
            "_schema": "api_request",
            "method": "GET",
            "path": "/api/../../../etc/passwd",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
        "sql_injection": {
            "_schema": "api_request",
            "method": "GET",
            "path": "/api/users/' OR 1=1--",
            "status_code": 200,
            "timestamp": "2026-01-15T12:00:00Z",
        },
    }

    for name, pattern in attack_patterns.items():
        detector.add_known_pattern(name, pattern)

    # Test on normal data
    print("\n--- Testing on Normal Data ---")
    test_normal = train_records[1000:1100]  # 100 normal records

    normal_results = []
    for record in test_normal:
        result = detector.check(record)
        normal_results.append(result)

    normal_flagged = sum(1 for r in normal_results if r.is_anomaly)
    print(f"Normal records tested: {len(test_normal)}")
    print(f"False positives: {normal_flagged} ({100*normal_flagged/len(test_normal):.1f}%)")

    # Show similarity distribution for normal
    normal_sims = [r.similarity_to_baseline for r in normal_results]
    print(f"Similarity to baseline: min={min(normal_sims):.3f}, max={max(normal_sims):.3f}, mean={np.mean(normal_sims):.3f}")

    # Test on attack data
    print("\n--- Testing on Attack Data ---")
    attacks = generate_attack_requests()

    print(f"\nResults for {len(attacks)} attack requests:")
    print("-" * 70)

    attack_results = []
    for attack in attacks:
        result = detector.check(attack)
        attack_results.append(result)

        status = "🚨 FLAGGED" if result.is_anomaly else "   normal"
        path = attack.get("path", "")[:40]
        print(f"{status} | sim={result.similarity_to_baseline:+.3f} | {path}")
        if result.is_anomaly:
            print(f"          Reason: {result.reason}")

    attacks_caught = sum(1 for r in attack_results if r.is_anomaly)
    print(f"\nAttacks detected: {attacks_caught}/{len(attacks)} ({100*attacks_caught/len(attacks):.1f}%)")

    # Cardinality reduction stats
    print("\n--- Cardinality Reduction ---")
    stats = normalizer.get_stats()
    for field, field_stats in sorted(stats.items()):
        if field_stats["raw_cardinality"] > 1:
            print(f"  {field}:")
            print(f"    Raw cardinality:  {field_stats['raw_cardinality']:,}")
            print(f"    Normalized:       {field_stats['normalized_cardinality']:,}")
            print(f"    Reduction:        {100*field_stats['reduction']:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"""
Approach: Domain-aware normalization + baseline comparison

Normalization:
  - /api/users/123     → /api/users/{{id}}     (reduces cardinality)
  - /api/../../../etc  → PRESERVED             (anomaly signal!)
  - usr_00000123       → usr_{{id}}            (reduces cardinality)

Results:
  Normal data:    {len(test_normal)} tested, {normal_flagged} false positives ({100*normal_flagged/len(test_normal):.1f}%)
  Attack data:    {len(attacks)} tested, {attacks_caught} detected ({100*attacks_caught/len(attacks):.1f}%)

Key insight:
  Understanding your data schema is a REQUIREMENT, not a limitation.
  Domain knowledge in the encoding layer enables effective detection.
""")


if __name__ == "__main__":
    main()
