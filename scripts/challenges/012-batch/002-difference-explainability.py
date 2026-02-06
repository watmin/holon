#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 002: Difference Vector Explainability
=============================================================================

Building on 001's significance detection, use holon's vector primitives to
EXPLAIN what changed:

1. difference(prior, recent) → "what's new/different"
2. resonance(packet_vec, difference) → "is this packet part of what changed?"
3. For each field-value: similarity(field_vec, difference) → importance score

This gives us actionable explanations:
  "The anomaly is driven by {src_port: 53} with importance 0.87"
  "Secondary contributors: {protocol: UDP} (0.65), {payload_size: large} (0.42)"

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/002-difference-explainability.py
"""

import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from enum import Enum
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
DECAY = 0.995
WARMUP_PACKETS = 200
WINDOW_SIZE = 50


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class FieldContribution:
    """How much a field-value contributes to the anomaly."""
    field: str
    value: Any
    importance: float  # Similarity to difference vector
    explanation: str


@dataclass
class DetectionResult:
    packet_num: int
    is_anomalous: bool
    confidence: float

    # Similarity scores
    prior_similarity: float
    recent_similarity: float

    # Vector-derived
    divergence: float  # How different is recent from prior
    difference_magnitude: float  # How large is the difference vector

    # Explainability: what's driving this anomaly?
    top_contributors: List[FieldContribution]

    # Simple explanation
    explanation: str


# =============================================================================
# DIFFERENCE-BASED DETECTOR
# =============================================================================

class DifferenceDetector:
    """
    Use vector difference operations for detection and explanation.

    Core idea:
    - difference(prior, recent) captures "what changed"
    - For each field-value in current packet, check similarity to difference
    - High similarity = this field-value is part of what's new/different
    """

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Accumulators
        self.prior_accum = self.encoder.create_accumulator()
        self.recent_accum = self.encoder.create_accumulator()
        self.prior_count = 0

        # Cached normalized vectors
        self._prior_norm = None
        self._recent_norm = None
        self._difference = None

        # Field-value cache for explanation
        self._field_value_vecs = {}

        # Anomaly state
        self.anomaly_history = deque(maxlen=WINDOW_SIZE)

    def _encode_field_value(self, field: str, value: Any) -> np.ndarray:
        """Encode a single field-value pair."""
        key = (field, str(value))
        if key not in self._field_value_vecs:
            structure = {field: value}
            self._field_value_vecs[key] = self.encoder.encode_data(structure)
        return self._field_value_vecs[key]

    def _update_cached_vectors(self):
        """Update cached normalized vectors and difference."""
        self._prior_norm = self.encoder.normalize_accumulator(self.prior_accum)
        self._recent_norm = self.encoder.normalize_accumulator(self.recent_accum)

        # Use holon's difference primitive
        self._difference = self.store.difference(self._prior_norm, self._recent_norm)

    def _explain_packet(self, packet: dict) -> List[FieldContribution]:
        """
        Explain which field-values in this packet contribute to the anomaly.

        Uses the difference vector: fields with high similarity to the difference
        are "new/unusual" components.
        """
        if self._difference is None:
            return []

        contributions = []

        # For each field-value in the packet, check similarity to difference
        field_values = [
            ("protocol", packet.get("protocol")),
            ("src_port", packet.get("src_port")),
            ("dst_port", packet.get("dst_port")),
            ("tcp_flags", packet.get("flags")),
            ("icmp_type", packet.get("icmp_type")),
            ("payload_size", self._bucket_size(packet.get("payload_size", 0))),
        ]

        for field, value in field_values:
            if value is None:
                continue

            field_vec = self._encode_field_value(field, value)

            # How similar is this field-value to what changed?
            importance = cosine_similarity(field_vec, self._difference)

            # Also check similarity to prior (high = normal, low = novel)
            prior_sim = cosine_similarity(field_vec, self._prior_norm)

            # Importance is high if: similar to difference AND dissimilar from prior
            adjusted_importance = importance * (1.0 - prior_sim)

            if adjusted_importance > 0.05:  # Threshold for relevance
                if prior_sim < 0.3:
                    explanation = f"{field}={value} is NOVEL (prior_sim={prior_sim:.2f})"
                elif importance > 0.3:
                    explanation = f"{field}={value} is part of the SHIFT"
                else:
                    explanation = f"{field}={value} contributes slightly"

                contributions.append(FieldContribution(
                    field=field,
                    value=value,
                    importance=adjusted_importance,
                    explanation=explanation,
                ))

        # Sort by importance
        contributions.sort(key=lambda c: c.importance, reverse=True)
        return contributions[:5]  # Top 5

    def _bucket_size(self, size: int) -> str:
        if size == 0:
            return "none"
        elif size < 64:
            return "tiny"
        elif size < 256:
            return "small"
        elif size < 1024:
            return "medium"
        else:
            return "large"

    def process(self, packet: dict) -> DetectionResult:
        """Process a packet with difference-based detection."""
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        # Encode packet
        packet_vec = self.encoder.encode_data(packet)

        if is_warmup:
            self.prior_accum = self.encoder.accumulate(self.prior_accum, packet_vec)
            self.prior_count += 1

            if self.packet_count == self.warmup_packets:
                self.warmup_complete = True
                self.recent_accum = self.prior_accum.copy()
                self._update_cached_vectors()

            return DetectionResult(
                packet_num=self.packet_count,
                is_anomalous=False,
                confidence=0.0,
                prior_similarity=0.5,
                recent_similarity=0.5,
                divergence=0.0,
                difference_magnitude=0.0,
                top_contributors=[],
                explanation="Warming up...",
            )

        # Update recent with decay
        self.recent_accum = DECAY * self.recent_accum + packet_vec.astype(np.float64)

        # Update cached vectors periodically (every 10 packets for efficiency)
        if self.packet_count % 10 == 0:
            self._update_cached_vectors()

        # Compute similarities
        prior_sim = cosine_similarity(packet_vec, self._prior_norm)
        recent_sim = cosine_similarity(packet_vec, self._recent_norm)

        # Divergence: how different is recent from prior
        divergence = 1.0 - cosine_similarity(self._prior_norm, self._recent_norm)

        # Difference magnitude
        diff_magnitude = float(np.linalg.norm(self._difference))

        # Anomaly detection:
        # 1. Low similarity to prior (packet is unusual vs baseline)
        # 2. High divergence (traffic pattern has shifted)
        is_anomalous = prior_sim < 0.4 or divergence > 0.30

        # Track anomaly state
        self.anomaly_history.append(is_anomalous)

        # Require sustained anomaly (reduce noise)
        recent_anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = is_anomalous and recent_anomaly_rate > 0.5

        # Confidence based on how far from normal
        confidence = max(0, (0.5 - prior_sim) / 0.5) if prior_sim < 0.5 else 0
        confidence = max(confidence, divergence)

        # Explain the anomaly
        contributors = self._explain_packet(packet) if is_anomalous else []

        # Build explanation
        if not is_anomalous:
            explanation = "Normal traffic"
        elif contributors:
            top = contributors[0]
            explanation = f"Anomaly: {top.explanation}"
            if len(contributors) > 1:
                explanation += f" (+ {len(contributors)-1} other factors)"
        else:
            explanation = f"Anomaly: traffic divergence={divergence:.0%}"

        return DetectionResult(
            packet_num=self.packet_count,
            is_anomalous=is_anomalous,
            confidence=confidence,
            prior_similarity=prior_sim,
            recent_similarity=recent_sim,
            divergence=divergence,
            difference_magnitude=diff_magnitude,
            top_contributors=contributors,
            explanation=explanation,
        )


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def generate_normal_packet(rng: random.Random) -> dict:
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.8, 0.18, 0.02])[0]

    if proto == "TCP":
        return {
            "protocol": "TCP",
            "src_port": rng.randint(49152, 65535),
            "dst_port": rng.choice([80, 443, 8080, 22]),
            "flags": rng.choices(["PA", "A", "SA", "S"], weights=[0.4, 0.3, 0.2, 0.1])[0],
            "payload_size": rng.randint(0, 1500),
        }
    elif proto == "UDP":
        return {
            "protocol": "UDP",
            "src_port": rng.randint(49152, 65535),
            "dst_port": rng.choice([53, 443, 123]),
            "payload_size": rng.randint(20, 512),
        }
    else:
        return {
            "protocol": "ICMP",
            "icmp_type": rng.choice([0, 8]),
            "payload_size": 64,
        }


def generate_syn_flood(rng: random.Random) -> dict:
    return {
        "protocol": "TCP",
        "src_port": rng.randint(1, 65535),
        "dst_port": 80,
        "flags": "S",
        "payload_size": 0,
    }


def generate_dns_reflection(rng: random.Random) -> dict:
    return {
        "protocol": "UDP",
        "src_port": 53,
        "dst_port": rng.randint(49152, 65535),
        "payload_size": rng.randint(256, 4096),
    }


def generate_ntp_amplification(rng: random.Random) -> dict:
    return {
        "protocol": "UDP",
        "src_port": 123,
        "dst_port": rng.randint(49152, 65535),
        "payload_size": rng.randint(300, 500),
    }


def generate_port_scan(rng: random.Random) -> dict:
    return {
        "protocol": "TCP",
        "src_port": 45000,
        "dst_port": rng.randint(1, 1024),
        "flags": "S",
        "payload_size": 0,
    }


def generate_icmp_flood(rng: random.Random) -> dict:
    return {
        "protocol": "ICMP",
        "icmp_type": 8,
        "payload_size": 64,
    }


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_attack(
    name: str,
    attack_generator,
    attack_fraction: float = 0.9,
    normal_count: int = 500,
    attack_count: int = 500,
):
    print(f"\n{'='*70}")
    print(f"Attack: {name}")
    print(f"{'='*70}")

    detector = DifferenceDetector(warmup_packets=200)
    rng = random.Random(42)

    stream = []
    for _ in range(normal_count):
        stream.append((generate_normal_packet(rng), "normal"))
    for _ in range(attack_count):
        if rng.random() < attack_fraction:
            stream.append((attack_generator(rng), "attack"))
        else:
            stream.append((generate_normal_packet(rng), "normal"))
    for _ in range(100):
        stream.append((generate_normal_packet(rng), "normal"))

    results = []
    first_detection = None

    for i, (packet, label) in enumerate(stream):
        result = detector.process(packet)
        results.append((result, label))

        if result.is_anomalous and first_detection is None and label == "attack":
            first_detection = i

    # Metrics
    post_warmup = [(r, l) for r, l in results if r.packet_num > 200]

    tp = sum(1 for r, l in post_warmup if l == "attack" and r.is_anomalous)
    fp = sum(1 for r, l in post_warmup if l == "normal" and r.is_anomalous)
    fn = sum(1 for r, l in post_warmup if l == "attack" and not r.is_anomalous)
    tn = sum(1 for r, l in post_warmup if l == "normal" and not r.is_anomalous)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\nMetrics:")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.3f}")

    if first_detection:
        print(f"  First detection at packet {first_detection} (delay: {first_detection - 500})")

    # Show sample explanations
    attack_results = [(r, l) for r, l in post_warmup if l == "attack" and r.is_anomalous]
    if attack_results:
        print(f"\nSample explanations (difference-based):")
        for r, _ in attack_results[:3]:
            print(f"  [{r.packet_num}] {r.explanation}")
            if r.top_contributors:
                for c in r.top_contributors[:3]:
                    print(f"    → {c.field}={c.value}: importance={c.importance:.2f}")

    return {"attack": name, "f1": f1, "precision": precision, "recall": recall}


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 002: Difference Vector Explainability")
    print("="*70)
    print("""
    Using holon's vector primitives for explanation:

    1. difference(prior, recent) → captures "what changed"
    2. For each field-value: similarity(field_vec, difference)
       → High similarity = this field-value is part of what's new

    This gives us:
      "src_port=53 is NOVEL (importance=0.87)"
      "protocol=UDP contributes (importance=0.42)"
    """)

    results = []

    results.append(evaluate_attack("SYN Flood", generate_syn_flood))
    results.append(evaluate_attack("DNS Reflection", generate_dns_reflection))
    results.append(evaluate_attack("NTP Amplification", generate_ntp_amplification))
    results.append(evaluate_attack("Port Scan", generate_port_scan))
    results.append(evaluate_attack("ICMP Flood", generate_icmp_flood))

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Attack':<20} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-"*50)
    for r in results:
        print(f"{r['attack']:<20} {r['f1']:>8.3f} {r['precision']:>10.1%} {r['recall']:>8.1%}")

    avg_f1 = np.mean([r['f1'] for r in results])
    print("-"*50)
    print(f"{'Average':<20} {avg_f1:>8.3f}")

    print("""

VECTOR OPERATIONS USED:
=======================

1. difference(prior, recent)
   - Computes: recent - prior (thresholded to bipolar)
   - Returns a vector representing "what changed"

2. similarity(field_vec, difference)
   - How much does this field-value contribute to the change?
   - High = this field-value is part of what's new/unusual

3. similarity(packet_vec, prior)
   - How similar is this packet to normal baseline?
   - Low = packet is anomalous

4. Accumulators (frequency-preserving)
   - accumulate() preserves frequency information
   - normalize_accumulator() for queries

This approach uses holon's primitives instead of counters,
giving us:
- Vectorized significance detection
- Mathematically grounded explanations
- No hardcoded domain knowledge
    """)


if __name__ == "__main__":
    main()
