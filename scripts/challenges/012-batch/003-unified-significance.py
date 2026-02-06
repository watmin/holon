#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 003: Unified Significance Detector
=============================================================================

Combines the best of 001 (per-field tracking) and 002 (difference vectors):

1. Per-field accumulators track value distributions
2. difference(prior, recent) for explanations
3. Multiple detection signals:
   - Field divergence: any field's distribution shifted
   - Traffic divergence: overall pattern shifted
   - Novelty: new values appeared

4. Adaptive thresholds based on baseline variability

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/003-unified-significance.py
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
class FieldStats:
    """Per-field statistics for significance detection."""
    field: str
    divergence: float  # How different is recent from prior
    concentration: float  # How concentrated is recent
    concentration_change: float  # Change from prior concentration
    dominant_value: Any
    prior_dominant: Any
    novel_values: List[Any]
    is_significant: bool
    explanation: str


@dataclass
class PacketExplanation:
    """Explanation of why a packet is anomalous."""
    field: str
    value: Any
    importance: float
    reason: str


@dataclass
class DetectionResult:
    packet_num: int
    is_anomalous: bool
    confidence: float

    # Packet-level
    prior_similarity: float

    # Field-level stats
    field_stats: Dict[str, FieldStats]

    # Traffic-level
    traffic_divergence: float

    # Explanation
    packet_explanations: List[PacketExplanation]
    summary: str


# =============================================================================
# FIELD TRACKER (from 001, enhanced)
# =============================================================================

class FieldTracker:
    """Track per-field value distributions."""

    def __init__(self, field: str, encoder, dimensions: int):
        self.field = field
        self.encoder = encoder
        self.dimensions = dimensions

        # Accumulators
        self.prior_accum = encoder.create_accumulator()
        self.recent_accum = encoder.create_accumulator()

        # Value tracking
        self.prior_values = set()
        self.prior_counts = defaultdict(int)
        self.prior_total = 0

        self.recent_values = set()
        self.recent_counts = defaultdict(int)
        self.recent_total = 0

        # Frozen state
        self.frozen = False
        self.prior_concentration = 0.0
        self.prior_dominant = None

    def _encode(self, value: Any) -> np.ndarray:
        return self.encoder.encode_data({self.field: value})

    def observe(self, value: Any, is_warmup: bool):
        if value is None:
            return

        vec = self._encode(value)

        if is_warmup:
            self.prior_accum = self.encoder.accumulate(self.prior_accum, vec)
            self.prior_values.add(value)
            self.prior_counts[value] += 1
            self.prior_total += 1
        else:
            self.recent_accum = DECAY * self.recent_accum + vec.astype(np.float64)
            self.recent_values.add(value)
            self.recent_counts[value] += 1
            self.recent_total += 1

            # Periodic cleanup
            if self.recent_total > 200:
                for k in list(self.recent_counts.keys()):
                    self.recent_counts[k] = self.recent_counts[k] // 2
                    if self.recent_counts[k] == 0:
                        del self.recent_counts[k]
                self.recent_total = sum(self.recent_counts.values())

    def freeze(self):
        self.frozen = True

        if self.prior_total > 0 and self.prior_counts:
            dom = max(self.prior_counts.items(), key=lambda x: x[1])
            self.prior_dominant = dom[0]
            self.prior_concentration = dom[1] / self.prior_total

        self.recent_accum = self.prior_accum.copy()
        self.recent_values = self.prior_values.copy()
        self.recent_counts = self.prior_counts.copy()
        self.recent_total = self.prior_total

    def compute_stats(self) -> FieldStats:
        if not self.frozen:
            return FieldStats(
                field=self.field,
                divergence=0.0,
                concentration=0.0,
                concentration_change=0.0,
                dominant_value=None,
                prior_dominant=None,
                novel_values=[],
                is_significant=False,
                explanation="Warming up"
            )

        # Divergence
        prior_norm = self.encoder.normalize_accumulator(self.prior_accum)
        recent_norm = self.encoder.normalize_accumulator(self.recent_accum)
        divergence = 1.0 - cosine_similarity(prior_norm, recent_norm)

        # Concentration
        if self.recent_total > 0 and self.recent_counts:
            dom = max(self.recent_counts.items(), key=lambda x: x[1])
            dominant_value = dom[0]
            concentration = dom[1] / self.recent_total
        else:
            dominant_value = None
            concentration = 0.0

        concentration_change = concentration - self.prior_concentration

        # Novel values
        novel_values = list(self.recent_values - self.prior_values)

        # Is this field significant?
        is_significant = False
        explanation = f"{self.field}: normal"

        # Significant if: high divergence + (new dominant value OR concentration increase OR novelty)
        if divergence > 0.20:
            if dominant_value != self.prior_dominant and concentration > 0.5:
                is_significant = True
                explanation = f"{self.field} → {dominant_value} ({concentration:.0%}, was {self.prior_concentration:.0%} on {self.prior_dominant})"
            elif concentration_change > 0.20:
                is_significant = True
                explanation = f"{self.field} concentrated {self.prior_concentration:.0%}→{concentration:.0%}"
            elif len(novel_values) > 2:
                is_significant = True
                explanation = f"{self.field} has {len(novel_values)} novel values"

        return FieldStats(
            field=self.field,
            divergence=divergence,
            concentration=concentration,
            concentration_change=concentration_change,
            dominant_value=dominant_value,
            prior_dominant=self.prior_dominant,
            novel_values=novel_values[:5],
            is_significant=is_significant,
            explanation=explanation,
        )


# =============================================================================
# UNIFIED DETECTOR
# =============================================================================

class UnifiedSignificanceDetector:
    """
    Combines per-field tracking with vector difference explanations.
    """

    MONITORED_FIELDS = [
        "protocol", "src_port", "dst_port", "tcp_flags", "icmp_type", "payload_size"
    ]

    def __init__(self, warmup: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.warmup = warmup
        self.packet_count = 0
        self.warmup_complete = False

        # Per-field trackers
        self.field_trackers = {
            f: FieldTracker(f, self.encoder, DIMENSIONS)
            for f in self.MONITORED_FIELDS
        }

        # Overall traffic
        self.prior_traffic = self.encoder.create_accumulator()
        self.recent_traffic = self.encoder.create_accumulator()

        # Cached vectors
        self._prior_norm = None
        self._recent_norm = None
        self._difference = None

        # Anomaly smoothing
        self.anomaly_history = deque(maxlen=30)

        # Baseline variability (for adaptive thresholds)
        self.baseline_sims = []
        self.baseline_mean = 0.7
        self.baseline_std = 0.1

    def _extract_fields(self, packet: dict) -> Dict[str, Any]:
        """Extract monitored fields from packet."""
        fields = {}

        fields["protocol"] = packet.get("protocol")
        fields["src_port"] = packet.get("src_port")
        fields["dst_port"] = packet.get("dst_port")

        if packet.get("protocol") == "TCP":
            fields["tcp_flags"] = packet.get("flags")

        if packet.get("protocol") == "ICMP":
            fields["icmp_type"] = packet.get("icmp_type")

        size = packet.get("payload_size", 0)
        fields["payload_size"] = "none" if size == 0 else "small" if size < 256 else "large"

        return fields

    def _update_caches(self):
        """Update cached normalized vectors."""
        self._prior_norm = self.encoder.normalize_accumulator(self.prior_traffic)
        self._recent_norm = self.encoder.normalize_accumulator(self.recent_traffic)
        self._difference = self.store.difference(self._prior_norm, self._recent_norm)

    def _explain_packet(self, packet: dict, fields: Dict[str, Any]) -> List[PacketExplanation]:
        """Explain which fields contribute most to anomaly."""
        if self._difference is None:
            return []

        explanations = []

        for field, value in fields.items():
            if value is None:
                continue

            field_vec = self.encoder.encode_data({field: value})

            # Importance = similarity to difference vector
            importance = cosine_similarity(field_vec, self._difference)

            # Prior similarity (low = novel)
            prior_sim = cosine_similarity(field_vec, self._prior_norm)

            # Combine: important if in difference AND not in prior
            adjusted = importance * max(0, 0.5 - prior_sim)

            if adjusted > 0.02:
                reason = "novel" if prior_sim < 0.2 else "shifted"
                explanations.append(PacketExplanation(
                    field=field,
                    value=value,
                    importance=adjusted,
                    reason=reason,
                ))

        explanations.sort(key=lambda e: e.importance, reverse=True)
        return explanations[:5]

    def process(self, packet: dict) -> DetectionResult:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup

        # Extract fields
        fields = self._extract_fields(packet)

        # Observe per-field
        for field, value in fields.items():
            if field in self.field_trackers:
                self.field_trackers[field].observe(value, is_warmup)

        # Encode packet
        packet_vec = self.encoder.encode_data(packet)

        if is_warmup:
            self.prior_traffic = self.encoder.accumulate(self.prior_traffic, packet_vec)

            if self.packet_count == self.warmup:
                self.warmup_complete = True
                for t in self.field_trackers.values():
                    t.freeze()
                self.recent_traffic = self.prior_traffic.copy()
                self._update_caches()

                # Compute baseline variability
                self.baseline_mean = np.mean(self.baseline_sims) if self.baseline_sims else 0.7
                self.baseline_std = np.std(self.baseline_sims) if self.baseline_sims else 0.1
            else:
                # Track baseline similarity variability
                if self.packet_count > 50:
                    prior_norm = self.encoder.normalize_accumulator(self.prior_traffic)
                    sim = cosine_similarity(packet_vec, prior_norm)
                    self.baseline_sims.append(sim)

            return DetectionResult(
                packet_num=self.packet_count,
                is_anomalous=False,
                confidence=0.0,
                prior_similarity=0.5,
                field_stats={},
                traffic_divergence=0.0,
                packet_explanations=[],
                summary="Warming up..."
            )

        # Update recent
        self.recent_traffic = DECAY * self.recent_traffic + packet_vec.astype(np.float64)

        # Update caches periodically
        if self.packet_count % 10 == 0:
            self._update_caches()

        # Compute stats
        prior_sim = cosine_similarity(packet_vec, self._prior_norm)
        traffic_div = 1.0 - cosine_similarity(self._prior_norm, self._recent_norm)

        # Per-field stats
        field_stats = {f: t.compute_stats() for f, t in self.field_trackers.items()}

        # Significant fields
        sig_fields = [s for s in field_stats.values() if s.is_significant]

        # Adaptive threshold based on baseline variability
        # Anomaly if packet is > 2 std below baseline mean similarity
        sim_threshold = self.baseline_mean - 2 * self.baseline_std

        # Detection logic
        is_anomalous = (
            prior_sim < sim_threshold or  # Packet is unusual
            traffic_div > 0.30 or  # Traffic shifted
            len(sig_fields) >= 2  # Multiple fields changed
        )

        # Smooth with history
        self.anomaly_history.append(is_anomalous)
        anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = is_anomalous and anomaly_rate > 0.4

        # Explain
        explanations = self._explain_packet(packet, fields) if is_anomalous else []

        # Confidence
        confidence = max(0, (sim_threshold - prior_sim) / 0.3) if prior_sim < sim_threshold else 0
        confidence = max(confidence, traffic_div)

        # Summary
        if not is_anomalous:
            summary = "Normal"
        elif sig_fields:
            top = sig_fields[0]
            summary = f"ALERT: {top.explanation}"
        elif traffic_div > 0.3:
            summary = f"ALERT: Traffic diverged {traffic_div:.0%}"
        else:
            summary = f"ALERT: Unusual packet (sim={prior_sim:.2f})"

        return DetectionResult(
            packet_num=self.packet_count,
            is_anomalous=is_anomalous,
            confidence=confidence,
            prior_similarity=prior_sim,
            field_stats=field_stats,
            traffic_divergence=traffic_div,
            packet_explanations=explanations,
            summary=summary,
        )


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def gen_normal(rng: random.Random) -> dict:
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.8, 0.18, 0.02])[0]
    if proto == "TCP":
        return {"protocol": "TCP", "src_port": rng.randint(49152, 65535),
                "dst_port": rng.choice([80, 443, 8080, 22]),
                "flags": rng.choices(["PA", "A", "SA", "S"], weights=[0.4, 0.3, 0.2, 0.1])[0],
                "payload_size": rng.randint(0, 1500)}
    elif proto == "UDP":
        return {"protocol": "UDP", "src_port": rng.randint(49152, 65535),
                "dst_port": rng.choice([53, 443, 123]), "payload_size": rng.randint(20, 512)}
    else:
        return {"protocol": "ICMP", "icmp_type": rng.choice([0, 8]), "payload_size": 64}


def gen_syn_flood(rng: random.Random) -> dict:
    return {"protocol": "TCP", "src_port": rng.randint(1, 65535), "dst_port": 80,
            "flags": "S", "payload_size": 0}


def gen_dns_refl(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 53, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(256, 4096)}


def gen_ntp_amp(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 123, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(300, 500)}


def gen_port_scan(rng: random.Random) -> dict:
    return {"protocol": "TCP", "src_port": 45000, "dst_port": rng.randint(1, 1024),
            "flags": "S", "payload_size": 0}


def gen_icmp_flood(rng: random.Random) -> dict:
    return {"protocol": "ICMP", "icmp_type": 8, "payload_size": 64}


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate(name: str, attack_gen, attack_frac: float = 0.9):
    print(f"\n{'='*70}")
    print(f"Attack: {name}")
    print(f"{'='*70}")

    detector = UnifiedSignificanceDetector()
    rng = random.Random(42)

    stream = []
    for _ in range(500):
        stream.append((gen_normal(rng), "normal"))
    for _ in range(500):
        if rng.random() < attack_frac:
            stream.append((attack_gen(rng), "attack"))
        else:
            stream.append((gen_normal(rng), "normal"))
    for _ in range(100):
        stream.append((gen_normal(rng), "normal"))

    first_detect = None
    results = []

    for i, (pkt, label) in enumerate(stream):
        r = detector.process(pkt)
        results.append((r, label))
        if r.is_anomalous and first_detect is None and label == "attack":
            first_detect = i

    post = [(r, l) for r, l in results if r.packet_num > 200]
    tp = sum(1 for r, l in post if l == "attack" and r.is_anomalous)
    fp = sum(1 for r, l in post if l == "normal" and r.is_anomalous)
    fn = sum(1 for r, l in post if l == "attack" and not r.is_anomalous)
    tn = sum(1 for r, l in post if l == "normal" and not r.is_anomalous)

    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(0.001, prec + rec)

    print(f"\nMetrics: TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  Precision: {prec:.1%}, Recall: {rec:.1%}, F1: {f1:.3f}")
    if first_detect:
        print(f"  Detection delay: {first_detect - 500} packets")

    # Sample alerts
    alerts = [(r, l) for r, l in post if l == "attack" and r.is_anomalous][:3]
    if alerts:
        print(f"\nSample alerts:")
        for r, _ in alerts:
            print(f"  [{r.packet_num}] {r.summary}")
            for e in r.packet_explanations[:2]:
                print(f"    → {e.field}={e.value}: {e.reason} (imp={e.importance:.2f})")

    return {"attack": name, "f1": f1, "prec": prec, "rec": rec}


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 003: Unified Significance Detector")
    print("="*70)
    print("""
    Combines:
    - Per-field tracking (from 001)
    - Difference vector explanations (from 002)
    - Adaptive thresholds based on baseline variability

    Zero hardcoded domain knowledge.
    """)

    results = []
    results.append(evaluate("SYN Flood", gen_syn_flood))
    results.append(evaluate("DNS Reflection", gen_dns_refl))
    results.append(evaluate("NTP Amplification", gen_ntp_amp))
    results.append(evaluate("Port Scan", gen_port_scan))
    results.append(evaluate("ICMP Flood", gen_icmp_flood))

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Attack':<20} {'F1':>8} {'Precision':>10} {'Recall':>8}")
    print("-"*50)
    for r in results:
        print(f"{r['attack']:<20} {r['f1']:>8.3f} {r['prec']:>10.1%} {r['rec']:>8.1%}")
    print("-"*50)
    print(f"{'Average':<20} {np.mean([r['f1'] for r in results]):>8.3f}")


if __name__ == "__main__":
    main()
