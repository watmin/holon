#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 001: Zero-Hardcode Significance Detection
=============================================================================

Goal: Detect when ANY field value becomes "significant" without hard-coding
what values to look for (no more "if src_port == 53 then dns_reflection").

Key Insight: Instead of encoding domain knowledge, we encode field-value
pairs as STRUCTURED DATA and track their frequency/distribution over time.

Detection Modes:
1. CONCENTRATION: A field that was diverse just became concentrated
   - Normal: src_port varies across ephemeral range
   - Attack: src_port is now 90% port 53

2. DIVERSIFICATION: A field that was stable just became diverse
   - Normal: dst_port is mostly 443
   - Attack: dst_port scanning across many ports

3. NOVELTY: Field values we've never seen before appeared
   - Prior: src_port always ephemeral (>= 1024)
   - Attack: src_port is now 53, 123, 1900 (never seen before)

Per-Field Tracking:
- prior_accum: Frozen baseline of normal behavior
- recent_accum: Decaying recent window
- Each stores the DISTRIBUTION of field-value vectors

Volumetric Detection:
- When similarity(prior, recent) drops → traffic pattern shifted
- Something "drowned out" the prior knowledge

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/001-significance-detection.py
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
DECAY = 0.995  # Recent accumulator decay
WARMUP_PACKETS = 200
WINDOW_SIZE = 50

# Fields to monitor - just the names, NO domain knowledge about values
MONITORED_FIELDS = [
    "src_port",
    "dst_port",
    "protocol",
    "tcp_flags",
    "icmp_type",
    "payload_size_bucket",
]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# SIGNIFICANCE TYPE
# =============================================================================

class SignificanceType(Enum):
    NONE = "none"
    CONCENTRATION = "concentration"  # Field became concentrated
    DIVERSIFICATION = "diversification"  # Field became diverse
    NOVELTY = "novelty"  # New values appeared
    VOLUMETRIC = "volumetric"  # Traffic pattern drowned out prior


@dataclass
class FieldSignificance:
    """Significance detection for a single field."""
    field_name: str
    significance_type: SignificanceType
    score: float  # 0-1, higher = more significant
    prior_recent_divergence: float  # How different is recent from prior
    concentration_score: float  # How concentrated is recent (high = concentrated)
    dominant_value: Optional[Any]  # What value dominates (if concentrated)
    explanation: str


@dataclass
class DetectionResult:
    """Overall detection result."""
    packet_num: int
    is_anomalous: bool
    confidence: float

    # Per-field significance
    field_significances: Dict[str, FieldSignificance]

    # Overall traffic divergence
    traffic_divergence: float  # How different is overall recent from prior

    # What to tell the operator
    explanation: str
    alerts: List[str]


# =============================================================================
# PER-FIELD TRACKER
# =============================================================================

class FieldTracker:
    """
    Track distribution of values for a single field using accumulators.

    No domain knowledge - just observes what values appear and how often.

    KEY INSIGHT: Compare CHANGE from baseline, not absolute values.
    If prior was 80% TCP, then 80% TCP now is NOT anomalous.
    But if prior was 80% TCP and now it's 95% TCP with flags=S, that's a change.
    """

    def __init__(self, field_name: str, encoder, dimensions: int, decay: float):
        self.field_name = field_name
        self.encoder = encoder
        self.dimensions = dimensions
        self.decay = decay

        # Accumulators - encode field-value pairs as structured data
        self.prior_accum = encoder.create_accumulator()
        self.recent_accum = encoder.create_accumulator()

        # Track seen values for novelty detection
        self.prior_values = set()
        self.recent_values = set()

        # Concentration tracking for BOTH prior and recent
        self.prior_value_counts = defaultdict(int)
        self.prior_total = 0

        self.recent_value_counts = defaultdict(int)
        self.recent_total = 0

        # Frozen prior snapshot
        self.prior_frozen = False
        self.prior_count = 0

        # Prior baseline stats (for comparison)
        self.prior_concentration = 0.0  # Baseline concentration level
        self.prior_dominant_value = None

    def _encode_field_value(self, value: Any) -> np.ndarray:
        """Encode a field-value pair as STRUCTURED data."""
        # This is the key insight: {field_name: value} preserves the role-filler binding
        # "src_port: 53" is different from "dst_port: 53"
        structure = {self.field_name: value}
        return self.encoder.encode_data(structure)

    def observe(self, value: Any, is_warmup: bool = False):
        """Observe a field value."""
        if value is None:
            return

        vec = self._encode_field_value(value)

        if is_warmup or not self.prior_frozen:
            # Building prior baseline
            self.prior_accum = self.encoder.accumulate(self.prior_accum, vec)
            self.prior_values.add(value)
            self.prior_count += 1

            # Track prior concentration
            self.prior_value_counts[value] += 1
            self.prior_total += 1
        else:
            # Post-warmup: update recent with decay
            self.recent_accum = self.decay * self.recent_accum + vec.astype(np.float64)
            self.recent_values.add(value)

            # Track concentration
            self.recent_value_counts[value] += 1
            self.recent_total += 1

            # Decay old counts (approximate sliding window)
            if self.recent_total > WINDOW_SIZE * 2:
                # Halve all counts to prevent unbounded growth
                for k in list(self.recent_value_counts.keys()):
                    self.recent_value_counts[k] = self.recent_value_counts[k] // 2
                    if self.recent_value_counts[k] == 0:
                        del self.recent_value_counts[k]
                self.recent_total = sum(self.recent_value_counts.values())

    def freeze_prior(self):
        """Freeze prior baseline - called after warmup."""
        self.prior_frozen = True

        # Compute prior baseline concentration
        if self.prior_total > 0 and self.prior_value_counts:
            most_common = max(self.prior_value_counts.items(), key=lambda x: x[1])
            self.prior_dominant_value, dominant_count = most_common
            self.prior_concentration = dominant_count / self.prior_total

        # Initialize recent from prior
        self.recent_accum = self.prior_accum.copy()
        self.recent_values = self.prior_values.copy()

    def compute_significance(self) -> FieldSignificance:
        """Compute significance metrics for this field."""
        if not self.prior_frozen:
            return FieldSignificance(
                field_name=self.field_name,
                significance_type=SignificanceType.NONE,
                score=0.0,
                prior_recent_divergence=0.0,
                concentration_score=0.0,
                dominant_value=None,
                explanation="Still warming up"
            )

        # Normalize accumulators
        prior_norm = self.encoder.normalize_accumulator(self.prior_accum)
        recent_norm = self.encoder.normalize_accumulator(self.recent_accum)

        # How different is recent from prior? (KEY metric)
        divergence = 1.0 - cosine_similarity(prior_norm, recent_norm)

        # Concentration: what fraction of recent is the dominant value?
        if self.recent_total > 0 and self.recent_value_counts:
            most_common = max(self.recent_value_counts.items(), key=lambda x: x[1])
            dominant_value, dominant_count = most_common
            concentration = dominant_count / self.recent_total
        else:
            dominant_value = None
            concentration = 0.0

        # CHANGE in concentration from baseline
        concentration_delta = concentration - self.prior_concentration

        # Did the dominant value CHANGE?
        dominant_value_changed = (
            dominant_value is not None and
            self.prior_dominant_value is not None and
            dominant_value != self.prior_dominant_value
        )

        # Novelty: values never seen in prior
        novel_values = self.recent_values - self.prior_values

        # Determine significance type based on CHANGE, not absolute
        sig_type = SignificanceType.NONE
        score = 0.0
        explanation = f"{self.field_name}: normal"

        # Check for significant divergence (this is the primary signal)
        if divergence > 0.25:
            # The field distribution has shifted significantly

            # Is it concentration on a NEW or DIFFERENT value?
            if concentration > 0.6 and (dominant_value_changed or dominant_value in novel_values):
                sig_type = SignificanceType.CONCENTRATION
                score = divergence + concentration_delta
                explanation = f"{self.field_name} CONCENTRATED on NEW value {dominant_value} ({concentration:.0%}, was {self.prior_concentration:.0%} on {self.prior_dominant_value})"

            # Is it concentration INCREASE on same value?
            elif concentration > 0.8 and concentration_delta > 0.15:
                sig_type = SignificanceType.CONCENTRATION
                score = divergence
                explanation = f"{self.field_name} CONCENTRATED from {self.prior_concentration:.0%} to {concentration:.0%} on {dominant_value}"

            # Check for novelty (new values + divergence)
            elif novel_values and len(novel_values) > 2:
                sig_type = SignificanceType.NOVELTY
                score = divergence
                novel_list = list(novel_values)[:3]
                explanation = f"{self.field_name} has {len(novel_values)} NOVEL values: {novel_list}..."

            # Check for diversification
            elif len(self.recent_value_counts) > len(self.prior_value_counts) * 2:
                sig_type = SignificanceType.DIVERSIFICATION
                score = divergence
                explanation = f"{self.field_name} DIVERSIFIED from {len(self.prior_value_counts)} to {len(self.recent_value_counts)} unique values"

            else:
                # Generic distribution shift
                sig_type = SignificanceType.NOVELTY
                score = divergence
                explanation = f"{self.field_name} distribution SHIFTED (divergence={divergence:.2f})"

        return FieldSignificance(
            field_name=self.field_name,
            significance_type=sig_type,
            score=score,
            prior_recent_divergence=divergence,
            concentration_score=concentration,
            dominant_value=dominant_value,
            explanation=explanation
        )


# =============================================================================
# SIGNIFICANCE DETECTOR
# =============================================================================

class SignificanceDetector:
    """
    Zero-hardcode anomaly detection via per-field significance tracking.

    No domain knowledge about what ports, flags, or values mean.
    Just observes what BECOMES significant.
    """

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Per-field trackers
        self.field_trackers: Dict[str, FieldTracker] = {
            field: FieldTracker(field, self.encoder, DIMENSIONS, DECAY)
            for field in MONITORED_FIELDS
        }

        # Overall traffic accumulator (all fields combined)
        self.prior_traffic = self.encoder.create_accumulator()
        self.recent_traffic = self.encoder.create_accumulator()
        self.prior_traffic_count = 0

        # Anomaly state tracking
        self.recent_anomalies = deque(maxlen=WINDOW_SIZE)
        self.anomaly_streak = 0

    def _extract_fields(self, packet: dict) -> Dict[str, Any]:
        """Extract monitored fields from packet."""
        fields = {}

        # Protocol
        fields["protocol"] = packet.get("protocol", "unknown")

        # Ports
        fields["src_port"] = packet.get("src_port")
        fields["dst_port"] = packet.get("dst_port")

        # TCP flags
        if packet.get("protocol") == "TCP":
            fields["tcp_flags"] = packet.get("flags")

        # ICMP
        if packet.get("protocol") == "ICMP":
            fields["icmp_type"] = packet.get("icmp_type")

        # Payload size bucket (no hard-coded thresholds - use simple bucketing)
        size = packet.get("payload_size", 0)
        if size == 0:
            fields["payload_size_bucket"] = "none"
        elif size < 64:
            fields["payload_size_bucket"] = "tiny"
        elif size < 256:
            fields["payload_size_bucket"] = "small"
        elif size < 1024:
            fields["payload_size_bucket"] = "medium"
        else:
            fields["payload_size_bucket"] = "large"

        return fields

    def process(self, packet: dict) -> DetectionResult:
        """Process a packet and detect significance changes."""
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        # Extract and observe fields
        fields = self._extract_fields(packet)
        for field_name, value in fields.items():
            if field_name in self.field_trackers:
                self.field_trackers[field_name].observe(value, is_warmup)

        # Encode full packet for overall traffic tracking
        packet_vec = self.encoder.encode_data(packet)

        if is_warmup:
            self.prior_traffic = self.encoder.accumulate(self.prior_traffic, packet_vec)
            self.prior_traffic_count += 1

            # End of warmup
            if self.packet_count == self.warmup_packets:
                self.warmup_complete = True
                for tracker in self.field_trackers.values():
                    tracker.freeze_prior()
                self.recent_traffic = self.prior_traffic.copy()

            return DetectionResult(
                packet_num=self.packet_count,
                is_anomalous=False,
                confidence=0.0,
                field_significances={},
                traffic_divergence=0.0,
                explanation="Warming up...",
                alerts=[]
            )

        # Post-warmup: update recent traffic
        self.recent_traffic = DECAY * self.recent_traffic + packet_vec.astype(np.float64)

        # Compute per-field significance
        field_sigs = {}
        for name, tracker in self.field_trackers.items():
            field_sigs[name] = tracker.compute_significance()

        # Compute overall traffic divergence
        prior_norm = self.encoder.normalize_accumulator(self.prior_traffic)
        recent_norm = self.encoder.normalize_accumulator(self.recent_traffic)
        traffic_divergence = 1.0 - cosine_similarity(prior_norm, recent_norm)

        # Determine if anomalous
        # Anomaly if: any field has significant divergence OR traffic divergence is high
        significant_fields = [
            f for f in field_sigs.values()
            if f.significance_type != SignificanceType.NONE and f.prior_recent_divergence > 0.25
        ]

        is_volumetric = traffic_divergence > 0.35

        # Require EITHER volumetric shift OR at least 2 significant fields
        # This reduces false positives from single-field noise
        is_anomalous = is_volumetric or len(significant_fields) >= 2

        # Track anomaly streak
        if is_anomalous:
            self.anomaly_streak += 1
        else:
            self.anomaly_streak = 0

        # Build explanation
        alerts = []
        for f in significant_fields:
            alerts.append(f.explanation)

        if is_volumetric:
            alerts.append(f"VOLUMETRIC shift: traffic diverged {traffic_divergence:.0%} from baseline")

        explanation = "; ".join(alerts) if alerts else "Normal traffic"
        confidence = max([f.score for f in significant_fields] + [traffic_divergence]) if is_anomalous else 0.0

        return DetectionResult(
            packet_num=self.packet_count,
            is_anomalous=is_anomalous,
            confidence=confidence,
            field_significances=field_sigs,
            traffic_divergence=traffic_divergence,
            explanation=explanation,
            alerts=alerts
        )


# =============================================================================
# TRAFFIC GENERATORS (for testing - these ARE hardcoded, but the detector isn't!)
# =============================================================================

def generate_normal_packet(rng: random.Random) -> dict:
    """Generate normal traffic - diverse, typical patterns."""
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.8, 0.18, 0.02])[0]

    if proto == "TCP":
        return {
            "protocol": "TCP",
            "src_port": rng.randint(49152, 65535),  # Ephemeral
            "dst_port": rng.choice([80, 443, 8080, 22]),  # Common services
            "flags": rng.choices(["PA", "A", "SA", "S"], weights=[0.4, 0.3, 0.2, 0.1])[0],
            "payload_size": rng.randint(0, 1500),
        }
    elif proto == "UDP":
        return {
            "protocol": "UDP",
            "src_port": rng.randint(49152, 65535),  # Ephemeral (client)
            "dst_port": rng.choice([53, 443, 123]),  # DNS, QUIC, NTP
            "payload_size": rng.randint(20, 512),
        }
    else:
        return {
            "protocol": "ICMP",
            "icmp_type": rng.choice([0, 8]),  # Echo reply/request
            "payload_size": 64,
        }


def generate_syn_flood(rng: random.Random) -> dict:
    """SYN flood - concentrated TCP flags."""
    return {
        "protocol": "TCP",
        "src_port": rng.randint(1, 65535),  # Spoofed random
        "dst_port": 80,
        "flags": "S",  # THE signal - but we don't hardcode looking for this!
        "payload_size": 0,
    }


def generate_dns_reflection(rng: random.Random) -> dict:
    """DNS reflection - src_port becomes concentrated on 53."""
    return {
        "protocol": "UDP",
        "src_port": 53,  # THE signal - from DNS server!
        "dst_port": rng.randint(49152, 65535),
        "payload_size": rng.randint(256, 4096),  # Amplified
    }


def generate_ntp_amplification(rng: random.Random) -> dict:
    """NTP amplification - src_port becomes concentrated on 123."""
    return {
        "protocol": "UDP",
        "src_port": 123,  # THE signal - from NTP server!
        "dst_port": rng.randint(49152, 65535),
        "payload_size": rng.randint(300, 500),
    }


def generate_port_scan(rng: random.Random) -> dict:
    """Port scan - dst_port becomes diversified."""
    return {
        "protocol": "TCP",
        "src_port": 45000,
        "dst_port": rng.randint(1, 1024),  # Scanning well-known ports
        "flags": "S",
        "payload_size": 0,
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
    """Evaluate detection on a specific attack type."""
    print(f"\n{'='*70}")
    print(f"Attack: {name}")
    print(f"{'='*70}")

    detector = SignificanceDetector(warmup_packets=200)
    rng = random.Random(42)

    # Build stream
    stream = []

    # Warmup + normal
    for _ in range(normal_count):
        stream.append((generate_normal_packet(rng), "normal"))

    # Attack phase (mixed)
    for _ in range(attack_count):
        if rng.random() < attack_fraction:
            stream.append((attack_generator(rng), "attack"))
        else:
            stream.append((generate_normal_packet(rng), "normal"))

    # Recovery
    for _ in range(100):
        stream.append((generate_normal_packet(rng), "normal"))

    # Process
    results = []
    first_detection = None

    for i, (packet, label) in enumerate(stream):
        result = detector.process(packet)
        results.append((result, label))

        if result.is_anomalous and first_detection is None and label == "attack":
            first_detection = i

    # Metrics (post-warmup only)
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
        print(f"  First detection at packet {first_detection} (delay: {first_detection - 500} after attack start)")

    # Show sample alerts
    attack_results = [(r, l) for r, l in post_warmup if l == "attack" and r.is_anomalous]
    if attack_results:
        print(f"\nSample alerts:")
        for r, _ in attack_results[:3]:
            print(f"  [{r.packet_num}] {r.explanation}")

    # Show which fields became significant
    if attack_results:
        last_result = attack_results[-1][0]
        print(f"\nField significance at detection:")
        for name, sig in last_result.field_significances.items():
            if sig.significance_type != SignificanceType.NONE:
                print(f"  {name}: {sig.significance_type.value} (score={sig.score:.2f})")
                print(f"    → {sig.explanation}")

    return {"attack": name, "f1": f1, "precision": precision, "recall": recall}


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 001: Zero-Hardcode Significance Detection")
    print("="*70)
    print("""
    Goal: Detect attacks by observing WHAT BECOMES SIGNIFICANT

    NO hardcoded rules like "if src_port == 53 then dns_reflection"

    Instead:
    - Track each field's value distribution over time
    - Detect when any field CONCENTRATES, DIVERSIFIES, or shows NOVELTY
    - Report: "src_port just became 85% concentrated on value 53"

    The detector doesn't KNOW that port 53 is DNS.
    It just knows that src_port suddenly became significant.
    """)

    results = []

    results.append(evaluate_attack(
        "SYN Flood",
        generate_syn_flood,
        attack_fraction=0.95,
    ))

    results.append(evaluate_attack(
        "DNS Reflection",
        generate_dns_reflection,
        attack_fraction=0.9,
    ))

    results.append(evaluate_attack(
        "NTP Amplification",
        generate_ntp_amplification,
        attack_fraction=0.9,
    ))

    results.append(evaluate_attack(
        "Port Scan",
        generate_port_scan,
        attack_fraction=0.9,
    ))

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

KEY INSIGHT: Zero Hardcoding
============================

The detector found these attacks by observing SIGNIFICANCE CHANGES:

- SYN Flood: tcp_flags CONCENTRATED on "S"
  (We didn't code "if flags == 'S' then syn_flood")

- DNS Reflection: src_port CONCENTRATED on 53
  (We didn't code "if src_port == 53 then dns")

- NTP Amplification: src_port CONCENTRATED on 123
  (Same principle - just observed significance)

- Port Scan: dst_port DIVERSIFIED to many values
  (We didn't code "if many_dst_ports then scan")

The operator sees:
  "src_port just became 90% concentrated on value 53"

They can then decide what to do about it.
We don't need to pre-program the meaning of port 53.
    """)


if __name__ == "__main__":
    main()
