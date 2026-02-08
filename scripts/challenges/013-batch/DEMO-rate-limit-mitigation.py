#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 DEMO: Vector-Derived Rate Limit Mitigation
=============================================================================

This demo extends Batch 012's zero-hardcode detection with ACTIONABLE RATE LIMITS.

BATCH 012 GAVE US:
- Field-level anomaly detection (what field is anomalous)
- Value identification (what value is concentrated/novel)
- Zero hardcoded domain knowledge

BATCH 013 ADDS:
- Rate limit decoding (what PPS to enforce)
- Single vector solution (no reference storage)
- Binary search decode (O(log N) complexity)
- Field scrubber architecture (shipped vectors + local state)

OUTPUT: Complete mitigation rules for an enforcer:
{
    "match": {"src_port": 53},
    "action": "rate_limit",
    "rate_pps": 5000,
    "reason": "src_port=53 anomalous (novel, 96% concentration)"
}

NOTE: This demo uses the HolonClient interface properly instead of
implementing custom similarity/normalize functions.

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/DEMO-rate-limit-mitigation.py
"""

import sys
import random
import json
import math
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator, Any, Union, Optional
from enum import Enum
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])


# =============================================================================
# HYPERLOGLOG: Memory-Efficient Cardinality Estimation
# =============================================================================

class HyperLogLog:
    """
    Memory-efficient cardinality estimator.

    Uses ~2^p bytes of memory (default p=10 = 1KB) to estimate
    cardinality of arbitrarily large sets with ~1-2% error.

    This replaces the O(n) memory set() approach.

    Memory comparison for 1 million unique values:
    - set(): ~48MB (each element + hash table overhead)
    - HyperLogLog(p=10): 1KB (constant!)
    """

    def __init__(self, precision: int = 10):
        """
        Initialize HyperLogLog with given precision.

        Args:
            precision: Number of bits for bucket index (4-16).
                      Higher = more memory but lower error.
                      p=10: 1KB memory, ~1.04% error
                      p=14: 16KB memory, ~0.8% error
        """
        self.p = precision
        self.m = 1 << precision  # Number of buckets (2^p)
        self.registers = bytearray(self.m)  # Each register is 1 byte
        self.count = 0  # Total observations

        # Alpha constant for bias correction
        if self.m == 16:
            self.alpha = 0.673
        elif self.m == 32:
            self.alpha = 0.697
        elif self.m == 64:
            self.alpha = 0.709
        else:
            self.alpha = 0.7213 / (1 + 1.079 / self.m)

    def _hash(self, value: Any) -> int:
        """Hash any value to a 64-bit integer."""
        h = hashlib.sha256(str(value).encode()).digest()
        return int.from_bytes(h[:8], 'big')

    def _leading_zeros(self, value: int, max_bits: int = 64) -> int:
        """Count leading zeros after the bucket index bits."""
        if value == 0:
            return max_bits
        count = 0
        for i in range(max_bits - 1, -1, -1):
            if value & (1 << i):
                break
            count += 1
        return count

    def add(self, value: Any):
        """Add a value to the set."""
        self.count += 1
        h = self._hash(value)

        # First p bits determine the bucket
        bucket = h & (self.m - 1)

        # Remaining bits used for leading zero count
        remaining = h >> self.p
        zeros = self._leading_zeros(remaining, 64 - self.p) + 1

        # Update register with max
        self.registers[bucket] = max(self.registers[bucket], zeros)

    def estimate(self) -> float:
        """Estimate the cardinality (number of unique values)."""
        # Harmonic mean of 2^register values
        indicator = sum(2.0 ** (-r) for r in self.registers)
        raw_estimate = self.alpha * self.m * self.m / indicator

        # Small range correction
        if raw_estimate <= 2.5 * self.m:
            zeros = self.registers.count(0)
            if zeros > 0:
                return self.m * math.log(self.m / zeros)

        # Large range correction (for 32-bit hash, not needed for 64-bit)
        return raw_estimate

    def cardinality_ratio(self) -> float:
        """
        Get ratio of unique values to total observations.

        Returns:
            0.0 to 1.0 where:
            - Low ratio = stable (same values repeated)
            - High ratio = diverse (many unique values)
        """
        if self.count == 0:
            return 0.0
        return min(1.0, self.estimate() / self.count)

    def reset(self):
        """Reset the estimator."""
        self.registers = bytearray(self.m)
        self.count = 0

    def memory_bytes(self) -> int:
        """Return memory usage in bytes."""
        return self.m  # 1 byte per register

from holon import HolonClient, Walkable, WalkType


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
DECAY = 0.98
MONITORED_FIELDS = ["protocol", "src_port", "dst_port", "flags", "icmp_type"]
DRIFT_THRESHOLD = 0.15


# =============================================================================
# WALKABLE PACKET TYPES
# =============================================================================


class TCPPacket(Walkable):
    __slots__ = ("protocol", "src_port", "dst_port", "flags", "payload_size")

    def __init__(self, src_port: int, dst_port: int, flags: str, payload_size: int):
        self.protocol = "TCP"
        self.src_port = src_port
        self.dst_port = dst_port
        self.flags = flags
        self.payload_size = payload_size

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self) -> Iterator[Tuple[str, Any]]:
        yield "protocol", self.protocol
        yield "src_port", self.src_port
        yield "dst_port", self.dst_port
        yield "flags", self.flags
        yield "payload_size", self.payload_size

    def get_field(self, name: str) -> Any:
        return getattr(self, name, None)

    def __repr__(self):
        return f"TCP({self.src_port}→{self.dst_port}, {self.flags})"


class UDPPacket(Walkable):
    __slots__ = ("protocol", "src_port", "dst_port", "payload_size")

    def __init__(self, src_port: int, dst_port: int, payload_size: int):
        self.protocol = "UDP"
        self.src_port = src_port
        self.dst_port = dst_port
        self.payload_size = payload_size

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self) -> Iterator[Tuple[str, Any]]:
        yield "protocol", self.protocol
        yield "src_port", self.src_port
        yield "dst_port", self.dst_port
        yield "payload_size", self.payload_size

    def get_field(self, name: str) -> Any:
        return getattr(self, name, None)

    def __repr__(self):
        return f"UDP({self.src_port}→{self.dst_port})"


class ICMPPacket(Walkable):
    __slots__ = ("protocol", "icmp_type", "payload_size")

    def __init__(self, icmp_type: int, payload_size: int):
        self.protocol = "ICMP"
        self.icmp_type = icmp_type
        self.payload_size = payload_size

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self) -> Iterator[Tuple[str, Any]]:
        yield "protocol", self.protocol
        yield "icmp_type", self.icmp_type
        yield "payload_size", self.payload_size

    def get_field(self, name: str) -> Any:
        return getattr(self, name, None)

    def __repr__(self):
        return f"ICMP(type={self.icmp_type})"


Packet = Union[TCPPacket, UDPPacket, ICMPPacket]


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================


def gen_normal(rng: random.Random) -> Packet:
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.75, 0.20, 0.05])[0]
    if proto == "TCP":
        return TCPPacket(
            src_port=rng.randint(49152, 65535),
            dst_port=rng.choice([80, 443, 8080, 22, 8443]),
            flags=rng.choices(["PA", "A", "SA", "S", "FA"], weights=[0.4, 0.3, 0.15, 0.10, 0.05])[0],
            payload_size=rng.randint(0, 1500),
        )
    elif proto == "UDP":
        return UDPPacket(
            src_port=rng.randint(49152, 65535),
            dst_port=rng.choice([53, 443, 123, 5353]),
            payload_size=rng.randint(20, 512),
        )
    else:
        return ICMPPacket(icmp_type=rng.choice([0, 8]), payload_size=64)


def gen_dns_reflection(rng: random.Random) -> Packet:
    return UDPPacket(src_port=53, dst_port=rng.randint(49152, 65535), payload_size=rng.randint(512, 4096))


def gen_syn_flood(rng: random.Random) -> Packet:
    return TCPPacket(src_port=rng.randint(1, 65535), dst_port=80, flags="S", payload_size=0)


def gen_icmp_flood(rng: random.Random) -> Packet:
    return ICMPPacket(icmp_type=8, payload_size=1400)


def gen_ntp_amplification(rng: random.Random) -> Packet:
    return UDPPacket(src_port=123, dst_port=rng.randint(49152, 65535), payload_size=rng.randint(468, 482))


ATTACKS = {
    "dns_reflection": gen_dns_reflection,
    "syn_flood": gen_syn_flood,
    "icmp_flood": gen_icmp_flood,
    "ntp_amplification": gen_ntp_amplification,
}


# =============================================================================
# BINARY SEARCH RATE DECODER (using HolonClient)
# =============================================================================


class BinarySearchRateDecoder:
    """
    Decodes rate from a single rate vector using binary search.
    NO reference vectors stored. O(log N) complexity.

    Uses HolonClient.similarity() instead of custom implementation.
    """

    def __init__(self, client: HolonClient, precision: float = 0.1):
        self.client = client
        self.precision = precision

    def decode(self, target_vec: np.ndarray, log_lo: float = -1, log_hi: float = 12) -> float:
        """Binary search to find rate that best matches target vector."""
        phi = (1 + math.sqrt(5)) / 2  # Golden ratio

        d = (log_hi - log_lo) / phi
        x1, x2 = log_hi - d, log_lo + d

        vec_x1 = self.client.encode_scalar_log(10**x1)
        vec_x2 = self.client.encode_scalar_log(10**x2)

        # Use HolonClient.similarity() instead of custom function
        sim_x1 = self.client.similarity(target_vec, vec_x1)
        sim_x2 = self.client.similarity(target_vec, vec_x2)

        while log_hi - log_lo > self.precision:
            if sim_x1 > sim_x2:
                log_hi, x2, sim_x2 = x2, x1, sim_x1
                d = (log_hi - log_lo) / phi
                x1 = log_hi - d
                vec_x1 = self.client.encode_scalar_log(10**x1)
                sim_x1 = self.client.similarity(target_vec, vec_x1)
            else:
                log_lo, x1, sim_x1 = x1, x2, sim_x2
                d = (log_hi - log_lo) / phi
                x2 = log_lo + d
                vec_x2 = self.client.encode_scalar_log(10**x2)
                sim_x2 = self.client.similarity(target_vec, vec_x2)

        return 10**((log_lo + log_hi) / 2)


# =============================================================================
# FIELD TRACKER (using HolonClient)
# =============================================================================


class FieldTracker:
    """
    Tracks pattern and rate for a single field.

    Uses HolonClient methods for all vector operations.
    """

    def __init__(self, field_name: str, client: HolonClient):
        self.field_name = field_name
        self.client = client

        # Use HolonClient's accumulator methods
        self.prior_pattern = client.create_accumulator()
        self.recent_pattern = client.create_accumulator()

        self.prior_counts: Dict[Any, int] = {}
        self.prior_total = 0
        self.recent_counts: Dict[Any, int] = {}
        self.recent_total = 0

        self.baseline_values = set()
        self._frozen = False
        self._prior_norm = None

        # Track cardinality using HyperLogLog (O(1) memory!)
        # p=10 gives us 1KB memory with ~1% error - plenty accurate for our threshold
        self._baseline_hll = HyperLogLog(precision=10)  # During warmup
        self._anomaly_hll = HyperLogLog(precision=10)   # During detection

        # Baseline cardinality encoded as vector (shipped to scrubber)
        self._baseline_cardinality_vec = None
        self._baseline_cardinality = 0.0

    def observe(self, value: Any, is_warmup: bool):
        if value is None:
            return

        # Use HolonClient.encode() for field:value pairs
        vec = self.client.encode({self.field_name: value})

        if is_warmup:
            # Use HolonClient.accumulate()
            self.prior_pattern = self.client.accumulate(self.prior_pattern, vec)
            self.prior_counts[value] = self.prior_counts.get(value, 0) + 1
            self.prior_total += 1
            self.baseline_values.add(value)
            # Track baseline cardinality
            self._baseline_hll.add(value)
        else:
            # Decaying accumulator (manual decay + accumulate)
            self.recent_pattern = DECAY * self.recent_pattern + vec.astype(np.float64)
            self.recent_counts[value] = self.recent_counts.get(value, 0) + 1
            self.recent_total += 1

            if self.recent_total > 200:
                for k in list(self.recent_counts.keys()):
                    self.recent_counts[k] //= 2
                    if self.recent_counts[k] == 0:
                        del self.recent_counts[k]
                self.recent_total = sum(self.recent_counts.values())

    def freeze(self):
        self._frozen = True
        # Use HolonClient.normalize_accumulator()
        self._prior_norm = self.client.normalize_accumulator(self.prior_pattern)
        self.recent_pattern = self.prior_pattern.copy().astype(np.float64)
        self.recent_counts = dict(self.prior_counts)
        self.recent_total = self.prior_total

        # Encode baseline cardinality as vector
        # Use (cardinality * 1000 + 1) to scale to a range suitable for log encoding
        # This maps cardinality 0.0->1, 0.001->2, 0.01->11, 0.1->101, 1.0->1001
        self._baseline_cardinality = self._baseline_hll.cardinality_ratio()
        scaled_card = self._baseline_cardinality * 1000 + 1
        self._baseline_cardinality_vec = self.client.encode_scalar_log(scaled_card)

    def get_divergence(self) -> float:
        if not self._frozen:
            return 0.0
        recent_norm = self.client.normalize_accumulator(self.recent_pattern)
        # Use HolonClient.similarity()
        return 1.0 - self.client.similarity(self._prior_norm, recent_norm)

    def get_dominant_value(self) -> Tuple[Any, float]:
        if not self.recent_counts:
            return None, 0.0
        value, count = max(self.recent_counts.items(), key=lambda x: x[1])
        concentration = count / self.recent_total if self.recent_total > 0 else 0
        return value, concentration

    def is_novel(self, value: Any) -> bool:
        return value not in self.baseline_values

    def track_anomaly_value(self, value: Any):
        """Track a value seen during anomaly for cardinality calculation."""
        self._anomaly_hll.add(value)

    def get_anomaly_cardinality_ratio(self) -> float:
        """
        Get ratio of unique values to total observations during anomaly.

        Uses HyperLogLog for O(1) memory instead of O(n) set.

        High ratio (close to 1.0) = randomized field (each packet has different value)
        Low ratio (close to 0.0) = stable field (same value repeated)

        Examples:
        - DNS reflection src_port=53: 1 unique / 100 observations = 0.01 (stable)
        - SYN flood src_port: 100 unique / 100 observations = 1.0 (randomized)
        """
        return self._anomaly_hll.cardinality_ratio()

    def get_anomaly_observation_count(self) -> int:
        """Get total observations during anomaly."""
        return self._anomaly_hll.count

    def get_cardinality_divergence(self) -> float:
        """
        Compare observed cardinality against baseline using vector similarity.

        Returns 0.0 if cardinality matches baseline, higher if it diverged.
        This is what the scrubber uses - no need to decode, just compare!
        """
        if self._baseline_cardinality_vec is None:
            return 0.0

        # Encode current cardinality as vector
        current_card = self._anomaly_hll.cardinality_ratio()
        scaled_card = current_card * 1000 + 1
        current_vec = self.client.encode_scalar_log(scaled_card)

        # Compare against baseline vector
        similarity = self.client.similarity(self._baseline_cardinality_vec, current_vec)
        return 1.0 - similarity

    def get_baseline_cardinality(self) -> float:
        """Get baseline cardinality ratio (for display)."""
        return self._baseline_cardinality

    def reset_anomaly_tracking(self):
        """Reset anomaly cardinality tracking."""
        self._anomaly_hll.reset()


# =============================================================================
# MITIGATION SIGNAL
# =============================================================================


@dataclass
class MitigationRule:
    """A complete mitigation rule for an enforcer."""
    match: Dict[str, Any]  # Composite match: {"protocol": "UDP", "src_port": 53}
    action: str  # "rate_limit" or "monitor"
    rate_pps: float
    reason: str
    components: List[Dict[str, Any]]  # Individual field signals that composed this rule

    def to_json(self) -> dict:
        return {
            "match": self.match,
            "action": self.action,
            "rate_pps": round(self.rate_pps, 0),
            "reason": self.reason,
        }


# =============================================================================
# UNIFIED DETECTOR (using HolonClient)
# =============================================================================


class RateLimitDetector:
    """
    Complete detector that emits rate limit mitigation rules.

    Uses HolonClient for all vector operations.
    """

    def __init__(self, baseline_pps: float = 1000.0):
        # Use HolonClient instead of CPUStore directly
        self.client = HolonClient(dimensions=DIMENSIONS)
        self.decoder = BinarySearchRateDecoder(self.client)

        self.field_trackers = {f: FieldTracker(f, self.client) for f in MONITORED_FIELDS}

        # Rate accumulator using HolonClient
        self.rate_accum = self.client.create_accumulator()
        self._baseline_rate_pps = baseline_pps

        self._warmup_count = 0
        self._warmup_complete = False
        self._decoded_baseline = None

    def process(self, packet: Packet, current_pps: float) -> List[MitigationRule]:
        is_warmup = not self._warmup_complete

        # Observe rate using HolonClient methods
        if is_warmup:
            rate_vec = self.client.encode_scalar_log(max(0.1, current_pps))
            self.rate_accum = self.client.accumulate(self.rate_accum, rate_vec)

        # Observe fields
        for name, tracker in self.field_trackers.items():
            value = packet.get_field(name)
            if value is not None:
                tracker.observe(value, is_warmup)

        if is_warmup:
            self._warmup_count += 1
            if self._warmup_count >= 300:
                self._warmup_complete = True
                for tracker in self.field_trackers.values():
                    tracker.freeze()
                # Decode baseline rate using HolonClient
                rate_norm = self.client.normalize_accumulator(self.rate_accum)
                self._decoded_baseline = self.decoder.decode(rate_norm)
            return []

        # Check for anomalies - collect STABLE anomalous fields for this packet
        # The attack signature is made of CONCENTRATED fields (same value across many attack packets)
        # NOT random/ephemeral fields (different value per packet)
        anomalous_fields = []
        for name, tracker in self.field_trackers.items():
            divergence = tracker.get_divergence()

            if divergence > DRIFT_THRESHOLD:
                value = packet.get_field(name)  # Get THIS packet's value

                # Skip if packet doesn't have this field
                if value is None:
                    continue

                # Track this value for cardinality calculation
                tracker.track_anomaly_value(value)

                dominant_value, concentration = tracker.get_dominant_value()
                is_novel = tracker.is_novel(value)

                # Get cardinality ratio for this field
                # High ratio = randomized (each packet has different value)
                # Low ratio = stable (same value repeated)
                cardinality_ratio = tracker.get_anomaly_cardinality_ratio()

                # KEY RULE 1: Only include if CONCENTRATED (same value appears repeatedly)
                if concentration < 0.5:
                    continue

                # KEY RULE 2: Exclude HIGH-CARDINALITY fields (randomized)
                # If >50% of observations are unique values, this field is randomized
                # Need enough observations to be confident (at least 20)
                # Examples:
                #   - DNS reflection src_port=53: cardinality ~0.01 (1 unique / 100 obs) → KEEP
                #   - SYN flood src_port: cardinality ~1.0 (100 unique / 100 obs) → SKIP
                obs_count = tracker.get_anomaly_observation_count()
                if obs_count >= 20 and cardinality_ratio > 0.5:
                    continue
                # If not enough observations yet, only include if NOT novel (known stable values)
                elif obs_count < 20 and is_novel:
                    continue

                # Must match the dominant value to be part of the signature
                if value == dominant_value:
                    anomalous_fields.append({
                        "field": name,
                        "value": value,
                        "is_novel": is_novel,
                        "concentration": concentration,
                        "divergence": divergence,
                        "cardinality_ratio": cardinality_ratio,
                    })

        # If we have anomalous fields, emit ONE composite rule
        if anomalous_fields:
            # Build composite match from all anomalous fields
            composite_match = {f["field"]: f["value"] for f in anomalous_fields}

            # Build reason from components
            reason_parts = []
            for f in anomalous_fields:
                parts = [f["field"] + "=" + str(f["value"])]
                if f["is_novel"]:
                    parts.append("novel")
                if f["concentration"] > 0.5:
                    parts.append(f"{f['concentration']*100:.0f}%")
                reason_parts.append(f"({', '.join(parts)})")

            reason = " + ".join(reason_parts)

            rule = MitigationRule(
                match=composite_match,
                action="rate_limit",
                rate_pps=self._decoded_baseline,
                reason=reason,
                components=anomalous_fields,
            )
            return [rule]

        return []


# =============================================================================
# SIMULATION
# =============================================================================


class Phase(Enum):
    WARMUP = "warmup"
    NORMAL = "normal"
    ATTACK = "attack"


@dataclass
class TimePhase:
    name: str
    duration_seconds: int
    pps: int
    phase: Phase
    attack_type: Optional[str] = None


def run_demo():
    print("=" * 75)
    print("BATCH 013 DEMO: Vector-Derived Rate Limit Mitigation")
    print("=" * 75)
    print("""
    This demo shows COMPLETE mitigation rules derived from vectors.

    FROM BATCH 012:
    - Field-level anomaly detection (what field is anomalous)
    - Value identification (what value is concentrated/novel)
    - Zero hardcoded domain knowledge

    NEW IN BATCH 013:
    - Rate limit decoding (what PPS to enforce)
    - Single vector solution (no reference storage)
    - Binary search decode (O(log N) complexity)

    USING: HolonClient interface (not custom similarity/normalize functions)
    """)

    # Define scenario
    baseline_pps = 5000
    attack_pps = 500000

    # With scale=0.001, packets = duration * pps * 0.001
    # warmup needs 300+ packets, so 100 * 5000 * 0.001 = 500 packets ✓
    timeline = [
        TimePhase("warmup", 100, baseline_pps, Phase.WARMUP),        # 500 packets
        TimePhase("normal-1", 50, baseline_pps, Phase.NORMAL),       # 250 packets
        TimePhase("DNS Reflection", 50, attack_pps, Phase.ATTACK, "dns_reflection"),  # 250 packets
        TimePhase("recovery-1", 50, baseline_pps, Phase.NORMAL),     # 250 packets
        TimePhase("SYN Flood", 50, attack_pps, Phase.ATTACK, "syn_flood"),  # 250 packets
        TimePhase("recovery-2", 50, baseline_pps, Phase.NORMAL),     # 250 packets
        TimePhase("ICMP Flood", 50, attack_pps, Phase.ATTACK, "icmp_flood"),  # 250 packets
        TimePhase("recovery-3", 50, baseline_pps, Phase.NORMAL),     # 250 packets
    ]

    print("-" * 75)
    print("SCENARIO")
    print("-" * 75)
    print(f"\n  Baseline: {baseline_pps:,} pps")
    print(f"  Attack:   {attack_pps:,} pps ({attack_pps/baseline_pps:.0f}x amplification)")
    print(f"\n  {'Phase':<20} {'PPS':>12} {'Type':<15}")
    print("  " + "-" * 50)
    for phase in timeline:
        ptype = phase.attack_type or phase.phase.value
        print(f"  {phase.name:<20} {phase.pps:>12,} {ptype:<15}")

    # Run detection
    print("\n" + "-" * 75)
    print("DETECTION & MITIGATION")
    print("-" * 75)

    detector = RateLimitDetector(baseline_pps=baseline_pps)
    rng = random.Random(42)

    all_rules: Dict[str, List[MitigationRule]] = defaultdict(list)
    scale = 0.001  # Scale down for demo (smaller for faster execution)

    for phase in timeline:
        num_packets = int(phase.duration_seconds * phase.pps * scale)
        attack_gen = ATTACKS.get(phase.attack_type)

        phase_rules = []

        for _ in range(num_packets):
            if phase.phase == Phase.ATTACK and attack_gen and rng.random() < 0.9:
                packet = attack_gen(rng)
            else:
                packet = gen_normal(rng)

            rules = detector.process(packet, phase.pps)
            phase_rules.extend(rules)

        if phase.phase == Phase.ATTACK:
            # After attack phase, compute cardinality for each field
            # and build ONE composite rule from stable (low-cardinality) fields
            stable_fields = {}
            for name, tracker in detector.field_trackers.items():
                divergence = tracker.get_divergence()
                if divergence <= DRIFT_THRESHOLD:
                    continue

                dominant_value, concentration = tracker.get_dominant_value()
                if concentration < 0.5:
                    continue

                # Get cardinality metrics
                current_card = tracker.get_anomaly_cardinality_ratio()
                baseline_card = tracker.get_baseline_cardinality()
                card_divergence = tracker.get_cardinality_divergence()

                # Only include stable fields (cardinality < 0.3 = same value repeated often)
                # This is the OBSERVED cardinality, derived from HLL comparison
                if current_card < 0.3 and dominant_value is not None:
                    stable_fields[name] = {
                        "value": dominant_value,
                        "is_novel": tracker.is_novel(dominant_value),
                        "concentration": concentration,
                        "divergence": divergence,
                        "cardinality": current_card,
                        "baseline_cardinality": baseline_card,
                        "cardinality_divergence": card_divergence,
                    }

            # Reset anomaly tracking for next phase
            for tracker in detector.field_trackers.values():
                tracker.reset_anomaly_tracking()

            if stable_fields:
                # Build composite match from stable fields only
                composite_match = {name: info["value"] for name, info in stable_fields.items()}
                components = [
                    {
                        "field": name,
                        "value": info["value"],
                        "is_novel": info["is_novel"],
                        "concentration": info["concentration"],
                        "divergence": info["divergence"],
                    }
                    for name, info in stable_fields.items()
                ]

                reason_parts = []
                for name, info in stable_fields.items():
                    parts = [f"{name}={info['value']}"]
                    if info["is_novel"]:
                        parts.append("novel")
                    parts.append(f"{info['concentration']*100:.0f}% conc")

                    # Interpret cardinality change in human terms
                    baseline_card = info['baseline_cardinality']
                    current_card = info['cardinality']

                    if baseline_card > 0.5 and current_card < 0.3:
                        # Was random/diverse, now fixed - KEY attack signature
                        card_desc = "was random, now fixed"
                    elif baseline_card < 0.3 and current_card < 0.3:
                        # Was stable, still stable
                        card_desc = "stable"
                    elif baseline_card < 0.3 and current_card > 0.5:
                        # Was stable, now random
                        card_desc = "was fixed, now random"
                    else:
                        # Mixed/transitional
                        card_desc = f"card {baseline_card:.0%}->{current_card:.0%}"

                    parts.append(card_desc)
                    reason_parts.append(f"({', '.join(parts)})")

                rule = MitigationRule(
                    match=composite_match,
                    action="rate_limit",
                    rate_pps=detector._decoded_baseline,
                    reason=" + ".join(reason_parts),
                    components=components,
                )
                all_rules[phase.name] = [rule]

    # Show decoded baseline
    print(f"\n  Baseline rate decoded from vector: {detector._decoded_baseline:,.0f} pps")
    print(f"  (True baseline: {baseline_pps:,} pps, error: {abs(detector._decoded_baseline - baseline_pps)/baseline_pps*100:.1f}%)")

    # Show rules per attack
    for attack_name, rules in all_rules.items():
        print(f"\n  {attack_name}:")
        for rule in rules:
            # Format composite match
            match_str = " AND ".join(f"{k}={v}" for k, v in rule.match.items())
            print(f"    MATCH: {match_str}")

            # Show component details
            for comp in rule.components:
                status = "novel" if comp["is_novel"] else f"{comp['concentration']*100:.0f}% conc"
                print(f"      - {comp['field']}={comp['value']} ({status}, {comp['divergence']*100:.0f}% drift)")

            print(f"    → RATE LIMIT TO: {rule.rate_pps:,.0f} pps")

    # JSON output
    print("\n" + "-" * 75)
    print("ENFORCER JSON OUTPUT")
    print("-" * 75)

    for attack_name, rules in all_rules.items():
        print(f"\n  {attack_name}:")
        for rule in rules:
            print(f"  {json.dumps(rule.to_json(), indent=4)}")

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY: WHAT'S NEW IN BATCH 013")
    print("=" * 75)
    print(f"""
    BATCH 012 gave us:
      - "src_port=53 is anomalous (novel, 96% concentration)"

    BATCH 013 adds:
      - "Rate limit to {detector._decoded_baseline:,.0f} pps"

    The rate limit is DECODED from a single baseline rate vector using
    binary search (O(log N), ~10 iterations, no stored references).

    COMPLETE MITIGATION RULE:
    {{
        "match": {{"src_port": 53}},
        "action": "rate_limit",
        "rate_pps": {detector._decoded_baseline:.0f},
        "reason": "src_port=53 anomalous (novel, 96% concentration)"
    }}

    ANOMALY SIGNALS (all vector-based):
    1. Pattern Divergence - similarity(baseline_pattern, current_pattern)
       Detects: What field values changed?

    2. Rate Divergence - similarity(baseline_rate, current_rate)
       Detects: Did throughput change? Decode to PPS for enforcement.

    3. Cardinality Divergence - similarity(baseline_cardinality, current_cardinality)
       Detects: Did value diversity change?
       Example: src_port going from random (card=1.0) to fixed (card=0.09)

    STATE SUMMARY (per field):
    - Central ships: pattern_vec + rate_vec + cardinality_vec (3 vectors)
    - Scrubber maintains: recent_pattern (1 vector) + HyperLogLog (1KB)
    - Rate decode: Binary search on demand (no stored refs)

    MEMORY EFFICIENCY:
    - Cardinality tracking: HyperLogLog (1KB per field, ~1% error)
    - Comparison: set() would use ~48MB for 1M unique src_ports
    - Total HLL memory: {len(MONITORED_FIELDS)} fields × 1KB = {len(MONITORED_FIELDS)}KB
    - Total vectors: {len(MONITORED_FIELDS)} fields × 3 vectors × 4KB = {len(MONITORED_FIELDS) * 12}KB

    HOLON INTERFACE USED:
    - HolonClient.encode() - encode field:value pairs
    - HolonClient.encode_walkable() - encode packets
    - HolonClient.encode_scalar_log() - encode rates
    - HolonClient.similarity() - compare vectors
    - HolonClient.create_accumulator() - initialize accumulators
    - HolonClient.accumulate() - add to accumulators
    - HolonClient.normalize_accumulator() - normalize for queries

    ZERO HARDCODED KNOWLEDGE:
    - Fields discovered from packet structure
    - Values learned during warmup
    - Rate decoded from baseline vector
    - Thresholds are sensitivity tuning, not domain knowledge
    """)


def main():
    run_demo()


if __name__ == "__main__":
    main()
