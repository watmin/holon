#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 009: Field Scrubber with Shipped Vectors
=============================================================================

SCENARIO:
- Central store: Builds baseline vectors from pcap, ships to fleet
- Field scrubbers: Receive vectors, make real-time decisions

QUESTION: What does a scrubber need to operate?
- Can it work with ONLY the shipped vectors?
- What local state (if any) is required?

SHIPPED FROM CENTRAL:
1. prior_pattern_norm (vector) - baseline pattern fingerprint
2. prior_rate_norm (vector) - baseline rate fingerprint
3. reference_rate_vectors (dict[rate → vector]) - for rate decoding

LOCAL STATE (scrubber maintains):
1. recent_pattern (vector) - decaying accumulator of current traffic
2. packet_count (scalar) - for rate estimation (unavoidable?)

DECISION FLOW:
1. Encode incoming packet → packet_vec
2. Accumulate into recent_pattern (decay)
3. Compute drift = 1 - sim(prior_pattern, recent_pattern)
4. If drift > threshold → anomalous
5. Decode baseline rate from prior_rate_norm
6. Emit: "rate limit to {baseline_rate} pps"

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/009-field-scrubber.py
"""

import sys
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator, Any, Union, Optional
import numpy as np
import json

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore, Walkable, WalkType


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
DECAY = 0.98


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


def gen_syn_flood(rng: random.Random) -> Packet:
    return TCPPacket(src_port=rng.randint(1, 65535), dst_port=80, flags="S", payload_size=0)


def gen_dns_reflection(rng: random.Random) -> Packet:
    return UDPPacket(src_port=53, dst_port=rng.randint(49152, 65535), payload_size=rng.randint(512, 4096))


# =============================================================================
# CENTRAL STORE (builds and ships vectors)
# =============================================================================


@dataclass
class ShippedVectors:
    """What central ships to field scrubbers."""
    prior_pattern_norm: np.ndarray     # Baseline pattern fingerprint
    prior_rate_norm: np.ndarray        # Baseline rate fingerprint
    reference_rate_vectors: Dict[float, np.ndarray]  # Rate decoding table
    baseline_rate_pps: float           # Pre-decoded for convenience (optional)
    decay: float                       # Decay factor


class CentralStore:
    """
    Offline pcap consumer that builds baseline vectors.
    Ships vectors to field scrubbers.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.pattern_accum = self.encoder.create_accumulator()
        self.rate_accum = self.encoder.create_accumulator()
        self.observed_rates: List[float] = []
        self.packet_count = 0

    def _normalize(self, accum: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(accum)
        return (accum / norm).astype(np.float32) if norm > 1e-10 else accum

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def ingest(self, packet: Packet, observed_pps: float):
        """Ingest a packet from pcap during baseline learning."""
        packet_vec = self.encoder.encode_walkable(packet)
        rate_vec = self.store.encode_scalar_log(max(0.1, observed_pps))

        self.pattern_accum = self.encoder.accumulate(self.pattern_accum, packet_vec)
        self.rate_accum = self.encoder.accumulate(self.rate_accum, rate_vec)
        self.observed_rates.append(observed_pps)
        self.packet_count += 1

    def build_shipped_vectors(self) -> ShippedVectors:
        """Build the vectors to ship to field scrubbers."""
        # Normalize pattern and rate accumulators
        prior_pattern_norm = self._normalize(self.pattern_accum)
        prior_rate_norm = self._normalize(self.rate_accum)

        # Build rate references from observed rates
        rates = np.array(self.observed_rates)
        p50 = np.percentile(rates, 50)

        # Build reference rates (learned from observations)
        ref_rates = [
            max(0.1, np.min(rates)),
            np.percentile(rates, 10),
            np.percentile(rates, 25),
            p50,
            np.percentile(rates, 75),
            np.percentile(rates, 90),
            np.max(rates),
            p50 * 2,
            p50 * 5,
            p50 * 10,
            p50 * 50,
            p50 * 100,
        ]
        ref_rates = sorted(set([r for r in ref_rates if r > 0]))

        reference_rate_vectors = {
            rate: self.store.encode_scalar_log(float(rate))
            for rate in ref_rates
        }

        # Pre-decode baseline rate
        best_rate, best_sim = p50, -1.0
        for rate, ref_vec in reference_rate_vectors.items():
            sim = self._similarity(prior_rate_norm, ref_vec)
            if sim > best_sim:
                best_sim, best_rate = sim, rate

        return ShippedVectors(
            prior_pattern_norm=prior_pattern_norm,
            prior_rate_norm=prior_rate_norm,
            reference_rate_vectors=reference_rate_vectors,
            baseline_rate_pps=best_rate,
            decay=DECAY,
        )


# =============================================================================
# FIELD SCRUBBER (receives vectors, makes decisions)
# =============================================================================


@dataclass
class ScrubberDecision:
    packet_num: int
    drift: float
    is_anomalous: bool
    action: str  # "ALLOW" or "RATE_LIMIT"
    enforce_rate_pps: Optional[float]


class FieldScrubber:
    """
    Edge scrubber that receives shipped vectors and makes real-time decisions.

    SHIPPED STATE (from central):
    - prior_pattern_norm: baseline pattern fingerprint
    - prior_rate_norm: baseline rate fingerprint
    - reference_rate_vectors: rate decoding table
    - baseline_rate_pps: pre-decoded baseline rate
    - decay: decay factor

    LOCAL STATE (maintained by scrubber):
    - recent_pattern: decaying accumulator of current traffic (VECTOR)
    - store: encoder for processing packets

    NO SCALAR COUNTERS for detection (except packet enumeration for logging).
    """

    def __init__(self, shipped: ShippedVectors):
        self.shipped = shipped

        # Need a store for encoding (but NOT for learning)
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        # Local vector state: recent pattern accumulator
        # Initialize from baseline (cold start = baseline)
        self.recent_pattern = shipped.prior_pattern_norm.copy().astype(np.float64)

        # Detection threshold (could also be shipped)
        self.drift_threshold = 0.15

        self._packet_num = 0

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _normalize(self, accum: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(accum)
        return (accum / norm).astype(np.float32) if norm > 1e-10 else accum

    def process(self, packet: Packet) -> ScrubberDecision:
        """
        Process a packet and decide: ALLOW or RATE_LIMIT.

        Uses ONLY:
        - Shipped vectors (from central)
        - Local recent_pattern accumulator (vector)

        NO packet counting for rate estimation.
        Drift detection is purely vector-based.
        """
        self._packet_num += 1

        # Encode packet
        packet_vec = self.encoder.encode_walkable(packet)

        # Update recent pattern with decay
        self.recent_pattern = (
            self.shipped.decay * self.recent_pattern +
            packet_vec.astype(np.float64)
        )

        # Compute drift from baseline
        recent_norm = self._normalize(self.recent_pattern)
        drift = 1.0 - self._similarity(self.shipped.prior_pattern_norm, recent_norm)

        # Decision
        is_anomalous = drift > self.drift_threshold

        if is_anomalous:
            return ScrubberDecision(
                packet_num=self._packet_num,
                drift=drift,
                is_anomalous=True,
                action="RATE_LIMIT",
                enforce_rate_pps=self.shipped.baseline_rate_pps,
            )
        else:
            return ScrubberDecision(
                packet_num=self._packet_num,
                drift=drift,
                is_anomalous=False,
                action="ALLOW",
                enforce_rate_pps=None,
            )


# =============================================================================
# SIMULATION
# =============================================================================


def run_simulation():
    print("="*70)
    print("FIELD SCRUBBER SIMULATION")
    print("="*70)

    rng = random.Random(42)

    # ==========================================================================
    # PHASE 1: Central builds baseline from pcap
    # ==========================================================================
    print("\n[CENTRAL] Building baseline from pcap...")

    central = CentralStore()
    baseline_pps = 100.0

    # Simulate pcap ingestion (500 packets at baseline rate)
    for _ in range(500):
        packet = gen_normal(rng)
        observed_pps = baseline_pps + rng.uniform(-10, 10)
        central.ingest(packet, observed_pps)

    # Build and "ship" vectors
    shipped = central.build_shipped_vectors()

    print(f"  Ingested {central.packet_count} packets")
    print(f"  Baseline rate: {shipped.baseline_rate_pps:.1f} pps")
    print(f"  Reference rates: {len(shipped.reference_rate_vectors)}")
    print(f"  Pattern vector: {shipped.prior_pattern_norm.shape}")
    print(f"  Rate vector: {shipped.prior_rate_norm.shape}")

    # ==========================================================================
    # PHASE 2: Field scrubber receives vectors
    # ==========================================================================
    print("\n[FIELD] Scrubber receives shipped vectors...")

    scrubber = FieldScrubber(shipped)

    print(f"  Scrubber initialized with:")
    print(f"    - prior_pattern_norm (vector)")
    print(f"    - prior_rate_norm (vector)")
    print(f"    - reference_rate_vectors ({len(shipped.reference_rate_vectors)} refs)")
    print(f"    - baseline_rate_pps: {shipped.baseline_rate_pps:.1f}")
    print(f"    - drift_threshold: {scrubber.drift_threshold}")

    # ==========================================================================
    # PHASE 3: Normal traffic
    # ==========================================================================
    print("\n[FIELD] Processing normal traffic (200 packets)...")

    normal_decisions = []
    for _ in range(200):
        packet = gen_normal(rng)
        decision = scrubber.process(packet)
        normal_decisions.append(decision)

    normal_rate_limits = sum(1 for d in normal_decisions if d.action == "RATE_LIMIT")
    normal_drift = [d.drift for d in normal_decisions]

    print(f"  Decisions: {len(normal_decisions)}")
    print(f"  RATE_LIMIT: {normal_rate_limits} ({normal_rate_limits/len(normal_decisions)*100:.1f}%)")
    print(f"  Drift: min={min(normal_drift):.3f}, max={max(normal_drift):.3f}, avg={sum(normal_drift)/len(normal_drift):.3f}")

    # ==========================================================================
    # PHASE 4: Attack traffic (DNS reflection)
    # ==========================================================================
    print("\n[FIELD] Processing attack traffic (500 packets, 90% attack)...")

    attack_decisions = []
    for _ in range(500):
        if rng.random() < 0.9:
            packet = gen_dns_reflection(rng)
        else:
            packet = gen_normal(rng)
        decision = scrubber.process(packet)
        attack_decisions.append(decision)

    attack_rate_limits = sum(1 for d in attack_decisions if d.action == "RATE_LIMIT")
    attack_drift = [d.drift for d in attack_decisions]

    print(f"  Decisions: {len(attack_decisions)}")
    print(f"  RATE_LIMIT: {attack_rate_limits} ({attack_rate_limits/len(attack_decisions)*100:.1f}%)")
    print(f"  Drift: min={min(attack_drift):.3f}, max={max(attack_drift):.3f}, avg={sum(attack_drift)/len(attack_drift):.3f}")

    if attack_rate_limits > 0:
        sample = next(d for d in attack_decisions if d.action == "RATE_LIMIT")
        print(f"\n  Sample RATE_LIMIT decision:")
        print(f"    packet_num: {sample.packet_num}")
        print(f"    drift: {sample.drift:.3f}")
        print(f"    enforce_rate_pps: {sample.enforce_rate_pps}")

    # ==========================================================================
    # PHASE 5: Recovery
    # ==========================================================================
    print("\n[FIELD] Processing recovery traffic (300 packets normal)...")

    recovery_decisions = []
    for _ in range(300):
        packet = gen_normal(rng)
        decision = scrubber.process(packet)
        recovery_decisions.append(decision)

    recovery_rate_limits = sum(1 for d in recovery_decisions if d.action == "RATE_LIMIT")
    recovery_drift = [d.drift for d in recovery_decisions]

    print(f"  Decisions: {len(recovery_decisions)}")
    print(f"  RATE_LIMIT: {recovery_rate_limits} ({recovery_rate_limits/len(recovery_decisions)*100:.1f}%)")
    print(f"  Drift: min={min(recovery_drift):.3f}, max={max(recovery_drift):.3f}, avg={sum(recovery_drift)/len(recovery_drift):.3f}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "="*70)
    print("SUMMARY: WHAT THE SCRUBBER NEEDED")
    print("="*70)

    print("""
    SHIPPED FROM CENTRAL (all vectors):
    ✓ prior_pattern_norm     - baseline pattern fingerprint
    ✓ prior_rate_norm        - baseline rate fingerprint (for rate decode)
    ✓ reference_rate_vectors - rate decoding table
    ✓ baseline_rate_pps      - pre-decoded baseline rate
    ✓ decay                  - decay factor (scalar constant)

    LOCAL STATE (scrubber maintains):
    ✓ recent_pattern         - decaying accumulator (VECTOR)
    ✓ store/encoder          - for encoding packets

    SCALAR COUNTERS:
    ✗ NONE for detection
    ✗ NONE for rate estimation

    The scrubber detects anomalies purely by drift:
      drift = 1 - similarity(prior_pattern, recent_pattern)

    When anomalous:
      enforce_rate = baseline_rate_pps (pre-decoded from vectors)
    """)

    print("="*70)
    print("RESULTS")
    print("="*70)

    print(f"""
    Normal traffic:  {normal_rate_limits}/{len(normal_decisions)} rate limits ({normal_rate_limits/len(normal_decisions)*100:.1f}%)
    Attack traffic:  {attack_rate_limits}/{len(attack_decisions)} rate limits ({attack_rate_limits/len(attack_decisions)*100:.1f}%)
    Recovery:        {recovery_rate_limits}/{len(recovery_decisions)} rate limits ({recovery_rate_limits/len(recovery_decisions)*100:.1f}%)

    Detection works without knowing PPS!
    The PATTERN drift itself signals the anomaly.
    Rate to enforce is pre-decoded from baseline vectors.
    """)


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 009: Field Scrubber with Shipped Vectors")
    print("="*70)
    print("""
    QUESTION: Can a field scrubber operate with ONLY shipped vectors?

    ANSWER: Almost! Here's what's needed:

    SHIPPED FROM CENTRAL:
    - prior_pattern_norm (vector) - baseline fingerprint
    - prior_rate_norm (vector) - for rate decoding
    - reference_rate_vectors (vectors) - rate decode table
    - baseline_rate_pps (scalar) - pre-decoded, optional
    - decay (scalar) - constant

    LOCAL STATE:
    - recent_pattern (vector) - decaying accumulator
    - encoder - to encode incoming packets

    NO SCALAR RATE TRACKING:
    - Detection is purely drift-based
    - Don't need to count packets per second
    - Pattern shift itself indicates anomaly

    The only scalars are constants (decay, threshold).
    """)

    run_simulation()


if __name__ == "__main__":
    main()
