#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 001: Similarity as Rate Factor
=============================================================================

HYPOTHESIS: similarity(packet, baseline) IS the rate limit.

No counters. No thresholds. No discrete categories.
Just: rate_factor = similarity(packet_vec, baseline_vec)

If this works, we can tell an enforcer:
  "Allow this packet class at {rate_factor} of baseline rate"

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/001-similarity-as-rate.py
"""

import sys
import random
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Any, Union
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore, Walkable, WalkType


# =============================================================================
# CONFIGURATION - Minimal, no magic numbers for thresholds
# =============================================================================

DIMENSIONS = 4096
WARMUP_PACKETS = 500
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
    """Generate normal traffic - mixed protocols, normal patterns."""
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
        return ICMPPacket(
            icmp_type=rng.choice([0, 8]),
            payload_size=64,
        )


def gen_syn_flood(rng: random.Random) -> Packet:
    """SYN flood - all SYN flags, single destination."""
    return TCPPacket(
        src_port=rng.randint(1, 65535),
        dst_port=80,
        flags="S",
        payload_size=0,
    )


def gen_dns_reflection(rng: random.Random) -> Packet:
    """DNS reflection - UDP from port 53 with large payloads."""
    return UDPPacket(
        src_port=53,
        dst_port=rng.randint(49152, 65535),
        payload_size=rng.randint(512, 4096),
    )


def gen_icmp_flood(rng: random.Random) -> Packet:
    """ICMP flood - echo requests with large payloads."""
    return ICMPPacket(
        icmp_type=8,
        payload_size=1400,
    )


ATTACK_GENERATORS = {
    "syn_flood": gen_syn_flood,
    "dns_reflection": gen_dns_reflection,
    "icmp_flood": gen_icmp_flood,
}


# =============================================================================
# VECTOR-ONLY RATE DETECTOR
# =============================================================================


class VectorRateDetector:
    """
    Rate limiting derived purely from vector similarity.

    NO counters. NO thresholds. NO discrete state.

    rate_factor = similarity(packet, baseline)

    That's it. The rate factor IS the similarity.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        # State lives ONLY in these vectors
        self.prior_accum = self.encoder.create_accumulator()
        self.recent_accum = self.encoder.create_accumulator()

        # Normalized versions (cached for efficiency, but derived from accumulators)
        self._prior_norm = None
        self._recent_norm = None

        # Warmup tracking - we need SOME way to know when to freeze
        # But this is the ONLY scalar state we keep
        self._warmup_count = 0
        self._warmup_complete = False

    def _normalize(self, accum: np.ndarray) -> np.ndarray:
        """Normalize accumulator to unit vector."""
        norm = np.linalg.norm(accum)
        if norm < 1e-10:
            return accum
        return (accum / norm).astype(np.float32)

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        a = a.astype(np.float64).flatten()
        b = b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def process(self, packet: Packet) -> dict:
        """
        Process a packet and return rate information.

        Returns:
            {
                "rate_factor": float,     # 0.0 to 1.0 - the rate limit
                "drift": float,           # How much traffic has shifted
                "phase": str,             # "warmup" or "active"
            }
        """
        # Encode packet
        packet_vec = self.encoder.encode_walkable(packet)

        # During warmup, just accumulate
        if not self._warmup_complete:
            self.prior_accum = self.encoder.accumulate(self.prior_accum, packet_vec)
            self._warmup_count += 1

            if self._warmup_count >= WARMUP_PACKETS:
                self._warmup_complete = True
                self._prior_norm = self._normalize(self.prior_accum)
                self.recent_accum = self.prior_accum.copy()
                self._recent_norm = self._prior_norm.copy()

            return {
                "rate_factor": 1.0,  # Allow everything during warmup
                "drift": 0.0,
                "phase": "warmup",
            }

        # Active phase - update recent accumulator with decay
        self.recent_accum = DECAY * self.recent_accum + packet_vec.astype(np.float64)
        self._recent_norm = self._normalize(self.recent_accum)

        # THE KEY INSIGHT: rate_factor IS the similarity
        # How similar is this packet to what we learned during baseline?
        rate_factor = self._similarity(packet_vec, self._prior_norm)

        # Clamp to [0, 1] - similarity can be negative for very anomalous packets
        rate_factor = max(0.0, min(1.0, rate_factor))

        # Drift = how much has traffic shifted overall?
        drift = 1.0 - self._similarity(self._prior_norm, self._recent_norm)

        return {
            "rate_factor": rate_factor,
            "drift": drift,
            "phase": "active",
        }


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_experiment(attack_type: str, attack_fraction: float = 0.8):
    """
    Run an experiment with mixed normal and attack traffic.

    Measures rate_factor distribution for normal vs attack packets.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {attack_type}")
    print(f"{'='*70}")

    detector = VectorRateDetector()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Phase 1: Warmup (pure normal traffic)
    print(f"\n[WARMUP] Processing {WARMUP_PACKETS} normal packets...")
    for _ in range(WARMUP_PACKETS):
        packet = gen_normal(rng)
        detector.process(packet)

    # Phase 2: Normal traffic (post-warmup baseline)
    print(f"\n[NORMAL] Processing 200 normal packets...")
    normal_rates = []
    for _ in range(200):
        packet = gen_normal(rng)
        result = detector.process(packet)
        normal_rates.append(result["rate_factor"])

    # Phase 3: Attack traffic (mixed)
    print(f"\n[ATTACK] Processing 500 packets ({attack_fraction:.0%} attack)...")
    attack_rates = []
    mixed_normal_rates = []
    for _ in range(500):
        if rng.random() < attack_fraction:
            packet = attack_gen(rng)
            result = detector.process(packet)
            attack_rates.append(result["rate_factor"])
        else:
            packet = gen_normal(rng)
            result = detector.process(packet)
            mixed_normal_rates.append(result["rate_factor"])

    # Phase 4: Recovery (normal traffic)
    print(f"\n[RECOVERY] Processing 200 normal packets...")
    recovery_rates = []
    recovery_drifts = []
    for _ in range(200):
        packet = gen_normal(rng)
        result = detector.process(packet)
        recovery_rates.append(result["rate_factor"])
        recovery_drifts.append(result["drift"])

    # Analysis
    print(f"\n{'='*70}")
    print("RATE FACTOR DISTRIBUTION (0.0 = block, 1.0 = allow)")
    print("="*70)

    def stats(values: List[float], label: str):
        if not values:
            return
        arr = np.array(values)
        print(f"\n{label}:")
        print(f"  Count: {len(arr)}")
        print(f"  Mean:  {arr.mean():.3f}")
        print(f"  Std:   {arr.std():.3f}")
        print(f"  Min:   {arr.min():.3f}")
        print(f"  Max:   {arr.max():.3f}")
        print(f"  P10:   {np.percentile(arr, 10):.3f}")
        print(f"  P50:   {np.percentile(arr, 50):.3f}")
        print(f"  P90:   {np.percentile(arr, 90):.3f}")

    stats(normal_rates, "NORMAL (post-warmup)")
    stats(attack_rates, "ATTACK")
    stats(mixed_normal_rates, "NORMAL (during attack)")
    stats(recovery_rates, "RECOVERY")

    # Separation analysis
    if normal_rates and attack_rates:
        normal_mean = np.mean(normal_rates)
        attack_mean = np.mean(attack_rates)
        separation = normal_mean - attack_mean

        print(f"\n{'='*70}")
        print("SEPARATION ANALYSIS")
        print("="*70)
        print(f"  Normal mean:  {normal_mean:.3f}")
        print(f"  Attack mean:  {attack_mean:.3f}")
        print(f"  Separation:   {separation:.3f}")

        # Check if there's clear separation
        normal_min = np.min(normal_rates)
        attack_max = np.max(attack_rates)
        clean_separation = normal_min > attack_max

        print(f"\n  Normal min:   {normal_min:.3f}")
        print(f"  Attack max:   {attack_max:.3f}")
        print(f"  Clean separation: {'YES' if clean_separation else 'NO (overlap)'}")

        # Suggested rate factor threshold (if we had to pick one)
        # But the POINT is we don't need a threshold - the rate IS the similarity
        if separation > 0.1:
            midpoint = (normal_min + attack_max) / 2
            print(f"\n  If using threshold: {midpoint:.3f}")
            print(f"  But rate_factor IS the rate - no threshold needed!")

    # Drift analysis
    print(f"\n{'='*70}")
    print("DRIFT ANALYSIS (traffic divergence)")
    print("="*70)
    print(f"  Final drift: {recovery_drifts[-1]:.3f}")
    print(f"  Recovery drift range: {min(recovery_drifts):.3f} - {max(recovery_drifts):.3f}")

    return {
        "attack_type": attack_type,
        "normal_mean": np.mean(normal_rates),
        "attack_mean": np.mean(attack_rates),
        "separation": np.mean(normal_rates) - np.mean(attack_rates),
        "clean_separation": np.min(normal_rates) > np.max(attack_rates),
    }


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 001: Similarity as Rate Factor")
    print("="*70)
    print("""
    HYPOTHESIS: rate_factor = similarity(packet, baseline)

    No counters. No thresholds. No discrete categories.
    The rate factor IS the similarity.

    If normal packets get rate_factor ≈ 0.8-1.0
    And attack packets get rate_factor ≈ 0.0-0.3
    Then we can tell an enforcer: "Allow at {rate_factor} of baseline"
    """)

    results = []
    for attack_type in ATTACK_GENERATORS:
        result = run_experiment(attack_type)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Rate Factor Separation by Attack Type")
    print("="*70)
    print(f"\n{'Attack':<20} {'Normal':>10} {'Attack':>10} {'Sep':>10} {'Clean?':>10}")
    print("-"*60)
    for r in results:
        print(f"{r['attack_type']:<20} {r['normal_mean']:>10.3f} {r['attack_mean']:>10.3f} "
              f"{r['separation']:>10.3f} {'YES' if r['clean_separation'] else 'NO':>10}")

    # Interpretation
    avg_separation = np.mean([r['separation'] for r in results])
    all_clean = all(r['clean_separation'] for r in results)

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("="*70)
    print(f"""
    Average separation: {avg_separation:.3f}
    All clean separation: {all_clean}

    WHAT THIS MEANS FOR RATE LIMITING:

    If similarity IS the rate factor, then:
    - Normal packets → rate_factor ≈ {np.mean([r['normal_mean'] for r in results]):.2f} → allow at ~{np.mean([r['normal_mean'] for r in results])*100:.0f}% rate
    - Attack packets → rate_factor ≈ {np.mean([r['attack_mean'] for r in results]):.2f} → allow at ~{np.mean([r['attack_mean'] for r in results])*100:.0f}% rate

    An enforcer can use this directly:
    - rate_factor > 0.7 → allow freely
    - rate_factor 0.3-0.7 → moderate throttle
    - rate_factor < 0.3 → heavy throttle/block

    But even better: the enforcer just uses the rate_factor AS IS.
    No categories. Pure continuous rate limiting.
    """)


if __name__ == "__main__":
    main()
