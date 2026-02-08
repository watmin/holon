#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 002: Drift (Accumulator Divergence) as Rate Factor
=============================================================================

FINDING FROM 001: Individual packet similarity doesn't work for SYN floods
because attack packets are MORE homogeneous than normal traffic.

NEW HYPOTHESIS: rate_factor = 1.0 - drift
where drift = 1.0 - similarity(prior_accum, recent_accum)

The DRIFT tells us "how much has traffic shifted from baseline?"
- Low drift → traffic is normal → allow at full rate
- High drift → traffic has shifted → reduce rate

This is STATE IN THE VECTOR - the accumulated pattern IS the state.

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/002-drift-as-rate.py
"""

import sys
import random
from dataclasses import dataclass
from typing import List, Tuple, Iterator, Any, Union
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore, Walkable, WalkType


# =============================================================================
# CONFIGURATION
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
    return TCPPacket(
        src_port=rng.randint(1, 65535),
        dst_port=80,
        flags="S",
        payload_size=0,
    )


def gen_dns_reflection(rng: random.Random) -> Packet:
    return UDPPacket(
        src_port=53,
        dst_port=rng.randint(49152, 65535),
        payload_size=rng.randint(512, 4096),
    )


def gen_icmp_flood(rng: random.Random) -> Packet:
    return ICMPPacket(icmp_type=8, payload_size=1400)


ATTACK_GENERATORS = {
    "syn_flood": gen_syn_flood,
    "dns_reflection": gen_dns_reflection,
    "icmp_flood": gen_icmp_flood,
}


# =============================================================================
# DRIFT-BASED RATE DETECTOR
# =============================================================================


class DriftRateDetector:
    """
    Rate limiting derived from accumulator drift.

    drift = 1.0 - similarity(prior_accum, recent_accum)
    rate_factor = 1.0 - drift = similarity(prior, recent)

    The ACCUMULATED PATTERN is the state.
    When traffic shifts, drift increases, rate_factor decreases.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        # State lives ONLY in these vectors
        self.prior_accum = self.encoder.create_accumulator()
        self.recent_accum = self.encoder.create_accumulator()

        self._prior_norm = None
        self._warmup_count = 0
        self._warmup_complete = False

    def _normalize(self, accum: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(accum)
        if norm < 1e-10:
            return accum
        return (accum / norm).astype(np.float32)

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64).flatten()
        b = b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def process(self, packet: Packet) -> dict:
        """
        Process a packet and return drift-based rate information.

        Returns:
            {
                "rate_factor": float,     # similarity(prior, recent) - THE rate signal
                "drift": float,           # 1.0 - rate_factor
                "phase": str,
            }
        """
        packet_vec = self.encoder.encode_walkable(packet)

        if not self._warmup_complete:
            self.prior_accum = self.encoder.accumulate(self.prior_accum, packet_vec)
            self._warmup_count += 1

            if self._warmup_count >= WARMUP_PACKETS:
                self._warmup_complete = True
                self._prior_norm = self._normalize(self.prior_accum)
                self.recent_accum = self.prior_accum.copy()

            return {"rate_factor": 1.0, "drift": 0.0, "phase": "warmup"}

        # Update recent accumulator with decay
        self.recent_accum = DECAY * self.recent_accum + packet_vec.astype(np.float64)

        # THE KEY: rate_factor = similarity between frozen prior and evolving recent
        recent_norm = self._normalize(self.recent_accum)
        rate_factor = self._similarity(self._prior_norm, recent_norm)
        rate_factor = max(0.0, min(1.0, rate_factor))

        drift = 1.0 - rate_factor

        return {"rate_factor": rate_factor, "drift": drift, "phase": "active"}


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_experiment(attack_type: str, attack_fraction: float = 0.9):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {attack_type}")
    print(f"{'='*70}")

    detector = DriftRateDetector()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Track rate_factor over time
    timeline = []

    # Phase 1: Warmup
    print(f"\n[WARMUP] {WARMUP_PACKETS} packets...")
    for i in range(WARMUP_PACKETS):
        packet = gen_normal(rng)
        result = detector.process(packet)
        timeline.append(("warmup", i, result["rate_factor"], result["drift"], "normal"))

    # Phase 2: Normal (200 packets)
    print(f"[NORMAL] 200 packets...")
    for i in range(200):
        packet = gen_normal(rng)
        result = detector.process(packet)
        timeline.append(("normal", WARMUP_PACKETS + i, result["rate_factor"], result["drift"], "normal"))

    # Phase 3: Attack (500 packets)
    print(f"[ATTACK] 500 packets ({attack_fraction:.0%} attack)...")
    for i in range(500):
        if rng.random() < attack_fraction:
            packet = attack_gen(rng)
            label = "attack"
        else:
            packet = gen_normal(rng)
            label = "normal"
        result = detector.process(packet)
        timeline.append(("attack", WARMUP_PACKETS + 200 + i, result["rate_factor"], result["drift"], label))

    # Phase 4: Recovery (300 packets)
    print(f"[RECOVERY] 300 packets...")
    for i in range(300):
        packet = gen_normal(rng)
        result = detector.process(packet)
        timeline.append(("recovery", WARMUP_PACKETS + 700 + i, result["rate_factor"], result["drift"], "normal"))

    # Analysis
    print(f"\n{'='*70}")
    print("RATE FACTOR (= accumulator similarity) BY PHASE")
    print("="*70)

    for phase in ["normal", "attack", "recovery"]:
        phase_data = [(rf, d) for (p, i, rf, d, l) in timeline if p == phase]
        if not phase_data:
            continue
        rates = [rf for rf, d in phase_data]
        drifts = [d for rf, d in phase_data]

        print(f"\n{phase.upper()}:")
        print(f"  Rate Factor: mean={np.mean(rates):.3f}, min={np.min(rates):.3f}, max={np.max(rates):.3f}")
        print(f"  Drift:       mean={np.mean(drifts):.3f}, min={np.min(drifts):.3f}, max={np.max(drifts):.3f}")

    # Key measurements
    normal_rates = [rf for (p, i, rf, d, l) in timeline if p == "normal"]
    attack_rates = [rf for (p, i, rf, d, l) in timeline if p == "attack"]
    recovery_rates = [rf for (p, i, rf, d, l) in timeline if p == "recovery"]

    # During attack - what's the rate factor?
    attack_min = min(attack_rates)
    attack_mean = np.mean(attack_rates)

    # First packet of recovery vs last
    recovery_first = recovery_rates[0]
    recovery_last = recovery_rates[-1]

    print(f"\n{'='*70}")
    print("KEY METRICS")
    print("="*70)
    print(f"  Normal rate_factor:    {np.mean(normal_rates):.3f}")
    print(f"  Attack rate_factor:    {attack_mean:.3f} (min: {attack_min:.3f})")
    print(f"  Recovery start:        {recovery_first:.3f}")
    print(f"  Recovery end:          {recovery_last:.3f}")
    print(f"  Recovery improvement:  {recovery_last - recovery_first:.3f}")

    # The signal
    normal_stable = np.mean(normal_rates)
    attack_lowest = attack_min
    separation = normal_stable - attack_lowest

    print(f"\n  SEPARATION: {separation:.3f}")
    print(f"  Normal stable at ~{normal_stable:.2f}, Attack drops to ~{attack_lowest:.2f}")

    # Timeline visualization (text-based)
    print(f"\n{'='*70}")
    print("RATE FACTOR TIMELINE (sample every 50 packets)")
    print("="*70)

    for i in range(0, len(timeline), 50):
        phase, idx, rf, drift, label = timeline[i]
        bar = "█" * int(rf * 40)
        print(f"  {idx:4d} [{phase:8s}] {rf:.3f} |{bar}")

    return {
        "attack_type": attack_type,
        "normal_rate": np.mean(normal_rates),
        "attack_rate_min": attack_min,
        "attack_rate_mean": attack_mean,
        "recovery_rate": recovery_last,
        "separation": separation,
    }


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 002: Drift as Rate Factor")
    print("="*70)
    print("""
    HYPOTHESIS: rate_factor = similarity(prior_accum, recent_accum)

    The DRIFT between accumulators IS the rate signal.
    - High similarity (low drift) → normal traffic → high rate
    - Low similarity (high drift) → shifted traffic → low rate

    The state IS the vector. No counters needed.
    """)

    results = []
    for attack_type in ATTACK_GENERATORS:
        result = run_experiment(attack_type)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Drift-Based Rate Factor")
    print("="*70)
    print(f"\n{'Attack':<18} {'Normal':>10} {'Atk Min':>10} {'Atk Mean':>10} {'Recovery':>10} {'Sep':>8}")
    print("-"*68)
    for r in results:
        print(f"{r['attack_type']:<18} {r['normal_rate']:>10.3f} {r['attack_rate_min']:>10.3f} "
              f"{r['attack_rate_mean']:>10.3f} {r['recovery_rate']:>10.3f} {r['separation']:>8.3f}")

    avg_sep = np.mean([r['separation'] for r in results])
    avg_normal = np.mean([r['normal_rate'] for r in results])
    avg_attack = np.mean([r['attack_rate_min'] for r in results])

    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print("="*70)
    print(f"""
    Average normal rate_factor: {avg_normal:.3f}
    Average attack rate_factor: {avg_attack:.3f} (at worst)
    Average separation:         {avg_sep:.3f}

    WHAT THIS MEANS:

    The drift-based rate_factor naturally:
    - Stays high (~{avg_normal:.2f}) during normal traffic
    - Drops dramatically (~{avg_attack:.2f}) during attacks
    - Recovers automatically as normal traffic resumes

    ENFORCER GUIDANCE:

    An enforcer can use this directly:
    - rate_factor ≈ 1.0 → "Traffic matches baseline, allow fully"
    - rate_factor ≈ 0.5 → "Traffic shifted 50%, throttle to 50%"
    - rate_factor ≈ 0.2 → "Traffic shifted 80%, throttle to 20%"

    The rate_factor IS the allowed proportion of baseline rate.
    No thresholds. No categories. Pure vector-derived rate.
    """)


if __name__ == "__main__":
    main()
