#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 003: Difference Vector as Rate-Limit Pattern
=============================================================================

We have: rate_factor = similarity(prior, recent) → how much to throttle

Now we need: WHAT to throttle

HYPOTHESIS: difference(recent, prior) = the anomaly signature

The difference vector captures "what's new/different" between baseline
and current traffic. We can use this to:
1. Identify WHICH packets should be throttled
2. Match future packets against the anomaly pattern

packet_match = similarity(packet, difference_vec)
- High match → this packet matches the anomaly → apply rate_factor
- Low match → this packet is normal → allow fully

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/003-difference-as-pattern.py
"""

import sys
import random
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
    return TCPPacket(src_port=rng.randint(1, 65535), dst_port=80, flags="S", payload_size=0)


def gen_dns_reflection(rng: random.Random) -> Packet:
    return UDPPacket(src_port=53, dst_port=rng.randint(49152, 65535), payload_size=rng.randint(512, 4096))


def gen_icmp_flood(rng: random.Random) -> Packet:
    return ICMPPacket(icmp_type=8, payload_size=1400)


ATTACK_GENERATORS = {
    "syn_flood": gen_syn_flood,
    "dns_reflection": gen_dns_reflection,
    "icmp_flood": gen_icmp_flood,
}


# =============================================================================
# DIFFERENCE-PATTERN RATE DETECTOR
# =============================================================================


class DifferencePatternDetector:
    """
    Uses the DIFFERENCE between prior and recent to identify what to throttle.

    Two signals:
    1. rate_factor = similarity(prior, recent) → global throttle level
    2. pattern_match = similarity(packet, difference) → does this packet match anomaly?

    Combined: packet_rate = 1.0 - (pattern_match * (1.0 - rate_factor))

    If traffic is normal (rate_factor ≈ 1.0), all packets pass.
    If traffic shifted and packet matches the difference, throttle it.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

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
        Process packet and return both rate_factor and pattern_match.

        Returns:
            {
                "rate_factor": float,      # Global: how much has traffic shifted
                "pattern_match": float,    # Packet: does this match the difference?
                "packet_rate": float,      # Combined: effective rate for this packet
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

            return {
                "rate_factor": 1.0,
                "pattern_match": 0.0,
                "packet_rate": 1.0,
                "phase": "warmup",
            }

        # Update recent
        self.recent_accum = DECAY * self.recent_accum + packet_vec.astype(np.float64)
        recent_norm = self._normalize(self.recent_accum)

        # Global rate_factor (from experiment 002)
        rate_factor = self._similarity(self._prior_norm, recent_norm)
        rate_factor = max(0.0, min(1.0, rate_factor))

        # THE KEY: Compute difference vector
        # difference = what's in recent that's NOT in prior
        difference = self.store.difference(recent_norm, self._prior_norm)

        # How much does this packet match the difference?
        pattern_match = self._similarity(packet_vec, difference)
        # Clamp to [0, 1] - we only care about positive matches
        pattern_match = max(0.0, min(1.0, pattern_match))

        # Combined packet rate:
        # If rate_factor is 1.0 (normal), packet_rate = 1.0 for all
        # If rate_factor < 1.0 and packet matches difference, reduce its rate
        # packet_rate = 1.0 - (pattern_match * (1.0 - rate_factor))
        #
        # Example: rate_factor=0.3, pattern_match=0.8
        # packet_rate = 1.0 - (0.8 * 0.7) = 1.0 - 0.56 = 0.44
        drift = 1.0 - rate_factor
        packet_rate = 1.0 - (pattern_match * drift)
        packet_rate = max(0.0, min(1.0, packet_rate))

        return {
            "rate_factor": rate_factor,
            "pattern_match": pattern_match,
            "packet_rate": packet_rate,
            "phase": "active",
        }


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_experiment(attack_type: str, attack_fraction: float = 0.9):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {attack_type}")
    print(f"{'='*70}")

    detector = DifferencePatternDetector()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Warmup
    for _ in range(WARMUP_PACKETS):
        detector.process(gen_normal(rng))

    # Normal phase - collect baseline
    print(f"\n[NORMAL] 200 packets...")
    normal_results = []
    for _ in range(200):
        packet = gen_normal(rng)
        result = detector.process(packet)
        normal_results.append(("normal", result))

    # Attack phase - mixed traffic
    print(f"[ATTACK] 500 packets ({attack_fraction:.0%} attack)...")
    attack_results = []
    for _ in range(500):
        if rng.random() < attack_fraction:
            packet = attack_gen(rng)
            label = "attack"
        else:
            packet = gen_normal(rng)
            label = "normal"
        result = detector.process(packet)
        attack_results.append((label, result))

    # Recovery phase
    print(f"[RECOVERY] 300 packets...")
    recovery_results = []
    for _ in range(300):
        packet = gen_normal(rng)
        result = detector.process(packet)
        recovery_results.append(("normal", result))

    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS: Rate Factor vs Pattern Match vs Packet Rate")
    print("="*70)

    def analyze(results: List[Tuple[str, dict]], label_filter: str = None):
        filtered = [(l, r) for l, r in results if label_filter is None or l == label_filter]
        if not filtered:
            return None
        rates = [r["rate_factor"] for _, r in filtered]
        matches = [r["pattern_match"] for _, r in filtered]
        packet_rates = [r["packet_rate"] for _, r in filtered]
        return {
            "count": len(filtered),
            "rate_factor": {"mean": np.mean(rates), "min": np.min(rates), "max": np.max(rates)},
            "pattern_match": {"mean": np.mean(matches), "min": np.min(matches), "max": np.max(matches)},
            "packet_rate": {"mean": np.mean(packet_rates), "min": np.min(packet_rates), "max": np.max(packet_rates)},
        }

    # Normal phase (all normal)
    normal_stats = analyze(normal_results)
    print(f"\n[NORMAL PHASE] - All normal packets")
    print(f"  rate_factor:   {normal_stats['rate_factor']['mean']:.3f}")
    print(f"  pattern_match: {normal_stats['pattern_match']['mean']:.3f}")
    print(f"  packet_rate:   {normal_stats['packet_rate']['mean']:.3f}")

    # Attack phase - attack packets only
    attack_attack_stats = analyze(attack_results, "attack")
    if attack_attack_stats:
        print(f"\n[ATTACK PHASE] - Attack packets")
        print(f"  rate_factor:   {attack_attack_stats['rate_factor']['mean']:.3f} (min: {attack_attack_stats['rate_factor']['min']:.3f})")
        print(f"  pattern_match: {attack_attack_stats['pattern_match']['mean']:.3f} (max: {attack_attack_stats['pattern_match']['max']:.3f})")
        print(f"  packet_rate:   {attack_attack_stats['packet_rate']['mean']:.3f} (min: {attack_attack_stats['packet_rate']['min']:.3f})")

    # Attack phase - normal packets (collateral damage check)
    attack_normal_stats = analyze(attack_results, "normal")
    if attack_normal_stats:
        print(f"\n[ATTACK PHASE] - Normal packets (collateral)")
        print(f"  rate_factor:   {attack_normal_stats['rate_factor']['mean']:.3f}")
        print(f"  pattern_match: {attack_normal_stats['pattern_match']['mean']:.3f}")
        print(f"  packet_rate:   {attack_normal_stats['packet_rate']['mean']:.3f}")

    # Recovery phase
    recovery_stats = analyze(recovery_results)
    print(f"\n[RECOVERY PHASE] - All normal packets")
    print(f"  rate_factor:   {recovery_stats['rate_factor']['mean']:.3f} (end: {recovery_results[-1][1]['rate_factor']:.3f})")
    print(f"  pattern_match: {recovery_stats['pattern_match']['mean']:.3f}")
    print(f"  packet_rate:   {recovery_stats['packet_rate']['mean']:.3f}")

    # KEY QUESTION: Separation between attack and normal packets
    print(f"\n{'='*70}")
    print("SEPARATION: Attack Packets vs Normal Packets (during attack)")
    print("="*70)

    if attack_attack_stats and attack_normal_stats:
        # Pattern match separation
        attack_match = attack_attack_stats["pattern_match"]["mean"]
        normal_match = attack_normal_stats["pattern_match"]["mean"]
        match_sep = attack_match - normal_match

        # Packet rate separation
        attack_rate = attack_attack_stats["packet_rate"]["mean"]
        normal_rate = attack_normal_stats["packet_rate"]["mean"]
        rate_sep = normal_rate - attack_rate

        print(f"\n  Pattern Match:")
        print(f"    Attack packets: {attack_match:.3f}")
        print(f"    Normal packets: {normal_match:.3f}")
        print(f"    Separation:     {match_sep:.3f}")

        print(f"\n  Packet Rate (effective throttle):")
        print(f"    Attack packets: {attack_rate:.3f} → throttle to {attack_rate*100:.0f}%")
        print(f"    Normal packets: {normal_rate:.3f} → throttle to {normal_rate*100:.0f}%")
        print(f"    Separation:     {rate_sep:.3f}")

        # Clean separation check
        attack_max_rate = attack_attack_stats["packet_rate"]["max"]
        normal_min_rate = attack_normal_stats["packet_rate"]["min"]
        clean = normal_min_rate > attack_max_rate

        print(f"\n  Clean separation: {'YES' if clean else 'NO (overlap)'}")
        print(f"    Attack max packet_rate: {attack_max_rate:.3f}")
        print(f"    Normal min packet_rate: {normal_min_rate:.3f}")

    return {
        "attack_type": attack_type,
        "attack_packet_rate": attack_attack_stats["packet_rate"]["mean"] if attack_attack_stats else 0,
        "normal_packet_rate": attack_normal_stats["packet_rate"]["mean"] if attack_normal_stats else 1,
        "separation": rate_sep if attack_attack_stats and attack_normal_stats else 0,
    }


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 003: Difference as Rate-Limit Pattern")
    print("="*70)
    print("""
    HYPOTHESIS: difference(recent, prior) = the anomaly signature

    Two signals (both from vectors):
    1. rate_factor = similarity(prior, recent) → global throttle
    2. pattern_match = similarity(packet, difference) → packet matches anomaly?

    Combined: packet_rate = 1.0 - (pattern_match * drift)

    This gives PER-PACKET rate limits:
    - Normal packets during attack → high packet_rate (less throttled)
    - Attack packets during attack → low packet_rate (heavily throttled)
    - All packets during normal → high packet_rate (allow all)
    """)

    results = []
    for attack_type in ATTACK_GENERATORS:
        result = run_experiment(attack_type)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: Per-Packet Rate Limits")
    print("="*70)
    print(f"\n{'Attack':<18} {'Attack Rate':>12} {'Normal Rate':>12} {'Separation':>12}")
    print("-"*56)
    for r in results:
        print(f"{r['attack_type']:<18} {r['attack_packet_rate']:>12.3f} {r['normal_packet_rate']:>12.3f} {r['separation']:>12.3f}")

    print(f"""

    WHAT THIS MEANS FOR ENFORCERS:

    The difference vector IS the pattern to match.
    The packet_rate IS the rate limit for that packet.

    Enforcer logic:
    1. Receive difference_vec and global rate_factor
    2. For each packet: pattern_match = similarity(packet, difference_vec)
    3. packet_rate = 1.0 - (pattern_match * (1.0 - rate_factor))
    4. Apply rate limit: allow packet with probability packet_rate

    ALL derived from vector operations. No discrete rules.
    """)


if __name__ == "__main__":
    main()
