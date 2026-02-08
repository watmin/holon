#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 005: Accumulator Magnitude as Rate Signal
=============================================================================

Key insight: The MAGNITUDE of an accumulator reflects frequency.

If we track per-field accumulators with decay:
- High frequency pattern → high magnitude (accumulates faster than decay)
- Low frequency pattern → low magnitude (decay dominates)

The RATIO of current magnitude to baseline magnitude IS the rate ratio.

rate_ratio = ||recent_accum|| / ||prior_accum||

If rate_ratio = 100, we're seeing 100x the traffic.
rate_factor = 1 / rate_ratio = 0.01 → throttle to 1%

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/005-accumulator-magnitude-as-rate.py
"""

import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterator, Any, Union, Optional
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore, Walkable, WalkType


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
WARMUP_PACKETS = 500
DECAY = 0.98
MONITORED_FIELDS = ["protocol", "src_port", "dst_port", "flags", "icmp_type"]


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
# FIELD TRACKER WITH MAGNITUDE-BASED RATE
# =============================================================================


class MagnitudeFieldTracker:
    """
    Tracks per-field patterns using accumulator magnitude as rate signal.

    Key insight:
    - prior_accum is frozen baseline (magnitude = baseline "mass")
    - recent_accum evolves with decay (magnitude = current "mass")
    - Ratio of magnitudes ≈ ratio of rates

    If we see 100x more traffic with same pattern:
    - recent magnitude will be ~100x prior magnitude
    - rate_factor = prior_mag / recent_mag ≈ 0.01
    """

    def __init__(self, field_name: str, encoder):
        self.field_name = field_name
        self.encoder = encoder

        # Pattern accumulator
        self.prior_accum = encoder.create_accumulator()
        self.recent_accum = encoder.create_accumulator()

        # Track baseline values
        self.baseline_values = set()

        self._frozen = False
        self._prior_magnitude = 0.0

    def _encode(self, value: Any) -> np.ndarray:
        return self.encoder.encode_data({self.field_name: value})

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64).flatten()
        b = b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _normalize(self, accum: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(accum)
        if norm < 1e-10:
            return accum
        return (accum / norm).astype(np.float32)

    def observe(self, value: Any, is_warmup: bool):
        if value is None:
            return

        vec = self._encode(value)

        if is_warmup:
            self.prior_accum = self.encoder.accumulate(self.prior_accum, vec)
            self.baseline_values.add(value)
        else:
            self.recent_accum = DECAY * self.recent_accum + vec.astype(np.float64)

    def freeze(self):
        self._frozen = True
        self._prior_magnitude = np.linalg.norm(self.prior_accum)
        # Initialize recent to match prior (same "mass")
        self.recent_accum = self.prior_accum.copy().astype(np.float64)

    def get_status(self) -> dict:
        if not self._frozen:
            return {"field": self.field_name, "status": "warmup"}

        # Pattern divergence (direction change)
        prior_norm = self._normalize(self.prior_accum)
        recent_norm = self._normalize(self.recent_accum)
        pattern_divergence = 1.0 - self._similarity(prior_norm, recent_norm)

        # Magnitude ratio (rate signal)
        recent_magnitude = np.linalg.norm(self.recent_accum)
        magnitude_ratio = recent_magnitude / self._prior_magnitude if self._prior_magnitude > 0 else 1.0

        # Rate factor: inverse of magnitude ratio (capped)
        # If magnitude_ratio = 10, rate_factor = 0.1 (throttle to 10%)
        # If magnitude_ratio = 1, rate_factor = 1.0 (no throttle)
        rate_factor = min(1.0, 1.0 / magnitude_ratio) if magnitude_ratio > 0 else 1.0

        return {
            "field": self.field_name,
            "pattern_divergence": pattern_divergence,
            "magnitude_ratio": magnitude_ratio,
            "prior_magnitude": self._prior_magnitude,
            "recent_magnitude": recent_magnitude,
            "rate_factor": rate_factor,
        }

    def is_novel(self, value: Any) -> bool:
        return value not in self.baseline_values


# =============================================================================
# MAGNITUDE-BASED RATE DETECTOR
# =============================================================================


@dataclass
class RateLimitSignal:
    field: str
    value: Any
    is_novel: bool
    pattern_divergence: float
    magnitude_ratio: float
    rate_factor: float
    explanation: str


class MagnitudeRateDetector:
    """
    Uses accumulator magnitude ratio as rate signal.

    For each field:
    - pattern_divergence → is this field anomalous?
    - magnitude_ratio → how much more traffic are we seeing?
    - rate_factor = 1/magnitude_ratio → how much to throttle
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.field_trackers = {
            f: MagnitudeFieldTracker(f, self.encoder) for f in MONITORED_FIELDS
        }

        self._warmup_count = 0
        self._warmup_complete = False

    def process(self, packet: Packet) -> dict:
        is_warmup = not self._warmup_complete

        # Update per-field trackers
        for field_name, tracker in self.field_trackers.items():
            value = packet.get_field(field_name)
            if value is not None:
                tracker.observe(value, is_warmup)

        # Handle warmup
        if is_warmup:
            self._warmup_count += 1
            if self._warmup_count >= WARMUP_PACKETS:
                self._warmup_complete = True
                for tracker in self.field_trackers.values():
                    tracker.freeze()
            return {"phase": "warmup", "signals": []}

        # Collect signals from anomalous fields
        signals = []
        for field_name, tracker in self.field_trackers.items():
            status = tracker.get_status()
            pattern_div = status.get("pattern_divergence", 0)

            # Only emit for anomalous fields
            if pattern_div > 0.15:
                value = packet.get_field(field_name)
                signal = RateLimitSignal(
                    field=field_name,
                    value=value,
                    is_novel=tracker.is_novel(value) if value else False,
                    pattern_divergence=pattern_div,
                    magnitude_ratio=status["magnitude_ratio"],
                    rate_factor=status["rate_factor"],
                    explanation=f"{field_name}: mag_ratio={status['magnitude_ratio']:.2f}, "
                               f"rate_factor={status['rate_factor']:.3f}",
                )
                signals.append(signal)

        return {"phase": "active", "signals": signals}


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_experiment(attack_type: str, attack_multiplier: float = 100.0):
    """
    Run experiment with rate multiplier during attack.

    attack_multiplier simulates processing N packets per "time unit" during attack
    vs 1 packet per time unit during normal.
    """
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {attack_type} (rate multiplier: {attack_multiplier}x)")
    print(f"{'='*70}")

    detector = MagnitudeRateDetector()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Warmup - 1 packet per "time unit"
    print(f"\n[WARMUP] {WARMUP_PACKETS} packets...")
    for _ in range(WARMUP_PACKETS):
        detector.process(gen_normal(rng))

    # Normal - 1 packet per time unit
    print(f"[NORMAL] 200 packets...")
    normal_signals = []
    for _ in range(200):
        result = detector.process(gen_normal(rng))
        normal_signals.extend(result["signals"])

    # Attack - N packets per time unit (simulating higher rate)
    # We process `attack_multiplier` packets per iteration to simulate higher rate
    print(f"[ATTACK] 500 time units × {attack_multiplier}x rate = {int(500 * attack_multiplier)} packets...")
    attack_signals = []
    attack_field_stats = defaultdict(list)

    for t in range(500):
        # Process multiple packets per time unit to simulate higher rate
        for _ in range(int(attack_multiplier)):
            if rng.random() < 0.9:
                packet = attack_gen(rng)
            else:
                packet = gen_normal(rng)
            result = detector.process(packet)

            for sig in result["signals"]:
                attack_signals.append(sig)
                attack_field_stats[sig.field].append({
                    "magnitude_ratio": sig.magnitude_ratio,
                    "rate_factor": sig.rate_factor,
                })

    # Recovery - 1 packet per time unit
    print(f"[RECOVERY] 300 packets...")
    recovery_signals = []
    for _ in range(300):
        result = detector.process(gen_normal(rng))
        recovery_signals.extend(result["signals"])

    # Analysis
    print(f"\n{'='*70}")
    print("MAGNITUDE-BASED RATE SIGNALS")
    print("="*70)

    print(f"\n[NORMAL] Signals: {len(normal_signals)}")
    print(f"[ATTACK] Signals: {len(attack_signals)}")

    if attack_field_stats:
        for field, stats in attack_field_stats.items():
            mag_ratios = [s["magnitude_ratio"] for s in stats]
            rate_factors = [s["rate_factor"] for s in stats]

            print(f"\n  {field}:")
            print(f"    Signals: {len(stats)}")
            print(f"    Magnitude ratio: mean={np.mean(mag_ratios):.1f}, max={np.max(mag_ratios):.1f}")
            print(f"    Rate factor:     mean={np.mean(rate_factors):.4f}, min={np.min(rate_factors):.4f}")

    print(f"\n[RECOVERY] Signals: {len(recovery_signals)}")
    if recovery_signals:
        # Check final recovery state
        final_signals = recovery_signals[-10:] if len(recovery_signals) >= 10 else recovery_signals
        final_rate_factors = [s.rate_factor for s in final_signals]
        print(f"    Final rate factors: {np.mean(final_rate_factors):.3f}")

    # Summary
    print(f"\n{'='*70}")
    print("RATE LIMIT INTERPRETATION")
    print("="*70)

    if attack_field_stats:
        # Get the most affected field
        most_affected = max(attack_field_stats.items(), key=lambda x: len(x[1]))
        field_name = most_affected[0]
        stats = most_affected[1]

        final_mag_ratio = stats[-1]["magnitude_ratio"]
        final_rate_factor = stats[-1]["rate_factor"]

        print(f"""
    Field: {field_name}
    Final magnitude ratio: {final_mag_ratio:.1f}x
    Final rate factor: {final_rate_factor:.4f}

    INTERPRETATION:
    - Magnitude ratio = {final_mag_ratio:.0f}x means we're seeing {final_mag_ratio:.0f}x the traffic
    - Rate factor = {final_rate_factor:.4f} means "allow at {final_rate_factor*100:.2f}% of current rate"
    - If current rate is {attack_multiplier}x baseline, effective rate = {final_rate_factor * attack_multiplier:.2f}x baseline

    ENFORCER MESSAGE:
    "Rate limit {field_name} traffic to {final_rate_factor*100:.2f}% of current rate"
    or equivalently:
    "Allow {field_name} at 1/{final_mag_ratio:.0f} of current rate"
        """)

    return {
        "attack_type": attack_type,
        "attack_multiplier": attack_multiplier,
        "signals": len(attack_signals),
        "final_rate_factor": final_rate_factor if attack_field_stats else 1.0,
    }


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 005: Accumulator Magnitude as Rate Signal")
    print("="*70)
    print("""
    KEY INSIGHT: Accumulator magnitude reflects frequency.

    With decay:
    - recent_accum = decay * recent_accum + new_vec
    - High rate → accumulator grows (additions > decay)
    - Low rate → accumulator shrinks (decay > additions)

    magnitude_ratio = ||recent|| / ||prior||
    rate_factor = 1 / magnitude_ratio

    If we're seeing 100x the traffic:
    - magnitude_ratio ≈ 100
    - rate_factor ≈ 0.01 → throttle to 1%

    ALL from vector operations. No counters.
    """)

    for attack_type in ATTACK_GENERATORS:
        run_experiment(attack_type, attack_multiplier=100.0)


if __name__ == "__main__":
    main()
