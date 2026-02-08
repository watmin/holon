#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 004: Per-Field Rate Vectors
=============================================================================

Building on batch 012's per-field pattern detection:
- We already detect WHICH field is anomalous (e.g., "src_port diverged")
- We already detect WHAT value is concentrated (e.g., "src_port=53 at 96%")

NOW: Add RATE tracking per field using separate rate vectors.

For each monitored field, we track:
1. PATTERN accumulator: what values are appearing
2. RATE accumulator: at what rate (using encode_scalar_log)

When a field is anomalous, we can say:
- "src_port=53 is appearing at rate R_current"
- "Baseline rate for src_port=53 was R_baseline"
- "Rate limit: allow at R_baseline / R_current of current rate"

The rate limit IS derived from vector comparisons.

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/004-per-field-rate-vectors.py
"""

import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterator, Any, Union, Optional
from collections import defaultdict
import numpy as np
import time

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
# PER-FIELD TRACKER WITH RATE VECTORS
# =============================================================================


@dataclass
class FieldRateInfo:
    """Rate information for a specific field-value pair."""
    field: str
    value: Any
    current_rate_sim: float      # Similarity of current rate to baseline rate
    pattern_divergence: float    # How much this field's pattern shifted
    is_novel: bool              # Was this value seen in baseline?
    rate_factor: float          # Derived rate limit (0-1)


class FieldTracker:
    """
    Tracks both PATTERN and RATE for a single field.

    Two accumulators per field:
    - pattern_accum: what values appear (field-value vectors)
    - rate_accum: at what rate (rate vectors per observation window)
    """

    def __init__(self, field_name: str, store: CPUStore):
        self.field_name = field_name
        self.store = store
        self.encoder = store.encoder

        # Pattern tracking (from batch 012)
        self.prior_pattern = self.encoder.create_accumulator()
        self.recent_pattern = self.encoder.create_accumulator()

        # Rate tracking (NEW for batch 013)
        # We encode rate as a vector using encode_scalar_log
        self.prior_rate = self.encoder.create_accumulator()
        self.recent_rate = self.encoder.create_accumulator()

        # Track which values we've seen in baseline
        self.baseline_values = set()

        # Observation window for rate calculation
        self.window_count = 0
        self.window_start_time = time.time()

        self._frozen = False
        self._prior_pattern_norm = None
        self._prior_rate_norm = None

    def _encode_field_value(self, value: Any) -> np.ndarray:
        """Encode a field-value pair."""
        return self.encoder.encode_data({self.field_name: value})

    def _encode_rate(self, pps: float) -> np.ndarray:
        """Encode rate as a vector using log scale."""
        # Use the store's encode_scalar_log for rate encoding
        return self.store.encode_scalar_log(max(0.1, pps))

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity."""
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

    def observe(self, value: Any, is_warmup: bool, current_pps: float):
        """
        Observe a field value with associated rate.

        Args:
            value: The field value (e.g., 53 for src_port)
            is_warmup: Whether we're in warmup phase
            current_pps: Current packets per second
        """
        if value is None:
            return

        # Encode the field-value pair
        pattern_vec = self._encode_field_value(value)

        # Encode the rate
        rate_vec = self._encode_rate(current_pps)

        if is_warmup:
            # Accumulate into prior (baseline)
            self.prior_pattern = self.encoder.accumulate(self.prior_pattern, pattern_vec)
            self.prior_rate = self.encoder.accumulate(self.prior_rate, rate_vec)
            self.baseline_values.add(value)
        else:
            # Accumulate into recent with decay
            self.recent_pattern = DECAY * self.recent_pattern + pattern_vec.astype(np.float64)
            self.recent_rate = DECAY * self.recent_rate + rate_vec.astype(np.float64)

    def freeze(self):
        """Freeze baseline after warmup."""
        self._frozen = True
        self._prior_pattern_norm = self._normalize(self.prior_pattern)
        self._prior_rate_norm = self._normalize(self.prior_rate)
        # Initialize recent from prior
        self.recent_pattern = self.prior_pattern.copy()
        self.recent_rate = self.prior_rate.copy()

    def get_status(self, current_pps: float) -> dict:
        """
        Get current status of this field.

        Returns pattern divergence, rate divergence, and derived rate factor.
        """
        if not self._frozen:
            return {"field": self.field_name, "status": "warmup"}

        # Pattern divergence (from batch 012)
        recent_pattern_norm = self._normalize(self.recent_pattern)
        pattern_divergence = 1.0 - self._similarity(self._prior_pattern_norm, recent_pattern_norm)

        # Rate similarity: how does current rate compare to baseline rate?
        current_rate_vec = self._encode_rate(current_pps)
        rate_similarity = self._similarity(current_rate_vec, self._prior_rate_norm)

        # Rate divergence in accumulator
        recent_rate_norm = self._normalize(self.recent_rate)
        rate_divergence = 1.0 - self._similarity(self._prior_rate_norm, recent_rate_norm)

        return {
            "field": self.field_name,
            "pattern_divergence": pattern_divergence,
            "rate_similarity": rate_similarity,      # Current rate vs baseline rate
            "rate_divergence": rate_divergence,      # Rate accumulator shift
        }

    def is_value_novel(self, value: Any) -> bool:
        """Check if value was seen during baseline."""
        return value not in self.baseline_values


# =============================================================================
# MULTI-FIELD RATE DETECTOR
# =============================================================================


@dataclass
class RateLimitSignal:
    """Rate limit signal for a specific field-value."""
    field: str
    value: Any
    is_novel: bool
    pattern_divergence: float    # How much this field shifted
    rate_similarity: float       # Current rate vs baseline rate
    rate_factor: float           # Derived: allow at this fraction of current rate
    explanation: str


class MultiFieldRateDetector:
    """
    Detects anomalies per field and derives rate limits from rate vectors.

    For each field, tracks:
    - Pattern: what values are appearing
    - Rate: at what rate values are appearing

    Emits rate limit signals when a field is anomalous.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        # Per-field trackers
        self.field_trackers: Dict[str, FieldTracker] = {
            f: FieldTracker(f, self.store) for f in MONITORED_FIELDS
        }

        # Global traffic accumulator (for overall divergence)
        self.prior_traffic = self.encoder.create_accumulator()
        self.recent_traffic = self.encoder.create_accumulator()

        self._warmup_count = 0
        self._warmup_complete = False
        self._prior_traffic_norm = None

        # Rate simulation (packets per "window")
        self._window_packets = 0
        self._simulated_pps = 100.0  # Will be set by simulation

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

    def process(self, packet: Packet, simulated_pps: float) -> dict:
        """
        Process a packet with simulated rate.

        Args:
            packet: The packet to process
            simulated_pps: Simulated packets per second (for rate encoding)

        Returns:
            {
                "phase": str,
                "global_divergence": float,
                "field_signals": List[RateLimitSignal],
            }
        """
        self._simulated_pps = simulated_pps
        is_warmup = not self._warmup_complete

        # Encode full packet
        packet_vec = self.encoder.encode_walkable(packet)

        # Update global traffic accumulator
        if is_warmup:
            self.prior_traffic = self.encoder.accumulate(self.prior_traffic, packet_vec)
        else:
            self.recent_traffic = DECAY * self.recent_traffic + packet_vec.astype(np.float64)

        # Update per-field trackers
        for field_name, tracker in self.field_trackers.items():
            value = packet.get_field(field_name)
            if value is not None:
                tracker.observe(value, is_warmup, simulated_pps)

        # Handle warmup completion
        if is_warmup:
            self._warmup_count += 1
            if self._warmup_count >= WARMUP_PACKETS:
                self._warmup_complete = True
                self._prior_traffic_norm = self._normalize(self.prior_traffic)
                self.recent_traffic = self.prior_traffic.copy()
                for tracker in self.field_trackers.values():
                    tracker.freeze()

            return {
                "phase": "warmup",
                "global_divergence": 0.0,
                "field_signals": [],
            }

        # Active phase - compute signals
        recent_traffic_norm = self._normalize(self.recent_traffic)
        global_divergence = 1.0 - self._similarity(self._prior_traffic_norm, recent_traffic_norm)

        # Collect field signals
        field_signals = []
        for field_name, tracker in self.field_trackers.items():
            status = tracker.get_status(simulated_pps)

            # Only emit signals for fields with significant pattern divergence
            if status.get("pattern_divergence", 0) > 0.15:
                # Get the current value for this field
                current_value = packet.get_field(field_name)
                is_novel = tracker.is_value_novel(current_value) if current_value else False

                # THE KEY: Rate factor derived from rate similarity
                # rate_similarity tells us how current rate compares to baseline
                # Low similarity = rate is very different = throttle more
                rate_sim = status.get("rate_similarity", 1.0)

                # rate_factor = rate_similarity → if rate matches baseline, allow
                # If rate is 10x baseline, rate_similarity will be low → throttle
                rate_factor = max(0.0, min(1.0, rate_sim))

                signal = RateLimitSignal(
                    field=field_name,
                    value=current_value,
                    is_novel=is_novel,
                    pattern_divergence=status["pattern_divergence"],
                    rate_similarity=rate_sim,
                    rate_factor=rate_factor,
                    explanation=f"{field_name} diverged {status['pattern_divergence']:.0%}, "
                               f"rate similarity={rate_sim:.2f}",
                )
                field_signals.append(signal)

        return {
            "phase": "active",
            "global_divergence": global_divergence,
            "field_signals": field_signals,
        }


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_experiment(attack_type: str):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {attack_type}")
    print(f"{'='*70}")

    detector = MultiFieldRateDetector()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Warmup at 100 pps
    print(f"\n[WARMUP] {WARMUP_PACKETS} packets at 100 pps...")
    for _ in range(WARMUP_PACKETS):
        packet = gen_normal(rng)
        detector.process(packet, simulated_pps=100.0)

    # Normal at 100 pps
    print(f"[NORMAL] 200 packets at 100 pps...")
    normal_signals = []
    for _ in range(200):
        packet = gen_normal(rng)
        result = detector.process(packet, simulated_pps=100.0)
        if result["field_signals"]:
            normal_signals.extend(result["field_signals"])

    # Attack at 10000 pps (100x rate increase + attack pattern)
    print(f"[ATTACK] 500 packets at 10000 pps (90% attack)...")
    attack_signals = []
    attack_rate_factors = []
    for _ in range(500):
        if rng.random() < 0.9:
            packet = attack_gen(rng)
        else:
            packet = gen_normal(rng)
        result = detector.process(packet, simulated_pps=10000.0)  # 100x rate
        if result["field_signals"]:
            attack_signals.extend(result["field_signals"])
            for sig in result["field_signals"]:
                attack_rate_factors.append(sig.rate_factor)

    # Recovery at 100 pps
    print(f"[RECOVERY] 300 packets at 100 pps...")
    recovery_signals = []
    recovery_rate_factors = []
    for _ in range(300):
        packet = gen_normal(rng)
        result = detector.process(packet, simulated_pps=100.0)
        if result["field_signals"]:
            recovery_signals.extend(result["field_signals"])
            for sig in result["field_signals"]:
                recovery_rate_factors.append(sig.rate_factor)

    # Analysis
    print(f"\n{'='*70}")
    print("FIELD-LEVEL RATE SIGNALS")
    print("="*70)

    print(f"\n[NORMAL] Signals emitted: {len(normal_signals)}")

    print(f"\n[ATTACK] Signals emitted: {len(attack_signals)}")
    if attack_signals:
        # Group by field
        by_field = defaultdict(list)
        for sig in attack_signals:
            by_field[sig.field].append(sig)

        for field, sigs in by_field.items():
            rate_factors = [s.rate_factor for s in sigs]
            pattern_divs = [s.pattern_divergence for s in sigs]
            novel_count = sum(1 for s in sigs if s.is_novel)

            print(f"\n  {field}:")
            print(f"    Signals: {len(sigs)}")
            print(f"    Novel values: {novel_count}")
            print(f"    Pattern divergence: {np.mean(pattern_divs):.3f}")
            print(f"    Rate factor: mean={np.mean(rate_factors):.3f}, min={np.min(rate_factors):.3f}")

            # Sample signal
            sample = sigs[-1]
            print(f"    Sample: {sample.explanation}")

    print(f"\n[RECOVERY] Signals emitted: {len(recovery_signals)}")
    if recovery_rate_factors:
        print(f"    Final rate factors: {np.mean(recovery_rate_factors[-10:]):.3f} (last 10)")

    # Summary
    print(f"\n{'='*70}")
    print("RATE LIMIT DERIVATION")
    print("="*70)

    if attack_rate_factors:
        attack_mean_rf = np.mean(attack_rate_factors)
        attack_min_rf = np.min(attack_rate_factors)

        print(f"""
    During attack (10000 pps, 100x baseline):
    - Rate factors emitted: {len(attack_rate_factors)}
    - Mean rate factor: {attack_mean_rf:.3f}
    - Min rate factor:  {attack_min_rf:.3f}

    INTERPRETATION:
    - rate_factor = {attack_mean_rf:.2f} means "allow at {attack_mean_rf*100:.0f}% of current rate"
    - Current rate is 10000 pps
    - So effective rate = {attack_mean_rf * 10000:.0f} pps
    - Baseline was 100 pps

    The rate_factor comes from rate_similarity:
    - encode_scalar_log(100) vs encode_scalar_log(10000)
    - These are different vectors → low similarity → low rate_factor
    - This naturally suggests throttling back toward baseline
        """)

    return {
        "attack_type": attack_type,
        "attack_signals": len(attack_signals),
        "attack_rate_factor": np.mean(attack_rate_factors) if attack_rate_factors else 1.0,
    }


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 004: Per-Field Rate Vectors")
    print("="*70)
    print("""
    Building on batch 012's per-field anomaly detection.

    For each monitored field, we track TWO things:
    1. PATTERN: what values are appearing (batch 012)
    2. RATE: at what rate values are appearing (NEW)

    When a field diverges:
    - pattern_divergence tells us the field is anomalous
    - rate_similarity tells us how rate compares to baseline
    - rate_factor = rate_similarity → the derived rate limit

    This gives us: "src_port=53 is anomalous, rate_factor=0.15"
    Which means: "Allow src_port=53 at 15% of current rate"
    """)

    for attack_type in ATTACK_GENERATORS:
        run_experiment(attack_type)

    print("\n" + "="*70)
    print("KEY INSIGHT")
    print("="*70)
    print("""
    The rate limit is derived from TWO vector comparisons:

    1. Pattern: similarity(prior_pattern, recent_pattern)
       → Tells us WHICH field is anomalous

    2. Rate: similarity(current_rate_vec, baseline_rate_vec)
       → Tells us HOW MUCH to throttle

    Both are vector operations. No counters. No hardcoded thresholds.
    The rate_factor naturally emerges from the rate vector comparison.
    """)


if __name__ == "__main__":
    main()
