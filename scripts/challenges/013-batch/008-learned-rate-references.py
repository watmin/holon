#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 008: Learn Rate References from Observation
=============================================================================

PROBLEM: Hardcoded REFERENCE_RATES = [1, 5, 10, 50, 100, ...] is magic.

We don't know ahead of time what rates an app will produce.
- A low-traffic service might baseline at 2 pps
- A high-traffic API might baseline at 50,000 pps

SOLUTION: Learn reference rates from baseline observation.

During warmup:
1. Observe actual PPS values
2. Compute distribution: min, p25, p50, p75, max
3. Generate references spanning the observed range
4. Also project "attack scale" references: 2x, 10x, 100x baseline

Now reference rates are DISCOVERED, not hardcoded.

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/008-learned-rate-references.py
"""

import sys
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Iterator, Any, Union, Optional
from collections import defaultdict
import numpy as np
import json

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
# LEARNED RATE DECODER
# =============================================================================


class LearnedRateDecoder:
    """
    Builds reference rates from observed baseline traffic.

    Instead of hardcoded [1, 10, 100, 1000, ...]:
    1. Observe actual rates during warmup
    2. Compute percentiles (p10, p25, p50, p75, p90)
    3. Project attack scales (2x, 5x, 10x, 50x, 100x median)
    4. Build reference vectors from these learned rates
    """

    def __init__(self, store: CPUStore):
        self.store = store
        self.observed_rates: List[float] = []
        self.reference_vectors: Dict[float, np.ndarray] = {}
        self.reference_rates: List[float] = []
        self._frozen = False

    def observe(self, pps: float):
        """Observe a rate during warmup."""
        if not self._frozen:
            self.observed_rates.append(pps)

    def freeze(self):
        """Build reference rates from observations."""
        if not self.observed_rates:
            # Fallback if no observations
            self.reference_rates = [1, 10, 100, 1000, 10000]
        else:
            # Compute distribution from observed rates
            rates = np.array(self.observed_rates)

            # Get percentiles of observed rates
            p10 = np.percentile(rates, 10)
            p25 = np.percentile(rates, 25)
            p50 = np.percentile(rates, 50)  # median
            p75 = np.percentile(rates, 75)
            p90 = np.percentile(rates, 90)
            min_rate = max(0.1, np.min(rates))
            max_rate = np.max(rates)

            # Build reference rates spanning observed range
            # Plus projections for attack detection
            base_refs = [min_rate, p10, p25, p50, p75, p90, max_rate]

            # Project attack scales (multiples of median)
            attack_scales = [2, 5, 10, 50, 100]
            attack_refs = [p50 * scale for scale in attack_scales]

            # Also include some below-baseline references
            below_refs = [p50 / 10, p50 / 2] if p50 > 1 else []

            # Combine and deduplicate
            all_refs = set(base_refs + attack_refs + below_refs)
            self.reference_rates = sorted([r for r in all_refs if r > 0])

        # Build reference vectors
        for rate in self.reference_rates:
            self.reference_vectors[rate] = self.store.encode_scalar_log(float(rate))

        self._frozen = True

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def decode(self, rate_vec: np.ndarray) -> Tuple[float, float]:
        """
        Decode a rate vector to the closest reference rate.
        Returns (decoded_rate, confidence).
        """
        if not self._frozen:
            return (100.0, 0.0)

        best_rate, best_sim = 100.0, -1.0
        for rate, ref_vec in self.reference_vectors.items():
            sim = self._similarity(rate_vec, ref_vec)
            if sim > best_sim:
                best_sim, best_rate = sim, rate

        return best_rate, best_sim

    def get_baseline_rate(self) -> float:
        """Get the median observed baseline rate."""
        if self.observed_rates:
            return float(np.median(self.observed_rates))
        return 100.0

    def describe(self) -> str:
        """Describe the learned reference rates."""
        if not self._frozen:
            return "Not frozen yet"

        baseline = self.get_baseline_rate()
        return (
            f"Observed {len(self.observed_rates)} rate samples\n"
            f"  Baseline median: {baseline:.1f} pps\n"
            f"  Reference rates: {[round(r, 1) for r in self.reference_rates]}"
        )


# =============================================================================
# UNIFIED FIELD TRACKER (with learned decoder)
# =============================================================================


class UnifiedFieldTracker:
    def __init__(self, field_name: str, store: CPUStore, decoder: LearnedRateDecoder):
        self.field_name = field_name
        self.store = store
        self.encoder = store.encoder
        self.decoder = decoder

        self.prior_pattern = self.encoder.create_accumulator()
        self.recent_pattern = self.encoder.create_accumulator()
        self.prior_rate = self.encoder.create_accumulator()
        self.recent_rate = self.encoder.create_accumulator()

        self.prior_counts: Dict[Any, int] = {}
        self.prior_total = 0
        self.recent_counts: Dict[Any, int] = {}
        self.recent_total = 0

        self.baseline_values = set()
        self._frozen = False

    def _encode_field_value(self, value: Any) -> np.ndarray:
        return self.encoder.encode_data({self.field_name: value})

    def _encode_rate(self, pps: float) -> np.ndarray:
        return self.store.encode_scalar_log(max(0.1, pps))

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _normalize(self, accum: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(accum)
        return (accum / norm).astype(np.float32) if norm > 1e-10 else accum

    def observe(self, value: Any, is_warmup: bool, current_pps: float):
        if value is None:
            return

        pattern_vec = self._encode_field_value(value)
        rate_vec = self._encode_rate(current_pps)

        if is_warmup:
            self.prior_pattern = self.encoder.accumulate(self.prior_pattern, pattern_vec)
            self.prior_rate = self.encoder.accumulate(self.prior_rate, rate_vec)
            self.prior_counts[value] = self.prior_counts.get(value, 0) + 1
            self.prior_total += 1
            self.baseline_values.add(value)
        else:
            self.recent_pattern = DECAY * self.recent_pattern + pattern_vec.astype(np.float64)
            self.recent_rate = DECAY * self.recent_rate + rate_vec.astype(np.float64)
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
        self._prior_pattern_norm = self._normalize(self.prior_pattern)
        self._prior_rate_norm = self._normalize(self.prior_rate)
        self.recent_pattern = self.prior_pattern.copy().astype(np.float64)
        self.recent_rate = self.prior_rate.copy().astype(np.float64)
        self.recent_counts = dict(self.prior_counts)
        self.recent_total = self.prior_total

    def get_signal(self, current_pps: float) -> Optional[dict]:
        if not self._frozen:
            return None

        recent_pattern_norm = self._normalize(self.recent_pattern)
        pattern_divergence = 1.0 - self._similarity(self._prior_pattern_norm, recent_pattern_norm)

        if pattern_divergence < 0.15:
            return None

        if self.recent_counts:
            dominant_value, dominant_count = max(self.recent_counts.items(), key=lambda x: x[1])
            concentration = dominant_count / self.recent_total if self.recent_total > 0 else 0
        else:
            dominant_value, concentration = None, 0

        is_novel = dominant_value not in self.baseline_values if dominant_value else False

        # Decode rates using learned decoder
        baseline_rate_pps, baseline_conf = self.decoder.decode(self._prior_rate_norm)
        current_rate_vec = self._encode_rate(current_pps)
        current_rate_pps, current_conf = self.decoder.decode(current_rate_vec)

        return {
            "field": self.field_name,
            "value": dominant_value,
            "is_novel": is_novel,
            "pattern_divergence": round(pattern_divergence, 3),
            "concentration": round(concentration, 3),
            "baseline_rate_pps": round(baseline_rate_pps, 1),
            "current_rate_pps": round(current_rate_pps, 1),
            "enforce_rate_pps": round(baseline_rate_pps, 1),
        }


# =============================================================================
# DETECTOR WITH LEARNED REFERENCES
# =============================================================================


class LearnedRateLimitDetector:
    """
    Detector that learns rate references from observation.
    No hardcoded reference rates.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.decoder = LearnedRateDecoder(self.store)

        self.field_trackers = {
            f: UnifiedFieldTracker(f, self.store, self.decoder)
            for f in MONITORED_FIELDS
        }

        self._warmup_count = 0
        self._warmup_complete = False

    def process(self, packet: Packet, simulated_pps: float) -> List[dict]:
        is_warmup = not self._warmup_complete

        # Observe rate during warmup
        if is_warmup:
            self.decoder.observe(simulated_pps)

        for field_name, tracker in self.field_trackers.items():
            value = packet.get_field(field_name)
            if value is not None:
                tracker.observe(value, is_warmup, simulated_pps)

        if is_warmup:
            self._warmup_count += 1
            if self._warmup_count >= WARMUP_PACKETS:
                self._warmup_complete = True
                self.decoder.freeze()
                for tracker in self.field_trackers.values():
                    tracker.freeze()
            return []

        signals = []
        for tracker in self.field_trackers.values():
            signal = tracker.get_signal(simulated_pps)
            if signal:
                signals.append(signal)

        return signals


# =============================================================================
# EXPERIMENTS
# =============================================================================


def test_rate_learning(baseline_pps: float = 100.0, variance: float = 20.0):
    """Test that we learn rate references from observation."""
    print("="*70)
    print(f"RATE REFERENCE LEARNING TEST")
    print(f"Baseline: {baseline_pps} pps (±{variance} variance)")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)
    decoder = LearnedRateDecoder(store)
    rng = random.Random(42)

    # Simulate warmup observations with some variance
    print(f"\nObserving {WARMUP_PACKETS} rate samples...")
    for _ in range(WARMUP_PACKETS):
        # Simulate rate with natural variance
        observed_pps = baseline_pps + rng.uniform(-variance, variance)
        decoder.observe(observed_pps)

    decoder.freeze()

    print(f"\n{decoder.describe()}")

    # Test decoding at various rates
    print("\nDecoding test:")
    print(f"{'Input PPS':>12} {'Decoded PPS':>14} {'Confidence':>12}")
    print("-"*45)

    test_rates = [
        baseline_pps / 10,
        baseline_pps / 2,
        baseline_pps,
        baseline_pps * 2,
        baseline_pps * 10,
        baseline_pps * 100,
    ]

    for pps in test_rates:
        vec = store.encode_scalar_log(float(pps))
        decoded, conf = decoder.decode(vec)
        print(f"{pps:>12.1f} {decoded:>14.1f} {conf:>12.3f}")


def run_experiment(attack_type: str, baseline_pps: float = 100.0, attack_pps: float = 10000.0):
    print(f"\n{'='*70}")
    print(f"ATTACK: {attack_type}")
    print(f"Baseline: {baseline_pps} pps → Attack: {attack_pps} pps")
    print("="*70)

    detector = LearnedRateLimitDetector()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Warmup with slight rate variance (realistic)
    for _ in range(WARMUP_PACKETS):
        simulated_pps = baseline_pps + rng.uniform(-10, 10)
        detector.process(gen_normal(rng), simulated_pps=simulated_pps)

    # Show learned references
    print(f"\nLearned rate references:")
    print(f"  {detector.decoder.describe()}")

    # Normal
    for _ in range(200):
        detector.process(gen_normal(rng), simulated_pps=baseline_pps)

    # Attack
    all_signals = []
    for _ in range(500):
        if rng.random() < 0.9:
            packet = attack_gen(rng)
        else:
            packet = gen_normal(rng)
        signals = detector.process(packet, simulated_pps=attack_pps)
        all_signals.extend(signals)

    # Collect unique signals
    final_signals = {}
    for sig in all_signals:
        final_signals[sig["field"]] = sig

    print(f"\nRATE LIMIT SIGNALS:")
    print("-"*70)

    for field, sig in sorted(final_signals.items()):
        status = "NOVEL" if sig["is_novel"] else f"{sig['concentration']*100:.0f}% concentration"
        print(f"""
  {sig['field']}={sig['value']}
    Status:           {status}
    Pattern shift:    {sig['pattern_divergence']*100:.0f}%
    Baseline rate:    {sig['baseline_rate_pps']} pps
    Current rate:     {sig['current_rate_pps']} pps
    → ENFORCE:        {sig['enforce_rate_pps']} pps""")

    # JSON output
    print(f"\n{'='*70}")
    print("ENFORCER JSON:")
    print("="*70)

    enforcer_rules = []
    for field, sig in final_signals.items():
        if field == "src_port" and isinstance(sig["value"], int) and sig["value"] >= 49152:
            continue
        enforcer_rules.append({
            "match": {sig["field"]: sig["value"]},
            "action": "rate_limit",
            "rate_pps": sig["enforce_rate_pps"],
            "reason": f"{sig['field']}={sig['value']} anomalous (baseline: {sig['baseline_rate_pps']} pps)",
        })

    print(json.dumps(enforcer_rules, indent=2))


def test_different_baselines():
    """Test that learning works at different baseline rates."""
    print("\n" + "="*70)
    print("TESTING DIFFERENT BASELINE RATES")
    print("="*70)
    print("""
    Different apps have different traffic profiles.
    The detector should learn from whatever baseline it observes.
    """)

    for baseline in [10.0, 500.0, 50000.0]:
        test_rate_learning(baseline_pps=baseline, variance=baseline * 0.1)
        print()


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 008: Learn Rate References from Observation")
    print("="*70)
    print("""
    BEFORE: Hardcoded REFERENCE_RATES = [1, 5, 10, 50, 100, ...]

    AFTER: Learn reference rates from baseline observations:
    1. Observe actual PPS values during warmup
    2. Compute distribution: min, p10, p25, p50, p75, p90, max
    3. Project attack scales: 2x, 5x, 10x, 50x, 100x median
    4. Build reference vectors from LEARNED rates

    Zero hardcoded rate knowledge. Works for any app.
    """)

    # First, show learning works for different baselines
    test_different_baselines()

    # Then run attack experiments
    for attack_type in ATTACK_GENERATORS:
        run_experiment(attack_type)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
    Rate references are now LEARNED, not hardcoded:

    BEFORE:
      REFERENCE_RATES = [1, 5, 10, 50, 100, 500, 1000, ...]  # Magic!

    AFTER:
      - Observe actual rates during warmup
      - Build references from percentiles: p10, p25, p50, p75, p90
      - Project attack scales from observed median

    Works for any baseline:
      - 10 pps service → learns references around 10 pps
      - 50,000 pps API → learns references around 50,000 pps

    ZERO HARDCODED RATE KNOWLEDGE.
    """)


if __name__ == "__main__":
    main()
