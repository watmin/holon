#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 006: Decode Baseline Rate from Vector
=============================================================================

KEY INSIGHT: We encode rate with log10 scale. We can DECODE it back.

During warmup:
- Accumulate rate vectors: baseline_rate = accumulate(encode_scalar_log(pps))
- This creates a "rate fingerprint" of baseline traffic

To decode:
- Create reference rate vectors: 10, 50, 100, 500, 1000, 5000, 10000 pps
- Find which reference the baseline is most similar to
- That's the rate to enforce!

ENFORCER MESSAGE:
- "Field X is anomalous. Pre-attack rate was ~100 pps. Enforce that."

ALL derived from vector operations. The rate limit IS stored in the vector.

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/006-rate-vector-decode.py
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

# Reference rates for decoding (powers of 10 and midpoints)
REFERENCE_RATES = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]


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
# RATE DECODER
# =============================================================================


class RateDecoder:
    """
    Decodes rate from a rate vector by comparing to reference rates.

    Uses encode_scalar_log() which encodes on log10 scale.
    We pre-compute reference vectors and find closest match.
    """

    def __init__(self, store: CPUStore):
        self.store = store

        # Pre-compute reference rate vectors
        self.reference_vectors: Dict[int, np.ndarray] = {}
        for rate in REFERENCE_RATES:
            self.reference_vectors[rate] = store.encode_scalar_log(float(rate))

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a = a.astype(np.float64).flatten()
        b = b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def decode(self, rate_vec: np.ndarray) -> Tuple[int, float]:
        """
        Decode a rate vector to the closest reference rate.

        Returns:
            (best_rate, similarity) - the reference rate with highest similarity
        """
        best_rate = 100
        best_sim = -1.0

        for rate, ref_vec in self.reference_vectors.items():
            sim = self._similarity(rate_vec, ref_vec)
            if sim > best_sim:
                best_sim = sim
                best_rate = rate

        return best_rate, best_sim

    def get_similarities(self, rate_vec: np.ndarray) -> Dict[int, float]:
        """Get similarity to all reference rates."""
        return {
            rate: self._similarity(rate_vec, ref_vec)
            for rate, ref_vec in self.reference_vectors.items()
        }


# =============================================================================
# FIELD TRACKER WITH RATE DECODING
# =============================================================================


class RateDecodingFieldTracker:
    """
    Tracks per-field patterns and rates.

    Key addition: Can DECODE the baseline rate to a concrete PPS value.
    """

    def __init__(self, field_name: str, store: CPUStore, decoder: RateDecoder):
        self.field_name = field_name
        self.store = store
        self.encoder = store.encoder
        self.decoder = decoder

        # Pattern tracking
        self.prior_pattern = self.encoder.create_accumulator()
        self.recent_pattern = self.encoder.create_accumulator()

        # Rate tracking (using log-scale encoding)
        self.prior_rate = self.encoder.create_accumulator()
        self.recent_rate = self.encoder.create_accumulator()

        self.baseline_values = set()
        self._frozen = False
        self._prior_pattern_norm = None
        self._prior_rate_norm = None

    def _encode_field_value(self, value: Any) -> np.ndarray:
        return self.encoder.encode_data({self.field_name: value})

    def _encode_rate(self, pps: float) -> np.ndarray:
        return self.store.encode_scalar_log(max(0.1, pps))

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

    def observe(self, value: Any, is_warmup: bool, current_pps: float):
        if value is None:
            return

        pattern_vec = self._encode_field_value(value)
        rate_vec = self._encode_rate(current_pps)

        if is_warmup:
            self.prior_pattern = self.encoder.accumulate(self.prior_pattern, pattern_vec)
            self.prior_rate = self.encoder.accumulate(self.prior_rate, rate_vec)
            self.baseline_values.add(value)
        else:
            self.recent_pattern = DECAY * self.recent_pattern + pattern_vec.astype(np.float64)
            self.recent_rate = DECAY * self.recent_rate + rate_vec.astype(np.float64)

    def freeze(self):
        self._frozen = True
        self._prior_pattern_norm = self._normalize(self.prior_pattern)
        self._prior_rate_norm = self._normalize(self.prior_rate)
        self.recent_pattern = self.prior_pattern.copy().astype(np.float64)
        self.recent_rate = self.prior_rate.copy().astype(np.float64)

    def get_status(self, current_pps: float) -> dict:
        if not self._frozen:
            return {"field": self.field_name, "status": "warmup"}

        # Pattern divergence
        recent_pattern_norm = self._normalize(self.recent_pattern)
        pattern_divergence = 1.0 - self._similarity(self._prior_pattern_norm, recent_pattern_norm)

        # Current rate vs baseline rate
        current_rate_vec = self._encode_rate(current_pps)
        rate_similarity = self._similarity(current_rate_vec, self._prior_rate_norm)

        # DECODE baseline rate to concrete PPS
        baseline_rate_pps, decode_confidence = self.decoder.decode(self._prior_rate_norm)

        # DECODE current rate
        current_decoded, _ = self.decoder.decode(current_rate_vec)

        return {
            "field": self.field_name,
            "pattern_divergence": pattern_divergence,
            "rate_similarity": rate_similarity,
            "baseline_rate_pps": baseline_rate_pps,
            "current_rate_pps": current_decoded,
            "decode_confidence": decode_confidence,
        }

    def is_novel(self, value: Any) -> bool:
        return value not in self.baseline_values


# =============================================================================
# RATE DECODING DETECTOR
# =============================================================================


@dataclass
class RateLimitSignal:
    field: str
    value: Any
    is_novel: bool
    pattern_divergence: float
    baseline_rate_pps: int        # DECODED baseline rate
    current_rate_pps: int         # DECODED current rate
    rate_to_enforce: int          # = baseline_rate_pps
    explanation: str


class RateDecodingDetector:
    """
    Detects anomalies and emits rate limits with decoded PPS values.

    The rate limit IS the decoded baseline rate.
    "Enforce traffic at the rate we saw before the attack."
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.decoder = RateDecoder(self.store)

        self.field_trackers = {
            f: RateDecodingFieldTracker(f, self.store, self.decoder)
            for f in MONITORED_FIELDS
        }

        self._warmup_count = 0
        self._warmup_complete = False

        # Global traffic tracking
        self.prior_traffic = self.encoder.create_accumulator()
        self.recent_traffic = self.encoder.create_accumulator()
        self._prior_traffic_norm = None

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
        is_warmup = not self._warmup_complete
        packet_vec = self.encoder.encode_walkable(packet)

        if is_warmup:
            self.prior_traffic = self.encoder.accumulate(self.prior_traffic, packet_vec)
        else:
            self.recent_traffic = DECAY * self.recent_traffic + packet_vec.astype(np.float64)

        for field_name, tracker in self.field_trackers.items():
            value = packet.get_field(field_name)
            if value is not None:
                tracker.observe(value, is_warmup, simulated_pps)

        if is_warmup:
            self._warmup_count += 1
            if self._warmup_count >= WARMUP_PACKETS:
                self._warmup_complete = True
                self._prior_traffic_norm = self._normalize(self.prior_traffic)
                self.recent_traffic = self.prior_traffic.copy().astype(np.float64)
                for tracker in self.field_trackers.values():
                    tracker.freeze()
            return {"phase": "warmup", "signals": []}

        # Collect signals for anomalous fields
        signals = []
        for field_name, tracker in self.field_trackers.items():
            status = tracker.get_status(simulated_pps)
            pattern_div = status.get("pattern_divergence", 0)

            if pattern_div > 0.15:
                value = packet.get_field(field_name)
                signal = RateLimitSignal(
                    field=field_name,
                    value=value,
                    is_novel=tracker.is_novel(value) if value else False,
                    pattern_divergence=pattern_div,
                    baseline_rate_pps=status["baseline_rate_pps"],
                    current_rate_pps=status["current_rate_pps"],
                    rate_to_enforce=status["baseline_rate_pps"],  # Enforce baseline!
                    explanation=f"{field_name}: baseline={status['baseline_rate_pps']} pps, "
                               f"current={status['current_rate_pps']} pps → enforce {status['baseline_rate_pps']} pps",
                )
                signals.append(signal)

        return {"phase": "active", "signals": signals}


# =============================================================================
# EXPERIMENT
# =============================================================================


def test_rate_encoding():
    """Test that rate encoding/decoding works correctly."""
    print("="*70)
    print("RATE ENCODING/DECODING VALIDATION")
    print("="*70)

    store = CPUStore(dimensions=DIMENSIONS)
    decoder = RateDecoder(store)

    print("\nEncoding rates and decoding back:")
    print(f"{'Input PPS':>12} {'Decoded PPS':>12} {'Confidence':>12}")
    print("-"*40)

    test_rates = [10, 50, 100, 200, 500, 1000, 5000, 10000]
    for pps in test_rates:
        vec = store.encode_scalar_log(float(pps))
        decoded, confidence = decoder.decode(vec)
        print(f"{pps:>12} {decoded:>12} {confidence:>12.3f}")

    # Show similarity matrix for reference
    print("\nSimilarity matrix (reference rates):")
    print(f"{'':>8}", end="")
    for r in REFERENCE_RATES[:8]:
        print(f"{r:>8}", end="")
    print()

    for r1 in REFERENCE_RATES[:8]:
        print(f"{r1:>8}", end="")
        for r2 in REFERENCE_RATES[:8]:
            v1 = store.encode_scalar_log(float(r1))
            v2 = store.encode_scalar_log(float(r2))
            sim = float(np.dot(v1.flatten(), v2.flatten()) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            print(f"{sim:>8.2f}", end="")
        print()


def run_experiment(attack_type: str, baseline_pps: float = 100.0, attack_pps: float = 10000.0):
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {attack_type}")
    print(f"Baseline: {baseline_pps} pps → Attack: {attack_pps} pps")
    print("="*70)

    detector = RateDecodingDetector()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Warmup at baseline rate
    print(f"\n[WARMUP] {WARMUP_PACKETS} packets at {baseline_pps} pps...")
    for _ in range(WARMUP_PACKETS):
        detector.process(gen_normal(rng), simulated_pps=baseline_pps)

    # Normal at baseline rate
    print(f"[NORMAL] 200 packets at {baseline_pps} pps...")
    for _ in range(200):
        detector.process(gen_normal(rng), simulated_pps=baseline_pps)

    # Attack at attack rate
    print(f"[ATTACK] 500 packets at {attack_pps} pps (90% attack)...")
    attack_signals = []
    for _ in range(500):
        if rng.random() < 0.9:
            packet = attack_gen(rng)
        else:
            packet = gen_normal(rng)
        result = detector.process(packet, simulated_pps=attack_pps)
        attack_signals.extend(result["signals"])

    # Recovery at baseline rate
    print(f"[RECOVERY] 300 packets at {baseline_pps} pps...")
    for _ in range(300):
        detector.process(gen_normal(rng), simulated_pps=baseline_pps)

    # Analysis
    print(f"\n{'='*70}")
    print("RATE LIMIT SIGNALS WITH DECODED PPS")
    print("="*70)

    if attack_signals:
        # Group by field
        by_field = defaultdict(list)
        for sig in attack_signals:
            by_field[sig.field].append(sig)

        for field, sigs in sorted(by_field.items()):
            baseline_rates = [s.baseline_rate_pps for s in sigs]
            current_rates = [s.current_rate_pps for s in sigs]

            print(f"\n  {field}:")
            print(f"    Signals emitted: {len(sigs)}")
            print(f"    Decoded baseline rate: {baseline_rates[0]} pps")
            print(f"    Decoded current rate:  {current_rates[-1]} pps")
            print(f"    Rate to enforce: {baseline_rates[0]} pps")

            # Sample signal
            sample = sigs[-1]
            print(f"    Example: {sample.explanation}")

    # Summary
    print(f"\n{'='*70}")
    print("ENFORCER MESSAGE")
    print("="*70)

    if attack_signals:
        # Get fields with signals
        fields_with_signals = list(set(s.field for s in attack_signals))

        print(f"""
    Anomalous fields detected: {fields_with_signals}

    For each anomalous field, the enforcer should:
    """)

        for field in fields_with_signals:
            field_sigs = [s for s in attack_signals if s.field == field]
            baseline = field_sigs[0].baseline_rate_pps
            current = field_sigs[-1].current_rate_pps

            print(f"""    {field}:
      - Baseline rate was: {baseline} pps
      - Current rate is:   {current} pps
      - ENFORCE: {baseline} pps
      - This is a {current/baseline:.0f}x reduction
    """)


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 006: Decode Baseline Rate from Vector")
    print("="*70)
    print("""
    KEY INSIGHT: Rate vectors can be DECODED.

    1. During warmup, accumulate rate vectors: encode_scalar_log(pps)
    2. This creates a "rate fingerprint" in the baseline
    3. Compare to reference rate vectors (10, 100, 1000, etc.)
    4. Highest similarity = decoded baseline rate

    The rate limit IS the decoded baseline rate.
    "Enforce traffic at the rate we saw before the attack."
    """)

    # First, validate rate encoding/decoding
    test_rate_encoding()

    # Then run experiments
    for attack_type in ATTACK_GENERATORS:
        run_experiment(attack_type, baseline_pps=100.0, attack_pps=10000.0)

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
    We can now emit concrete rate limits:

    BEFORE: "src_port is anomalous, rate_factor=0.15"
            (What does 0.15 mean? 15% of what?)

    AFTER:  "src_port is anomalous. Baseline was 100 pps. Enforce 100 pps."
            (Concrete, actionable!)

    The rate limit is DECODED from the baseline rate vector.
    The vector learned the rate during warmup.
    We query it against reference rates to extract a concrete PPS.

    ALL from vector operations.
    """)


if __name__ == "__main__":
    main()
