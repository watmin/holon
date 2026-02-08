#!/usr/bin/env python3
"""
=============================================================================
BATCH 013 - CHALLENGE 007: Unified Rate Limit Signal
=============================================================================

Combines batch 012 field-level anomaly detection with rate vector decoding.

OUTPUT: A complete rate limit signal containing:
1. WHAT is anomalous (field + value from batch 012)
2. WHY it's anomalous (novelty, concentration, divergence)
3. AT WHAT RATE to limit (decoded from baseline rate vector)

Example signal:
{
    "field": "src_port",
    "value": 53,
    "is_novel": true,
    "pattern_divergence": 0.85,
    "concentration": 0.96,
    "baseline_rate_pps": 100,
    "current_rate_pps": 10000,
    "enforce_rate_pps": 100,
    "explanation": "src_port=53 (novel) at 96% concentration. Enforce 100 pps (baseline)."
}

An enforcer can directly apply this:
- Match: src_port=53
- Action: rate limit
- Rate: 100 pps

Run: ./scripts/run_with_venv.sh python scripts/challenges/013-batch/007-unified-rate-limit-signal.py
"""

import sys
import random
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
    def __init__(self, store: CPUStore):
        self.store = store
        self.reference_vectors = {
            rate: store.encode_scalar_log(float(rate))
            for rate in REFERENCE_RATES
        }

    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-10 or nb < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def decode(self, rate_vec: np.ndarray) -> int:
        best_rate, best_sim = 100, -1.0
        for rate, ref_vec in self.reference_vectors.items():
            sim = self._similarity(rate_vec, ref_vec)
            if sim > best_sim:
                best_sim, best_rate = sim, rate
        return best_rate


# =============================================================================
# UNIFIED FIELD TRACKER
# =============================================================================


class UnifiedFieldTracker:
    """
    Tracks pattern AND rate for a field.
    Combines batch 012 pattern detection with batch 013 rate decoding.
    """

    def __init__(self, field_name: str, store: CPUStore, decoder: RateDecoder):
        self.field_name = field_name
        self.store = store
        self.encoder = store.encoder
        self.decoder = decoder

        # Pattern accumulator (from batch 012)
        self.prior_pattern = self.encoder.create_accumulator()
        self.recent_pattern = self.encoder.create_accumulator()

        # Rate accumulator (from batch 013)
        self.prior_rate = self.encoder.create_accumulator()
        self.recent_rate = self.encoder.create_accumulator()

        # Value counting (for concentration)
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

            # Periodic decay of counts
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
        """
        Get rate limit signal if field is anomalous.
        Returns None if field is normal.
        """
        if not self._frozen:
            return None

        # Pattern divergence
        recent_pattern_norm = self._normalize(self.recent_pattern)
        pattern_divergence = 1.0 - self._similarity(self._prior_pattern_norm, recent_pattern_norm)

        # Only emit if divergence is significant
        if pattern_divergence < 0.15:
            return None

        # Get dominant value and concentration
        if self.recent_counts:
            dominant_value, dominant_count = max(self.recent_counts.items(), key=lambda x: x[1])
            concentration = dominant_count / self.recent_total if self.recent_total > 0 else 0
        else:
            dominant_value, concentration = None, 0

        # Is dominant value novel?
        is_novel = dominant_value not in self.baseline_values if dominant_value else False

        # Decode rates
        baseline_rate_pps = self.decoder.decode(self._prior_rate_norm)
        current_rate_vec = self._encode_rate(current_pps)
        current_rate_pps = self.decoder.decode(current_rate_vec)

        # The enforce rate IS the baseline rate
        enforce_rate_pps = baseline_rate_pps

        return {
            "field": self.field_name,
            "value": dominant_value,
            "is_novel": is_novel,
            "pattern_divergence": round(pattern_divergence, 3),
            "concentration": round(concentration, 3),
            "baseline_rate_pps": baseline_rate_pps,
            "current_rate_pps": current_rate_pps,
            "enforce_rate_pps": enforce_rate_pps,
        }


# =============================================================================
# UNIFIED DETECTOR
# =============================================================================


class UnifiedRateLimitDetector:
    """
    Complete detector that emits unified rate limit signals.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.decoder = RateDecoder(self.store)

        self.field_trackers = {
            f: UnifiedFieldTracker(f, self.store, self.decoder)
            for f in MONITORED_FIELDS
        }

        self._warmup_count = 0
        self._warmup_complete = False

    def process(self, packet: Packet, simulated_pps: float) -> List[dict]:
        """
        Process packet and return list of rate limit signals.
        """
        is_warmup = not self._warmup_complete

        for field_name, tracker in self.field_trackers.items():
            value = packet.get_field(field_name)
            if value is not None:
                tracker.observe(value, is_warmup, simulated_pps)

        if is_warmup:
            self._warmup_count += 1
            if self._warmup_count >= WARMUP_PACKETS:
                self._warmup_complete = True
                for tracker in self.field_trackers.values():
                    tracker.freeze()
            return []

        # Collect signals from all anomalous fields
        signals = []
        for tracker in self.field_trackers.values():
            signal = tracker.get_signal(simulated_pps)
            if signal:
                signals.append(signal)

        return signals


# =============================================================================
# EXPERIMENT
# =============================================================================


def run_experiment(attack_type: str, baseline_pps: float = 100.0, attack_pps: float = 10000.0):
    print(f"\n{'='*70}")
    print(f"ATTACK: {attack_type}")
    print(f"Baseline: {baseline_pps} pps → Attack: {attack_pps} pps")
    print("="*70)

    detector = UnifiedRateLimitDetector()
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Warmup
    for _ in range(WARMUP_PACKETS):
        detector.process(gen_normal(rng), simulated_pps=baseline_pps)

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

    # Collect unique signals (last one per field)
    final_signals = {}
    for sig in all_signals:
        final_signals[sig["field"]] = sig

    # Output
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

    # JSON output for enforcer
    print(f"\n{'='*70}")
    print("ENFORCER JSON:")
    print("="*70)

    enforcer_rules = []
    for field, sig in final_signals.items():
        # Skip ephemeral ports
        if field == "src_port" and isinstance(sig["value"], int) and sig["value"] >= 49152:
            continue

        enforcer_rules.append({
            "match": {sig["field"]: sig["value"]},
            "action": "rate_limit",
            "rate_pps": sig["enforce_rate_pps"],
            "reason": f"{sig['field']}={sig['value']} anomalous (baseline: {sig['baseline_rate_pps']} pps)",
        })

    print(json.dumps(enforcer_rules, indent=2))


def main():
    print("="*70)
    print("BATCH 013 - CHALLENGE 007: Unified Rate Limit Signal")
    print("="*70)
    print("""
    Combines batch 012 field-level anomaly detection with rate decoding.

    Each signal contains:
    - WHAT: field=value that's anomalous
    - WHY: novelty, concentration, pattern shift
    - RATE: decoded baseline rate to enforce

    The enforcer receives a concrete rule:
    {
        "match": {"src_port": 53},
        "action": "rate_limit",
        "rate_pps": 100
    }
    """)

    for attack_type in ATTACK_GENERATORS:
        run_experiment(attack_type)

    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
    We now emit COMPLETE rate limit signals:

    1. WHAT is anomalous:
       - Field: src_port, dst_port, protocol, flags, etc.
       - Value: 53, 80, "S", "UDP", etc.
       - Novel: was this value seen in baseline?
       - Concentration: how dominant is this value?

    2. HOW MUCH to limit:
       - baseline_rate_pps: decoded from baseline rate vector
       - current_rate_pps: decoded from current rate vector
       - enforce_rate_pps: = baseline_rate_pps

    3. ALL from vector operations:
       - Pattern: similarity(prior_pattern, recent_pattern) → divergence
       - Rate: decode(prior_rate_vec) → baseline PPS
       - No counters except for concentration tracking

    An enforcer can directly apply these rules with no interpretation needed.
    """)


if __name__ == "__main__":
    main()
