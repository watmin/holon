#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 DEMO: Zero-Hardcode Detection with WALKABLE Interface
=============================================================================

This is the same zero-hardcode anomaly detector as DEMO-zero-hardcode-detection.py,
but using the NEW WALKABLE INTERFACE instead of dict serialization.

WHAT'S DIFFERENT:
- Packets are typed structs (TCPPacket, UDPPacket, ICMPPacket)
- Uses encode_walkable() instead of encode_data()
- ZERO serialization - structs are walked directly
- Type safety without sacrificing flexibility

WHAT'S THE SAME:
- Detection logic unchanged
- Zero hardcoded domain knowledge
- 100% attack recall, ~4% false positive rate

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/DEMO-walkable-detection.py
"""

import sys
import random
from dataclasses import dataclass
from typing import Optional, Iterator, Tuple, Any, Union
from collections import deque
from enum import Enum
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore, Walkable, WalkType


# =============================================================================
# WALKABLE PACKET TYPES
# =============================================================================


class TCPPacket(Walkable):
    """TCP packet - implements Walkable for zero-serialization encoding."""

    __slots__ = ("protocol", "src_port", "dst_port", "flags", "payload_size")

    def __init__(
        self,
        src_port: int,
        dst_port: int,
        flags: str,
        payload_size: int,
    ):
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

    def __repr__(self):
        return f"TCP({self.src_port}→{self.dst_port}, {self.flags})"


class UDPPacket(Walkable):
    """UDP packet - implements Walkable for zero-serialization encoding."""

    __slots__ = ("protocol", "src_port", "dst_port", "payload_size")

    def __init__(
        self,
        src_port: int,
        dst_port: int,
        payload_size: int,
    ):
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

    def __repr__(self):
        return f"UDP({self.src_port}→{self.dst_port})"


class ICMPPacket(Walkable):
    """ICMP packet - implements Walkable for zero-serialization encoding."""

    __slots__ = ("protocol", "icmp_type", "payload_size")

    def __init__(
        self,
        icmp_type: int,
        payload_size: int,
    ):
        self.protocol = "ICMP"
        self.icmp_type = icmp_type
        self.payload_size = payload_size

    def walk_type(self) -> WalkType:
        return WalkType.MAP

    def walk_map_items(self) -> Iterator[Tuple[str, Any]]:
        yield "protocol", self.protocol
        yield "icmp_type", self.icmp_type
        yield "payload_size", self.payload_size

    def __repr__(self):
        return f"ICMP(type={self.icmp_type})"


# Type alias for any packet
Packet = Union[TCPPacket, UDPPacket, ICMPPacket]


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
WARMUP_PACKETS = 400
DECAY = 0.98


# =============================================================================
# FROZEN Z-SCORE BASELINE
# =============================================================================


class FrozenBaseline:
    """
    Learns mean/std during warmup, then FREEZES.
    All subsequent comparisons are against the frozen baseline.
    """

    def __init__(self):
        self.samples = []
        self.frozen = False
        self.mean = 0.0
        self.std = 1.0

    def observe(self, value: float):
        if not self.frozen:
            self.samples.append(value)

    def freeze(self):
        self.frozen = True
        if self.samples:
            self.mean = np.mean(self.samples)
            self.std = max(np.std(self.samples) if len(self.samples) > 1 else 0.05, 0.02)

    def z_score(self, value: float) -> float:
        return (value - self.mean) / self.std


# =============================================================================
# ZERO-HARDCODE DETECTOR (WALKABLE VERSION)
# =============================================================================


class WalkableDetector:
    """
    Anomaly detector using the WALKABLE interface.

    Key difference from the original:
    - Uses encode_walkable() instead of encode_data()
    - Accepts typed Packet objects instead of dicts
    - ZERO serialization overhead
    """

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Rate tracking
        self.rate_accum = self.encoder.create_accumulator()
        self._rate_norm = None
        self.rate_baseline = FrozenBaseline()

        # Pattern tracking
        self.pattern_accum = self.encoder.create_accumulator()
        self._pattern_norm = None
        self.pattern_baseline = FrozenBaseline()

        # Recent pattern tracking
        self.recent_pattern = self.encoder.create_accumulator()

        # State machine
        self.in_anomaly_state = False
        self.anomaly_window = deque(maxlen=15)
        self.consecutive_normal = 0

    def process(self, packet: Packet, pps: float) -> dict:
        """
        Process a typed packet and detect anomalies.

        Args:
            packet: A Walkable packet (TCPPacket, UDPPacket, or ICMPPacket)
            pps: Packets per second (rate)

        Returns:
            Detection result with explanation
        """
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        # WALKABLE ENCODING - no dict conversion needed!
        packet_vec = self.encoder.encode_walkable(packet)
        rate_vec = self.store.encode_scalar_log(pps)

        if is_warmup:
            return self._warmup_phase(packet_vec, rate_vec)
        else:
            return self._detection_phase(packet, packet_vec, rate_vec, pps)

    def _warmup_phase(self, packet_vec, rate_vec) -> dict:
        """Learn what's normal."""
        self.rate_accum = self.encoder.accumulate(self.rate_accum, rate_vec)
        self.pattern_accum = self.encoder.accumulate(self.pattern_accum, packet_vec)

        if self.packet_count > 100:
            temp_rate = self.encoder.normalize_accumulator(self.rate_accum)
            temp_pattern = self.encoder.normalize_accumulator(self.pattern_accum)

            rate_sim = self.store.similarity(rate_vec, temp_rate, metric="cosine")
            pattern_sim = self.store.similarity(packet_vec, temp_pattern, metric="cosine")

            self.rate_baseline.observe(rate_sim)
            self.pattern_baseline.observe(pattern_sim)

        if self.packet_count == self.warmup_packets:
            self._freeze_baselines()

        return {
            "packet_num": self.packet_count,
            "is_anomalous": False,
            "phase": "warmup",
            "progress": f"{self.packet_count}/{self.warmup_packets}",
        }

    def _freeze_baselines(self):
        """Lock baselines after warmup."""
        self.warmup_complete = True
        self._rate_norm = self.encoder.normalize_accumulator(self.rate_accum)
        self._pattern_norm = self.encoder.normalize_accumulator(self.pattern_accum)
        self.recent_pattern = self.pattern_accum.copy()
        self.rate_baseline.freeze()
        self.pattern_baseline.freeze()

    def _detection_phase(self, packet: Packet, packet_vec, rate_vec, pps) -> dict:
        """Detect anomalies against frozen baseline."""

        self.recent_pattern = DECAY * self.recent_pattern + packet_vec.astype(np.float64)

        rate_sim = self.store.similarity(rate_vec, self._rate_norm, metric="cosine")
        pattern_sim = self.store.similarity(packet_vec, self._pattern_norm, metric="cosine")

        rate_z = self.rate_baseline.z_score(rate_sim)
        pattern_z = self.pattern_baseline.z_score(pattern_sim)

        rate_anomalous = rate_z < -2.5
        pattern_anomalous = pattern_z < -2.0
        rate_confirms = rate_z < -0.5

        raw_anomaly = rate_anomalous or (pattern_anomalous and rate_confirms)

        if raw_anomaly:
            self.consecutive_normal = 0
        else:
            self.consecutive_normal += 1

        if self.consecutive_normal >= 5:
            for _ in range(3):
                self.anomaly_window.append(0)
        else:
            self.anomaly_window.append(1 if raw_anomaly else 0)

        fraction = sum(self.anomaly_window) / len(self.anomaly_window) if self.anomaly_window else 0

        if not self.in_anomaly_state:
            if fraction > 0.5:
                self.in_anomaly_state = True
        else:
            if fraction < 0.2:
                self.in_anomaly_state = False

        is_anomalous = self.in_anomaly_state

        explanation = self._build_explanation(
            packet, rate_z, pattern_z, rate_anomalous, pattern_anomalous, pps
        )

        return {
            "packet_num": self.packet_count,
            "is_anomalous": is_anomalous,
            "phase": "detection",
            "rate_z": rate_z,
            "pattern_z": pattern_z,
            "rate_anomalous": rate_anomalous,
            "pattern_anomalous": pattern_anomalous,
            "explanation": explanation,
        }

    def _build_explanation(self, packet: Packet, rate_z, pattern_z, rate_anom, pattern_anom, pps):
        """Build human-readable explanation from typed packet."""
        if not (rate_anom or pattern_anom):
            return "Traffic matches learned baseline"

        parts = []

        if rate_anom:
            parts.append(f"Rate ({pps:.0f} pps) is {abs(rate_z):.1f} std below baseline")

        if pattern_anom:
            # Build field description from Walkable interface
            fields = []
            for key, value in packet.walk_map_items():
                if key != "payload_size":
                    fields.append(f"{key}={value}")
            parts.append(f"Pattern [{', '.join(fields)}] is {abs(pattern_z):.1f} std below baseline")

        return "; ".join(parts)


# =============================================================================
# TRAFFIC SIMULATION (WALKABLE VERSION)
# =============================================================================


class Phase(Enum):
    CALM = "calm"
    ATTACK = "attack"


@dataclass
class TimePhase:
    name: str
    duration_seconds: int
    packets_per_second: int
    phase_type: Phase
    attack_type: str = None
    attack_fraction: float = 0.95


def gen_normal(rng: random.Random) -> Packet:
    """Generate normal traffic as typed Packet objects."""
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.8, 0.18, 0.02])[0]

    if proto == "TCP":
        return TCPPacket(
            src_port=rng.randint(49152, 65535),
            dst_port=rng.choice([80, 443, 8080, 22]),
            flags=rng.choices(["PA", "A", "SA", "S"], weights=[0.4, 0.3, 0.2, 0.1])[0],
            payload_size=rng.randint(0, 1500),
        )
    elif proto == "UDP":
        return UDPPacket(
            src_port=rng.randint(49152, 65535),
            dst_port=rng.choice([53, 443, 123]),
            payload_size=rng.randint(20, 512),
        )
    else:
        return ICMPPacket(
            icmp_type=rng.choice([0, 8]),
            payload_size=64,
        )


# Attack generators - return typed Packet objects
def gen_dns_reflection(rng: random.Random) -> UDPPacket:
    return UDPPacket(
        src_port=53,
        dst_port=rng.randint(49152, 65535),
        payload_size=rng.randint(256, 4096),
    )


def gen_syn_flood(rng: random.Random) -> TCPPacket:
    return TCPPacket(
        src_port=rng.randint(1, 65535),
        dst_port=80,
        flags="S",
        payload_size=0,
    )


def gen_ntp_amplification(rng: random.Random) -> UDPPacket:
    return UDPPacket(
        src_port=123,
        dst_port=rng.randint(49152, 65535),
        payload_size=rng.randint(468, 482),
    )


def gen_udp_flood(rng: random.Random) -> UDPPacket:
    return UDPPacket(
        src_port=rng.randint(1, 65535),
        dst_port=rng.randint(1, 65535),
        payload_size=rng.randint(0, 1400),
    )


ATTACKS = {
    "dns_reflection": gen_dns_reflection,
    "syn_flood": gen_syn_flood,
    "ntp_amplification": gen_ntp_amplification,
    "udp_flood": gen_udp_flood,
}


# =============================================================================
# DEMO
# =============================================================================


def run_demo():
    print("=" * 75)
    print("BATCH 012 DEMO: Zero-Hardcode Detection with WALKABLE Interface")
    print("=" * 75)
    print("""
    This is the WALKABLE version of zero-hardcode detection.

    KEY DIFFERENCES FROM DICT VERSION:
    1. Packets are TYPED STRUCTS (TCPPacket, UDPPacket, ICMPPacket)
    2. Uses encode_walkable() instead of encode_data()
    3. ZERO SERIALIZATION - structs are walked directly
    4. Type safety with IDE autocomplete and error checking

    THE SAME:
    - Detection logic unchanged
    - Zero hardcoded domain knowledge
    - Same accuracy (100% recall, ~4% FP)

    WHY WALKABLE?
    - No JSON/dict conversion overhead
    - Works with your existing data models
    - Ready for Rust port (trait Walkable { ... })
    """)

    # Demo the Walkable packets
    print("\n" + "-" * 75)
    print("WALKABLE PACKET TYPES")
    print("-" * 75)

    tcp = TCPPacket(src_port=54321, dst_port=443, flags="PA", payload_size=1200)
    udp = UDPPacket(src_port=53, dst_port=12345, payload_size=512)
    icmp = ICMPPacket(icmp_type=8, payload_size=64)

    print(f"\n  TCPPacket:  {tcp}")
    print(f"  UDPPacket:  {udp}")
    print(f"  ICMPPacket: {icmp}")

    print("\n  Walking TCPPacket:")
    for key, value in tcp.walk_map_items():
        print(f"    {key}: {value}")

    # Compare encoding methods
    print("\n" + "-" * 75)
    print("WALKABLE vs DICT ENCODING")
    print("-" * 75)

    store = CPUStore(dimensions=DIMENSIONS)

    # Create equivalent dict
    tcp_dict = {
        "protocol": "TCP",
        "src_port": 54321,
        "dst_port": 443,
        "flags": "PA",
        "payload_size": 1200,
    }

    vec_walkable = store.encoder.encode_walkable(tcp)
    vec_dict = store.encoder.encode_data(tcp_dict)

    # They should be identical!
    identical = np.array_equal(vec_walkable, vec_dict)
    similarity = store.similarity(vec_walkable, vec_dict, metric="cosine")

    print(f"\n  TCPPacket (Walkable) vs dict encoding:")
    print(f"    Vectors identical: {identical}")
    print(f"    Similarity: {similarity:.6f}")

    if identical:
        print("\n  ✓ Walkable produces IDENTICAL vectors to dict encoding!")
    else:
        print(f"\n  Vectors differ slightly (similarity: {similarity:.6f})")

    # Performance note
    print("""
  PERFORMANCE NOTE:
  In Python, dict encoding is faster than Walkable due to CPython optimizations.
  The Walkable interface provides:
  - Type safety and IDE support (worth the overhead in development)
  - Zero-cost abstraction in Rust (trait dispatch is free)
  - Avoids JSON string serialization (the original bottleneck)
    """)

    # Define attack scenario
    timeline = [
        TimePhase("warmup", 600, 100, Phase.CALM),
        TimePhase("DNS Attack", 30, 100000, Phase.ATTACK, "dns_reflection"),
        TimePhase("recovery-1", 300, 100, Phase.CALM),
        TimePhase("SYN Flood", 45, 80000, Phase.ATTACK, "syn_flood"),
        TimePhase("recovery-2", 300, 100, Phase.CALM),
        TimePhase("NTP Attack", 30, 100000, Phase.ATTACK, "ntp_amplification"),
        TimePhase("final", 300, 100, Phase.CALM),
    ]

    print("\n" + "-" * 75)
    print("ATTACK SIMULATION (using Walkable packets)")
    print("-" * 75)
    print("\n  Timeline:")
    print(f"  {'Phase':<15} {'PPS':>10} {'Type':<12} {'Packets':>10}")
    print("  " + "-" * 50)

    scale = 0.005
    rng = random.Random(42)

    for phase in timeline:
        scaled = int(phase.duration_seconds * phase.packets_per_second * scale)
        ptype = phase.attack_type or "normal"
        print(f"  {phase.name:<15} {phase.packets_per_second:>10,} {ptype:<12} {scaled:>10,}")

    # Run detection
    print("\n" + "-" * 75)
    print("DETECTION RESULTS")
    print("-" * 75)

    first_calm_packets = int(timeline[0].duration_seconds * timeline[0].packets_per_second * scale)
    warmup_packets = min(first_calm_packets - 10, 400)

    detector = WalkableDetector(warmup_packets=warmup_packets)

    results = []
    sample_alerts = []
    packet_types_seen = {"TCP": 0, "UDP": 0, "ICMP": 0}

    for phase in timeline:
        scaled_packets = max(1, int(phase.duration_seconds * phase.packets_per_second * scale))
        phase_detections = 0

        attack_gen = ATTACKS.get(phase.attack_type) if phase.attack_type else None

        for i in range(scaled_packets):
            # Generate TYPED packet
            if phase.phase_type == Phase.ATTACK and attack_gen and rng.random() < phase.attack_fraction:
                packet = attack_gen(rng)
            else:
                packet = gen_normal(rng)

            # Track packet types
            packet_types_seen[packet.protocol] += 1

            # Detect using Walkable interface
            result = detector.process(packet, phase.packets_per_second)

            if detector.warmup_complete:
                if result["is_anomalous"]:
                    phase_detections += 1

                    if len(sample_alerts) < 5 and result.get("explanation"):
                        sample_alerts.append({
                            "packet": result["packet_num"],
                            "phase": phase.name,
                            "packet_type": type(packet).__name__,
                            "packet_repr": repr(packet),
                            "explanation": result["explanation"],
                        })

        detection_rate = phase_detections / scaled_packets if scaled_packets > 0 else 0

        if phase.name == "warmup":
            status = "LEARNING"
        elif phase.phase_type == Phase.ATTACK:
            status = "DETECTED" if detection_rate > 0.5 else "MISSED"
        else:
            status = "CLEAN" if detection_rate < 0.05 else "FP"

        results.append({
            "name": phase.name,
            "phase_type": phase.phase_type,
            "packets": scaled_packets,
            "detections": phase_detections,
            "detection_rate": detection_rate,
            "status": status,
        })

    # Print results
    print(f"\n  {'Phase':<15} {'Packets':>10} {'Detected':>10} {'Rate':>10} {'Status':>12}")
    print("  " + "-" * 60)

    for r in results:
        marker = {"DETECTED": "✓", "CLEAN": "✓", "LEARNING": "○", "FP": "⚠", "MISSED": "✗"}
        print(f"  {r['name']:<15} {r['packets']:>10,} {r['detections']:>10,} "
              f"{r['detection_rate']:>9.0%} {marker.get(r['status'], '?')} {r['status']}")

    # Metrics
    attack_phases = [r for r in results if r["phase_type"] == Phase.ATTACK]
    calm_phases = [r for r in results if r["phase_type"] == Phase.CALM and r["status"] != "LEARNING"]

    attack_detected = sum(r["detections"] for r in attack_phases)
    attack_total = sum(r["packets"] for r in attack_phases)
    attack_recall = attack_detected / attack_total if attack_total > 0 else 0

    fp = sum(r["detections"] for r in calm_phases)
    fp_total = sum(r["packets"] for r in calm_phases)
    fp_rate = fp / fp_total if fp_total > 0 else 0

    print("  " + "-" * 60)
    print(f"  {'ATTACK RECALL':<37} {attack_recall:>9.0%}")
    print(f"  {'FALSE POSITIVE RATE':<37} {fp_rate:>9.0%}")

    # Packet type breakdown
    print("\n" + "-" * 75)
    print("PACKET TYPES PROCESSED (typed Walkable structs)")
    print("-" * 75)
    total_packets = sum(packet_types_seen.values())
    for ptype, count in packet_types_seen.items():
        pct = count / total_packets * 100 if total_packets > 0 else 0
        print(f"  {ptype + 'Packet':<15} {count:>8,} ({pct:>5.1f}%)")

    # Sample alerts
    print("\n" + "-" * 75)
    print("SAMPLE ALERTS (with typed packet info)")
    print("-" * 75)

    for alert in sample_alerts[:3]:
        print(f"\n  Packet #{alert['packet']} ({alert['phase']}):")
        print(f"    Type: {alert['packet_type']}")
        print(f"    Repr: {alert['packet_repr']}")
        print(f"    Alert: {alert['explanation']}")

    # Summary
    print("\n" + "=" * 75)
    print("SUMMARY: WALKABLE DETECTION")
    print("=" * 75)
    print(f"""
    RESULTS:
    - Attack Recall: {attack_recall:.0%}
    - False Positive Rate: {fp_rate:.0%}

    WALKABLE BENEFITS:
    1. TYPE SAFETY - TCPPacket, UDPPacket, ICMPPacket with proper fields
    2. IDE SUPPORT - autocomplete, type checking, refactoring
    3. NO JSON PARSING - avoids string→parse→dict overhead
    4. RUST READY - trait Walkable is zero-cost (unlike Python iterators)
    5. WORKS WITH YOUR MODELS - no manual dict conversion needed

    ENCODING EQUIVALENCE:
    - Walkable structs produce IDENTICAL vectors to dicts
    - Same detection quality, better developer experience

    HOLON API USED:
    - encoder.encode_walkable(packet)  # NEW! Walks structs directly
    - store.encode_scalar_log(pps)     # Log-scale rate encoding
    - store.similarity(v1, v2)         # Vector comparison
    """)


if __name__ == "__main__":
    run_demo()
