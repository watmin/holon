#!/usr/bin/env python3
"""
Challenge 011-008: Realistic Streaming Deployment

Architecture for production-like packet analysis:

PRIOR KNOWLEDGE (frozen, from historical analysis):
├── baseline_vectors: per-perspective normalized vectors
├── expected_values: common field values and their vectors
├── byte_distributions: expected bytes at each position
└── cardinality_baselines: normal cardinality per field

RECENT KNOWLEDGE (adaptive, from recent traffic):
├── recent_accumulators: decaying per-perspective vectors
├── recent_cardinality: sliding window cardinality
└── checkpoints: periodic snapshots of "known good" recent state

STREAMING INTERFACE:
  packet → parse → compare to prior + recent → detect → identify culprits → update recent

CULPRIT IDENTIFICATION:
When anomaly detected, report:
  - Which perspective triggered
  - Which specific field/value is unexpected
  - What was expected vs what was seen
  - Confidence level
"""

import sys
import time
import random
import json
import pickle
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import Counter, deque
from pathlib import Path

import numpy as np

from scapy.all import IP, TCP, UDP, ICMP, Raw, Packet, raw as scapy_raw

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
RECENT_DECAY = 0.995  # ~140 packet half-life


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# PRIOR KNOWLEDGE (frozen baseline from training)
# =============================================================================

@dataclass
class FieldExpectation:
    """Expected values for a field."""
    field_name: str
    common_values: List[str]  # Most common values seen
    value_frequencies: Dict[str, float]  # Value → frequency
    cardinality: float  # Expected cardinality


@dataclass
class BytePositionExpectation:
    """Expected bytes at each position."""
    position: int
    common_bytes: List[int]  # Most common bytes
    byte_frequencies: Dict[int, float]  # Byte → frequency


class PriorKnowledge:
    """
    Frozen baseline knowledge from historical traffic analysis.

    This represents what "normal" looks like - trained offline
    and loaded at runtime.
    """

    def __init__(self, vm: DeterministicVectorManager):
        self.vm = vm
        self.encoder = Encoder(vector_manager=vm)

        # Per-perspective baseline vectors (accumulated from training)
        self.baseline_vectors: Dict[str, np.ndarray] = {}

        # Expected field values
        self.field_expectations: Dict[str, FieldExpectation] = {}

        # Expected byte positions (for payload)
        self.byte_expectations: Dict[int, BytePositionExpectation] = {}

        # Training stats
        self.training_packets = 0

    def train(self, packets: List[Packet]):
        """Build prior knowledge from historical packets."""
        print(f"Training prior knowledge on {len(packets)} packets...")

        # Accumulators for building baselines
        l3_accum = np.zeros(DIMENSIONS, dtype=np.float64)
        l4_accum = np.zeros(DIMENSIONS, dtype=np.float64)
        payload_accum = np.zeros(DIMENSIONS, dtype=np.float64)

        # Field value counters
        field_counters: Dict[str, Counter] = {
            "src_prefix": Counter(),
            "dst_prefix": Counter(),
            "src_port": Counter(),
            "dst_port": Counter(),
            "protocol": Counter(),
            "flags": Counter(),
        }

        # Byte position counters
        byte_counters: Dict[int, Counter] = {i: Counter() for i in range(64)}

        for pkt in packets:
            self.training_packets += 1

            # Parse and encode
            parsed = self._parse(pkt)
            l3_vec = self._encode_l3(parsed)
            l4_vec = self._encode_l4(parsed)
            payload_vec = self._encode_payload(parsed)

            # Accumulate (simple sum for training)
            l3_accum += l3_vec.astype(np.float64)
            l4_accum += l4_vec.astype(np.float64)
            payload_accum += payload_vec.astype(np.float64)

            # Count field values
            if parsed["src_prefix"]:
                field_counters["src_prefix"][parsed["src_prefix"]] += 1
            if parsed["dst_prefix"]:
                field_counters["dst_prefix"][parsed["dst_prefix"]] += 1
            if parsed["src_port"]:
                field_counters["src_port"][str(parsed["src_port"])] += 1
            if parsed["dst_port"]:
                field_counters["dst_port"][str(parsed["dst_port"])] += 1
            if parsed["protocol"]:
                field_counters["protocol"][parsed["protocol"]] += 1
            if parsed["flags"] is not None:
                field_counters["flags"][str(parsed["flags"])] += 1

            # Count byte positions
            for i, byte in enumerate(parsed["payload"][:64]):
                byte_counters[i][byte] += 1

        # Normalize baseline vectors
        for name, accum in [("l3", l3_accum), ("l4", l4_accum), ("payload", payload_accum)]:
            norm = np.linalg.norm(accum)
            if norm > 1e-10:
                self.baseline_vectors[name] = (accum / norm).astype(np.float32)
            else:
                self.baseline_vectors[name] = np.zeros(DIMENSIONS, dtype=np.float32)

        # Build field expectations
        for field_name, counter in field_counters.items():
            if counter:
                total = sum(counter.values())
                common = [v for v, _ in counter.most_common(10)]
                freqs = {v: c/total for v, c in counter.items()}
                card = len(counter) / total

                self.field_expectations[field_name] = FieldExpectation(
                    field_name=field_name,
                    common_values=common,
                    value_frequencies=freqs,
                    cardinality=card,
                )

        # Build byte position expectations
        for pos, counter in byte_counters.items():
            if counter:
                total = sum(counter.values())
                common = [b for b, _ in counter.most_common(5)]
                freqs = {b: c/total for b, c in counter.items()}

                self.byte_expectations[pos] = BytePositionExpectation(
                    position=pos,
                    common_bytes=common,
                    byte_frequencies=freqs,
                )

        print(f"  Trained on {self.training_packets} packets")
        print(f"  Fields tracked: {list(self.field_expectations.keys())}")
        print(f"  Byte positions tracked: {len(self.byte_expectations)}")

    def _parse(self, pkt: Packet) -> dict:
        """Parse packet into field dict."""
        result = {
            "src_ip": "", "dst_ip": "",
            "src_prefix": "", "dst_prefix": "",
            "src_port": 0, "dst_port": 0,
            "protocol": "", "flags": None,
            "payload": b"",
        }

        if IP in pkt:
            result["src_ip"] = pkt[IP].src
            result["dst_ip"] = pkt[IP].dst
            src_parts = pkt[IP].src.split(".")
            dst_parts = pkt[IP].dst.split(".")
            result["src_prefix"] = f"{src_parts[0]}.{src_parts[1]}"
            result["dst_prefix"] = f"{dst_parts[0]}.{dst_parts[1]}"

        if TCP in pkt:
            result["protocol"] = "TCP"
            result["src_port"] = pkt[TCP].sport
            result["dst_port"] = pkt[TCP].dport
            result["flags"] = int(pkt[TCP].flags)
        elif UDP in pkt:
            result["protocol"] = "UDP"
            result["src_port"] = pkt[UDP].sport
            result["dst_port"] = pkt[UDP].dport
        elif ICMP in pkt:
            result["protocol"] = "ICMP"

        if Raw in pkt:
            result["payload"] = bytes(pkt[Raw].load)

        return result

    def _encode_l3(self, parsed: dict) -> np.ndarray:
        if not parsed["src_prefix"]:
            return np.zeros(DIMENSIONS, dtype=np.int8)
        features = {
            "src_prefix": parsed["src_prefix"],
            "dst_prefix": parsed["dst_prefix"],
        }
        return self.encoder.encode_data(features)

    def _encode_l4(self, parsed: dict) -> np.ndarray:
        features = {"protocol": parsed["protocol"]}
        if parsed["protocol"] in ("TCP", "UDP"):
            features["src_port_type"] = "well_known" if parsed["src_port"] < 1024 else "high"
            features["dst_port"] = parsed["dst_port"]
        if parsed["protocol"] == "TCP" and parsed["flags"] is not None:
            features["flags"] = parsed["flags"]
        return self.encoder.encode_data(features)

    def _encode_payload(self, parsed: dict) -> np.ndarray:
        payload = parsed["payload"]
        if not payload:
            return np.zeros(DIMENSIONS, dtype=np.int8)

        features = {
            "size": len(payload),
            "unique_bytes": len(set(payload)),
        }
        # First 4 bytes as atoms
        for i, byte in enumerate(payload[:4]):
            features[f"byte_{i}"] = byte

        return self.encoder.encode_data(features)

    def save(self, path: str):
        """Save prior knowledge to disk."""
        data = {
            "baseline_vectors": {k: v.tobytes() for k, v in self.baseline_vectors.items()},
            "field_expectations": {k: asdict(v) for k, v in self.field_expectations.items()},
            "byte_expectations": {k: asdict(v) for k, v in self.byte_expectations.items()},
            "training_packets": self.training_packets,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved prior knowledge to {path}")

    def load(self, path: str):
        """Load prior knowledge from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.baseline_vectors = {
            k: np.frombuffer(v, dtype=np.float32).copy()
            for k, v in data["baseline_vectors"].items()
        }
        self.field_expectations = {
            k: FieldExpectation(**v)
            for k, v in data["field_expectations"].items()
        }
        self.byte_expectations = {
            int(k): BytePositionExpectation(**v)
            for k, v in data["byte_expectations"].items()
        }
        self.training_packets = data["training_packets"]
        print(f"Loaded prior knowledge ({self.training_packets} training packets)")


# =============================================================================
# RECENT KNOWLEDGE (adaptive from streaming)
# =============================================================================

class RecentKnowledge:
    """
    Adaptive knowledge from recent traffic.

    Decays over time, tracks recent patterns.
    """

    def __init__(self, decay: float = RECENT_DECAY, window_size: int = 100):
        self.decay = decay
        self.window_size = window_size

        # Per-perspective accumulators
        self.accumulators: Dict[str, np.ndarray] = {
            "l3": np.zeros(DIMENSIONS, dtype=np.float64),
            "l4": np.zeros(DIMENSIONS, dtype=np.float64),
            "payload": np.zeros(DIMENSIONS, dtype=np.float64),
        }

        # Recent field values (sliding window)
        self.field_windows: Dict[str, deque] = {
            "src_prefix": deque(maxlen=window_size),
            "dst_prefix": deque(maxlen=window_size),
            "src_port": deque(maxlen=window_size),
            "dst_port": deque(maxlen=window_size),
            "flags": deque(maxlen=window_size),
        }

        self.packet_count = 0

    def update(self, perspective: str, vector: np.ndarray, weight: float = 1.0):
        """Update accumulator for a perspective."""
        if perspective in self.accumulators:
            self.accumulators[perspective] = (
                self.decay * self.accumulators[perspective] +
                weight * vector.astype(np.float64)
            )

    def record_field(self, field: str, value: str):
        """Record a field value."""
        if field in self.field_windows:
            self.field_windows[field].append(value)

    def get_normalized(self, perspective: str) -> np.ndarray:
        """Get normalized accumulator vector."""
        if perspective not in self.accumulators:
            return np.zeros(DIMENSIONS, dtype=np.float32)
        accum = self.accumulators[perspective]
        norm = np.linalg.norm(accum)
        if norm < 1e-10:
            return np.zeros(DIMENSIONS, dtype=np.float32)
        return (accum / norm).astype(np.float32)

    def get_field_frequency(self, field: str, value: str) -> float:
        """Get frequency of value in recent window."""
        if field not in self.field_windows:
            return 0.0
        window = self.field_windows[field]
        if not window:
            return 0.0
        return window.count(value) / len(window)

    def get_field_cardinality(self, field: str) -> float:
        """Get cardinality in recent window."""
        if field not in self.field_windows:
            return 1.0
        window = self.field_windows[field]
        if len(window) < 10:
            return 1.0
        return len(set(window)) / len(window)

    def increment_count(self):
        self.packet_count += 1


# =============================================================================
# CULPRIT IDENTIFICATION
# =============================================================================

@dataclass
class Culprit:
    """Identifies what specifically is strange about a packet."""
    field: str
    observed_value: str
    expected_values: List[str]
    observed_frequency: float  # How often we've seen this value
    severity: str  # "low", "medium", "high"
    explanation: str


@dataclass
class DetectionResult:
    """Full detection result with culprit identification."""
    packet_num: int
    is_anomalous: bool

    # Similarity to prior (baseline)
    prior_l3_sim: float
    prior_l4_sim: float
    prior_payload_sim: float

    # Similarity to recent
    recent_l3_sim: float
    recent_l4_sim: float
    recent_payload_sim: float

    # Culprits (what's strange)
    culprits: List[Culprit]

    # Summary
    anomaly_score: float  # 0-1, higher = more anomalous
    primary_reason: str


# =============================================================================
# STREAMING DETECTOR
# =============================================================================

class StreamingDetector:
    """
    Production-ready streaming packet detector.

    Uses prior knowledge (frozen baseline) + recent knowledge (adaptive)
    to detect anomalies and identify culprits.
    """

    def __init__(
        self,
        prior: PriorKnowledge,
        prior_weight: float = 0.6,  # How much to weight prior vs recent
        anomaly_threshold: float = 0.5,
    ):
        self.prior = prior
        self.recent = RecentKnowledge()
        self.prior_weight = prior_weight
        self.anomaly_threshold = anomaly_threshold

        self.vm = prior.vm
        self.encoder = prior.encoder

    def process(self, pkt: Packet) -> DetectionResult:
        """Process a single packet in streaming fashion."""
        self.recent.increment_count()
        packet_num = self.recent.packet_count

        # Parse
        parsed = self.prior._parse(pkt)

        # Encode each perspective
        l3_vec = self.prior._encode_l3(parsed)
        l4_vec = self.prior._encode_l4(parsed)
        payload_vec = self.prior._encode_payload(parsed)

        # Compare to PRIOR knowledge (frozen baseline)
        prior_l3_sim = cosine_similarity(l3_vec, self.prior.baseline_vectors.get("l3", np.zeros(DIMENSIONS)))
        prior_l4_sim = cosine_similarity(l4_vec, self.prior.baseline_vectors.get("l4", np.zeros(DIMENSIONS)))
        prior_payload_sim = cosine_similarity(payload_vec, self.prior.baseline_vectors.get("payload", np.zeros(DIMENSIONS)))

        # Compare to RECENT knowledge (adaptive)
        recent_l3_sim = cosine_similarity(l3_vec, self.recent.get_normalized("l3"))
        recent_l4_sim = cosine_similarity(l4_vec, self.recent.get_normalized("l4"))
        recent_payload_sim = cosine_similarity(payload_vec, self.recent.get_normalized("payload"))

        # Identify culprits (what's strange)
        culprits = self._identify_culprits(parsed)

        # Compute anomaly score (weighted blend of prior and recent deviations)
        # Weight perspectives by reliability (L4 and Payload most reliable for attack detection)
        prior_deviation = 1.0 - (0.2 * prior_l3_sim + 0.4 * prior_l4_sim + 0.4 * prior_payload_sim)
        recent_deviation = 1.0 - (0.2 * recent_l3_sim + 0.4 * recent_l4_sim + 0.4 * recent_payload_sim)

        # Anomaly if deviant from PRIOR (even if matches recent - recent might be poisoned)
        # But also consider recent for context
        anomaly_score = self.prior_weight * prior_deviation + (1 - self.prior_weight) * recent_deviation

        # Boost score if we have identified HIGH severity culprits
        high_culprits = [c for c in culprits if c.severity == "high"]
        culprit_boost = min(0.3, len(high_culprits) * 0.15)
        anomaly_score = min(1.0, anomaly_score + culprit_boost)

        is_anomalous = anomaly_score > self.anomaly_threshold

        # Determine primary reason
        if culprits:
            primary_reason = culprits[0].explanation
        elif prior_l3_sim < 0.3:
            primary_reason = "Unusual source/destination pattern"
        elif prior_l4_sim < 0.3:
            primary_reason = "Unusual port/protocol pattern"
        elif prior_payload_sim < 0.3:
            primary_reason = "Unusual payload structure"
        else:
            primary_reason = "Normal traffic"

        # Update recent knowledge (with reduced weight if anomalous)
        weight = 0.1 if is_anomalous else 1.0
        self.recent.update("l3", l3_vec, weight)
        self.recent.update("l4", l4_vec, weight)
        self.recent.update("payload", payload_vec, weight)

        # Record field values for cardinality tracking
        if parsed["src_prefix"]:
            self.recent.record_field("src_prefix", parsed["src_prefix"])
        if parsed["dst_prefix"]:
            self.recent.record_field("dst_prefix", parsed["dst_prefix"])
        if parsed["src_port"]:
            self.recent.record_field("src_port", str(parsed["src_port"]))
        if parsed["dst_port"]:
            self.recent.record_field("dst_port", str(parsed["dst_port"]))
        if parsed["flags"] is not None:
            self.recent.record_field("flags", str(parsed["flags"]))

        return DetectionResult(
            packet_num=packet_num,
            is_anomalous=is_anomalous,
            prior_l3_sim=prior_l3_sim,
            prior_l4_sim=prior_l4_sim,
            prior_payload_sim=prior_payload_sim,
            recent_l3_sim=recent_l3_sim,
            recent_l4_sim=recent_l4_sim,
            recent_payload_sim=recent_payload_sim,
            culprits=culprits,
            anomaly_score=anomaly_score,
            primary_reason=primary_reason,
        )

    def _identify_culprits(self, parsed: dict) -> List[Culprit]:
        """Identify what specific values are unusual."""
        culprits = []

        # High-cardinality fields to skip for culprit identification
        # (ephemeral ports are expected to be unique)
        skip_if_high_cardinality = {"src_port"}

        # Check each field against prior expectations
        field_checks = [
            ("src_prefix", parsed["src_prefix"]),
            ("dst_prefix", parsed["dst_prefix"]),
            ("src_port", str(parsed["src_port"]) if parsed["src_port"] else None),
            ("dst_port", str(parsed["dst_port"]) if parsed["dst_port"] else None),
            ("protocol", parsed["protocol"]),
            ("flags", str(parsed["flags"]) if parsed["flags"] is not None else None),
        ]

        for field, value in field_checks:
            if not value or field not in self.prior.field_expectations:
                continue

            expectation = self.prior.field_expectations[field]

            # Skip high-cardinality fields UNLESS the value is structurally unusual
            if field in skip_if_high_cardinality and expectation.cardinality > 0.5:
                # For src_port, only flag if it's a well-known port (unusual for source)
                if field == "src_port":
                    port = int(value)
                    if port < 1024:  # Well-known port as source = suspicious (reflection)
                        culprits.append(Culprit(
                            field=field,
                            observed_value=value,
                            expected_values=["ephemeral (49152-65535)"],
                            observed_frequency=0.0,
                            severity="high",
                            explanation=f"src_port={value} is a well-known port (possible reflection attack)",
                        ))
                continue

            # Check if value is in common values
            if value not in expectation.common_values:
                # Check frequency in training
                freq = expectation.value_frequencies.get(value, 0.0)

                if freq < 0.01:  # Rare or never seen
                    severity = "high" if freq == 0 else "medium"

                    culprits.append(Culprit(
                        field=field,
                        observed_value=value,
                        expected_values=expectation.common_values[:5],
                        observed_frequency=freq,
                        severity=severity,
                        explanation=f"{field}={value} is {'never' if freq == 0 else 'rarely'} seen (expected: {expectation.common_values[:3]})",
                    ))

        # Check byte positions for payload
        if parsed["payload"]:
            for i, byte in enumerate(parsed["payload"][:8]):
                if i in self.prior.byte_expectations:
                    exp = self.prior.byte_expectations[i]
                    freq = exp.byte_frequencies.get(byte, 0.0)

                    if freq < 0.01 and byte not in exp.common_bytes:
                        culprits.append(Culprit(
                            field=f"payload_byte_{i}",
                            observed_value=f"0x{byte:02x}",
                            expected_values=[f"0x{b:02x}" for b in exp.common_bytes[:3]],
                            observed_frequency=freq,
                            severity="medium",
                            explanation=f"Byte {i}=0x{byte:02x} unexpected (expected: {[f'0x{b:02x}' for b in exp.common_bytes[:3]]})",
                        ))

        # Sort by severity
        severity_order = {"high": 0, "medium": 1, "low": 2}
        culprits.sort(key=lambda c: severity_order.get(c.severity, 3))

        return culprits[:5]  # Top 5 culprits


# =============================================================================
# PACKET GENERATOR
# =============================================================================

class PacketGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _random_ip(self, prefix: str = "192.168") -> str:
        return f"{prefix}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}"

    def _ephemeral(self) -> int:
        return self.rng.randint(49152, 65535)

    def normal_http(self) -> Packet:
        payload = b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n"
        return (
            IP(src=self._random_ip(), dst=self._random_ip("10.0"))
            / TCP(sport=self._ephemeral(), dport=80, flags="PA")
            / Raw(load=payload)
        )

    def normal_https(self) -> Packet:
        # TLS-like
        payload = bytes([0x16, 0x03, 0x01]) + bytes([self.rng.randint(0, 255) for _ in range(50)])
        return (
            IP(src=self._random_ip(), dst=self._random_ip("10.0"))
            / TCP(sport=self._ephemeral(), dport=443, flags="PA")
            / Raw(load=payload)
        )

    def normal_dns(self) -> Packet:
        payload = bytes([self.rng.randint(0, 255) for _ in range(20)])
        return (
            IP(src=self._random_ip(), dst="8.8.8.8")
            / UDP(sport=self._ephemeral(), dport=53)
            / Raw(load=payload)
        )

    def syn_flood(self, target: str = "10.0.1.100") -> Packet:
        return (
            IP(src=self._random_ip("172.16"), dst=target)
            / TCP(sport=self.rng.randint(1, 65535), dport=443, flags="S")
        )

    def dns_reflection(self, victim: str = "10.0.1.100") -> Packet:
        payload = bytes([self.rng.randint(0, 255) for _ in range(512)])
        return (
            IP(src="8.8.8.8", dst=victim)
            / UDP(sport=53, dport=self._ephemeral())
            / Raw(load=payload)
        )

    def unknown_protocol(self) -> Packet:
        # Traffic on unusual port with strange payload
        payload = bytes([0xDE, 0xAD, 0xBE, 0xEF]) + bytes([self.rng.randint(0, 255) for _ in range(100)])
        return (
            IP(src=self._random_ip("192.168"), dst=self._random_ip("192.168"))
            / TCP(sport=self._ephemeral(), dport=31337, flags="PA")
            / Raw(load=payload)
        )


# =============================================================================
# MAIN DEMO
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 011-008: Realistic Streaming Deployment")
    print("=" * 80)
    print("""
ARCHITECTURE:
  Prior Knowledge (frozen baseline)
  + Recent Knowledge (adaptive from stream)
  + Culprit Identification (what's strange)
  = Production-ready anomaly detection
""")

    gen = PacketGenerator(seed=42)

    # ==========================================================================
    # PHASE 1: Build Prior Knowledge (offline training)
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 1: Training Prior Knowledge (offline)")
    print("=" * 60)

    vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
    prior = PriorKnowledge(vm)

    # Generate training traffic (historical baseline)
    training_packets = []
    for _ in range(1000):
        pkt = gen.rng.choice([
            gen.normal_http,
            gen.normal_https,
            gen.normal_dns,
        ])()
        training_packets.append(pkt)

    prior.train(training_packets)

    # Show what we learned
    print("\n  Field expectations learned:")
    for field, exp in prior.field_expectations.items():
        print(f"    {field}: top values = {exp.common_values[:3]}, cardinality = {exp.cardinality:.3f}")

    print("\n  Byte position expectations (first 4):")
    for i in range(4):
        if i in prior.byte_expectations:
            exp = prior.byte_expectations[i]
            print(f"    Position {i}: common bytes = {[f'0x{b:02x}' for b in exp.common_bytes[:3]]}")

    # Optionally save/load (simulate deployment)
    # prior.save("/tmp/prior_knowledge.pkl")
    # prior.load("/tmp/prior_knowledge.pkl")

    # ==========================================================================
    # PHASE 2: Streaming Detection
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 2: Streaming Detection (online)")
    print("=" * 60)

    detector = StreamingDetector(prior, prior_weight=0.7, anomaly_threshold=0.45)

    # Simulate streaming with phases
    phases = [
        ("Normal traffic", [gen.normal_http, gen.normal_https, gen.normal_dns], 200),
        ("SYN flood starts", [gen.syn_flood], 100),
        ("DNS reflection", [gen.dns_reflection], 100),
        ("Unknown protocol", [gen.unknown_protocol], 50),
        ("Normal resumes", [gen.normal_http, gen.normal_https], 100),
    ]

    all_results = []

    for phase_name, generators, count in phases:
        print(f"\n--- {phase_name} ({count} packets) ---")

        phase_results = []
        anomaly_count = 0

        for _ in range(count):
            pkt = gen.rng.choice(generators)()
            result = detector.process(pkt)
            phase_results.append(result)

            if result.is_anomalous:
                anomaly_count += 1

        all_results.extend(phase_results)

        # Summary for phase
        anomaly_rate = 100 * anomaly_count / count
        print(f"  Anomaly rate: {anomaly_count}/{count} ({anomaly_rate:.0f}%)")

        # Show sample culprits
        anomalies = [r for r in phase_results if r.is_anomalous]
        if anomalies:
            sample = anomalies[0]
            print(f"  Sample anomaly (score={sample.anomaly_score:.2f}):")
            print(f"    Prior similarities: L3={sample.prior_l3_sim:.2f}, L4={sample.prior_l4_sim:.2f}, Payload={sample.prior_payload_sim:.2f}")
            print(f"    Primary reason: {sample.primary_reason}")
            if sample.culprits:
                print(f"    Culprits identified:")
                for c in sample.culprits[:3]:
                    print(f"      [{c.severity}] {c.explanation}")

    # ==========================================================================
    # PHASE 3: Detailed Culprit Analysis
    # ==========================================================================
    print("\n" + "=" * 60)
    print("PHASE 3: Culprit Analysis Examples")
    print("=" * 60)

    # Create fresh detector for clean analysis
    detector2 = StreamingDetector(prior, prior_weight=0.7, anomaly_threshold=0.45)

    # Warmup with some normal traffic
    for _ in range(50):
        detector2.process(gen.normal_http())

    test_cases = [
        ("Normal HTTP", gen.normal_http()),
        ("SYN Flood packet", gen.syn_flood()),
        ("DNS Reflection", gen.dns_reflection()),
        ("Unknown Protocol (port 31337)", gen.unknown_protocol()),
    ]

    for name, pkt in test_cases:
        result = detector2.process(pkt)

        status = "🚨 ANOMALY" if result.is_anomalous else "✅ NORMAL"
        print(f"\n{name}:")
        print(f"  {status} (score: {result.anomaly_score:.2f})")
        print(f"  Prior sim: L3={result.prior_l3_sim:.2f}, L4={result.prior_l4_sim:.2f}, Payload={result.prior_payload_sim:.2f}")

        if result.culprits:
            print(f"  Culprits:")
            for c in result.culprits:
                print(f"    [{c.severity}] {c.field}={c.observed_value}")
                print(f"           Expected: {c.expected_values[:3]}")
        else:
            print(f"  No specific culprits identified")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("DEPLOYMENT SUMMARY")
    print("=" * 80)
    print("""
PRIOR KNOWLEDGE (frozen, from training):
  ✓ Baseline vectors per perspective (L3, L4, Payload)
  ✓ Expected field values and frequencies
  ✓ Expected byte distributions per position
  ✓ Serializable to disk for deployment

RECENT KNOWLEDGE (adaptive, from stream):
  ✓ Decaying accumulators track recent patterns
  ✓ Sliding window cardinality per field
  ✓ Can detect drift from prior over time

CULPRIT IDENTIFICATION:
  ✓ Pinpoints specific unusual fields/values
  ✓ Shows expected vs observed
  ✓ Severity rating (high/medium/low)
  ✓ Human-readable explanations

STREAMING INTERFACE:
  ✓ Process packets one at a time
  ✓ O(1) per packet (vector ops)
  ✓ Update recent knowledge after each packet
  ✓ Reduce update weight for anomalies (resist poisoning)

DETECTION STRATEGY:
  - Compare to PRIOR (frozen baseline) - primary signal
  - Compare to RECENT (adaptive) - context signal
  - Identify CULPRITS (specific unusual values)
  - Boost score if specific culprits found
""")


if __name__ == "__main__":
    main()
