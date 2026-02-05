#!/usr/bin/env python3
"""
Challenge 011-002: Scoped PCAP Field Vectors

Apply per-component accumulator approach to network packets:

    src_addr_accumulator  → learns normal source IP patterns
    dst_addr_accumulator  → learns normal destination patterns
    src_port_accumulator  → learns normal source port patterns
    dst_port_accumulator  → learns normal dest port patterns
    proto_accumulator     → learns normal protocol distribution
    size_accumulator      → learns normal payload sizes

This enables:
    - Per-field anomaly detection ("unusual source port")
    - Better DDoS classification ("SYN flood = abnormal flag pattern")
    - Explainability ("dst_port anomaly detected")

Comparison to batch 010: single-vector encoding vs scoped fields
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import deque

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.995  # Faster decay for packet streams
WARMUP_PACKETS = 500


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# FIELD EXTRACTION
# =============================================================================

def bucket_port(port: int) -> str:
    """Bucket port into categories for lower cardinality."""
    if port == 0:
        return "zero"
    elif port < 1024:
        return "well_known"
    elif port < 49152:
        return "registered"
    else:
        return "ephemeral"


def bucket_size(size: int) -> str:
    """Bucket payload size."""
    if size == 0:
        return "empty"
    elif size < 64:
        return "tiny"
    elif size < 256:
        return "small"
    elif size < 1024:
        return "medium"
    elif size < 1500:
        return "large"
    else:
        return "jumbo"


def ip_prefix(ip: str, mask: int = 16) -> str:
    """Extract IP prefix for grouping."""
    parts = ip.split(".")
    if mask == 8:
        return f"{parts[0]}.0.0.0/8"
    elif mask == 16:
        return f"{parts[0]}.{parts[1]}.0.0/16"
    elif mask == 24:
        return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
    return ip


@dataclass
class PacketFields:
    """Decomposed packet into scoped fields."""
    src_addr: dict
    dst_addr: dict
    src_port: dict
    dst_port: dict
    protocol: dict
    size: dict
    flags: dict  # TCP flags or ICMP type/code

    # Raw for debugging
    raw: dict = None


def extract_fields(packet: dict) -> PacketFields:
    """Extract scoped fields from packet."""
    proto = packet.get("protocol", "TCP").upper()
    src_ip = packet.get("src_ip", "0.0.0.0")
    dst_ip = packet.get("dst_ip", "0.0.0.0")
    src_port = packet.get("src_port", 0)
    dst_port = packet.get("dst_port", 0)
    payload_size = packet.get("payload_size", 0)

    # Source address field
    src_addr_field = {
        "prefix_16": ip_prefix(src_ip, 16),
        "prefix_24": ip_prefix(src_ip, 24),
    }

    # Destination address field
    dst_addr_field = {
        "prefix_16": ip_prefix(dst_ip, 16),
        "prefix_24": ip_prefix(dst_ip, 24),
    }

    # Source port field
    src_port_field = {
        "bucket": bucket_port(src_port),
        "is_wellknown": src_port < 1024,
        "is_dns": src_port == 53,
        "is_ntp": src_port == 123,
    }

    # Dest port field
    dst_port_field = {
        "bucket": bucket_port(dst_port),
        "is_http": dst_port in (80, 8080),
        "is_https": dst_port == 443,
        "is_ssh": dst_port == 22,
    }

    # Protocol field
    proto_field = {
        "protocol": proto,
        "is_tcp": proto == "TCP",
        "is_udp": proto == "UDP",
        "is_icmp": proto == "ICMP",
    }

    # Size field
    size_field = {
        "bucket": bucket_size(payload_size),
        "is_empty": payload_size == 0,
    }

    # Flags field (protocol-specific)
    if proto == "TCP":
        flags = packet.get("flags", 0)
        flags_field = {
            "raw_flags": flags,
            "is_syn": (flags & 0x02) != 0,
            "is_ack": (flags & 0x10) != 0,
            "is_fin": (flags & 0x01) != 0,
            "is_rst": (flags & 0x04) != 0,
            "is_syn_only": flags == 0x02,
        }
    elif proto == "ICMP":
        flags_field = {
            "icmp_type": packet.get("icmp_type", 0),
            "icmp_code": packet.get("icmp_code", 0),
            "is_echo": packet.get("icmp_type", 0) in (0, 8),
        }
    else:
        flags_field = {"none": True}

    return PacketFields(
        src_addr=src_addr_field,
        dst_addr=dst_addr_field,
        src_port=src_port_field,
        dst_port=dst_port_field,
        protocol=proto_field,
        size=size_field,
        flags=flags_field,
        raw=packet,
    )


# =============================================================================
# SCOPED ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    def __init__(self, dimensions: int, decay: float = DECAY_FACTOR):
        self.dimensions = dimensions
        self.decay = decay
        self.accumulator = np.zeros(dimensions, dtype=np.float64)
        self.count = 0

    def update(self, vector: np.ndarray, weight: float = 1.0):
        self.accumulator = self.decay * self.accumulator + weight * vector.astype(np.float64)
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)


# =============================================================================
# SCOPED DETECTOR
# =============================================================================

@dataclass
class FieldScore:
    """Score for a single field."""
    name: str
    similarity: float
    threshold: float
    is_anomalous: bool
    features: dict


@dataclass
class ScopedPacketResult:
    """Detection result with per-field breakdown."""
    is_flagged: bool
    field_scores: List[FieldScore]
    anomalous_fields: List[str]
    explanation: str
    is_warmup: bool = False


class ScopedPacketDetector:
    """
    Packet anomaly detector with per-field accumulators.

    Each field (src_addr, dst_port, protocol, etc.) has its own
    accumulator that learns normal patterns independently.
    """

    # Field thresholds (tunable per-field)
    DEFAULT_THRESHOLDS = {
        "src_addr": 0.40,
        "dst_addr": 0.40,
        "src_port": 0.45,
        "dst_port": 0.45,
        "protocol": 0.50,
        "size": 0.45,
        "flags": 0.45,
    }

    # Field weights for importance
    DEFAULT_WEIGHTS = {
        "src_addr": 0.10,
        "dst_addr": 0.10,
        "src_port": 0.20,  # Source port is key for reflection detection
        "dst_port": 0.15,
        "protocol": 0.15,
        "size": 0.10,
        "flags": 0.20,  # Flags key for SYN flood detection
    }

    def __init__(
        self,
        decay: float = DECAY_FACTOR,
        warmup: int = WARMUP_PACKETS,
        voting_threshold: int = 2,
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
        self.encoder = Encoder(vector_manager=self.vm)

        self.decay = decay
        self.warmup = warmup
        self.voting_threshold = voting_threshold
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

        # Per-field accumulators
        self.accumulators = {
            name: DecayingAccumulator(DIMENSIONS, decay)
            for name in self.DEFAULT_THRESHOLDS
        }

        self.packets_seen = 0

    def process(self, packet: dict) -> ScopedPacketResult:
        """Process packet with per-field analysis."""
        self.packets_seen += 1
        is_warmup = self.packets_seen <= self.warmup

        # Extract fields
        fields = extract_fields(packet)

        # Score each field
        field_scores = []
        field_vecs = {}

        for name, features in [
            ("src_addr", fields.src_addr),
            ("dst_addr", fields.dst_addr),
            ("src_port", fields.src_port),
            ("dst_port", fields.dst_port),
            ("protocol", fields.protocol),
            ("size", fields.size),
            ("flags", fields.flags),
        ]:
            vec = self.encoder.encode_data(features)
            field_vecs[name] = vec

            model = self.accumulators[name].get_normalized()
            sim = cosine_similarity(vec, model) if self.packets_seen > 1 else 1.0

            threshold = self.thresholds[name]
            is_anomalous = sim < threshold

            field_scores.append(FieldScore(
                name=name,
                similarity=sim,
                threshold=threshold,
                is_anomalous=is_anomalous,
                features=features,
            ))

        # Voting aggregation
        anomalous_fields = [f.name for f in field_scores if f.is_anomalous]
        n_anomalous = len(anomalous_fields)
        is_flagged = n_anomalous >= self.voting_threshold

        if is_warmup:
            is_flagged = False

        # Generate explanation
        if is_flagged:
            explanation = f"ANOMALY: {n_anomalous} fields anomalous - {anomalous_fields}"
        else:
            explanation = f"NORMAL: {n_anomalous} anomalous (threshold={self.voting_threshold})"

        # Update accumulators
        weight = 0.1 if is_flagged else 1.0
        for name, vec in field_vecs.items():
            self.accumulators[name].update(vec, weight)

        return ScopedPacketResult(
            is_flagged=is_flagged,
            field_scores=field_scores,
            anomalous_fields=anomalous_fields,
            explanation=explanation,
            is_warmup=is_warmup,
        )


# =============================================================================
# TRAFFIC GENERATOR
# =============================================================================

class TrafficGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _random_ip(self) -> str:
        return f"192.168.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}"

    def _ephemeral(self) -> int:
        return self.rng.randint(49152, 65535)

    def normal_packet(self) -> dict:
        """Generate normal network traffic."""
        proto = self.rng.choices(["TCP", "UDP", "ICMP"], weights=[0.80, 0.18, 0.02])[0]

        base = {
            "src_ip": self._random_ip(),
            "dst_ip": self._random_ip(),
            "protocol": proto,
            "payload_size": self.rng.randint(0, 1400),
        }

        if proto == "TCP":
            base["src_port"] = self._ephemeral()
            base["dst_port"] = self.rng.choice([443, 80, 8080, 22])
            # Normal TCP has mostly ACK, some SYN, some FIN
            base["flags"] = self.rng.choices(
                [0x10, 0x12, 0x18, 0x02, 0x11],  # ACK, SYN-ACK, ACK-PSH, SYN, FIN-ACK
                weights=[0.50, 0.15, 0.25, 0.05, 0.05]
            )[0]
        elif proto == "UDP":
            base["src_port"] = self._ephemeral()
            base["dst_port"] = self.rng.choice([443, 53, 123])
        else:  # ICMP
            base["icmp_type"] = self.rng.choice([0, 8])  # Echo reply/request
            base["icmp_code"] = 0

        return base

    def syn_flood(self) -> dict:
        """SYN flood attack."""
        return {
            "src_ip": f"{self.rng.randint(1, 223)}.{self.rng.randint(0, 255)}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}",
            "dst_ip": "192.168.1.100",  # Target
            "protocol": "TCP",
            "src_port": self.rng.randint(1, 65535),
            "dst_port": 443,
            "flags": 0x02,  # SYN only
            "payload_size": 0,
        }

    def dns_reflection(self) -> dict:
        """DNS amplification/reflection."""
        return {
            "src_ip": f"8.8.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}",  # DNS servers
            "dst_ip": "192.168.1.100",  # Victim
            "protocol": "UDP",
            "src_port": 53,  # FROM DNS server (reflection!)
            "dst_port": self._ephemeral(),
            "payload_size": self.rng.randint(512, 4096),  # Amplified response
        }

    def ntp_amplification(self) -> dict:
        """NTP amplification attack."""
        return {
            "src_ip": f"129.{self.rng.randint(0, 255)}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}",
            "dst_ip": "192.168.1.100",
            "protocol": "UDP",
            "src_port": 123,  # FROM NTP server
            "dst_port": self._ephemeral(),
            "payload_size": self.rng.randint(468, 468),  # monlist response
        }

    def port_scan(self) -> dict:
        """Port scanning."""
        return {
            "src_ip": "10.0.0.99",  # Scanner
            "dst_ip": "192.168.1.100",
            "protocol": "TCP",
            "src_port": self._ephemeral(),
            "dst_port": self.rng.randint(1, 1024),  # Scanning well-known ports
            "flags": 0x02,  # SYN
            "payload_size": 0,
        }

    def icmp_flood(self) -> dict:
        """ICMP flood."""
        return {
            "src_ip": f"{self.rng.randint(1, 223)}.{self.rng.randint(0, 255)}.{self.rng.randint(0, 255)}.{self.rng.randint(1, 254)}",
            "dst_ip": "192.168.1.100",
            "protocol": "ICMP",
            "icmp_type": 8,
            "icmp_code": 0,
            "payload_size": self.rng.randint(56, 1472),
        }

    def unusual_port(self) -> dict:
        """Connection to unusual port."""
        return {
            "src_ip": self._random_ip(),
            "dst_ip": self._random_ip(),
            "protocol": "TCP",
            "src_port": self._ephemeral(),
            "dst_port": self.rng.choice([4444, 31337, 6667, 1337]),  # Suspicious ports
            "flags": 0x02,
            "payload_size": 0,
        }

    def generate_mixed_stream(self, n: int, attack_ratio: float = 0.05) -> List[Tuple[dict, Optional[str]]]:
        """Generate stream with mixed attacks."""
        stream = []
        attacks = ["syn_flood", "dns_reflection", "ntp_amplification", "port_scan", "icmp_flood", "unusual_port"]
        attack_funcs = {
            "syn_flood": self.syn_flood,
            "dns_reflection": self.dns_reflection,
            "ntp_amplification": self.ntp_amplification,
            "port_scan": self.port_scan,
            "icmp_flood": self.icmp_flood,
            "unusual_port": self.unusual_port,
        }

        for _ in range(n):
            if self.rng.random() < attack_ratio:
                attack = self.rng.choice(attacks)
                packet = attack_funcs[attack]()
                stream.append((packet, attack))
            else:
                stream.append((self.normal_packet(), None))

        return stream


# =============================================================================
# MAIN
# =============================================================================

def evaluate(detector: ScopedPacketDetector, stream: list, name: str):
    """Evaluate detector on stream."""
    results = []
    start = time.time()

    for packet, attack in stream:
        result = detector.process(packet)
        results.append((result, attack))

    elapsed = time.time() - start
    throughput = len(stream) / elapsed

    # Post-warmup metrics
    post_warmup = [(r, a) for r, a in results if not r.is_warmup]

    tp = sum(1 for r, a in post_warmup if a is not None and r.is_flagged)
    fp = sum(1 for r, a in post_warmup if a is None and r.is_flagged)
    fn = sum(1 for r, a in post_warmup if a is not None and not r.is_flagged)
    tn = sum(1 for r, a in post_warmup if a is None and not r.is_flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\n--- {name} ---")
    print(f"  Throughput: {throughput:.0f} pkt/sec")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision:.1%}, Recall: {recall:.1%}, F1: {f1:.3f}")

    # By attack type
    attack_stats = {}
    for r, a in post_warmup:
        if a:
            if a not in attack_stats:
                attack_stats[a] = {"detected": 0, "missed": 0, "fields": []}
            if r.is_flagged:
                attack_stats[a]["detected"] += 1
                attack_stats[a]["fields"].extend(r.anomalous_fields)
            else:
                attack_stats[a]["missed"] += 1

    print(f"  By attack type:")
    for attack, stats in sorted(attack_stats.items()):
        total = stats["detected"] + stats["missed"]
        rate = stats["detected"] / total if total else 0
        # Most common triggering fields
        from collections import Counter
        field_counts = Counter(stats["fields"]).most_common(3)
        fields_str = ", ".join(f"{f}({c})" for f, c in field_counts)
        print(f"    {attack}: {stats['detected']}/{total} ({rate:.0%}) - triggers: {fields_str}")

    return f1, precision, recall, throughput, attack_stats


def main():
    print("=" * 80)
    print("Challenge 011-002: Scoped PCAP Field Vectors")
    print("=" * 80)
    print("""
Per-field accumulators for network packets:
  - src_addr: source IP patterns
  - dst_addr: destination IP patterns
  - src_port: source port patterns (key for reflection!)
  - dst_port: destination port patterns
  - protocol: TCP/UDP/ICMP distribution
  - size: payload size patterns
  - flags: TCP flags / ICMP type patterns

Detection: Flag if >= N fields are anomalous (voting)
""")

    # Generate stream
    generator = TrafficGenerator(seed=42)
    stream = generator.generate_mixed_stream(10000, attack_ratio=0.05)

    actual_attacks = sum(1 for _, a in stream if a is not None)
    print(f"Stream: {len(stream)} packets, {actual_attacks} attacks ({100*actual_attacks/len(stream):.1f}%)")

    # Test different voting thresholds
    results = {}
    for threshold in [1, 2, 3]:
        detector = ScopedPacketDetector(voting_threshold=threshold, warmup=WARMUP_PACKETS)
        f1, p, r, tput, attack_stats = evaluate(detector, stream, f"Voting threshold={threshold}")
        results[threshold] = {"f1": f1, "precision": p, "recall": r, "throughput": tput, "attack_stats": attack_stats}

    # Summary
    print("\n" + "=" * 80)
    print("COMPARISON: Voting Thresholds")
    print("=" * 80)
    print(f"\n{'Threshold':<12} {'F1':<10} {'Precision':<12} {'Recall':<10}")
    print("-" * 50)
    for t, m in results.items():
        print(f"{t:<12} {m['f1']:<10.3f} {m['precision']:<12.1%} {m['recall']:<10.1%}")

    # Show which fields trigger for each attack
    print("\n--- Field Triggers by Attack Type ---")
    best = max(results.items(), key=lambda x: x[1]["f1"])
    for attack, stats in sorted(best[1]["attack_stats"].items()):
        from collections import Counter
        field_counts = Counter(stats["fields"]).most_common()
        print(f"  {attack}:")
        for field, count in field_counts:
            pct = 100 * count / max(1, stats["detected"])
            print(f"    {field}: {count} ({pct:.0f}% of detections)")

    # Sample detections
    print("\n--- Sample Detections (threshold=2) ---")
    detector = ScopedPacketDetector(voting_threshold=2)

    # Warmup
    for packet, _ in stream[:WARMUP_PACKETS]:
        detector.process(packet)

    test_cases = [
        (generator.normal_packet(), None),
        (generator.syn_flood(), "syn_flood"),
        (generator.dns_reflection(), "dns_reflection"),
        (generator.port_scan(), "port_scan"),
        (generator.unusual_port(), "unusual_port"),
    ]

    for packet, attack in test_cases:
        result = detector.process(packet)
        status = "🚨 FLAGGED" if result.is_flagged else "✅ ALLOWED"
        expected = f"(expected: {attack if attack else 'normal'})"
        print(f"\n{status} {expected}")
        print(f"  Protocol: {packet.get('protocol')}, dst_port: {packet.get('dst_port')}, src_port: {packet.get('src_port')}")
        print(f"  Anomalous fields: {result.anomalous_fields}")
        for f in result.field_scores:
            flag = "⚠️" if f.is_anomalous else "✓"
            print(f"    {flag} {f.name}: sim={f.similarity:.3f}")

    print("\n" + "=" * 80)
    print("LEARNINGS")
    print("=" * 80)
    print("""
Key observations from per-field PCAP analysis:

1. src_port is KEY for reflection detection
   - Normal: ephemeral ports (49152+)
   - Reflection: well-known ports (53, 123) as SOURCE

2. flags field catches SYN floods
   - Normal: mix of ACK, SYN-ACK, etc.
   - SYN flood: pure SYN (0x02)

3. dst_port catches unusual port connections
   - Normal: 443, 80, 22
   - Suspicious: 4444, 31337, etc.

4. Field-level detection provides natural explanations
   - "src_port anomaly" = likely reflection
   - "flags anomaly" = likely SYN flood

Next: Black-box byte-level encoding (no protocol knowledge)
""")


if __name__ == "__main__":
    main()
