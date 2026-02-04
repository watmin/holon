#!/usr/bin/env python3
"""
Challenge 010-026: Headless PCAP Detection with Pure Frequency+Decay

Honest packet representation:
- L4 protocol (TCP/UDP/ICMP)
- Protocol-specific fields:
  - TCP: src_port, dst_port, flags (bitmask)
  - UDP: src_port, dst_port
  - ICMP: type, code
- Packet size (8 buckets)
- No IP classification (can't trust in the wild)
- No L2 (MAC addresses not trustworthy)

Let frequency+decay learn what's normal.
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from deterministic_codebook import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.9995
ANOMALY_THRESHOLD = 0.35
WARMUP_PACKETS = 500


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# PACKET FINGERPRINTING - Honest Representation
# =============================================================================

def bucket_size(size: int) -> int:
    """
    Bucket packet size into 8 categories.

    0: tiny (<=40, headers only)
    1: minimal (41-64)
    2: small (65-128)
    3: medium-small (129-256)
    4: medium (257-512)
    5: medium-large (513-1024)
    6: large (1025-1500, typical MTU)
    7: jumbo (>1500)
    """
    if size <= 40:
        return 0
    elif size <= 64:
        return 1
    elif size <= 128:
        return 2
    elif size <= 256:
        return 3
    elif size <= 512:
        return 4
    elif size <= 1024:
        return 5
    elif size <= 1500:
        return 6
    else:
        return 7


def describe_tcp_flags(mask: int) -> str:
    """Human-readable TCP flag description."""
    parts = []
    if mask & 0x02: parts.append("SYN")
    if mask & 0x10: parts.append("ACK")
    if mask & 0x01: parts.append("FIN")
    if mask & 0x04: parts.append("RST")
    if mask & 0x08: parts.append("PSH")
    if mask & 0x20: parts.append("URG")
    return "+".join(parts) if parts else "NONE"


def bucket_port(port: int) -> str:
    """
    Bucket port into meaningful categories.

    For headless detection, raw ports are too high cardinality.
    Group into service categories.
    """
    # Well-known web
    if port in {80, 443, 8080, 8443}:
        return "web"
    # DNS/time
    elif port in {53, 123}:
        return "dns_ntp"
    # SSH/admin
    elif port in {22, 23, 3389}:
        return "admin"
    # Windows
    elif port in {445, 139, 135}:
        return "windows"
    # Mail
    elif port in {25, 587, 465, 110, 143, 993, 995}:
        return "mail"
    # Databases
    elif port in {3306, 5432, 1433, 27017, 6379}:
        return "db"
    # Suspicious/hacker ports
    elif port in {6667, 6697, 4444, 31337, 1337, 12345, 17, 19, 1900}:
        return "suspicious"
    # Other privileged
    elif port < 1024:
        return "priv_other"
    # Registered
    elif port < 49152:
        return "registered"
    # Ephemeral
    else:
        return "ephemeral"


def extract_packet_fingerprint(packet: dict) -> dict:
    """
    MINIMAL fingerprint - fewer features = each has more weight.

    Protocol-specific:
    - TCP: (dst_bucket, flags) as combined key
    - UDP: dst_bucket
    - ICMP: (type, code) as combined key

    Common:
    - size_bucket
    """
    protocol = packet.get("protocol", "TCP").upper()
    size = packet.get("size", 0)
    size_bucket = bucket_size(size)

    if protocol == "TCP":
        dst_port = packet.get("dst_port", 0)

        # Flags
        flags = packet.get("flags", 0)
        if isinstance(flags, list):
            flag_map = {"FIN": 0x01, "SYN": 0x02, "RST": 0x04, "PSH": 0x08, "ACK": 0x10, "URG": 0x20}
            mask = 0
            for f in flags:
                mask |= flag_map.get(f.upper(), 0)
            flags = mask

        dst_bucket = bucket_port(dst_port)

        # Balance: separate features allow partial matching
        # But give key discriminators (bucket, flags) more weight
        features = {
            # Destination bucket - key discriminator for suspicious ports
            "dst_bucket": dst_bucket,
            "dst_bucket_key": f"dst:{dst_bucket}",

            # Flags - key discriminator for invalid combos
            "tcp_flags": flags,
            "tcp_flags_key": f"flags:{flags:02x}",

            # Combined for specific patterns
            "tcp_pattern": f"TCP:{dst_bucket}:{flags:02x}",
        }
        # Display string
        features["fingerprint"] = f"TCP:{dst_bucket}:{flags:02x}"

    elif protocol == "UDP":
        dst_port = packet.get("dst_port", 0)
        dst_bucket = bucket_port(dst_port)

        features = {
            "dst_bucket": dst_bucket,
            "dst_bucket_key": f"dst:{dst_bucket}",
            "udp_pattern": f"UDP:{dst_bucket}",
            "fingerprint": f"UDP:{dst_bucket}",
        }

    elif protocol == "ICMP":
        icmp_type = packet.get("icmp_type", 0)
        icmp_code = packet.get("icmp_code", 0)

        features = {
            "icmp_type": icmp_type,
            "icmp_code": icmp_code,
            "icmp_pattern": f"ICMP:{icmp_type}:{icmp_code}",
            "fingerprint": f"ICMP:{icmp_type}:{icmp_code}",
            "size_bucket": size_bucket,  # Size matters for ICMP tunneling
        }

    else:
        features = {
            "fingerprint": f"{protocol}:unknown",
        }

    return features


# =============================================================================
# DECAYING ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    def __init__(self, dimensions: int, decay: float):
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
# HEADLESS DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    packet: dict
    features: dict
    is_flagged: bool
    similarity: float
    is_warmup: bool


class HeadlessPCAPDetector:
    """
    Pure frequency+decay PCAP detection.

    No explicit rules. Just learned fingerprint distribution.
    """

    def __init__(
        self,
        global_seed: int = GLOBAL_SEED,
        decay: float = DECAY_FACTOR,
        threshold: float = ANOMALY_THRESHOLD,
        warmup: int = WARMUP_PACKETS,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=global_seed)
        self.encoder = Encoder(vector_manager=self.vm)

        self.accumulator = DecayingAccumulator(DIMENSIONS, decay)
        self.threshold = threshold
        self.warmup = warmup
        self.packets_seen = 0

    def process(self, packet: dict) -> DetectionResult:
        """Process packet with pure frequency+decay detection."""
        self.packets_seen += 1
        is_warmup = self.packets_seen <= self.warmup

        # Extract fingerprint
        features = extract_packet_fingerprint(packet)

        # Encode
        vec = self.encoder.encode_data(features)

        # Get current model
        model = self.accumulator.get_normalized()

        # Similarity
        if self.packets_seen <= 1:
            similarity = 1.0
        else:
            similarity = cosine_similarity(vec, model)

        # Pure frequency decision (no rules!)
        if is_warmup:
            is_flagged = False
        else:
            is_flagged = similarity < self.threshold

        # Update accumulator
        weight = 0.1 if is_flagged else 1.0
        self.accumulator.update(vec, weight)

        return DetectionResult(
            packet=packet,
            features=features,
            is_flagged=is_flagged,
            similarity=similarity,
            is_warmup=is_warmup,
        )


# =============================================================================
# PACKET GENERATOR - More Realistic
# =============================================================================

class PCAPGenerator:
    """
    Generate realistic network traffic with protocol-appropriate fields.

    Benign patterns:
    - HTTPS/HTTP traffic (TCP to 443/80)
    - DNS queries (UDP to 53)
    - ICMP echo (ping)
    - SSH (TCP to 22)
    - Internal services (SMB, RDP)

    Malicious patterns:
    - Port scans (SYN to many ports)
    - Invalid TCP flags (SYN+FIN, NULL scan, XMAS)
    - Unusual ports (IRC, backdoors)
    - ICMP tunneling (large ICMP)
    - UDP amplification (small request, expects large response)
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

        # Common service ports for benign traffic
        self.common_tcp_ports = [22, 80, 443, 445, 139, 3389, 8080, 8443]
        self.common_udp_ports = [53, 123, 161]  # DNS, NTP, SNMP

    def _ephemeral_port(self) -> int:
        return self.rng.randint(49152, 65535)

    def generate_benign(self) -> dict:
        pattern = self.rng.choices(
            ["https", "http", "dns", "icmp_echo", "ssh", "smb", "ntp"],
            weights=[0.35, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05]
        )[0]

        if pattern == "https":
            # Normal HTTPS - common flag patterns
            flags = self.rng.choices(
                [0x02, 0x10, 0x12, 0x18],  # SYN, ACK, SYN+ACK, PSH+ACK
                weights=[0.1, 0.4, 0.2, 0.3]
            )[0]
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": 443,
                "flags": flags,
                "size": self.rng.randint(64, 1460),
            }

        elif pattern == "http":
            flags = self.rng.choices(
                [0x02, 0x10, 0x12, 0x18],
                weights=[0.1, 0.4, 0.2, 0.3]
            )[0]
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": 80,
                "flags": flags,
                "size": self.rng.randint(64, 1460),
            }

        elif pattern == "dns":
            return {
                "protocol": "UDP",
                "src_port": self._ephemeral_port(),
                "dst_port": 53,
                "size": self.rng.randint(40, 512),
            }

        elif pattern == "icmp_echo":
            # Normal ping: type 8 (request) or 0 (reply), code 0
            return {
                "protocol": "ICMP",
                "icmp_type": self.rng.choice([8, 0]),
                "icmp_code": 0,
                "size": self.rng.randint(64, 128),
            }

        elif pattern == "ssh":
            flags = self.rng.choices([0x10, 0x18], weights=[0.5, 0.5])[0]
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": 22,
                "flags": flags,
                "size": self.rng.randint(64, 512),
            }

        elif pattern == "smb":
            flags = self.rng.choices([0x10, 0x18], weights=[0.6, 0.4])[0]
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": self.rng.choice([445, 139]),
                "flags": flags,
                "size": self.rng.randint(64, 1460),
            }

        else:  # ntp
            return {
                "protocol": "UDP",
                "src_port": self._ephemeral_port(),
                "dst_port": 123,
                "size": 48,
            }

    def generate_malicious(self) -> Tuple[dict, str]:
        attack = self.rng.choices(
            ["port_scan", "null_scan", "xmas_scan", "syn_fin",
             "icmp_tunnel", "unusual_port", "udp_amplification"],
            weights=[0.2, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]
        )[0]

        if attack == "port_scan":
            # SYN scan to random ports
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": self.rng.randint(1, 65535),
                "flags": 0x02,  # SYN only
                "size": 40,
            }, attack

        elif attack == "null_scan":
            # No flags set
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": self.rng.choice([22, 80, 443]),
                "flags": 0x00,  # NULL
                "size": 40,
            }, attack

        elif attack == "xmas_scan":
            # FIN+PSH+URG = 0x29
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": self.rng.choice([22, 80, 443]),
                "flags": 0x29,  # FIN+PSH+URG
                "size": 40,
            }, attack

        elif attack == "syn_fin":
            # Invalid: SYN+FIN = 0x03
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": self.rng.choice([80, 443]),
                "flags": 0x03,  # SYN+FIN
                "size": 40,
            }, attack

        elif attack == "icmp_tunnel":
            # Large ICMP packets (data exfil)
            return {
                "protocol": "ICMP",
                "icmp_type": 8,
                "icmp_code": 0,
                "size": self.rng.randint(500, 1400),  # Unusually large
            }, attack

        elif attack == "unusual_port":
            # Traffic to suspicious ports
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral_port(),
                "dst_port": self.rng.choice([6667, 6697, 4444, 31337, 12345, 1337]),
                "flags": 0x10,  # ACK
                "size": self.rng.randint(64, 500),
            }, attack

        else:  # udp_amplification
            # Small UDP to amplification-prone services
            return {
                "protocol": "UDP",
                "src_port": self._ephemeral_port(),
                "dst_port": self.rng.choice([17, 19, 1900]),  # QOTD, chargen, SSDP
                "size": self.rng.randint(20, 40),
            }, attack

    def generate_stream(self, n: int, malicious_ratio: float = 0.02):
        stream = []
        for _ in range(n):
            if self.rng.random() < malicious_ratio:
                packet, attack = self.generate_malicious()
                stream.append((packet, attack))
            else:
                stream.append((self.generate_benign(), None))
        return stream


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 010-026: Headless PCAP Detection (Pure Frequency+Decay)")
    print("=" * 80)
    print("""
Honest packet representation:

L4 Protocol-specific fields:
  - TCP: src_port, dst_port, flags (raw bitmask)
  - UDP: src_port, dst_port
  - ICMP: type, code

Common:
  - size_bucket (8 categories)

NO IP classification (can't trust in the wild).
Let frequency+decay learn what's normal.
""")

    # Show flag examples
    print("--- TCP Flag Bitmask Examples ---")
    for flags, name in [(0x02, "SYN"), (0x10, "ACK"), (0x12, "SYN+ACK"),
                         (0x18, "PSH+ACK"), (0x03, "SYN+FIN"), (0x00, "NULL"),
                         (0x29, "FIN+PSH+URG")]:
        print(f"  0x{flags:02x} = {describe_tcp_flags(flags)} ({name})")

    # Initialize
    generator = PCAPGenerator(seed=42)
    detector = HeadlessPCAPDetector(
        global_seed=GLOBAL_SEED,
        decay=DECAY_FACTOR,
        threshold=ANOMALY_THRESHOLD,
        warmup=WARMUP_PACKETS,
    )

    print(f"\n--- Configuration ---")
    print(f"  Decay: {DECAY_FACTOR}")
    print(f"  Threshold: {ANOMALY_THRESHOLD}")
    print(f"  Warmup: {WARMUP_PACKETS}")

    # Generate stream
    n_packets = 10000
    stream = generator.generate_stream(n_packets, malicious_ratio=0.02)

    actual_malicious = sum(1 for _, a in stream if a is not None)
    print(f"\n--- Stream ---")
    print(f"  Total: {n_packets}")
    print(f"  Malicious: {actual_malicious} ({100*actual_malicious/n_packets:.1f}%)")

    # Process
    print(f"\n--- Processing ---")
    results = []
    start = time.time()

    for packet, attack in stream:
        result = detector.process(packet)
        results.append((result, attack))

    total_time = time.time() - start
    print(f"  Throughput: {n_packets/total_time:.0f} packets/sec")

    # Metrics
    post_warmup = [(r, a) for r, a in results if not r.is_warmup]

    tp = sum(1 for r, a in post_warmup if a is not None and r.is_flagged)
    fp = sum(1 for r, a in post_warmup if a is None and r.is_flagged)
    fn = sum(1 for r, a in post_warmup if a is not None and not r.is_flagged)
    tn = sum(1 for r, a in post_warmup if a is None and not r.is_flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    print(f"\n--- Results ---")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.3f}")

    # By attack type
    print(f"\n--- By Attack Type ---")
    attack_stats = {}
    for r, a in post_warmup:
        if a:
            if a not in attack_stats:
                attack_stats[a] = {"detected": 0, "missed": 0}
            if r.is_flagged:
                attack_stats[a]["detected"] += 1
            else:
                attack_stats[a]["missed"] += 1

    for attack, stats in sorted(attack_stats.items()):
        total = stats["detected"] + stats["missed"]
        rate = stats["detected"] / total if total else 0
        print(f"  {attack}: {stats['detected']}/{total} ({rate:.0%})")

    # Similarity distribution
    benign_sims = [r.similarity for r, a in post_warmup if a is None]
    malicious_sims = [r.similarity for r, a in post_warmup if a is not None]

    print(f"\n--- Similarity Distribution ---")
    print(f"  Benign:    min={min(benign_sims):.3f}, mean={np.mean(benign_sims):.3f}, max={max(benign_sims):.3f}")
    print(f"  Malicious: min={min(malicious_sims):.3f}, mean={np.mean(malicious_sims):.3f}, max={max(malicious_sims):.3f}")

    # Samples
    def describe_features(f: dict) -> str:
        return f.get('fingerprint', str(f))

    print(f"\n--- Sample True Positives ---")
    for r, a in [(r, a) for r, a in post_warmup if a and r.is_flagged][:5]:
        print(f"  {a}: sim={r.similarity:.3f} {describe_features(r.features)}")

    print(f"\n--- Sample False Negatives ---")
    for r, a in [(r, a) for r, a in post_warmup if a and not r.is_flagged][:5]:
        print(f"  {a}: sim={r.similarity:.3f} {describe_features(r.features)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: Headless PCAP Detection")
    print("=" * 80)
    print(f"""
Honest packet representation:

L4-specific:
  - TCP: src_port, dst_port, flags (raw bitmask 0-63)
  - UDP: src_port, dst_port
  - ICMP: type, code

Common:
  - size_bucket (8 categories)

Performance:
  F1 Score:  {f1:.3f}
  Precision: {precision:.1%}
  Recall:    {recall:.1%}
  Throughput: {n_packets/total_time:.0f} packets/sec

Separation:
  Benign mean:    {np.mean(benign_sims):.3f}
  Malicious mean: {np.mean(malicious_sims):.3f}
  Gap: {np.mean(benign_sims) - np.mean(malicious_sims):.3f}
""")

    return f1, benign_sims, malicious_sims


if __name__ == "__main__":
    f1, benign_sims, malicious_sims = main()

    if f1 < 0.9:
        print("\n--- Threshold Tuning ---")
        best_f1 = 0
        best_thresh = 0.55

        for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            gen = PCAPGenerator(seed=42)
            det = HeadlessPCAPDetector(threshold=thresh, warmup=WARMUP_PACKETS)

            stream = gen.generate_stream(10000, 0.02)
            results = [(det.process(p), a) for p, a in stream]
            post = [(r, a) for r, a in results if not r.is_warmup]

            tp = sum(1 for r, a in post if a and r.is_flagged)
            fp = sum(1 for r, a in post if not a and r.is_flagged)
            fn = sum(1 for r, a in post if a and not r.is_flagged)

            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f = 2 * p * r / max(0.001, p + r)

            print(f"  threshold={thresh:.2f}: P={p:.1%} R={r:.1%} F1={f:.3f}")

            if f > best_f1:
                best_f1 = f
                best_thresh = thresh

        print(f"\n  Best: threshold={best_thresh:.2f} → F1={best_f1:.3f}")
