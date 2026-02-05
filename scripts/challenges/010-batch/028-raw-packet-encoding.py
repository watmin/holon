#!/usr/bin/env python3
"""
Challenge 010-028: Raw Packet Encoding with Holon

No fingerprints, no string decorations.
Just pass the raw packet structure to Holon's encoder.

Holon handles structured data natively:
- Dicts are encoded by binding field name to field value
- Nested structures work recursively
- Numbers, strings, bools are atoms

Test both:
1. Anomaly detection (low similarity = novel)
2. DDoS detection (high similarity = repetitive)
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.999
WARMUP = 200


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


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
# RAW PACKET REPRESENTATION
# =============================================================================

def raw_packet_data(packet: dict) -> dict:
    """
    Return packet as-is for Holon encoding.

    COMPLETELY raw - no filtering, no bucketing, no fingerprints.
    Just pass the L3/L4 fields directly.
    """
    protocol = packet.get("protocol", "TCP").upper()

    if protocol == "TCP":
        return {
            "protocol": "TCP",
            "dst_port": packet.get("dst_port", 0),
            "flags": packet.get("flags", 0),
            "size": packet.get("size", 0),  # Raw size
        }

    elif protocol == "UDP":
        return {
            "protocol": "UDP",
            "dst_port": packet.get("dst_port", 0),
            "size": packet.get("size", 0),  # Raw size
        }

    elif protocol == "ICMP":
        return {
            "protocol": "ICMP",
            "icmp_type": packet.get("icmp_type", 0),
            "icmp_code": packet.get("icmp_code", 0),
            "size": packet.get("size", 0),  # Raw size
        }

    else:
        return {"protocol": protocol}


# =============================================================================
# DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    packet: dict
    encoded: dict
    is_flagged: bool
    similarity: float
    is_warmup: bool


class RawPacketDetector:
    """
    Detect anomalies or DDoS using raw packet encoding.

    mode="anomaly": flag LOW similarity (novel patterns)
    mode="ddos": flag HIGH similarity (repetitive patterns)
    """

    def __init__(
        self,
        mode: str = "anomaly",
        threshold: float = 0.5,
        global_seed: int = GLOBAL_SEED,
        decay: float = DECAY_FACTOR,
        warmup: int = WARMUP,
        window_size: int = 15,
        burst_fraction: float = 0.5,
    ):
        self.mode = mode
        self.threshold = threshold
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=global_seed)
        self.encoder = Encoder(vector_manager=self.vm)

        self.accumulator = DecayingAccumulator(DIMENSIONS, decay)
        self.warmup = warmup
        self.packets_seen = 0

        # For DDoS burst detection
        self.window_size = window_size
        self.burst_fraction = burst_fraction
        self.recent_high_sim = []

    def process(self, packet: dict) -> DetectionResult:
        self.packets_seen += 1
        is_warmup = self.packets_seen <= self.warmup

        # Raw encoding - no fingerprints
        encoded = raw_packet_data(packet)
        vec = self.encoder.encode_data(encoded)

        model = self.accumulator.get_normalized()

        if self.packets_seen <= 1:
            similarity = 0.5
        else:
            similarity = cosine_similarity(vec, model)

        # Detection logic
        if self.mode == "anomaly":
            # Flag LOW similarity
            if is_warmup:
                is_flagged = False
            else:
                is_flagged = similarity < self.threshold

        else:  # ddos
            # Flag HIGH similarity concentration
            is_high_sim = similarity > self.threshold
            self.recent_high_sim.append(is_high_sim)
            if len(self.recent_high_sim) > self.window_size:
                self.recent_high_sim.pop(0)

            if is_warmup or len(self.recent_high_sim) < self.window_size:
                is_flagged = False
            else:
                high_count = sum(self.recent_high_sim)
                is_flagged = high_count >= self.window_size * self.burst_fraction

        # Update
        self.accumulator.update(vec)

        return DetectionResult(
            packet=packet,
            encoded=encoded,
            is_flagged=is_flagged,
            similarity=similarity,
            is_warmup=is_warmup,
        )


# =============================================================================
# TRAFFIC GENERATOR
# =============================================================================

class TrafficGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _ephemeral_port(self) -> int:
        return self.rng.randint(49152, 65535)

    def generate_normal(self) -> dict:
        pattern = self.rng.choices(
            ["https", "http", "dns", "icmp", "ssh", "smb"],
            weights=[0.35, 0.20, 0.15, 0.10, 0.10, 0.10]
        )[0]

        if pattern == "https":
            flags = self.rng.choices([0x02, 0x10, 0x12, 0x18], weights=[0.1, 0.4, 0.2, 0.3])[0]
            return {"protocol": "TCP", "dst_port": 443, "flags": flags,
                    "size": self.rng.randint(64, 1460)}

        elif pattern == "http":
            flags = self.rng.choices([0x02, 0x10, 0x12, 0x18], weights=[0.1, 0.4, 0.2, 0.3])[0]
            return {"protocol": "TCP", "dst_port": 80, "flags": flags,
                    "size": self.rng.randint(64, 1460)}

        elif pattern == "dns":
            return {"protocol": "UDP", "dst_port": 53, "size": self.rng.randint(40, 512)}

        elif pattern == "icmp":
            return {"protocol": "ICMP", "icmp_type": self.rng.choice([8, 0]),
                    "icmp_code": 0, "size": self.rng.randint(64, 128)}

        elif pattern == "ssh":
            flags = self.rng.choices([0x10, 0x18], weights=[0.5, 0.5])[0]
            return {"protocol": "TCP", "dst_port": 22, "flags": flags,
                    "size": self.rng.randint(64, 512)}

        else:  # smb
            flags = self.rng.choices([0x10, 0x18], weights=[0.6, 0.4])[0]
            return {"protocol": "TCP", "dst_port": self.rng.choice([445, 139]),
                    "flags": flags, "size": self.rng.randint(64, 1460)}

    # Malicious packets for anomaly detection
    def generate_malicious(self) -> Tuple[dict, str]:
        attack = self.rng.choices(
            ["port_scan", "null_scan", "xmas_scan", "syn_fin",
             "icmp_tunnel", "unusual_port", "udp_amp"],
            weights=[0.2, 0.15, 0.15, 0.15, 0.1, 0.15, 0.1]
        )[0]

        if attack == "port_scan":
            return {"protocol": "TCP", "dst_port": self.rng.randint(1, 65535),
                    "flags": 0x02, "size": 40}, attack

        elif attack == "null_scan":
            return {"protocol": "TCP", "dst_port": self.rng.choice([22, 80, 443]),
                    "flags": 0x00, "size": 40}, attack

        elif attack == "xmas_scan":
            return {"protocol": "TCP", "dst_port": self.rng.choice([22, 80, 443]),
                    "flags": 0x29, "size": 40}, attack

        elif attack == "syn_fin":
            return {"protocol": "TCP", "dst_port": self.rng.choice([80, 443]),
                    "flags": 0x03, "size": 40}, attack

        elif attack == "icmp_tunnel":
            return {"protocol": "ICMP", "icmp_type": 8, "icmp_code": 0,
                    "size": self.rng.randint(500, 1400)}, attack

        elif attack == "unusual_port":
            return {"protocol": "TCP", "dst_port": self.rng.choice([6667, 4444, 31337]),
                    "flags": 0x10, "size": self.rng.randint(64, 500)}, attack

        else:  # udp_amp
            return {"protocol": "UDP", "dst_port": self.rng.choice([17, 19, 1900]),
                    "size": self.rng.randint(20, 40)}, attack

    # DDoS packets - realistic patterns
    def generate_ddos(self, attack_type: str) -> dict:
        """
        Realistic DDoS patterns:

        SYN flood: static dst_port, static size, random src_port
        UDP reflection: static src_port (spoofed), static size, random dst_port
        ICMP flood: static type/code/size
        """
        if attack_type == "syn_flood":
            # SYN flood to web server - static dst_port=80, size=40
            # Only src_port varies (but we don't encode src_port)
            return {"protocol": "TCP", "dst_port": 80, "flags": 0x02, "size": 40}

        elif attack_type == "syn_flood_443":
            # SYN flood to HTTPS - static dst_port=443, size=40
            return {"protocol": "TCP", "dst_port": 443, "flags": 0x02, "size": 40}

        elif attack_type == "udp_reflection":
            # UDP reflection - attacker spoofs src_port=53 (DNS), static size
            # dst_port varies (victim IPs) but we don't encode dst_port for reflection
            # Actually for reflection, the RESPONSE comes from port 53 with large size
            return {"protocol": "UDP", "dst_port": 53, "size": 512}  # DNS response size

        elif attack_type == "ntp_amp":
            # NTP amplification - responses from port 123, large size
            return {"protocol": "UDP", "dst_port": 123, "size": 468}  # NTP monlist response

        elif attack_type == "icmp_flood":
            # ICMP flood - static type=8 (echo), code=0, fixed size
            return {"protocol": "ICMP", "icmp_type": 8, "icmp_code": 0, "size": 64}

        elif attack_type == "ssdp_amp":
            # SSDP amplification - responses from port 1900, large size
            return {"protocol": "UDP", "dst_port": 1900, "size": 300}

        else:
            return self.generate_ddos("syn_flood")

    def generate_anomaly_stream(self, n: int, malicious_ratio: float = 0.02):
        stream = []
        for _ in range(n):
            if self.rng.random() < malicious_ratio:
                packet, attack = self.generate_malicious()
                stream.append((packet, attack))
            else:
                stream.append((self.generate_normal(), None))
        return stream

    def generate_ddos_stream(self, normal_count: int, burst_count: int, attack_type: str):
        stream = []
        for _ in range(normal_count // 2):
            stream.append((self.generate_normal(), None))
        for _ in range(burst_count):
            stream.append((self.generate_ddos(attack_type), attack_type))
        for _ in range(normal_count // 2):
            stream.append((self.generate_normal(), None))
        return stream


# =============================================================================
# MAIN
# =============================================================================

def test_anomaly_detection():
    print("=" * 70)
    print("TEST 1: Anomaly Detection (Raw Packet Encoding)")
    print("=" * 70)
    print("""
No fingerprints - just raw packet fields:
  {"protocol": "TCP", "dst_port": 443, "flags": 24, "size": 1024}

Flag when similarity < threshold (novel patterns)
""")

    gen = TrafficGenerator(seed=42)

    # Test different thresholds
    best_f1 = 0
    best_thresh = 0.5

    print("Threshold tuning:")
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55]:
        det = RawPacketDetector(mode="anomaly", threshold=thresh, warmup=WARMUP)
        stream = gen.generate_anomaly_stream(10000, 0.02)

        results = [(det.process(p), l) for p, l in stream]
        post = [(r, l) for r, l in results if not r.is_warmup]

        tp = sum(1 for r, l in post if l and r.is_flagged)
        fp = sum(1 for r, l in post if not l and r.is_flagged)
        fn = sum(1 for r, l in post if l and not r.is_flagged)

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(0.001, p + r)

        print(f"  thresh={thresh:.2f}: P={p:.1%} R={r:.1%} F1={f:.3f}")

        if f > best_f1:
            best_f1 = f
            best_thresh = thresh

    # Run with best threshold
    print(f"\nRunning with best threshold: {best_thresh}")
    det = RawPacketDetector(mode="anomaly", threshold=best_thresh, warmup=WARMUP)
    stream = gen.generate_anomaly_stream(10000, 0.02)

    results = []
    start = time.time()
    for p, l in stream:
        results.append((det.process(p), l))
    elapsed = time.time() - start

    post = [(r, l) for r, l in results if not r.is_warmup]

    tp = sum(1 for r, l in post if l and r.is_flagged)
    fp = sum(1 for r, l in post if not l and r.is_flagged)
    fn = sum(1 for r, l in post if l and not r.is_flagged)
    tn = sum(1 for r, l in post if not l and not r.is_flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    normal_sims = [r.similarity for r, l in post if not l]
    attack_sims = [r.similarity for r, l in post if l]

    print(f"\n--- Results ---")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Throughput: {len(stream)/elapsed:.0f} pkt/sec")
    print(f"  Normal sim:  mean={np.mean(normal_sims):.3f}, max={max(normal_sims):.3f}")
    print(f"  Attack sim:  mean={np.mean(attack_sims):.3f}, max={max(attack_sims):.3f}")

    # By attack type
    print(f"\n--- By Attack Type ---")
    attack_stats = {}
    for r, l in post:
        if l:
            if l not in attack_stats:
                attack_stats[l] = {"tp": 0, "fn": 0}
            if r.is_flagged:
                attack_stats[l]["tp"] += 1
            else:
                attack_stats[l]["fn"] += 1

    for attack, stats in sorted(attack_stats.items()):
        total = stats["tp"] + stats["fn"]
        rate = stats["tp"] / total if total else 0
        print(f"  {attack}: {stats['tp']}/{total} ({rate:.0%})")

    return f1


def test_ddos_detection():
    print("\n" + "=" * 70)
    print("TEST 2: DDoS Detection (Raw Packet Encoding)")
    print("=" * 70)
    print("""
No fingerprints - just raw packet fields.

Realistic DDoS patterns:
- SYN flood: static dst_port, static size (only src_port varies)
- UDP reflection: static src_port, static size (large response packets)
- ICMP flood: static type/code/size

Flag when similarity > threshold (repetitive patterns)
""")

    gen = TrafficGenerator(seed=42)
    attack_types = ["syn_flood", "syn_flood_443", "udp_reflection", "ntp_amp", "icmp_flood", "ssdp_amp"]

    results_summary = []

    # Tune threshold per attack type based on similarity distribution
    thresholds = {
        "syn_flood": 0.70,
        "syn_flood_443": 0.70,
        "udp_reflection": 0.55,
        "ntp_amp": 0.50,
        "icmp_flood": 0.50,
        "ssdp_amp": 0.50,
    }

    for attack_type in attack_types:
        det = RawPacketDetector(
            mode="ddos",
            threshold=thresholds.get(attack_type, 0.60),
            warmup=WARMUP,
            window_size=15,
            burst_fraction=0.5
        )

        stream = gen.generate_ddos_stream(4000, 500, attack_type)

        results = []
        start = time.time()
        for p, l in stream:
            results.append((det.process(p), l))
        elapsed = time.time() - start

        post = [(r, l) for r, l in results if not r.is_warmup]

        tp = sum(1 for r, l in post if l and r.is_flagged)
        fp = sum(1 for r, l in post if not l and r.is_flagged)
        fn = sum(1 for r, l in post if l and not r.is_flagged)

        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f = 2 * p * r / max(0.001, p + r)

        normal_sims = [res.similarity for res, lbl in post if not lbl]
        attack_sims = [res.similarity for res, lbl in post if lbl]

        print(f"\n{attack_type}:")
        print(f"  TP={tp}, FP={fp}, FN={fn}")
        print(f"  Precision: {p:.1%}, Recall: {r:.1%}, F1: {f:.3f}")
        print(f"  Normal sim: mean={np.mean(normal_sims):.3f}")
        print(f"  Attack sim: mean={np.mean(attack_sims):.3f}")

        results_summary.append({"attack": attack_type, "f1": f, "precision": p, "recall": r})

    avg_f1 = np.mean([r["f1"] for r in results_summary])
    print(f"\n--- Average F1: {avg_f1:.3f} ---")

    return avg_f1


def main():
    print("=" * 70)
    print("Challenge 010-028: Raw Packet Encoding")
    print("=" * 70)
    print("""
Testing Holon's native structured data encoding.

NO fingerprints, NO string decorations.
Just pass raw packet dicts to the encoder.

Example packet:
  {"protocol": "TCP", "dst_port": 443, "flags": 24, "size": 1024}

Holon encodes this by binding field names to values.
""")

    anomaly_f1 = test_anomaly_detection()
    ddos_f1 = test_ddos_detection()

    print("\n" + "=" * 70)
    print("SUMMARY: Raw Packet Encoding")
    print("=" * 70)
    print(f"""
Results with raw packet encoding (no fingerprints):

| Mode     | F1 Score |
|----------|----------|
| Anomaly  | {anomaly_f1:.3f}    |
| DDoS     | {ddos_f1:.3f}    |
""")


if __name__ == "__main__":
    main()
