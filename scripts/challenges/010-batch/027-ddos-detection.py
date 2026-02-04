#!/usr/bin/env python3
"""
Challenge 010-027: DDoS Detection via High Similarity

Flip the anomaly detection logic:
- Normal traffic: diverse fingerprints → accumulator is spread
- DDoS attack: repetitive fingerprints → accumulator concentrates

Detection: Flag when similarity is TOO HIGH (packet matches accumulated pattern too well)

Attack types:
- SYN flood: many SYN packets to same port
- UDP flood: many UDP packets to same port
- ICMP flood: many echo requests
- Amplification: small requests to amplification-prone services
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from deterministic_codebook import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.999  # Faster decay to track recent patterns
HIGH_SIMILARITY_THRESHOLD = 0.70  # Flag if similarity > this
WARMUP_PACKETS = 200


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# PACKET FINGERPRINTING (same as 026)
# =============================================================================

def bucket_size(size: int) -> int:
    if size <= 40: return 0
    elif size <= 64: return 1
    elif size <= 128: return 2
    elif size <= 256: return 3
    elif size <= 512: return 4
    elif size <= 1024: return 5
    elif size <= 1500: return 6
    else: return 7


def bucket_port(port: int) -> str:
    if port in {80, 443, 8080, 8443}: return "web"
    elif port in {53, 123}: return "dns_ntp"
    elif port in {22, 23, 3389}: return "admin"
    elif port in {445, 139, 135}: return "windows"
    elif port in {25, 587, 465, 110, 143, 993, 995}: return "mail"
    elif port in {3306, 5432, 1433, 27017, 6379}: return "db"
    elif port in {6667, 6697, 4444, 31337, 1337, 12345, 17, 19, 1900}: return "suspicious"
    elif port < 1024: return "priv_other"
    elif port < 49152: return "registered"
    else: return "ephemeral"


def extract_packet_fingerprint(packet: dict) -> dict:
    """Extract structural fingerprint."""
    protocol = packet.get("protocol", "TCP").upper()
    size = packet.get("size", 0)
    size_bucket = bucket_size(size)

    if protocol == "TCP":
        dst_port = packet.get("dst_port", 0)
        flags = packet.get("flags", 0)
        if isinstance(flags, list):
            flag_map = {"FIN": 0x01, "SYN": 0x02, "RST": 0x04, "PSH": 0x08, "ACK": 0x10, "URG": 0x20}
            mask = 0
            for f in flags:
                mask |= flag_map.get(f.upper(), 0)
            flags = mask

        dst_bucket = bucket_port(dst_port)

        features = {
            "dst_bucket": dst_bucket,
            "dst_bucket_key": f"dst:{dst_bucket}",
            "tcp_flags": flags,
            "tcp_flags_key": f"flags:{flags:02x}",
            "tcp_pattern": f"TCP:{dst_bucket}:{flags:02x}",
            "fingerprint": f"TCP:{dst_bucket}:{flags:02x}",
        }

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
            "size_bucket": size_bucket,
        }

    else:
        features = {"fingerprint": f"{protocol}:unknown"}

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
# DDOS DETECTOR - HIGH SIMILARITY = SUSPICIOUS
# =============================================================================

@dataclass
class DetectionResult:
    packet: dict
    features: dict
    is_flagged: bool
    similarity: float
    is_warmup: bool


class DDoSDetector:
    """
    Detect DDoS by flagging HIGH similarity concentration.

    Normal traffic: diverse → low/medium similarity, occasional high
    DDoS: repetitive → sustained high similarity (burst)

    Uses a sliding window to detect CONCENTRATION of high-similarity packets.
    """

    def __init__(
        self,
        global_seed: int = GLOBAL_SEED,
        decay: float = DECAY_FACTOR,
        threshold: float = HIGH_SIMILARITY_THRESHOLD,
        warmup: int = WARMUP_PACKETS,
        window_size: int = 15,
        burst_fraction: float = 0.5,  # 50% of window must be high-sim
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=global_seed)
        self.encoder = Encoder(vector_manager=self.vm)

        self.accumulator = DecayingAccumulator(DIMENSIONS, decay)
        self.threshold = threshold
        self.warmup = warmup
        self.packets_seen = 0

        # Sliding window for burst detection
        self.window_size = window_size
        self.burst_fraction = burst_fraction
        self.recent_high_sim = []  # Track recent high-sim flags

    def process(self, packet: dict) -> DetectionResult:
        """Process packet - flag sustained HIGH similarity as DDoS."""
        self.packets_seen += 1
        is_warmup = self.packets_seen <= self.warmup

        features = extract_packet_fingerprint(packet)
        vec = self.encoder.encode_data(features)
        model = self.accumulator.get_normalized()

        if self.packets_seen <= 1:
            similarity = 0.0
        else:
            similarity = cosine_similarity(vec, model)

        # Track high-similarity in sliding window
        is_high_sim = similarity > self.threshold
        self.recent_high_sim.append(is_high_sim)
        if len(self.recent_high_sim) > self.window_size:
            self.recent_high_sim.pop(0)

        # Burst detection: flag if fraction of window is high-sim
        if is_warmup or len(self.recent_high_sim) < self.window_size:
            is_flagged = False
        else:
            high_sim_count = sum(self.recent_high_sim)
            is_flagged = high_sim_count >= self.window_size * self.burst_fraction

        # Update accumulator
        self.accumulator.update(vec)

        return DetectionResult(
            packet=packet,
            features=features,
            is_flagged=is_flagged,
            similarity=similarity,
            is_warmup=is_warmup,
        )


# =============================================================================
# TRAFFIC GENERATOR
# =============================================================================

class TrafficGenerator:
    """Generate normal traffic and DDoS attacks."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _ephemeral_port(self) -> int:
        return self.rng.randint(49152, 65535)

    def generate_normal(self) -> dict:
        """Diverse normal traffic."""
        pattern = self.rng.choices(
            ["https", "http", "dns", "icmp", "ssh", "smb", "ntp", "mail", "db"],
            weights=[0.30, 0.15, 0.15, 0.05, 0.10, 0.10, 0.05, 0.05, 0.05]
        )[0]

        if pattern == "https":
            flags = self.rng.choices([0x02, 0x10, 0x12, 0x18], weights=[0.1, 0.4, 0.2, 0.3])[0]
            return {"protocol": "TCP", "src_port": self._ephemeral_port(), "dst_port": 443,
                    "flags": flags, "size": self.rng.randint(64, 1460)}

        elif pattern == "http":
            flags = self.rng.choices([0x02, 0x10, 0x12, 0x18], weights=[0.1, 0.4, 0.2, 0.3])[0]
            return {"protocol": "TCP", "src_port": self._ephemeral_port(), "dst_port": 80,
                    "flags": flags, "size": self.rng.randint(64, 1460)}

        elif pattern == "dns":
            return {"protocol": "UDP", "src_port": self._ephemeral_port(), "dst_port": 53,
                    "size": self.rng.randint(40, 512)}

        elif pattern == "icmp":
            return {"protocol": "ICMP", "icmp_type": self.rng.choice([8, 0]), "icmp_code": 0,
                    "size": self.rng.randint(64, 128)}

        elif pattern == "ssh":
            flags = self.rng.choices([0x10, 0x18], weights=[0.5, 0.5])[0]
            return {"protocol": "TCP", "src_port": self._ephemeral_port(), "dst_port": 22,
                    "flags": flags, "size": self.rng.randint(64, 512)}

        elif pattern == "smb":
            flags = self.rng.choices([0x10, 0x18], weights=[0.6, 0.4])[0]
            return {"protocol": "TCP", "src_port": self._ephemeral_port(),
                    "dst_port": self.rng.choice([445, 139]), "flags": flags,
                    "size": self.rng.randint(64, 1460)}

        elif pattern == "ntp":
            return {"protocol": "UDP", "src_port": self._ephemeral_port(), "dst_port": 123, "size": 48}

        elif pattern == "mail":
            flags = self.rng.choices([0x10, 0x18], weights=[0.5, 0.5])[0]
            return {"protocol": "TCP", "src_port": self._ephemeral_port(),
                    "dst_port": self.rng.choice([25, 587, 143]), "flags": flags,
                    "size": self.rng.randint(64, 1460)}

        else:  # db
            flags = self.rng.choices([0x10, 0x18], weights=[0.5, 0.5])[0]
            return {"protocol": "TCP", "src_port": self._ephemeral_port(),
                    "dst_port": self.rng.choice([3306, 5432]), "flags": flags,
                    "size": self.rng.randint(64, 1460)}

    def generate_ddos(self, attack_type: str) -> dict:
        """Generate DDoS attack packet - SAME fingerprint repeatedly."""

        if attack_type == "syn_flood":
            # SYN flood to web server
            return {"protocol": "TCP", "src_port": self._ephemeral_port(), "dst_port": 80,
                    "flags": 0x02, "size": 40}

        elif attack_type == "udp_flood":
            # UDP flood
            return {"protocol": "UDP", "src_port": self._ephemeral_port(), "dst_port": 53,
                    "size": 64}

        elif attack_type == "icmp_flood":
            # ICMP echo flood
            return {"protocol": "ICMP", "icmp_type": 8, "icmp_code": 0, "size": 64}

        elif attack_type == "ntp_amplification":
            # NTP amplification
            return {"protocol": "UDP", "src_port": self._ephemeral_port(), "dst_port": 123,
                    "size": 48}

        elif attack_type == "syn_flood_https":
            # SYN flood to HTTPS
            return {"protocol": "TCP", "src_port": self._ephemeral_port(), "dst_port": 443,
                    "flags": 0x02, "size": 40}

        else:
            return self.generate_ddos("syn_flood")

    def generate_stream_with_burst(
        self,
        normal_count: int = 5000,
        burst_count: int = 500,
        attack_type: str = "syn_flood"
    ) -> List[Tuple[dict, str]]:
        """
        Generate stream: normal traffic, then DDoS burst, then more normal.

        Returns list of (packet, label) where label is None for normal or attack_type for DDoS.
        """
        stream = []

        # First half normal
        for _ in range(normal_count // 2):
            stream.append((self.generate_normal(), None))

        # DDoS burst
        for _ in range(burst_count):
            stream.append((self.generate_ddos(attack_type), attack_type))

        # Second half normal
        for _ in range(normal_count // 2):
            stream.append((self.generate_normal(), None))

        return stream


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("Challenge 010-027: DDoS Detection via High Similarity")
    print("=" * 80)
    print("""
INVERTED LOGIC:
- Normal traffic: diverse fingerprints → LOW similarity to accumulated model
- DDoS attack: repetitive fingerprints → HIGH similarity (same pattern repeats)

Detection: Flag when similarity > threshold (too repetitive)
""")

    generator = TrafficGenerator(seed=42)

    # Test each attack type
    attack_types = ["syn_flood", "udp_flood", "icmp_flood", "ntp_amplification", "syn_flood_https"]

    results_summary = []

    for attack_type in attack_types:
        print(f"\n{'='*60}")
        print(f"Testing: {attack_type}")
        print(f"{'='*60}")

        detector = DDoSDetector(
            global_seed=GLOBAL_SEED,
            decay=DECAY_FACTOR,
            threshold=HIGH_SIMILARITY_THRESHOLD,
            warmup=WARMUP_PACKETS,
        )

        # Generate stream with burst
        stream = generator.generate_stream_with_burst(
            normal_count=4000,
            burst_count=500,
            attack_type=attack_type
        )

        # Process
        results = []
        start = time.time()
        for packet, label in stream:
            result = detector.process(packet)
            results.append((result, label))
        total_time = time.time() - start

        # Metrics (post-warmup)
        post_warmup = [(r, l) for r, l in results if not r.is_warmup]

        tp = sum(1 for r, l in post_warmup if l is not None and r.is_flagged)
        fp = sum(1 for r, l in post_warmup if l is None and r.is_flagged)
        fn = sum(1 for r, l in post_warmup if l is not None and not r.is_flagged)
        tn = sum(1 for r, l in post_warmup if l is None and not r.is_flagged)

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(0.001, precision + recall)

        # Similarity distribution
        normal_sims = [r.similarity for r, l in post_warmup if l is None]
        attack_sims = [r.similarity for r, l in post_warmup if l is not None]

        print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"  Precision: {precision:.1%}")
        print(f"  Recall:    {recall:.1%}")
        print(f"  F1 Score:  {f1:.3f}")
        print(f"  Throughput: {len(stream)/total_time:.0f} packets/sec")
        print(f"  Normal sim:  min={min(normal_sims):.3f}, mean={np.mean(normal_sims):.3f}, max={max(normal_sims):.3f}")
        print(f"  Attack sim:  min={min(attack_sims):.3f}, mean={np.mean(attack_sims):.3f}, max={max(attack_sims):.3f}")

        results_summary.append({
            "attack": attack_type,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "attack_mean_sim": np.mean(attack_sims),
            "normal_mean_sim": np.mean(normal_sims),
        })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY: DDoS Detection Results")
    print("=" * 80)
    print(f"\nThreshold: {HIGH_SIMILARITY_THRESHOLD} (flag if similarity > this)")
    print(f"Decay: {DECAY_FACTOR}")
    print()

    print(f"{'Attack Type':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Attack Sim':>12} {'Normal Sim':>12}")
    print("-" * 72)
    for r in results_summary:
        print(f"{r['attack']:<20} {r['f1']:>8.3f} {r['precision']:>10.1%} {r['recall']:>8.1%} {r['attack_mean_sim']:>12.3f} {r['normal_mean_sim']:>12.3f}")

    avg_f1 = np.mean([r['f1'] for r in results_summary])
    print("-" * 72)
    print(f"{'Average':<20} {avg_f1:>8.3f}")

    return results_summary


if __name__ == "__main__":
    results = main()

    # Threshold tuning
    print("\n" + "=" * 80)
    print("Threshold Tuning")
    print("=" * 80)

    for thresh in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        gen = TrafficGenerator(seed=42)
        all_f1s = []

        for attack_type in ["syn_flood", "udp_flood", "icmp_flood"]:
            det = DDoSDetector(threshold=thresh, warmup=WARMUP_PACKETS)
            stream = gen.generate_stream_with_burst(4000, 500, attack_type)
            results = [(det.process(p), l) for p, l in stream]
            post = [(r, l) for r, l in results if not r.is_warmup]

            tp = sum(1 for r, l in post if l and r.is_flagged)
            fp = sum(1 for r, l in post if not l and r.is_flagged)
            fn = sum(1 for r, l in post if l and not r.is_flagged)

            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f = 2 * p * r / max(0.001, p + r)
            all_f1s.append(f)

        avg = np.mean(all_f1s)
        print(f"  threshold={thresh:.2f}: avg F1={avg:.3f}")
