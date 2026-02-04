#!/usr/bin/env python3
"""
Challenge 010-029: Realistic DDoS Detection

Key insights:

NORMAL TRAFFIC:
- TCP: dst_port 443/80, mostly ACK flags, ephemeral src_port
- UDP: consistent dst_port (game/app), ephemeral src_port, bounded size
- ICMP: rare/minimal

DDoS SIGNALS:
- UDP reflection: concentration of src_port FROM well-known ports (53, 123, 1900)
  Normal: src_port=ephemeral, dst_port=service
  Attack: src_port=53 (DNS server responding), dst_port=victim

- TCP SYN flood: concentration of SYN flag (not ACK)
  Normal: mostly ACK, some SYN
  Attack: massive SYN concentration

- ICMP flood: unusual ICMP volume (since normally rare)

Each test is ISOLATED - fresh detector to avoid pollution.
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from deterministic_codebook import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.999
WARMUP = 500


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

    def update(self, vector: np.ndarray):
        self.accumulator = self.decay * self.accumulator + vector.astype(np.float64)
        self.count += 1

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)


# =============================================================================
# RAW PACKET ENCODING
# =============================================================================

def encode_packet(packet: dict) -> dict:
    """
    Encode packet for Holon.

    Include src_port to detect reflection attacks!
    """
    protocol = packet.get("protocol", "TCP").upper()

    if protocol == "TCP":
        return {
            "protocol": "TCP",
            "src_port": packet.get("src_port", 0),
            "dst_port": packet.get("dst_port", 0),
            "flags": packet.get("flags", 0),
        }

    elif protocol == "UDP":
        return {
            "protocol": "UDP",
            "src_port": packet.get("src_port", 0),  # KEY for reflection detection
            "dst_port": packet.get("dst_port", 0),
        }

    elif protocol == "ICMP":
        return {
            "protocol": "ICMP",
            "icmp_type": packet.get("icmp_type", 0),
            "icmp_code": packet.get("icmp_code", 0),
        }

    return {"protocol": protocol}


# =============================================================================
# DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    packet: dict
    is_flagged: bool
    similarity: float
    is_warmup: bool


class DDoSDetector:
    """
    Detect DDoS via similarity patterns.

    Two modes:
    - "high_sim": Flag high similarity concentration (e.g., SYN flood)
    - "low_sim": Flag low similarity concentration (e.g., UDP reflection)
    """

    def __init__(
        self,
        mode: str = "high_sim",  # or "low_sim"
        threshold: float = 0.70,
        warmup: int = WARMUP,
        window_size: int = 20,
        burst_fraction: float = 0.5,
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
        self.encoder = Encoder(vector_manager=self.vm)

        self.accumulator = DecayingAccumulator(DIMENSIONS, DECAY_FACTOR)
        self.mode = mode
        self.threshold = threshold
        self.warmup = warmup
        self.packets_seen = 0

        self.window_size = window_size
        self.burst_fraction = burst_fraction
        self.recent_trigger = []

    def process(self, packet: dict) -> DetectionResult:
        self.packets_seen += 1
        is_warmup = self.packets_seen <= self.warmup

        encoded = encode_packet(packet)
        vec = self.encoder.encode_data(encoded)
        model = self.accumulator.get_normalized()

        similarity = cosine_similarity(vec, model) if self.packets_seen > 1 else 0.5

        # Mode-specific trigger
        if self.mode == "high_sim":
            is_trigger = similarity > self.threshold
        else:  # low_sim
            is_trigger = similarity < self.threshold

        self.recent_trigger.append(is_trigger)
        if len(self.recent_trigger) > self.window_size:
            self.recent_trigger.pop(0)

        # Burst detection
        if is_warmup or len(self.recent_trigger) < self.window_size:
            is_flagged = False
        else:
            trigger_count = sum(self.recent_trigger)
            is_flagged = trigger_count >= self.window_size * self.burst_fraction

        self.accumulator.update(vec)

        return DetectionResult(
            packet=packet,
            is_flagged=is_flagged,
            similarity=similarity,
            is_warmup=is_warmup,
        )


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

class RealisticTrafficGenerator:
    """Generate realistic normal traffic for warmup."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _ephemeral(self) -> int:
        return self.rng.randint(49152, 65535)

    def generate_normal_tcp(self) -> dict:
        """
        Normal TCP traffic:
        - dst_port: mostly 443, some 80
        - flags: mostly ACK (0x10), some SYN+ACK (0x12), PSH+ACK (0x18)
        - src_port: ephemeral
        """
        dst_port = self.rng.choices([443, 80], weights=[0.7, 0.3])[0]
        flags = self.rng.choices(
            [0x10, 0x12, 0x18, 0x02],  # ACK, SYN+ACK, PSH+ACK, SYN
            weights=[0.5, 0.2, 0.25, 0.05]  # SYN is rare in established traffic
        )[0]

        return {
            "protocol": "TCP",
            "src_port": self._ephemeral(),
            "dst_port": dst_port,
            "flags": flags,
        }

    def generate_normal_udp(self) -> dict:
        """
        Normal UDP traffic (e.g., game service, DNS client):
        - dst_port: consistent service port
        - src_port: ephemeral (client)
        """
        dst_port = self.rng.choice([443, 8080, 27015, 7777])  # QUIC, game servers

        return {
            "protocol": "UDP",
            "src_port": self._ephemeral(),
            "dst_port": dst_port,
        }

    def generate_normal_icmp(self) -> dict:
        """Normal ICMP - ping (rare)."""
        return {
            "protocol": "ICMP",
            "icmp_type": self.rng.choice([8, 0]),  # echo request/reply
            "icmp_code": 0,
        }

    def generate_normal_stream(self, n: int) -> List[dict]:
        """
        Generate normal traffic mix.

        Heavy TCP (80%), some UDP (18%), rare ICMP (2%)
        """
        stream = []
        for _ in range(n):
            proto = self.rng.choices(
                ["TCP", "UDP", "ICMP"],
                weights=[0.80, 0.18, 0.02]
            )[0]

            if proto == "TCP":
                stream.append(self.generate_normal_tcp())
            elif proto == "UDP":
                stream.append(self.generate_normal_udp())
            else:
                stream.append(self.generate_normal_icmp())

        return stream


class DDoSAttackGenerator:
    """Generate DDoS attack traffic."""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _ephemeral(self) -> int:
        return self.rng.randint(49152, 65535)

    def syn_flood(self) -> dict:
        """
        SYN flood:
        - flags: SYN (0x02) - THE signal
        - dst_port: target service (80/443)
        - src_port: randomized (spoofed)
        """
        return {
            "protocol": "TCP",
            "src_port": self.rng.randint(1, 65535),  # Spoofed, random
            "dst_port": 443,
            "flags": 0x02,  # SYN only
        }

    def dns_reflection(self) -> dict:
        """
        DNS reflection:
        - src_port: 53 (FROM DNS server) - THE signal
        - dst_port: victim's ephemeral port
        """
        return {
            "protocol": "UDP",
            "src_port": 53,  # From DNS server!
            "dst_port": self._ephemeral(),  # Victim's port
        }

    def ntp_reflection(self) -> dict:
        """
        NTP amplification:
        - src_port: 123 (FROM NTP server)
        - dst_port: victim's port
        """
        return {
            "protocol": "UDP",
            "src_port": 123,  # From NTP server!
            "dst_port": self._ephemeral(),
        }

    def ssdp_reflection(self) -> dict:
        """
        SSDP amplification:
        - src_port: 1900 (FROM SSDP service)
        """
        return {
            "protocol": "UDP",
            "src_port": 1900,
            "dst_port": self._ephemeral(),
        }

    def icmp_flood(self) -> dict:
        """
        ICMP flood:
        - High volume of ICMP (normally rare)
        """
        return {
            "protocol": "ICMP",
            "icmp_type": 8,
            "icmp_code": 0,
        }


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_isolated_test(
    attack_name: str,
    attack_generator,
    normal_gen: RealisticTrafficGenerator,
    mode: str = "high_sim",
    warmup_count: int = 500,
    normal_count: int = 2000,
    attack_count: int = 500,
    threshold: float = 0.70,
):
    """
    Run isolated DDoS detection test.

    1. Warmup with normal traffic
    2. More normal traffic (post-warmup baseline)
    3. Attack burst
    4. Back to normal (recovery)
    """
    print(f"\n{'='*60}")
    print(f"Testing: {attack_name} (mode={mode})")
    print(f"{'='*60}")

    # Fresh detector
    detector = DDoSDetector(mode=mode, threshold=threshold, warmup=warmup_count)

    # Build stream
    stream = []

    # Phase 1: Warmup (normal)
    for p in normal_gen.generate_normal_stream(warmup_count):
        stream.append((p, "warmup"))

    # Phase 2: Post-warmup normal
    for p in normal_gen.generate_normal_stream(normal_count // 2):
        stream.append((p, "normal"))

    # Phase 3: Attack burst
    for _ in range(attack_count):
        stream.append((attack_generator(), "attack"))

    # Phase 4: Recovery (normal)
    for p in normal_gen.generate_normal_stream(normal_count // 2):
        stream.append((p, "normal"))

    # Process
    results = []
    for packet, label in stream:
        result = detector.process(packet)
        results.append((result, label))

    # Metrics (exclude warmup)
    post_warmup = [(r, l) for r, l in results if l != "warmup"]

    tp = sum(1 for r, l in post_warmup if l == "attack" and r.is_flagged)
    fp = sum(1 for r, l in post_warmup if l == "normal" and r.is_flagged)
    fn = sum(1 for r, l in post_warmup if l == "attack" and not r.is_flagged)
    tn = sum(1 for r, l in post_warmup if l == "normal" and not r.is_flagged)

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(0.001, precision + recall)

    # Similarity distributions
    normal_sims = [r.similarity for r, l in post_warmup if l == "normal"]
    attack_sims = [r.similarity for r, l in post_warmup if l == "attack"]

    print(f"  Stream: {warmup_count} warmup + {normal_count} normal + {attack_count} attack")
    print(f"  Mode: {mode}, Threshold: {threshold}")
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Precision: {precision:.1%}")
    print(f"  Recall:    {recall:.1%}")
    print(f"  F1 Score:  {f1:.3f}")
    print(f"  Normal sim: mean={np.mean(normal_sims):.3f}, max={max(normal_sims):.3f}")
    print(f"  Attack sim: mean={np.mean(attack_sims):.3f}, max={max(attack_sims):.3f}")

    return {"attack": attack_name, "f1": f1, "precision": precision, "recall": recall,
            "attack_sim": np.mean(attack_sims), "normal_sim": np.mean(normal_sims)}


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("Challenge 010-029: Realistic DDoS Detection")
    print("=" * 70)
    print("""
Key signals:

1. SYN flood: concentration of SYN flag (normal is mostly ACK)
2. UDP reflection: src_port FROM well-known ports (53, 123, 1900)
   (Normal UDP has ephemeral src_port)
3. ICMP flood: unusual ICMP volume (normally rare)

Each test is ISOLATED (fresh detector).
""")

    normal_gen = RealisticTrafficGenerator(seed=42)
    attack_gen = DDoSAttackGenerator(seed=42)

    results = []

    # Test each attack type with appropriate mode

    # SYN flood: high similarity (SYN flag seen in normal, just concentrated)
    results.append(run_isolated_test(
        "SYN Flood",
        attack_gen.syn_flood,
        normal_gen,
        mode="high_sim",
        threshold=0.65,
    ))

    # UDP reflections: LOW similarity (src_port from service ports is NOVEL)
    results.append(run_isolated_test(
        "DNS Reflection",
        attack_gen.dns_reflection,
        normal_gen,
        mode="low_sim",  # src_port=53 never seen in normal
        threshold=0.45,
    ))

    results.append(run_isolated_test(
        "NTP Amplification",
        attack_gen.ntp_reflection,
        normal_gen,
        mode="low_sim",
        threshold=0.45,
    ))

    results.append(run_isolated_test(
        "SSDP Amplification",
        attack_gen.ssdp_reflection,
        normal_gen,
        mode="low_sim",
        threshold=0.45,
    ))

    # ICMP flood: Since ICMP is rare (2%), the attack packets are somewhat novel
    # But they're also consistent (same type/code) so similarity increases during burst
    # Try high_sim with higher threshold
    results.append(run_isolated_test(
        "ICMP Flood",
        attack_gen.icmp_flood,
        normal_gen,
        mode="high_sim",
        threshold=0.70,  # Higher threshold for clearer separation
    ))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Realistic DDoS Detection")
    print("=" * 70)
    print(f"\n{'Attack':<20} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Attack Sim':>12} {'Normal Sim':>12}")
    print("-" * 72)
    for r in results:
        print(f"{r['attack']:<20} {r['f1']:>8.3f} {r['precision']:>10.1%} {r['recall']:>8.1%} {r['attack_sim']:>12.3f} {r['normal_sim']:>12.3f}")

    avg_f1 = np.mean([r['f1'] for r in results])
    print("-" * 72)
    print(f"{'Average':<20} {avg_f1:>8.3f}")

    return results


if __name__ == "__main__":
    main()
