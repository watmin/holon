#!/usr/bin/env python3
"""
Challenge 010-030: Two-Phase DDoS Detection

Realistic DDoS simulation:
1. Normal traffic (learning phase)
2. DDoS arrives - bad traffic DOMINATES (95%+), drowns out good traffic

Two-phase detection:
1. DETECT: "Something is wrong" - traffic pattern changed
2. CLASSIFY: What kind of DDoS is it?

Detection signal: The VARIANCE of similarity drops dramatically during DDoS
- Normal: diverse traffic → similarity varies
- DDoS: homogeneous traffic → similarity becomes stable/high
"""

import sys
import time
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import deque

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import DeterministicVectorManager
from holon.encoder import Encoder


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096
DECAY_FACTOR = 0.995  # Faster decay to adapt to changes


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a.astype(np.float64), b.astype(np.float64)) / (norm_a * norm_b))


# =============================================================================
# ACCUMULATOR
# =============================================================================

class DecayingAccumulator:
    def __init__(self, dimensions: int, decay: float):
        self.dimensions = dimensions
        self.decay = decay
        self.accumulator = np.zeros(dimensions, dtype=np.float64)

    def update(self, vector: np.ndarray):
        self.accumulator = self.decay * self.accumulator + vector.astype(np.float64)

    def get_normalized(self) -> np.ndarray:
        norm = np.linalg.norm(self.accumulator)
        if norm < 1e-10:
            return np.zeros(self.dimensions, dtype=np.float32)
        return (self.accumulator / norm).astype(np.float32)


# =============================================================================
# PACKET ENCODING
# =============================================================================

def encode_packet(packet: dict) -> dict:
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
            "src_port": packet.get("src_port", 0),
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
# TWO-PHASE DETECTOR
# =============================================================================

@dataclass
class DetectionState:
    packet_num: int
    similarity: float
    sim_variance: float
    sim_mean: float
    ddos_detected: bool
    attack_type: Optional[str]
    phase: str  # "learning", "normal", "ddos_detected"


class TwoPhaseDetector:
    """
    Two-phase DDoS detection:

    Phase 1 - DETECT:
    Monitor similarity variance. Normal traffic has high variance (diverse).
    DDoS traffic has low variance (homogeneous).
    When variance drops significantly → DDoS detected.

    Phase 2 - CLASSIFY:
    Once DDoS detected, analyze the dominant pattern to classify attack type.
    """

    def __init__(
        self,
        learning_period: int = 500,
        window_size: int = 50,
        variance_threshold: float = 0.01,  # Low variance = DDoS
    ):
        self.vm = DeterministicVectorManager(dimensions=DIMENSIONS, global_seed=GLOBAL_SEED)
        self.encoder = Encoder(vector_manager=self.vm)
        self.accumulator = DecayingAccumulator(DIMENSIONS, DECAY_FACTOR)

        self.learning_period = learning_period
        self.window_size = window_size
        self.variance_threshold = variance_threshold

        self.packet_count = 0
        self.recent_sims = deque(maxlen=window_size)
        self.baseline_variance = None
        self.ddos_detected = False

        # For classification
        self.recent_packets = deque(maxlen=100)

    def process(self, packet: dict) -> DetectionState:
        self.packet_count += 1

        # Encode and compute similarity
        encoded = encode_packet(packet)
        vec = self.encoder.encode_data(encoded)
        model = self.accumulator.get_normalized()

        sim = cosine_similarity(vec, model) if self.packet_count > 1 else 0.5
        self.recent_sims.append(sim)
        self.recent_packets.append(packet)

        # Compute current stats
        if len(self.recent_sims) >= 10:
            current_var = np.var(list(self.recent_sims))
            current_mean = np.mean(list(self.recent_sims))
        else:
            current_var = 1.0
            current_mean = 0.5

        # Determine phase
        if self.packet_count <= self.learning_period:
            phase = "learning"
            self.ddos_detected = False
            attack_type = None
            # Establish baseline at end of learning
            if self.packet_count == self.learning_period:
                self.baseline_variance = current_var
        else:
            # Post-learning: check for DDoS
            if not self.ddos_detected:
                # DDoS detection: variance drops to fraction of baseline
                # AND similarity mean is high (homogeneous traffic)
                if (self.baseline_variance and
                    current_var < self.baseline_variance * 0.3 and
                    current_mean > 0.7):
                    self.ddos_detected = True
                    phase = "ddos_detected"
                    attack_type = self._classify_attack()
                else:
                    phase = "normal"
                    attack_type = None
            else:
                phase = "ddos_detected"
                attack_type = self._classify_attack()

        # Update accumulator
        self.accumulator.update(vec)

        return DetectionState(
            packet_num=self.packet_count,
            similarity=sim,
            sim_variance=current_var,
            sim_mean=current_mean,
            ddos_detected=self.ddos_detected,
            attack_type=attack_type,
            phase=phase,
        )

    def _classify_attack(self) -> str:
        """
        Classify attack type based on dominant packet pattern.
        """
        if len(self.recent_packets) < 10:
            return "unknown"

        packets = list(self.recent_packets)

        # Count protocol distribution
        tcp_count = sum(1 for p in packets if p.get("protocol") == "TCP")
        udp_count = sum(1 for p in packets if p.get("protocol") == "UDP")
        icmp_count = sum(1 for p in packets if p.get("protocol") == "ICMP")

        total = len(packets)

        # Check for ICMP flood
        if icmp_count / total > 0.8:
            return "icmp_flood"

        # Check for TCP attacks
        if tcp_count / total > 0.8:
            # Check flags
            syn_count = sum(1 for p in packets
                          if p.get("protocol") == "TCP" and p.get("flags") == 0x02)
            if syn_count / tcp_count > 0.8:
                return "syn_flood"
            return "tcp_flood"

        # Check for UDP attacks
        if udp_count / total > 0.8:
            # Check for reflection (src_port is well-known service)
            well_known_src = sum(1 for p in packets
                                if p.get("protocol") == "UDP"
                                and p.get("src_port", 0) in {53, 123, 1900, 11211})
            if well_known_src / udp_count > 0.5:
                # Identify specific reflection type
                src_ports = [p.get("src_port") for p in packets
                            if p.get("protocol") == "UDP"]
                from collections import Counter
                most_common = Counter(src_ports).most_common(1)[0][0]
                if most_common == 53:
                    return "dns_reflection"
                elif most_common == 123:
                    return "ntp_amplification"
                elif most_common == 1900:
                    return "ssdp_amplification"
                return "udp_reflection"
            return "udp_flood"

        return "mixed_attack"


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

class TrafficGenerator:
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    def _ephemeral(self) -> int:
        return self.rng.randint(49152, 65535)

    def normal_packet(self) -> dict:
        """Diverse normal traffic."""
        proto = self.rng.choices(["TCP", "UDP", "ICMP"], weights=[0.80, 0.18, 0.02])[0]

        if proto == "TCP":
            return {
                "protocol": "TCP",
                "src_port": self._ephemeral(),
                "dst_port": self.rng.choice([443, 80, 8080]),
                "flags": self.rng.choices([0x10, 0x12, 0x18, 0x02], weights=[0.5, 0.2, 0.25, 0.05])[0],
            }
        elif proto == "UDP":
            return {
                "protocol": "UDP",
                "src_port": self._ephemeral(),
                "dst_port": self.rng.choice([443, 53, 123]),
            }
        else:
            return {
                "protocol": "ICMP",
                "icmp_type": self.rng.choice([8, 0]),
                "icmp_code": 0,
            }

    def syn_flood(self) -> dict:
        return {
            "protocol": "TCP",
            "src_port": self.rng.randint(1, 65535),
            "dst_port": 443,
            "flags": 0x02,
        }

    def dns_reflection(self) -> dict:
        return {
            "protocol": "UDP",
            "src_port": 53,
            "dst_port": self._ephemeral(),
        }

    def ntp_amplification(self) -> dict:
        return {
            "protocol": "UDP",
            "src_port": 123,
            "dst_port": self._ephemeral(),
        }

    def icmp_flood(self) -> dict:
        return {
            "protocol": "ICMP",
            "icmp_type": 8,
            "icmp_code": 0,
        }

    def generate_realistic_stream(
        self,
        normal_count: int = 1000,
        ddos_count: int = 2000,
        ddos_ratio: float = 0.95,  # 95% bad during DDoS
        attack_type: str = "syn_flood",
    ) -> List[Tuple[dict, str]]:
        """
        Generate realistic DDoS stream:
        1. Normal phase
        2. DDoS phase (bad traffic dominates, some good still comes through)
        """
        stream = []

        # Attack generator
        attack_gen = {
            "syn_flood": self.syn_flood,
            "dns_reflection": self.dns_reflection,
            "ntp_amplification": self.ntp_amplification,
            "icmp_flood": self.icmp_flood,
        }.get(attack_type, self.syn_flood)

        # Phase 1: Normal traffic
        for _ in range(normal_count):
            stream.append((self.normal_packet(), "normal"))

        # Phase 2: DDoS dominates
        for _ in range(ddos_count):
            if self.rng.random() < ddos_ratio:
                stream.append((attack_gen(), attack_type))
            else:
                stream.append((self.normal_packet(), "normal_during_ddos"))

        return stream


# =============================================================================
# MAIN
# =============================================================================

def run_test(attack_type: str, gen: TrafficGenerator):
    print(f"\n{'='*70}")
    print(f"Testing: {attack_type}")
    print(f"{'='*70}")

    detector = TwoPhaseDetector(
        learning_period=500,
        window_size=50,
        variance_threshold=0.01,
    )

    stream = gen.generate_realistic_stream(
        normal_count=1000,
        ddos_count=3000,
        ddos_ratio=0.95,
        attack_type=attack_type,
    )

    results = []
    detection_packet = None
    classified_as = None

    start_time = time.time()
    for packet, label in stream:
        state = detector.process(packet)
        results.append((state, label))

        if state.ddos_detected and detection_packet is None:
            detection_packet = state.packet_num
            classified_as = state.attack_type

    elapsed = time.time() - start_time
    throughput = len(stream) / elapsed

    # Metrics
    print(f"\nStream: {len(stream)} packets")
    print(f"  Normal phase: 1000 packets")
    print(f"  DDoS phase: 3000 packets (95% attack, 5% normal)")
    print(f"  Throughput: {throughput:,.0f} packets/sec")

    if detection_packet:
        print(f"\n✓ DDoS DETECTED at packet #{detection_packet}")
        print(f"  Detection delay: {detection_packet - 1000} packets after DDoS started")
        print(f"  Classified as: {classified_as}")
        print(f"  Correct classification: {classified_as == attack_type}")
    else:
        print(f"\n✗ DDoS NOT DETECTED")

    # Show variance trajectory
    print(f"\nVariance trajectory (sampled):")
    for i, (state, label) in enumerate(results):
        if i % 500 == 0 or (state.ddos_detected and i == detection_packet - 1):
            marker = "<<<" if i == detection_packet - 1 else ""
            print(f"  Packet {i}: var={state.sim_variance:.4f}, mean={state.sim_mean:.3f}, phase={state.phase} {marker}")

    return {
        "attack": attack_type,
        "detected": detection_packet is not None,
        "detection_delay": detection_packet - 1000 if detection_packet else None,
        "classified_as": classified_as,
        "correct": classified_as == attack_type if classified_as else False,
        "throughput": throughput,
    }


def main():
    print("=" * 70)
    print("Challenge 010-030: Two-Phase DDoS Detection")
    print("=" * 70)
    print("""
Realistic DDoS simulation:
- Phase 1: Normal traffic (learning)
- Phase 2: DDoS dominates (95% attack traffic drowns good traffic)

Two-phase detection:
1. DETECT: Variance of similarity drops (homogeneous attack traffic)
2. CLASSIFY: Analyze dominant pattern to identify attack type
""")

    gen = TrafficGenerator(seed=42)

    attack_types = ["syn_flood", "dns_reflection", "ntp_amplification", "icmp_flood"]

    results = []
    for attack in attack_types:
        results.append(run_test(attack, gen))

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Attack':<20} {'Detected':<10} {'Delay':<10} {'Classified As':<20} {'Throughput':<12}")
    print("-" * 80)
    for r in results:
        delay = f"{r['detection_delay']}" if r['detection_delay'] else "N/A"
        classified = r['classified_as'] or "N/A"
        detected = "✓" if r['detected'] else "✗"
        throughput = f"{r['throughput']:,.0f}/s"
        print(f"{r['attack']:<20} {detected:<10} {delay:<10} {classified:<20} {throughput:<12}")

    detection_rate = sum(1 for r in results if r['detected']) / len(results)
    classification_rate = sum(1 for r in results if r['correct']) / len(results)
    avg_throughput = np.mean([r['throughput'] for r in results])
    print("-" * 80)
    print(f"Detection rate: {detection_rate:.0%}")
    print(f"Classification accuracy: {classification_rate:.0%}")
    print(f"Average throughput: {avg_throughput:,.0f} packets/sec")


if __name__ == "__main__":
    main()
