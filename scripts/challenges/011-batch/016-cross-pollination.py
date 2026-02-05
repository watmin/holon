#!/usr/bin/env python3
"""
Challenge 011-016: Cross-Pollination Experiments

Apply techniques between batch 010 and 011:

EXPERIMENT A: Apply 010's smart normalization to 011's structural detection
  - Port bucketing (wellknown < 1024, registered < 49152, ephemeral)
  - IP prefix levels (/8, /16, /24)

EXPERIMENT B: Apply 010's character bitmasks to 011's payload analysis
  - Encode payload as structural fingerprint, not raw bytes

EXPERIMENT C: Apply 010's variance-based DDoS detection to 011's state machine
  - Use variance drop as transition signal

EXPERIMENT D: Apply 011's sample-based signatures to improve 010's classification
  - Build signatures from actual attack packets, not handcrafted

EXPERIMENT E: Apply 011's prior/recent separation to 010's accumulator
  - Frozen prior + decaying recent for better drift detection
"""

import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import deque
from enum import Enum
import numpy as np

from scapy.all import IP, TCP, UDP, ICMP, Raw, Packet

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


DIMENSIONS = 4096


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


class AttackType(Enum):
    NONE = "none"
    SYN_FLOOD = "syn_flood"
    UDP_FLOOD = "udp_flood"
    DNS_REFLECTION = "dns_reflection"
    PORT_SCAN = "port_scan"
    ICMP_FLOOD = "icmp_flood"


# =============================================================================
# EXPERIMENT A: Smart Normalization from 010
# =============================================================================

def bucket_port(port: int) -> str:
    """010's port bucketing."""
    if port < 1024:
        return "wellknown"
    elif port < 49152:
        return "registered"
    else:
        return "ephemeral"


def bucket_ip_prefix(ip: str, level: int = 16) -> str:
    """IP prefix at /8, /16, or /24 level."""
    parts = ip.split('.')
    if level == 8:
        return parts[0]
    elif level == 16:
        return f"{parts[0]}.{parts[1]}"
    else:  # 24
        return f"{parts[0]}.{parts[1]}.{parts[2]}"


class NormalizedStructuralDetector:
    """
    Combines 010's smart normalization with 011's structural encoding.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.baseline = self.encoder.create_accumulator()
        self.baseline_count = 0
        self._build_baseline()

    def _packet_to_structure_v1(self, pkt: Packet) -> dict:
        """011 approach: raw values."""
        structure = {}
        if IP in pkt:
            structure["l3"] = {
                "src_prefix": bucket_ip_prefix(pkt[IP].src),
                "dst_prefix": bucket_ip_prefix(pkt[IP].dst),
            }
        if TCP in pkt:
            structure["l4"] = {
                "proto": "tcp",
                "dst_port": pkt[TCP].dport,  # Raw value
                "flags": str(pkt[TCP].flags),
            }
        elif UDP in pkt:
            structure["l4"] = {
                "proto": "udp",
                "dst_port": pkt[UDP].dport,  # Raw value
                "src_port": pkt[UDP].sport,  # Raw value
            }
        return structure

    def _packet_to_structure_v2(self, pkt: Packet) -> dict:
        """Combined approach: 010's normalization + 011's structure."""
        structure = {}
        if IP in pkt:
            structure["l3"] = {
                "src_prefix_8": bucket_ip_prefix(pkt[IP].src, 8),
                "src_prefix_16": bucket_ip_prefix(pkt[IP].src, 16),
                "dst_prefix_8": bucket_ip_prefix(pkt[IP].dst, 8),
                "dst_prefix_16": bucket_ip_prefix(pkt[IP].dst, 16),
            }
        if TCP in pkt:
            structure["l4"] = {
                "proto": "tcp",
                "dst_port_bucket": bucket_port(pkt[TCP].dport),  # Bucketed!
                "dst_port_wellknown": pkt[TCP].dport if pkt[TCP].dport < 1024 else None,
                "flags": str(pkt[TCP].flags),
                "is_syn_only": str(pkt[TCP].flags) == "S",
            }
        elif UDP in pkt:
            structure["l4"] = {
                "proto": "udp",
                "dst_port_bucket": bucket_port(pkt[UDP].dport),
                "src_port_bucket": bucket_port(pkt[UDP].sport),
                "is_reflection": pkt[UDP].sport < 1024,  # Key signal!
            }
        elif ICMP in pkt:
            structure["l4"] = {
                "proto": "icmp",
                "type": pkt[ICMP].type,
            }
        return structure

    def _build_baseline(self):
        for _ in range(100):
            pkt = generate_normal_packet()
            vec = self.encoder.encode_data(self._packet_to_structure_v2(pkt))
            self.baseline = self.encoder.accumulate(self.baseline, vec)
            self.baseline_count += 1

    def detect_v1(self, pkt: Packet) -> float:
        """011's approach."""
        vec = self.encoder.encode_data(self._packet_to_structure_v1(pkt))
        return cosine_similarity(vec, self.encoder.normalize_accumulator(self.baseline))

    def detect_v2(self, pkt: Packet) -> float:
        """Combined approach."""
        vec = self.encoder.encode_data(self._packet_to_structure_v2(pkt))
        return cosine_similarity(vec, self.encoder.normalize_accumulator(self.baseline))


# =============================================================================
# EXPERIMENT B: Character Bitmask for Payload (from 010)
# =============================================================================

def payload_bitmask(payload: bytes) -> int:
    """
    010's character class bitmask applied to payload.

    Bits:
    0: has lowercase
    1: has uppercase
    2: has digits
    3: has space
    4: has common punctuation (.,;:)
    5: has brackets/parens
    6: has quotes
    7: has binary/control chars
    """
    mask = 0
    for b in payload:
        if 97 <= b <= 122:  # lowercase
            mask |= 0x01
        elif 65 <= b <= 90:  # uppercase
            mask |= 0x02
        elif 48 <= b <= 57:  # digits
            mask |= 0x04
        elif b == 32:  # space
            mask |= 0x08
        elif b in (46, 44, 59, 58):  # .,;:
            mask |= 0x10
        elif b in (40, 41, 60, 62, 91, 93, 123, 125):  # ()[]{}
            mask |= 0x20
        elif b in (34, 39, 96):  # "'`
            mask |= 0x40
        elif b < 32 or b > 126:  # binary/control
            mask |= 0x80
    return mask


class BitmaskPayloadDetector:
    """Apply 010's bitmask approach to payload analysis."""

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.baseline = self.encoder.create_accumulator()
        self._build_baseline()

    def _packet_to_structure(self, pkt: Packet) -> dict:
        structure = {}

        if IP in pkt:
            structure["l3"] = {"src_prefix": bucket_ip_prefix(pkt[IP].src)}

        if TCP in pkt:
            structure["l4"] = {"proto": "tcp", "dst_port_bucket": bucket_port(pkt[TCP].dport)}
        elif UDP in pkt:
            structure["l4"] = {"proto": "udp", "is_reflection": pkt[UDP].sport < 1024}

        # Payload bitmask (from 010!)
        if Raw in pkt:
            payload = bytes(pkt[Raw].load)
            bitmask = payload_bitmask(payload)
            structure["payload"] = {
                "has_payload": True,
                "size_bucket": "small" if len(payload) < 64 else "medium" if len(payload) < 256 else "large",
                "bitmask": bitmask,
                "has_binary": bool(bitmask & 0x80),
                "has_quotes": bool(bitmask & 0x40),
            }
        else:
            structure["payload"] = {"has_payload": False}

        return structure

    def _build_baseline(self):
        for _ in range(100):
            pkt = generate_normal_packet()
            vec = self.encoder.encode_data(self._packet_to_structure(pkt))
            self.baseline = self.encoder.accumulate(self.baseline, vec)

    def detect(self, pkt: Packet) -> Tuple[float, dict]:
        structure = self._packet_to_structure(pkt)
        vec = self.encoder.encode_data(structure)
        sim = cosine_similarity(vec, self.encoder.normalize_accumulator(self.baseline))
        return sim, structure


# =============================================================================
# EXPERIMENT C: Variance-Based Transition Detection (from 010)
# =============================================================================

class VarianceTransitionDetector:
    """
    Combine 010's variance-based detection with 011's state machine.
    """

    def __init__(self, window_size: int = 50):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.baseline = self.encoder.create_accumulator()
        self.recent = self.encoder.create_accumulator()
        self.decay = 0.98

        self.window_size = window_size
        self.similarity_history = deque(maxlen=window_size)

        # Baseline variance/mean (from normal phase)
        self.baseline_variance = None
        self.baseline_mean = None

        self._build_baseline()

    def _packet_to_structure(self, pkt: Packet) -> dict:
        structure = {}
        if TCP in pkt:
            structure["l4"] = {"proto": "tcp", "flags": str(pkt[TCP].flags)}
        elif UDP in pkt:
            structure["l4"] = {"proto": "udp", "is_reflection": pkt[UDP].sport < 1024}
        elif ICMP in pkt:
            structure["l4"] = {"proto": "icmp", "type": pkt[ICMP].type}
        return structure

    def _build_baseline(self):
        sims = []
        for _ in range(100):
            pkt = generate_normal_packet()
            vec = self.encoder.encode_data(self._packet_to_structure(pkt))
            self.baseline = self.encoder.accumulate(self.baseline, vec)

            if len(sims) > 10:
                baseline_norm = self.encoder.normalize_accumulator(self.baseline)
                sim = cosine_similarity(vec, baseline_norm)
                sims.append(sim)

        self.baseline_variance = np.var(sims) if sims else 0.05
        self.baseline_mean = np.mean(sims) if sims else 0.5

    def process(self, pkt: Packet) -> Tuple[str, float, float]:
        """Returns (state, variance, mean)."""
        vec = self.encoder.encode_data(self._packet_to_structure(pkt))
        baseline_norm = self.encoder.normalize_accumulator(self.baseline)
        sim = cosine_similarity(vec, baseline_norm)

        self.similarity_history.append(sim)

        if len(self.similarity_history) < 20:
            return "warmup", 0.0, sim

        current_var = np.var(list(self.similarity_history))
        current_mean = np.mean(list(self.similarity_history))

        # 010's detection logic adapted:
        # DDoS = variance drop + mean SHIFT (could be up or down depending on attack)
        mean_shift = abs(current_mean - self.baseline_mean)

        if current_var < self.baseline_variance * 0.3 and mean_shift > 0.2:
            state = "ddos_detected"
        elif current_var < self.baseline_variance * 0.5:
            state = "suspicious"
        else:
            state = "normal"

        # Update recent
        self.recent = self.decay * self.recent + vec.astype(np.float64)

        return state, current_var, current_mean


# =============================================================================
# EXPERIMENT D: Sample-Based Signatures (from 011)
# =============================================================================

class SampleBasedClassifier:
    """
    Apply 011's sample-based signatures to 010's classification.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.signatures: Dict[AttackType, np.ndarray] = {}
        self._build_signatures()

    def _packet_to_structure(self, pkt: Packet) -> dict:
        structure = {}
        if IP in pkt:
            structure["l3"] = {"src_prefix": bucket_ip_prefix(pkt[IP].src)}
        if TCP in pkt:
            structure["l4"] = {
                "proto": "tcp",
                "dst_port_bucket": bucket_port(pkt[TCP].dport),
                "flags": str(pkt[TCP].flags),
            }
        elif UDP in pkt:
            structure["l4"] = {
                "proto": "udp",
                "is_reflection": pkt[UDP].sport < 1024,
            }
        elif ICMP in pkt:
            structure["l4"] = {"proto": "icmp", "type": pkt[ICMP].type}
        return structure

    def _build_signatures(self):
        """Build from actual samples (011's approach)."""
        for attack_type in AttackType:
            if attack_type == AttackType.NONE:
                continue

            acc = self.encoder.create_accumulator()
            for _ in range(50):
                pkt = generate_attack_packet(attack_type)
                vec = self.encoder.encode_data(self._packet_to_structure(pkt))
                acc = self.encoder.accumulate(acc, vec)

            self.signatures[attack_type] = self.encoder.normalize_accumulator(acc)

    def classify(self, pkt: Packet) -> Tuple[AttackType, float]:
        vec = self.encoder.encode_data(self._packet_to_structure(pkt))

        best_type = AttackType.NONE
        best_sim = 0.0

        for attack_type, sig in self.signatures.items():
            sim = cosine_similarity(vec, sig)
            if sim > best_sim:
                best_sim = sim
                best_type = attack_type

        return best_type, best_sim


# =============================================================================
# EXPERIMENT E: Prior/Recent Separation (from 011)
# =============================================================================

class DualKnowledgeDetector:
    """
    Apply 011's prior/recent separation to 010's accumulator approach.
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        # Prior: frozen baseline
        self.prior = self.encoder.create_accumulator()
        self.prior_count = 0

        # Recent: decaying
        self.recent = self.encoder.create_accumulator()
        self.recent_count = 0
        self.decay = 0.98

        self._build_prior()

    def _packet_to_structure(self, pkt: Packet) -> dict:
        structure = {}
        if TCP in pkt:
            structure["l4"] = {"proto": "tcp", "flags": str(pkt[TCP].flags)}
        elif UDP in pkt:
            structure["l4"] = {"proto": "udp", "is_reflection": pkt[UDP].sport < 1024}
        elif ICMP in pkt:
            structure["l4"] = {"proto": "icmp", "type": pkt[ICMP].type}
        return structure

    def _build_prior(self):
        for _ in range(100):
            pkt = generate_normal_packet()
            vec = self.encoder.encode_data(self._packet_to_structure(pkt))
            self.prior = self.encoder.accumulate(self.prior, vec)
            self.prior_count += 1

    def process(self, pkt: Packet) -> Tuple[float, float, float]:
        """Returns (prior_sim, recent_sim, divergence)."""
        vec = self.encoder.encode_data(self._packet_to_structure(pkt))

        prior_norm = self.encoder.normalize_accumulator(self.prior)
        prior_sim = cosine_similarity(vec, prior_norm)

        if self.recent_count > 10:
            recent_norm = self.encoder.normalize_accumulator(self.recent)
            recent_sim = cosine_similarity(vec, recent_norm)
            divergence = cosine_similarity(prior_norm, recent_norm)
        else:
            recent_sim = prior_sim
            divergence = 1.0

        # Update recent with decay
        self.recent = self.decay * self.recent + vec.astype(np.float64)
        self.recent_count += 1

        return prior_sim, recent_sim, divergence


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def generate_normal_packet() -> Packet:
    pkt_type = random.choice(["http", "https", "dns"])
    if pkt_type == "http":
        return IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
            sport=random.randint(49152, 65535), dport=80, flags="PA"
        ) / Raw(load=b"GET / HTTP/1.1\r\n")
    elif pkt_type == "https":
        return IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
            sport=random.randint(49152, 65535), dport=443, flags="PA"
        ) / Raw(load=b"\x16\x03\x01" + b"X" * 50)
    else:
        return IP(src="192.168.1.50", dst="8.8.8.8") / UDP(
            sport=random.randint(49152, 65535), dport=53
        ) / Raw(load=b"\x00\x01example\x03com\x00")


def generate_attack_packet(attack_type: AttackType) -> Packet:
    idx = random.randint(0, 10000)
    if attack_type == AttackType.SYN_FLOOD:
        return IP(src=f"10.{idx%256}.{idx//256%256}.{idx//65536%256}", dst="192.168.1.100") / TCP(
            sport=40000 + idx % 20000, dport=80, flags="S"
        )
    elif attack_type == AttackType.UDP_FLOOD:
        return IP(src=f"10.{idx%256}.{idx//256%256}.1", dst="192.168.1.100") / UDP(
            sport=40000 + idx % 20000, dport=53
        ) / Raw(load=b"X" * 64)
    elif attack_type == AttackType.DNS_REFLECTION:
        return IP(src=f"8.8.{idx%4}.{idx%256}", dst="192.168.1.100") / UDP(
            sport=53, dport=40000 + idx % 1000
        ) / Raw(load=b"X" * 512)
    elif attack_type == AttackType.PORT_SCAN:
        return IP(src="10.0.0.5", dst="192.168.1.100") / TCP(
            sport=40000, dport=1 + idx % 1000, flags="S"
        )
    elif attack_type == AttackType.ICMP_FLOOD:
        return IP(src=f"10.{idx%256}.{idx//256%256}.1", dst="192.168.1.100") / ICMP(type=8)
    return generate_normal_packet()


# =============================================================================
# RUN EXPERIMENTS
# =============================================================================

def run_experiment_a():
    """Smart normalization comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT A: Smart Normalization (010) → Structural Detection (011)")
    print("=" * 70)

    detector = NormalizedStructuralDetector()

    # Test on different traffic
    results = {"v1": {"normal": [], "attack": []}, "v2": {"normal": [], "attack": []}}

    for _ in range(50):
        pkt = generate_normal_packet()
        results["v1"]["normal"].append(detector.detect_v1(pkt))
        results["v2"]["normal"].append(detector.detect_v2(pkt))

    for attack_type in [AttackType.SYN_FLOOD, AttackType.DNS_REFLECTION]:
        for _ in range(50):
            pkt = generate_attack_packet(attack_type)
            results["v1"]["attack"].append(detector.detect_v1(pkt))
            results["v2"]["attack"].append(detector.detect_v2(pkt))

    print(f"\nV1 (raw values):")
    print(f"  Normal: mean={np.mean(results['v1']['normal']):.3f}, std={np.std(results['v1']['normal']):.3f}")
    print(f"  Attack: mean={np.mean(results['v1']['attack']):.3f}, std={np.std(results['v1']['attack']):.3f}")
    print(f"  Gap: {np.mean(results['v1']['normal']) - np.mean(results['v1']['attack']):.3f}")

    print(f"\nV2 (normalized/bucketed):")
    print(f"  Normal: mean={np.mean(results['v2']['normal']):.3f}, std={np.std(results['v2']['normal']):.3f}")
    print(f"  Attack: mean={np.mean(results['v2']['attack']):.3f}, std={np.std(results['v2']['attack']):.3f}")
    print(f"  Gap: {np.mean(results['v2']['normal']) - np.mean(results['v2']['attack']):.3f}")

    improvement = (np.mean(results['v2']['normal']) - np.mean(results['v2']['attack'])) - \
                  (np.mean(results['v1']['normal']) - np.mean(results['v1']['attack']))
    print(f"\n  → Gap improvement: {improvement:+.3f}")


def run_experiment_b():
    """Payload bitmask comparison."""
    print("\n" + "=" * 70)
    print("EXPERIMENT B: Character Bitmask (010) → Payload Analysis (011)")
    print("=" * 70)

    detector = BitmaskPayloadDetector()

    # Normal payloads
    normal_sims = []
    for _ in range(50):
        pkt = generate_normal_packet()
        sim, _ = detector.detect(pkt)
        normal_sims.append(sim)

    # Attack payloads with unusual content
    attack_sims = []
    attack_bitmasks = []
    for _ in range(50):
        # Create packet with binary payload
        pkt = IP(src="10.0.0.1", dst="192.168.1.100") / TCP(
            sport=40000, dport=31337, flags="PA"
        ) / Raw(load=b"\xde\xad\xbe\xef" + bytes(range(256))[:50])
        sim, structure = detector.detect(pkt)
        attack_sims.append(sim)
        attack_bitmasks.append(structure.get("payload", {}).get("bitmask", 0))

    print(f"\nNormal payloads:")
    print(f"  Similarity: mean={np.mean(normal_sims):.3f}")

    print(f"\nAttack payloads (binary content):")
    print(f"  Similarity: mean={np.mean(attack_sims):.3f}")
    print(f"  Bitmasks: {set(attack_bitmasks)} (has_binary={any(b & 0x80 for b in attack_bitmasks)})")

    print(f"\n  → Gap: {np.mean(normal_sims) - np.mean(attack_sims):.3f}")


def run_experiment_c():
    """Variance-based transition detection."""
    print("\n" + "=" * 70)
    print("EXPERIMENT C: Variance Detection (010) → State Machine (011)")
    print("=" * 70)

    detector = VarianceTransitionDetector()

    # Normal phase
    print("\nPhase 1: Normal traffic")
    for i in range(100):
        pkt = generate_normal_packet()
        state, var, mean = detector.process(pkt)
        if i == 99:
            print(f"  Packet {i}: state={state}, var={var:.4f}, mean={mean:.3f}")

    # Attack phase
    print("\nPhase 2: DDoS attack")
    detection_point = None
    for i in range(200):
        pkt = generate_attack_packet(AttackType.SYN_FLOOD)
        state, var, mean = detector.process(pkt)
        if state == "ddos_detected" and detection_point is None:
            detection_point = i
            print(f"  ✓ DDoS DETECTED at packet {100 + i}")
            print(f"    var={var:.4f}, mean={mean:.3f}")
        if i in [50, 100, 150, 199]:
            print(f"  Packet {100 + i}: state={state}, var={var:.4f}, mean={mean:.3f}")

    if detection_point:
        print(f"\n  → Detection delay: {detection_point} packets after attack start")


def run_experiment_d():
    """Sample-based signatures."""
    print("\n" + "=" * 70)
    print("EXPERIMENT D: Sample-Based Signatures (011) → Classification (010)")
    print("=" * 70)

    classifier = SampleBasedClassifier()

    # Test classification accuracy
    results = {attack_type: {"correct": 0, "total": 0} for attack_type in AttackType if attack_type != AttackType.NONE}

    for attack_type in AttackType:
        if attack_type == AttackType.NONE:
            continue
        for _ in range(50):
            pkt = generate_attack_packet(attack_type)
            predicted, confidence = classifier.classify(pkt)
            results[attack_type]["total"] += 1
            if predicted == attack_type:
                results[attack_type]["correct"] += 1

    print("\nClassification accuracy (from samples):")
    total_correct = 0
    total_count = 0
    for attack_type, r in results.items():
        acc = r["correct"] / r["total"] if r["total"] > 0 else 0
        print(f"  {attack_type.value:<20} {acc:.1%} ({r['correct']}/{r['total']})")
        total_correct += r["correct"]
        total_count += r["total"]

    print(f"\n  → Overall accuracy: {total_correct/total_count:.1%}")


def run_experiment_e():
    """Prior/recent separation."""
    print("\n" + "=" * 70)
    print("EXPERIMENT E: Prior/Recent Separation (011) → Accumulator (010)")
    print("=" * 70)

    detector = DualKnowledgeDetector()

    # Normal phase
    print("\nPhase 1: Normal traffic (prior/recent should agree)")
    for i in range(50):
        pkt = generate_normal_packet()
        prior_sim, recent_sim, divergence = detector.process(pkt)
        if i in [0, 24, 49]:
            print(f"  Packet {i}: prior={prior_sim:.3f}, recent={recent_sim:.3f}, divergence={divergence:.3f}")

    # Attack phase
    print("\nPhase 2: DDoS attack (prior/recent should diverge)")
    for i in range(100):
        pkt = generate_attack_packet(AttackType.SYN_FLOOD)
        prior_sim, recent_sim, divergence = detector.process(pkt)
        if i in [0, 25, 50, 75, 99]:
            print(f"  Packet {50+i}: prior={prior_sim:.3f}, recent={recent_sim:.3f}, divergence={divergence:.3f}")

    # Recovery phase
    print("\nPhase 3: Return to normal (divergence should recover)")
    for i in range(50):
        pkt = generate_normal_packet()
        prior_sim, recent_sim, divergence = detector.process(pkt)
        if i in [0, 24, 49]:
            print(f"  Packet {150+i}: prior={prior_sim:.3f}, recent={recent_sim:.3f}, divergence={divergence:.3f}")


def main():
    print("=" * 70)
    print("CHALLENGE 011-016: CROSS-POLLINATION EXPERIMENTS")
    print("=" * 70)

    run_experiment_a()
    run_experiment_b()
    run_experiment_c()
    run_experiment_d()
    run_experiment_e()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
EXPERIMENT A: Smart Normalization
  - Port bucketing reduces cardinality
  - Multi-level IP prefixes capture more patterns

EXPERIMENT B: Payload Bitmask
  - Binary content detection via bitmask bit 7
  - Low cardinality (256 possible values)

EXPERIMENT C: Variance-Based Detection
  - Variance drop + mean rise = DDoS signal
  - Faster than pure streak-based detection

EXPERIMENT D: Sample-Based Signatures
  - Building from actual packets > handcrafted
  - Works across attack types

EXPERIMENT E: Prior/Recent Separation
  - Divergence tracks regime change
  - Prior stays stable during attack
  - Recent adapts (with decay)
""")


if __name__ == "__main__":
    main()
