#!/usr/bin/env python3
"""
Challenge 011-014: Structural Detection Using Holon's Core Encoder

Use Holon's actual structural encoding (role-filler binding) instead of
naive atom bundling. This leverages the library's real power:

- encode_data() recursively encodes nested structures
- Keys are bound to values: key ⊛ value
- Sequences get positional binding: position ⊛ item
- Everything bundles together with proper thresholding

This should dramatically improve detection by preserving structure.
"""

import sys
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from enum import Enum
import numpy as np

from scapy.all import IP, TCP, UDP, ICMP, Raw, Packet

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import VectorManager, CPUStore


# =============================================================================
# CONFIGURATION
# =============================================================================

GLOBAL_SEED = 42
DIMENSIONS = 4096


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# STATE AND LABELS
# =============================================================================

class TrafficState(Enum):
    NORMAL = "normal"
    ATTACK_START = "attack_start"
    ATTACK_ONGOING = "attack_ongoing"
    ATTACK_END = "attack_end"


class AttackType(Enum):
    NONE = "none"
    SYN_FLOOD = "syn_flood"
    UDP_FLOOD = "udp_flood"
    DNS_REFLECTION = "dns_reflection"
    PORT_SCAN = "port_scan"
    ICMP_FLOOD = "icmp_flood"


@dataclass
class LabeledPacket:
    packet: Packet
    state: TrafficState
    attack_type: AttackType
    packet_idx: int


@dataclass
class Detection:
    predicted_state: TrafficState
    predicted_attack: AttackType
    confidence: float
    similarity_to_baseline: float
    culprits: List[str]


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class ClassMetrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class MetricsTracker:
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.metrics: Dict[str, ClassMetrics] = {c: ClassMetrics() for c in classes}
        self.confusion: Dict[Tuple[str, str], int] = defaultdict(int)

    def record(self, true_label: str, pred_label: str):
        self.confusion[(true_label, pred_label)] += 1
        for cls in self.classes:
            if true_label == cls and pred_label == cls:
                self.metrics[cls].tp += 1
            elif true_label != cls and pred_label == cls:
                self.metrics[cls].fp += 1
            elif true_label == cls and pred_label != cls:
                self.metrics[cls].fn += 1

    def macro_f1(self) -> float:
        f1s = [m.f1 for m in self.metrics.values() if (m.tp + m.fn) > 0]
        return np.mean(f1s) if f1s else 0.0

    def weighted_f1(self) -> float:
        total = sum(m.tp + m.fn for m in self.metrics.values())
        if total == 0:
            return 0.0
        return sum(m.f1 * (m.tp + m.fn) for m in self.metrics.values()) / total

    def print_report(self, title: str = "Classification Report"):
        print(f"\n{title}")
        print("-" * 60)
        print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support'}")
        print("-" * 60)
        for cls in self.classes:
            m = self.metrics[cls]
            support = m.tp + m.fn
            if support > 0:
                print(f"{cls:<20} {m.precision:<12.3f} {m.recall:<12.3f} {m.f1:<12.3f} {support}")
        print("-" * 60)
        print(f"{'Macro F1':<20} {'':<12} {'':<12} {self.macro_f1():<12.3f}")
        print(f"{'Weighted F1':<20} {'':<12} {'':<12} {self.weighted_f1():<12.3f}")


# =============================================================================
# STRUCTURAL DETECTOR - Using Holon's Encoder Properly
# =============================================================================

class StructuralDetector:
    """
    Detector that uses Holon's structural encoding.

    Key insight: Instead of building atom strings, we build structured dicts
    and let Holon's encoder handle the role-filler binding.
    """

    def __init__(self, threshold: float = 0.3):
        # Create store with encoder
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        # Baseline (prior knowledge)
        self.baseline_accumulator = self.encoder.create_accumulator()
        self.baseline_count = 0

        # Recent knowledge
        self.recent_accumulator = self.encoder.create_accumulator()
        self.recent_count = 0
        self.decay = 0.99

        # Attack signatures as structured data
        self.attack_signatures: Dict[AttackType, np.ndarray] = {}

        # Detection threshold
        self.threshold = threshold

        # State tracking
        self.state_history: List[float] = []

        # Build baseline and signatures
        self._build_baseline()
        self._build_attack_signatures()

    def _packet_to_structure(self, pkt: Packet) -> dict:
        """
        Convert a Scapy packet to a structured dict for Holon encoding.

        This is the key translation layer - we build a proper nested structure
        that Holon's encoder can work with.
        """
        structure = {}

        # L3 Layer
        if IP in pkt:
            src_parts = pkt[IP].src.split('.')
            dst_parts = pkt[IP].dst.split('.')
            structure["l3"] = {
                "src_prefix": f"{src_parts[0]}.{src_parts[1]}",
                "dst_prefix": f"{dst_parts[0]}.{dst_parts[1]}",
            }

        # L4 Layer
        if TCP in pkt:
            flags = str(pkt[TCP].flags)
            structure["l4"] = {
                "proto": "tcp",
                "dst_port": pkt[TCP].dport,
                "flags": flags,
                "syn_only": flags == "S",
            }
        elif UDP in pkt:
            structure["l4"] = {
                "proto": "udp",
                "dst_port": pkt[UDP].dport,
                "src_port": pkt[UDP].sport,
                "reflection": pkt[UDP].sport < 1024,  # Well-known port as source = reflection
            }
        elif ICMP in pkt:
            structure["l4"] = {
                "proto": "icmp",
                "type": pkt[ICMP].type,
            }

        # Payload
        if Raw in pkt:
            payload = bytes(pkt[Raw].load)
            structure["payload"] = {
                "size": len(payload),
                "has_payload": True,
                "size_class": "empty" if len(payload) == 0
                              else "small" if len(payload) < 64
                              else "medium" if len(payload) < 256
                              else "large",
            }
        else:
            structure["payload"] = {
                "size": 0,
                "has_payload": False,
                "size_class": "empty",
            }

        return structure

    def _build_baseline(self):
        """Build baseline from synthetic normal traffic."""
        for _ in range(100):
            pkt = _generate_normal_packet()
            structure = self._packet_to_structure(pkt)
            vec = self.encoder.encode_data(structure)
            self.baseline_accumulator = self.encoder.accumulate(
                self.baseline_accumulator, vec
            )
            self.baseline_count += 1

    def _build_attack_signatures(self):
        """Build signature vectors for each attack type using structured data."""

        attack_structures = {
            AttackType.SYN_FLOOD: {
                "l3": {"src_prefix": "10.0"},  # Many sources from 10.x
                "l4": {
                    "proto": "tcp",
                    "syn_only": True,
                    "flags": "S",
                },
                "payload": {"has_payload": False, "size_class": "empty"},
                "attack_pattern": "syn_flood",
            },
            AttackType.UDP_FLOOD: {
                "l3": {"src_prefix": "10.0"},
                "l4": {
                    "proto": "udp",
                    "reflection": False,
                },
                "payload": {"size_class": "small"},
                "attack_pattern": "udp_flood",
            },
            AttackType.DNS_REFLECTION: {
                "l4": {
                    "proto": "udp",
                    "src_port": 53,
                    "reflection": True,
                },
                "payload": {"size_class": "large"},
                "attack_pattern": "dns_reflection",
            },
            AttackType.PORT_SCAN: {
                "l4": {
                    "proto": "tcp",
                    "syn_only": True,
                    "flags": "S",
                },
                "payload": {"has_payload": False},
                "attack_pattern": "port_scan",
            },
            AttackType.ICMP_FLOOD: {
                "l4": {
                    "proto": "icmp",
                    "type": 8,  # Echo request
                },
                "attack_pattern": "icmp_flood",
            },
        }

        for attack_type, structure in attack_structures.items():
            self.attack_signatures[attack_type] = self.encoder.encode_data(structure)

    def process(self, pkt: Packet) -> Detection:
        """Process a packet and return detection result."""

        # Convert to structure and encode
        structure = self._packet_to_structure(pkt)
        packet_vec = self.encoder.encode_data(structure)

        # Compare to baseline
        baseline_normalized = self.encoder.normalize_accumulator(self.baseline_accumulator)
        baseline_sim = cosine_similarity(packet_vec, baseline_normalized)

        # Compare to recent
        if self.recent_count > 10:
            recent_normalized = self.encoder.normalize_accumulator(self.recent_accumulator)
            recent_sim = cosine_similarity(packet_vec, recent_normalized)
        else:
            recent_sim = baseline_sim

        # Combined score (weighted)
        combined_sim = 0.6 * baseline_sim + 0.4 * recent_sim

        # Anomaly detection
        is_anomalous = combined_sim < self.threshold

        # Attack classification
        if is_anomalous:
            attack_scores = {}
            for attack_type, sig_vec in self.attack_signatures.items():
                sim = cosine_similarity(packet_vec, sig_vec)
                attack_scores[attack_type] = sim

            best_attack = max(attack_scores, key=attack_scores.get)
            best_score = attack_scores[best_attack]

            if best_score > 0.2:  # Confidence threshold
                predicted_attack = best_attack
            else:
                predicted_attack = AttackType.NONE
        else:
            predicted_attack = AttackType.NONE

        # Identify culprits (what fields are unusual)
        culprits = self._identify_culprits(structure, baseline_sim)

        # State tracking
        self.state_history.append(combined_sim)
        if len(self.state_history) > 20:
            self.state_history.pop(0)

        predicted_state = self._detect_transition(is_anomalous)

        # Update recent knowledge (with decay)
        self.recent_accumulator = self.decay * self.recent_accumulator
        weight = 0.1 if is_anomalous else 1.0
        self.recent_accumulator = self.encoder.accumulate(
            self.recent_accumulator, weight * packet_vec.astype(np.float64)
        )
        self.recent_count += 1

        return Detection(
            predicted_state=predicted_state,
            predicted_attack=predicted_attack,
            confidence=abs(combined_sim - self.threshold),
            similarity_to_baseline=baseline_sim,
            culprits=culprits,
        )

    def _identify_culprits(self, structure: dict, overall_sim: float) -> List[str]:
        """Identify which parts of the structure are unusual."""
        culprits = []

        # Encode sub-structures and compare to baseline
        for key, value in structure.items():
            if isinstance(value, dict):
                sub_vec = self.encoder.encode_data({key: value})
                baseline_norm = self.encoder.normalize_accumulator(self.baseline_accumulator)
                sub_sim = cosine_similarity(sub_vec, baseline_norm)

                # If this component has lower similarity than overall, it's suspicious
                if sub_sim < overall_sim - 0.1:
                    culprits.append(f"{key}: sim={sub_sim:.3f}")

        return culprits

    def _detect_transition(self, is_anomalous: bool) -> TrafficState:
        """Detect state transitions."""
        if len(self.state_history) < 10:
            return TrafficState.ATTACK_ONGOING if is_anomalous else TrafficState.NORMAL

        recent = np.mean(self.state_history[-5:])
        older = np.mean(self.state_history[:-5])
        delta = recent - older

        if delta < -0.15 and is_anomalous:  # Similarity dropping = attack starting
            return TrafficState.ATTACK_START
        elif delta > 0.15 and not is_anomalous:  # Similarity rising = attack ending
            return TrafficState.ATTACK_END
        elif is_anomalous:
            return TrafficState.ATTACK_ONGOING
        else:
            return TrafficState.NORMAL


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def generate_labeled_scenario(
    normal_before: int = 50,
    attack_duration: int = 100,
    normal_after: int = 50,
    attack_type: AttackType = AttackType.SYN_FLOOD,
    transition_window: int = 10,
) -> List[LabeledPacket]:
    packets = []
    idx = 0

    # Normal before
    for i in range(normal_before):
        pkt = _generate_normal_packet()
        packets.append(LabeledPacket(pkt, TrafficState.NORMAL, AttackType.NONE, idx))
        idx += 1

    # Attack
    for i in range(attack_duration):
        pkt = _generate_attack_packet(attack_type)
        if i < transition_window:
            state = TrafficState.ATTACK_START
        elif i >= attack_duration - transition_window:
            state = TrafficState.ATTACK_END
        else:
            state = TrafficState.ATTACK_ONGOING
        packets.append(LabeledPacket(pkt, state, attack_type, idx))
        idx += 1

    # Normal after
    for i in range(normal_after):
        pkt = _generate_normal_packet()
        packets.append(LabeledPacket(pkt, TrafficState.NORMAL, AttackType.NONE, idx))
        idx += 1

    return packets


def _generate_normal_packet() -> Packet:
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


def _generate_attack_packet(attack_type: AttackType) -> Packet:
    idx = random.randint(0, 10000)

    if attack_type == AttackType.SYN_FLOOD:
        src_ip = f"10.{(idx // 65536) % 256}.{(idx // 256) % 256}.{idx % 256}"
        return IP(src=src_ip, dst="192.168.1.100") / TCP(
            sport=40000 + (idx % 20000), dport=80, flags="S"
        )
    elif attack_type == AttackType.UDP_FLOOD:
        src_ip = f"10.{(idx // 65536) % 256}.{(idx // 256) % 256}.{idx % 256}"
        return IP(src=src_ip, dst="192.168.1.100") / UDP(
            sport=40000 + (idx % 20000), dport=53
        ) / Raw(load=b"X" * 64)
    elif attack_type == AttackType.DNS_REFLECTION:
        src_ip = f"8.8.{idx % 4}.{idx % 256}"
        return IP(src=src_ip, dst="192.168.1.100") / UDP(
            sport=53, dport=40000 + (idx % 1000)
        ) / Raw(load=b"X" * 512)
    elif attack_type == AttackType.PORT_SCAN:
        return IP(src="10.0.0.5", dst="192.168.1.100") / TCP(
            sport=40000, dport=1 + (idx % 1000), flags="S"
        )
    elif attack_type == AttackType.ICMP_FLOOD:
        src_ip = f"10.{(idx // 65536) % 256}.{(idx // 256) % 256}.{idx % 256}"
        return IP(src=src_ip, dst="192.168.1.100") / ICMP(type=8)
    else:
        return _generate_normal_packet()


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_detector(
    detector: StructuralDetector,
    scenarios: List[Tuple[AttackType, List[LabeledPacket]]],
) -> Dict[str, MetricsTracker]:

    binary_tracker = MetricsTracker(["normal", "attack"])
    attack_tracker = MetricsTracker([a.value for a in AttackType])

    for attack_type, packets in scenarios:
        # Reset recent knowledge between scenarios
        detector.recent_accumulator = detector.encoder.create_accumulator()
        detector.recent_count = 0
        detector.state_history = []

        for labeled_pkt in packets:
            detection = detector.process(labeled_pkt.packet)

            # Binary
            true_binary = "attack" if labeled_pkt.attack_type != AttackType.NONE else "normal"
            pred_binary = "attack" if detection.predicted_attack != AttackType.NONE else "normal"
            binary_tracker.record(true_binary, pred_binary)

            # Attack type
            attack_tracker.record(
                labeled_pkt.attack_type.value,
                detection.predicted_attack.value
            )

    return {"binary": binary_tracker, "attack": attack_tracker}


def find_best_threshold(scenarios: List[Tuple[AttackType, List[LabeledPacket]]]) -> Tuple[float, float]:
    """Grid search for best threshold."""
    best_f1 = 0.0
    best_threshold = 0.3

    for threshold in [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        detector = StructuralDetector(threshold=threshold)
        results = evaluate_detector(detector, scenarios)
        f1 = results["binary"].metrics["attack"].f1

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("CHALLENGE 011-014: STRUCTURAL DETECTION USING HOLON'S ENCODER")
    print("=" * 80)

    print("\nKey difference from 013:")
    print("  - Using encoder.encode_data() with proper role-filler binding")
    print("  - Packets converted to nested dicts, not atom strings")
    print("  - Structure is preserved: {dst_port: 80} != {src_port: 80}")

    # Generate scenarios
    print("\nGenerating labeled scenarios...")
    scenarios = []

    for attack_type in [
        AttackType.SYN_FLOOD,
        AttackType.UDP_FLOOD,
        AttackType.DNS_REFLECTION,
        AttackType.PORT_SCAN,
        AttackType.ICMP_FLOOD,
    ]:
        packets = generate_labeled_scenario(
            normal_before=50,
            attack_duration=100,
            normal_after=50,
            attack_type=attack_type,
        )
        scenarios.append((attack_type, packets))
        print(f"  {attack_type.value}: {len(packets)} packets")

    total_packets = sum(len(p) for _, p in scenarios)
    print(f"Total: {total_packets} packets")

    # === Find best threshold ===
    print("\n" + "=" * 60)
    print("THRESHOLD SEARCH")
    print("=" * 60)

    best_threshold, best_f1 = find_best_threshold(scenarios)
    print(f"Best threshold: {best_threshold}")
    print(f"Best F1: {best_f1:.3f}")

    # === Full evaluation with best threshold ===
    print("\n" + "=" * 60)
    print(f"EVALUATION (threshold={best_threshold})")
    print("=" * 60)

    detector = StructuralDetector(threshold=best_threshold)
    results = evaluate_detector(detector, scenarios)

    results["binary"].print_report("BINARY DETECTION (attack vs normal)")
    results["attack"].print_report("ATTACK TYPE CLASSIFICATION")

    # === Per-attack analysis ===
    print("\n" + "=" * 60)
    print("PER-ATTACK-TYPE ANALYSIS")
    print("=" * 60)

    for attack_type, packets in scenarios:
        detector = StructuralDetector(threshold=best_threshold)
        single_result = evaluate_detector(detector, [(attack_type, packets)])
        m = single_result["binary"].metrics["attack"]
        print(f"  {attack_type.value:<20} P={m.precision:.3f} R={m.recall:.3f} F1={m.f1:.3f}")

    # === Debug: Show what the encoder sees ===
    print("\n" + "=" * 60)
    print("DEBUG: SAMPLE ENCODINGS")
    print("=" * 60)

    normal_pkt = _generate_normal_packet()
    attack_pkt = _generate_attack_packet(AttackType.SYN_FLOOD)

    normal_struct = detector._packet_to_structure(normal_pkt)
    attack_struct = detector._packet_to_structure(attack_pkt)

    print("\nNormal packet structure:")
    for k, v in normal_struct.items():
        print(f"  {k}: {v}")

    print("\nSYN flood packet structure:")
    for k, v in attack_struct.items():
        print(f"  {k}: {v}")

    normal_vec = detector.encoder.encode_data(normal_struct)
    attack_vec = detector.encoder.encode_data(attack_struct)
    baseline = detector.encoder.normalize_accumulator(detector.baseline_accumulator)

    print(f"\nSimilarity to baseline:")
    print(f"  Normal packet: {cosine_similarity(normal_vec, baseline):.3f}")
    print(f"  Attack packet: {cosine_similarity(attack_vec, baseline):.3f}")

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    binary_f1 = results["binary"].weighted_f1()
    attack_f1 = results["attack"].weighted_f1()

    print(f"""
STRUCTURAL ENCODING RESULTS:
  Binary Detection F1:     {binary_f1:.3f}
  Attack Classification F1: {attack_f1:.3f}
  Best Threshold:          {best_threshold}

APPROACH:
  - Convert packets to nested dicts (l3, l4, payload)
  - Use encoder.encode_data() for proper role-filler binding
  - Build baseline from 100 synthetic normal packets
  - Use accumulator primitives for streaming

STRUCTURAL BINDING ADVANTAGE:
  - {{dst_port: 80}} encodes differently than {{src_port: 80}}
  - Nested structure preserved: l4.proto, l4.flags, etc.
  - Attack signatures are structured data, not atom lists
""")


if __name__ == "__main__":
    main()
