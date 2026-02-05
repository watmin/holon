#!/usr/bin/env python3
"""
Challenge 011-015: Three Dimensions of Detection

Focus on the user's three key dimensions:
1. TRANSITION: Attack beginning vs ending (state machine)
2. CLASSIFICATION: What kind of attack
3. KNOWLEDGE: Prior, recent, compositional - how to blend them

Using Holon's proper structural encoding from 014.
"""

import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict, deque
from enum import Enum
import numpy as np

from scapy.all import IP, TCP, UDP, ICMP, Raw, Packet

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import VectorManager, CPUStore


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).flatten()
    b = b.astype(np.float64).flatten()
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# =============================================================================
# LABELS
# =============================================================================

class TransitionState(Enum):
    """Dimension 1: Transition detection."""
    STABLE_NORMAL = "stable_normal"
    ATTACK_BEGINNING = "attack_beginning"  # Transition normal→attack
    STABLE_ATTACK = "stable_attack"
    ATTACK_ENDING = "attack_ending"  # Transition attack→normal


class AttackType(Enum):
    """Dimension 2: Attack classification."""
    NONE = "none"
    SYN_FLOOD = "syn_flood"
    UDP_FLOOD = "udp_flood"
    DNS_REFLECTION = "dns_reflection"
    PORT_SCAN = "port_scan"
    ICMP_FLOOD = "icmp_flood"


@dataclass
class LabeledPacket:
    packet: Packet
    transition: TransitionState
    attack_type: AttackType
    packet_idx: int


@dataclass
class Detection:
    # Dimension 1: Transition
    transition: TransitionState
    transition_confidence: float

    # Dimension 2: Classification
    attack_type: AttackType
    attack_confidence: float

    # Dimension 3: Knowledge scores
    prior_score: float  # Similarity to frozen baseline
    recent_score: float  # Similarity to recent traffic
    compositional_score: float  # Divergence signal

    # Debug
    is_anomalous: bool


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

    def record(self, true_label: str, pred_label: str):
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

    def print_report(self, title: str):
        print(f"\n{title}")
        print("-" * 65)
        print(f"{'Class':<20} {'Prec':<10} {'Recall':<10} {'F1':<10} {'Support'}")
        print("-" * 65)
        for cls in self.classes:
            m = self.metrics[cls]
            support = m.tp + m.fn
            if support > 0:
                print(f"{cls:<20} {m.precision:<10.3f} {m.recall:<10.3f} {m.f1:<10.3f} {support}")
        print("-" * 65)
        print(f"{'Macro F1':<20} {'':<10} {'':<10} {self.macro_f1():<10.3f}")


# =============================================================================
# THREE-DIMENSIONAL DETECTOR
# =============================================================================

class ThreeDimensionalDetector:
    """
    Detector focusing on three dimensions:
    1. Transition detection (beginning/ending)
    2. Attack classification
    3. Knowledge composition (prior/recent/compositional)
    """

    def __init__(self, threshold: float = 0.4):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.threshold = threshold

        # === DIMENSION 3: KNOWLEDGE SOURCES ===

        # Prior knowledge (frozen baseline from training)
        self.prior_accumulator = self.encoder.create_accumulator()
        self.prior_count = 0

        # Recent knowledge (adaptive, decaying)
        self.recent_accumulator = self.encoder.create_accumulator()
        self.recent_count = 0
        self.decay = 0.98

        # Compositional: difference between prior and recent
        # This captures "what's changing"

        # === DIMENSION 2: ATTACK SIGNATURES ===
        self.attack_signatures: Dict[AttackType, np.ndarray] = {}

        # === DIMENSION 1: TRANSITION STATE MACHINE ===
        self.current_state = TransitionState.STABLE_NORMAL
        self.score_history: deque = deque(maxlen=30)
        self.anomaly_streak = 0
        self.normal_streak = 0

        # Initialize
        self._build_prior_knowledge()
        self._build_attack_signatures()

    def _packet_to_structure(self, pkt: Packet) -> dict:
        """Convert packet to nested structure for Holon encoding."""
        structure = {}

        if IP in pkt:
            src_parts = pkt[IP].src.split('.')
            dst_parts = pkt[IP].dst.split('.')
            structure["l3"] = {
                "src_net": f"{src_parts[0]}.{src_parts[1]}",
                "dst_net": f"{dst_parts[0]}.{dst_parts[1]}",
            }

        if TCP in pkt:
            flags = str(pkt[TCP].flags)
            structure["l4"] = {
                "proto": "tcp",
                "dst_port": pkt[TCP].dport,
                "dst_port_class": "wellknown" if pkt[TCP].dport < 1024 else "high",
                "flags": flags,
                "is_syn": "S" in flags and "A" not in flags,
            }
        elif UDP in pkt:
            structure["l4"] = {
                "proto": "udp",
                "dst_port": pkt[UDP].dport,
                "src_port_class": "wellknown" if pkt[UDP].sport < 1024 else "ephemeral",
            }
        elif ICMP in pkt:
            structure["l4"] = {
                "proto": "icmp",
                "type": pkt[ICMP].type,
            }

        if Raw in pkt:
            size = len(pkt[Raw].load)
            structure["payload"] = {
                "present": True,
                "size_class": "empty" if size == 0 else "small" if size < 64
                             else "medium" if size < 256 else "large",
            }
        else:
            structure["payload"] = {"present": False, "size_class": "empty"}

        return structure

    def _build_prior_knowledge(self):
        """Build frozen baseline from synthetic normal traffic."""
        for _ in range(200):  # More samples for better baseline
            pkt = _generate_normal_packet()
            vec = self.encoder.encode_data(self._packet_to_structure(pkt))
            self.prior_accumulator = self.encoder.accumulate(self.prior_accumulator, vec)
            self.prior_count += 1

    def _build_attack_signatures(self):
        """Build distinct signatures for each attack type."""
        # Build signatures from actual attack samples (more accurate)
        attack_samples = {
            AttackType.SYN_FLOOD: [_generate_attack_packet(AttackType.SYN_FLOOD) for _ in range(50)],
            AttackType.UDP_FLOOD: [_generate_attack_packet(AttackType.UDP_FLOOD) for _ in range(50)],
            AttackType.DNS_REFLECTION: [_generate_attack_packet(AttackType.DNS_REFLECTION) for _ in range(50)],
            AttackType.PORT_SCAN: [_generate_attack_packet(AttackType.PORT_SCAN) for _ in range(50)],
            AttackType.ICMP_FLOOD: [_generate_attack_packet(AttackType.ICMP_FLOOD) for _ in range(50)],
        }

        for attack_type, samples in attack_samples.items():
            acc = self.encoder.create_accumulator()
            for pkt in samples:
                vec = self.encoder.encode_data(self._packet_to_structure(pkt))
                acc = self.encoder.accumulate(acc, vec)
            self.attack_signatures[attack_type] = self.encoder.normalize_accumulator(acc)

    def process(self, pkt: Packet) -> Detection:
        """Process packet and return three-dimensional detection."""

        structure = self._packet_to_structure(pkt)
        packet_vec = self.encoder.encode_data(structure)

        # === DIMENSION 3: KNOWLEDGE SCORES ===

        # Prior score (similarity to frozen baseline)
        prior_norm = self.encoder.normalize_accumulator(self.prior_accumulator)
        prior_score = cosine_similarity(packet_vec, prior_norm)

        # Recent score (similarity to recent traffic)
        if self.recent_count > 5:
            recent_norm = self.encoder.normalize_accumulator(self.recent_accumulator)
            recent_score = cosine_similarity(packet_vec, recent_norm)
        else:
            recent_score = prior_score

        # Compositional score: how different is recent from prior?
        # This captures "drift" or "shift" in traffic patterns
        if self.recent_count > 5:
            compositional_score = cosine_similarity(prior_norm, recent_norm)
        else:
            compositional_score = 1.0  # No divergence yet

        # Anomaly detection using prior (frozen baseline)
        is_anomalous = prior_score < self.threshold

        # === DIMENSION 2: ATTACK CLASSIFICATION ===
        if is_anomalous:
            attack_scores = {}
            for attack_type, sig_vec in self.attack_signatures.items():
                sim = cosine_similarity(packet_vec, sig_vec)
                attack_scores[attack_type] = sim

            # Sort by score
            sorted_attacks = sorted(attack_scores.items(), key=lambda x: -x[1])
            best_attack, best_score = sorted_attacks[0]

            # Confidence based on separation from second-best
            if len(sorted_attacks) > 1:
                second_score = sorted_attacks[1][1]
                attack_confidence = best_score - second_score
            else:
                attack_confidence = best_score

            if best_score > 0.15:
                predicted_attack = best_attack
            else:
                predicted_attack = AttackType.NONE
        else:
            predicted_attack = AttackType.NONE
            attack_confidence = 1.0 - prior_score  # High prior = high confidence it's normal

        # === DIMENSION 1: TRANSITION DETECTION ===
        self.score_history.append(prior_score)

        if is_anomalous:
            self.anomaly_streak += 1
            self.normal_streak = 0
        else:
            self.normal_streak += 1
            self.anomaly_streak = 0

        transition = self._detect_transition(prior_score)
        transition_confidence = self._compute_transition_confidence()

        # Update recent knowledge (with decay)
        self.recent_accumulator *= self.decay
        weight = 0.1 if is_anomalous else 1.0
        self.recent_accumulator = self.encoder.accumulate(
            self.recent_accumulator, weight * packet_vec.astype(np.float64)
        )
        self.recent_count += 1

        return Detection(
            transition=transition,
            transition_confidence=transition_confidence,
            attack_type=predicted_attack,
            attack_confidence=attack_confidence,
            prior_score=prior_score,
            recent_score=recent_score,
            compositional_score=compositional_score,
            is_anomalous=is_anomalous,
        )

    def _detect_transition(self, current_score: float) -> TransitionState:
        """
        Detect state transitions using streaks.

        State machine matches realistic labeling:
        - stable_normal → attack_beginning: when anomaly streak starts
        - attack_beginning → stable_attack: when attack is sustained
        - stable_attack → attack_ending: when normal packets return
        - attack_ending → stable_normal: when recovery is confirmed
        """
        is_normal = current_score >= self.threshold

        if len(self.score_history) < 3:
            if is_normal:
                return TransitionState.STABLE_NORMAL
            return TransitionState.STABLE_ATTACK

        # State machine
        if self.current_state == TransitionState.STABLE_NORMAL:
            if self.anomaly_streak >= 3:
                self.current_state = TransitionState.ATTACK_BEGINNING

        elif self.current_state == TransitionState.ATTACK_BEGINNING:
            if self.anomaly_streak >= 15:  # Confirmed attack
                self.current_state = TransitionState.STABLE_ATTACK
            elif self.normal_streak >= 3:  # False alarm
                self.current_state = TransitionState.STABLE_NORMAL

        elif self.current_state == TransitionState.STABLE_ATTACK:
            if self.normal_streak >= 3:  # Attack ending, recovery starting
                self.current_state = TransitionState.ATTACK_ENDING

        elif self.current_state == TransitionState.ATTACK_ENDING:
            if self.normal_streak >= 15:  # Confirmed recovery
                self.current_state = TransitionState.STABLE_NORMAL
            elif self.anomaly_streak >= 3:  # Attack resumed
                self.current_state = TransitionState.STABLE_ATTACK

        return self.current_state

    def _compute_transition_confidence(self) -> float:
        """Compute confidence in transition detection."""
        if len(self.score_history) < 5:
            return 0.5

        scores = list(self.score_history)
        recent = np.mean(scores[-5:])
        variance = np.var(scores[-10:]) if len(scores) >= 10 else np.var(scores)

        # Low variance = high confidence in current state
        return 1.0 - min(variance * 10, 0.5)

    def reset(self):
        """Reset for new scenario (keep prior, reset recent)."""
        self.recent_accumulator = self.encoder.create_accumulator()
        self.recent_count = 0
        self.score_history.clear()
        self.anomaly_streak = 0
        self.normal_streak = 0
        self.current_state = TransitionState.STABLE_NORMAL


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def generate_labeled_scenario(
    normal_before: int = 50,
    attack_duration: int = 100,
    normal_after: int = 50,
    attack_type: AttackType = AttackType.SYN_FLOOD,
    transition_window: int = 15,
) -> List[LabeledPacket]:
    """
    Generate labeled scenario with realistic transition labeling.

    Transitions are labeled based on what's actually detectable:
    - attack_beginning: First N packets AFTER attack starts (still attack packets)
    - attack_ending: First N packets AFTER attack ends (normal packets, during recovery)

    This matches what a detector can actually observe:
    - Beginning: anomaly streak starts
    - Ending: normal streak starts after sustained attack
    """
    packets = []
    idx = 0

    # Normal before
    for _ in range(normal_before):
        pkt = _generate_normal_packet()
        packets.append(LabeledPacket(pkt, TransitionState.STABLE_NORMAL, AttackType.NONE, idx))
        idx += 1

    # Attack phase
    for i in range(attack_duration):
        pkt = _generate_attack_packet(attack_type)

        if i < transition_window:
            trans = TransitionState.ATTACK_BEGINNING
        else:
            trans = TransitionState.STABLE_ATTACK

        packets.append(LabeledPacket(pkt, trans, attack_type, idx))
        idx += 1

    # Normal after - first N are "attack_ending" (recovery phase)
    for i in range(normal_after):
        pkt = _generate_normal_packet()

        if i < transition_window:
            trans = TransitionState.ATTACK_ENDING  # Recovery phase
        else:
            trans = TransitionState.STABLE_NORMAL

        packets.append(LabeledPacket(pkt, trans, AttackType.NONE, idx))
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

def evaluate(
    detector: ThreeDimensionalDetector,
    scenarios: List[Tuple[AttackType, List[LabeledPacket]]],
) -> Dict[str, MetricsTracker]:

    # Trackers for each dimension
    transition_tracker = MetricsTracker([t.value for t in TransitionState])
    attack_tracker = MetricsTracker([a.value for a in AttackType])
    binary_tracker = MetricsTracker(["normal", "attack"])

    for attack_type, packets in scenarios:
        detector.reset()

        for labeled_pkt in packets:
            detection = detector.process(labeled_pkt.packet)

            # Dimension 1: Transition
            transition_tracker.record(
                labeled_pkt.transition.value,
                detection.transition.value
            )

            # Dimension 2: Attack type
            attack_tracker.record(
                labeled_pkt.attack_type.value,
                detection.attack_type.value
            )

            # Binary detection
            true_bin = "attack" if labeled_pkt.attack_type != AttackType.NONE else "normal"
            pred_bin = "attack" if detection.attack_type != AttackType.NONE else "normal"
            binary_tracker.record(true_bin, pred_bin)

    return {
        "transition": transition_tracker,
        "attack": attack_tracker,
        "binary": binary_tracker,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("CHALLENGE 011-015: THREE DIMENSIONS OF DETECTION")
    print("=" * 80)

    print("""
Three dimensions:
  1. TRANSITION:  stable_normal → attack_beginning → stable_attack → attack_ending
  2. CLASSIFICATION: syn_flood, udp_flood, dns_reflection, port_scan, icmp_flood
  3. KNOWLEDGE: prior (frozen), recent (adaptive), compositional (divergence)
""")

    # Generate scenarios
    scenarios = []
    for attack_type in AttackType:
        if attack_type == AttackType.NONE:
            continue
        packets = generate_labeled_scenario(
            normal_before=50,
            attack_duration=100,
            normal_after=50,
            attack_type=attack_type,
        )
        scenarios.append((attack_type, packets))
        print(f"  {attack_type.value}: {len(packets)} packets")

    print(f"\nTotal: {sum(len(p) for _, p in scenarios)} packets")

    # Evaluate
    detector = ThreeDimensionalDetector(threshold=0.4)
    results = evaluate(detector, scenarios)

    # Reports
    print("\n" + "=" * 65)
    print("DIMENSION 1: TRANSITION DETECTION")
    print("=" * 65)
    results["transition"].print_report("Transition States")

    print("\n" + "=" * 65)
    print("DIMENSION 2: ATTACK CLASSIFICATION")
    print("=" * 65)
    results["attack"].print_report("Attack Types")

    print("\n" + "=" * 65)
    print("DIMENSION 3: KNOWLEDGE COMPOSITION")
    print("=" * 65)
    results["binary"].print_report("Binary Detection")

    # Knowledge analysis
    print("\nKnowledge source analysis:")
    detector2 = ThreeDimensionalDetector(threshold=0.4)

    # Run one scenario to collect scores
    _, packets = scenarios[0]
    detector2.reset()

    prior_scores = []
    recent_scores = []
    comp_scores = []

    for pkt in packets:
        det = detector2.process(pkt.packet)
        prior_scores.append(det.prior_score)
        recent_scores.append(det.recent_score)
        comp_scores.append(det.compositional_score)

    # Split by phase
    normal1 = prior_scores[:50]
    attack = prior_scores[50:150]
    normal2 = prior_scores[150:]

    print(f"\n  Prior knowledge scores (SYN flood scenario):")
    print(f"    Normal (before): mean={np.mean(normal1):.3f}, std={np.std(normal1):.3f}")
    print(f"    Attack:          mean={np.mean(attack):.3f}, std={np.std(attack):.3f}")
    print(f"    Normal (after):  mean={np.mean(normal2):.3f}, std={np.std(normal2):.3f}")

    comp1 = comp_scores[:50]
    comp_attack = comp_scores[50:150]
    comp2 = comp_scores[150:]

    print(f"\n  Compositional (prior vs recent divergence):")
    print(f"    Normal (before): mean={np.mean(comp1):.3f}")
    print(f"    Attack:          mean={np.mean(comp_attack):.3f}")
    print(f"    Normal (after):  mean={np.mean(comp2):.3f}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    trans_f1 = results["transition"].macro_f1()
    attack_f1 = results["attack"].macro_f1()
    binary_f1 = results["binary"].weighted_f1()

    print(f"""
DIMENSION SCORES:
  1. Transition F1:      {trans_f1:.3f}
  2. Attack Class F1:    {attack_f1:.3f}
  3. Binary Detection:   {binary_f1:.3f}

KNOWLEDGE SOURCES:
  - Prior:         Frozen baseline, high for normal (~0.85), low for attack (~0.1)
  - Recent:        Adaptive, tracks current traffic pattern
  - Compositional: Divergence between prior/recent, detects regime change

STATE MACHINE:
  stable_normal → attack_beginning (3+ anomalous) → stable_attack (10+ anomalous)
  stable_attack → attack_ending (3+ normal) → stable_normal (10+ normal)
""")


if __name__ == "__main__":
    main()
