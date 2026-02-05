#!/usr/bin/env python3
"""
Challenge 011-017: Integrated Detector

Combines the best techniques from batches 010 and 011:

FROM BATCH 010:
- Smart normalization (port bucketing, IP prefix levels)
- Character class bitmasks for payload
- Rule-based detection for known attacks
- Variance-based DDoS detection

FROM BATCH 011:
- Structural encoding via encoder.encode_data()
- Prior/recent/compositional knowledge separation
- State machine for transition detection
- Sample-based signatures for classification
- Per-field culprit identification
"""

import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Set
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


# =============================================================================
# ENUMS AND DATACLASSES
# =============================================================================

class TransitionState(Enum):
    STABLE_NORMAL = "stable_normal"
    ATTACK_BEGINNING = "attack_beginning"
    STABLE_ATTACK = "stable_attack"
    ATTACK_ENDING = "attack_ending"


class AttackType(Enum):
    NONE = "none"
    SYN_FLOOD = "syn_flood"
    UDP_FLOOD = "udp_flood"
    DNS_REFLECTION = "dns_reflection"
    PORT_SCAN = "port_scan"
    ICMP_FLOOD = "icmp_flood"
    BINARY_PAYLOAD = "binary_payload"


@dataclass
class Detection:
    # Binary
    is_anomalous: bool
    confidence: float

    # Transition (from 011)
    transition_state: TransitionState

    # Classification (from 011 sample-based)
    attack_type: AttackType
    attack_confidence: float

    # Knowledge scores (from 011)
    prior_score: float
    recent_score: float
    divergence: float

    # Variance-based (from 010)
    variance: float
    mean_sim: float

    # Rule-based (from 010)
    rule_triggered: Optional[str]

    # Culprits (from 011)
    culprits: List[str]


# =============================================================================
# NORMALIZATION FUNCTIONS (FROM 010)
# =============================================================================

def bucket_port(port: int) -> str:
    if port < 1024:
        return "wellknown"
    elif port < 49152:
        return "registered"
    else:
        return "ephemeral"


def bucket_ip_prefix(ip: str, level: int = 16) -> str:
    parts = ip.split('.')
    if level == 8:
        return parts[0]
    elif level == 16:
        return f"{parts[0]}.{parts[1]}"
    else:
        return f"{parts[0]}.{parts[1]}.{parts[2]}"


def payload_bitmask(payload: bytes) -> int:
    """Character class bitmask from 010."""
    mask = 0
    for b in payload:
        if 97 <= b <= 122: mask |= 0x01  # lowercase
        elif 65 <= b <= 90: mask |= 0x02  # uppercase
        elif 48 <= b <= 57: mask |= 0x04  # digits
        elif b == 32: mask |= 0x08  # space
        elif b in (46, 44, 59, 58): mask |= 0x10  # punctuation
        elif b in (40, 41, 60, 62, 91, 93, 123, 125): mask |= 0x20  # brackets
        elif b in (34, 39, 96): mask |= 0x40  # quotes
        elif b < 32 or b > 126: mask |= 0x80  # binary/control
    return mask


# =============================================================================
# RULE-BASED CHECKS (FROM 010)
# =============================================================================

def check_reflection(pkt: Packet) -> Optional[str]:
    """Check for reflection attack (well-known port as source)."""
    if UDP in pkt and pkt[UDP].sport < 1024:
        if pkt[UDP].sport == 53:
            return "dns_reflection"
        elif pkt[UDP].sport == 123:
            return "ntp_reflection"
    return None


def check_syn_flood(pkt: Packet) -> Optional[str]:
    """Check for SYN flood (SYN-only from suspicious source)."""
    if TCP in pkt:
        flags = str(pkt[TCP].flags)
        if flags == "S" and IP in pkt:
            src_prefix = bucket_ip_prefix(pkt[IP].src, 8)
            if src_prefix == "10":  # RFC1918 space attacking
                return "syn_flood"
    return None


def check_binary_payload(pkt: Packet) -> Optional[str]:
    """Check for binary/suspicious payload on unexpected ports."""
    if Raw in pkt:
        # Skip well-known ports that normally have binary (TLS, DNS)
        if TCP in pkt and pkt[TCP].dport in (443, 8443):
            return None  # Expected binary (TLS)
        if UDP in pkt and pkt[UDP].dport == 53:
            return None  # Expected binary (DNS)

        bitmask = payload_bitmask(bytes(pkt[Raw].load))
        # Only flag if ONLY binary (no text) on unexpected port
        if (bitmask & 0x80) and not (bitmask & 0x07):
            return "binary_payload"
    return None


# =============================================================================
# INTEGRATED DETECTOR
# =============================================================================

class IntegratedDetector:
    """
    Best of both batches:
    - 010: normalization, bitmasks, rules, variance
    - 011: structural encoding, knowledge separation, state machine, sample signatures
    """

    def __init__(self):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        # === KNOWLEDGE SOURCES (from 011) ===
        self.prior = self.encoder.create_accumulator()
        self.prior_count = 0

        self.recent = self.encoder.create_accumulator()
        self.recent_count = 0
        self.decay = 0.98

        # === VARIANCE TRACKING (from 010) ===
        self.sim_history = deque(maxlen=50)
        self.baseline_variance = 0.03
        self.baseline_mean = 0.75

        # === STATE MACHINE (from 011) ===
        self.current_state = TransitionState.STABLE_NORMAL
        self.anomaly_streak = 0
        self.normal_streak = 0

        # === SIGNATURES (from 011 - sample-based) ===
        self.signatures: Dict[AttackType, np.ndarray] = {}

        # Initialize
        self._build_prior()
        self._build_signatures()

    def _packet_to_structure(self, pkt: Packet) -> dict:
        """
        Combined structure using 010's normalization and 011's nesting.
        """
        structure = {}

        # L3 with multi-level prefixes (from 010)
        if IP in pkt:
            structure["l3"] = {
                "src_prefix_8": bucket_ip_prefix(pkt[IP].src, 8),
                "src_prefix_16": bucket_ip_prefix(pkt[IP].src, 16),
                "dst_prefix_8": bucket_ip_prefix(pkt[IP].dst, 8),
                "dst_prefix_16": bucket_ip_prefix(pkt[IP].dst, 16),
            }

        # L4 with port bucketing (from 010)
        if TCP in pkt:
            structure["l4"] = {
                "proto": "tcp",
                "dst_port_bucket": bucket_port(pkt[TCP].dport),
                "src_port_bucket": bucket_port(pkt[TCP].sport),
                "flags": str(pkt[TCP].flags),
                "is_syn_only": str(pkt[TCP].flags) == "S",
            }
        elif UDP in pkt:
            structure["l4"] = {
                "proto": "udp",
                "dst_port_bucket": bucket_port(pkt[UDP].dport),
                "src_port_bucket": bucket_port(pkt[UDP].sport),
                "is_reflection": pkt[UDP].sport < 1024,
            }
        elif ICMP in pkt:
            structure["l4"] = {
                "proto": "icmp",
                "type": pkt[ICMP].type,
            }

        # Payload with bitmask (from 010)
        if Raw in pkt:
            payload = bytes(pkt[Raw].load)
            bitmask = payload_bitmask(payload)
            structure["payload"] = {
                "has_payload": True,
                "size_bucket": "small" if len(payload) < 64 else "medium" if len(payload) < 256 else "large",
                "bitmask": bitmask,
                "has_binary": bool(bitmask & 0x80),
                "has_text": bool(bitmask & 0x07),  # lower/upper/digits
            }
        else:
            structure["payload"] = {"has_payload": False}

        return structure

    def _build_prior(self):
        """Build frozen baseline from normal traffic."""
        sims = []
        for _ in range(200):
            pkt = generate_normal_packet()
            vec = self.encoder.encode_data(self._packet_to_structure(pkt))
            self.prior = self.encoder.accumulate(self.prior, vec)
            self.prior_count += 1

            if self.prior_count > 50:
                prior_norm = self.encoder.normalize_accumulator(self.prior)
                sims.append(cosine_similarity(vec, prior_norm))

        if sims:
            self.baseline_variance = np.var(sims)
            self.baseline_mean = np.mean(sims)

    def _build_signatures(self):
        """Build sample-based signatures (from 011)."""
        for attack_type in AttackType:
            if attack_type == AttackType.NONE:
                continue

            acc = self.encoder.create_accumulator()
            for _ in range(50):
                pkt = generate_attack_packet(attack_type)
                vec = self.encoder.encode_data(self._packet_to_structure(pkt))
                acc = self.encoder.accumulate(acc, vec)

            self.signatures[attack_type] = self.encoder.normalize_accumulator(acc)

    def process(self, pkt: Packet) -> Detection:
        """Process packet with integrated detection."""

        structure = self._packet_to_structure(pkt)
        vec = self.encoder.encode_data(structure)

        # === KNOWLEDGE SCORES (from 011) ===
        prior_norm = self.encoder.normalize_accumulator(self.prior)
        prior_score = cosine_similarity(vec, prior_norm)

        if self.recent_count > 20:
            recent_norm = self.encoder.normalize_accumulator(self.recent)
            recent_score = cosine_similarity(vec, recent_norm)
            divergence = cosine_similarity(prior_norm, recent_norm)
        else:
            recent_score = prior_score
            divergence = 1.0

        # === RULE-BASED DETECTION (from 010) ===
        rule_triggered = (
            check_reflection(pkt) or
            check_syn_flood(pkt) or
            check_binary_payload(pkt)
        )

        # === VARIANCE-BASED DETECTION (from 010) ===
        self.sim_history.append(prior_score)
        if len(self.sim_history) >= 20:
            current_var = np.var(list(self.sim_history))
            current_mean = np.mean(list(self.sim_history))
        else:
            current_var = self.baseline_variance
            current_mean = self.baseline_mean

        # === ANOMALY DETECTION ===
        # Combine: rule OR similarity threshold OR variance anomaly
        similarity_anomaly = prior_score < 0.4
        variance_anomaly = (current_var < self.baseline_variance * 0.3 and
                           abs(current_mean - self.baseline_mean) > 0.15)

        is_anomalous = bool(rule_triggered) or similarity_anomaly or variance_anomaly

        # === STREAKS ===
        if is_anomalous:
            self.anomaly_streak += 1
            self.normal_streak = 0
        else:
            self.normal_streak += 1
            self.anomaly_streak = 0

        # === STATE MACHINE (from 011) ===
        transition_state = self._update_state()

        # === CLASSIFICATION (from 011 - sample-based) ===
        if is_anomalous:
            attack_type, attack_confidence = self._classify(vec)
        else:
            attack_type = AttackType.NONE
            attack_confidence = 1.0 - prior_score

        # === CULPRIT IDENTIFICATION (from 011) ===
        culprits = self._identify_culprits(structure, prior_score)

        # === UPDATE RECENT (from 011) ===
        weight = 0.1 if is_anomalous else 1.0
        self.recent = self.decay * self.recent + weight * vec.astype(np.float64)
        self.recent_count += 1

        confidence = abs(prior_score - 0.4) if similarity_anomaly else 0.5

        return Detection(
            is_anomalous=is_anomalous,
            confidence=confidence,
            transition_state=transition_state,
            attack_type=attack_type,
            attack_confidence=attack_confidence,
            prior_score=prior_score,
            recent_score=recent_score,
            divergence=divergence,
            variance=current_var,
            mean_sim=current_mean,
            rule_triggered=rule_triggered,
            culprits=culprits,
        )

    def _update_state(self) -> TransitionState:
        """State machine from 011."""
        if self.current_state == TransitionState.STABLE_NORMAL:
            if self.anomaly_streak >= 3:
                self.current_state = TransitionState.ATTACK_BEGINNING
        elif self.current_state == TransitionState.ATTACK_BEGINNING:
            if self.anomaly_streak >= 15:
                self.current_state = TransitionState.STABLE_ATTACK
            elif self.normal_streak >= 3:
                self.current_state = TransitionState.STABLE_NORMAL
        elif self.current_state == TransitionState.STABLE_ATTACK:
            if self.normal_streak >= 3:
                self.current_state = TransitionState.ATTACK_ENDING
        elif self.current_state == TransitionState.ATTACK_ENDING:
            if self.normal_streak >= 15:
                self.current_state = TransitionState.STABLE_NORMAL
            elif self.anomaly_streak >= 3:
                self.current_state = TransitionState.STABLE_ATTACK

        return self.current_state

    def _classify(self, vec: np.ndarray) -> Tuple[AttackType, float]:
        """Sample-based classification from 011."""
        best_type = AttackType.NONE
        best_sim = 0.0

        for attack_type, sig in self.signatures.items():
            sim = cosine_similarity(vec, sig)
            if sim > best_sim:
                best_sim = sim
                best_type = attack_type

        return best_type, best_sim

    def _identify_culprits(self, structure: dict, overall_sim: float) -> List[str]:
        """Per-field culprit identification from 011."""
        culprits = []

        l4 = structure.get("l4", {})

        # Check for specific suspicious features
        if l4.get("is_syn_only"):
            culprits.append("TCP SYN-only (possible flood)")

        if l4.get("is_reflection"):
            culprits.append(f"UDP reflection (src_port < 1024)")

        payload = structure.get("payload", {})
        if payload.get("has_binary"):
            culprits.append("Binary payload content")

        if overall_sim < 0.3:
            culprits.append(f"Very low prior similarity: {overall_sim:.2f}")

        return culprits

    def reset(self):
        """Reset for new scenario (keep prior, reset recent)."""
        self.recent = self.encoder.create_accumulator()
        self.recent_count = 0
        self.sim_history.clear()
        self.anomaly_streak = 0
        self.normal_streak = 0
        self.current_state = TransitionState.STABLE_NORMAL


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def generate_normal_packet() -> Packet:
    pkt_type = random.choice(["http", "https", "dns"])
    if pkt_type == "http":
        return IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
            sport=random.randint(49152, 65535), dport=80, flags="PA"
        ) / Raw(load=b"GET /index.html HTTP/1.1\r\nHost: example.com\r\n\r\n")
    elif pkt_type == "https":
        return IP(src="192.168.1.50", dst="93.184.216.34") / TCP(
            sport=random.randint(49152, 65535), dport=443, flags="PA"
        ) / Raw(load=b"\x16\x03\x03" + b"ClientHello" + b"X" * 40)
    else:
        return IP(src="192.168.1.50", dst="8.8.8.8") / UDP(
            sport=random.randint(49152, 65535), dport=53
        ) / Raw(load=b"\x00\x01\x01\x00\x00\x01example\x03com\x00")


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
    elif attack_type == AttackType.BINARY_PAYLOAD:
        return IP(src="10.0.0.1", dst="192.168.1.100") / TCP(
            sport=40000, dport=31337, flags="PA"
        ) / Raw(load=b"\xde\xad\xbe\xef" + bytes(range(256))[:50])

    return generate_normal_packet()


# =============================================================================
# EVALUATION
# =============================================================================

@dataclass
class Metrics:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

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


def evaluate():
    print("=" * 70)
    print("CHALLENGE 011-017: INTEGRATED DETECTOR EVALUATION")
    print("=" * 70)

    detector = IntegratedDetector()

    binary_metrics = Metrics()
    classification_correct = 0
    classification_total = 0

    # Test scenarios
    scenarios = [
        ("Normal traffic", 100, AttackType.NONE),
        ("SYN flood", 100, AttackType.SYN_FLOOD),
        ("DNS reflection", 100, AttackType.DNS_REFLECTION),
        ("Port scan", 100, AttackType.PORT_SCAN),
        ("ICMP flood", 100, AttackType.ICMP_FLOOD),
        ("Binary payload", 50, AttackType.BINARY_PAYLOAD),
    ]

    for name, count, attack_type in scenarios:
        detector.reset()

        print(f"\n--- {name} ---")

        # Warmup with normal traffic
        for _ in range(30):
            pkt = generate_normal_packet()
            detector.process(pkt)

        detections = []
        classifications = []

        for i in range(count):
            if attack_type == AttackType.NONE:
                pkt = generate_normal_packet()
            else:
                pkt = generate_attack_packet(attack_type)

            det = detector.process(pkt)
            detections.append(det.is_anomalous)
            classifications.append(det.attack_type)

            # Metrics
            is_attack = attack_type != AttackType.NONE
            if is_attack and det.is_anomalous:
                binary_metrics.tp += 1
            elif is_attack and not det.is_anomalous:
                binary_metrics.fn += 1
            elif not is_attack and det.is_anomalous:
                binary_metrics.fp += 1
            else:
                binary_metrics.tn += 1

            # Classification
            if is_attack:
                classification_total += 1
                if det.attack_type == attack_type:
                    classification_correct += 1

            # Show sample
            if i == count - 1:
                print(f"  Final packet: anomalous={det.is_anomalous}, "
                      f"type={det.attack_type.value}, "
                      f"state={det.transition_state.value}")
                if det.rule_triggered:
                    print(f"  Rule triggered: {det.rule_triggered}")
                if det.culprits:
                    print(f"  Culprits: {det.culprits}")

        detection_rate = sum(detections) / len(detections) if detections else 0
        print(f"  Detection rate: {detection_rate:.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nBinary Detection:")
    print(f"  Precision: {binary_metrics.precision:.3f}")
    print(f"  Recall:    {binary_metrics.recall:.3f}")
    print(f"  F1 Score:  {binary_metrics.f1:.3f}")

    print(f"\nAttack Classification:")
    print(f"  Accuracy:  {classification_correct/classification_total:.1%} ({classification_correct}/{classification_total})")

    print(f"""
INTEGRATED TECHNIQUES:
  From 010:
    ✓ Port bucketing (wellknown/registered/ephemeral)
    ✓ Multi-level IP prefixes (/8, /16)
    ✓ Payload character bitmask
    ✓ Rule-based detection (reflection, SYN flood, binary)
    ✓ Variance-based anomaly detection

  From 011:
    ✓ Structural encoding via encode_data()
    ✓ Prior/recent knowledge separation
    ✓ State machine for transitions
    ✓ Sample-based attack signatures
    ✓ Per-field culprit identification
""")


if __name__ == "__main__":
    evaluate()
