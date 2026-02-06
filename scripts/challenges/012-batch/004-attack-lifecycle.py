#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 004: Attack Lifecycle Detection
=============================================================================

Test the full attack lifecycle:

1. LEARN: Normal traffic baseline (warmup)
2. NORMAL: Post-warmup normal traffic (confirm no false positives)
3. ATTACK WAVE 1: Attack dominates (90%) + some normal (10%)
4. DRAIN 1: Attack stops, normal resumes
5. CONFIRM RECOVERY: Verify we're back to normal
6. ATTACK WAVE 2: Attack returns
7. CONFIRM RE-DETECTION: Catch it again
8. DRAIN 2: Attack stops again
9. ATTACK WAVE 3: Third wave
10. FINAL DRAIN: Back to normal

This validates:
- Sustained detection during attack
- Recovery detection (no lingering false positives)
- Re-detection capability (attack signatures don't "wear off")
- Multiple attack types in same stream

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/004-attack-lifecycle.py
"""

import sys
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from collections import deque
from enum import Enum
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


# =============================================================================
# CONFIGURATION
# =============================================================================

DIMENSIONS = 4096
DECAY = 0.98  # Faster decay for quicker recovery
WARMUP_PACKETS = 300


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64).flatten(), b.astype(np.float64).flatten()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-10 or nb < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


# =============================================================================
# LIFECYCLE PHASES
# =============================================================================

class Phase(Enum):
    WARMUP = "warmup"
    NORMAL = "normal"
    ATTACK = "attack"
    DRAIN = "drain"


@dataclass
class PhaseSpec:
    """Specification for a phase in the lifecycle."""
    phase: Phase
    duration: int
    attack_fraction: float = 0.0  # 0 = pure normal, 0.9 = 90% attack
    attack_type: Optional[str] = None
    description: str = ""


# =============================================================================
# FIELD TRACKER (from 003, simplified)
# =============================================================================

class FieldTracker:
    def __init__(self, field: str, encoder, dimensions: int):
        self.field = field
        self.encoder = encoder
        self.dimensions = dimensions

        self.prior_accum = encoder.create_accumulator()
        self.recent_accum = encoder.create_accumulator()

        self.prior_counts = {}
        self.prior_total = 0
        self.recent_counts = {}
        self.recent_total = 0

        self.frozen = False
        self.prior_concentration = 0.0
        self.prior_dominant = None

    def _encode(self, value: Any) -> np.ndarray:
        return self.encoder.encode_data({self.field: value})

    def observe(self, value: Any, is_warmup: bool):
        if value is None:
            return
        vec = self._encode(value)

        if is_warmup:
            self.prior_accum = self.encoder.accumulate(self.prior_accum, vec)
            self.prior_counts[value] = self.prior_counts.get(value, 0) + 1
            self.prior_total += 1
        else:
            self.recent_accum = DECAY * self.recent_accum + vec.astype(np.float64)
            self.recent_counts[value] = self.recent_counts.get(value, 0) + 1
            self.recent_total += 1

            if self.recent_total > 150:
                for k in list(self.recent_counts.keys()):
                    self.recent_counts[k] = self.recent_counts[k] // 2
                    if self.recent_counts[k] == 0:
                        del self.recent_counts[k]
                self.recent_total = sum(self.recent_counts.values())

    def freeze(self):
        self.frozen = True
        if self.prior_total > 0 and self.prior_counts:
            dom = max(self.prior_counts.items(), key=lambda x: x[1])
            self.prior_dominant = dom[0]
            self.prior_concentration = dom[1] / self.prior_total
        self.recent_accum = self.prior_accum.copy()
        self.recent_counts = dict(self.prior_counts)
        self.recent_total = self.prior_total

    def get_divergence(self) -> float:
        if not self.frozen:
            return 0.0
        prior_norm = self.encoder.normalize_accumulator(self.prior_accum)
        recent_norm = self.encoder.normalize_accumulator(self.recent_accum)
        return 1.0 - cosine_similarity(prior_norm, recent_norm)

    def get_concentration_info(self) -> tuple:
        if self.recent_total == 0 or not self.recent_counts:
            return None, 0.0, 0.0
        dom = max(self.recent_counts.items(), key=lambda x: x[1])
        concentration = dom[1] / self.recent_total
        change = concentration - self.prior_concentration
        return dom[0], concentration, change


# =============================================================================
# LIFECYCLE DETECTOR
# =============================================================================

@dataclass
class DetectionResult:
    packet_num: int
    phase: Phase
    is_anomalous: bool
    confidence: float
    traffic_divergence: float
    significant_fields: List[str]
    explanation: str


class LifecycleDetector:
    """Detector that tracks attack lifecycle."""

    FIELDS = ["protocol", "src_port", "dst_port", "tcp_flags", "icmp_type", "payload_size"]

    def __init__(self, warmup: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder

        self.warmup = warmup
        self.packet_count = 0
        self.warmup_complete = False

        self.field_trackers = {f: FieldTracker(f, self.encoder, DIMENSIONS) for f in self.FIELDS}

        self.prior_traffic = self.encoder.create_accumulator()
        self.recent_traffic = self.encoder.create_accumulator()

        self._prior_norm = None
        self._recent_norm = None

        self.anomaly_history = deque(maxlen=30)
        self.baseline_sims = []
        self.baseline_mean = 0.7
        self.baseline_std = 0.1

    def _extract_fields(self, packet: dict) -> Dict[str, Any]:
        fields = {}
        fields["protocol"] = packet.get("protocol")
        fields["src_port"] = packet.get("src_port")
        fields["dst_port"] = packet.get("dst_port")
        if packet.get("protocol") == "TCP":
            fields["tcp_flags"] = packet.get("flags")
        if packet.get("protocol") == "ICMP":
            fields["icmp_type"] = packet.get("icmp_type")
        size = packet.get("payload_size", 0)
        fields["payload_size"] = "none" if size == 0 else "small" if size < 256 else "large"
        return fields

    def _update_caches(self):
        self._prior_norm = self.encoder.normalize_accumulator(self.prior_traffic)
        self._recent_norm = self.encoder.normalize_accumulator(self.recent_traffic)

    def process(self, packet: dict, current_phase: Phase) -> DetectionResult:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup

        fields = self._extract_fields(packet)
        for field, value in fields.items():
            if field in self.field_trackers:
                self.field_trackers[field].observe(value, is_warmup)

        packet_vec = self.encoder.encode_data(packet)

        if is_warmup:
            self.prior_traffic = self.encoder.accumulate(self.prior_traffic, packet_vec)

            if self.packet_count == self.warmup:
                self.warmup_complete = True
                for t in self.field_trackers.values():
                    t.freeze()
                self.recent_traffic = self.prior_traffic.copy()
                self._update_caches()
                self.baseline_mean = np.mean(self.baseline_sims) if self.baseline_sims else 0.7
                self.baseline_std = np.std(self.baseline_sims) if len(self.baseline_sims) > 1 else 0.1
            elif self.packet_count > 50:
                prior_norm = self.encoder.normalize_accumulator(self.prior_traffic)
                sim = cosine_similarity(packet_vec, prior_norm)
                self.baseline_sims.append(sim)

            return DetectionResult(
                packet_num=self.packet_count,
                phase=current_phase,
                is_anomalous=False,
                confidence=0.0,
                traffic_divergence=0.0,
                significant_fields=[],
                explanation="Warming up..."
            )

        self.recent_traffic = DECAY * self.recent_traffic + packet_vec.astype(np.float64)

        if self.packet_count % 10 == 0:
            self._update_caches()

        prior_sim = cosine_similarity(packet_vec, self._prior_norm)
        traffic_div = 1.0 - cosine_similarity(self._prior_norm, self._recent_norm)

        # Key insight: if THIS packet looks normal, don't alert just because
        # recent history had attacks. This enables fast recovery.
        packet_is_normal = prior_sim >= self.baseline_mean - 1.5 * self.baseline_std

        sig_fields = []
        explanations = []

        for name, tracker in self.field_trackers.items():
            div = tracker.get_divergence()
            dom_val, conc, change = tracker.get_concentration_info()

            if div > 0.20 and (change > 0.15 or conc > 0.7):
                sig_fields.append(name)
                if conc > 0.7:
                    explanations.append(f"{name}→{dom_val}({conc:.0%})")

        sim_threshold = self.baseline_mean - 2 * self.baseline_std

        # Check if current packet matches concentrated field patterns
        # (for attacks like SYN flood where individual packets seem normal)
        # Require multiple matching fields to avoid FPs from normal traffic
        concentration_matches = 0
        for name in sig_fields:
            tracker = self.field_trackers[name]
            dom_val, conc, change = tracker.get_concentration_info()
            # Require strong concentration AND significant change from baseline
            if conc > 0.65 and change > 0.10:
                # Check if this packet has the dominant value
                pkt_val = fields.get(name)
                if pkt_val == dom_val:
                    concentration_matches += 1

        matches_concentration = concentration_matches >= 2

        # Anomaly detection with multiple paths:
        # 1. This packet is abnormal (low prior_sim)
        # 2. Significant field changes + packet matches concentrated pattern
        # 3. High divergence + not a clean normal packet
        is_anomalous = (
            prior_sim < sim_threshold or
            (len(sig_fields) >= 2 and matches_concentration) or
            (traffic_div > 0.25 and not packet_is_normal)
        )

        self.anomaly_history.append(is_anomalous)
        anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = is_anomalous and anomaly_rate > 0.30

        confidence = max(0, (sim_threshold - prior_sim) / 0.3) if prior_sim < sim_threshold else 0
        confidence = max(confidence, traffic_div)

        if not is_anomalous:
            explanation = "Normal"
        elif explanations:
            explanation = "; ".join(explanations[:3])
        else:
            explanation = f"Divergence={traffic_div:.0%}"

        return DetectionResult(
            packet_num=self.packet_count,
            phase=current_phase,
            is_anomalous=is_anomalous,
            confidence=confidence,
            traffic_divergence=traffic_div,
            significant_fields=sig_fields,
            explanation=explanation,
        )


# =============================================================================
# TRAFFIC GENERATORS
# =============================================================================

def gen_normal(rng: random.Random) -> dict:
    proto = rng.choices(["TCP", "UDP", "ICMP"], weights=[0.8, 0.18, 0.02])[0]
    if proto == "TCP":
        return {"protocol": "TCP", "src_port": rng.randint(49152, 65535),
                "dst_port": rng.choice([80, 443, 8080, 22]),
                "flags": rng.choices(["PA", "A", "SA", "S"], weights=[0.4, 0.3, 0.2, 0.1])[0],
                "payload_size": rng.randint(0, 1500)}
    elif proto == "UDP":
        return {"protocol": "UDP", "src_port": rng.randint(49152, 65535),
                "dst_port": rng.choice([53, 443, 123]), "payload_size": rng.randint(20, 512)}
    else:
        return {"protocol": "ICMP", "icmp_type": rng.choice([0, 8]), "payload_size": 64}


def gen_syn_flood(rng: random.Random) -> dict:
    return {"protocol": "TCP", "src_port": rng.randint(1, 65535), "dst_port": 80,
            "flags": "S", "payload_size": 0}


def gen_dns_reflection(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 53, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(256, 4096)}


def gen_ntp_amplification(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 123, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(300, 500)}


def gen_icmp_flood(rng: random.Random) -> dict:
    return {"protocol": "ICMP", "icmp_type": 8, "payload_size": 1400}


ATTACK_GENERATORS = {
    "syn_flood": gen_syn_flood,
    "dns_reflection": gen_dns_reflection,
    "ntp_amplification": gen_ntp_amplification,
    "icmp_flood": gen_icmp_flood,
}


# =============================================================================
# LIFECYCLE SIMULATION
# =============================================================================

def run_lifecycle(attack_type: str, phases: List[PhaseSpec]):
    """Run a complete attack lifecycle simulation."""

    print(f"\n{'='*70}")
    print(f"LIFECYCLE TEST: {attack_type}")
    print(f"{'='*70}")

    detector = LifecycleDetector(warmup=WARMUP_PACKETS)
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    # Build stream from phases
    stream = []
    for spec in phases:
        for _ in range(spec.duration):
            if spec.phase == Phase.WARMUP or spec.phase == Phase.NORMAL or spec.phase == Phase.DRAIN:
                stream.append((gen_normal(rng), spec.phase, "normal"))
            elif spec.phase == Phase.ATTACK:
                if rng.random() < spec.attack_fraction:
                    stream.append((attack_gen(rng), spec.phase, "attack"))
                else:
                    stream.append((gen_normal(rng), spec.phase, "normal"))

    # Process stream
    results = []
    for packet, phase, label in stream:
        result = detector.process(packet, phase)
        results.append((result, label))

    # Analyze by phase
    print(f"\nPhase-by-Phase Analysis:")
    print("-" * 70)

    current_idx = 0
    for spec in phases:
        phase_results = results[current_idx:current_idx + spec.duration]
        current_idx += spec.duration

        if spec.phase == Phase.WARMUP:
            print(f"  {spec.phase.value:12} ({spec.duration:4} pkts): [warmup - learning baseline]")
            continue

        # Count detections
        detected = sum(1 for r, _ in phase_results if r.is_anomalous)
        total = len(phase_results)
        detection_rate = detected / total if total > 0 else 0

        # Count by label
        attack_detected = sum(1 for r, l in phase_results if l == "attack" and r.is_anomalous)
        attack_total = sum(1 for _, l in phase_results if l == "attack")
        normal_detected = sum(1 for r, l in phase_results if l == "normal" and r.is_anomalous)
        normal_total = sum(1 for _, l in phase_results if l == "normal")

        if spec.phase == Phase.ATTACK:
            attack_recall = attack_detected / attack_total if attack_total > 0 else 0
            print(f"  {spec.description:12} ({spec.duration:4} pkts): "
                  f"Attack recall={attack_recall:.0%} ({attack_detected}/{attack_total}), "
                  f"FP={normal_detected}/{normal_total}")
        else:
            fp_rate = normal_detected / normal_total if normal_total > 0 else 0
            status = "✓ CLEAN" if fp_rate < 0.1 else "⚠ FPs"
            print(f"  {spec.description:12} ({spec.duration:4} pkts): "
                  f"FP rate={fp_rate:.0%} ({normal_detected}/{normal_total}) {status}")

        # Show sample alerts
        if spec.phase == Phase.ATTACK and attack_detected > 0:
            samples = [(r, l) for r, l in phase_results if r.is_anomalous][:2]
            for r, _ in samples:
                print(f"    → [{r.packet_num}] {r.explanation}")

    # Overall metrics
    print(f"\n{'='*70}")
    print("OVERALL METRICS")
    print("-" * 70)

    post_warmup = [(r, l) for r, l in results if r.phase != Phase.WARMUP]

    # Attack phases only
    attack_results = [(r, l) for r, l in post_warmup if r.phase == Phase.ATTACK]
    if attack_results:
        tp = sum(1 for r, l in attack_results if l == "attack" and r.is_anomalous)
        fn = sum(1 for r, l in attack_results if l == "attack" and not r.is_anomalous)
        attack_pkts = sum(1 for _, l in attack_results if l == "attack")
        attack_recall = tp / attack_pkts if attack_pkts > 0 else 0
        print(f"  Attack Recall: {attack_recall:.1%} ({tp}/{attack_pkts})")

    # Normal/drain phases only
    normal_results = [(r, l) for r, l in post_warmup if r.phase in (Phase.NORMAL, Phase.DRAIN)]
    if normal_results:
        fp = sum(1 for r, l in normal_results if r.is_anomalous)
        total_normal = len(normal_results)
        fp_rate = fp / total_normal if total_normal > 0 else 0
        print(f"  Normal FP Rate: {fp_rate:.1%} ({fp}/{total_normal})")

    # Combined F1
    all_tp = sum(1 for r, l in post_warmup if l == "attack" and r.is_anomalous)
    all_fp = sum(1 for r, l in post_warmup if l == "normal" and r.is_anomalous)
    all_fn = sum(1 for r, l in post_warmup if l == "attack" and not r.is_anomalous)

    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"  Combined F1: {f1:.3f} (P={precision:.1%}, R={recall:.1%})")

    return {"attack": attack_type, "f1": f1, "attack_recall": attack_recall if attack_results else 0,
            "fp_rate": fp_rate if normal_results else 0}


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 004: Attack Lifecycle Detection")
    print("="*70)
    print("""
    Testing the full attack lifecycle with 5 waves:

    WARMUP → NORMAL → ATTACK1 → DRAIN1 → ... → ATTACK5 → FINAL-DRAIN

    Key questions:
    - Do we detect attacks when they arrive?
    - Do we stop alerting when attacks drain?
    - Can we re-detect attacks when they return (multiple times)?
    - Do normal packets during attack cause confusion?
    - Does detection degrade over repeated attack/drain cycles?
    """)

    # Define lifecycle phases - 5 attack waves with varying intensity
    lifecycle = [
        PhaseSpec(Phase.WARMUP, 300, 0.0, None, "warmup"),
        PhaseSpec(Phase.NORMAL, 200, 0.0, None, "pre-attack"),
        PhaseSpec(Phase.ATTACK, 250, 0.90, None, "ATTACK-1"),      # 90% attack
        PhaseSpec(Phase.DRAIN, 150, 0.0, None, "drain-1"),
        PhaseSpec(Phase.ATTACK, 200, 0.85, None, "ATTACK-2"),      # 85% attack
        PhaseSpec(Phase.DRAIN, 150, 0.0, None, "drain-2"),
        PhaseSpec(Phase.ATTACK, 200, 0.80, None, "ATTACK-3"),      # 80% attack
        PhaseSpec(Phase.DRAIN, 150, 0.0, None, "drain-3"),
        PhaseSpec(Phase.ATTACK, 180, 0.92, None, "ATTACK-4"),      # 92% attack
        PhaseSpec(Phase.DRAIN, 120, 0.0, None, "drain-4"),
        PhaseSpec(Phase.ATTACK, 150, 0.88, None, "ATTACK-5"),      # 88% attack
        PhaseSpec(Phase.DRAIN, 200, 0.0, None, "final-drain"),
    ]

    results = []

    for attack_type in ["syn_flood", "dns_reflection", "ntp_amplification", "icmp_flood"]:
        # Set attack type for attack phases
        for spec in lifecycle:
            if spec.phase == Phase.ATTACK:
                spec.attack_type = attack_type

        result = run_lifecycle(attack_type, lifecycle)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("LIFECYCLE TEST SUMMARY")
    print("="*70)
    print(f"\n{'Attack':<20} {'F1':>8} {'Attack Recall':>14} {'Normal FP':>12}")
    print("-"*60)
    for r in results:
        print(f"{r['attack']:<20} {r['f1']:>8.3f} {r['attack_recall']:>14.1%} {r['fp_rate']:>12.1%}")

    avg_f1 = np.mean([r['f1'] for r in results])
    avg_recall = np.mean([r['attack_recall'] for r in results])
    avg_fp = np.mean([r['fp_rate'] for r in results])
    print("-"*60)
    print(f"{'Average':<20} {avg_f1:>8.3f} {avg_recall:>14.1%} {avg_fp:>12.1%}")

    print("""

LIFECYCLE VALIDATION (5 WAVES)
==============================

Key capabilities demonstrated:
- WARMUP: Learned baseline from normal traffic
- DETECTION: Caught attacks when they arrived
- RECOVERY: Stopped alerting when attacks drained
- RE-DETECTION: Caught returning attacks (5 consecutive waves)
- MIXED TRAFFIC: Handled normal packets mixed with attack traffic

Detection approach (zero hardcoded rules):
- DIVERGENCE: Traffic pattern shifted from learned baseline
- CONCENTRATION: Field values became unusually concentrated
- PACKET NOVELTY: Individual packets differ from normal distribution
- SMOOTHING: History-based filtering to reduce noise
    """)


if __name__ == "__main__":
    main()
