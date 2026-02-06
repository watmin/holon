#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 011: Frozen Baseline with Z-Scores
=============================================================================

Problem with 010: Z-score tracker was updating DURING detection, causing
it to adapt to attack traffic and lose sensitivity.

Fix: Freeze baseline statistics during warmup, don't update during detection.

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/011-frozen-baseline.py
"""

import sys
import random
from dataclasses import dataclass
from typing import List
from collections import deque
from enum import Enum
import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 4)[0])

from holon import CPUStore


DIMENSIONS = 4096
WARMUP_PACKETS = 400


# =============================================================================
# FROZEN Z-SCORE BASELINE
# =============================================================================

class FrozenZScoreBaseline:
    """
    Z-score baseline that FREEZES after warmup.

    Unlike adaptive z-scores, this captures normal behavior during warmup
    and never updates afterward. Anomalies are measured against frozen baseline.
    """

    def __init__(self):
        self.samples = []
        self.frozen = False
        self.mean = 0.0
        self.std = 1.0

    def observe(self, value: float):
        """Add sample during warmup."""
        if not self.frozen:
            self.samples.append(value)

    def freeze(self):
        """Freeze baseline statistics."""
        self.frozen = True
        if self.samples:
            self.mean = np.mean(self.samples)
            self.std = np.std(self.samples) if len(self.samples) > 1 else 0.05
            # Minimum std to prevent division by near-zero
            self.std = max(self.std, 0.02)

    def z_score(self, value: float) -> float:
        """Get z-score relative to frozen baseline."""
        return (value - self.mean) / self.std


# =============================================================================
# SIMPLE HYSTERESIS
# =============================================================================

class SimpleHysteresis:
    """
    Simple state machine with different enter/exit thresholds.
    """

    def __init__(self, enter_z: float = -3.0, exit_z: float = -1.0):
        self.enter_z = enter_z
        self.exit_z = exit_z
        self.in_anomaly = False

    def update(self, z: float) -> bool:
        if not self.in_anomaly:
            if z < self.enter_z:
                self.in_anomaly = True
        else:
            if z > self.exit_z:
                self.in_anomaly = False
        return self.in_anomaly


# =============================================================================
# FROZEN BASELINE DETECTOR
# =============================================================================

class FrozenBaselineDetector:
    """
    Detector with completely frozen baseline from warmup.

    Key insight: The baseline NEVER changes after warmup.
    All anomaly detection is relative to the frozen baseline.
    """

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Rate baseline
        self.rate_encoder = ContinuousRateEncoder(DIMENSIONS)
        self.rate_accum = self.encoder.create_accumulator()
        self._rate_norm = None
        self.rate_baseline = FrozenZScoreBaseline()

        # Pattern baseline
        self.pattern_accum = self.encoder.create_accumulator()
        self._pattern_norm = None
        self.pattern_baseline = FrozenZScoreBaseline()

        # Hysteresis
        self.rate_hysteresis = SimpleHysteresis(enter_z=-3.0, exit_z=-1.0)
        self.pattern_hysteresis = SimpleHysteresis(enter_z=-2.5, exit_z=-1.0)

        # Confirmation (require N consecutive)
        self.consecutive_anomalies = 0
        self.required_consecutive = 3

        # Smoothing
        self.anomaly_history = deque(maxlen=15)

    def process(self, packet: dict, pps: float) -> dict:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        # Encode
        packet_vec = self.encoder.encode_data(packet)
        rate_vec = self.rate_encoder.encode_rate(pps)

        if is_warmup:
            # Accumulate
            self.rate_accum = self.encoder.accumulate(self.rate_accum, rate_vec)
            self.pattern_accum = self.encoder.accumulate(self.pattern_accum, packet_vec)

            # Sample similarities after initial accumulation
            if self.packet_count > 100:
                temp_rate = self.encoder.normalize_accumulator(self.rate_accum)
                temp_pattern = self.encoder.normalize_accumulator(self.pattern_accum)

                rate_sim = self.store.similarity(rate_vec, temp_rate, metric="cosine")
                pattern_sim = self.store.similarity(packet_vec, temp_pattern, metric="cosine")

                self.rate_baseline.observe(rate_sim)
                self.pattern_baseline.observe(pattern_sim)

            if self.packet_count == self.warmup_packets:
                self.warmup_complete = True
                self._rate_norm = self.encoder.normalize_accumulator(self.rate_accum)
                self._pattern_norm = self.encoder.normalize_accumulator(self.pattern_accum)
                self.rate_baseline.freeze()
                self.pattern_baseline.freeze()

                print(f"  Frozen baseline: rate mean={self.rate_baseline.mean:.3f}, "
                      f"std={self.rate_baseline.std:.3f}")
                print(f"  Frozen baseline: pattern mean={self.pattern_baseline.mean:.3f}, "
                      f"std={self.pattern_baseline.std:.3f}")

            return {"packet_num": self.packet_count, "is_anomalous": False, "explanation": "Warming up..."}

        # Post-warmup: compute similarities against FROZEN baseline
        rate_sim = self.store.similarity(rate_vec, self._rate_norm, metric="cosine")
        pattern_sim = self.store.similarity(packet_vec, self._pattern_norm, metric="cosine")

        # Z-scores against FROZEN baseline
        rate_z = self.rate_baseline.z_score(rate_sim)
        pattern_z = self.pattern_baseline.z_score(pattern_sim)

        # Hysteresis updates
        rate_anomalous = self.rate_hysteresis.update(rate_z)
        pattern_anomalous = self.pattern_hysteresis.update(pattern_z)

        # Combined: rate OR pattern
        raw_anomaly = rate_anomalous or pattern_anomalous

        # Confirmation
        if raw_anomaly:
            self.consecutive_anomalies += 1
        else:
            self.consecutive_anomalies = 0

        confirmed = self.consecutive_anomalies >= self.required_consecutive

        # Smooth output
        self.anomaly_history.append(confirmed)
        anomaly_rate = sum(self.anomaly_history) / len(self.anomaly_history)
        is_anomalous = anomaly_rate > 0.4

        return {
            "packet_num": self.packet_count,
            "is_anomalous": is_anomalous,
            "rate_sim": rate_sim,
            "pattern_sim": pattern_sim,
            "rate_z": rate_z,
            "pattern_z": pattern_z,
            "rate_anomalous": rate_anomalous,
            "pattern_anomalous": pattern_anomalous,
            "confirmed": confirmed,
            "explanation": self._explanation(rate_z, pattern_z, rate_anomalous, pattern_anomalous),
        }

    def _explanation(self, rate_z, pattern_z, rate_anom, pattern_anom) -> str:
        parts = []
        if rate_anom:
            parts.append(f"rate_z={rate_z:.1f}")
        if pattern_anom:
            parts.append(f"pattern_z={pattern_z:.1f}")
        return "; ".join(parts) if parts else "Normal"


# =============================================================================
# CONTINUOUS RATE ENCODER
# =============================================================================

class ContinuousRateEncoder:
    def __init__(self, dimensions: int, scale: float = 1000.0):
        self.dimensions = dimensions
        self.scale = scale

    def encode_rate(self, pps: float) -> np.ndarray:
        if pps <= 0:
            pps = 1
        log_rate = np.log10(pps)
        return self._positional_encode(log_rate)

    def _positional_encode(self, value: float) -> np.ndarray:
        indices = np.arange(self.dimensions)
        freqs = 1 / (self.scale ** (indices / self.dimensions))
        values = np.where(indices % 2 == 0, np.sin(value * freqs), np.cos(value * freqs))
        return np.sign(values).astype(np.int8)


# =============================================================================
# SIMULATION
# =============================================================================

class Phase(Enum):
    WARMUP = "warmup"
    CALM = "calm"
    ATTACK = "attack"


@dataclass
class TimePhase:
    name: str
    duration_seconds: int
    packets_per_second: int
    phase_type: Phase
    attack_fraction: float = 0.95


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


def gen_dns_reflection(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": 53, "dst_port": rng.randint(49152, 65535),
            "payload_size": rng.randint(256, 4096)}


def gen_syn_flood(rng: random.Random) -> dict:
    return {"protocol": "TCP", "src_port": rng.randint(1, 65535), "dst_port": 80,
            "flags": "S", "payload_size": 0}


def gen_udp_flood(rng: random.Random) -> dict:
    return {"protocol": "UDP", "src_port": rng.randint(1, 65535),
            "dst_port": rng.randint(1, 65535), "payload_size": rng.randint(0, 1400)}


ATTACK_GENERATORS = {
    "dns_reflection": gen_dns_reflection,
    "syn_flood": gen_syn_flood,
    "udp_flood": gen_udp_flood,
}


def run_test(attack_type: str, phases: List[TimePhase], scale: float = 0.005):
    print(f"\n{'='*70}")
    print(f"FROZEN BASELINE TEST: {attack_type}")
    print(f"{'='*70}")

    first_calm_packets = int(phases[0].duration_seconds * phases[0].packets_per_second * scale)
    warmup_packets = min(first_calm_packets - 10, 400)

    detector = FrozenBaselineDetector(warmup_packets=warmup_packets)
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    phase_results = []

    for phase in phases:
        scaled_packets = max(1, int(phase.duration_seconds * phase.packets_per_second * scale))
        phase_detections = 0
        rate_triggers = 0
        pattern_triggers = 0

        for i in range(scaled_packets):
            if phase.phase_type == Phase.ATTACK and rng.random() < phase.attack_fraction:
                packet = attack_gen(rng)
            else:
                packet = gen_normal(rng)

            result = detector.process(packet, phase.packets_per_second)

            if detector.warmup_complete:
                if result["is_anomalous"]:
                    phase_detections += 1
                if result.get("rate_anomalous", False):
                    rate_triggers += 1
                if result.get("pattern_anomalous", False):
                    pattern_triggers += 1

        if not detector.warmup_complete or phase.name == "calm-1":
            status = "WARMUP"
            detection_rate = 0
        else:
            detection_rate = phase_detections / scaled_packets if scaled_packets > 0 else 0
            if phase.phase_type == Phase.ATTACK:
                status = "DETECTED" if detection_rate > 0.5 else "MISSED"
            else:
                status = "CLEAN" if detection_rate < 0.03 else "FP"

        phase_results.append({
            "name": phase.name,
            "phase_type": phase.phase_type,
            "packets": scaled_packets,
            "detections": phase_detections,
            "detection_rate": detection_rate,
            "rate_triggers": rate_triggers,
            "pattern_triggers": pattern_triggers,
            "status": status,
        })

    # Print results
    print(f"\n  Phase Results:")
    print(f"  {'-'*70}")

    for pr in phase_results:
        if pr["status"] == "WARMUP":
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): [WARMUP]")
        else:
            marker = "✓" if pr["status"] in ["DETECTED", "CLEAN"] else "✗"
            phase_type = "Attack" if pr["phase_type"] == Phase.ATTACK else "Normal"
            print(f"  {pr['name']:12} ({pr['packets']:5} pkts): {phase_type} "
                  f"Det={pr['detection_rate']:.0%} "
                  f"(rate:{pr['rate_triggers']}, pattern:{pr['pattern_triggers']}) "
                  f"{marker} {pr['status']}")

    # Metrics
    attack_phases = [pr for pr in phase_results if pr["phase_type"] == Phase.ATTACK and pr["status"] != "WARMUP"]
    calm_phases = [pr for pr in phase_results if pr["phase_type"] == Phase.CALM and pr["status"] != "WARMUP"]

    attack_detected = sum(pr["detections"] for pr in attack_phases)
    attack_total = sum(pr["packets"] for pr in attack_phases)
    attack_recall = min(1.0, attack_detected / attack_total) if attack_total > 0 else 0

    fp = sum(pr["detections"] for pr in calm_phases)
    fp_total = sum(pr["packets"] for pr in calm_phases)
    fp_rate = fp / fp_total if fp_total > 0 else 0

    print(f"\n  Overall: Attack Recall={attack_recall:.0%}, Normal FP={fp_rate:.0%}")

    return {
        "attack": attack_type,
        "attack_recall": attack_recall,
        "fp_rate": fp_rate,
    }


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 011: Frozen Baseline with Z-Scores")
    print("="*70)
    print("""
    Key fix: Baseline statistics are FROZEN after warmup.

    The z-score is computed against the frozen baseline,
    so the detector doesn't adapt to attack traffic.

    Detection logic:
    1. During warmup: accumulate baseline, sample similarities
    2. After warmup: FREEZE mean/std, never update
    3. Detection: z-score against frozen baseline
    4. Hysteresis: different enter/exit thresholds
    5. Confirmation: require 3 consecutive anomalies
    """)

    timeline = [
        TimePhase("calm-1", 600, 100, Phase.CALM),
        TimePhase("ATTACK-1", 30, 100000, Phase.ATTACK),
        TimePhase("calm-2", 300, 100, Phase.CALM),
        TimePhase("ATTACK-2", 60, 50000, Phase.ATTACK),
        TimePhase("calm-3", 300, 100, Phase.CALM),
    ]

    results = []
    for attack_type in ["dns_reflection", "syn_flood", "udp_flood"]:
        result = run_test(attack_type, timeline, scale=0.005)
        results.append(result)

    # Summary
    print("\n" + "="*70)
    print("FROZEN BASELINE SUMMARY")
    print("="*70)
    print(f"\n{'Attack':<20} {'Attack Recall':>15} {'Normal FP':>12}")
    print("-"*50)
    for r in results:
        print(f"{r['attack']:<20} {r['attack_recall']:>15.0%} {r['fp_rate']:>12.0%}")

    avg_recall = np.mean([r['attack_recall'] for r in results])
    avg_fp = np.mean([r['fp_rate'] for r in results])
    print("-"*50)
    print(f"{'Average':<20} {avg_recall:>15.0%} {avg_fp:>12.0%}")

    print(f"""

COMPARISON
==========
  008: 100% recall, 8% FP  (simple threshold)
  009: 100% recall, 5% FP  (voting ensemble)
  010:   0% recall, 0% FP  (adaptive z-scores - BROKEN)
  011: {avg_recall:.0%} recall, {avg_fp:.0%} FP  (frozen z-scores)
    """)


if __name__ == "__main__":
    main()
