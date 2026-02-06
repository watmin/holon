#!/usr/bin/env python3
"""
=============================================================================
BATCH 012 - CHALLENGE 013: Best Combination
=============================================================================

Combining the best elements from previous experiments:
- Frozen z-score baselines (011) - proven to work
- Voting from multiple signals (009) - reduces noise
- Gating logic - pattern requires rate confirmation
- Window-based smoothing - faster recovery than EWMA

Key insight: The 5% FP from 009/011 came from pattern signals during
recovery when rate was back to normal. Gate those out.

Run: ./scripts/run_with_venv.sh python scripts/challenges/012-batch/013-best-combination.py
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


class FrozenBaseline:
    def __init__(self):
        self.samples = []
        self.frozen = False
        self.mean = 0.0
        self.std = 1.0

    def observe(self, value: float):
        if not self.frozen:
            self.samples.append(value)

    def freeze(self):
        self.frozen = True
        if self.samples:
            self.mean = np.mean(self.samples)
            self.std = max(np.std(self.samples) if len(self.samples) > 1 else 0.05, 0.02)

    def z_score(self, value: float) -> float:
        return (value - self.mean) / self.std

    def is_anomalous(self, value: float, threshold: float = -2.5) -> bool:
        return self.z_score(value) < threshold


class ContinuousRateEncoder:
    def __init__(self, dimensions: int, scale: float = 1000.0):
        self.dimensions = dimensions
        self.scale = scale

    def encode_rate(self, pps: float) -> np.ndarray:
        if pps <= 0:
            pps = 1
        log_rate = np.log10(pps)
        indices = np.arange(self.dimensions)
        freqs = 1 / (self.scale ** (indices / self.dimensions))
        values = np.where(indices % 2 == 0, np.sin(log_rate * freqs), np.cos(log_rate * freqs))
        return np.sign(values).astype(np.int8)


class BestCombinationDetector:
    """
    Combines:
    1. Rate z-score with frozen baseline
    2. Pattern z-score with frozen baseline
    3. Residual detection (resonance-based)
    4. Gating: pattern anomalies require rate confirmation
    5. Window smoothing for output
    """

    def __init__(self, warmup_packets: int = WARMUP_PACKETS):
        self.store = CPUStore(dimensions=DIMENSIONS)
        self.encoder = self.store.encoder
        self.warmup_packets = warmup_packets
        self.packet_count = 0
        self.warmup_complete = False

        # Rate tracking
        self.rate_encoder = ContinuousRateEncoder(DIMENSIONS)
        self.rate_accum = self.encoder.create_accumulator()
        self._rate_norm = None
        self.rate_baseline = FrozenBaseline()

        # Pattern tracking
        self.pattern_accum = self.encoder.create_accumulator()
        self._pattern_norm = None
        self.pattern_baseline = FrozenBaseline()

        # Window-based smoothing (NOT EWMA)
        self.anomaly_window = deque(maxlen=20)

    def process(self, packet: dict, pps: float) -> dict:
        self.packet_count += 1
        is_warmup = self.packet_count <= self.warmup_packets

        packet_vec = self.encoder.encode_data(packet)
        rate_vec = self.rate_encoder.encode_rate(pps)

        if is_warmup:
            self.rate_accum = self.encoder.accumulate(self.rate_accum, rate_vec)
            self.pattern_accum = self.encoder.accumulate(self.pattern_accum, packet_vec)

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

            return {"packet_num": self.packet_count, "is_anomalous": False}

        # Compute similarities
        rate_sim = self.store.similarity(rate_vec, self._rate_norm, metric="cosine")
        pattern_sim = self.store.similarity(packet_vec, self._pattern_norm, metric="cosine")

        # Z-scores
        rate_z = self.rate_baseline.z_score(rate_sim)
        pattern_z = self.pattern_baseline.z_score(pattern_sim)

        # Detection logic with gating
        # Rate anomaly: z < -2.5 (strong)
        rate_is_anomalous = rate_z < -2.5

        # Pattern anomaly: z < -2.0 (moderate)
        pattern_is_anomalous = pattern_z < -2.0

        # Rate is "somewhat elevated" (for gating): z < -0.5
        rate_confirms = rate_z < -0.5

        # GATED LOGIC:
        # 1. Rate anomaly alone → anomaly (volumetric)
        # 2. Pattern anomaly + rate confirms → anomaly
        # 3. Pattern anomaly + rate normal → NOT anomaly (recovery artifact)

        raw_anomaly = rate_is_anomalous or (pattern_is_anomalous and rate_confirms)

        # Window-based smoothing
        self.anomaly_window.append(1 if raw_anomaly else 0)
        anomaly_fraction = sum(self.anomaly_window) / len(self.anomaly_window)

        # Threshold for final decision
        is_anomalous = anomaly_fraction > 0.4

        return {
            "packet_num": self.packet_count,
            "is_anomalous": is_anomalous,
            "rate_z": rate_z,
            "pattern_z": pattern_z,
            "rate_anomalous": rate_is_anomalous,
            "pattern_anomalous": pattern_is_anomalous,
            "rate_confirms": rate_confirms,
            "raw": raw_anomaly,
            "fraction": anomaly_fraction,
        }


# =============================================================================
# SIMULATION
# =============================================================================

class Phase(Enum):
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


ATTACK_GENERATORS = {
    "dns_reflection": lambda rng: {"protocol": "UDP", "src_port": 53,
                                    "dst_port": rng.randint(49152, 65535),
                                    "payload_size": rng.randint(256, 4096)},
    "syn_flood": lambda rng: {"protocol": "TCP", "src_port": rng.randint(1, 65535),
                               "dst_port": 80, "flags": "S", "payload_size": 0},
    "udp_flood": lambda rng: {"protocol": "UDP", "src_port": rng.randint(1, 65535),
                               "dst_port": rng.randint(1, 65535),
                               "payload_size": rng.randint(0, 1400)},
}


def run_test(attack_type: str, phases: List[TimePhase], scale: float = 0.005):
    print(f"\n{'='*70}")
    print(f"BEST COMBINATION TEST: {attack_type}")
    print(f"{'='*70}")

    first_calm_packets = int(phases[0].duration_seconds * phases[0].packets_per_second * scale)
    warmup_packets = min(first_calm_packets - 10, 400)

    detector = BestCombinationDetector(warmup_packets=warmup_packets)
    rng = random.Random(42)
    attack_gen = ATTACK_GENERATORS[attack_type]

    phase_results = []

    for phase in phases:
        scaled_packets = max(1, int(phase.duration_seconds * phase.packets_per_second * scale))
        phase_detections = 0
        rate_triggers = 0
        pattern_triggers = 0
        raw_triggers = 0

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
                if result.get("raw", False):
                    raw_triggers += 1

        if not detector.warmup_complete or phase.name == "calm-1":
            status = "WARMUP"
            detection_rate = 0
        else:
            detection_rate = phase_detections / scaled_packets if scaled_packets > 0 else 0
            if phase.phase_type == Phase.ATTACK:
                status = "DETECTED" if detection_rate > 0.5 else "MISSED"
            else:
                status = "CLEAN" if detection_rate < 0.02 else "FP"

        phase_results.append({
            "name": phase.name,
            "phase_type": phase.phase_type,
            "packets": scaled_packets,
            "detections": phase_detections,
            "detection_rate": detection_rate,
            "rate_triggers": rate_triggers,
            "pattern_triggers": pattern_triggers,
            "raw_triggers": raw_triggers,
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
                  f"(rate:{pr['rate_triggers']}, pat:{pr['pattern_triggers']}, raw:{pr['raw_triggers']}) "
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

    return {"attack": attack_type, "attack_recall": attack_recall, "fp_rate": fp_rate}


def main():
    print("="*70)
    print("BATCH 012 - CHALLENGE 013: Best Combination")
    print("="*70)
    print("""
    Combining best elements:

    1. Frozen z-score baselines (from 011)
    2. Gating logic:
       - Rate anomaly (z < -2.5) → always anomaly
       - Pattern anomaly (z < -2.0) + rate confirms (z < -0.5) → anomaly
       - Pattern anomaly alone → NOT anomaly (recovery artifact)
    3. Window-based smoothing (not EWMA)
       - 20-packet window
       - Threshold > 0.4 for final decision
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
    print("FINAL SUMMARY")
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

PROGRESSION
===========
  008: 100% recall,  8% FP  (simple positional encoding)
  009: 100% recall,  5% FP  (voting ensemble)
  011: 100% recall,  5% FP  (frozen z-scores)
  013: {avg_recall:.0%} recall, {avg_fp:.0%} FP  (gated + frozen z-scores)

KEY IMPROVEMENTS
================
1. Positional encoding for continuous rate (no magic bands)
2. Frozen z-score baselines (don't adapt to attacks)
3. Gating: pattern anomalies require rate confirmation
4. Window smoothing for stable output
    """)


if __name__ == "__main__":
    main()
